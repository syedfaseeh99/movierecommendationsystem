import ast
import os
import pandas as pd
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert
from dotenv import load_dotenv


class MovieRecommenderSystem:
    def __init__(self):
        self._load_config()
        self.engine = create_engine(self.conn_string)
        self.conn = None
        self.movies_df = None
        self.similarity_matrix = None

    def _load_config(self):
        """Load database configuration from .env or use defaults"""
        env_path = os.path.join(os.getcwd(), '.env')
        if not os.path.exists(env_path):
            with open(env_path, 'w') as f:
                f.write("DB_USER=postgres\nDB_PASSWORD=faseehpg\nDB_HOST=localhost\nDB_NAME=movierecs")

        load_dotenv(env_path)
        self.conn_string = (
            f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
            f"{os.getenv('DB_PASSWORD', 'faseehpg')}@"
            f"{os.getenv('DB_HOST', 'localhost')}:5432/"
            f"{os.getenv('DB_NAME', 'movierecs')}"
        )

    def connect(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME', 'movierecs'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'faseehpg'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=5432
        )
        return self.conn

    def _extract_names(self, json_str):
        """Helper function to parse JSON strings"""
        try:
            return [i['name'] for i in ast.literal_eval(json_str)]
        except:
            return []

    def _get_top_3_cast(self, cast_json):
        """Helper function to get top 3 cast members"""
        try:
            return [i['name'] for i in ast.literal_eval(cast_json)][:3]
        except:
            return []

    def _get_directors(self, crew_json):
        """Helper function to get directors"""
        try:
            return [i['name'] for i in ast.literal_eval(crew_json) if i['job'] == 'Director']
        except:
            return []

    def load_and_preprocess_data(self, movies_path='movies.csv', credits_path='credits.csv'):
        """Load and preprocess the raw data"""
        movies = pd.read_csv(movies_path)
        credits = pd.read_csv(credits_path)
        movies = movies.merge(credits, on='title')

        # Process JSON columns
        movies['genres'] = movies['genres'].apply(self._extract_names)
        movies['keywords'] = movies['keywords'].apply(self._extract_names)
        movies['cast'] = movies['cast'].apply(self._get_top_3_cast)
        movies['crew'] = movies['crew'].apply(self._get_directors)

        return movies

    def initialize_database(self, movies):
        """Initialize database tables with movie data"""
        with self.engine.begin() as conn:
            # Clear existing tables
            tables = ['movie_crew', 'movie_cast', 'movie_keywords',
                      'movie_genres', 'keywords', 'genres', 'movies']
            for table in tables:
                conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))

            # Load movies
            movies[['id', 'title', 'overview']].rename(columns={'id': 'movie_id'}) \
                .to_sql('movies', conn, if_exists='append', index=False)

            # Load genres
            pd.DataFrame({'genre_name': list(set(g for sublist in movies['genres'] for g in sublist))}) \
                .to_sql('genres', conn, if_exists='append', index=False)

            print("✅ Database initialized successfully")


class MovieRecommender:
    def __init__(self, connection):
        self.connection = connection
        self.movies_df = None
        self.similarity_matrix = None
        self.title_to_idx = {}

    def load_data(self):
        """Load data from database for recommendations"""
        try:
            query = """
                    SELECT m.movie_id, \
                           m.title, \
                           COALESCE(m.overview, '')                                  AS overview, \
                           COALESCE(m.poster_url, '')                                AS poster_url, \
                           COALESCE( \
                                   (SELECT string_agg(g.genre_name, ' ') \
                                    FROM movie_genres mg \
                                             JOIN genres g ON mg.genre_id = g.genre_id \
                                    WHERE mg.movie_id = m.movie_id), '')             AS genres, \
                           COALESCE( \
                                   (SELECT string_agg(p.name, ' ') \
                                    FROM (SELECT p.name
                                          FROM movie_cast mc
                                                   JOIN people p ON mc.person_id = p.person_id
                                          WHERE mc.movie_id = m.movie_id
                                          ORDER BY mc.cast_order LIMIT 3) AS p), '') AS top_cast, \
                           COALESCE( \
                                   (SELECT string_agg(p.name, ' ') \
                                    FROM movie_crew mc \
                                             JOIN people p ON mc.person_id = p.person_id \
                                    WHERE mc.movie_id = m.movie_id \
                                      AND mc.job = 'Director'), '')                  AS directors, \
                           COALESCE( \
                                   (SELECT string_agg(k.keyword_name, ' ') \
                                    FROM movie_keywords mk \
                                             JOIN keywords k ON mk.keyword_id = k.keyword_id \
                                    WHERE mk.movie_id = m.movie_id), '')             AS keywords
                    FROM movies m
                    GROUP BY m.movie_id, m.title, m.overview, m.poster_url \
                    """

            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query)
                self.movies_df = pd.DataFrame(cursor.fetchall())

            if not self.movies_df.empty:
                self.movies_df['combined_features'] = (
                        self.movies_df['overview'] + ' ' +
                        self.movies_df['genres'] + ' ' +
                        self.movies_df['keywords'] + ' ' +
                        self.movies_df['top_cast'] + ' ' +
                        self.movies_df['directors']
                )

                self.movies_df['title_lower'] = self.movies_df['title'].str.lower()
                self.title_to_idx = dict(zip(
                    self.movies_df['title_lower'],
                    self.movies_df.index
                ))
                return True
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    @lru_cache(maxsize=1)
    def build_similarity_matrix(self):
        """Build the similarity matrix using TF-IDF"""
        if not self.load_data():
            return None

        tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_features=5000
        )

        try:
            tfidf_matrix = tfidf.fit_transform(self.movies_df['combined_features'])
            self.similarity_matrix = cosine_similarity(tfidf_matrix)
            return self.similarity_matrix
        except Exception as e:
            print(f"Error building similarity matrix: {e}")
            return None

    def recommend(self, movie_title, top_n=5):
        """Get recommendations for a movie"""
        if self.similarity_matrix is None:
            if self.build_similarity_matrix() is None:
                return pd.DataFrame()

        movie_title_lower = movie_title.lower()
        if movie_title_lower not in self.title_to_idx:
            return pd.DataFrame()

        movie_idx = self.title_to_idx[movie_title_lower]
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
        movie_indices = [i[0] for i in sim_scores]

        recommendations = self.movies_df.iloc[movie_indices][['movie_id', 'title', 'poster_url']].copy()
        recommendations['similarity'] = [round(i[1], 3) for i in sim_scores]

        return recommendations.reset_index(drop=True)