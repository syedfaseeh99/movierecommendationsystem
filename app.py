import streamlit as st
import pandas as pd
from movierecommender import MovieRecommenderSystem, MovieRecommender
import psycopg2
from psycopg2.extras import execute_values

# Configure Streamlit page
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)
st.markdown("""
    <style>
    html, body, [class*="st-"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    .stButton>button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: none;
        border-radius: 5px;
        padding: 0.5em 1em;
    }
    .stButton>button:hover {
        background-color: #dddddd !important;
        color: #000000 !important;
    }
    .stSelectbox div, .stTextInput input {
        background-color: #121212 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 5px;
    }
    .stSidebar, .css-1d391kg {  /* Sidebar background */
        background-color: #111111 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Database connection parameters
DB_CONFIG = {
    "dbname": "movierecs",
    "user": "postgres",
    "password": "faseehpg",
    "host": "localhost",
    "port": "5432"
}

# Initialize connection to database
@st.cache_resource
def init_connection():
    return psycopg2.connect(**DB_CONFIG)

# Initialize MovieRecommender
@st.cache_resource
def get_recommender():
    conn = init_connection()
    recommender = MovieRecommender(conn)
    recommender.load_data()
    recommender.build_similarity_matrix()
    return recommender

# App UI
st.title("🎬 Movie Recommendation System")
st.write("Discover movies similar to your favorites!")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.recommender = None
    st.session_state.recommendations = None

# Sidebar for database setup
with st.sidebar:
    st.header("Database Setup")
    if st.button("Initialize Database"):
        with st.spinner("Loading and preprocessing data..."):
            try:
                system = MovieRecommenderSystem()
                movies = system.load_and_preprocess_data()

                # Connect to the database
                conn = init_connection()
                with conn.cursor() as cur:
                    # Remove duplicates from input data
                    movies = movies.drop_duplicates(subset='id')
                    data = [tuple(x) for x in movies[['id', 'title', 'overview']].to_numpy()]

                    # Clear existing data
                    cur.execute("TRUNCATE TABLE movies CASCADE")

                    # Batch insert data safely (no duplicate movie_ids)
                    execute_values(
                        cur,
                        """
                        INSERT INTO movies (movie_id, title, overview)
                        VALUES %s
                        ON CONFLICT (movie_id) DO UPDATE SET
                            title = EXCLUDED.title,
                            overview = EXCLUDED.overview
                        """,
                        data
                    )

                    # Reset sequence
                    cur.execute("""
                        SELECT setval(
                            pg_get_serial_sequence('movies', 'movie_id'),
                            COALESCE((SELECT MAX(movie_id) FROM movies), 1),
                            false
                        )
                    """)
                    conn.commit()

                st.session_state.initialized = True
                st.session_state.recommender = get_recommender()
                st.success("Database initialized successfully!")

            except Exception as e:
                st.error(f"Error initializing database: {str(e)}")
                st.session_state.initialized = False

    if st.session_state.initialized:
        st.success("✅ Database ready")

# Main recommendation interface
if st.session_state.initialized and st.session_state.recommender:
    # Get movie titles for dropdown
    @st.cache_data
    def get_movie_titles():
        with st.session_state.recommender.connection.cursor() as cur:
            cur.execute("SELECT title FROM movies ORDER BY title")
            return [row[0] for row in cur.fetchall()]

    movie_titles = get_movie_titles()

    # Movie selector
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        options=movie_titles
    )

    if st.button('Show Recommendation'):
        with st.spinner('Finding similar movies...'):
            try:
                # 🔧 Call the correct method
                recommendations = st.session_state.recommender.recommend(selected_movie, top_n=5)

                if not recommendations.empty:
                    st.subheader(f"Movies similar to {selected_movie}:")
                    cols = st.columns(len(recommendations))
                    for i, row in recommendations.iterrows():
                        with cols[i]:
                            st.write(f"**{row['title']}**")
                            st.write(f"Similarity: {row['similarity']:.2f}")
                            if row['poster_url']:
                                st.image(row['poster_url'], use_column_width=True)
                else:
                    st.warning("No recommendations found for the selected movie.")
            except Exception as e:
                st.error(f"Error getting recommendations: {str(e)}")
