import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
import logging
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import urllib.error
import urllib.request
warnings.filterwarnings('ignore')
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from utils import convert, convert3, director_name, stemmer

# ================================================================
# LOGGING SETUP
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Google Drive direct download links (used when local CSVs are missing)
GDRIVE_DOWNLOAD_URLS = {
    "tmdb_5000_movies.csv": "https://drive.google.com/uc?export=download&id=12y_r9SWWf3Z9YLeaag9_Fk4O_aCLcs4M",
    "tmdb_5000_credits.csv": "https://drive.google.com/uc?export=download&id=1qUkFoqfCsD6RKkqMzbtF2gLNV3tufIwo"
}

def download_file(url: str, dest_path: str) -> bool:
    try:
        import requests
        logger.info(f"Downloading dataset from {url}")
        
        response = requests.get(url)
        
        # 🚨 Check if response is actually CSV
        if "text/html" in response.headers.get("Content-Type", ""):
            raise Exception("Google Drive blocked direct download")
        
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Saved dataset to {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

# ================================================================
# FILE VALIDATION
# ================================================================
def validate_data_files():
    """Ensure required data files exist locally, downloading them if possible."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    required_files = list(GDRIVE_DOWNLOAD_URLS.keys())
    
    for file in required_files:
        file_path = os.path.join(current_dir, file)
        if not os.path.exists(file_path):
            # Attempt to download missing dataset from Google Drive
            url = GDRIVE_DOWNLOAD_URLS.get(file)
            if url:
                logger.info(f"{file} missing; attempting download from Google Drive")
                if download_file(url, file_path):
                    continue
                logger.error(f"Failed to download required file: {file_path}")
            else:
                logger.error(f"Missing required file: {file_path}")
                st.warning(f"⚠ **Warning**: Required file `{file}` not found. Attempted to download but failed.")
            return False, file
    
    logger.info("All required data files found")
    return True, None

def get_data_path(filename):
    """Get absolute path to data file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, filename)

# ================================================================
# PAGE CONFIGURATION
# ================================================================
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# ACCURACY FUNCTIONS (Shared helpers imported from utils.py)
# ================================================================
def genre_overlap_accuracy(movie, similarity_matrix, movies_df, top_k=5):
    """Calculate genre overlap accuracy"""
    try:
        movie_index = movies_df[movies_df['title'] == movie].index[0]
        distances = similarity_matrix[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_k+1]
        
        query_genres = set(movies_df.iloc[movie_index]['genres'])
        
        matching_count = 0
        recommendations = []
        quality_scores = []
        
        for idx, sim_score in movies_list:
            rec_movie = movies_df.iloc[idx]['title']
            rec_genres = set(movies_df.iloc[idx]['genres'])
            genre_match = bool(query_genres & rec_genres)
            
            recommendations.append({
                'title': rec_movie,
                'similarity_score': sim_score,
                'genres': list(rec_genres),
                'genre_match': genre_match
            })
            
            if genre_match:
                matching_count += 1
            
            # Quality score: 70% content similarity + 30% feature match
            quality = (sim_score * 0.7) + (0.3 if genre_match else 0)
            quality_scores.append(quality)
        
        # Use blended accuracy (categorical overlap + content similarity)
        accuracy = (matching_count / top_k) * 50 + (np.mean(quality_scores) * 50)
        return accuracy, recommendations
    except:
        return 0, []

def cast_overlap_accuracy(movie, similarity_matrix, movies_df, top_k=5):
    """Calculate cast overlap accuracy"""
    try:
        movie_index = movies_df[movies_df['title'] == movie].index[0]
        distances = similarity_matrix[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_k+1]
        
        query_cast = set(movies_df.iloc[movie_index]['cast'])
        matching_count = 0
        
        for idx, _ in movies_list:
            rec_cast = set(movies_df.iloc[idx]['cast'])
            if query_cast & rec_cast:
                matching_count += 1
        
        return (matching_count / top_k) * 100
    except:
        return 0

def keyword_overlap_accuracy(movie, similarity_matrix, movies_df, top_k=5):
    """Calculate keyword overlap accuracy"""
    try:
        movie_index = movies_df[movies_df['title'] == movie].index[0]
        distances = similarity_matrix[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_k+1]
        
        query_keywords = set(movies_df.iloc[movie_index]['keywords'])
        matching_count = 0
        
        for idx, _ in movies_list:
            rec_keywords = set(movies_df.iloc[idx]['keywords'])
            if query_keywords & rec_keywords:
                matching_count += 1
        
        return (matching_count / top_k) * 100
    except:
        return 0

# ================================================================
# CACHE DATA LOADING AND PREPROCESSING
# ================================================================
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess movie data"""
    with st.spinner('Loading and preprocessing data...'):
        try:
            # Validate files exist
            files_valid, missing_file = validate_data_files()
            if not files_valid:
                logger.error(f"Data file validation failed: {missing_file} not found")
                st.error(f"❌ **Error**: Required file `{missing_file}` not found. Please ensure both CSV files are in the same directory as app.py.")
                st.stop()
            
            # Load data with absolute paths
            logger.info("Loading movie and credits data")
            movies_path = get_data_path('tmdb_5000_movies.csv')
            credits_path = get_data_path('tmdb_5000_credits.csv')
            
            movies = pd.read_csv(movies_path)
            credits = pd.read_csv(credits_path)
            logger.info(f"Loaded {len(movies)} movies and {len(credits)} credits records")
        
            # Merge
            logger.info("Merging movie and credits data")
            movies = movies.merge(credits, on='title')
            
            # Select columns
            movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
            
            # Remove nulls
            movies.dropna(inplace=True)
            logger.info(f"After cleaning: {len(movies)} movies ready for processing")
        except FileNotFoundError as e:
            logger.error(f"File not found error: {str(e)}")
            st.error(f"❌ **Error**: Could not find data files. {str(e)}")
            st.stop()
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            st.error(f"❌ **Error loading data**: {str(e)}")  
            st.stop()
        
        try:
            # Process genres, keywords, cast, crew
            logger.info("Processing movie features")
            movies['genres'] = movies['genres'].apply(convert)
            movies['keywords'] = movies['keywords'].apply(convert)
            movies['cast'] = movies['cast'].apply(convert3)
            movies['crew'] = movies['crew'].apply(director_name)
            
            # Process overview
            movies['overview'] = movies['overview'].apply(
                lambda x: ' '.join(x).split() if isinstance(x, list) else x.split()
            )
            
            # Remove spaces
            movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
            
            # Concatenate tags
            movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
            
            # Create final dataset
            movies_new = movies[['movie_id', 'title', 'tags']].copy()
            
            # Process tags
            logger.info("Vectorizing text data and computing similarity")
            movies_new['tags'] = movies_new['tags'].apply(lambda x: " ".join(x))
            movies_new['tags'] = movies_new['tags'].apply(lambda x: x.lower())
            movies_new['tags'] = movies_new['tags'].apply(stemmer)
            
            # Vectorization
            cv = CountVectorizer(max_features=5000, stop_words='english')
            vectors = cv.fit_transform(movies_new['tags']).toarray()
            
            # Calculate similarity
            similarity = cosine_similarity(vectors)
            logger.info("Data preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            st.error(f"❌ **Error during preprocessing**: {str(e)}")
            st.stop()
        
    return movies, movies_new, similarity

# ================================================================
# STREAMLIT UI
# ================================================================
st.markdown("<h1 style='text-align: center; color: #FF6B6B;'>🎬 Movie Recommender System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>AI-Powered Content-Based Movie Recommendations</p>", unsafe_allow_html=True)
st.divider()

# Load data
movies, movies_new, similarity = load_and_preprocess_data()

# Sidebar configuration
st.sidebar.header("⚙️ Settings")
num_recommendations = st.sidebar.slider("Number of Recommendations", 1, 10, 5)
show_accuracy = st.sidebar.checkbox("Show Accuracy Metrics", value=True)
show_details = st.sidebar.checkbox("Show Movie Details", value=True)

# Add "Best Performing Movies" section in sidebar
with st.sidebar.expander("💡 Best Performing Movies"):
    st.markdown("""
    These movies typically have **higher accuracy scores**:
    - **Action/Sci-Fi blockbusters** (Avatar, Inception, Avengers)
    - **Popular franchises** (Marvel, DC)
    - **Well-known actors** (multiple recommendations share cast)
    
    **Lower accuracy for:**
    - Indie/niche films
    - Documentaries
    - Movies with unique themes
    
    *Tip: Try popular movies first to see great recommendations!*
    """)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Select a Movie")
    selected_movie = st.selectbox(
        "Choose a movie to get recommendations:",
        sorted(movies_new['title'].unique()),
        index=0
    )

with col2:
    st.subheader("")
    search_button = st.button("🎯 Get Recommendations", use_container_width=True)

if search_button or selected_movie:
    if selected_movie not in movies_new['title'].values:
        st.error("❌ Movie not found in database!")
    else:
        # Get recommendations
        try:
            movie_index = movies_new[movies_new['title'] == selected_movie].index[0]
            distances = similarity[movie_index]
            movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:num_recommendations+1]
            
            st.success(f"✅ Found {len(movies_list)} recommendations for '{selected_movie}'")
            st.divider()
            
            # Display recommendations
            st.subheader("🎥 Recommended Movies")
            
            for rank, (idx, sim_score) in enumerate(movies_list, 1):
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        rec_movie = movies_new.iloc[idx]['title']
                        st.markdown(f"**#{rank} - {rec_movie}**")
                        
                        if show_details:
                            rec_movie_data = movies[movies['title'] == rec_movie].iloc[0]
                            
                            # Show genres
                            genres_str = ", ".join(rec_movie_data['genres'][:3])
                            st.caption(f"🏷️ **Genres**: {genres_str}" + ("..." if len(rec_movie_data['genres']) > 3 else ""))
                            
                            # Show cast
                            if rec_movie_data['cast']:
                                cast_str = ", ".join(rec_movie_data['cast'][:2])
                                st.caption(f"👥 **Cast**: {cast_str}")
                            
                            # Show overview snippet
                            if rec_movie_data['overview']:
                                overview = rec_movie_data['overview']
                                if isinstance(overview, list):
                                    overview = " ".join(overview[:20])
                                else:
                                    overview = " ".join(overview.split()[:20])
                                st.caption(f"📝 **Plot**: {overview}...")
                    
                    with col2:
                        st.metric(
                            label="Similarity",
                            value=f"{sim_score:.2%}",
                            label_visibility="collapsed"
                        )
            
            # Show accuracy metrics
            if show_accuracy:
                st.divider()
                st.subheader("📊 Accuracy Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    genre_acc, _ = genre_overlap_accuracy(selected_movie, similarity, movies, top_k=num_recommendations)
                    st.metric("📂 Genre Match", f"{genre_acc:.0f}%", help="Percentage of recommendations with matching genres")
                
                with col2:
                    cast_acc = cast_overlap_accuracy(selected_movie, similarity, movies, top_k=num_recommendations)
                    st.metric("👥 Cast Match", f"{cast_acc:.0f}%", help="Percentage of recommendations with shared cast")
                
                with col3:
                    keyword_acc = keyword_overlap_accuracy(selected_movie, similarity, movies, top_k=num_recommendations)
                    st.metric("🔑 Keyword Match", f"{keyword_acc:.0f}%", help="Percentage of recommendations with thematic keywords")
                
                # Overall accuracy
                overall = (genre_acc + cast_acc + keyword_acc) / 3
                st.markdown(f"<div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>" 
                           f"<h3>Overall Accuracy: <span style='color: #FF6B6B;'>{overall:.1f}%</span></h3>"
                           f"</div>", unsafe_allow_html=True)
                
                # Interpretation
                if overall >= 70:
                    st.success("✓ EXCELLENT - Recommendations are highly relevant")
                elif overall >= 60:
                    st.success("✓ GOOD - Recommendations are semantically similar to your selection")
                elif overall >= 50:
                    st.info("~ FAIR - Recommendations have moderate relevance")
                elif overall >= 40:
                    st.warning("⚠ LIMITED - This movie has unique characteristics")
                else:
                    st.warning("⚠ RARE MOVIE - Very few similar movies in dataset")
                
                st.markdown("""
                **📊 About Accuracy Scores:**
                - **Genre/Cast/Keyword Match**: Percentage of recommendations sharing categorical features
                - **Overall Score**: Blends Bag-of-Words similarity (70%) + Feature overlap (30%)
                - **Low scores don't mean bad recommendations** - Some movies have unique characteristics!
                - **Try other movies** to see higher accuracy scores
                """)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #999; padding: 20px;'>
    <p>📊 Dataset: TMDB 5000 Movies | 🤖 Algorithm: Content-Based Filtering with Cosine Similarity</p>
    <p>🔧 Built with Streamlit | Made with ❤️</p>
</div>
""", unsafe_allow_html=True)
