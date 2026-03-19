# 🎬 Movie Recommender - Streamlit Web App

A beautiful, interactive web application for movie recommendations using content-based filtering with cosine similarity.

## 📋 Features

- **🎯 Movie Search & Selection**: Choose from 4800+ movies in the TMDB dataset
- **🎬 Smart Recommendations**: Get up to 10 personalized movie recommendations
- **📊 Accuracy Metrics**: View genre match, cast match, and keyword similarity scores
- **🔍 Detailed Movie Info**: See genres, cast, overview, and similarity scores
- **📈 Interactive Dashboard**: Real-time metrics and visualization
- **🎨 Beautiful UI**: Modern, responsive Streamlit interface

## 🛠️ Installation & Setup

### 1. Navigate to the ML Projects directory
```bash
cd "C:\Users\admin\Desktop\python\ML Projects"
```

### 2. Install required packages
```bash
pip install -r requirements.txt
```

### 3. Download and setup data files (automatic)
The app can automatically download the required CSV files from Google Drive the first time it runs.

If you prefer, you can still download them manually and place them in the project directory.

- **Movies Dataset**: [tmdb_5000_movies.csv](https://drive.google.com/file/d/12y_r9SWWf3Z9YLeaag9_Fk4O_aCLcs4M/view?usp=sharing)
- **Credits Dataset**: [tmdb_5000_credits.csv](https://drive.google.com/file/d/1qUkFoqfCsD6RKkqMzbtF2gLNV3tufIwo/view?usp=sharing)

**Manual Download Instructions (if automatic download fails):**
1. Click the links above to download the CSV files
2. Place both files (`tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`) in the same directory as `app.py`
3. Ensure the files are not renamed or modified

## 🚀 Running the App

### Option 1: Using Streamlit CLI (Recommended)
```bash
streamlit run app.py
```

### Option 2: Using Python
```bash
python -m streamlit run app.py
```

The app will start at `http://localhost:8501` in your default browser.

## 📖 How to Use

1. **Select a Movie**: Use the dropdown to choose a movie from the database
2. **Configure Settings** (Sidebar):
   - Adjust the number of recommendations (1-10)
   - Toggle "Show Accuracy Metrics" to see match percentages
   - Toggle "Show Movie Details" to see genres, cast, and plot summaries
3. **Click Get Recommendations**: The app will display similar movies with scores
4. **View Metrics**: Check accuracy scores to understand recommendation quality

## 📊 Accuracy Metrics Explained

- **Genre Match** (%): Percentage of recommendations sharing genres with the selected movie
  - Higher = Better thematic similarity

- **Cast Match** (%): Percentage of recommendations with shared actors
  - Indicates if popular actors appear in recommended movies

- **Keyword Match** (%): Percentage of recommendations with thematic keywords
  - Measures semantic content similarity

- **Overall Accuracy**: Average of all three metrics
  - ✓ 70%+ = GOOD
  - ∼ 50-70% = FAIR
  - ✗ <50% = POOR

## 🔧 Algorithm Details

**Method**: Content-Based Filtering with Cosine Similarity

**Feature Extraction**:
1. Movie overview text (plot summary)
2. Genres (action, drama, comedy, etc.)
3. Keywords (themes, topics)
4. Cast (top 3 actors)
5. Director(s)

**Processing Pipeline**:
1. Text preprocessing and lowercasing
2. Porter Stemming (word normalization)
3. Bag of Words vectorization (5000 features)
4. Cosine similarity computation
5. Ranking by similarity score

## 📁 Project Structure
```
ML Projects/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── MovieRecommender.ipynb         # Original Jupyter notebook
├── tmdb_5000_movies.csv           # Movie metadata
├── tmdb_5000_credits.csv          # Cast & crew data
└── README.md                       # This file
```

## 🐛 Troubleshooting

**Issue**: Module not found error
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: CSV files not found or failed to download
- **Solution**: Ensure you have network access and retry. If the app cannot download the datasets automatically, download them manually using the links above and place them in the same directory as `app.py`.

**Issue**: App runs slowly on first load
- **Solution**: This is normal - the app preprocesses the entire dataset. Subsequent runs use caching.

**Issue**: Can't access the app
- **Solution**: Check if `http://localhost:8501` is accessible. Try refreshing the page.

## 📈 Performance

- **Data Loading**: ~5-10 seconds (first time, then cached)
- **Recommendations**: <1 second
- **Metrics Calculation**: <1 second
- **Total Response Time**: <2 seconds per query

## 🎨 Customization

Edit the Streamlit configuration in `app.py`:
- Change the app title/icon in `st.set_page_config()`
- Adjust color scheme by modifying CSS in markdown
- Modify default settings in the sidebar

## 💡 Future Enhancements

- [ ] Hybrid recommendations (collaborative + content-based)
- [ ] User ratings integration
- [ ] Movie visualization (posters, trailers)
- [ ] Export recommendations as CSV
- [ ] Dark/Light mode toggle
- [ ] Multi-language support

## 📄 License

This project uses the TMDB 5000 dataset from Kaggle.

## 👨‍💻 Author

Created as part of ML movie recommendation project

---

**Enjoy discovering your next favorite movie! 🍿🎬**
