# AI-Based Movie Recommendation System

**Student:** Sachin Dnyaneshwar Gavhad  
**Department:** Artificial Intelligence and Data Science (AIDS)  
**Academic Year:** 2026–27

A content-based movie recommendation system built with Python, Pandas, Scikit-learn, TF-IDF, cosine similarity, and Streamlit.

## Dataset
Download the TMDB 5000 Movie Dataset from:
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Put these two files in the same folder as `app.py`:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

## Run on Windows
1. Install Python 3.10+.
2. Open Command Prompt in this folder.
3. Run `python -m pip install -r requirements.txt`
4. Run `python -m streamlit run app.py`
5. Open the local URL shown in the terminal.

## Method
Dataset loading → preprocessing → feature engineering → TF-IDF → cosine similarity → top-N recommendations.

The similarity score is text similarity, not a quality rating. Add screenshots of your running application and output before submitting the final report.
