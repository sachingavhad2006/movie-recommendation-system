# Notebook version: AI-Based Movie Recommendation System
# Put both CSV files in the same folder as this notebook.

import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
key = "movie_id" if "movie_id" in credits.columns else "id"
data = movies.merge(credits, left_on="id", right_on=key, how="left", suffixes=("", "_credits"))
title_col = "title" if "title" in data.columns else "title_x"
data = data.rename(columns={title_col: "title"})

def parse_names(value):
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        items = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(items, list):
        return []
    return [str(x["name"]).replace(" ", "") for x in items if isinstance(x, dict) and x.get("name")]

for col in ["genres", "keywords"]:
    if col in data.columns:
        data[col] = data[col].apply(parse_names)
if "cast" in data.columns:
    data["cast"] = data["cast"].apply(lambda x: parse_names(x)[:3])
if "crew" in data.columns:
    def directors(value):
        if not isinstance(value, str) or not value.strip():
            return []
        try:
            people = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
        return [str(p["name"]).replace(" ", "") for p in people
                if isinstance(p, dict) and p.get("job") == "Director" and p.get("name")]
    data["crew"] = data["crew"].apply(directors)
data["overview"] = data["overview"].fillna("").astype(str).apply(str.split)
feature_cols = [c for c in ["overview", "genres", "keywords", "cast", "crew"] if c in data.columns]
data["tags"] = data[feature_cols].apply(
    lambda row: " ".join(" ".join(v) if isinstance(v, list) else str(v) for v in row), axis=1
).str.lower()
tfidf = TfidfVectorizer(stop_words="english", max_features=50000)
vectors = tfidf.fit_transform(data["tags"].fillna(""))
similarity = cosine_similarity(vectors, dense_output=False)

def recommend(movie_title, n=5):
    matches = data.index[data["title"].str.casefold() == movie_title.casefold()].tolist()
    if not matches:
        print("Movie not found. Check the title spelling.")
        return
    idx = matches[0]
    scores = similarity[idx].toarray().ravel()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    print(f"Movies similar to: {data.iloc[idx]['title']}")
    for rank, (i, score) in enumerate([(i, s) for i, s in ranked if i != idx][:n], 1):
        print(f"{rank}. {data.iloc[i]['title']} (similarity: {score:.3f})")

recommend("Avatar", 5)
