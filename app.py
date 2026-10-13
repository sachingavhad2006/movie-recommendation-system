import ast
from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="MovieMatch AI", page_icon="🎬", layout="wide")
st.title("🎬 MovieMatch AI")
st.subheader("AI-Based Movie Recommendation System")
st.write("Choose a movie and discover similar movies using TF-IDF and cosine similarity.")

BASE = Path(__file__).parent
MOVIES_FILE = BASE / "tmdb_5000_movies.csv"
CREDITS_FILE = BASE / "tmdb_5000_credits.csv"

@st.cache_data(show_spinner=False)
def load_model_data():
    missing = [p.name for p in [MOVIES_FILE, CREDITS_FILE] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset file(s): " + ", ".join(missing) +
                                ". Put both CSV files next to app.py.")
    movies = pd.read_csv(MOVIES_FILE)
    credits = pd.read_csv(CREDITS_FILE)
    key = "movie_id" if "movie_id" in credits.columns else "id"
    data = movies.merge(credits, left_on="id", right_on=key, how="left", suffixes=("", "_credits"))
    title_col = "title" if "title" in data.columns else "title_x"
    data = data.rename(columns={title_col: "title"})
    cols = [c for c in ["title", "genres", "keywords", "overview", "cast", "crew"] if c in data.columns]
    data = data[cols].copy()
    data["title"] = data["title"].fillna("Unknown").astype(str)

    def parse_list(value):
        if not isinstance(value, str) or not value.strip():
            return []
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
        if not isinstance(parsed, list):
            return []
        return [str(x["name"]).replace(" ", "") for x in parsed
                if isinstance(x, dict) and x.get("name")]

    for col in ["genres", "keywords"]:
        if col in data.columns:
            data[col] = data[col].apply(parse_list)
    if "cast" in data.columns:
        data["cast"] = data["cast"].apply(lambda x: parse_list(x)[:3])
    if "crew" in data.columns:
        def directors(value):
            if not isinstance(value, str) or not value.strip():
                return []
            try:
                crew = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return []
            return [str(p["name"]).replace(" ", "") for p in crew
                    if isinstance(p, dict) and p.get("job") == "Director" and p.get("name")]
        data["crew"] = data["crew"].apply(directors)
    data["overview"] = data["overview"].fillna("").astype(str).apply(str.split) if "overview" in data.columns else [[] for _ in range(len(data))]
    feature_cols = [c for c in ["overview", "genres", "keywords", "cast", "crew"] if c in data.columns]
    data["tags"] = data[feature_cols].apply(
        lambda row: " ".join(" ".join(v) if isinstance(v, list) else str(v) for v in row), axis=1
    ).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    tfidf = TfidfVectorizer(stop_words="english", max_features=50000)
    vectors = tfidf.fit_transform(data["tags"].fillna(""))
    similarity = cosine_similarity(vectors, dense_output=False)
    return data, similarity

def recommend(title, data, similarity, n):
    found = data.index[data["title"].str.casefold() == title.casefold()].tolist()
    if not found:
        found = data.index[data["title"].str.casefold().str.contains(title.casefold(), regex=False, na=False)].tolist()
    if not found:
        return None, []
    idx = found[0]
    scores = similarity[idx].toarray().ravel()
    order = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return data.iloc[idx]["title"], [(data.iloc[i]["title"], float(score)) for i, score in order if i != idx][:n]

try:
    with st.spinner("Loading dataset and building model..."):
        data, similarity = load_model_data()
except Exception as e:
    st.error(str(e))
    st.info("Required files: tmdb_5000_movies.csv and tmdb_5000_credits.csv")
    st.stop()

a, b, c = st.columns(3)
a.metric("Movies loaded", f"{len(data):,}")
b.metric("Recommendation type", "Content-based")
c.metric("Algorithm", "TF-IDF + cosine similarity")
titles = sorted(data["title"].dropna().unique().tolist(), key=str.casefold)
default_idx = titles.index("Avatar") if "Avatar" in titles else 0
selected = st.selectbox("Select a movie", titles, index=default_idx)
n = st.slider("Number of recommendations", 3, 10, 5)
if st.button("Recommend movies", type="primary"):
    chosen, recs = recommend(selected, data, similarity, n)
    if not recs:
        st.warning("No recommendations found. Try another movie.")
    else:
        st.success(f"Movies similar to {chosen}")
        for rank, (movie, score) in enumerate(recs, 1):
            st.markdown(f"**{rank}. {movie}** — similarity score: `{score:.3f}`")
            st.progress(max(0.0, min(1.0, score)))

with st.expander("How this system works"):
    st.markdown("""
    1. Load and merge the two TMDB CSV files.
    2. Parse genres, keywords, cast and director; handle missing overview values.
    3. Combine features into a movie text profile called `tags`.
    4. Convert text into numerical features with TF-IDF.
    5. Compare movies using cosine similarity.
    6. Return the top-N similar movies, excluding the selected movie.

    This is a content-based recommender. Similarity scores measure metadata similarity, not movie quality.
    """)
st.caption("College ML mini-project • Python • Pandas • Scikit-learn • Streamlit")
