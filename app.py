import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")

# Dark glass style
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f172a,#020617);
    color:white;
}
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding:20px;
    border-radius:15px;
}
.title {
    text-align:center;
    font-size:35px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("movies.csv")

# Convert text to vectors
cv = CountVectorizer()
vectors = cv.fit_transform(df["genre"]).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)

# -----------------------------
# RECOMMEND FUNCTION
# -----------------------------
def recommend(movie):
    index = df[df["title"] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]
    return [df.iloc[i[0]].title for i in movies_list]

# -----------------------------
# UI
# -----------------------------
st.markdown('<div class="title">🎬 AI Movie Recommendation System</div>', unsafe_allow_html=True)
st.write("")

st.markdown('<div class="glass">', unsafe_allow_html=True)

selected_movie = st.selectbox(
    "Choose a movie you like:",
    df["title"].values
)

if st.button("✨ Recommend Movies"):

    recommendations = recommend(selected_movie)

    st.write("### 🍿 Recommended Movies:")
    for movie in recommendations:
        st.success(movie)

st.markdown('</div>', unsafe_allow_html=True)
