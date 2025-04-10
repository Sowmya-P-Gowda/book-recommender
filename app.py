import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load saved models and data
popular_df = pickle.load(open('popular_books_test.pkl', 'rb'))
cosine_sim = pickle.load(open('cosine_sim_test.pkl', 'rb'))
tfidf_matrix = pickle.load(open('tfidf_matrix_test.pkl', 'rb'))
user_item_matrix = pickle.load(open('user_item_matrix_test.pkl', 'rb'))
collab_sim_df = pickle.load(open('collab_similarity_df_test.pkl', 'rb'))
ratings = pickle.load(open('test_data.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))

# -------------------- HYBRID RECOMMENDER --------------------
def hybrid_recommend(book_title, top_n=5):
    # Content-based filtering
    if book_title not in books['title'].values:
        return []

    idx = books[books['title'] == book_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    content_indices = [i[0] for i in sim_scores]
    content_recs = books.iloc[content_indices][['title', 'author']].copy()
    content_recs['method'] = 'Content'

    # Collaborative filtering
    if book_title in collab_sim_df.columns:
        collab_scores = collab_sim_df[book_title].sort_values(ascending=False)[1:top_n + 1]
        collab_titles = collab_scores.index.tolist()
        collab_recs = books[books['title'].isin(collab_titles)][['title', 'author']].copy()
        collab_recs['method'] = 'Collaborative'
    else:
        collab_recs = pd.DataFrame(columns=['title', 'author', 'method'])

    # Combine and return
    combined = pd.concat([content_recs, collab_recs]).drop_duplicates(subset='title').head(top_n)
    return combined

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="📚 Book Recommender", layout="wide")
st.title("📚 Hybrid Book Recommendation System")

book_list = books['title'].dropna().unique().tolist()
book_choice = st.selectbox("Select a Book Title", sorted(book_list))

if st.button("Recommend"):
    with st.spinner("Fetching recommendations..."):
        recommendations = hybrid_recommend(book_choice)
        if not recommendations.empty:
            st.subheader("🔍 Top Recommendations:")
            st.table(recommendations)
        else:
            st.warning("No recommendations found. Try a different book.")