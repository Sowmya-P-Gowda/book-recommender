import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load preprocessed test data
popular_books = pickle.load(open("popular_books_test.pkl", "rb"))
cosine_sim = pickle.load(open("cosine_sim_test.pkl", "rb"))
tfidf_matrix = pickle.load(open("tfidf_matrix_test.pkl", "rb"))
user_item_matrix = pickle.load(open("user_item_matrix_test.pkl", "rb"))
collab_similarity_df = pickle.load(open("collab_similarity_df_test.pkl", "rb"))
test_data = pickle.load(open("test_data.pkl", "rb"))

# Create index mapping for book titles (based on filtered test set)
indices = pd.Series(popular_books.index, index=popular_books['Book-Title']).drop_duplicates()

# Hybrid Recommender Function
def hybrid_recommend(book_title, top_n=5):
    if book_title not in indices:
        return ["❌ Book not found in the recommendation model."]
    
    idx = indices[book_title]

    # Content-based recommendations
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    book_indices = [i[0] for i in sim_scores]
    content_recs = popular_books.iloc[book_indices][['Book-Title', 'Book-Author']]
    content_recs['method'] = 'content'

    # Collaborative filtering recommendations (user-user)
    user_rated_books = test_data[test_data['Book-Title'] == book_title]
    if user_rated_books.empty:
        collab_recs = pd.DataFrame(columns=['Book-Title', 'Book-Author', 'method'])
    else:
        user_id = user_rated_books.iloc[0]['User-ID']
        if user_id in collab_similarity_df.index:
            similar_users = collab_similarity_df[user_id].sort_values(ascending=False)[1:top_n+1].index
            similar_user_books = test_data[test_data['User-ID'].isin(similar_users)]
            similar_user_books = similar_user_books[similar_user_books['Book-Title'] != book_title]
            top_books = similar_user_books.groupby('Book-Title')['Book-Rating'].mean().sort_values(ascending=False).head(top_n).index
            collab_recs = popular_books[popular_books['Book-Title'].isin(top_books)][['Book-Title', 'Book-Author']]
            collab_recs['method'] = 'collaborative'
        else:
            collab_recs = pd.DataFrame(columns=['Book-Title', 'Book-Author', 'method'])

    # Merge recommendations
    all_recs = pd.concat([content_recs, collab_recs]).drop_duplicates(subset='Book-Title').head(top_n)
    return all_recs

# Streamlit Interface
st.set_page_config(page_title="📚 Book Recommender", layout="centered")
st.title("📚 Hybrid Book Recommender App")

# Dropdown of available books
book_list = popular_books['Book-Title'].dropna().unique().tolist()
book_choice = st.selectbox("Choose a book to get recommendations:", sorted(book_list))

# Recommend button
if st.button("Recommend"):
    with st.spinner("Fetching recommendations..."):
        results = hybrid_recommend(book_choice)
        if isinstance(results, list):
            st.warning(results[0])
        else:
            st.success(f"Recommendations based on **{book_choice}**:")
            st.dataframe(results.reset_index(drop=True))
