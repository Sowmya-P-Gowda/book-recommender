import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# --- Set page configuration (MUST be the first Streamlit command) ---
st.set_page_config(page_title="📚 Book Recommender", layout="centered")

# --- Load Data ---
popular_books_test = pickle.load(open("popular_books_test.pkl", "rb"))
cosine_sim_test = pickle.load(open("cosine_sim_test.pkl", "rb"))
tfidf_matrix_test = pickle.load(open("tfidf_matrix_test.pkl", "rb"))
user_item_matrix_test = pickle.load(open("user_item_matrix_test.pkl", "rb"))
collab_similarity_df_test = pickle.load(open("collab_similarity_df_test.pkl", "rb"))
test_data = pickle.load(open("test_data.pkl", "rb"))
books = pickle.load(open("books.pkl", "rb"))

# --- Simulated User Login ---
st.sidebar.header("📚 Book Recommender Login")
user_ids = test_data['User-ID'].unique().tolist()
selected_user = st.sidebar.selectbox("Select User ID (Simulated Login)", user_ids)

# --- Hybrid Recommendation Function ---
def hybrid_recommend_test(book_title_test, user_id_test=None, number=5):
    book_title_test = book_title_test.strip()
    unique_books_test = popular_books_test.drop_duplicates(subset='Book-Title', keep='first').reset_index(drop=True)
    matches_test = unique_books_test[unique_books_test['Book-Title'] == book_title_test]

    if matches_test.empty:
        st.warning(f"Book titled '{book_title_test}' not found.")
        st.write("\nSome available books:")
        st.write(unique_books_test['Book-Title'].sample(n=min(10, len(unique_books_test))).to_list())
        return

    idx = matches_test.index[0]
    input_book_test = unique_books_test.iloc[idx]
    st.subheader(f"Recommendations based on: {input_book_test['Book-Title']}")

    # --- Content-Based Filtering ---
    sim_scores_test = list(enumerate(cosine_sim_test[idx]))
    sim_scores_test = sorted(sim_scores_test, key=lambda x: x[1], reverse=True)

    content_dict_test = {}
    for i, score in sim_scores_test:
        isbn = unique_books_test.iloc[i]['ISBN']
        if isbn != input_book_test['ISBN']:
            content_dict_test[isbn] = score

    # --- Collaborative Filtering ---
    user_ids_who_rated_test = [user_id_test] if user_id_test else []
    collab_applicable_test = False
    collaborative_dict_test = {}

    for user_id in user_ids_who_rated_test:
        if user_id in collab_similarity_df_test.index:
            similar_users = collab_similarity_df_test.loc[user_id].sort_values(ascending=False)[1:11]
            collab_applicable_test = True

            for sim_user in similar_users.index:
                if sim_user in user_item_matrix_test.index:
                    user_ratings = user_item_matrix_test.loc[sim_user]
                    rated_books = user_ratings[user_ratings > 0].sort_values(ascending=False)

                    for isbn, rating in rated_books.items():
                        collaborative_dict_test[isbn] = collaborative_dict_test.get(isbn, 0) + rating
            break

    if not collab_applicable_test:
        st.info("Collaborative filtering not available for this book. Showing content-based results only.")
        final_scores_test = content_dict_test
    else:
        max_content = max(content_dict_test.values()) if content_dict_test else 1
        max_collab = max(collaborative_dict_test.values()) if collaborative_dict_test else 1

        final_scores_test = {}
        for isbn in set(content_dict_test.keys()).union(collaborative_dict_test.keys()):
            content_score = content_dict_test.get(isbn, 0) / max_content
            collab_score = collaborative_dict_test.get(isbn, 0) / max_collab
            final_scores_test[isbn] = 0.5 * content_score + 0.5 * collab_score

    sorted_scores_test = sorted(final_scores_test.items(), key=lambda x: x[1], reverse=True)
    recommended_books_test = [isbn for isbn, _ in sorted_scores_test[:number]]

    if not recommended_books_test:
        st.warning("No recommendations could be generated.")
        return

    for isbn in recommended_books_test:
        book_rows_test = books[books['ISBN'] == isbn]
        if not book_rows_test.empty:
            book_info = book_rows_test.iloc[0]
            st.write(f"**{book_info['Book-Title']}** by *{book_info['Book-Author']}*")
        else:
            st.write(f"ISBN: {isbn} (No metadata found)")

    # --- Evaluation (Precision & Recall) ---
    liked_books = test_data[(test_data['User-ID'] == user_id_test) & (test_data['Book-Rating'] >= 7)]
    actual_liked_isbns = set(liked_books['ISBN'])

    recommended_isbns = set(recommended_books_test)
    relevant_recommendations = recommended_isbns.intersection(actual_liked_isbns)

    precision = len(relevant_recommendations) / len(recommended_isbns) if recommended_isbns else 0
    recall = len(relevant_recommendations) / len(actual_liked_isbns) if actual_liked_isbns else 0

    st.markdown("---")
    st.subheader(f"📊 Evaluation for User {user_id_test}")
    st.write(f"**Precision@{number}**: {precision:.4f}")
    st.write(f"**Recall@{number}**: {recall:.4f}")

# --- Main Interface ---
st.title("📖 Hybrid Book Recommendation System")

book_choice = st.selectbox("Select a Book to Get Recommendations", popular_books_test['Book-Title'].unique())
top_n = st.slider("How many recommendations do you want?", min_value=1, max_value=10, value=5)

if st.button("Get Recommendations"):
    hybrid_recommend_test(book_choice, user_id_test=selected_user, number=top_n)
