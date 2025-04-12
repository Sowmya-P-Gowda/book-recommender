import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load all required files
popular_books_test = pickle.load(open("popular_books_test.pkl", "rb"))
cosine_sim_test = pickle.load(open("cosine_sim_test.pkl", "rb"))
tfidf_matrix_test = pickle.load(open("tfidf_matrix_test.pkl", "rb"))
user_item_matrix_test = pickle.load(open("user_item_matrix_test.pkl", "rb"))
collab_similarity_df_test = pickle.load(open("collab_similarity_df_test.pkl", "rb"))
test_data = pickle.load(open("test_data.pkl", "rb"))
books = pickle.load(open("books.pkl", "rb"))

# Preprocess
unique_books_test = popular_books_test.drop_duplicates(subset='Book-Title', keep='first').reset_index(drop=True)

# Hybrid Recommender Function for Streamlit
def hybrid_recommend_test(book_title_test, number=5):
    book_title_test = book_title_test.strip()
    matches_test = unique_books_test[unique_books_test['Book-Title'] == book_title_test]

    if matches_test.empty:
        return [], f"❌ Book titled '{book_title_test}' not found.", None

    idx = matches_test.index[0]
    input_book_test = unique_books_test.iloc[idx]

    # -------- Content-Based Filtering --------
    sim_scores_test = list(enumerate(cosine_sim_test[idx]))
    sim_scores_test = sorted(sim_scores_test, key=lambda x: x[1], reverse=True)

    content_dict_test = {}
    for i, score in sim_scores_test:
        isbn = unique_books_test.iloc[i]['ISBN']
        if isbn != input_book_test['ISBN']:
            content_dict_test[isbn] = score

    # -------- Collaborative Filtering --------
    user_ids_who_rated_test = test_data[test_data['ISBN'] == input_book_test['ISBN']]['User-ID'].values

    collab_applicable_test = False
    collaborative_dict_test = {}

    for user_id_test in user_ids_who_rated_test:
        if user_id_test in collab_similarity_df_test.index:
            similar_users = collab_similarity_df_test.loc[user_id_test].sort_values(ascending=False)[1:11]
            collab_applicable_test = True

            for sim_user in similar_users.index:
                if sim_user in user_item_matrix_test.index:
                    user_ratings = user_item_matrix_test.loc[sim_user]
                    rated_books = user_ratings[user_ratings > 0].sort_values(ascending=False)

                    for isbn, rating in rated_books.items():
                        collaborative_dict_test[isbn] = collaborative_dict_test.get(isbn, 0) + rating
            break

    if not collab_applicable_test:
        final_scores_test = content_dict_test
        evaluation = None
    else:
        max_content = max(content_dict_test.values()) if content_dict_test else 1
        max_collab = max(collaborative_dict_test.values()) if collaborative_dict_test else 1

        final_scores_test = {}
        for isbn in set(content_dict_test.keys()).union(collaborative_dict_test.keys()):
            content_score = content_dict_test.get(isbn, 0) / max_content
            collab_score = collaborative_dict_test.get(isbn, 0) / max_collab
            final_scores_test[isbn] = 0.5 * content_score + 0.5 * collab_score

        # Evaluation (only if collab was used)
        liked_books = test_data[(test_data['User-ID'] == user_id_test) & (test_data['Book-Rating'] >= 7)]
        actual_liked_isbns = set(liked_books['ISBN'])

        recommended_isbns = set(final_scores_test.keys())
        relevant_recommendations = recommended_isbns.intersection(actual_liked_isbns)

        precision = len(relevant_recommendations) / len(recommended_isbns) if recommended_isbns else 0
        recall = len(relevant_recommendations) / len(actual_liked_isbns) if actual_liked_isbns else 0
        evaluation = (user_id_test, precision, recall)

    sorted_scores_test = sorted(final_scores_test.items(), key=lambda x: x[1], reverse=True)
    recommended_books_test = [isbn for isbn, _ in sorted_scores_test[:number]]

    results = []
    for isbn in recommended_books_test:
        book_rows_test = books[books['ISBN'] == isbn]
        if not book_rows_test.empty:
            book_info = book_rows_test.iloc[0]
            results.append({
                "Book-Title": book_info['Book-Title'],
                "Book-Author": book_info['Book-Author']
            })
        else:
            results.append({"Book-Title": f"ISBN: {isbn}", "Book-Author": "Unknown"})

    return results, None, evaluation

# ---- Streamlit UI ----
st.set_page_config(page_title="📚 Book Recommender", layout="centered")
st.title("📚 Hybrid Book Recommender")

book_titles = unique_books_test['Book-Title'].dropna().unique().tolist()
book_choice = st.selectbox("Choose a book:", sorted(book_titles))

if st.button("Recommend"):
    with st.spinner("Generating recommendations..."):
        recs, warning, eval_data = hybrid_recommend_test(book_choice)

        if warning:
            st.warning(warning)
        else:
            st.success(f"Top Recommendations based on **{book_choice}**:")
            st.table(pd.DataFrame(recs))

            if eval_data:
                user_id, precision, recall = eval_data
                st.markdown("### 📊 Evaluation ")
                

