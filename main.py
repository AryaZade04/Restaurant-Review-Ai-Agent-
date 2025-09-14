# main.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("restaurant_reviews.csv")
    df['Review'] = df['Review'].astype(str)
    return df

df = load_data()

st.title("🍕 Restaurant Review AI App")
st.write("Ask any question about the restaurant, pizzas, or customer experience:")

# -----------------------------
# User Input
# -----------------------------
question = st.text_input("Your question:")

# -----------------------------
# Simple Semantic Search
# -----------------------------
def get_relevant_reviews(question, df, top_k=5):
    """Return top_k most relevant reviews based on cosine similarity."""
    vectorizer = TfidfVectorizer(stop_words='english')
    all_texts = df['Review'].tolist() + [question]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # similarity between question and all reviews
    sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    sim_scores = sim.flatten()
    
    top_indices = sim_scores.argsort()[::-1][:top_k]
    return df.iloc[top_indices], sim_scores[top_indices]

# -----------------------------
# Generate Answer
# -----------------------------
def generate_answer(question, df):
    relevant_reviews, scores = get_relevant_reviews(question, df)
    answer = "Based on customer reviews:\n\n"
    for idx, row in relevant_reviews.iterrows():
        answer += f"- {row['Title']} ({row['Rating']}/5): {row['Review'][:200]}...\n"
    return answer

# -----------------------------
# Display Answer
# -----------------------------
if question:
    answer = generate_answer(question, df)
    st.markdown(answer)
