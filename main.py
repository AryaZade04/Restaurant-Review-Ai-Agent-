# main.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("realistic_restaurant_reviews.csv")
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
# OpenAI Setup
# -----------------------------
# Make sure you add your OpenAI API key in Streamlit secrets or as env variable
# st.secrets["OPENAI_API_KEY"] or os.environ["OPENAI_API_KEY"]
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# -----------------------------
# Simple Semantic Search
# -----------------------------
def get_relevant_reviews(question, df, top_k=5):
    """Return top_k most relevant reviews based on cosine similarity."""
    vectorizer = TfidfVectorizer(stop_words='english')
    all_texts = df['Review'].tolist() + [question]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    sim_scores = sim.flatten()
    
    top_indices = sim_scores.argsort()[::-1][:top_k]
    return df.iloc[top_indices]

# -----------------------------
# Generate Answer Using OpenAI
# -----------------------------
def generate_answer(question, df):
    relevant_reviews = get_relevant_reviews(question, df)
    # Combine reviews as context
    context = "\n\n".join(relevant_reviews['Review'].tolist())
    
    prompt = f"""
You are an expert restaurant reviewer. Based on the following customer reviews, answer the question concisely:

Customer Reviews:
{context}

Question: {question}

Answer:
"""
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=250,
            temperature=0.7,
        )
        answer = response['choices'][0]['text'].strip()
    except Exception as e:
        answer = f"Error generating answer: {e}"
    return answer

# -----------------------------
# Display Answer
# -----------------------------
if question:
    with st.spinner("Generating answer..."):
        answer = generate_answer(question, df)
    st.markdown(f"**Answer:** {answer}")


