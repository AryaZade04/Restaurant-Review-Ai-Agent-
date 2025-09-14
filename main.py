import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# Simple in-memory retriever for demo
# -----------------------------
def get_reviews(question: str):
    # For demo purposes, these are sample reviews
    sample_reviews = """
    1. The pizza was amazing, especially the Margherita!
    2. Great ambiance, but the service was slow.
    3. Loved the cheese burst pizza, would visit again.
    4. The pepperoni pizza was slightly overcooked, but overall good.
    5. Affordable prices and friendly staff.
    """
    return sample_reviews

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🍕 Restaurant Review AI App")

# Initialize Ollama model
model = OllamaLLM(model="llama3.2")

# Define the prompt template
template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Input box for the user's question
question = st.text_input("Ask a question about the restaurant:")

# Button to get AI answer
if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        reviews = get_reviews(question)  # Fetch reviews
        result = chain.invoke({"reviews": reviews, "question": question})
        st.subheader("AI Answer:")
        st.write(result)
