import streamlit as st
import os
import re
import numpy as np
import faiss
import PyPDF2
import requests

# =============================================
# CONFIGURATION
# =============================================
GROQ_API_KEY = "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"  # ✅ Your Groq API key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_EMBED_URL = "https://api.groq.com/openai/v1/embeddings"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# =============================================
# FUNCTIONS
# =============================================

@st.cache_data
def extract_text_from_pdf(uploaded_file):
    """Extract clean text from PDF."""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return re.sub(r'\s+', ' ', text)

@st.cache_data
def create_embeddings(text_chunks):
    """Create embeddings using Groq's API."""
    embeddings = []
    for chunk in text_chunks:
        data = {
            "model": "text-embedding-3-small",
            "input": chunk
        }
        response = requests.post(GROQ_EMBED_URL, headers=HEADERS, json=data)
        if response.status_code == 200:
            emb = np.array(response.json()["data"][0]["embedding"], dtype="float32")
            embeddings.append(emb)
        else:
            raise Exception(f"Embedding API error: {response.text}")
    return np.vstack(embeddings)

@st.cache_data
def build_faiss_index(embeddings):
    """Build FAISS index for vector search."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def answer_query(query, chunks, index, embeddings):
    """Retrieve top chunks and generate an answer using Groq."""
    # Get query embedding
    data = {"model": "text-embedding-3-small", "input": query}
    response = requests.post(GROQ_EMBED_URL, headers=HEADERS, json=data)
    if response.status_code != 200:
        return f"⚠️ Embedding Error: {response.text}"
    
    query_vector = np.array(response.json()["data"][0]["embedding"], dtype="float32").reshape(1, -1)

    # Retrieve top chunks
    distances, indices = index.search(query_vector, k=3)
    retrieved_chunks = [chunks[i] for i in indices[0]]

    # Prepare LLM prompt
    prompt = (
        "You are a helpful research assistant. Based on the following context from a research paper, "
        "answer the user's query concisely and accurately.\n\n"
        f"Context:\n{retrieved_chunks}\n\nQuery: {query}\nAnswer:"
    )

    # Call Groq LLM (Llama3-70B)
    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }
    llm_response = requests.post(GROQ_API_URL, headers=HEADERS, json=data)

    if llm_response.status_code == 200:
        return llm_response.json()["choices"][0]["message"]["content"]
    else:
        return f"⚠️ LLM Error: {llm_response.text}"

# =============================================
# STREAMLIT APP UI
# =============================================
st.set_page_config(page_title="📘 IntelliRAG: Research Paper Analysis Agent", layout="wide")

st.title("📘 IntelliRAG: Research Paper Analysis Agent")
st.write("Upload a research paper PDF and ask anything — powered by **Groq Llama 3 + FAISS retrieval**.")

uploaded_file = st.file_uploader("📂 Upload a Research Paper (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("🧠 Extracting and embedding text..."):
        text = extract_text_from_pdf(uploaded_file)
        text_chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
        embeddings = create_embeddings(text_chunks)
        index = build_faiss_index(embeddings)
    st.success("✅ Paper processed successfully! Now ask your questions:")

    user_query = st.text_input("🔍 Ask a question about the paper:")

    if user_query:
        with st.spinner("💭 Thinking with Groq..."):
            answer = answer_query(user_query, text_chunks, index, embeddings)
        st.markdown("### 🧠 Answer:")
        st.write(answer)
