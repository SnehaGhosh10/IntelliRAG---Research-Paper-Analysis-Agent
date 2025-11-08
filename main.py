import streamlit as st
import os
import re
import numpy as np
import faiss
import PyPDF2
from openai import OpenAI

# =============================================
# CONFIGURATION
# =============================================
API_KEY = "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"  # ✅ Your Gemini / OpenAI-style key
os.environ["OPENAI_API_KEY"] = API_KEY
client = OpenAI(api_key=API_KEY)

# =============================================
# FUNCTIONS
# =============================================

@st.cache_data
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    text = re.sub(r'\s+', ' ', text)
    return text

@st.cache_data
def create_embeddings(text_chunks):
    """Uses OpenAI Embeddings (no torch needed)."""
    embeddings = []
    for chunk in text_chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",  # ✅ Lightweight & free-tier friendly
            input=chunk
        )
        emb = np.array(response.data[0].embedding, dtype="float32")
        embeddings.append(emb)
    return np.vstack(embeddings)

@st.cache_data
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def answer_query(query, chunks, index, embeddings):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_vector = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)

    distances, indices = index.search(query_vector, k=3)
    retrieved_chunks = [chunks[i] for i in indices[0]]

    prompt = (
        "You are a helpful research assistant. Based on the following context from a research paper, "
        "answer the user's query concisely.\n\n"
        f"Context:\n{retrieved_chunks}\n\nQuery: {query}\nAnswer:"
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content

# =============================================
# STREAMLIT APP UI
# =============================================
st.set_page_config(page_title="📘 IntelliRAG: Research Paper Analysis Agent", layout="wide")

st.title("📘 IntelliRAG: Research Paper Analysis Agent")
st.write("Upload a research paper PDF and ask anything — powered by **Gen AI** + **FAISS** retrieval.")

uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting and embedding text..."):
        text = extract_text_from_pdf(uploaded_file)
        text_chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
        embeddings = create_embeddings(text_chunks)
        index = build_faiss_index(embeddings)
    st.success("✅ Paper processed successfully! Ask your questions below:")

    user_query = st.text_input("🔍 Ask a question about the paper:")

    if user_query:
        with st.spinner("Thinking..."):
            answer = answer_query(user_query, text_chunks, index, embeddings)
        st.markdown("### 🧠 Answer:")
        st.write(answer)
