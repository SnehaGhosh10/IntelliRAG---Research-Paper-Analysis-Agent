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
    """Extract and clean text from PDF."""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=1500, overlap=200):
    """Split long text into smaller chunks with overlap."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def safe_embed_text(chunk):
    """Safely create embeddings for a text chunk using Groq."""
    data = {
        "model": "text-embedding-3-large",
        "input": chunk[:6000]  # Truncate to prevent token overflow
    }
    response = requests.post(GROQ_EMBED_URL, headers=HEADERS, json=data)
    if response.status_code == 200:
        return np.array(response.json()["data"][0]["embedding"], dtype="float32")
    else:
        return None

@st.cache_data
def create_embeddings(text_chunks):
    """Generate embeddings for all chunks."""
    embeddings = []
    for i, chunk in enumerate(text_chunks):
        emb = safe_embed_text(chunk)
        if emb is not None:
            embeddings.append(emb)
        else:
            st.warning(f"⚠ Skipped chunk {i+1} due to API error.")
    return np.vstack(embeddings)

@st.cache_data
def build_faiss_index(embeddings):
    """Build FAISS index."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def summarize_paper(full_text):
    """Summarize the uploaded research paper using Groq LLM."""
    prompt = f"""
    You are a research summarization expert.
    Summarize the following research paper text in 3 paragraphs:
    - 1st paragraph: Research objective and problem statement.
    - 2nd paragraph: Methodology and major findings.
    - 3rd paragraph: Conclusion and real-world impact.
    
    Keep it concise and factual.
    
    Text:
    {full_text[:8000]}
    """

    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(GROQ_API_URL, headers=HEADERS, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return "⚠ Could not generate summary."

def answer_query(query, chunks, index):
    """Answer user query based on top retrieved chunks."""
    # Embed query
    q_data = {"model": "text-embedding-3-large", "input": query}
    q_response = requests.post(GROQ_EMBED_URL, headers=HEADERS, json=q_data)
    if q_response.status_code != 200:
        return f"⚠ Embedding Error: {q_response.text}"

    q_vector = np.array(q_response.json()["data"][0]["embedding"], dtype="float32").reshape(1, -1)

    # Retrieve top chunks
    distances, indices = index.search(q_vector, k=3)
    context = "\n\n".join([chunks[i] for i in indices[0]])

    # Smart query prompt
    prompt = f"""
    You are an intelligent research assistant.
    Use the context below from a research paper to answer the user question concisely, 
    citing evidence from the context if possible.

    Context:
    {context}

    Question:
    {query}

    Answer clearly, factually, and in 3–4 sentences:
    """

    llm_data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }
    llm_response = requests.post(GROQ_API_URL, headers=HEADERS, json=llm_data)
    if llm_response.status_code == 200:
        return llm_response.json()["choices"][0]["message"]["content"].strip()
    else:
        return f"⚠ LLM Error: {llm_response.text}"

# =============================================
# STREAMLIT APP UI
# =============================================
st.set_page_config(page_title="📘 IntelliRAG: Research Paper Analysis Agent", layout="wide")

st.title("📘 IntelliRAG: Research Paper Analysis Agent")
st.caption("Powered by Groq Llama 3 + FAISS — Upload, Summarize, and Query your Research Paper.")

uploaded_file = st.file_uploader("📂 Upload a Research Paper (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("🧠 Extracting and analyzing your paper..."):
        try:
            text = extract_text_from_pdf(uploaded_file)
            text_chunks = chunk_text(text)
            embeddings = create_embeddings(text_chunks)
            index = build_faiss_index(embeddings)
            st.success("✅ Paper processed successfully!")

            # Auto-generate summary
            st.subheader("📄 Paper Summary")
            summary = summarize_paper(text)
            st.write(summary)

        except Exception as e:
            st.error(f"❌ Error while processing the file: {e}")
            st.stop()

    st.divider()
    st.subheader("💬 Ask a Question About the Paper")
    user_query = st.text_input("Type your question here:")

    if user_query:
        with st.spinner("🤖 Thinking with Groq..."):
            answer = answer_query(user_query, text_chunks, index)
        st.markdown("### 🧠 Answer:")
        st.write(answer)
