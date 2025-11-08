import streamlit as st
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import re
import os

# =========================================================
# 🔧 CONFIGURATION
# =========================================================
API_KEY = "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"  # ✅ Using your provided key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

PROJECT_INFO = """
- Uses **Groq LLM** for intelligent summarization  
- Employs **RAG pipeline** for contextual retrieval  
- Embeds documents using **SentenceTransformer**  
- Vector search powered by **FAISS**  
- Persistent **conversation memory**
"""

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Avoid torch duplicate errors

# =========================================================
# ⚙️ INITIALIZATION
# =========================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # ✅ Force CPU

embedder = load_embedder()
embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)
documents = []
chat_history = []

# =========================================================
# 🧩 HELPER FUNCTIONS
# =========================================================
def clean_text(text):
    """Cleans and normalizes extracted text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=500):
    """Splits text into overlapping chunks for better retrieval."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 50):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def extract_text_from_file(file):
    """Extracts text from PDF or TXT file."""
    if file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        return text
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    return ""

def embed_and_index(texts):
    """Converts text chunks to embeddings and stores in FAISS."""
    global documents
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    index.add(np.array(embeddings, dtype="float32"))
    documents.extend(texts)

def search_documents(query, k=3):
    """Finds top-k similar document chunks for a query."""
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(np.array(query_embedding, dtype="float32"), k)
    retrieved_docs = [documents[i] for i in I[0] if i < len(documents)]
    return "\n".join(retrieved_docs)

def query_groq(context, query, history):
    """Generates contextual answer using Groq API."""
    conversation = "\n".join([f"User: {q}\nAI: {a}" for q, a in history[-3:]])
    prompt = f"""
    You are an intelligent research assistant analyzing academic papers.

    Context from relevant documents:
    {context}

    Previous conversation:
    {conversation}

    Question:
    {query}

    Provide a clear, concise, and well-reasoned answer with research insight.
    """

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Groq API Error: {e}"

# =========================================================
# 🖥️ STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Research Paper Analyzer", page_icon="📘", layout="wide")
st.title("📘 AI-Powered Research Paper Analyzer (RAG + Groq + FAISS)")
st.markdown("Upload your research papers or paste text and ask smart research questions!")

with st.sidebar:
    st.header("🧠 Project Overview")
    st.markdown(PROJECT_INFO)
    st.divider()
    st.markdown("👩‍💻 *Final Year Project by Sneha Ghosh*")

# =========================================================
# 📄 DOCUMENT UPLOAD + PROCESSING
# =========================================================
uploaded_files = st.file_uploader("📄 Upload Research Papers (PDF/Text)", type=["pdf", "txt"], accept_multiple_files=True)
input_text = st.text_area("🧩 Or paste text content below:", height=150)

if st.button("🔍 Process Documents"):
    new_texts = []

    for file in uploaded_files:
        text = extract_text_from_file(file)
        if text:
            text = clean_text(text)
            chunks = chunk_text(text)
            new_texts.extend(chunks)

    if input_text.strip():
        new_texts.extend(chunk_text(input_text.strip()))

    if new_texts:
        embed_and_index(new_texts)
        st.success(f"✅ {len(new_texts)} text chunks indexed successfully!")
    else:
        st.warning("⚠️ No valid text found in uploaded files or input.")

# =========================================================
# 💬 QUESTION ANSWERING
# =========================================================
query = st.text_input("💬 Ask a question about your research paper:")

if st.button("🚀 Get Answer"):
    if not documents:
        st.warning("Please process documents first!")
    elif not query.strip():
        st.warning("Enter a question to analyze!")
    else:
        context = search_documents(query)
        answer = query_groq(context, query, chat_history)
        st.subheader("🧠 Answer:")
        st.write(answer)
        chat_history.append((query, answer))

# =========================================================
# 🧠 MEMORY + SUMMARY
# =========================================================
if chat_history:
    st.markdown("### 🗂️ Conversation Memory:")
    for i, (q, a) in enumerate(chat_history):
        st.markdown(f"**Q{i+1}:** {q}")
        st.markdown(f"**A{i+1}:** {a}")

    if st.button("📊 Summarize Insights"):
        full_convo = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])
        summary_prompt = f"""
        Summarize the key research insights and findings discussed below into concise bullet points:
        {full_convo}
        """
        summary = query_groq("", summary_prompt, [])
        st.subheader("🔍 Summary of Research Insights:")
        st.write(summary)
