import streamlit as st
from groq import Groq
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import re

# ============================================
# CONFIGURATION
# ============================================
API_KEY = "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"  # Directly used key (no env file)
os.environ["GROQ_API_KEY"] = API_KEY

client = Groq(api_key=API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = 384

index = faiss.IndexFlatL2(embedding_dim)
documents = []
doc_embeddings = []
chat_history = []


# ============================================
# HELPER FUNCTIONS
# ============================================
def clean_text(text):
    """Cleans and normalizes extracted text"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=500):
    """Splits text into overlapping chunks for better RAG retrieval"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 50):  # 50-word overlap
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def embed_and_index(texts):
    """Converts text chunks to embeddings and stores in FAISS"""
    global documents, doc_embeddings
    embeddings = embedder.encode(texts)
    index.add(np.array(embeddings, dtype="float32"))
    documents.extend(texts)
    doc_embeddings.append(embeddings)

def query_groq(context, query, history):
    """Generates contextual answer using Groq"""
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

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


# ============================================
# STREAMLIT UI
# ============================================
st.set_page_config(page_title="Research Paper Analyzer", page_icon="📘", layout="wide")
st.title("📘 AI-Powered Research Paper Analyzer (RAG + Groq + FAISS)")
st.markdown("Upload PDFs or text and ask deep research questions intelligently.")

with st.sidebar:
    st.header("🧠 Project Overview")
    st.markdown("""
    - Uses **Groq LLM** for fast & smart responses  
    - Employs **RAG pipeline** for knowledge retrieval  
    - Embeds documents using **SentenceTransformer**  
    - Vector search powered by **FAISS**  
    - Persistent **conversation memory**
    """)
    st.divider()
    st.markdown("👩‍💻 *Final Year Project by Sneha Ghosh*")

uploaded_files = st.file_uploader("📄 Upload Research Papers (PDF/Text)", type=["pdf", "txt"], accept_multiple_files=True)
input_text = st.text_area("🧩 Or paste text content below:", height=150)

if st.button("🔍 Process Documents"):
    new_texts = []

    # Extract text from PDF
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(file)
            text = "".join([page.extract_text() for page in pdf_reader.pages])
            text = clean_text(text)
            chunks = chunk_text(text)
            new_texts.extend(chunks)
        elif file.name.endswith(".txt"):
            text = file.read().decode("utf-8")
            chunks = chunk_text(text)
            new_texts.extend(chunks)

    if input_text.strip():
        new_texts.extend(chunk_text(input_text.strip()))

    if new_texts:
        embed_and_index(new_texts)
        st.success(f"✅ {len(new_texts)} text chunks indexed successfully!")
    else:
        st.warning("No valid text found in uploaded files or input.")

# ============================================
# QUESTION ANSWERING
# ============================================
query = st.text_input("💬 Ask a question about your research paper:")

if st.button("🚀 Get Answer"):
    if not documents:
        st.warning("Please process documents first!")
    elif not query.strip():
        st.warning("Enter a question to analyze!")
    else:
        # Find top relevant chunks
        query_embedding = embedder.encode([query])
        D, I = index.search(np.array(query_embedding, dtype="float32"), k=3)
        retrieved_docs = [documents[i] for i in I[0]]
        context = "\n".join(retrieved_docs)

        # Generate Groq answer
        answer = query_groq(context, query, chat_history)

        st.subheader("🧠 Answer:")
        st.write(answer)

        # Save chat memory
        chat_history.append((query, answer))

# ============================================
# MEMORY + INSIGHT SUMMARY
# ============================================
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

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": summary_prompt}],
        )
        summary = response.choices[0].message.content
        st.subheader("🔍 Summary of Research Insights:")
        st.write(summary)
