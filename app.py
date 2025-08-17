import os
import google.generativeai as genai
import gradio as gr
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import shutil

# -----------------------
# 1. Setup Gemini API
# -----------------------
GEMINI_API_KEY = "AIzaSyAvbHvuJ7GbrLxPVcrdKjzVa2ADBrHs_dc"

if not GEMINI_API_KEY:
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")

genai.configure(api_key=GEMINI_API_KEY)

# -----------------------
# 2. Paths for Storage
# -----------------------
DOC_UPLOAD_DIR = r"C:\Users\sonta\Downloads\trag\doc_uploads"
DB_FILE_PATH = r"C:\Users\sonta\Downloads\trag\db\db.pkl"

os.makedirs(DOC_UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_FILE_PATH), exist_ok=True)

# -----------------------
# 3. Embedding Model
# -----------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------
# 4. Vector Store Functions
# -----------------------
def load_or_create_index():
    if os.path.exists(DB_FILE_PATH):
        with open(DB_FILE_PATH, "rb") as f:
            index, texts = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(384)  # 384 dims for MiniLM
        texts = []
    return index, texts

def save_index(index, texts):
    with open(DB_FILE_PATH, "wb") as f:
        pickle.dump((index, texts), f)

def build_vector_store(pdf_path):
    index, texts = load_or_create_index()

    reader = PdfReader(pdf_path)
    new_texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            new_texts.append(text)

    if new_texts:
        embeddings = embedder.encode(new_texts, convert_to_tensor=False)
        index.add(embeddings)
        texts.extend(new_texts)
        save_index(index, texts)
        return f"✅ Added {len(new_texts)} chunks from {os.path.basename(pdf_path)}"
    else:
        return f"⚠️ No text found in {os.path.basename(pdf_path)}"

def search_docs(query, k=3):
    if not os.path.exists(DB_FILE_PATH):
        return "❌ No vector store found. Please upload a document first."

    with open(DB_FILE_PATH, "rb") as f:
        index, texts = pickle.load(f)

    query_emb = embedder.encode([query], convert_to_tensor=False)
    distances, indices = index.search(query_emb, k)
    return [texts[i] for i in indices[0]]

# -----------------------
# 5. RAG Pipeline
# -----------------------
def rag_query(query):
    docs = search_docs(query)
    if isinstance(docs, str):
        return docs

    context = "\n\n".join(docs)
    prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# -----------------------
# 6. Gradio UI
# -----------------------
def upload_and_build(pdf_file):
    saved_pdf_path = os.path.join(DOC_UPLOAD_DIR, os.path.basename(pdf_file.name))
    shutil.copy(pdf_file.name, saved_pdf_path)
    return build_vector_store(saved_pdf_path)

with gr.Blocks() as demo:
    gr.Markdown("# 📚 RAG App with Gemini API ")

    with gr.Tab("1️⃣ Upload & Add to Vector Store"):
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        build_btn = gr.Button("Add to Store")
        build_output = gr.Textbox(label="Status")
        build_btn.click(fn=upload_and_build, inputs=pdf_input, outputs=build_output)

    with gr.Tab("2️⃣ Ask Questions"):
        query_input = gr.Textbox(label="Enter your question")
        query_btn = gr.Button("Ask")
        query_output = gr.Textbox(label="Answer")
        query_btn.click(fn=rag_query, inputs=query_input, outputs=query_output)

demo.launch()