import os
import json
import uuid
import csv
import io
import numpy as np
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import pdfplumber

# ============================================
# CONFIG
# ============================================

STORAGE_DIR = Path("storage")
DOCS_FILE   = STORAGE_DIR / "documents.json"
CHUNKS_FILE = STORAGE_DIR / "chunks.pkl"
FAISS_FILE  = STORAGE_DIR / "index.faiss"

MAX_TXT_SIZE  = 500 * 1024
MAX_PDF_SIZE  = 500 * 1024
MAX_CSV_ROWS  = 50

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================
# STORAGE HELPERS
# ============================================

def init_storage():
    STORAGE_DIR.mkdir(exist_ok=True)
    if not DOCS_FILE.exists():
        DOCS_FILE.write_text("[]")
    if not CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump([], f)

def get_all_docs() -> list:
    init_storage()
    try:
        return json.loads(DOCS_FILE.read_text())
    except:
        return []

def get_chunks() -> list:
    try:
        with open(CHUNKS_FILE, "rb") as f:
            return pickle.load(f)
    except:
        return []

def save_chunks(chunks: list):
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

def load_faiss():
    if FAISS_FILE.exists():
        return faiss.read_index(str(FAISS_FILE))
    return None

def save_faiss(index):
    faiss.write_index(index, str(FAISS_FILE))

# ============================================
# FAISS — TXT + PDF (RAG)
# ============================================

def add_to_faiss(txt_chunks: list):
    if not txt_chunks:
        return
    vectors = embedder.encode(
        [c["text"] for c in txt_chunks],
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype("float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-9)
    index = load_faiss()
    if index is None:
        index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    save_faiss(index)

def rebuild_faiss(remaining_chunks: list):
    txt_chunks = [c for c in remaining_chunks if c["type"] in ["txt", "pdf"]]
    if not txt_chunks:
        if FAISS_FILE.exists():
            FAISS_FILE.unlink()
        return
    vectors = embedder.encode(
        [c["text"] for c in txt_chunks],
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype("float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-9)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    save_faiss(index)

def search_faiss(question: str, top_k: int = 4) -> list[dict]:
    index = load_faiss()
    all_chunks = get_chunks()
    txt_chunks = [c for c in all_chunks if c["type"] in ["txt", "pdf"]]

    if index is None or not txt_chunks:
        return []

    q_vector = embedder.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype("float32")
    norm = np.linalg.norm(q_vector, axis=1, keepdims=True)
    q_vector = q_vector / np.maximum(norm, 1e-9)

    distances, indices = index.search(q_vector, min(top_k, len(txt_chunks)))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(txt_chunks) and dist > 0.20:
            results.append(txt_chunks[idx])
    return results

# ============================================
# FILE PARSERS
# ============================================

def parse_txt(content: bytes) -> list[str]:
    """
    Heading-based chunking.
    Each section (heading + content) = one chunk.
    Falls back to word-count for plain prose.
    """
    text = content.decode("utf-8", errors="ignore").strip()
    chunks = []
    current_section = []

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        is_heading = (
            (stripped == stripped.upper() and len(stripped) > 3 and not stripped.replace(" ", "").isdigit())
            or stripped.startswith("## ")
            or stripped.startswith("# ")
            or stripped.endswith(":")
            or (set(stripped) <= {"-", "=", " "} and len(stripped) > 3)
        )

        if is_heading and current_section:
            chunk = " ".join(current_section).strip()
            if len(chunk) > 30:
                chunks.append(chunk)
            current_section = [stripped]
        else:
            current_section.append(stripped)

    if current_section:
        chunk = " ".join(current_section).strip()
        if len(chunk) > 30:
            chunks.append(chunk)

    # fallback for plain prose with no headings
    if len(chunks) <= 1:
        words = text.split()
        chunks = []
        for i in range(0, len(words), 100):
            chunk = " ".join(words[i:i + 120]).strip()
            if chunk:
                chunks.append(chunk)

    return chunks


def parse_pdf(content: bytes) -> list[str]:
    """
    Page-based chunking.
    Each page = one chunk.
    """
    chunks = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text or len(text.strip()) < 30:
                continue
            chunks.append(text.strip())
    return chunks


def parse_csv(content: bytes) -> list[str]:
    text = content.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        if len(rows) >= MAX_CSV_ROWS:
            break
        row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v and v.strip())
        if row_text.strip():
            rows.append(row_text)
    return rows

# ============================================
# UPLOAD & DELETE
# ============================================

def process_upload(filename: str, content: bytes) -> dict:
    init_storage()
    existing = get_all_docs()

    if any(d["name"] == filename for d in existing):
        raise ValueError(f"'{filename}' already uploaded. Delete it first.")

    ext = filename.split(".")[-1].lower()

    if ext == "txt" and len(content) > MAX_TXT_SIZE:
        raise ValueError("TXT file too large. Maximum size is 500KB.")
    if ext == "pdf" and len(content) > MAX_PDF_SIZE:
        raise ValueError("PDF file too large. Maximum size is 500KB.")

    if ext == "txt":
        chunk_texts = parse_txt(content)
    elif ext == "pdf":
        chunk_texts = parse_pdf(content)
    elif ext == "csv":
        chunk_texts = parse_csv(content)
        if len(chunk_texts) > MAX_CSV_ROWS:
            raise ValueError(f"CSV too large. Maximum {MAX_CSV_ROWS} rows allowed.")
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Use TXT, PDF or CSV.")

    if not chunk_texts:
        raise ValueError("Could not extract content from file.")

    doc_id = str(uuid.uuid4())
    chunk_dicts = [
        {"doc_id": doc_id, "doc_name": filename, "type": ext, "text": text}
        for text in chunk_texts
    ]

    existing_chunks = get_chunks()
    save_chunks(existing_chunks + chunk_dicts)

    if ext in ["txt", "pdf"]:
        add_to_faiss(chunk_dicts)

    doc = {
        "id":           doc_id,
        "name":         filename,
        "type":         ext,
        "total_chunks": len(chunk_dicts),
        "uploaded_at":  datetime.now().isoformat(),
        "size":         len(content)
    }
    existing.append(doc)
    DOCS_FILE.write_text(json.dumps(existing, indent=2))
    return doc


def delete_doc(doc_id: str) -> bool:
    try:
        docs = get_all_docs()
        doc = next((d for d in docs if d["id"] == doc_id), None)
        if not doc:
            return False

        DOCS_FILE.write_text(json.dumps(
            [d for d in docs if d["id"] != doc_id], indent=2
        ))

        remaining = [c for c in get_chunks() if c["doc_id"] != doc_id]
        save_chunks(remaining)

        if doc["type"] in ["txt", "pdf"]:
            rebuild_faiss(remaining)

        return True
    except:
        return False

# ============================================
# AI — GROQ
# ============================================

client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.groq.com/openai/v1"
)

def ask(question: str, mode: str = "docs") -> dict:
    if not get_all_docs():
        return {
            "reply": "Hi! I'm Zeta, your company knowledge assistant. No documents found, please upload some documents for information.",
            "sources": []
        }

    # handle casual/general chat — redirect politely
    casual_keywords = ["how are you", "who are you", "what are you", "hello", "hi", "hey",
                       "good morning", "good evening", "what's up", "thanks", "thank you"]
    if any(kw in question.lower() for kw in casual_keywords):
        return {
            "reply": "Hi! I'm Zeta, your company knowledge assistant. I'm here to help you find information from your uploaded documents and data. What would you like to know?",
            "sources": []
        }

    context = ""
    sources = []

    if mode == "docs":
        txt_results = search_faiss(question, top_k=4)
        if not txt_results:
            return {
                "reply": "I couldn't find anything relevant in the uploaded documents. Try rephrasing your question.",
                "sources": []
            }
        context = "\n\n".join(
            f"[{c['doc_name']}]\n{c['text']}"
            for c in txt_results
        )
        sources = list({c["doc_name"] for c in txt_results})

        system_prompt = f"""You are Zeta, an intelligent company knowledge assistant.
Answer using ONLY the context below. Never guess or make up information.

Rules:
- Be direct and complete — do not cut your answer short
- Include ALL relevant details, lists, bullet points from the context
- Quote exact rules, policies, numbers, and dates when present
- Always mention which document your answer came from
- If the information is genuinely not in the context say: "This information is not in the uploaded documents."
- Never say you "couldn't find" something if the context clearly contains it

CONTEXT:
{context}
"""

    elif mode == "tables":
        all_chunks = get_chunks()
        csv_chunks = [c for c in all_chunks if c["type"] == "csv"]

        if not csv_chunks:
            return {
                "reply": "No table data uploaded yet. Please upload a CSV file first.",
                "sources": []
            }

        csv_by_doc = {}
        for c in csv_chunks:
            csv_by_doc.setdefault(c["doc_name"], []).append(c["text"])

        for doc_name, rows in csv_by_doc.items():
            context += f"[{doc_name}]\n" + "\n".join(rows) + "\n\n"
            sources.append(doc_name)

        system_prompt = f"""You are Zeta, an intelligent company data assistant.
You are searching through structured table data to answer the user's question.

Rules:
- Search through ALL records carefully before answering
- Return ALL matching records with complete details
- For comparisons (highest, lowest, best, worst) — scan every row and compute correctly
- For aggregations (total, average, count) — calculate accurately
- Always mention which file the data came from
- If not found say: "No matching records found in the uploaded tables."

TABLE DATA:
{context}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question}
        ],
        max_tokens=800,
        temperature=0.1
    )

    usage = response.usage
    print(f"Mode: {mode} | Tokens → prompt: {usage.prompt_tokens} | answer: {usage.completion_tokens} | total: {usage.total_tokens}")

    return {
        "reply":   response.choices[0].message.content,
        "sources": sources
    }