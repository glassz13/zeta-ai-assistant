# Zeta — Intelligent Knowledge Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-F55036?style=for-the-badge)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF?style=for-the-badge)](https://faiss.ai)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Zeta** is an intelligent company knowledge assistant that lets teams query their internal documents and data using plain English — no SQL, no search bars, no manual digging.

>[![Live Demo](https://glassz13-zeta-ai.hf.space) 

---

## What is Zeta?

Most companies drown in documents — HR policies, meeting notes, client CSVs, product reports. Finding anything means hunting through folders or bothering a colleague. Zeta fixes this.

Upload your files. Ask questions. Get answers — with sources.

Zeta uses **RAG (Retrieval-Augmented Generation)** to search TXT and PDF files semantically, and sends structured CSV data directly to the LLM for precise table queries. Two modes, one interface.

---

## Features

- **RAG · v2** — semantic search over TXT and PDF files using FAISS + sentence-transformers
- **Tables · v1** — direct LLM querying over CSV data (employees, clients, inventory)
- **Smart chunking** — heading-based chunking for TXT files, page-based chunking for PDFs
- **Multi-file support** — upload multiple TXT, PDF, and CSV files at once
- **Persistent storage** — files survive server restarts locally (pickle + FAISS index)
- **Source attribution** — every answer cites which file it came from
- **Casual chat handling** — politely redirects off-topic questions back to its purpose
- **File limits** — TXT/PDF max 500KB · CSV max 50 rows
- **Clean UI** — chat interface inspired by modern AI products, zero frameworks
- **Docker ready** — single container deployment

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| LLM | LLaMA 3.1 8B via Groq API |
| Vector Search | FAISS (cosine similarity) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| PDF Parsing | pdfplumber |
| Frontend | Vanilla HTML + CSS + JS |
| Deployment | Docker + HuggingFace Spaces |

---

## How It Works

### RAG Mode (TXT + PDF)
```
Upload TXT  → heading-based chunking (each section = one chunk)
Upload PDF  → page-based chunking (each page = one chunk)
            → embed with sentence-transformers → store in FAISS index
            → question asked → embed question → cosine similarity search
            → top 4 chunks retrieved → sent to LLaMA 3.1 → answer with source
```

### Tables Mode (CSV)
```
Upload CSV → parse rows → store as text chunks
→ question asked → ALL rows sent directly to LLaMA 3.1
→ LLM reads and reasons over full table → precise answer
```

### Why two different approaches?

Semantic search (FAISS) works great for prose — policies, reports, notes. But CSV data has no semantic meaning — `salary: 85000` doesn't embed meaningfully. Sending full rows to the LLM is far more accurate for structured data.

### Why heading-based chunking for TXT?

Word-count chunking splits sections mid-content — a `KEY ACHIEVEMENTS` heading and its bullet points can end up in separate chunks, causing the retrieval to miss them entirely. Heading-based chunking keeps each section intact as one unit, so FAISS finds the right section every time. Plain prose TXT files with no headings automatically fall back to word-count chunking.

### Why page-based chunking for PDF?

PDF pages are natural document boundaries. Page-based chunking preserves the layout and context of each page without splitting mid-paragraph or mid-table.

---

## Project Structure
```
zeta/
├── main.py            ← FastAPI routes
├── services.py        ← RAG logic, FAISS, chunking, file parsing, Groq
├── index.html         ← Full frontend (single file)
├── requirements.txt
├── Dockerfile
├── demo_files/        ← Sample files for local testing
│   ├── hr_policy.txt
│   ├── product_catalog.txt
│   └── employee_data.csv
└── storage/           ← Auto-created on first run (gitignored)
    ├── documents.json
    ├── chunks.pkl
    └── index.faiss
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/zeta.git
cd zeta
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key
Get a free key at [console.groq.com](https://console.groq.com)
```bash
# Mac / Linux
export GROQ_API_KEY=your-gsk-key-here

# Windows CMD
set GROQ_API_KEY=your-gsk-key-here

# Windows PowerShell
$env:GROQ_API_KEY="your-gsk-key-here"
```

### 4. Run
```bash
uvicorn main:app --reload
```

### 5. Open
```
http://localhost:8000
```

Upload any file from `demo_files/` to start testing immediately.

---

## Demo Files

Sample files are included in `demo_files/` to quickly test Zeta locally.

| File | Type | Use with |
|---|---|---|
| `hr_policy.txt` | Company HR policy | RAG · v2 |
| `product_catalog.txt` | Product information | RAG · v2 |
| `employee_data.csv` | Employee records (30 rows) | Tables · v1 |

---

## File Limits

| Type | Limit | Reason |
|---|---|---|
| TXT | 500 KB | Stays within FAISS + token budget |
| PDF | 500 KB | pdfplumber extraction stays fast |
| CSV | 50 rows | Full rows sent to LLM — 50 rows ≈ ~400 tokens |

**Why 50 rows for CSV?** Zeta sends every CSV row to the LLM on every query. This gives the model full context for comparisons and aggregations but has a token cost. 50 rows keeps queries well within Groq's free tier limits.

---

## Docker
```bash
# Build
docker build -t zeta .

# Run
docker run -p 7860:7860 -e GROQ_API_KEY=your-gsk-key-here zeta
```

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## HuggingFace Deployment

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Docker** as the SDK
3. Push this repo to the Space
4. Add `GROQ_API_KEY` in Space Settings → Repository Secrets
5. Space builds and deploys automatically

> **Note:** HuggingFace Spaces free tier does not have persistent disk storage. Uploaded files reset when the Space restarts. Re-upload files from `demo_files/` each session.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve frontend |
| `POST` | `/upload` | Upload TXT, PDF, or CSV |
| `GET` | `/documents` | List uploaded files |
| `DELETE` | `/documents/{id}` | Delete a file |
| `POST` | `/ask` | Ask a question |

### Example
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the maternity leave policy?", "mode": "docs"}'
```

---

## Future Plans

### CSV & Data Analytics Engine
Replace raw row sending with a proper analytics layer:
- **pandas** for in-memory data processing
- **SQL-style queries** generated by the LLM, executed against DataFrames
- **Plotly** charts rendered directly in the chat
- Support for CSVs with 10,000+ rows

### Persistent Database (Supabase)
Replace local pickle/FAISS storage with Supabase:
- **PostgreSQL** for document metadata
- **pgvector** for embeddings (replacing FAISS)
- Files persist across HuggingFace restarts
- Multi-user support with auth

### More Planned
- User authentication and per-team knowledge bases
- Chat history per session
- Cross-file reasoning ("compare Q3 report with Q4 report")
- Slack / Teams integration
- Admin dashboard for file management

---

## Why I Built This

Most RAG tutorials show a single PDF chatbot. Zeta goes further — hybrid retrieval (semantic for prose, direct for structured data), smart chunking strategies per file type, persistent multi-file storage, and a production-grade UI. Built to understand and demonstrate the full RAG pipeline end to end.

---

## License

MIT — use it, fork it, build on it.

---

<p align="center">Built with FastAPI · FAISS · Groq · LLaMA 3.1</p>
