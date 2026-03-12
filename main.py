import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from services import process_upload, get_all_docs, delete_doc, ask

# ============================================
# APP SETUP
# ============================================

app = FastAPI(title="Veba — Intelligent Knowledge Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# MODELS
# ============================================

class QuestionRequest(BaseModel):
    question: str
    mode: str = "docs"

# ============================================
# ROUTES
# ============================================

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed = ["csv", "txt","pdf"]
    ext = file.filename.split(".")[-1].lower()

    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"File type .{ext} not supported. Use CSV or TXT.")

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty.")

    try:
        doc = process_upload(file.filename, content)
        return {
            "message": "File uploaded successfully.",
            "doc": {
                "id": doc["id"],
                "name": doc["name"],
                "type": doc["type"],
                "total_chunks": doc["total_chunks"],
                "uploaded_at": doc["uploaded_at"]
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.get("/documents")
def list_docs():
    docs = get_all_docs()
    return {
        "docs": [
            {
                "id": d["id"],
                "name": d["name"],
                "type": d["type"],
                "total_chunks": d.get("total_chunks", 0),
                "uploaded_at": d["uploaded_at"]
            }
            for d in docs
        ]
    }

@app.delete("/documents/{doc_id}")
def remove_doc(doc_id: str):
    success = delete_doc(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"message": "Document deleted successfully."}

@app.post("/ask")
def ask_question(body: QuestionRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if body.mode not in ["docs", "tables"]:
        raise HTTPException(status_code=400, detail="Mode must be 'docs' or 'tables'.")
    try:
        result = ask(body.question, body.mode)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")