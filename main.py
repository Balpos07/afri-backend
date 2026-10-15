from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from pipeline import MedicalVoicePipeline
from io import BytesIO
from urllib.parse import quote
import os

import learning
from admin_routes import router as admin_router
from db import init_db

class ChatRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    text: str
    helpful: bool
    matched_symptom: str | None = None

app = FastAPI(title="Naija Medical Voice Assistant (Pidgin)")

# Comma-separated list of allowed frontend origins; "*" until ALLOWED_ORIGINS is set
allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

pipeline_instance = None

def reload_pipeline_kb():
    # Nothing to refresh if the pipeline hasn't loaded yet; it reads the DB on init
    if pipeline_instance is not None:
        pipeline_instance.reload_kb()

learning.kb_changed_hook = reload_pipeline_kb

@app.on_event("startup")
def setup_database():
    try:
        init_db()
        seeded = learning.seed_kb()
        if seeded:
            print(f"Seeded knowledge base with {seeded} entries from knowledge_base.json")
    except Exception as e:
        print(f"WARNING: database unavailable ({e}). Learning and admin features are disabled.")

app.include_router(admin_router)

def get_pipeline():
    global pipeline_instance
    if pipeline_instance is None:
        print("Initializing MedicalVoicePipeline for the first time...")
        pipeline_instance = MedicalVoicePipeline()
    return pipeline_instance

@app.post("/voice")
async def voice_assistant(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    pipeline = get_pipeline()
    result = await run_in_threadpool(pipeline.process_voice, audio_bytes)
    
    input_text_clean = result["input_text"].replace("\n", " ").strip()
    response_text_clean = result["response_text"].replace("\n", " ").strip()
    matched_symptom = (result.get("matched_symptom") or "").replace("\n", " ").strip()
    matched_phrase = (result.get("matched_phrase") or "").replace("\n", " ").strip()

    # URL-encode text to safely send Pidgin characters in HTTP headers
    return StreamingResponse(
        BytesIO(result["audio"]),
        media_type="audio/wav",
        headers={
            "X-Input-Text": quote(input_text_clean),
            "X-Response-Text": quote(response_text_clean),
            "X-Matched-Symptom": quote(matched_symptom),
            "X-Matched-Phrase": quote(matched_phrase)
        }
    )

@app.post("/chat")
async def chat_assistant(request: ChatRequest):
    pipeline = get_pipeline()
    result = await run_in_threadpool(pipeline.process_chat, request.text)
    
    input_text_clean = result["input_text"].replace("\n", " ").strip()
    response_text_clean = result["response_text"].replace("\n", " ").strip()
    matched_symptom = (result.get("matched_symptom") or "").replace("\n", " ").strip()
    matched_phrase = (result.get("matched_phrase") or "").replace("\n", " ").strip()

    # Return exact same StreamingResponse and Headers as /voice
    return StreamingResponse(
        BytesIO(result["audio"]),
        media_type="audio/wav",
        headers={
            "X-Input-Text": quote(input_text_clean),
            "X-Response-Text": quote(response_text_clean),
            "X-Matched-Symptom": quote(matched_symptom),
            "X-Matched-Phrase": quote(matched_phrase)
        }
    )

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    await run_in_threadpool(learning.record_feedback, request.text, request.helpful, request.matched_symptom)
    return {"ok": True}

@app.get("/")
def serve_ui():
    return FileResponse("index.html")

@app.get("/status")
def health():
    # Return basic status immediately without waiting for model load
    if pipeline_instance is None:
        return {"status": "starting", "message": "Models are loading or waiting for first request..."}
    return {"status": "running", "device": pipeline_instance.device}