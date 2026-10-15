from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from pipeline import MedicalVoicePipeline
from io import BytesIO
from urllib.parse import quote
import os

import config
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

# Bump this when deploying so the startup log shows which build is running
APP_VERSION = "learning-v1"

def log_startup_report(db_ok, db_error=None):
    """Print one block to the container log: build, routes, database and config."""
    lines = ["", "=" * 60, f"AFRI BACKEND STARTED  (build: {APP_VERSION})", "=" * 60]

    lines.append("Endpoints loaded:")
    routes = sorted(
        (route.path, ",".join(sorted(m for m in route.methods if m not in ("HEAD", "OPTIONS"))))
        for route in app.routes
        if getattr(route, "methods", None) and route.path not in ("/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc")
    )
    for path, methods in routes:
        lines.append(f"  {methods:<8} {path}")
    admin_count = sum(1 for path, _ in routes if path.startswith("/admin"))
    lines.append(f"  -> {len(routes)} endpoints ({admin_count} admin)")
    lines.append(f"  /admin routes: {'LOADED' if admin_count else 'MISSING'}   /feedback: "
                 f"{'LOADED' if any(p == '/feedback' for p, _ in routes) else 'MISSING'}")

    lines.append("Database:")
    if db_ok:
        try:
            from sqlalchemy import func, select
            from db import Admin, Candidate, KBEntry, SessionLocal, engine
            with SessionLocal() as s:
                lines.append(f"  connected to {engine.url.render_as_string(hide_password=True)}")
                lines.append(f"  admins: {s.scalar(select(func.count()).select_from(Admin))}, "
                             f"KB entries: {s.scalar(select(func.count()).select_from(KBEntry))}, "
                             f"pending candidates: "
                             f"{s.scalar(select(func.count()).select_from(Candidate).where(Candidate.status == 'pending'))}")
        except Exception as e:
            lines.append(f"  connected, but could not read tables: {e}")
    else:
        lines.append(f"  NOT AVAILABLE: {db_error}")

    # Names only: values are never printed
    lines.append("Configuration:")
    for name in ("DATABASE_URL", "JWT_SECRET", "YARNGPT_API_KEY", "ALLOWED_ORIGINS"):
        source = "environment" if os.getenv(name) else "config.py default"
        lines.append(f"  {name}: {'set (' + source + ')' if config.get(name) else 'NOT SET'}")
    lines.append("=" * 60)
    print("\n".join(lines), flush=True)

@app.on_event("startup")
def setup_database():
    db_ok, db_error = True, None
    try:
        init_db()
        seeded = learning.seed_kb()
        if seeded:
            print(f"Seeded knowledge base with {seeded} entries from {learning.KB_JSON.name}", flush=True)
    except Exception as e:
        db_ok, db_error = False, e
        print(f"WARNING: database unavailable ({e}). Learning and admin features are disabled.", flush=True)
    log_startup_report(db_ok, db_error)

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