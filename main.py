from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.concurrency import run_in_threadpool
from pipeline import MedicalVoicePipeline
from io import BytesIO
from urllib.parse import quote

app = FastAPI(title="Naija Medical Voice Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Allows all origins (good for testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

pipeline_instance = None

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

    # URL-encode text to safely send Yoruba characters in HTTP headers
    return StreamingResponse(
        BytesIO(result["audio"]),
        media_type="audio/wav",
        headers={
            "X-Input-Text": quote(input_text_clean),
            "X-Response-Text": quote(response_text_clean)
        }
    )

@app.get("/")
def serve_ui():
    return FileResponse("index.html")

@app.get("/status")
def health():
    # Return basic status immediately without waiting for model load
    if pipeline_instance is None:
        return {"status": "starting", "message": "Models are loading or waiting for first request..."}
    return {"status": "running", "device": pipeline_instance.device}