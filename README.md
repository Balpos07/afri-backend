# AfriHealth - Naija Medical Voice Assistant

A multilingual medical voice assistant powered by AI that provides health recommendations in **Yoruba** language. Users can speak their symptoms, and the system transcribes, analyzes, and responds with medical guidance using text-to-speech.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Data Flow](#data-flow)
3. [Architecture](#architecture)
4. [Libraries & Models](#libraries--models)
5. [Installation](#installation)
6. [API Endpoints](#api-endpoints)
7. [Performance Metrics](#performance-metrics)
8. [Project Structure](#project-structure)

---

## 🎯 Project Overview

**AfriHealth** is a FastAPI-based medical voice assistant designed to provide healthcare recommendations in Yoruba (Nigerian language). The system bridges the gap between modern AI technology and African language speakers by offering a completely voice-based medical consultation interface.

### Key Features:
- 🗣️ **Voice Input Processing**: Accepts audio files for transcription
- 🔄 **Real-time Processing**: Three-stage pipeline with performance tracking
- 🌍 **Multilingual Support**: Yoruba language focus with symptom matching
- 🏥 **Knowledge-Based Recommendations**: Symptom-to-recommendation matching
- 🎙️ **Voice Output**: Text-to-speech responses in Yoruba
- 📊 **Performance Monitoring**: Detailed timing for each stage

---

## 📊 Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER INPUT (Audio File)                    │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  STAGE 1: TRANSCRIPTION    │
        │  (Yoruba ASR Model)        │
        │  - Load audio (librosa)    │
        │  - Convert to 16kHz        │
        │  - ASR inference           │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │    INPUT TEXT (Yoruba)     │
        │  "Ara ń yan mi" (itching)  │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  STAGE 2: KNOWLEDGE LOOKUP     │
        │  - Text cleanup                │
        │  - Exact matching              │
        │  - Fuzzy substring matching    │
        │  - Threshold check (≥60%)      │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  RECOMMENDATION TEXT (Yoruba)  │
        │  From knowledge_base.json      │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  STAGE 3: TEXT-TO-SPEECH       │
        │  - YarnGPT API Call            │
        │  - Voice: "Idera"              │
        │  - Format: WAV @ 24kHz         │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │   AUDIO RESPONSE (WAV File)    │
        │   + Response Metadata          │
        └────────────────────────────────┘
```

### Pipeline Stages:

**STAGE 1: Automatic Speech Recognition (ASR)**
- Input: Audio bytes (various formats)
- Process: Audio preprocessing (16kHz mono), transformer inference
- Output: Transcribed text in Yoruba
- Time: ~1-3 seconds

**STAGE 2: Knowledge Matching**
- Input: Transcribed text (Yoruba)
- Process: 
  1. Text cleanup (remove punctuation)
  2. Symptom lookup from `knowledge_base.json`
  3. Exact matching first (fast path)
  4. Fuzzy substring matching with SequenceMatcher (similarity threshold ≥60%)
- Output: Medical recommendation in Yoruba or fallback message
- Time: ~0.1-0.5 seconds

**STAGE 3: Text-to-Speech (TTS)**
- Input: Recommendation text (Yoruba)
- Process: External API call to YarnGPT with speaker "Idera"
- Output: WAV audio file (24kHz)
- Time: ~2-5 seconds

---

## 🏗️ Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Web Server                       │
│                  (Naija Medical Voice)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐        │
│  │  /voice Endpoint │         │  /status         │        │
│  │  (POST)          │         │  /health         │        │
│  │  File Upload     │         │  Check pipeline  │        │
│  └────────┬─────────┘         └──────────────────┘        │
│           │                                                │
│           ▼                                                │
│  ┌──────────────────────────────────────────────┐        │
│  │   MedicalVoicePipeline (Singleton)          │        │
│  ├──────────────────────────────────────────────┤        │
│  │  • ASR Model (NCAIR1/Yoruba-ASR)            │        │
│  │  • Knowledge Base (JSON)                     │        │
│  │  • Device Management (CUDA/CPU)             │        │
│  └──────────────────────────────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Classes

#### `MedicalVoicePipeline`
**Location**: `pipeline.py`

**Responsibilities**:
- Initialize all ML models and knowledge base on startup
- Manage GPU/CPU device selection
- Execute three-stage processing pipeline

**Key Methods**:
- `__init__()` - Model initialization with performance tracking
- `transcribe(audio_bytes)` - ASR transcription
- `generate_response(user_text)` - Knowledge base matching
- `text_to_speech(text)` - API-based TTS
- `process_voice(audio_bytes)` - Main orchestration method
- `fuzzy_substring_score()` - Similarity scoring

#### FastAPI Application
**Location**: `main.py`

**Routes**:
- `POST /voice` - Main voice processing endpoint
- `GET /` - Serves UI (index.html)
- `GET /status` - Health check endpoint

**Middleware**:
- CORS enabled for all origins (development mode)

---

## 📚 Libraries & Models

### Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| **fastapi** | 0.109.0 | Web framework |
| **uvicorn** | 0.25.0 | ASGI server |
| **transformers** | 4.36.2 | Hugging Face models (ASR) |
| **torch** | 2.1.2 | Deep learning framework |
| **torchaudio** | 2.1.2 | Audio processing |
| **librosa** | 0.10.1 | Audio feature extraction |
| **soundfile** | ≥0.12.1 | Audio file I/O |
| **requests** | 2.31.0 | HTTP client (TTS API) |
| **numpy** | <2.0.0 | Numerical computing |
| **python-multipart** | 0.0.6 | File upload handling |

### ML Models

#### 1. Automatic Speech Recognition (ASR)
- **Model**: `NCAIR1/Yoruba-ASR`
- **Framework**: Hugging Face Transformers (Wav2Vec2-based)
- **Source**: Hugging Face Hub
- **Language**: Yoruba
- **Input**: Audio (16kHz mono)
- **Output**: Transcribed text
- **Device**: GPU (CUDA) or CPU (with fallback)
- **Cache**: Downloaded locally and cached offline

#### 2. Text-to-Speech (TTS)
- **Service**: YarnGPT API
- **Voice**: "Idera" (Yoruba speaker)
- **Endpoint**: `https://yarngpt.ai/api/v1/tts`
- **Output Format**: WAV @ 24kHz
- **Authentication**: Bearer token (API key)

### Knowledge Base
- **File**: `knowledge_base.json`
- **Format**: JSON array
- **Entries**: Medical symptoms with:
  - English symptom name
  - Yoruba variations (list)
  - Medical recommendations (Yoruba text)
- **Size**: 5+ symptom categories documented

**Example Entry**:
```json
{
  "symptom": "itching",
  "yoruba": ["Ara yíyun", "Ara ń yun mi", "Ara ń yan mi"],
  "recommendation": "Láti tọ́jú ara yíyun, lo búlúù (powder)..."
}
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB+ RAM (16GB recommended for GPU)

### Steps

1. **Clone/Setup Project**
   ```bash
   cd c:\Users\HomePC\Downloads\AfriHealth
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or: source venv/bin/activate  # Linux/Mac
   ```

3. **Install Dependencies**
   ```bash
   pip install -r Requirements.txt
   ```

4. **Download Models** (First Time)
   ```bash
   # Models will auto-download on first run, or pre-download:
   python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='NCAIR1/Yoruba-ASR')"
   ```

5. **Set API Key**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your YarnGPT API key:
     ```
     YARNGPT_API_KEY=your_actual_api_key_here
     ```
   - The app will read from the `YARNGPT_API_KEY` environment variable

6. **Run Server**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

7. **Access UI**
   - Open browser: `http://localhost:8000`
   - Server will start with model initialization (~10-30 seconds on first run)

---

## 🔌 API Endpoints

### 1. Voice Processing
```
POST /voice
```

**Request:**
- Body: `multipart/form-data`
- Field: `file` (audio file - WAV, MP3, etc.)

**Response:**
- Media Type: `audio/wav`
- Headers:
  - `X-Input-Text`: URL-encoded transcription
  - `X-Response-Text`: URL-encoded recommendation
- Body: WAV audio bytes (24kHz)

**Example:**
```bash
curl -X POST "http://localhost:8000/voice" \
  -F "file=@audio.wav"
```

**Response Headers:**
```
X-Input-Text: Ara%20n%20yan%20mi
X-Response-Text: L%C3%A1ti%20t%E1%BB%8D%CC%81j%C3%BA%20ara...
Content-Type: audio/wav
```

### 2. UI Serving
```
GET /
```
Serves `index.html` for web interface.

### 3. Health Check
```
GET /status
```

**Response (Starting):**
```json
{
  "status": "starting",
  "message": "Models are loading or waiting for first request..."
}
```

**Response (Running):**
```json
{
  "status": "running",
  "device": "cuda"  // or "cpu"
}
```

---

## 📊 Performance Metrics

### Initialization Times
```
Stage 1 (ASR Model Load):         ~5-20s (depending on cache)
Stage 2 (Knowledge Base Load):    <1s
────────────────────────────────────────
TOTAL INITIALIZATION:              ~5-20s
```

### Per-Request Processing
```
Stage 1 (Transcription):
  - Audio preprocessing:           ~0.5-1s
  - ASR inference:                 ~1-2s
  - Subtotal:                      ~1.5-3s

Stage 2 (Knowledge Matching):
  - Text cleanup:                  <0.1s
  - Symptom matching:              ~0.05-0.3s
  - Subtotal:                      ~0.1-0.5s

Stage 3 (Text-to-Speech):
  - API call + response:           ~2-5s
  - Subtotal:                      ~2-5s
────────────────────────────────────────
TOTAL REQUEST TIME:                ~3.6-8.5s
```

### Matching Algorithm Performance
- **Exact Match**: O(n) - very fast (immediate return)
- **Fuzzy Match**: O(n*m) where:
  - n = number of knowledge base entries (~5-50)
  - m = average text length (20-100 chars)
- **Threshold**: 60% similarity (tunable)

### Resource Usage
- **GPU Memory**: ~2-4GB (model loaded once)
- **CPU Memory**: ~1-2GB (baseline)
- **Disk**: ~500MB (model cache)

---

## 📁 Project Structure

```
AfriHealth/
├── main.py                    # FastAPI application & routing
├── pipeline.py                # MedicalVoicePipeline class
├── knowledge_base.json        # Symptom-recommendation database
├── Requirements.txt           # Python dependencies
├── README.md                  # This file
└── index.html                 # Web UI (optional)
```

### File Descriptions

#### `main.py` (52 lines)
- FastAPI app initialization
- CORS middleware configuration
- Global pipeline singleton management
- Route handlers for `/voice`, `/`, `/status`
- Uses thread pool to avoid blocking async operations

#### `pipeline.py` (330 lines)
- Core `MedicalVoicePipeline` class
- Three-stage processing pipeline
- ASR transcription with librosa
- Knowledge base matching with fuzzy logic
- TTS API integration
- Performance timing and logging

#### `knowledge_base.json`
- JSON array of symptom entries
- Each entry contains:
  - `symptom`: English identifier
  - `yoruba`: Array of Yoruba variations
  - `recommendation`: Medical advice in Yoruba
- Currently indexed with 5+ symptoms

#### `Requirements.txt`
- Pinned versions for reproducibility
- No version ranges specified (strict dependency locking)

---

## 🔧 Configuration & Customization

### Environment Variables

Create a `.env` file in the project root (use `.env.example` as a template):

```
YARNGPT_API_KEY=your_api_key_here
```

The application will automatically load these variables at startup.

### Model Switching
To use a different ASR model, modify `pipeline.py` line 34:
```python
self.asr = pipeline(
    "automatic-speech-recognition",
    model="YOUR_MODEL_NAME",  # Change here
    device=self.device
)
```

### TTS Voice Selection
Change speaker in `pipeline.py` line 225:
```python
payload = {
    "text": text,
    "voice": "YOUR_VOICE",  # Change "Idera" to another voice
    "response_format": "wav"
}
```

### Matching Threshold
Adjust similarity threshold in `pipeline.py` line 194:
```python
if highest_score >= 0.60:  # Change 0.60 to desired threshold (0-1)
```

### Adding New Symptoms
Add entries to `knowledge_base.json`:
```json
{
  "symptom": "cough",
  "yoruba": ["Sisan", "Ohun iku", "Sisan tutu"],
  "recommendation": "Remedies in Yoruba..."
}
```

---

## 🐛 Debugging & Troubleshooting

### Enable Debug Logging
The pipeline prints detailed timing and matching information to stdout. Check console output for:
- ASR transcription results
- Knowledge base matches and similarity scores
- API call status
- Performance metrics

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Model download fails | No internet/offline mode | Disable `TRANSFORMERS_OFFLINE` or pre-cache models |
| CUDA out of memory | Model too large for GPU | Set device to CPU or increase GPU memory |
| No match found | Low similarity score | Lower threshold or add more symptom variations |
| API key invalid | Expired/incorrect key | Update `api_key` in `pipeline.py` |
| Slow response | Network latency to TTS API | Consider local TTS model alternative |

---

## 🚀 Deployment

### Development
```bash
uvicorn main:app --reload
```

### Production
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

### Docker (Optional)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r Requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

---

## 📄 License & Attribution

- **Yoruba ASR Model**: NCAIR1 (Hugging Face)
- **TTS Service**: YarnGPT API
- **Framework**: FastAPI, PyTorch

---

## 📞 Support & Contribution

For issues or improvements, please document:
1. **Input**: What Yoruba text did you say?
2. **Expected**: What recommendation should it give?
3. **Actual**: What did it actually return?
4. **Similarity**: What was the matching score?

---

## 🎓 Key Learnings

This project demonstrates:
- ✅ **Multi-stage ML pipelines** with performance monitoring
- ✅ **Async FastAPI** patterns for AI services
- ✅ **Language-specific NLP** (Yoruba ASR)
- ✅ **Fuzzy matching algorithms** for fault-tolerant systems
- ✅ **Singleton pattern** for resource management
- ✅ **Production-ready** error handling and timing

---

**Last Updated**: May 27, 2026  
**Status**: Fully Documented ✅
