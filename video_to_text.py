import os
import tempfile

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import whisper

app = FastAPI(title="Video to Text AI (Whisper)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
model = whisper.load_model(MODEL_NAME)


# -------------------------
# Home Page
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "v2t.html",
        {
            "request": request,
            "transcript": None,
            "error": None,
            "model_name": MODEL_NAME
        },
    )


# -------------------------
# Transcription (Form Submit)
# -------------------------
@app.post("/transcribe", response_class=HTMLResponse)
async def transcribe_video(request: Request, file: UploadFile = File(...)):

    transcript = None
    error = None
    tmp_path = None

    try:
        if not file.filename:
            error = "No file uploaded."
        else:
            allowed_ext = {".mp4", ".mov", ".avi", ".mkv"}
            _, ext = os.path.splitext(file.filename.lower())

            if ext not in allowed_ext:
                error = f"Unsupported file type {ext}"
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp_path = tmp.name
                    tmp.write(await file.read())

                result = model.transcribe(tmp_path)
                transcript = result.get("text", "")

    except Exception as e:
        error = str(e)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return templates.TemplateResponse(
        "v2t.html",
        {
            "request": request,
            "transcript": transcript,
            "error": error,
            "model_name": MODEL_NAME
        },
    )
