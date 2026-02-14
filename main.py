from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
from datetime import datetime

from video_to_text import transcribe_video
from pdf_to_text import extract_pdf_text
from summarizer import generate_notes
from concept_graph import extract_concepts
from quiz_generator import generate_quiz
from evaluator import evaluate_answer
from weak_topic import detect_weak_topics
from spaced_repetition import next_review

# ✅ 1) Create app FIRST
app = FastAPI()

# ✅ 2) Then add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production put your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Pydantic models for JSON body
class TextRequest(BaseModel):
    text: str

class EvaluateRequest(BaseModel):
    student_answer: str
    correct_answer: str

class WeakTopicsRequest(BaseModel):
    results: dict

class ReviewRequest(BaseModel):
    score: float


# 1️⃣ Video to Text
@app.post("/video-to-text/")
async def video_to_text(file: UploadFile = File(...)):
    path = f"temp_{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = transcribe_video(path)
    return {"transcript": text}


# 2️⃣ PDF to Text
@app.post("/pdf-to-text/")
async def pdf_to_text(file: UploadFile = File(...)):
    path = f"temp_{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_pdf_text(path)
    return {"text": text}


# 3️⃣ Generate Notes (JSON body)
@app.post("/generate-notes/")
async def notes(req: TextRequest):
    summary = generate_notes(req.text)
    return {"notes": summary}


# 4️⃣ Concept Graph (JSON body)
@app.post("/concepts/")
async def concepts(req: TextRequest):
    concept_list = extract_concepts(req.text)
    return {"concepts": concept_list}


# 5️⃣ Quiz Generator (JSON body)
@app.post("/generate-quiz/")
async def quiz(req: TextRequest):
    questions = generate_quiz(req.text)
    return {"quiz": questions}


# 6️⃣ Answer Evaluation (JSON body)
@app.post("/evaluate/")
async def evaluate(req: EvaluateRequest):
    score = evaluate_answer(req.student_answer, req.correct_answer)
    return {"similarity_score": score}


# 7️⃣ Weak Topic Detection (JSON body)
@app.post("/weak-topics/")
async def weak(req: WeakTopicsRequest):
    weak_topics = detect_weak_topics(req.results)
    return {"weak_topics": weak_topics}


# 8️⃣ Spaced Repetition (JSON body)
@app.post("/next-review/")
async def review(req: ReviewRequest):
    today = datetime.now()
    next_date = next_review(today, req.score)
    return {"next_review_date": next_date.isoformat()}
