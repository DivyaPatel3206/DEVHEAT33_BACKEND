from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util

app = FastAPI(title="AI Semantic Answer Evaluator")

# -----------------------------
# Load model once (cached globally)
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Schemas
# -----------------------------
class EvaluateRequest(BaseModel):
    correct_answer: str = Field(..., min_length=1)
    student_answer: str = Field(..., min_length=1)


class EvaluateResponse(BaseModel):
    similarity: float          # 0 to 1
    percentage: float          # 0 to 100
    verdict: str               # Excellent/Good/Partial/Needs Improvement


# -----------------------------
# Core logic
# -----------------------------
def evaluate_answer(student_answer: str, correct_answer: str) -> float:
    emb1 = model.encode(student_answer, convert_to_tensor=True)
    emb2 = model.encode(correct_answer, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2)
    # score is a 1x1 tensor
    return float(score.item())


def classify(score: float) -> str:
    if score > 0.85:
        return "Excellent Match ✅"
    elif score > 0.65:
        return "Good Answer 👍"
    elif score > 0.40:
        return "Partially Correct ⚠️"
    else:
        return "Needs Improvement ❌"


# -----------------------------
# Endpoints
# -----------------
