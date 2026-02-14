from datetime import date, datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Spaced Repetition Review Scheduler API")


# -----------------------------
# Core Function
# -----------------------------
def next_review(last_review_date: date, score: float) -> date:
    if score > 0.8:
        days_gap = 5
    elif score > 0.6:
        days_gap = 3
    else:
        days_gap = 1
    return last_review_date + timedelta(days=days_gap)


def feedback(score: float) -> str:
    if score > 0.8:
        return "Excellent retention 🧠"
    elif score > 0.6:
        return "Moderate retention 👍"
    else:
        return "Needs reinforcement 📚"


class NextReviewRequest(BaseModel):
    last_review_date: date = Field(..., description="YYYY-MM-DD")
    score_percent: int = Field(..., ge=0, le=100, description="0 to 100")


@app.get("/")
def root():
    return {"message": "Spaced Repetition Scheduler API running"}


@app.post("/next-review")
def compute_next_review(req: NextReviewRequest):
    try:
        score = req.score_percent / 100.0
        nxt = next_review(req.last_review_date, score)

        today = date.today()
        days_left = (nxt - today).days

        if days_left > 0:
            status = f"⏳ Review in {days_left} day(s)"
        elif days_left == 0:
            status = "📖 Review is TODAY!"
        else:
            status = "⚠️ Review date has already passed!"

        return {
            "last_review_date": req.last_review_date.isoformat(),
            "score_percent": req.score_percent,
            "score": round(score, 4),
            "next_review_date": nxt.isoformat(),
            "next_review_human": nxt.strftime("%A, %d %B %Y"),
            "days_left": days_left,
            "status": status,
            "performance_feedback": feedback(score),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate next review: {str(e)}")
