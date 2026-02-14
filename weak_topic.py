from typing import Dict, List, Optional, Tuple, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Weak Topic Detector API")


def detect_weak_topics(results: Dict[str, float], threshold: float = 0.6) -> List[str]:
    return [topic for topic, score in results.items() if score < threshold]


def parse_raw_lines(raw: str) -> Tuple[Dict[str, float], List[str]]:
    """
    Parse lines like:
      OSI Model=0.55
      TCP/IP=0.8
    Returns: (results_dict, bad_lines)
    """
    results: Dict[str, float] = {}
    bad_lines: List[str] = []

    lines = [line.strip() for line in (raw or "").splitlines() if line.strip()]
    for line in lines:
        if "=" not in line:
            bad_lines.append(line)
            continue

        topic, score_str = line.split("=", 1)
        topic = topic.strip()
        try:
            score = float(score_str.strip())
            results[topic] = score
        except ValueError:
            bad_lines.append(line)

    return results, bad_lines


class WeakTopicsRequest(BaseModel):
    threshold: float = Field(0.6, ge=0.0, le=1.0)
    # Option A: send dict directly
    results: Optional[Dict[str, float]] = None
    # Option B: send text block
    raw: Optional[str] = None


@app.get("/")
def root():
    return {"message": "Weak Topic Detector API running"}


@app.post("/weak-topics")
def weak_topics(req: WeakTopicsRequest):
    # Decide input source
    results: Dict[str, float] = {}
    bad_lines: List[str] = []

    if req.results and len(req.results) > 0:
        results = req.results
    elif req.raw and req.raw.strip():
        results, bad_lines = parse_raw_lines(req.raw)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'results' (dict) or 'raw' (multiline text).",
        )

    # Validate score range (0..1) like your UI says
    invalid_scores = {k: v for k, v in results.items() if not (0.0 <= v <= 1.0)}
    if invalid_scores:
        raise HTTPException(
            status_code=400,
            detail={"message": "Scores must be between 0 and 1", "invalid_scores": invalid_scores},
        )

    weak = detect_weak_topics(results, threshold=req.threshold)

    return {
        "threshold": req.threshold,
        "all_scores": results,
        "weak_topics": [{"topic": t, "score": results[t]} for t in weak],
        "bad_lines": bad_lines,  # only filled when using raw input
        "counts": {"total_topics": len(results), "weak_count": len(weak)},
    }
