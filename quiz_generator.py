import json
import re
from typing import List, Dict, Any, Optional, Tuple

import torch
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


app = FastAPI(title="MCQ Quiz Coach (Jinja2)")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Load FLAN-T5 once (global cache)
# -----------------------------
MODEL_NAME = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

DIFF_MAP = {"Easy": 0, "Medium": 1, "Hard": 2}
DIFF_INV = {v: k for k, v in DIFF_MAP.items()}

DEFAULT_TEXT = """Networking Devices:
Router, Switch, Hub, Gateway, NIC, Host, Client-Server model.
IP addressing, TCP/UDP, DNS, DHCP, HTTP/HTTPS, firewalls, NAT.
OSI model layers and packet forwarding, routing tables, LAN/WAN.
"""


# -----------------------------
# Recursive difficulty adjust
# -----------------------------
def rec_binary_adjust(low: int, high: int, current: int, score_ratio: float) -> int:
    if score_ratio >= 0.75:
        if current >= high:
            return current
        new_low = current + 1
        next_level = (new_low + high) // 2
        return rec_binary_adjust(new_low, high, next_level, score_ratio) if next_level != current else next_level

    if score_ratio <= 0.40:
        if current <= low:
            return current
        new_high = current - 1
        next_level = (low + new_high) // 2
        return rec_binary_adjust(low, new_high, next_level, score_ratio) if next_level != current else next_level

    return current


# -----------------------------
# JSON helper
# -----------------------------
def extract_json_block(text: str) -> Optional[str]:
    match = re.search(r"\[\s*\{.*?\}\s*\]", text, flags=re.S)
    return match.group(0) if match else None


# -----------------------------
# Non-JSON MCQ parser
# -----------------------------
def parse_mcqs_from_text(raw: str) -> Optional[List[Dict[str, Any]]]:
    raw = (raw or "").strip()
    if not raw:
        return None

    # --- Single-line format
    one = re.sub(r"\s+", " ", raw)
    chunks = re.split(r"(?=Question\s*:)", one, flags=re.I)
    mcqs: List[Dict[str, Any]] = []

    for c in chunks:
        c = c.strip()
        if not c:
            continue

        qmatch = re.search(r"Question\s*:\s*(.+?)\s*Options\s*:\s*", c, flags=re.I)
        if not qmatch:
            continue
        question = qmatch.group(1).strip()

        omatch = re.search(
            r"Options\s*:\s*A\s*(.+?)\s*B\s*(.+?)\s*C\s*(.+?)\s*D\s*(.+?)(?:\s*Answer\s*:|\Z)",
            c, flags=re.I
        )
        if not omatch:
            continue

        opts = {
            "A": omatch.group(1).strip(" ."),
            "B": omatch.group(2).strip(" ."),
            "C": omatch.group(3).strip(" ."),
            "D": omatch.group(4).strip(" ."),
        }
        am = re.search(r"Answer\s*:\s*([ABCD])", c, flags=re.I)
        ans = am.group(1).upper() if am else "A"

        if all(opts.values()):
            mcqs.append({"question": question, "options": opts, "answer": ans})

    if mcqs:
        return mcqs

    # --- Multiline blocks
    txt = raw.replace("\r\n", "\n").replace("\r", "\n")
    txt = txt.replace("Options:", "\nOptions:\n").replace("Answer:", "\nAnswer:")

    blocks = re.split(r"\n\s*(?=Q\d+\s*:)", txt, flags=re.I)
    mcqs = []

    for b in blocks:
        b = b.strip()
        if not b:
            continue

        qm = re.search(r"Q\d+\s*:\s*(.+?)(?=\n\s*A[\)\.])", b, flags=re.I | re.S)
        if not qm:
            continue
        question = re.sub(r"\s+", " ", qm.group(1)).strip()

        def get_opt(letter: str) -> str:
            m = re.search(
                rf"\n\s*{letter}[\)\.]\s*(.+?)(?=\n\s*[ABCD][\)\.]|\n\s*Answer\s*:|\Z)",
                b, flags=re.I | re.S
            )
            return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""

        opts = {k: get_opt(k) for k in ["A", "B", "C", "D"]}
        am = re.search(r"Answer\s*:\s*([ABCD])", b, flags=re.I)
        ans = am.group(1).upper() if am else "A"

        if all(opts.values()):
            mcqs.append({"question": question, "options": opts, "answer": ans})

    return mcqs if mcqs else None


# -----------------------------
# Model generation
# -----------------------------
def generate_raw_output(source_text: str, n: int, difficulty_label: str) -> str:
    difficulty_desc = {
        "Easy": "basic factual questions directly from the text",
        "Medium": "conceptual questions requiring understanding of the text",
        "Hard": "application/tricky distractors, deeper reasoning",
    }[difficulty_label]

    prompt = f"""
Generate EXACTLY {n} multiple-choice questions from the text.

Return ONLY valid JSON (no markdown, no extra words) in this exact schema:

[
  {{
    "question": "....?",
    "options": {{"A":"...","B":"...","C":"...","D":"..."}},
    "answer": "A"
  }}
]

Rules:
- Make options short and clear
- Answer must be one of A/B/C/D
- Difficulty: {difficulty_label} ({difficulty_desc})

TEXT:
{source_text[:1200]}
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=600,
            num_beams=1,          # ✅ important
            do_sample=True,       # ✅ important
            temperature=0.9,      # ✅ helps creativity
            top_p=0.95,
            repetition_penalty=1.15,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def safe_generate_mcqs(source_text: str, n: int, difficulty_label: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    raw = generate_raw_output(source_text, n, difficulty_label)

    # JSON directly
    try:
        data = json.loads(raw)
        if isinstance(data, list) and data:
            return data, raw
    except Exception:
        pass

    # JSON block inside
    block = extract_json_block(raw)
    if block:
        try:
            data = json.loads(block)
            if isinstance(data, list) and data:
                return data, raw
        except Exception:
            pass

    # Parse text formats
    parsed = parse_mcqs_from_text(raw)
    return parsed, raw


def generate_with_retries(topic_text: str, num_q: int, difficulty: str, max_tries: int = 5):
    final_mcqs: List[Dict[str, Any]] = []
    raw_outputs: List[str] = []
    tries = 0

    while len(final_mcqs) < num_q and tries < max_tries:
        tries += 1
        mcqs, raw = safe_generate_mcqs(topic_text, num_q, difficulty)
        raw_outputs.append(f"--- TRY {tries} ---\n{raw}")

        if mcqs:
            existing = {q.get("question", "").lower().strip() for q in final_mcqs}
            for q in mcqs:
                qq = (q.get("question", "") or "").lower().strip()
                opts = q.get("options", {})
                ans = (q.get("answer", "A") or "A").upper()

                if (
                    qq
                    and qq not in existing
                    and isinstance(opts, dict)
                    and set(opts.keys()) >= {"A", "B", "C", "D"}
                    and all(str(opts.get(k, "")).strip() for k in ["A", "B", "C", "D"])
                    and ans in {"A", "B", "C", "D"}
                ):
                    final_mcqs.append(
                        {"question": q["question"], "options": {k: opts[k] for k in ["A", "B", "C", "D"]}, "answer": ans}
                    )
                    existing.add(qq)

                if len(final_mcqs) >= num_q:
                    break

    return final_mcqs, "\n\n".join(raw_outputs)


# -----------------------------
# Routes (Jinja2 UI)
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "mcq.html",
        {
            "request": request,
            "topic_text": DEFAULT_TEXT,
            "num_q": 5,
            "difficulty": "Medium",
            "mcqs": None,
            "score": None,
            "recommended": None,
            "answer_key": None,
            "raw_debug": None,
            "error": None,
        },
    )


@app.post("/generate", response_class=HTMLResponse)
def generate_quiz(
    request: Request,
    topic_text: str = Form(...),
    num_q: int = Form(5),
    difficulty: str = Form("Medium"),
):
    difficulty = difficulty.title()
    if difficulty not in DIFF_MAP:
        difficulty = "Medium"

    if not topic_text.strip():
        return templates.TemplateResponse(
            "mcq.html",
            {
                "request": request,
                "topic_text": topic_text,
                "num_q": num_q,
                "difficulty": difficulty,
                "mcqs": None,
                "score": None,
                "recommended": None,
                "answer_key": None,
                "raw_debug": None,
                "error": "Please paste some topic text first.",
            },
        )

    mcqs, raw_debug = generate_with_retries(topic_text, num_q, difficulty, max_tries=5)

    if not mcqs:
        return templates.TemplateResponse(
            "mcq.html",
            {
                "request": request,
                "topic_text": topic_text,
                "num_q": num_q,
                "difficulty": difficulty,
                "mcqs": None,
                "score": None,
                "recommended": None,
                "answer_key": None,
                "raw_debug": raw_debug,
                "error": "Could not extract MCQs after retries. See debug output.",
            },
        )

    return templates.TemplateResponse(
        "mcq.html",
        {
            "request": request,
            "topic_text": topic_text,
            "num_q": num_q,
            "difficulty": difficulty,
            "mcqs": mcqs,
            "score": None,
            "recommended": None,
            "answer_key": None,
            "raw_debug": raw_debug,
            "error": None,
        },
    )


@app.post("/submit", response_class=HTMLResponse)
async def submit_quiz(
    request: Request,
    topic_text: str = Form(...),
    num_q: int = Form(...),
    difficulty: str = Form(...),
    mcqs_json: str = Form(...),
):
    difficulty = difficulty.title()
    if difficulty not in DIFF_MAP:
        difficulty = "Medium"

    try:
        mcqs = json.loads(mcqs_json)
        if not isinstance(mcqs, list) or not mcqs:
            raise ValueError("Invalid MCQs payload")
    except Exception:
        return templates.TemplateResponse(
            "mcq.html",
            {
                "request": request,
                "topic_text": topic_text,
                "num_q": num_q,
                "difficulty": difficulty,
                "mcqs": None,
                "score": None,
                "recommended": None,
                "answer_key": None,
                "raw_debug": None,
                "error": "MCQs data was corrupted. Generate again.",
            },
        )

    form = await request.form()

    correct = 0
    total = len(mcqs)
    answer_key = {}

    for idx, q in enumerate(mcqs, start=1):
        correct_ans = (q.get("answer", "A") or "A").upper()
        if correct_ans not in {"A", "B", "C", "D"}:
            correct_ans = "A"

        answer_key[idx] = correct_ans

        user_ans = (form.get(f"ans_{idx}", "") or "").upper()
        if user_ans == correct_ans:
            correct += 1

    score_ratio = correct / total if total else 0.0
    new_level = rec_binary_adjust(0, 2, DIFF_MAP[difficulty], score_ratio)
    recommended = DIFF_INV[new_level]

    score_text = f"{correct}/{total} ({score_ratio*100:.1f}%)"

    return templates.TemplateResponse(
        "mcq.html",
        {
            "request": request,
            "topic_text": topic_text,
            "num_q": num_q,
            "difficulty": difficulty,
            "mcqs": mcqs,
            "score": score_text,
            "recommended": recommended,
            "answer_key": answer_key,
            "raw_debug": None,
            "error": None,
        },
    )
