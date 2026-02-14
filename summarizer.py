from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

app = FastAPI(title="Study Notes Generator (Flan-T5)")

# Load model once (cached globally)
generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base"
)

class NotesRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max_length: int = Field(300, ge=32, le=1024)

@app.get("/")
def root():
    return {"message": "Study Notes Generator is running", "model": "google/flan-t5-base"}

@app.post("/generate-notes")
def generate_notes(req: NotesRequest):
    try:
        prompt = f"Create structured study notes from this:\n{req.text}"
        result = generator(prompt, max_length=req.max_length, do_sample=False)
        # result looks like: [{'generated_text': '...'}]
        return {"notes": result[0]["generated_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

