import fitz  # PyMuPDF

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="PDF to Text Extractor")

# (Optional for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "pdf_ui.html",
        {"request": request},
    )


def extract_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    parts = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    return "".join(parts)


@app.post("/extract-pdf-text")
async def extract_pdf_text(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        extracted_text = extract_pdf_text_from_bytes(pdf_bytes)

        return JSONResponse(
            {
                "filename": file.filename,
                "text_length": len(extracted_text),
                "text": extracted_text,
            }
        )

    except fitz.fitz.FileDataError:
        raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
