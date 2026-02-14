import re
import io

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


app = FastAPI(title="IT Concept Graph (Jinja2 UI)")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Load NER model once
# -----------------------------
ner = pipeline(
    "token-classification",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",
)

# -----------------------------
# Keyword extraction (TF-IDF)
# -----------------------------
def extract_keywords(text: str, k: int = 12):
    cleaned = re.sub(r"\s+", " ", text.strip())
    if len(cleaned) < 10:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=2000
    )
    tfidf = vectorizer.fit_transform([cleaned])
    feats = vectorizer.get_feature_names_out()
    scores = tfidf.toarray()[0]

    pairs = sorted(zip(feats, scores), key=lambda x: x[1], reverse=True)
    return [w for w, s in pairs[:k] if s > 0]

# -----------------------------
# Extract concepts (NER + keywords)
# -----------------------------
def extract_concepts(text: str, max_ner: int = 12, max_kw: int = 12):
    ner_entities = ner(text[:2000])

    ner_concepts = []
    for e in ner_entities:
        w = e.get("word", "").strip()
        if w:
            ner_concepts.append(w)

    kw_concepts = extract_keywords(text, k=max_kw)

    all_concepts = ner_concepts[:max_ner] + kw_concepts

    seen = set()
    final = []
    for c in all_concepts:
        c2 = c.strip()
        if c2 and c2.lower() not in seen:
            seen.add(c2.lower())
            final.append(c2)

    return final

# -----------------------------
# Build graph + render image bytes
# -----------------------------
def build_concept_graph(concepts):
    G = nx.Graph()
    for c in concepts:
        G.add_node(c)

    for i in range(len(concepts) - 1):
        G.add_edge(concepts[i], concepts[i + 1])

    for i in range(len(concepts) - 3):
        if i % 2 == 0:
            G.add_edge(concepts[i], concepts[i + 3])

    return G

def draw_graph_png_bytes(G):
    fig = plt.figure(figsize=(12, 7))
    pos = nx.spring_layout(G, k=0.7, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=1200)
    nx.draw_networkx_edges(G, pos, width=1.2)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# -----------------------------
# Schema used by API endpoints
# -----------------------------
class ConceptGraphRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max_nodes: int = Field(18, ge=1, le=100)
    max_ner: int = Field(10, ge=0, le=50)
    max_kw: int = Field(12, ge=0, le=50)

# -----------------------------
# Jinja UI routes
# -----------------------------
DEFAULT_TEXT = """Networking Devices:
Router, Switch, Hub, Gateway, NIC, Host, Client-Server model.
IP addressing, TCP/UDP, DNS, DHCP, HTTP/HTTPS, firewalls, NAT.
OSI model layers and packet forwarding, routing tables, LAN/WAN.
"""

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "concept_graph.html",
        {
            "request": request,
            "text": DEFAULT_TEXT,
            "max_nodes": 18,
            "max_ner": 10,
            "max_kw": 12,
            "concepts": None,
            "error": None,
            "show_graph": False,
        },
    )

@app.post("/generate", response_class=HTMLResponse)
def generate_ui(
    request: Request,
    text: str = Form(...),
    max_nodes: int = Form(18),
    max_ner: int = Form(10),
    max_kw: int = Form(12),
):
    try:
        if not text.strip():
            raise ValueError("Please paste some topic text first.")

        concepts = extract_concepts(text, max_ner=max_ner, max_kw=max_kw)[:max_nodes]

        return templates.TemplateResponse(
            "concept_graph.html",
            {
                "request": request,
                "text": text,
                "max_nodes": max_nodes,
                "max_ner": max_ner,
                "max_kw": max_kw,
                "concepts": concepts,
                "error": None,
                "show_graph": True,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "concept_graph.html",
            {
                "request": request,
                "text": text,
                "max_nodes": max_nodes,
                "max_ner": max_ner,
                "max_kw": max_kw,
                "concepts": None,
                "error": str(e),
                "show_graph": False,
            },
        )

# -----------------------------
# Image endpoint (used by <img src="...">)
# -----------------------------
@app.get("/concept-graph/image")
def concept_graph_image(
    text: str,
    max_nodes: int = 18,
    max_ner: int = 10,
    max_kw: int = 12,
):
    try:
        concepts = extract_concepts(text, max_ner=max_ner, max_kw=max_kw)[:max_nodes]
        G = build_concept_graph(concepts)
        png_bytes = draw_graph_png_bytes(G)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render graph image: {str(e)}")
