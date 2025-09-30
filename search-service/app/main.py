import os
import requests
from fastapi import FastAPI, HTTPException
from typing import List
from .schemas import UpsertItem, SearchRequest, SearchResponse, SearchResult
from .qdrant_client import client, ensure_collection, build_filter, COLLECTION
from qdrant_client.http import models as qm

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "3072"))

app = FastAPI(title="RealEstate Search Service", version="0.2.0")

@app.on_event("startup")
def startup_event():
    ensure_collection()

@app.get("/health")
def health():
    return {"status": "ok", "collection": COLLECTION, "embedding_model": EMBEDDING_MODEL}


def get_embedding(text: str) -> List[float]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY não configurada")
    r = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=30
    )
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Erro OpenAI: {r.text}")
    emb = r.json()["data"][0]["embedding"]
    if len(emb) != VECTOR_SIZE:
        raise HTTPException(status_code=500, detail=f"Dimensão do embedding ({len(emb)}) difere do VECTOR_SIZE ({VECTOR_SIZE})")
    return emb

@app.post("/upsert")
def upsert(item: UpsertItem):
    vec = get_embedding(item.text)
    client.upsert(
        collection_name=COLLECTION,
        points=[qm.PointStruct(id=item.id, vector=vec, payload=item.payload)]
    )
    return {"status": "ok", "id": item.id}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    vec = get_embedding(req.query_text)
    qfilter = build_filter(req.filters.dict() if req.filters else None)
    res = client.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=min(max(req.top_k, 1), 50),
        query_filter=qfilter
    )
    return SearchResponse(
        results=[
            SearchResult(id=str(p.id), score=float(p.score), payload=p.payload or {})
            for p in res
        ]
    )
