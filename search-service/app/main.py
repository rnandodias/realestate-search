import os
import requests
from fastapi import FastAPI, HTTPException, Header, Depends
from typing import List, Optional
from .schemas import UpsertItem, SearchRequest, SearchResponse, SearchResult
from .qdrant_client import client, ensure_collection, build_filter, COLLECTION
from qdrant_client.http import models as qm

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "3072"))
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")  # << novo

app = FastAPI(title="RealEstate Search Service", version="0.2.0")

@app.on_event("startup")
def startup_event():
    ensure_collection()

@app.get("/health")
def health():
    return {"status": "ok", "collection": COLLECTION, "embedding_model": EMBEDDING_MODEL}

def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """
    Se SEARCH_API_KEY não estiver definido no ambiente, não exigimos header (modo dev).
    Se estiver definido, validamos o header X-API-Key.
    """
    if not SEARCH_API_KEY:
        return
    if x_api_key != SEARCH_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

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

@app.post("/upsert", dependencies=[Depends(require_api_key)])
def upsert(item: UpsertItem):
    vec = get_embedding(item.text)
    client.upsert(
        collection_name=COLLECTION,
        points=[qm.PointStruct(id=item.id, vector=vec, payload=item.payload)]
    )
    return {"status": "ok", "id": item.id}

@app.post("/search", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
def search(req: SearchRequest):
    vec = get_embedding(req.query_text)
    qfilter = build_filter(req.filters or None)

    # paginação: compat top_k + novos limit/offset
    offset = max(0, int(req.offset or 0))
    limit = int(req.limit or req.top_k or 10)
    limit = min(max(limit, 1), 100)  # clamp 1..100

    res = client.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=limit,
        offset=offset,
        query_filter=qfilter
    )

    results = [
        SearchResult(id=str(p.id), score=float(p.score), payload=p.payload or {})
        for p in res
    ]
    next_off = offset + len(results) if len(results) == limit else None

    return SearchResponse(results=results, next_offset=next_off)
