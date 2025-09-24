import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pymongo import MongoClient
import requests
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_DB_URL  = os.getenv("VECTOR_DB_URL", "http://127.0.0.1:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "imoveis_v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "3072"))

DOCDB_URI = os.getenv("DOCDB_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


def build_search_corpus(doc: Dict[str,Any]) -> str:
    ad = doc.get("ad", {})
    amen = doc.get("amenidades", [])
    return (
        f"[Título]: {ad.get('title','')}\n"
        f"[Descrição]: {ad.get('description','')}\n"
        f"[Localização]: {doc.get('bairro','')}, {doc.get('cidade','')}, {doc.get('estado','')}\n"
        f"[Comodidades]: {', '.join(amen)}\n"
        f"[Detalhes]: {doc.get('quartos')} quartos; {doc.get('banheiros')} banheiros; {doc.get('vagas')} vagas; {doc.get('area_util_m2')} m²\n"
    ).strip()


def get_embedding(text: str) -> List[float]:
    r = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=60
    )
    r.raise_for_status()
    emb = r.json()["data"][0]["embedding"]
    if len(emb) != VECTOR_SIZE:
        raise RuntimeError(f"Embedding dim {len(emb)} != VECTOR_SIZE {VECTOR_SIZE}")
    return emb


def upsert_qdrant(points: List[Dict[str,Any]]):
    r = requests.put(
        f"{VECTOR_DB_URL}/collections/{QDRANT_COLLECTION}/points?wait=true",
        json={"points": points},
        timeout=120
    )
    r.raise_for_status()


def main():
    client = MongoClient(DOCDB_URI)
    coll = client[DB_NAME][COLLECTION_NAME]

    since = datetime.utcnow() - timedelta(days=1)
    cursor = coll.find({"updated_at": {"$gte": since}})

    batch = []
    for doc in cursor:
        pid = str(doc["_id"])
        text = build_search_corpus(doc)
        vec = get_embedding(text)
        payload = {
            "imovel_id": pid,
            "cidade": doc.get("cidade"),
            "bairro": doc.get("bairro"),
            "preco": doc.get("preco"),
            "quartos": doc.get("quartos"),
            "banheiros": doc.get("banheiros"),
            "vagas": doc.get("vagas"),
            "area_util_m2": doc.get("area_util_m2"),
            "updated_at": datetime.utcnow().isoformat()
        }
        batch.append({"id": pid, "vector": vec, "payload": payload})
        if len(batch) >= 128:
            upsert_qdrant(batch)
            batch.clear()

    if batch:
        upsert_qdrant(batch)

if __name__ == "__main__":
    main()