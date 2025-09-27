import os
from datetime import datetime
from typing import Dict, Any, List
from pymongo import MongoClient
import requests
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
MONGODB_URI      = os.getenv("MONGODB_URI")  # mongodb+srv://<user>:<pass>@cluster/...
DB_NAME          = os.getenv("DB_NAME")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "imoveis_v1")
VECTOR_DB_URL    = os.getenv("VECTOR_DB_URL", "http://127.0.0.1:6333")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
VECTOR_SIZE      = int(os.getenv("VECTOR_SIZE", "3072"))

# ====== Helpers ======

def build_search_corpus(doc: Dict[str, Any]) -> str:
    """Monta um texto rico usando seu schema (title, description, city, neighborhood, etc.)."""
    parts = []
    # Campos textuais principais
    parts.append(f"[Título]: {doc.get('title','')}")
    parts.append(f"[Descrição]: {doc.get('description','')}")
    # Localização
    loc = f"{doc.get('street','')}, {doc.get('streetNumber','')} - {doc.get('neighborhood','')}, {doc.get('city','')}"
    parts.append(f"[Localização]: {loc}")
    # Tipos & uso
    parts.append(f"[Tipos]: propertyType={doc.get('propertyType','')}; unitType={doc.get('unitType','')}; usageType={doc.get('usageType','')}")
    # Medidas
    parts.append(
        f"[Detalhes]: area={doc.get('usableArea')}; quartos={doc.get('bedrooms')}; banheiros={doc.get('bathrooms')}; preço={doc.get('price')}"
    )
    # Metadados
    parts.append(f"[Status]: {doc.get('status','')}; Portal={doc.get('portal','')}; Seller={doc.get('sellerName','')}/{doc.get('sellerTier','')}")
    return "
".join(parts)


def get_embedding(text: str) -> List[float]:
    r = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=60,
    )
    r.raise_for_status()
    emb = r.json()["data"][0]["embedding"]
    if len(emb) != VECTOR_SIZE:
        raise RuntimeError(f"Embedding dim {len(emb)} != VECTOR_SIZE {VECTOR_SIZE}")
    return emb


def ensure_collection_qdrant():
    import requests as rq
    # cria coleção se não existir
    info = rq.get(f"{VECTOR_DB_URL}/collections").json()
    names = {c["name"] if isinstance(c, dict) else c.get("name") for c in info.get("result", {}).get("collections", [])}
    if QDRANT_COLLECTION not in names:
        rq.put(
            f"{VECTOR_DB_URL}/collections/{QDRANT_COLLECTION}",
            json={
                "vectors": {"size": VECTOR_SIZE, "distance": "Cosine"}
            },
            timeout=30,
        ).raise_for_status()


def upsert_qdrant(points: List[Dict[str, Any]]):
    import requests as rq
    resp = rq.put(
        f"{VECTOR_DB_URL}/collections/{QDRANT_COLLECTION}/points?wait=true",
        json={"points": points},
        timeout=120,
    )
    resp.raise_for_status()


# ====== Main ======

def main():
    ensure_collection_qdrant()
    client = MongoClient(MONGODB_URI)
    coll = client[DB_NAME][COLLECTION_NAME]

    cursor = coll.find({}, limit=5000)  # ajuste o limite conforme necessário

    batch: List[Dict[str, Any]] = []
    for doc in cursor:
        pid = str(doc.get("id") or doc.get("_id"))
        text = build_search_corpus(doc)
        vec = get_embedding(text)

        payload = {
            "id": pid,
            "portal": doc.get("portal"),
            "title": doc.get("title"),
            "description": doc.get("description"),
            "propertyType": doc.get("propertyType"),
            "unitType": doc.get("unitType"),
            "usageType": doc.get("usageType"),
            "usableArea": doc.get("usableArea"),
            "bedrooms": doc.get("bedrooms"),
            "bathrooms": doc.get("bathrooms"),
            "city": doc.get("city"),
            "neighborhood": doc.get("neighborhood"),
            "street": doc.get("street"),
            "streetNumber": doc.get("streetNumber"),
            "lat": doc.get("lat"),
            "lon": doc.get("lon"),
            "status": doc.get("status"),
            "sellerName": doc.get("sellerName"),
            "sellerTier": doc.get("sellerTier"),
            "link": doc.get("link"),
            "price": doc.get("price"),
            "monthlyCondo": doc.get("monthlyCondo"),
            "yearlyIptu": doc.get("yearlyIptu"),
            "updated_at": datetime.utcnow().isoformat(),
        }
        batch.append({"id": pid, "vector": vec, "payload": payload})

        if len(batch) >= 128:
            upsert_qdrant(batch)
            batch.clear()

    if batch:
        upsert_qdrant(batch)


if __name__ == "__main__":
    main()