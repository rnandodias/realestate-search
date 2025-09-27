import os
import math
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Union

import requests
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
MONGODB_URI       = os.getenv("MONGODB_URI")  # mongodb+srv://<user>:<pass>@cluster/...
DB_NAME           = os.getenv("DB_NAME")
COLLECTION_NAME   = os.getenv("COLLECTION_NAME")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "imoveis_v1")
VECTOR_DB_URL     = os.getenv("VECTOR_DB_URL", "http://127.0.0.1:6333")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
VECTOR_SIZE       = int(os.getenv("VECTOR_SIZE", "3072"))
BATCH_UPSERT      = int(os.getenv("BATCH_UPSERT", "128"))

# ========= Helpers =========

def build_search_corpus(doc: Dict[str, Any]) -> str:
    parts = []
    parts.append(f"[Título]: {doc.get('title','')}")
    parts.append(f"[Descrição]: {doc.get('description','')}")
    loc = f"{doc.get('street','')}, {doc.get('streetNumber','')} - {doc.get('neighborhood','')}, {doc.get('city','')}"
    parts.append(f"[Localização]: {loc}")
    parts.append(f"[Tipos]: propertyType={doc.get('propertyType','')}; unitType={doc.get('unitType','')}; usageType={doc.get('usageType','')}")
    parts.append(f"[Detalhes]: area={doc.get('usableArea')}; quartos={doc.get('bedrooms')}; banheiros={doc.get('bathrooms')}; preço={doc.get('price')}")
    parts.append(f"[Status]: {doc.get('status','')}; Portal={doc.get('portal','')}; Seller={doc.get('sellerName','')}/{doc.get('sellerTier','')}")
    return "\n".join(parts)


def _clean_value(v: Any):
    if isinstance(v, dict):
        return {k: _clean_value(vv) for k, vv in v.items()}
    if isinstance(v, list):
        return [_clean_value(x) for x in v]
    if isinstance(v, (int, float)):
        return v if math.isfinite(v) else None
    return v


def build_payload(doc: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "src_id": str(doc.get("_id")),  # preserva _id original
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
    return _clean_value(payload)


def to_point_id(raw: Any) -> Union[int, str]:
    """Qdrant aceita ID inteiro ou UUID string.
    Usaremos SEMPRE um UUIDv5 determinístico baseado no _id do MongoDB.
    """
    if raw is None:
        return str(uuid.uuid4())
    s = str(raw)
    # Se já for UUID, use direto
    try:
        return str(uuid.UUID(s))
    except Exception:
        # Gera UUID5 determinístico com namespace OID (estável e reproduzível)
        return str(uuid.uuid5(uuid.NAMESPACE_OID, s))


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
    info = rq.get(f"{VECTOR_DB_URL}/collections", timeout=15).json()
    names = {c.get("name") if isinstance(c, dict) else c for c in info.get("result", {}).get("collections", [])}
    if QDRANT_COLLECTION not in names:
        rq.put(
            f"{VECTOR_DB_URL}/collections/{QDRANT_COLLECTION}",
            json={"vectors": {"size": VECTOR_SIZE, "distance": "Cosine"}},
            timeout=30,
        ).raise_for_status()


def upsert_qdrant(points: List[Dict[str, Any]]):
    import requests as rq
    resp = rq.put(
        f"{VECTOR_DB_URL}/collections/{QDRANT_COLLECTION}/points?wait=true",
        json={"points": points},
        timeout=120,
    )
    if not resp.ok:
        print("[qdrant] status=", resp.status_code)
        try:
            print("[qdrant] body=", resp.json())
        except Exception:
            print("[qdrant] text=", resp.text)
        resp.raise_for_status()


# ========= Main =========

def main():
    ensure_collection_qdrant()

    client = MongoClient(MONGODB_URI)
    coll = client[DB_NAME][COLLECTION_NAME]

    projection = {
        "_id": 1,
        "portal": 1,
        "title": 1,
        "description": 1,
        "propertyType": 1,
        "unitType": 1,
        "usageType": 1,
        "usableArea": 1,
        "bedrooms": 1,
        "bathrooms": 1,
        "city": 1,
        "neighborhood": 1,
        "street": 1,
        "streetNumber": 1,
        "lat": 1,
        "lon": 1,
        "status": 1,
        "sellerName": 1,
        "sellerTier": 1,
        "link": 1,
        "price": 1,
        "monthlyCondo": 1,
        "yearlyIptu": 1,
    }

    cursor = coll.find({}, projection=projection, batch_size=500)

    batch: List[Dict[str, Any]] = []
    processed = 0

    for doc in cursor:
        raw_id = doc.get("_id")  # usa SEMPRE o _id do MongoDB; ignore campo 'id' externo
        pid = to_point_id(raw_id)
        text = build_search_corpus(doc)
        try:
            vec = get_embedding(text)
        except Exception as e:
            print(f"[warn] embedding falhou para _id={raw_id}: {e}")
            continue

        payload = build_payload(doc)
        batch.append({"id": pid, "vector": vec, "payload": payload})

        if len(batch) >= BATCH_UPSERT:
            upsert_qdrant(batch)
            batch.clear()
            processed += BATCH_UPSERT
            if processed % 1024 == 0:
                print(f"[info] upsert total: {processed}")

    if batch:
        upsert_qdrant(batch)
        processed += len(batch)

    print(f"[done] total processados: {processed}")


if __name__ == "__main__":
    main()
