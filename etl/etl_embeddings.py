import os
import math
import uuid
import re
from datetime import datetime
from typing import Dict, Any, List, Union

import requests
from pymongo import MongoClient
from dotenv import load_dotenv
from html import unescape

load_dotenv()

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
MONGODB_URI       = os.getenv("MONGODB_URI")  # mongodb+srv://...
DB_NAME           = os.getenv("DB_NAME")
COLLECTION_NAME   = os.getenv("COLLECTION_NAME")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "imoveis_v1")
VECTOR_DB_URL     = os.getenv("VECTOR_DB_URL", "http://127.0.0.1:6333")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
VECTOR_SIZE       = int(os.getenv("VECTOR_SIZE", "3072"))
BATCH_UPSERT      = int(os.getenv("BATCH_UPSERT", "128"))

# ========= Helpers =========

def strip_html(s: str) -> str:
    if not s:
        return ""
    s = unescape(s)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def build_search_corpus(doc: Dict[str, Any]) -> str:
    parts = []
    parts.append(f"[Título]: {doc.get('title','')}")
    parts.append(f"[Descrição]: {strip_html(doc.get('description',''))}")
    loc = f"{doc.get('street','')}, {doc.get('streetNumber','')} - {doc.get('neighborhood','')}, {doc.get('city','')}"
    parts.append(f"[Localização]: {loc}")
    parts.append(f"[Tipos]: propertyType={doc.get('propertyType','')}; unitType={doc.get('unitType','')}; usageType={doc.get('usageType','')}")
    parts.append(
        f"[Detalhes]: área útil={doc.get('usableArea') or ''} m²; área total={doc.get('totalArea') or ''} m²; "
        f"quartos={doc.get('bedrooms') or 0}; suítes={doc.get('suites') or 0}; banheiros={doc.get('bathrooms') or 0}; "
        f"vagas={doc.get('parkingSpaces') or 0}; preço={doc.get('price') or ''}"
    )
    am = doc.get('amenities') or []
    if am:
        parts.append(f"[Amenidades]: {', '.join(am)}")
    parts.append(f"[Status]: {doc.get('status','')}; Portal={doc.get('portal','')}; Anunciante={doc.get('sellerName','')}/{doc.get('sellerTier','')}")
    # monthlyCondo / yearlyIptu intentionally NOT included in embedding
    return "\n".join(parts)

def _clean_value(v: Any):
    if isinstance(v, dict):
        return {k: _clean_value(vv) for k, vv in v.items()}
    if isinstance(v, list):
        return [_clean_value(x) for x in v]
    if isinstance(v, (int, float)):
        return v if math.isfinite(v) else None
    return v

def _lc(x: Any):
    return x.lower().strip() if isinstance(x, str) else x

def _lc_list(xs):
    if not xs:
        return None
    out = []
    for v in xs:
        if isinstance(v, str) and v.strip():
            out.append(v.lower().strip())
    return out or None

def build_payload(doc: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        # original fields
        "src_id": str(doc.get("_id")),
        "id": doc.get("id"),
        "portal": doc.get("portal"),
        "title": doc.get("title"),
        "description": doc.get("description"),
        "propertyType": doc.get("propertyType"),
        "unitType": doc.get("unitType"),
        "usageType": doc.get("usageType"),
        "usableArea": doc.get("usableArea"),
        "totalArea": doc.get("totalArea"),
        "bedrooms": doc.get("bedrooms"),
        "bathrooms": doc.get("bathrooms"),
        "suites": doc.get("suites"),
        "parkingSpaces": doc.get("parkingSpaces"),
        "amenities": doc.get("amenities"),
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
        "imageUrl": doc.get("imageUrl"),
        "updated_at": datetime.utcnow().isoformat(),
    }

    # case-insensitive mirror keys ( *_lc ) for filtering
    payload.update({
        "city_lc": _lc(doc.get("city")),
        "neighborhood_lc": _lc(doc.get("neighborhood")),
        "propertyType_lc": _lc(doc.get("propertyType")),
        "unitType_lc": _lc(doc.get("unitType")),
        "usageType_lc": _lc(doc.get("usageType")),
        "status_lc": _lc(doc.get("status")),
        "sellerTier_lc": _lc(doc.get("sellerTier")),
        "amenities_lc": _lc_list(doc.get("amenities")),
    })

    return _clean_value(payload)

def to_point_id(raw: Any) -> Union[int, str]:
    if raw is None:
        return str(uuid.uuid4())
    s = str(raw)
    try:
        return str(uuid.UUID(s))
    except Exception:
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
        try:
            print("[qdrant]", resp.status_code, resp.json())
        except Exception:
            print("[qdrant]", resp.status_code, resp.text)
        resp.raise_for_status()

def main():
    ensure_collection_qdrant()

    client = MongoClient(MONGODB_URI)
    coll = client[DB_NAME][COLLECTION_NAME]

    projection = {
        "_id": 1,
        "id": 1,
        "portal": 1,
        "title": 1,
        "description": 1,
        "propertyType": 1,
        "unitType": 1,
        "usageType": 1,
        "usableArea": 1,
        "totalArea": 1,
        "bedrooms": 1,
        "bathrooms": 1,
        "suites": 1,
        "parkingSpaces": 1,
        "amenities": 1,
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
        "imageUrl": 1,
    }

    cursor = coll.find({}, projection=projection, batch_size=500)

    batch: List[Dict[str, Any]] = []
    processed = 0

    for doc in cursor:
        pid = to_point_id(doc.get("_id"))
        text = build_search_corpus(doc)
        try:
            vec = get_embedding(text)
        except Exception as e:
            print(f"[warn] embedding falhou para _id={doc.get('_id')}: {e}")
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
