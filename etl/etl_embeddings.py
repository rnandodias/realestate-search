# etl/etl_embeddings.py
import os
import math
import uuid
import re
import time
from datetime import datetime
from typing import Dict, Any, List, Union, Optional

import requests
from pymongo import MongoClient
from dotenv import load_dotenv
from html import unescape

load_dotenv()

# === Config via .env ===
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
MONGODB_URI       = os.getenv("MONGODB_URI")           # ex: mongodb+srv://...
DB_NAME           = os.getenv("DB_NAME")
COLLECTION_NAME   = os.getenv("COLLECTION_NAME")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "imoveis_v1")
VECTOR_DB_URL     = os.getenv("VECTOR_DB_URL", "http://127.0.0.1:6333")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
VECTOR_SIZE       = int(os.getenv("VECTOR_SIZE", "3072"))
BATCH_UPSERT      = int(os.getenv("BATCH_UPSERT", "128"))
PRINT_EVERY       = int(os.getenv("PRINT_EVERY", "1000"))  # imprime a cada N pontos enviados

# === Helpers ===
def strip_html(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unescape(s)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def build_search_corpus(doc: Dict[str, Any]) -> str:
    """Texto semântico usado no embedding (rua já considerada aqui)."""
    parts = []
    parts.append(f"[Título]: {doc.get('title','')}")
    parts.append(f"[Descrição]: {strip_html(doc.get('description',''))}")

    loc = f"{doc.get('street','')}, {doc.get('streetNumber','')} - {doc.get('neighborhood','')}, {doc.get('city','')}"
    parts.append(f"[Localização]: {loc}")

    parts.append(
        f"[Tipos]: propertyType={doc.get('propertyType','')}; "
        f"unitType={doc.get('unitType','')}; usageType={doc.get('usageType','')}"
    )

    parts.append(
        f"[Detalhes]: área útil={doc.get('usableArea') or ''} m²; área total={doc.get('totalArea') or ''} m²; "
        f"quartos={doc.get('bedrooms') or 0}; suítes={doc.get('suites') or 0}; banheiros={doc.get('bathrooms') or 0}; "
        f"vagas={doc.get('parkingSpaces') or 0}; preço={doc.get('price') or ''}"
    )

    am = doc.get('amenities') or []
    if am:
        parts.append(f"[Amenidades]: {', '.join(am)}")

    parts.append(
        f"[Status]: {doc.get('status','')}; Portal={doc.get('portal','')}; "
        f"Anunciante={doc.get('sellerName','')}/{doc.get('sellerTier','')}"
    )

    # monthlyCondo / yearlyIptu intencionalmente NÃO entram no embedding (pedido seu)
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
    """Payload salvo no Qdrant (inclui espelhos *_lc para filtros case-insensitive)."""
    payload = {
        # campos originais
        "src_id": str(doc.get("_id")),   # preserva o _id do Mongo
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

        # espelho geo para filtros nativos (se você usar geo.near/far)
        "location": {
            "lat": doc.get("lat"),
            "lon": doc.get("lon"),
        },
    }

    # espelhos *_lc para filtros case-insensitive
    payload.update({
        "city_lc": _lc(doc.get("city")),
        "neighborhood_lc": _lc(doc.get("neighborhood")),
        "street_lc": _lc(doc.get("street")),           # <-- ADICIONADO
        "propertyType_lc": _lc(doc.get("propertyType")),
        "unitType_lc": _lc(doc.get("unitType")),
        "usageType_lc": _lc(doc.get("usageType")),
        "status_lc": _lc(doc.get("status")),
        "sellerTier_lc": _lc(doc.get("sellerTier")),
        "amenities_lc": _lc_list(doc.get("amenities")),
    })

    return _clean_value(payload)

def to_point_id(raw: Any) -> Union[int, str]:
    """Usa UUIDv5 determinístico a partir do _id do Mongo.
       (Se já for UUID válido, mantém.)"""
    if raw is None:
        return str(uuid.uuid4())
    s = str(raw)
    try:
        return str(uuid.UUID(s))
    except Exception:
        return str(uuid.uuid5(uuid.NAMESPACE_OID, s))

# === OpenAI ===
def get_embedding(text: str) -> List[float]:
    r = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI {r.status_code}: {r.text}")
    emb = r.json()["data"][0]["embedding"]
    if len(emb) != VECTOR_SIZE:
        raise RuntimeError(f"Embedding dim {len(emb)} != VECTOR_SIZE {VECTOR_SIZE}")
    return emb

# === Qdrant (HTTP API) ===
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
            body = resp.json()
        except Exception:
            body = resp.text
        raise RuntimeError(f"Qdrant upsert {resp.status_code}: {body}")

# === Main ===
def main():
    if not all([OPENAI_API_KEY, MONGODB_URI, DB_NAME, COLLECTION_NAME]):
        raise SystemExit("[erro] Variáveis .env ausentes: verifique OPENAI_API_KEY, MONGODB_URI, DB_NAME, COLLECTION_NAME")

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
    processed = 0           # pontos efetivamente enviados
    embedded = 0            # embeddings gerados
    skipped = 0             # pulados (ex: erro 451/422 etc.)
    upsert_fail = 0         # falhas de upsert

    print("[start] ETL embeddings → Qdrant")
    t0 = time.time()

    for doc in cursor:
        pid = to_point_id(doc.get("_id"))
        text = build_search_corpus(doc)

        try:
            vec = get_embedding(text)
            embedded += 1
        except Exception as e:
            skipped += 1
            print(f"[warn] embedding falhou para _id={doc.get('_id')}: {e}")
            continue

        payload = build_payload(doc)
        batch.append({"id": pid, "vector": vec, "payload": payload})

        if len(batch) >= BATCH_UPSERT:
            try:
                upsert_qdrant(batch)
            except Exception as e:
                print(f"[warn] upsert falhou (tentando novamente): {e}")
                time.sleep(1.0)
                try:
                    upsert_qdrant(batch)
                except Exception as e2:
                    upsert_fail += len(batch)
                    print(f"[error] upsert falhou novamente: {e2}")
                    batch.clear()
                    continue

            processed += len(batch)
            batch.clear()

            if processed % PRINT_EVERY == 0:
                elapsed = time.time() - t0
                print(f"[info] upsert parcial: {processed} | embedded={embedded} | skipped={skipped} | fail={upsert_fail} | {elapsed:.1f}s")

    # flush final
    if batch:
        try:
            upsert_qdrant(batch)
            processed += len(batch)
        except Exception as e:
            print(f"[error] upsert final falhou: {e}")
            upsert_fail += len(batch)
        batch.clear()

    elapsed = time.time() - t0
    print(f"[done] total enviados={processed} | embedded={embedded} | skipped={skipped} | upsert_fail={upsert_fail} | tempo={elapsed:.1f}s")

if __name__ == "__main__":
    main()
