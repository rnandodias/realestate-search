import os
import math
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Union, Iterable

import requests
from pymongo import MongoClient
from pymongo.collection import Collection
from dotenv import load_dotenv

load_dotenv()

# ==== Config (env) ====
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
MONGODB_URI       = os.getenv("MONGODB_URI")
DB_NAME           = os.getenv("DB_NAME", "koortimativaDB")
COLLECTION_NAME   = os.getenv("COLLECTION_NAME", "buscaVoz")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "imoveis_v1")
VECTOR_DB_URL     = os.getenv("VECTOR_DB_URL", "http://127.0.0.1:6333")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
VECTOR_SIZE       = int(os.getenv("VECTOR_SIZE", "3072"))
BATCH_UPSERT      = int(os.getenv("BATCH_UPSERT", "128"))
BATCH_EMBED       = int(os.getenv("BATCH_EMBED", "32"))
MODE              = os.getenv("MODE", "pending")  # pending|updated|all|ids_file
RESET_QDRANT      = os.getenv("RESET_QDRANT", "false").lower() == "true"
EMBED_INDEX_COLL  = os.getenv("EMBED_INDEX_COLL", "embeddings_index")
EMBED_FAIL_COLL   = os.getenv("EMBED_FAIL_COLL", "embeddings_failures")
SAMPLE_SIZE       = int(os.getenv("SAMPLE_SIZE", "0"))
IDS_FILE          = os.getenv("IDS_FILE")

# ==== Helpers ====

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
        "src_id": str(doc.get("_id")),
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


def to_point_id(raw: Any) -> str:
    s = str(raw) if raw is not None else str(uuid.uuid4())
    try:
        return str(uuid.UUID(s))
    except Exception:
        return str(uuid.uuid5(uuid.NAMESPACE_OID, s))


# ==== Embeddings ====

def request_embeddings(texts: List[str]) -> List[List[float]]:
    assert OPENAI_API_KEY, "OPENAI_API_KEY ausente"
    for attempt in range(5):
        try:
            r = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={"model": EMBEDDING_MODEL, "input": texts},
                timeout=120,
            )
            if r.status_code in {429, 500, 502, 503, 504}:
                # backoff exponencial simples
                import time
                sleep = min(60, 2 ** attempt)
                time.sleep(sleep)
                continue
            if r.status_code in {400, 401, 403, 451}:
                raise requests.HTTPError(f"{r.status_code} permanent", response=r)
            r.raise_for_status()
            data = r.json()["data"]
            vectors = [item["embedding"] for item in data]
            for v in vectors:
                if len(v) != VECTOR_SIZE:
                    raise RuntimeError(f"Embedding dim {len(v)} != VECTOR_SIZE {VECTOR_SIZE}")
            return vectors
        except requests.HTTPError as e:
            if getattr(e, "response", None) and e.response is not None:
                code = e.response.status_code
                if code in {400, 401, 403, 451}:
                    # não retentável
                    raise
            # outros erros: seguem o loop (retentáveis)
    raise RuntimeError("Falha ao obter embeddings após retries")


# ==== Qdrant ====

def ensure_collection_qdrant():
    import requests as rq
    if RESET_QDRANT:
        rq.delete(f"{VECTOR_DB_URL}/collections/{QDRANT_COLLECTION}")
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
        timeout=180,
    )
    if not resp.ok:
        print("[qdrant] status=", resp.status_code)
        try:
            print("[qdrant] body=", resp.json())
        except Exception:
            print("[qdrant] text=", resp.text)
        resp.raise_for_status()


# ==== Index bookkeeping ====

def mark_success(idx_coll: Collection, _id, model: str, vdim: int):
    idx_coll.update_one(
        {"_id": _id},
        {"$set": {"embedded_at": datetime.utcnow(), "embedding_model": model, "vector_size": vdim}},
        upsert=True,
    )

def mark_failure(fail_coll: Collection, _id, code: int, msg: str):
    fail_coll.update_one(
        {"_id": _id},
        {"$set": {"error_code": code, "error_msg": msg, "last_seen_at": datetime.utcnow()},
         "$setOnInsert": {"first_seen_at": datetime.utcnow()},
         "$inc": {"attempts": 1}},
        upsert=True,
    )


# ==== Data sources ====
PROJECTION = {
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
    "updatedAt": 1,
}


def cursor_all(coll: Collection):
    if SAMPLE_SIZE > 0:
        return coll.aggregate([
            {"$sample": {"size": SAMPLE_SIZE}},
            {"$project": PROJECTION}
        ], allowDiskUse=True)
    return coll.find({}, projection=PROJECTION, batch_size=500)


def cursor_pending(coll: Collection, idx_coll: Collection):
    pipeline = [
        {"$lookup": {"from": EMBED_INDEX_COLL, "localField": "_id", "foreignField": "_id", "as": "idx"}},
        {"$match": {"idx": {"$size": 0}}},
        {"$project": PROJECTION},
    ]
    if SAMPLE_SIZE > 0:
        pipeline.insert(0, {"$sample": {"size": SAMPLE_SIZE}})
    return coll.aggregate(pipeline, allowDiskUse=True)


def cursor_updated(coll: Collection, idx_coll: Collection):
    pipeline = [
        {"$lookup": {"from": EMBED_INDEX_COLL, "localField": "_id", "foreignField": "_id", "as": "idx"}},
        {"$unwind": {"path": "$idx", "preserveNullAndEmptyArrays": True}},
        {"$match": {"$expr": {"$gt": ["$updatedAt", "$idx.embedded_at"]}}},
        {"$project": PROJECTION},
    ]
    if SAMPLE_SIZE > 0:
        pipeline.insert(0, {"$sample": {"size": SAMPLE_SIZE}})
    return coll.aggregate(pipeline, allowDiskUse=True)


def cursor_ids_file(coll: Collection, path: str):
    ids = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return coll.find({"_id": {"$in": ids}}, projection=PROJECTION, batch_size=500)


# ==== Main ====

def main():
    ensure_collection_qdrant()

    client = MongoClient(MONGODB_URI)
    coll = client[DB_NAME][COLLECTION_NAME]
    idx_coll = client[DB_NAME][EMBED_INDEX_COLL]
    fail_coll = client[DB_NAME][EMBED_FAIL_COLL]

    # escolher fonte
    if MODE == "pending":
        cur = cursor_pending(coll, idx_coll)
    elif MODE == "updated":
        cur = cursor_updated(coll, idx_coll)
    elif MODE == "ids_file" and IDS_FILE:
        cur = cursor_ids_file(coll, IDS_FILE)
    else:
        cur = cursor_all(coll)

    batch_points: List[Dict[str, Any]] = []
    batch_texts: List[str] = []
    batch_ids: List[Any] = []

    processed = 0
    doc_cache: Dict[str, Any] = {}

    for doc in cur:
        raw_id = doc.get("_id")
        doc_cache[str(raw_id)] = doc
        batch_texts.append(build_search_corpus(doc))
        batch_ids.append(raw_id)

        # processa embeddings em lote
        if len(batch_texts) >= BATCH_EMBED:
            try:
                vectors = request_embeddings(batch_texts)
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else -1
                for _id in batch_ids:
                    mark_failure(fail_coll, _id, code, "embedding permanent error")
                batch_texts.clear(); batch_ids.clear()
                continue
            for _id, vec in zip(batch_ids, vectors):
                pid = to_point_id(_id)
                payload = build_payload(doc_cache[str(_id)])
                batch_points.append({"id": pid, "vector": vec, "payload": payload})
                mark_success(idx_coll, _id, EMBEDDING_MODEL, VECTOR_SIZE)
            batch_texts.clear(); batch_ids.clear()

        # upsert periódico
        if len(batch_points) >= BATCH_UPSERT:
            upsert_qdrant(batch_points)
            processed += len(batch_points)
            batch_points.clear()
            if processed % 1024 == 0:
                print(f"[info] upsert total: {processed}")

    # flush finais
    if batch_texts:
        try:
            vectors = request_embeddings(batch_texts)
            for _id, vec in zip(batch_ids, vectors):
                pid = to_point_id(_id)
                payload = build_payload(doc_cache[str(_id)])
                batch_points.append({"id": pid, "vector": vec, "payload": payload})
                mark_success(idx_coll, _id, EMBEDDING_MODEL, VECTOR_SIZE)
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else -1
            for _id in batch_ids:
                mark_failure(fail_coll, _id, code, "embedding permanent error (flush)")

    if batch_points:
        upsert_qdrant(batch_points)
        processed += len(batch_points)

    print(f"[done] total upserted: {processed}")


if __name__ == "__main__":
    main()