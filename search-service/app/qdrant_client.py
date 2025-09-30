import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "imoveis_v1")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "3072"))
DISTANCE_ENV = os.getenv("VECTOR_DISTANCE", "COSINE").upper()
if DISTANCE_ENV not in {"COSINE", "EUCLID", "DOT"}:
    DISTANCE_ENV = "COSINE"

client = QdrantClient(url=QDRANT_URL)


def ensure_collection():
    if COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qm.VectorParams(size=VECTOR_SIZE, distance=qm.Distance[DISTANCE_ENV])
        )


def _lc(x: str | None) -> str | None:
    return x.lower().strip() if isinstance(x, str) else None


def _lc_list(xs):
    if not xs:
        return None
    out = []
    for v in xs:
        if isinstance(v, str) and v.strip():
            out.append(v.lower().strip())
    return out or None


def build_filter(payload_filters: dict | None) -> qm.Filter | None:
    """
    Filters expect EN field names from API and match against *_lc fields in payload
    (created by ETL) to be case-insensitive.
    """
    if not payload_filters:
        return None

    must: list[qm.FieldCondition] = []

    f = payload_filters

    # categorical (lists) → match any on *_lc keys
    def match_any(key_lc: str, values):
        vals = _lc_list(values)
        if vals:
            must.append(qm.FieldCondition(key=key_lc, match=qm.MatchAny(any=vals)))

    # categorical (single) → match value on *_lc keys
    def match_val(key_lc: str, value):
        v = _lc(value)
        if v:
            must.append(qm.FieldCondition(key=key_lc, match=qm.MatchValue(value=v)))

    # city / neighborhoods / street
    if f.get("city"):
        match_val("city_lc", f.get("city"))
    if f.get("neighborhoods"):
        match_any("neighborhood_lc", f.get("neighborhoods"))
    if f.get("street"):
        match_any("street_lc", f.get("street"))

    # property/unit/usage/status/sellerTier
    if f.get("propertyType"):
        match_any("propertyType_lc", f.get("propertyType"))
    if f.get("unitType"):
        match_any("unitType_lc", f.get("unitType"))
    if f.get("usageType"):
        match_any("usageType_lc", f.get("usageType"))
    if f.get("status"):
        match_any("status_lc", f.get("status"))
    if f.get("sellerTier"):
        match_any("sellerTier_lc", f.get("sellerTier"))

    # amenities (list, case-insensitive)
    if f.get("amenities"):
        match_any("amenities_lc", f.get("amenities"))

    # numeric ranges map 1:1 to payload numeric keys
    def add_range(field: str, rng):
        if not rng:
            return
        gte = rng.get("min")
        lte = rng.get("max")
        if gte is None and lte is None:
            return
        must.append(qm.FieldCondition(key=field, range=qm.Range(gte=gte, lte=lte)))

    add_range("price",          f.get("price"))
    add_range("usableArea",     f.get("usableArea"))
    add_range("totalArea",      f.get("totalArea"))
    add_range("bedrooms",       f.get("bedrooms"))
    add_range("bathrooms",      f.get("bathrooms"))
    add_range("suites",         f.get("suites"))
    add_range("parkingSpaces",  f.get("parkingSpaces"))

    return qm.Filter(must=must) if must else None