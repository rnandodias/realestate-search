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


def build_filter(payload_filters: dict | None) -> qm.Filter | None:
    if not payload_filters:
        return None

    must: list[qm.FieldCondition] = []

    # equals/any
    city = payload_filters.get("city")
    if city:
        must.append(qm.FieldCondition(key="city", match=qm.MatchValue(value=city)))

    neighborhoods = payload_filters.get("neighborhoods") or payload_filters.get("neighborhood")
    if neighborhoods:
        if isinstance(neighborhoods, list):
            must.append(qm.FieldCondition(key="neighborhood", match=qm.MatchAny(any=neighborhoods)))
        else:
            must.append(qm.FieldCondition(key="neighborhood", match=qm.MatchValue(value=neighborhoods)))

    property_type = payload_filters.get("propertyType")
    if property_type:
        must.append(qm.FieldCondition(key="propertyType", match=qm.MatchValue(value=property_type)))

    unit_type = payload_filters.get("unitType")
    if unit_type:
        must.append(qm.FieldCondition(key="unitType", match=qm.MatchValue(value=unit_type)))

    usage_type = payload_filters.get("usageType")
    if usage_type:
        must.append(qm.FieldCondition(key="usageType", match=qm.MatchValue(value=usage_type)))

    status = payload_filters.get("status")
    if status:
        must.append(qm.FieldCondition(key="status", match=qm.MatchValue(value=status)))

    # ranges
    def add_range(field: str, rng):
        if not rng:
            return
        gte = rng.get("min")
        lte = rng.get("max")
        if gte is None and lte is None:
            return
        must.append(qm.FieldCondition(key=field, range=qm.Range(gte=gte, lte=lte)))

    add_range("price", payload_filters.get("price"))
    add_range("usableArea", payload_filters.get("usableArea"))
    add_range("bedrooms", payload_filters.get("bedrooms"))
    add_range("bathrooms", payload_filters.get("bathrooms"))

    return qm.Filter(must=must) if must else None