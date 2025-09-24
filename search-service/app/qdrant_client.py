import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "imoveis_v1")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "3072"))  # compat c/ text-embedding-3-large
DISTANCE = os.getenv("VECTOR_DISTANCE", "Cosine")

client = QdrantClient(url=QDRANT_URL)

def ensure_collection():
    if COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qm.VectorParams(size=VECTOR_SIZE, distance=getattr(qm.Distance, DISTANCE))
        )


def build_filter(payload_filters: dict | None) -> qm.Filter | None:
    if not payload_filters:
        return None

    must: list[qm.FieldCondition] = []

    cidade = payload_filters.get("cidade")
    if cidade:
        must.append(qm.FieldCondition(key="cidade", match=qm.MatchValue(value=cidade)))

    bairros = payload_filters.get("bairros")
    if bairros:
        must.append(qm.FieldCondition(key="bairro", match=qm.MatchAny(any=bairros)))

    # Ranges simples
    def add_range(field: str, rng: dict | None, to_int: bool = False):
        if not rng:
            return
        gte = int(rng.get("min")) if to_int and rng.get("min") is not None else rng.get("min")
        lte = int(rng.get("max")) if to_int and rng.get("max") is not None else rng.get("max")
        if gte is None and lte is None:
            return
        must.append(qm.FieldCondition(
            key=field,
            range=qm.Range(gte=gte, lte=lte)
        ))

    add_range("preco", payload_filters.get("orcamento"))
    add_range("quartos", payload_filters.get("quartos"), to_int=True)
    add_range("banheiros", payload_filters.get("banheiros"), to_int=True)
    add_range("vagas", payload_filters.get("vagas"), to_int=True)
    add_range("area_util_m2", payload_filters.get("area_util_m2"))

    if not must:
        return None
    return qm.Filter(must=must)