from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# ===== Filters in EN (as requested) =====
class Range(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None

class Filters(BaseModel):
    # categorical
    city: Optional[str] = None
    neighborhoods: Optional[List[str]] = None
    propertyType: Optional[List[str]] = None  # e.g., ["Unidade"]
    unitType: Optional[List[str]] = None      # e.g., ["Apartamento"]
    usageType: Optional[List[str]] = None     # e.g., ["Residencial"]

    # numeric ranges
    price: Optional[Range] = None
    usableArea: Optional[Range] = None
    totalArea: Optional[Range] = None
    bedrooms: Optional[Range] = None
    bathrooms: Optional[Range] = None
    suites: Optional[Range] = None
    parkingSpaces: Optional[Range] = None

    # multi-value (case-insensitive)
    amenities: Optional[List[str]] = None

    # optional extras
    status: Optional[List[str]] = None        # e.g., ["ACTIVE"]
    sellerTier: Optional[List[str]] = None    # e.g., ["gold","diamond"]


class UpsertItem(BaseModel):
    id: str
    text: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query_text: str
    filters: Optional[Filters] = None
    top_k: int = 10


class SearchResult(BaseModel):
    id: str
    score: float
    payload: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[SearchResult]