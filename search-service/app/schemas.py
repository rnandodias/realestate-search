from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class Range(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None

class Filters(BaseModel):
    intent: Optional[str] = None  # "alugar" | "comprar"
    cidade: Optional[str] = None
    bairros: Optional[List[str]] = None
    orcamento: Optional[Range] = None
    quartos: Optional[Range] = None
    banheiros: Optional[Range] = None
    vagas: Optional[Range] = None
    area_util_m2: Optional[Range] = None
    amenidades: Optional[List[str]] = None
    tipo_imovel: Optional[List[str]] = None

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