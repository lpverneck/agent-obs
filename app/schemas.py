from pydantic import BaseModel, Field


class KBDocument(BaseModel):
    text: str
    id: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    n_results: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    id: str
    text: str
    distance: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
