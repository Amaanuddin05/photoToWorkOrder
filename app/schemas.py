"""
Pydantic schemas for request and response models.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_id: str
    loaded: bool
    embedding_dim: Optional[int] = None


class DependencyDetail(BaseModel):
    url: str
    reachable: bool
    status_code: Optional[int] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class HealthDepsResponse(BaseModel):
    status: str = "ok"
    dependencies: Dict[str, DependencyDetail]


class EmbedTextRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of texts to embed")


class EmbedTextResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="L2-normalized embedding vectors")
    dim: int = Field(..., description="Embedding dimension")
    truncated: List[bool] = Field(..., description="Boolean flag per text indicating if token truncation occurred")


class EmbedImageResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="L2-normalized embedding vectors")
    dim: int = Field(..., description="Embedding dimension")


class CandidateInput(BaseModel):
    id: Union[str, int, float] = Field(..., description="Unique identifier for candidate")
    text: str = Field(..., description="Text representation / visual line of candidate")


class CandidateRankResult(BaseModel):
    id: Union[str, int, float]
    score: float
    rank: int

