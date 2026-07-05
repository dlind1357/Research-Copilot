from pydantic import BaseModel
from typing import List, Optional


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    top_k: Optional[int] = 5  # Number of context chunks to retrieve


class Citation(BaseModel):
    chunk_id: str
    text: str
    distance: float
    metadata: dict = {}


class ChatResponse(BaseModel):
    response: str
    citations: List[Citation] = []
    context_used: bool = False
    num_chunks_retrieved: int = 0


class EvaluationRequest(BaseModel):
    query: str
    answer: str
    contexts: List[dict] = []
    citations: List[dict] = []


class EvaluationMetricDetail(BaseModel):
    score: float
    reason: str


class EvaluationResponse(BaseModel):
    metrics: dict
    details: dict
