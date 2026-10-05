"""FastAPI request and response Pydantic schemas for SCRAG."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """User request for asking a question to SCRAG."""

    question: str = Field(..., min_length=2, description="The query to answer")
    max_retries: Optional[int] = Field(
        default=None, ge=1, le=5, description="Override maximum self-correction loops"
    )


class QueryResponse(BaseModel):
    """Grounded, verified response from SCRAG."""

    question: str
    answer: str
    citations: List[str]
    status: str
    retrieval_action: Optional[str]
    web_search_used: bool
    retry_count: int
    knowledge_strips_count: int
    trace: List[Dict[str, Any]]


class IndexDocumentRequest(BaseModel):
    """Request to index new text content into SCRAG vector store."""

    text: str = Field(..., min_length=10, description="Raw text content to ingest")
    source: str = Field(default="api_upload", description="Document source label")


class IndexResponse(BaseModel):
    """Response returned after indexing documents."""

    status: str
    indexed_chunks: int
    source: str


class BenchmarkResponse(BaseModel):
    """Response returned from comparative evaluation run."""

    status: str
    scenarios_evaluated: int
    summary: str


class HealthResponse(BaseModel):
    """System health and model configuration status."""

    status: str
    project: str
    version: str
    execution_mode: str
    is_local: bool
    generator_model: str
    critic_model: str
    embedding_model: str
    indexed_documents_count: int
