"""State definitions and Pydantic schemas for the Self-Correcting RAG (SCRAG) pipeline."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field


class RetrievalAction(str, Enum):
    """Action pathways determined by the SCRAG pre-generation evaluator."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    AMBIGUOUS = "ambiguous"


class RetrievalGrade(BaseModel):
    """Relevance grade for a retrieved document."""

    document_id: str = Field(description="Unique identifier of document chunk")
    source: str = Field(default="internal", description="Document source origin")
    score: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0 to 1.0")
    is_relevant: bool = Field(description="Whether document contains relevant facts")
    reasoning: str = Field(description="Explanation for grading judgment")


class KnowledgeStrip(BaseModel):
    """Granular knowledge strip decomposed from retrieved documents."""

    strip_id: str = Field(description="Unique identifier for strip")
    source: str = Field(description="Originating document file or URL")
    content: str = Field(description="Atomic text strip (1-2 sentences)")
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    is_relevant: bool = Field(default=True)
    is_external: bool = Field(
        default=False, description="True if retrieved from web search fallback"
    )


class CritiqueType(str, Enum):
    """Type of post-generation reflection critique."""

    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"


class CritiqueResult(BaseModel):
    """Evaluation result from SCRAG post-generation reflection."""

    critique_type: CritiqueType
    score: float = Field(ge=0.0, le=1.0, description="Evaluation score 0.0 to 1.0")
    passed: bool = Field(description="Whether the draft satisfies criterion")
    reasoning: str = Field(description="Detailed explanation of the critique")
    hallucinated_claims: List[str] = Field(
        default_factory=list,
        description="List of ungrounded or contradictory assertions",
    )
    suggested_fix: Optional[str] = Field(
        default=None, description="Guidance for regeneration prompt"
    )


class TraceEvent(BaseModel):
    """Audit log entry capturing node actions and transitions."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    node: str = Field(description="SCRAG graph node name")
    action: str = Field(description="Operation executed")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata and metrics"
    )


class GraphState(TypedDict, total=False):
    """The central state dictionary passed between all nodes in the SCRAG workflow."""

    # User inputs
    question: str
    rewritten_query: Optional[str]

    # Retrieval stage
    raw_documents: List[Dict[str, Any]]
    document_grades: List[Dict[str, Any]]
    retrieval_action: Optional[str]  # RetrievalAction value

    # Knowledge refinement
    knowledge_strips: List[Dict[str, Any]]

    # External web fallback
    web_search_needed: bool
    web_documents: List[Dict[str, Any]]

    # Generation stage
    generation: Optional[str]
    citations: List[str]

    # Post-generation critiques
    faithfulness_critique: Optional[Dict[str, Any]]
    relevance_critique: Optional[Dict[str, Any]]

    # Loop control & circuit breaker
    retry_count: int
    max_retries: int
    status: str  # "in_progress", "completed", "fallback"

    # Observability
    trace_history: List[Dict[str, Any]]


def create_initial_state(
    question: str, max_retries: int = 2
) -> GraphState:
    """Helper to initialize a clean SCRAG GraphState."""
    return {
        "question": question,
        "rewritten_query": None,
        "raw_documents": [],
        "document_grades": [],
        "retrieval_action": None,
        "knowledge_strips": [],
        "web_search_needed": False,
        "web_documents": [],
        "generation": None,
        "citations": [],
        "faithfulness_critique": None,
        "relevance_critique": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "status": "in_progress",
        "trace_history": [
            TraceEvent(
                node="start",
                action="initialize",
                details={"question": question, "max_retries": max_retries},
            ).model_dump()
        ],
    }
