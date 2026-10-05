"""Unit tests for SCRAG configuration and state models."""

import pytest
from pydantic import ValidationError

from scrag.config import SCRAGSettings
from scrag.state import (
    CritiqueResult,
    CritiqueType,
    KnowledgeStrip,
    RetrievalAction,
    RetrievalGrade,
    TraceEvent,
    create_initial_state,
)


def test_default_settings():
    """Verify default SCRAG settings load properly."""
    settings = SCRAGSettings()
    assert settings.project_name == "Self-Correcting RAG (SCRAG)"
    assert settings.execution_mode == "auto"
    assert settings.crag_upper_threshold == 0.70
    assert settings.crag_lower_threshold == 0.30
    assert settings.max_retry_count == 2
    assert settings.search_provider == "duckduckgo"


def test_threshold_validation():
    """Ensure invalid threshold order raises a ValidationError."""
    with pytest.raises(ValidationError):
        SCRAGSettings(crag_upper_threshold=0.40, crag_lower_threshold=0.60)


def test_retrieval_grade_model():
    """Verify RetrievalGrade validation."""
    grade = RetrievalGrade(
        document_id="doc_1",
        score=0.85,
        is_relevant=True,
        reasoning="Document explicitly mentions the answer.",
    )
    assert grade.score == 0.85
    assert grade.is_relevant is True


def test_knowledge_strip_model():
    """Verify KnowledgeStrip model creation and defaults."""
    strip = KnowledgeStrip(
        strip_id="strip_1",
        source="test.txt",
        content="This is an atomic sentence strip.",
        score=0.9,
    )
    assert strip.is_external is False
    assert strip.is_relevant is True


def test_critique_result_model():
    """Verify CritiqueResult handles hallucination claims."""
    critique = CritiqueResult(
        critique_type=CritiqueType.FAITHFULNESS,
        score=0.4,
        passed=False,
        reasoning="Claims about 2024 results were not in the provided documents.",
        hallucinated_claims=["Won gold in 2024"],
        suggested_fix="Remove claim regarding 2024 victory.",
    )
    assert critique.passed is False
    assert len(critique.hallucinated_claims) == 1


def test_create_initial_state():
    """Verify initial GraphState structure and trace initialization."""
    state = create_initial_state("What is SCRAG?", max_retries=3)
    assert state["question"] == "What is SCRAG?"
    assert state["max_retries"] == 3
    assert state["retry_count"] == 0
    assert state["status"] == "in_progress"
    assert state["raw_documents"] == []
    assert len(state["trace_history"]) == 1
    assert state["trace_history"][0]["node"] == "start"
