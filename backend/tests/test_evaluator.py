"""Unit tests for SCRAG document grading, strip refinement, query rewriting, and search."""

import pytest

from scrag.config import SCRAGSettings
from scrag.evaluator.grader import DocumentGrader
from scrag.evaluator.refiner import KnowledgeRefiner
from scrag.evaluator.rewriter import QueryRewriter
from scrag.retrieval.loaders import DocumentChunk
from scrag.search.web import WebSearchClient
from scrag.state import RetrievalAction


@pytest.fixture
def evaluator_setup():
    settings = SCRAGSettings(
        crag_upper_threshold=0.70,
        crag_lower_threshold=0.30,
        crag_strip_top_k=3,
    )
    grader = DocumentGrader(settings=settings, use_llm=False)
    refiner = KnowledgeRefiner(settings=settings, grader=grader)
    rewriter = QueryRewriter(settings=settings, use_llm=False)
    searcher = WebSearchClient(settings=settings)
    return {
        "settings": settings,
        "grader": grader,
        "refiner": refiner,
        "rewriter": rewriter,
        "searcher": searcher,
    }


def test_grader_correct_action(evaluator_setup):
    grader = evaluator_setup["grader"]
    question = "What is the purpose of the SCRAG retrieval evaluator?"

    relevant_chunk = DocumentChunk(
        chunk_id="chunk_1",
        content="The purpose of the SCRAG retrieval evaluator is to estimate document relevance and trigger corrective actions.",
        metadata={"source": "scrag_paper.txt"},
    )
    grades, action = grader.grade_all(question, [relevant_chunk])
    assert len(grades) == 1
    assert grades[0].is_relevant is True
    assert action == RetrievalAction.CORRECT


def test_grader_incorrect_action(evaluator_setup):
    grader = evaluator_setup["grader"]
    question = "Who won the FIFA World Cup in 2022?"

    irrelevant_chunk = DocumentChunk(
        chunk_id="chunk_2",
        content="Photosynthesis is the biological process used by plants to convert light energy into chemical energy.",
        metadata={"source": "biology.txt"},
    )
    grades, action = grader.grade_all(question, [irrelevant_chunk])
    assert len(grades) == 1
    assert grades[0].is_relevant is False
    assert action == RetrievalAction.INCORRECT


def test_knowledge_refiner_decompose_and_recompose(evaluator_setup):
    refiner = evaluator_setup["refiner"]
    question = "What is SCRAG and how does it refine knowledge?"

    chunk = DocumentChunk(
        chunk_id="doc_1",
        content=(
            "SCRAG stands for Self Correcting Retrieval Augmented Generation. "
            "It introduces an evaluator to assess document quality. "
            "Bananas are rich in potassium and popular among athletes. "
            "A decompose-then-recompose algorithm selectively filters out noisy strips."
        ),
        metadata={"source": "scrag_doc.txt"},
    )

    strips = refiner.decompose([chunk])
    assert len(strips) >= 2

    refined = refiner.filter_and_recompose(question, strips, top_k=2)
    assert len(refined) <= 2
    recomposed_text = refiner.format_for_generation(refined)
    assert "bananas" not in recomposed_text.lower()
    assert "scrag" in recomposed_text.lower()


def test_query_rewriter(evaluator_setup):
    rewriter = evaluator_setup["rewriter"]
    question = "Can you please explain what are the main features of LangGraph?"
    query = rewriter.rewrite(question)
    assert len(query) > 0
    assert "please" not in query.lower()
    assert "langgraph" in query.lower()
