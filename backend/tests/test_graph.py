"""End-to-end tests for the SCRAG state machine."""

import chromadb
import pytest

from scrag.config import SCRAGSettings
from scrag.critic.faithfulness import FaithfulnessCritic
from scrag.critic.relevance import AnswerRelevanceCritic
from scrag.evaluator.grader import DocumentGrader
from scrag.evaluator.refiner import KnowledgeRefiner
from scrag.evaluator.rewriter import QueryRewriter
from scrag.generator.generator import GroundedGenerator
from scrag.graph.nodes import SCRAGNodes
from scrag.graph.workflow import SelfCorrectingRAGApp
from scrag.retrieval.dense import DenseRetriever, FallbackHashEmbeddings
from scrag.retrieval.hybrid import HybridRetriever
from scrag.retrieval.loaders import DocumentChunk
from scrag.retrieval.sparse import BM25Retriever
from scrag.search.web import WebSearchClient
from scrag.state import RetrievalAction


@pytest.fixture
def test_environment():
    """Sets up an isolated, deterministic environment for SCRAG graph testing."""
    settings = SCRAGSettings(
        chroma_persist_directory="",
        crag_upper_threshold=0.60,
        crag_lower_threshold=0.30,
        crag_strip_top_k=3,
        max_retry_count=2,
        search_provider="duckduckgo",
    )

    client = chromadb.EphemeralClient()
    dense = DenseRetriever(
        settings=settings,
        embeddings=FallbackHashEmbeddings(dim=128),
        client=client,
    )
    sparse = BM25Retriever(settings=settings)
    retriever = HybridRetriever(
        settings=settings, dense_retriever=dense, sparse_retriever=sparse
    )

    grader = DocumentGrader(settings=settings, use_llm=False)
    refiner = KnowledgeRefiner(settings=settings, grader=grader)
    rewriter = QueryRewriter(settings=settings, use_llm=False)
    web_searcher = WebSearchClient(settings=settings)
    generator = GroundedGenerator(settings=settings, use_llm=False)
    faithfulness = FaithfulnessCritic(settings=settings, use_llm=False)
    relevance = AnswerRelevanceCritic(settings=settings, use_llm=False)

    nodes = SCRAGNodes(
        settings=settings,
        retriever=retriever,
        grader=grader,
        refiner=refiner,
        rewriter=rewriter,
        web_searcher=web_searcher,
        generator=generator,
        faithfulness_critic=faithfulness,
        relevance_critic=relevance,
    )

    app = SelfCorrectingRAGApp(settings=settings, nodes=nodes)
    return {"app": app, "retriever": retriever, "settings": settings}


def test_pathway_correct_internal_knowledge(test_environment):
    """Test standard successful SCRAG CORRECT pathway."""
    app = test_environment["app"]
    retriever = test_environment["retriever"]

    chunk = DocumentChunk(
        chunk_id="doc_correct_1",
        content=(
            "LangGraph is designed for building stateful, "
            "cyclic multi-actor agent workflows. It overcomes the limitation of linear DAGs."
        ),
        metadata={"source": "langgraph_manual.txt"},
    )
    retriever.add_documents([chunk])

    question = "What is LangGraph designed for?"
    result = app.run(question)

    assert result["status"] == "completed"
    assert result["retrieval_action"] == RetrievalAction.CORRECT.value
    assert len(result["knowledge_strips"]) > 0
    assert result["generation"] is not None
    assert len(result["citations"]) > 0

    visited_nodes = [t["node"] for t in result["trace_history"]]
    assert "retrieve" in visited_nodes
    assert "grade_documents" in visited_nodes
    assert "refine_internal" in visited_nodes
    assert "generate" in visited_nodes
    assert "critique" in visited_nodes


def test_pathway_web_search_fallback(test_environment):
    """Test SCRAG INCORRECT pathway triggering web search fallback."""
    app = test_environment["app"]
    retriever = test_environment["retriever"]

    chunk = DocumentChunk(
        chunk_id="doc_irrelevant_1",
        content="The lifecycle of frogs includes egg, tadpole, and adult frog stages in freshwater habitats.",
        metadata={"source": "biology.txt"},
    )
    retriever.add_documents([chunk])

    question = "Who is the CEO of Apple Inc in 2026?"
    result = app.run(question)

    assert result["retrieval_action"] == RetrievalAction.INCORRECT.value
    assert result["web_search_needed"] is True
    assert result["rewritten_query"] is not None

    visited_nodes = [t["node"] for t in result["trace_history"]]
    assert "rewrite_query" in visited_nodes
    assert "web_search" in visited_nodes
    assert result["generation"] is not None


def test_circuit_breaker_max_retries(test_environment):
    """Test circuit breaker stops infinite looping and yields fallback."""
    app = test_environment["app"]

    question = "Explain the completely non-existent theory of quantum marshmallow acceleration."
    result = app.run(question, max_retries=1)

    assert result["status"] in ["completed", "fallback"]
    assert result["retry_count"] <= 1
    assert result["generation"] is not None
