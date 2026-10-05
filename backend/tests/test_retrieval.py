"""Tests for SCRAG document loaders and hybrid retrieval."""

import chromadb
import pytest

from scrag.config import SCRAGSettings
from scrag.retrieval.dense import DenseRetriever, FallbackHashEmbeddings
from scrag.retrieval.hybrid import HybridRetriever
from scrag.retrieval.loaders import DocumentChunk, DocumentLoader
from scrag.retrieval.sparse import BM25Retriever


@pytest.fixture
def sample_corpus():
    text = (
        "Corrective Retrieval Augmented Generation (CRAG) is a framework that designs "
        "corrective strategies to improve the robustness of generation. A retrieval evaluator "
        "assesses document relevance and triggers actions: Correct, Incorrect, or Ambiguous.\n\n"
        "Self-Reflective Retrieval-Augmented Generation (Self-RAG) trains an LM to output "
        "special reflection tokens like Retrieve, ISREL, ISSUP, and ISUSE. It evaluates factuality "
        "and citation groundedness after each generation segment.\n\n"
        "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
        "It supports cyclical graphs, human-in-the-loop workflows, and state persistence."
    )
    return text


def test_document_loader(sample_corpus):
    loader = DocumentLoader(chunk_size=150, chunk_overlap=20)
    chunks = loader.load_text(sample_corpus, source="test_doc.md")

    assert len(chunks) >= 3
    for chunk in chunks:
        assert isinstance(chunk, DocumentChunk)
        assert "source" in chunk.metadata
        assert chunk.metadata["source"] == "test_doc.md"
        assert len(chunk.content) > 0


def test_dense_retriever(sample_corpus):
    loader = DocumentLoader(chunk_size=200, chunk_overlap=20)
    chunks = loader.load_text(sample_corpus, source="sample.txt")

    settings = SCRAGSettings(
        chroma_persist_directory="", chroma_collection_name="test_dense_coll"
    )
    client = chromadb.EphemeralClient()
    dense = DenseRetriever(
        settings=settings,
        embeddings=FallbackHashEmbeddings(dim=128),
        client=client,
    )
    dense.clear()

    dense.add_documents(chunks)
    assert dense.count() == len(chunks)

    results = dense.search("reflection tokens in Self-RAG", top_k=2)
    assert len(results) > 0
    assert any("Self-RAG" in r.content or "reflection" in r.content for r in results)


def test_bm25_retriever(sample_corpus):
    loader = DocumentLoader(chunk_size=200, chunk_overlap=20)
    chunks = loader.load_text(sample_corpus, source="sample.txt")

    bm25 = BM25Retriever()
    bm25.add_documents(chunks)

    results = bm25.search("LangGraph stateful", top_k=2)
    assert len(results) > 0
    assert "LangGraph" in results[0].content
    assert results[0].metadata["bm25_score"] > 0


def test_hybrid_retriever_rrf(sample_corpus):
    loader = DocumentLoader(chunk_size=200, chunk_overlap=20)
    chunks = loader.load_text(sample_corpus, source="sample.txt")

    settings = SCRAGSettings(chroma_persist_directory="")
    client = chromadb.EphemeralClient()
    dense = DenseRetriever(
        settings=settings,
        embeddings=FallbackHashEmbeddings(dim=128),
        client=client,
    )
    sparse = BM25Retriever(settings=settings)
    hybrid = HybridRetriever(
        settings=settings, dense_retriever=dense, sparse_retriever=sparse
    )

    hybrid.add_documents(chunks)
    results = hybrid.search("CRAG evaluator actions", top_k=3)

    assert len(results) > 0
    assert "rrf_score" in results[0].metadata
    assert results[0].metadata["rrf_score"] > 0.0
    top_content = results[0].content
    assert "CRAG" in top_content or "evaluator" in top_content
