"""Hybrid retrieval engine combining Dense (ChromaDB) and Sparse (BM25) via RRF."""

from typing import Dict, List, Optional

from scrag.config import SCRAGSettings, get_settings
from scrag.retrieval.dense import DenseRetriever
from scrag.retrieval.loaders import DocumentChunk
from scrag.retrieval.sparse import BM25Retriever


class HybridRetriever:
    """Orchestrates hybrid retrieval using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        settings: Optional[SCRAGSettings] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[BM25Retriever] = None,
    ):
        self.settings = settings or get_settings()
        self.dense = dense_retriever or DenseRetriever(settings=self.settings)
        self.sparse = sparse_retriever or BM25Retriever(settings=self.settings)

    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add chunks to both dense vector store and BM25 index."""
        if not chunks:
            return
        self.dense.add_documents(chunks)
        self.sparse.add_documents(chunks)

    def search(
        self, query: str, top_k: Optional[int] = None
    ) -> List[DocumentChunk]:
        """Execute hybrid search using Reciprocal Rank Fusion."""
        k = top_k or self.settings.hybrid_top_k
        fetch_k = max(k * 2, 10)
        rrf_constant = self.settings.rrf_k

        # 1. Dense search
        dense_hits = self.dense.search(query, top_k=fetch_k)

        # 2. Sparse BM25 search
        sparse_hits = self.sparse.search(query, top_k=fetch_k)

        # 3. Compute RRF scores: RRF_Score = sum(1 / (rrf_k + rank))
        rrf_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, DocumentChunk] = {}

        for rank, chunk in enumerate(dense_hits):
            chunk_id = chunk.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (
                1.0 / (rrf_constant + rank + 1)
            )
            chunk_lookup[chunk_id] = chunk

        for rank, chunk in enumerate(sparse_hits):
            chunk_id = chunk.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (
                1.0 / (rrf_constant + rank + 1)
            )
            if chunk_id not in chunk_lookup:
                chunk_lookup[chunk_id] = chunk

        # 4. Rank by fused RRF score
        sorted_chunk_ids = sorted(
            rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
        )

        final_results: List[DocumentChunk] = []
        for cid in sorted_chunk_ids[:k]:
            item = chunk_lookup[cid].model_copy(deep=True)
            item.metadata["rrf_score"] = rrf_scores[cid]
            final_results.append(item)

        return final_results

    def clear(self) -> None:
        """Clear both indices."""
        self.dense.clear()
        self.sparse.clear()
