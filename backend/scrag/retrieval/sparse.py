"""Sparse keyword retrieval engine using BM25Okapi."""

import re
from typing import List, Optional

from rank_bm25 import BM25Okapi

from scrag.config import SCRAGSettings, get_settings
from scrag.retrieval.loaders import DocumentChunk


def simple_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """Keyword-based sparse retrieval using the BM25 algorithm."""

    def __init__(self, settings: Optional[SCRAGSettings] = None):
        self.settings = settings or get_settings()
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25Okapi] = None

    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Tokenize and build BM25 index over provided chunks."""
        if not chunks:
            return

        self.chunks.extend(chunks)
        tokenized_corpus = [simple_tokenize(c.content) for c in self.chunks]
        self.bm25 = BM25Okapi(
            tokenized_corpus,
            k1=self.settings.bm25_k1,
            b=self.settings.bm25_b,
        )

    def search(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Query BM25 index and return top scoring DocumentChunks."""
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = simple_tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

        results: List[DocumentChunk] = []
        for idx in ranked_indices[: min(top_k, len(ranked_indices))]:
            if scores[idx] > 0.0:
                chunk = self.chunks[idx].model_copy(deep=True)
                chunk.metadata["bm25_score"] = float(scores[idx])
                results.append(chunk)

        return results

    def count(self) -> int:
        """Total chunks in BM25 index."""
        return len(self.chunks)

    def clear(self) -> None:
        """Reset index."""
        self.chunks = []
        self.bm25 = None
