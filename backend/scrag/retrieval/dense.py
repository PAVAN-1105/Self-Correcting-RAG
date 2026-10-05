"""Dense vector retrieval engine using ChromaDB and local or cloud embeddings."""

from typing import List, Optional

import chromadb
from chromadb.api import ClientAPI
from langchain_core.embeddings import Embeddings

from scrag.config import SCRAGSettings, get_settings
from scrag.retrieval.loaders import DocumentChunk


class FallbackHashEmbeddings(Embeddings):
    """Deterministic offline hash-based embedding for testing and dry runs without models."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> List[float]:
        import hashlib
        import math

        vec = [0.0] * self.dim
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


class DenseRetriever:
    """Manages dense semantic search using ChromaDB."""

    def __init__(
        self,
        settings: Optional[SCRAGSettings] = None,
        embeddings: Optional[Embeddings] = None,
        client: Optional[ClientAPI] = None,
    ):
        self.settings = settings or get_settings()

        if embeddings is not None:
            self.embeddings = embeddings
        else:
            from scrag.llm_factory import get_embedding_model
            self.embeddings = get_embedding_model(self.settings)

        if client is not None:
            self.client = client
        elif self.settings.chroma_persist_directory:
            self.client = chromadb.PersistentClient(
                path=self.settings.chroma_persist_directory
            )
        else:
            self.client = chromadb.EphemeralClient()

        self.collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Embed and index document chunks into ChromaDB."""
        if not chunks:
            return

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embeddings.embed_documents(texts)
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def search(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Perform semantic similarity search for a query."""
        if self.count() == 0:
            return []

        query_vector = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[DocumentChunk] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return chunks

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i, chunk_id in enumerate(ids):
            content = docs[i] if i < len(docs) else ""
            meta = dict(metas[i]) if i < len(metas) else {}
            if i < len(distances):
                meta["similarity_score"] = 1.0 - distances[i]

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=meta,
                )
            )
        return chunks

    def count(self) -> int:
        """Return total number of indexed chunks."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete all documents in collection."""
        self.client.delete_collection(name=self.settings.chroma_collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
