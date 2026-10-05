"""Factory for creating local (Ollama / HuggingFace) or cloud (OpenAI) models."""

import logging
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from scrag.config import SCRAGSettings, get_settings

logger = logging.getLogger("SCRAG.LLMFactory")


class LocalSentenceEmbeddings(Embeddings):
    """Local SentenceTransformer embeddings running directly on Apple Silicon / CPU."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()


def get_generator_llm(settings: Optional[SCRAGSettings] = None) -> Optional[BaseChatModel]:
    """Instantiate the Grounded Generator model (Ollama local or OpenAI)."""
    cfg = settings or get_settings()

    if cfg.is_local:
        try:
            from langchain_ollama import ChatOllama
            logger.info(f"Using local Ollama generator: {cfg.ollama_generator_model}")
            return ChatOllama(
                model=cfg.ollama_generator_model,
                base_url=cfg.ollama_base_url,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning(f"Could not initialize ChatOllama generator: {e}")
            return None

    if cfg.openai_api_key:
        from langchain_openai import ChatOpenAI
        logger.info(f"Using OpenAI generator: {cfg.openai_generator_model}")
        return ChatOpenAI(
            model=cfg.openai_generator_model,
            temperature=0.0,
            openai_api_key=cfg.openai_api_key,
        )

    return None


def get_critic_llm(settings: Optional[SCRAGSettings] = None) -> Optional[BaseChatModel]:
    """Instantiate the Evaluator/Critic model (Ollama local or OpenAI)."""
    cfg = settings or get_settings()

    if cfg.is_local:
        try:
            from langchain_ollama import ChatOllama
            logger.info(f"Using local Ollama critic: {cfg.ollama_critic_model}")
            return ChatOllama(
                model=cfg.ollama_critic_model,
                base_url=cfg.ollama_base_url,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning(f"Could not initialize ChatOllama critic: {e}")
            return None

    if cfg.openai_api_key:
        from langchain_openai import ChatOpenAI
        logger.info(f"Using OpenAI critic: {cfg.openai_critic_model}")
        return ChatOpenAI(
            model=cfg.openai_critic_model,
            temperature=0.0,
            openai_api_key=cfg.openai_api_key,
        )

    return None


def get_embedding_model(settings: Optional[SCRAGSettings] = None) -> Embeddings:
    """Instantiate the Embedding model (Local SentenceTransformers or OpenAI)."""
    cfg = settings or get_settings()

    if cfg.is_local:
        try:
            return LocalSentenceEmbeddings(model_name=cfg.local_embedding_model)
        except Exception as e:
            logger.warning(f"Could not load local sentence transformer: {e}")

    if not cfg.is_local and cfg.openai_api_key:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=cfg.openai_embedding_model,
            openai_api_key=cfg.openai_api_key,
        )

    # Fallback deterministic hash embedding for testing
    from scrag.retrieval.dense import FallbackHashEmbeddings
    return FallbackHashEmbeddings(dim=384)
