"""Configuration settings for Self-Correcting RAG (SCRAG)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SCRAGSettings(BaseSettings):
    """Configuration settings for the SCRAG pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Project metadata
    project_name: str = "Self-Correcting RAG (SCRAG)"
    version: str = "1.0.0"
    base_dir: Path = Path(__file__).resolve().parent.parent

    # Execution Mode: "local" (Ollama + local embeddings), "api" (OpenAI + Tavily), or "auto"
    execution_mode: Literal["local", "api", "auto"] = Field(
        default="auto",
        description="Execution mode: 'local' (100% offline, free), 'api' (cloud APIs), or 'auto' (API if key present, else local)",
    )

    # Provider overrides if user wants specific control
    model_provider: Literal["ollama", "openai", "auto"] = Field(
        default="auto",
        description="Model provider: 'ollama', 'openai', or 'auto'",
    )
    embedding_provider: Literal["local", "openai", "auto"] = Field(
        default="auto",
        description="Embedding provider: 'local' (sentence-transformers), 'openai', or 'auto'",
    )

    # Ollama Local Configuration (Used when mode is 'local' or 'auto' without API key)
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    ollama_generator_model: str = Field(
        default="qwen3:8b", alias="OLLAMA_GENERATOR_MODEL"
    )
    ollama_critic_model: str = Field(
        default="qwen2.5-coder:7b", alias="OLLAMA_CRITIC_MODEL"
    )

    # Cloud OpenAI Configuration (Used when mode is 'api' or 'auto' with key)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_generator_model: str = Field(
        default="gpt-4o", alias="OPENAI_GENERATOR_MODEL"
    )
    openai_critic_model: str = Field(
        default="gpt-4o-mini", alias="OPENAI_CRITIC_MODEL"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )

    # Embeddings Configuration
    local_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace local model path",
    )

    # Web Search Configuration: "duckduckgo" (Free, zero API key) or "tavily"
    search_provider: Literal["duckduckgo", "tavily"] = Field(
        default="duckduckgo",
        description="'duckduckgo' (free, zero API key) or 'tavily'",
    )
    tavily_api_key: Optional[str] = Field(default=None, alias="TAVILY_API_KEY")

    # Local Chroma Vector Store
    chroma_persist_directory: str = Field(
        default="./data/chroma_db", alias="CHROMA_PERSIST_DIRECTORY"
    )
    chroma_collection_name: str = Field(
        default="scrag_knowledge_base", alias="CHROMA_COLLECTION_NAME"
    )

    # Hybrid Retrieval Parameters
    hybrid_top_k: int = Field(default=5, ge=1, le=20)
    bm25_k1: float = Field(default=1.5, ge=0.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)

    # CRAG Evaluation Thresholds
    crag_upper_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Threshold above which retrieval is CORRECT",
    )
    crag_lower_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Threshold below which retrieval is INCORRECT",
    )
    crag_strip_top_k: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Top knowledge strips recomposed for context",
    )

    # Self-RAG Reflection Thresholds & Guardrails
    faithfulness_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Score required to pass hallucination check",
    )
    relevance_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Score required to pass answer utility check",
    )
    max_retry_count: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Max self-correction loops before circuit breaker fallback",
    )

    @property
    def is_local(self) -> bool:
        """Determine if system should execute with local models."""
        if self.execution_mode == "local" or self.model_provider == "ollama":
            return True
        if self.execution_mode == "api" or self.model_provider == "openai":
            return False
        # auto mode: use local if no OpenAI API key is set
        return not bool(self.openai_api_key and self.openai_api_key.strip())

    @field_validator("crag_lower_threshold")
    @classmethod
    def validate_threshold_order(cls, v: float, info) -> float:
        upper = info.data.get("crag_upper_threshold", 0.70)
        if v >= upper:
            raise ValueError(
                f"crag_lower_threshold ({v}) must be strictly less than crag_upper_threshold ({upper})"
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> SCRAGSettings:
    """Return cached SCRAG application settings."""
    return SCRAGSettings()
