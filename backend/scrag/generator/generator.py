"""Grounded generator synthesizing factual responses strictly from knowledge strips."""

import re
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from scrag.config import SCRAGSettings, get_settings


class GenerationResult(BaseModel):
    """Output from the grounded generator with citations."""

    text: str = Field(description="The synthesized answer")
    citations: List[str] = Field(
        default_factory=list, description="Extracted citation bracket markers [1], [2]"
    )


class GroundedGenerator:
    """Synthesizes answers grounded strictly on verified knowledge strips."""

    def __init__(
        self,
        settings: Optional[SCRAGSettings] = None,
        llm: Optional[BaseChatModel] = None,
        use_llm: bool = True,
    ):
        self.settings = settings or get_settings()

        if not use_llm:
            self.llm = None
        elif llm is not None:
            self.llm = llm
        else:
            from scrag.llm_factory import get_generator_llm

            self.llm = get_generator_llm(self.settings)

    def generate(
        self,
        question: str,
        formatted_context: str,
        critique_feedback: Optional[str] = None,
    ) -> GenerationResult:
        """Synthesize a response strictly grounded on the provided context strips."""
        if self.llm is not None:
            system_prompt = (
                "You are an expert, highly rigorous factual assistant in a Self-Correcting RAG (SCRAG) pipeline. "
                "Synthesize a clear, concise, and complete answer to the user query based ONLY on the provided context strips.\n"
                "RULES:\n"
                "1. Every factual statement must cite its supporting strip using bracket notation like [1], [2].\n"
                "2. DO NOT extrapolate, assume, or introduce information not supported by the context.\n"
                "3. If the context does not contain sufficient facts to fully answer, state honestly what is known and what is missing."
            )

            if critique_feedback:
                system_prompt += (
                    f"\n\nCRITICAL ATTENTION - PREVIOUS DRAFT CRITIQUE:\n"
                    f"A previous draft was rejected for the following reason:\n{critique_feedback}\n"
                    f"You MUST address this critique and ensure all claims are strictly faithful to the evidence."
                )

            user_prompt = (
                f"Question: {question}\n\n"
                f"Verified Knowledge Strips:\n{formatted_context}\n\n"
                f"Grounded Answer with Citations:"
            )

            try:
                response = self.llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                )
                raw_text = (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                )
                citations = list(set(re.findall(r"\[\d+\]", raw_text)))
                return GenerationResult(text=raw_text.strip(), citations=citations)
            except Exception:
                pass

        return self._heuristic_generate(question, formatted_context, critique_feedback)

    def _heuristic_generate(
        self,
        question: str,
        formatted_context: str,
        critique_feedback: Optional[str] = None,
    ) -> GenerationResult:
        """Deterministic response generation for offline testing."""
        lines = [line.strip() for line in formatted_context.split("\n") if line.strip()]
        content_lines = [
            l for l in lines if not l.startswith("[") and not l.startswith("(")
        ]

        if not content_lines:
            return GenerationResult(
                text="Based on available sources, insufficient data was found to answer the query.",
                citations=[],
            )

        summary = " ".join(content_lines[:2])
        ans_text = f"According to verified evidence [1], {summary}"
        return GenerationResult(text=ans_text, citations=["[1]"])
