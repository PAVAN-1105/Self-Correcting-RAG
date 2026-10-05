"""SCRAG Query Rewriter: transforms natural language questions into search keywords."""

import re
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from scrag.config import SCRAGSettings, get_settings


class QueryRewriteOutput(BaseModel):
    """Structured output for search query reformulation."""

    search_query: str = Field(
        description="Concise keywords separated by space, optimized for search engines"
    )
    intent_summary: str = Field(
        description="Brief summary of what factual information is being sought"
    )


class QueryRewriter:
    """Reformulates queries for web search fallback when initial retrieval fails."""

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
            from scrag.llm_factory import get_critic_llm

            raw_llm = get_critic_llm(self.settings)
            if raw_llm is not None:
                try:
                    self.llm = raw_llm.with_structured_output(QueryRewriteOutput)
                except Exception:
                    self.llm = raw_llm
            else:
                self.llm = None

    def rewrite(self, question: str) -> str:
        """Rewrite question into targeted search keywords."""
        if self.llm is not None:
            system_prompt = (
                "You are an expert query optimizer for search engines. "
                "Transform the user's question into a concise, high-signal web search query "
                "consisting of keywords. Omit conversational filler, punctuation, and pleasantries. "
                "Focus on entities, key actions, and technical terms."
            )
            user_prompt = f"Original Question: {question}"

            try:
                res = self.llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                )
                if isinstance(res, QueryRewriteOutput):
                    return res.search_query.strip()
                elif hasattr(res, "content"):
                    return str(res.content).strip()
            except Exception:
                pass

        return self._heuristic_rewrite(question)

    def _heuristic_rewrite(self, question: str) -> str:
        """Remove question boilerplate and extract content keywords."""
        fillers = [
            r"\bcan you tell me\b",
            r"\bwhat is\b",
            r"\bwhat are\b",
            r"\bhow does\b",
            r"\bwhy is\b",
            r"\bexplain\b",
            r"\bdescribe\b",
            r"\bplease\b",
            r"\babout\b",
        ]
        q = question.lower()
        for f in fillers:
            q = re.sub(f, "", q)

        words = re.findall(r"\b\w{3,}\b", q)
        return " ".join(words) if words else question
