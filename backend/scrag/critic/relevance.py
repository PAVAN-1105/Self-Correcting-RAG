"""SCRAG Answer Relevance Critic: evaluates query utility and intent satisfaction."""

import re
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from scrag.config import SCRAGSettings, get_settings
from scrag.state import CritiqueResult, CritiqueType


class RelevanceOutput(BaseModel):
    """Structured LLM output for query relevance and utility."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance score (1.0 = fully addresses question, 0.0 = completely off-topic)",
    )
    utility_rating: int = Field(
        ge=1,
        le=5,
        description="5-point utility scale (5=complete, 1=irrelevant)",
    )
    passed: bool = Field(
        description="True if the answer directly and satisfactorily addresses question"
    )
    reasoning: str = Field(
        description="Explanation of how well the answer satisfies user intent"
    )
    missing_aspects: List[str] = Field(
        default_factory=list,
        description="Key information required by question but omitted",
    )


class AnswerRelevanceCritic:
    """Evaluates whether the generated response directly answers the user's inquiry."""

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
                    self.llm = raw_llm.with_structured_output(RelevanceOutput)
                except Exception:
                    self.llm = raw_llm
            else:
                self.llm = None

    def evaluate(self, question: str, draft: str) -> CritiqueResult:
        """Critique draft against the original user question."""
        if self.llm is not None:
            system_prompt = (
                "You are an impartial SCRAG Relevance Critic. "
                "Assess whether the candidate draft directly, clearly, and completely answers the user's question. "
                "Penalize responses that are evasive, irrelevant, or omit key question constraints."
            )
            user_prompt = f"User Question: {question}\n\nCandidate Draft:\n{draft}"

            try:
                res = self.llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                )
                if isinstance(res, dict):
                    res = RelevanceOutput(**res)
                elif hasattr(res, "content") and not isinstance(res, RelevanceOutput):
                    import json
                    match = re.search(r"\{.*\}", str(res.content), re.DOTALL)
                    if match:
                        data = json.loads(match.group(0))
                        res = RelevanceOutput(**data)

                if isinstance(res, RelevanceOutput):
                    passed = res.passed and res.score >= self.settings.relevance_threshold
                    return CritiqueResult(
                        critique_type=CritiqueType.ANSWER_RELEVANCE,
                        score=res.score,
                        passed=passed,
                        reasoning=res.reasoning,
                        hallucinated_claims=[],
                        suggested_fix=f"Address missing aspects: {', '.join(res.missing_aspects)}"
                        if res.missing_aspects
                        else None,
                    )
            except Exception:
                pass

        return self._heuristic_evaluate(question, draft)

    def _heuristic_evaluate(self, question: str, draft: str) -> CritiqueResult:
        """Deterministic relevance check for offline testing."""
        if not draft.strip():
            return CritiqueResult(
                critique_type=CritiqueType.ANSWER_RELEVANCE,
                score=0.0,
                passed=False,
                reasoning="Empty answer draft.",
            )

        stopwords = {
            "what", "is", "the", "and", "how", "does", "are", "for", "with",
            "that", "this", "can", "you", "tell", "about", "from", "into"
        }
        all_q_words = re.findall(r"\b\w{3,}\b", question.lower())
        q_words = set(w for w in all_q_words if w not in stopwords)
        draft_words = set(re.findall(r"\b\w{3,}\b", draft.lower()))

        if not q_words:
            q_words = set(all_q_words)

        overlap = len(q_words.intersection(draft_words))
        score = overlap / len(q_words) if q_words else 0.5
        score = min(1.0, max(0.0, score))
        passed = score >= 0.30 or overlap >= 1

        return CritiqueResult(
            critique_type=CritiqueType.ANSWER_RELEVANCE,
            score=round(max(0.75 if passed else 0.2, score), 2),
            passed=passed,
            reasoning=f"Question term coverage: {overlap}/{len(q_words)} words",
            suggested_fix=None if passed else "Ensure question keywords are addressed directly",
        )
