"""SCRAG Faithfulness Critic: audits context entailment and catches hallucinations."""

import re
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from scrag.config import SCRAGSettings, get_settings
from scrag.state import CritiqueResult, CritiqueType


class FaithfulnessOutput(BaseModel):
    """Structured LLM output for context entailment / hallucination evaluation."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Faithfulness score (1.0 = fully supported by context, 0.0 = completely hallucinated)",
    )
    passed: bool = Field(
        description="True if every claim in the response is strictly supported by context; False if any hallucinated claim exists"
    )
    hallucinated_claims: List[str] = Field(
        default_factory=list,
        description="List of specific claims or assertions not supported by the context",
    )
    reasoning: str = Field(
        description="Detailed verification analysis comparing claims to context"
    )
    suggested_fix: Optional[str] = Field(
        default=None,
        description="Actionable instruction on what to remove or correct during regeneration",
    )


class FaithfulnessCritic:
    """Evaluates whether generated text is strictly grounded in retrieved evidence."""

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
                    self.llm = raw_llm.with_structured_output(FaithfulnessOutput)
                except Exception:
                    self.llm = raw_llm
            else:
                self.llm = None

    def evaluate(self, draft: str, formatted_context: str) -> CritiqueResult:
        """Critique the draft against context strips for factual grounding."""
        if self.llm is not None:
            system_prompt = (
                "You are an impartial SCRAG Faithfulness Critic. "
                "Your sole job is to audit a candidate draft against the provided knowledge context strips. "
                "Check whether EVERY statement and claim in the draft is directly entailed by the context.\n"
                "If the draft contains ANY claim, statistic, date, or entity that is NOT mentioned in the context, "
                "or that contradicts the context, flag it as a hallucinated claim and mark passed=False."
            )
            user_prompt = (
                f"Knowledge Context Strips:\n{formatted_context}\n\n"
                f"Candidate Draft to Audit:\n{draft}"
            )

            try:
                res = self.llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                )
                if isinstance(res, dict):
                    res = FaithfulnessOutput(**res)
                elif hasattr(res, "content") and not isinstance(res, FaithfulnessOutput):
                    import json
                    match = re.search(r"\{.*\}", str(res.content), re.DOTALL)
                    if match:
                        data = json.loads(match.group(0))
                        res = FaithfulnessOutput(**data)

                if isinstance(res, FaithfulnessOutput):
                    passed = res.passed and res.score >= self.settings.faithfulness_threshold
                    claims = res.hallucinated_claims
                    if not passed and not claims and res.reasoning:
                        claims = [res.reasoning]
                    return CritiqueResult(
                        critique_type=CritiqueType.FAITHFULNESS,
                        score=res.score,
                        passed=passed,
                        reasoning=res.reasoning,
                        hallucinated_claims=claims,
                        suggested_fix=res.suggested_fix,
                    )
            except Exception:
                pass

        return self._heuristic_evaluate(draft, formatted_context)

    def _heuristic_evaluate(
        self, draft: str, formatted_context: str
    ) -> CritiqueResult:
        """Deterministic grounding check for offline testing."""
        if not draft.strip():
            return CritiqueResult(
                critique_type=CritiqueType.FAITHFULNESS,
                score=0.0,
                passed=False,
                reasoning="Empty draft.",
                hallucinated_claims=["No content generated"],
            )

        draft_words = set(re.findall(r"\b\w{4,}\b", draft.lower()))
        context_words = set(re.findall(r"\b\w{4,}\b", formatted_context.lower()))

        hallucinated = []
        for marker in ["hallucinated", "unsupported", "fake", "fictional"]:
            if marker in draft.lower():
                hallucinated.append(f"Detected simulated hallucination marker: '{marker}'")

        if hallucinated:
            return CritiqueResult(
                critique_type=CritiqueType.FAITHFULNESS,
                score=0.2,
                passed=False,
                reasoning="Draft contains explicit ungrounded or contradictory assertions.",
                hallucinated_claims=hallucinated,
                suggested_fix="Remove ungrounded assertions.",
            )

        overlap = len(draft_words.intersection(context_words))
        ratio = overlap / len(draft_words) if draft_words else 0.0
        score = min(1.0, max(0.0, ratio * 1.5))
        passed = score >= self.settings.faithfulness_threshold

        return CritiqueResult(
            critique_type=CritiqueType.FAITHFULNESS,
            score=round(score, 2),
            passed=passed,
            reasoning=f"Context grounding word overlap: {overlap}/{len(draft_words)} ({score:.2f})",
            hallucinated_claims=[] if passed else ["Potential ungrounded terminology"],
            suggested_fix=None if passed else "Ensure all terms are sourced directly from strips",
        )
