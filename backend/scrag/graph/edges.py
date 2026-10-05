"""Conditional routing edges and circuit breaker guardrails for SCRAG."""

import logging
from typing import Literal

from scrag.state import GraphState, RetrievalAction

logger = logging.getLogger("SCRAG.Edges")


def route_retrieval_action(
    state: GraphState,
) -> Literal["refine_internal", "rewrite_query", "refine_then_web"]:
    """Conditional Edge 1: Routes after pre-generation document grading.

    - CORRECT: High confidence -> refine internal strips (k_in)
    - INCORRECT: Low confidence -> rewrite query & fallback to web search (k_ex)
    - AMBIGUOUS: Uncertain -> refine internal strips AND trigger web search (k_in + k_ex)
    """
    action = state.get("retrieval_action", RetrievalAction.INCORRECT.value)
    logger.info(f"[SCRAG Edge] Action evaluated as: '{action}'")

    if action == RetrievalAction.CORRECT.value:
        return "refine_internal"
    elif action == RetrievalAction.AMBIGUOUS.value:
        return "refine_then_web"
    else:  # INCORRECT
        return "rewrite_query"


def route_after_critique(
    state: GraphState,
) -> Literal["end", "regenerate", "re_retrieve", "fallback"]:
    """Conditional Edge 2: Routes after post-generation reflection critique.

    - If both Faithfulness and Relevance pass -> "end"
    - If retry_count >= max_retries -> "fallback" (circuit breaker)
    - If Faithfulness fails (hallucinations detected) -> "regenerate" (stricter penalty)
    - If Relevance fails (query unaddressed / facts missing) -> "re_retrieve" (rewrite query)
    """
    faith = state.get("faithfulness_critique", {}) or {}
    rel = state.get("relevance_critique", {}) or {}
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    faith_passed = faith.get("passed", False)
    rel_passed = rel.get("passed", False)

    logger.info(
        f"[SCRAG Edge] Faithfulness: {faith_passed}, Relevance: {rel_passed}, "
        f"Retry: {retry_count}/{max_retries}"
    )

    if faith_passed and rel_passed:
        logger.info("[SCRAG Edge] Draft PASSED all reflection checks -> Complete.")
        return "end"

    if retry_count >= max_retries:
        if not state.get("web_search_needed", False):
            logger.warning("[SCRAG Edge] Local retries exhausted. Escalating to web search!")
            return "escalate_to_web"
        logger.warning(
            f"[SCRAG Edge] Retry limit ({max_retries}) reached -> Fallback."
        )
        return "fallback"

    if not faith_passed:
        logger.info("[SCRAG Edge] Hallucination detected -> Triggering Regeneration.")
        return "regenerate"
    else:
        logger.info("[SCRAG Edge] Low relevance/utility -> Triggering Query Reformulation.")
        return "re_retrieve"
