"""Execution nodes for the SCRAG state machine."""

import logging
from typing import Any, Dict, List, Optional

from scrag.config import SCRAGSettings, get_settings
from scrag.critic.faithfulness import FaithfulnessCritic
from scrag.critic.relevance import AnswerRelevanceCritic
from scrag.evaluator.grader import DocumentGrader
from scrag.evaluator.refiner import KnowledgeRefiner
from scrag.evaluator.rewriter import QueryRewriter
from scrag.generator.generator import GroundedGenerator
from scrag.retrieval.hybrid import HybridRetriever
from scrag.retrieval.loaders import DocumentChunk
from scrag.search.web import WebSearchClient
from scrag.state import GraphState, KnowledgeStrip, RetrievalAction, TraceEvent

logger = logging.getLogger("SCRAG.Nodes")


class SCRAGNodes:
    """Encapsulates all node functions with shared dependencies."""

    def __init__(
        self,
        settings: Optional[SCRAGSettings] = None,
        retriever: Optional[HybridRetriever] = None,
        grader: Optional[DocumentGrader] = None,
        refiner: Optional[KnowledgeRefiner] = None,
        rewriter: Optional[QueryRewriter] = None,
        web_searcher: Optional[WebSearchClient] = None,
        generator: Optional[GroundedGenerator] = None,
        faithfulness_critic: Optional[FaithfulnessCritic] = None,
        relevance_critic: Optional[AnswerRelevanceCritic] = None,
    ):
        self.settings = settings or get_settings()
        self.retriever = retriever or HybridRetriever(settings=self.settings)
        self.grader = grader or DocumentGrader(settings=self.settings)
        self.refiner = refiner or KnowledgeRefiner(
            settings=self.settings, grader=self.grader
        )
        self.rewriter = rewriter or QueryRewriter(settings=self.settings)
        self.web_searcher = web_searcher or WebSearchClient(settings=self.settings)
        self.generator = generator or GroundedGenerator(settings=self.settings)
        self.faithfulness_critic = faithfulness_critic or FaithfulnessCritic(
            settings=self.settings
        )
        self.relevance_critic = relevance_critic or AnswerRelevanceCritic(
            settings=self.settings
        )

    def retrieve(self, state: GraphState) -> Dict[str, Any]:
        """Node 1: Retrieve candidate documents using Hybrid (Dense + BM25) search."""
        query = state.get("rewritten_query") or state["question"]
        logger.info(f"[SCRAG Retrieve] Searching for: '{query}'")

        chunks = self.retriever.search(query, top_k=self.settings.hybrid_top_k)
        raw_docs = [chunk.model_dump() for chunk in chunks]

        trace = list(state.get("trace_history", []))
        trace.append(
            TraceEvent(
                node="retrieve",
                action="hybrid_search",
                details={"query": query, "retrieved_count": len(chunks)},
            ).model_dump()
        )

        return {"raw_documents": raw_docs, "trace_history": trace}

    def grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """Node 2: Evaluate retrieval relevance and classify into CRAG action."""
        question = state["question"]
        raw_docs = state.get("raw_documents", [])
        chunks = [DocumentChunk(**doc) for doc in raw_docs]

        logger.info(f"[SCRAG Grade] Evaluating {len(chunks)} chunks...")
        grades, action = self.grader.grade_all(question, chunks)

        trace = list(state.get("trace_history", []))
        trace.append(
            TraceEvent(
                node="grade_documents",
                action="retrieval_evaluation",
                details={
                    "action": action.value,
                    "grades": [g.model_dump() for g in grades],
                },
            ).model_dump()
        )

        return {
            "document_grades": [g.model_dump() for g in grades],
            "retrieval_action": action.value,
            "web_search_needed": action
            in [RetrievalAction.INCORRECT, RetrievalAction.AMBIGUOUS],
            "trace_history": trace,
        }

    def refine_internal(self, state: GraphState) -> Dict[str, Any]:
        """Node 3A: Decompose relevant internal documents into clean knowledge strips (k_in)."""
        question = state["question"]
        raw_docs = state.get("raw_documents", [])
        grades = state.get("document_grades", [])

        relevant_doc_ids = {
            g["document_id"] for g in grades if g.get("is_relevant")
        }
        relevant_chunks = [
            DocumentChunk(**d)
            for d in raw_docs
            if d["chunk_id"] in relevant_doc_ids
        ]

        logger.info(
            f"[SCRAG Refine Internal] Decomposing {len(relevant_chunks)} chunks..."
        )
        decomposed = self.refiner.decompose(relevant_chunks, is_external=False)
        refined_strips = self.refiner.filter_and_recompose(
            question, decomposed, top_k=self.settings.crag_strip_top_k
        )

        trace = list(state.get("trace_history", []))
        trace.append(
            TraceEvent(
                node="refine_internal",
                action="decompose_and_recompose",
                details={"internal_strips_count": len(refined_strips)},
            ).model_dump()
        )

        return {
            "knowledge_strips": [s.model_dump() for s in refined_strips],
            "trace_history": trace,
        }

    def rewrite_query(self, state: GraphState) -> Dict[str, Any]:
        """Node 3B: Rewrite the query into targeted keywords for web search fallback."""
        question = state["question"]
        logger.info("[SCRAG Rewrite] Optimizing query for search engines...")
        rewritten = self.rewriter.rewrite(question)

        trace = list(state.get("trace_history", []))
        trace.append(
            TraceEvent(
                node="rewrite_query",
                action="query_reformulation",
                details={"original": question, "rewritten": rewritten},
            ).model_dump()
        )

        return {"rewritten_query": rewritten, "trace_history": trace}

    def web_search(self, state: GraphState) -> Dict[str, Any]:
        """Node 3C: Fallback search to external web and extract knowledge strips (k_ex)."""
        search_query = state.get("rewritten_query") or state["question"]
        logger.info(f"[SCRAG Web Search] Querying web for: '{search_query}'")

        web_chunks = self.web_searcher.search(search_query, max_results=4)
        logger.info(f"[SCRAG Web Search] Fetched {len(web_chunks)} passages.")

        web_strips = self.refiner.decompose(web_chunks, is_external=True)
        refined_web = self.refiner.filter_and_recompose(
            state["question"], web_strips, top_k=self.settings.crag_strip_top_k
        )

        current_strips = [
            KnowledgeStrip(**s) for s in state.get("knowledge_strips", [])
        ]
        combined = current_strips + refined_web

        trace = list(state.get("trace_history", []))
        trace.append(
            TraceEvent(
                node="web_search",
                action="external_retrieval",
                details={
                    "search_query": search_query,
                    "web_chunks": len(web_chunks),
                    "refined_web_strips": len(refined_web),
                },
            ).model_dump()
        )

        return {
            "web_documents": [c.model_dump() for c in web_chunks],
            "knowledge_strips": [s.model_dump() for s in combined],
            "trace_history": trace,
        }

    def generate_answer(self, state: GraphState) -> Dict[str, Any]:
        """Node 4: Synthesize grounded response using recomposed knowledge strips."""
        question = state["question"]
        strips = [KnowledgeStrip(**s) for s in state.get("knowledge_strips", [])]
        formatted_context = self.refiner.format_for_generation(strips)

        feedback = None
        if state.get("faithfulness_critique") and not state["faithfulness_critique"].get("passed"):
            feedback = state["faithfulness_critique"].get("suggested_fix")

        logger.info(
            f"[SCRAG Generate] Synthesizing draft (Retry: {state.get('retry_count', 0)}) with {len(strips)} strips..."
        )
        gen_result = self.generator.generate(
            question=question,
            formatted_context=formatted_context,
            critique_feedback=feedback,
        )

        trace = list(state.get("trace_history", []))
        trace.append(
            TraceEvent(
                node="generate",
                action="grounded_synthesis",
                details={
                    "citations": gen_result.citations,
                    "retry_count": state.get("retry_count", 0),
                },
            ).model_dump()
        )

        return {
            "generation": gen_result.text,
            "citations": gen_result.citations,
            "trace_history": trace,
        }

    def critique_output(self, state: GraphState) -> Dict[str, Any]:
        """Node 5: Post-generation reflection: Faithfulness and Answer Relevance."""
        question = state["question"]
        draft = state.get("generation", "")
        strips = [KnowledgeStrip(**s) for s in state.get("knowledge_strips", [])]
        formatted_context = self.refiner.format_for_generation(strips)

        logger.info("[SCRAG Critique] Evaluating Faithfulness and Relevance...")
        faith_critique = self.faithfulness_critic.evaluate(draft, formatted_context)
        rel_critique = self.relevance_critic.evaluate(question, draft)

        trace = list(state.get("trace_history", []))
        trace.append(
            TraceEvent(
                node="critique",
                action="self_reflection",
                details={
                    "faithfulness_passed": faith_critique.passed,
                    "faithfulness_score": faith_critique.score,
                    "relevance_passed": rel_critique.passed,
                    "relevance_score": rel_critique.score,
                },
            ).model_dump()
        )

        return {
            "faithfulness_critique": faith_critique.model_dump(),
            "relevance_critique": rel_critique.model_dump(),
            "trace_history": trace,
        }

    def fallback(self, state: GraphState) -> Dict[str, Any]:
        """Node 6: Circuit breaker fallback when max retries are exceeded."""
        logger.warning(
            f"[SCRAG Fallback] Max retries ({state.get('retry_count', 0)}) reached. Providing graceful response."
        )

        generation = state.get("generation", "")
        fallback_text = (
            f"{generation}\n\n"
            f"[SCRAG Disclaimer]: This response was formulated after {state.get('retry_count', 0)} "
            f"self-correction attempts. Some claims could not be verified with complete factual certainty "
            f"against verified internal or external sources."
        )

        trace = list(state.get("trace_history", []))
        trace.append(
            TraceEvent(
                node="fallback",
                action="circuit_breaker_triggered",
                details={"retry_count": state.get("retry_count", 0)},
            ).model_dump()
        )

        return {
            "generation": fallback_text,
            "status": "fallback",
            "trace_history": trace,
        }
