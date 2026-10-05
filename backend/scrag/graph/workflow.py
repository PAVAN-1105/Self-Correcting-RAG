"""SCRAG LangGraph StateGraph compilation and execution runner."""

import logging
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from scrag.config import SCRAGSettings, get_settings
from scrag.graph.edges import route_after_critique, route_retrieval_action
from scrag.graph.nodes import SCRAGNodes
from scrag.state import GraphState, TraceEvent, create_initial_state

logger = logging.getLogger("SCRAG.Workflow")


def create_scrag_graph(
    nodes: Optional[SCRAGNodes] = None, settings: Optional[SCRAGSettings] = None
) -> Any:
    """Construct and compile the cyclical SCRAG state machine."""
    cfg = settings or get_settings()
    node_handlers = nodes or SCRAGNodes(settings=cfg)

    # 1. Initialize StateGraph
    workflow = StateGraph(GraphState)

    # 2. Add Node Workers
    workflow.add_node("retrieve", node_handlers.retrieve)
    workflow.add_node("grade_documents", node_handlers.grade_documents)
    workflow.add_node("refine_internal", node_handlers.refine_internal)
    workflow.add_node("rewrite_query", node_handlers.rewrite_query)
    workflow.add_node("web_search", node_handlers.web_search)
    workflow.add_node("generate_answer", node_handlers.generate_answer)
    workflow.add_node("critique_output", node_handlers.critique_output)
    workflow.add_node("fallback", node_handlers.fallback)

    def increment_retry_node(state: GraphState) -> Dict[str, Any]:
        new_count = state.get("retry_count", 0) + 1
        trace = list(state.get("trace_history", []))
        trace.append(
            TraceEvent(
                node="increment_retry",
                action="retry_loop",
                details={"new_retry_count": new_count},
            ).model_dump()
        )
        return {"retry_count": new_count, "trace_history": trace}

    workflow.add_node("increment_retry", increment_retry_node)

    # 3. Add Edges & Conditional Routing
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        route_retrieval_action,
        {
            "refine_internal": "refine_internal",
            "rewrite_query": "rewrite_query",
            "refine_then_web": "refine_internal",
        },
    )

    def route_after_internal_refinement(state: GraphState) -> str:
        if state.get("web_search_needed"):
            return "rewrite_query"
        return "generate_answer"

    workflow.add_conditional_edges(
        "refine_internal",
        route_after_internal_refinement,
        {
            "rewrite_query": "rewrite_query",
            "generate_answer": "generate_answer",
        },
    )

    workflow.add_edge("rewrite_query", "web_search")
    workflow.add_edge("web_search", "generate_answer")
    workflow.add_edge("generate_answer", "critique_output")

    workflow.add_conditional_edges(
        "critique_output",
        route_after_critique,
        {
            "end": END,
            "regenerate": "increment_retry",
            "re_retrieve": "increment_retry",
            "escalate_to_web": "rewrite_query",
            "fallback": "fallback",
        },
    )

    def route_retry_target(state: GraphState) -> str:
        faith = state.get("faithfulness_critique", {}) or {}
        if not faith.get("passed", False):
            return "generate_answer"
        return "rewrite_query"

    workflow.add_conditional_edges(
        "increment_retry",
        route_retry_target,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
        },
    )

    workflow.add_edge("fallback", END)

    # 4. Compile Graph
    return workflow.compile()


class SelfCorrectingRAGApp:
    """High-level runner wrapping the compiled SCRAG state machine."""

    def __init__(
        self,
        settings: Optional[SCRAGSettings] = None,
        nodes: Optional[SCRAGNodes] = None,
    ):
        self.settings = settings or get_settings()
        self.nodes = nodes or SCRAGNodes(settings=self.settings)
        self.graph = create_scrag_graph(nodes=self.nodes, settings=self.settings)

    def run(self, question: str, max_retries: Optional[int] = None) -> GraphState:
        """Execute the end-to-end SCRAG pipeline synchronously."""
        retries = max_retries if max_retries is not None else self.settings.max_retry_count
        initial_state = create_initial_state(question=question, max_retries=retries)

        logger.info(f"Initiating SCRAG session for query: '{question}'")
        final_state = self.graph.invoke(initial_state)

        if final_state.get("status") == "in_progress":
            final_state["status"] = "completed"

        return final_state
