"""Self-Correcting RAG (SCRAG) Package."""

from scrag.config import SCRAGSettings, get_settings
from scrag.graph.workflow import SelfCorrectingRAGApp, create_scrag_graph
from scrag.state import GraphState, KnowledgeStrip, RetrievalAction, create_initial_state

__version__ = "1.0.0"

__all__ = [
    "SelfCorrectingRAGApp",
    "create_scrag_graph",
    "SCRAGSettings",
    "get_settings",
    "GraphState",
    "KnowledgeStrip",
    "RetrievalAction",
    "create_initial_state",
]
