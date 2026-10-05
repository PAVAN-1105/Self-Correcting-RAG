"""Comparative benchmark: Naive Baseline RAG vs Self-Correcting RAG (SCRAG)."""

import time
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from scrag.config import SCRAGSettings, get_settings
from scrag.graph.workflow import SelfCorrectingRAGApp
from scrag.retrieval.hybrid import HybridRetriever
from scrag.retrieval.loaders import DocumentChunk

console = Console()


class NaiveRAGBaseline:
    """Standard, one-shot linear RAG without self-correction (control baseline)."""

    def __init__(self, retriever: HybridRetriever, settings: SCRAGSettings):
        self.retriever = retriever
        self.settings = settings
        from scrag.generator.generator import GroundedGenerator
        self.generator = GroundedGenerator(settings=settings, use_llm=not settings.is_local)

    def query(self, question: str) -> Dict[str, Any]:
        start_time = time.time()
        # 1. Blind retrieval
        chunks = self.retriever.search(question, top_k=self.settings.hybrid_top_k)
        raw_context = "\n\n".join([c.content for c in chunks])

        # 2. Blind generation (no grading, no filtering, no reflection)
        res = self.generator.generate(question, raw_context)
        elapsed = time.time() - start_time

        return {
            "question": question,
            "generation": res.text,
            "citations": res.citations,
            "retrieved_count": len(chunks),
            "noise_filtered": False,
            "web_fallback_used": False,
            "reflection_performed": False,
            "elapsed_seconds": round(elapsed, 3),
        }


def run_comparative_benchmark(
    app: Optional[SelfCorrectingRAGApp] = None,
    settings: Optional[SCRAGSettings] = None,
) -> List[Dict[str, Any]]:
    """Execute comparative benchmark evaluating Naive RAG vs SCRAG across test scenarios."""
    cfg = settings or get_settings()
    scrag_app = app or SelfCorrectingRAGApp(settings=cfg)

    # Ingest representative corpus
    test_docs = [
        DocumentChunk(
            chunk_id="scrag_paper_01",
            content=(
                "Self-Correcting RAG (SCRAG) integrates pre-generation retrieval grading with "
                "post-generation self-reflection. It decomposes documents into knowledge strips "
                "to eliminate irrelevant noise and trigger autonomous web search fallbacks."
            ),
            metadata={"source": "scrag_overview.txt"},
        ),
        DocumentChunk(
            chunk_id="noisy_distractor_02",
            content=(
                "Bananas are curved, edible fruits produced by several kinds of large herbaceous plants. "
                "They are an excellent source of vitamin B6 and dietary fiber, consumed widely worldwide."
            ),
            metadata={"source": "botany.txt"},
        ),
        DocumentChunk(
            chunk_id="langgraph_core_03",
            content=(
                "LangGraph is a library designed for building stateful, cyclic multi-agent applications. "
                "Unlike linear LangChain chains, LangGraph allows conditional routing and recovery loops."
            ),
            metadata={"source": "langgraph_manual.txt"},
        ),
    ]

    scrag_app.nodes.retriever.clear()
    scrag_app.nodes.retriever.add_documents(test_docs)

    naive_rag = NaiveRAGBaseline(retriever=scrag_app.nodes.retriever, settings=cfg)

    test_queries = [
        {
            "query": "How does SCRAG eliminate irrelevant noise?",
            "scenario": "In-Corpus Query (High Relevance)",
        },
        {
            "query": "What are the latest updates on quantum computing breakthroughs in 2026?",
            "scenario": "Out-of-Corpus Query (Requires Web Fallback)",
        },
        {
            "query": "Why is LangGraph superior to linear chains for self-correcting agents?",
            "scenario": "Multi-Factor Synthesis (Groundedness Check)",
        },
    ]

    results = []

    table = Table(
        title="Comparative Benchmark: Naive Baseline RAG vs Self-Correcting RAG (SCRAG)",
        border_style="cyan",
    )
    table.add_column("Scenario", style="magenta", no_wrap=True)
    table.add_column("Pipeline", style="bold")
    table.add_column("Noise Filtered?", style="yellow")
    table.add_column("Web Fallback?", style="blue")
    table.add_column("Citations", style="green")
    table.add_column("Reflection Status", style="bold")

    for item in test_queries:
        q = item["query"]
        scenario = item["scenario"]

        # Run Naive RAG
        naive_res = naive_rag.query(q)

        # Run SCRAG
        scrag_res = scrag_app.run(q)

        noise_filtered = len(scrag_res.get("knowledge_strips", [])) > 0
        web_used = scrag_res.get("web_search_needed", False)
        citations_count = len(scrag_res.get("citations", []))
        scrag_status = scrag_res.get("status", "completed").upper()

        table.add_row(
            scenario,
            "Naive RAG",
            "[red]NO (Raw Chunks)[/red]",
            "[red]NO (Fails on missing)[/red]",
            str(len(naive_res["citations"])),
            "[red]NONE (Blind Output)[/red]",
        )
        table.add_row(
            "",
            "[bold cyan]SCRAG[/bold cyan]",
            "[green]YES (Strips Recomposed)[/green]" if noise_filtered else "N/A",
            "[green]YES (DuckDuckGo/Tavily)[/green]" if web_used else "[dim]No (Local OK)[/dim]",
            f"[bold green]{citations_count}[/bold green]",
            f"[bold green]{scrag_status}[/bold green]",
        )
        table.add_section()

        results.append({"naive": naive_res, "scrag": scrag_res})

    console.print(table)
    return results
