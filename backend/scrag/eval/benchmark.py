"""Comparative benchmark: Naive Baseline RAG vs Self-Correcting RAG (SCRAG)."""

import time
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from scrag.config import SCRAGSettings, get_settings
from scrag.graph.workflow import SelfCorrectingRAGApp
from scrag.state import create_initial_state
from scrag.retrieval.hybrid import HybridRetriever
from scrag.retrieval.loaders import DocumentChunk

import scrag.logging # Forces the rich logging configuration to load
console = Console()


class NaiveRAGBaseline:
    """Standard, one-shot linear RAG without self-correction (control baseline)."""

    def __init__(self, retriever: HybridRetriever, settings: SCRAGSettings):
        self.retriever = retriever
        self.settings = settings
        from scrag.generator.generator import GroundedGenerator
        self.generator = GroundedGenerator(settings=settings, use_llm=True)

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

class PureSelfRAGBaseline:
    def __init__(self, nodes, settings):
        self.nodes = nodes
        self.settings = settings

    def query(self, question: str) -> Dict[str, Any]:
        start_time = time.time()
        state = create_initial_state(question=question, max_retries=2)
        
        state.update(self.nodes.retrieve(state))
        
        while state.get("retry_count", 0) <= 2:
            state.update(self.nodes.generate_answer(state))
            state.update(self.nodes.critique_output(state))
            
            faith = state.get("faithfulness_critique", {}).get("passed", False)
            rel = state.get("relevance_critique", {}).get("passed", False)
            
            if faith and rel:
                state["status"] = "completed"
                break
            else:
                state["retry_count"] = state.get("retry_count", 0) + 1
                
        if state.get("status") != "completed":
            state.update(self.nodes.fallback(state))
            
        elapsed = time.time() - start_time
        return {
            "question": question,
            "generation": state.get("generation"),
            "citations": state.get("citations", []),
            "retrieved_count": len(state.get("retrieved_documents", [])),
            "noise_filtered": False,
            "web_fallback_used": False,
            "reflection_performed": True,
            "elapsed_seconds": round(elapsed, 3),
            "status": state.get("status")
        }

class PureCRAGBaseline:
    def __init__(self, nodes, settings):
        self.nodes = nodes
        self.settings = settings

    def query(self, question: str) -> Dict[str, Any]:
        start_time = time.time()
        state = create_initial_state(question=question, max_retries=0)
        
        state.update(self.nodes.retrieve(state))
        state.update(self.nodes.grade_documents(state))
        
        action = state.get("retrieval_action")
        if action in ["refine_internal", "refine_then_web"]:
            state.update(self.nodes.refine_internal(state))
            
        if action in ["rewrite_query", "refine_then_web"]:
            state.update(self.nodes.rewrite_query(state))
            state.update(self.nodes.web_search(state))
            
        state.update(self.nodes.generate_answer(state))
        state["status"] = "completed"
        
        elapsed = time.time() - start_time
        return {
            "question": question,
            "generation": state.get("generation"),
            "citations": state.get("citations", []),
            "retrieved_count": len(state.get("retrieved_documents", [])),
            "noise_filtered": action in ["refine_internal", "refine_then_web"],
            "web_fallback_used": state.get("web_search_needed", False),
            "reflection_performed": False,
            "elapsed_seconds": round(elapsed, 3),
            "status": state.get("status")
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
    self_rag = PureSelfRAGBaseline(nodes=scrag_app.nodes, settings=cfg)
    crag_rag = PureCRAGBaseline(nodes=scrag_app.nodes, settings=cfg)

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
        title="Comparative Benchmark: 4-Way Pipeline Architecture Shootout",
        border_style="cyan",
    )
    table.add_column("Scenario", style="magenta", no_wrap=True)
    table.add_column("Pipeline", style="bold")
    table.add_column("Noise Filtered?", style="yellow")
    table.add_column("Web Fallback?", style="blue")
    table.add_column("Citations", style="green")
    table.add_column("Reflection", style="bold")

    for item in test_queries:
        q = item["query"]
        scenario = item["scenario"]

        naive_res = naive_rag.query(q)
        crag_res = crag_rag.query(q)
        self_res = self_rag.query(q)
        
        start_time = time.time()
        scrag_res = scrag_app.run(q)
        scrag_res["elapsed_seconds"] = time.time() - start_time

        table.add_row(
            scenario,
            "Naive RAG",
            "NO (Raw Chunks)",
            "NO",
            str(len(naive_res["citations"])),
            "NONE (Blind Output)",
        )
        table.add_row(
            "",
            "Pure CRAG",
            "[green]YES[/green]" if crag_res["noise_filtered"] else "N/A",
            "[green]YES[/green]" if crag_res["web_fallback_used"] else "No",
            str(len(crag_res["citations"])),
            "NONE (Blind Output)",
        )
        table.add_row(
            "",
            "Pure Self-RAG",
            "NO (Raw Chunks)",
            "NO",
            str(len(self_res["citations"])),
            f"[green]YES ({self_res.get('status')})[/green]",
        )
        table.add_row(
            "",
            "[bold cyan]SCRAG (Ours)[/bold cyan]",
            "[green]YES (Strips)[/green]" if len(scrag_res.get("knowledge_strips", [])) > 0 else "N/A",
            "[green]YES (Search)[/green]" if scrag_res.get("web_search_needed") else "No",
            f"[bold green]{len(scrag_res.get('citations', []))}[/bold green]",
            f"[bold green]YES ({scrag_res.get('status', 'completed')})[/bold green]",
        )
        table.add_section()

        results.append({
            "naive": naive_res, 
            "crag": crag_res, 
            "self": self_res, 
            "scrag": scrag_res
        })

    console.print(table)
    
    # ---------------------------------------------------------
    # Build Aggregate Scorecard
    # ---------------------------------------------------------
    scorecard = Table(title="\nFinal Scorecard & Metrics (Aggregate)", border_style="magenta")
    scorecard.add_column("Pipeline", style="bold")
    scorecard.add_column("Faithfulness", justify="right", style="green")
    scorecard.add_column("First-Try Efficiency", justify="right", style="cyan")
    scorecard.add_column("Groundedness", justify="right", style="yellow")
    scorecard.add_column("Noise Filtering", justify="right", style="blue")
    scorecard.add_column("Avg Latency", justify="right", style="white")

    pipelines = ["naive", "crag", "self", "scrag"]
    pipeline_names = ["Naive RAG", "Pure CRAG", "Pure Self-RAG", "SCRAG (Ours)"]
    
    total = len(results)
    for key, name in zip(pipelines, pipeline_names):
        # 1. Faithfulness
        if key in ["naive", "crag"]:
            faith_str = "[dim]Untested (Blind)[/dim]"
        else:
            faithful = sum(1 for r in results if r[key].get("status", "") != "fallback")
            faith_pct = int((faithful / total) * 100)
            faith_str = f"{faith_pct}%"
            
        # 2. First-Try Efficiency
        if key in ["naive", "crag"]:
            eff_str = "100%"
        else:
            first_try = sum(1 for r in results if r[key].get("retry_count", 0) == 0)
            eff_pct = int((first_try / total) * 100)
            eff_str = f"{eff_pct}%"
            
        # 3. Groundedness (Has citations)
        grounded = sum(1 for r in results if len(r[key].get("citations", [])) > 0)
        ground_pct = int((grounded / total) * 100)
        
        # 4. Noise Filtering Rate
        filtered = sum(
            1 for r in results 
            if (r[key].get("noise_filtered", False) if key != "scrag" else len(r[key].get("knowledge_strips", [])) > 0)
        )
        filt_pct = int((filtered / total) * 100)
        
        # 5. Avg Latency
        avg_lat = sum(r[key].get("elapsed_seconds", 0) for r in results) / total
        
        scorecard.add_row(
            name,
            faith_str,
            eff_str,
            f"{ground_pct}%",
            f"{filt_pct}%",
            f"{avg_lat:.2f}s"
        )
        
    console.print(scorecard)
    return results

if __name__ == "__main__":
    import asyncio
    import sys
    
    # Run the benchmark directly when this file is executed
    try:
        run_comparative_benchmark()
    except KeyboardInterrupt:
        console.print("\n[bold red]Benchmark interrupted by user.[/bold red]")
        sys.exit(1)
