"""Structured logging and Rich terminal output for Self-Correcting RAG (SCRAG)."""

import logging
import sys
from typing import Any, Dict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("SCRAG")


def print_node_banner(node_name: str, description: str) -> None:
    """Print an eye-catching banner for LangGraph node entry."""
    console.print(
        Panel(
            f"[bold cyan]{description}[/bold cyan]",
            title=f"[bold yellow]SCRAG Node: {node_name}[/bold yellow]",
            border_style="bright_blue",
            expand=False,
        )
    )


def print_evaluation_summary(grades: list[Dict[str, Any]], action: str) -> None:
    """Print pre-generation retrieval evaluation table."""
    table = Table(title="SCRAG Retrieval Evaluation Summary", border_style="magenta")
    table.add_column("Doc ID", style="cyan", no_wrap=True)
    table.add_column("Score", style="yellow")
    table.add_column("Relevant?", style="bold")
    table.add_column("Reasoning", style="italic")

    for g in grades:
        status_str = "[green]YES[/green]" if g.get("is_relevant") else "[red]NO[/red]"
        table.add_row(
            str(g.get("document_id", "N/A")),
            f"{g.get('score', 0.0):.2f}",
            status_str,
            str(g.get("reasoning", "")),
        )

    console.print(table)
    color = (
        "green"
        if action == "correct"
        else ("yellow" if action == "ambiguous" else "red")
    )
    console.print(f"Triggered Action: [bold {color}]{action.upper()}[/bold {color}]\n")


def print_critique_summary(
    faithfulness: Dict[str, Any], relevance: Dict[str, Any]
) -> None:
    """Print post-generation reflection critique outcomes."""
    table = Table(title="SCRAG Reflection Critique", border_style="green")
    table.add_column("Axis", style="cyan")
    table.add_column("Score", style="yellow")
    table.add_column("Verdict", style="bold")
    table.add_column("Reasoning", style="italic")

    f_verdict = (
        "[green]PASSED[/green]"
        if faithfulness.get("passed")
        else "[red]FAILED (Hallucination)[/red]"
    )
    table.add_row(
        "Faithfulness (Grounding)",
        f"{faithfulness.get('score', 0.0):.2f}",
        f_verdict,
        faithfulness.get("reasoning", ""),
    )

    r_verdict = (
        "[green]PASSED[/green]"
        if relevance.get("passed")
        else "[red]FAILED (Off-topic/Incomplete)[/red]"
    )
    table.add_row(
        "Answer Relevance (Utility)",
        f"{relevance.get('score', 0.0):.2f}",
        r_verdict,
        relevance.get("reasoning", ""),
    )

    console.print(table)


def print_final_answer(generation: str, citations: list[str], status: str) -> None:
    """Print the final verified output with citation metadata."""
    title_color = "green" if status == "completed" else "yellow"
    citation_str = ", ".join(citations) if citations else "None"
    content = (
        f"{generation}\n\n"
        f"[bold cyan]Citations:[/bold cyan] {citation_str}\n"
        f"[bold cyan]Status:[/bold cyan] {status.upper()}"
    )
    console.print(
        Panel(
            content,
            title=f"[bold {title_color}]SCRAG Final Verified Answer[/bold {title_color}]",
            border_style=title_color,
            expand=False,
        )
    )
