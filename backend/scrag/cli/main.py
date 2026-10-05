"""Interactive Command Line Interface for Self-Correcting RAG (SCRAG)."""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from scrag import __version__
from scrag.config import SCRAGSettings, get_settings
from scrag.graph.workflow import SelfCorrectingRAGApp
from scrag.logging import (
    print_critique_summary,
    print_evaluation_summary,
    print_final_answer,
    print_node_banner,
)
from scrag.retrieval.loaders import DocumentLoader

console = Console()


def run_interactive_session(app: SelfCorrectingRAGApp):
    """Run an interactive conversational loop with SCRAG in terminal."""
    console.print(
        Panel.fit(
            f"[bold cyan]Self-Correcting RAG (SCRAG) v{__version__}[/bold cyan]\n"
            f"[green]Mode: {'100% Local (Ollama)' if app.settings.is_local else 'Cloud API (OpenAI)'}[/green]\n"
            f"[dim]Type 'exit', 'quit', or ':q' to exit. Type ':benchmark' to run tests.[/dim]",
            border_style="bright_blue",
        )
    )

    while True:
        try:
            question = Prompt.ask("\n[bold yellow]SCRAG Query[/bold yellow]")
            if not question or question.strip().lower() in ["exit", "quit", ":q"]:
                console.print("[dim]Exiting SCRAG. Goodbye![/dim]")
                break

            if question.strip() == ":benchmark":
                from scrag.eval.benchmark import run_comparative_benchmark

                run_comparative_benchmark(app=app, settings=app.settings)
                continue

            with console.status("[bold green]Executing Self-Correcting RAG pipeline...[/bold green]"):
                state = app.run(question.strip())

            # Display Execution Summary
            if state.get("document_grades"):
                print_evaluation_summary(
                    state["document_grades"], state.get("retrieval_action", "unknown")
                )

            if state.get("faithfulness_critique") and state.get("relevance_critique"):
                print_critique_summary(
                    state["faithfulness_critique"], state["relevance_critique"]
                )

            print_final_answer(
                generation=state.get("generation", "No response formulated."),
                citations=state.get("citations", []),
                status=state.get("status", "completed"),
            )

        except KeyboardInterrupt:
            console.print("\n[dim]Session terminated.[/dim]")
            break
        except Exception as e:
            console.print(f"[bold red]Error during pipeline execution:[/bold red] {e}")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="SCRAG: Self-Correcting RAG Assistant CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: query
    query_parser = subparsers.add_parser("query", help="Ask a single question to SCRAG")
    query_parser.add_argument("question", type=str, help="The query string")
    query_parser.add_argument(
        "--retries", type=int, default=None, help="Max self-correction retries"
    )
    query_parser.add_argument(
        "--mode",
        choices=["local", "api", "auto"],
        default="auto",
        help="Model execution mode",
    )

    # Command: index
    index_parser = subparsers.add_parser(
        "index", help="Index a text, markdown, or directory into knowledge base"
    )
    index_parser.add_argument("path", type=str, help="Path to file or folder")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI REST server")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")

    # Command: benchmark
    subparsers.add_parser("benchmark", help="Run comparative benchmark")

    # Command: interactive (default)
    subparsers.add_parser("interactive", help="Start interactive terminal session")

    args = parser.parse_args()

    # Load settings
    settings = get_settings()
    if hasattr(args, "mode") and args.mode:
        settings.execution_mode = args.mode

    app = SelfCorrectingRAGApp(settings=settings)

    if args.command == "query":
        console.print(f"[bold cyan]Querying SCRAG:[/bold cyan] {args.question}")
        state = app.run(args.question, max_retries=args.retries)
        if state.get("document_grades"):
            print_evaluation_summary(
                state["document_grades"], state.get("retrieval_action", "unknown")
            )
        if state.get("faithfulness_critique") and state.get("relevance_critique"):
            print_critique_summary(
                state["faithfulness_critique"], state["relevance_critique"]
            )
        print_final_answer(
            generation=state.get("generation", ""),
            citations=state.get("citations", []),
            status=state.get("status", "completed"),
        )

    elif args.command == "index":
        target = Path(args.path)
        loader = DocumentLoader()
        if target.is_dir():
            chunks = loader.load_directory(target)
        else:
            chunks = loader.load_file(target)

        app.nodes.retriever.add_documents(chunks)
        console.print(
            f"[bold green]Successfully indexed {len(chunks)} chunks into SCRAG![/bold green]"
        )

    elif args.command == "benchmark":
        from scrag.eval.benchmark import run_comparative_benchmark

        run_comparative_benchmark(app=app, settings=settings)

    elif args.command == "serve":
        import uvicorn

        console.print(
            f"[bold green]Launching SCRAG FastAPI server on {args.host}:{args.port}...[/bold green]"
        )
        uvicorn.run("scrag.api.server:app", host=args.host, port=args.port, reload=True)

    else:
        # Default to interactive
        run_interactive_session(app)


if __name__ == "__main__":
    main()
