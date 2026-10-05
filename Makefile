.PHONY: help dev api ui cli query benchmark test ingest db index_file index_url kill

# The default target if you just run 'make'
help:
	@echo "========================================"
	@echo " SCRAG (Self-Correcting RAG) Shortcuts"
	@echo "========================================"
	@echo "🚀 Application Servers:"
	@echo "  make dev         : Run BOTH frontend and backend servers together"
	@echo "  make api         : Run ONLY the FastAPI backend server"
	@echo "  make ui          : Run ONLY the Next.js frontend UI"
	@echo "  make kill        : Forcefully kill all backend and frontend processes (clears ports 8000/3000)"
	@echo ""
	@echo "🛠️  CLI Commands:"
	@echo "  make cli         : Start the interactive terminal session"
	@echo "  make query       : Ask a single query (Usage: make query Q=\"your question\")"
	@echo "  make benchmark   : Run the comparative automated benchmark suite"
	@echo ""
	@echo "🗄️  Data & Testing:"
	@echo "  make ingest      : Run the Wikipedia bulk ingestion script"
	@echo "  make index_file  : Index a local file or folder (Usage: make index_file FILE=\"/path/to/file\")"
	@echo "  make index_url   : Index a webpage URL (Usage: make index_url URL=\"https://...\")"
	@echo "  make db          : View local ChromaDB statistics and sample data"
	@echo "  make test        : Run the Pytest test suite"
	@echo "========================================"

# Run both servers concurrently and gracefully kill both on Ctrl+C
dev:
	@echo "Starting both FastAPI Backend and Next.js Frontend..."
	@bash -c "trap 'kill 0' SIGINT; (cd backend && source ../.venv/bin/activate && python -m scrag.cli.main serve) & (cd frontend && npm run dev) & wait"

kill:
	@echo "Forcefully killing processes on ports 3000 and 8000..."
	@-lsof -t -i :3000 | xargs kill -9 2>/dev/null || true
	@-lsof -t -i :8000 | xargs kill -9 2>/dev/null || true
	@echo "Processes killed. Ports are now free."

api:
	cd backend && source ../.venv/bin/activate && python -m scrag.cli.main serve

ui:
	cd frontend && npm run dev

cli:
	cd backend && source ../.venv/bin/activate && python -m scrag.cli.main interactive

# Allow passing a query via `make query Q="my question"`
query:
	@if [ -z "$(Q)" ]; then echo "Error: Please provide a query using Q=\"...\" (e.g., make query Q=\"What is ML?\")"; exit 1; fi
	cd backend && source ../.venv/bin/activate && python -m scrag.cli.main query "$(Q)"

benchmark:
	cd backend && source ../.venv/bin/activate && python -m scrag.cli.main benchmark

test:
	cd backend && source ../.venv/bin/activate && pytest -v

ingest:
	cd backend && source ../.venv/bin/activate && python data/ingest_knowledge.py

index_file:
	@if [ -z "$(FILE)" ]; then echo "Error: Please provide a file or folder path using FILE=\"...\" (e.g., make index_file FILE=\"./README.md\")"; exit 1; fi
	cd backend && source ../.venv/bin/activate && python -m scrag.cli.main index --path "$(FILE)"

index_url:
	@if [ -z "$(URL)" ]; then echo "Error: Please provide a webpage URL using URL=\"...\" (e.g., make index_url URL=\"https://...\")"; exit 1; fi
	cd backend && source ../.venv/bin/activate && python data/ingest_knowledge.py "$(URL)"

db:
	cd backend && source ../.venv/bin/activate && python data/view_db.py
