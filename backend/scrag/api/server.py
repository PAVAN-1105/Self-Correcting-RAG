import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from scrag import __version__
from scrag.api.schemas import (
    BenchmarkResponse,
    HealthResponse,
    IndexDocumentRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
)
from scrag.config import SCRAGSettings, get_settings
from scrag.graph.workflow import SelfCorrectingRAGApp
from scrag.retrieval.loaders import DocumentLoader

# Global application instance
scrag_app: Optional[SelfCorrectingRAGApp] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SCRAG engine on startup."""
    global scrag_app
    settings = get_settings()
    scrag_app = SelfCorrectingRAGApp(settings=settings)
    yield


app = FastAPI(
    title="Self-Correcting RAG (SCRAG) API",
    description=(
        "Production REST API for Self-Correcting RAG with pre-generation evaluation, "
        "decompose-then-recompose knowledge strips, autonomous web fallback, "
        "and post-generation reflection critique."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Check service health and model provider configurations."""
    settings = get_settings()
    doc_count = scrag_app.nodes.retriever.dense.count() if scrag_app else 0

    return HealthResponse(
        status="healthy",
        project=settings.project_name,
        version=settings.version,
        execution_mode=settings.execution_mode,
        is_local=settings.is_local,
        generator_model=settings.ollama_generator_model
        if settings.is_local
        else settings.openai_generator_model,
        critic_model=settings.ollama_critic_model
        if settings.is_local
        else settings.openai_critic_model,
        embedding_model=settings.local_embedding_model
        if settings.is_local
        else settings.openai_embedding_model,
        indexed_documents_count=doc_count,
    )


@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG"])
def query_scrag(request: QueryRequest) -> QueryResponse:
    """Execute synchronous self-correcting RAG pipeline on user question."""
    if not scrag_app:
        raise HTTPException(status_code=503, detail="SCRAG engine not initialized")

    result = scrag_app.run(
        question=request.question, max_retries=request.max_retries
    )

    return QueryResponse(
        question=result["question"],
        answer=result.get("generation") or "No response could be formulated.",
        citations=result.get("citations", []),
        status=result.get("status", "completed"),
        retrieval_action=result.get("retrieval_action"),
        web_search_used=result.get("web_search_needed", False),
        retry_count=result.get("retry_count", 0),
        knowledge_strips_count=len(result.get("knowledge_strips", [])),
        trace=result.get("trace_history", []),
    )


@app.post("/api/v1/query/stream", tags=["RAG"])
async def stream_query_scrag(request: QueryRequest) -> StreamingResponse:
    """Stream real-time graph node transition events via Server-Sent Events (SSE)."""
    if not scrag_app:
        raise HTTPException(status_code=503, detail="SCRAG engine not initialized")

    from scrag.state import create_initial_state

    async def event_generator() -> AsyncGenerator[str, None]:
        initial_state = create_initial_state(
            question=request.question,
            max_retries=request.max_retries or scrag_app.settings.max_retry_count,
        )

        running_state = {}
        for event in scrag_app.graph.stream(initial_state):
            for node_name, node_state in event.items():
                running_state.update(node_state)
                payload = {
                    "node": node_name,
                    "retry_count": running_state.get("retry_count", 0),
                    "action": running_state.get("retrieval_action"),
                    "status": "running"
                }
                yield f"data: {json.dumps(payload)}\n\n"

        if running_state:
            strips = running_state.get("knowledge_strips", [])
            formatted_citations = []
            
            for s in strips:
                if isinstance(s, dict):
                    source = s.get("source", "Unknown")
                    content = s.get("content", "")
                    formatted_citations.append({"content": f"{source}: {content}"})
                elif hasattr(s, "model_dump"):
                    s_dict = s.model_dump()
                    source = s_dict.get("source", "Unknown")
                    content = s_dict.get("content", "")
                    formatted_citations.append({"content": f"{source}: {content}"})
                else:
                    formatted_citations.append({"content": str(s)})

            final_payload = {
                "node": "FINAL_RESULT",
                "answer": running_state.get("generation", "No generation produced by the pipeline."),
                "citations": formatted_citations
            }
            yield f"data: {json.dumps(final_payload)}\n\n"
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/v1/index", response_model=IndexResponse, tags=["Indexing"])
def index_document(request: IndexDocumentRequest) -> IndexResponse:
    """Chunk and index raw text content into local Chroma vector store and BM25 index."""
    if not scrag_app:
        raise HTTPException(status_code=503, detail="SCRAG engine not initialized")

    loader = DocumentLoader()
    try:
        chunks = loader.load_text(text=request.text, source=request.source)
        scrag_app.nodes.retriever.add_documents(chunks)

        return IndexResponse(
            status="success",
            indexed_chunks=len(chunks),
            source=request.source,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/benchmark", response_model=BenchmarkResponse, tags=["Evaluation"])
def run_benchmark() -> BenchmarkResponse:
    """Execute automated comparative benchmark comparing Naive RAG vs SCRAG."""
    if not scrag_app:
        raise HTTPException(status_code=503, detail="SCRAG engine not initialized")

    from scrag.eval.benchmark import run_comparative_benchmark

    results = run_comparative_benchmark(app=scrag_app, settings=scrag_app.settings)

    return BenchmarkResponse(
        status="completed",
        scenarios_evaluated=len(results),
        summary=f"Successfully evaluated {len(results)} scenarios across Naive RAG and SCRAG.",
    )
