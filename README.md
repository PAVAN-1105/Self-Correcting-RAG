# Self-Correcting RAG (SCRAG) Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

> **SCRAG** is an advanced, production-grade **Self-Correcting Retrieval-Augmented Generation** engine. It unites pre-generation document grading and granular knowledge strip refinement (**CRAG**) with post-generation dual-axis self-reflection (**Self-RAG**) inside a stateful, cyclical **LangGraph** architecture.

---

## 1. Why SCRAG?

Traditional RAG systems follow a naive, linear pipeline:
$$\text{Query} \longrightarrow \text{Dense Vector Retrieval} \longrightarrow \text{Prompt Augmentation} \longrightarrow \text{LLM Output}$$

In production, linear RAG fails silently due to:
1. **The Distraction Problem (Retriever Noise)**: Low-quality retrievers inject tangential facts. Studies show LLMs are easily distracted by irrelevant context, causing them to hallucinate.
2. **Context Contamination**: Whole chunks (500–1000 tokens) are dumped into prompts, containing 80% non-essential noise.
3. **Knowledge Boundary Failure**: If the internal vector database lacks answers, standard RAG forces generation from poor similarity matches.
4. **Generation Hallucination & Fact Drift**: Models frequently hallucinate unsupported claims. Standard RAG provides zero post-generation verification.

### The SCRAG Solution
- **Pre-Generation Evaluation**: A dedicated `DocumentGrader` evaluates retrieval confidence (`CORRECT`, `INCORRECT`, `AMBIGUOUS`).
  - **Correct**: Decomposes documents into atomic 1–2 sentence knowledge strips and filters out noise ($k_{in}$).
  - **Incorrect**: Rewrites the query and triggers autonomous web search fallback ($k_{ex}$).
  - **Ambiguous**: Combines filtered internal strips and web search ($k_{in} + k_{ex}$).
- **Post-Generation Reflection**: A dual-axis `Critic` validates the generated draft before user delivery:
  - **Faithfulness (Hallucination Check)**: Confirms every claim is entailed by the evidence.
  - **Answer Relevance (Utility Check)**: Confirms the response directly resolves user intent.
  - **Self-Healing Loop**: If unfaithful or irrelevant, SCRAG loops back with surgical feedback to regenerate or re-retrieve. A circuit breaker (`max_retries <= 2`) prevents infinite recursion.

---

## 2. Local Models vs. Cloud APIs: In-Depth Comparison

SCRAG includes a unified **Execution Mode Toggle** (`local`, `api`, or `auto`):

| Evaluation Dimension | Local Mode (Ollama + Sentence-Transformers) | Cloud API Mode (OpenAI + Tavily) |
| :--- | :--- | :--- |
| **Cost** | **100% Free** (Zero API fees, unlimited queries) | Pay-per-token (3–6 calls per query can accumulate) |
| **Data Privacy** | **100% Private** (All data stays on your local machine) | Data transmitted to external servers |
| **Hardware Required** | Requires 8GB–16GB RAM / Apple Silicon Metal GPU | Runs on any lightweight machine (cloud offloaded) |
| **Offline Capability** | **100% Offline** (Zero internet required for local corpus) | Requires constant, low-latency internet connection |
| **Query Latency** | ~3–8s on Apple Silicon (sequential local model calls) | ~1–3s (frontier data center throughput) |
| **Reasoning Fidelity** | High (e.g., Qwen 2.5 7B/14B, Llama 3.1 8B) | State-of-the-art (`gpt-4o`, `gpt-4o-mini`) |
| **Structured Output** | JSON schema adherence with regex fallback | Strict JSON mode native enforcement |

### Recommendation
* **Default Recommended Mode: `auto`**:
  * If `OPENAI_API_KEY` is present in your environment, SCRAG seamlessly leverages cloud models for maximum throughput.
  * If no API key is detected, SCRAG **automatically runs 100% locally on Ollama and sentence-transformers with zero setup!**

---

## 3. Project Architecture

```text
scrag/
├── __init__.py            # Primary exports (SelfCorrectingRAGApp, SCRAGSettings)
├── config.py              # Configuration with local/cloud toggle and thresholds
├── state.py               # Typed GraphState, KnowledgeStrip, and Critique models
├── logging.py             # Rich console formatting and audit trail logger
├── llm_factory.py         # Dynamic model factory (Ollama vs. OpenAI)
│
├── retrieval/             # Stage 1: Document Ingestion & Hybrid Search
│   ├── loaders.py         # Multi-format document chunker
│   ├── dense.py           # ChromaDB local vector store
│   ├── sparse.py          # BM25 keyword indexer
│   └── hybrid.py          # Reciprocal Rank Fusion (RRF)
│
├── evaluator/             # Stage 2: Pre-generation Document Grading & Strip Refinement
│   ├── grader.py          # Relevance evaluator (Correct / Incorrect / Ambiguous)
│   ├── refiner.py         # Decompose-then-recompose knowledge strip engine
│   └── rewriter.py        # Search query reformulation
│
├── search/                # Stage 3: External Search Fallback
│   └── web.py             # DuckDuckGo (Free, zero API key) + Tavily fallback
│
├── generator/             # Stage 4: Grounded Answer Synthesis
│   └── generator.py       # Strict synthesis from knowledge strips with [1], [2] citations
│
├── critic/                # Stage 5: Post-generation Reflection
│   ├── faithfulness.py    # Hallucination critic (checks context entailment)
│   └── relevance.py       # Answer relevance critic (checks query intent fulfillment)
│
├── graph/                 # Stage 6: LangGraph Cyclical State Machine
│   ├── nodes.py           # SCRAG pipeline nodes
│   ├── edges.py           # Conditional routers & circuit breaker
│   └── workflow.py        # Graph assembly & runner
│
├── api/                   # Stage 7: FastAPI REST API & SSE Streaming
│   ├── schemas.py
│   └── server.py
│
└── cli/                   # Stage 8: Interactive Terminal Runner
    └── main.py
```

---

## 4. Quickstart Guide

### Prerequisites
- Python 3.11+
- (Optional for Local Mode): [Ollama](https://ollama.com/) installed with any model:
  ```bash
  ollama pull qwen2.5-coder:7b
  ollama pull qwen3:8b
  ```

### Installation
```bash
# 1. Clone repository
git clone <repo_url>
cd Self_Correcting_Rag

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration (`.env`)
Create a `.env` file (or copy from `.env.example`):
```bash
# Execution mode: "auto" (default), "local", or "api"
SCRAG_EXECUTION_MODE=auto

# Local Ollama Settings (Zero API keys needed!)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_GENERATOR_MODEL=qwen3:8b
OLLAMA_CRITIC_MODEL=qwen2.5-coder:7b

# Cloud OpenAI Settings (Optional)
OPENAI_API_KEY=your_key_here
OPENAI_GENERATOR_MODEL=gpt-4o
OPENAI_CRITIC_MODEL=gpt-4o-mini
```

---

## 5. Usage

### 1. Interactive Terminal Session (CLI)
```bash
./.venv/bin/python -m scrag.cli.main interactive
```

### 2. Ask a Single Question
```bash
./.venv/bin/python -m scrag.cli.main query "What is LangGraph and how does it support cyclical self-correction?"
```

### 3. Index Documents into Vector Store
```bash
./.venv/bin/python -m scrag.cli.main index path/to/document.txt
```

### 4. Run Comparative Benchmark (Naive RAG vs. SCRAG)
```bash
./.venv/bin/python -m scrag.cli.main benchmark
```

### 5. Launch FastAPI REST Server
```bash
./.venv/bin/python -m scrag.cli.main serve --port 8000
```
Then visit the interactive OpenAPI Swagger documentation at: **`http://localhost:8000/docs`**

---

## 6. Testing & Verification

Run the comprehensive unit and integration test suite:
```bash
./.venv/bin/pytest tests/ -v
```

All 24 test cases validate:
- Configuration validation and auto-mode detection
- Ingestion and chunking metadata preservation
- ChromaDB dense vector indexing and BM25 sparse keyword indexing
- Reciprocal Rank Fusion (RRF)
- Pre-generation CRAG evaluation (Correct, Incorrect, Ambiguous)
- Decompose-then-recompose knowledge strip extraction
- Grounded generation with citation tags
- Post-generation Self-RAG reflection (Faithfulness and Relevance)
- LangGraph cyclical state machine routing and circuit breaker retries
- FastAPI REST endpoints (`/health`, `/api/v1/index`, `/api/v1/query`)
