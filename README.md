# RAG Architecture

A modular Retrieval-Augmented Generation (RAG) pipeline built in Python.

## Project Structure

```
rag/
├── README.md              — Project description, setup guide, how to run
├── pyproject.toml         — Project metadata & dependencies (uv/pip)
├── config.yaml            — Configuration for models, chunk size, DB settings
├── .env                   — API keys and secrets (not pushed to GitHub)
├── .gitignore             — Git ignore rules
│
├── src/                   — Application source code
│   ├── ingestion/         — Load data from PDFs, CSVs, websites
│   │   └── loader.py
│   ├── chunking/          — Split text into small chunks
│   │   └── chunker.py
│   ├── embeddings/        — Convert text chunks into embeddings
│   │   └── embedder.py
│   ├── vectordb/          — Vector database operations (ChromaDB, Pinecone, FAISS)
│   │   └── vector_store.py
│   ├── retrieval/         — Retrieve relevant chunks via similarity search
│   │   └── retriever.py
│   ├── prompts/           — Prompt templates
│   │   └── prompt_templates.py
│   ├── llm/               — LLM calls (OpenAI, Gemini, Claude)
│   │   └── llm_client.py
│   ├── api/               — API endpoints (FastAPI)
│   │   └── routes.py
│   └── utils/             — Helper functions & common utilities
│       └── helpers.py
│
├── tests/                 — Unit tests and integration tests
│   └── test_app.py
├── logs/                  — Log files for debugging and monitoring
│   └── app.log
└── main.py                — Entry point of the application
```

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure

- Copy `.env` and fill in your API keys
- Edit `config.yaml` to choose your LLM provider, embedding model, vector DB, etc.

### 3. Run the API server

```bash
uv run main.py
```

The API will be available at `http://localhost:8000`. Check health at `/health`.

### 4. Run tests

```bash
uv run pytest tests/
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `ingestion` | Load documents from PDFs, CSVs, text files, URLs |
| `chunking` | Split documents into overlapping chunks |
| `embeddings` | Generate vector embeddings via OpenAI / HuggingFace |
| `vectordb` | Store & query vectors in ChromaDB, Pinecone, or FAISS |
| `retrieval` | Retrieve top-k relevant chunks for a query |
| `prompts` | Manage and format prompt templates |
| `llm` | Call LLMs (OpenAI GPT, Google Gemini, Anthropic Claude) |
| `api` | FastAPI endpoints for querying and ingestion |
| `utils` | Config loading, logging setup, helper functions |

## License

MIT
