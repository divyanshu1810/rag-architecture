"""
API Routes Module

FastAPI endpoints for the RAG application.
"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# -- Request / Response models --

class QueryRequest(BaseModel):
    """Request model for RAG query."""
    question: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    answer: str
    sources: list[str]


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    file_path: Optional[str] = None
    directory_path: Optional[str] = None
    url: Optional[str] = None


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    status: str
    documents_processed: int
    chunks_created: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str


# -- App factory --

def create_app(pipeline: Optional[dict] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        pipeline: Dict containing the initialized RAG pipeline components:
            loader, chunker, embedder, vector_store, retriever, llm_client
    """
    app = FastAPI(
        title="RAG API",
        description="Retrieval-Augmented Generation API",
        version="0.1.0",
    )

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(status="healthy", version="0.1.0")

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        """Query the RAG system with a question."""
        if not pipeline:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline not initialized.",
            )

        retriever = pipeline["retriever"]
        llm_client = pipeline["llm_client"]

        try:
            # 1. Retrieve relevant chunks
            results = retriever.retrieve_with_scores(request.question)
            context_chunks = [r["content"] for r in results]

            if not context_chunks:
                return QueryResponse(
                    answer="No relevant documents found for your question.",
                    sources=[],
                )

            # 2. Build prompt from context + question
            from src.prompts import PromptTemplates

            prompt = PromptTemplates.format_rag_prompt(
                context_chunks=context_chunks,
                question=request.question,
            )
            system_prompt = PromptTemplates.get_system_prompt()

            # 3. Generate answer via LLM
            answer = llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
            )

            return QueryResponse(
                answer=answer,
                sources=context_chunks[:request.top_k],
            )

        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(request: IngestRequest):
        """Ingest documents into the vector store."""
        if not pipeline:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline not initialized.",
            )

        loader = pipeline["loader"]
        chunker = pipeline["chunker"]
        embedder = pipeline["embedder"]
        vector_store = pipeline["vector_store"]

        try:
            # 1. Load documents
            documents = []
            if request.file_path:
                doc = loader.load_file(request.file_path)
                documents.append(doc)
            elif request.directory_path:
                documents = loader.load_directory(request.directory_path)
            elif request.url:
                doc = loader.load_from_url(request.url)
                documents.append(doc)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Provide file_path, directory_path, or url.",
                )

            if not documents:
                return IngestResponse(
                    status="warning",
                    documents_processed=0,
                    chunks_created=0,
                )

            # 2. Chunk documents
            chunks = chunker.chunk_documents(documents)

            # 3. Generate embeddings
            embeddings = embedder.embed_batch(chunks)

            # 4. Store in vector DB
            vector_store.add_documents(
                texts=chunks,
                embeddings=embeddings,
            )

            logger.info(
                f"Ingested {len(documents)} docs → {len(chunks)} chunks"
            )

            return IngestResponse(
                status="success",
                documents_processed=len(documents),
                chunks_created=len(chunks),
            )

        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return app
