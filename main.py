"""
RAG Application — Entry Point

This is the main entry point for the Retrieval-Augmented Generation pipeline.
Run this file to start the API server or execute the pipeline directly.
"""

import logging

import uvicorn
from dotenv import load_dotenv

from src.utils import load_config, setup_logging
from src.ingestion import DocumentLoader
from src.chunking import TextChunker
from src.embeddings import Embedder
from src.vectordb import VectorStore
from src.retrieval import Retriever
from src.prompts import PromptTemplates
from src.llm import LLMClient
from src.api import create_app

logger = logging.getLogger(__name__)


def build_pipeline(config: dict):
    """Build and return all RAG pipeline components from config."""
    # Ingestion
    loader = DocumentLoader()

    # Chunking
    chunk_cfg = config.get("chunking", {})
    chunker = TextChunker(
        chunk_size=chunk_cfg.get("chunk_size", 512),
        chunk_overlap=chunk_cfg.get("chunk_overlap", 50),
        method=chunk_cfg.get("method", "recursive"),
    )

    # Embeddings
    embed_cfg = config.get("embeddings", {})
    embedder = Embedder(
        provider=embed_cfg.get("provider", "openai"),
        model=embed_cfg.get("model", "text-embedding-3-small"),
        dimension=embed_cfg.get("dimension", 1536),
    )

    # Vector Store
    vec_cfg = config.get("vectordb", {})
    vector_store = VectorStore(
        provider=vec_cfg.get("provider", "chromadb"),
        collection_name=vec_cfg.get("collection_name", "rag_collection"),
        persist_directory=vec_cfg.get("persist_directory", "./data/vectordb"),
        dimension=embed_cfg.get("dimension", 1536),
    )

    # Retrieval
    ret_cfg = config.get("retrieval", {})
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=ret_cfg.get("top_k", 5),
        search_type=ret_cfg.get("search_type", "similarity"),
    )

    # LLM
    llm_cfg = config.get("llm", {})
    llm_client = LLMClient(
        provider=llm_cfg.get("provider", "openai"),
        model=llm_cfg.get("model", "gpt-4o-mini"),
        temperature=llm_cfg.get("temperature", 0.7),
        max_tokens=llm_cfg.get("max_tokens", 1024),
    )

    return {
        "loader": loader,
        "chunker": chunker,
        "embedder": embedder,
        "vector_store": vector_store,
        "retriever": retriever,
        "llm_client": llm_client,
    }


def main():
    """Main entry point."""
    # Load environment variables from .env
    load_dotenv()

    # Load config
    config = load_config("config.yaml")

    # Setup logging
    log_cfg = config.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("log_file", "logs/app.log"),
    )

    logger.info("Starting RAG application...")

    # Build pipeline components
    pipeline = build_pipeline(config)
    logger.info("Pipeline components initialized successfully")

    # Start API server with pipeline injected
    api_cfg = config.get("api", {})
    app = create_app(pipeline=pipeline)
    uvicorn.run(
        app,
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
    )


if __name__ == "__main__":
    main()
