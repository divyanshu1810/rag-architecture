"""
Retriever Module

Retrieve relevant chunks using similarity search against the vector store.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieve relevant document chunks for a given query."""

    def __init__(
        self,
        embedder=None,
        vector_store=None,
        top_k: int = 5,
        search_type: str = "similarity",
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.search_type = search_type

    def retrieve(self, query: str) -> List[str]:
        """Retrieve the most relevant chunks for a query."""
        if not self.embedder or not self.vector_store:
            raise RuntimeError(
                "Retriever requires both an Embedder and a VectorStore"
            )

        logger.info(f"Retrieving top {self.top_k} chunks for query: {query[:80]}...")

        # 1. Embed the query
        query_embedding = self.embedder.embed_text(query)

        # 2. Search the vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        # 3. Extract documents from results
        documents = results.get("documents", [[]])[0]
        logger.info(f"Retrieved {len(documents)} relevant chunks")
        return documents

    def retrieve_with_scores(
        self, query: str
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks along with relevance scores."""
        if not self.embedder or not self.vector_store:
            raise RuntimeError(
                "Retriever requires both an Embedder and a VectorStore"
            )

        query_embedding = self.embedder.embed_text(query)
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        scored_results = []
        for doc, dist, meta in zip(documents, distances, metadatas):
            scored_results.append({
                "content": doc,
                "distance": dist,
                "metadata": meta,
            })

        return scored_results
