"""
Vector Store Module

Handle vector database operations (ChromaDB, Pinecone, FAISS).
"""

import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """Manage vector storage and retrieval."""

    def __init__(
        self,
        provider: str = "chromadb",
        collection_name: str = "rag_collection",
        persist_directory: str = "./data/vectordb",
        dimension: int = 1536,
    ):
        self.provider = provider
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.dimension = dimension
        self._collection = None
        self._index = None  # For FAISS
        self._documents = []  # For FAISS document store
        self._metadatas = []  # For FAISS metadata store
        self._ids = []  # For FAISS id store

    def _get_collection(self):
        """Initialize and return the vector DB collection."""
        if self._collection is not None or self._index is not None:
            return self._collection or self._index

        if self.provider == "chromadb":
            return self._init_chromadb()
        elif self.provider == "faiss":
            return self._init_faiss()
        elif self.provider == "pinecone":
            return self._init_pinecone()
        else:
            raise ValueError(f"Unsupported vector DB provider: {self.provider}")

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add documents with their embeddings to the vector store."""
        self._get_collection()

        if ids is None:
            existing = len(self._ids) if self.provider == "faiss" else 0
            ids = [f"doc_{existing + i}" for i in range(len(texts))]

        if metadatas is None:
            metadatas = [{} for _ in texts]

        if self.provider == "chromadb":
            self._collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

        elif self.provider == "faiss":
            vectors = np.array(embeddings, dtype="float32")
            self._index.add(vectors)
            self._documents.extend(texts)
            self._metadatas.extend(metadatas)
            self._ids.extend(ids)
            self._save_faiss()

        elif self.provider == "pinecone":
            vectors = [
                {
                    "id": doc_id,
                    "values": emb,
                    "metadata": {**(meta or {}), "text": text},
                }
                for doc_id, emb, text, meta in zip(
                    ids, embeddings, texts, metadatas
                )
            ]
            # Pinecone upsert in batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                self._collection.upsert(vectors=batch)

        logger.info(f"Added {len(texts)} documents to {self.provider} vector store")

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Query the vector store for similar documents.

        Returns a dict with keys: documents, distances, metadatas, ids
        (matching ChromaDB's format for consistency).
        """
        self._get_collection()

        if self.provider == "chromadb":
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
            return results

        elif self.provider == "faiss":
            query_vec = np.array([query_embedding], dtype="float32")
            distances, indices = self._index.search(query_vec, top_k)

            docs = []
            metas = []
            result_ids = []
            for idx in indices[0]:
                if idx < len(self._documents) and idx >= 0:
                    docs.append(self._documents[idx])
                    metas.append(self._metadatas[idx])
                    result_ids.append(self._ids[idx])

            return {
                "documents": [docs],
                "distances": [distances[0].tolist()],
                "metadatas": [metas],
                "ids": [result_ids],
            }

        elif self.provider == "pinecone":
            results = self._collection.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
            )

            docs = []
            distances = []
            metas = []
            result_ids = []
            for match in results.get("matches", []):
                meta = match.get("metadata", {})
                docs.append(meta.pop("text", ""))
                distances.append(match.get("score", 0.0))
                metas.append(meta)
                result_ids.append(match.get("id", ""))

            return {
                "documents": [docs],
                "distances": [distances],
                "metadatas": [metas],
                "ids": [result_ids],
            }

        raise ValueError(f"Unsupported provider: {self.provider}")

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        logger.warning(f"Deleting collection: {self.collection_name}")

        if self.provider == "chromadb":
            try:
                import chromadb

                client = chromadb.PersistentClient(path=self.persist_directory)
                client.delete_collection(name=self.collection_name)
                self._collection = None
            except Exception as e:
                logger.error(f"Failed to delete ChromaDB collection: {e}")

        elif self.provider == "faiss":
            self._index = None
            self._documents = []
            self._metadatas = []
            self._ids = []
            # Remove persisted files
            faiss_dir = Path(self.persist_directory) / self.collection_name
            for f in ["index.faiss", "store.pkl"]:
                p = faiss_dir / f
                if p.exists():
                    p.unlink()

        elif self.provider == "pinecone":
            try:
                self._collection.delete(delete_all=True)
            except Exception as e:
                logger.error(f"Failed to delete Pinecone index: {e}")

    def count(self) -> int:
        """Return the number of documents in the store."""
        self._get_collection()

        if self.provider == "chromadb":
            return self._collection.count()
        elif self.provider == "faiss":
            return self._index.ntotal
        elif self.provider == "pinecone":
            stats = self._collection.describe_index_stats()
            return stats.get("total_vector_count", 0)

        return 0

    # -- Private initializers --

    def _init_chromadb(self):
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = client.get_or_create_collection(
                name=self.collection_name
            )
            logger.info(
                f"ChromaDB collection '{self.collection_name}' ready "
                f"at {self.persist_directory}"
            )
            return self._collection
        except ImportError:
            raise ImportError(
                "chromadb package required. Install with: uv add chromadb"
            )

    def _init_faiss(self):
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu package required. Install with: uv add faiss-cpu"
            )

        faiss_dir = Path(self.persist_directory) / self.collection_name
        faiss_dir.mkdir(parents=True, exist_ok=True)

        index_path = faiss_dir / "index.faiss"
        store_path = faiss_dir / "store.pkl"

        if index_path.exists() and store_path.exists():
            # Load existing index
            self._index = faiss.read_index(str(index_path))
            with open(store_path, "rb") as f:
                store = pickle.load(f)
            self._documents = store["documents"]
            self._metadatas = store["metadatas"]
            self._ids = store["ids"]
            logger.info(
                f"Loaded FAISS index from {faiss_dir} "
                f"({self._index.ntotal} vectors)"
            )
        else:
            # Create new index
            self._index = faiss.IndexFlatL2(self.dimension)
            self._documents = []
            self._metadatas = []
            self._ids = []
            logger.info(
                f"Created new FAISS index (dim={self.dimension}) "
                f"at {faiss_dir}"
            )

        return self._index

    def _save_faiss(self):
        """Persist FAISS index and document store to disk."""
        try:
            import faiss
        except ImportError:
            return

        faiss_dir = Path(self.persist_directory) / self.collection_name
        faiss_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(faiss_dir / "index.faiss"))
        with open(faiss_dir / "store.pkl", "wb") as f:
            pickle.dump(
                {
                    "documents": self._documents,
                    "metadatas": self._metadatas,
                    "ids": self._ids,
                },
                f,
            )
        logger.debug("FAISS index saved to disk")

    def _init_pinecone(self):
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError(
                "pinecone package required. Install with: uv add pinecone"
            )

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY environment variable is required"
            )

        pc = Pinecone(api_key=api_key)

        # Check if index exists, create if not
        existing = [idx.name for idx in pc.list_indexes()]
        if self.collection_name not in existing:
            pc.create_index(
                name=self.collection_name,
                dimension=self.dimension,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            )
            logger.info(f"Created Pinecone index: {self.collection_name}")

        self._collection = pc.Index(self.collection_name)
        logger.info(f"Connected to Pinecone index: {self.collection_name}")
        return self._collection
