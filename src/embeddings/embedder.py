"""
Embedder Module

Convert text chunks into vector embeddings using various providers.
Supports OpenAI, Google Gemini, and HuggingFace sentence-transformers.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings for text chunks."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
    ):
        self.provider = provider
        self.model = model
        self.dimension = dimension
        self._client = None

    def _get_client(self):
        """Lazy-initialize the embedding client."""
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            return self._init_openai()
        elif self.provider == "gemini":
            return self._init_gemini()
        elif self.provider == "huggingface":
            return self._init_huggingface()
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        client = self._get_client()

        if self.provider == "openai":
            response = client.embeddings.create(
                input=text,
                model=self.model,
            )
            embedding = response.data[0].embedding

        elif self.provider == "gemini":
            result = client.embed_content(
                model=f"models/{self.model}",
                content=text,
            )
            embedding = result["embedding"]

        elif self.provider == "huggingface":
            embedding = client.encode(text).tolist()

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        logger.debug(f"Generated embedding of dimension {len(embedding)}")
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of text strings."""
        if not texts:
            return []

        client = self._get_client()
        logger.info(f"Embedding batch of {len(texts)} texts with {self.provider}")

        if self.provider == "openai":
            response = client.embeddings.create(
                input=texts,
                model=self.model,
            )
            embeddings = [item.embedding for item in response.data]

        elif self.provider == "gemini":
            embeddings = []
            for text in texts:
                result = client.embed_content(
                    model=f"models/{self.model}",
                    content=text,
                )
                embeddings.append(result["embedding"])

        elif self.provider == "huggingface":
            import numpy as np

            vectors = client.encode(texts)
            embeddings = [v.tolist() for v in vectors]

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    # -- Private initializers --

    def _init_openai(self):
        try:
            from openai import OpenAI

            self._client = OpenAI()
            logger.info(f"Initialized OpenAI embedding client: {self.model}")
            return self._client
        except ImportError:
            raise ImportError(
                "openai package required. Install with: uv add openai"
            )

    def _init_gemini(self):
        try:
            import google.generativeai as genai
            import os

            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY environment variable is required for Gemini"
                )

            genai.configure(api_key=api_key)
            self._client = genai
            logger.info(f"Initialized Gemini embedding client: {self.model}")
            return self._client
        except ImportError:
            raise ImportError(
                "google-generativeai package required. "
                "Install with: uv add google-generativeai"
            )

    def _init_huggingface(self):
        try:
            from sentence_transformers import SentenceTransformer

            self._client = SentenceTransformer(self.model)
            self.dimension = self._client.get_sentence_embedding_dimension()
            logger.info(
                f"Initialized HuggingFace embedding client: {self.model} "
                f"(dim={self.dimension})"
            )
            return self._client
        except ImportError:
            raise ImportError(
                "sentence-transformers package required. "
                "Install with: uv add sentence-transformers"
            )
