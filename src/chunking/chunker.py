"""
Text Chunker Module

Split text into small, overlapping chunks for embedding.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class TextChunker:
    """Split documents into smaller chunks for processing."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        method: str = "recursive",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.method = method

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using the configured method."""
        if not text or not text.strip():
            return []

        logger.info(
            f"Chunking text ({len(text)} chars) with method={self.method}, "
            f"size={self.chunk_size}, overlap={self.chunk_overlap}"
        )

        if self.method == "recursive":
            return self._recursive_split(text)
        elif self.method == "character":
            return self._character_split(text)
        elif self.method == "token":
            return self._token_split(text)
        else:
            raise ValueError(f"Unknown chunking method: {self.method}")

    def chunk_documents(self, documents: List[str]) -> List[str]:
        """Chunk a list of documents."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc)
            all_chunks.extend(chunks)

        logger.info(
            f"Created {len(all_chunks)} chunks from {len(documents)} documents"
        )
        return all_chunks

    # -- Private methods --

    def _recursive_split(self, text: str) -> List[str]:
        """Split text recursively by paragraphs, then sentences, then characters."""
        separators = ["\n\n", "\n", ". ", " ", ""]
        return self._split_with_separators(text, separators)

    def _character_split(self, text: str) -> List[str]:
        """Simple character-based splitting with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        return chunks

    def _token_split(self, text: str) -> List[str]:
        """Split text by approximate token count (whitespace-based)."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        return chunks

    def _split_with_separators(
        self, text: str, separators: List[str]
    ) -> List[str]:
        """Recursively split text using a hierarchy of separators."""
        if not separators:
            return self._character_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        if not separator:
            return self._character_split(text)

        parts = text.split(separator)
        chunks = []
        current_chunk = ""

        for part in parts:
            candidate = (
                current_chunk + separator + part if current_chunk else part
            )

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if len(part) > self.chunk_size:
                    sub_chunks = self._split_with_separators(
                        part, remaining_separators
                    )
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk and current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks
