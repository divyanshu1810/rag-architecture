"""
Tests for the RAG application.
"""

from src.chunking import TextChunker


class TestTextChunker:
    """Test suite for the TextChunker."""

    def test_character_split(self):
        chunker = TextChunker(chunk_size=20, chunk_overlap=5, method="character")
        text = "This is a test string that should be split into multiple chunks."
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
        assert all(len(c) <= 20 for c in chunks)

    def test_empty_text(self):
        chunker = TextChunker()
        chunks = chunker.chunk_text("")
        assert chunks == []

    def test_recursive_split(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10, method="recursive")
        text = (
            "First paragraph content here.\n\n"
            "Second paragraph with more content.\n\n"
            "Third paragraph to ensure splitting."
        )
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1

    def test_chunk_documents(self):
        chunker = TextChunker(chunk_size=30, chunk_overlap=5, method="character")
        docs = ["Short doc.", "Another short document for testing."]
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) >= 2
