"""
Prompt Templates Module

Store and manage prompt templates for the RAG pipeline.
"""

from typing import List, Optional


class PromptTemplates:
    """Collection of prompt templates for the RAG system."""

    RAG_SYSTEM_PROMPT = (
        "You are a helpful assistant that answers questions based on the "
        "provided context. If the context does not contain enough information "
        "to answer the question, say so clearly. Do not make up information."
    )

    RAG_USER_PROMPT = (
        "Context:\n"
        "---\n"
        "{context}\n"
        "---\n\n"
        "Question: {question}\n\n"
        "Answer the question based only on the context provided above."
    )

    CONVERSATIONAL_PROMPT = (
        "Context:\n"
        "---\n"
        "{context}\n"
        "---\n\n"
        "Chat History:\n"
        "{chat_history}\n\n"
        "Question: {question}\n\n"
        "Answer the question based on the context and chat history."
    )

    @classmethod
    def format_rag_prompt(
        cls, context_chunks: List[str], question: str
    ) -> str:
        """Format the RAG prompt with context and question."""
        context = "\n\n".join(context_chunks)
        return cls.RAG_USER_PROMPT.format(
            context=context, question=question
        )

    @classmethod
    def format_conversational_prompt(
        cls,
        context_chunks: List[str],
        question: str,
        chat_history: Optional[str] = None,
    ) -> str:
        """Format prompt with context, question, and chat history."""
        context = "\n\n".join(context_chunks)
        history = chat_history or "No previous conversation."
        return cls.CONVERSATIONAL_PROMPT.format(
            context=context, question=question, chat_history=history
        )

    @classmethod
    def get_system_prompt(cls) -> str:
        """Return the system prompt for RAG."""
        return cls.RAG_SYSTEM_PROMPT
