"""
LLM Client Module

Handle LLM calls (OpenAI, Gemini, Claude).
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with Large Language Models."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        """Lazy-initialize the LLM client."""
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            return self._init_openai()
        elif self.provider == "gemini":
            return self._init_gemini()
        elif self.provider == "claude":
            return self._init_claude()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a response from the LLM."""
        client = self._get_client()

        logger.info(f"Generating response with {self.provider}/{self.model}")

        if self.provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content

        elif self.provider == "gemini":
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            response = client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )
            return response.text

        elif self.provider == "claude":
            messages = [{"role": "user", "content": prompt}]
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = client.messages.create(**kwargs)
            return response.content[0].text

        raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a response with conversation history."""
        client = self._get_client()

        if self.provider == "openai":
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)

            response = client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content

        elif self.provider == "gemini":
            # Gemini uses a chat session for multi-turn
            chat = client.start_chat(
                history=[
                    {"role": m["role"], "parts": [m["content"]]}
                    for m in messages[:-1]
                ]
            )
            response = chat.send_message(
                messages[-1]["content"],
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )
            return response.text

        elif self.provider == "claude":
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = client.messages.create(**kwargs)
            return response.content[0].text

        raise ValueError(f"Unsupported provider: {self.provider}")

    # -- Private initializers --

    def _init_openai(self):
        try:
            from openai import OpenAI

            self._client = OpenAI()
            logger.info(f"Initialized OpenAI client: {self.model}")
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
            self._client = genai.GenerativeModel(self.model)
            logger.info(f"Initialized Gemini client: {self.model}")
            return self._client
        except ImportError:
            raise ImportError(
                "google-generativeai package required. "
                "Install with: uv add google-generativeai"
            )

    def _init_claude(self):
        try:
            import anthropic

            self._client = anthropic.Anthropic()
            logger.info(f"Initialized Claude client: {self.model}")
            return self._client
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: uv add anthropic"
            )
