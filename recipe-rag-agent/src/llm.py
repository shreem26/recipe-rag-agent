"""
llm.py
------
Provider-agnostic LLM wrapper used for the generation step of the RAG
pipeline.

Defaults to **xAI Grok**, which is OpenAI-SDK compatible. Because everything goes through
the OpenAI client with a configurable base_url, you can point this at
any OpenAI-compatible provider by changing two env vars:

    xAI Grok (default):
        LLM_BASE_URL=https://api.x.ai/v1
        LLM_MODEL=grok-3-mini

    OpenRouter (has free models):
        LLM_BASE_URL=https://openrouter.ai/api/v1
        LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free

    Ollama (fully local, no key needed):
        LLM_BASE_URL=http://localhost:11434/v1
        LLM_MODEL=llama3.1
        LLM_API_KEY=ollama

    OpenAI:
        LLM_BASE_URL=https://api.openai.com/v1
        LLM_MODEL=gpt-4o-mini
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-3-mini"


class LLMNotConfigured(RuntimeError):
    """Raised when no API key is available."""


# Backwards-compatible alias (earlier versions of this project were
# Anthropic-only and raised ClaudeNotConfigured).
ClaudeNotConfigured = LLMNotConfigured


def get_base_url() -> str:
    return os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)


def get_model() -> str:
    return os.environ.get("LLM_MODEL", DEFAULT_MODEL)


def get_api_key() -> str:
    """
    Look for a key in LLM_API_KEY first, then fall back to the common
    provider-specific variable names so an existing key just works.
    """
    for var in ("LLM_API_KEY", "XAI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        key = os.environ.get(var)
        if key and not key.startswith("your-"):
            return key
    raise LLMNotConfigured(
        "No API key found.\n\n"
        "Get an xAI API key at https://console.x.ai/, "
        "then copy .env.example to .env and set:\n"
        "    LLM_API_KEY=xai-your-api-key-here\n\n"
        "See README.md for other free options (OpenRouter, or fully local Ollama)."
    )


def get_client():
    from openai import OpenAI

    return OpenAI(api_key=get_api_key(), base_url=get_base_url())


def generate(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    max_tokens: int = 2000,
    temperature: float = 0.4,
) -> str:
    """Single-turn completion call. Returns the assistant's text reply."""
    client = get_client()
    response = client.chat.completions.create(
        model=model or get_model(),
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()
