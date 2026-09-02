"""
LLM service with pluggable backends and a strict fallback chain.

Backends: "groq" (primary) -> "hf_api" (fallback) -> "mock" (safety net).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from utils.logger import get_logger

logger = get_logger(__name__)


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...


class RateLimitError(Exception):
    """Raised when a backend is unavailable and the next backend should be tried."""


class MockLLM(BaseLLM):
    def generate(self, prompt: str) -> str:
        query_line = ""
        for line in prompt.splitlines():
            if line.strip().startswith("QUERY:"):
                query_line = line.strip().replace("QUERY:", "").strip()
                break
        return (
            f'{{"final_answer": "Based on available evidence, research on {query_line!r} '
            f'shows supportive findings.", '
            f'"key_claims": ["Evidence supports the primary hypothesis.", '
            f'"Some variability exists across studies.", '
            f'"Further research may be needed."], '
            f'"summary": "The retrieved studies suggest a generally supportive pattern '
            f'for the query topic. Multiple papers report positive associations, though '
            f'some variability exists. Interpret findings with appropriate caution."}}'
        )


class GroqLLM(BaseLLM):
    _API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model_id: str) -> None:
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._model_id = model_id
        self._client = httpx.Client(timeout=60.0)
        logger.info(f"Groq LLM backend ready: {model_id}")

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.3,
        }
        try:
            response = self._client.post(self._API_URL, json=payload, headers=self._headers)
            if response.status_code in (401, 429, 503):
                logger.warning(f"Groq {response.status_code} -> triggering fallback")
                raise RateLimitError(f"Groq unavailable: {response.status_code}")
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except RateLimitError:
            raise
        except Exception as exc:
            logger.warning(f"Groq request failed: {exc} -> triggering fallback")
            raise RateLimitError(str(exc))


class HuggingFaceAPILLM(BaseLLM):
    _API_URL = "https://router.huggingface.co/v1/chat/completions"

    def __init__(self, model_id: str, api_token: str) -> None:
        self._url = self._API_URL
        self._model_id = model_id
        self._headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=90.0)
        logger.info(f"HuggingFace router backend ready: {model_id}")

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.3,
        }
        try:
            response = self._client.post(self._url, json=payload, headers=self._headers)
            if response.status_code in (400, 401, 403, 404, 410, 429, 503):
                logger.warning(f"HuggingFace router {response.status_code} -> triggering fallback to mock")
                raise RateLimitError(f"HF API unavailable: {response.status_code} {response.text[:200]}")
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except RateLimitError:
            raise
        except Exception as exc:
            logger.warning(f"HuggingFace router failed: {exc} -> triggering fallback to mock")
            raise RateLimitError(str(exc))


class FallbackLLM(BaseLLM):
    """Tries backends in order. MockLLM is expected to be the final backend."""

    def __init__(self, backends: list[BaseLLM]) -> None:
        self._backends = backends

    def generate(self, prompt: str) -> str:
        for backend in self._backends:
            try:
                result = backend.generate(prompt)
                if result:
                    return result
            except RateLimitError as exc:
                logger.info(f"Backend {type(backend).__name__} failed ({exc}), trying next...")
                continue
        logger.error("All LLM backends failed")
        return ""


def _has_real_secret(value: str) -> bool:
    return bool(value and "your_" not in value.lower())


def build_llm() -> BaseLLM:
    from app.config import get_settings

    settings = get_settings()
    backend = settings.LLM_BACKEND
    mock = MockLLM()

    if backend == "groq":
        backends: list[BaseLLM] = []

        if _has_real_secret(settings.GROQ_API_KEY):
            backends.append(GroqLLM(api_key=settings.GROQ_API_KEY, model_id=settings.GROQ_MODEL_ID))
        else:
            logger.warning("GROQ_API_KEY missing or placeholder -> skipping Groq backend")

        if _has_real_secret(settings.HF_API_TOKEN):
            backends.append(
                HuggingFaceAPILLM(model_id=settings.HF_API_MODEL_ID, api_token=settings.HF_API_TOKEN)
            )

        if not backends:
            logger.warning("No live LLM credentials configured -> using Mock")
            return mock

        backends.append(mock)
        logger.info(f"LLM strategy: {' -> '.join(type(b).__name__ for b in backends)}")
        return FallbackLLM(backends=backends)

    if backend == "hf_api":
        if not _has_real_secret(settings.HF_API_TOKEN):
            logger.warning("HF_API_TOKEN missing or placeholder -> using Mock")
            return mock
        return FallbackLLM(
            backends=[
                HuggingFaceAPILLM(model_id=settings.HF_API_MODEL_ID, api_token=settings.HF_API_TOKEN),
                mock,
            ]
        )

    logger.info("LLM backend: Mock")
    return mock
