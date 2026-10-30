"""LLM provider interface and the Google Gemini-backed implementation.

The intent parser depends only on `LLMProvider`, so tests can substitute a
fake provider and never make network calls. The ``google-genai`` SDK is
imported lazily so the rest of CamAgent works without it installed, and the
MCP server starts without an API key: the key is only read on the first
``parse_request`` call.

The provider is a thin transport. It sends the system prompt, the user's text
and the response schema to Gemini and returns the model's text. It never
interprets, validates or executes anything; that stays in deterministic
Python code (app/llm/intent.py and the safety layers behind it).
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Protocol

DEFAULT_MODEL = "gemini-3.8-flash"
MODEL_ENV_VAR = "CAMAGENT_MODEL"
API_KEY_ENV_VAR = "GEMINI_API_KEY"
# The google-genai SDK itself also reads GOOGLE_API_KEY; accept it as a
# fallback so a key set under that name is not reported as "not set".
FALLBACK_API_KEY_ENV_VAR = "GOOGLE_API_KEY"

# Finish reasons that mean Gemini refused or filtered the reply. Anything
# other than a normal stop is treated as a failure: the parser must reject
# rather than guess from partial output.
_DECLINED_FINISH_REASONS = frozenset(
    {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}
)
_NORMAL_FINISH_REASONS = frozenset({"", "STOP", "FINISH_REASON_UNSPECIFIED"})


class LLMProviderError(RuntimeError):
    """Raised when the provider cannot return a usable completion."""


class LLMProvider(Protocol):
    """Anything that can turn a system prompt and user text into raw text."""

    def complete(
        self,
        system: str,
        prompt: str,
        response_schema: Mapping[str, Any] | None = None,
    ) -> str: ...


def _enum_name(value: Any) -> str:
    """Name of an SDK enum member, or the plain string the SDK may return."""
    if value is None:
        return ""
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    return str(value).upper()


def _api_key_from_env() -> str:
    """The configured API key, or an empty string. Never logged."""
    for name in (API_KEY_ENV_VAR, FALLBACK_API_KEY_ENV_VAR):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _redact(text: str, secret: str | None) -> str:
    """Make sure an API key never appears in an error message."""
    if secret and secret in text:
        return text.replace(secret, "***")
    return text


def _describe_failure(exc: BaseException) -> str:
    """Turn an SDK exception into a message that says what to fix."""
    code = getattr(exc, "code", None)
    message = getattr(exc, "message", None)
    status = getattr(exc, "status", None)
    detail = message if isinstance(message, str) and message else str(exc) or type(exc).__name__
    if isinstance(code, int):
        # The Gemini API reports an invalid key as HTTP 400 ("API key not
        # valid"), a key without access as 403, and some proxies as 401.
        looks_like_auth = (
            code in (401, 403)
            or status in ("UNAUTHENTICATED", "PERMISSION_DENIED")
            or "api key" in detail.lower()
            or "API_KEY" in detail
        )
        if looks_like_auth:
            return (
                f"Gemini API authentication failed (HTTP {code}); "
                f"check {API_KEY_ENV_VAR}: {detail}"
            )
        if code == 429:
            return f"Gemini API rate limit or quota exceeded (HTTP 429): {detail}"
        if code >= 500:
            return f"Gemini API is unavailable (HTTP {code}): {detail}"
        return f"Gemini API rejected the request (HTTP {code}): {detail}"
    return f"LLM request failed: {type(exc).__name__}: {detail}"


def _extract_text(response: Any) -> str:
    """Return the text of the first candidate, or raise LLMProviderError."""
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = _enum_name(getattr(feedback, "block_reason", None))
    if block_reason and block_reason != "BLOCKED_REASON_UNSPECIFIED":
        raise LLMProviderError(f"LLM declined to answer the request (prompt blocked: {block_reason})")

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise LLMProviderError("LLM returned an empty response")
    candidate = candidates[0]

    finish_reason = _enum_name(getattr(candidate, "finish_reason", None))
    if finish_reason == "MAX_TOKENS":
        raise LLMProviderError("LLM response was truncated")
    if finish_reason in _DECLINED_FINISH_REASONS:
        raise LLMProviderError(f"LLM declined to answer the request (finish reason: {finish_reason})")
    if finish_reason not in _NORMAL_FINISH_REASONS:
        raise LLMProviderError(f"LLM stopped unexpectedly (finish reason: {finish_reason})")

    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []
    text = "".join(
        part.text
        for part in parts
        if isinstance(getattr(part, "text", None), str) and not getattr(part, "thought", False)
    )
    if not text.strip():
        raise LLMProviderError("LLM returned an empty response")
    return text


class GeminiProvider:
    """Call Gemini through the official ``google-genai`` SDK.

    The API key is read from the ``GEMINI_API_KEY`` environment variable on
    first use and passed to the SDK client; it is never logged or included in
    error messages. The model defaults to ``gemini-3.8-flash`` and can be
    overridden with the ``CAMAGENT_MODEL`` environment variable or the
    ``model`` argument. A pre-built ``client`` can be injected for tests.
    """

    def __init__(
        self,
        model: str | None = None,
        client: Any = None,
        max_output_tokens: int = 2048,
    ) -> None:
        self.model = model or os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL
        self.max_output_tokens = max_output_tokens
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise LLMProviderError(
                    "the 'google-genai' package is required for GeminiProvider; "
                    "install it with 'pip install google-genai'"
                ) from exc
            api_key = _api_key_from_env()
            if not api_key:
                raise LLMProviderError(
                    f"{API_KEY_ENV_VAR} is not set in the environment of the process "
                    "running CamAgent (an MCP client must pass it in its env block, "
                    f"or MCP Inspector with -e {API_KEY_ENV_VAR}=...); create a key "
                    "in Google AI Studio"
                )
            try:
                self._client = genai.Client(api_key=api_key)
            except Exception as exc:
                raise LLMProviderError(
                    _redact(f"could not create the Gemini client: {exc}", api_key)
                ) from exc
        return self._client

    def complete(
        self,
        system: str,
        prompt: str,
        response_schema: Mapping[str, Any] | None = None,
    ) -> str:
        client = self._get_client()
        config: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": self.max_output_tokens,
        }
        if response_schema is not None:
            # Constrained decoding: Gemini must reply with JSON matching the
            # envelope schema. The parser still validates every field itself.
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = dict(response_schema)

        secret = _api_key_from_env()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:  # SDK errors vary by version; surface uniformly.
            raise LLMProviderError(_redact(_describe_failure(exc), secret)) from exc

        return _extract_text(response)
