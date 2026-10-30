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

# Bounded exponential backoff for transient failures, applied by the SDK's
# own retry support (``HttpOptions.retry_options``), not a custom loop. Google
# documents 408, 429 and 5xx as the responses worth retrying; 4 attempts with
# delays of about 1s, 2s and 4s (plus jitter, capped at 10s) keeps a bad
# outage to roughly 20 seconds before a provider_error is returned.
RETRY_ATTEMPTS = 4  # including the first request
RETRY_INITIAL_DELAY_SECONDS = 1.0
RETRY_MAX_DELAY_SECONDS = 10.0
RETRY_EXP_BASE = 2.0
RETRY_JITTER_SECONDS = 1.0  # uniform random extra delay in [0, jitter]
RETRYABLE_HTTP_STATUS_CODES = (408, 429, *range(500, 600))

# Per-attempt HTTP timeout. The SDK's default is *no* timeout, so a request
# that Gemini accepts but never answers would block the calling thread (and
# an MCP tool call) forever, and the retry sequence would never start. With
# a timeout each attempt gives up, counts as transient and is retried, so the
# whole call is bounded by ``max_request_duration_seconds()``.
REQUEST_TIMEOUT_SECONDS = 20.0

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


def retry_options() -> dict[str, Any]:
    """SDK ``HttpRetryOptions`` for the Gemini client, as a plain dict."""
    return {
        "attempts": RETRY_ATTEMPTS,
        "initial_delay": RETRY_INITIAL_DELAY_SECONDS,
        "max_delay": RETRY_MAX_DELAY_SECONDS,
        "exp_base": RETRY_EXP_BASE,
        "jitter": RETRY_JITTER_SECONDS,
        "http_status_codes": list(RETRYABLE_HTTP_STATUS_CODES),
    }


def http_options() -> dict[str, Any]:
    """SDK ``HttpOptions`` for the Gemini client: timeout plus retry policy."""
    return {
        "timeout": int(REQUEST_TIMEOUT_SECONDS * 1000),  # the SDK takes milliseconds
        "retry_options": retry_options(),
    }


def max_request_duration_seconds() -> float:
    """Upper bound on one ``complete()`` call: every attempt times out and
    every backoff wait is as long as the policy allows."""
    waits = sum(
        min(RETRY_INITIAL_DELAY_SECONDS * RETRY_EXP_BASE**i + RETRY_JITTER_SECONDS, RETRY_MAX_DELAY_SECONDS)
        for i in range(RETRY_ATTEMPTS - 1)
    )
    return RETRY_ATTEMPTS * REQUEST_TIMEOUT_SECONDS + waits


def _is_http_timeout(exc: BaseException) -> bool:
    """True for the HTTP client's timeout exceptions (httpx), checked by name
    so the SDK and its HTTP client need not be imported here."""
    return any(
        cls.__module__.split(".")[0] == "httpx" and cls.__name__.endswith(("Timeout", "TimeoutException"))
        for cls in type(exc).__mro__
    )


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
    if _is_http_timeout(exc):
        return (
            f"Gemini API did not respond within {REQUEST_TIMEOUT_SECONDS:g}s per attempt "
            f"({RETRY_ATTEMPTS} attempts made): {type(exc).__name__}: {detail}"
        )
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

    Each HTTP attempt is limited to ``REQUEST_TIMEOUT_SECONDS``. Transient
    failures (HTTP 408, 429 and 5xx, timeouts and connection errors) are
    retried by the SDK with bounded exponential backoff (see
    ``retry_options``), so one call never takes longer than
    ``max_request_duration_seconds()``. Once the attempts are exhausted the
    last error is reported as an ``LLMProviderError`` exactly as before; a
    failure is never turned into a successful completion.
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
                self._client = genai.Client(api_key=api_key, http_options=http_options())
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
