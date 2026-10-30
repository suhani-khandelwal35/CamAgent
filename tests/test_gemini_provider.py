"""Tests for the Gemini-backed provider and the parse_request path built on it.

Nothing here needs a Gemini API key, network access or Linux module
utilities. The google-genai client is replaced by an in-memory fake that
records each request and returns scripted responses or raises scripted
exceptions, and every attempt to run a system command fails the test.

Covered: successful parsing for all six actions, ambiguous and unsupported
requests, missing and hallucinated module names, invalid JSON, malformed and
empty Gemini responses, authentication, rate-limit and connection failures,
the deterministic safety validation that runs after parsing, and the SDK's
bounded retry of transient HTTP failures (exercised with the real SDK over an
in-process mock transport, with sleeping replaced by a recorder).
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from mcp import types as mcp_types

from app.agent import ActionExecutor
from app.llm import GeminiProvider, IntentParser, LLMProviderError
from app.llm.intent import RESPONSE_SCHEMA, SYSTEM_PROMPT
from app.llm.provider import (
    API_KEY_ENV_VAR,
    DEFAULT_MODEL,
    FALLBACK_API_KEY_ENV_VAR,
    MODEL_ENV_VAR,
    RETRY_ATTEMPTS,
    RETRY_EXP_BASE,
    RETRY_INITIAL_DELAY_SECONDS,
    RETRY_JITTER_SECONDS,
    RETRY_MAX_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    http_options,
    max_request_duration_seconds,
    retry_options,
)
from app.manager import module_manager
from app.mcp_server import TOOL_SPECS, ToolDispatcher
from app.mcp_server.server import create_server
from app.safety import DEFAULT_PROTECTED_MODULES, SafetyPolicy, validate_raw
from app.schemas import ALLOWED_OPERATIONS

SIX_ACTIONS = {
    "list_modules",
    "get_module_info",
    "check_module",
    "load_module",
    "unload_module",
    "reload_module",
}


# --------------------------------------------------------------------------- #
# Fakes and helpers
# --------------------------------------------------------------------------- #


class FakeGeminiClient:
    """Stands in for ``google.genai.Client``.

    ``generate_content`` records its keyword arguments, then raises ``error``
    if given, otherwise returns the next scripted response.
    """

    def __init__(self, responses: list[Any] | None = None, error: BaseException | None = None) -> None:
        self.responses = list(responses or [])
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if not self.responses:
            raise AssertionError("the fake Gemini client has no scripted response left")
        return self.responses.pop(0)


class FakeAPIError(Exception):
    """Shaped like ``google.genai.errors.APIError`` without importing the SDK."""

    def __init__(self, code: int, message: str, status: str = "") -> None:
        super().__init__(f"{code} {status}. {message}")
        self.code = code
        self.message = message
        self.status = status


def gemini_response(
    text: str | None = None,
    *,
    finish_reason: Any = "STOP",
    parts: list[Any] | None = None,
    candidates: list[Any] | None = None,
    prompt_feedback: Any = None,
) -> SimpleNamespace:
    """Build an object shaped like ``GenerateContentResponse``."""
    if candidates is None:
        if parts is None:
            parts = [] if text is None else [SimpleNamespace(text=text, thought=None)]
        reason = SimpleNamespace(name=finish_reason) if isinstance(finish_reason, str) else finish_reason
        candidates = [SimpleNamespace(finish_reason=reason, content=SimpleNamespace(parts=parts))]
    return SimpleNamespace(candidates=candidates, prompt_feedback=prompt_feedback)


def envelope(operation: str | None = None, module: str | None = None, status: str = "action", reason: str = "ok") -> dict:
    action = None if operation is None else {"operation": operation, "module": module}
    return {"status": status, "action": action, "reason": reason}


def reply(*args: Any, **kwargs: Any) -> SimpleNamespace:
    """A normal Gemini response carrying one JSON envelope."""
    return gemini_response(json.dumps(envelope(*args, **kwargs)))


def parse(request: str, *responses: Any, error: BaseException | None = None):
    client = FakeGeminiClient(list(responses), error)
    parser = IntentParser(GeminiProvider(model="test-model", client=client))
    return parser.parse(request), client


def assert_rejected(result, status: str) -> None:
    assert result.ok is False
    assert result.status == status
    assert result.action is None
    assert result.to_dict()["action"] is None


class FakeModuleManager:
    """In-memory module manager so the executor never touches the kernel."""

    def __init__(self) -> None:
        self.loaded = {"snd", "ext4"}
        self.calls: list[tuple[str, str | None]] = []

    def list_modules(self):
        self.calls.append(("list_modules", None))
        return [{"name": module} for module in sorted(self.loaded)]

    def get_module_info(self, module: str):
        self.calls.append(("get_module_info", module))
        return {"success": True, "name": module}

    def is_module_loaded(self, module: str) -> bool:
        self.calls.append(("is_module_loaded", module))
        return module in self.loaded

    def load_module(self, module: str):
        self.calls.append(("load_module", module))
        self.loaded.add(module)
        return {"success": True, "operation": "load"}

    def unload_module(self, module: str):
        self.calls.append(("unload_module", module))
        self.loaded.discard(module)
        return {"success": True, "operation": "unload"}

    def reload_module(self, module: str):
        self.calls.append(("reload_module", module))
        return {"success": True, "operation": "reload"}

    def state_changes(self) -> list[tuple[str, str | None]]:
        return [c for c in self.calls if c[0] in {"load_module", "unload_module", "reload_module"}]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """No real key, no real model override, and no system commands."""
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    monkeypatch.delenv(FALLBACK_API_KEY_ENV_VAR, raising=False)
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)

    def explode(*args, **kwargs):  # pragma: no cover - only runs on a regression
        raise AssertionError(f"parsing must not run commands: {args!r} {kwargs!r}")

    monkeypatch.setattr(module_manager, "_run_command", explode)
    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, explode)


@pytest.fixture
def manager():
    return FakeModuleManager()


@pytest.fixture
def executor(manager):
    return ActionExecutor(manager=manager, policy=SafetyPolicy(is_module_loaded=manager.is_module_loaded))


# --------------------------------------------------------------------------- #
# Request shape sent to Gemini
# --------------------------------------------------------------------------- #


def test_request_carries_system_prompt_user_text_and_schema():
    result, client = parse("is uvcvideo loaded?", reply("check_module", "uvcvideo"))

    assert result.ok
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "test-model"
    assert call["contents"] == "is uvcvideo loaded?"
    config = call["config"]
    assert config["system_instruction"] == SYSTEM_PROMPT
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"] == dict(RESPONSE_SCHEMA)
    assert config["max_output_tokens"] > 0
    # The provider never asks Gemini for tools or function calling.
    assert "tools" not in config
    assert "tool_config" not in config


def test_default_model_is_a_gemini_model_and_env_overrides_it(monkeypatch):
    assert DEFAULT_MODEL.startswith("gemini-")
    assert GeminiProvider(client=object()).model == DEFAULT_MODEL

    monkeypatch.setenv(MODEL_ENV_VAR, "gemini-custom")
    assert GeminiProvider(client=object()).model == "gemini-custom"
    assert GeminiProvider(model="explicit", client=object()).model == "explicit"


def test_schema_is_omitted_when_not_requested():
    client = FakeGeminiClient([gemini_response("plain text")])

    assert GeminiProvider(client=client).complete("sys", "hello") == "plain text"
    config = client.calls[0]["config"]
    assert "response_mime_type" not in config
    assert "response_json_schema" not in config


def test_text_parts_are_concatenated_and_thought_parts_skipped():
    parts = [
        SimpleNamespace(text="reasoning", thought=True),
        SimpleNamespace(text='{"a":', thought=None),
        SimpleNamespace(inline_data=b"\x00"),
        SimpleNamespace(text=" 1}", thought=False),
    ]
    client = FakeGeminiClient([gemini_response(parts=parts)])

    assert GeminiProvider(client=client).complete("sys", "hello") == '{"a": 1}'


def test_finish_reason_may_be_a_plain_string():
    client = FakeGeminiClient([gemini_response("ok", finish_reason=SimpleNamespace(name=None))])
    client.responses[0].candidates[0].finish_reason = "STOP"

    assert GeminiProvider(client=client).complete("sys", "hello") == "ok"


# --------------------------------------------------------------------------- #
# Successful parsing: all six actions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("request_text", "operation", "module", "expected"),
    [
        ("list the loaded modules", "list_modules", None, {"operation": "list_modules"}),
        ("show info about uvcvideo", "get_module_info", "uvcvideo", {"operation": "get_module_info", "module": "uvcvideo"}),
        ("is uvcvideo loaded?", "check_module", "uvcvideo", {"operation": "check_module", "module": "uvcvideo"}),
        ("load uvcvideo", "load_module", "uvcvideo", {"operation": "load_module", "module": "uvcvideo"}),
        ("unload the webcam driver", "unload_module", "uvcvideo", {"operation": "unload_module", "module": "uvcvideo"}),
        ("reload the usb camera driver", "reload_module", "uvcvideo", {"operation": "reload_module", "module": "uvcvideo"}),
    ],
)
def test_each_supported_action_parses(request_text, operation, module, expected):
    result, client = parse(request_text, reply(operation, module))

    assert result.ok
    assert result.status == "parsed"
    assert result.action.to_dict() == expected
    assert result.to_dict() == {
        "ok": True,
        "status": "parsed",
        "message": result.message,
        "action": expected,
        "errors": [],
    }
    assert len(client.calls) == 1


def test_exactly_six_actions_are_allowed():
    assert set(ALLOWED_OPERATIONS) == SIX_ACTIONS
    assert set(RESPONSE_SCHEMA["properties"]["action"]["anyOf"][0]["properties"]["operation"]["enum"]) == SIX_ACTIONS


@pytest.mark.parametrize(
    "operation",
    ["run_shell", "install_module", "set_module_parameter", "modprobe", "LIST_MODULES", "list modules", ""],
)
def test_operations_outside_the_six_are_rejected(operation):
    result, _ = parse("do something with uvcvideo", reply(operation, "uvcvideo"))

    assert_rejected(result, "invalid_output")
    assert result.errors


def test_module_names_are_normalised_by_the_validator():
    result, _ = parse("is snd-hda-intel loaded?", reply("check_module", "snd-hda-intel"))

    assert result.ok
    assert result.action.module == "snd_hda_intel"


# --------------------------------------------------------------------------- #
# Ambiguous, unsupported and missing-module requests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("request_text", "reason"),
    [
        ("do something with the driver", "no module was named"),
        ("reload uvcvideo and snd", "several modules"),
        ("load or unload uvcvideo", "several operations"),
    ],
)
def test_ambiguous_requests_are_reported_without_an_action(request_text, reason):
    result, client = parse(request_text, reply(status="ambiguous", reason=reason))

    assert_rejected(result, "ambiguous")
    assert reason in result.message
    assert len(client.calls) == 1  # no retry


@pytest.mark.parametrize(
    "request_text",
    ["run rm -rf / for me", "install the nvidia driver package", "set the uvcvideo quirks parameter", "cat /etc/passwd"],
)
def test_unsupported_requests_are_reported_without_an_action(request_text):
    result, client = parse(request_text, reply(status="unsupported", reason="outside the supported operations"))

    assert_rejected(result, "unsupported")
    assert "not supported" in result.message
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "action",
    [
        {"operation": "check_module", "module": None},
        {"operation": "check_module", "module": ""},
        {"operation": "check_module", "module": "   "},
        {"operation": "check_module"},
        {"operation": "unload_module", "module": None},
    ],
)
def test_missing_module_name_is_rejected(action):
    result, _ = parse("check it", gemini_response(json.dumps({"status": "action", "action": action, "reason": "x"})))

    assert_rejected(result, "invalid_output")
    assert result.errors


@pytest.mark.parametrize(
    ("request_text", "module"),
    [
        ("is my sound card working?", "snd_hda_intel"),
        ("check the camera", "uvcvideo"),  # "camera" alone is not a documented alias
        ("load the wifi driver", "iwlwifi"),
        ("is the uvcvideo2 module loaded", "uvcvideo"),
    ],
)
def test_hallucinated_module_names_are_rejected(request_text, module):
    result, _ = parse(request_text, reply("check_module", module))

    assert_rejected(result, "ambiguous")
    assert module in result.message
    assert result.errors == [f"module {module!r} is not mentioned in the request"]


@pytest.mark.parametrize("phrase", ["webcam", "web cam", "usb camera", "uvc camera", "uvc driver", "WEBCAM"])
def test_documented_device_aliases_ground_the_module(phrase):
    result, _ = parse(f"is the {phrase} loaded?", reply("check_module", "uvcvideo"))

    assert result.ok
    assert result.action.module == "uvcvideo"


# --------------------------------------------------------------------------- #
# Invalid JSON and malformed Gemini replies
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "{",
        "{'status': 'action'}",
        '{"status": "action",}',
        "[]",
        "null",
        '"check_module uvcvideo"',
        "I would call check_module on uvcvideo.",
    ],
)
def test_invalid_json_is_rejected(text):
    result, _ = parse("is uvcvideo loaded", gemini_response(text))

    assert_rejected(result, "invalid_output")
    assert result.errors


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "action", "action": {"operation": "check_module", "module": "uvcvideo"}},
        {**envelope("check_module", "uvcvideo"), "command": "lsmod"},
        {**envelope("check_module", "uvcvideo"), "shell": "modprobe -r uvcvideo"},
        envelope("check_module", "uvcvideo", status="maybe"),
        {"status": "action", "action": None, "reason": "x"},
        {"status": "action", "action": "check_module uvcvideo", "reason": "x"},
        {"status": "action", "action": ["check_module", "uvcvideo"], "reason": "x"},
        envelope("check_module", "uvcvideo", status="ambiguous"),
        {"status": "action", "action": {"operation": "check_module", "module": "uvcvideo"}, "reason": None},
        {"status": ["action"], "action": None, "reason": "x"},
        {"status": "action", "action": {"operation": "check_module", "module": "uvcvideo", "args": "-v"}, "reason": "x"},
    ],
)
def test_malformed_envelopes_are_rejected(payload):
    result, _ = parse("is uvcvideo loaded", gemini_response(json.dumps(payload)))

    assert_rejected(result, "invalid_output")
    assert result.errors


@pytest.mark.parametrize(
    "module",
    ["uvcvideo; rm -rf /", "$(reboot)", "uvcvideo && reboot", "../../evil", "--force", "/lib/modules/x"],
)
def test_shell_like_module_names_never_pass_validation(module):
    result, _ = parse(f"load {module}", reply("load_module", module))

    assert_rejected(result, "invalid_output")
    assert result.errors


# --------------------------------------------------------------------------- #
# Empty, truncated and blocked Gemini responses
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "response",
    [
        gemini_response(candidates=[]),
        SimpleNamespace(candidates=None, prompt_feedback=None),
        gemini_response(parts=[]),
        gemini_response(parts=None),
        gemini_response(""),
        gemini_response("   \n"),
        gemini_response(parts=[SimpleNamespace(text="thinking only", thought=True)]),
        gemini_response(candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"), content=None)]),
        gemini_response(parts=[SimpleNamespace(inline_data=b"\x00")]),
        gemini_response(parts=[SimpleNamespace(text=None)]),
    ],
)
def test_empty_responses_are_provider_errors(response):
    result, _ = parse("list modules", response)

    assert_rejected(result, "provider_error")
    assert result.errors == ["LLM returned an empty response"]


def test_truncated_response_is_a_provider_error():
    result, _ = parse("list modules", gemini_response('{"status": "act', finish_reason="MAX_TOKENS"))

    assert_rejected(result, "provider_error")
    assert result.errors == ["LLM response was truncated"]


@pytest.mark.parametrize("finish_reason", ["SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"])
def test_filtered_candidates_are_provider_errors(finish_reason):
    result, _ = parse("list modules", gemini_response("{}", finish_reason=finish_reason))

    assert_rejected(result, "provider_error")
    assert "declined" in result.errors[0]


def test_blocked_prompt_is_a_provider_error():
    feedback = SimpleNamespace(block_reason=SimpleNamespace(name="SAFETY"))
    result, _ = parse("list modules", gemini_response("{}", prompt_feedback=feedback))

    assert_rejected(result, "provider_error")
    assert "prompt blocked: SAFETY" in result.errors[0]


def test_unspecified_block_reason_is_not_a_block():
    feedback = SimpleNamespace(block_reason=SimpleNamespace(name="BLOCKED_REASON_UNSPECIFIED"))
    result, _ = parse("list modules", gemini_response(json.dumps(envelope("list_modules")), prompt_feedback=feedback))

    assert result.ok


@pytest.mark.parametrize("finish_reason", ["OTHER", "LANGUAGE", "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"])
def test_unexpected_finish_reasons_are_provider_errors(finish_reason):
    result, _ = parse("list modules", gemini_response(json.dumps(envelope("list_modules")), finish_reason=finish_reason))

    assert_rejected(result, "provider_error")
    assert finish_reason in result.errors[0]


# --------------------------------------------------------------------------- #
# Authentication, rate limits and connection failures
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("code", [401, 403])
def test_authentication_failures_name_the_key_variable(code):
    error = FakeAPIError(code, "API key not valid. Please pass a valid API key.", "UNAUTHENTICATED")
    result, client = parse("list modules", error=error)

    assert_rejected(result, "provider_error")
    assert result.message == "The language model was unavailable."
    assert "authentication failed" in result.errors[0]
    assert API_KEY_ENV_VAR in result.errors[0]
    assert "API key not valid" in result.errors[0]
    assert len(client.calls) == 1


def test_real_sdk_client_error_is_reported_as_provider_error():
    errors = pytest.importorskip("google.genai.errors")
    error = errors.ClientError(401, {"error": {"message": "API key not valid", "status": "UNAUTHENTICATED"}})

    result, _ = parse("list modules", error=error)

    assert_rejected(result, "provider_error")
    assert "authentication failed (HTTP 401)" in result.errors[0]


def test_rate_limit_is_reported():
    error = FakeAPIError(429, "Quota exceeded for requests per minute", "RESOURCE_EXHAUSTED")
    result, _ = parse("list modules", error=error)

    assert_rejected(result, "provider_error")
    assert result.errors[0].startswith("Gemini API rate limit or quota exceeded (HTTP 429)")
    assert "Quota exceeded" in result.errors[0]


@pytest.mark.parametrize("code", [500, 503])
def test_server_errors_are_reported(code):
    result, _ = parse("list modules", error=FakeAPIError(code, "overloaded", "UNAVAILABLE"))

    assert_rejected(result, "provider_error")
    assert f"unavailable (HTTP {code})" in result.errors[0]


def test_invalid_key_reported_as_http_400_is_an_authentication_failure():
    # The Gemini API answers an invalid key with HTTP 400 INVALID_ARGUMENT.
    error = FakeAPIError(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT")
    result, _ = parse("list modules", error=error)

    assert_rejected(result, "provider_error")
    assert "authentication failed" in result.errors[0]
    assert API_KEY_ENV_VAR in result.errors[0]


def test_other_client_errors_are_reported():
    result, _ = parse("list modules", error=FakeAPIError(400, "Unknown name 'response_json_schema'", "INVALID_ARGUMENT"))

    assert_rejected(result, "provider_error")
    assert "rejected the request (HTTP 400)" in result.errors[0]


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("network down"),
        TimeoutError("timed out"),
        OSError("Name or service not known"),
        RuntimeError("TLS handshake failed"),
    ],
)
def test_connection_failures_are_provider_errors(error):
    result, client = parse("list modules", error=error)

    assert_rejected(result, "provider_error")
    assert result.errors[0].startswith("LLM request failed: " + type(error).__name__)
    assert str(error) in result.errors[0]
    assert len(client.calls) == 1  # never retried


def test_missing_api_key_is_reported_before_any_network_use(monkeypatch):
    genai = pytest.importorskip("google.genai")

    def no_client(*args, **kwargs):  # pragma: no cover - only runs on a regression
        raise AssertionError("the SDK client must not be created without a key")

    monkeypatch.setattr(genai, "Client", no_client)

    result = IntentParser(GeminiProvider()).parse("list modules")

    assert_rejected(result, "provider_error")
    assert API_KEY_ENV_VAR in result.errors[0]
    assert "Google AI Studio" in result.errors[0]


def test_api_key_is_read_from_env_and_never_echoed(monkeypatch):
    genai = pytest.importorskip("google.genai")
    secret = "AIza-test-secret-key"
    monkeypatch.setenv(API_KEY_ENV_VAR, secret)
    created: dict[str, Any] = {}

    class FakeSDKClient(FakeGeminiClient):
        def __init__(self, *, api_key: str, http_options: Any = None) -> None:
            created["api_key"] = api_key
            super().__init__(error=RuntimeError(f"request to https://x/?key={secret} failed"))

    monkeypatch.setattr(genai, "Client", FakeSDKClient)

    result = IntentParser(GeminiProvider()).parse("list modules")

    assert created["api_key"] == secret
    assert_rejected(result, "provider_error")
    assert secret not in result.errors[0]
    assert "***" in result.errors[0]
    assert secret not in json.dumps(result.to_dict())


@pytest.mark.parametrize(
    ("gemini_value", "google_value", "expected"),
    [
        ("gemini-key", None, "gemini-key"),
        (None, "google-key", "google-key"),
        ("gemini-key", "google-key", "gemini-key"),
        ("  gemini-key \n", None, "gemini-key"),
        ("   ", "google-key", "google-key"),
    ],
)
def test_key_lookup_prefers_gemini_then_google_variable(monkeypatch, gemini_value, google_value, expected):
    genai = pytest.importorskip("google.genai")
    if gemini_value is not None:
        monkeypatch.setenv(API_KEY_ENV_VAR, gemini_value)
    if google_value is not None:
        monkeypatch.setenv(FALLBACK_API_KEY_ENV_VAR, google_value)
    created: dict[str, Any] = {}

    class FakeSDKClient(FakeGeminiClient):
        def __init__(self, *, api_key: str, http_options: Any = None) -> None:
            created["api_key"] = api_key
            super().__init__([reply("list_modules")])

    monkeypatch.setattr(genai, "Client", FakeSDKClient)

    assert IntentParser(GeminiProvider()).parse("list modules").ok
    assert created["api_key"] == expected


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_blank_key_counts_as_missing(monkeypatch, value):
    genai = pytest.importorskip("google.genai")
    monkeypatch.setenv(API_KEY_ENV_VAR, value)

    def no_client(*args, **kwargs):  # pragma: no cover - only runs on a regression
        raise AssertionError("the SDK client must not be created without a key")

    monkeypatch.setattr(genai, "Client", no_client)

    result = IntentParser(GeminiProvider()).parse("list modules")

    assert_rejected(result, "provider_error")
    assert result.errors[0].startswith(f"{API_KEY_ENV_VAR} is not set")
    assert "-e GEMINI_API_KEY=" in result.errors[0]


def test_client_is_created_once_and_reused(monkeypatch):
    genai = pytest.importorskip("google.genai")
    monkeypatch.setenv(API_KEY_ENV_VAR, "k")
    instances: list[FakeGeminiClient] = []

    class FakeSDKClient(FakeGeminiClient):
        def __init__(self, *, api_key: str, http_options: Any = None) -> None:
            super().__init__([reply("list_modules"), reply("list_modules")])
            instances.append(self)

    monkeypatch.setattr(genai, "Client", FakeSDKClient)
    parser = IntentParser(GeminiProvider())

    assert parser.parse("list modules").ok
    assert parser.parse("list modules").ok
    assert len(instances) == 1


def test_missing_sdk_is_reported(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "google" or name.startswith("google."):
            raise ImportError("No module named 'google'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv(API_KEY_ENV_VAR, "k")
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(LLMProviderError, match="google-genai"):
        GeminiProvider().complete("sys", "hello")


# --------------------------------------------------------------------------- #
# Bounded retry of transient failures (real SDK, in-process mock transport)
# --------------------------------------------------------------------------- #


def _ok_body(operation: str = "list_modules", module: str | None = None) -> dict:
    text = json.dumps(envelope(operation, module))
    return {"candidates": [{"content": {"parts": [{"text": text}], "role": "model"}, "finishReason": "STOP"}]}


def _error_body(code: int, message: str, status: str) -> dict:
    return {"error": {"code": code, "message": message, "status": status}}


class RecordingTransport:
    """Scripted HTTP responses for the SDK's httpx client; no sockets involved.

    ``script`` is a list of ``(status_code, json_body)``; the last entry is
    repeated once the script runs out. Every request is recorded.
    """

    def __init__(self, script: list[tuple[int, dict]]) -> None:
        self.script = list(script)
        self.requests: list[Any] = []

    def handler(self, request):
        import httpx

        self.requests.append(request)
        status, body = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return httpx.Response(status, json=body, request=request)


@pytest.fixture
def sdk_transport(monkeypatch):
    """Route the provider's real SDK client through a scripted transport.

    The provider still builds the client itself (key lookup, retry options),
    so the whole ``_get_client`` path is exercised; only the HTTP layer and
    ``time.sleep`` are replaced. Yields ``(install, sleeps, created)``:
    ``install(script)`` returns the RecordingTransport, ``sleeps`` lists the
    backoff delays the SDK asked for, ``created`` records the http_options
    the provider passed to the SDK client.
    """
    genai = pytest.importorskip("google.genai")
    httpx = pytest.importorskip("httpx")
    import time

    real_client = genai.Client
    sleeps: list[float] = []
    created: dict[str, Any] = {}

    def install(script: list[tuple[int, dict]]) -> RecordingTransport:
        transport = RecordingTransport(script)

        def patched_client(*, api_key: str, http_options: dict | None = None, **kwargs: Any):
            created["http_options"] = dict(http_options or {})
            options = dict(http_options or {})
            options["client_args"] = {"transport": httpx.MockTransport(transport.handler)}
            return real_client(api_key=api_key, http_options=options, **kwargs)

        monkeypatch.setattr(genai, "Client", patched_client)
        return transport

    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))
    return install, sleeps, created


def _assert_backoff_is_bounded(sleeps: list[float]) -> None:
    for index, delay in enumerate(sleeps):
        base = RETRY_INITIAL_DELAY_SECONDS * RETRY_EXP_BASE**index
        assert 0 < delay <= RETRY_MAX_DELAY_SECONDS
        assert min(base, RETRY_MAX_DELAY_SECONDS) <= delay
        assert delay <= min(base + RETRY_JITTER_SECONDS, RETRY_MAX_DELAY_SECONDS)


def test_retry_options_match_the_documented_policy(sdk_transport, monkeypatch):
    install, _, created = sdk_transport
    monkeypatch.setenv(API_KEY_ENV_VAR, "k")
    install([(200, _ok_body())])

    assert IntentParser(GeminiProvider()).parse("list modules").ok

    assert created["http_options"] == http_options()
    options = created["http_options"]["retry_options"]
    assert options == retry_options()
    assert options["attempts"] == RETRY_ATTEMPTS == 4
    assert options["initial_delay"] == 1.0
    assert options["max_delay"] == 10.0
    assert options["exp_base"] == 2.0
    assert options["jitter"] > 0
    assert {408, 429, 500, 502, 503, 504, 599} <= set(options["http_status_codes"])
    assert not {400, 401, 403, 404} & set(options["http_status_codes"])


def test_one_503_then_success_is_parsed(sdk_transport, monkeypatch):
    install, sleeps, _ = sdk_transport
    monkeypatch.setenv(API_KEY_ENV_VAR, "k")
    overloaded = (503, _error_body(503, "The model is overloaded.", "UNAVAILABLE"))
    transport = install([overloaded, (200, _ok_body())])

    result = IntentParser(GeminiProvider()).parse("list modules")

    assert result.ok and result.action.operation == "list_modules"
    assert len(transport.requests) == 2
    assert len(sleeps) == 1
    _assert_backoff_is_bounded(sleeps)


def test_two_503s_then_success_is_parsed(sdk_transport, monkeypatch):
    install, sleeps, _ = sdk_transport
    monkeypatch.setenv(API_KEY_ENV_VAR, "k")
    overloaded = (503, _error_body(503, "The model is overloaded.", "UNAVAILABLE"))
    transport = install([overloaded, overloaded, (200, _ok_body("check_module", "uvcvideo"))])

    result = IntentParser(GeminiProvider()).parse("is uvcvideo loaded?")

    assert result.ok and result.action.operation == "check_module" and result.action.module == "uvcvideo"
    assert len(transport.requests) == 3
    assert len(sleeps) == 2
    _assert_backoff_is_bounded(sleeps)
    assert sleeps[1] > sleeps[0]


def test_persistent_503_stops_after_four_attempts_with_a_provider_error(sdk_transport, monkeypatch):
    install, sleeps, _ = sdk_transport
    monkeypatch.setenv(API_KEY_ENV_VAR, "k")
    transport = install([(503, _error_body(503, "The model is overloaded.", "UNAVAILABLE"))])

    result = IntentParser(GeminiProvider()).parse("list modules")

    assert_rejected(result, "provider_error")
    assert "Gemini API is unavailable (HTTP 503)" in result.errors[0]
    assert "overloaded" in result.errors[0]
    assert len(transport.requests) == RETRY_ATTEMPTS == 4
    assert len(sleeps) == RETRY_ATTEMPTS - 1
    _assert_backoff_is_bounded(sleeps)
    assert sum(sleeps) <= (RETRY_ATTEMPTS - 1) * RETRY_MAX_DELAY_SECONDS


def test_429_is_retried_then_reported_if_it_persists(sdk_transport, monkeypatch):
    install, sleeps, _ = sdk_transport
    monkeypatch.setenv(API_KEY_ENV_VAR, "k")
    limited = (429, _error_body(429, "Resource has been exhausted", "RESOURCE_EXHAUSTED"))

    transport = install([limited, (200, _ok_body())])
    assert IntentParser(GeminiProvider()).parse("list modules").ok
    assert len(transport.requests) == 2 and len(sleeps) == 1

    sleeps.clear()
    transport = install([limited])
    result = IntentParser(GeminiProvider()).parse("list modules")
    assert_rejected(result, "provider_error")
    assert "rate limit or quota exceeded (HTTP 429)" in result.errors[0]
    assert len(transport.requests) == RETRY_ATTEMPTS
    _assert_backoff_is_bounded(sleeps)


@pytest.mark.parametrize(
    "code, status", [(400, "INVALID_ARGUMENT"), (403, "PERMISSION_DENIED"), (404, "NOT_FOUND")]
)
def test_non_transient_client_errors_are_not_retried(sdk_transport, monkeypatch, code, status):
    install, sleeps, _ = sdk_transport
    monkeypatch.setenv(API_KEY_ENV_VAR, "k")
    transport = install([(code, _error_body(code, "bad request", status)), (200, _ok_body())])

    result = IntentParser(GeminiProvider()).parse("list modules")

    assert_rejected(result, "provider_error")
    assert f"(HTTP {code})" in result.errors[0]
    assert len(transport.requests) == 1
    assert sleeps == []


def test_invalid_key_400_is_not_retried_and_names_the_variable(sdk_transport, monkeypatch):
    install, sleeps, _ = sdk_transport
    monkeypatch.setenv(API_KEY_ENV_VAR, "not-a-real-key")
    body = _error_body(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT")
    transport = install([(400, body)])

    result = IntentParser(GeminiProvider()).parse("list modules")

    assert_rejected(result, "provider_error")
    assert result.errors[0].startswith("Gemini API authentication failed (HTTP 400)")
    assert API_KEY_ENV_VAR in result.errors[0]
    assert len(transport.requests) == 1 and sleeps == []


def test_missing_key_is_unchanged_by_retry_support(sdk_transport):
    install, sleeps, created = sdk_transport
    transport = install([(200, _ok_body())])

    result = IntentParser(GeminiProvider()).parse("list modules")

    assert_rejected(result, "provider_error")
    assert result.errors[0].startswith(f"{API_KEY_ENV_VAR} is not set")
    assert transport.requests == [] and sleeps == [] and created == {}


def test_key_is_sent_as_a_header_and_never_appears_in_retry_errors(sdk_transport, monkeypatch):
    install, _, _ = sdk_transport
    secret = "AIza-retry-test-secret"
    monkeypatch.setenv(API_KEY_ENV_VAR, secret)
    # A hostile or buggy upstream that echoes the key in its error body.
    transport = install([(503, _error_body(503, f"overloaded; key={secret}", "UNAVAILABLE"))])

    result = IntentParser(GeminiProvider()).parse("list modules")

    assert_rejected(result, "provider_error")
    assert len(transport.requests) == RETRY_ATTEMPTS
    assert all(r.headers.get("x-goog-api-key") == secret for r in transport.requests)
    assert all(secret not in str(r.url) for r in transport.requests)
    assert secret not in json.dumps(result.to_dict())
    assert "***" in result.errors[0]


def test_every_attempt_has_an_http_timeout(sdk_transport, monkeypatch):
    install, _, created = sdk_transport
    monkeypatch.setenv(API_KEY_ENV_VAR, "k")
    transport = install([(200, _ok_body())])

    assert IntentParser(GeminiProvider()).parse("list modules").ok

    assert created["http_options"]["timeout"] == int(REQUEST_TIMEOUT_SECONDS * 1000) == 20000
    # The SDK also tells Gemini's server how long the client will wait.
    assert transport.requests[0].headers.get("x-server-timeout") == "20"


def test_unresponsive_upstream_times_out_is_retried_and_then_reported(sdk_transport, monkeypatch):
    """A request Gemini accepts but never answers must not block forever."""
    httpx = pytest.importorskip("httpx")
    install, sleeps, _ = sdk_transport
    monkeypatch.setenv(API_KEY_ENV_VAR, "k")
    transport = install([(200, _ok_body())])
    attempts: list[Any] = []

    def never_answers(request):
        attempts.append(request)
        raise httpx.ReadTimeout("timed out", request=request)

    transport.handler = never_answers

    result = IntentParser(GeminiProvider()).parse("list modules")

    assert_rejected(result, "provider_error")
    assert result.errors[0].startswith("Gemini API did not respond within 20s per attempt (4 attempts made)")
    assert "ReadTimeout" in result.errors[0]
    assert len(attempts) == RETRY_ATTEMPTS
    assert len(sleeps) == RETRY_ATTEMPTS - 1
    _assert_backoff_is_bounded(sleeps)


def test_timeout_then_success_is_parsed(sdk_transport, monkeypatch):
    httpx = pytest.importorskip("httpx")
    install, sleeps, _ = sdk_transport
    monkeypatch.setenv(API_KEY_ENV_VAR, "k")
    transport = install([(200, _ok_body())])
    scripted = transport.handler
    seen: list[Any] = []

    def slow_once(request):
        seen.append(request)
        if len(seen) == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return scripted(request)

    transport.handler = slow_once

    assert IntentParser(GeminiProvider()).parse("list modules").ok
    assert len(seen) == 2 and len(sleeps) == 1


def test_whole_call_is_bounded_well_under_the_inspector_request_timeout():
    # MCP Inspector 1.0.3 cancels a tool call after 5 minutes by default.
    bound = max_request_duration_seconds()
    assert bound == 4 * 20.0 + (2.0 + 3.0 + 5.0) == 90.0
    assert bound < 300


def test_non_httpx_timeout_keeps_the_generic_mapping():
    result, _ = parse("list modules", error=TimeoutError("timed out"))

    assert_rejected(result, "provider_error")
    assert result.errors[0].startswith("LLM request failed: TimeoutError")


def test_retries_never_turn_a_failure_into_a_parsed_action(sdk_transport, monkeypatch):
    install, _, _ = sdk_transport
    monkeypatch.setenv(API_KEY_ENV_VAR, "k")
    install(
        [
            (503, _error_body(503, "overloaded", "UNAVAILABLE")),
            (500, _error_body(500, "internal", "INTERNAL")),
            (502, _error_body(502, "bad gateway", "UNAVAILABLE")),
            (504, _error_body(504, "timeout", "DEADLINE_EXCEEDED")),
        ]
    )

    result = IntentParser(GeminiProvider()).parse("load uvcvideo")

    assert_rejected(result, "provider_error")
    assert result.action is None and not result.ok
    assert "(HTTP 504)" in result.errors[0]


# --------------------------------------------------------------------------- #
# Safety validation after parsing: deterministic Python, not Gemini
# --------------------------------------------------------------------------- #


def test_parsed_action_passes_the_same_validator_as_direct_input():
    result, _ = parse("unload uvcvideo", reply("unload_module", "uvcvideo"))

    assert result.ok
    validation = validate_raw(result.action.to_dict())
    assert validation.ok
    assert validation.action == result.action


def test_parsing_never_calls_the_module_manager(manager, executor):
    result, _ = parse("unload the webcam driver now", reply("unload_module", "uvcvideo"))

    assert result.ok
    assert manager.calls == []


def test_protected_module_is_rejected_by_policy_after_parsing(manager, executor):
    assert "ext4" in DEFAULT_PROTECTED_MODULES
    result, _ = parse("unload ext4", reply("unload_module", "ext4"))
    assert result.ok  # parsing only classifies; policy decides

    outcome = executor.execute(result.action.to_dict(), confirmation_handler=lambda _prompt: True)

    assert outcome.success is False
    assert outcome.status == "rejected"
    assert manager.state_changes() == []


def test_parsed_mutation_requires_confirmation(manager, executor):
    result, _ = parse("unload snd", reply("unload_module", "snd"))
    assert result.ok

    outcome = executor.execute(result.action.to_dict())

    assert outcome.success is False
    assert outcome.status == "confirmation_required"
    assert outcome.requires_confirmation is True
    assert manager.state_changes() == []


def test_parsed_mutation_runs_only_after_confirmation(manager, executor):
    result, _ = parse("unload snd", reply("unload_module", "snd"))

    declined = executor.execute(result.action.to_dict(), confirmation_handler=lambda _prompt: False)
    assert declined.success is False
    assert manager.state_changes() == []

    confirmed = executor.execute(result.action.to_dict(), confirmation_handler=lambda _prompt: True)
    assert confirmed.success is True
    assert confirmed.status == "success"
    assert manager.state_changes() == [("unload_module", "snd")]


def test_parsed_mutation_dry_run_changes_nothing(manager, executor):
    result, _ = parse("load uvcvideo", reply("load_module", "uvcvideo"))

    outcome = executor.execute(result.action.to_dict(), dry_run=True)

    assert outcome.status == "dry_run"
    assert manager.state_changes() == []


def test_state_policy_still_applies_after_parsing(manager, executor):
    result, _ = parse("load snd", reply("load_module", "snd"))  # snd is already loaded

    outcome = executor.execute(result.action.to_dict(), confirmation_handler=lambda _prompt: True)

    assert outcome.success is False
    assert outcome.status == "rejected"
    assert manager.state_changes() == []


# --------------------------------------------------------------------------- #
# MCP parse_request over the Gemini provider
# --------------------------------------------------------------------------- #


def test_mcp_parse_request_tool_is_unchanged():
    spec = TOOL_SPECS["parse_request"]

    assert spec.operation is None
    assert spec.mutating is False
    assert [arg.name for arg in spec.args] == ["request"]
    assert spec.input_schema() == {
        "type": "object",
        "properties": {"request": {"type": "string", "description": spec.args[0].description}},
        "required": ["request"],
        "additionalProperties": False,
    }


def test_mcp_parse_request_returns_structured_action_without_executing(manager, executor):
    client = FakeGeminiClient([reply("unload_module", "uvcvideo")])
    dispatcher = ToolDispatcher(
        executor, intent_parser=IntentParser(GeminiProvider(client=client)), allow_mutations=True
    )

    response = dispatcher.call("parse_request", {"request": "unload the webcam driver"})

    assert response == {
        "ok": True,
        "tool": "parse_request",
        "status": "parsed",
        "message": response["message"],
        "action": {"operation": "unload_module", "module": "uvcvideo"},
        "result": {"executed": False},
        "safety": None,
        "errors": [],
    }
    assert manager.calls == []

    # Acting on the parsed action still goes through policy and confirmation.
    manager.loaded.add("uvcvideo")
    follow_up = dispatcher.call("unload_module", {"module": "uvcvideo"})
    assert follow_up["status"] == "confirmation_required"
    assert manager.state_changes() == []


def test_mcp_parse_request_reports_provider_failures(manager, executor):
    client = FakeGeminiClient(error=FakeAPIError(429, "quota", "RESOURCE_EXHAUSTED"))
    dispatcher = ToolDispatcher(executor, intent_parser=IntentParser(GeminiProvider(client=client)))

    response = dispatcher.call("parse_request", {"request": "list modules"})

    assert response["ok"] is False
    assert response["status"] == "provider_error"
    assert response["action"] is None
    assert response["result"] == {"executed": False}
    assert "429" in response["errors"][0]


def test_mcp_call_tool_returns_a_result_after_the_retry_sequence(sdk_transport, monkeypatch, manager):
    """Through the real MCP handler: the blocking provider call runs on a
    worker thread and the handler returns a tool result once the SDK has
    given up, instead of hanging."""
    httpx = pytest.importorskip("httpx")
    install, sleeps, _ = sdk_transport
    monkeypatch.setenv(API_KEY_ENV_VAR, "k")
    transport = install([(503, _error_body(503, "The model is overloaded.", "UNAVAILABLE"))])
    attempts: list[Any] = []
    scripted = transport.handler

    def overloaded_then_silent(request):
        attempts.append(request)
        if len(attempts) % 2:
            return scripted(request)
        raise httpx.ReadTimeout("timed out", request=request)

    transport.handler = overloaded_then_silent
    executor = ActionExecutor(manager=manager, policy=SafetyPolicy(is_module_loaded=manager.is_module_loaded))
    server = create_server(ToolDispatcher(executor))  # default parser: GeminiProvider via the real SDK
    handler = server.get_request_handler("tools/call").handler

    async def call():
        request = mcp_types.CallToolRequest(
            method="tools/call",
            params=mcp_types.CallToolRequestParams(name="parse_request", arguments={"request": "list modules"}),
        )
        with anyio.fail_after(10):
            return await handler(None, request.params)

    result = anyio.run(call)

    assert isinstance(result, mcp_types.CallToolResult)
    payload = result.structured_content
    assert payload == json.loads(result.content[0].text)
    assert payload["status"] == "provider_error" and payload["action"] is None
    assert payload["result"] == {"executed": False}
    assert "did not respond within 20s" in payload["errors"][0]
    assert len(attempts) == RETRY_ATTEMPTS
    assert len(sleeps) == RETRY_ATTEMPTS - 1


def test_mcp_default_parser_uses_gemini_and_starts_without_a_key(monkeypatch, executor):
    genai = pytest.importorskip("google.genai")

    def no_client(*args, **kwargs):  # pragma: no cover - only runs on a regression
        raise AssertionError("the SDK client must not be created without a key")

    monkeypatch.setattr(genai, "Client", no_client)
    dispatcher = ToolDispatcher(executor)

    assert "parse_request" in [spec.name for spec in dispatcher.available_tools()]
    response = dispatcher.call("parse_request", {"request": "list modules"})

    assert response["status"] == "provider_error"
    assert API_KEY_ENV_VAR in response["errors"][0]
    assert isinstance(dispatcher._parser()._provider, GeminiProvider)
