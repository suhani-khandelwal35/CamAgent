import json
from types import SimpleNamespace

import pytest

from app.llm import GeminiProvider, IntentParser, LLMProviderError, module_is_grounded
from app.llm.intent import MAX_REQUEST_LENGTH, RESPONSE_SCHEMA, SYSTEM_PROMPT
from app.schemas import Operation


class FakeProvider:
    """Predictable stand-in for the LLM: returns a fixed reply or raises."""

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def complete(self, system, prompt, response_schema=None):
        self.calls.append((system, prompt, response_schema))
        if self.error is not None:
            raise self.error
        if isinstance(self.reply, str):
            return self.reply
        return json.dumps(self.reply)


def envelope(operation=None, module=None, status="action", reason="ok"):
    action = None if operation is None else {"operation": operation, "module": module}
    return {"status": status, "action": action, "reason": reason}


def parse(request, reply):
    provider = FakeProvider(reply)
    return IntentParser(provider).parse(request), provider


@pytest.mark.parametrize(
    ("request_text", "reply", "expected"),
    [
        (
            "check whether the webcam driver is loaded",
            envelope("check_module", "uvcvideo"),
            {"operation": "check_module", "module": "uvcvideo"},
        ),
        (
            "show me info about uvcvideo",
            envelope("get_module_info", "uvcvideo"),
            {"operation": "get_module_info", "module": "uvcvideo"},
        ),
        (
            "what modules are loaded right now?",
            envelope("list_modules", None),
            {"operation": "list_modules"},
        ),
        (
            "please reload the UVC driver",
            envelope("reload_module", "uvcvideo"),
            {"operation": "reload_module", "module": "uvcvideo"},
        ),
        (
            "is snd-hda-intel loaded?",
            envelope("check_module", "snd-hda-intel"),
            {"operation": "check_module", "module": "snd_hda_intel"},
        ),
    ],
)
def test_valid_intents_produce_schema_actions(request_text, reply, expected):
    result, _ = parse(request_text, reply)

    assert result.ok
    assert result.status == "parsed"
    assert result.action.to_dict() == expected
    assert result.to_dict()["action"] == expected


def test_provider_receives_prompt_and_response_schema():
    result, provider = parse("is uvcvideo loaded", envelope("check_module", "uvcvideo"))

    assert result.ok
    system, prompt, schema = provider.calls[0]
    assert prompt == "is uvcvideo loaded"
    assert schema is RESPONSE_SCHEMA
    for operation in Operation:
        assert operation.value in system
    assert system == SYSTEM_PROMPT


def test_fenced_json_reply_is_accepted():
    reply = "```json\n" + json.dumps(envelope("check_module", "uvcvideo")) + "\n```"

    result, _ = parse("is the webcam loaded", reply)

    assert result.ok
    assert result.action.operation is Operation.CHECK_MODULE


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "not json at all",
        "Sure! Here is the action: {}",
        "[]",
        '"check_module"',
        {"status": "action", "action": {"operation": "check_module", "module": "uvcvideo"}},
        {**envelope("check_module", "uvcvideo"), "command": "lsmod"},
        envelope("check_module", "uvcvideo", status="maybe"),
        {"status": "action", "action": None, "reason": "x"},
        {"status": "action", "action": "check_module uvcvideo", "reason": "x"},
        envelope("check_module", "uvcvideo", status="unsupported"),
        {"status": "action", "action": {"operation": "check_module", "module": "uvcvideo"}, "reason": 5},
    ],
)
def test_malformed_model_output_is_rejected(reply):
    result, _ = parse("is the webcam loaded", reply)

    assert not result.ok
    assert result.status == "invalid_output"
    assert result.action is None
    assert result.errors


@pytest.mark.parametrize(
    "action",
    [
        {"operation": "run_shell", "module": "uvcvideo"},
        {"operation": "check_module", "module": "uvcvideo", "flags": "-f"},
        {"operation": "check_module", "module": None},
        {"operation": "check_module", "module": 42},
        {"operation": "list_modules", "module": "uvcvideo"},
        {"operation": "load_module", "module": "uvcvideo; rm -rf /"},
        {"operation": "load_module", "module": "../../evil"},
        {"operation": "unload_module", "module": "--force"},
    ],
)
def test_actions_failing_schema_or_validation_are_rejected(action):
    reply = {"status": "action", "action": action, "reason": "x"}

    result, _ = parse("do something with uvcvideo", reply)

    assert not result.ok
    assert result.status == "invalid_output"
    assert result.action is None


def test_non_text_model_output_is_rejected():
    class ObjectProvider(FakeProvider):
        def complete(self, system, prompt, response_schema=None):
            return {"status": "action"}

    result = IntentParser(ObjectProvider()).parse("list modules")

    assert result.status == "invalid_output"


@pytest.mark.parametrize("status", ["unsupported", "ambiguous"])
def test_model_rejections_are_reported(status):
    result, _ = parse("run rm -rf / for me", envelope(status=status, reason="not a module operation"))

    assert not result.ok
    assert result.status == status
    assert "not a module operation" in result.message


def test_invented_module_is_rejected_as_ambiguous():
    result, _ = parse("is my sound card working?", envelope("check_module", "snd_hda_intel"))

    assert not result.ok
    assert result.status == "ambiguous"
    assert result.action is None


@pytest.mark.parametrize("request_text", ["", "   ", None, 42, "x" * (MAX_REQUEST_LENGTH + 1)])
def test_invalid_requests_never_reach_the_provider(request_text):
    provider = FakeProvider(envelope("list_modules"))

    result = IntentParser(provider).parse(request_text)

    assert result.status == "invalid_request"
    assert provider.calls == []


def test_provider_errors_are_reported():
    provider = FakeProvider(error=LLMProviderError("rate limited"))

    result = IntentParser(provider).parse("list modules")

    assert result.status == "provider_error"
    assert result.errors == ["rate limited"]


@pytest.mark.parametrize(
    ("request_text", "module", "expected"),
    [
        ("is uvcvideo loaded", "uvcvideo", True),
        ("is UVCVIDEO loaded", "uvcvideo", True),
        ("is snd-hda-intel loaded", "snd_hda_intel", True),
        ("check the webcam", "uvcvideo", True),
        ("check the web-cam", "uvcvideo", False),
        ("is snd_hda_intel loaded", "snd", False),
        ("is uvcvideo2 loaded", "uvcvideo", False),
        ("check the camera", "uvcvideo", False),
    ],
)
def test_module_grounding(request_text, module, expected):
    assert module_is_grounded(request_text, module) is expected


# --------------------------------------------------------------------------- #
# GeminiProvider over a fake google-genai client (no key, no network)
# --------------------------------------------------------------------------- #


def fake_gemini_client(finish_reason="STOP", parts=None, prompt_feedback=None):
    """A stand-in for google.genai.Client that records generate_content calls."""
    calls = []
    if parts is None:
        parts = [SimpleNamespace(text='{"ok": 1}', thought=None)]
    candidate = SimpleNamespace(
        finish_reason=SimpleNamespace(name=finish_reason),
        content=SimpleNamespace(parts=parts),
    )
    response = SimpleNamespace(candidates=[candidate], prompt_feedback=prompt_feedback)

    def generate_content(**kwargs):
        calls.append(kwargs)
        return response

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return client, calls


def test_gemini_provider_requests_structured_output():
    client, calls = fake_gemini_client()
    provider = GeminiProvider(model="test-model", client=client)

    text = provider.complete("system", "prompt", {"type": "object"})

    assert text == '{"ok": 1}'
    request = calls[0]
    assert request["model"] == "test-model"
    assert request["contents"] == "prompt"
    config = request["config"]
    assert config["system_instruction"] == "system"
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"] == {"type": "object"}


def test_gemini_provider_model_comes_from_environment(monkeypatch):
    monkeypatch.setenv("CAMAGENT_MODEL", "env-model")

    assert GeminiProvider(client=object()).model == "env-model"


@pytest.mark.parametrize(
    ("finish_reason", "parts", "prompt_feedback"),
    [
        ("SAFETY", None, None),
        ("MAX_TOKENS", None, None),
        ("STOP", [], None),
        ("STOP", [SimpleNamespace(text="only a thought", thought=True)], None),
        ("STOP", None, SimpleNamespace(block_reason=SimpleNamespace(name="PROHIBITED_CONTENT"))),
    ],
)
def test_gemini_provider_rejects_unusable_responses(finish_reason, parts, prompt_feedback):
    client, _ = fake_gemini_client(finish_reason, parts, prompt_feedback)

    with pytest.raises(LLMProviderError):
        GeminiProvider(client=client).complete("system", "prompt")


def test_gemini_provider_wraps_request_failures():
    def generate_content(**kwargs):
        raise ConnectionError("network down")

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    with pytest.raises(LLMProviderError, match="network down"):
        GeminiProvider(client=client).complete("system", "prompt")
