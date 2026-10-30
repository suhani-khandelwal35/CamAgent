"""Edge-case tests for the natural-language intent parser (app/llm/intent.py).

Every test uses a fake LLM provider: no network, no Gemini API key and no
Linux module operations. The tests pin down the parser's documented contract:
reject rather than guess, never retry, never execute.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import types

import pytest

from app.llm import (
    DEVICE_ALIASES,
    IntentParser,
    IntentResult,
    LLMProviderError,
    module_is_grounded,
)
from app.llm.intent import (
    ENVELOPE_FIELDS,
    ENVELOPE_STATUSES,
    MAX_REQUEST_LENGTH,
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
)
from app.manager import module_manager
from app.safety import MAX_MODULE_NAME_LENGTH
from app.schemas import ALLOWED_OPERATIONS, Operation

INVALID_OUTPUT_SCHEMA_MESSAGE = (
    "The language model returned output that does not match the action schema."
)
INVALID_OUTPUT_VALIDATION_MESSAGE = (
    "The language model proposed an action that failed validation."
)


# --------------------------------------------------------------------------- #
# Fakes and helpers
# --------------------------------------------------------------------------- #


class FakeProvider:
    """Stand-in LLM.

    A dict/list reply is JSON-encoded; anything else (str, bytes, None, ...) is
    handed back exactly as given so tests can simulate non-text output.
    """

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.calls: list[tuple[str, str, object]] = []

    def complete(self, system, prompt, response_schema=None):
        self.calls.append((system, prompt, response_schema))
        if self.error is not None:
            raise self.error
        if isinstance(self.reply, (dict, list)):
            return json.dumps(self.reply)
        return self.reply


class SequenceProvider(FakeProvider):
    """Returns a different canned reply on each call."""

    def __init__(self, replies):
        super().__init__()
        self.replies = list(replies)

    def complete(self, system, prompt, response_schema=None):
        self.reply = self.replies.pop(0)
        return super().complete(system, prompt, response_schema)


def action_reply(operation, module=None, reason="ok"):
    return {"status": "action", "action": {"operation": operation, "module": module}, "reason": reason}


def rejection_reply(status, reason="unclear"):
    return {"status": status, "action": None, "reason": reason}


def parse(request, reply=None, error=None):
    provider = FakeProvider(reply, error)
    return IntentParser(provider).parse(request), provider


def assert_rejected(result, status):
    assert result.status == status
    assert result.ok is False
    assert result.action is None
    assert result.to_dict()["action"] is None


@pytest.fixture(autouse=True)
def forbid_system_operations(monkeypatch):
    """Fail loudly if anything in these tests tries to run a Linux command.

    All real module operations go through ``module_manager._run_command`` and
    ``subprocess``; parsing must never reach either.
    """

    def explode(*args, **kwargs):  # pragma: no cover - only runs on a regression
        raise AssertionError(f"intent parsing must not run commands: {args!r} {kwargs!r}")

    monkeypatch.setattr(module_manager, "_run_command", explode)
    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, explode)


# --------------------------------------------------------------------------- #
# Ambiguous and incomplete natural-language requests
# --------------------------------------------------------------------------- #

INCOMPLETE_REQUESTS = [
    "load the module",
    "reload",
    "is it loaded?",
    "check the driver",
    "unload that thing again",
    "fix my camera",
    "?",
]


@pytest.mark.parametrize("request_text", INCOMPLETE_REQUESTS)
def test_incomplete_request_marked_ambiguous_by_model_is_rejected(request_text):
    result, provider = parse(request_text, rejection_reply("ambiguous", "no module was named"))

    assert_rejected(result, "ambiguous")
    assert result.message == "Request is ambiguous: no module was named"
    assert result.errors == []
    assert len(provider.calls) == 1


@pytest.mark.parametrize("request_text", INCOMPLETE_REQUESTS)
@pytest.mark.parametrize("module", ["uvcvideo", "snd_hda_intel", "nvidia"])
def test_model_may_not_guess_a_module_for_an_incomplete_request(request_text, module):
    # The model "helpfully" fills in a module the user never mentioned.
    result, _ = parse(request_text, action_reply("load_module", module))

    assert_rejected(result, "ambiguous")
    assert result.errors == [f"module {module!r} is not mentioned in the request"]
    assert result.message == (
        f"Request does not name module {module!r}; please name the module explicitly."
    )


@pytest.mark.parametrize(
    ("request_text", "reason"),
    [
        ("reload uvcvideo and snd_hda_intel", "request names two modules"),
        ("load then unload uvcvideo", "request asks for two operations"),
        ("do something with uvcvideo", "operation is unclear"),
    ],
)
def test_model_ambiguity_verdict_is_honoured_even_when_a_module_is_named(request_text, reason):
    result, _ = parse(request_text, rejection_reply("ambiguous", reason))

    assert_rejected(result, "ambiguous")
    assert result.message == f"Request is ambiguous: {reason}"


@pytest.mark.parametrize(
    ("request_text", "reason"),
    [
        ("run lsmod | grep snd", "shell commands are not supported"),
        ("install the nvidia driver package", "installing packages is not supported"),
        ("set uvcvideo quirks=0x100", "module parameters cannot be changed"),
        ("edit /etc/modprobe.d/blacklist.conf", "editing files is not supported"),
    ],
)
def test_unsupported_requests_are_reported_as_not_supported(request_text, reason):
    result, _ = parse(request_text, rejection_reply("unsupported", reason))

    assert_rejected(result, "unsupported")
    assert result.message == f"Request is not supported: {reason}"
    assert result.errors == []


@pytest.mark.parametrize(("status", "label"), [("ambiguous", "ambiguous"), ("unsupported", "not supported")])
@pytest.mark.parametrize("reason", ["", "   ", "\n\t"])
def test_blank_reason_gets_a_placeholder(status, label, reason):
    result, _ = parse("do the thing", rejection_reply(status, reason))

    assert_rejected(result, status)
    assert result.message == f"Request is {label}: no reason given"


def test_reason_whitespace_is_trimmed_in_message():
    result, _ = parse("do the thing", rejection_reply("ambiguous", "  which module?  \n"))

    assert result.message == "Request is ambiguous: which module?"


# --------------------------------------------------------------------------- #
# Request pre-validation (before the model is involved)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "request_text",
    [b"list modules", ["list modules"], {"request": "list modules"}, 1.5, True, object()],
)
def test_non_string_requests_are_rejected_without_calling_the_model(request_text):
    provider = FakeProvider(action_reply("list_modules"))

    result = IntentParser(provider).parse(request_text)

    assert_rejected(result, "invalid_request")
    assert result.message == "Request must be non-empty text."
    assert provider.calls == []


def test_request_is_stripped_before_reaching_the_model():
    result, provider = parse("  \n list modules \t\n", action_reply("list_modules"))

    assert result.ok
    assert provider.calls[0][1] == "list modules"


def test_request_exactly_at_length_limit_is_accepted():
    body = "x" * MAX_REQUEST_LENGTH

    result, provider = parse(body, rejection_reply("unsupported", "gibberish"))

    assert result.status == "unsupported"
    assert provider.calls[0][1] == body


def test_surrounding_whitespace_does_not_count_towards_length_limit():
    body = "x" * MAX_REQUEST_LENGTH

    result, provider = parse("   " + body + "   ", rejection_reply("unsupported", "gibberish"))

    assert result.status == "unsupported"
    assert provider.calls[0][1] == body


def test_over_long_request_message_names_the_limit():
    result, provider = parse("x" * (MAX_REQUEST_LENGTH + 1))

    assert_rejected(result, "invalid_request")
    assert str(MAX_REQUEST_LENGTH) in result.message
    assert provider.calls == []


# --------------------------------------------------------------------------- #
# Malformed JSON and invalid model responses
# --------------------------------------------------------------------------- #

VALID_ENVELOPE_JSON = json.dumps(rejection_reply("ambiguous", "x"))


@pytest.mark.parametrize(
    "reply",
    [
        '{"status": "action"',  # truncated
        "{'status': 'ambiguous', 'action': None, 'reason': 'x'}",  # Python repr, not JSON
        '{"status": "ambiguous", "action": null, "reason": "x",}',  # trailing comma
        VALID_ENVELOPE_JSON + "{}",  # two concatenated documents
        VALID_ENVELOPE_JSON + "\nDone!",  # trailing prose
        "﻿" + VALID_ENVELOPE_JSON,  # byte-order mark
        "Here you go:\n```json\n" + VALID_ENVELOPE_JSON + "\n```",  # prose before a fence
        "```json\n```",  # empty fence
        "```json\n" + VALID_ENVELOPE_JSON,  # unterminated fence
        "// classification\n" + VALID_ENVELOPE_JSON,  # comment
        "status: ambiguous\naction: null\nreason: x",  # YAML
        "<action><operation>list_modules</operation></action>",  # XML
    ],
)
def test_unparseable_replies_are_invalid_output(reply):
    result, _ = parse("list modules", reply)

    assert_rejected(result, "invalid_output")
    assert result.message == INVALID_OUTPUT_SCHEMA_MESSAGE
    assert len(result.errors) == 1
    assert result.errors[0].startswith("model output is not valid JSON")


@pytest.mark.parametrize("reply", ["null", "true", "false", "0", "1.5", "NaN", "[]", '["action"]', '"ambiguous"'])
def test_valid_json_that_is_not_an_object_is_invalid_output(reply):
    result, _ = parse("list modules", reply)

    assert_rejected(result, "invalid_output")
    assert result.errors == ["model output must be a JSON object"]


@pytest.mark.parametrize("reply", [None, b'{"status": "ambiguous", "action": null, "reason": "x"}', 42, 3.5, True])
def test_non_text_replies_are_invalid_output(reply):
    result, _ = parse("list modules", reply)

    assert_rejected(result, "invalid_output")
    assert result.errors == [f"model output must be text, got {type(reply).__name__}"]


def test_duplicate_json_keys_do_not_bypass_shape_checks():
    # json.loads keeps the last duplicate, so this becomes status=action with no action.
    reply = '{"status": "ambiguous", "status": "action", "action": null, "reason": "x"}'

    result, _ = parse("list modules", reply)

    assert_rejected(result, "invalid_output")
    assert result.errors == ["model output with status 'action' must include an action object"]


def test_double_encoded_action_is_rejected():
    reply = {
        "status": "action",
        "action": json.dumps({"operation": "list_modules", "module": None}),
        "reason": "x",
    }

    result, _ = parse("list modules", reply)

    assert_rejected(result, "invalid_output")
    assert result.errors == ["model output with status 'action' must include an action object"]


@pytest.mark.parametrize(
    ("envelope", "error"),
    [
        ({}, "model output is missing field(s): ['action', 'reason', 'status']"),
        ({"status": "ambiguous", "action": None}, "model output is missing field(s): ['reason']"),
        ({"status": "ambiguous", "reason": "x"}, "model output is missing field(s): ['action']"),
        ({"action": None, "reason": "x"}, "model output is missing field(s): ['status']"),
        # Unexpected fields are reported before missing ones.
        ({"extra": 1}, "model output has unexpected field(s): ['extra']"),
        ({**rejection_reply("ambiguous"), "confidence": 0.9}, "model output has unexpected field(s): ['confidence']"),
        # Field names are case-sensitive.
        ({**rejection_reply("ambiguous"), "Status": "action"}, "model output has unexpected field(s): ['Status']"),
        (
            {**rejection_reply("ambiguous"), "command": "lsmod", "shell": True},
            "model output has unexpected field(s): ['command', 'shell']",
        ),
        ({**rejection_reply("AMBIGUOUS")}, "model output has unknown status 'AMBIGUOUS'"),
        ({**rejection_reply(" ambiguous")}, "model output has unknown status ' ambiguous'"),
        # The parser's own result statuses are not valid envelope statuses.
        ({**rejection_reply("parsed")}, "model output has unknown status 'parsed'"),
        ({**rejection_reply("invalid_output")}, "model output has unknown status 'invalid_output'"),
        ({**rejection_reply("")}, "model output has unknown status ''"),
        ({**rejection_reply(None)}, "model output has unknown status None"),
        ({**rejection_reply(1)}, "model output has unknown status 1"),
        ({**rejection_reply(True)}, "model output has unknown status True"),
        # Unhashable statuses must be rejected, not crash the set lookup.
        ({**rejection_reply([])}, "model output has unknown status []"),
        ({**rejection_reply(["action"])}, "model output has unknown status ['action']"),
        ({**rejection_reply({})}, "model output has unknown status {}"),
        ({**rejection_reply({"value": "action"})}, "model output has unknown status {'value': 'action'}"),
        ({"status": "ambiguous", "action": None, "reason": None}, "model output 'reason' must be a string"),
        ({"status": "ambiguous", "action": None, "reason": ["x"]}, "model output 'reason' must be a string"),
        ({"status": "ambiguous", "action": None, "reason": 7}, "model output 'reason' must be a string"),
        (
            {"status": "action", "action": [], "reason": "x"},
            "model output with status 'action' must include an action object",
        ),
        (
            {"status": "action", "action": ["list_modules"], "reason": "x"},
            "model output with status 'action' must include an action object",
        ),
        (
            {"status": "action", "action": True, "reason": "x"},
            "model output with status 'action' must include an action object",
        ),
        (
            {"status": "ambiguous", "action": {"operation": "list_modules", "module": None}, "reason": "x"},
            "model output with status 'ambiguous' must not include an action",
        ),
        (
            {"status": "unsupported", "action": {}, "reason": "x"},
            "model output with status 'unsupported' must not include an action",
        ),
        (
            {"status": "unsupported", "action": "none", "reason": "x"},
            "model output with status 'unsupported' must not include an action",
        ),
    ],
)
def test_envelope_shape_errors_are_reported_precisely(envelope, error):
    result, _ = parse("list modules", envelope)

    assert_rejected(result, "invalid_output")
    assert result.message == INVALID_OUTPUT_SCHEMA_MESSAGE
    assert result.errors == [error]


# --------------------------------------------------------------------------- #
# Unsupported operations and action names
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "operation",
    [
        "lsmod",
        "modprobe",
        "rmmod",
        "insmod",
        "exec",
        "run_shell",
        "list-modules",  # hyphen is not normalised for operations
        "list modules",
        "listmodules",
        "LoadModule",
        "load_modules",
        "check_module; rm -rf /",
        "delete_module",
        "",
        "   ",
    ],
)
def test_unsupported_operation_names_are_rejected(operation):
    result, _ = parse("check uvcvideo", action_reply(operation, "uvcvideo"))

    assert_rejected(result, "invalid_output")
    assert result.message == INVALID_OUTPUT_VALIDATION_MESSAGE
    assert len(result.errors) == 1
    assert result.errors[0].startswith(f"Unsupported operation {operation!r}")
    for allowed in ALLOWED_OPERATIONS:
        assert allowed in result.errors[0]


@pytest.mark.parametrize("operation", [None, 7, 1.5, True, ["check_module"], {"name": "check_module"}])
def test_non_string_operations_are_rejected(operation):
    result, _ = parse("check uvcvideo", action_reply(operation, "uvcvideo"))

    assert_rejected(result, "invalid_output")
    assert result.message == INVALID_OUTPUT_VALIDATION_MESSAGE
    assert result.errors == [f"'operation' must be a string, got {type(operation).__name__}"]


@pytest.mark.parametrize("operation", ["CHECK_MODULE", "Check_Module", " check_module ", "check_module\n"])
def test_operation_case_and_surrounding_whitespace_are_normalised(operation):
    result, _ = parse("check uvcvideo", action_reply(operation, "uvcvideo"))

    assert result.ok
    assert result.action.operation is Operation.CHECK_MODULE
    assert result.action.to_dict() == {"operation": "check_module", "module": "uvcvideo"}


# --------------------------------------------------------------------------- #
# Missing required fields and unexpected fields in the action
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("action", "error"),
    [
        ({}, "Action is missing required field 'operation'"),
        ({"module": "uvcvideo"}, "Action is missing required field 'operation'"),
        ({"operation": "check_module"}, "Operation 'check_module' requires a 'module' name"),
        ({"operation": "check_module", "module": None}, "Operation 'check_module' requires a 'module' name"),
        ({"operation": "load_module", "module": ""}, "'module' must not be empty"),
        ({"operation": "load_module", "module": "   \n"}, "'module' must not be empty"),
        ({"operation": "load_module", "module": 42}, "'module' must be a string, got int"),
        ({"operation": "load_module", "module": True}, "'module' must be a string, got bool"),
        ({"operation": "load_module", "module": ["uvcvideo"]}, "'module' must be a string, got list"),
        ({"operation": "load_module", "module": {"name": "uvcvideo"}}, "'module' must be a string, got dict"),
        ({"operation": "list_modules", "module": "uvcvideo"}, "Operation 'list_modules' does not take a 'module'"),
        ({"operation": "list_modules", "module": ""}, "Operation 'list_modules' does not take a 'module'"),
        (
            {"operation": "check_module", "module": "uvcvideo", "force": True},
            "Unexpected field(s) in action: ['force']. Allowed fields: ['module', 'operation']",
        ),
        (
            {"operation": "check_module", "module": "uvcvideo", "modules": ["snd"]},
            "Unexpected field(s) in action: ['modules']. Allowed fields: ['module', 'operation']",
        ),
        (
            {"operation": "check_module", "module": "uvcvideo", "Operation": "load_module"},
            "Unexpected field(s) in action: ['Operation']. Allowed fields: ['module', 'operation']",
        ),
        (
            {"operation": "check_module", "module": "uvcvideo", "args": ["-f"], "confirm": True},
            "Unexpected field(s) in action: ['args', 'confirm']. Allowed fields: ['module', 'operation']",
        ),
        (
            {"operation": "check_module", "module": "uvcvideo", "reason": "x", "status": "action"},
            "Unexpected field(s) in action: ['reason', 'status']. Allowed fields: ['module', 'operation']",
        ),
    ],
)
def test_missing_and_unexpected_action_fields_are_rejected(action, error):
    result, _ = parse("check uvcvideo", {"status": "action", "action": action, "reason": "x"})

    assert_rejected(result, "invalid_output")
    assert result.message == INVALID_OUTPUT_VALIDATION_MESSAGE
    assert result.errors == [error]


# --------------------------------------------------------------------------- #
# Unknown or invalid module names
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("module", "fragment"),
    [
        ("uvcvideo; rm -rf /", "forbidden character"),
        ("uvc video", "forbidden character"),
        ("uvcvideo\tsnd", "forbidden character"),
        ("sudo modprobe uvcvideo", "forbidden character"),
        ("$(reboot)", "forbidden character"),
        ("`id`", "forbidden character"),
        ("uvcvideo|tee", "forbidden character"),
        ("uvcvideo&", "forbidden character"),
        ("../../etc/passwd", "forbidden character"),
        ("a..b", "must not contain '..'"),
        ("-r", "must not start with '-'"),
        ("--force", "must not start with '-'"),
        ("-", "must not start with '-'"),
        (".", "must start with a letter or digit"),
        ("_uvcvideo", "must start with a letter or digit"),
        ("uvcvidéo", "may only contain letters, digits"),
        ("uvc\x00video", "may only contain letters, digits"),
        ("uvcvideo​", "may only contain letters, digits"),  # zero-width space
        ("m" * (MAX_MODULE_NAME_LENGTH + 1), "too long"),
    ],
)
def test_malformed_module_names_are_rejected_before_grounding(module, fragment):
    # The request mentions nothing useful; validation must fail first anyway.
    result, _ = parse("load it", action_reply("load_module", module))

    assert_rejected(result, "invalid_output")
    assert result.message == INVALID_OUTPUT_VALIDATION_MESSAGE
    assert result.errors
    assert any(fragment in error for error in result.errors)


@pytest.mark.parametrize(
    ("request_text", "module"),
    [
        ("is uvcvideo loaded?", "snd_hda_intel"),  # a different module than named
        ("is uvcvideo loaded?", "uvcvideo2"),  # superstring of the named module
        ("is uvcvideo loaded?", "uvc"),  # prefix of the named module
        ("is uvcvideo loaded?", "video"),  # suffix of the named module
        ("is uvcvideo2 loaded?", "uvcvideo"),  # named module is a prefix of the request token
        ("is snd.uvcvideo loaded?", "uvcvideo"),  # dotted prefix is part of the token
        ("check the webcam", "snd_hda_intel"),  # alias belongs to another module
        ("check the camera", "uvcvideo"),  # not a documented alias
        ("check my web-cam", "uvcvideo"),  # alias spelling must match exactly
        ("check the webcams", "uvcvideo"),  # alias is matched whole-word
        ("check the usb  camera", "uvcvideo"),  # double space breaks the phrase
        ("list modules", "uvcvideo"),
    ],
)
def test_well_formed_but_ungrounded_module_is_ambiguous(request_text, module):
    result, _ = parse(request_text, action_reply("check_module", module))

    assert_rejected(result, "ambiguous")
    assert result.errors == [f"module {module!r} is not mentioned in the request"]
    assert result.message == (
        f"Request does not name module {module!r}; please name the module explicitly."
    )


def test_ungrounded_module_error_uses_canonical_name():
    result, _ = parse("is uvcvideo loaded?", action_reply("check_module", "snd-hda-intel"))

    assert_rejected(result, "ambiguous")
    assert result.errors == ["module 'snd_hda_intel' is not mentioned in the request"]


def test_parser_does_not_check_whether_a_named_module_exists():
    # Existence is the module manager's job; the parser only needs the name to
    # be well-formed and grounded in the request.
    result, _ = parse(
        "load the module totally_made_up_mod",
        action_reply("load_module", "totally_made_up_mod"),
    )

    assert result.ok
    assert result.action.module == "totally_made_up_mod"


def test_list_modules_needs_no_grounding():
    result, _ = parse("what's loaded?", action_reply("list_modules"))

    assert result.ok
    assert result.action.module is None
    assert result.action.to_dict() == {"operation": "list_modules"}


# --------------------------------------------------------------------------- #
# Module aliases and normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("module", "phrase"),
    [(module, phrase) for module, phrases in DEVICE_ALIASES.items() for phrase in phrases],
)
def test_every_documented_alias_grounds_only_its_module(module, phrase):
    assert module_is_grounded(f"please check the {phrase}", module)
    assert module_is_grounded(f"please check the {phrase.upper()}", module)
    assert module_is_grounded(f"{phrase}?", module)
    assert module_is_grounded(f"({phrase})", module)
    assert not module_is_grounded(f"please check the {phrase}", "snd_hda_intel")
    # The prompt must tell the model about every alias the parser will accept.
    assert phrase in SYSTEM_PROMPT
    assert module in SYSTEM_PROMPT


@pytest.mark.parametrize(
    "request_text",
    [
        "is the webcam working?",
        "check my WEBCAM",
        "is the web cam driver loaded",
        "reload the USB camera",
        "info on the uvc camera",
        "restart the UVC driver",
        "check the webcam.",
    ],
)
def test_alias_phrases_ground_uvcvideo_end_to_end(request_text):
    result, _ = parse(request_text, action_reply("check_module", "uvcvideo"))

    assert result.ok
    assert result.action.module == "uvcvideo"


@pytest.mark.parametrize(
    ("request_text", "model_module", "expected"),
    [
        ("is snd-hda-intel loaded", "snd-hda-intel", "snd_hda_intel"),
        ("is snd_hda_intel loaded", "snd-hda-intel", "snd_hda_intel"),  # request uses '_', model '-'
        ("is snd-hda-intel loaded", "snd_hda_intel", "snd_hda_intel"),  # request uses '-', model '_'
        ("is snd_hda-intel loaded", "snd-hda_intel", "snd_hda_intel"),  # mixed on both sides
        ("is uvcvideo loaded", "  uvcvideo  ", "uvcvideo"),
        ("is uvcvideo loaded", "uvcvideo\n", "uvcvideo"),
        ("is UVCVIDEO loaded", "uvcvideo", "uvcvideo"),
        ("is uvcvideo.ko loaded", "uvcvideo", "uvcvideo"),  # file suffix does not hide the name
        ("is /lib/modules/uvcvideo loaded", "uvcvideo", "uvcvideo"),
    ],
)
def test_module_names_are_canonicalised_and_grounded_across_spellings(request_text, model_module, expected):
    result, _ = parse(request_text, action_reply("check_module", model_module))

    assert result.ok
    assert result.action.module == expected
    assert result.message == f"Parsed request as check_module {expected}."


@pytest.mark.parametrize(
    ("request_text", "module", "expected"),
    [
        ("load a.b", "a.b", True),
        ("load a.b", "a_b", False),
        ("load a+b", "a+b", True),  # regex metacharacters in the name are escaped
        ("load ab", "a+b", False),
        ("load a-b", "a_b", True),
        ("load a_b", "a-b", True),
    ],
)
def test_grounding_is_literal_and_treats_dash_and_underscore_alike(request_text, module, expected):
    assert module_is_grounded(request_text, module) is expected


# --------------------------------------------------------------------------- #
# Unexpected LLM response formats and exceptions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "template",
    [
        "{json}",
        "\n\n  {json}  \n",
        "```{json}```",
        "```\n{json}\n```",
        "```json\n{json}\n```",
        "```JSON\n{json}\n```",
        "```json\r\n{json}\r\n```",
        "  ```json\n{json}\n```  ",
    ],
)
def test_whitespace_and_fence_variants_around_valid_json_are_accepted(template):
    reply = template.format(json=json.dumps(action_reply("list_modules")))

    result, _ = parse("list modules", reply)

    assert result.ok
    assert result.action.operation is Operation.LIST_MODULES


def test_pretty_printed_json_is_accepted():
    reply = json.dumps(action_reply("check_module", "uvcvideo"), indent=4)

    result, _ = parse("check uvcvideo", reply)

    assert result.ok


class CustomProviderError(LLMProviderError):
    pass


@pytest.mark.parametrize(
    "error",
    [
        LLMProviderError("rate limited"),
        LLMProviderError("LLM response was truncated"),
        LLMProviderError("LLM declined to answer the request"),
        LLMProviderError(""),
        CustomProviderError("subclass failure"),
    ],
)
def test_provider_errors_including_subclasses_are_reported(error):
    result, provider = parse("list modules", error=error)

    assert_rejected(result, "provider_error")
    assert result.message == "The language model was unavailable."
    assert result.errors == [str(error)]
    assert len(provider.calls) == 1


@pytest.mark.parametrize("error", [ValueError("bad"), TimeoutError("slow"), RuntimeError("boom"), KeyError("k")])
def test_unexpected_provider_exceptions_are_not_masked(error):
    # Providers must raise LLMProviderError (see app/llm/provider.py). Anything
    # else is a programming error and must surface instead of being reported as
    # a parsed or merely failed intent. (The MCP dispatcher's catch-all turns it
    # into an "internal_error" response without leaking details.)
    provider = FakeProvider(error=error)

    with pytest.raises(type(error)):
        IntentParser(provider).parse("list modules")


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        rejection_reply("maybe"),
        action_reply("run_shell", "uvcvideo"),
        action_reply("check_module", "uvcvideo; id"),
        action_reply("check_module", "snd_hda_intel"),  # ungrounded
    ],
)
def test_parser_never_retries_or_reprompts_on_bad_output(reply):
    result, provider = parse("check uvcvideo", reply)

    assert not result.ok
    assert len(provider.calls) == 1


def test_parser_is_stateless_across_calls():
    provider = SequenceProvider(
        [
            "garbage",
            action_reply("check_module", "uvcvideo"),
            rejection_reply("unsupported", "no"),
            action_reply("list_modules"),
        ]
    )
    parser = IntentParser(provider)

    first = parser.parse("check uvcvideo")
    second = parser.parse("check uvcvideo")
    third = parser.parse("rm -rf /")
    fourth = parser.parse("list modules")

    assert [r.status for r in (first, second, third, fourth)] == [
        "invalid_output",
        "parsed",
        "unsupported",
        "parsed",
    ]
    assert second.action.module == "uvcvideo"
    assert fourth.action.module is None
    assert [call[1] for call in provider.calls] == ["check uvcvideo", "check uvcvideo", "rm -rf /", "list modules"]


def test_every_call_sends_the_same_system_prompt_and_schema():
    provider = SequenceProvider([rejection_reply("ambiguous"), action_reply("list_modules")])
    parser = IntentParser(provider)

    parser.parse("hm")
    parser.parse("list modules")

    assert all(system == SYSTEM_PROMPT for system, _, _ in provider.calls)
    assert all(schema is RESPONSE_SCHEMA for _, _, schema in provider.calls)


# --------------------------------------------------------------------------- #
# Result object contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("request_text", "reply", "status"),
    [
        ("", None, "invalid_request"),
        ("list modules", "nope", "invalid_output"),
        ("list modules", action_reply("list_modules", "uvcvideo"), "invalid_output"),
        ("list modules", rejection_reply("unsupported"), "unsupported"),
        ("list modules", rejection_reply("ambiguous"), "ambiguous"),
        ("list modules", action_reply("check_module", "uvcvideo"), "ambiguous"),
        ("list modules", action_reply("list_modules"), "parsed"),
    ],
)
def test_to_dict_has_a_fixed_shape_for_every_status(request_text, reply, status):
    result, _ = parse(request_text, reply)
    data = result.to_dict()

    assert result.status == status
    assert set(data) == {"ok", "status", "message", "action", "errors"}
    assert data["ok"] is (status == "parsed")
    assert data["status"] == status
    assert isinstance(data["message"], str) and data["message"]
    assert isinstance(data["errors"], list)
    if status == "parsed":
        assert data["action"] == {"operation": "list_modules"}
    else:
        assert data["action"] is None


def test_provider_error_result_is_reported_in_to_dict():
    result, _ = parse("list modules", error=LLMProviderError("down"))

    assert result.to_dict() == {
        "ok": False,
        "status": "provider_error",
        "message": "The language model was unavailable.",
        "action": None,
        "errors": ["down"],
    }


def test_to_dict_errors_are_a_copy():
    result, _ = parse("list modules", "nope")

    result.to_dict()["errors"].append("tampered")

    assert "tampered" not in result.errors
    assert result.to_dict()["errors"] == result.errors


def test_intent_result_is_immutable():
    result, _ = parse("list modules", action_reply("list_modules"))

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = "parsed"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.action = None  # type: ignore[misc]


def test_ok_requires_both_parsed_status_and_an_action():
    assert IntentResult("parsed", "m").ok is False
    assert IntentResult("ambiguous", "m").ok is False


# --------------------------------------------------------------------------- #
# Parsing never executes system operations
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("operation", ALLOWED_OPERATIONS)
def test_parsing_any_valid_action_runs_no_commands(operation):
    # The autouse fixture raises if subprocess or the module manager is touched.
    module = None if operation == "list_modules" else "uvcvideo"

    result, _ = parse(f"{operation.replace('_', ' ')} uvcvideo", action_reply(operation, module))

    assert result.ok
    assert result.action.operation.value == operation
    assert result.to_dict() == {
        "ok": True,
        "status": "parsed",
        "message": f"Parsed request as {result.action.describe()}.",
        "action": result.action.to_dict(),
        "errors": [],
    }


def test_parsing_a_mutating_request_only_describes_the_action():
    result, _ = parse("unload the webcam driver now", action_reply("unload_module", "uvcvideo"))

    assert result.ok
    assert result.action.to_dict() == {"operation": "unload_module", "module": "uvcvideo"}
    # No execution outcome, state, or confirmation flags are part of a parse result.
    assert not {"result", "executed", "confirmed", "safety"} & set(result.to_dict())


def test_intent_module_has_no_execution_dependencies():
    import app.llm.intent as intent

    forbidden_prefixes = ("subprocess", "os", "shlex", "app.manager", "app.agent", "app.orchestrator")
    for name, value in vars(intent).items():
        origin = value.__name__ if isinstance(value, types.ModuleType) else getattr(value, "__module__", None)
        if isinstance(origin, str):
            assert not origin.startswith(forbidden_prefixes), f"{name} comes from {origin}"


# --------------------------------------------------------------------------- #
# Schema and prompt stay in step with the parser
# --------------------------------------------------------------------------- #


def test_response_schema_matches_parser_constants():
    assert RESPONSE_SCHEMA["type"] == "object"
    assert RESPONSE_SCHEMA["additionalProperties"] is False
    assert set(RESPONSE_SCHEMA["required"]) == ENVELOPE_FIELDS
    assert set(RESPONSE_SCHEMA["properties"]) == ENVELOPE_FIELDS
    assert set(RESPONSE_SCHEMA["properties"]["status"]["enum"]) == ENVELOPE_STATUSES

    action_object, action_null = RESPONSE_SCHEMA["properties"]["action"]["anyOf"]
    assert action_null == {"type": "null"}
    assert action_object["additionalProperties"] is False
    assert set(action_object["required"]) == {"operation", "module"}
    assert set(action_object["properties"]) == {"operation", "module"}
    assert tuple(action_object["properties"]["operation"]["enum"]) == ALLOWED_OPERATIONS
    assert action_object["properties"]["module"]["type"] == ["string", "null"]


def test_system_prompt_documents_every_status_and_operation():
    for status in ENVELOPE_STATUSES:
        assert f'"{status}"' in SYSTEM_PROMPT
    for operation in ALLOWED_OPERATIONS:
        assert operation in SYSTEM_PROMPT
    assert "never run anything" in SYSTEM_PROMPT
