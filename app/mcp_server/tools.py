"""Tool catalog, argument validation and dispatch for the MCP server.

This module has no MCP dependency: the server passes it the raw tool name and
argument mapping exactly as the client sent them, and gets back a response
dictionary with the same keys for every outcome:

    {"ok", "tool", "status", "message", "action", "result", "safety", "errors"}

Arguments are checked strictly (no unknown keys, no type coercion, module
names validated) before anything is routed. Module operations are routed to
`ActionExecutor`, so the validator, safety policy and confirmation gate always
apply; `parse_request` is routed to the intent parser and never executes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from app.agent import ActionExecutor, ExecutionResult
from app.llm import GeminiProvider, IntentParser, IntentResult
from app.llm.intent import MAX_REQUEST_LENGTH
from app.manager import module_manager
from app.safety import module_name_errors
from app.schemas import Operation

logger = logging.getLogger("camagent.mcp")


@dataclass(frozen=True)
class ArgSpec:
    name: str
    type: type
    description: str
    required: bool = True
    default: Any = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args: tuple[ArgSpec, ...] = ()
    operation: Operation | None = None
    mutating: bool = False

    def input_schema(self) -> dict[str, Any]:
        json_types = {str: "string", bool: "boolean"}
        properties: dict[str, Any] = {}
        for arg in self.args:
            prop: dict[str, Any] = {"type": json_types[arg.type], "description": arg.description}
            if not arg.required:
                prop["default"] = arg.default
            properties[arg.name] = prop
        return {
            "type": "object",
            "properties": properties,
            "required": [arg.name for arg in self.args if arg.required],
            "additionalProperties": False,
        }


_MODULE = ArgSpec("module", str, "Kernel module name, for example 'uvcvideo'.")
_CONFIRM = ArgSpec(
    "confirm",
    bool,
    "Must be true to perform the change. Set only after the user explicitly approves it.",
    required=False,
    default=False,
)
_DRY_RUN = ArgSpec(
    "dry_run",
    bool,
    "Preview the change and its safety checks without performing it.",
    required=False,
    default=False,
)

TOOL_SPECS: Mapping[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec("list_modules", "List the kernel modules that are currently loaded.", (), Operation.LIST_MODULES),
        ToolSpec(
            "get_module_info",
            "Show modinfo details (description, author, dependencies, parameters) for one module.",
            (_MODULE,),
            Operation.GET_MODULE_INFO,
        ),
        ToolSpec("check_module", "Check whether one kernel module is currently loaded.", (_MODULE,), Operation.CHECK_MODULE),
        ToolSpec(
            "parse_request",
            "Translate a natural-language request into a structured module action without executing it.",
            (ArgSpec("request", str, "The user's request, for example 'is the webcam driver loaded?'."),),
        ),
        ToolSpec(
            "load_module",
            "Load a kernel module. Requires confirm=true; use dry_run=true to preview.",
            (_MODULE, _CONFIRM, _DRY_RUN),
            Operation.LOAD_MODULE,
            mutating=True,
        ),
        ToolSpec(
            "unload_module",
            "Unload a kernel module. Requires confirm=true; use dry_run=true to preview.",
            (_MODULE, _CONFIRM, _DRY_RUN),
            Operation.UNLOAD_MODULE,
            mutating=True,
        ),
        ToolSpec(
            "reload_module",
            "Unload and reload a kernel module. Requires confirm=true; use dry_run=true to preview.",
            (_MODULE, _CONFIRM, _DRY_RUN),
            Operation.RELOAD_MODULE,
            mutating=True,
        ),
    )
}

READ_ONLY_TOOLS = tuple(name for name, spec in TOOL_SPECS.items() if not spec.mutating)
MUTATING_TOOLS = tuple(name for name, spec in TOOL_SPECS.items() if spec.mutating)


@dataclass(frozen=True)
class ToolResponse:
    ok: bool
    tool: str
    status: str
    message: str
    action: dict[str, Any] | None = None
    result: Any = None
    safety: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "status": self.status,
            "message": self.message,
            "action": self.action,
            "result": self.result,
            "safety": self.safety,
            "errors": list(self.errors),
        }


class ToolCallError(Exception):
    """A tool call that cannot be routed (unknown tool or invalid arguments)."""

    def __init__(self, status: str, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.errors = errors or []


def validate_arguments(spec: ToolSpec, arguments: Any) -> dict[str, Any]:
    """Return the arguments with defaults filled in, or raise ToolCallError."""
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        raise ToolCallError(
            "invalid_arguments",
            "Tool arguments must be a JSON object.",
            [f"arguments must be an object, got {type(arguments).__name__}"],
        )

    errors: list[str] = []
    known = {arg.name for arg in spec.args}
    unknown = sorted(str(key) for key in arguments if key not in known)
    if unknown:
        errors.append(f"unexpected argument(s): {unknown}")

    values: dict[str, Any] = {}
    for arg in spec.args:
        if arg.name not in arguments:
            if arg.required:
                errors.append(f"missing required argument '{arg.name}'")
            else:
                values[arg.name] = arg.default
            continue

        value = arguments[arg.name]
        # `type(...) is` so that, for example, 1 is not accepted as True.
        if type(value) is not arg.type:
            errors.append(
                f"argument '{arg.name}' must be a {arg.type.__name__}, got {type(value).__name__}"
            )
            continue
        values[arg.name] = value

    if isinstance(values.get("module"), str):
        errors.extend(module_name_errors(values["module"]))
    if isinstance(values.get("request"), str):
        if not values["request"].strip():
            errors.append("argument 'request' must not be empty")
        elif len(values["request"]) > MAX_REQUEST_LENGTH:
            errors.append(f"argument 'request' must be at most {MAX_REQUEST_LENGTH} characters")
    if values.get("confirm") and values.get("dry_run"):
        errors.append("'confirm' and 'dry_run' cannot both be true")

    if errors:
        raise ToolCallError("invalid_arguments", f"Invalid arguments for tool '{spec.name}'.", errors)
    return values


class CheckedModuleManager:
    """Module manager adapter that reports lsmod failures instead of hiding them.

    The plain manager treats an lsmod failure as "no modules loaded", which
    would make check_module answer "not loaded" when the state is unknown.
    """

    list_modules = staticmethod(module_manager.list_modules_checked)
    get_module_info = staticmethod(module_manager.get_module_info)
    is_module_loaded = staticmethod(module_manager.verify_module_loaded)
    load_module = staticmethod(module_manager.load_module)
    unload_module = staticmethod(module_manager.unload_module)
    reload_module = staticmethod(module_manager.reload_module)


class ToolDispatcher:
    """Validate tool calls and route them to the executor or intent parser."""

    def __init__(
        self,
        executor: ActionExecutor | None = None,
        intent_parser: IntentParser | Callable[[], IntentParser] | None = None,
        allow_mutations: bool = False,
    ) -> None:
        self._executor = executor or ActionExecutor(manager=CheckedModuleManager())
        self._intent_parser = intent_parser
        self.allow_mutations = allow_mutations

    def available_tools(self) -> list[ToolSpec]:
        return [spec for spec in TOOL_SPECS.values() if self.allow_mutations or not spec.mutating]

    def _parser(self) -> IntentParser:
        # Created on first use so the server starts without LLM credentials.
        if self._intent_parser is None:
            self._intent_parser = IntentParser(GeminiProvider())
        elif not isinstance(self._intent_parser, IntentParser):
            self._intent_parser = self._intent_parser()
        return self._intent_parser

    def call(self, name: Any, arguments: Any = None) -> dict[str, Any]:
        tool = name if isinstance(name, str) else repr(name)
        try:
            spec = self._lookup(name)
            values = validate_arguments(spec, arguments)
            if spec.operation is None:
                return self._from_intent(tool, self._parser().parse(values["request"]))
            return self._execute(spec, values)
        except ToolCallError as exc:
            return ToolResponse(False, tool, exc.status, exc.message, errors=exc.errors).to_dict()
        except Exception:
            # Never leak tracebacks or internal state to the client.
            logger.exception("tool %s failed unexpectedly", tool)
            return ToolResponse(
                False, tool, "internal_error", "The tool failed unexpectedly.", errors=["internal error"]
            ).to_dict()

    def _lookup(self, name: Any) -> ToolSpec:
        spec = TOOL_SPECS.get(name) if isinstance(name, str) else None
        if spec is None:
            raise ToolCallError(
                "unknown_tool",
                f"Unknown tool {name!r}.",
                [f"available tools: {[s.name for s in self.available_tools()]}"],
            )
        if spec.mutating and not self.allow_mutations:
            raise ToolCallError(
                "tool_disabled",
                f"Tool {name!r} is disabled; module changes are not enabled on this server.",
            )
        return spec

    def _execute(self, spec: ToolSpec, values: dict[str, Any]) -> dict[str, Any]:
        assert spec.operation is not None
        raw_action: dict[str, Any] = {"operation": spec.operation.value}
        if "module" in values:
            raw_action["module"] = values["module"]

        options: dict[str, Any] = {}
        if spec.mutating:
            options["dry_run"] = values["dry_run"]
            # With no handler the executor reports "confirmation_required"
            # and never calls the manager.
            options["confirmation_handler"] = (lambda _prompt: True) if values["confirm"] else None

        return self._from_execution(spec.name, self._executor.execute(raw_action, **options))

    @staticmethod
    def _from_execution(tool: str, result: ExecutionResult) -> dict[str, Any]:
        return ToolResponse(
            ok=result.success,
            tool=tool,
            status=result.status,
            message=result.message,
            action=result.action,
            result=result.result,
            safety={
                "risk_level": result.risk_level,
                "requires_confirmation": result.requires_confirmation,
                "confirmed": result.confirmed,
            },
            errors=list(result.errors),
        ).to_dict()

    @staticmethod
    def _from_intent(tool: str, intent: IntentResult) -> dict[str, Any]:
        action = intent.action.to_dict() if intent.action is not None else None
        return ToolResponse(
            ok=intent.ok,
            tool=tool,
            status=intent.status,
            message=intent.message,
            action=action,
            result={"executed": False},
            errors=list(intent.errors),
        ).to_dict()
