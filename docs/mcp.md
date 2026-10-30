# MCP server and natural-language integration

CamAgent exposes its module manager to MCP clients (such as Claude Desktop or
Claude Code) as a fixed set of tools. Every tool call is validated and routed
through the same validator, safety policy, and confirmation gate as the rest of
CamAgent. There is no tool for running shell commands, and tool arguments never
reach a shell.

## Components

| Component | Responsibility |
| --- | --- |
| `app/mcp_server/server.py` | MCP protocol adapter: lists tools, receives calls over stdio, and returns results. Uses the low-level MCP server so arguments arrive unmodified. |
| `app/mcp_server/tools.py` | Tool catalog, strict argument validation, routing to the executor or intent parser, and the response envelope. Has no MCP dependency. |
| `app/agent/executor.py` | Runs schema validation, the safety policy, dry-run, and confirmation before calling the module manager. |
| `app/llm/intent.py` | Turns a natural-language request into one structured action, or rejects it. Never executes anything. |
| `app/llm/provider.py` | LLM provider interface and the Google Gemini implementation (`google-genai` SDK). |

Request flow for a module tool:

```text
MCP client
    ↓  tools/call (name, raw arguments)
Tool dispatcher: known tool? mutations enabled? arguments valid?
    ↓
ActionExecutor: schema → module-name validation → safety policy
    ↓            → dry run or confirmation gate
Module manager (lsmod / modinfo / modprobe, no shell)
```

## Tools

The server always exposes the read-only tools below. The state-changing tools
are exposed only when `CAMAGENT_MCP_ALLOW_MUTATIONS` is enabled (see
[Configuration](#configuration)).

| Tool | Arguments | Changes state | Description |
| --- | --- | --- | --- |
| `list_modules` | none | No | List loaded kernel modules. |
| `get_module_info` | `module` | No | Show `modinfo` details for one module. |
| `check_module` | `module` | No | Report whether one module is loaded. |
| `parse_request` | `request` | No | Translate natural language into a structured action without running it. |
| `load_module` | `module`, `confirm`, `dry_run` | Yes | Load a module. |
| `unload_module` | `module`, `confirm`, `dry_run` | Yes | Unload a module. |
| `reload_module` | `module`, `confirm`, `dry_run` | Yes | Unload and reload a module. |

### Input rules

Arguments must be a JSON object that matches the tool's input schema exactly:

- Unknown arguments are rejected (`additionalProperties: false`).
- Types are not converted: `"confirm": "true"` or `"confirm": 1` is rejected.
- `module` must be a valid kernel module name: letters, digits, `_`, `-`,
  `.`; it must start with a letter or digit; it may be at most 55 characters;
  and it may contain no whitespace, shell metacharacters, path separators, or
  `..`. Hyphens are normalized to underscores, as `modprobe` does.
- `request` must be non-empty and at most 500 characters.
- `confirm` and `dry_run` default to `false` and cannot both be `true`.

State-changing tools only act when called with `confirm: true`. Without it the
response status is `confirmation_required` and the module manager is not
called. Use `dry_run: true` to preview what would happen and whether the safety
policy allows it. The policy still applies when a call is confirmed. For
example, `ext4`, `xfs`, `btrfs`, `overlay`, and `dm_mod` cannot be unloaded or
reloaded. Loading an already-loaded module, or unloading one that isn't
loaded, is also rejected.

### Response format

Every call returns the same object, both as `structuredContent` and as JSON
text content. The MCP `isError` flag is set whenever `ok` is `false`.

```json
{
  "ok": true,
  "tool": "check_module",
  "status": "success",
  "message": "check_module completed.",
  "action": {"operation": "check_module", "module": "uvcvideo"},
  "result": {"loaded": true},
  "safety": {"risk_level": "read_only", "requires_confirmation": false, "confirmed": false},
  "errors": []
}
```

| Field | Meaning |
| --- | --- |
| `ok` | `true` only for `success`, `dry_run`, or a `parsed` request. |
| `tool` | The tool that was called. |
| `status` | Outcome code; see below. |
| `message` | Short human-readable summary. |
| `action` | The validated structured action, or `null` if none was produced. |
| `result` | Tool output: the module list, `modinfo` fields, `{"loaded": bool}`, the manager result, a dry-run preview, or `{"executed": false}` for `parse_request`. |
| `safety` | Risk level and confirmation details for module tools; `null` otherwise. |
| `errors` | Specific reasons for a failure. |

Status codes:

| Status | Meaning |
| --- | --- |
| `success` | The operation ran and succeeded. |
| `dry_run` | Preview only; nothing was executed. |
| `parsed` | `parse_request` produced a valid action (not executed). |
| `invalid_arguments` | Arguments did not match the tool's input schema or rules. |
| `unknown_tool` | No tool with that name. |
| `tool_disabled` | A state-changing tool was called while mutations are disabled. |
| `rejected` | Schema validation or the safety policy refused the action. |
| `confirmation_required` | A state-changing call was made without `confirm: true`. |
| `error` | The module manager failed, for example because `lsmod` is unavailable. |
| `permission_error` | A utility or module-state check reported insufficient permissions. |
| `manager_error` | The module manager reported an unsuccessful operation. |
| `internal_error` | Unexpected failure; details are logged on the server, not returned. |
| `unsupported`, `ambiguous` | `parse_request` could not map the request to one safe action. |
| `invalid_output`, `provider_error`, `invalid_request` | `parse_request` failed: bad model output, LLM unavailable, or empty or overlong request text. |

## Natural-language parsing

`parse_request` sends the request text to Gemini with a fixed system prompt
and a JSON response schema (constrained decoding), and requires a JSON reply
in this form:

```json
{"status": "action", "action": {"operation": "check_module", "module": "uvcvideo"}, "reason": "..."}
```

The `action` object is the existing structured action schema and goes through
the same validation as any other action. A request is rejected rather than
guessed when the model reports it as unsupported or ambiguous, when the reply
is malformed, when the action fails validation, or when the module is not
named in the request. The only exception is the documented device aliases in
`app/llm/intent.py`: `webcam`, `web cam`, `usb camera`, `uvc camera`, and
`uvc driver` all map to `uvcvideo`.

`parse_request` never executes the action it returns. A client that wants to
act on it calls the matching tool, so the normal validation, policy, and
confirmation rules apply. Gemini is given no tools and no function-calling
configuration, so it cannot request a shell command; the schema only allows
the six operations and a module name, and a reply outside that shape is
rejected with `invalid_output`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `CAMAGENT_MCP_ALLOW_MUTATIONS` | unset (disabled) | Set to `1`, `true`, `yes`, or `on` to expose `load_module`, `unload_module`, and `reload_module`. |
| `GEMINI_API_KEY` | none | Google AI Studio API key for `parse_request`. Only needed when `parse_request` is used. See the README section [Gemini API key](../README.md#gemini-api-key) for how to obtain one and set it on Windows and Linux. |
| `CAMAGENT_MODEL` | `gemini-3.8-flash` | Gemini model used by `parse_request`. |

The server starts without an API key. If none is available, `parse_request`
returns `provider_error` (the `errors` entry names `GEMINI_API_KEY`) and the
other tools work as usual. The key is read from the environment only and is
never written to logs or error messages.

## Running the server

Install the dependencies from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start the server over stdio:

```bash
python -m app.mcp_server
```

The server communicates over stdin/stdout, so it is normally launched by an
MCP client rather than run by hand. Logs go to stderr.

### Privileges

Read-only tools run as an ordinary user. Loading and unloading modules usually
requires root. CamAgent never elevates privileges itself; without the needed
permission, the operation fails and the error is returned. Only enable
mutations, or run the server with elevated privileges, on machines where the
connected client is trusted to request module changes.

## Connecting a client

Replace `/path/to/CamAgent` with the absolute path to this repository. Setting
`PYTHONPATH` lets the client start the server from any working directory.

### Claude Code

```bash
claude mcp add camagent \
  -e PYTHONPATH=/path/to/CamAgent \
  -- /path/to/CamAgent/.venv/bin/python -m app.mcp_server
```

Add `-e CAMAGENT_MCP_ALLOW_MUTATIONS=1` to expose the state-changing tools, and
`-e GEMINI_API_KEY=...` to enable `parse_request`.

### Claude Desktop and other JSON-configured clients

```json
{
  "mcpServers": {
    "camagent": {
      "command": "/path/to/CamAgent/.venv/bin/python",
      "args": ["-m", "app.mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/CamAgent",
        "CAMAGENT_MCP_ALLOW_MUTATIONS": "0"
      }
    }
  }
}
```

### MCP Inspector

[MCP Inspector](https://github.com/modelcontextprotocol/inspector) is a
browser-based client for trying tools by hand. It needs Node.js. The
instructions below pin **Inspector 1.0.3** (the v1 interface). Inspector 2.x
is a different application: its launcher has no `-e` flag, it starts a stdio
server with only a fixed safe subset of variables plus the `env` of a
configured server entry (the shell's own variables are not inherited), and
servers are managed through a config or catalog file. Pinning the version
keeps the interface stable and keeps the environment rules below valid.
Unpinned `npx @modelcontextprotocol/inspector` installs 2.x.

How 1.0.3 passes the environment to the server, from its source: the proxy
spawns the server with the MCP SDK's safe subset (`PATH`, `SYSTEMROOT`,
`APPDATA`, `TEMP`, ... on Windows), then the Inspector's own process
environment, then whatever the UI's **Environment Variables** panel
contains. `-e NAME=value` pre-fills that panel. Consequences:

- A variable set in the Command Prompt window that runs `npx` reaches the
  server, and so does a variable passed with `-e`.
- A `GEMINI_API_KEY` row with an empty value in the panel overrides the shell
  and makes CamAgent report `GEMINI_API_KEY is not set`. Fill it in or
  remove the row, then press **Connect** (or **Restart**) again.
- Variables set in the shell after the Inspector started are not seen until
  the Inspector is restarted. Panel changes apply on the next connect, since
  every connect starts a new server process.
- The proxy's console prints the panel's variable names, never values, as
  `env: [names: ...]`, which shows what was sent.

Windows Command Prompt, from the repository root:

```bat
cd /d C:\path\to\CamAgent
set GEMINI_API_KEY=<your key>
set CAMAGENT_MODEL=gemini-3.8-flash
npx -y @modelcontextprotocol/inspector@1.0.3 -e GEMINI_API_KEY=%GEMINI_API_KEY% -e CAMAGENT_MODEL=%CAMAGENT_MODEL% -- py -3.13 -m app.mcp_server
```

`set` takes no quotes and no spaces around `=`. `%GEMINI_API_KEY%` expands
only in the window where `set` ran; `setx` and the System Properties dialog
affect only windows opened afterwards. When the variable is not set, Command
Prompt passes the literal text `%GEMINI_API_KEY%` and Gemini rejects it with
an authentication error. For a screen recording, keep the key off the
command line: run `set GEMINI_API_KEY=...` before recording, then `cls`, and
launch without `-e GEMINI_API_KEY=...`; the proxy still inherits it from the
window.

Linux and macOS, with the virtual environment active:

```bash
cd /path/to/CamAgent
export GEMINI_API_KEY="<your key>"
npx -y @modelcontextprotocol/inspector@1.0.3 -e GEMINI_API_KEY="$GEMINI_API_KEY" -e CAMAGENT_MODEL=gemini-3.8-flash -- python -m app.mcp_server
```

Add `-e CAMAGENT_MCP_ALLOW_MUTATIONS=1` to see the state-changing tools too.
Then, in the browser page the command opens (its URL carries the session
token printed in the console):

1. Keep **Transport Type** at `STDIO`. **Command** shows `py` and
   **Arguments** `-3.13 -m app.mcp_server`. Open **Environment Variables**
   and check that `GEMINI_API_KEY` is listed with a value. Press
   **Connect**.
2. Open the **Tools** tab and press **List Tools**. The four read-only tools
   appear (seven with mutations enabled).
3. Select `parse_request`, type a request such as
   `is the webcam driver loaded?` in the `request` field, and press
   **Run Tool**.

Expected result: the response envelope with `"status": "parsed"`,
`"action": {"operation": "check_module", "module": "uvcvideo"}`, and
`"result": {"executed": false}`. Nothing is executed; to act on it, call
`check_module` with `{"module": "uvcvideo"}`. A request Gemini cannot map
returns `ambiguous` or `unsupported` with `"action": null`.

A `parse_request` call normally returns in a few seconds. While Gemini is overloaded it can take up to 90 seconds (4 attempts of at most 20 seconds each, plus backoff waits) before the tool result arrives; the Inspector's default Request Timeout is 5 minutes (Configuration, `MCP_SERVER_REQUEST_TIMEOUT`), so the spinner ends with a result rather than a client-side timeout.

`provider_error` entries you may see in `errors`:

| `errors` entry | Meaning |
| --- | --- |
| `GEMINI_API_KEY is not set ...` | The server process has no key: the variable was not in the window that launched the Inspector, the Inspector is 2.x, or the panel holds an empty `GEMINI_API_KEY` row. |
| `Gemini API authentication failed (HTTP 400 or 403) ...` | A key reached Gemini and was rejected: a typo, the literal `%GEMINI_API_KEY%`, or a revoked key. The key value is never included. |
| `Gemini API is unavailable (HTTP 503) ...` | Gemini's side: the model is temporarily overloaded, and the SDK's automatic retries (4 attempts, exponential backoff with jitter, delays capped at 10s) did not get through. Not an application error. Retry, or set `CAMAGENT_MODEL` to another current model. |
| `Gemini API rate limit or quota exceeded (HTTP 429) ...` | The key's quota is exhausted for now. Wait, or use another model. |
| `Gemini API did not respond within 20s per attempt (4 attempts made) ...` | Gemini accepted each request but never answered. Every attempt is capped at 20 seconds and retried, so the call ends within 90 seconds. Google's side; retry later or use another model. |

The same check from the command line, without the browser, uses the
Inspector's CLI mode (flags before the command, no `--`):

```bat
npx -y @modelcontextprotocol/inspector@1.0.3 --cli -e GEMINI_API_KEY=%GEMINI_API_KEY% py -3.13 -m app.mcp_server --method tools/call --tool-name parse_request --tool-arg "request=is the webcam driver loaded?"
```

### Testing with the MCP Python client

Run this from the repository root with the virtual environment's Python:

```python
import os
import sys

import anyio
from mcp import Client, StdioServerParameters

async def main():
    # The MCP client starts the server with a minimal environment unless `env`
    # is given; pass it so CAMAGENT_MCP_ALLOW_MUTATIONS and GEMINI_API_KEY
    # exported in your shell reach the server process.
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "app.mcp_server"], env=dict(os.environ)
    )
    async with Client(params) as client:
        print([tool.name for tool in (await client.list_tools()).tools])
        result = await client.call_tool("check_module", {"module": "uvcvideo"})
        print(result.structured_content)

anyio.run(main)
```

## Troubleshooting

| Symptom | Cause | What to do |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'mcp'` on start | The `mcp` package is not installed in the Python the client launches | Install `requirements.txt` into the virtual environment and point the client at that environment's `python`. |
| `ModuleNotFoundError: No module named 'app'` on start | The server was launched from another working directory | Set `PYTHONPATH` to the repository root in the client configuration, as in the examples above. |
| `load_module`, `unload_module`, and `reload_module` are missing from the tool list, or return `tool_disabled` | Mutations are disabled | Set `CAMAGENT_MCP_ALLOW_MUTATIONS=1` in the server's environment and restart it. |
| `confirmation_required` | The call did not include `confirm: true` | Ask the user, then repeat the call with `confirm: true`. Use `dry_run: true` to preview first. |
| `invalid_arguments` with `argument 'confirm' must be a bool, got str` | The client sent `"true"` as a string | Send a JSON boolean. Values are never coerced. |
| `rejected` | The safety policy refused the action (protected module, already loaded, not loaded) | Check the `errors` field; these actions are not retried. |
| `error` with `FileNotFoundError` in `errors` | `lsmod`, `modinfo`, or `modprobe` is not installed, or the host is not Linux | Install the `kmod` utilities on a Linux host. |
| `permission_error` | A utility reported insufficient permissions | Run the server with the privileges the operation needs; it never elevates itself. |
| `provider_error` from `parse_request` | `GEMINI_API_KEY` is not set or invalid, the Gemini quota is exhausted, the network is down, or the `google-genai` package is missing | Read the `errors` field. Set `GEMINI_API_KEY` for the server process (with `-e` or the `env` block) and install `google-genai`. Other tools are unaffected. |
| `internal_error` | An unexpected failure | Details are written to the server's stderr log, not returned to the client. |

## Integration status

Working:

- MCP server over stdio with the read-only tools and opt-in state-changing
  tools listed above.
- Strict argument validation, safety policy, dry-run, and confirmation gating
  for every tool call.
- Natural-language parsing via `parse_request` with the Google Gemini
  provider (`google-genai` SDK, `GEMINI_API_KEY`), plus an `LLMProvider`
  interface for alternative or mocked providers.
- `lsmod` failures reported as errors rather than as "module not loaded".
- CLI and MCP actions routed through the same `ActionExecutor` validation,
  state policy, confirmation, dispatch, and post-operation verification path.
- Unit tests that mock the LLM and module manager, and protocol tests that use
  an in-process MCP client.

Planned or not yet available:

- Natural-language input in the interactive CLI (`python -m app.cli` still
  accepts only commands and JSON).
- Streamable HTTP or SSE transports and authentication for remote clients.
  Only stdio is supported.
- Interactive confirmation through MCP elicitation. Confirmation is currently
  the explicit `confirm` argument, which the client must set after asking the
  user.
- Additional device aliases beyond the webcam driver, and module parameters
  for `load_module`.
