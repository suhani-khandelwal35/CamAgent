# CamAgent

CamAgent inspects and manages Linux kernel modules through a guarded interactive
command-line interface and an MCP server. Module changes require explicit
confirmation and are reported only after checking the resulting module state.

Contents:

- [Requirements](#requirements)
- [Installation](#installation)
- [Running the CLI](#running-the-cli)
- [Expected output](#expected-output)
- [Safety behaviour](#safety-behaviour)
- [Gemini API key](#gemini-api-key)
- [MCP server](#mcp-server)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Current limitations](#current-limitations)

For a step-by-step demo and pre-release walk-through with expected outputs,
see [docs/release-checklist.md](docs/release-checklist.md).

## Requirements

- Python 3.10 or newer. The code uses `X | None` type syntax at runtime and
  every dependency in `requirements.txt` requires Python 3.10 or newer.
- A Linux host with `lsmod`, `modinfo`, and `modprobe` on `PATH` for any
  command that touches kernel modules.
- Appropriate system permissions for module changes (typically root). CamAgent
  never elevates privileges itself.
- Optional: the `mcp` package to run the MCP server, and the `google-genai`
  package plus a Gemini API key to use the `parse_request` MCP tool. Both
  are listed in `requirements.txt`; the CLI works without them.

The CLI and tests start on other platforms, but every module command needs the
Linux utilities above and fails with a command error without them (see
[Troubleshooting](#troubleshooting)).

## Installation

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead of
`source .venv/bin/activate`.

`requirements.txt` installs `pytest`, `google-genai`, and `mcp`. There is no
package to install for CamAgent itself: run it from the project directory with
`python -m ...` so that the `app` package is importable.

## Running the CLI

Start the interactive CLI:

```bash
python -m app.cli
```

The prompt is `camagent> `. Commands are case-insensitive and take at most one
module name. Type `help` to print the command list.

| Command | Structured JSON form | What it does | Linux utility | Confirmation |
| --- | --- | --- | --- | --- |
| `list` | `{"command": "list"}` | List loaded modules with size and users | `lsmod` | No |
| `info <module>` | `{"command": "info", "module": "uvcvideo"}` | Show module metadata | `modinfo` | No |
| `loaded <module>` | `{"command": "loaded", "module": "uvcvideo"}` | Report whether a module is loaded | `lsmod` | No |
| `load <module>` | `{"command": "load", "module": "uvcvideo"}` | Load a module | `lsmod`, `modprobe` | Yes |
| `unload <module>` | `{"command": "unload", "module": "uvcvideo"}` | Unload a module | `lsmod`, `modprobe -r` | Yes |
| `reload <module>` | `{"command": "reload", "module": "uvcvideo"}` | Unload, then load a module | `lsmod`, `modprobe -r`, `modprobe` | Yes |
| `help` | | Print the command list | | |
| `exit`, `quit` | | Leave the CLI | | |

Notes on input:

- A JSON request must be an object with only `command` and, for module
  commands, `module`. Any other key is rejected.
- Plain-text input is split like a shell command line, so `load snd; reboot`
  is rejected as "Commands accept at most one module name." and a quoted
  `load 'snd; reboot'` reaches the validator, which rejects the name.
- Blank input is ignored. `exit`, `quit`, Ctrl-D, or Ctrl-C at the prompt
  leaves the CLI with `Goodbye.`.
- The CLI accepts commands and JSON only. It does not accept natural-language
  requests; `reload the webcam driver` is an error. Natural language is
  available through the MCP `parse_request` tool (see [MCP server](#mcp-server)).

### Confirmation prompt

`load`, `unload`, and `reload` first check the live module state, then ask:

```text
Confirm load_module uvcvideo? Risk: disruptive. [y/N]
```

Only `y` or `yes` (any letter case, surrounding spaces ignored) proceeds.
Anything else, an empty answer, Ctrl-D, or Ctrl-C declines, and nothing is
executed. The prompt is never shown for a request that validation or the safety
policy has already rejected.

## Expected output

Every command prints a one-line summary followed by a JSON block. The summary
starts with one of five labels:

| Label | Meaning |
| --- | --- |
| `SUCCESS:` | The operation ran. For `load`, `unload`, and `reload`, the module state was re-checked and matches. |
| `VALIDATION REJECTED:` | The module name is unsafe, or the safety policy refused the action (protected module, already loaded, not loaded). Nothing ran. |
| `DECLINED:` | A state-changing command was not confirmed. Nothing ran. |
| `PERMISSION ERROR:` | A utility reported insufficient permissions. |
| `FAILURE:` | A utility was missing, failed, returned malformed output, or the post-operation state check did not match. |

The JSON block always contains `success`, `operation`, and `module`. It adds
`modules` for `list`, `loaded` for `loaded`, the `modinfo` fields for `info`,
`verification` after a state change, and `error` with the first specific error
when something went wrong.

Listing modules:

```text
camagent> list
SUCCESS: list_modules completed.
{
  "success": true,
  "operation": "list",
  "module": null,
  "modules": [
    {
      "name": "snd",
      "size": "16384",
      "used_by": [
        0
      ]
    }
  ]
}
```

Checking whether a module is loaded:

```text
camagent> loaded uvcvideo
SUCCESS: check_module completed.
{
  "success": true,
  "operation": "loaded",
  "module": "uvcvideo",
  "loaded": false
}
```

Loading a module with confirmation. The verification block shows the requested
state and the state observed afterwards:

```text
camagent> load uvcvideo
Confirm load_module uvcvideo? Risk: disruptive. [y/N] yes
SUCCESS: module loaded successfully
{
  "success": true,
  "operation": "load",
  "module": "uvcvideo",
  "error": null,
  "verification": {
    "expected_loaded": true,
    "actual_loaded": true,
    "passed": true
  }
}
```

Common error responses:

```text
camagent> unload ext4
VALIDATION REJECTED: Action was rejected by safety policy.
{
  "success": false,
  "operation": "unload",
  "module": "ext4",
  "error": "module 'ext4' is protected from unload_module"
}

camagent> load snd
VALIDATION REJECTED: Action was rejected by safety policy.
{
  "success": false,
  "operation": "load",
  "module": "snd",
  "error": "module 'snd' is already loaded"
}

camagent> load -r
VALIDATION REJECTED: Action failed validation.
{
  "success": false,
  "operation": "load",
  "module": "-r",
  "error": "module name must not start with '-' (looks like a command flag)"
}

camagent> unload snd
Confirm unload_module snd? Risk: disruptive. [y/N] n
DECLINED: Execution was not confirmed.
{
  "success": false,
  "operation": "unload",
  "module": "snd",
  "error": "operation was declined"
}

camagent> info nope
FAILURE: modinfo: ERROR: Module nope not found.
{
  "success": false,
  "operation": "info",
  "module": "nope",
  "error": "modinfo: ERROR: Module nope not found."
}

camagent> load uvcvideo
Confirm load_module uvcvideo? Risk: disruptive. [y/N] y
PERMISSION ERROR: modprobe: ERROR: could not insert 'uvcvideo': Operation not permitted
{
  "success": false,
  "operation": "load",
  "module": "uvcvideo",
  "error": "PermissionDenied: modprobe: ERROR: could not insert 'uvcvideo': Operation not permitted"
}
```

Input that does not parse is reported on one line and nothing is executed,
for example `Error: Unknown command: bogus` or
`Error: 'load' requires a module name.`.

## Safety behaviour

The CLI and the MCP server share one execution path
(`app/agent/executor.py`), so the rules below apply to both. See
[docs/architecture.md](docs/architecture.md) for the component layout.

1. **Operation schema.** Only `list_modules`, `get_module_info`,
   `check_module`, `load_module`, `unload_module`, and `reload_module` exist.
   Nothing accepts a shell command, and utilities are invoked with argument
   lists, never through a shell.
2. **Module-name validation.** A name must start with a letter or digit;
   contain only letters, digits, `_`, `-`, and `.`; be at most 55 characters;
   and contain no whitespace, shell metacharacters, path separators, or `..`.
   Names starting with `-` are rejected so they can never be read as a
   `modprobe` flag. Hyphens are normalised to underscores, as `modprobe` does,
   so `snd-hda-intel` and `snd_hda_intel` are the same module.
3. **Protected modules.** `ext4`, `xfs`, `btrfs`, `overlay`, and `dm_mod`
   cannot be unloaded or reloaded. They can still be listed, inspected,
   checked, and loaded. This check runs before any utility is called.
4. **Live state checks.** Before a change, `lsmod` is consulted. Loading an
   already-loaded module, or unloading or reloading a module that is not
   loaded, is rejected. If `lsmod` fails, the request is reported as an error
   or permission error rather than guessed.
5. **Confirmation.** The CLI prompts as described above. MCP clients must pass
   `confirm: true`; without it the response status is `confirmation_required`
   and nothing runs.
6. **Dry run (MCP only).** MCP state-changing tools accept `dry_run: true`,
   which reports what would happen and whether policy allows it without
   running anything. The CLI has no dry-run option.
7. **Post-operation verification.** After `modprobe` succeeds, `lsmod` is
   consulted again. A successful command whose observed state does not match
   the request is reported as a failure, never as success.

## Gemini API key

Only the MCP `parse_request` tool talks to an LLM, through Google's Gemini API
and the official `google-genai` SDK. The CLI, the other MCP tools, and the test
suite never contact the Gemini API, and the MCP server starts without a key:
the key is read the first time `parse_request` is called.

Gemini only interprets the request and returns a structured action. Every
action it proposes goes through the same schema validation, module-name
validation, safety policy, and confirmation gate as any other input, in
deterministic Python code. Gemini is never asked for, and cannot trigger,
shell commands.

### Getting a key from Google AI Studio

1. Sign in at [Google AI Studio](https://aistudio.google.com/) with a Google
   account.
2. Open the [API keys page](https://aistudio.google.com/apikey) and choose
   **Create API key**. Select or create a Google Cloud project if asked.
3. Copy the key. You can create another one at any time, and you should
   delete a key as soon as it may have leaked.

### Configuring `GEMINI_API_KEY`

CamAgent reads the key from the `GEMINI_API_KEY` environment variable of the
process that runs the MCP server. Set it in the shell that launches the
server, or in the MCP client's `env` block (see [MCP server](#mcp-server)).

Windows PowerShell, for the current window only:

```powershell
$env:GEMINI_API_KEY = "<your key>"
python -m app.mcp_server
```

Windows, stored for your user account (takes effect in new windows):

```powershell
[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "<your key>", "User")
```

Windows Command Prompt: `set GEMINI_API_KEY=<your key>` for the current
window, or `setx GEMINI_API_KEY "<your key>"` to store it.

Linux (and macOS), for the current shell only:

```bash
export GEMINI_API_KEY="<your key>"
python -m app.mcp_server
```

To keep it across shells, add the `export` line to `~/.bashrc`, `~/.zshrc`,
or `~/.profile`, or to a systemd unit's `Environment=` if the server runs as a
service.

[.env.example](.env.example) lists every variable CamAgent reads. You can copy
it to `.env` to keep your own values; `.env` is ignored by Git. CamAgent does
not load `.env` by itself, so the variable still has to be set in the
environment as shown above. Never commit a key to the repository or paste one
into documentation or MCP client configuration that is shared.

Related settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | none | Google AI Studio API key for `parse_request`. Only needed when `parse_request` is used. |
| `GOOGLE_API_KEY` | none | Fallback for `GEMINI_API_KEY`, the other name the `google-genai` SDK accepts. Ignored when `GEMINI_API_KEY` is set. |
| `CAMAGENT_MODEL` | `gemini-3.8-flash` | Gemini model used by `parse_request`, read when the first request is parsed. Any current Gemini API model that supports JSON structured output can be set here. |

When the key is missing, `parse_request` returns status `provider_error`
with an error starting `GEMINI_API_KEY is not set in the environment of the
process running CamAgent`, and every other tool keeps working. "The process
running CamAgent" is the server subprocess that the MCP client starts, so
the variable must reach that process: through the client's `env` block, or
for MCP Inspector through `-e` or its Environment Variables panel (see
[docs/mcp.md](docs/mcp.md#mcp-inspector), which also explains why the pinned
Inspector 1.0.3 is used). An invalid key, an exhausted quota, or a network
failure is also reported as `provider_error`, with the HTTP status in the
`errors` field. When the `google-genai` package is not installed, the error
text is `the 'google-genai' package is required for GeminiProvider; install
it with 'pip install google-genai'`. The key itself never appears in any
error message.

## MCP server

The MCP server exposes CamAgent to MCP clients such as Claude Desktop or
Claude Code. The only transport is stdio: the client launches the server as a
subprocess and talks to it over stdin and stdout. Logs go to stderr.

```bash
python -m app.mcp_server
```

Tools:

| Tool | Changes state | Available by default | Requires |
| --- | --- | --- | --- |
| `list_modules` | No | Yes | |
| `get_module_info` | No | Yes | `module` |
| `check_module` | No | Yes | `module` |
| `parse_request` | No | Yes | `request`, `GEMINI_API_KEY` |
| `load_module` | Yes | Only with `CAMAGENT_MCP_ALLOW_MUTATIONS` | `module`, `confirm: true` (or `dry_run: true`) |
| `unload_module` | Yes | Only with `CAMAGENT_MCP_ALLOW_MUTATIONS` | `module`, `confirm: true` (or `dry_run: true`) |
| `reload_module` | Yes | Only with `CAMAGENT_MCP_ALLOW_MUTATIONS` | `module`, `confirm: true` (or `dry_run: true`) |

`parse_request` turns a natural-language sentence into a structured action and
never executes it; the client then calls the matching tool, so confirmation and
policy still apply. Set `CAMAGENT_MCP_ALLOW_MUTATIONS` to `1`, `true`, `yes`,
or `on` to expose the state-changing tools. Any other value, or leaving it
unset, hides them, and calling a hidden tool returns status `tool_disabled`.

Arguments are checked strictly: unknown arguments are rejected, types are not
converted (`"confirm": "true"` is rejected because it is a string), and
`module` must pass the validation in [Safety behaviour](#safety-behaviour).

Register the server with Claude Code (replace the path with the absolute path
to this repository):

```bash
claude mcp add camagent \
  -e PYTHONPATH=/path/to/CamAgent \
  -- /path/to/CamAgent/.venv/bin/python -m app.mcp_server
```

Add `-e CAMAGENT_MCP_ALLOW_MUTATIONS=1` to expose the state-changing tools and
`-e GEMINI_API_KEY=...` to enable `parse_request`.

[docs/mcp.md](docs/mcp.md) documents every tool's inputs, the response
envelope and status codes, the JSON configuration for Claude Desktop, how to
try `parse_request` in MCP Inspector, and a Python client example.

## Tests

Run the unit tests with:

```bash
python -m pytest
```

Tests mock Linux command interactions and the Gemini client, and never load or
unload real drivers or call the Gemini API; no `GEMINI_API_KEY` or network
access is needed. The MCP protocol tests are
skipped when the `mcp` package is not installed. The end-to-end tests in
`tests/test_end_to_end_workflows.py` drive the real CLI, executor, module
manager, and MCP dispatcher over a scripted stand-in for `lsmod`, `modinfo`,
and `modprobe`.

### Opt-in Linux integration checks

The Linux integration checks exercise real, read-only `lsmod`, `modinfo`, and
loaded-state inspection. They are skipped by default and on non-Linux hosts.
They never call `modprobe` or change kernel state.

Prerequisites:

- Linux with `lsmod` and `modinfo` installed and available on `PATH`
- A module with metadata available to `modinfo`; the default is `loop`
- Python dependencies installed as above

From the project directory, explicitly enable the checks:

```bash
CAMAGENT_RUN_LINUX_INTEGRATION=1 python -m pytest tests/test_linux_integration.py -v
```

To select a different module that exists on the test host, set
`CAMAGENT_TEST_MODULE`, for example:

```bash
CAMAGENT_RUN_LINUX_INTEGRATION=1 CAMAGENT_TEST_MODULE=uvcvideo \
  python -m pytest tests/test_linux_integration.py -v
```

The checks expect listing to return a list of parsed module records, metadata
inspection to succeed for the selected module, and the state check to agree
with the loaded-module listing. Missing utilities cause the relevant check to
be skipped. A selected module without metadata causes the inspection check to
fail; choose a module known to be installed for the running kernel.

These checks only cover listing, inspection, and state queries. Failure cases
such as missing commands, permission errors, and nonzero utility exit codes
are covered by mocked unit tests, not by intentionally breaking the host.
State-changing operations are also covered with mocks; they are not part of
the Linux integration suite. See [docs/linux-modules.md](docs/linux-modules.md)
for validation details and limitations.

## Troubleshooting

| Symptom | Cause | What to do |
| --- | --- | --- |
| `python3: command not found`, or `python -m app.cli` fails at import with a `TypeError` or `SyntaxError` | Python is missing or older than 3.10 | Install Python 3.10 or newer and recreate the virtual environment. Check with `python3 --version`. |
| `ModuleNotFoundError: No module named 'app'` | Started from outside the project directory | Run `python -m app.cli` from the repository root, or set `PYTHONPATH` to the repository as in the MCP example. |
| `ModuleNotFoundError: No module named 'mcp'` when starting `python -m app.mcp_server` | Dependencies not installed in the active environment | Activate the virtual environment and run `python -m pip install -r requirements.txt`. The CLI does not need `mcp`. |
| `FAILURE: Module manager operation failed.` with `FileNotFoundError` in `error` | `lsmod` is not installed, or you are not on Linux | Install the kernel module utilities (the `kmod` package on most distributions) and run on a Linux host. |
| `FAILURE:` with `modinfo command is unavailable` or `modprobe command is unavailable` | `modinfo` or `modprobe` is missing | Install the `kmod` utilities. |
| `PERMISSION ERROR:` on `load`, `unload`, or `reload` | The kernel refused the change for the current user | Run with sufficient privileges (usually root). CamAgent does not use `sudo` for you. |
| `PERMISSION ERROR: Permission denied while checking module state.` | `lsmod` itself could not be run | Fix permissions on the utility; the change is not attempted until the state is known. |
| `FAILURE:` with `lsmod returned malformed output` | `lsmod` printed something other than the standard `Module Size Used by` table | Check that `lsmod` is the standard `kmod` tool and not an alias or wrapper. CamAgent never treats unreadable output as "no modules loaded". |
| `FAILURE: modinfo returned no recognized metadata fields` | `modinfo` exited successfully but printed none of the expected fields | Check the module name and the `modinfo` installation. |
| `FAILURE: The requested module state could not be verified.` | `modprobe` exited successfully but `lsmod` does not show the requested state | Inspect `dmesg` and the module's dependencies; the result is reported as a failure on purpose. |
| `VALIDATION REJECTED:` with a module-name error | The name contains characters CamAgent refuses | Use the module's plain name, for example `uvcvideo`, not a path or `.ko` file. |
| MCP response `tool_disabled` | State-changing tools are hidden | Start the server with `CAMAGENT_MCP_ALLOW_MUTATIONS=1`. |
| MCP response `provider_error` from `parse_request` | `GEMINI_API_KEY` is not set or invalid, the Gemini quota is exhausted, the network is down, or the `google-genai` package is missing | Read the `errors` field: it names the missing variable, the HTTP status, or the missing package. Set `GEMINI_API_KEY` in the server's environment (see [Gemini API key](#gemini-api-key)) and install `google-genai`. The other tools are unaffected. |
| `provider_error` with `Gemini API is unavailable (HTTP 503)` | Gemini reported the model as temporarily overloaded, and it stayed overloaded through CamAgent's automatic retries (4 attempts with exponential backoff, about 1s, 2s and 4s apart) | A transient condition on Google's side, not a CamAgent error. Retry later, or point `CAMAGENT_MODEL` at another current model. |
| `provider_error` with `Gemini API did not respond within 20s per attempt` | Gemini accepted the connection but sent no reply within 20 seconds, on all 4 attempts | Also Google's side, seen during overload. Each attempt is capped at 20 seconds and the whole call at 90 seconds, so `parse_request` always returns. Retry later, or point `CAMAGENT_MODEL` at another current model. |

## Current limitations

- The CLI accepts commands and JSON only. Natural-language input is available
  only through the MCP `parse_request` tool.
- The MCP server supports stdio only. There is no HTTP or SSE transport and no
  authentication for remote clients.
- MCP confirmation is the explicit `confirm` argument. There is no interactive
  confirmation through MCP elicitation.
- Dry run exists only for the MCP state-changing tools, not in the CLI.
- `parse_request` recognises one set of device aliases (`webcam`, `web cam`,
  `usb camera`, `uvc camera`, `uvc driver`, all meaning `uvcvideo`). Any other
  module must be named explicitly in the request.
- `parse_request` does not check whether a module exists. Existence is only
  discovered when the matching tool runs.
- Module parameters cannot be passed to `load_module`.
- The protected-module list is replaceable in code (`SafetyPolicy`) but there
  is no configuration file or environment variable for it.
- CamAgent never elevates privileges, and its validation and protected-module
  policy do not replace operating-system access controls.
- The Linux integration checks are read-only. Whether a given module can be
  loaded or unloaded on a particular host is not something the test suite can
  establish.
