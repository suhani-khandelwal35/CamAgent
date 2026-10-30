# Demo and release checklist

A reproducible walk-through of CamAgent for a demo or a release review. Each
step gives the exact command and the result you should observe. Steps are
labelled:

- **Mocked**: the Linux utilities or the LLM are replaced by a stand-in inside
  the snippet. Runs on any host. Nothing touches the kernel or the Gemini API.
- **Real Linux**: runs the real `lsmod` and `modinfo` on a Linux host. Still
  read-only. No step in this checklist loads, unloads, or reloads a module.
- **Review**: a manual check with no command output to compare.

Completing this checklist is evidence for a release only when every expected
result below was actually observed. Record what was run, on which host, in
[Status](#status-as-of-2026-10-30) and keep that section honest.

All commands run from the repository root with the virtual environment active.
Multi-line Python snippets use `python - <<'EOF'`, which is bash syntax; on
Windows PowerShell save the snippet to a file and run `python file.py` with
`PYTHONPATH` set to the repository root.

## 1. Environment prerequisites and setup

- [ ] Python 3.10 or newer:

  ```bash
  python3 --version
  ```

  Expected: `Python 3.10` or later. Older versions fail at import time.

- [ ] Virtual environment and dependencies:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  python -m pip check
  ```

  Expected: the install completes and `pip check` prints
  `No broken requirements found.`

- [ ] Linux utilities present (Real Linux; skip on other hosts):

  ```bash
  command -v lsmod modinfo modprobe
  ```

  Expected: three paths, typically under `/usr/sbin` or `/sbin`.

## 2. CLI read-only operations

**Real Linux.** These commands run `lsmod` and `modinfo` and change nothing.

- [ ] Start the CLI and list, inspect, and check:

  ```bash
  python -m app.cli
  ```

  ```text
  camagent> list
  camagent> info loop
  camagent> loaded loop
  camagent> exit
  ```

  Expected: each command prints a `SUCCESS:` line followed by JSON. `list`
  returns `modules` with `name`, `size`, and `used_by` per entry; `info`
  returns the `modinfo` fields; `loaded` returns `"loaded": true` or `false`,
  matching whether `loop` appears in `lsmod`. Pick another module with
  `modinfo` metadata if `loop` is unavailable on the host.

  On a host without the utilities (for example macOS or Windows) the same
  commands print `FAILURE: Module manager operation failed.` with a
  `FileNotFoundError` in `error`. That demonstrates error handling but does
  not count as passing this step.

## 3. Intent parsing without execution

**Mocked.** The LLM is replaced by a class returning a fixed classification,
so no API key is needed and no network call is made.

- [ ] Parse a natural-language request through the MCP dispatcher:

  ```bash
  python - <<'EOF'
  import json
  from app.llm import IntentParser
  from app.mcp_server import ToolDispatcher

  class FakeLLM:
      """Stands in for the Gemini provider: returns a fixed classification."""
      def complete(self, system, prompt, response_schema=None):
          return json.dumps({
              "status": "action",
              "action": {"operation": "check_module", "module": "uvcvideo"},
              "reason": "the webcam driver is uvcvideo",
          })

  dispatcher = ToolDispatcher(intent_parser=IntentParser(FakeLLM()))
  response = dispatcher.call("parse_request", {"request": "is the webcam driver loaded?"})
  print(json.dumps(response, indent=2))
  EOF
  ```

  Expected: `"status": "parsed"`, `"action": {"operation": "check_module",
  "module": "uvcvideo"}`, and `"result": {"executed": false}`. No utility is
  run; the parsed action is only returned.

- [ ] Optional, with a real key: set `GEMINI_API_KEY`, start the MCP server
  as in section 4, and call `parse_request` from a client (for example MCP
  Inspector, see [mcp.md](mcp.md#mcp-inspector)). Expected: the same envelope
  shape with `"result": {"executed": false}`. Not required for the release
  and not covered by the automated tests.

## 4. MCP server start and tool discovery

**Real stdio transport, any host.** The snippet launches `python -m
app.mcp_server` as a subprocess and talks to it over stdin/stdout, which is
the only supported transport.

- [ ] Default server (read-only tools):

  ```bash
  python - <<'EOF'
  import os
  import sys

  import anyio
  from mcp import Client, StdioServerParameters

  async def main():
      params = StdioServerParameters(
          command=sys.executable, args=["-m", "app.mcp_server"], env=dict(os.environ)
      )
      async with Client(params) as client:
          print([tool.name for tool in (await client.list_tools()).tools])
          result = await client.call_tool("check_module", {"module": "uvcvideo"})
          print(result.structured_content)

  anyio.run(main)
  EOF
  ```

  Expected first line:

  ```text
  ['list_modules', 'get_module_info', 'check_module', 'parse_request']
  ```

  Expected second line on Linux: a dict with `'status': 'success'` and
  `'result': {'loaded': True}` or `False`. On a host without `lsmod` the dict
  has `'status': 'error'` and a `FileNotFoundError` in `errors`; tool
  discovery still succeeded.

- [ ] Server with state-changing tools exposed:

  ```bash
  CAMAGENT_MCP_ALLOW_MUTATIONS=1 python - <<'EOF'
  # same snippet as above
  EOF
  ```

  Expected first line lists seven tools, ending with `'load_module',
  'unload_module', 'reload_module'`. The `env=dict(os.environ)` argument is
  what carries the variable to the server subprocess; without it the MCP
  client starts the server with a minimal environment and the list stays at
  four tools. Do not call the three state-changing tools with `confirm: true`
  during the demo.

- [ ] Registering with an MCP client follows [mcp.md](mcp.md#connecting-a-client).
  Expected after registration: the client's tool list shows the same four (or
  seven) tools.

## 5. Safe rejection of invalid requests

**Any host, real code.** Every request below is refused before any utility
runs, so the output is identical on Linux and elsewhere.

- [ ] CLI rejections:

  ```bash
  python -m app.cli
  ```

  ```text
  camagent> unload ext4
  camagent> load -r
  camagent> {"command": "load", "module": "snd; reboot"}
  camagent> load snd; reboot
  camagent> reload the webcam driver
  camagent> exit
  ```

  Expected, in order:

  | Input | Output |
  | --- | --- |
  | `unload ext4` | `VALIDATION REJECTED: Action was rejected by safety policy.` with error `module 'ext4' is protected from unload_module`. No confirmation prompt appears. |
  | `load -r` | `VALIDATION REJECTED: Action failed validation.` with error `module name must not start with '-' (looks like a command flag)` |
  | JSON with `snd; reboot` | `VALIDATION REJECTED: Action failed validation.` with error `module name contains forbidden character(s): [' ', ';']` |
  | `load snd; reboot` | `Error: Commands accept at most one module name.` |
  | `reload the webcam driver` | `Error: Commands accept at most one module name.` (the CLI has no natural-language input) |

- [ ] MCP rejections:

  ```bash
  python - <<'EOF'
  from app.mcp_server import ToolDispatcher

  read_only = ToolDispatcher()                  # default: mutations disabled
  full = ToolDispatcher(allow_mutations=True)

  calls = [
      (read_only, "load_module", {"module": "uvcvideo", "confirm": True}),
      (full, "check_module", {"module": "snd; reboot"}),
      (full, "load_module", {"module": "uvcvideo", "confirm": "true"}),
      (full, "unload_module", {"module": "ext4", "confirm": True}),
      (full, "run_shell", {"command": "reboot"}),
  ]
  for dispatcher, tool, arguments in calls:
      response = dispatcher.call(tool, arguments)
      print(f"{tool:14} {response['status']:18} {response['errors']}")
  EOF
  ```

  Expected:

  ```text
  load_module    tool_disabled      []
  check_module   invalid_arguments  ["module name contains forbidden character(s): [' ', ';']"]
  load_module    invalid_arguments  ["argument 'confirm' must be a bool, got str"]
  unload_module  rejected           ["module 'ext4' is protected from unload_module"]
  run_shell      unknown_tool       ["available tools: [...]"]
  ```

## 6. Confirmation and dry-run behaviour

**Mocked.** `lsmod` is replaced by a stand-in that reports `snd` as loaded.
The stand-in raises if anything tries to run `modprobe`, so the demo proves
that none of these paths executes a change.

- [ ] Dry run, missing confirmation, policy rejection, and a declined CLI
  confirmation:

  ```bash
  python - <<'EOF'
  import subprocess
  from app import cli
  from app.manager import module_manager
  from app.mcp_server import ToolDispatcher

  loaded = {"snd"}

  def fake_run(args):
      """Stands in for lsmod. Any attempt to run modprobe fails the demo."""
      if args[0] == "lsmod":
          output = "Module Size Used by\n" + "".join(f"{name} 16384 0\n" for name in sorted(loaded))
          return subprocess.CompletedProcess(args, 0, output, "")
      raise AssertionError(f"the demo must never run {args}")

  module_manager._run_command = fake_run

  mcp = ToolDispatcher(allow_mutations=True)
  for tool, arguments in [
      ("load_module", {"module": "uvcvideo", "dry_run": True}),
      ("load_module", {"module": "uvcvideo"}),
      ("unload_module", {"module": "ext4", "confirm": True}),
  ]:
      response = mcp.call(tool, arguments)
      print(f"{response['status']:22} {response['message']}  result={response['result']}")

  answers = iter(["unload snd", "n", "exit"])
  cli.run_cli(input_fn=lambda prompt: next(answers), output_fn=print)
  EOF
  ```

  Expected:

  ```text
  dry_run                Would perform load_module uvcvideo. Risk: disruptive. Confirmation will be required before execution.  result={'would_execute': True, 'summary': '...'}
  confirmation_required  Execution blocked because confirmation was not provided.  result=None
  rejected               Action was rejected by safety policy.  result=None
  CamAgent interactive CLI. Type 'help' for commands.
  DECLINED: Execution was not confirmed.
  {
    "success": false,
    "operation": "unload",
    "module": "snd",
    "error": "operation was declined"
  }
  Goodbye.
  ```

  The snippet exits without an `AssertionError`, which shows `modprobe` was
  never attempted. The interactive equivalent is `unload snd` in the CLI
  followed by answering `n` at `Confirm unload_module snd? Risk: disruptive. [y/N]`.

## 7. Test suite

**Mocked.** All tests replace Linux commands and the Gemini client; none loads
or unloads a module or contacts the Gemini API, and no `GEMINI_API_KEY` is
needed.

- [ ] Full suite:

  ```bash
  python -m pytest -q
  ```

  Expected for a release: every test passes and no file fails to collect.
  See [Status](#status-as-of-2026-10-30) for the current result.

- [ ] Focused groups. Counts are as of 2026-10-30 and grow as tests are added;
  the requirement is `0 failed`.

  | Group | Command | Expected |
  | --- | --- | --- |
  | Safety: schema, validator, policy, confirmation, executor, shared CLI/MCP policy | `python -m pytest -q tests/test_actions.py tests/test_validator.py tests/test_policy.py tests/test_confirmation.py tests/test_executor.py tests/test_shared_safety.py` | `149 passed` |
  | Module manager and CLI adapter | `python -m pytest -q tests/test_module_manager.py tests/test_cli.py` | `56 passed` |
  | MCP: dispatcher, argument validation, protocol | `python -m pytest -q tests/test_mcp_tools.py tests/test_mcp_validation.py tests/test_mcp_server.py tests/test_mcp_protocol_edge_cases.py` | `545 passed` (protocol files are skipped if `mcp` is missing) |
  | Intent parsing and Gemini provider | `python -m pytest -q tests/test_intent.py tests/test_intent_edge_cases.py tests/test_gemini_provider.py` | `427 passed` |
  | End-to-end workflows | `python -m pytest -q tests/test_end_to_end_workflows.py` | `134 passed` |

## 8. Opt-in Linux integration checks

**Real Linux, read-only.** Runs real `lsmod` and `modinfo`; never `modprobe`.
Skipped automatically on other hosts and without the switch.

- [ ] On a Linux host with the utilities installed:

  ```bash
  CAMAGENT_RUN_LINUX_INTEGRATION=1 python -m pytest tests/test_linux_integration.py -v
  ```

  Expected: `3 passed`. Each test name is printed with `PASSED`. If `loop`
  has no `modinfo` metadata on the host, rerun with
  `CAMAGENT_TEST_MODULE=<installed module>`.

  On any other host, or without the variable, the same command prints
  `3 skipped` with the reason
  `requires Linux and CAMAGENT_RUN_LINUX_INTEGRATION=1`. That is not a pass.

## 9. Release review

**Review.**

- [ ] Working tree: `git status --short` prints nothing, and
  `git log --oneline -5` shows the intended commits.
- [ ] Whitespace: `git diff --check` on the release range prints nothing.
- [ ] Documentation: every relative link in `README.md` and `docs/*.md`
  resolves, the command tables match `python -m app.cli` and `help`, and the
  tool and status tables in [mcp.md](mcp.md) match the output of sections 4
  and 5.
- [ ] Dependencies: `requirements.txt` lists `pytest`, `google-genai`, and
  `mcp`; `python -m pip check` reports no broken requirements; every
  third-party import in `app/` is covered (`anyio`, `mcp`, `mcp_types` come
  with `mcp`; `google.genai` is imported lazily and only by `parse_request`).
- [ ] No leftover LLM-provider configuration: `git grep -i anthropic` prints
  nothing, and `.env` files are ignored (`git check-ignore .env` prints
  `.env`).
- [ ] Known limitations in [README.md](../README.md#current-limitations) are
  still accurate: commands-only CLI, stdio-only MCP, explicit `confirm`
  argument instead of elicitation, dry run only over MCP, one device alias set,
  no module parameters, no privilege elevation.
- [ ] Nothing in the release notes claims that module loading or unloading was
  verified on real hardware unless someone did that deliberately outside this
  checklist.

## Status as of 2026-10-30

Recorded from a Windows 11 host with Python 3.13.7 after migrating
`parse_request` from the Claude API to the Google Gemini API (`google-genai`
2.34.0, `mcp` 2.3.0). Update this section whenever the checklist is rerun.

Run and observed as expected:

- Section 1: `pip install -r requirements.txt` and `pip check` clean
  (`No broken requirements found.`). Linux utilities not present on this host.
- Section 3: mocked intent demo printed `parsed`, `check_module uvcvideo`,
  `executed: false`.
- Section 4: stdio tool discovery against the real server listed four tools
  by default and seven with `CAMAGENT_MCP_ALLOW_MUTATIONS=1`. With no
  `GEMINI_API_KEY` in the server's environment, `parse_request` returned
  `provider_error` with `GEMINI_API_KEY is not set; ...` in `errors` and
  `result: {"executed": false}`; invalid arguments and a module name with
  `;` returned `invalid_arguments`.
- Section 7, full suite: `python -m pytest -q` exited 0 with
  `1311 passed, 3 skipped` (the 3 skips are the Linux integration checks).
  Focused groups: 149, 56, 545, 427, and 134 passed, 0 failed.
- Section 9: `git diff --check` printed nothing; no `anthropic` or
  `ANTHROPIC_API_KEY` reference remains in code, configuration, or
  documentation; `.env` is ignored and `.env.example` is tracked.

Not run:

- Section 2 (real Linux read-only CLI), section 5, section 6, and section 8
  (Linux integration checks) were not rerun for this status; no Linux host
  was available and sections 5 and 6 are unaffected by the provider change
  but should be rerun before a release. Section 8 prints `3 skipped` here.
- Section 3 with a real `GEMINI_API_KEY` was not run: no key was available,
  so the live request shape (`response_json_schema` with the envelope
  schema against `gemini-3.8-flash`) is verified only against the SDK's
  signature and mocked responses, not against the Gemini service.
