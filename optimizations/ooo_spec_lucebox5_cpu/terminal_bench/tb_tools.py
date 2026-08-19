"""Shared tool contract for the terminal-bench tool-speculation experiment.

Both the harbor agent (client side) and the dflash server-side speculative
executor import this module, so a predicted read-only call produces exactly the
same bytes the client would have produced by executing it itself.

Tools run inside the task's docker container via `docker exec`.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import time
from typing import Any

PROTOCOL = "dflash.tool-speculation.v1"
STATE_FILE = "/home/lucebox5/tbspec/state/current_container.json"
DOCKER = "/usr/bin/docker"

# Tools eligible for server-side speculation (read-only / idempotent).
READ_ONLY_TOOLS = ("read_file", "list_dir", "search_files")

RAW_STDOUT_CAP = 20000
RAW_STDERR_CAP = 4000
READ_ONLY_TIMEOUT_SEC = 30
DEFAULT_BASH_TIMEOUT_SEC = 120
MAX_BASH_TIMEOUT_SEC = 600

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a bash command in the task container and return its output. "
                "Use for anything that changes state (install, compile, run tests, "
                "git, moving files) or that the read-only tools cannot do."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to run."},
                    "timeout_sec": {
                        "type": "integer",
                        "description": "Optional timeout in seconds (default 120, max 600).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a text file (read-only). Returns numbered lines. "
                "Defaults to lines 1-200; pass start_line/end_line for other ranges."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or workdir-relative file path."},
                    "start_line": {"type": "integer", "description": "First line to show (1-based, default 1)."},
                    "end_line": {"type": "integer", "description": "Last line to show (default start_line+199)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List a directory (read-only), like `ls -la`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default '.')."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Search file contents recursively with grep -rn (read-only). "
                "Returns at most 200 matching lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression (grep -E syntax)."},
                    "path": {"type": "string", "description": "File or directory to search (default '.')."},
                    "include": {"type": "string", "description": "Optional filename glob, e.g. '*.py'."},
                },
                "required": ["pattern", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given full content (parent dirs are created).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write."},
                    "content": {"type": "string", "description": "Full file content."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "Call this exactly once when the task is fully done and verified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "One or two sentences on what was done."},
                },
                "required": ["summary"],
            },
        },
    },
]

TOOL_NAMES = tuple(t["function"]["name"] for t in TOOLS)


def canonical_call(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps({"name": name, "arguments": arguments}, sort_keys=True, separators=(",", ":"))


def call_sha256(name: str, arguments: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_call(name, arguments).encode()).hexdigest()


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def shell_command(name: str, arguments: dict[str, Any]) -> tuple[str, int]:
    """Map a tool call to (bash command string, timeout seconds)."""
    if name == "read_file":
        path = str(arguments.get("path", ""))
        start = max(1, _int(arguments.get("start_line"), 1))
        end = _int(arguments.get("end_line"), start + 199)
        if end < start:
            end = start
        if end - start > 999:
            end = start + 999
        cmd = f"nl -ba -- {shlex.quote(path)} | sed -n '{start},{end}p'"
        return cmd, READ_ONLY_TIMEOUT_SEC
    if name == "list_dir":
        path = str(arguments.get("path") or ".")
        return f"ls -la -- {shlex.quote(path)}", READ_ONLY_TIMEOUT_SEC
    if name == "search_files":
        pattern = str(arguments.get("pattern", ""))
        path = str(arguments.get("path") or ".")
        include = arguments.get("include")
        inc = f" --include={shlex.quote(str(include))}" if include else ""
        cmd = f"grep -rnE{inc} -e {shlex.quote(pattern)} -- {shlex.quote(path)} | head -n 200"
        return cmd, READ_ONLY_TIMEOUT_SEC
    if name == "bash":
        timeout = _int(arguments.get("timeout_sec"), DEFAULT_BASH_TIMEOUT_SEC)
        timeout = max(1, min(timeout, MAX_BASH_TIMEOUT_SEC))
        return str(arguments.get("command", "")), timeout
    if name == "write_file":
        import base64

        path = str(arguments.get("path", ""))
        content = str(arguments.get("content", ""))
        b64 = base64.b64encode(content.encode()).decode()
        q = shlex.quote(path)
        cmd = (
            f"mkdir -p -- \"$(dirname -- {q})\" && printf %s {shlex.quote(b64)} | base64 -d > {q} "
            f"&& printf 'wrote %s bytes to %s\\n' \"$(wc -c < {q})\" {q}"
        )
        return cmd, READ_ONLY_TIMEOUT_SEC
    raise ValueError(f"unknown tool {name!r}")


def docker_exec_argv(cid: str, workdir: str | None, user: str | None, command: str, timeout: int) -> list[str]:
    argv = [DOCKER, "exec"]
    if workdir:
        argv += ["-w", workdir]
    if user:
        argv += ["-u", str(user)]
    argv += [cid, "timeout", "-k", "5", str(timeout), "bash", "-lc", command]
    return argv


def cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def run_tool_sync(cid: str, workdir: str | None, user: str | None, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool call inside the container synchronously and return raw output."""
    command, timeout = shell_command(name, arguments)
    argv = docker_exec_argv(cid, workdir, user, command, timeout)
    started = time.perf_counter()
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=timeout + 15)
        rc = proc.returncode
        out = proc.stdout.decode("utf-8", "replace")
        err = proc.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as exc:
        rc = 124
        out = (exc.stdout or b"").decode("utf-8", "replace")
        err = (exc.stderr or b"").decode("utf-8", "replace") + "\n[docker exec timed out]"
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "tool_name": name,
        "call_sha256": call_sha256(name, arguments),
        "rc": rc,
        "stdout": cap(out, RAW_STDOUT_CAP),
        "stderr": cap(err, RAW_STDERR_CAP),
        "elapsed_ms": elapsed_ms,
        "side_effects": name not in READ_ONLY_TOOLS,
    }


def format_result(raw: dict[str, Any], visible_chars: int = 6000) -> str:
    """Render a raw tool result into the string the model sees (identical for both paths)."""
    out = raw.get("stdout") or ""
    err = raw.get("stderr") or ""
    rc = raw.get("rc", 0)
    if len(out) > visible_chars:
        head = out[: int(visible_chars * 0.7)]
        tail = out[-int(visible_chars * 0.3):]
        out = (head + f"\n...[{len(out) - len(head) - len(tail)} chars omitted; output too long. "
               "For files use read_file with start_line/end_line ranges; for commands narrow the output]...\n" + tail)
    parts = []
    if out.strip():
        parts.append(out.rstrip("\n"))
    if err.strip():
        parts.append("[stderr]\n" + cap(err, 1500).rstrip("\n"))
    if rc != 0:
        parts.append(f"[exit code {rc}]")
    if not parts:
        parts.append("[no output]")
    return "\n".join(parts)


def load_state() -> dict[str, Any]:
    with open(STATE_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)

