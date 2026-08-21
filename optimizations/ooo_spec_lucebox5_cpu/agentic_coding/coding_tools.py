#!/usr/bin/env python3
"""Read-only coding tools shared by the benchmark client and server executor."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

PROTOCOL = "dflash.tool-speculation.v1"
MAX_FILE_BYTES = 1_048_576
MAX_READ_LINES = 240
MAX_MATCHES = 100
MAX_LISTED_FILES = 200
MAX_SCANNED_FILES = 20_000
MAX_SCANNED_BYTES = 128 * 1024 * 1024
MAX_RETURNED_TEXT_BYTES = 128 * 1024
MAX_PATH_CHARS = 4096
MAX_RETURNED_PATH_CHARS = 512
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "deps",
    "dist",
    "node_modules",
    "third_party",
    "vendor",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a bounded line range from a UTF-8 text file in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Find a literal string in repository text files and return matching lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "path": {"type": "string", "minLength": 1},
                    "case_sensitive": {"type": "boolean"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_MATCHES},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List repository files beneath a path using a bounded glob.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "glob": {"type": "string", "minLength": 1},
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_LISTED_FILES,
                    },
                },
                "additionalProperties": False,
            },
        },
    },
]

TOOL_NAMES = frozenset(tool["function"]["name"] for tool in TOOLS)
TOOL_ARGUMENTS = {
    "read_file": frozenset({"path", "start_line", "end_line"}),
    "search_code": frozenset({"query", "path", "case_sensitive", "max_results"}),
    "list_files": frozenset({"path", "glob", "max_results"}),
}


class ToolInputError(ValueError):
    """Raised when a model-generated tool call violates the read-only contract."""


def canonical_call_sha256(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(
        {"name": name, "arguments": arguments},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _workspace_root(root: str | os.PathLike[str] | None) -> Path:
    raw = root if root is not None else os.environ.get("DFLASH_TOOL_WORKSPACE", os.getcwd())
    resolved = Path(raw).resolve(strict=True)
    if not resolved.is_dir():
        raise ToolInputError("workspace root is not a directory")
    return resolved


def _confined_path(root: Path, raw_path: Any, *, expect: str) -> Path:
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or len(raw_path) > MAX_PATH_CHARS
        or "\x00" in raw_path
    ):
        raise ToolInputError("path must be a non-empty string")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise ToolInputError("path must be relative to the workspace")
    try:
        resolved = (root / candidate).resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise ToolInputError("path is missing or escapes the workspace") from exc
    if expect == "file" and not resolved.is_file():
        raise ToolInputError("path is not a file")
    if expect == "directory" and not resolved.is_dir():
        raise ToolInputError("path is not a directory")
    if expect == "file_or_directory" and not (resolved.is_file() or resolved.is_dir()):
        raise ToolInputError("path is not a file or directory")
    return resolved


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _bounded_int(value: Any, default: int, low: int, high: int, name: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ToolInputError(f"{name} must be an integer from {low} to {high}")
    return value


def _text_file(path: Path) -> str | None:
    with path.open("rb") as source:
        data = source.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        return None
    if b"\x00" in data:
        return None
    return data.decode("utf-8", errors="replace")


def read_file(arguments: dict[str, Any], root: Path) -> dict[str, Any]:
    path = _confined_path(root, arguments.get("path"), expect="file")
    start = _bounded_int(arguments.get("start_line"), 1, 1, 10_000_000, "start_line")
    end = _bounded_int(
        arguments.get("end_line"), start + MAX_READ_LINES - 1, start, 10_000_000, "end_line"
    )
    if end - start + 1 > MAX_READ_LINES:
        raise ToolInputError(f"read_file is limited to {MAX_READ_LINES} lines")
    text = _text_file(path)
    if text is None:
        raise ToolInputError("file is binary or exceeds the size limit")
    lines = text.splitlines()
    selected = lines[start - 1 : end]
    actual_end = start + len(selected) - 1 if selected else start - 1
    numbered = "\n".join(
        f"{line_number}: {line}"
        for line_number, line in enumerate(selected, start=start)
    )
    encoded = numbered.encode("utf-8")
    content_truncated = len(encoded) > MAX_RETURNED_TEXT_BYTES
    if content_truncated:
        numbered = encoded[:MAX_RETURNED_TEXT_BYTES].decode("utf-8", errors="ignore")
    return {
        "path": _relative(root, path),
        "start_line": start,
        "end_line": actual_end,
        "total_lines": len(lines),
        "content": numbered,
        "content_truncated": content_truncated,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _candidate_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for current_root, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = sorted(
            directory
            for directory in dirs
            if directory not in SKIP_DIRS and not directory.startswith(".")
        )
        for filename in sorted(files):
            if filename.startswith("."):
                continue
            yield Path(current_root) / filename


def search_code(arguments: dict[str, Any], root: Path) -> dict[str, Any]:
    query = arguments.get("query")
    if not isinstance(query, str) or not query or len(query) > 256 or "\n" in query:
        raise ToolInputError("query must be one non-empty line of at most 256 characters")
    raw_path = arguments.get("path", ".")
    search_root = _confined_path(root, raw_path, expect="file_or_directory")
    case_sensitive = arguments.get("case_sensitive", True)
    if not isinstance(case_sensitive, bool):
        raise ToolInputError("case_sensitive must be a boolean")
    limit = _bounded_int(arguments.get("max_results"), 40, 1, MAX_MATCHES, "max_results")
    needle = query if case_sensitive else query.casefold()
    matches: list[dict[str, Any]] = []
    truncated = False
    scanned_files = 0
    scanned_bytes = 0
    for path in _candidate_files(search_root):
        if scanned_files >= MAX_SCANNED_FILES:
            truncated = True
            break
        scanned_files += 1
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
            file_bytes = resolved.stat().st_size
            if file_bytes > MAX_FILE_BYTES:
                continue
            if scanned_bytes + file_bytes > MAX_SCANNED_BYTES:
                truncated = True
                break
            text = _text_file(resolved)
        except (OSError, RuntimeError, ValueError):
            continue
        if text is None:
            continue
        scanned_bytes += file_bytes
        for line_number, line in enumerate(text.splitlines(), start=1):
            haystack = line if case_sensitive else line.casefold()
            if needle not in haystack:
                continue
            if len(matches) >= limit:
                truncated = True
                break
            matches.append(
                {
                    "path": _relative(root, resolved),
                    "line": line_number,
                    "text": line[:500],
                }
            )
        if truncated:
            break
    return {
        "query": query,
        "matches": matches,
        "truncated": truncated,
        "scanned_files": scanned_files,
        "scanned_bytes": scanned_bytes,
    }


def list_files(arguments: dict[str, Any], root: Path) -> dict[str, Any]:
    directory = _confined_path(root, arguments.get("path", "."), expect="directory")
    # Models sometimes serialize an omitted optional glob as an empty string.
    # It has the same unambiguous meaning as the documented default.
    raw_pattern = arguments.get("glob")
    pattern = "*" if raw_pattern is None or raw_pattern == "" else raw_pattern
    if not isinstance(pattern, str) or not pattern or len(pattern) > 256:
        raise ToolInputError("glob must be a non-empty string of at most 256 characters")
    limit = _bounded_int(
        arguments.get("max_results"), 200, 1, MAX_LISTED_FILES, "max_results"
    )
    files: list[str] = []
    truncated = False
    scanned_files = 0
    for path in _candidate_files(directory):
        if scanned_files >= MAX_SCANNED_FILES:
            truncated = True
            break
        scanned_files += 1
        try:
            resolved = path.resolve(strict=True)
            relative_to_directory = resolved.relative_to(directory).as_posix()
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            continue
        if not fnmatch.fnmatch(relative_to_directory, pattern) and not fnmatch.fnmatch(
            resolved.name, pattern
        ):
            continue
        if len(files) >= limit:
            truncated = True
            break
        relative = _relative(root, resolved)
        if len(relative) <= MAX_RETURNED_PATH_CHARS:
            files.append(relative)
    return {
        "path": _relative(root, directory) or ".",
        "glob": pattern,
        "files": files,
        "truncated": truncated,
        "scanned_files": scanned_files,
    }


def execute_tool(
    name: Any,
    arguments: dict[str, Any],
    workspace: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(name, str) or name not in TOOL_NAMES or not isinstance(arguments, dict):
        return {"tool_name": name, "ok": False, "error": "unsupported or invalid tool call"}
    try:
        unknown = set(arguments) - TOOL_ARGUMENTS[name]
        if unknown:
            names = ", ".join(sorted(str(item) for item in unknown))
            raise ToolInputError(f"unknown arguments for {name}: {names}")
        root = _workspace_root(workspace)
        handlers = {
            "read_file": read_file,
            "search_code": search_code,
            "list_files": list_files,
        }
        value = handlers[name](arguments, root)
        return {
            "tool_name": name,
            "ok": True,
            "value": value,
            "call_sha256": canonical_call_sha256(name, arguments),
        }
    except (OSError, ToolInputError) as exc:
        return {"tool_name": name, "ok": False, "error": str(exc)}


def format_tool_result(result: dict[str, Any]) -> str:
    content = result.get("value") if result.get("ok") is True and "value" in result else result
    if isinstance(content, str):
        return content
    return json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
