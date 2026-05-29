#!/usr/bin/env python3
"""Mine a Claude Code or Codex session JSONL into a luce-bench fixture case.

This is the v0 collector behind the ``agent_recorded`` evaluation area.
It walks one session file (Claude Code's
``~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`` or Codex's
``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``), extracts the first
user message + the sequence of tool_use calls the assistant made, and
emits ONE case dict on stdout.

Two output shapes:

* ``--out <path>``: write the case as a single JSON object.
* default: print to stdout (round-trippable JSON).

Cases use the schema described in
``luce-bench/src/lucebench/areas/agent_recorded.py`` — see that module's
docstring for the field-by-field contract and the rationale for each
``verifier.type``.

PII strip pass:

* Replace ``$HOME`` with ``<HOME>`` everywhere in strings.
* Mask values for any string that looks like a credential
  (regex over ``token``, ``key``, ``secret``, ``api_?key``, ``bearer``).
* Drop tool_result bodies — keep only ``length`` + sha256 of the first
  100 chars so a future grader can still join results to calls if it
  needs to, without shipping raw output.
* Drop assistant ``reasoning_content`` / inner-monologue blobs.

The "non-trivial" filter (used by ``--scan``) requires at least two
tool calls, a non-empty first-user message of length >= 80 chars, and at
least one Edit/Write/Bash call. This skips the queue-operation / trivial
"what's 2+2" style entries that don't represent real agent work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

HOME = os.path.expanduser("~")

# Regexes used during PII redaction. We strip everything that *looks*
# like a secret rather than try to detect specific credential formats —
# false positives here (a variable named "key" in code) are way cheaper
# than leaking a token.
_REDACT_KEY_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|bearer|authorization|password|"
    r"openai_api_key|anthropic_api_key|github_token|gh_token|ssh[_-]?key)\b"
)
# Substring-match against the *value* of a string when the surrounding
# key wasn't suspicious. Generic enough to catch ssh keys / JWTs /
# base64-looking blobs that happen to be embedded in tool inputs.
_REDACT_VALUE_RE = re.compile(
    r"(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|"
    r"AKIA[0-9A-Z]{16}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,})"
)


def _sha8(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:16]


def _scrub(value: Any) -> Any:
    """Recursively scrub strings of HOME paths and credential-looking values."""
    if isinstance(value, str):
        out = value.replace(HOME, "<HOME>")
        if _REDACT_VALUE_RE.search(out):
            out = _REDACT_VALUE_RE.sub("<REDACTED>", out)
        return out
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: ("<REDACTED>" if _REDACT_KEY_RE.search(k) else _scrub(v)) for k, v in value.items()}
    return value


def _short_hash(s: str | None) -> str | None:
    if not s:
        return None
    return _sha8(s)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ── Claude Code session shape ────────────────────────────────────────


def _is_claude_session(path: Path) -> bool:
    """Heuristic: does the first decodable line look like a Claude record?"""
    for rec in _read_jsonl(path):
        # Claude lines always carry top-level sessionId + type ∈
        # {user,assistant,attachment,…}. Codex lines carry payload+type=session_meta.
        if "sessionId" in rec and rec.get("type") in (
            "user",
            "assistant",
            "attachment",
            "ai-title",
            "queue-operation",
            "last-prompt",
        ):
            return True
        return False
    return False


def _extract_claude(path: Path) -> dict[str, Any] | None:
    """Mine a Claude Code session JSONL into a case dict (or None if it lacks
    a usable user message / tool trace)."""
    session_id: str | None = None
    cwd: str | None = None
    git_branch: str | None = None
    first_user_message: str | None = None
    first_user_timestamp: str | None = None
    tool_calls: list[dict[str, Any]] = []
    files_touched: set[str] = set()
    bash_count = 0

    for rec in _read_jsonl(path):
        if session_id is None and isinstance(rec.get("sessionId"), str):
            session_id = rec["sessionId"]
        if cwd is None and isinstance(rec.get("cwd"), str):
            cwd = rec["cwd"]
        if git_branch is None and isinstance(rec.get("gitBranch"), str):
            git_branch = rec["gitBranch"]

        t = rec.get("type")
        msg = rec.get("message")

        if t == "user" and isinstance(msg, dict) and first_user_message is None:
            content = msg.get("content")
            text: str | None = None
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in (None, "text"):
                        cand = block.get("text") or block.get("content")
                        if isinstance(cand, str):
                            text = cand
                            break
            if text and text.strip():
                # Strip tool_result echoes that Claude lifts into the user
                # role — those are hidden continuation tokens, not real
                # user prompts.
                if not text.lstrip().startswith("<tool_use_") and not text.lstrip().startswith("[Request"):
                    first_user_message = text
                    first_user_timestamp = rec.get("timestamp")

        if t == "assistant" and isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use":
                        name = block.get("name") or "?"
                        inp = block.get("input") or {}
                        if not isinstance(inp, dict):
                            inp = {}
                        call_args: dict[str, Any] = {}
                        # Per-tool argument summarization. The grader
                        # never needs the full args — only stable hashes
                        # of the key fields and the file path so future
                        # versions can do replay/equivalence checks.
                        fp = inp.get("file_path") or inp.get("path") or inp.get("notebook_path")
                        if isinstance(fp, str):
                            call_args["file_path"] = _scrub(fp)
                            files_touched.add(_scrub(fp))
                        if "old_string" in inp:
                            call_args["old_string_hash"] = _short_hash(str(inp.get("old_string")))
                        if "new_string" in inp:
                            call_args["new_string_hash"] = _short_hash(str(inp.get("new_string")))
                        if "content" in inp and name in ("Write", "NotebookEdit"):
                            call_args["content_hash"] = _short_hash(str(inp.get("content")))
                        if "command" in inp:
                            cmd = _scrub(str(inp.get("command")))
                            call_args["command_hash"] = _short_hash(cmd)
                            call_args["command_head"] = cmd[:80]
                            bash_count += 1
                        if "pattern" in inp:
                            call_args["pattern"] = _scrub(str(inp.get("pattern")))[:80]
                        if "url" in inp:
                            call_args["url_hash"] = _short_hash(str(inp.get("url")))
                        if "description" in inp:
                            call_args["description"] = _scrub(str(inp.get("description")))[:100]
                        tool_calls.append({"tool": name, "args": call_args})

    if not session_id or not first_user_message:
        return None
    if not tool_calls:
        return None

    return _build_case(
        source="claude-code",
        session_id=session_id,
        first_user_message=first_user_message,
        timestamp=first_user_timestamp,
        cwd=cwd,
        git_ref=None,  # Claude doesn't store the commit; only branch.
        git_branch=git_branch,
        tool_calls=tool_calls,
        files_touched=files_touched,
        bash_count=bash_count,
    )


# ── Codex session shape ──────────────────────────────────────────────


def _extract_codex(path: Path) -> dict[str, Any] | None:
    session_id: str | None = None
    cwd: str | None = None
    git_branch: str | None = None
    git_commit: str | None = None
    first_user_message: str | None = None
    first_user_timestamp: str | None = None
    tool_calls: list[dict[str, Any]] = []
    files_touched: set[str] = set()
    bash_count = 0

    for rec in _read_jsonl(path):
        t = rec.get("type")
        payload = rec.get("payload") or {}

        if t == "session_meta" and isinstance(payload, dict):
            session_id = payload.get("id") or session_id
            cwd = payload.get("cwd") or cwd
            git = payload.get("git")
            if isinstance(git, dict):
                git_branch = git.get("branch") or git_branch
                git_commit = git.get("commit_hash") or git_commit
            continue

        if t != "response_item" or not isinstance(payload, dict):
            continue
        inner_type = payload.get("type")
        role = payload.get("role")

        if inner_type == "message" and role == "user" and first_user_message is None:
            for block in payload.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "input_text":
                    text = block.get("text") or ""
                    # Skip AGENTS.md / sandbox boilerplate that Codex prepends
                    # before the real user prompt.
                    if (
                        text.strip()
                        and not text.lstrip().startswith("# AGENTS.md")
                        and not text.lstrip().startswith("<permissions instructions>")
                        and not text.lstrip().startswith("<environment_context>")
                    ):
                        first_user_message = text
                        first_user_timestamp = rec.get("timestamp")
                        break

        if inner_type == "function_call":
            name = payload.get("name") or "?"
            try:
                args = json.loads(payload.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            call_args: dict[str, Any] = {}
            cmd = args.get("cmd") or args.get("command")
            if isinstance(cmd, str):
                cmd_s = _scrub(cmd)
                call_args["command_hash"] = _short_hash(cmd_s)
                call_args["command_head"] = cmd_s[:80]
                bash_count += 1
            wd = args.get("workdir")
            if isinstance(wd, str):
                call_args["workdir"] = _scrub(wd)
            # Codex's apply_patch is `name == "apply_patch"` with a `patch`
            # argument holding the unified-diff envelope. We pull the file
            # paths out so files_touched stays comparable across sources.
            patch = args.get("patch") or args.get("diff")
            if isinstance(patch, str):
                call_args["patch_hash"] = _short_hash(patch)
                for m in re.finditer(r"^\*\*\* (?:Update|Add|Delete) File: (.+)$", patch, re.MULTILINE):
                    files_touched.add(_scrub(m.group(1).strip()))
                for m in re.finditer(r"^(?:---|\+\+\+) [ab]/(.+)$", patch, re.MULTILINE):
                    files_touched.add(_scrub(m.group(1).strip()))
            # Map codex tool names to a shape the grader can compare to
            # Claude's. exec_command / shell → Bash; apply_patch stays as-is.
            tool_norm = {
                "exec_command": "Bash",
                "shell": "Bash",
                "container_exec": "Bash",
            }.get(name, name)
            tool_calls.append({"tool": tool_norm, "args": call_args})

    if not session_id or not first_user_message or not tool_calls:
        return None

    return _build_case(
        source="codex",
        session_id=session_id,
        first_user_message=first_user_message,
        timestamp=first_user_timestamp,
        cwd=cwd,
        git_ref=git_commit,
        git_branch=git_branch,
        tool_calls=tool_calls,
        files_touched=files_touched,
        bash_count=bash_count,
    )


# ── Common case builder ──────────────────────────────────────────────


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str, n: int = 32) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return s[:n]


def _build_case(
    *,
    source: str,
    session_id: str,
    first_user_message: str,
    timestamp: str | None,
    cwd: str | None,
    git_ref: str | None,
    git_branch: str | None,
    tool_calls: list[dict[str, Any]],
    files_touched: set[str],
    bash_count: int,
) -> dict[str, Any]:
    scrubbed_prompt = _scrub(first_user_message)
    scrubbed_cwd = _scrub(cwd) if cwd else None

    # Stable case id: short hash of (session_id, first_user_message).
    # Doesn't change if the session is re-extracted; varies between
    # otherwise-similar sessions that asked different questions.
    id_seed = f"{session_id}|{first_user_message}"
    case_hash = hashlib.sha256(id_seed.encode("utf-8")).hexdigest()[:10]
    date_tag = (timestamp or "")[:10] if timestamp else "undated"
    slug = _slug(first_user_message[:64])
    case_id = f"{source.replace('-code','')}-{date_tag}-{slug}-{case_hash}"

    # Bag-of-tools (deduped, preserving order of first-appearance) is
    # what the grader checks against the candidate's response — the v0
    # verifier doesn't care about ordering or arg parity, only "did the
    # model name a tool we know was needed".
    expected_tools: list[str] = []
    for tc in tool_calls:
        if tc["tool"] not in expected_tools:
            expected_tools.append(tc["tool"])

    # `min_tool_calls` is a *lower* bound used by future replay graders;
    # the v0 text-only grader uses the bag-of-tools only.
    min_tool_calls = max(2, min(len(tool_calls), 4))

    return {
        "id": case_id,
        "source": source,
        "prompt": scrubbed_prompt,
        "initial_state": {
            "cwd": scrubbed_cwd,
            "git_ref": git_ref,
            "git_branch": git_branch,
            "files_referenced": sorted(files_touched)[:20],
        },
        "reference_trace": {
            "tool_calls": tool_calls[:50],  # cap so a 500-call session doesn't blow the fixture
            "outcome": {
                "files_modified": sorted(files_touched)[:20],
                "commands_run_count": bash_count,
                "total_tool_calls": len(tool_calls),
            },
        },
        "verifier": {
            "type": "tool-schema-coverage",
            "expected_tools": expected_tools,
            "min_tool_calls": min_tool_calls,
            "expected_files_touched": sorted(files_touched)[:8],
        },
    }


# ── Scanner + CLI ────────────────────────────────────────────────────


def _is_nontrivial(case: dict[str, Any]) -> bool:
    """Heuristic 'this looks like real agent work' filter.

    Requires:

    * Prompt at least 80 chars (skips one-word toy prompts).
    * At least 2 tool calls.
    * At least one Edit / Write / Bash / apply_patch / NotebookEdit call
      (skips pure-read sessions which are easy to grade but boring).
    """
    if len(case["prompt"]) < 80:
        return False
    if case["reference_trace"]["outcome"]["total_tool_calls"] < 2:
        return False
    write_like = {"Edit", "Write", "Bash", "apply_patch", "NotebookEdit", "MultiEdit"}
    tools = {tc["tool"] for tc in case["reference_trace"]["tool_calls"]}
    return bool(tools & write_like)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Path to a single session JSONL file. If omitted, --scan is required.",
    )
    ap.add_argument(
        "--scan",
        action="store_true",
        help="Walk ~/.claude/projects and ~/.codex/sessions, emit a JSON array "
        "of every non-trivial case (one per session). Writes to stdout or --out.",
    )
    ap.add_argument("--out", type=Path, default=None, help="Optional output path.")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="When --scan: stop after this many non-trivial cases.",
    )
    args = ap.parse_args(argv)

    if args.scan:
        cases = list(_scan_all(limit=args.limit))
        out = {"schema": "lucebox-bench-agent-recorded-v1", "cases": cases}
        text = json.dumps(out, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
            print(
                f"wrote {len(cases)} cases to {args.out} ({len(text)} bytes)",
                file=sys.stderr,
            )
        else:
            sys.stdout.write(text)
        return 0

    if not args.path:
        ap.error("provide a session path or use --scan")

    if not args.path.exists():
        print(f"no such file: {args.path}", file=sys.stderr)
        return 2

    if _is_claude_session(args.path):
        case = _extract_claude(args.path)
    else:
        case = _extract_codex(args.path)

    if case is None:
        print("no usable case extracted (no user message + tool calls)", file=sys.stderr)
        return 3

    text = json.dumps(case, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        sys.stdout.write(text + "\n")
    return 0


def _scan_all(*, limit: int | None = None) -> Iterable[dict[str, Any]]:
    """Walk both source dirs, yield non-trivial cases.

    Ordering: Claude top-level sessions first (one file per session id at
    the directory root — the ``subagents/`` subdir contains spawned
    sub-agents which inherit a queue-operation prompt that is rarely a
    real user task, so we skip those by depth=2). Then Codex sessions.
    """
    seen_ids: set[str] = set()
    yielded = 0

    claude_root = Path(HOME) / ".claude" / "projects"
    if claude_root.exists():
        for f in sorted(claude_root.glob("*/*.jsonl")):
            if limit is not None and yielded >= limit:
                return
            try:
                if not _is_claude_session(f):
                    continue
                case = _extract_claude(f)
            except Exception as e:  # noqa: BLE001 - one bad file shouldn't kill the scan
                print(f"skip {f}: {e}", file=sys.stderr)
                continue
            if case is None or case["id"] in seen_ids or not _is_nontrivial(case):
                continue
            seen_ids.add(case["id"])
            yielded += 1
            yield case

    codex_roots = [Path(HOME) / ".codex" / "sessions", Path(HOME) / ".config" / "codex" / "sessions"]
    for root in codex_roots:
        if not root.exists():
            continue
        for f in sorted(root.rglob("*.jsonl")):
            if limit is not None and yielded >= limit:
                return
            try:
                case = _extract_codex(f)
            except Exception as e:  # noqa: BLE001
                print(f"skip {f}: {e}", file=sys.stderr)
                continue
            if case is None or case["id"] in seen_ids or not _is_nontrivial(case):
                continue
            seen_ids.add(case["id"])
            yielded += 1
            yield case


if __name__ == "__main__":
    raise SystemExit(main())
