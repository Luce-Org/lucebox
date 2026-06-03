#!/usr/bin/env python3
"""Mine a Claude Code or Codex session JSONL into a luce-bench fixture case.

This is the collector behind the ``agent_recorded`` evaluation area.
It walks one session file (Claude Code's
``~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`` or Codex's
``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``).

Two output modes:

* **v1 single-prompt** (default) — extracts the first user message +
  the sequence of tool_use calls the assistant made, emits ONE case
  dict for the ``tool-schema-coverage`` verifier. Schema:
  ``lucebox-bench-agent-recorded-v1``.

* **multi-turn replay** (``--multi-turn``) — walks the whole session
  in record order, building up an OpenAI-shaped ``messages`` list, and
  emits a case snapshot every time cumulative approx token count
  crosses a bucket threshold (default ``8K,16K,32K,64K,100K,128K``;
  override with ``--buckets``). Each case is consumable as a chat
  completions request against the running lucebox server, intended for
  the ``prefill-and-decode`` verifier the autotune sweep uses to score
  "this max_ctx setting actually serves this trace." Schema:
  ``lucebox-bench-agent-recorded-multi-turn-v0``.

* ``--scan`` (single-prompt mode only) — walks both source dirs and
  emits a JSON array of every non-trivial case.

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
    """Heuristic: do the early records look like Claude session entries?

    Walks up to the first 32 decodable records — any one of them with a
    top-level ``sessionId`` AND a Claude-shaped ``type`` is enough to
    confirm. Claude sessions can lead with ``permission-mode``,
    ``queue-operation``, or other meta records before the first user
    turn; a previous version returned False on any single non-matching
    record at the head, misrouting every Claude session that didn't
    happen to start with a user line. Codex sessions never carry
    ``sessionId`` at the top level (the session id lives under
    ``payload.id`` of the ``session_meta`` record), so the presence of
    a top-level ``sessionId`` is the cleanest discriminator.
    """
    for i, rec in enumerate(_read_jsonl(path)):
        if i >= 32:
            break
        if "sessionId" in rec and rec.get("type") in (
            "user",
            "assistant",
            "attachment",
            "ai-title",
            "queue-operation",
            "last-prompt",
            "permission-mode",
            "file-history-snapshot",
            "system",
        ):
            return True
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


# ── Multi-turn replay slicing ────────────────────────────────────────
#
# A separate output mode from the v1 single-turn case. The motivation is
# the autotune sweep's "given a context of length n, the trace of n -
# reply_budget must prefill and decode" success criterion: we need cases
# whose total prompt-token count lands in target buckets {8K, 16K, 32K,
# 64K, 100K, 128K}. The v1 extractor's first-user-message-only output
# tops out near 3K tokens — too small to exercise the long-context path.
#
# This walks one session in record order, building up an OpenAI-shaped
# ``messages`` list, and emits a case snapshot every time the cumulative
# approximate token count crosses a bucket threshold. PII scrub +
# tool-result-body hashing inherited from v1; thinking blocks are
# dropped (they're not what the assistant turn sends to the next
# request and they bloat the prompt with content the model didn't
# actually emit to the chat history).
#
# Output schema: ``lucebox-bench-agent-recorded-multi-turn-v0``. A
# fixture file contains one case per bucket that the source session
# reached; buckets the session never reached are simply absent. The
# ``agent_recorded`` area is the consumer (see its module docstring for
# the case shape the area expects).


DEFAULT_BUCKETS: tuple[int, ...] = (8_192, 16_384, 32_768, 65_536, 100_000, 131_072)

# Chars-per-token heuristic for bucket-crossing detection. Real tokenizer
# would be more accurate but introduces a heavyweight dependency; the
# bucket boundaries are coarse (factor-of-2 spacing) so a 4x rough scale
# is fine for bucket assignment. The case records the actual char count
# so consumers can re-derive a tighter token estimate.
_CHARS_PER_TOKEN = 4


def _claude_record_to_message(rec: dict[str, Any]) -> tuple[str, str] | None:
    """Map one Claude session record to (role, text) for replay, or None.

    Thinking blocks are dropped — they aren't part of the chat history
    a subsequent request would send. Tool-use blocks are serialized as
    bracketed text so the model sees them as part of the assistant's
    text stream (it would otherwise see nothing for an all-tool-use
    turn). Tool-result blocks in user records likewise become bracketed
    text so the replay flattens into pure user/assistant text — the
    fixture is for prefill-and-decode probing, not faithful semantic
    replay.
    """
    t = rec.get("type")
    msg = rec.get("message")
    if not isinstance(msg, dict):
        return None

    if t == "user":
        c = msg.get("content")
        if isinstance(c, str):
            stripped = c.lstrip()
            # Skip Claude's hidden continuation prompts that lift tool
            # results back into the user role — these aren't real turns.
            if stripped.startswith("<tool_use_") or stripped.startswith("[Request"):
                return None
            return ("user", c) if c.strip() else None
        if isinstance(c, list):
            parts: list[str] = []
            for b in c:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    txt = b.get("text") or b.get("content")
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt)
                elif bt == "tool_result":
                    body = b.get("content")
                    body_text: str = ""
                    if isinstance(body, str):
                        body_text = body
                    elif isinstance(body, list):
                        for sub in body:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                t_text = sub.get("text")
                                if isinstance(t_text, str):
                                    body_text += t_text
                    if body_text.strip():
                        parts.append(f"[tool result] {body_text}")
            return ("user", "\n\n".join(parts)) if parts else None
        return None

    if t == "assistant":
        c = msg.get("content")
        if not isinstance(c, list):
            return None
        parts = []
        for b in c:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                txt = b.get("text", "")
                if isinstance(txt, str) and txt:
                    parts.append(txt)
            elif bt == "tool_use":
                name = b.get("name", "?")
                inp = b.get("input", {}) or {}
                try:
                    args_str = json.dumps(inp, ensure_ascii=False)[:400]
                except (TypeError, ValueError):
                    args_str = "{}"
                parts.append(f"[Tool: {name}({args_str})]")
            # thinking blocks: dropped on purpose
        return ("assistant", "\n\n".join(parts)) if parts else None

    return None


def _codex_record_to_message(rec: dict[str, Any]) -> tuple[str, str] | None:
    """Map one Codex response_item record to (role, text) for replay."""
    if rec.get("type") != "response_item":
        return None
    payload = rec.get("payload") or {}
    if not isinstance(payload, dict):
        return None
    inner_type = payload.get("type")
    role = payload.get("role")

    if inner_type == "message":
        content = payload.get("content") or []
        parts: list[str] = []
        for b in content:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt in ("input_text", "output_text", "text"):
                txt = b.get("text") or ""
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt)
        if not parts:
            return None
        if role == "user":
            return ("user", "\n\n".join(parts))
        if role in ("assistant", "model"):
            return ("assistant", "\n\n".join(parts))
        return None

    if inner_type == "function_call":
        name = payload.get("name") or "?"
        args_raw = payload.get("arguments") or "{}"
        if isinstance(args_raw, str):
            args_str = args_raw[:400]
        else:
            try:
                args_str = json.dumps(args_raw, ensure_ascii=False)[:400]
            except (TypeError, ValueError):
                args_str = "{}"
        return ("assistant", f"[Tool: {name}({args_str})]")

    if inner_type == "function_call_output":
        out = payload.get("output") or ""
        if isinstance(out, str) and out.strip():
            return ("user", f"[tool result] {out}")
        return None

    return None


def _append_message(messages: list[dict[str, str]], role: str, text: str) -> int:
    """Append text to messages list, collapsing consecutive same-role turns.

    Returns the number of chars added (used for cumulative-length tracking
    in the bucket walker).
    """
    if messages and messages[-1]["role"] == role:
        sep = "\n\n"
        added = len(sep) + len(text)
        messages[-1]["content"] += sep + text
    else:
        messages.append({"role": role, "content": text})
        added = len(text)
    return added


def _build_multi_turn_case(
    *,
    source: str,
    session_id: str,
    bucket_tokens: int,
    actual_chars: int,
    messages: list[dict[str, str]],
    source_path: Path,
    cwd: str | None,
    git_branch: str | None,
    timestamp: str | None,
) -> dict[str, Any]:
    """Assemble one multi-turn case snapshot at a bucket crossing."""
    scrubbed_messages = [
        {"role": m["role"], "content": _scrub(m["content"])} for m in messages
    ]
    scrubbed_cwd = _scrub(cwd) if cwd else None
    actual_tokens = actual_chars // _CHARS_PER_TOKEN

    # Stable id: source session + target bucket. Same session re-extracted
    # produces the same id per bucket; different buckets from the same
    # session are distinct cases.
    id_seed = f"{session_id}|bucket-{bucket_tokens}"
    case_hash = hashlib.sha256(id_seed.encode("utf-8")).hexdigest()[:10]
    date_tag = (timestamp or "")[:10] if timestamp else "undated"
    case_id = f"{source.replace('-code', '')}-{date_tag}-multiturn-{bucket_tokens}-{case_hash}"

    return {
        "id": case_id,
        "source": source,
        "kind": "multi-turn-replay",
        "messages": scrubbed_messages,
        "context_tokens_approx": actual_tokens,
        "context_chars": actual_chars,
        "target_bucket_tokens": bucket_tokens,
        "n_messages": len(scrubbed_messages),
        "source_session_id": session_id,
        "source_session_path": _scrub(str(source_path)),
        "initial_state": {
            "cwd": scrubbed_cwd,
            "git_branch": git_branch,
        },
        "verifier": {
            "type": "prefill-and-decode",
            "min_response_chars": 1,
            "max_wall_seconds": 300,
        },
    }


def _append_cost(messages: list[dict[str, str]], role: str, text: str) -> int:
    """Predict how many chars ``_append_message`` would add — used to
    detect bucket crossings *before* committing the append."""
    if messages and messages[-1]["role"] == role:
        return len("\n\n") + len(text)
    return len(text)


def _slice_session_multi_turn(
    *,
    source: str,
    path: Path,
    buckets: list[int],
    record_iter: Iterable[dict[str, Any]],
    decode: Any,
    meta_pick: Any,
) -> list[dict[str, Any]]:
    """Shared multi-turn slicing loop for Claude and Codex sessions.

    The contract guaranteed by this loop: every emitted case satisfies
    ``context_tokens_approx <= target_bucket_tokens``. We achieve this by
    snapshotting the messages list *before* appending the record that
    would cross a bucket — the case becomes "the longest prefix that
    fits under this bucket." Cases where the very first record would
    overshoot the smallest bucket (yielding an empty snapshot) are
    skipped.

    ``record_iter`` yields one record dict per session line. ``decode``
    takes a record and returns ``(role, text)`` or ``None``.
    ``meta_pick`` takes a record and a current metadata dict and
    updates the dict in place (session_id / cwd / git_branch /
    first_timestamp).
    """
    sorted_buckets = sorted(buckets)
    messages: list[dict[str, str]] = []
    cum_chars = 0
    next_idx = 0
    cases: list[dict[str, Any]] = []
    meta: dict[str, str | None] = {
        "session_id": None,
        "cwd": None,
        "git_branch": None,
        "first_timestamp": None,
    }

    for rec in record_iter:
        meta_pick(rec, meta)
        rt = decode(rec)
        if rt is None:
            continue
        role, text = rt
        cost = _append_cost(messages, role, text)
        projected_tokens = (cum_chars + cost) // _CHARS_PER_TOKEN

        # Emit cases for every bucket this record would push us past.
        # Snapshot is the *current* (pre-append) state, which by
        # induction satisfies cum_chars/4 <= sorted_buckets[next_idx-1]
        # < sorted_buckets[next_idx].
        while next_idx < len(sorted_buckets) and projected_tokens > sorted_buckets[next_idx]:
            if meta["session_id"] is not None and messages:
                cases.append(
                    _build_multi_turn_case(
                        source=source,
                        session_id=meta["session_id"],
                        bucket_tokens=sorted_buckets[next_idx],
                        actual_chars=cum_chars,
                        messages=messages,
                        source_path=path,
                        cwd=meta["cwd"],
                        git_branch=meta["git_branch"],
                        timestamp=meta["first_timestamp"],
                    )
                )
            next_idx += 1

        if next_idx >= len(sorted_buckets):
            break

        cum_chars += _append_message(messages, role, text)

    # Tail flush: if the session ends without crossing some buckets but
    # the final state fits under one or more remaining buckets, emit a
    # single case for the largest such bucket — that case is the
    # session's "max length" snapshot, useful for the largest-fitting
    # sweep cell. Skip if next_idx is already past every bucket.
    if next_idx < len(sorted_buckets) and messages and meta["session_id"] is not None:
        final_tokens = cum_chars // _CHARS_PER_TOKEN
        # Largest bucket the final state fits under.
        fit_idx = next_idx
        while fit_idx < len(sorted_buckets) and sorted_buckets[fit_idx] >= final_tokens:
            fit_idx += 1
        fit_idx -= 1  # last bucket where final_tokens <= bucket
        if fit_idx >= next_idx:
            cases.append(
                _build_multi_turn_case(
                    source=source,
                    session_id=meta["session_id"],
                    bucket_tokens=sorted_buckets[fit_idx],
                    actual_chars=cum_chars,
                    messages=messages,
                    source_path=path,
                    cwd=meta["cwd"],
                    git_branch=meta["git_branch"],
                    timestamp=meta["first_timestamp"],
                )
            )

    return cases


def _claude_meta_pick(rec: dict[str, Any], meta: dict[str, str | None]) -> None:
    if meta["session_id"] is None and isinstance(rec.get("sessionId"), str):
        meta["session_id"] = rec["sessionId"]
    if meta["cwd"] is None and isinstance(rec.get("cwd"), str):
        meta["cwd"] = rec["cwd"]
    if meta["git_branch"] is None and isinstance(rec.get("gitBranch"), str):
        meta["git_branch"] = rec["gitBranch"]
    if meta["first_timestamp"] is None and isinstance(rec.get("timestamp"), str):
        meta["first_timestamp"] = rec["timestamp"]


def _codex_meta_pick(rec: dict[str, Any], meta: dict[str, str | None]) -> None:
    if meta["first_timestamp"] is None and isinstance(rec.get("timestamp"), str):
        meta["first_timestamp"] = rec["timestamp"]
    if rec.get("type") == "session_meta":
        payload = rec.get("payload") or {}
        if isinstance(payload, dict):
            if meta["session_id"] is None:
                meta["session_id"] = payload.get("id") or meta["session_id"]
            if meta["cwd"] is None:
                meta["cwd"] = payload.get("cwd") or meta["cwd"]
            git = payload.get("git")
            if isinstance(git, dict) and meta["git_branch"] is None:
                meta["git_branch"] = git.get("branch") or meta["git_branch"]


def _slice_claude_multi_turn(path: Path, buckets: list[int]) -> list[dict[str, Any]]:
    """Walk a Claude session, emit one replay case per token-count bucket."""
    return _slice_session_multi_turn(
        source="claude-code",
        path=path,
        buckets=buckets,
        record_iter=_read_jsonl(path),
        decode=_claude_record_to_message,
        meta_pick=_claude_meta_pick,
    )


def _slice_codex_multi_turn(path: Path, buckets: list[int]) -> list[dict[str, Any]]:
    """Walk a Codex rollout, emit one replay case per token-count bucket."""
    return _slice_session_multi_turn(
        source="codex",
        path=path,
        buckets=buckets,
        record_iter=_read_jsonl(path),
        decode=_codex_record_to_message,
        meta_pick=_codex_meta_pick,
    )


def _parse_buckets(spec: str) -> list[int]:
    """Parse a comma-separated bucket spec like ``8K,16K,32K,64K,100K,128K``.

    Accepts plain integers, ``K`` (×1024), and ``M`` (×1024²) suffixes.
    """
    out: list[int] = []
    for raw in spec.split(","):
        s = raw.strip()
        if not s:
            continue
        mult = 1
        if s[-1].lower() == "k":
            mult = 1024
            s = s[:-1]
        elif s[-1].lower() == "m":
            mult = 1024 * 1024
            s = s[:-1]
        try:
            out.append(int(float(s) * mult))
        except ValueError as e:
            raise ValueError(f"bad bucket spec: {raw!r}") from e
    if not out:
        raise ValueError("empty bucket spec")
    return sorted(out)


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
    ap.add_argument(
        "--multi-turn",
        action="store_true",
        help="Emit multi-turn replay cases (one per token-count bucket) "
        "from a single session, instead of the v1 first-prompt-only case. "
        "Requires a session path; not compatible with --scan.",
    )
    ap.add_argument(
        "--buckets",
        type=str,
        default="8K,16K,32K,64K,100K,128K",
        help="Comma-separated target token counts for --multi-turn slicing "
        "(accepts K/M suffixes; default: 8K,16K,32K,64K,100K,128K).",
    )
    args = ap.parse_args(argv)

    if args.scan and args.multi_turn:
        ap.error("--multi-turn is not compatible with --scan")

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

    if args.multi_turn:
        try:
            buckets = _parse_buckets(args.buckets)
        except ValueError as e:
            print(f"--buckets: {e}", file=sys.stderr)
            return 2
        if _is_claude_session(args.path):
            cases = _slice_claude_multi_turn(args.path, buckets)
        else:
            cases = _slice_codex_multi_turn(args.path, buckets)
        if not cases:
            print(
                "no buckets reached (session too short, or no usable records)",
                file=sys.stderr,
            )
            return 3
        out = {
            "schema": "lucebox-bench-agent-recorded-multi-turn-v0",
            "buckets": buckets,
            "source_session_path": _scrub(str(args.path)),
            "cases": cases,
        }
        text = json.dumps(out, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
            print(
                f"wrote {len(cases)} bucketed cases to {args.out} "
                f"({len(text)} bytes); buckets reached: "
                f"{[c['target_bucket_tokens'] for c in cases]}",
                file=sys.stderr,
            )
        else:
            sys.stdout.write(text + "\n")
        return 0

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
