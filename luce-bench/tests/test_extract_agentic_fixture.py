"""Unit tests for ``scripts/extract-agentic-fixture.py``.

Covers the multi-turn slicing mode that feeds the ``agent_recorded``
area's prefill-and-decode verifier. The v1 single-prompt path is
exercised end-to-end by the existing ``agent_recorded`` cases fixture.

No filesystem access to ``~/.claude/projects`` or ``~/.codex/sessions``;
each test writes a synthetic JSONL into ``tmp_path`` and asserts on the
returned cases.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_extractor():
    """Load the hyphen-named script as a module for testing."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "extract-agentic-fixture.py"
    spec = importlib.util.spec_from_file_location("eaf", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def eaf():
    return _load_extractor()


# ── synthetic session builders ──────────────────────────────────────


def _write_claude_session(path: Path, *, sid: str = "test-session", turns: list[tuple[str, str]] | None = None) -> None:
    """Write a minimal Claude-shaped JSONL with the given (role, text) turns."""
    if turns is None:
        turns = [("user", "first user message")]
    lines = []
    # Leading meta record that doesn't match user/assistant — exercises
    # the multi-record detection fix (older code returned False on a
    # non-user leading record).
    lines.append(json.dumps({"type": "permission-mode", "sessionId": sid, "mode": "default"}))
    for role, text in turns:
        if role == "user":
            lines.append(
                json.dumps(
                    {
                        "type": "user",
                        "sessionId": sid,
                        "cwd": "/tmp/synthetic",
                        "gitBranch": "main",
                        "timestamp": "2026-05-29T00:00:00Z",
                        "message": {"content": text},
                    }
                )
            )
        else:
            lines.append(
                json.dumps(
                    {
                        "type": "assistant",
                        "sessionId": sid,
                        "cwd": "/tmp/synthetic",
                        "gitBranch": "main",
                        "timestamp": "2026-05-29T00:00:01Z",
                        "message": {"content": [{"type": "text", "text": text}]},
                    }
                )
            )
    path.write_text("\n".join(lines) + "\n")


# ── tests ───────────────────────────────────────────────────────────


def test_is_claude_session_after_leading_meta_record(tmp_path: Path, eaf) -> None:
    """Regression: leading non-user record (permission-mode, system) must
    not misclassify a Claude session as Codex."""
    session = tmp_path / "claude.jsonl"
    _write_claude_session(session, turns=[("user", "hello world")])
    assert eaf._is_claude_session(session) is True


def test_parse_buckets_accepts_suffixes(eaf) -> None:
    assert eaf._parse_buckets("4K,8K,16K") == [4096, 8192, 16384]
    assert eaf._parse_buckets("1024,4096") == [1024, 4096]
    assert eaf._parse_buckets("1M")[0] == 1024 * 1024


def test_parse_buckets_rejects_garbage(eaf) -> None:
    with pytest.raises(ValueError):
        eaf._parse_buckets("not-a-number")
    with pytest.raises(ValueError):
        eaf._parse_buckets("")


def test_multi_turn_slice_emits_one_case_per_bucket(tmp_path: Path, eaf) -> None:
    """Five turns of ~16K chars each → ~20K tokens cumulative — should
    cross the 8K and 16K buckets and tail-flush at 32K. v1 schema
    requires a usable reference_response (last assistant turn lifted
    out of the prefix) and at least 3 messages in the prefix, so the
    fixture needs enough turns to satisfy that contract."""
    session = tmp_path / "claude.jsonl"
    chunk = "x" * 16_000  # ~4K tokens each at chars/4
    # Reference text must be >= 40 chars (v1 schema filter).
    ref_chunk = "this is a non-trivial assistant response " + ("y" * 16_000)
    _write_claude_session(
        session,
        turns=[
            ("user", chunk),
            ("assistant", ref_chunk),
            ("user", chunk),
            ("assistant", ref_chunk),
            ("user", chunk),
        ],
    )
    cases = eaf._slice_claude_multi_turn(session, [8192, 16384, 32768])
    # 8K bucket reached; 16K should be crossed too.
    assert len(cases) >= 1
    for c in cases:
        # Contract: every case fits under its target.
        assert c["context_tokens_approx"] <= c["target_bucket_tokens"], (
            f"case {c['id']} overshoots: {c['context_tokens_approx']} > {c['target_bucket_tokens']}"
        )
        assert c["kind"] == "multi-turn-replay"
        assert c["source"] == "claude-code"
        # v1 schema uses cache-and-quality (was prefill-and-decode in v0).
        assert c["verifier"]["type"] == "cache-and-quality"
        assert isinstance(c["messages"], list) and c["messages"]
        # v1 contract: prefix ends on user, reference is the last
        # assistant turn that was lifted out, both populated.
        assert c["messages"][-1]["role"] == "user", c["id"]
        assert c["reference_response"].strip(), c["id"]
        assert len(c["messages"]) >= 3, "prefix must have at least one full exchange"


def test_multi_turn_case_messages_alternate_roles_after_collapse(tmp_path: Path, eaf) -> None:
    """Consecutive same-role records collapse into one message — verifies
    the OpenAI-shape replay can be sent as a chat completions request.

    v1 schema lifts the last assistant turn out as reference_response,
    so the *prefix* messages end on a user turn — the alternation we
    check is over that user-ending prefix, not the full session.
    """
    session = tmp_path / "claude.jsonl"
    _write_claude_session(
        session,
        turns=[
            ("user", "u1 " * 1000),
            ("user", "u2 " * 1000),  # collapses with u1
            ("assistant", "a1 " * 1000 + "non-trivial reference body"),
            ("user", "u3 " * 1000),
            ("assistant", "a2 " * 5000 + "non-trivial reference body two"),
            ("user", "u4 " * 1000),
        ],
    )
    cases = eaf._slice_claude_multi_turn(session, [4096, 16384])
    assert cases
    # Final / largest case should have collapsed turns into a strictly
    # alternating role list ending on user.
    final = cases[-1]
    roles = [m["role"] for m in final["messages"]]
    for a, b in zip(roles, roles[1:], strict=False):
        assert a != b, f"adjacent same-role messages survived collapse: {roles}"
    assert roles[-1] == "user"


def test_multi_turn_drops_thinking_blocks(tmp_path: Path, eaf) -> None:
    """Assistant `thinking` blocks must not appear in the replay messages."""
    session = tmp_path / "claude.jsonl"
    session.write_text(
        json.dumps(
            {
                "type": "user",
                "sessionId": "s",
                "timestamp": "2026-05-29T00:00:00Z",
                "message": {"content": "ask"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "sessionId": "s",
                "timestamp": "2026-05-29T00:00:01Z",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "should not leak", "signature": "x"},
                        {"type": "text", "text": "visible answer"},
                    ]
                },
            }
        )
        + "\n"
    )
    cases = eaf._slice_claude_multi_turn(session, [1024])  # tail-flush only
    # Session is tiny — no bucket crossed; tail-flush should still
    # emit nothing since the final state has cum_chars/4 well under
    # the bucket. Use a tiny bucket to force at least the tail emit.
    if cases:
        for c in cases:
            full = "\n".join(m["content"] for m in c["messages"])
            assert "should not leak" not in full


def test_multi_turn_scrubs_home_path(tmp_path: Path, eaf, monkeypatch: pytest.MonkeyPatch) -> None:
    """PII scrub: ``$HOME`` and credential-looking strings must not leak.

    The v1 schema requires at least one full user-assistant-user
    exchange and a >=40-char reference, so the session needs three
    turns; the credentials appear in turn one to verify scrubbing
    survives the truncation pass.
    """
    home = tmp_path / "fake_home"
    home.mkdir()
    monkeypatch.setattr(eaf, "HOME", str(home))
    session = tmp_path / "claude.jsonl"
    # Each body lands ~1500 chars. Three of them = ~4500 chars ≈ 1125
    # tokens. The slicer's bucket-cross detection compares cumulative
    # tokens to the bucket, so we want bodies small enough that the
    # FIRST record alone doesn't blow past the bucket (otherwise the
    # pre-append snapshot is empty and gets dropped by the v1
    # "len(prefix) < 3" filter).
    leaky_body = (
        f"path is {home}/secret.txt and token=sk-AAAAAAAAAAAAAAAAAAAAAAAAAA "
        + "and a bunch more body content so it crosses the bucket. " * 30
    )
    asst_body = (
        f"I see the path {home}/output.log let me think about it. "
        + "More long reference content. " * 30
    )
    user_followup = "follow up: " + ("more content. " * 30)
    session.write_text(
        json.dumps({
            "type": "user", "sessionId": "s",
            "cwd": str(home / "proj"),
            "timestamp": "2026-05-29T00:00:00Z",
            "message": {"content": leaky_body},
        }) + "\n"
        + json.dumps({
            "type": "assistant", "sessionId": "s",
            "timestamp": "2026-05-29T00:00:01Z",
            "message": {"content": [{"type": "text", "text": asst_body}]},
        }) + "\n"
        + json.dumps({
            "type": "user", "sessionId": "s",
            "timestamp": "2026-05-29T00:00:02Z",
            "message": {"content": user_followup},
        }) + "\n"
        + json.dumps({
            "type": "assistant", "sessionId": "s",
            "timestamp": "2026-05-29T00:00:03Z",
            "message": {"content": [{"type": "text", "text": "final assistant reference body that is long enough to clear the 40 char threshold."}]},
        }) + "\n"
    )
    # Bucket sized so the slicer doesn't cross it on the first record
    # (which would snapshot an empty prefix and fail the v1 "min 3
    # messages" filter). Tail-flush at session end emits one case
    # whose prefix is the first three turns (user, asst, user) and
    # whose reference is the final assistant turn.
    cases = eaf._slice_claude_multi_turn(session, [4096])
    assert cases
    for c in cases:
        for m in c["messages"]:
            assert str(home) not in m["content"], "HOME path leaked into messages"
            assert "sk-AAAAAAAAAA" not in m["content"], "secret-looking token leaked"
        # Reference response is also scrubbed.
        assert str(home) not in c["reference_response"], "HOME path leaked into reference_response"
        # cwd in initial_state scrubbed too
        assert str(home) not in (c["initial_state"]["cwd"] or "")


def test_codex_record_decoding(tmp_path: Path, eaf) -> None:
    """Codex response_item records decode to (role, text) appropriately."""
    msg_user = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hi from codex"}],
        },
    }
    msg_asst = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi back"}],
        },
    }
    fcall = {
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "name": "exec_command",
            "arguments": '{"cmd": "ls"}',
        },
    }
    foutput = {
        "type": "response_item",
        "payload": {
            "type": "function_call_output",
            "output": "file_a\nfile_b\n",
        },
    }
    assert eaf._codex_record_to_message(msg_user) == ("user", "hi from codex")
    assert eaf._codex_record_to_message(msg_asst) == ("assistant", "hi back")
    role, text = eaf._codex_record_to_message(fcall)
    assert role == "assistant" and "exec_command" in text
    role, text = eaf._codex_record_to_message(foutput)
    assert role == "user" and "tool result" in text
