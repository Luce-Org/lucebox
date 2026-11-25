"""Client-side thinking-control unit tests.

Covers ``lucebench._thinking`` end-to-end without firing a real HTTP
request:

  * Injection resolution order — card > family map > skip.
  * Auto-mode behavior against a lucebox-confirmed server vs a
    bare provider gateway.
  * Idempotency + non-mutation of the caller's messages list.
  * The post-run verifier flips honored=False when reasoning_tokens /
    reasoning_content contradict the requested mode, and respects the
    5% slack threshold.

No HTTP fixtures — the runner stays out of scope here; that's covered
by the existing ``test_runner.py`` integration. These tests pin the
shape of the helper functions so a future refactor (e.g. moving the
resolution into a class) keeps the contract.
"""

from __future__ import annotations

from lucebench._thinking import (
    FAMILY_TOKENS,
    maybe_inject_thinking_token,
    verify_thinking_control,
)

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def _msgs(text: str = "What is 2+2?") -> list[dict[str, str]]:
    """Minimal two-turn message list — system + user."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]


def _qwen_card() -> dict:
    """Synthetic Qwen3.6 card carrying explicit thinking_control tokens."""
    return {
        "name": "Qwen3.6 27B (test)",
        "thinking_control": {
            "think_prompt_token": "/think",
            "nothink_prompt_token": "/no_think",
            "injection_point": "user_turn_suffix",
        },
    }


# ────────────────────────────────────────────────────────────────────
# 1-2. Card-driven injection — both modes
# ────────────────────────────────────────────────────────────────────


def test_inject_qwen_card_appends_token_think():
    msgs = _msgs()
    out, info = maybe_inject_thinking_token(
        msgs,
        mode="think",
        model_id="qwen3.6-27b",
        card=_qwen_card(),
        control_flag="on",
        server_honors_api_flags=False,
    )
    # Last user turn carries the appended token.
    assert out[-1]["content"].endswith("/think")
    assert out[-1]["content"] == "What is 2+2? /think"
    # Info block reflects what landed.
    assert info == {"active": True, "token": "/think", "source": "card"}


def test_inject_qwen_card_appends_token_nothink():
    msgs = _msgs()
    out, info = maybe_inject_thinking_token(
        msgs,
        mode="nothink",
        model_id="qwen3.6-27b",
        card=_qwen_card(),
        control_flag="on",
        server_honors_api_flags=False,
    )
    assert out[-1]["content"] == "What is 2+2? /no_think"
    assert info == {"active": True, "token": "/no_think", "source": "card"}


# ────────────────────────────────────────────────────────────────────
# 3. Family-map fallback
# ────────────────────────────────────────────────────────────────────


def test_inject_falls_back_to_family_map():
    msgs = _msgs()
    out_think, info_think = maybe_inject_thinking_token(
        msgs,
        mode="think",
        model_id="qwen3.6-27b",
        card=None,
        control_flag="on",
        server_honors_api_flags=False,
    )
    assert out_think[-1]["content"].endswith("/think")
    assert info_think["source"] == "family_map"
    assert info_think["token"] == "/think"
    out_no, info_no = maybe_inject_thinking_token(
        _msgs(),
        mode="nothink",
        model_id="qwen3.6-27b",
        card=None,
        control_flag="on",
        server_honors_api_flags=False,
    )
    assert out_no[-1]["content"].endswith("/no_think")
    assert info_no["source"] == "family_map"
    assert info_no["token"] == "/no_think"
    # Sanity: the longest-prefix match resolves to qwen3.6, not qwen3.
    assert "qwen3.6" in FAMILY_TOKENS


# ────────────────────────────────────────────────────────────────────
# 4. Unknown families silently skip
# ────────────────────────────────────────────────────────────────────


def test_inject_skipped_for_unknown_family():
    msgs = _msgs()
    out, info = maybe_inject_thinking_token(
        msgs,
        mode="nothink",
        model_id="mystery-model-7b",
        card=None,
        control_flag="on",
        server_honors_api_flags=False,
    )
    # Messages unchanged.
    assert out[-1]["content"] == "What is 2+2?"
    assert info == {"active": False, "token": None, "source": "none"}


# ────────────────────────────────────────────────────────────────────
# 5-7. Control-flag matrix — auto / on / off
# ────────────────────────────────────────────────────────────────────


def test_auto_skips_lucebox_server():
    """auto + server_honors_api_flags=True → no injection (server enforces)."""
    msgs = _msgs()
    out, info = maybe_inject_thinking_token(
        msgs,
        mode="nothink",
        model_id="qwen3.6-27b",
        card=_qwen_card(),
        control_flag="auto",
        server_honors_api_flags=True,
    )
    assert out[-1]["content"] == "What is 2+2?"
    assert info["active"] is False


def test_auto_engages_no_props():
    """auto + server_honors_api_flags=False (e.g. OpenRouter) → inject."""
    msgs = _msgs()
    out, info = maybe_inject_thinking_token(
        msgs,
        mode="nothink",
        model_id="qwen3.6-27b",
        card=None,
        control_flag="auto",
        server_honors_api_flags=False,
    )
    assert out[-1]["content"].endswith("/no_think")
    assert info["active"] is True
    assert info["source"] == "family_map"


def test_on_forces_injection_even_for_lucebox():
    """control=on forces injection regardless of server detection."""
    msgs = _msgs()
    out, info = maybe_inject_thinking_token(
        msgs,
        mode="think",
        model_id="qwen3.6-27b",
        card=_qwen_card(),
        control_flag="on",
        server_honors_api_flags=True,
    )
    assert out[-1]["content"].endswith("/think")
    assert info["active"] is True


# ────────────────────────────────────────────────────────────────────
# 8. Off is a hard no-op
# ────────────────────────────────────────────────────────────────────


def test_off_never_injects():
    msgs = _msgs()
    out, info = maybe_inject_thinking_token(
        msgs,
        mode="nothink",
        model_id="qwen3.6-27b",
        card=_qwen_card(),
        control_flag="off",
        server_honors_api_flags=False,
    )
    assert out[-1]["content"] == "What is 2+2?"
    assert info == {"active": False, "token": None, "source": "none"}


# ────────────────────────────────────────────────────────────────────
# 9-10. Verifier: contradiction vs slack
# ────────────────────────────────────────────────────────────────────


def test_verify_flags_contradicting_run():
    """nothink run with reasoning_tokens > 0 on every row → not honored."""
    rows = [
        {
            "case_id": f"c-{i}",
            "reasoning_tokens": 42,
            "reasoning_content": "thinking...",
        }
        for i in range(10)
    ]
    honored, contradicting = verify_thinking_control(rows, "nothink")
    assert honored is False
    assert contradicting == 10


def test_verify_honors_run_within_threshold():
    """4/92 contradicting → < 5% → honored=True."""
    # 92 nothink rows — 4 carry stray reasoning, 88 are clean.
    rows = []
    for i in range(4):
        rows.append({"case_id": f"bad-{i}", "reasoning_tokens": 17})
    for i in range(88):
        rows.append({"case_id": f"ok-{i}", "reasoning_tokens": 0, "reasoning_content": ""})
    honored, contradicting = verify_thinking_control(rows, "nothink")
    # 4/92 = 4.3% < 5% slack → still honored.
    assert honored is True
    assert contradicting == 4


# ────────────────────────────────────────────────────────────────────
# 11-12. Idempotency + non-mutation
# ────────────────────────────────────────────────────────────────────


def test_inject_idempotent():
    """Token already at end-of-string → no double-append."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "What is 2+2? /no_think"},
    ]
    out, info = maybe_inject_thinking_token(
        msgs,
        mode="nothink",
        model_id="qwen3.6-27b",
        card=_qwen_card(),
        control_flag="on",
        server_honors_api_flags=False,
    )
    # Still just one occurrence — the second call is a no-op for content.
    assert out[-1]["content"].count("/no_think") == 1
    assert out[-1]["content"] == "What is 2+2? /no_think"
    # But ``info.active`` stays True so the result.json reflects intent.
    assert info["active"] is True


def test_inject_does_not_mutate_input_messages():
    """The caller's messages list is not modified in place."""
    msgs = _msgs()
    orig_last_content = msgs[-1]["content"]
    orig_len = len(msgs)
    orig_last_dict_id = id(msgs[-1])
    out, _info = maybe_inject_thinking_token(
        msgs,
        mode="nothink",
        model_id="qwen3.6-27b",
        card=_qwen_card(),
        control_flag="on",
        server_honors_api_flags=False,
    )
    # Caller's list reference: unchanged length + last message untouched.
    assert len(msgs) == orig_len
    assert msgs[-1]["content"] == orig_last_content
    # And the returned list points at a new last-message dict — proves
    # the helper shallow-copied rather than mutating.
    assert id(out[-1]) != orig_last_dict_id
    assert out[-1]["content"].endswith("/no_think")
