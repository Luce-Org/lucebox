"""Client-side thinking-control fallback.

Why this exists
---------------
luce-bench's runner ships three API-side thinking flags on every
request — ``chat_template_kwargs.enable_thinking``,
``thinking: {type: enabled|disabled}``, and ``reasoning_effort``.
luce-dflash + several modern stacks honor at least one of them, but
provider gateways (OpenRouter on 2026-05-27 with Qwen3.6-27B) silently
ignore ALL three: a ``--no-think`` sweep came back with reasoning text
on 83 of 92 rows. Pure API-side control is therefore not enough.

The mitigation here is *prompt-level injection* — append the model
family's documented in-band token (``/think`` / ``/no_think`` for the
Qwen3.x line) to the last user turn. This is what Qwen's own tech
report (arXiv 2505.09388) and the upstream chat template both treat
as authoritative; servers that strip our API flags still pass the
token through to the template.

This module is *augmentation*, not replacement:

  * API flags in ``runner.run_case`` keep firing — stacks that honor
    them (luce-dflash, vLLM with the qwen3 template, …) still benefit.
  * The injection only fires when the operator opts in (CLI
    ``--prompt-thinking-control on``), or in ``auto`` mode when the
    preflight couldn't confirm a lucebox server (``/props`` absent or
    not surfacing ``model_card_source``).
  * The post-run verifier counts rows whose ``reasoning_tokens`` /
    ``reasoning_content`` contradicts the requested mode and flips the
    canonical result's ``thinking_control_honored`` flag accordingly,
    so an OpenRouter-shaped failure surfaces as a single warning line
    rather than 92 quiet contaminated rows.

Family detection is longest-prefix on a lowercased model id: ``qwen3.6``
wins over ``qwen3`` for ``qwen3.6-27b``. Adding a new family is one
entry in :data:`FAMILY_TOKENS`.
"""

from __future__ import annotations

from typing import Any

# In-band tokens documented by upstream model cards. Add new entries
# alphabetically; longest-prefix wins so ``qwen3.6`` resolves before
# ``qwen3``. Keep the keys lowercase — ``_detect_family`` lowercases the
# model id before lookup.
FAMILY_TOKENS: dict[str, dict[str, str]] = {
    "qwen3":   {"think": "/think", "nothink": "/no_think"},
    "qwen3.5": {"think": "/think", "nothink": "/no_think"},
    "qwen3.6": {"think": "/think", "nothink": "/no_think"},
}

# Slack allowed on the contradiction count before we declare a run
# "not honored". 5% covers the occasional 1-2 rows where a non-thinking
# model still emits a few reasoning tokens (e.g. a wrap-up scratch pad
# at the end of decode) without flipping the headline on noise alone.
_VERIFY_SLACK = 0.05


def _detect_family(model_id: str) -> str | None:
    """Longest-prefix family match on a lowercased model id.

    Returns the family key (e.g. ``"qwen3.6"``) when a known entry's
    name appears as a substring after the leading provider slug, or
    None if no family matches. We scan in length-descending order so
    ``"qwen3.6-27b"`` resolves to ``"qwen3.6"`` rather than ``"qwen3"``.

    Lenient enough to handle the shapes seen in the wild:
      * ``qwen3.6-27b``           — bare HF id
      * ``Qwen/Qwen3.6-27B``      — HF org/repo
      * ``qwen/qwen3.6-27b:free`` — OpenRouter route
    """
    if not isinstance(model_id, str) or not model_id:
        return None
    needle = model_id.lower()
    for fam in sorted(FAMILY_TOKENS.keys(), key=len, reverse=True):
        if fam in needle:
            return fam
    return None


def _resolve_tokens(
    model_id: str,
    card: dict[str, Any] | None,
) -> tuple[dict[str, str] | None, str]:
    """Pick the {think, nothink} token pair for this run.

    Resolution order:

      1. ``card["thinking_control"]`` with ``think_prompt_token`` +
         ``nothink_prompt_token`` (explicit per-card override).
      2. :data:`FAMILY_TOKENS` indexed by longest-prefix family match
         on the model id.
      3. Nothing — return ``(None, "none")`` so the caller skips.

    Returns ``(tokens, source)`` where ``source`` is one of
    ``"card"``, ``"family_map"``, or ``"none"``. The source string
    lands in the result.json's ``thinking_control_injection.source``
    field so an operator reading a report can tell whether the
    injection was driven by a hand-curated card or a default-family
    guess.
    """
    if isinstance(card, dict):
        tc = card.get("thinking_control")
        if isinstance(tc, dict):
            think_tok = tc.get("think_prompt_token")
            nothink_tok = tc.get("nothink_prompt_token")
            if isinstance(think_tok, str) and isinstance(nothink_tok, str):
                return (
                    {"think": think_tok, "nothink": nothink_tok},
                    "card",
                )
    fam = _detect_family(model_id)
    if fam is not None:
        return FAMILY_TOKENS[fam], "family_map"
    return None, "none"


def maybe_inject_thinking_token(
    messages: list[dict[str, Any]],
    *,
    mode: str,
    model_id: str,
    card: dict[str, Any] | None,
    control_flag: str,
    server_honors_api_flags: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return ``(messages, info)`` with the family/card token injected.

    Resolution order:

      1. ``control_flag == "off"`` → no injection, regardless of
         anything else. Restores pre-feature behavior.
      2. ``control_flag == "auto"`` AND ``server_honors_api_flags`` is
         True → no injection. The server (luce-dflash) already enforces
         control server-side via the chat template; injecting would
         double the marker and risk template confusion.
      3. ``control_flag == "on"`` (or ``"auto"`` + flags-not-honored) →
         resolve a token via ``card.thinking_control`` then
         :data:`FAMILY_TOKENS`. If nothing resolves, skip — we don't
         have a safe default for unknown families.
      4. Mutate the LAST user-turn's content (appending ``" <token>"``
         with a single space separator). The input ``messages`` list
         is NOT modified in place; we return a new list with the last
         message shallow-copied so callers can keep their own ref.
      5. Idempotent — if the resolved token already appears at the end
         of the last user message (modulo trailing whitespace), the
         function is a no-op for that segment but still reports
         ``active=True`` in the info dict so the result.json reflects
         the operator's intent.

    Parameters
    ----------
    messages
        The OpenAI-shape messages list about to be POSTed.
    mode
        Either ``"think"`` or ``"nothink"`` — the operator's requested
        reasoning state.
    model_id
        The model id we'll send in the request body. Used for
        longest-prefix family lookup when no card is supplied.
    card
        Optional model-card dict (the shape under
        ``share/model_cards/<name>.json``). When the card carries a
        ``thinking_control`` block, its explicit tokens win over the
        family map.
    control_flag
        ``"auto"`` / ``"on"`` / ``"off"`` from the CLI flag.
    server_honors_api_flags
        True when the preflight confirmed the server respects API-side
        thinking flags (currently: a /props response surfaces
        ``model_card_source``, indicating a lucebox stack). False when
        /props was absent or didn't surface the field.

    Returns
    -------
    (new_messages, info)
        ``new_messages`` is the (possibly augmented) message list to
        send. ``info`` is a dict suitable for embedding in the result
        as ``thinking_control_injection``:

        ``{"active": bool, "token": str | None, "source": str}``

        ``source`` is one of ``"card"``, ``"family_map"``, ``"none"``.
    """
    info_inactive = {"active": False, "token": None, "source": "none"}

    if control_flag not in {"auto", "on", "off"}:
        # Defensive: argparse should already constrain this. Fall back
        # to a no-op rather than raise — surfacing an injection bug as
        # a benchmark abort would be a worse failure mode.
        return list(messages), info_inactive

    if control_flag == "off":
        return list(messages), info_inactive

    if control_flag == "auto" and server_honors_api_flags:
        # luce-dflash / any server that the preflight confirmed honors
        # API-side flags. The chat template already enforces control;
        # injecting again would risk duplicate markers in the rendered
        # prompt.
        return list(messages), info_inactive

    if mode not in {"think", "nothink"}:
        return list(messages), info_inactive

    tokens, source = _resolve_tokens(model_id, card)
    if tokens is None:
        # Unknown family with no card override — we don't have a safe
        # default. Skip rather than guess.
        return list(messages), info_inactive

    token = tokens.get(mode)
    if not isinstance(token, str) or not token:
        return list(messages), info_inactive

    if not messages:
        return list(messages), info_inactive

    # Locate the last user turn. Walking from the end means we land on
    # the operator's actual question even when the area or future
    # callers stack multiple user turns (rare today but cheap to be
    # forward-compatible).
    new_messages = list(messages)
    last_user_idx: int | None = None
    for i in range(len(new_messages) - 1, -1, -1):
        m = new_messages[i]
        if isinstance(m, dict) and m.get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        # No user turn (shouldn't happen with run_case but defensive).
        return new_messages, info_inactive

    last_msg = dict(new_messages[last_user_idx])
    content = last_msg.get("content")
    if not isinstance(content, str):
        # Multi-part content (vision etc.) is out of scope for the v1
        # text-only injection — skip rather than try to coerce.
        return new_messages, info_inactive

    # Idempotent: if the token already trails the message (modulo
    # whitespace), do not append again. Use rstrip on the haystack so
    # an earlier injection that left a "\n" between the user text and
    # the token still counts as present.
    stripped = content.rstrip()
    if stripped.endswith(token):
        new_messages[last_user_idx] = last_msg  # shallow copy already
        last_msg["content"] = content  # unchanged but explicit
        return new_messages, {"active": True, "token": token, "source": source}

    last_msg["content"] = content + " " + token if content else token
    new_messages[last_user_idx] = last_msg
    return new_messages, {"active": True, "token": token, "source": source}


def verify_thinking_control(
    rows: list[dict[str, Any]],
    requested_mode: str,
) -> tuple[bool, int]:
    """Post-run sanity check: did the server honor the requested mode?

    A row "contradicts" the requested mode when:

      * ``requested_mode == "nothink"`` and the row carries reasoning
        — non-zero ``reasoning_tokens`` (preferred) or non-empty
        ``reasoning_content``. Either signal is enough; OpenRouter's
        Qwen3 routes have been seen to ship reasoning text without a
        token count.
      * ``requested_mode == "think"`` and the row carries NO reasoning
        — zero/missing ``reasoning_tokens`` AND empty ``reasoning_content``.

    Returns ``(honored, contradicting_count)`` where ``honored`` is
    True iff ``contradicting / n < 5%`` (see :data:`_VERIFY_SLACK`).
    The slack lets one or two genuinely-stop-on-first-token nothink
    rows off the hook without masking a real provider-ignoring-flags
    failure (which presents as 80%+ contradicting on Qwen3.6).

    Both legacy (``thinking_tokens``) and current (``reasoning_tokens``)
    field names are read; whichever is present wins, with the new name
    preferred when both are set.
    """
    if requested_mode not in {"think", "nothink"}:
        return True, 0
    if not rows:
        return True, 0

    contradicting = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        # reasoning_tokens (current schema) takes precedence over
        # thinking_tokens (legacy) when both are set — the runner only
        # writes the new field, but the verifier also runs through the
        # regrade CLI which loads historical files.
        tokens = r.get("reasoning_tokens")
        if tokens is None:
            tokens = r.get("thinking_tokens")
        text = r.get("reasoning_content")
        has_reasoning_tokens = isinstance(tokens, int) and tokens > 0
        has_reasoning_text = isinstance(text, str) and bool(text.strip())
        if requested_mode == "nothink":
            if has_reasoning_tokens or has_reasoning_text:
                contradicting += 1
        else:  # think
            if not has_reasoning_tokens and not has_reasoning_text:
                contradicting += 1

    fraction = contradicting / len(rows)
    honored = fraction < _VERIFY_SLACK
    return honored, contradicting


__all__ = [
    "FAMILY_TOKENS",
    "maybe_inject_thinking_token",
    "verify_thinking_control",
]
