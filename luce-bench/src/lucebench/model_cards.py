"""Client-side model-card registry — resolve a card by model id.

Why this exists
---------------
luce-bench's thinking-control machinery (``_thinking.py``) is only correct
when it uses the *right* tokens for the *right* model. Historically a card
only arrived from ``/props.model_card`` on a lucebox server; against
OpenRouter / MLX the client fell back to a hardcoded Qwen family guess and
knew nothing about per-model terminators, reply reserves, or effort tiers.

This module gives luce-bench a **card registry** resolvable by model id,
mirroring the canonical cards in repo ``share/model_cards/``. The cards are
embedded at build time (see ``pyproject.toml`` force-include) into the wheel
as package data under ``lucebench/_model_cards/`` so the standalone PyPI
build carries them without reaching the repo tree.

Resolution order (see :func:`resolve_card`):
  1. ``/props.model_card`` — authoritative when present (lucebox server told
     us the exact card it loaded). source = ``"props"``.
  2. Bundled registry, keyed by :func:`normalize_model_card_stem`. source =
     ``"bundled"``.
  3. Nothing → ``(None, "none")``. The family fallback stays in
     ``_thinking.py`` (it operates on the model id, not a card).

A bundled-card id match is a HINT, not proof: OpenRouter ids, aliases, and
quant routes can normalize to a known stem while serving behavior that
disagrees with the card. The match selects *which* card to read; whether to
*trust* it for thinking control is gated separately (see
:func:`card_is_thinking_capable` and the capability gate in ``cli.py``).

TODO (Tier 2, deferred — see docs/client-thinking-budget.md):
  * client-side abort + re-prompt continuation
  * char/4 mid-stream reasoning-token estimate
  * CI drift guard comparing packaged card hashes to share/model_cards/
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

# Provider prefixes stripped before normalization. OpenRouter / HF route ids
# carry an ``org/`` segment (``qwen/qwen3.6-27b:free``) that is not part of
# the card stem. We only strip a leading ``<segment>/`` once — the canonical
# stem never contains a slash.
#
# Route / quant suffixes stripped after the last segment: OpenRouter appends
# ``:free`` / ``:nitro`` / ``:floor`` etc. after a colon. The card stem never
# contains a colon, so everything from the first colon on is dropped.


def normalize_model_card_stem(model_id: str) -> str:
    """Mirror the server's ``normalize_model_card_stem`` with client extras.

    The server (``server/src/server/model_card.cpp``) normalizes a clean
    ``general.name`` from GGUF metadata: lowercase, ``space``/``tab``/``_`` →
    ``-``, keep ``[a-z0-9.-]``, drop other punctuation. luce-bench works from
    a *model id* sent on the wire (``qwen/qwen3.6-27b:free``), so before
    applying that rule we additionally:

      * strip a leading provider/org prefix (``qwen/`` → ``""``), keeping
        only the final path segment, and
      * strip a route/quant suffix introduced by a colon (``:free`` etc.).

    Examples::

        qwen/qwen3.6-27b:free   → qwen3.6-27b
        Qwen/Qwen3.6-27B        → qwen3.6-27b
        qwen3.6-27b             → qwen3.6-27b
        Qwen3.6 27B             → qwen3.6-27b
        unknown-model-7b        → unknown-model-7b
    """
    if not isinstance(model_id, str):
        return ""
    s = model_id
    # Strip route/quant suffix: everything from the first ':' on.
    colon = s.find(":")
    if colon != -1:
        s = s[:colon]
    # Strip provider/org prefix: keep only the last '/'-segment.
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    # Core rule — byte-for-byte the server's loop.
    out: list[str] = []
    for ch in s:
        if ch in (" ", "\t", "_"):
            out.append("-")
        elif "A" <= ch <= "Z":
            out.append(ch.lower())
        elif ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch in (".", "-"):
            out.append(ch)
        # else: silently drop punctuation
    return "".join(out)


@lru_cache(maxsize=1)
def _bundled_cards() -> dict[str, dict[str, Any]]:
    """Load every bundled card into a ``{stem: card_dict}`` map.

    Reads the package data embedded at build time under
    ``lucebench/_model_cards/*.json`` (``_schema.json`` excluded). The map
    is keyed by the file stem, which is already the server's normalized
    form. Missing package (e.g. an editable install that hasn't run the
    force-include) yields an empty map rather than raising — the resolver
    then falls through to ``"none"``.
    """
    from importlib import resources

    cards: dict[str, dict[str, Any]] = {}
    try:
        root = resources.files("lucebench").joinpath("_model_cards")
    except (ModuleNotFoundError, AttributeError):
        return cards
    if not root.is_dir():
        return cards
    for entry in root.iterdir():
        name = entry.name
        if not name.endswith(".json") or name == "_schema.json":
            continue
        stem = name[: -len(".json")]
        try:
            data = json.loads(entry.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if isinstance(data, dict):
            cards[stem] = data
    return cards


def card_is_thinking_capable(card: dict[str, Any] | None) -> bool:
    """True iff the card carries thinking fields the feature can act on.

    The capability gate: think/nothink token injection (and, later, client
    budgeting) only activate for a model that resolves to a thinking-capable
    card. A card is thinking-capable when it has a ``thinking_control`` block
    OR a ``thinking_terminator_hint`` — either is evidence of a thinking
    channel. Everything else runs plain, never injected.
    """
    if not isinstance(card, dict):
        return False
    tc = card.get("thinking_control")
    if isinstance(tc, dict) and tc:
        return True
    hint = card.get("thinking_terminator_hint")
    return isinstance(hint, str) and bool(hint)


def card_sampling(card: dict[str, Any] | None) -> dict[str, Any]:
    """Return the card's ``sampling`` block, or ``{}`` when absent.

    The sampling block carries the model's recommended decode params
    (``temperature``, ``top_p``, ``top_k``, ``min_p``, ``presence_penalty``,
    ``repetition_penalty``). luce-bench applies these by default against
    card-less servers (OpenRouter / MLX) so the model runs with its own
    recommended sampling instead of the provider's defaults; the CLI's
    ``--no-card-sampling`` opts out, and explicit ``--temperature`` /
    ``--top-p`` / ``--top-k`` override per field. Returns a fresh dict the
    caller may mutate.
    """
    if not isinstance(card, dict):
        return {}
    sampling = card.get("sampling")
    if isinstance(sampling, dict):
        return dict(sampling)
    return {}


def resolve_card(
    model_id: str,
    props_model_card: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Resolve a model card + its provenance for ``model_id``.

    Resolution order:
      1. ``props_model_card`` given → ``(card, "props")`` — authoritative.
      2. Bundled registry lookup by normalized stem → ``(card, "bundled")``.
      3. Nothing → ``(None, "none")``.

    The family fallback (``_thinking.FAMILY_TOKENS``) is intentionally NOT
    consulted here — it keys off the model id, not a card, and stays in
    ``_thinking.py``. ``source`` lands in the result rows as ``card_source``.
    """
    if isinstance(props_model_card, dict) and props_model_card:
        return props_model_card, "props"
    stem = normalize_model_card_stem(model_id)
    if stem:
        card = _bundled_cards().get(stem)
        if card is not None:
            return card, "bundled"
    return None, "none"


__all__ = [
    "normalize_model_card_stem",
    "resolve_card",
    "card_is_thinking_capable",
    "card_sampling",
]
