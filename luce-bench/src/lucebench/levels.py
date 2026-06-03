"""Curated snapshot levels — area sets keyed by ``level0..level3``.

The snapshot primitive runs one or more areas in sequence; this module
encodes which areas belong to each tier so the CLI surface stays tiny
(``--level level1``) and the level → area mapping isn't duplicated.

Each entry is a ``(area_name, n_cap)`` tuple. ``n_cap`` is the
per-area ``--questions`` cap (``None`` = run all cases for that area).
"""

from __future__ import annotations

LEVELS: dict[str, list[tuple[str, int | None]]] = {
    "level0": [
        ("smoke", None),
    ],
    "level1": [
        ("smoke", None),
        ("code", 5),
        ("gsm8k", 5),
        ("agent", 2),
        ("longctx", 1),
    ],
    "level2": [
        ("smoke", None),
        ("code", None),
        ("gsm8k", None),
        ("truthfulqa-mc1", None),
        ("hellaswag", None),
        ("agent", None),
        ("longctx", None),
    ],
    "level3": [
        ("smoke", None),
        ("code", None),
        ("gsm8k", None),
        ("truthfulqa-mc1", None),
        ("hellaswag", None),
        ("agent", None),
        ("longctx", None),
        ("ds4-eval", None),
        ("agent_recorded", None),
        ("forge", None),
    ],
}


def resolve_level(level: str) -> list[tuple[str, int | None]]:
    """Return the ``(area, n_cap)`` list for ``level`` (e.g. ``"level1"``).

    Raises ``ValueError`` on an unknown level so callers can surface a
    crisp argparse error instead of a KeyError stack.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; want one of {list(LEVELS)}")
    return list(LEVELS[level])
