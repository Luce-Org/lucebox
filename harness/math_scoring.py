"""Shared math-scoring helpers for answer equivalence.

Canonical home for ``_extract_boxed``, ``_normalize_math``, and
``_math_equiv``.  Every harness script that scores MATH / GSM8K
answers should import from here rather than maintaining a local copy.
"""

from __future__ import annotations

import re


def _extract_boxed(text: str) -> str | None:
    """Extract the last \\boxed{...} from a string, handling nested braces."""
    results = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        start = idx + len("\\boxed{")
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start : j - 1].strip())
        i = j
    return results[-1] if results else None


def _normalize_math(s: str | None) -> str:
    """Normalize a math answer string for comparison."""
    if s is None:
        return ""
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # Strip currency $ (e.g. "$18" -> "18")
    if re.match(r"^\$\d", s):
        s = s[1:]
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", s)
    for cmd in [r"\left", r"\right", r"\displaystyle"]:
        s = s.replace(cmd, "")
    s = s.replace(r"\tfrac", r"\frac")
    s = s.replace(r"\dfrac", r"\frac")
    for unit in [
        " cm", " m", " km", " kg", " g", " s", " ms",
        " degrees", " degree", "\u00b0", " inches", " feet",
        " square units", " units", " dollars",
    ]:
        if s.lower().rstrip(".").endswith(unit):
            s = s[: len(s) - len(unit) - (1 if s.endswith(".") else 0)]
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(".,")
    return s


def _math_equiv(pred: str | None, gold: str | None) -> bool:
    """Check if two math answers are equivalent."""
    if pred is None or gold is None:
        return False
    p = _normalize_math(pred)
    g = _normalize_math(gold)
    if p == g:
        return True
    p_c = re.sub(r"\s*\\frac", r"\\frac", p)
    g_c = re.sub(r"\s*\\frac", r"\\frac", g)
    if p_c == g_c:
        return True
    try:
        pf = float(p.replace(",", ""))
        gf = float(g.replace(",", ""))
        return abs(pf - gf) < 1e-6
    except (ValueError, TypeError):
        pass
    mixed_pat = re.compile(r"^(\d+)\s*\\frac\s*\{(\d+)\}\s*\{(\d+)\}$")
    for s, other in [(p, g), (g, p)]:
        m = mixed_pat.match(s)
        if m:
            try:
                val = float(m.group(1)) + float(m.group(2)) / float(m.group(3))
                oval = float(other.replace(",", ""))
                if abs(val - oval) < 1e-6:
                    return True
            except (ValueError, ZeroDivisionError):
                pass
    frac_pat = re.compile(r"^\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}$")
    for s, other in [(p, g), (g, p)]:
        m = frac_pat.search(s)
        if m:
            try:
                val = float(m.group(1)) / float(m.group(2))
                oval = float(other.replace(",", ""))
                if abs(val - oval) < 1e-6:
                    return True
            except (ValueError, ZeroDivisionError):
                pass
    return False
