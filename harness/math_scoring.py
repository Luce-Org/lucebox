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


def _brace_group_at(s: str, k: int) -> tuple[str, int] | None:
    """Return (contents, end_index) of a {...} group starting at k, else None."""
    if k >= len(s) or s[k] != "{":
        return None
    depth = 0
    for p in range(k, len(s)):
        if s[p] == "{":
            depth += 1
        elif s[p] == "}":
            depth -= 1
            if depth == 0:
                return s[k + 1 : p], p + 1
    return None


def _expand_frac(s: str) -> str:
    """Rewrite \\frac{A}{B} (and \\dfrac/\\tfrac) to A/B so LaTeX and plain
    forms compare equal. Handles nested braces in A/B."""
    out = s
    for _ in range(32):
        m = re.search(r"\\(?:d|t)?frac\s*\{", out)
        if m is None:
            break
        i = m.start()
        g1 = _brace_group_at(out, out.index("{", i))
        if g1 is None:
            break
        a, e1 = g1
        g2 = _brace_group_at(out, e1)
        if g2 is None:
            break
        b, e2 = g2
        out = out[:i] + f"{a}/{b}" + out[e2:]
    return out


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
    s = _expand_frac(s)
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", s)
    for cmd in [r"\left", r"\right", r"\displaystyle", r"\,", r"\;", r"\:", r"\!", r"\ "]:
        s = s.replace(cmd, "")
    s = s.replace(r"\tfrac", r"\frac")
    s = s.replace(r"\dfrac", r"\frac")
    for unit in [
        " cm",
        " m",
        " km",
        " kg",
        " g",
        " s",
        " ms",
        " degrees",
        " degree",
        "\u00b0",
        " inches",
        " feet",
        " square units",
        " units",
        " dollars",
    ]:
        if s.lower().rstrip(".").endswith(unit):
            s = s[: len(s) - len(unit) - (1 if s.endswith(".") else 0)]
    # Interval/coordinate spacing: "[2, 5)" == "[2,5)".
    s = re.sub(r",\s+", ",", s)
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
