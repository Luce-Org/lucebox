"""Shared math-scoring helpers for answer equivalence.

Canonical home for ``_extract_boxed``, ``_normalize_math``, and
``_math_equiv``.  Every harness script that scores MATH / GSM8K
answers should import from here rather than maintaining a local copy.
"""

from __future__ import annotations

import math
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


_FRAC_RE = re.compile(r"^([+-]?)\s*\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}$")
_MIXED_RE = re.compile(r"^([+-]?)\s*(\d+)\s*\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}$")


def _parse_numeric_value(s: str) -> float | None:
    """Parse a normalized math answer string into a numeric float value.

    Handles integers, decimal floats, signed simple fractions, and signed
    mixed fractions. Returns None for symbolic or non-numeric expressions.
    """
    if not s:
        return None

    # 1. Plain float or integer (with optional commas)
    clean = s.replace(",", "").strip()
    try:
        val = float(clean)
        if math.isfinite(val):
            return val
    except (ValueError, TypeError):
        pass

    # 2. Mixed fraction: [+-]? \d+ \frac{\d+}{\d+}
    m = _MIXED_RE.match(s)
    if m:
        try:
            sign = -1.0 if m.group(1) == "-" else 1.0
            whole = float(m.group(2))
            num = float(m.group(3).replace(",", "").strip())
            den = float(m.group(4).replace(",", "").strip())
            if den == 0:
                return None
            val = sign * (whole + num / den)
            if math.isfinite(val):
                return val
        except (ValueError, ZeroDivisionError):
            pass

    # 3. Simple fraction: [+-]? \frac{num}{den}
    m = _FRAC_RE.match(s)
    if m:
        try:
            sign = -1.0 if m.group(1) == "-" else 1.0
            num = float(m.group(2).replace(",", "").strip())
            den = float(m.group(3).replace(",", "").strip())
            if den == 0:
                return None
            val = sign * (num / den)
            if math.isfinite(val):
                return val
        except (ValueError, ZeroDivisionError):
            pass

    return None


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

    val_p = _parse_numeric_value(p)
    val_g = _parse_numeric_value(g)
    if val_p is not None and val_g is not None:
        return abs(val_p - val_g) < 1e-6

    return False
