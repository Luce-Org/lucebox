"""Shared math-scoring helpers for answer equivalence.

Canonical home for ``_extract_boxed``, ``_normalize_math``, and
``_math_equiv``.  Every harness script that scores MATH / GSM8K
answers should import from here rather than maintaining a local copy.
"""

from __future__ import annotations

import re
from fractions import Fraction


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


_COMMA_NUM_RE = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
_FRAC_RE = re.compile(r"^([+-]?)\s*\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}$")
_MIXED_RE = re.compile(r"^([+-]?)\s*([0-9,]+)\s*\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}$")


def _strip_valid_commas(s: str) -> str | None:
    """Validate comma placement and remove separators if valid.

    Accepts correctly grouped thousands (e.g. '1,000', '12,345', '1,234,567').
    Rejects malformed grouping (e.g. '1,2', '12,34', '1,23,456').
    Returns s unchanged if no commas are present.
    """
    if "," not in s:
        return s
    if _COMMA_NUM_RE.match(s):
        return s.replace(",", "")
    return None


def _parse_numeric_value(s: str) -> Fraction | None:
    """Parse a normalized math answer string into an exact Fraction.

    Handles integers, decimal numbers, signed simple fractions, and signed
    mixed fractions. Enforces strict thousands comma validation.
    Returns None for symbolic or non-numeric expressions.
    """
    if not s:
        return None

    # 1. Plain float or integer (with comma validation)
    clean = _strip_valid_commas(s.strip())
    if clean is not None:
        try:
            return Fraction(clean)
        except (ValueError, ZeroDivisionError):
            pass

    # 2. Mixed fraction: [+-]? whole \frac{num}{den}
    m = _MIXED_RE.match(s)
    if m:
        try:
            sign = -1 if m.group(1) == "-" else 1
            whole_str = _strip_valid_commas(m.group(2).strip())
            num_str = _strip_valid_commas(m.group(3).strip())
            den_str = _strip_valid_commas(m.group(4).strip())
            if whole_str is None or num_str is None or den_str is None:
                return None
            whole = Fraction(whole_str)
            num = Fraction(num_str)
            den = Fraction(den_str)
            if den == 0:
                return None
            return sign * (whole + num / den)
        except (ValueError, ZeroDivisionError):
            pass

    # 3. Simple fraction: [+-]? \frac{num}{den}
    m = _FRAC_RE.match(s)
    if m:
        try:
            sign = -1 if m.group(1) == "-" else 1
            num_str = _strip_valid_commas(m.group(2).strip())
            den_str = _strip_valid_commas(m.group(3).strip())
            if num_str is None or den_str is None:
                return None
            num = Fraction(num_str)
            den = Fraction(den_str)
            if den == 0:
                return None
            return sign * (num / den)
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
