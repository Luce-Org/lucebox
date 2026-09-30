"""Deterministic graders: numeric, math, choice, label, json_keys, python_tests, ifeval.

Every grader returns a ``GradeResult``. The LLM judge lives in ``judge.py``.
Graders only ever see the answer with ``<think>`` blocks removed.
"""

from __future__ import annotations

import json
import os
import re
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any

from . import ifeval_checks
from .common import strip_think


@dataclass
class GradeResult:
    correct: bool
    score: float
    grader: str
    detail: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- math


def extract_boxed(text: str) -> str | None:
    """Last ``\\boxed{...}`` (or ``\\fbox{...}``) content, handling nested braces."""
    results = []
    for macro in ("\\boxed{", "\\fbox{"):
        i = 0
        while True:
            idx = text.find(macro, i)
            if idx == -1:
                break
            start = idx + len(macro)
            depth, j = 1, start
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                results.append((idx, text[start : j - 1].strip()))
            i = j
    return max(results)[1] if results else None


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?|-?\.\d+")


def extract_last_number(text: str) -> str | None:
    nums = _NUM_RE.findall(text)
    return nums[-1] if nums else None


def _to_float(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").replace("\\%", "").rstrip("%").strip()
    s = s.rstrip(".")
    m = re.fullmatch(r"(-?)\\[dt]?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", s)
    if m:
        try:
            v = float(m.group(2)) / float(m.group(3))
            return -v if m.group(1) else v
        except (ValueError, ZeroDivisionError):
            return None
    m = re.fullmatch(r"(-?[\d.]+)/([\d.]+)", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def normalize_math(s: str | None) -> str:
    if s is None:
        return ""
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s.strip("$").strip()
    if re.match(r"^\\\$\d|^\$\d", s):
        s = s.lstrip("\\$")
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\s*\{([^}]*)\}", r"\1", s)
    for cmd in (r"\left", r"\right", r"\displaystyle", r"\!", r"\,", r"\;", r"\ "):
        s = s.replace(cmd, "")
    s = s.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("°", "")
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\{([^}]*)\}(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", s)
    for unit in (" cm", " m", " km", " kg", " g", " s", " degrees", " degree", " inches",
                 " feet", " square units", " units", " dollars", " cents"):
        if s.lower().rstrip(".").endswith(unit):
            s = s.rstrip(".")[: -len(unit)]
    if re.fullmatch(r"[a-zA-Z]\s*=\s*.+", s):  # "x = 5" -> "5"
        s = s.split("=", 1)[1]
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".,")
    return s


def math_equiv(pred: str | None, gold: str | None, tol: float = 1e-6) -> bool:
    if pred is None or gold is None:
        return False
    p, g = normalize_math(pred), normalize_math(gold)
    if p == g:
        return True
    pf, gf = _to_float(p), _to_float(g)
    if pf is not None and gf is not None:
        return abs(pf - gf) <= tol * max(1.0, abs(gf))
    return False


_ANSWER_IS_RE = re.compile(
    r"(?:final answer|the answer)\s*(?:is|:)?\s*[:=]?\s*\$?\s*(-?[\d,]*\.?\d+(?:/\d+)?)",
    re.IGNORECASE,
)


def extract_numeric_answer(text: str) -> str | None:
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    m = re.findall(r"####\s*(-?[\d,.]+)", text)
    if m:
        return m[-1]
    m = _ANSWER_IS_RE.findall(text)
    if m:
        return m[-1]
    return extract_last_number(text)


def grade_numeric(text: str, grader: dict[str, Any]) -> GradeResult:
    tol = float(grader.get("tolerance", 1e-6))
    pred = extract_numeric_answer(text)
    gold = str(grader["answer"])
    pf, gf = _to_float(normalize_math(pred) if pred else None), _to_float(gold)
    ok = pf is not None and gf is not None and abs(pf - gf) <= tol * max(1.0, abs(gf))
    if not ok and pred is not None:
        ok = math_equiv(pred, gold, tol)
    return GradeResult(ok, float(ok), "numeric", {"pred": pred, "gold": gold})


def grade_math(text: str, grader: dict[str, Any]) -> GradeResult:
    """Competition math: boxed answer, LaTeX-normalised string or numeric match."""
    pred = extract_boxed(text)
    source = "boxed"
    if pred is None:
        pred, source = extract_last_number(text), "last_number"
    ok = math_equiv(pred, str(grader["answer"]))
    return GradeResult(ok, float(ok), "math", {"pred": pred, "gold": grader["answer"], "from": source})


# --------------------------------------------------------------------------- choice / label

_CHOICE_PATTERNS = [
    r"answer is\s*:?\s*\(?\**([A-J])\b",
    r"answer\s*:\s*\(?\**([A-J])\b",
    r"\\boxed\{\s*\(?([A-J])\)?\s*\}",
    r"^\s*\(?([A-J])\)?[.)]?\s*$",
    r"\boption\s*\(?([A-J])\b",
    r"\(([A-J])\)",
]


def extract_choice(text: str) -> str | None:
    for pat in _CHOICE_PATTERNS:
        m = re.findall(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m[-1].upper()
    return None


def grade_choice(text: str, grader: dict[str, Any]) -> GradeResult:
    pred = extract_choice(text)
    gold = str(grader["answer"]).upper()
    ok = pred == gold
    return GradeResult(ok, float(ok), "choice", {"pred": pred, "gold": gold})


def grade_label(text: str, grader: dict[str, Any]) -> GradeResult:
    """Classification: exactly one of ``labels`` must appear, and it must be ``answer``."""
    lowered = text.lower()
    labels = [lbl.lower() for lbl in grader["labels"]]
    found = [lbl for lbl in labels if re.search(rf"\b{re.escape(lbl)}\b", lowered)]
    ok = found == [str(grader["answer"]).lower()]
    return GradeResult(ok, float(ok), "label", {"found": found, "gold": grader["answer"]})


# --------------------------------------------------------------------------- json


def extract_json_value(text: str) -> Any:
    """Parse the first JSON object/array in ``text`` (fenced block or bare)."""
    candidates = re.findall(r"```(?:json)?\s*\n?(.*?)```", text, flags=re.DOTALL)
    candidates.append(text)
    decoder = json.JSONDecoder()
    for cand in candidates:
        cand = cand.strip()
        for i, ch in enumerate(cand):
            if ch in "{[":
                try:
                    value, _ = decoder.raw_decode(cand[i:])
                    return value
                except json.JSONDecodeError:
                    continue
    raise ValueError("no JSON value found")


def _norm_val(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v)).strip().lower()


def grade_json_keys(text: str, grader: dict[str, Any]) -> GradeResult:
    """JSON object with the required keys; optional ``values`` must match (case-insensitive)."""
    try:
        obj = extract_json_value(text)
    except ValueError:
        return GradeResult(False, 0.0, "json_keys", {"error": "no_json"})
    if isinstance(obj, list) and grader.get("list_ok") and obj and isinstance(obj[0], dict):
        obj = obj[0]
    if not isinstance(obj, dict):
        return GradeResult(False, 0.0, "json_keys", {"error": "not_object"})
    missing = [k for k in grader["keys"] if k not in obj]
    wrong = {
        k: obj.get(k)
        for k, v in (grader.get("values") or {}).items()
        if k in obj and _norm_val(obj[k]) != _norm_val(v)
    }
    ok = not missing and not wrong
    return GradeResult(ok, float(ok), "json_keys", {"missing": missing, "wrong_values": wrong})


# --------------------------------------------------------------------------- python tests


def extract_code(text: str) -> str:
    """Longest fenced python block, else the whole answer."""
    blocks = re.findall(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", text, flags=re.DOTALL)
    if not blocks:
        # Unterminated fence (truncated answer).
        m = re.search(r"```(?:python|py)?[ \t]*\n(.*)", text, flags=re.DOTALL)
        blocks = [m.group(1)] if m else [text]
    return max(blocks, key=len)


def _limit_resources() -> None:  # pragma: no cover - runs in the child
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
    resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
    os.setsid()


# Loaded before the candidate program: IP sockets refuse to connect. Unprivileged
# network namespaces (unshare -rn) are not available on every host, so this is the
# portable guard; it stops accidental network use, not a hostile program.
_GUARD = """
import socket as _s
def _deny(self, addr, *a, _ip=(_s.AF_INET, _s.AF_INET6), _orig=_s.socket.connect, **k):
    if self.family in _ip:
        raise OSError("network disabled in eval sandbox")
    return _orig(self, addr, *a, **k)
_s.socket.connect = _deny
_s.socket.connect_ex = _deny
del _s, _deny
"""
_RUNNER = "exec(open('guard.py').read()); import runpy; runpy.run_path('prog.py', run_name='__main__')"


def run_python_tests(program: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Run ``program`` in a temp dir, isolated interpreter, rlimits, no IP networking."""
    with tempfile.TemporaryDirectory(prefix="route_eval_") as tmp:
        with open(os.path.join(tmp, "prog.py"), "w") as f:
            f.write(program)
        with open(os.path.join(tmp, "guard.py"), "w") as f:
            f.write(_GUARD)
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "PYTHONHASHSEED": "0"}
        try:
            r = subprocess.run(
                [sys.executable, "-I", "-c", _RUNNER],
                cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout,
                preexec_fn=_limit_resources, stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        tail = (r.stderr or r.stdout)[-800:]
        return r.returncode == 0, tail


def build_program(code: str, grader: dict[str, Any]) -> str:
    entry = grader.get("entry_point")
    prefix = grader.get("prompt_prefix") or ""
    if entry and prefix and not re.search(rf"def\s+{re.escape(entry)}\s*\(", code):
        # The model returned only the function body: splice it under the given signature.
        body = code if code.startswith((" ", "\t")) else "\n".join(
            "    " + ln for ln in code.splitlines())
        code = prefix + body
    setup = grader.get("setup") or ""
    program = f"{setup}\n{code}\n\n{grader['tests']}\n"
    if entry and "def check(" in grader["tests"]:
        program += f"\ncheck({entry})\n"
    return program


def grade_python_tests(text: str, grader: dict[str, Any], timeout: float = 10.0) -> GradeResult:
    code = extract_code(text)
    ok, tail = run_python_tests(build_program(code, grader), timeout=float(grader.get("timeout", timeout)))
    return GradeResult(ok, float(ok), "python_tests", {"stderr_tail": "" if ok else tail})


# --------------------------------------------------------------------------- ifeval


def grade_ifeval(text: str, grader: dict[str, Any]) -> GradeResult:
    results = ifeval_checks.check_all(
        text, grader["instruction_id_list"], grader["kwargs"], prompt=grader.get("prompt", ""))
    ok = all(results.values())
    return GradeResult(ok, float(ok), "ifeval_loose", {"per_instruction": results})


# --------------------------------------------------------------------------- dispatch

DETERMINISTIC = {
    "numeric": grade_numeric,
    "math": grade_math,
    "choice": grade_choice,
    "label": grade_label,
    "json_keys": grade_json_keys,
    "python_tests": grade_python_tests,
    "ifeval": grade_ifeval,
}


def grade_deterministic(raw_text: str, grader: dict[str, Any]) -> GradeResult:
    fn = DETERMINISTIC[grader["type"]]
    return fn(strip_think(raw_text), grader)
