"""HumanEval-style code-completion cases + grader for `--area code`.

Mirrors the conventions of `bench_ds4_eval.py`: data lives in a JSON
fixture, this module exposes a CASES list + a grader. `bench_http_capability.py`
dispatches into this module when `--area code` is selected so HumanEval
runs flow through the same harness as ds4-eval (provider/server_info
capture, --parallel, --host-label, unified row schema).

Grading is intentionally lightweight: a completion PASSes if the joined
prompt + completion parses as syntactically valid Python. This catches
the obvious "model regressed to noise" failure mode without requiring a
sandboxed test runner. Real HumanEval pass@1 (executing unit tests)
remains a follow-up; the existing CLI `bench_he.py` runs that flow
under test_dflash and is not part of this --area path.

The 10-prompt set is identical to `bench_he_http.py`'s PROMPTS (the
canonical HumanEval-style mid-function completion set that lucebox
autotune uses). Vendored as JSON so the fixture stays language-agnostic.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
FIXTURE_PATH = SCRIPT_DIR / "fixtures" / "humaneval" / "cases.json"


def load_humaneval_cases(path: Path = FIXTURE_PATH) -> list[dict[str, Any]]:
    """Load the vendored HumanEval-style case set.

    Each case is shaped to match the bench_http_capability run loop:
    area / source / id / kind / prompt / answer (None — we don't grade
    against a reference solution, just against "parses").
    """
    payload = json.loads(path.read_text())
    out = []
    for raw in payload["cases"]:
        out.append({
            "area": "code",
            "source": "HumanEval-port",
            "id": raw["id"],
            "kind": raw.get("kind", "code-completion"),
            "prompt": raw["prompt"],
            # No reference answer — grading is by-parse, see grade_completion.
            "answer": None,
            "domain": "code",
            "title": raw["id"],
        })
    return out


HE_CASES = load_humaneval_cases()


def grade_completion(prompt: str, completion: str) -> dict[str, Any]:
    """Decide whether ``completion`` is a coherent continuation of ``prompt``.

    Grading rules (decode-only, no execution):

    * **parse_pass**: ``prompt + completion`` parses as valid Python
      via ``ast.parse``. This is the bench's headline PASS signal.
    * **nonempty**: completion stripped is at least 8 chars long
      (filters out empty / whitespace-only / single-token responses).
    * **mentions_return_or_yield**: heuristic for "the model produced
      real function body content" rather than just continuing
      whitespace. Recorded but not used for pass/fail.

    The grader is intentionally permissive: HumanEval pass@1 (executing
    the function against test cases) requires a sandbox and isn't worth
    the complexity for an in-process bench. The "parses cleanly" signal
    is enough to catch the failure modes we actually care about: model
    regressed to noise / model produced markdown instead of code / model
    timed out / model spat out a tool-call envelope by mistake.
    """
    stripped = (completion or "").strip()
    nonempty = len(stripped) >= 8
    # Chat-template renders sometimes strip leading whitespace from the
    # model's response, so a prompt ending with ``for`` and a completion
    # starting with ``i in range(...)`` would re-join as ``fori in
    # range`` and fail to parse. Try a few naive separators before
    # declaring it broken — we're only trying to detect "the model
    # produced obvious noise", not strict whitespace fidelity.
    parse_pass = False
    for sep in ("", " ", "\n"):
        try:
            ast.parse(prompt + sep + completion)
            parse_pass = True
            break
        except (SyntaxError, ValueError):
            continue
    return {
        "graded_pass": parse_pass and nonempty,
        "strict_pass": parse_pass and nonempty,
        "format_pass": nonempty,
        "semantic_pass": parse_pass and nonempty,
        "semantic_hint": "return" in stripped or "yield" in stripped,
        "status": "passed" if (parse_pass and nonempty) else "failed",
        "ok": parse_pass and nonempty,
    }
