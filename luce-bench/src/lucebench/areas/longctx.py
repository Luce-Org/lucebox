"""Long-context frontier cases for `--area longctx`.

Mirrors the convention of `bench_ds4_eval.py` and `bench_humaneval.py`:
data lives here, the dispatcher in `bench_http_capability.py` is thin.

The case set is ported from `bench_http_frontiers.py`'s frontier probe —
deterministic prompts sized to hit a target token frontier (2k → 64k),
ending in a fixed instruction the grader checks for. Each frontier is a
single case so the regular bench harness can record per-case timings,
provider, server_info, etc. through the normal row schema.

Grading is single-line:

* **format_pass**: response starts with ``Risk:`` (the instruction asks
  for "exactly one sentence beginning with 'Risk:'").

We don't grade meaningful "risk" content because the prompt's haystack
is deliberately generic — what we're measuring is whether the model
follows the instruction-after-long-context pattern at all, not whether
the risk analysis is correct.

The standalone `bench_http_frontiers.py` still exists for lucebox's
autotune flow (which expects its specific CSV output format); this
module is the unified-harness entry. Phase 2 of the integration will
re-point autotune at the unified harness and we can delete the
standalone then.
"""

from __future__ import annotations

import re
from typing import Any

# See lucebench.areas.ds4_eval.GRADER_VERSION for the bump policy.
GRADER_VERSION = 1

# Reuse the corpus blocks from the upstream frontier probe so prompts
# remain bit-identical when comparing autotune-shelled output against
# the unified-harness output.
_CORPUS_BLOCKS = [
    "You are auditing a repository for a local inference server. "
    "Track API compatibility, tool-call behavior, startup configuration, "
    "benchmark fidelity, and Docker reproducibility.\n",
    "File: lucebox/lucebox/smoke.py\n"
    "The smoke check must prove /props is populated, text streams, and tools "
    "are emitted in OpenAI format.\n",
    "File: dflash/scripts/server.py\n"
    "The server renders Qwen chat templates, streams SSE deltas, parses XML "
    "tool calls, and reports runtime properties.\n",
    "Review note: preserve patch isolation so /props, Docker startup, uv "
    "bootstrap, and benchmark harness changes can be split later.\n",
]

# Frontiers we sweep. 2k → 32k covers the common HTTP-client workload
# range; 64k probes the long-context regime that exposes spec-decode
# acceptance + KV-quant memory pressure on bragi-class 24 GB cards.
FRONTIER_TARGETS = [2048, 4096, 8192, 16384, 32768, 65536]
CHARS_PER_TOKEN = 4
_INSTRUCTION = (
    "Final instruction: write exactly one sentence beginning with 'Risk:' "
    "that summarizes the highest-risk reliability issue."
)


def _make_prompt(target_tokens: int) -> str:
    """Generate a deterministic prompt sized to approximate ``target_tokens``.

    Uses the rough ``chars / 4`` token estimate (matching the upstream
    frontier probe). The actual prompt_tokens reported by the server is
    captured in the bench row, so the harness can correct any drift.
    """
    target_chars = max(256, target_tokens * CHARS_PER_TOKEN)
    pieces: list[str] = []
    i = 0
    while sum(len(p) for p in pieces) < target_chars:
        pieces.append(f"[chunk {i:05d}] {_CORPUS_BLOCKS[i % len(_CORPUS_BLOCKS)]}")
        i += 1
    body = "".join(pieces)[:target_chars]
    return (
        "Use the following repository context to answer the final instruction. "
        "Do not call tools for this benchmark.\n\n"
        f"{body}\n\n" + _INSTRUCTION
    )


def _build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for target in FRONTIER_TARGETS:
        # Pretty-print the size in the case id (4k / 8k / 16k …) for grep-ability.
        if target >= 1024:
            label = f"{target // 1024}k"
        else:
            label = str(target)
        cases.append(
            {
                "area": "longctx",
                "source": "long-context-frontier",
                "id": f"frontier-{label}",
                "kind": "longctx-frontier",
                "prompt": _make_prompt(target),
                "answer": None,
                "domain": "longctx",
                "title": f"long-context frontier {label} tokens",
                "target_tokens": target,
            }
        )
    return cases


LONGCTX_CASES = _build_cases()

_RISK_PREFIX = re.compile(r"^\s*Risk\s*:\s*\S", re.IGNORECASE)


def grade_longctx(prompt: str, completion: str) -> dict[str, Any]:
    """Pass if the model's visible reply begins with ``Risk:`` (its instruction)."""
    text = (completion or "").lstrip()
    starts_with_risk = bool(_RISK_PREFIX.match(text))
    nonempty = len(text.strip()) >= 8
    return {
        "graded_pass": starts_with_risk and nonempty,
        "strict_pass": starts_with_risk and nonempty,
        "format_pass": starts_with_risk,
        "semantic_pass": starts_with_risk and nonempty,
        "semantic_hint": "risk" in text.lower(),
        "status": "passed" if (starts_with_risk and nonempty) else "failed",
        "ok": starts_with_risk and nonempty,
    }


def grade_longctx_case(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Wrap grade_longctx to match the lucebench.cli runner shape."""
    completion = row.get("content") or ""
    g = grade_longctx(case["prompt"], completion)
    return {
        "pass": g["graded_pass"],
        "given": "risk_prefix_ok" if g["format_pass"] else "risk_prefix_missing",
        "correct": "starts-with-Risk:",
        "status": g["status"],
        "format_pass": g["format_pass"],
        "semantic_hint": False,
    }
