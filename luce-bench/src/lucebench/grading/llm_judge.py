"""Claude Sonnet quality judge for the multi-turn agent_recorded area.

Compares a candidate model's response to Claude's actual response from
the same point in the original session. Returns a 3-bin verdict:

* ``suitable_match`` — the candidate's response is equivalent in
  quality and intent to the reference. Maps to ``pass``.
* ``divergent_but_valid`` — the candidate took a different but
  reasonable approach to the same task. Maps to ``pass``.
* ``off_track`` — the candidate is irrelevant, refused, or
  fundamentally wrong. Maps to ``fail``.

Why this judge instead of substring matching
============================================

The single-turn ``agent_recorded_v1`` area uses substring matching
against Claude's actual tool list (e.g. ``Edit``, ``Bash``). That biases
against models that solve the task with a different but valid tool —
e.g. a model that proposes ``write_file`` instead of ``Edit`` is graded
``fail`` even when the proposed approach is sound. The judge is
specifically allowed to mark ``divergent_but_valid`` so that valid
non-Claude-shaped approaches pass.

API cost & caching
==================

Every judge invocation costs an Anthropic Sonnet call. Results are
cached on disk under ``~/.cache/luce-bench/judge/`` keyed by
``case_id + sha256(prompt + reference + candidate)``, so a re-run with
identical inputs hits the cache and bills $0. Cache hits return
``{"cached": true, ...}`` for visibility.

For the default fixture (48 cases × cold+warm = 96 judge calls per
full run): with Sonnet 4.6 list pricing (~$3/Mtok input,
$15/Mtok output) and judge prompts averaging ~5K input + ~150 output
tokens, a full cold judge pass is roughly $0.30 -- $1.50 depending on
case sizes. Cached re-runs are free.

Environment requirement
=======================

``ANTHROPIC_API_KEY`` must be set. If absent the judge raises
``JudgeUnavailable`` so callers can surface a crisp error pointing at
the env var rather than a generic SDK auth failure.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

# The judge model id. Pinned to the latest Sonnet at the time of writing
# (claude-sonnet-4-6); bump in lockstep with the Anthropic SDK roll. A
# constant rather than env-driven by default so the grader version is a
# stable function of the source tree.
JUDGE_MODEL = "claude-sonnet-4-6"

# The cache root. Lives outside the repo so re-running the bench across
# different worktrees / branches still hits the same cache when the
# inputs are byte-identical.
CACHE_ROOT = Path(os.path.expanduser("~/.cache/luce-bench/judge"))

# Verdict-to-pass mapping. The judge prompt explicitly lists these three
# bins; the mapping below is the only place that turns them into the
# binary pass/fail the area surfaces.
_PASS_VERDICTS = frozenset({"suitable_match", "divergent_but_valid"})
_FAIL_VERDICTS = frozenset({"off_track"})
_VALID_VERDICTS = _PASS_VERDICTS | _FAIL_VERDICTS


class JudgeUnavailable(RuntimeError):
    """Raised when the judge cannot be invoked at all (missing API key,
    missing SDK, etc.). Distinguishable from a per-call failure so the
    area can short-circuit instead of looping through every case."""


@dataclasses.dataclass
class JudgeVerdict:
    """One judge result. ``passed`` is the bin -> pass/fail mapping. The
    raw ``verdict`` is preserved so callers can grade further (e.g. count
    suitable vs divergent separately)."""

    passed: bool
    verdict: str
    rationale: str
    cached: bool = False
    model: str = JUDGE_MODEL
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass": self.passed,
            "verdict": self.verdict,
            "rationale": self.rationale,
            "cached": self.cached,
            "model": self.model,
            "error": self.error,
        }


_JUDGE_SYSTEM_PROMPT = """You are an evaluation judge for agentic-coding model outputs.

You will be given:
  1. The full conversation so far (a user-assistant transcript where the model is acting as a coding agent).
  2. The REFERENCE response that Claude (the original session driver) produced at this point.
  3. The CANDIDATE response from the model under test.

Your job: rate the CANDIDATE relative to the REFERENCE on a 3-bin scale.

Bins:
  - "suitable_match": the candidate is functionally equivalent to the reference. It addresses the same task with comparable quality. Specific tool calls or wording may differ, but the work proposed/executed is the same scope and the substance is right.
  - "divergent_but_valid": the candidate takes a *different* approach than the reference, but the approach is reasonable and would plausibly solve the user's task. Examples: proposing a different sequence of tools, asking a clarifying question instead of acting, suggesting a different file structure. The candidate is engaged and competent, just not Claude-shaped.
  - "off_track": the candidate is irrelevant, refuses the task without justification, hallucinates context that wasn't in the conversation, or is so short / incoherent it doesn't constitute an engagement. Refusals on legitimate non-harmful coding tasks are off_track.

Output STRICT JSON only, in this shape, with no surrounding prose:

{"verdict": "<bin>", "rationale": "<one-to-three-sentence explanation>"}

Be calibrated. Do not penalize the candidate for failing to literally use Claude's tool names — verb-level equivalence is enough.
Do not pass the candidate just because it produced fluent prose — it must engage with the task."""


def _stable_hash(prompt_text: str, reference: str, candidate: str) -> str:
    """SHA256 of the three inputs; first 16 hex chars used as cache key.

    Truncation to 16 chars (64 bits) is fine — the cache is local-only
    and a collision means the wrong cached verdict comes back, which
    we'd notice the first time we ran a regression test.
    """
    h = hashlib.sha256()
    h.update(prompt_text.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(reference.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(candidate.encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


def _safe_case_id(case_id: str) -> str:
    """Filesystem-safe slug of the case id for the cache filename."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", case_id)[:120]


def _cache_path(case_id: str, key: str) -> Path:
    return CACHE_ROOT / f"{_safe_case_id(case_id)}__{key}.json"


def _format_conversation(messages: list[dict[str, Any]]) -> str:
    """Render the multi-turn conversation as a single string for the judge.

    Keeps it compact — the judge sees the full history but doesn't need
    nested JSON; flat ``role: ...`` blocks is enough context for it to
    grade the candidate's response in context. Long tool-result blobs
    in the history would otherwise dominate the judge prompt; we
    truncate per-turn at 4000 chars so the judge prompt stays bounded.
    """
    lines: list[str] = []
    for m in messages:
        role = (m.get("role") or "?").upper()
        content = m.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if len(content) > 4000:
            content = content[:4000] + f"\n... [truncated {len(content) - 4000} chars]"
        lines.append(f"[{role}]\n{content}")
    return "\n\n".join(lines)


def _build_judge_user_message(
    *,
    messages: list[dict[str, Any]],
    reference_response: str,
    candidate_response: str,
) -> str:
    """Assemble the user-message sent to the judge model."""
    convo = _format_conversation(messages)
    # Both responses are wrapped in fence-like markers so the judge can
    # tell where they end even if they contain the word "REFERENCE".
    return (
        "Here is the conversation so far. The next assistant response is "
        "what we are grading.\n\n"
        "===== CONVERSATION =====\n"
        f"{convo}\n"
        "===== END CONVERSATION =====\n\n"
        "===== REFERENCE RESPONSE (what Claude produced) =====\n"
        f"{reference_response}\n"
        "===== END REFERENCE =====\n\n"
        "===== CANDIDATE RESPONSE (the model under test) =====\n"
        f"{candidate_response}\n"
        "===== END CANDIDATE =====\n\n"
        "Output the verdict JSON now."
    )


def _parse_judge_output(text: str) -> tuple[str, str]:
    """Extract ``(verdict, rationale)`` from the judge's raw text.

    The judge is told to output strict JSON. In practice models
    sometimes wrap the JSON in a markdown fence or prefix it with a
    line of prose. We strip both, then JSON-parse. On any failure we
    map the verdict to ``off_track`` so a malformed judge response
    biases against the candidate (safer than letting a junk response
    silently pass).
    """
    cleaned = text.strip()
    # Strip Markdown code fence if present.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    # Find the first { … } object — tolerate leading prose.
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m is None:
        return ("off_track", f"judge returned non-JSON: {text[:200]!r}")
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return ("off_track", f"judge returned malformed JSON: {text[:200]!r}")
    verdict = obj.get("verdict")
    rationale = obj.get("rationale") or ""
    if verdict not in _VALID_VERDICTS:
        return ("off_track", f"unrecognized verdict {verdict!r}; rationale: {rationale}")
    return (verdict, rationale)


def _load_cached(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
    except OSError:
        # Cache write failure is non-fatal — the judge result is still
        # valid, we just won't have it next time.
        pass


def judge_response(
    *,
    case_id: str,
    messages: list[dict[str, Any]],
    reference_response: str,
    candidate_response: str,
    model: str = JUDGE_MODEL,
    cache_root: Path = CACHE_ROOT,
    _client_factory: Any = None,
) -> JudgeVerdict:
    """Grade a candidate response against the reference using Claude Sonnet.

    ``_client_factory`` is an opt-in seam for tests — when set, the judge
    uses ``_client_factory()`` instead of constructing a real
    ``anthropic.Anthropic()`` client. Production callers should never
    pass this argument; it exists so tests can verify the caching and
    parsing logic without hitting the API.

    Cache miss: synchronously calls the Anthropic API and writes the
    result to disk. Cache hit: returns the cached verdict with
    ``cached=True``.

    Raises ``JudgeUnavailable`` if the API key is missing AND the test
    seam wasn't used. Per-call API errors are caught and returned as a
    JudgeVerdict with ``passed=False, verdict="off_track", error=...``.
    """
    user_msg = _build_judge_user_message(
        messages=messages,
        reference_response=reference_response,
        candidate_response=candidate_response,
    )
    key = _stable_hash(user_msg, reference_response, candidate_response)
    path = cache_root / f"{_safe_case_id(case_id)}__{key}.json"
    cached = _load_cached(path)
    if cached is not None:
        return JudgeVerdict(
            passed=bool(cached.get("pass")),
            verdict=str(cached.get("verdict") or "off_track"),
            rationale=str(cached.get("rationale") or ""),
            cached=True,
            model=str(cached.get("model") or model),
            error=cached.get("error"),
        )

    # No cache. We need to call the API. Allow the test seam first.
    if _client_factory is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise JudgeUnavailable(
                "ANTHROPIC_API_KEY is not set. The agent_recorded judge "
                "requires Anthropic credentials. Export ANTHROPIC_API_KEY "
                "before running this area."
            )
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as e:
            raise JudgeUnavailable(
                "anthropic SDK is not importable. `pip install anthropic`."
            ) from e
        client = anthropic.Anthropic()
    else:
        client = _client_factory()

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=_JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        # The SDK returns content as a list of blocks; the judge only
        # speaks text so we grab the first text block.
        raw_text = ""
        for block in getattr(response, "content", None) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                raw_text += text
        verdict, rationale = _parse_judge_output(raw_text)
        passed = verdict in _PASS_VERDICTS
        result = JudgeVerdict(
            passed=passed,
            verdict=verdict,
            rationale=rationale,
            cached=False,
            model=model,
            error=None,
        )
    except Exception as e:  # noqa: BLE001 - we surface the error string
        result = JudgeVerdict(
            passed=False,
            verdict="off_track",
            rationale=f"judge invocation failed: {type(e).__name__}: {e}",
            cached=False,
            model=model,
            error=f"{type(e).__name__}: {e}",
        )

    _write_cache(path, result.to_dict())
    return result
