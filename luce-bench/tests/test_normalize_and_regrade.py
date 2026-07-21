"""Tests for the canonical schema, normalizer, and ``luce-bench regrade`` CLI.

Each test is the smallest fixture that exercises one specific contract
the user has been bitten by:

* legacy id/output/ok rows must normalize cleanly into the canonical
  case_id/content/graded shape (no silent field drops);
* current 0.2.7 case_id/content rows must normalize without
  reinterpretation (a roundtrip is a fixed point);
* pass_rate unit auto-detection: 0.576 stays 0.576, 77.17 becomes 0.7717;
* a re-grade at the same GRADER_VERSION reproduces the source rates
  exactly (Δ = 0pp);
* bumping GRADER_VERSION flags the rows as belonging to a different
  comparison group;
* mixed grader_versions in a single regrade markdown surface the
  mismatch banner, not a fake-comparable row;
* semantic_hint=True alongside strict_pass=False does NOT inflate the
  headline strict_pass_rate.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from lucebench import schema
from lucebench.areas import ds4_eval
from lucebench.normalize import _normalize_pass_rate, normalize_result
from lucebench.regrade import _format_markdown, regrade_result
from lucebench.schema import CanonicalResult


def _fake_choice_row_legacy(*, case_id: str, output: str, ok: bool) -> dict[str, Any]:
    """Build a row in the pre-0.2.5 result.json shape."""
    return {
        "id": case_id,
        "source": "GPQA Diamond",
        "kind": "choice",
        "prompt": "...",
        "output": output,
        "reasoning_content": "",
        "ok": ok,
        "graded_pass": ok,
        "strict_pass": ok,
        "format_pass": True,
        "semantic_hint": ok,
        "semantic_pass": False,
        "given": "B" if ok else "A",
        "correct": ["B"],
        "status": "passed" if ok else "failed",
        "wall_s": 1.5,
        "completion_tokens": 200,
        "thinking_tokens": 50,
        "http_status": 200,
    }


def _fake_choice_row_current(*, case_id: str, content: str, passed: bool) -> dict[str, Any]:
    """Build a row in the 0.2.5+ result.json shape."""
    return {
        "case_id": case_id,
        "source": "GPQA Diamond",
        "kind": "choice",
        "content": content,
        "reasoning_content": "",
        "pass": passed,
        "graded": {
            "pass": passed,
            "given": "B" if passed else "A",
            "correct": "B",
            "status": "passed" if passed else "failed",
            "format_pass": True,
            "semantic_hint": passed,
        },
        "wall_seconds": 1.5,
        "completion_tokens": 200,
        "reasoning_tokens": 50,
        "http_status": 200,
    }


def _real_ds4_case_id(kind: str = "choice") -> str:
    """Pick a known ds4 case id of the requested kind from the fixture.

    Used so re-grade can find the case in DS4_EVAL_CASES by id. Tests
    don't need to fabricate matching content; they just need a row that
    references a real case so grade_case has something to compare to.
    """
    for c in ds4_eval.DS4_EVAL_CASES:
        if c.get("kind") == kind:
            return str(c["id"])
    raise RuntimeError(f"no ds4 case of kind={kind!r} in fixture")


# ──────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────


def test_schema_version_is_pinned():
    """SCHEMA_VERSION is an int and surfaces in CanonicalResult defaults.

    Bumping the constant must show up in default-constructed results so
    a downstream consumer can refuse a newer-than-known schema.
    """
    assert isinstance(schema.SCHEMA_VERSION, int)
    assert CanonicalResult().schema_version == schema.SCHEMA_VERSION


def test_schema_version_is_v2_or_higher():
    """The host-block landed in v2 — the floor moved off v1."""
    assert schema.SCHEMA_VERSION >= 2


def test_host_info_dataclass_defaults():
    """``HostInfo()`` defaults: every field optional, ``gpus`` is an empty list."""
    from lucebench.schema import GpuInfo, HostInfo

    h = HostInfo()
    assert h.os_pretty is None
    assert h.kernel is None
    assert h.wsl_version is None
    assert h.docker_version is None
    assert h.nvidia_driver is None
    assert h.nvidia_ctk_version is None
    assert h.cpu_model is None
    assert h.nproc is None
    assert h.ram_gb is None
    assert h.gpus == []
    assert h.cuda_visible_devices is None
    assert h.source is None
    assert h.collector is None
    assert h.collected_at is None
    # GpuInfo defaults mirror.
    g = GpuInfo()
    assert g.index is None
    assert g.name is None
    assert g.vram_gb is None


def test_canonical_result_with_host_none_loads_cleanly():
    """A pre-v2 dict (no host field) and an explicit host=None both load."""
    canon = schema.from_dict({"area": "ds4-eval", "rows": []})
    assert canon.host is None
    canon2 = schema.from_dict({"area": "ds4-eval", "host": None, "rows": []})
    assert canon2.host is None


def test_each_area_module_exports_grader_version():
    """Every area the regrade CLI knows about must declare GRADER_VERSION.

    Adding a new area without bumping/declaring this constant breaks the
    comparability guard rail.
    """
    from lucebench.areas import (
        agent,
        agent_recorded,
        gsm8k,
        hellaswag,
        humaneval,
        longctx,
        smoke,
        truthfulqa_mc1,
    )

    for mod in (
        ds4_eval,
        smoke,
        gsm8k,
        hellaswag,
        truthfulqa_mc1,
        humaneval,
        longctx,
        agent,
        agent_recorded,
    ):
        assert isinstance(getattr(mod, "GRADER_VERSION", None), int)


# ──────────────────────────────────────────────────────────────────────
# Normalizer
# ──────────────────────────────────────────────────────────────────────


def test_legacy_schema_normalizes():
    """Legacy id/output/ok/graded_pass rows fold into CanonicalRow.

    Specifically: case_id is picked up from `id`, content from `output`,
    wall_seconds from `wall_s`, reasoning_tokens from `thinking_tokens`,
    and graded.strict_pass survives from strict_pass.
    """
    raw = {
        "area": "ds4-eval",
        "suite": "ds4-eval",
        "lucebench_version": "0.2.4",
        "thinking_enabled": False,
        "pass_rate": 0.5,
        "strict_pass_rate": 0.5,
        "rows": [
            _fake_choice_row_legacy(case_id="A", output="Answer: B", ok=True),
            _fake_choice_row_legacy(case_id="B", output="Answer: A", ok=False),
        ],
    }
    canon = normalize_result(raw)
    assert canon.area == "ds4-eval"
    assert canon.n == 2
    assert canon.rows[0].case_id == "A"
    assert canon.rows[0].content == "Answer: B"
    assert canon.rows[0].wall_seconds == 1.5
    assert canon.rows[0].reasoning_tokens == 50
    assert canon.rows[0].graded["pass"] is True
    assert canon.rows[0].graded["strict_pass"] is True
    assert canon.rows[1].graded["pass"] is False
    assert canon.strict_pass_rate == pytest.approx(0.5)


def test_current_schema_normalizes():
    """0.2.7 case_id/content/graded rows pass through to CanonicalRow.

    No remapping should happen — the canonical shape IS the current
    shape, modulo the rates being recomputed straight from the rows
    rather than read from the writer's aggregates.
    """
    raw = {
        "area": "ds4-eval",
        "lucebench_version": "0.2.7",
        "think": True,
        "pass_rate": 50.0,  # percent in current shape
        "rows": [
            _fake_choice_row_current(case_id="X", content="Answer: B", passed=True),
            _fake_choice_row_current(case_id="Y", content="Answer: A", passed=False),
        ],
    }
    canon = normalize_result(raw)
    assert canon.area == "ds4-eval"
    assert canon.mode == "think"
    assert canon.rows[0].case_id == "X"
    assert canon.rows[0].content == "Answer: B"
    assert canon.strict_pass_rate == pytest.approx(0.5)
    # The unit-detection heuristic tags the normalisation in metrics.
    assert canon.metrics["pass_rate_unit"] == "normalized_from_legacy_percent"
    assert canon.metrics["declared_strict_pass_rate"] == pytest.approx(0.5)


def test_pass_rate_unit_detection():
    """Range-based heuristic: ≤1 → fraction; (1,100] → percent → /100."""
    assert _normalize_pass_rate(0.5761) == (pytest.approx(0.5761), "fraction")
    assert _normalize_pass_rate(0.0) == (0.0, "fraction")
    assert _normalize_pass_rate(1.0) == (1.0, "fraction")
    frac, tag = _normalize_pass_rate(77.17)
    assert frac == pytest.approx(0.7717)
    assert tag == "normalized_from_legacy_percent"
    assert _normalize_pass_rate(None) == (0.0, "missing")
    assert _normalize_pass_rate(-1) == (0.0, "missing")
    # > 100 is almost certainly a bug; we clamp at 1.0 and tag it.
    assert _normalize_pass_rate(150.0)[1] == "clamped"


def test_normalizer_drops_dead_semantic_pass_rate():
    """Top-level semantic_passed / semantic_pass_rate must not survive load.

    Legacy files carry these but they're always 0.0 — leaving them in
    has burned the user multiple times. A REAL judge under
    metrics["semantic_judge"][judge_id] is preserved.
    """
    raw = {
        "area": "ds4-eval",
        "rows": [_fake_choice_row_legacy(case_id="A", output="Answer: B", ok=True)],
        "semantic_pass_rate": 0.0,
        "semantic_passed": 0,
    }
    canon = normalize_result(raw)
    assert "semantic_pass_rate" not in canon.metrics
    assert "semantic_passed" not in canon.metrics

    # Now with a real judge wired in: keep it.
    raw_with_judge = {
        "area": "ds4-eval",
        "rows": [_fake_choice_row_legacy(case_id="A", output="Answer: B", ok=True)],
        "metrics": {"semantic_judge": {"gpt5-grader": {"pass_rate": 0.8}}},
        "semantic_pass_rate": 0.0,
    }
    canon2 = normalize_result(raw_with_judge)
    assert canon2.metrics["semantic_judge"]["gpt5-grader"]["pass_rate"] == 0.8


def test_thinking_control_inferred_from_rows_when_no_flag():
    """nothink + any row with reasoning_tokens>0 → honored=False.

    Compliant rows (reasoning_tokens=0) do NOT count as contradicting;
    only rows that actually spent reasoning budget while the run was in
    nothink mode do.
    """
    raw = {
        "area": "ds4-eval",
        "think": False,
        "rows": [
            {
                **_fake_choice_row_current(case_id="A", content="X", passed=True),
                "reasoning_tokens": 0,
            },
            {
                **_fake_choice_row_current(case_id="B", content="Y", passed=False),
                "reasoning_tokens": 200,
            },
        ],
    }
    canon = normalize_result(raw)
    assert canon.mode == "nothink"
    assert canon.thinking_control_honored is False
    assert canon.contradicting_rows == 1


def test_normalize_picks_up_explicit_host_block():
    """A result file with a top-level ``host`` block surfaces it on CanonicalResult."""
    raw = {
        "area": "ds4-eval",
        "host": {
            "os_pretty": "Ubuntu 22.04.3 LTS",
            "wsl_version": "wsl2",
            "gpus": [
                {
                    "index": 0,
                    "name": "NVIDIA RTX 5090",
                    "sm": "12.0",
                    "vram_gb": 24,
                    "power_limit_w": 175,
                }
            ],
            "source": "props",
            "collected_at": "2026-05-28T20:31:42Z",
        },
        "rows": [_fake_choice_row_current(case_id="A", content="B", passed=True)],
    }
    canon = normalize_result(raw)
    assert canon.host is not None
    assert canon.host.source == "props"
    assert canon.host.wsl_version == "wsl2"
    assert len(canon.host.gpus) == 1
    assert canon.host.gpus[0].name == "NVIDIA RTX 5090"


def test_normalize_historical_result_gets_unknown_host():
    """Pre-v2 result files (no ``host`` block) load with ``HostInfo(source="unknown")``."""
    raw = {
        "area": "ds4-eval",
        "rows": [_fake_choice_row_legacy(case_id="A", output="Answer: B", ok=True)],
    }
    canon = normalize_result(raw)
    assert canon.host is not None
    assert canon.host.source == "unknown"
    assert canon.host.os_pretty is None
    assert canon.host.gpus == []


def test_normalize_extracts_host_from_props_helper():
    """``extract_host_from_props`` lifts ``/props.host`` into a HostInfo."""
    from lucebench.normalize import extract_host_from_props

    props = {
        "host": {
            "os_pretty": "Ubuntu 22.04.3 LTS",
            "kernel": "6.6.87.2-microsoft-standard-WSL2",
            "wsl_version": "wsl2",
            "nvidia_driver": "596.36",
            "nvidia_ctk_version": "1.16.2",
            "gpus": [{"index": 0, "name": "NVIDIA RTX 5090", "vram_gb": 24}],
            "source": "lucebox.sh",
        }
    }
    h = extract_host_from_props(props)
    assert h is not None
    assert h.wsl_version == "wsl2"
    assert h.nvidia_ctk_version == "1.16.2"
    assert h.gpus[0].vram_gb == 24
    # Source is relabeled to "props" — the upstream "lucebox.sh" label
    # describes how the SERVER obtained the data and lives in collector.
    assert h.source == "props"
    assert h.collector == "lucebox.sh"
    # None / missing host → None.
    assert extract_host_from_props(None) is None
    assert extract_host_from_props({}) is None
    assert extract_host_from_props({"host": None}) is None


def test_semantic_hint_does_not_contribute_to_strict_pass():
    """A semantic_hint=True row with strict_pass=False stays a fail.

    Defensive against any future refactor that conflates the diagnostic
    hint into the headline strict_pass_rate.
    """
    raw = {
        "area": "ds4-eval",
        "rows": [
            {
                **_fake_choice_row_current(case_id="A", content="A", passed=False),
                "graded": {
                    "pass": False,
                    "given": "A",
                    "correct": "B",
                    "status": "failed",
                    "format_pass": True,
                    "semantic_hint": True,
                },
            },
        ],
    }
    canon = normalize_result(raw)
    assert canon.strict_pass_rate == 0.0
    assert canon.semantic_hint_rate == 1.0


# ──────────────────────────────────────────────────────────────────────
# Regrade
# ──────────────────────────────────────────────────────────────────────


def test_regrade_matches_when_grader_version_same(monkeypatch):
    """Re-grading a synthetic ds4-eval row at the same GRADER_VERSION
    reproduces the headline rate exactly (Δ = 0pp).
    """
    case_id = _real_ds4_case_id("choice")
    # Pick the actual canonical answer so the row is a strict pass.
    case = next(c for c in ds4_eval.DS4_EVAL_CASES if c["id"] == case_id)
    expected = case["answer"]
    raw = {
        "area": "ds4-eval",
        "rows": [
            {
                "case_id": case_id,
                "source": case["source"],
                "kind": "choice",
                "content": f"Some reasoning... Answer: {expected}",
                "reasoning_content": "",
                "graded": {
                    "pass": True,
                    "given": expected,
                    "correct": expected,
                    "format_pass": True,
                    "semantic_hint": True,
                    "status": "passed",
                },
                "wall_seconds": 1.0,
                "completion_tokens": 10,
            }
        ],
    }
    canon = normalize_result(raw)
    regraded = regrade_result(canon)
    assert regraded.grader_version == f"ds4-eval={ds4_eval.GRADER_VERSION}"
    assert regraded.strict_pass_rate == canon.strict_pass_rate == 1.0
    assert regraded.metrics["regrade_status"] == "ok"


def test_regrade_differs_when_grader_version_bumped(monkeypatch):
    """A simulated future grader change (different answer) makes the
    re-grade flag the run as not-comparable to its source.

    We monkeypatch the grader to "always fail" and confirm the re-graded
    rate flips to 0.0 even when the source rate was 1.0 — i.e. the
    re-grade actually re-ran the grader rather than trusting the row's
    cached `graded` block.
    """
    case_id = _real_ds4_case_id("choice")
    case = next(c for c in ds4_eval.DS4_EVAL_CASES if c["id"] == case_id)
    expected = case["answer"]

    raw = {
        "area": "ds4-eval",
        "rows": [
            {
                "case_id": case_id,
                "source": case["source"],
                "kind": "choice",
                "content": f"Answer: {expected}",
                "reasoning_content": "",
                "graded": {
                    "pass": True,
                    "given": expected,
                    "correct": expected,
                    "format_pass": True,
                    "semantic_hint": True,
                    "status": "passed",
                },
                "wall_seconds": 1.0,
                "completion_tokens": 5,
            }
        ],
    }
    canon = normalize_result(raw)
    assert canon.strict_pass_rate == 1.0

    # Simulate a future grader by patching the area's grade_case and
    # bumping its GRADER_VERSION. The regrade CLI rebuilds the area map
    # at import; we have to also patch the cached _AREAS entry.
    from lucebench import regrade as regrade_mod

    def always_fail(case, row):
        return {
            "pass": False,
            "given": "?",
            "correct": str(case.get("answer")),
            "status": "failed",
            "format_pass": False,
            "semantic_hint": False,
        }

    monkeypatch.setattr(ds4_eval, "GRADER_VERSION", 99)
    # Re-point the _AREAS entry so the regrade picks up our shim.
    new_entry = (ds4_eval, always_fail, regrade_mod._AREAS["ds4-eval"][2])
    monkeypatch.setitem(regrade_mod._AREAS, "ds4-eval", new_entry)

    regraded = regrade_mod.regrade_result(canon)
    assert regraded.strict_pass_rate == 0.0
    assert regraded.grader_version == "ds4-eval=99"
    # Δ vs declared (1.0) is the drift signal — surfaces in markdown.
    declared = regraded.metrics["declared_strict_pass_rate"]
    assert declared == 1.0


def test_report_refuses_mixed_grader_versions(tmp_path):
    """When two regraded results carry different grader_versions, the
    markdown must surface the mismatch banner and place them in
    separate per-version tables — never a single comparison row.
    """
    a = CanonicalResult(
        area="ds4-eval",
        grader_version="ds4-eval=1",
        n=1,
        strict_pass_rate=0.5,
        rows=[],
    )
    b = CanonicalResult(
        area="ds4-eval",
        grader_version="ds4-eval=2",
        n=1,
        strict_pass_rate=0.7,
        rows=[],
    )
    md = _format_markdown(
        [
            (tmp_path / "a/result.json", a, a),
            (tmp_path / "b/result.json", b, b),
        ]
    )
    assert "[grader-version mismatch]" in md
    assert "## grader_version: `ds4-eval=1`" in md
    assert "## grader_version: `ds4-eval=2`" in md


def test_regrade_writes_canonical_json_next_to_source(tmp_path):
    """The regrade CLI writes regraded.json with canonical fields next
    to each source result.json (default behaviour; --no-per-input opts
    out).
    """
    case_id = _real_ds4_case_id("choice")
    case = next(c for c in ds4_eval.DS4_EVAL_CASES if c["id"] == case_id)
    src = tmp_path / "run-A" / "result.json"
    src.parent.mkdir(parents=True)
    src.write_text(
        json.dumps(
            {
                "area": "ds4-eval",
                "rows": [
                    {
                        "case_id": case_id,
                        "source": case["source"],
                        "kind": "choice",
                        "content": f"Answer: {case['answer']}",
                        "graded": {
                            "pass": True,
                            "format_pass": True,
                            "semantic_hint": True,
                        },
                    }
                ],
            }
        )
    )

    from lucebench.regrade import main as regrade_main

    rc = regrade_main([str(tmp_path / "run-A"), "--out", str(tmp_path / "report.md")])
    assert rc == 0
    regraded_path = tmp_path / "run-A" / "regraded.json"
    assert regraded_path.exists()
    payload = json.loads(regraded_path.read_text())
    assert payload["schema_version"] == schema.SCHEMA_VERSION
    assert payload["grader_version"] == f"ds4-eval={ds4_eval.GRADER_VERSION}"
    assert payload["strict_pass_rate"] == pytest.approx(1.0)

    # Markdown + sibling JSON aggregate both written when --out is .md.
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.json").exists()


def test_regrade_skips_props_and_metadata_files(tmp_path):
    """A sweep run dir with props.json / command.sh next to result.json
    must not pick those up as graders' inputs (regression: an earlier
    draft of the resolver picked everything ending in .json).
    """
    src = tmp_path / "run-A" / "result.json"
    src.parent.mkdir(parents=True)
    case_id = _real_ds4_case_id("choice")
    case = next(c for c in ds4_eval.DS4_EVAL_CASES if c["id"] == case_id)
    src.write_text(
        json.dumps(
            {
                "area": "ds4-eval",
                "rows": [
                    {
                        "case_id": case_id,
                        "kind": "choice",
                        "content": f"Answer: {case['answer']}",
                        "graded": {"pass": True, "format_pass": True},
                    }
                ],
            }
        )
    )
    (tmp_path / "run-A" / "props.json").write_text('{"arch": "qwen35"}')
    (tmp_path / "run-A" / "command.sh").write_text("# anything")

    from lucebench.regrade import main as regrade_main

    rc = regrade_main(
        [str(tmp_path / "run-A"), "--out", str(tmp_path / "r.md"), "--no-per-input"]
    )
    assert rc == 0
    md = (tmp_path / "r.md").read_text()
    # Exactly one grader-version table → exactly one input picked up.
    assert md.count("## grader_version:") == 1
    assert "props" not in md.lower() or "props.json" not in md  # negative assert
