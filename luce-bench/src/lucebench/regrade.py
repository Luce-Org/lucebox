"""``luce-bench regrade`` — re-score stored raw outputs with the pinned grader.

Goal
----
Take any historical ``result.json`` (legacy or current shape), normalise
it, re-run each row through the area's canonical ``grade_case`` at the
CURRENT ``GRADER_VERSION``, and emit:

  * ``<run-dir>/regraded.json`` — one canonical result per input run
  * ``--out report.md`` — a markdown comparison table (percent at this
    layer; canonical JSON underneath is fractions)
  * ``--out report.json`` — aggregate of every regraded run

The CLI is also a guard rail: it refuses to put two runs in the same
"comparable" table row unless their ``grader_version``s match. If they
don't, the markdown surfaces a ``[grader-version mismatch]`` row banner
and the per-grader-version tables are emitted separately.

CLI shape
---------
``luce-bench regrade <run-dir|file> [<more> ...]``
``luce-bench regrade --glob 'snapshots/**/result.json'``

``regrade`` is wired into ``lucebench.cli`` as a subcommand so existing
``lucebench`` invocations stay backwards-compatible (no positional
clash).
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lucebench import __version__
from lucebench.areas import (
    agent,
    agent_recorded,
    ds4_eval,
    gsm8k,
    hellaswag,
    humaneval,
    longctx,
    smoke,
    truthfulqa_mc1,
)
from lucebench.normalize import load_result
from lucebench.schema import CanonicalResult, CanonicalRow

# Map canonical area names → (module, grade_fn, case_id_field, case_lookup).
# Each entry is what `regrade_result` needs to re-grade a row:
#   * module — to read GRADER_VERSION dynamically
#   * grade  — the canonical grader (case, row) -> dict
#   * cases  — list[dict] of cases keyed by case_id
_AREAS = {
    "ds4-eval": (ds4_eval, ds4_eval.grade_case, ds4_eval.DS4_EVAL_CASES),
    "smoke": (smoke, smoke.grade_smoke_case, smoke.load_smoke_cases()),
    "gsm8k": (gsm8k, gsm8k.grade_gsm8k_case, gsm8k.load_gsm8k_cases()),
    "hellaswag": (hellaswag, hellaswag.grade_hellaswag_case, hellaswag.load_hellaswag_cases()),
    "truthfulqa-mc1": (
        truthfulqa_mc1,
        truthfulqa_mc1.grade_truthfulqa_mc1_case,
        truthfulqa_mc1.load_truthfulqa_mc1_cases(),
    ),
    "code": (humaneval, humaneval.grade_humaneval_case, humaneval.load_humaneval_cases()),
    "longctx": (longctx, longctx.grade_longctx_case, longctx.LONGCTX_CASES),
    "agent": (agent, agent.grade_agent_case, agent.load_agent_cases()),
    "agent_recorded": (
        agent_recorded,
        agent_recorded.grade_agent_recorded_case,
        agent_recorded.load_agent_recorded_cases(),
    ),
}


def _case_lookup(area: str) -> dict[str, dict[str, Any]]:
    """Build a {case_id: case_dict} map for the given area.

    Uses ``case["id"]`` as the key — that's the upstream-stable id all
    ds4-eval rows reference (ds4_eval_cases.json keeps it across version
    bumps). Returns an empty dict for areas the regrade CLI doesn't know
    how to drive (forge, which has its own runner shape).
    """
    entry = _AREAS.get(area)
    if entry is None:
        return {}
    _mod, _grade, cases = entry
    return {str(c.get("id")): c for c in cases if c.get("id")}


def _grader_version_tag(area: str) -> str:
    """Render a ``"<area>=<N>"`` tag for the current grader.

    Used both for the result's ``grader_version`` field and for the
    same-grader-version comparison guard rail. Returns ``"<area>=?"``
    when the area has no GRADER_VERSION attribute (forge, etc).
    """
    entry = _AREAS.get(area)
    if entry is None:
        return f"{area}=?"
    mod, _grade, _cases = entry
    return f"{area}={getattr(mod, 'GRADER_VERSION', '?')}"


def regrade_result(canon: CanonicalResult) -> CanonicalResult:
    """Re-grade every row in a CanonicalResult using the current grader.

    Returns a NEW CanonicalResult — the input is left untouched so the
    caller can do baseline-vs-regraded diffing. ``grader_version`` is
    overwritten with the current value.

    Areas the regrade CLI doesn't know how to drive (e.g. ``forge``, which
    runs its own scenario loop rather than ``grade_case``) pass through
    unchanged with a ``grader_version="<area>=?"`` tag so downstream
    consumers can see they weren't re-scored.
    """
    entry = _AREAS.get(canon.area)
    if entry is None:
        # Area we can't re-grade — preserve the input but tag it so the
        # comparison row in the markdown shows the regrade was a no-op.
        regraded = CanonicalResult(**{k: v for k, v in asdict(canon).items() if k != "rows"})
        regraded.rows = list(canon.rows)
        regraded.grader_version = _grader_version_tag(canon.area)
        regraded.metrics = dict(canon.metrics or {})
        regraded.metrics["regrade_status"] = "skipped_unknown_area"
        return regraded

    _mod, grade_fn, _cases = entry
    cases_by_id = _case_lookup(canon.area)

    new_rows: list[CanonicalRow] = []
    strict_passes = 0
    format_passes = 0
    hints = 0
    missing_cases = 0

    for row in canon.rows:
        case = cases_by_id.get(row.case_id)
        if case is None:
            # No matching case in the fixture — leave the row's old grade
            # intact, count it as a miss, and don't fold it into the rates.
            missing_cases += 1
            new_rows.append(row)
            continue
        # The grader reads from a row-like dict with content +
        # reasoning_content (see ds4_eval.grade_case). Build that shape
        # straight from the CanonicalRow.
        row_for_grader = {
            "content": row.content,
            "reasoning_content": row.reasoning_content,
            "completion_tokens": row.completion_tokens,
            "reasoning_tokens": row.reasoning_tokens,
        }
        graded = grade_fn(case, row_for_grader)
        new_row = CanonicalRow(
            **{k: v for k, v in asdict(row).items() if k != "graded"},
        )
        new_row.graded = dict(graded)
        # Preserve strict_pass alongside `pass` so legacy-style consumers
        # that key on it keep working.
        new_row.graded.setdefault("strict_pass", new_row.graded.get("pass", False))
        new_rows.append(new_row)
        if new_row.graded.get("pass") or new_row.graded.get("strict_pass"):
            strict_passes += 1
        if new_row.graded.get("format_pass"):
            format_passes += 1
        if new_row.graded.get("semantic_hint"):
            hints += 1

    graded_n = len(new_rows) - missing_cases
    strict_rate = strict_passes / graded_n if graded_n else 0.0
    format_rate = format_passes / graded_n if graded_n else 0.0
    hint_rate = hints / graded_n if graded_n else 0.0

    metrics = dict(canon.metrics or {})
    metrics["regrade_status"] = "ok" if missing_cases == 0 else "partial"
    metrics["regrade_missing_cases"] = missing_cases
    # Stash the run's originally-declared strict rate so the markdown
    # can show "was X.XX before regrade" without a second pass through
    # the source file.
    metrics.setdefault("declared_strict_pass_rate", canon.strict_pass_rate)

    return CanonicalResult(
        schema_version=canon.schema_version,
        lucebench_version=__version__,
        grader_version=_grader_version_tag(canon.area),
        area=canon.area,
        model=canon.model,
        quant=canon.quant,
        serving_url=canon.serving_url,
        mode=canon.mode,
        seed=canon.seed,
        n=len(new_rows),
        started_at=canon.started_at,
        thinking_control_requested=canon.thinking_control_requested,
        thinking_control_honored=canon.thinking_control_honored,
        contradicting_rows=canon.contradicting_rows,
        strict_pass_rate=strict_rate,
        format_pass_rate=format_rate,
        semantic_hint_rate=hint_rate,
        metrics=metrics,
        rows=new_rows,
    )


def _resolve_inputs(args: argparse.Namespace) -> list[Path]:
    """Expand --glob + positional paths into a list of result.json files.

    Accepts both directories (we search for ``result.json`` /
    ``ds4-eval.json`` / etc. inside) and individual files. De-dupes the
    result so the same file isn't regraded twice.
    """
    seen: dict[Path, None] = {}

    def _add(p: Path) -> None:
        rp = p.resolve()
        if rp not in seen:
            seen[rp] = None

    for pattern in args.glob_patterns or []:
        for match in _glob.glob(pattern, recursive=True):
            mp = Path(match)
            if mp.is_file():
                _add(mp)

    # Names a sweep directory legitimately contains. Anything else
    # (props.json, command.sh, the renamed `regraded.json` from a
    # previous run) is excluded so a `regrade <dir>` invocation doesn't
    # accidentally pick up server metadata and try to grade it.
    sweep_area_files = {
        "ds4-eval.json",
        "code.json",
        "longctx.json",
        "agent.json",
        "agent_recorded.json",
        "forge.json",
        "smoke.json",
        "gsm8k.json",
        "hellaswag.json",
        "truthfulqa-mc1.json",
    }

    for p in args.paths or []:
        if not p.exists():
            print(f"luce-bench regrade: {p} does not exist", file=sys.stderr)
            continue
        if p.is_file():
            _add(p)
            continue
        # Directory: prefer a sole `result.json` (single-area runs);
        # otherwise pick up per-area sweep files by known name only.
        # Walking with `*.json` is wrong — run dirs typically also carry
        # `props.json`, server metadata, and stale `regraded.json` outputs
        # from previous regrades.
        result_file = p / "result.json"
        if result_file.is_file():
            _add(result_file)
            continue
        for c in sorted(p.glob("*.json")):
            if c.name in sweep_area_files:
                _add(c)

    return list(seen.keys())


def _label_for_path(path: Path) -> str:
    """Render a short human-readable label for the markdown table.

    Strategy: use the parent dir name when the file is named
    ``result.json`` / ``regraded.json``; otherwise use ``parent/stem``.
    """
    if path.name in {"result.json", "regraded.json"}:
        return path.parent.name
    return f"{path.parent.name}/{path.stem}"


def _stack_label(canon: CanonicalResult) -> str:
    """One-word identifier for the serving stack ('dflash', 'openrouter', …).

    Sniffs the URL — local 8080 / 1236 are luce-dflash; openrouter.ai is
    OpenRouter; otherwise empty. The point is *not* to be exhaustive but
    to give the markdown a second column the human eye can group on.
    """
    url = canon.serving_url.lower()
    if not url:
        return ""
    if "openrouter" in url:
        return "openrouter"
    if "127.0.0.1" in url or "localhost" in url:
        return "local"
    if url.startswith("http://") or url.startswith("https://"):
        return url.split("://", 1)[1].split("/", 1)[0]
    return url[:24]


def _format_markdown(
    regraded: list[tuple[Path, CanonicalResult, CanonicalResult]],
) -> str:
    """Render the comparison markdown.

    Groups by ``grader_version`` so runs you can directly compare share
    a table. Any cross-version comparison is replaced with a loud
    ``[grader-version mismatch]`` banner — the rule is "comparable
    requires same grader_version", and this enforces it visibly.

    Columns: label | model | stack | area | mode | n |
             strict_pass | declared_before | Δ | format_pass |
             semantic_hint (diag) | grader_ver | tc_honored.
    """
    groups: dict[str, list[tuple[Path, CanonicalResult, CanonicalResult]]] = {}
    for src, original, regraded_run in regraded:
        groups.setdefault(regraded_run.grader_version, []).append((src, original, regraded_run))

    lines: list[str] = []
    lines.append("# luce-bench regrade")
    lines.append("")
    lines.append(
        f"- inputs: {len(regraded)}; grader-version groups: {len(groups)}"
    )
    if len(groups) > 1:
        lines.append(
            "- **[grader-version mismatch]** — runs span multiple grader_versions; "
            "see per-version tables below. Direct comparison across versions is "
            "not valid; re-grade older runs to bring them to the current version."
        )
    lines.append("")

    # Per-group tables
    for gv in sorted(groups.keys()):
        entries = groups[gv]
        lines.append(f"## grader_version: `{gv}`")
        lines.append("")
        lines.append(
            "| label | model | stack | area | mode | n | strict_pass | declared_before | "
            "Δ | format_pass | semantic_hint (diag) | tc_honored |"
        )
        lines.append(
            "|---|---|---|---|---|---|---|---|---|---|---|---|"
        )
        for src, original, run in entries:
            label = _label_for_path(src)
            declared = float(
                run.metrics.get("declared_strict_pass_rate") or original.strict_pass_rate or 0.0
            )
            delta = run.strict_pass_rate - declared
            delta_str = f"{delta * 100:+.2f}pp" if abs(delta) > 1e-9 else "0.00pp"
            tc = "honored" if run.thinking_control_honored else f"NO ({run.contradicting_rows})"
            lines.append(
                "| {label} | {model} | {stack} | {area} | {mode} | {n} | "
                "{strict:.2f}% | {decl:.2f}% | {delta} | {fmt:.2f}% | {hint:.2f}% | {tc} |".format(
                    label=label,
                    model=run.model or "—",
                    stack=_stack_label(run) or "—",
                    area=run.area,
                    mode=run.mode,
                    n=run.n,
                    strict=run.strict_pass_rate * 100,
                    decl=declared * 100,
                    delta=delta_str,
                    fmt=run.format_pass_rate * 100,
                    hint=run.semantic_hint_rate * 100,
                    tc=tc,
                )
            )
        lines.append("")

    # Footnote on what these columns mean — the user has been bitten by
    # misreading semantic_hint_rate, so spell it out in the report itself.
    lines.append("---")
    lines.append("")
    lines.append("Notes:")
    lines.append(
        "- **strict_pass** — the headline score (fraction of cases whose "
        "extracted answer matches the canonical answer at the current "
        "grader_version)."
    )
    lines.append(
        "- **declared_before** — what the source run's result.json reported "
        "(normalised to a percent). Δ vs strict_pass = silent grader drift."
    )
    lines.append(
        "- **semantic_hint (diag)** — DIAGNOSTIC ONLY: did the expected "
        "answer appear anywhere in content/reasoning. NOT the score; never "
        "compare runs on this column."
    )
    lines.append(
        "- **tc_honored** — `honored` means the server's reasoning matched "
        "the requested think/nothink mode; `NO (N)` means N rows "
        "contradicted the requested mode (e.g. nothink with reasoning "
        "tokens > 0)."
    )
    return "\n".join(lines)


def _run(args: argparse.Namespace) -> int:
    inputs = _resolve_inputs(args)
    if not inputs:
        print("luce-bench regrade: no inputs resolved", file=sys.stderr)
        return 2

    regraded: list[tuple[Path, CanonicalResult, CanonicalResult]] = []
    for path in inputs:
        try:
            original = load_result(path)
        except Exception as e:
            print(f"luce-bench regrade: {path}: load failed: {e}", file=sys.stderr)
            continue
        run = regrade_result(original)
        regraded.append((path, original, run))

        # Per-input regraded.json. Sits alongside the source result.json
        # by default; --regraded-suffix lets the caller redirect.
        if not args.no_per_input:
            out_path = path.with_name(args.regraded_filename)
            out_path.write_text(run.to_json())

    if not regraded:
        print("luce-bench regrade: all inputs failed to load", file=sys.stderr)
        return 3

    markdown = _format_markdown(regraded)

    if args.out:
        out = args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".json":
            payload = {
                "schema_version": regraded[0][2].schema_version,
                "lucebench_version": __version__,
                "n_runs": len(regraded),
                "runs": [run.to_dict() for _src, _orig, run in regraded],
            }
            out.write_text(json.dumps(payload, indent=2, default=str))
            print(f"luce-bench regrade: wrote {out}", file=sys.stderr)
        else:
            out.write_text(markdown + "\n")
            print(f"luce-bench regrade: wrote {out}", file=sys.stderr)
            # Also emit a sibling JSON aggregate next to the markdown.
            json_sibling = out.with_suffix(".json")
            payload = {
                "schema_version": regraded[0][2].schema_version,
                "lucebench_version": __version__,
                "n_runs": len(regraded),
                "runs": [run.to_dict() for _src, _orig, run in regraded],
            }
            json_sibling.write_text(json.dumps(payload, indent=2, default=str))
            print(f"luce-bench regrade: wrote {json_sibling}", file=sys.stderr)
    else:
        print(markdown)

    return 0


def main(argv: list[str] | None = None) -> int:
    """``luce-bench regrade`` CLI entrypoint.

    Also wired in via ``lucebench.cli`` as the ``regrade`` subcommand,
    but stays runnable standalone (``python -m lucebench.regrade``) so
    tests can drive it without going through argparse-subparsers.
    """
    ap = argparse.ArgumentParser(
        prog="luce-bench regrade",
        description=(
            "Re-score stored luce-bench result.json files at the current "
            "grader_version. Loads both legacy (id/output/ok) and current "
            "(case_id/content/pass) shapes; emits canonical regraded.json "
            "per input, plus an aggregate markdown report. Refuses to "
            "place runs with mismatched grader_versions in the same "
            "comparison row — that's the point."
        ),
    )
    ap.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Run dirs or result JSON files. Directories are searched "
        "for `result.json` + per-area sweep files.",
    )
    ap.add_argument(
        "--glob",
        action="append",
        dest="glob_patterns",
        default=[],
        help="Glob pattern (recursive ** supported) to add inputs from. "
        "Repeatable.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. .md → markdown table (+ a sibling .json aggregate); "
        ".json → JSON aggregate only. Omit to print markdown to stdout.",
    )
    ap.add_argument(
        "--regraded-filename",
        default="regraded.json",
        help="Per-input filename written next to each source result.json "
        "(default: regraded.json). Set to e.g. `regraded-v1.json` to keep "
        "multiple re-grades side by side.",
    )
    ap.add_argument(
        "--no-per-input",
        action="store_true",
        help="Skip writing per-input regraded.json files (only emit the "
        "aggregate markdown + JSON).",
    )
    args = ap.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
