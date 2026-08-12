"""Run orchestration: per-area runners + the multi-area sweep driver.

Split out of cli.py. These functions own the "run cases → grade → write
``<area>.json`` → roll up a summary row" flow for the standard, forge, and
agent_recorded areas, plus the ``--areas`` sweep that chains them and writes
the combined ``_summary.{json,md}``. cli.py stays the arg-parse + dispatch
shell and re-exports these names (the snapshot subcommand and tests import
them from ``lucebench.cli``).

Shared low-level helpers (AREAS, select_cases, the auth/JSON/terse helpers)
still live in cli.py; we import them here. cli.py imports this module at the
bottom of its file so that by the time this ``from lucebench.cli import ...``
runs, those names are already defined on the partially-initialised module —
a deliberate, working resolution of the two-way dependency.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

from lucebench import __version__
from lucebench._display import format_row
from lucebench._thinking import verify_thinking_control
from lucebench.cli import (
    AREAS,
    _AuthEnvEmpty,
    _forge_available,
    _resolve_auth_header,
    _row_is_unreachable,
    _summarize_injection,
    _terse_rows,
    _write_area_json,
    select_cases,
)
from lucebench.model_cards import card_is_thinking_capable
from lucebench.runner import run_case


def _run_forge_area_to_dir(
    *,
    out_root: Path,
    url: str,
    model: str,
    auth_header: str,
    timeout: int,
    max_tokens: int | None,
    questions: int | None,
) -> dict[str, Any] | None:
    """Drive the forge area + write ``<out_root>/forge.json``.

    Returns the per-area summary row (the dict appended to
    ``summary_areas``) or ``None`` if the forge runner raised
    ``SystemExit`` (e.g. no anthropic SDK installed).
    """
    from lucebench.areas.forge import run_forge_area

    max_tokens_forge = max_tokens if max_tokens is not None else 4096
    print(
        f"\n[lucebench] === area=forge max_tokens={max_tokens_forge} ===",
        flush=True,
    )
    try:
        forge_rows, forge_summary = run_forge_area(
            url=url,
            model=model,
            max_tokens=max_tokens_forge,
            timeout_s=timeout,
            auth_header=auth_header,
            questions=questions,
        )
    except SystemExit as exc:
        print(f"[lucebench] forge: {exc}", file=sys.stderr, flush=True)
        return None
    _write_area_json(
        out_root / "forge.json",
        area="forge",
        url=url,
        model=model,
        summary=forge_summary,
        rows=forge_rows,
    )
    print(
        f"[lucebench] area=forge pass_rate={forge_summary.get('pass_rate', 0):.2f}% "
        f"({forge_summary.get('n_pass', 0)}/{forge_summary.get('n_scenarios', 0)})",
        flush=True,
    )
    return {
        "area": "forge",
        "n": forge_summary.get("n_scenarios", 0),
        "pass": forge_summary.get("n_pass", 0),
        "rate": forge_summary.get("pass_rate", 0.0),
        "wall_total": sum(r.get("wall_seconds") or 0 for r in forge_rows),
        "wall_median": (
            statistics.median([r.get("wall_seconds") or 0 for r in forge_rows])
            if forge_rows
            else 0
        ),
    }


def _run_agent_recorded_to_dir(
    *,
    out_root: Path,
    url: str,
    model: str,
    auth_header: str,
    timeout: int,
    max_tokens: int | None,
    questions: int | None,
    restart_between_cases: bool = True,
    replay_mode: str = "sequential",
    mock_judge: Any = None,
) -> tuple[dict[str, Any] | None, bool]:
    """Drive the multi-turn agent_recorded area + write ``<out_root>/agent_recorded.json``.

    Mirrors ``_run_standard_area_to_dir`` — the area owns its own
    cold/warm loop and judge invocation, so the standard ``run_case``
    path doesn't fit. Returns ``(row, aborted)`` where ``aborted`` is
    True for hard configuration failures (e.g. missing ANTHROPIC_API_KEY
    → ``JudgeUnavailable``) that should fail the sweep instead of being
    silently dropped from the summary.
    """
    from lucebench.areas.agent_recorded import run_agent_recorded_area
    from lucebench.grading.llm_judge import JudgeUnavailable

    max_tokens_eff = max_tokens if max_tokens is not None else 512
    print(
        f"\n[lucebench] === area=agent_recorded max_tokens={max_tokens_eff} "
        f"restart_between_cases={restart_between_cases} replay_mode={replay_mode} ===",
        flush=True,
    )
    try:
        rows, summary = run_agent_recorded_area(
            url=url,
            model=model,
            max_tokens=max_tokens_eff,
            auth_header=auth_header,
            timeout_s=timeout,
            restart_between_cases=restart_between_cases,
            questions=questions,
            mock_judge=mock_judge,
            replay_mode=replay_mode,
        )
    except JudgeUnavailable as exc:
        # Missing judge credentials are a configuration error, not a
        # per-case hiccup — abort the sweep rather than reporting success
        # with the area silently missing from the summary.
        print(
            f"[lucebench] agent_recorded aborted: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return None, True
    except Exception as exc:  # noqa: BLE001
        # Any other unexpected failure also aborts — letting the sweep
        # exit 0 with agent_recorded missing was hiding real breakage.
        print(
            f"[lucebench] agent_recorded aborted: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return None, True

    _write_area_json(
        out_root / "agent_recorded.json",
        area="agent_recorded",
        url=url,
        model=model,
        summary=summary,
        rows=rows,
    )
    print(
        f"[lucebench] area=agent_recorded pass_rate={summary.get('pass_rate', 0):.2f}% "
        f"({summary.get('n_pass', 0)}/{summary.get('n_judged', 0)}) "
        f"cache_speedup_p50={summary.get('cache_speedup_p50')} "
        f"cache_hit_rate={summary.get('cache_hit_rate')}% "
        f"cold_prefill_p50={summary.get('cold_prefill_p50')}s "
        f"warm_prefill_p50={summary.get('warm_prefill_p50')}s "
        f"final_prefill_p50={summary.get('final_prefill_p50')}s",
        flush=True,
    )
    turn_walls = [
        turn.get("wall_s") or 0
        for r in rows
        for turn in (r.get("turns") or [])
    ]
    cold_walls = [r["cold"].get("wall_s") or 0 for r in rows]
    warm_walls = [r["warm"].get("wall_s") or 0 for r in rows]
    walls = turn_walls or (cold_walls + warm_walls)
    return {
        "area": "agent_recorded",
        "n": summary.get("n", 0),
        "pass": summary.get("n_pass", 0),
        "rate": summary.get("pass_rate", 0.0),
        "wall_total": sum(walls),
        "wall_median": statistics.median(walls) if walls else 0,
    }, False


def _run_standard_area_to_dir(
    area: str,
    *,
    out_root: Path,
    url: str,
    model: str,
    auth_header: str,
    timeout: int,
    max_tokens: int | None,
    think: bool | None,
    sampling: dict[str, Any] | None,
    sampling_source: str | None,
    questions: int | None,
    no_fail_fast: bool,
    prompt_thinking_control: str,
    server_honors_api_flags: bool,
    reasoning_effort: str = "high",
    thinking_budget_tokens: int | None = None,
    client_thinking_budget: int | None = None,
    model_card: dict[str, Any] | None = None,
    card_source: str | None = None,
    card_stem: str | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    """Drive a single stdlib area into ``<out_root>/<area>.json``.

    Returns ``(summary_row, aborted)`` where ``aborted`` is ``True`` when
    the fail-fast guard tripped on the first case (server unreachable).
    """
    cfg = AREAS[area]
    cases = cfg["load"]()
    cases = select_cases(cases, questions=questions)
    chosen_max_tokens = max_tokens if max_tokens is not None else cfg["default_max_tokens"]
    chosen_think = think if think is not None else cfg["default_thinking"]
    print(
        f"\n[lucebench] === area={area} cases={len(cases)} think={chosen_think} "
        f"max_tokens={chosen_max_tokens} ===",
        flush=True,
    )

    # Capability gate (see single-area path): only inject think/nothink
    # tokens for a thinking-capable card; otherwise force the flag off so
    # neither the card nor the family-map fallback injects.
    effective_thinking_control = (
        prompt_thinking_control if card_is_thinking_capable(model_card) else "off"
    )

    sampling = sampling or {}
    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        row = run_case(
            url=url,
            case=case,
            timeout_s=timeout,
            max_tokens=chosen_max_tokens,
            think=chosen_think,
            model=model,
            auth_header=auth_header,
            temperature=sampling.get("temperature"),
            top_p=sampling.get("top_p"),
            top_k=sampling.get("top_k"),
            min_p=sampling.get("min_p"),
            presence_penalty=sampling.get("presence_penalty"),
            repetition_penalty=sampling.get("repetition_penalty"),
            sampling_source=sampling_source,
            thinking_control_flag=effective_thinking_control,
            server_honors_api_flags=server_honors_api_flags,
            reasoning_effort=reasoning_effort,
            thinking_budget_tokens=thinking_budget_tokens,
            client_thinking_budget=client_thinking_budget,
            model_card=model_card,
            card_source=card_source,
            card_stem=card_stem,
        )
        graded = cfg["grade"](case, row)
        row["pass"] = graded.get("pass", False)
        row["graded"] = graded
        rows.append(row)
        print(format_row(idx, row, graded), flush=True)
        if idx == 1 and not no_fail_fast and _row_is_unreachable(row):
            print(
                f"\n[lucebench] sweep aborted — server at {url} appears "
                f"unreachable (case 1 raised {row.get('error')!r}). "
                "Pass --no-fail-fast to keep going anyway.",
                file=sys.stderr,
                flush=True,
            )
            return None, True

    pass_n = sum(1 for r in rows if r["pass"])
    rate = 100 * pass_n / len(rows) if rows else 0
    walls = [r.get("wall_seconds") or 0 for r in rows]
    wall_total = sum(walls)
    wall_median = statistics.median(walls) if walls else 0
    print(
        f"[lucebench] area={area} pass_rate={rate:.2f}% "
        f"({pass_n}/{len(rows)}) wall_total={wall_total:.0f}s",
        flush=True,
    )

    requested_mode = "think" if chosen_think else "nothink"
    honored, contradicting = verify_thinking_control(rows, requested_mode)
    injection_summary = _summarize_injection(rows)
    if not honored:
        host = url.split("://", 1)[1].split("/", 1)[0] if "://" in url else url
        print(
            f"[lucebench] WARNING: thinking control not honored at {host} — "
            f"{contradicting}/{len(rows)} rows in {requested_mode} mode have "
            f"non-empty reasoning. Consider --prompt-thinking-control=on or "
            f"pick a model card with an explicit thinking_control block.",
            file=sys.stderr,
            flush=True,
        )

    terse = _terse_rows(rows)
    (out_root / f"{area}.json").write_text(
        json.dumps(
            {
                "lucebench_version": __version__,
                "area": area,
                "url": url,
                "model": model,
                "think": chosen_think,
                "max_tokens": chosen_max_tokens,
                "n": len(rows),
                "pass": pass_n,
                "pass_rate": rate,
                "wall_total": wall_total,
                "wall_median": wall_median,
                "thinking_control_requested": requested_mode,
                "thinking_control_honored": honored,
                "contradicting_rows": contradicting,
                "thinking_control_injection": injection_summary,
                "rows": terse,
            },
            indent=2,
        )
    )
    return (
        {
            "area": area,
            "n": len(rows),
            "pass": pass_n,
            "rate": rate,
            "wall_total": wall_total,
            "wall_median": wall_median,
        },
        False,
    )


def write_sweep_summary(
    out_root: Path,
    *,
    name: str,
    url: str,
    model: str,
    summary_areas: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``_summary.json`` + ``_summary.md`` to ``out_root`` and return the JSON payload.

    ``extra`` is shallow-merged into the JSON payload — used by the
    snapshot subcommand to record ``level`` next to the area roll-up so
    downstream tools (``submit-baseline``) can validate the snapshot
    against the requested tier.
    """
    summary: dict[str, Any] = {
        "lucebench_version": __version__,
        "name": name,
        "url": url,
        "model": model,
        "areas": summary_areas,
    }
    if extra:
        summary.update(extra)
    (out_root / "_summary.json").write_text(json.dumps(summary, indent=2))

    md_lines = [
        f"# luce-bench sweep — {name}",
        "",
        f"- url:   `{url}`",
        f"- model: `{model}`",
        f"- lucebench v{__version__}",
        "",
        "| area | n | pass | rate | wall_total | wall_median |",
        "|------|---|------|------|------------|-------------|",
    ]
    for a in summary_areas:
        md_lines.append(
            f"| {a['area']} | {a['n']} | {a['pass']} | "
            f"{a['rate']:.1f}% | {a['wall_total']:.0f}s | {a['wall_median']:.1f}s |"
        )
    (out_root / "_summary.md").write_text("\n".join(md_lines) + "\n")
    return summary


def _run_sweep(args) -> int:
    """Run every stdlib area in sequence, write per-area + combined JSON.

    Layout:
        <out_dir>/<name>/
            ds4-eval.json
            code.json
            longctx.json
            agent.json
            forge.json       # only when [forge] is installed; skipped with a hint otherwise
            _summary.json    # {areas: [{area, n, pass, rate, wall_s}, ...]}
            _summary.md
    """
    import datetime as _dt

    name = args.name or _dt.date.today().isoformat() + "-sweep"
    out_root = args.out_dir / name
    out_root.mkdir(parents=True, exist_ok=True)

    # The set of areas to run is supplied by main() in args.areas_list
    # (computed from --areas, with back-compat for --area).
    sweep_areas = list(args.areas_list)
    forge_ok, forge_reason = _forge_available()
    try:
        auth_header = _resolve_auth_header(args, on_empty="error")
    except _AuthEnvEmpty as e:
        print(str(e), file=sys.stderr)
        return 2

    print(
        f"[lucebench] sweep name={name} "
        f"areas={','.join(sweep_areas)} url={args.url} model={args.model} "
        f"out={out_root}",
        flush=True,
    )

    if "forge" in sweep_areas and not forge_ok:
        print(
            f"[lucebench] forge: skipped — {forge_reason}",
            file=sys.stderr,
            flush=True,
        )
        sweep_areas = [a for a in sweep_areas if a != "forge"]

    summary_areas: list[dict[str, Any]] = []
    for area in sweep_areas:
        if area == "forge":
            row = _run_forge_area_to_dir(
                out_root=out_root,
                url=args.url,
                model=args.model,
                auth_header=auth_header,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                questions=args.questions,
            )
            if row is not None:
                summary_areas.append(row)
            continue
        if area == "agent_recorded":
            row, aborted = _run_agent_recorded_to_dir(
                out_root=out_root,
                url=args.url,
                model=args.model,
                auth_header=auth_header,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                questions=args.questions,
                restart_between_cases=getattr(
                    args, "agent_recorded_restart", True
                ),
                replay_mode=getattr(args, "agent_recorded_replay_mode", "sequential"),
            )
            if aborted:
                return 3
            if row is not None:
                summary_areas.append(row)
            continue

        row, aborted = _run_standard_area_to_dir(
            area,
            out_root=out_root,
            url=args.url,
            model=args.model,
            auth_header=auth_header,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            think=args.think,
            sampling=getattr(args, "sampling", {}),
            sampling_source=getattr(args, "sampling_source", "none"),
            questions=args.questions,
            no_fail_fast=args.no_fail_fast,
            prompt_thinking_control=getattr(args, "prompt_thinking_control", "off"),
            server_honors_api_flags=getattr(args, "server_honors_api_flags", False),
            reasoning_effort=getattr(args, "reasoning_effort", "high"),
            thinking_budget_tokens=getattr(args, "thinking_budget_tokens", None),
            client_thinking_budget=getattr(args, "client_thinking_budget", None),
            model_card=getattr(args, "resolved_card", None),
            card_source=getattr(args, "card_source", None),
            card_stem=getattr(args, "card_stem", None),
        )
        if aborted:
            return 3
        if row is not None:
            summary_areas.append(row)

    # Called for its side effect: writes _summary.json / _summary.md under
    # out_root. The returned payload is not needed here.
    write_sweep_summary(
        out_root,
        name=name,
        url=args.url,
        model=args.model,
        summary_areas=summary_areas,
    )

    md_text = (out_root / "_summary.md").read_text()
    print(f"\n[lucebench] sweep complete → {out_root}", flush=True)
    print(md_text.rstrip(), flush=True)
    return 0


