"""Command-line entry point: ``lucebench --area X --url Y --model Z``.

Minimal dispatcher around lucebench.runner. The full-featured
multi-area runner from luce-dflash (parallelism, forge, agent areas,
sampling-from-card, per-area max_tokens defaults) is the bigger
brother; this CLI exists so external users can `pip install
luce-bench` and benchmark any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from lucebench import __version__
from lucebench.areas import agent, ds4_eval, humaneval, longctx
from lucebench.runner import run_case


def resolve_model(url: str, auth_header: str = "", timeout_s: int = 10) -> str | None:
    """Pick a model id by probing the server's /v1/models endpoint.

    Returns:
      * the single model id if the server exposes exactly one
      * None if the server exposes zero, multiple, or doesn't speak the
        OpenAI /v1/models shape

    The caller decides whether to fall back to a hard default or error
    out. We deliberately don't pick one when multiple are exposed —
    silently picking would mask user mistakes (e.g. forgetting to set
    --model when a gateway exposes 200+ models).
    """
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/models", headers={"Accept": "application/json"}
    )
    if auth_header:
        req.add_header("Authorization", auth_header)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list) or len(models) != 1:
        return None
    entry = models[0]
    if not isinstance(entry, dict):
        return None
    mid = entry.get("id")
    return mid if isinstance(mid, str) and mid else None


AREAS = {
    "ds4-eval": {
        "load": ds4_eval.load_ds4_eval_cases,
        "grade": ds4_eval.grade_case,
        "default_max_tokens": ds4_eval.DS4_EVAL_MAX_TOKENS,
        "default_thinking": True,
    },
    "code": {
        "load": humaneval.load_humaneval_cases,
        "grade": humaneval.grade_humaneval_case,
        "default_max_tokens": 2048,
        "default_thinking": False,
    },
    "longctx": {
        "load": lambda: longctx.LONGCTX_CASES,
        "grade": longctx.grade_longctx_case,
        "default_max_tokens": 256,
        "default_thinking": False,
    },
    "agent": {
        "load": agent.load_agent_cases,
        "grade": agent.grade_agent_case,
        "default_max_tokens": 4096,
        "default_thinking": False,
    },
}


def select_cases(
    cases: list[dict],
    *,
    questions: int | None = None,
    case_id: str | None = None,
    case_index: int | None = None,
    sources: list[str] | None = None,
) -> list[dict]:
    """Filter cases by id / index / source / count."""
    out = list(cases)
    if sources:
        out = [c for c in out if c.get("source") in sources]
    if case_id:
        out = [c for c in out if c.get("id") == case_id]
    if case_index is not None:
        out = out[case_index : case_index + 1] if 0 <= case_index < len(out) else []
    if questions:
        out = out[:questions]
    return out


def format_row(idx: int, row: dict, graded: dict) -> str:
    src = row.get("source") or "?"
    cid = row.get("case_id") or "?"
    verdict = "PASS" if graded.get("pass") else "FAIL"
    given = graded.get("given") or "?"
    correct = graded.get("correct") or "?"
    wall = row.get("wall_seconds") or 0
    timings = row.get("timings") or {}
    tps = timings.get("decode_tokens_per_sec") or 0
    return (
        f"  {idx:3d} {verdict} {src:14s} {cid:24s} "
        f"given={given:20s} correct={correct:20s} "
        f"wall={wall:.2f}s {tps:.0f}tps"
    )


# Substrings in row["error"] that mean the server is unreachable — fail-fast
# triggers on the first row matching any of these unless --no-fail-fast is set.
_UNREACHABLE_ERRORS = (
    "ConnectionRefusedError",
    "ConnectionResetError",
    "Name or service not known",
    "Temporary failure in name resolution",
    "No route to host",
    "Connection refused",
    "URLError",
)


def _row_is_unreachable(row: dict) -> bool:
    """True if row["error"] looks like a connection-level failure.

    Used by the sweep's fail-fast guard. Timeouts and HTTP errors are
    deliberately excluded — those are per-request failures, not a
    server-down signal.
    """
    err = row.get("error") or ""
    return any(marker in err for marker in _UNREACHABLE_ERRORS)


def _forge_available() -> tuple[bool, str | None]:
    """Probe whether the `[forge]` extra is installed without importing it eagerly.

    Returns (available, reason) where reason is a short string the
    sweep prints when forge is skipped. Lazy import keeps the default
    install free of the anthropic dep.
    """
    try:
        import anthropic  # noqa: F401

        return True, None
    except ImportError:
        return False, "anthropic SDK not installed — `pip install 'luce-bench[forge]'`"


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

    # Default sweep covers the stdlib areas. Forge is added when the
    # `[forge]` extra is available; otherwise we print a "skipped"
    # hint so users know how to opt in.
    sweep_areas = ["ds4-eval", "code", "longctx", "agent"]
    forge_ok, forge_reason = _forge_available()
    if forge_ok:
        sweep_areas.append("forge")
    auth_header = ""
    if args.auth_env:
        token = os.environ.get(args.auth_env, "")
        if not token:
            print(f"--auth-env {args.auth_env}: env var is empty or unset", file=sys.stderr)
            return 2
        auth_header = f"Bearer {token}"

    print(
        f"[lucebench] v{__version__} sweep name={name} "
        f"areas={','.join(sweep_areas)} url={args.url} model={args.model} "
        f"out={out_root}",
        flush=True,
    )

    if not forge_ok:
        print(
            f"[lucebench] forge: skipped — {forge_reason}",
            file=sys.stderr,
            flush=True,
        )

    summary_areas: list[dict[str, Any]] = []
    for area in sweep_areas:
        if area == "forge":
            # Forge has its own runner (recording AnthropicClient), so dispatch
            # separately. Still emit per-area JSON for symmetry with the others.
            from lucebench.areas.forge import run_forge_area

            max_tokens_forge = args.max_tokens if args.max_tokens is not None else 4096
            print(
                f"\n[lucebench] === area=forge max_tokens={max_tokens_forge} ===",
                flush=True,
            )
            try:
                forge_rows, forge_summary = run_forge_area(
                    url=args.url,
                    model=args.model,
                    max_tokens=max_tokens_forge,
                    timeout_s=args.timeout,
                    auth_header=auth_header,
                    questions=args.questions,
                )
            except SystemExit as exc:
                print(f"[lucebench] forge: {exc}", file=sys.stderr, flush=True)
                continue
            (out_root / "forge.json").write_text(
                json.dumps(
                    {
                        "lucebench_version": __version__,
                        "area": "forge",
                        "url": args.url,
                        "model": args.model,
                        **forge_summary,
                        "rows": forge_rows,
                    },
                    indent=2,
                    default=str,
                )
            )
            summary_areas.append(
                {
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
            )
            print(
                f"[lucebench] area=forge pass_rate={forge_summary.get('pass_rate', 0):.2f}% "
                f"({forge_summary.get('n_pass', 0)}/{forge_summary.get('n_scenarios', 0)})",
                flush=True,
            )
            continue

        cfg = AREAS[area]
        cases = cfg["load"]()
        cases = select_cases(cases, questions=args.questions)
        max_tokens = args.max_tokens if args.max_tokens is not None else cfg["default_max_tokens"]
        think = args.think if args.think is not None else cfg["default_thinking"]
        print(
            f"\n[lucebench] === area={area} cases={len(cases)} think={think} "
            f"max_tokens={max_tokens} ===",
            flush=True,
        )

        rows: list[dict[str, Any]] = []
        aborted = False
        for idx, case in enumerate(cases, start=1):
            row = run_case(
                url=args.url,
                case=case,
                timeout_s=args.timeout,
                max_tokens=max_tokens,
                think=think,
                model=args.model,
                auth_header=auth_header,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            graded = cfg["grade"](case, row)
            row["pass"] = graded.get("pass", False)
            row["graded"] = graded
            rows.append(row)
            print(format_row(idx, row, graded), flush=True)
            # Fail-fast: if the very first case looks like the server is
            # unreachable, abort the sweep rather than wasting timeouts
            # on the remaining ~91 cases per area * 4 areas. Skip the
            # guard when --no-fail-fast is set (CI / chaos tests).
            if idx == 1 and not args.no_fail_fast and _row_is_unreachable(row):
                print(
                    f"\n[lucebench] sweep aborted — server at {args.url} appears "
                    f"unreachable (case 1 raised {row.get('error')!r}). "
                    "Pass --no-fail-fast to keep going anyway.",
                    file=sys.stderr,
                    flush=True,
                )
                aborted = True
                break
        if aborted:
            return 3

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

        # Per-area JSON
        terse = [{k: v for k, v in r.items() if k != "_response"} for r in rows]
        (out_root / f"{area}.json").write_text(
            json.dumps(
                {
                    "lucebench_version": __version__,
                    "area": area,
                    "url": args.url,
                    "model": args.model,
                    "think": think,
                    "max_tokens": max_tokens,
                    "n": len(rows),
                    "pass": pass_n,
                    "pass_rate": rate,
                    "wall_total": wall_total,
                    "wall_median": wall_median,
                    "rows": terse,
                },
                indent=2,
            )
        )
        summary_areas.append(
            {
                "area": area,
                "n": len(rows),
                "pass": pass_n,
                "rate": rate,
                "wall_total": wall_total,
                "wall_median": wall_median,
            }
        )

    # Combined summary
    summary = {
        "lucebench_version": __version__,
        "name": name,
        "url": args.url,
        "model": args.model,
        "areas": summary_areas,
    }
    (out_root / "_summary.json").write_text(json.dumps(summary, indent=2))

    md_lines = [
        f"# luce-bench sweep — {name}",
        "",
        f"- url:   `{args.url}`",
        f"- model: `{args.model}`",
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

    print(f"\n[lucebench] sweep complete → {out_root}", flush=True)
    print("\n".join(md_lines), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="lucebench",
        description="Capability benchmarks for chat-completion endpoints.",
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument(
        "--url",
        "--base-url",
        dest="url",
        default="http://127.0.0.1:8080",
        help="Server base URL (default: http://127.0.0.1:8080).",
    )
    ap.add_argument(
        "--model",
        default="default",
        help="Model identifier sent in the request body. "
        "When left as the literal string 'default', "
        "the CLI queries `<base-url>/v1/models` and "
        "auto-picks the single exposed model. If the "
        "server exposes zero or multiple, it falls back "
        "to the literal 'default' (which most servers "
        "404 on — pass --model explicitly for gateways).",
    )
    ap.add_argument(
        "--area",
        choices=sorted(set(AREAS) | {"forge"}),
        help="Evaluation area to run. Required unless --sweep is set.",
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="Run all stdlib areas (ds4-eval, code, longctx, agent) "
        "in sequence. Forge requires --area forge explicitly "
        "since it needs the [forge] extra.",
    )
    ap.add_argument(
        "--name",
        default=None,
        help="Label for snapshot directory under --out-dir. "
        "Common pattern: machine + model tag, e.g. "
        "`bragi-gemma4-26b-2026-05-26`.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./snapshots"),
        help="Root directory for sweep snapshots. Each area writes "
        "<out-dir>/<name>/<area>.json and a combined "
        "_summary.json. Default: ./snapshots",
    )
    ap.add_argument(
        "--questions", type=int, default=None, help="Limit to first N cases (after other filters)."
    )
    ap.add_argument("--case-id", default=None, help="Run only the case with this ID.")
    ap.add_argument(
        "--case-index",
        type=int,
        default=None,
        help="Run only the case at this position (after source filter).",
    )
    ap.add_argument(
        "--sources",
        default=None,
        help="Comma-separated source filter (e.g. AIME2025,GPQA Diamond).",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Per-request decode cap (overrides area default).",
    )
    ap.add_argument("--think", dest="think", action="store_true", default=None)
    ap.add_argument("--no-think", dest="think", action="store_false")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=300, help="Per-request wall timeout (s).")
    ap.add_argument(
        "--auth-env",
        default=None,
        help="Env var name to read auth bearer token from "
        "(e.g. OPENAI_API_KEY, OPENROUTER_API_KEY).",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write the per-case rows as a JSON array to this path.",
    )
    ap.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="In --sweep mode, keep going even when the first case can't reach "
        "the server. Default behavior aborts on connection-refused-style "
        "errors to avoid burning ~92 timeouts per area on a typo'd URL.",
    )
    ap.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Run up to N cases concurrently. Default 1 "
        "(sequential). Safe to raise for stateless HTTP "
        "gateways (OpenRouter); leave at 1 for single-GPU "
        "local servers since concurrent requests just queue.",
    )
    args = ap.parse_args()
    if args.parallel < 1:
        ap.error("--parallel must be >= 1")
    if not args.area and not args.sweep:
        ap.error("one of --area or --sweep is required")
    if args.sweep and args.area:
        ap.error("--area and --sweep are mutually exclusive — pick one")

    # /v1/models auto-resolution. Only fires when the user left --model
    # at the literal default; an explicit value (even if wrong) is
    # respected so gateways with hundreds of models stay predictable.
    if args.model == "default":
        auth_for_probe = ""
        if args.auth_env:
            token = os.environ.get(args.auth_env, "")
            if token:
                auth_for_probe = f"Bearer {token}"
        resolved = resolve_model(args.url, auth_header=auth_for_probe)
        if resolved:
            print(
                f"[lucebench] --model default → resolved to '{resolved}' via {args.url}/v1/models",
                flush=True,
            )
            args.model = resolved
        else:
            print(
                f"[lucebench] --model default: /v1/models at {args.url} "
                "didn't expose exactly one model — sending 'default' as-is. "
                "Most servers will 404 on this; pass --model explicitly.",
                file=sys.stderr,
                flush=True,
            )

    # ── Sweep mode: run all stdlib areas sequentially, write into a
    # snapshot dir keyed on --name (default: today's date + a sweep tag).
    if args.sweep:
        return _run_sweep(args)

    # Forge takes a completely different path — it owns its own runner
    # (recording AnthropicClient + scenario driver) instead of using
    # run_case + a grader. Dispatch early.
    if args.area == "forge":
        from lucebench.areas.forge import run_forge_area

        max_tokens = args.max_tokens if args.max_tokens is not None else 4096
        auth_header = ""
        if args.auth_env:
            token = os.environ.get(args.auth_env, "")
            if not token:
                ap.error(f"--auth-env {args.auth_env}: env var is empty or unset")
            auth_header = f"Bearer {token}"
        rows, summary = run_forge_area(
            url=args.url,
            model=args.model,
            max_tokens=max_tokens,
            timeout_s=args.timeout,
            auth_header=auth_header,
            tags=None,
            names=None,
            questions=args.questions,
        )
        for idx, r in enumerate(rows, start=1):
            verdict = "PASS" if r.get("pass") else "FAIL"
            print(
                f"  {idx:3d} {verdict} forge   {r['case_id']:32s} "
                f"wall={r['wall_seconds']:.2f}s "
                f"calls={len(r.get('iterations') or [])}",
                flush=True,
            )
        print(
            f"\n[lucebench] forge pass_rate={summary['pass_rate']:.2f}% "
            f"({summary['n_pass']}/{summary['n_scenarios']})",
            flush=True,
        )
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(
                    {
                        "lucebench_version": __version__,
                        "area": "forge",
                        "url": args.url,
                        "model": args.model,
                        **summary,
                        "rows": rows,
                    },
                    indent=2,
                    default=str,
                )
            )
            print(f"[lucebench] wrote {len(rows)} rows to {args.json_out}", flush=True)
        return 0

    cfg = AREAS[args.area]
    cases = cfg["load"]()
    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None
    selected = select_cases(
        cases,
        questions=args.questions,
        case_id=args.case_id,
        case_index=args.case_index,
        sources=sources,
    )
    if not selected:
        ap.error("no cases selected by the supplied filters")

    max_tokens = args.max_tokens if args.max_tokens is not None else cfg["default_max_tokens"]
    think = args.think if args.think is not None else cfg["default_thinking"]

    auth_header = ""
    if args.auth_env:
        token = os.environ.get(args.auth_env, "")
        if not token:
            ap.error(f"--auth-env {args.auth_env}: env var is empty or unset")
        auth_header = f"Bearer {token}"

    print(
        f"[lucebench] v{__version__} area={args.area} cases={len(selected)} "
        f"url={args.url} model={args.model} think={think} max_tokens={max_tokens}",
        flush=True,
    )

    def _do(idx_case):
        idx, case = idx_case
        row = run_case(
            url=args.url,
            case=case,
            timeout_s=args.timeout,
            max_tokens=max_tokens,
            think=think,
            model=args.model,
            auth_header=auth_header,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        graded = cfg["grade"](case, row)
        row["pass"] = graded.get("pass", False)
        row["graded"] = graded
        row["_idx"] = idx
        return row, graded

    rows: list[dict[str, Any]] = []
    if args.parallel > 1:
        # Parallel runner: stateless HTTP gateways (OpenRouter etc.) can
        # serve many concurrent requests. Local single-GPU servers just
        # queue them. Output streams "as completed" but the JSON-out rows
        # are sorted back to selection order so snapshots stay deterministic.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(_do, (i, c)): (i, c) for i, c in enumerate(selected, start=1)}
            for fut in as_completed(futures):
                row, graded = fut.result()
                rows.append(row)
                print(format_row(row["_idx"], row, graded), flush=True)
        rows.sort(key=lambda r: r["_idx"])
    else:
        for idx, case in enumerate(selected, start=1):
            row, graded = _do((idx, case))
            rows.append(row)
            print(format_row(idx, row, graded), flush=True)
    for r in rows:
        r.pop("_idx", None)

    pass_n = sum(1 for r in rows if r["pass"])
    rate = 100 * pass_n / len(rows) if rows else 0
    walls = [r.get("wall_seconds") or 0 for r in rows]
    print(
        f"\n[lucebench] pass_rate={rate:.2f}% ({pass_n}/{len(rows)}) "
        f"wall_total={sum(walls):.0f}s wall_median={statistics.median(walls):.1f}s",
        flush=True,
    )

    if args.json_out:
        # Drop the raw _response blob from JSON-out by default to keep file size sane.
        terse = [{k: v for k, v in r.items() if k != "_response"} for r in rows]
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "lucebench_version": __version__,
                    "area": args.area,
                    "url": args.url,
                    "model": args.model,
                    "think": think,
                    "max_tokens": max_tokens,
                    "n": len(rows),
                    "pass": pass_n,
                    "pass_rate": rate,
                    "rows": terse,
                },
                indent=2,
            )
        )
        print(f"[lucebench] wrote {len(rows)} rows to {args.json_out}", flush=True)

    return 0 if pass_n == len(rows) or os.environ.get("LUCEBENCH_PASS_RATE_GATE") is None else 1


if __name__ == "__main__":
    sys.exit(main())
