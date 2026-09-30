"""Fit per-arm latency profiles for brick_skill from measured answers (runs/<name>/answers.jsonl).

For each arm (model + thinking mode), over the answers of the chosen split:

* ttft_ms + ttft_ms_per_token * prompt_tokens: least squares of ttft_ms on prompt_tokens
  (slope clamped at >= 0; the intercept is the median TTFT when the slope cannot be fitted);
* tpot_ms: median of (total_ms - ttft_ms) / (completion_tokens - 1);
* out_tokens.default: mean completion_tokens. For a thinking arm this includes the reasoning
  tokens, because luce_server counts every generated token in usage.completion_tokens;
* out_tokens.easy/medium/hard: completion tokens averaged with the Brick complexity probs as
  weights (sum_q p_class(q) * tokens(q) / sum_q p_class(q)), when runs/<name>/brick_probs.jsonl
  has probs for the chosen --service;
* backends.<model>.service_ms: mean total_ms of every answer on that backend (all its arms),
  used for the live queue term;
* chars_per_token: prompt characters / prompt tokens over all answers, so the router can
  estimate prompt tokens without a tokenizer.

Collect with --concurrency 1 (the default) so these are unloaded latencies; the router adds
the live queue on top.

    uv run python -m eval.fit_latency --run pilot        # -> runs/pilot/latency_profile.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from lucerouter.latency import CLASSES

from .common import BACKENDS_PATH, PROMPTS_PATH, RUNS_DIR, latest_by_key, load_arms, read_jsonl


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """(intercept, slope >= 0) of y on x; falls back to (median y, 0)."""
    if len(xs) >= 2 and len(set(xs)) >= 2:
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        if slope >= 0:
            return max(0.0, my - slope * mx), slope
    return statistics.median(ys), 0.0


def _usable(row: dict[str, Any]) -> bool:
    return (not row.get("error") and row.get("total_ms") is not None and row.get("ttft_ms") is not None
            and row.get("completion_tokens"))


def fit_profile(answers: list[dict[str, Any]], prompts: dict[str, dict[str, Any]],
                arms: dict[str, dict[str, Any]], complexity: dict[str, dict[str, float]] | None = None,
                ids: set[str] | None = None) -> dict[str, Any]:
    """Latency profile JSON (the format lucerouter.latency.LatencyProfile reads)."""
    from lucerouter.types import RouteRequest

    rows = [r for r in latest_by_key(answers).values()
            if _usable(r) and r["id"] in prompts and (ids is None or r["id"] in ids)]
    complexity = complexity or {}
    out: dict[str, Any] = {"arms": {}, "backends": {}}
    chars = toks = 0
    for r in rows:
        if r.get("prompt_tokens"):
            chars += RouteRequest(messages=prompts[r["id"]]["messages"]).prompt_chars()
            toks += r["prompt_tokens"]
    out["chars_per_token"] = round(chars / toks, 3) if toks else None
    for arm, spec in arms.items():
        mine = [r for r in rows if r["model"] == spec["model"] and bool(r["thinking"]) == spec["thinking"]]
        if not mine:
            continue
        with_pt = [r for r in mine if r.get("prompt_tokens") is not None]
        a, b = _ols([float(r["prompt_tokens"]) for r in with_pt], [float(r["ttft_ms"]) for r in with_pt]) \
            if with_pt else (statistics.median(r["ttft_ms"] for r in mine), 0.0)
        tpots = [(r["total_ms"] - r["ttft_ms"]) / (r["completion_tokens"] - 1)
                 for r in mine if r["completion_tokens"] > 1]
        out_tokens = {"default": round(statistics.fmean(r["completion_tokens"] for r in mine), 1)}
        weighted = [r for r in mine if r["id"] in complexity]
        if weighted:
            for c in CLASSES:
                w = sum(complexity[r["id"]][c] for r in weighted)
                if w > 1e-9:
                    out_tokens[c] = round(sum(complexity[r["id"]][c] * r["completion_tokens"]
                                              for r in weighted) / w, 1)
        out["arms"][arm] = {"ttft_ms": round(a, 2), "ttft_ms_per_token": round(b, 5),
                            "tpot_ms": round(statistics.median(tpots), 3) if tpots else 0.0,
                            "out_tokens": out_tokens, "n": len(mine)}
    for model in {spec["model"] for spec in arms.values()}:
        totals = [r["total_ms"] for r in rows if r["model"] == model]
        if totals:
            out["backends"][model] = {"service_ms": round(statistics.fmean(totals), 1), "n": len(totals)}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--prompts", type=Path, default=PROMPTS_PATH)
    ap.add_argument("--config", type=Path, default=BACKENDS_PATH)
    ap.add_argument("--split", choices=["train", "test", "all"], default="train")
    ap.add_argument("--service", default="brick-max", help="Brick service whose cached probs weight out_tokens")
    ap.add_argument("--out", type=Path, default=None, help="default runs/<name>/latency_profile.json")
    args = ap.parse_args(argv)
    if not (args.run or args.run_dir):
        ap.error("--run or --run-dir required")
    run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / args.run
    prompts = {p["id"]: p for p in read_jsonl(args.prompts)}
    ids = {i for i, p in prompts.items() if args.split == "all" or p["split"] == args.split}
    complexity = {r["id"]: r["probs"] for r in read_jsonl(run_dir / "brick_probs.jsonl")
                  if r.get("service") == args.service and r.get("probs") and not r.get("error")}
    profile = fit_profile(read_jsonl(run_dir / "answers.jsonl"), prompts, load_arms(args.config),
                          complexity, ids)
    profile.update(run=str(run_dir), split=args.split, complexity_service=args.service if complexity else None)
    out = args.out or run_dir / "latency_profile.json"
    out.write_text(json.dumps(profile, indent=2) + "\n")
    for arm, a in profile["arms"].items():
        print(f"  {arm:<18} ttft {a['ttft_ms']:.0f} ms + {a['ttft_ms_per_token']:.3f}/tok, "
              f"tpot {a['tpot_ms']:.1f} ms, out {a['out_tokens']} (n={a['n']})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
