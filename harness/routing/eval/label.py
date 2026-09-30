"""Join grades + answers into per-prompt, per-arm routing labels and a headroom report.

An arm is a (model, thinking) route target from config/backends.json "arms" (default:
qwen35-2b off, qwen38-27b off, qwen38-27b-think = the 27B with thinking on).

runs/<name>/labels.jsonl rows:

    {id, split, source, category, text, quality: {arm: score}, correct: {arm: bool},
     latency: {arm: total_ms}, ttft: {arm: ttft_ms}, cheapest_ok: arm|null, complete}

``cheapest_ok`` is the lowest-rank arm whose answer is correct (null if none); arm rank
orders arms by cost, which for the default arms is also the order of measured latency.
``complete`` is true when every requested arm has a grade for the prompt.
Also writes runs/<name>/headroom.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import PROMPTS_PATH, RUNS_DIR, latest_by_key, load_arms, read_jsonl, write_jsonl


def last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
            return str(c or "")
    return ""


def build_labels(
    prompts: list[dict[str, Any]],
    grades: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    arms: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """``arms`` ({arm: {"model", "thinking"}}) must be ordered cheapest first (rank order)."""
    g_by = latest_by_key(grades)
    a_by = latest_by_key(answers)
    models = list(arms)
    labels = []
    for p in prompts:
        quality, correct, latency, ttft = {}, {}, {}, {}
        for m in models:
            key = (p["id"], arms[m]["model"], bool(arms[m]["thinking"]))
            g = g_by.get(key)
            if g is None:
                continue
            quality[m] = float(g["score"])
            correct[m] = bool(g["correct"])
            a = a_by.get(key)
            if a is not None and a.get("total_ms") is not None:
                latency[m] = float(a["total_ms"])
                ttft[m] = a.get("ttft_ms")
        if not quality:
            continue
        cheapest = pool_cheapest_ok({"correct": correct}, models)
        labels.append({
            "id": p["id"], "split": p["split"], "source": p["source"], "category": p["category"],
            "text": last_user_text(p["messages"]),
            "quality": quality, "correct": correct, "latency": latency, "ttft": ttft,
            "cheapest_ok": cheapest, "complete": len(quality) == len(models),
        })
    return labels


# --------------------------------------------------------------------------- headroom


def _mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def pool_cheapest_ok(label: dict[str, Any], pool: list[str]) -> str | None:
    """Smallest model in ``pool`` (ordered smallest -> largest) whose answer was correct."""
    return next((m for m in pool if label["correct"].get(m)), None)


def oracle_choice(label: dict[str, Any], pool: list[str]) -> str:
    """Cheapest correct model in the pool; if none, the best-scoring (ties -> smaller)."""
    cheapest = pool_cheapest_ok(label, pool)
    if cheapest:
        return cheapest
    return max(pool, key=lambda m: (label["quality"].get(m, 0.0), -pool.index(m)))


def policy_stats(labels: list[dict[str, Any]], choose, models: list[str]) -> dict[str, Any]:
    q, lat = [], []
    for lab in labels:
        m = choose(lab)
        q.append(lab["quality"][m])
        if m in lab["latency"]:
            lat.append(lab["latency"][m])
    return {"quality": _mean(q), "mean_latency_ms": _mean(lat), "median_latency_ms": _median(lat),
            "n": len(labels)}


def outcome_shares(labels: list[dict[str, Any]], models: list[str]) -> dict[str, float]:
    """Share of prompts per model being ok, plus 'only largest ok' and 'none ok'."""
    n = len(labels) or 1
    out = {f"{m}_ok": sum(lab["correct"].get(m, False) for lab in labels) / n for m in models}
    largest = models[-1]
    out[f"only_{largest}_ok"] = sum(
        lab["correct"].get(largest, False) and not any(lab["correct"].get(m) for m in models[:-1])
        for lab in labels) / n
    out["none_ok"] = sum(not any(lab["correct"].values()) for lab in labels) / n
    for m in models:
        out[f"cheapest_is_{m}"] = sum(lab["cheapest_ok"] == m for lab in labels) / n
    return out


def headroom(labels: list[dict[str, Any]], models: list[str]) -> dict[str, Any]:
    complete = [lab for lab in labels if lab["complete"]]
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lab in complete:
        by_cat[lab["category"]].append(lab)
    lat = {m: [lab["latency"][m] for lab in complete if m in lab["latency"]] for m in models}
    ttft = {m: [lab["ttft"][m] for lab in complete if lab["ttft"].get(m) is not None] for m in models}
    policies = {"oracle": policy_stats(complete, lambda lab: oracle_choice(lab, models), models)}
    for m in models:
        policies[f"always_{m}"] = policy_stats(complete, lambda lab, m=m: m, models)
    return {
        "models": models,
        "n_prompts": len(labels),
        "n_complete": len(complete),
        "overall": outcome_shares(complete, models),
        "by_category": {c: {"n": len(v), **outcome_shares(v, models)} for c, v in sorted(by_cat.items())},
        "latency": {m: {"mean_ms": _mean(lat[m]), "median_ms": _median(lat[m]),
                        "mean_ttft_ms": _mean(ttft[m])} for m in models},
        "policies": policies,
    }


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _ms(x: float | None) -> str:
    return "-" if x is None else f"{x:.0f}"


def headroom_markdown(h: dict[str, Any]) -> str:
    models = h["models"]
    largest = models[-1]
    lines = [f"# Routing headroom\n\n{h['n_complete']} prompts with grades for every arm "
             f"({h['n_prompts']} labelled). Columns are arms (model + thinking mode).\n", "## Who answers acceptably\n"]
    cols = [f"{m}_ok" for m in models] + [f"only_{largest}_ok", "none_ok"]
    lines.append("| category | n | " + " | ".join(cols) + " |")
    lines.append("|---" * (len(cols) + 2) + "|")
    lines.append("| **all** | " + str(h["n_complete"]) + " | "
                 + " | ".join(_pct(h["overall"][c]) for c in cols) + " |")
    for cat, row in h["by_category"].items():
        lines.append(f"| {cat} | {row['n']} | " + " | ".join(_pct(row[c]) for c in cols) + " |")
    lines += ["", "## Latency per arm (total request time)\n",
              "| arm | mean ms | median ms | mean TTFT ms |", "|---|---|---|---|"]
    for m in models:
        L = h["latency"][m]
        lines.append(f"| {m} | {_ms(L['mean_ms'])} | {_ms(L['median_ms'])} | {_ms(L['mean_ttft_ms'])} |")
    lines += ["", "## Policies\n", "| policy | quality | mean latency ms | median latency ms |",
              "|---|---|---|---|"]
    for name, p in h["policies"].items():
        lines.append(f"| {name} | {_pct(p['quality'])} | {_ms(p['mean_latency_ms'])} | "
                     f"{_ms(p['median_latency_ms'])} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--prompts", type=Path, default=PROMPTS_PATH)
    ap.add_argument("--backends", default=None)
    ap.add_argument("--arms", default=None, help="comma list of arms, cheapest first (default: config arms by rank)")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / (args.run or "")
    if not (args.run or args.run_dir):
        ap.error("--run or --run-dir required")
    arms = load_arms(args.backends)
    if args.arms:
        arms = {a: arms[a] for a in args.arms.split(",")}
    models = list(arms)
    labels = build_labels(read_jsonl(args.prompts), read_jsonl(run_dir / "grades.jsonl"),
                          read_jsonl(run_dir / "answers.jsonl"), arms)
    write_jsonl(run_dir / "labels.jsonl", labels)
    h = headroom(labels, models)
    (run_dir / "headroom.json").write_text(json.dumps(h, indent=2) + "\n")
    md = headroom_markdown(h)
    (run_dir / "headroom.md").write_text(md)
    print(md)
    print(f"wrote {run_dir / 'labels.jsonl'} ({len(labels)} rows), headroom.md, headroom.json")


if __name__ == "__main__":
    main()
