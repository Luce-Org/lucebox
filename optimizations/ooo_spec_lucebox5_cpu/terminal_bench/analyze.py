#!/usr/bin/env python3
"""Aggregate paired control/spec results for a run tag."""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys
from datetime import datetime

ROOT = "/home/lucebox5/tbspec"


def parse_dt(value):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load(tag: str):
    rows = []
    for result_path in sorted(glob.glob(f"{ROOT}/jobs/{tag}__*__*/*/result.json")):
        job = result_path.split("/")[-3]
        _, task, arm = job.rsplit("__", 2)
        result = json.load(open(result_path))
        trial_dir = os.path.dirname(result_path)
        trace_path = os.path.join(trial_dir, "agent", "tbspec_trace.json")
        trace = json.load(open(trace_path)) if os.path.exists(trace_path) else {}
        ae = result.get("agent_execution") or {}
        started, finished = parse_dt(ae.get("started_at")), parse_dt(ae.get("finished_at"))
        agent_sec = (finished - started).total_seconds() if started and finished else None
        verifier = result.get("verifier_result") or {}
        rewards = verifier.get("rewards") or {}
        reward = None
        if isinstance(rewards, dict) and rewards:
            reward = list(rewards.values())[0]
        totals = trace.get("totals") or {}
        turns = trace.get("turns") or []
        pred_ms = [t.get("spec", {}).get("predictor_wall_ms") for t in turns if isinstance(t.get("spec"), dict) and t["spec"].get("predictor_wall_ms") is not None]
        rows.append({
            "task": task, "arm": arm, "reward": reward, "agent_sec": agent_sec,
            "wall_ms": trace.get("wall_ms"), "stop": trace.get("stop_reason"),
            "turns": len(turns), "tool_calls": totals.get("tool_calls"), "readonly": totals.get("readonly_calls"),
            "hits": totals.get("spec_hits"), "spec_status": totals.get("spec_status"),
            "model_ms": totals.get("model_wall_ms"), "tool_ms": totals.get("tool_wall_ms"),
            "hit_saved_ms": totals.get("hit_saved_ms"), "predictor_ms": totals.get("predictor_wall_ms"),
            "predictor_p50_ms": statistics.median(pred_ms) if pred_ms else None,
            "prompt_tokens": totals.get("prompt_tokens"), "completion_tokens": totals.get("completion_tokens"),
            "exception": (result.get("exception_info") or {}).get("exception_type"),
        })
    return rows


def main():
    tag = sys.argv[1]
    rows = load(tag)
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], {})[r["arm"]] = r
    print(f"{'task':32} {'arm':8} {'reward':>6} {'agent_s':>8} {'turns':>5} {'tools':>5} {'ro':>4} {'hits':>4} {'model_s':>8} {'tool_s':>7} {'pred_s':>7} {'ptok':>7} {'ctok':>6} stop/spec")
    for task, arms in sorted(by_task.items()):
        for arm in ("control", "spec"):
            r = arms.get(arm)
            if not r:
                continue
            f = lambda v, d=1: "-" if v is None else f"{v:.{d}f}"
            print(f"{task:32} {arm:8} {str(r['reward']):>6} {f(r['agent_sec']):>8} {r['turns']:>5} {str(r['tool_calls']):>5} {str(r['readonly']):>4} {str(r['hits']):>4} "
                  f"{f((r['model_ms'] or 0)/1000):>8} {f((r['tool_ms'] or 0)/1000):>7} {f((r['predictor_ms'] or 0)/1000):>7} {str(r['prompt_tokens']):>7} {str(r['completion_tokens']):>6} {r['stop']} {r['spec_status']} {r['exception'] or ''}")
    ratios = []
    print()
    for task, arms in sorted(by_task.items()):
        c, s = arms.get("control"), arms.get("spec")
        if c and s and c["agent_sec"] and s["agent_sec"]:
            ratio = c["agent_sec"] / s["agent_sec"]
            ratios.append(ratio)
            print(f"{task:32} control {c['agent_sec']:8.1f}s (r={c['reward']}, {c['turns']}t)  spec {s['agent_sec']:8.1f}s (r={s['reward']}, {s['turns']}t)  speedup x{ratio:.3f}  hits {s['hits']}/{s['readonly']}")
    if ratios:
        print(f"\npairs={len(ratios)} paired speedup median x{statistics.median(ratios):.3f}  mean x{statistics.mean(ratios):.3f}  min x{min(ratios):.3f}  max x{max(ratios):.3f}")
        cs = sum(arms['control']['agent_sec'] for arms in by_task.values() if 'control' in arms and 'spec' in arms and arms['control']['agent_sec'] and arms['spec']['agent_sec'])
        ss = sum(arms['spec']['agent_sec'] for arms in by_task.values() if 'control' in arms and 'spec' in arms and arms['control']['agent_sec'] and arms['spec']['agent_sec'])
        print(f"total control {cs:.0f}s  total spec {ss:.0f}s  aggregate x{cs/ss:.3f}")


if __name__ == "__main__":
    main()

