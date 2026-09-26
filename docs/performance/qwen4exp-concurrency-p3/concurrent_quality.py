#!/usr/bin/env python3
"""Run the existing client_test_runner cases/scorers in concurrent waves of four."""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "harness"))
import client_test_runner as ctr  # noqa: E402

URL = os.environ.get("P3_URL", "http://127.0.0.1:8935")
MODEL = "qwen4exp"
SUITES = ("he", "gsm", "math", "recall")


def score(suite: str, case: dict, result: dict) -> tuple[bool, str]:
    if not result.get("ok") or not result.get("text"):
        return False, result.get("error", f"HTTP {result.get('status', '?')}")
    text = result["text"]
    if suite == "he" and "gold_test" in case:
        return ctr._score_he_response(text, case["entry_point"], case["gold_test"])
    if suite == "gsm":
        return ctr._score_gsm_response(text, case["gold_answer"])
    if suite == "math":
        return ctr._score_math_response(text, case["gold_answer"])
    wanted = case.get("expect_contains", [])
    if isinstance(wanted, str):
        wanted = [wanted]
    misses = [s for s in wanted if s not in text]
    rx = case.get("expect_regex")
    rx_bad = bool(rx) and ctr.re.search(rx, text) is None
    return not misses and not rx_bad, f"missing={misses}" if misses or rx_bad else "correct"


def main() -> int:
    # Round-robin suites so even recall's two cases run alongside three other
    # active requests; every prompt is scored by the stock harness functions.
    loaded = {s: ctr._load_bench_prompts(s) for s in SUITES}
    cases = []
    for row in range(max(map(len, loaded.values()))):
        for suite in SUITES:
            if row < len(loaded[suite]):
                cases.append((suite, loaded[suite][row]))

    records = []
    for start in range(0, len(cases), 4):
        wave = cases[start : start + 4]
        barrier = threading.Barrier(len(wave))

        def run(item):
            suite, case = item
            barrier.wait()
            t0 = time.perf_counter()
            result = ctr._run_bench_case(URL, MODEL, case)
            result["client_wall_s"] = round(time.perf_counter() - t0, 3)
            correct, detail = score(suite, case, result)
            result.update(suite=suite, correct=correct, score_detail=detail)
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run, item) for item in wave]
            rows = [future.result() for future in futures]
        for result in rows:
            records.append(result)
            print(
                f"[concurrent-quality] {result['suite']} {result['id']} "
                f"ok={result.get('ok')} correct={result['correct']} "
                f"ttft={result.get('ttft_s')} wall={result.get('wall_s')} "
                f"detail={result['score_detail']}",
                flush=True,
            )

    summary = {}
    for suite in SUITES:
        rows = [r for r in records if r["suite"] == suite]
        ncorrect = sum(bool(r["correct"]) for r in rows)
        summary[suite] = {"correct": ncorrect, "total": len(rows)}
    payload = {"url": URL, "max_active": 4, "scoring": "client_test_runner.py", "summary": summary, "results": records}
    out = Path(os.environ.get("P3_QUALITY_JSON", "concurrent-quality.json"))
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("[concurrent-quality] summary=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if all(v["correct"] == v["total"] for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
