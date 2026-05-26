#!/usr/bin/env python3
"""
Multi-hop QA bench: ee7 at 32K/64K/128K.
Tests whether keep_ratio=0.05 + EARLY_EXIT_N=7 preserves multi-hop reasoning
across spatially-separated chain links.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import requests

REPO = Path(__file__).resolve().parents[2]
SERVER_BIN = REPO / "dflash/build/dflash_server"
TARGET = Path("/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf")
DRAFTER = Path("/home/peppi/models/Qwen3-0.6B-BF16.gguf")
PORT = 18099
BASE_URL = f"http://127.0.0.1:{PORT}"

CONTEXTS = [32768, 65536, 131072]
CONDITION_SPECS = {"ee7": (7, 7)}  # (EARLY_EXIT_N, SCORE_LAYERS)


def start_server(condition, log_path, compression_mode="always", keep_ratio=0.05):
    # server accepts off|auto|always; allow "none" as alias for "off"
    srv_compression = "off" if compression_mode == "none" else compression_mode
    max_ctx = 139264
    env = os.environ.copy()
    env["GGML_CUDA_NO_VMM"] = "1"
    env["DFLASH27B_KV_K"] = "tq3_0"
    env["DFLASH27B_KV_V"] = "tq3_0"
    env.pop("PFLASH_DRAFTER_EARLY_EXIT_N", None)
    env.pop("PFLASH_DRAFTER_SCORE_LAYERS", None)
    if condition in CONDITION_SPECS:
        ee_n, sl = CONDITION_SPECS[condition]
        env["PFLASH_DRAFTER_EARLY_EXIT_N"] = str(ee_n)
        env["PFLASH_DRAFTER_SCORE_LAYERS"] = str(sl)
    for k in ("DFLASH_COMPRESS_ANCHOR_TRANSITIVE", "DFLASH_COMPRESS_ANCHOR_MAX_ITERS",
              "DFLASH_COMPRESS_RARE_MAX_FREQ", "DFLASH_COMPRESS_ANCHOR_NGRAM",
              "DFLASH_COMPRESS_CASCADE_MIN_ANCHOR_FRAC", "DFLASH_COMPRESS_MAX_FORCED_RATIO"):
        if k in os.environ:
            env[k] = os.environ[k]

    cmd = [
        str(SERVER_BIN), str(TARGET),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-ctx", str(max_ctx),
        "--prefill-compression", srv_compression,
        "--prefill-keep-ratio", str(keep_ratio),
        "--prefill-drafter", str(DRAFTER),
    ]
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=f, env=env)
    return proc


def wait_server(proc, timeout=180):
    for _ in range(timeout):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
        if proc.poll() is not None:
            return False
    return False


def stop_server(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    time.sleep(3)


def extract_metrics_from_log(log_path):
    metrics = {"drafter_fwd_s": None, "tail_score_s": None}
    ggml_assert_count = 0
    try:
        with open(log_path) as f:
            for line in f:
                m = re.search(r"\[drafter\]\s+forward\+score in ([\d.]+)s", line)
                if m:
                    metrics["drafter_fwd_s"] = float(m.group(1))
                m2 = re.search(r"tail.?score\s+([\d.]+)s", line, re.IGNORECASE)
                if m2:
                    metrics["tail_score_s"] = float(m2.group(1))
                if "ggml_view_3d" in line and "assert" in line.lower():
                    ggml_assert_count += 1
    except Exception:
        pass
    metrics["ggml_assert_count"] = ggml_assert_count
    return metrics


def score_response(answer: int, response_text: str) -> bool:
    return bool(re.search(rf"\b{answer}\b", response_text))


def run_one_case(condition, ctx, case, case_idx, results_dir, compression_mode="always", keep_ratio=0.05):
    log_path = results_dir / f"{condition}_{ctx}_case{case_idx}_server.log"
    proc = start_server(condition, log_path, compression_mode=compression_mode, keep_ratio=keep_ratio)
    result = {
        "latency_s": None, "text": "", "passed": False, "error": None,
        "drafter_fwd_s": None, "tail_score_s": None, "ggml_assert_count": 0,
    }
    try:
        if not wait_server(proc, timeout=180):
            tail = ""
            try:
                with open(log_path) as f:
                    tail = "".join(f.readlines()[-30:])
            except Exception:
                pass
            result["error"] = f"server_start_failed: {tail[:500]}"
            return result

        payload = {
            "model": "dflash",
            "messages": [{"role": "user", "content": case["prompt"]}],
            "max_tokens": 64,
            "stream": False,
            "temperature": 0.0,
        }
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=600)
            result["latency_s"] = time.perf_counter() - t0
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            result["text"] = text[:300]
            result["passed"] = score_response(case["answer"], text)
        except Exception as e:
            result["latency_s"] = time.perf_counter() - t0
            result["error"] = str(e)
    finally:
        stop_server(proc)

    m = extract_metrics_from_log(log_path)
    result.update(m)
    return result


def run_condition_ctx(condition, ctx, cases, results_dir, compression_mode="always", keep_ratio=0.05):
    print(f"\n[bench] condition={condition} ctx={ctx} ({len(cases)} cases, one server per case)", flush=True)
    case_results = []
    for i, case in enumerate(cases):
        ans = case.get("answer", "?")
        print(f"  case {i}: ntok={case.get('n_tokens', ctx)} ans={ans} hops={case.get('hops', '?')}", flush=True)
        r = run_one_case(condition, ctx, case, i, results_dir, compression_mode=compression_mode, keep_ratio=keep_ratio)
        case_results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        drafter_s = f"{r['drafter_fwd_s']:.3f}s" if r["drafter_fwd_s"] else "N/A"
        latency_s = f"{r['latency_s']:.2f}s" if r["latency_s"] is not None else "N/A"
        print(f"  case {i}: latency={latency_s} drafter={drafter_s} [{status}] resp={r['text'][:80]!r}", flush=True)
        if r["error"]:
            print(f"  case {i}: error={r['error'][:200]}", flush=True)
        if r.get("ggml_assert_count", 0) > 0:
            print(f"  [STOP] ggml_view_3d assert detected in case {i} log — Bug #42 regressed!", flush=True)
            sys.exit(1)

    drafter_times = [r["drafter_fwd_s"] for r in case_results if r["drafter_fwd_s"] is not None]
    pass_count = sum(1 for r in case_results if r["passed"])

    if pass_count == 0 and len(case_results) > 0:
        print(f"\n[STOP] ee7 at ctx={ctx} scored 0/{len(cases)} — degradation IS the finding.", flush=True)
        print("  Example responses:", flush=True)
        for i, r in enumerate(case_results[:2]):
            print(f"  case {i}: {r['text']!r}", flush=True)
        # Don't exit — let caller emit the summary and report

    return {
        "condition": condition,
        "ctx": ctx,
        "case_results": case_results,
        "drafter_times_s": drafter_times,
        "drafter_p50_s": median(drafter_times) if drafter_times else None,
        "pass_count": pass_count,
        "total": len(cases),
        "ggml_asserts": sum(r.get("ggml_assert_count", 0) for r in case_results),
    }


def write_summary(all_results, results_dir, compression_mode="always", keep_ratio=0.05):
    lines = []
    lines.append("# ee7 Multi-Hop QA Bench (RULER-style)\n")
    lines.append(f"Config: compression_mode={compression_mode} keep_ratio={keep_ratio} EARLY_EXIT_N=7 SCORE_LAYERS=7\n\n")
    lines.append("## Results\n\n")
    lines.append("| ctx | pass/N | drafter_p50 | ggml_asserts |\n")
    lines.append("|---|---|---|---|\n")

    any_zero = False
    any_assert = False
    for r in all_results:
        ctx = r["ctx"]
        pn = f"{r['pass_count']}/{r['total']}"
        dp50 = f"{r['drafter_p50_s']:.3f}s" if r.get("drafter_p50_s") else "N/A"
        ga = r.get("ggml_asserts", 0)
        lines.append(f"| {ctx} | {pn} | {dp50} | {ga} |\n")
        if r["pass_count"] == 0:
            any_zero = True
        if ga > 0:
            any_assert = True

    lines.append("\n## Failure Examples\n\n")
    for r in all_results:
        if r["pass_count"] < r["total"]:
            lines.append(f"### ctx={r['ctx']}\n\n")
            for i, cr in enumerate(r["case_results"]):
                if not cr["passed"]:
                    lines.append(f"- case {i}: response={cr['text']!r}\n")

    lines.append("\n## Decision\n\n")
    if any_assert:
        lines.append("STOP: ggml_view_3d assert detected — Bug #42 regressed.\n")
    elif any_zero:
        lines.append("STOP: ee7 scored 0/N at one or more contexts — multi-hop degradation confirmed. Do NOT proceed to ee3 sweep without investigation.\n")
    else:
        lines.append("PROCEED: ee7 passes multi-hop at all tested contexts. ee3 sweep is next gate.\n")

    summary_path = results_dir / "SUMMARY.md"
    with open(summary_path, "w") as f:
        f.writelines(lines)
    print(f"\n[done] {summary_path}", flush=True)
    print("".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="dflash/bench/results/2026-05-25_ee7_multihop")
    ap.add_argument("--cases-dir", default="/tmp")
    ap.add_argument("--contexts", nargs="+", type=int, default=CONTEXTS)
    ap.add_argument("--compression-mode", default="always")
    ap.add_argument("--keep-ratio", type=float, default=0.05)
    args = ap.parse_args()

    results_dir = Path(args.out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cases_by_ctx = {}
    for ctx in args.contexts:
        f_path = Path(args.cases_dir) / f"multihop_{ctx}.jsonl"
        if not f_path.exists():
            sys.exit(f"[error] missing {f_path}")
        with open(f_path) as f:
            cases_by_ctx[ctx] = [json.loads(l) for l in f]
        print(f"[init] {len(cases_by_ctx[ctx])} cases for ctx={ctx}", flush=True)

    all_results = []
    for condition in CONDITION_SPECS:
        for ctx in args.contexts:
            result = run_condition_ctx(condition, ctx, cases_by_ctx[ctx], results_dir,
                                       compression_mode=args.compression_mode, keep_ratio=args.keep_ratio)
            all_results.append(result)
            with open(results_dir / "raw_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    write_summary(all_results, results_dir, compression_mode=args.compression_mode, keep_ratio=args.keep_ratio)


if __name__ == "__main__":
    main()
