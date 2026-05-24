#!/usr/bin/env python3
"""
N-sweep NIAH: baseline + ee3 + ee5 + ee7 at 32K / 64K / 128K.

Each condition sets PFLASH_DRAFTER_EARLY_EXIT_N / PFLASH_DRAFTER_SCORE_LAYERS in the
server environment. One server boot per (condition, context, case) triplet to avoid
the Bug #42 ggml_view_3d assert that fires when a single server sees >1 sequence length.

Conditions:
  baseline  -- no early-exit (full 28 drafter layers)
  ee3       -- EARLY_EXIT_N=3  SCORE_LAYERS=3
  ee5       -- EARLY_EXIT_N=5  SCORE_LAYERS=5
  ee7       -- EARLY_EXIT_N=7  SCORE_LAYERS=7  (production reference)

Usage:
  python3 dflash/bench/run_ee_n_sweep_niah.py [--out-dir PATH] [--cases-dir PATH] \
      [--conditions COND ...] [--contexts CTX ...]

Case files:
  {cases_dir}/niah_32768.jsonl
  {cases_dir}/niah_65536.jsonl
  {cases_dir}/niah_131072.jsonl

Generate them once via:
  python3 pflash/tests/niah_gen.py --context 32768  --n 3 -o /tmp/niah_32768.jsonl
  python3 pflash/tests/niah_gen.py --context 65536  --n 3 -o /tmp/niah_65536.jsonl
  python3 pflash/tests/niah_gen.py --context 131072 --n 3 -o /tmp/niah_131072.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import time
import re
import requests
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[2]
SERVER_BIN = REPO / "dflash/build/dflash_server"
TARGET = Path("/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf")
DRAFTER = Path("/home/peppi/models/Qwen3-0.6B-BF16.gguf")
PORT = 18099
BASE_URL = f"http://127.0.0.1:{PORT}"

# N=0 → baseline (unset, full layers)
CONDITION_SPECS = {
    "baseline": (0, 0),
    "ee3":      (3, 3),
    "ee5":      (5, 5),
    "ee7":      (7, 7),
}

DEFAULT_CONTEXTS = [32768, 65536, 131072]
DEFAULT_CONDITIONS = ["baseline", "ee3", "ee5", "ee7"]


def start_server(condition: str, ctx: int, log_path: Path):
    early_n, score_n = CONDITION_SPECS[condition]
    max_ctx = 139264
    env = os.environ.copy()
    env["GGML_CUDA_NO_VMM"] = "1"
    env["DFLASH27B_KV_K"] = "tq3_0"
    env["DFLASH27B_KV_V"] = "tq3_0"
    env.pop("PFLASH_DRAFTER_EARLY_EXIT_N", None)
    env.pop("PFLASH_DRAFTER_SCORE_LAYERS", None)
    if early_n > 0:
        env["PFLASH_DRAFTER_EARLY_EXIT_N"] = str(early_n)
        env["PFLASH_DRAFTER_SCORE_LAYERS"] = str(score_n)

    cmd = [
        str(SERVER_BIN), str(TARGET),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-ctx", str(max_ctx),
        "--prefill-compression", "always",
        "--prefill-keep-ratio", "0.05",
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


def extract_metrics(log_path: Path) -> dict:
    metrics = {
        "drafter_fwd_s": None,
        "tail_score_s": None,
        "a_compute_s": None,
        "fp_s": None,
    }
    try:
        with open(log_path) as f:
            for line in f:
                m = re.search(r"\[drafter\]\s+forward\+score in ([\d.]+)s", line)
                if m:
                    metrics["drafter_fwd_s"] = float(m.group(1))
                m2 = re.search(r"tail.?score\s+([\d.]+)s", line, re.IGNORECASE)
                if m2:
                    metrics["tail_score_s"] = float(m2.group(1))
                m3 = re.search(r"A_compute\s+([\d.]+)s", line)
                if m3:
                    metrics["a_compute_s"] = float(m3.group(1))
                m4 = re.search(r"\bFP\b\s+([\d.]+)s", line)
                if m4:
                    metrics["fp_s"] = float(m4.group(1))
    except Exception:
        pass
    return metrics


def run_one_case(condition: str, ctx: int, case: dict, case_idx: int, results_dir: Path) -> dict:
    log_path = results_dir / f"{condition}_{ctx}_case{case_idx}_server.log"
    proc = start_server(condition, ctx, log_path)
    result = {
        "ttft_s": None, "text": "", "found": False, "error": None,
        "drafter_fwd_s": None, "tail_score_s": None,
        "a_compute_s": None, "fp_s": None,
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
            result["ttft_s"] = time.perf_counter() - t0
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            result["text"] = text[:300]
            result["found"] = case["answer"] in text
        except Exception as e:
            result["ttft_s"] = time.perf_counter() - t0
            result["error"] = str(e)
    finally:
        stop_server(proc)

    result.update(extract_metrics(log_path))
    return result


def run_condition_ctx(condition: str, ctx: int, cases: list, results_dir: Path) -> dict:
    print(f"\n[sweep] condition={condition} ctx={ctx} ({len(cases)} cases)", flush=True)
    case_results = []
    for i, case in enumerate(cases):
        print(f"  case {i}: ans={case.get('answer', '?')}", flush=True)
        r = run_one_case(condition, ctx, case, i, results_dir)
        case_results.append(r)
        status = "OK" if r["found"] else "FAIL"
        drafter_s = f"{r['drafter_fwd_s']:.3f}s" if r["drafter_fwd_s"] else "N/A"
        ttft_s = f"{r['ttft_s']:.2f}s" if r["ttft_s"] is not None else "N/A"
        print(f"  case {i}: ttft={ttft_s} drafter={drafter_s} [{status}]", flush=True)
        if r.get("error"):
            print(f"  case {i}: error={r['error'][:200]}", flush=True)

    drafter_times = [r["drafter_fwd_s"] for r in case_results if r["drafter_fwd_s"]]
    tail_times = [r["tail_score_s"] for r in case_results if r["tail_score_s"]]
    ttfts = [r["ttft_s"] for r in case_results if r["ttft_s"] is not None]
    a_compute_times = [r["a_compute_s"] for r in case_results if r["a_compute_s"]]
    fp_times = [r["fp_s"] for r in case_results if r["fp_s"]]
    niah_pass = sum(1 for c in case_results if c["found"])

    return {
        "condition": condition,
        "ctx": ctx,
        "case_results": case_results,
        "drafter_p50_s": median(drafter_times) if drafter_times else None,
        "tail_score_p50_s": median(tail_times) if tail_times else None,
        "ttft_p50_s": median(ttfts) if ttfts else None,
        "a_compute_p50_s": median(a_compute_times) if a_compute_times else None,
        "fp_p50_s": median(fp_times) if fp_times else None,
        "niah_pass": niah_pass,
        "niah_total": len(cases),
    }


def main():
    ap = argparse.ArgumentParser(
        description="N-sweep NIAH bench: baseline/ee3/ee5/ee7 × 32K/64K/128K"
    )
    ap.add_argument("--out-dir", default="dflash/bench/results/2026-05-25_ee_n_sweep")
    ap.add_argument("--cases-dir", default="/tmp",
                    help="Directory containing niah_{ctx}.jsonl files")
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS,
                    choices=list(CONDITION_SPECS.keys()),
                    help="Conditions to run (default: all four)")
    ap.add_argument("--contexts", nargs="+", type=int, default=DEFAULT_CONTEXTS)
    args = ap.parse_args()

    results_dir = Path(args.out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cases_by_ctx: dict[int, list] = {}
    for ctx in args.contexts:
        f_path = Path(args.cases_dir) / f"niah_{ctx}.jsonl"
        if not f_path.exists():
            sys.exit(
                f"[error] missing {f_path}\n"
                f"Generate with: python3 pflash/tests/niah_gen.py --context {ctx} --n 3 -o {f_path}"
            )
        with open(f_path) as f:
            cases_by_ctx[ctx] = [json.loads(line) for line in f if line.strip()]
        print(f"[init] {len(cases_by_ctx[ctx])} cases for ctx={ctx}", flush=True)

    all_results = []
    for condition in args.conditions:
        for ctx in args.contexts:
            r = run_condition_ctx(condition, ctx, cases_by_ctx[ctx], results_dir)
            all_results.append(r)
            with open(results_dir / "raw_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    baseline_times = {
        r["ctx"]: r["drafter_p50_s"]
        for r in all_results
        if r["condition"] == "baseline" and r.get("drafter_p50_s")
    }

    print("\n=== SUMMARY TABLE ===")
    print(f"{'ctx':>7}  {'condition':>10}  {'drafter_p50':>12}  {'tail_score':>10}  {'NIAH':>6}  {'speedup':>8}")

    rows = []
    for r in all_results:
        ctx, cond = r["ctx"], r["condition"]
        dp50 = r.get("drafter_p50_s")
        ts50 = r.get("tail_score_p50_s")
        niah = f"{r.get('niah_pass', 0)}/{r.get('niah_total', 0)}"
        if dp50 and ctx in baseline_times and cond != "baseline":
            speedup = f"{baseline_times[ctx] / dp50:.2f}x"
        else:
            speedup = "1.00x" if cond == "baseline" else "N/A"
        dp50_s = f"{dp50:.3f}s" if dp50 else "N/A"
        ts50_s = f"{ts50:.3f}s" if ts50 else "N/A"
        print(f"{ctx:>7}  {cond:>10}  {dp50_s:>12}  {ts50_s:>10}  {niah:>6}  {speedup:>8}")
        rows.append({
            "ctx": ctx, "condition": cond,
            "drafter_fwd_p50": dp50_s, "tail_score": ts50_s,
            "NIAH": niah, "speedup": speedup,
        })

    with open(results_dir / "SUMMARY.md", "w") as f:
        f.write("# ee N-sweep NIAH: baseline / ee3 / ee5 / ee7 @ 32K / 64K / 128K\n\n")
        f.write("## Results\n\n")
        f.write("| ctx | condition | drafter_p50 | tail_score | NIAH | speedup_vs_baseline |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in rows:
            f.write(f"| {row['ctx']} | {row['condition']} | {row['drafter_fwd_p50']} "
                    f"| {row['tail_score']} | {row['NIAH']} | {row['speedup']} |\n")
        f.write("\n## Verdict\n\n_Filled in post-run._\n")

    print(f"\n[done] {results_dir}/SUMMARY.md", flush=True)


if __name__ == "__main__":
    main()
