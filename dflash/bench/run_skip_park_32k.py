#!/usr/bin/env python3
"""
Task #47: ee7 + --prefill-skip-park empirical test at 32K.
Single condition. One server per case (ggml view_3d avoidance).
Decides whether park/unpark choreography can be eliminated at <=32K with ee7.
"""
import json, os, re, subprocess, sys, time
from pathlib import Path
import requests

REPO = Path(__file__).resolve().parents[2]
SERVER_BIN = REPO / "dflash/build/dflash_server"
TARGET = Path("/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf")
DRAFTER = Path("/home/peppi/models/Qwen3-0.6B-BF16.gguf")
NIAH_FILE = Path("/tmp/niah_32768.jsonl")
OUT = REPO / "dflash/bench/results/2026-05-22_skip_park_32k"
PORT = 18100
BASE_URL = f"http://127.0.0.1:{PORT}"

# ee7 baseline drafter_fwd at 32K from d3fbad3 (d7d476c bench)
EE7_PARK_BASELINE_S = 1.44


def start_server(log_path):
    env = os.environ.copy()
    env["GGML_CUDA_NO_VMM"] = "1"
    env["DFLASH_DRAFTER_EARLY_EXIT_N"] = "7"
    env["DFLASH_DRAFTER_SCORE_LAYERS"] = "7"
    cmd = [
        str(SERVER_BIN), str(TARGET),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-ctx", "36864",
        "--cache-type-k", "tq3_0",
        "--cache-type-v", "tq3_0",
        "--prefill-compression", "always",
        "--prefill-keep-ratio", "0.05",
        "--prefill-drafter", str(DRAFTER),
        "--prefill-skip-park",
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
    time.sleep(2)


def extract_log_metrics(log_path):
    metrics = {"drafter_fwd_s": None, "tail_score_s": None, "cuda_crash": False}
    try:
        with open(log_path) as f:
            content = f.read()
        # cuMemSetAccess crash
        if re.search(r"cuMemSetAccess|NOT_READY|CUDA_ERROR_NOT_READY", content):
            metrics["cuda_crash"] = True
        # [drafter] forward+score in X.XXXs
        m = re.search(r"\[drafter\]\s+forward\+score in ([\d.]+)s", content)
        if m:
            metrics["drafter_fwd_s"] = float(m.group(1))
        # tail-score
        m2 = re.search(r"tail.?score\s+([\d.]+)s", content, re.IGNORECASE)
        if m2:
            metrics["tail_score_s"] = float(m2.group(1))
    except Exception:
        pass
    return metrics


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    cases = []
    with open(NIAH_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    cases = cases[:3]
    print(f"[task47] loaded {len(cases)} NIAH cases from {NIAH_FILE}")

    # VRAM monitor
    vram_log = OUT / "vram.csv"
    vram_proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits", "--loop=1"],
        stdout=open(vram_log, "w"),
        stderr=subprocess.DEVNULL,
    )

    results = []
    cuda_crash_seen = False

    for idx, case in enumerate(cases):
        log_path = OUT / f"case{idx}_server.log"
        print(f"[task47] case {idx}: starting server...")
        proc = start_server(log_path)

        try:
            ok = wait_server(proc, timeout=180)
            if not ok:
                print(f"[task47] case {idx}: server start FAILED")
                metrics = extract_log_metrics(log_path)
                if metrics["cuda_crash"]:
                    print(f"[task47] VERDICT C: cuMemSetAccess crash on case {idx}")
                    cuda_crash_seen = True
                results.append({"pass": False, "answer": "", "expected": case["answer"],
                                 "drafter_fwd_s": None, "crash": True})
                continue

            # Check startup crash
            metrics = extract_log_metrics(log_path)
            if metrics["cuda_crash"]:
                print(f"[task47] VERDICT C: cuMemSetAccess crash during startup case {idx}")
                cuda_crash_seen = True
                results.append({"pass": False, "answer": "", "expected": case["answer"],
                                 "drafter_fwd_s": None, "crash": True})
                continue

            payload = {
                "model": "dflash",
                "messages": [{"role": "user", "content": case["prompt"]}],
                "max_tokens": 64,
                "stream": False,
                "temperature": 0.0,
            }
            t0 = time.perf_counter()
            try:
                r = requests.post(f"{BASE_URL}/v1/chat/completions",
                                   json=payload, timeout=600)
                elapsed = time.perf_counter() - t0
                data = r.json()
                text = data["choices"][0]["message"]["content"]
            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f"[task47] case {idx}: request error: {e}")
                text = ""

            expected = case["answer"]
            found = expected in text
            print(f"[task47] case {idx}: answer='{text[:80]}' expected={expected} "
                  f"PASS={found} wall={elapsed:.1f}s")

        finally:
            stop_server(proc)

        # Re-read metrics after request completes
        metrics = extract_log_metrics(log_path)
        results.append({
            "pass": found,
            "answer": text[:80],
            "expected": expected,
            "drafter_fwd_s": metrics["drafter_fwd_s"],
            "crash": False,
        })
        print(f"[task47] case {idx}: drafter_fwd={metrics['drafter_fwd_s']}s "
              f"tail_score={metrics['tail_score_s']}s")

    # Stop VRAM monitor
    vram_proc.terminate()
    vram_proc.wait()

    # Compute peak VRAM
    try:
        vals = [int(l.strip()) for l in open(vram_log) if l.strip().isdigit()]
        peak_mib = max(vals) if vals else 0
        peak_gb = peak_mib / 1024
    except Exception:
        peak_gb = 0.0

    # Aggregate
    n_pass = sum(1 for r in results if r["pass"])
    n_done = len(results)
    dfwds = [r["drafter_fwd_s"] for r in results if r["drafter_fwd_s"] is not None]
    mean_dfw = sum(dfwds) / len(dfwds) if dfwds else None

    mean_dfw_str = f"{mean_dfw:.2f}" if mean_dfw is not None else "N/A"
    print(f"\n[task47] RESULTS: NIAH {n_pass}/3, peak_vram={peak_gb:.1f}GB, "
          f"mean_drafter_fwd={mean_dfw_str}s")

    # Verdict
    if cuda_crash_seen:
        verdict = ("(C) cuMemSetAccess NOT_READY crash — historical skip_park bug recurs on 24 GB GPU,"
                   " choreography must stay")
    elif any(r["crash"] for r in results):
        verdict = "(C) Server crash mid-run — skip_park unsafe at 32K with ee7"
    elif n_pass >= 2 and peak_gb < 23.5 and mean_dfw is not None:
        verdict = ("(A) ee7 + skip-park works at 32K — recommend as opt-in config for <=32K workloads")
    else:
        verdict = ("(B) VRAM OOM or quality degradation — keep park/unpark at 32K")

    delta_str = "N/A"
    if mean_dfw is not None:
        d = mean_dfw - EE7_PARK_BASELINE_S
        pct = d / EE7_PARK_BASELINE_S * 100
        sign = "+" if d > 0 else ""
        delta_str = f"{sign}{d:.2f}s ({sign}{pct:.0f}%)"

    summary = f"""# ee7 + --prefill-skip-park 32K Experiment (Task #47)

Binary: d3fbad3 (layer-subset VRAM fix f157274 + guard bug fix d3fbad3)
GPU: NVIDIA GeForce RTX 3090 (24 GB)
Condition: ee7 (EARLY_EXIT_N=7, SCORE_LAYERS=7) + --prefill-skip-park + GGML_CUDA_NO_VMM=1
Context: 32768 tokens (niah_32768.jsonl, seeds from prior bench, ~32764 tok each)
prefill-keep-ratio=0.05, tq3_0 KV (--cache-type-k/v tq3_0)

## Results

| Metric | Value |
|---|---|
| Server startup | OK |
| Cases completed | {n_done} of 3 |
| NIAH retrieval | {n_pass}/3 |
| Peak VRAM | {peak_gb:.1f} GB |
| Mean drafter_fwd | {f"{mean_dfw:.2f}" if mean_dfw else "N/A"} s |
| ee7-with-park baseline (32K, d3fbad3) | {EE7_PARK_BASELINE_S:.2f} s |
| Delta vs baseline | {delta_str} |

## Per-case results
"""
    for i, r in enumerate(results):
        summary += f"- case {i}: NIAH={'PASS' if r['pass'] else 'FAIL'}, "
        summary += f"drafter_fwd={r['drafter_fwd_s']}s, answer='{r['answer'][:40]}'\n"

    summary += f"""
## Verdict

{verdict}
"""
    summary_path = OUT / "SUMMARY.md"
    summary_path.write_text(summary)
    print(summary)
    print(f"[task47] SUMMARY written to {summary_path}")
    return verdict


if __name__ == "__main__":
    main()
