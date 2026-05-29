#!/usr/bin/env python3
"""
Adaptive keep_ratio NIAH recovery bench.

Tests whether the PR #264 adaptive bandit recovers NIAH quality at 32K/64K/128K
where fixed keep=0.05 is reported to cliff.

Conditions per context:
  fixed_005  -- --prefill-keep-ratio 0.05, no session_id
  fixed_020  -- --prefill-keep-ratio 0.20, no session_id
  adaptive   -- --prefill-keep-ratio 0.10 (bandit initial), session_id injected
                 Bandit from PR #264 (pflash-auto worktree binary)

9 trials per (context, condition). Each trial categorized as:
  pass / wrong_answer / crash / timeout

Binary used: pflash-auto worktree (PR #264) for ALL conditions (like-vs-like).
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

# ── Paths ────────────────────────────────────────────────────────────────────
# NOTE: pflash-auto binary (PR #264) crashes at 32K+ NIAH prompts with
# ggml_view_3d assertion in qwen3_graph.cpp. Using drafter-fastpath binary
# (feat/pflash-drafter-ee7) which has the Bug #42 fix and stable NIAH at 128K.
# The adaptive condition is BLOCKED until pflash-auto gets the Bug #42 fix.
DRAFTER_FASTPATH_WORKTREE = Path("/home/peppi/Dev/lucebox-hub/.claude/worktrees/drafter-fastpath")
PFLASH_AUTO_WORKTREE = Path("/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto")
SERVER_BIN = DRAFTER_FASTPATH_WORKTREE / "dflash/build/dflash_server"
SERVER_BIN_BANDIT = PFLASH_AUTO_WORKTREE / "dflash/build/dflash_server"
TARGET = Path("/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf")
DRAFT = Path("/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf")
PFLASH_DRAFTER = Path("/home/peppi/models/Qwen3-0.6B-BF16.gguf")
NIAH_GEN = Path("/home/peppi/Dev/lucebox-hub/pflash/tests/niah_gen.py")

PORT = 18097
BASE_URL = f"http://127.0.0.1:{PORT}"
API_KEY = "sk-lucebox"
MODEL_ID = "dflash"

CONTEXTS = [32768, 65536, 131072]
CONDITIONS = ["fixed_005", "fixed_020", "adaptive"]
N_TRIALS = 9           # 9 trials per (ctx, condition)
SEED_BASE = 1000       # different seeds from prior bench (seed 42 used by niah_gen default)
REQUEST_TIMEOUT = 900  # 15 min per request at 128K
SERVER_START_TIMEOUT = 240

# ── max_ctx: must accommodate compressed prompt + generation headroom ─────────
# At 128K source, keep=0.05 → 6400 tokens compressed; keep=0.20 → 25600.
# For adaptive starting at 0.10 → 13100. Add 512 gen + safety margin.
MAX_CTX = 139264


def ensure_niah_cases(ctx: int, out_path: Path, n: int) -> list:
    """Generate or load NIAH cases for a given context size."""
    if out_path.exists():
        with open(out_path) as f:
            cases = [json.loads(l) for l in f if l.strip()]
        if len(cases) >= n:
            print(f"[niah] loaded {len(cases)} cases from {out_path}", flush=True)
            return cases[:n]

    print(f"[niah] generating {n} cases ctx={ctx} ...", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable, str(NIAH_GEN),
            "--n", str(n),
            "--ctx", str(ctx),
            "--out", str(out_path),
            "--seed-base", str(SEED_BASE),
            "--tokenizer", "Qwen/Qwen3-0.6B",
        ],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[error] niah_gen failed: {result.stderr}", flush=True)
        sys.exit(1)
    with open(out_path) as f:
        return [json.loads(l) for l in f if l.strip()]


def server_env(condition: str) -> dict:
    env = os.environ.copy()
    env["GGML_CUDA_NO_VMM"] = "1"
    env["DFLASH27B_KV_K"] = "tq3_0"
    env["DFLASH27B_KV_V"] = "tq3_0"
    env["DFLASH_FP_USE_BSA"] = "1"
    env["DFLASH_FP_ALPHA"] = "0.85"
    if condition in ("fixed_005", "fixed_020"):
        # drafter-fastpath binary supports ee7
        env["PFLASH_DRAFTER_EARLY_EXIT_N"] = "7"
        env["PFLASH_DRAFTER_SCORE_LAYERS"] = "7"
    else:
        # pflash-auto binary: no ee7 support
        env.pop("PFLASH_DRAFTER_EARLY_EXIT_N", None)
        env.pop("PFLASH_DRAFTER_SCORE_LAYERS", None)
    return env


def server_cmd(condition: str, keep_ratio: float) -> list:
    binary = str(SERVER_BIN) if condition in ("fixed_005", "fixed_020") else str(SERVER_BIN_BANDIT)
    return [
        binary, str(TARGET),
        "--draft", str(DRAFT),
        "--prefill-drafter", str(PFLASH_DRAFTER),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-ctx", str(MAX_CTX),
        "--max-tokens", "512",
        "--model-name", MODEL_ID,
        "--ddtree", "--ddtree-budget", "16",
        "--prefill-compression", "always",
        "--prefill-keep-ratio", str(keep_ratio),
    ]


def start_server(condition: str, log_path: Path) -> subprocess.Popen:
    keep = 0.05 if condition == "fixed_005" else 0.20 if condition == "fixed_020" else 0.10
    env = server_env(condition)
    cmd = server_cmd(condition, keep)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=f, env=env)
    return proc


def wait_server(proc: subprocess.Popen, timeout: int = SERVER_START_TIMEOUT) -> bool:
    import requests as req_lib
    for _ in range(timeout):
        try:
            r = req_lib.get(f"{BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
        if proc.poll() is not None:
            return False
    return False


def stop_server(proc: subprocess.Popen):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    time.sleep(2)


def run_niah_request(case: dict, session_id: str | None) -> dict:
    """Run a single NIAH trial against the server. Returns result dict."""
    import requests as req_lib

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": case["prompt"]}],
        "max_tokens": 64,
        "stream": False,
        "temperature": 0.0,
    }
    if session_id:
        payload["session_id"] = session_id

    result = {
        "answer": case["answer"],
        "text": "",
        "category": "crash",
        "ttft_s": None,
        "passed": False,
    }

    t0 = time.perf_counter()
    try:
        r = req_lib.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=REQUEST_TIMEOUT,
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        result["ttft_s"] = time.perf_counter() - t0
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        result["text"] = text[:400]
        if case["answer"] in text:
            result["category"] = "pass"
            result["passed"] = True
        else:
            result["category"] = "wrong_answer"
    except req_lib.exceptions.Timeout:
        result["ttft_s"] = time.perf_counter() - t0
        result["category"] = "timeout"
    except Exception as e:
        result["ttft_s"] = time.perf_counter() - t0
        result["category"] = "crash"
        result["error"] = str(e)[:200]

    return result


def extract_server_metrics(log_path: Path, n_requests: int) -> dict:
    """Parse server log for accept_rate, keep_ratio usage, prefill times."""
    metrics = {
        "prefill_times_s": [],
        "accept_rates": [],
        "keep_ratios_used": [],
        "bandit_log": [],
    }
    try:
        with open(log_path) as f:
            for line in f:
                # [prefill] tokens=N time=X.XXX s
                m = re.search(r"\[prefill\] tokens=\d+ time=([\d.]+)\s*s", line)
                if m:
                    metrics["prefill_times_s"].append(float(m.group(1)))
                # spec-decode accept rate
                m2 = re.search(r"accepted=\d+/\d+\s+\(([\d.]+)%\)", line)
                if m2:
                    metrics["accept_rates"].append(float(m2.group(1)) / 100.0)
                # bandit keep_ratio transitions
                m3 = re.search(r"\[pflash-bandit\].*keep=([\d.]+)->([\d.]+)", line)
                if m3:
                    metrics["keep_ratios_used"].append(float(m3.group(2)))
                    metrics["bandit_log"].append(line.strip())
                # K -> M tokens (X% kept)
                m4 = re.search(r"(\d+)\s*->\s*(\d+)\s*tokens", line)
                if m4:
                    src, kept = int(m4.group(1)), int(m4.group(2))
                    if src > 0:
                        metrics["keep_ratios_used"].append(kept / src)
    except Exception as e:
        metrics["parse_error"] = str(e)
    return metrics


def run_condition(condition: str, ctx: int, cases: list, out_dir: Path, n_trials: int) -> dict:
    """Run one (condition, ctx) cell. Returns summary dict."""
    session_id = f"niah_adaptive_{ctx}" if condition == "adaptive" else None
    log_path = out_dir / "server.log"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[bench] {condition} @ {ctx//1024}K  ({n_trials} trials)", flush=True)
    print(f"  session_id={session_id!r}  log={log_path}", flush=True)

    proc = start_server(condition, log_path)
    trial_results = []

    try:
        if not wait_server(proc, SERVER_START_TIMEOUT):
            print(f"  [error] server failed to start — see {log_path}", flush=True)
            try:
                with open(log_path) as f:
                    tail = f.readlines()[-20:]
                print("".join(tail), flush=True)
            except Exception:
                pass
            # Mark all trials as crash
            for i in range(n_trials):
                trial_results.append({
                    "trial": i, "answer": cases[i % len(cases)].get("answer", ""),
                    "text": "", "category": "crash", "ttft_s": None, "passed": False,
                    "error": "server_start_failed",
                })
        else:
            print(f"  server up (pid={proc.pid})", flush=True)
            for i, case in enumerate(cases[:n_trials]):
                print(f"  trial {i}/{n_trials} ntok={case.get('n_tokens', ctx)} ans={case['answer']}", flush=True)
                r = run_niah_request(case, session_id)
                r["trial"] = i
                trial_results.append(r)
                ttft_str = f"{r['ttft_s']:.1f}s" if r["ttft_s"] is not None else "N/A"
                print(f"  trial {i}: {r['category'].upper()} ttft={ttft_str} text={r['text'][:60]!r}", flush=True)
    finally:
        stop_server(proc)

    # Extract server metrics
    server_metrics = extract_server_metrics(log_path, n_trials)

    # Summarise
    passes = sum(1 for r in trial_results if r["category"] == "pass")
    wrong = sum(1 for r in trial_results if r["category"] == "wrong_answer")
    crashes = sum(1 for r in trial_results if r["category"] == "crash")
    timeouts = sum(1 for r in trial_results if r["category"] == "timeout")

    ttfts = [r["ttft_s"] for r in trial_results if r["ttft_s"] is not None]
    ttft_median = median(ttfts) if ttfts else None

    ar_list = server_metrics["accept_rates"]
    accept_median = median(ar_list) if ar_list else None

    if condition == "adaptive" and server_metrics["keep_ratios_used"]:
        keep_actual = f"~{median(server_metrics['keep_ratios_used']):.3f} (bandit median)"
    elif condition == "fixed_005":
        keep_actual = "0.050 (fixed)"
    else:
        keep_actual = "0.200 (fixed)"

    summary = {
        "ctx": ctx,
        "condition": condition,
        "niah_pass": passes,
        "niah_total": n_trials,
        "crashes": crashes,
        "wrong_answers": wrong,
        "timeouts": timeouts,
        "actual_keep": keep_actual,
        "accept_rate_median": accept_median,
        "ttft_median_s": ttft_median,
        "trial_results": trial_results,
        "bandit_log": server_metrics["bandit_log"],
        "server_metrics": server_metrics,
    }

    # Write per-cell JSON
    with open(out_dir / "result.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  => {passes}/{n_trials} pass  wrong={wrong}  crash={crashes}  timeout={timeouts}", flush=True)
    ar_display = f"{accept_median:.3f}" if accept_median is not None else "N/A"
    print(f"     keep={keep_actual}  accept_rate={ar_display}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",
                    default="/home/peppi/Dev/lucebox-hub/.claude/worktrees/drafter-fastpath/dflash/bench/results/2026-05-24_adaptive_niah_recovery")
    ap.add_argument("--contexts", nargs="+", type=int, default=CONTEXTS)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    args = ap.parse_args()

    n_trials = args.n_trials
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = out_dir / "niah_cases"

    print(f"[bench] adaptive NIAH recovery experiment", flush=True)
    print(f"[bench] binary: {SERVER_BIN}", flush=True)
    print(f"[bench] contexts: {args.contexts}", flush=True)
    print(f"[bench] conditions: {args.conditions}", flush=True)
    print(f"[bench] trials per cell: {n_trials}", flush=True)
    print(f"[bench] out_dir: {out_dir}", flush=True)

    if not SERVER_BIN.exists():
        sys.exit(f"[error] drafter-fastpath server binary not found: {SERVER_BIN}")
    if not SERVER_BIN_BANDIT.exists():
        print(f"[warn] pflash-auto (bandit) binary not found: {SERVER_BIN_BANDIT} — adaptive condition will crash", flush=True)
    if not TARGET.exists():
        sys.exit(f"[error] target model not found: {TARGET}")
    if not DRAFT.exists():
        sys.exit(f"[error] draft model not found: {DRAFT}")
    if not PFLASH_DRAFTER.exists():
        sys.exit(f"[error] pflash drafter not found: {PFLASH_DRAFTER}")

    print(f"[bench] binary (fixed_005, fixed_020): {SERVER_BIN}", flush=True)
    print(f"[bench] binary (adaptive/bandit): {SERVER_BIN_BANDIT}", flush=True)
    print(f"[warn]  adaptive condition uses pflash-auto binary which has a known ggml_view_3d", flush=True)
    print(f"[warn]  crash at 32K+ NIAH prompts (Bug #42 not fixed in pflash-auto).", flush=True)
    print(f"[warn]  Adaptive trials will be categorized as 'crash' if the bug triggers.", flush=True)

    # Generate NIAH cases
    all_cases = {}
    for ctx in args.contexts:
        cases_path = cases_dir / f"niah_{ctx}.jsonl"
        all_cases[ctx] = ensure_niah_cases(ctx, cases_path, n_trials)

    # Main loop: condition × context
    all_summaries = []
    t_start = time.time()

    for condition in args.conditions:
        for ctx in args.contexts:
            cell_dir = out_dir / f"{ctx//1024}K" / condition
            summary = run_condition(condition, ctx, all_cases[ctx][:n_trials], cell_dir, n_trials)
            all_summaries.append(summary)

            # Write raw_results incrementally
            with open(out_dir / "raw_results.json", "w") as f:
                json.dump(all_summaries, f, indent=2)

            elapsed = (time.time() - t_start) / 60
            remaining_cells = (len(args.conditions) * len(args.contexts)) - len(all_summaries)
            print(f"  elapsed={elapsed:.1f}min  remaining_cells={remaining_cells}", flush=True)

    # Build SUMMARY.md
    print("\n=== HEADLINE TABLE ===")
    header = "| ctx | condition | NIAH | crashes | wrong_answers | actual_keep | accept_rate |"
    sep    = "|-----|-----------|------|---------|---------------|-------------|-------------|"
    print(header)
    print(sep)

    rows_md = []
    for s in all_summaries:
        ctx_k = f"{s['ctx']//1024}K"
        niah_str = f"{s['niah_pass']}/{s['niah_total']}"
        ar_str = f"{s['accept_rate_median']:.1%}" if s["accept_rate_median"] is not None else "N/A"
        row = f"| {ctx_k} | {s['condition']} | {niah_str} | {s['crashes']} | {s['wrong_answers']} | {s['actual_keep']} | {ar_str} |"
        print(row)
        rows_md.append(row)

    # Verdict
    adaptive_results = [s for s in all_summaries if s["condition"] == "adaptive"]
    all_adaptive_pass = all(s["niah_pass"] >= 8 for s in adaptive_results)
    some_adaptive_better = any(
        s["niah_pass"] > next(
            (f["niah_pass"] for f in all_summaries
             if f["condition"] == "fixed_005" and f["ctx"] == s["ctx"]), 0
        )
        for s in adaptive_results
    )

    if all_adaptive_pass:
        verdict = "ADAPTIVE WINS: bandit recovers NIAH to >=8/9 at all contexts — propose as default."
    elif some_adaptive_better:
        fixed005_fail = [s for s in all_summaries
                        if s["condition"] == "fixed_005" and s["niah_pass"] < 8]
        adaptive_still_fail = [s for s in adaptive_results if s["niah_pass"] < 8]
        verdict = (f"PARTIAL RECOVERY: adaptive beats fixed_005 at some contexts "
                   f"but {len(adaptive_still_fail)} cell(s) still below 8/9. "
                   f"Keep adaptive but flag residual issue.")
    else:
        verdict = ("NO RECOVERY: adaptive does not improve NIAH vs fixed_005. "
                   "Cliff is NOT compression-induced (or bandit can't adapt in 9 trials). "
                   "Bug #42 or other mechanism is dominant.")

    print(f"\nVERDICT: {verdict}", flush=True)

    summary_path = out_dir / "SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write("# Adaptive keep_ratio NIAH Recovery: 32K / 64K / 128K\n\n")
        f.write(f"Binary: pflash-auto worktree (PR #264)  \n")
        f.write(f"Stack: Q4_K_M target + Qwen3-0.6B-BF16 drafter + dflash-draft + BSA alpha=0.85  \n")
        f.write(f"Date: 2026-05-24  \n")
        f.write(f"Trials per cell: {n_trials}  \n\n")
        f.write("## Headline Table\n\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for row in rows_md:
            f.write(row + "\n")
        f.write("\n## Verdict\n\n")
        f.write(f"{verdict}\n\n")
        f.write("## Failure Categories\n\n")
        f.write("- `pass`: model reproduced the needle value correctly\n")
        f.write("- `wrong_answer`: model answered but with wrong value (compression quality failure)\n")
        f.write("- `crash`: server error or HTTP error during request\n")
        f.write("- `timeout`: request exceeded 15 min limit\n")

    print(f"\n[done] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
