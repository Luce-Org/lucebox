#!/usr/bin/env python3
"""
Pass B: Agentic claude_code harness for ee7 broad validation.

Purpose: measure drafter_fwd, accept_rate, and OK_DONE across baseline/ee14/ee7
using the real claude_code client against a live dflash_server.
Run manually from the drafter-fastpath worktree; results land in
dflash/bench/results/2026-05-21_ee7_broad/. Keep as evidence trail for
multi-client validation (commit 764b18e).
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "dflash/bench/results/2026-05-21_ee7_broad"
HARNESS = REPO / "harness/clients/run_claude_code.sh"

CONDITIONS = [
    {
        "name": "baseline",
        "env_extra": {},
    },
    {
        "name": "ee14",
        "env_extra": {"PFLASH_DRAFTER_EARLY_EXIT_N": "14"},
    },
    {
        "name": "ee7",
        "env_extra": {
            "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
            "PFLASH_DRAFTER_SCORE_LAYERS": "7",
        },
    },
]


def run_condition(cond: dict) -> dict:
    name = cond["name"]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"ee7-passbv-{name}-{stamp}"
    run_dir = f"/tmp/lucebox-harness-runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    env = os.environ.copy()
    # Common server env (matches the spec)
    env.update({
        "MODEL_SERVER": "lucebox",
        "LUCEBOX_SERVER_BACKEND": "cpp",
        "DFLASH_SERVER_BIN": str(REPO / "dflash/build/dflash_server"),
        "TARGET": "/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf",
        "DRAFT": "/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf",
        "MAX_CTX": "98304",
        "MAX_TOKENS": "512",
        "GGML_CUDA_NO_VMM": "1",
        "DFLASH27B_KV_K": "tq3_0",
        "DFLASH27B_KV_V": "tq3_0",
        "VERIFY_MODE": "ddtree",
        "BUDGET": "16",
        "REPO_DIR": str(REPO),
        "RUN_DIR": "/tmp/lucebox-harness-runs",
        "EXTRA_SERVER_ARGS": (
            "--prefill-compression always --prefill-keep-ratio 0.05 "
            "--prefill-drafter /home/peppi/models/Qwen3-0.6B-BF16.gguf"
        ),
        "PROMPT_FILE": str(REPO / "harness/clients/prompts/decode_check.txt"),
        "CLAUDE_TIMEOUT": "600",
        "MARKER": "OK_DONE",
        "CLAUDE_TOOLS": "none",
        "PORT": "18098",
        "STAMP": run_name,
        "CLAUDE_BIN": "/home/peppi/.local/bin/claude",
    })
    # Clear any previous early-exit env
    env.pop("PFLASH_DRAFTER_EARLY_EXIT_N", None)
    env.pop("PFLASH_DRAFTER_SCORE_LAYERS", None)
    # Apply condition-specific env
    env.update(cond["env_extra"])

    print(f"\n[passbv] running condition={name} stamp={stamp}", flush=True)
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            ["bash", str(HARNESS)],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        elapsed = time.perf_counter() - t0
        print(f"[passbv] condition={name} rc={result.returncode} elapsed={elapsed:.1f}s", flush=True)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        print(f"[passbv] condition={name} TIMEOUT after {elapsed:.0f}s — marking as failed", flush=True)
        return {
            "condition": name,
            "drafter_fwd_s": None,
            "n_tokens": None,
            "accept_rate": None,
            "accept_detail": None,
            "ok_done": False,
            "run_dir": str(run_dir),
            "rc": -1,
            "error": "timeout",
        }

    server_log = Path(f"/tmp/lucebox-harness-runs/{run_name}/server.log")
    client_out = Path(f"/tmp/lucebox-harness-runs/{run_name}/claude-code.out")

    # Extract metrics
    drafter_fwd = None
    accept_rate = None
    ok_done = False
    n_tokens = None
    accept_str = None

    if server_log.exists():
        text = server_log.read_text()
        m = re.search(r"\[drafter\] forward\+score in ([\d.]+)s S=(\d+)", text)
        if m:
            drafter_fwd = float(m.group(1))
            n_tokens = int(m.group(2))
        m2 = re.search(r"accepted=(\d+/\d+) \(([\d.]+)%\)", text)
        if m2:
            accept_rate = m2.group(2) + "%"
            accept_str = m2.group(0)

    if client_out.exists():
        client_text = client_out.read_text()
        ok_done = "OK_DONE" in client_text

    print(f"  drafter_fwd={drafter_fwd}s S={n_tokens} accept={accept_str} ok_done={ok_done}", flush=True)

    return {
        "condition": name,
        "drafter_fwd_s": drafter_fwd,
        "n_tokens": n_tokens,
        "accept_rate": accept_rate,
        "accept_detail": accept_str,
        "ok_done": ok_done,
        "run_dir": str(run_dir),
        "rc": result.returncode,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Get baseline drafter time for speedup calculation
    results = []
    baseline_fwd = None

    for cond in CONDITIONS:
        r = run_condition(cond)
        results.append(r)
        if r["condition"] == "baseline" and r["drafter_fwd_s"] is not None:
            baseline_fwd = r["drafter_fwd_s"]

    # Print table
    print("\n=== PASS B TABLE ===")
    print(f"{'condition':>12}  {'drafter_fwd':>12}  {'accept_rate':>12}  {'OK_DONE':>8}  {'speedup':>8}")
    rows = []
    for r in results:
        cond = r["condition"]
        df = f"{r['drafter_fwd_s']:.2f}s" if r['drafter_fwd_s'] else "N/A"
        ar = r['accept_rate'] or "N/A"
        ok = "YES" if r['ok_done'] else "NO"
        speedup = "1.00x"
        if cond != "baseline" and r['drafter_fwd_s'] and baseline_fwd:
            speedup = f"{baseline_fwd / r['drafter_fwd_s']:.2f}x"
        print(f"{cond:>12}  {df:>12}  {ar:>12}  {ok:>8}  {speedup:>8}")
        rows.append({
            "condition": cond,
            "drafter_fwd": df,
            "accept_rate": ar,
            "OK_DONE": ok,
            "speedup": speedup,
        })

    # Write to results dir
    out = RESULTS_DIR / "SUMMARY_PASS_B.md"
    with open(out, "w") as f:
        f.write("# Pass B: Agentic claude_code Harness — ee7 vs ee14 vs baseline\n\n")
        f.write("| condition | drafter_fwd | accept_rate | OK_DONE | speedup_vs_baseline |\n")
        f.write("|---|---|---|---|---|\n")
        for row in rows:
            f.write(f"| {row['condition']} | {row['drafter_fwd']} | {row['accept_rate']} | {row['OK_DONE']} | {row['speedup']} |\n")
    print(f"\n[done] {out}", flush=True)
    return results


if __name__ == "__main__":
    main()
