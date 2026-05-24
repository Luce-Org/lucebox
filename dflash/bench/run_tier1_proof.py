#!/usr/bin/env python3
"""Tier 1 drafter speedup proof: measure drafter forward wall time under
baseline (BF16, 28 layers), Q8 (Q8_0, 28 layers), Q8+subset (Q8_0, 7 layers).

Measures at 32K and 64K context. Extracts drafter forward time from server
stderr log line: [drafter] forward+score in T.XXs

Strategy: start server once per condition (all contexts on same server),
restart between conditions. This reduces server-load overhead from 6 to 3.

Usage:
    python3 run_tier1_proof.py [--dry-run] [--n-reps N]
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
WORKTREE = Path("/home/peppi/Dev/lucebox-hub/.claude/worktrees/drafter-fastpath")
SERVER_BIN = WORKTREE / "dflash/build/dflash_server"
TARGET_MODEL = Path("/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf")
DRAFTER_BF16 = Path("/home/peppi/models/Qwen3-0.6B-BF16.gguf")
DRAFTER_Q8 = Path("/home/peppi/models/Qwen3-0.6B-Q8_0.gguf")

PORT = 18090
RESULTS_DIR = WORKTREE / "dflash/bench/results/2026-05-21_tier1_proof"

NEEDLE = "The secret code for the vault is AMBER-DELTA-9923."
NEEDLE_QUERY = "What is the secret code for the vault?"
NEEDLE_ANSWER_KEY = "AMBER-DELTA-9923"

FILLER = (
    "The Amazon rainforest covers over 5.5 million square kilometers. "
    "It represents over half of the world's remaining rainforests. "
    "The Amazon basin is home to an estimated 10% of all species on Earth. "
    "Scientists estimate that a new species is discovered in the Amazon every three days. "
    "The forest is a critical carbon sink storing billions of tons of CO2. "
    "Deforestation threatens both biodiversity and global climate stability. "
)

CONDITIONS = [
    {"name": "baseline", "drafter": DRAFTER_BF16, "score_layers": None},
    {"name": "q8",       "drafter": DRAFTER_Q8,   "score_layers": None},
    {"name": "q8_l7",    "drafter": DRAFTER_Q8,   "score_layers": 7},
]
# Max context needed - server max-ctx must cover the largest
MAX_CTX = 70000
CONTEXTS = [32768, 65536]


def build_niah_prompt(ctx_tokens: int) -> str:
    # Build haystack: ~ctx_tokens characters assuming ~3.5 char/token
    # Insert needle at ~50% depth
    chars_needed = int(ctx_tokens * 3.5)
    text = (FILLER * (chars_needed // len(FILLER) + 2))[:chars_needed]
    insert_at = len(text) // 2
    text = text[:insert_at] + "\n" + NEEDLE + "\n" + text[insert_at:]
    text = text[:chars_needed]
    return (
        f"Carefully read the following long document:\n\n{text}\n\n"
        f"Based on the document above, answer this question:\n{NEEDLE_QUERY}"
    )


def start_server(cond: dict) -> tuple:
    drafter = cond["drafter"]
    score_layers = cond["score_layers"]
    name = cond["name"]

    env = os.environ.copy()
    env["GGML_CUDA_NO_VMM"] = "1"
    if score_layers is not None:
        env["PFLASH_DRAFTER_SCORE_LAYERS"] = str(score_layers)
    elif "PFLASH_DRAFTER_SCORE_LAYERS" in env:
        del env["PFLASH_DRAFTER_SCORE_LAYERS"]

    cmd = [
        str(SERVER_BIN),
        str(TARGET_MODEL),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-ctx", str(MAX_CTX),
        "--prefill-compression", "always",
        "--prefill-keep-ratio", "0.05",
        "--prefill-drafter", str(drafter),
        "--prefill-skip-park",
    ]
    log_path = RESULTS_DIR / f"server_{name}.log"
    log_f = log_path.open("w")
    proc = subprocess.Popen(cmd, env=env, stderr=log_f, stdout=log_f,
                             preexec_fn=os.setsid)
    return proc, log_f, log_path


def wait_for_server(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=3)
            return True
        except Exception:
            time.sleep(2)
    return False


def chat(prompt: str, max_tokens: int = 64, timeout: float = 900) -> dict:
    body = {
        "model": "dflash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    t_first = None
    text_parts = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
            if content:
                if t_first is None:
                    t_first = time.perf_counter()
                text_parts.append(content)
    t_end = time.perf_counter()
    text = "".join(text_parts)
    return {
        "text": text,
        "ttft_s": (t_first - t0) if t_first else (t_end - t0),
        "total_s": t_end - t0,
        "found": NEEDLE_ANSWER_KEY.lower() in text.lower(),
    }


def extract_drafter_times_after(log_path: Path, after_idx: int) -> list[float]:
    """Extract [drafter] forward+score in T.XXs after a given index."""
    times = []
    try:
        text = log_path.read_text(errors="replace")
        all_matches = list(re.finditer(r'\[drafter\] forward\+score in ([\d.]+)s', text))
        for m in all_matches[after_idx:]:
            times.append(float(m.group(1)))
    except Exception:
        pass
    return times


def count_drafter_times(log_path: Path) -> int:
    try:
        text = log_path.read_text(errors="replace")
        return len(re.findall(r'\[drafter\] forward\+score in ([\d.]+)s', text))
    except Exception:
        return 0


def kill_server(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=15)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def p50(vals: list) -> float | None:
    if not vals:
        return None
    return sorted(vals)[len(vals) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-reps", type=int, default=3)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("Dry run: would run conditions:", [c["name"] for c in CONDITIONS])
        print("Contexts:", CONTEXTS)
        return

    n_reps = args.n_reps
    all_results = {}

    for cond in CONDITIONS:
        name = cond["name"]
        drafter = cond["drafter"]
        score_layers = cond["score_layers"]

        if not drafter.exists():
            print(f"[SKIP] Drafter not found: {drafter}", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"Condition: {name}  drafter={drafter.name}  score_layers={score_layers}", flush=True)
        print(f"{'='*60}", flush=True)

        proc, log_f, log_path = start_server(cond)
        print(f"Server PID {proc.pid}, log={log_path}", flush=True)

        if not wait_for_server(timeout=180):
            print("  ERROR: server did not start in time", flush=True)
            kill_server(proc)
            log_f.close()
            continue

        print("  Server ready", flush=True)

        for ctx in CONTEXTS:
            print(f"\n  ctx={ctx}", flush=True)
            prompt = build_niah_prompt(ctx)
            print(f"  prompt_chars={len(prompt)}", flush=True)

            rep_results = []
            drafter_times_before = count_drafter_times(log_path)

            for rep in range(n_reps):
                print(f"  rep {rep+1}/{n_reps} ...", end=" ", flush=True)
                try:
                    r = chat(prompt, max_tokens=64, timeout=900)
                    print(f"ttft={r['ttft_s']:.1f}s found={r['found']} text={r['text'][:80]!r}", flush=True)
                    rep_results.append(r)
                except Exception as e:
                    print(f"ERROR: {e}", flush=True)

            all_times = extract_drafter_times_after(log_path, 0)
            new_times = extract_drafter_times_after(log_path, drafter_times_before)
            print(f"  drafter forward times (new): {[f'{t:.2f}s' for t in new_times]}", flush=True)

            key = f"{name}|{ctx}"
            all_results[key] = {
                "condition": name,
                "ctx": ctx,
                "drafter": drafter.name,
                "score_layers": score_layers,
                "drafter_times_s": new_times,
                "drafter_p50_s": p50(new_times),
                "reps": rep_results,
                "ttft_p50_s": p50([r["ttft_s"] for r in rep_results]) if rep_results else None,
                "niah_pass_rate": (sum(r["found"] for r in rep_results) / len(rep_results)
                                   if rep_results else None),
            }

        kill_server(proc)
        log_f.close()
        # Wait for VRAM to fully release
        time.sleep(8)

    # Save raw results
    raw_path = RESULTS_DIR / "raw_results.json"
    with raw_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {raw_path}", flush=True)

    # Compute baseline reference times
    baseline_times = {}
    for ctx in CONTEXTS:
        k = f"baseline|{ctx}"
        if k in all_results and all_results[k]["drafter_p50_s"]:
            baseline_times[ctx] = all_results[k]["drafter_p50_s"]

    # Print summary table
    print("\n" + "="*80)
    print("RESULTS TABLE")
    print("="*80)
    print(f"{'Condition':<15} {'ctx':>6} {'drafter_p50':>12} {'ttft_p50':>10} {'niah%':>7} {'speedup':>8}")
    print("-"*80)

    for cond in CONDITIONS:
        nm = cond["name"]
        for ctx in CONTEXTS:
            k = f"{nm}|{ctx}"
            if k not in all_results:
                print(f"{nm:<15} {ctx:>6} {'N/A':>12} {'N/A':>10} {'N/A':>7} {'N/A':>8}")
                continue
            r = all_results[k]
            d = r["drafter_p50_s"]
            t = r["ttft_p50_s"]
            niah = r["niah_pass_rate"]
            speedup = ""
            if d and ctx in baseline_times and baseline_times[ctx]:
                speedup = f"{baseline_times[ctx]/d:.1f}x"
            print(f"{nm:<15} {ctx:>6} "
                  f"{(f'{d:.2f}s' if d else 'N/A'):>12} "
                  f"{(f'{t:.1f}s' if t else 'N/A'):>10} "
                  f"{(f'{niah:.0%}' if niah is not None else 'N/A'):>7} "
                  f"{speedup:>8}")

    print("="*80)

    write_summary(all_results, baseline_times)


def write_summary(all_results, baseline_times):
    summary_path = RESULTS_DIR / "SUMMARY.md"

    # Determine verdict
    verdict_lines = []
    for ctx in CONTEXTS:
        k = f"q8_l7|{ctx}"
        if k in all_results:
            r = all_results[k]
            d = r["drafter_p50_s"]
            niah = r["niah_pass_rate"]
            if d and ctx in baseline_times:
                spd = baseline_times[ctx] / d
                verdict_lines.append(
                    f"At {ctx//1024}K: {spd:.1f}x speedup, NIAH {niah:.0%}"
                    if niah is not None else
                    f"At {ctx//1024}K: {spd:.1f}x speedup, NIAH N/A"
                )

    with summary_path.open("w") as f:
        f.write("# Tier 1 Drafter Speedup Proof — 2026-05-21\n\n")
        f.write("## Setup\n\n")
        f.write(f"- Target: Qwen3.6-27B-Q4_K_M  GPU: RTX 3090 24GB\n")
        f.write(f"- Baseline: Qwen3-0.6B-BF16, 28 layers scored\n")
        f.write(f"- Q8: Qwen3-0.6B-Q8_0, 28 layers scored\n")
        f.write(f"- Q8+L7: Qwen3-0.6B-Q8_0, last 7 of 28 layers scored\n")
        f.write(f"- pflash keep_ratio=0.05, NIAH single-needle at 50% depth\n")
        f.write(f"- n_reps=3 per cell\n\n")
        f.write("## Results\n\n")
        f.write("| Condition | ctx | drafter_fwd_p50 | ttft_p50 | NIAH | speedup_vs_baseline |\n")
        f.write("|-----------|-----|-----------------|----------|------|---------------------|\n")
        for cond in CONDITIONS:
            nm = cond["name"]
            for ctx in CONTEXTS:
                k = f"{nm}|{ctx}"
                if k not in all_results:
                    f.write(f"| {nm} | {ctx//1024}K | N/A | N/A | N/A | N/A |\n")
                    continue
                r = all_results[k]
                d = r["drafter_p50_s"]
                t = r["ttft_p50_s"]
                niah = r["niah_pass_rate"]
                speedup = "—"
                if d and ctx in baseline_times and baseline_times[ctx]:
                    speedup = f"{baseline_times[ctx]/d:.1f}x"
                f.write(f"| {nm} | {ctx//1024}K | "
                        f"{f'{d:.2f}s' if d else 'N/A'} | "
                        f"{f'{t:.1f}s' if t else 'N/A'} | "
                        f"{f'{niah:.0%}' if niah is not None else 'N/A'} | "
                        f"{speedup} |\n")
        f.write("\n")
        f.write("## Verdict\n\n")
        if verdict_lines:
            for v in verdict_lines:
                f.write(f"- {v}\n")
            # Determine if light or dark
            k32 = "q8_l7|32768"
            if k32 in all_results:
                r = all_results[k32]
                if r.get("drafter_p50_s") and 32768 in baseline_times:
                    spd = baseline_times[32768] / r["drafter_p50_s"]
                    niah_ok = r.get("niah_pass_rate", 0) or 0
                    if spd >= 3.0 and niah_ok >= 0.67:
                        f.write("\n**Light at the end of the tunnel:** "
                                f"{spd:.1f}x drafter speedup at 32K with NIAH preserved.\n")
                    elif spd >= 2.0:
                        f.write("\n**Partial signal:** "
                                f"{spd:.1f}x drafter speedup, but below 3x target or NIAH degraded.\n")
                    else:
                        f.write("\n**Dark:** speedup insufficient or NIAH broken.\n")
        else:
            f.write("Run incomplete — no Q8+L7 results at 32K.\n")

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
