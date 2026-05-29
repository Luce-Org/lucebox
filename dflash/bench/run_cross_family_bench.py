#!/usr/bin/env python3
"""Cross-family drafter bench: Qwen3-0.6B (baseline) vs SmolLM2-360M and SmolLM2-135M.

Usage:
    SMOLLM2_360M=/home/peppi/models/SmolLM2-360M-BF16.gguf \
    SMOLLM2_135M=/home/peppi/models/SmolLM2-135M-BF16.gguf \
    python3 run_cross_family_bench.py [--dry-run] [--n-reps N]
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
from pathlib import Path

WORKTREE = Path("/home/peppi/Dev/lucebox-hub/.claude/worktrees/drafter-fastpath")
SERVER_BIN = WORKTREE / "dflash/build/dflash_server"
TARGET_MODEL = Path("/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf")
DRAFTER_QWEN3 = Path("/home/peppi/models/Qwen3-0.6B-BF16.gguf")
DRAFTER_SMOL360 = Path(os.environ.get("SMOLLM2_360M", "/home/peppi/models/SmolLM2-360M-BF16.gguf"))
DRAFTER_SMOL135 = Path(os.environ.get("SMOLLM2_135M", "/home/peppi/models/SmolLM2-135M-BF16.gguf"))

PORT = 18096
RESULTS_DIR = WORKTREE / "dflash/bench/results/2026-05-21_cross_family"
MAX_CTX = 70656
CONTEXTS = [32768, 65536]

# A_baseline reference from c7ef4f6 early_exit bench (warm p50)
BASELINE_WARM = {32768: 3.52, 65536: 7.28}

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
    {"name": "A_baseline",    "drafter": DRAFTER_QWEN3,  "early_exit_n": None},
    {"name": "E_smol_360",    "drafter": DRAFTER_SMOL360, "early_exit_n": None},
    {"name": "F_smol_360_ee16","drafter": DRAFTER_SMOL360, "early_exit_n": 16},
    {"name": "G_smol_135",    "drafter": DRAFTER_SMOL135, "early_exit_n": None},
]


def build_niah_prompt(ctx_tokens: int) -> str:
    chars_needed = int(ctx_tokens * 3.5)
    text = (FILLER * (chars_needed // len(FILLER) + 2))[:chars_needed]
    insert_at = len(text) // 2
    text = text[:insert_at] + "\n" + NEEDLE + "\n" + text[insert_at:]
    text = text[:chars_needed]
    return (
        f"Carefully read the following long document:\n\n{text}\n\n"
        f"Based on the document above, answer this question:\n{NEEDLE_QUERY}"
    )


def start_server(cond: dict, log_path: Path):
    env = os.environ.copy()
    env["GGML_CUDA_NO_VMM"] = "1"
    env["DFLASH27B_KV_K"] = "tq3_0"
    env["DFLASH27B_KV_V"] = "tq3_0"

    for var in ("PFLASH_DRAFTER_EARLY_EXIT_N", "PFLASH_DRAFTER_SCORE_LAYERS"):
        env.pop(var, None)

    if cond["early_exit_n"] is not None:
        env["PFLASH_DRAFTER_EARLY_EXIT_N"] = str(cond["early_exit_n"])

    cmd = [
        str(SERVER_BIN),
        str(TARGET_MODEL),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-ctx", str(MAX_CTX),
        "--prefill-compression", "always",
        "--prefill-keep-ratio", "0.05",
        "--prefill-drafter", str(cond["drafter"]),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w")
    proc = subprocess.Popen(cmd, env=env, stderr=log_f, stdout=log_f,
                             preexec_fn=os.setsid)
    return proc, log_f


def wait_for_server(timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=3)
            return True
        except Exception:
            time.sleep(3)
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
    try:
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
    except Exception as e:
        print(f"  [chat error] {e}", flush=True)
    t_end = time.perf_counter()
    text = "".join(text_parts)
    return {
        "text": text,
        "ttft_s": (t_first - t0) if t_first else (t_end - t0),
        "total_s": t_end - t0,
        "found": NEEDLE_ANSWER_KEY.lower() in text.lower(),
    }


def extract_drafter_fwd_times(log_text: str) -> list:
    # Match "[qwen3-0.6b-fp] forward X.XXs ..."
    return [float(m.group(1)) for m in re.finditer(r'\[qwen3-0\.6b-fp\] forward ([\d.]+)s', log_text)]


def extract_tail_score_times(log_text: str) -> list:
    return [float(m.group(1)) for m in re.finditer(r'tail-score ([\d.]+)s', log_text)]


def extract_a_compute_times(log_text: str) -> list:
    return [float(m.group(1)) for m in re.finditer(r'A_compute=([\d.]+)s', log_text)]


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


def p50(vals: list):
    if not vals:
        return None
    s = sorted(vals)
    return s[len(s) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-reps", type=int, default=3)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("Dry run: conditions =", [c["name"] for c in CONDITIONS])
        print("Contexts:", CONTEXTS)
        for c in CONDITIONS:
            print(f"  {c['name']}: drafter={c['drafter']}")
        return

    n_reps = args.n_reps
    all_results = {}
    prompts = {ctx: build_niah_prompt(ctx) for ctx in CONTEXTS}

    for cond in CONDITIONS:
        cond_name = cond["name"]
        print(f"\n=== {cond_name} (drafter={cond['drafter'].name}, ee={cond.get('early_exit_n')}) ===", flush=True)

        log_path = RESULTS_DIR / f"{cond_name}_server.log"
        proc, log_f = start_server(cond, log_path)

        print("  waiting for server...", flush=True)
        if not wait_for_server(300):
            print("  ERROR: server never became ready", flush=True)
            kill_server(proc)
            log_f.close()
            all_results[cond_name] = {"error": "server_start_failed"}
            continue

        print("  server ready", flush=True)
        cond_results = {}

        for ctx in CONTEXTS:
            print(f"  ctx={ctx}:", flush=True)
            prompt = prompts[ctx]

            fwd_times = []
            tail_times = []
            ac_times = []
            niah_hits = 0
            log_snapshots = []

            for rep in range(n_reps):
                log_before = log_path.stat().st_size if log_path.exists() else 0
                t_req_start = time.perf_counter()
                result = chat(prompt, max_tokens=64, timeout=600)
                t_req_end = time.perf_counter()
                found = result["found"]
                if found:
                    niah_hits += 1
                print(f"    rep{rep}: ttft={result['ttft_s']:.2f}s total={result['total_s']:.2f}s "
                      f"NIAH={'OK' if found else 'FAIL'} answer={result['text'][:80]!r}", flush=True)

                # Extract timing from log chunk written since this request
                time.sleep(0.5)  # allow log flush
                try:
                    with log_path.open() as lf:
                        lf.seek(log_before)
                        chunk = lf.read()
                    fwd = extract_drafter_fwd_times(chunk)
                    tail = extract_tail_score_times(chunk)
                    ac = extract_a_compute_times(chunk)
                    print(f"    rep{rep} log: fwd={fwd} tail={tail} A_compute={ac}", flush=True)
                    if fwd:
                        fwd_times.extend(fwd)
                    if tail:
                        tail_times.extend(tail)
                    if ac:
                        ac_times.extend(ac)
                except Exception as e:
                    print(f"    rep{rep} log parse error: {e}", flush=True)

            warm_fwd = p50(fwd_times[1:]) if len(fwd_times) > 1 else fwd_times[0] if fwd_times else None
            warm_tail = p50(tail_times[1:]) if len(tail_times) > 1 else tail_times[0] if tail_times else None
            warm_ac = p50(ac_times[1:]) if len(ac_times) > 1 else ac_times[0] if ac_times else None
            cold_fwd = fwd_times[0] if fwd_times else None

            speedup = None
            if warm_fwd and BASELINE_WARM.get(ctx):
                speedup = BASELINE_WARM[ctx] / warm_fwd

            cond_results[ctx] = {
                "fwd_cold": cold_fwd,
                "fwd_warm": warm_fwd,
                "tail_warm": warm_tail,
                "ac_warm": warm_ac,
                "niah": f"{niah_hits}/{n_reps}",
                "speedup_vs_A": speedup,
                "all_fwd": fwd_times,
                "all_tail": tail_times,
                "all_ac": ac_times,
            }
            print(f"  ctx={ctx} summary: fwd_warm={warm_fwd} tail_warm={warm_tail} "
                  f"NIAH={niah_hits}/{n_reps} speedup_vs_A={speedup:.2f}x" if speedup else
                  f"  ctx={ctx} summary: fwd_warm={warm_fwd} NIAH={niah_hits}/{n_reps}",
                  flush=True)

        kill_server(proc)
        log_f.close()
        print(f"  server stopped", flush=True)
        all_results[cond_name] = cond_results

        # Wait for GPU to clear
        time.sleep(5)

    # Write raw results
    raw_path = RESULTS_DIR / "raw_results.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results: {raw_path}", flush=True)

    # Write SUMMARY.md
    summary_path = RESULTS_DIR / "SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write("# Cross-Family Drafter Bench — 2026-05-21\n\n")
        f.write(f"GPU: RTX 3090 (24 GiB)\n")
        f.write(f"A_baseline reference: Qwen3-0.6B-BF16 warm p50 from c7ef4f6\n\n")
        f.write("| Condition | ctx | fwd_cold | fwd_warm | tail_warm | ac_warm | NIAH | speedup_vs_A |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for cond in CONDITIONS:
            cn = cond["name"]
            r = all_results.get(cn, {})
            for ctx in CONTEXTS:
                cr = r.get(ctx, {})
                if not cr:
                    f.write(f"| {cn} | {ctx//1024}K | - | - | - | - | - | - |\n")
                    continue
                fc = f"{cr.get('fwd_cold'):.2f}s" if cr.get('fwd_cold') else "-"
                fw = f"{cr.get('fwd_warm'):.2f}s" if cr.get('fwd_warm') else "-"
                tw = f"{cr.get('tail_warm'):.2f}s" if cr.get('tail_warm') else "-"
                aw = f"{cr.get('ac_warm'):.2f}s" if cr.get('ac_warm') else "-"
                niah = cr.get('niah', '-')
                sp = f"{cr.get('speedup_vs_A'):.2f}x" if cr.get('speedup_vs_A') else "-"
                f.write(f"| {cn} | {ctx//1024}K | {fc} | {fw} | {tw} | {aw} | {niah} | {sp} |\n")

        f.write("\n## Notes\n\n")
        f.write(f"- A_baseline warm reference: 32K={BASELINE_WARM[32768]}s, 64K={BASELINE_WARM[65536]}s (from c7ef4f6 early-exit bench)\n")
        f.write(f"- F_smol_360_ee16: SmolLM2-360M with EARLY_EXIT_N=16 (half of 32 layers)\n")
        f.write(f"- keep_ratio=0.05, TQ3_0 KV cache, BF16 drafters\n")

    print(f"Summary: {summary_path}", flush=True)
    print("\nFinal table:", flush=True)
    with open(summary_path) as f:
        print(f.read(), flush=True)


if __name__ == "__main__":
    main()
