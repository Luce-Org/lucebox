#!/usr/bin/env python3
"""Early-exit forward bench: baseline_ee, ee14, ee7 at 32K and 64K.

Usage:
    python3 run_early_exit_bench.py [--dry-run] [--n-reps N]
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
DRAFTER_BF16 = Path("/home/peppi/models/Qwen3-0.6B-BF16.gguf")

PORT = 18093
RESULTS_DIR = WORKTREE / "dflash/bench/results/2026-05-21_early_exit"
MAX_CTX = 70656
CONTEXTS = [32768, 65536]

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
    {"name": "baseline_ee", "early_exit_n": None, "score_layers": None},
    {"name": "ee14",        "early_exit_n": 14,   "score_layers": None},
    {"name": "ee7",         "early_exit_n": 7,    "score_layers": 7},
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

    if cond["early_exit_n"] is not None:
        env["PFLASH_DRAFTER_EARLY_EXIT_N"] = str(cond["early_exit_n"])
    elif "PFLASH_DRAFTER_EARLY_EXIT_N" in env:
        del env["PFLASH_DRAFTER_EARLY_EXIT_N"]

    if cond["score_layers"] is not None:
        env["PFLASH_DRAFTER_SCORE_LAYERS"] = str(cond["score_layers"])
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
        "--prefill-drafter", str(DRAFTER_BF16),
    ]
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


def extract_drafter_times_after(log_path: Path, after_idx: int) -> list:
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


def extract_layer_log(log_path: Path) -> str:
    """Extract the last qwen3-0.6b-fp forward summary line."""
    try:
        text = log_path.read_text(errors="replace")
        lines = re.findall(r'\[qwen3-0\.6b-fp\] forward .*', text)
        return lines[-1] if lines else ""
    except Exception:
        return ""


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
        print(f"\n{'='*60}", flush=True)
        print(f"Condition: {name}  early_exit_n={cond['early_exit_n']}  score_layers={cond['score_layers']}", flush=True)
        print(f"{'='*60}", flush=True)

        log_path = RESULTS_DIR / f"{name}_server.log"
        proc, log_f = start_server(cond, log_path)
        print(f"Server PID {proc.pid}, log={log_path}", flush=True)

        if not wait_for_server(timeout=300):
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

            new_times = extract_drafter_times_after(log_path, drafter_times_before)
            last_layer_log = extract_layer_log(log_path)
            print(f"  drafter forward times (new): {[f'{t:.2f}s' for t in new_times]}", flush=True)
            print(f"  last layer log: {last_layer_log[:200]}", flush=True)

            key = f"{name}|{ctx}"
            all_results[key] = {
                "condition": name,
                "ctx": ctx,
                "early_exit_n": cond["early_exit_n"],
                "score_layers": cond["score_layers"],
                "drafter_times_s": new_times,
                "drafter_p50_s": p50(new_times),
                "reps": rep_results,
                "ttft_p50_s": p50([r["ttft_s"] for r in rep_results]) if rep_results else None,
                "niah_pass_rate": (sum(r["found"] for r in rep_results) / len(rep_results)
                                   if rep_results else None),
                "last_layer_log": last_layer_log,
            }

        kill_server(proc)
        log_f.close()
        time.sleep(10)

    raw_path = RESULTS_DIR / "raw_results.json"
    with raw_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {raw_path}", flush=True)

    # Print summary table
    baseline_32k = all_results.get("baseline_ee|32768", {}).get("drafter_p50_s")
    baseline_64k = all_results.get("baseline_ee|65536", {}).get("drafter_p50_s")

    print("\n## Results Summary\n")
    header = f"{'Condition':<15} {'ctx':>6} {'drafter_p50':>12} {'ttft_p50':>10} {'NIAH':>6} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for cond in CONDITIONS:
        for ctx in CONTEXTS:
            key = f"{cond['name']}|{ctx}"
            r = all_results.get(key, {})
            dp50 = r.get("drafter_p50_s")
            tp50 = r.get("ttft_p50_s")
            niah = r.get("niah_pass_rate")
            base = baseline_32k if ctx == 32768 else baseline_64k
            speedup = f"{base/dp50:.2f}x" if dp50 and base else "n/a"
            niah_s = f"{niah*100:.0f}%" if niah is not None else "n/a"
            dp50_s = f"{dp50:.2f}s" if dp50 else "n/a"
            tp50_s = f"{tp50:.1f}s" if tp50 else "n/a"
            print(f"{cond['name']:<15} {ctx:>6} {dp50_s:>12} {tp50_s:>10} {niah_s:>6} {speedup:>8}")

    print(f"\nFull results: {raw_path}")


if __name__ == "__main__":
    main()
