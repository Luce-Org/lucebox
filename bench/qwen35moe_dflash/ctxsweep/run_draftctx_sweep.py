#!/usr/bin/env python3
"""
Draft-ctx VRAM ceiling sweep + recall horizon measurement.
Runs under flock -x /tmp/lucebox_gpu.lock.
"""

import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

BASE_DIR    = "/home/peppi/Dev/lucebox-hub"
SERVER_BIN  = f"{BASE_DIR}/server/build/dflash_server"
TARGET_MODEL = "/home/peppi/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
DRAFT_MODEL  = "/home/peppi/models/qwen3.6-35b-a3b-dflash-new/qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf"
CHAT_TMPL    = "/home/peppi/models/qwen3-coder-chat-template.jinja"
BENCH_DIR    = f"{BASE_DIR}/bench/qwen35moe_dflash/ctxsweep"
PORT         = 18081
HOST         = "127.0.0.1"
MAX_CTX      = 40960
FEAT_RING_CAP = 40960
HEALTH_TIMEOUT = 180  # seconds

NEEDLE_FILES = [
    ("needle_mid_06k.json", "06k"),
    ("needle_mid_08k.json", "08k"),
    ("needle_mid_12k.json", "12k"),
]

# Step 1 sweep
DRAFT_CTX_VALUES = [8192, 16384, 24576]

# ---- helpers ----

def health_check():
    url = f"http://{HOST}:{PORT}/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False

def wait_healthy(timeout=HEALTH_TIMEOUT):
    start = time.time()
    while time.time() - start < timeout:
        if health_check():
            return True
        time.sleep(2)
    return False

def launch_server(draft_ctx, log_path, extra_env=None):
    env = os.environ.copy()
    env["DFLASH_DRAFT_CTX_MAX"]  = str(draft_ctx)
    env["DFLASH_FEAT_RING_CAP"]   = str(FEAT_RING_CAP)
    if extra_env:
        env.update(extra_env)

    cmd = [
        SERVER_BIN,
        TARGET_MODEL,
        "--draft", DRAFT_MODEL,
        "--host", HOST,
        "--port", str(PORT),
        "--max-ctx", str(MAX_CTX),
        "--max-tokens", "200",
        "--fa-window", "0",
        "--cache-type-k", "q4_0",
        "--cache-type-v", "q4_0",
        "--chat-template-file", CHAT_TMPL,
        "--model-name", "luce-dflash",
        "--lazy-draft",
    ]

    log_fd = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_fd, stderr=log_fd, env=env)
    return proc, log_fd

def kill_server(proc, log_fd):
    if proc and proc.poll() is None:
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(4)
    if log_fd:
        log_fd.close()

def post_prompt(prompt_path):
    with open(prompt_path) as f:
        payload = json.load(f)
    data = json.dumps(payload).encode()
    url = f"http://{HOST}:{PORT}/v1/chat/completions"
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            body = r.read().decode()
            return json.loads(body), None
    except Exception as e:
        return None, str(e)

def parse_log_for_oom(log_path):
    try:
        with open(log_path) as f:
            content = f.read()
        oom_patterns = ["out of memory", "CUDA error", "cudaErrorMemoryAllocation",
                        "CUDA_ERROR_OUT_OF_MEMORY", "OOM", "alloc failed"]
        for p in oom_patterns:
            if p.lower() in content.lower():
                return True, p
        return False, None
    except Exception:
        return False, None

def parse_spec_stats_from_log(log_path, after_time=None):
    """Extract accept% and decode tok/s from server log lines after after_time."""
    try:
        with open(log_path) as f:
            lines = f.readlines()
    except Exception:
        return None, None, None

    accept_vals = []
    commit_vals = []
    decode_tps_vals = []

    for line in lines:
        # spec-decode stats line
        m = re.search(r'accept[_\s]?(?:rate|%|ratio)[\s=:]+([0-9.]+)', line, re.IGNORECASE)
        if m:
            accept_vals.append(float(m.group(1)))
        m = re.search(r'avg_commit[\s=:]+([0-9.]+)', line, re.IGNORECASE)
        if m:
            commit_vals.append(float(m.group(1)))
        # decode tok/s
        m = re.search(r'decode[_\s]?(?:speed|tps|tok/s|tokens/s)[\s=:]+([0-9.]+)', line, re.IGNORECASE)
        if m:
            decode_tps_vals.append(float(m.group(1)))
        # Also look for [spec-decode] lines
        m = re.search(r'\[spec-?decode\].*?accept[=:\s]+([0-9.]+)%?', line, re.IGNORECASE)
        if m:
            accept_vals.append(float(m.group(1)))

    accept = accept_vals[-1] if accept_vals else None
    commit = commit_vals[-1] if commit_vals else None
    decode_tps = decode_tps_vals[-1] if decode_tps_vals else None
    return accept, commit, decode_tps

def parse_response_stats(response):
    """Extract spec-decode stats from response body."""
    if not response:
        return None, None, None
    usage = response.get("usage", {})
    # Try common fields
    accept = usage.get("spec_accept_rate") or usage.get("accept_rate")
    commit = usage.get("avg_commit") or usage.get("spec_avg_commit")
    decode_tps = usage.get("decode_tps") or usage.get("tokens_per_second")
    return accept, commit, decode_tps

def check_output_for_marker(response):
    """Check if response reproduced luce_marker_widget."""
    if not response:
        return False
    choices = response.get("choices", [])
    if not choices:
        return False
    content = choices[0].get("message", {}).get("content", "")
    return "luce_marker_widget" in content and ("int gamma" in content or "alpha * 31" in content)

def get_tok_count(response):
    if not response:
        return None
    usage = response.get("usage", {})
    return usage.get("completion_tokens"), usage.get("prompt_tokens")


# ---- Step 1: VRAM ceiling sweep ----

def step1_vram_sweep():
    print("\n=== STEP 1: VRAM ceiling sweep ===")
    print(f"Testing DFLASH_DRAFT_CTX_MAX in {DRAFT_CTX_VALUES}")

    # Use ctx_016384.json as the ~18K prompt
    prompt_path = f"{BENCH_DIR}/ctx_016384.json"

    results = []

    for draft_ctx in DRAFT_CTX_VALUES:
        log_path = f"{BENCH_DIR}/vram_sweep_{draft_ctx}.log"
        print(f"\n--- draft_ctx={draft_ctx} ---")
        print(f"Log: {log_path}")

        proc, log_fd = launch_server(draft_ctx, log_path)
        print(f"Server PID: {proc.pid}")

        # Wait for health
        healthy = wait_healthy()
        if not healthy:
            # Check if it crashed (OOM?)
            oom, pat = parse_log_for_oom(log_path)
            print(f"Server did NOT become healthy. OOM={oom} ({pat})")
            kill_server(proc, log_fd)
            results.append({
                "draft_ctx": draft_ctx,
                "status": "LOAD_OOM" if oom else "LOAD_FAIL",
                "oom_pattern": pat,
            })
            continue

        print(f"Server healthy at draft_ctx={draft_ctx}")

        # Check OOM at load time
        oom, pat = parse_log_for_oom(log_path)
        if oom:
            print(f"OOM detected in log during startup: {pat}")
            kill_server(proc, log_fd)
            results.append({
                "draft_ctx": draft_ctx,
                "status": "LOAD_OOM",
                "oom_pattern": pat,
            })
            continue

        # Post ~18K prompt
        print(f"Posting ~18K prompt...")
        t0 = time.time()
        resp, err = post_prompt(prompt_path)
        elapsed = time.time() - t0
        print(f"Response in {elapsed:.1f}s")

        if err:
            print(f"Request error: {err}")
            oom, pat = parse_log_for_oom(log_path)
            kill_server(proc, log_fd)
            results.append({
                "draft_ctx": draft_ctx,
                "status": "REQUEST_OOM" if oom else "REQUEST_ERROR",
                "error": err,
                "oom_pattern": pat,
            })
            continue

        # Check for OOM in log after request
        oom, pat = parse_log_for_oom(log_path)
        if oom:
            print(f"OOM after request: {pat}")
            kill_server(proc, log_fd)
            results.append({
                "draft_ctx": draft_ctx,
                "status": "SERVE_OOM",
                "oom_pattern": pat,
            })
            continue

        # Parse stats
        accept, commit, decode_tps = parse_response_stats(resp)
        if accept is None and commit is None:
            # Try log
            accept, commit, decode_tps = parse_spec_stats_from_log(log_path)

        comp_toks, prompt_toks = get_tok_count(resp)
        print(f"accept={accept}, commit={commit}, decode_tps={decode_tps}")
        print(f"prompt_tokens={prompt_toks}, completion_tokens={comp_toks}")

        # Print first 200 chars of output
        choices = resp.get("choices", []) if resp else []
        output = choices[0].get("message", {}).get("content", "") if choices else ""
        print(f"Output (first 200): {output[:200]!r}")

        kill_server(proc, log_fd)
        results.append({
            "draft_ctx": draft_ctx,
            "status": "OK",
            "accept": accept,
            "commit": commit,
            "decode_tps": decode_tps,
            "prompt_tokens": prompt_toks,
            "completion_tokens": comp_toks,
        })
        print(f"draft_ctx={draft_ctx} -> OK")

    return results


# ---- Step 2: Recall horizon ----

def step2_recall_horizon(max_fitting_ctx):
    print(f"\n=== STEP 2: Recall horizon at draft_ctx={max_fitting_ctx} and 8192 ===")

    test_configs = [
        (max_fitting_ctx, {}),
    ]
    if max_fitting_ctx != 8192:
        test_configs.append((8192, {}))

    needle_prompts = [
        ("needle_mid_06k.json", "06k", 2593),   # task-stated marker dist
        ("needle_mid_08k.json", "08k", 4854),
        ("needle_mid_12k.json", "12k", 7987),   # our approx
    ]

    results = []

    for draft_ctx, extra_env in test_configs:
        log_path = f"{BENCH_DIR}/recall_{draft_ctx}.log"
        print(f"\n--- draft_ctx={draft_ctx} ---")

        proc, log_fd = launch_server(draft_ctx, log_path, extra_env)
        print(f"Server PID: {proc.pid}")

        healthy = wait_healthy()
        if not healthy:
            oom, pat = parse_log_for_oom(log_path)
            print(f"Server did NOT become healthy. OOM={oom} ({pat})")
            kill_server(proc, log_fd)
            for fname, tag, dist in needle_prompts:
                results.append({
                    "draft_ctx": draft_ctx,
                    "needle": tag,
                    "marker_dist_from_end": dist,
                    "status": "SERVER_FAIL",
                })
            continue

        print(f"Server healthy, running needle prompts...")

        for fname, tag, dist in needle_prompts:
            prompt_path = f"{BENCH_DIR}/{fname}"
            print(f"  Needle {tag} (marker ~{dist} tok from end)...")
            t0 = time.time()
            resp, err = post_prompt(prompt_path)
            elapsed = time.time() - t0

            if err:
                oom, pat = parse_log_for_oom(log_path)
                print(f"  Error: {err}, OOM={oom}")
                results.append({
                    "draft_ctx": draft_ctx,
                    "needle": tag,
                    "marker_dist_from_end": dist,
                    "status": "OOM" if oom else "ERROR",
                    "error": err,
                })
                continue

            accept, commit, decode_tps = parse_response_stats(resp)
            if accept is None:
                accept, commit, decode_tps = parse_spec_stats_from_log(log_path)

            recalled = check_output_for_marker(resp)
            choices = resp.get("choices", []) if resp else []
            output = choices[0].get("message", {}).get("content", "") if choices else ""
            comp_toks, prompt_toks = get_tok_count(resp)

            print(f"  accept={accept}, commit={commit}, decode_tps={decode_tps}")
            print(f"  recalled={recalled}, elapsed={elapsed:.1f}s")
            print(f"  output (first 200): {output[:200]!r}")

            results.append({
                "draft_ctx": draft_ctx,
                "needle": tag,
                "marker_dist_from_end": dist,
                "status": "OK",
                "accept": accept,
                "commit": commit,
                "decode_tps": decode_tps,
                "recalled": recalled,
                "prompt_tokens": prompt_toks,
                "completion_tokens": comp_toks,
                "elapsed_s": round(elapsed, 1),
            })

        kill_server(proc, log_fd)

    return results


# ---- Step 3: f16 mirror test ----

def step3_f16_mirror(oom_draft_ctx):
    """Re-run the first OOM draft_ctx with DFLASH_FEATURE_DTYPE=f16."""
    print(f"\n=== STEP 3: f16 mirror test at draft_ctx={oom_draft_ctx} ===")

    log_path = f"{BENCH_DIR}/vram_sweep_{oom_draft_ctx}_f16.log"
    prompt_path = f"{BENCH_DIR}/ctx_016384.json"

    extra_env = {"DFLASH_FEATURE_DTYPE": "f16"}
    proc, log_fd = launch_server(oom_draft_ctx, log_path, extra_env)
    print(f"Server PID: {proc.pid}, log: {log_path}")

    healthy = wait_healthy()
    if not healthy:
        oom, pat = parse_log_for_oom(log_path)
        print(f"Server did NOT become healthy. OOM={oom} ({pat})")
        kill_server(proc, log_fd)
        return {"draft_ctx": oom_draft_ctx, "dtype": "f16", "status": "LOAD_OOM" if oom else "LOAD_FAIL"}

    print(f"Server healthy with f16 mirror at draft_ctx={oom_draft_ctx}")

    # Post prompt
    resp, err = post_prompt(prompt_path)
    if err:
        oom, pat = parse_log_for_oom(log_path)
        print(f"Request error: {err}, OOM={oom}")
        kill_server(proc, log_fd)
        return {"draft_ctx": oom_draft_ctx, "dtype": "f16", "status": "OOM" if oom else "ERROR"}

    oom, pat = parse_log_for_oom(log_path)
    if oom:
        print(f"OOM after request: {pat}")
        kill_server(proc, log_fd)
        return {"draft_ctx": oom_draft_ctx, "dtype": "f16", "status": "SERVE_OOM"}

    accept, commit, decode_tps = parse_response_stats(resp)
    if accept is None:
        accept, commit, decode_tps = parse_spec_stats_from_log(log_path)

    print(f"f16 result: accept={accept}, commit={commit}, decode_tps={decode_tps}")
    kill_server(proc, log_fd)
    return {
        "draft_ctx": oom_draft_ctx,
        "dtype": "f16",
        "status": "OK",
        "accept": accept,
        "commit": commit,
        "decode_tps": decode_tps,
    }


def main():
    # Acquire GPU lock
    lock_fd = open("/tmp/lucebox_gpu.lock", "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print("GPU lock held by another process. Aborting.")
        sys.exit(1)

    try:
        # Step 1
        step1_results = step1_vram_sweep()

        print("\n=== STEP 1 SUMMARY ===")
        max_fitting_ctx = None
        first_oom_ctx = None
        for r in step1_results:
            status = r["status"]
            dc = r["draft_ctx"]
            print(f"  draft_ctx={dc}: {status}")
            if status == "OK" and (max_fitting_ctx is None or dc > max_fitting_ctx):
                max_fitting_ctx = dc
            if status in ("LOAD_OOM", "REQUEST_OOM", "SERVE_OOM") and first_oom_ctx is None:
                first_oom_ctx = dc

        print(f"Max fitting draft_ctx: {max_fitting_ctx}")
        print(f"First OOM draft_ctx: {first_oom_ctx}")

        # Step 2
        if max_fitting_ctx is not None:
            step2_results = step2_recall_horizon(max_fitting_ctx)
        else:
            print("No fitting draft_ctx found, skipping Step 2")
            step2_results = []

        # Step 3
        if first_oom_ctx is not None:
            step3_result = step3_f16_mirror(first_oom_ctx)
        else:
            print("No OOM arm found, skipping Step 3")
            step3_result = {"note": "no OOM arm"}

        # Save results
        results = {
            "step1": step1_results,
            "step2": step2_results,
            "step3": step3_result,
            "max_fitting_ctx": max_fitting_ctx,
            "first_oom_ctx": first_oom_ctx,
        }
        out_path = f"{BENCH_DIR}/draftctx_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")

        return results

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


if __name__ == "__main__":
    results = main()
    print("\n=== FINAL RESULTS ===")
    print(json.dumps(results, indent=2))
