#!/usr/bin/env python3
"""
2x2 isolation bench: FEATURE_DTYPE x DRAFT_CTX_MAX
Isolates which knob causes ctx_008192.json accept regression.
Requires: flock -x /tmp/lucebox_gpu.lock (acquired inside script).
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

BASE_DIR     = "/home/peppi/Dev/lucebox-hub"
SERVER_BIN   = f"{BASE_DIR}/server/build/dflash_server"
TARGET_MODEL = "/home/peppi/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
DRAFT_MODEL  = "/home/peppi/models/qwen3.6-35b-a3b-dflash-new/qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf"
CHAT_TMPL    = "/home/peppi/models/qwen3-coder-chat-template.jinja"
BENCH_DIR    = f"{BASE_DIR}/bench/qwen35moe_dflash/ctxsweep"
PORT         = 18081
HOST         = "127.0.0.1"
MAX_CTX      = 40960
FEAT_RING_CAP = 40960
HEALTH_TIMEOUT = 180

PROMPTS = [
    ("ctx_008192.json",    "ctx_008192"),
    ("needle_mid_06k.json","needle_06k"),
]

# 2x2 grid: (FEATURE_DTYPE, DRAFT_CTX_MAX_ENV)
CELLS = [
    ("f32", "2048"),
    ("f16", "2048"),
    ("f32", "8192"),
    ("f16", "8192"),
]

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

def launch_server(dtype, draft_ctx_max_str, log_path):
    env = os.environ.copy()
    env["DFLASH_FEAT_RING_CAP"]    = str(FEAT_RING_CAP)
    env["DFLASH_FEATURE_DTYPE"]    = dtype
    env["DFLASH_DRAFT_CTX_MAX"]    = draft_ctx_max_str

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
        try:
            log_fd.close()
        except Exception:
            pass

def post_prompt(prompt_path, timeout=300):
    with open(prompt_path) as f:
        payload = json.load(f)
    # Ensure temperature=0 and model name
    payload["temperature"] = 0
    payload["model"] = "luce-dflash"
    data = json.dumps(payload).encode()
    url = f"http://{HOST}:{PORT}/v1/chat/completions"
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode()
            return json.loads(body), None
    except Exception as e:
        return None, str(e)

def parse_last_spec_stats_from_log(log_path):
    """
    Parse the LAST [spec-decode] line in the log for accept%, AL, decode tok/s.
    Also detect gate_floor reason from [spec-gate] floor lines.
    Returns: (accept_pct, avg_commit, decode_tps, gate_floor_reason)
    """
    try:
        with open(log_path) as f:
            lines = f.readlines()
    except Exception:
        return None, None, None, None

    accept_pct = None
    avg_commit = None
    decode_tps = None
    gate_floor_reason = None

    for line in lines:
        # [spec-decode] tokens=89 time=0.815 s speed=109.23 tok/s steps=6 accepted=89/96 (92.7%) avg_commit=14.83
        m = re.search(r'\[spec-decode\].*?speed=([\d.]+) tok/s.*?\(([0-9.]+)%\).*?avg_commit=([\d.]+)', line)
        if m:
            decode_tps  = float(m.group(1))
            accept_pct  = float(m.group(2))
            avg_commit  = float(m.group(3))

        # [spec-gate] floor reason=slow ema_ratio=0.41 ...
        m2 = re.search(r'\[spec-gate\] floor reason=(\S+)', line)
        if m2:
            gate_floor_reason = m2.group(1)

        # [n-gate] floor reason=slow ...
        m3 = re.search(r'\[n-gate\] floor reason=(\S+)', line)
        if m3:
            gate_floor_reason = m3.group(1)

    return accept_pct, avg_commit, decode_tps, gate_floor_reason

def parse_chat_done_tps(log_path):
    """Parse decode TPS from DONE line: decode=X.Xs(YYY tok/s)"""
    try:
        with open(log_path) as f:
            content = f.read()
    except Exception:
        return None
    # Take the LAST DONE line
    matches = re.findall(r'decode=[\d.]+s\(([\d.]+)tok/s\)', content)
    if matches:
        return float(matches[-1])
    return None

def check_dflash_feature_line(log_path):
    """Return the mirror dtype and cap from log: [dflash-feature] mirror dtype=X cap=Y fc_in=Z"""
    try:
        with open(log_path) as f:
            content = f.read()
    except Exception:
        return None, None
    m = re.search(r'\[dflash-feature\] mirror dtype=(\S+) cap=(\d+)', content)
    if m:
        return m.group(1), int(m.group(2))
    return None, None

def check_draft_ctx_in_log(log_path):
    """Try to find the actual draft_ctx used from log lines."""
    # The backend logs a draft_ctx in spec-gate/n-gate lines occasionally.
    # Also look for any "draft_ctx=" log lines.
    try:
        with open(log_path) as f:
            content = f.read()
    except Exception:
        return None
    m = re.search(r'draft_ctx=(\d+)', content)
    if m:
        return int(m.group(1))
    return None


def run_cell(dtype, draft_ctx_max_str):
    tag = f"dt{dtype}_dc{draft_ctx_max_str}"
    log_path = f"{BENCH_DIR}/isolation2x2_{tag}.log"
    print(f"\n{'='*60}")
    print(f"CELL: FEATURE_DTYPE={dtype}  DRAFT_CTX_MAX={draft_ctx_max_str}")
    print(f"Log: {log_path}")
    print(f"{'='*60}")

    proc, log_fd = launch_server(dtype, draft_ctx_max_str, log_path)
    print(f"Server PID: {proc.pid}")

    healthy = wait_healthy()
    if not healthy:
        print("ERROR: Server did not become healthy within timeout")
        kill_server(proc, log_fd)
        return None

    # Verify mirror dtype from log
    mirror_dtype, mirror_cap = check_dflash_feature_line(log_path)
    print(f"Confirmed mirror dtype={mirror_dtype} cap={mirror_cap}")

    cell_results = []

    for prompt_file, prompt_tag in PROMPTS:
        prompt_path = f"{BENCH_DIR}/{prompt_file}"
        print(f"\n  Prompt: {prompt_tag} ({prompt_file})")
        t0 = time.time()
        resp, err = post_prompt(prompt_path)
        elapsed = time.time() - t0
        print(f"  Wall: {elapsed:.1f}s")

        if err:
            print(f"  ERROR: {err}")
            cell_results.append({
                "dtype": dtype,
                "draft_ctx_max": draft_ctx_max_str,
                "prompt": prompt_tag,
                "status": "ERROR",
                "error": err,
            })
            continue

        # Parse from log (after the last request)
        accept_pct, avg_commit, log_tps, gate_floor = parse_last_spec_stats_from_log(log_path)
        done_tps = parse_chat_done_tps(log_path)

        # Also get from response
        usage = resp.get("usage", {}) if resp else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        choices = resp.get("choices", []) if resp else []
        output = choices[0].get("message", {}).get("content", "") if choices else ""

        print(f"  accept%={accept_pct}  avg_commit={avg_commit}  spec-tps={log_tps}  done-tps={done_tps}")
        print(f"  gate_floor={gate_floor}")
        print(f"  prompt_tokens={prompt_tokens}  completion_tokens={completion_tokens}")
        print(f"  output[:100]: {output[:100]!r}")

        cell_results.append({
            "dtype": dtype,
            "draft_ctx_max": draft_ctx_max_str,
            "mirror_dtype_confirmed": mirror_dtype,
            "mirror_cap": mirror_cap,
            "prompt": prompt_tag,
            "status": "OK",
            "accept_pct": accept_pct,
            "avg_commit": avg_commit,
            "decode_tps_spec": log_tps,
            "decode_tps_done": done_tps,
            "gate_floor": gate_floor,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "wall_s": round(elapsed, 1),
        })

    kill_server(proc, log_fd)
    print(f"\nCell {tag} done.")
    return cell_results


def main():
    # Acquire GPU lock
    lock_fd = open("/tmp/lucebox_gpu.lock", "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print("ERROR: GPU lock held by another process. Aborting.")
        sys.exit(1)

    try:
        all_results = []
        for dtype, dc_str in CELLS:
            results = run_cell(dtype, dc_str)
            if results:
                all_results.extend(results)
            else:
                all_results.append({
                    "dtype": dtype,
                    "draft_ctx_max": dc_str,
                    "status": "SERVER_FAIL",
                })

        # Save raw results
        out_json = f"{BENCH_DIR}/isolation2x2_results.json"
        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nRaw results saved: {out_json}")

        # Print summary table
        print("\n" + "="*100)
        print("SUMMARY TABLE")
        print("="*100)
        hdr = f"{'FEATURE_DTYPE':<14} {'DRAFT_CTX_MAX':<14} {'prompt':<14} {'accept%':>8} {'AL':>7} {'decode tok/s':>13} {'gate_floor':<12}"
        print(hdr)
        print("-"*100)
        for r in all_results:
            if r.get("status") != "OK":
                print(f"{r.get('dtype','?'):<14} {r.get('draft_ctx_max','?'):<14} {'ERROR':<14}")
                continue
            accept_s   = f"{r['accept_pct']:.1f}%" if r.get('accept_pct') is not None else "---"
            al_s       = f"{r['avg_commit']:.2f}" if r.get('avg_commit') is not None else "---"
            tps_s      = f"{r.get('decode_tps_done') or r.get('decode_tps_spec') or '---':.1f}" if (r.get('decode_tps_done') or r.get('decode_tps_spec')) else "---"
            gate_s     = r.get('gate_floor') or "-"
            print(f"{r['dtype']:<14} {r['draft_ctx_max']:<14} {r['prompt']:<14} {accept_s:>8} {al_s:>7} {tps_s:>13} {gate_s:<12}")

        return all_results

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


if __name__ == "__main__":
    results = main()
    print("\n=== DONE ===")
