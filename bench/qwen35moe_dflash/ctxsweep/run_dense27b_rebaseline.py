#!/usr/bin/env python3
"""
Dense 27B cold-start rebaseline.
3 prompts × 1 config = 3 cells. One fresh server per cell. Never reuses a server.
Matching the MoE re-baseline protocol exactly for like-vs-like comparison.
"""
import subprocess
import time
import json
import os
import re
import signal
import sys
import datetime
import tempfile

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCH_DIR     = "/home/peppi/Dev/lucebox-hub/bench/qwen35moe_dflash/ctxsweep"
SERVER_BIN    = "/home/peppi/Dev/lucebox-hub/server/build/dflash_server"
TARGET_MODEL  = "/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf"
DRAFT_MODEL   = "/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-bf16-reconverted.gguf"
CHAT_TEMPLATE = "/home/peppi/models/qwen3.6-27b-chat-template.jinja"
GPU_LOCK      = "/tmp/lucebox_gpu.lock"

PORT     = 18081
MAX_CTX  = 40960
MAX_CTX_FALLBACK = 24576  # retry at this if load OOM at max_ctx

# ---------------------------------------------------------------------------
# Prompts: (name, filepath)  — identical to MoE rebaseline subset
# ---------------------------------------------------------------------------
PROMPTS = [
    ("ctx_008192",      f"{BENCH_DIR}/ctx_008192.json"),
    ("ctx_032768",      f"{BENCH_DIR}/ctx_032768.json"),
    ("needle_deep_12k", f"{BENCH_DIR}/needle_deep_12k.json"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def health_poll(port, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            result = subprocess.run(
                ["curl", "-sf", f"http://127.0.0.1:{port}/health"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and "ok" in result.stdout.lower():
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def launch_server(max_ctx, log_path):
    """Launch dflash_server with the dense-27B config. Returns (proc, log_fh)."""
    env = os.environ.copy()
    env["DFLASH_FEAT_RING_CAP"] = "40960"
    # Do NOT set DFLASH_FEATURE_DTYPE — f32 mirror is the default
    env.pop("DFLASH_FEATURE_DTYPE", None)
    env.pop("DFLASH_DRAFT_CTX_MAX", None)

    cmd = [
        SERVER_BIN,
        TARGET_MODEL,
        "--draft", DRAFT_MODEL,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-ctx", str(max_ctx),
        "--max-tokens", "200",
        "--fa-window", "0",
        "--cache-type-k", "q4_0",
        "--cache-type-v", "q4_0",
        "--chat-template-file", CHAT_TEMPLATE,
        "--model-name", "luce-dflash-27b",
        "--lazy-draft",
    ]
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_fh, stderr=log_fh,
        start_new_session=True
    )
    return proc, log_fh


def send_request(json_path):
    with open(json_path) as f:
        payload = json.load(f)
    payload["temperature"] = 0
    payload["max_tokens"] = 200

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(payload, tf)
        tmp_path = tf.name

    try:
        t0 = time.time()
        result = subprocess.run(
            ["curl", "-sf", "-X", "POST",
             f"http://127.0.0.1:{PORT}/v1/chat/completions",
             "-H", "Content-Type: application/json",
             "-d", f"@{tmp_path}",
             "--max-time", "600"],
            capture_output=True, text=True
        )
        elapsed = time.time() - t0
    finally:
        os.unlink(tmp_path)

    return result.stdout, elapsed


def kill_server(proc, log_fh):
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    try:
        subprocess.run(["fuser", "-k", f"{PORT}/tcp"], capture_output=True, timeout=5)
    except Exception:
        pass
    time.sleep(4)
    try:
        proc.wait(timeout=8)
    except Exception:
        pass
    log_fh.flush()
    log_fh.close()


def parse_server_log(log_path):
    result = {
        "spec_decode": None,
        "ar_decode": None,
        "spec_gate": None,
        "server_done": None,
        "oom": False,
        "load_error": None,
        "raw_lines": [],
    }
    try:
        with open(log_path) as f:
            lines = f.readlines()
    except Exception:
        return result

    for line in lines:
        s = line.strip()
        if "out of memory" in s.lower() or "OOM" in s or "CUDA error" in s.lower():
            result["oom"] = True
        # Architecture mismatch or drafter load failure
        if ("failed to load" in s.lower() or "architecture" in s.lower() and "mismatch" in s.lower()
                or "error loading model" in s.lower()):
            result["load_error"] = s
        if "[spec-decode]" in s and "tokens=" in s and "accepted=" in s:
            result["spec_decode"] = s
        elif "[ar-decode]" in s and "tokens=" in s and result["spec_decode"] is None:
            result["ar_decode"] = s
        if "[spec-gate]" in s:
            result["spec_gate"] = s
        if "[server] chat DONE" in s:
            result["server_done"] = s

    result["raw_lines"] = lines
    return result


def parse_accept(spec_line):
    if not spec_line:
        return None
    m = re.search(r'accepted=\d+/\d+\s+\(([0-9.]+)%\)', spec_line)
    return float(m.group(1)) if m else None


def parse_avg_commit(spec_line):
    if not spec_line:
        return None
    m = re.search(r'avg_commit=([0-9.]+)', spec_line)
    return float(m.group(1)) if m else None


def parse_decode_tps_from_done(server_done_line):
    if not server_done_line:
        return None
    m = re.search(r'decode=[0-9.]+s\(([0-9.]+)tok/s\)', server_done_line)
    return float(m.group(1)) if m else None


def parse_prefill_s(server_done_line):
    if not server_done_line:
        return None
    m = re.search(r'prefill=([0-9.]+)s', server_done_line)
    return float(m.group(1)) if m else None


def parse_wall_s(server_done_line):
    if not server_done_line:
        return None
    m = re.search(r'wall=([0-9.]+)s', server_done_line)
    return float(m.group(1)) if m else None


def parse_prompt_tok_from_done(server_done_line):
    if not server_done_line:
        return None
    m = re.search(r'\bin=(\d+)\b', server_done_line)
    return int(m.group(1)) if m else None


def check_needle_in_response(response_text):
    try:
        d = json.loads(response_text)
        content = d["choices"][0]["message"]["content"]
        return "luce_marker_widget" in content
    except Exception:
        return False


def run_cell(prompt_name, prompt_path, max_ctx, label_suffix=""):
    ts = datetime.datetime.now().strftime("%H%M%S")
    log_name = f"dense27b_{prompt_name}{label_suffix}_{ts}.log"
    log_path = os.path.join(BENCH_DIR, log_name)
    is_needle = "needle" in prompt_name

    print(f"\n{'='*70}")
    print(f"  CELL: dense27b x {prompt_name}{label_suffix}  max_ctx={max_ctx}")
    print(f"  log: {log_name}")
    print(f"{'='*70}")

    proc, log_fh = launch_server(max_ctx, log_path)
    print(f"  Server PID={proc.pid} launched, polling health (up to 180s)...")

    ready = health_poll(PORT, timeout=180)
    if not ready:
        # Peek at log for load error / OOM before declaring failure
        log_fh.flush()
        parsed_early = parse_server_log(log_path)
        if parsed_early["load_error"]:
            print(f"  FATAL: drafter load error: {parsed_early['load_error']}")
            kill_server(proc, log_fh)
            return {"prompt": prompt_name, "max_ctx": max_ctx, "error": "LOAD_FAIL",
                    "error_detail": parsed_early["load_error"], "log": log_name}
        if parsed_early["oom"]:
            print(f"  OOM detected during load at max_ctx={max_ctx}")
            kill_server(proc, log_fh)
            return {"prompt": prompt_name, "max_ctx": max_ctx, "error": "OOM_LOAD", "log": log_name}
        print("  ERROR: Server did not become healthy in 180s")
        kill_server(proc, log_fh)
        return {"prompt": prompt_name, "max_ctx": max_ctx, "error": "NOT_HEALTHY", "log": log_name}

    print("  Server ready. Sending one request...")
    response_text, req_elapsed = send_request(prompt_path)
    print(f"  Request done in {req_elapsed:.1f}s")

    kill_server(proc, log_fh)
    print("  Server killed.")

    parsed = parse_server_log(log_path)

    if parsed["load_error"]:
        print(f"  FATAL: drafter load error: {parsed['load_error']}")
        return {"prompt": prompt_name, "max_ctx": max_ctx, "error": "LOAD_FAIL",
                "error_detail": parsed["load_error"], "log": log_name}

    if parsed["oom"]:
        print("  OOM detected!")
        return {"prompt": prompt_name, "max_ctx": max_ctx, "error": "OOM", "log": log_name}

    accept_pct = parse_accept(parsed["spec_decode"])
    avg_commit = parse_avg_commit(parsed["spec_decode"])
    decode_tps = parse_decode_tps_from_done(parsed["server_done"])
    prefill_s  = parse_prefill_s(parsed["server_done"])
    wall_s     = parse_wall_s(parsed["server_done"])
    prompt_tok = parse_prompt_tok_from_done(parsed["server_done"])
    gate_line  = parsed["spec_gate"]
    is_ar      = parsed["spec_decode"] is None and parsed["ar_decode"] is None

    gate_floor_reason = "N/A"
    if gate_line:
        if "floor reason=" in gate_line:
            m = re.search(r'floor reason=(\S+)', gate_line)
            gate_floor_reason = m.group(1) if m else "floored"
        elif "held" in gate_line:
            gate_floor_reason = "held"
        elif "AR" in gate_line:
            gate_floor_reason = "AR"

    needle_hit = None
    if is_needle:
        needle_hit = check_needle_in_response(response_text)

    result = {
        "prompt":      prompt_name,
        "max_ctx":     max_ctx,
        "prompt_tok":  prompt_tok,
        "accept_pct":  accept_pct,
        "avg_commit":  avg_commit,
        "decode_tps":  decode_tps,
        "prefill_s":   prefill_s,
        "wall_s":      wall_s,
        "gate_floor":  gate_floor_reason,
        "ar_floor":    is_ar,
        "needle_hit":  needle_hit,
        "log":         log_name,
        "spec_line":   parsed["spec_decode"],
        "ar_line":     parsed["ar_decode"],
        "gate_line":   gate_line,
        "server_done": parsed["server_done"],
    }

    print(f"  accept={accept_pct}%  avg_commit={avg_commit}  decode_tps={decode_tps}tok/s")
    print(f"  prefill={prefill_s}s  wall={wall_s}s  prompt_tok={prompt_tok}")
    print(f"  gate={gate_floor_reason}  ar_floor={is_ar}")
    if is_needle:
        print(f"  needle_hit={needle_hit}")

    return result


def main():
    # Verify no server already on port
    chk = subprocess.run(["fuser", f"{PORT}/tcp"], capture_output=True, text=True)
    if chk.stdout.strip():
        print(f"ERROR: Port {PORT} already in use (PID {chk.stdout.strip()}). Aborting.")
        sys.exit(1)

    # GPU check
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.free,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    print(f"GPU: {gpu.stdout.strip()}")
    try:
        parts = gpu.stdout.strip().split(",")
        free_mb = int(parts[1].strip())
        if free_mb < 20000:
            print(f"WARNING: Only {free_mb}MB free — less than 20GB. Results may be contended.")
    except Exception:
        pass

    print(f"\nBinary md5: ", end="", flush=True)
    md5_out = subprocess.run(["md5sum", SERVER_BIN], capture_output=True, text=True)
    print(md5_out.stdout.strip())

    print(f"Target:  {TARGET_MODEL}")
    print(f"Drafter: {DRAFT_MODEL}")
    print(f"Max ctx: {MAX_CTX}")
    print(f"Port:    {PORT}  (flock: {GPU_LOCK})")

    results = []
    total = len(PROMPTS)

    for i, (prom_name, prom_path) in enumerate(PROMPTS):
        print(f"\n[{i+1}/{total}] Starting cell...")
        cell = run_cell(prom_name, prom_path, MAX_CTX)
        results.append(cell)

        # If OOM at max_ctx, retry at fallback ctx
        if cell.get("error") in ("OOM_LOAD", "OOM"):
            print(f"\n  OOM at max_ctx={MAX_CTX}. Retrying at max_ctx={MAX_CTX_FALLBACK}...")
            time.sleep(4)
            cell2 = run_cell(prom_name, prom_path, MAX_CTX_FALLBACK, label_suffix="_fallback")
            results.append(cell2)

        # If LOAD_FAIL, stop immediately — drafter can't load
        if cell.get("error") == "LOAD_FAIL":
            print(f"\nFATAL: Drafter failed to load. Stopping bench. Error: {cell.get('error_detail')}")
            break

        time.sleep(2)

    # Save results JSON
    results_path = os.path.join(BENCH_DIR, "dense27b_rebaseline_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Print summary table
    print("\n" + "="*110)
    print("DENSE 27B REBASELINE SUMMARY")
    print("="*110)
    print(f"{'Prompt':<22} {'ctx':<7} {'ptok':<8} {'accept%':<9} {'AL':<6} {'dec_tps':<10} {'prefill_s':<11} {'wall_s':<8} {'gate':<10} {'needle'}")
    print("-"*110)

    for r in results:
        if "error" in r:
            print(f"{r['prompt']:<22} {r.get('max_ctx','?'):<7} {'ERR':>8}  {r['error']}  {r.get('error_detail','')[:60]}")
            continue
        accept_str = f"{r['accept_pct']:.1f}%" if r['accept_pct'] is not None else ("AR_floor" if r['ar_floor'] else "N/A")
        al_str     = f"{r['avg_commit']:.2f}" if r['avg_commit'] is not None else "N/A"
        dtps_str   = f"{r['decode_tps']:.1f}" if r['decode_tps'] is not None else "N/A"
        pre_str    = f"{r['prefill_s']:.1f}s" if r['prefill_s'] is not None else "N/A"
        wall_str   = f"{r['wall_s']:.1f}s" if r['wall_s'] is not None else "N/A"
        needle_str = str(r['needle_hit']) if r['needle_hit'] is not None else "-"
        print(f"{r['prompt']:<22} {r.get('max_ctx','?'):<7} {str(r['prompt_tok'] or '?'):<8} {accept_str:<9} {al_str:<6} {dtps_str:<10} {pre_str:<11} {wall_str:<8} {r.get('gate_floor','?'):<10} {needle_str}")

    print()

    # Side-by-side comparison table vs MoE
    print("="*110)
    print("DENSE 27B-Q4 vs MoE 35B-A3B-Q3 — SAME PROMPTS")
    print("  MoE numbers: ctx_008192: 76.8%/AL12.86/174.6tok/s/4.4s prefill")
    print("               ctx_032768: 76.8%/AL12.86/171.3tok/s/17.1s prefill")
    print("               needle_deep_12k: 28.7%/AL5.29/78.3tok/s/7.5s prefill")
    print("="*110)
    print(f"{'Prompt':<22} {'DENSE 27B-Q4 accept':<22} {'DENSE 27B-Q4 decode':<22} {'MoE 35B-A3B-Q3 accept':<24} {'MoE 35B-A3B-Q3 decode'}")
    print("-"*110)
    moe_ref = {
        "ctx_008192":      ("76.8%", "174.6 tok/s"),
        "ctx_032768":      ("76.8%", "171.3 tok/s"),
        "needle_deep_12k": ("28.7%", "78.3 tok/s"),
    }
    for r in results:
        if "error" in r:
            continue
        pname = r["prompt"]
        if pname not in moe_ref:
            continue
        accept_str = f"{r['accept_pct']:.1f}%" if r['accept_pct'] is not None else ("AR_floor" if r.get('ar_floor') else "N/A")
        dtps_str   = f"{r['decode_tps']:.1f} tok/s" if r['decode_tps'] is not None else "N/A"
        moe_accept, moe_decode = moe_ref[pname]
        print(f"{pname:<22} {accept_str:<22} {dtps_str:<22} {moe_accept:<24} {moe_decode}")

    print()

    return results


if __name__ == "__main__":
    main()
