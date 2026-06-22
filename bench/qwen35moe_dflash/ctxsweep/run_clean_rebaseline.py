#!/usr/bin/env python3
"""
Clean cold-start benchmark: 3 configs × 4 prompts = 12 cells.
One fresh server per cell. Never reuses a server.
"""
import subprocess
import time
import json
import os
import re
import signal
import sys
import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCH_DIR = "/home/peppi/Dev/lucebox-hub/bench/qwen35moe_dflash/ctxsweep"
SERVER_BIN = "/home/peppi/Dev/lucebox-hub/server/build/dflash_server"
TARGET_MODEL = "/home/peppi/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
DRAFT_MODEL  = "/home/peppi/models/qwen3.6-35b-a3b-dflash-new/qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf"
CHAT_TEMPLATE = "/home/peppi/models/qwen3-coder-chat-template.jinja"
GPU_LOCK = "/tmp/lucebox_gpu.lock"

PORT = 18081
MAX_CTX = 40960

# ---------------------------------------------------------------------------
# Configs: (name, RING, DT)
# ---------------------------------------------------------------------------
CONFIGS = [
    ("C1_fixed",  40960, "f32"),  # corrected recipe candidate
    ("C0_cliff",  4096,  "f32"),  # old default / "before"
    ("Cf16",      40960, "f16"),  # confirm f16 regression cold
]

# ---------------------------------------------------------------------------
# Prompts: (name, filepath)
# ---------------------------------------------------------------------------
PROMPTS = [
    ("ctx_008192",      f"{BENCH_DIR}/ctx_008192.json"),
    ("ctx_032768",      f"{BENCH_DIR}/ctx_032768.json"),
    ("needle_mid_06k",  f"{BENCH_DIR}/needle_mid_06k.json"),
    ("needle_deep_12k", f"{BENCH_DIR}/needle_deep_12k.json"),
]


def count_prompt_tokens_approx(json_path):
    """Rough token estimate: chars/3.5."""
    with open(json_path) as f:
        d = json.load(f)
    content = ""
    for msg in d.get("messages", []):
        c = msg.get("content", "")
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    content += part.get("text", "")
        else:
            content += c
    return int(len(content) / 3.5)


def health_poll(port, timeout=120):
    """Poll /health until ready or timeout."""
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


def launch_server(ring, dtype, log_path):
    """Launch dflash_server directly (no flock wrapper — we serialize cells).
    Returns (proc, log_fh).
    """
    env = os.environ.copy()
    env["DFLASH_FEAT_RING_CAP"] = str(ring)
    env["DFLASH_FEATURE_DTYPE"] = dtype
    # Explicitly unset DFLASH_DRAFT_CTX_MAX (ignored on MoE anyway)
    env.pop("DFLASH_DRAFT_CTX_MAX", None)

    cmd = [
        SERVER_BIN,
        TARGET_MODEL,
        "--draft", DRAFT_MODEL,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-ctx", str(MAX_CTX),
        "--max-tokens", "200",
        "--fa-window", "0",
        "--cache-type-k", "q4_0",
        "--cache-type-v", "q4_0",
        "--chat-template-file", CHAT_TEMPLATE,
        "--model-name", "luce-dflash",
        "--lazy-draft",
    ]
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_fh, stderr=log_fh,
        start_new_session=True  # creates own process group for clean killpg
    )
    return proc, log_fh


def send_request(json_path):
    """POST the request, return (response_text, elapsed_s).
    Uses a temp file for the payload to avoid ARG_MAX issues with large prompts.
    """
    import tempfile
    with open(json_path) as f:
        payload = json.load(f)
    # Force temp=0, max_tokens=200
    payload["temperature"] = 0
    payload["max_tokens"] = 200

    # Write payload to a temp file so curl can use @file syntax
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
    """Kill server hard (entire process group), wait, close log."""
    # Kill entire process group to catch any child processes
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    # Also kill by port to be sure (catches orphans)
    try:
        subprocess.run(
            ["fuser", "-k", f"{PORT}/tcp"],
            capture_output=True, timeout=5
        )
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
    """Extract spec-decode, ar-decode, spec-gate, server DONE lines."""
    result = {
        "spec_decode": None,
        "ar_decode": None,
        "spec_gate": None,
        "server_done": None,
        "oom": False,
        "raw_lines": [],
    }
    try:
        with open(log_path) as f:
            lines = f.readlines()
    except Exception:
        return result

    for line in lines:
        line = line.strip()
        if "out of memory" in line.lower() or "OOM" in line or "CUDA error" in line.lower():
            result["oom"] = True
        if "[spec-decode]" in line and "tokens=" in line and "accepted=" in line:
            result["spec_decode"] = line
        elif "[ar-decode]" in line and "tokens=" in line and result["spec_decode"] is None:
            # Only record AR if no spec-decode (pure AR mode)
            result["ar_decode"] = line
        if "[spec-gate]" in line:
            result["spec_gate"] = line
        if "[server] chat DONE" in line:
            result["server_done"] = line

    result["raw_lines"] = lines
    return result


def parse_accept(spec_line):
    """Extract accept% from [spec-decode] line."""
    if not spec_line:
        return None
    m = re.search(r'accepted=\d+/\d+\s+\(([0-9.]+)%\)', spec_line)
    return float(m.group(1)) if m else None


def parse_avg_commit(spec_line):
    """Extract avg_commit from [spec-decode] line."""
    if not spec_line:
        return None
    m = re.search(r'avg_commit=([0-9.]+)', spec_line)
    return float(m.group(1)) if m else None


def parse_decode_tps(spec_line):
    """Extract decode tok/s from [spec-decode] or server_done line."""
    if not spec_line:
        return None
    m = re.search(r'speed=([0-9.]+)\s+tok/s', spec_line)
    return float(m.group(1)) if m else None


def parse_prefill_s(server_done_line):
    """Extract prefill_s from [server] chat DONE line."""
    if not server_done_line:
        return None
    m = re.search(r'prefill=([0-9.]+)s', server_done_line)
    return float(m.group(1)) if m else None


def parse_decode_tps_from_done(server_done_line):
    """Extract decode TPS from server DONE line: decode=Xs(Ytok/s)."""
    if not server_done_line:
        return None
    m = re.search(r'decode=[0-9.]+s\(([0-9.]+)tok/s\)', server_done_line)
    return float(m.group(1)) if m else None


def parse_prompt_tok_from_done(server_done_line):
    """Extract in= token count from server DONE line."""
    if not server_done_line:
        return None
    m = re.search(r'\bin=(\d+)\b', server_done_line)
    return int(m.group(1)) if m else None


def check_needle_in_response(response_text):
    """Check if luce_marker_widget appears in the model's response."""
    try:
        d = json.loads(response_text)
        content = d["choices"][0]["message"]["content"]
        return "luce_marker_widget" in content
    except Exception:
        return False


def run_cell(config_name, ring, dtype, prompt_name, prompt_path):
    """Run one cold-start cell. Returns dict of metrics."""
    ts = datetime.datetime.now().strftime("%H%M%S")
    log_name = f"clean_{config_name}_{prompt_name}_{ts}.log"
    log_path = os.path.join(BENCH_DIR, log_name)
    is_needle = "needle" in prompt_name

    print(f"\n{'='*70}")
    print(f"  CELL: {config_name} x {prompt_name}")
    print(f"  ring={ring} dtype={dtype}")
    print(f"  log: {log_name}")
    print(f"{'='*70}")

    # Launch server
    proc, log_fh = launch_server(ring, dtype, log_path)
    print(f"  Server PID={proc.pid} launched, polling health...")

    ready = health_poll(PORT, timeout=120)
    if not ready:
        print("  ERROR: Server did not become healthy in 120s")
        kill_server(proc, log_fh)
        return {
            "config": config_name, "prompt": prompt_name,
            "ring": ring, "dtype": dtype,
            "error": "server_not_healthy",
        }

    print("  Server ready. Sending one request...")
    response_text, req_elapsed = send_request(prompt_path)
    print(f"  Request done in {req_elapsed:.1f}s")

    # Kill immediately after the one request
    kill_server(proc, log_fh)
    print("  Server killed.")

    # Parse log
    parsed = parse_server_log(log_path)

    if parsed["oom"]:
        print("  OOM detected!")
        return {
            "config": config_name, "prompt": prompt_name,
            "ring": ring, "dtype": dtype,
            "error": "OOM",
        }

    accept_pct  = parse_accept(parsed["spec_decode"])
    avg_commit  = parse_avg_commit(parsed["spec_decode"])
    decode_tps  = parse_decode_tps_from_done(parsed["server_done"])
    prefill_s   = parse_prefill_s(parsed["server_done"])
    prompt_tok  = parse_prompt_tok_from_done(parsed["server_done"])
    gate_line   = parsed["spec_gate"]
    is_ar       = parsed["spec_decode"] is None

    # Gate floor reason
    gate_floor_reason = "N/A"
    if gate_line:
        if "floor reason=" in gate_line:
            m = re.search(r'floor reason=(\S+)', gate_line)
            gate_floor_reason = m.group(1) if m else "floored"
        elif "held" in gate_line:
            gate_floor_reason = "held"

    # Needle check
    needle_hit = None
    if is_needle:
        needle_hit = check_needle_in_response(response_text)

    result = {
        "config":         config_name,
        "prompt":         prompt_name,
        "ring":           ring,
        "dtype":          dtype,
        "prompt_tok":     prompt_tok,
        "accept_pct":     accept_pct,
        "avg_commit":     avg_commit,
        "decode_tps":     decode_tps,
        "prefill_s":      prefill_s,
        "gate_floor":     gate_floor_reason,
        "ar_floor":       is_ar,
        "needle_hit":     needle_hit,
        "log":            log_name,
        "spec_line":      parsed["spec_decode"],
        "gate_line":      gate_line,
        "server_done":    parsed["server_done"],
    }

    print(f"  accept={accept_pct}%  avg_commit={avg_commit}  decode_tps={decode_tps}")
    print(f"  prefill={prefill_s}s  prompt_tok={prompt_tok}  gate={gate_floor_reason}")
    if is_needle:
        print(f"  needle_hit={needle_hit}")

    return result


def main():
    print(f"Binary md5: ", end="", flush=True)
    md5_out = subprocess.run(["md5sum", SERVER_BIN], capture_output=True, text=True)
    print(md5_out.stdout.strip())

    results = []
    total = len(CONFIGS) * len(PROMPTS)
    cell_num = 0

    for cfg_name, ring, dtype in CONFIGS:
        for prom_name, prom_path in PROMPTS:
            cell_num += 1
            print(f"\n[{cell_num}/{total}] Starting cell...")
            cell = run_cell(cfg_name, ring, dtype, prom_name, prom_path)
            results.append(cell)
            # Brief pause between cells to let GPU settle
            time.sleep(2)

    # Save results JSON
    results_path = os.path.join(BENCH_DIR, "clean_rebaseline_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Print summary table
    print("\n" + "="*100)
    print("CLEAN REBASELINE — 12-CELL SUMMARY")
    print("="*100)
    print(f"{'Config':<12} {'Prompt':<20} {'ring':<8} {'dtype':<6} {'ptok':<8} {'accept%':<9} {'AL':<6} {'dec_tps':<9} {'prefill_s':<10} {'gate':<10} {'needle'}")
    print("-"*100)

    for r in results:
        if "error" in r:
            print(f"{r['config']:<12} {r['prompt']:<20} {r['ring']:<8} {r['dtype']:<6} {'ERR':>8} {r['error']}")
            continue
        accept_str  = f"{r['accept_pct']:.1f}%" if r['accept_pct'] is not None else ("AR_floor" if r['ar_floor'] else "N/A")
        commit_str  = f"{r['avg_commit']:.2f}" if r['avg_commit'] is not None else "-"
        tps_str     = f"{r['decode_tps']:.1f}" if r['decode_tps'] is not None else "-"
        pref_str    = f"{r['prefill_s']:.1f}" if r['prefill_s'] is not None else "-"
        ptok_str    = str(r['prompt_tok']) if r['prompt_tok'] else "-"
        needle_str  = ("YES" if r['needle_hit'] else "NO") if r['needle_hit'] is not None else "-"
        gate_str    = r['gate_floor'] or "-"
        print(f"{r['config']:<12} {r['prompt']:<20} {r['ring']:<8} {r['dtype']:<6} {ptok_str:>8} {accept_str:<9} {commit_str:<6} {tps_str:<9} {pref_str:<10} {gate_str:<10} {needle_str}")

    print("="*100)
    return results


if __name__ == "__main__":
    main()
