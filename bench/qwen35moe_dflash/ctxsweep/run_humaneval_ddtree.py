#!/usr/bin/env python3
"""
HumanEval DDTree benchmark driver.

Target: reproduce and beat blog numbers on Qwen3.6-27B-Q4_K_M + dflash-draft:
  - Benchmark: HumanEval/0-9 from server/eval/humaneval_plus/humanevalplus.jsonl
  - Config:    --ddtree (budget 22), Q4_0 KV, DFLASH_FEAT_RING_CAP=4096
  - n_gen=256, temperature=0 (greedy)
  - Blog targets: AR=37.78 / chain=112.82 / ddtree=129.52 tok/s, AL=8.31

Prompt source: REAL HumanEval+ dataset (server/eval/humaneval_plus/humanevalplus.jsonl)
               tasks HumanEval/0 through HumanEval/9 (first 10).

USAGE
-----
  # By default the script assumes a server is already running on PORT (18081).
  # Launch the server first (see launch_server_cmd() below or run_server flag).

  # Measure against a live server:
  python3 run_humaneval_ddtree.py

  # Also launch the server (requires GPU, flock /tmp/lucebox_gpu.lock):
  python3 run_humaneval_ddtree.py --run-server

  # Point at a different host/port:
  python3 run_humaneval_ddtree.py --url http://127.0.0.1:18081

  # Use a different server log for parsing (required when --run-server is off):
  python3 run_humaneval_ddtree.py --server-log /tmp/dflash_he_ddtree.log
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
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE         = Path(__file__).resolve().parent
REPO_ROOT    = HERE.parents[2]          # lucebox-hub/

HE_DATASET   = REPO_ROOT / "server/eval/humaneval_plus/humanevalplus.jsonl"
SERVER_BIN   = REPO_ROOT / "server/build/dflash_server"
TARGET_MODEL = Path("/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf")
DRAFT_MODEL  = Path("/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-bf16-reconverted.gguf")
CHAT_TMPL    = Path("/home/peppi/models/qwen3.6-27b-chat-template.jinja")
GPU_LOCK     = "/tmp/lucebox_gpu.lock"
RESULTS_PATH = HERE / "humaneval_ddtree_results.json"

PORT          = 18081                   # never :18099
DEFAULT_URL   = f"http://127.0.0.1:{PORT}/v1/chat/completions"
DEFAULT_LOG   = "/tmp/dflash_he_ddtree.log"
N_PROMPTS     = 10                      # HumanEval/0..9
MAX_TOKENS    = 256                     # matches blog n_gen=256
TEMPERATURE   = 0                       # greedy
DDTREE_BUDGET = 22                      # blog default
REQUEST_TIMEOUT = 300                   # seconds per request

# Blog reference numbers (for PASS/FAIL comparison)
BLOG_AR_TPS    = 37.78
BLOG_CHAIN_TPS = 112.82
BLOG_DDTREE_TPS = 129.52
BLOG_DDTREE_AL  = 8.31

# ---------------------------------------------------------------------------
# Prompt construction — REAL HumanEval+ dataset
# ---------------------------------------------------------------------------
def load_humaneval_prompts(n=N_PROMPTS):
    """
    Load first N prompts from the canonical HumanEval+ JSONL.
    Source: server/eval/humaneval_plus/humanevalplus.jsonl (164 tasks).
    Prompt format: bare function stub (Python), user asks for completion.
    """
    if not HE_DATASET.exists():
        raise FileNotFoundError(
            f"HumanEval+ dataset not found: {HE_DATASET}\n"
            "Expected at server/eval/humaneval_plus/humanevalplus.jsonl"
        )
    tasks = []
    with open(HE_DATASET) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
            if len(tasks) >= n:
                break
    if len(tasks) < n:
        raise ValueError(f"Dataset has only {len(tasks)} tasks, requested {n}")
    return tasks


def make_chat_payload(task):
    """
    Convert a HumanEval+ task dict into a chat-completions request body.

    The prompt is the raw function stub from the dataset. We instruct the
    model to complete the function body only (no explanation, no markdown).
    This matches the EvalPlus codegen convention for Python function completion.
    """
    stub = task["prompt"]
    user_msg = (
        "Complete the following Python function. "
        "Return ONLY the completed function body — no explanation, no markdown fences.\n\n"
        + stub
    )
    return {
        "model": "luce-dflash-27b",
        "messages": [{"role": "user", "content": user_msg}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }


# ---------------------------------------------------------------------------
# Server launch (GPU-gated, used only with --run-server)
# ---------------------------------------------------------------------------
def launch_server_cmd():
    """
    Return the full shell command that launches the dflash_server for this bench.
    NOT executed unless --run-server is passed.

    Flags:
      --ddtree           tree-verify ON  (server_main.cpp:431)
      --ddtree-budget 22 blog default    (server_main.cpp:434-435, backend_factory.h:54)
      --cache-type-k/v q4_0  Q4 KV quant (server_main.cpp:610-613)
      DFLASH_FEAT_RING_CAP=4096  rolling target_feat ring buffer matching blog config
      --lazy-draft       don't pre-load drafter weights into VRAM until first request
      --fa-window 0      unlimited attention window (default for this bench)
      flock /tmp/lucebox_gpu.lock  GPU exclusive access guard
    """
    env_prefix = f"DFLASH_FEAT_RING_CAP=4096"
    cmd = (
        f"flock {GPU_LOCK} "
        f"{env_prefix} "
        f"{SERVER_BIN} "
        f"{TARGET_MODEL} "
        f"--draft {DRAFT_MODEL} "
        f"--host 127.0.0.1 --port {PORT} "
        f"--max-ctx 8192 "
        f"--max-tokens {MAX_TOKENS} "
        f"--ddtree --ddtree-budget {DDTREE_BUDGET} "
        f"--cache-type-k q4_0 --cache-type-v q4_0 "
        f"--chat-template-file {CHAT_TMPL} "
        f"--model-name luce-dflash-27b "
        f"--lazy-draft"
    )
    return cmd


def launch_server(log_path):
    """Spawn the server in a child process. Returns (proc, log_fh)."""
    env = os.environ.copy()
    env["DFLASH_FEAT_RING_CAP"] = "4096"

    cmd = [
        str(SERVER_BIN),
        str(TARGET_MODEL),
        "--draft", str(DRAFT_MODEL),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-ctx", "8192",
        "--max-tokens", str(MAX_TOKENS),
        "--ddtree",
        "--ddtree-budget", str(DDTREE_BUDGET),
        "--cache-type-k", "q4_0",
        "--cache-type-v", "q4_0",
        "--chat-template-file", str(CHAT_TMPL),
        "--model-name", "luce-dflash-27b",
        "--lazy-draft",
    ]

    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_fh, stderr=log_fh,
        start_new_session=True
    )
    return proc, log_fh


def health_poll(port, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = subprocess.run(
                ["curl", "-sf", f"http://127.0.0.1:{port}/health"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0 and "ok" in r.stdout.lower():
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


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
    time.sleep(3)
    try:
        proc.wait(timeout=6)
    except Exception:
        pass
    log_fh.flush()
    log_fh.close()


# ---------------------------------------------------------------------------
# HTTP request
# ---------------------------------------------------------------------------
def post_request(url, payload):
    """POST chat-completions payload. Returns (response_json, wall_s)."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        data = json.loads(resp.read())
    return data, time.time() - t0


# ---------------------------------------------------------------------------
# Log parsing — same patterns as run_ctxsweep.py / run_dense27b_rebaseline.py
# ---------------------------------------------------------------------------

# [server] chat DONE ... prefill=0.42s decode=1.97s(129.9tok/s) error=-
# wall= field not emitted by this server; make it optional
RE_DONE   = re.compile(
    r'\[server\] chat DONE.*?prefill=([\d.]+)s\s+decode=([\d.]+)s\(([\d.]+)tok/s\)'
    r'(?:.*?wall=([\d.]+)s)?'
)
RE_DONE_IN = re.compile(r'\bin=(\d+)\b')
RE_DONE_OUT = re.compile(r'\bout=(\d+)\b')

# [spec-decode] tokens=256 accepted=213/256 (83.2%) avg_commit=8.53
RE_SPEC   = re.compile(
    r'\[spec-decode\].*?tokens=(\d+).*?accepted=(\d+)/(\d+).*?\(([\d.]+)%\).*?avg_commit=([\d.]+)'
)

# [ar-decode] tokens=256 decode=1.97s(37.3tok/s)
RE_AR     = re.compile(
    r'\[ar-decode\].*?tokens=(\d+).*?decode=[\d.]+s\(([\d.]+)tok/s\)'
)

# [spec-gate] ...
RE_GATE   = re.compile(r'\[spec-gate\].*')


def parse_log_tail(log_path, start_line):
    """Parse server log from start_line onward; return dict of per-request metrics."""
    try:
        with open(log_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return {}, len(lines) if 'lines' in dir() else start_line

    tail_lines = lines[start_line:]
    blob = "".join(tail_lines)
    new_end = len(lines)

    result = {
        "done_line":   None,
        "spec_line":   None,
        "ar_line":     None,
        "gate_line":   None,
        "decode_tps":  None,
        "prefill_s":   None,
        "decode_s":    None,
        "wall_s":      None,
        "prompt_tok":  None,
        "gen_tok":     None,
        "accept_pct":  None,
        "avg_commit":  None,
        "is_ar":       False,
    }

    for line in tail_lines:
        if "[server] chat DONE" in line:
            result["done_line"] = line.strip()
        if "[spec-decode]" in line and "accepted=" in line:
            result["spec_line"] = line.strip()
        if "[ar-decode]" in line and "tokens=" in line:
            result["ar_line"] = line.strip()
        if "[spec-gate]" in line:
            result["gate_line"] = line.strip()

    if result["done_line"]:
        m = RE_DONE.search(result["done_line"])
        if m:
            result["prefill_s"]  = float(m.group(1))
            result["decode_s"]   = float(m.group(2))
            result["decode_tps"] = float(m.group(3))
            result["wall_s"]     = float(m.group(4)) if m.group(4) else None
        m_in = RE_DONE_IN.search(result["done_line"])
        if m_in:
            result["prompt_tok"] = int(m_in.group(1))
        m_out = RE_DONE_OUT.search(result["done_line"])
        if m_out:
            result["gen_tok"] = int(m_out.group(1))

    if result["spec_line"]:
        m = RE_SPEC.search(result["spec_line"])
        if m:
            result["accept_pct"]  = float(m.group(4))
            result["avg_commit"]  = float(m.group(5))
    elif result["ar_line"]:
        result["is_ar"] = True
        m = RE_AR.search(result["ar_line"])
        if m:
            result["decode_tps"] = float(m.group(2))

    return result, new_end


# ---------------------------------------------------------------------------
# Main bench loop
# ---------------------------------------------------------------------------
def run_bench(url, server_log, tasks, verbose=True):
    """
    Drive N tasks against a live server; parse log for per-request metrics.

    Returns list of per-task result dicts.
    """
    # Count current log lines so we can slice from here onward per request
    log_line_cursor = 0
    if os.path.exists(server_log):
        with open(server_log) as f:
            log_line_cursor = sum(1 for _ in f)

    rows = []
    for i, task in enumerate(tasks):
        task_id    = task["task_id"]
        payload    = make_chat_payload(task)

        if verbose:
            print(f"\n[{i+1}/{len(tasks)}] {task_id}  ({task['entry_point']})")

        t0 = time.time()
        try:
            resp, wall_client = post_request(url, payload)
        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append({"task_id": task_id, "error": str(e)})
            continue
        # Give the server ~0.1 s to flush its log line
        time.sleep(0.15)

        metrics, log_line_cursor = parse_log_tail(server_log, log_line_cursor)

        row = {
            "task_id":      task_id,
            "entry_point":  task["entry_point"],
            "prompt_tok":   metrics.get("prompt_tok"),
            "gen_tok":      metrics.get("gen_tok"),
            "decode_tps":   metrics.get("decode_tps"),
            "prefill_s":    metrics.get("prefill_s"),
            "decode_s":     metrics.get("decode_s"),
            "wall_s":       metrics.get("wall_s"),
            "accept_pct":   metrics.get("accept_pct"),
            "avg_commit":   metrics.get("avg_commit"),    # = AL (avg tokens per spec step)
            "is_ar":        metrics.get("is_ar", False),
            "gate_line":    metrics.get("gate_line"),
            "client_wall_s": wall_client,
        }
        rows.append(row)

        if verbose:
            dtps  = f"{row['decode_tps']:.1f}" if row["decode_tps"] else "N/A"
            al    = f"{row['avg_commit']:.2f}"  if row["avg_commit"] else "N/A"
            acc   = f"{row['accept_pct']:.1f}%" if row["accept_pct"] else ("AR" if row["is_ar"] else "N/A")
            print(f"  decode={dtps} tok/s  AL={al}  accept={acc}  "
                  f"ptok={row['prompt_tok']}  gtok={row['gen_tok']}  "
                  f"prefill={row['prefill_s']}s  wall={row['wall_s']}s")

    return rows


def aggregate(rows):
    """Compute mean/max decode_tps and mean avg_commit (AL) over valid rows."""
    valid   = [r for r in rows if r.get("decode_tps") is not None and not r.get("error")]
    spec    = [r for r in valid if not r.get("is_ar")]
    ar_only = [r for r in valid if r.get("is_ar")]

    if not valid:
        return None

    all_tps   = [r["decode_tps"] for r in valid]
    spec_al   = [r["avg_commit"] for r in spec if r.get("avg_commit") is not None]

    return {
        "n_total":       len(rows),
        "n_valid":       len(valid),
        "n_spec":        len(spec),
        "n_ar":          len(ar_only),
        "mean_decode_tps": sum(all_tps) / len(all_tps),
        "max_decode_tps":  max(all_tps),
        "min_decode_tps":  min(all_tps),
        "mean_AL":         sum(spec_al) / len(spec_al) if spec_al else None,
        "max_AL":          max(spec_al) if spec_al else None,
    }


def print_summary(agg, rows):
    print("\n" + "=" * 70)
    print("HUMANEVAL DDTREE BENCH — SUMMARY")
    print("=" * 70)
    print(f"{'Task':<24} {'decode tok/s':>12} {'AL':>6} {'accept%':>8} "
          f"{'gtok':>5} {'prefill_s':>9} {'wall_s':>7}")
    print("-" * 70)
    for r in rows:
        if r.get("error"):
            print(f"{r['task_id']:<24}  ERROR: {r['error']}")
            continue
        dtps = f"{r['decode_tps']:.1f}" if r["decode_tps"] else "N/A"
        al   = f"{r['avg_commit']:.2f}"  if r["avg_commit"] else ("AR" if r["is_ar"] else "N/A")
        acc  = f"{r['accept_pct']:.1f}%" if r["accept_pct"] else ("AR" if r["is_ar"] else "N/A")
        gt   = str(r["gen_tok"] or "?")
        pre  = f"{r['prefill_s']:.2f}s"  if r["prefill_s"] else "N/A"
        wa   = f"{r['wall_s']:.2f}s"     if r["wall_s"]    else "N/A"
        print(f"{r['task_id']:<24} {dtps:>12} {al:>6} {acc:>8} {gt:>5} {pre:>9} {wa:>7}")

    if not agg:
        print("\nNo valid results to aggregate.")
        return

    print("-" * 70)
    print(f"{'MEAN':<24} {agg['mean_decode_tps']:>12.1f} "
          f"{agg['mean_AL'] or 0:>6.2f} "
          f"{'—':>8}  (n_valid={agg['n_valid']} spec={agg['n_spec']} ar={agg['n_ar']})")
    print(f"{'MAX':<24} {agg['max_decode_tps']:>12.1f} "
          f"{agg['max_AL'] or 0:>6.2f}")

    print("\n" + "=" * 70)
    print("COMPARISON vs BLOG TARGETS")
    print("=" * 70)
    print(f"  Blog AR (no ddtree):    {BLOG_AR_TPS:.2f} tok/s")
    print(f"  Blog chain:            {BLOG_CHAIN_TPS:.2f} tok/s")
    print(f"  Blog DDTree target:    {BLOG_DDTREE_TPS:.2f} tok/s  AL={BLOG_DDTREE_AL:.2f}")
    print()
    mean_tps = agg["mean_decode_tps"]
    mean_al  = agg["mean_AL"]
    tps_pass = mean_tps > BLOG_DDTREE_TPS
    al_pass  = (mean_al is not None) and (mean_al >= BLOG_DDTREE_AL)
    print(f"  This run mean decode:  {mean_tps:.2f} tok/s  {'PASS >129.52' if tps_pass else 'FAIL <129.52'}")
    if mean_al is not None:
        print(f"  This run mean AL:      {mean_al:.2f}        {'PASS >=8.31' if al_pass else 'FAIL <8.31'}")
    print(f"\n  OVERALL: {'PASS — beats blog DDTree target' if (tps_pass and al_pass) else 'NOT YET — check server log / flags'}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url",        default=DEFAULT_URL,  help="chat-completions URL")
    p.add_argument("--server-log", default=DEFAULT_LOG,  help="dflash_server stdout/stderr log path (for metric parsing)")
    p.add_argument("--n",          type=int, default=N_PROMPTS, help="Number of HumanEval prompts (default 10)")
    p.add_argument("--run-server", action="store_true",
                   help="Also launch the server (GPU required, flock GPU_LOCK)")
    p.add_argument("--quiet",      action="store_true",  help="Suppress per-request output")
    return p.parse_args()


def main():
    args = parse_args()

    # Print the launch command regardless — useful even without --run-server
    print("=" * 70)
    print("SERVER LAUNCH COMMAND (for reference / manual launch):")
    print("=" * 70)
    cmd_str = launch_server_cmd()
    print(f"  {cmd_str}")
    print()

    # Validate paths
    missing = []
    for p, label in [
        (HE_DATASET,   "HumanEval+ dataset"),
        (SERVER_BIN,   "dflash_server binary"),
        (TARGET_MODEL, "target model"),
        (DRAFT_MODEL,  "draft model"),
        (CHAT_TMPL,    "chat template"),
    ]:
        if not Path(p).exists():
            missing.append(f"  MISSING: {label}: {p}")
    if missing:
        for m in missing:
            print(m)
        if not args.run_server and Path(SERVER_BIN).exists() is False:
            print("\nNote: binary missing; build first with: cd server/build && make -j$(nproc)")
        if HE_DATASET in [str(m) for m in missing]:
            sys.exit("ERROR: HumanEval+ dataset missing — cannot proceed")

    # Load prompts
    tasks = load_humaneval_prompts(args.n)
    print(f"Loaded {len(tasks)} HumanEval+ prompts (source: REAL dataset, tasks 0-{len(tasks)-1})")
    print(f"  {HE_DATASET}")
    print(f"  First task: {tasks[0]['task_id']} ({tasks[0]['entry_point']})")
    print(f"  Last  task: {tasks[-1]['task_id']} ({tasks[-1]['entry_point']})")
    print()

    proc = None
    log_fh = None

    if args.run_server:
        print(f"--run-server: launching server (flock {GPU_LOCK})...")
        log_path = args.server_log
        proc, log_fh = launch_server(log_path)
        print(f"  Server PID={proc.pid}  log={log_path}")
        print("  Polling health (up to 180s)...")
        ok = health_poll(PORT)
        if not ok:
            print("  ERROR: server did not become healthy; aborting")
            kill_server(proc, log_fh)
            sys.exit(1)
        print("  Server ready.\n")
    else:
        print(f"Running against existing server at {args.url}")
        print(f"Parsing metrics from server log: {args.server_log}")
        if not os.path.exists(args.server_log):
            print(f"  WARNING: log file does not exist yet — metrics will be missing.")
            print(f"  Tip: launch server with stdout/stderr redirected to {args.server_log}")
        print()

    try:
        rows = run_bench(args.url, args.server_log, tasks, verbose=not args.quiet)
    finally:
        if proc is not None and log_fh is not None:
            kill_server(proc, log_fh)

    agg = aggregate(rows)
    print_summary(agg, rows)

    # Save results
    output = {
        "timestamp":   datetime.now().isoformat(),
        "config": {
            "target_model":  str(TARGET_MODEL),
            "draft_model":   str(DRAFT_MODEL),
            "chat_template": str(CHAT_TMPL),
            "port":          PORT,
            "max_tokens":    MAX_TOKENS,
            "temperature":   TEMPERATURE,
            "ddtree_budget": DDTREE_BUDGET,
            "dflash_feat_ring_cap": 4096,
            "cache_type_k":  "q4_0",
            "cache_type_v":  "q4_0",
            "prompt_source": str(HE_DATASET),
            "n_prompts":     len(tasks),
            "tasks":         [t["task_id"] for t in tasks],
        },
        "blog_targets": {
            "ar_tps":     BLOG_AR_TPS,
            "chain_tps":  BLOG_CHAIN_TPS,
            "ddtree_tps": BLOG_DDTREE_TPS,
            "ddtree_al":  BLOG_DDTREE_AL,
        },
        "aggregates": agg,
        "rows": rows,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
