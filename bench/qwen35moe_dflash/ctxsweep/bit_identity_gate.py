#!/usr/bin/env python3
"""
bit_identity_gate.py — AR-decode bit-identity merge gate for CUDA-graph refactors.

USAGE
-----
Before rebuilding for the refactor, copy the current binary:
  cp server/build/dflash_server server/build/dflash_server.golden

Then invoke the gate with:
  python3 bench/qwen35moe_dflash/ctxsweep/bit_identity_gate.py \\
    --golden-binary server/build/dflash_server.golden \\
    --test-binary   server/build/dflash_server \\
    --model         /path/to/model.gguf \\
    --stamp         20260622_120000

Context tiers tested: 4096, 32768, 71680 tokens.
A single divergence in any tier = GATE FAIL for that tier.
Overall PASS requires ALL tiers to be IDENTICAL.

DESIGN NOTES
------------
- Servers are launched on port 18081 (never :18099 which may be live).
- GPU lock: /tmp/lucebox_gpu.lock (exclusive flock, respects live sessions).
- AR-only mode: no --draft flag is passed. With no drafter loaded, the server
  runs pure autoregressive decode. This is apples-to-apples for the AR-graph
  refactor because spec-decode is NOT batch-invariant (changing draft depth K
  flips near-tie argmax via verify-batch FP reduction order).
- temperature=0 and seed=42 are sent in the request body. At temp=0 the
  sampler collapses to argmax so the seed only matters if the implementation
  has any stochastic tiebreak — treat any divergence as a real failure.
- stream=false: single-shot JSON response, easier to parse and compare.
- n_gen=128 tokens: enough to expose decode-path bugs without long wait.
- Prompt construction: concatenates ctx_*.json content to reach the target
  character count. Token ratio ~4 chars/token (empirically validated from
  existing ctx_*.json files).
"""

import argparse
import fcntl
import glob
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GATE_PORT       = 18081          # never touch :18099 (may be live user session)
GATE_HOST       = "127.0.0.1"
GPU_LOCK_PATH   = "/tmp/lucebox_gpu.lock"
SEED            = 42
N_GEN           = 128            # decode tokens per probe
SERVER_READY_TIMEOUT_S = 300     # seconds to wait for server health
CHARS_PER_TOKEN = 4.0            # empirical: ctx_032768.json = 131072 chars / 32768 tokens

CTXSWEEP_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def load_all_corpus() -> str:
    """Load all ctx_*.json content, sorted, and concatenate into one string."""
    files = sorted(glob.glob(os.path.join(CTXSWEEP_DIR, "ctx_*.json")))
    if not files:
        raise RuntimeError(
            f"No ctx_*.json files found in {CTXSWEEP_DIR}. "
            "Cannot build prompts."
        )
    parts = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        content = data["messages"][0]["content"]
        parts.append(content)
    return "\n".join(parts)


def build_prompt_content(target_tokens: int, corpus: str) -> str:
    """
    Return a content string of approximately target_tokens tokens.
    Truncates or repeats corpus as needed.
    """
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    if len(corpus) >= target_chars:
        return corpus[:target_chars]
    # Repeat corpus until we have enough chars.
    repeated = corpus
    while len(repeated) < target_chars:
        repeated = repeated + "\n" + corpus
    return repeated[:target_chars]


def build_request(model_name: str, content: str) -> bytes:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "seed": SEED,
        "max_tokens": N_GEN,
        "stream": False,
    }
    return json.dumps(payload).encode()


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def start_server(binary: str, model: str, log_path: str, extra_args: list[str]) -> subprocess.Popen:
    cmd = [
        binary, model,
        "--host", GATE_HOST,
        "--port", str(GATE_PORT),
        "--max-ctx", "131072",      # high enough for all tiers
        "--max-tokens", str(N_GEN),
        # NO --draft: forces pure AR decode.
    ] + extra_args
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=log_fh,
        start_new_session=True,
    )
    return proc


def wait_for_server(timeout_s: int = SERVER_READY_TIMEOUT_S) -> bool:
    url = f"http://{GATE_HOST}:{GATE_PORT}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model_name: str, content: str) -> str:
    """POST to the running server; return the assistant content text."""
    url = f"http://{GATE_HOST}:{GATE_PORT}/v1/chat/completions"
    body = build_request(model_name, content)
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read())
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response shape: {e}\n{json.dumps(result)[:400]}")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_outputs(golden: str, test: str) -> tuple[bool, str]:
    """
    Compare two output strings character by character.
    Returns (identical: bool, detail: str).
    """
    if golden == test:
        return True, "IDENTICAL"
    # Find first divergence.
    for i, (g, t) in enumerate(zip(golden, test)):
        if g != t:
            ctx_start = max(0, i - 20)
            snippet_g = repr(golden[ctx_start:i + 20])
            snippet_t = repr(test[ctx_start:i + 20])
            detail = (
                f"DIVERGE at char {i}: "
                f"golden={snippet_g} test={snippet_t}"
            )
            return False, detail
    # One is a prefix of the other.
    if len(golden) != len(test):
        detail = (
            f"LENGTH MISMATCH: golden={len(golden)} test={len(test)} chars. "
            f"Shorter ends at char {min(len(golden), len(test))}."
        )
        return False, detail
    return True, "IDENTICAL"


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def run_gate(
    golden_binary: str,
    test_binary:   str,
    model:         str,
    model_name:    str,
    tiers:         list[int],
    stamp:         str,
    extra_server_args: list[str],
    tmpdir:        str,
) -> dict:
    corpus = load_all_corpus()
    print(f"[gate] corpus loaded: {len(corpus):,} chars from {CTXSWEEP_DIR}/ctx_*.json")

    results = []
    overall_pass = True

    for tier in tiers:
        print(f"\n[gate] === tier {tier:,} tokens ===")
        content = build_prompt_content(tier, corpus)
        print(f"[gate] prompt content: {len(content):,} chars (~{len(content)/CHARS_PER_TOKEN:.0f} estimated tokens)")

        tier_result = {
            "tier_tokens": tier,
            "prompt_chars": len(content),
        }

        for role, binary in [("golden", golden_binary), ("test", test_binary)]:
            log_path = os.path.join(tmpdir, f"server_{role}_tier{tier}.log")
            print(f"[gate]   launching {role} binary: {binary}")
            proc = start_server(binary, model, log_path, extra_server_args)
            try:
                if not wait_for_server():
                    proc_log = open(log_path).read()[-2000:]
                    raise RuntimeError(
                        f"{role} server did not become healthy in {SERVER_READY_TIMEOUT_S}s.\n"
                        f"Log tail:\n{proc_log}"
                    )
                print(f"[gate]   {role} server ready, posting {N_GEN}-token inference...")
                t0 = time.time()
                output = run_inference(model_name, content)
                elapsed = time.time() - t0
                print(f"[gate]   {role} done in {elapsed:.1f}s, output={len(output)} chars")
                tier_result[f"{role}_output"] = output
                tier_result[f"{role}_elapsed_s"] = round(elapsed, 2)
            finally:
                stop_server(proc)
                # Brief pause so port 18081 is released before next launch.
                time.sleep(2)

        golden_out = tier_result.get("golden_output", "")
        test_out   = tier_result.get("test_output", "")
        identical, detail = compare_outputs(golden_out, test_out)

        tier_result["pass"] = identical
        tier_result["detail"] = detail
        results.append(tier_result)

        verdict = "PASS" if identical else "FAIL"
        print(f"[gate]   tier {tier:,}: {verdict} — {detail}")
        if not identical:
            overall_pass = False

    return {
        "stamp": stamp,
        "golden_binary": golden_binary,
        "test_binary": test_binary,
        "model": model,
        "n_gen": N_GEN,
        "seed": SEED,
        "temperature": 0,
        "mode": "pure_AR_no_draft",
        "tiers": results,
        "overall_pass": overall_pass,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AR-decode bit-identity gate for CUDA-graph decode refactors."
    )
    p.add_argument(
        "--golden-binary",
        default=os.path.join(
            os.path.dirname(CTXSWEEP_DIR), "..", "..", "server", "build", "dflash_server"
        ),
        help="Path to golden (known-good) binary (default: server/build/dflash_server).",
    )
    p.add_argument(
        "--test-binary",
        required=True,
        help="Path to refactored binary under test.",
    )
    p.add_argument(
        "--model",
        required=True,
        help="Path to target model GGUF.",
    )
    p.add_argument(
        "--model-name",
        default="luce-gate",
        help="Model name sent in API requests (default: luce-gate).",
    )
    p.add_argument(
        "--tiers",
        nargs="+",
        type=int,
        default=[4096, 32768, 71680],
        metavar="N",
        help="Context tiers in tokens (default: 4096 32768 71680).",
    )
    p.add_argument(
        "--n-gen",
        type=int,
        default=N_GEN,
        help=f"Tokens to generate per probe (default: {N_GEN}).",
    )
    p.add_argument(
        "--stamp",
        default=None,
        help="Timestamp string for the output JSON filename (e.g. 20260622_120000). "
             "Defaults to seconds since epoch.",
    )
    p.add_argument(
        "--output-dir",
        default=CTXSWEEP_DIR,
        help="Directory for result JSON (default: same dir as this script).",
    )
    p.add_argument(
        "--extra-server-arg",
        dest="extra_server_args",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra arg to pass to BOTH server binaries (repeatable). "
             "E.g. --extra-server-arg --cache-type-k --extra-server-arg f16",
    )
    p.add_argument(
        "--no-lock",
        action="store_true",
        help="Skip the GPU lock (for debugging without GPU).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Override module-level N_GEN if user passed --n-gen.
    global N_GEN
    N_GEN = args.n_gen

    stamp = args.stamp or str(int(time.time()))
    out_path = os.path.join(args.output_dir, f"bit_identity_{stamp}.json")
    tmpdir = os.path.join(args.output_dir, f".gate_tmp_{stamp}")
    os.makedirs(tmpdir, exist_ok=True)

    # Validate binaries.
    for label, path in [("golden", args.golden_binary), ("test", args.test_binary)]:
        if not os.path.isfile(path):
            print(f"[gate] ERROR: {label} binary not found: {path}", file=sys.stderr)
            return 2
        if not os.access(path, os.X_OK):
            print(f"[gate] ERROR: {label} binary not executable: {path}", file=sys.stderr)
            return 2

    if not os.path.isfile(args.model):
        print(f"[gate] ERROR: model not found: {args.model}", file=sys.stderr)
        return 2

    print(f"[gate] bit-identity gate stamp={stamp}")
    print(f"[gate] golden: {args.golden_binary}")
    print(f"[gate] test:   {args.test_binary}")
    print(f"[gate] model:  {args.model}")
    print(f"[gate] tiers:  {args.tiers}")
    print(f"[gate] n_gen:  {N_GEN}  seed={SEED}  temp=0  mode=pure_AR")
    print(f"[gate] port:   {GATE_PORT} (not :18099)")

    # Acquire GPU lock.
    lock_fd = None
    if not args.no_lock:
        print(f"[gate] acquiring GPU lock {GPU_LOCK_PATH} ...")
        lock_fd = open(GPU_LOCK_PATH, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        print("[gate] GPU lock acquired.")

    try:
        gate_result = run_gate(
            golden_binary      = args.golden_binary,
            test_binary        = args.test_binary,
            model              = args.model,
            model_name         = args.model_name,
            tiers              = args.tiers,
            stamp              = stamp,
            extra_server_args  = args.extra_server_args,
            tmpdir             = tmpdir,
        )
    finally:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    # Print summary table.
    print("\n" + "=" * 60)
    print(f"  BIT-IDENTITY GATE RESULTS  [{stamp}]")
    print("=" * 60)
    print(f"  {'TIER':>10}  {'RESULT':>8}  DETAIL")
    print("-" * 60)
    for t in gate_result["tiers"]:
        verdict = "PASS" if t["pass"] else "FAIL"
        detail = t["detail"]
        if len(detail) > 45:
            detail = detail[:42] + "..."
        print(f"  {t['tier_tokens']:>10,}  {verdict:>8}  {detail}")
    print("-" * 60)
    overall = "PASS" if gate_result["overall_pass"] else "FAIL"
    print(f"  {'OVERALL':>10}  {overall:>8}")
    print("=" * 60)

    # Save JSON.
    with open(out_path, "w") as fh:
        json.dump(gate_result, fh, indent=2)
    print(f"\n[gate] results saved: {out_path}")

    return 0 if gate_result["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
