#!/usr/bin/env python3
"""
HumanEval+ A/B for the dflash server. For each config, spawn a fresh server,
generate one completion per HumanEval+ task, then grade pass@1 by exec'ing the
EvalPlus tests in a subprocess sandbox.

This is the *capability* counterpart to quality_ab_simple.py: instead of
"are the bytes the same?", it asks "does the answer pass the unit tests?".
Because pass@1 is graded against hidden tests, two different completions can
both score 1.0 — so cache-induced output drift may or may not hurt this number.

Run:
    PFLASH_TARGET=...  PFLASH_DRAFT=...  PFLASH_BIN=...  PFLASH_DRAFTER=... \\
    python3 dflash/scripts/quality_humaneval_plus.py [--limit N]

Outputs:
    /tmp/hep_results/samples_<config>.jsonl  — one completion per task
    /tmp/hep_results/scores_<config>.json    — per-task pass/fail
    stdout markdown table comparing pass@1 across configs
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET = PROJECT_ROOT / "dflash/eval/humaneval_plus/humanevalplus.jsonl"

CONFIGS = {
    "baseline": {
        "flags": ["--prefix-cache-slots", "0", "--prefill-compression", "off"],
        "env":   {"DFLASH_FP_USE_BSA": "0"},
    },
    "prefix_on": {
        "flags": ["--prefix-cache-slots", "4", "--prefill-compression", "off"],
        "env":   {"DFLASH_FP_USE_BSA": "0"},
    },
    "compression_on": {
        "flags": ["--prefix-cache-slots", "0", "--prefill-compression", "always"],
        "env":   {"DFLASH_FP_USE_BSA": "0"},
    },
    "compression_auto": {
        "flags": ["--prefix-cache-slots", "0", "--prefill-compression", "auto"],
        "env":   {"DFLASH_FP_USE_BSA": "0"},
    },
    "all_on": {
        "flags": ["--prefix-cache-slots", "4", "--prefill-compression", "always"],
        "env":   {"DFLASH_FP_USE_BSA": "1"},
    },
}

PORT     = 8765
TARGET   = os.environ.get("PFLASH_TARGET", "/home/peppi/models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf")
DRAFT    = os.environ.get("PFLASH_DRAFT",  "/home/peppi/models/qwen3.6-27b-dflash/model.safetensors")
BIN      = os.environ.get("PFLASH_BIN",    "dflash/build/test_dflash")
DRAFTER  = os.environ.get("PFLASH_DRAFTER", str(Path.home() / "models/Qwen3-0.6B-BF16.gguf"))
SCRIPT   = PROJECT_ROOT / "dflash/scripts/server.py"

# Canonical EvalPlus chat-mode prompt (evalplus/codegen.py:222-223)
INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:"
)

EXEC_TIMEOUT = 10        # seconds per task grading
GEN_TIMEOUT  = 120       # seconds per chat request (was 180 — caused timeout artifacts)
MAX_TOKENS   = 512


def _do_chat_post(payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=GEN_TIMEOUT) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def chat_post(payload, timeout=GEN_TIMEOUT):
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(_do_chat_post, payload)
    try:
        return fut.result(timeout=timeout)
    except FutureTimeout:
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"chat_post wall-clock timeout after {timeout}s")
    finally:
        pool.shutdown(wait=False)


def wait_ready(proc, timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def spawn(name, cfg, log_path):
    flags = list(cfg["flags"])
    if "--prefill-compression" in flags:
        idx = flags.index("--prefill-compression")
        if idx + 1 < len(flags) and flags[idx + 1] != "off" and "--prefill-drafter" not in flags:
            flags += ["--prefill-drafter", DRAFTER]
    cmd = [
        sys.executable, "-u", str(SCRIPT),
        "--target", TARGET, "--draft", DRAFT, "--bin", BIN,
        "--port", str(PORT), *flags,
    ]
    env = {**os.environ, **cfg.get("env", {})}
    print(f"[{name}] spawn", flush=True)
    log = open(log_path, "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)


def load_dataset():
    with open(DATASET) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_completion(reply: str, entry_point: str, prompt: str) -> str:
    """Extract a complete Python solution from the model's reply.

    EvalPlus-canonical prompt asks for a self-contained script in a markdown
    code block. We:
      1. Find the *last* ```python ... ``` fenced block (later blocks tend to
         be the final answer; earlier ones are often planning sketches).
      2. If the block redefines the target function, return everything from
         that ``def`` onward — the prompt's signature gets replaced.
      3. Otherwise, return the body lines indented to match the prompt's
         signature, so prompt+body parses.
      4. On failure, return ``    pass  # empty`` so grading just fails the
         task instead of crashing.
    """
    fences = re.findall(r"```(?:python)?\s*\n(.*?)```", reply, re.DOTALL)
    code = fences[-1] if fences else reply

    # If the code defines our entry function, grade that whole script
    # by *replacing* the prompt with the model's full implementation.
    func_re = re.compile(rf"^(def\s+{re.escape(entry_point)}\s*\(.*?\)[^:]*:)",
                         re.MULTILINE | re.DOTALL)
    m = func_re.search(code)
    if m:
        return "##__FULL_SCRIPT__##\n" + code.strip()

    # Fallback: pull indented body lines after the first def we see.
    sig = re.search(r"^def\s+\w+\s*\(.*?\)[^:]*:\s*\n", code, re.MULTILINE | re.DOTALL)
    if sig:
        code = code[sig.end():]
    lines = code.split("\n")
    while lines and not lines[0].startswith((" ", "\t")):
        lines.pop(0)
    body = "\n".join(lines).rstrip()
    return body if body else "    pass  # empty completion"


def generate(name, cfg, tasks, samples_path):
    log = f"/tmp/hep_results/server_{name}.log"
    proc = spawn(name, cfg, log)
    try:
        if not wait_ready(proc):
            print(f"[{name}] server failed to start — see {log}", flush=True)
            return False
        print(f"[{name}] ready, generating {len(tasks)} completions", flush=True)
        with open(samples_path, "w") as out:
            for i, t in enumerate(tasks, 1):
                t0 = time.time()
                try:
                    user_msg = (
                        f"{INSTRUCTION_PREFIX}\n"
                        f"```\n{t['prompt'].strip()}\n```\n"
                        f"{RESPONSE_PREFIX}"
                    )
                    reply = chat_post({
                        "model": "luce-dflash",
                        "messages": [{"role": "user", "content": user_msg}],
                        "max_tokens": MAX_TOKENS,
                        "stream": False,
                        "chat_template_kwargs": {"enable_thinking": False},
                    })
                    err = None
                except Exception as e:
                    reply = ""
                    err = str(e)
                completion = extract_completion(reply, t["entry_point"], t["prompt"])
                dur = time.time() - t0
                row = {"task_id": t["task_id"], "completion": completion,
                       "raw_reply": reply, "err": err, "secs": round(dur, 2)}
                out.write(json.dumps(row) + "\n")
                out.flush()
                if i <= 3 or i % 5 == 0 or i == len(tasks):
                    print(f"  [{name}] {i}/{len(tasks)} {t['task_id']} {dur:.1f}s",
                          flush=True)
        return True
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()


# ----- grading via subprocess sandbox (one process per task) -----

GRADE_RUNNER = r'''
import json, sys, signal, traceback
def _h(*_): raise TimeoutError()
signal.signal(signal.SIGALRM, _h); signal.alarm(int(sys.argv[1]))
src = sys.stdin.read()
try:
    g = {}
    exec(src, g)
    print(json.dumps({"ok": True}))
except TimeoutError:
    print(json.dumps({"ok": False, "err": "timeout"}))
except Exception as e:
    print(json.dumps({"ok": False, "err": f"{type(e).__name__}: {e}"}))
'''


def grade_one(task, completion):
    """Run completion + task['test'] + check_call in a subprocess with a
    SIGALRM-based timeout. Returns dict with ok/err.

    If completion starts with the ##__FULL_SCRIPT__## sentinel we use the model's
    full script (it redefined the entry function); otherwise we treat completion
    as a function body and prepend the prompt's signature."""
    if completion.startswith("##__FULL_SCRIPT__##\n"):
        body = completion[len("##__FULL_SCRIPT__##\n"):]
        src = body + "\n" + task["test"] + f"\ncheck({task['entry_point']})\n"
    else:
        src = task["prompt"] + completion + "\n" + task["test"] + \
              f"\ncheck({task['entry_point']})\n"
    try:
        p = subprocess.run(
            [sys.executable, "-c", GRADE_RUNNER, str(EXEC_TIMEOUT)],
            input=src.encode(), capture_output=True, timeout=EXEC_TIMEOUT + 5)
    except subprocess.TimeoutExpired:
        return {"ok": False, "err": "subprocess-timeout"}
    out = (p.stdout or b"").decode().strip().splitlines()
    if not out:
        return {"ok": False, "err": "no-output", "stderr": (p.stderr or b"")[-200:].decode(errors="replace")}
    try:
        return json.loads(out[-1])
    except Exception:
        return {"ok": False, "err": "bad-json", "raw": out[-1]}


def grade_config(name, tasks, samples_path, scores_path):
    samples = {}
    sample_errs = {}
    with open(samples_path) as f:
        for line in f:
            s = json.loads(line)
            samples[s["task_id"]] = s
            if s.get("err"):
                sample_errs[s["task_id"]] = s["err"]
    rows = []
    passed = 0
    for i, t in enumerate(tasks, 1):
        s = samples.get(t["task_id"])
        if s is None or not s["completion"]:
            rows.append({"task_id": t["task_id"], "ok": False, "err": "no-sample"})
            continue
        r = grade_one(t, s["completion"])
        r["task_id"] = t["task_id"]
        rows.append(r)
        if r["ok"]:
            passed += 1
        if i <= 3 or i % 25 == 0 or i == len(tasks):
            print(f"  [{name}] graded {i}/{len(tasks)}  pass={passed}", flush=True)
    with open(scores_path, "w") as f:
        json.dump({"passed": passed, "total": len(rows), "rows": rows}, f, indent=2)
    return passed, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run the first N tasks (for smoke testing)")
    ap.add_argument("--configs", default=",".join(CONFIGS.keys()))
    ap.add_argument("--skip-generate", action="store_true",
                    help="Skip generation; only re-grade existing samples")
    args = ap.parse_args()

    out_dir = Path("/tmp/hep_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_dataset()
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"Loaded {len(tasks)} HumanEval+ tasks")

    chosen = [c.strip() for c in args.configs.split(",") if c.strip()]
    for name in chosen:
        if name not in CONFIGS:
            print(f"unknown config: {name}", file=sys.stderr)
            return 1

    results = {}
    for name in chosen:
        samples_path = out_dir / f"samples_{name}.jsonl"
        scores_path  = out_dir / f"scores_{name}.json"
        if not args.skip_generate:
            ok = generate(name, CONFIGS[name], tasks, samples_path)
            if not ok:
                print(f"ABORT: {name} did not generate")
                return 1
        passed, total = grade_config(name, tasks, samples_path, scores_path)
        results[name] = (passed, total)

    print()
    print("=== HumanEval+ pass@1 (greedy, raw) ===")
    print(f"{'config':18} {'pass':>6} {'total':>6} {'pass@1':>8}")
    base_passed_raw = results.get("baseline", (None, None))[0]
    for name, (passed, total) in results.items():
        rate = passed / total if total else 0.0
        delta = ""
        if base_passed_raw is not None and name != "baseline":
            d = (passed - base_passed_raw) / total
            delta = f"  Δ{d:+.3f} vs baseline"
        print(f"{name:18} {passed:>6} {total:>6} {rate:>8.3f}{delta}")

    # Drop-on-error: skip tasks where ANY config errored at the request layer.
    # Eliminates the "baseline timed out so it auto-fails" artefact.
    err_sets = {}
    for name in chosen:
        samples_path = out_dir / f"samples_{name}.jsonl"
        with open(samples_path) as f:
            err_sets[name] = {json.loads(l)["task_id"]
                              for l in f if json.loads(l).get("err")}
    union_errs = set().union(*err_sets.values())
    if union_errs:
        print(f"\n=== HumanEval+ pass@1 (drop {len(union_errs)} tasks where any config errored) ===")
        print(f"{'config':18} {'pass':>6} {'total':>6} {'pass@1':>8}")
        clean_results = {}
        for name in chosen:
            scores_path = out_dir / f"scores_{name}.json"
            data = json.loads(scores_path.read_text())
            kept = [r for r in data["rows"] if r["task_id"] not in union_errs]
            passed = sum(1 for r in kept if r["ok"])
            clean_results[name] = (passed, len(kept))
        base_passed_clean = clean_results.get("baseline", (None, None))[0]
        for name, (passed, total) in clean_results.items():
            rate = passed / total if total else 0.0
            delta = ""
            if base_passed_clean is not None and name != "baseline":
                d = (passed - base_passed_clean) / total if total else 0.0
                delta = f"  Δ{d:+.3f} vs baseline"
            print(f"{name:18} {passed:>6} {total:>6} {rate:>8.3f}{delta}")


if __name__ == "__main__":
    sys.exit(main() or 0)
