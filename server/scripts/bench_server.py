#!/usr/bin/env python3
"""HTTP server benchmark — exercises the C++ dflash_server with the same
workloads as bench_llm.py (short LLM prompts) and bench_agent.py (long
agentic prompts), but over HTTP via /v1/chat/completions streaming.

This answers: "does the C++ server perform as well as the raw CLI binaries?"

Workloads:
  he      — 10 HumanEval code-completion prompts (same as bench_daemon.py)
  gsm8k   — 10 GSM8K math word problems
  math500 — 10 MATH-500 problems (2048 max_tokens)
  agent   — SWE-bench Verified at 2K / 8K / 24K token buckets

Usage:
    # Start C++ server first:
    ./dflash/build/dflash_server dflash/models/Qwen3-0.6B-BF16.gguf --port 9099

    # Run all workloads:
    python3 dflash/scripts/bench_server.py --url http://localhost:9099

    # Run specific workloads:
    python3 dflash/scripts/bench_server.py --url http://localhost:9099 --workload he gsm8k

    # Quick smoke test (1 prompt per workload):
    python3 dflash/scripts/bench_server.py --url http://localhost:9099 --n-sample 1
"""
import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# Allow importing bench_he for its PROMPTS list.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_llm import _extract_boxed, _normalize_math, _math_equiv

N_SAMPLE = 10
N_GEN_DEFAULT = 256
N_GEN_MATH = 2048


# ── HTTP streaming client ─────────────────────────────────────────────────

def stream_chat(url: str, messages: list[dict], max_tokens: int,
                temperature: float = 0.0, timeout: float = 600.0,
                thinking: bool = False) -> dict:
    """POST /v1/chat/completions with stream=True.

    Returns dict with:
      n_tok, wall_s, ttft_s, decode_s, decode_tps, wall_tps,
      text, usage (if server returns it in final chunk).
    """
    body = {
        "model": "dflash",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if thinking:
        body["thinking"] = {"type": "enabled", "budget_tokens": 4096}

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream"},
    )
    t0 = time.perf_counter()
    t_first = 0.0
    t_last = 0.0
    n_tok = 0
    text_parts = []
    usage = None

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
            # Extract usage from final chunk if present.
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or ""
            if content or reasoning:
                if n_tok == 0:
                    t_first = time.perf_counter()
                n_tok += 1
                t_last = time.perf_counter()
                if content:
                    text_parts.append(content)

    wall_s = time.perf_counter() - t0
    ttft_s = (t_first - t0) if n_tok > 0 else wall_s
    decode_s = (t_last - t_first) if n_tok > 1 else 0.0
    decode_tps = (n_tok - 1) / decode_s if decode_s > 0 else 0.0
    wall_tps = n_tok / wall_s if wall_s > 0 else 0.0

    return {
        "n_tok": n_tok,
        "wall_s": wall_s,
        "ttft_s": ttft_s,
        "decode_s": decode_s,
        "decode_tps": decode_tps,
        "wall_tps": wall_tps,
        "text": "".join(text_parts),
        "usage": usage,
    }


# ── Workload: HumanEval ──────────────────────────────────────────────────

def workload_he(url: str, n_sample: int, n_gen: int, **_kw):
    """HumanEval code-completion prompts (same 10 as bench_he.py)."""
    from bench_he import PROMPTS
    prompts = PROMPTS[:n_sample]
    results = []
    for name, text in prompts:
        msgs = [{"role": "user", "content": text}]
        try:
            r = stream_chat(url, msgs, n_gen)
            results.append({"name": name, **r})
            _print_row(name, r)
        except Exception as e:
            print(f"  {name:28s}  FAILED: {e}", flush=True)
    return results


# ── Workload: GSM8K ──────────────────────────────────────────────────────

def _load_gsm8k(n_sample: int):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    n_sample = min(n_sample, len(ds))
    ds = ds.shuffle(seed=42).select(range(n_sample))
    return [
        {"name": f"gsm8k_{i:02d}",
         "prompt": f"Question: {row['question']}\nAnswer: "}
        for i, row in enumerate(ds)
    ]


def workload_gsm8k(url: str, n_sample: int, n_gen: int, **_kw):
    rows = _load_gsm8k(n_sample)
    results = []
    for row in rows:
        msgs = [{"role": "user", "content": row["prompt"]}]
        try:
            r = stream_chat(url, msgs, n_gen)
            results.append({"name": row["name"], **r})
            _print_row(row["name"], r)
        except Exception as e:
            print(f"  {row['name']:28s}  FAILED: {e}", flush=True)
    return results


# ── Workload: Math500 ────────────────────────────────────────────────────

def _load_math500(n_sample: int):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    n_sample = min(n_sample, len(ds))
    ds = ds.shuffle(seed=42).select(range(n_sample))
    return [
        {"name": f"math_{i:02d}",
         "prompt": f"Problem: {row['problem']}\nSolution: Put your final answer in \\boxed{{}}.\n",
         "answer": row["answer"]}
        for i, row in enumerate(ds)
    ]


def _score_math_text(text: str, gold_answer: str) -> tuple[bool, str]:
    """Score a math response text against the gold answer.

    Extracts \\boxed{} answers (after </think> for thinking models),
    with fallbacks for **bold** and $...$ patterns.
    Returns (correct, detail_str).
    """
    think_end = text.rfind("</think>")
    answer_text = text[think_end + len("</think>"):] if think_end >= 0 else text

    pred = _extract_boxed(answer_text)
    if not pred:
        pred = _extract_boxed(text)

    # Fallback: "the answer is **X**" patterns
    if pred is None:
        bold_pattern = re.compile(
            r'(?:answer\s+is|there\s+are|result\s+is|equals?|=)\s*\*\*(.+?)\*\*',
            re.IGNORECASE)
        m = bold_pattern.search(answer_text)
        if m:
            pred = m.group(1).strip().rstrip(".")

    # Fallback: last $...$ expression
    if pred is None:
        matches = re.findall(r'\$([^$]+)\$', answer_text)
        if matches:
            pred = matches[-1].strip()

    correct = _math_equiv(pred, gold_answer)
    pred_short = (pred[:60] + "…") if pred and len(pred) > 60 else pred
    gold_short = (gold_answer[:60] + "…") if len(gold_answer) > 60 else gold_answer
    if correct:
        detail = f"🎯 {pred_short}"
    elif pred:
        detail = f"✗ pred={pred_short} gold={gold_short}"
    else:
        detail = f"✗ no answer found, gold={gold_short}"
    return correct, detail


def workload_math500(url: str, n_sample: int, n_gen: int, thinking: bool = False, **_kw):
    rows = _load_math500(n_sample)
    gen = max(n_gen, N_GEN_MATH)  # Math needs longer generation
    results = []
    n_correct, n_scored = 0, 0
    for row in rows:
        msgs = [{"role": "user", "content": row["prompt"]}]
        try:
            r = stream_chat(url, msgs, gen, thinking=thinking)
            correct, detail = _score_math_text(r["text"], row["answer"])
            r["correct"] = correct
            r["score_detail"] = detail
            n_scored += 1
            if correct:
                n_correct += 1
            results.append({"name": row["name"], **r})
            _print_row(row["name"], r, score=detail)
        except Exception as e:
            print(f"  {row['name']:28s}  FAILED: {e}", flush=True)
    if n_scored:
        pct = n_correct / n_scored * 100
        print(f"\n  accuracy: {n_correct}/{n_scored} ({pct:.0f}%)")
    return results


# ── Workload: Agent (SWE-bench) ──────────────────────────────────────────

FIX_DIR = Path(__file__).resolve().parent / "fixtures"
SWE_PARQUET = FIX_DIR / "swe_bench" / "swe_bench_verified.parquet"
SYS_PROMPT_SMALL = FIX_DIR / "agent_prompts" / "codex_gpt52_codex.md"
SYS_PROMPT_LARGE = FIX_DIR / "agent_prompts" / "codex_gpt52.md"

AGENT_BUCKETS = {
    "2k":  {"target_chars": 6000,   "sys": SYS_PROMPT_SMALL},
    "8k":  {"target_chars": 24000,  "sys": SYS_PROMPT_LARGE},
    "24k": {"target_chars": 72000,  "sys": SYS_PROMPT_LARGE},
}


def _build_agent_user_msg(row: dict, target_chars: int) -> str:
    """Build a Codex-style user message padded to ~target_chars."""
    repo = row.get("repo", "unknown/repo")
    iid = row.get("instance_id", "unknown")
    problem = row.get("problem_statement", "") or ""
    patch = row.get("patch", "") or ""
    test_patch = row.get("test_patch", "") or ""
    hints = row.get("hints_text", "") or ""

    pool = "\n\n".join(p for p in (patch, test_patch, hints) if p)
    if not pool:
        pool = problem

    chunks = []
    chunk_size = max(2000, target_chars // 6)
    cur = 0
    idx = 1
    while cur < target_chars:
        offset = cur % max(1, len(pool))
        seg = pool[offset:offset + chunk_size]
        if not seg:
            seg = pool[:chunk_size]
        chunks.append(
            f'<tool_result tool="read_file" path="{repo}/_ctx_{idx}.py">\n{seg}\n</tool_result>'
        )
        cur += len(seg)
        idx += 1

    file_blocks = "\n\n".join(chunks)
    return (
        f"Repository: {repo}\nInstance: {iid}\n\n"
        f"## Issue\n{problem}\n\n"
        f"## Context I gathered\n"
        f"I ran `read_file` on the relevant modules:\n\n"
        f"{file_blocks}\n\n"
        f"## Task\nInvestigate the bug and reply with a single tool call "
        f"to `apply_patch` that fixes it. Keep the patch minimal."
    )


def workload_agent(url: str, n_sample: int, n_gen: int, bucket: str = "all", **_kw):
    if not SWE_PARQUET.is_file():
        print(f"  SKIP: SWE-bench parquet not found at {SWE_PARQUET}", flush=True)
        return []

    import pyarrow.parquet as pq
    df = pq.read_table(str(SWE_PARQUET)).to_pandas()

    bucket_keys = list(AGENT_BUCKETS) if bucket == "all" else [bucket]
    all_results = []

    for bk in bucket_keys:
        cfg = AGENT_BUCKETS[bk]
        sys_path = cfg["sys"]
        if not sys_path.is_file():
            print(f"  SKIP bucket {bk}: system prompt not found at {sys_path}", flush=True)
            continue

        sys_text = sys_path.read_text(encoding="utf-8")
        rows = df.sample(n=min(n_sample, len(df)), random_state=42).to_dict("records")

        print(f"\n  --- bucket {bk} (target ~{cfg['target_chars']} chars, n={len(rows)}) ---")
        for i, row in enumerate(rows):
            name = f"agent_{bk}_{i:02d}"
            user_msg = _build_agent_user_msg(row, cfg["target_chars"])
            msgs = [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": user_msg},
            ]
            try:
                r = stream_chat(url, msgs, n_gen)
                r["bucket"] = bk
                r["instance_id"] = row.get("instance_id", "")
                all_results.append({"name": name, **r})
                _print_row(name, r)
            except Exception as e:
                print(f"  {name:28s}  FAILED: {e}", flush=True)

    return all_results


# ── Output formatting ─────────────────────────────────────────────────────

def _print_header():
    print(f"  {'prompt':28s}  {'n_tok':>5s} {'wall_s':>7s} {'ttft_s':>7s} "
          f"{'dec_s':>7s} {'dec_tps':>8s} {'wall_tps':>9s}")
    print("  " + "-" * 80)


def _print_row(name: str, r: dict, score: str = ""):
    n = r["n_tok"]
    suffix = f"  {score}" if score else ""
    if n == 0:
        print(f"  {name:28s}  {n:5d} {r['wall_s']:7.2f}   --       --       --         --{suffix}",
              flush=True)
        return
    print(f"  {name:28s}  {n:5d} {r['wall_s']:7.2f} {r['ttft_s']:7.3f} "
          f"{r['decode_s']:7.2f} {r['decode_tps']:8.2f} {r['wall_tps']:9.2f}{suffix}",
          flush=True)


def _print_summary(label: str, results: list[dict]):
    if not results:
        return
    valid = [r for r in results if r["n_tok"] > 0]
    if not valid:
        print(f"\n  [{label}] no successful runs")
        return

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    n = len(valid)
    wall_tps = _mean([r["wall_tps"] for r in valid])
    dec_tps_list = [r["decode_tps"] for r in valid if r["decode_tps"] > 0]
    dec_tps = _mean(dec_tps_list) if dec_tps_list else 0.0
    ttft = _mean([r["ttft_s"] for r in valid])
    wall = _mean([r["wall_s"] for r in valid])
    tok = _mean([r["n_tok"] for r in valid])

    print(f"\n  [{label}] {n} prompts — mean: "
          f"n_tok={tok:.0f}  TTFT={ttft:.3f}s  "
          f"decode={dec_tps:.2f} tok/s  "
          f"wall={wall_tps:.2f} tok/s  "
          f"total={wall:.2f}s")
    if dec_tps_list:
        print(f"  [{label}] decode tok/s range: "
              f"{min(dec_tps_list):.2f} - {max(dec_tps_list):.2f}")


# ── Main ──────────────────────────────────────────────────────────────────

WORKLOADS = {
    "he":      ("HumanEval (code completion)", workload_he),
    "gsm8k":   ("GSM8K (math word problems)", workload_gsm8k),
    "math500": ("MATH-500 (hard math)", workload_math500),
    "agent":   ("SWE-bench agent (2K/8K/24K)", workload_agent),
}


def main():
    ap = argparse.ArgumentParser(
        description="HTTP server benchmark — exercises dflash_server with "
                    "bench_llm + bench_agent workloads over /v1/chat/completions")
    ap.add_argument("--url", default="http://localhost:9099",
                    help="Server base URL (default: http://localhost:9099)")
    ap.add_argument("--workload", nargs="+", choices=list(WORKLOADS) + ["all"],
                    default=["all"],
                    help="Which workloads to run (default: all)")
    ap.add_argument("--n-sample", type=int, default=N_SAMPLE,
                    help=f"Prompts per workload (default: {N_SAMPLE})")
    ap.add_argument("--n-gen", type=int, default=N_GEN_DEFAULT,
                    help=f"Max output tokens (default: {N_GEN_DEFAULT})")
    ap.add_argument("--agent-bucket", choices=["2k", "8k", "24k", "all"],
                    default="all", help="Agent bucket filter (default: all)")
    ap.add_argument("--warmup", action="store_true",
                    help="Run one warmup request before timing")
    ap.add_argument("--thinking", action="store_true",
                    help="Enable thinking/reasoning mode")
    ap.add_argument("--out", type=str, default=None,
                    help="Write JSON results to this file")
    args = ap.parse_args()

    # Validate server is reachable.
    try:
        urllib.request.urlopen(args.url.rstrip("/") + "/health", timeout=5)
    except Exception as e:
        print(f"ERROR: server not reachable at {args.url}: {e}", file=sys.stderr)
        sys.exit(1)

    if args.warmup:
        print("[bench-server] warmup...", flush=True)
        stream_chat(args.url, [{"role": "user", "content": "Hi"}], 16)

    wl_keys = list(WORKLOADS) if "all" in args.workload else args.workload

    print("=" * 88)
    print(f"  HTTP Server Benchmark — {args.url}")
    print(f"  workloads: {', '.join(wl_keys)}  n_sample={args.n_sample}  n_gen={args.n_gen}")
    print("=" * 88)

    all_results = {}
    for wk in wl_keys:
        label, fn = WORKLOADS[wk]
        print(f"\n{'─' * 88}")
        print(f"  {label}")
        print(f"{'─' * 88}")
        _print_header()
        try:
            results = fn(url=args.url, n_sample=args.n_sample, n_gen=args.n_gen,
                         bucket=args.agent_bucket, thinking=args.thinking)
        except ImportError as e:
            print(f"  SKIP {wk}: missing dependency — {e}", flush=True)
            results = []
        except FileNotFoundError as e:
            print(f"  SKIP {wk}: {e}", flush=True)
            results = []

        all_results[wk] = results
        _print_summary(wk, results)

    # Final summary
    print(f"\n{'=' * 88}")
    print("  SUMMARY")
    print(f"{'=' * 88}")
    print(f"  {'Workload':12s}  {'N':>3s}  {'TTFT':>7s}  {'dec_tps':>8s}  "
          f"{'wall_tps':>9s}  {'wall_s':>7s}")
    print("  " + "-" * 55)
    for wk in wl_keys:
        results = all_results.get(wk, [])
        valid = [r for r in results if r.get("n_tok", 0) > 0]
        if not valid:
            print(f"  {wk:12s}  {'--':>3s}  {'--':>7s}  {'--':>8s}  {'--':>9s}  {'--':>7s}")
            continue
        n = len(valid)

        def _m(key):
            vals = [r[key] for r in valid if r.get(key, 0) > 0]
            return sum(vals) / len(vals) if vals else 0.0

        print(f"  {wk:12s}  {n:3d}  {_m('ttft_s'):7.3f}  {_m('decode_tps'):8.2f}  "
              f"{_m('wall_tps'):9.2f}  {_m('wall_s'):7.2f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Wrote {out_path}")

    print()


if __name__ == "__main__":
    main()
