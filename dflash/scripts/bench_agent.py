"""
Agentic-workload benchmark — simulates Codex / Claude Code style requests.

Where ``bench_llm.py`` exercises 40-250 token prompts (and skips anything
``> 3500`` tokens) and reports decode tok/s only, real Codex / Claude Code
clients send 5K-30K input tokens (system prompt + tool definitions +
conversation history + tool-call file dumps) and the user-visible cost is
prefill-dominated. This bench:

  - Builds prompts from a real Codex system prompt fixture + a SWE-bench
    Verified row (problem_statement + patch/test_patch as synthesised
    "tool result" file context) padded to three length buckets:
    ~2K, ~8K, ~24K tokens.
  - Reports AR vs DFlash for each bucket: prefill_s, decode tok/s, TTFT,
    total latency, AL — and BOTH the decode-only speedup (today's
    RESULTS.md definition) and the user-visible total-latency speedup.
  - Parses the per-stage [timing] block from ``test_dflash`` for
    root-cause attribution.

Usage:
    python3 scripts/bench_agent.py                      # all buckets, n=5 each
    python3 scripts/bench_agent.py --bucket 8k          # one bucket
    python3 scripts/bench_agent.py --n-sample 1 --bucket 2k  # smoke

Same env vars as ``bench_llm.py``: ``DFLASH_TARGET``, ``DFLASH_DRAFT``,
``DFLASH_BIN``, ``DFLASH_BIN_AR``, ``DFLASH_TOKENIZER``.
"""
import argparse
import json
import os
import re
import struct
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN_SUFFIX = ".exe" if os.name == "nt" else ""
TARGET = os.environ.get(
    "DFLASH_TARGET",
    str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),
)
_LOCAL_DRAFT_FILE = ROOT / "models" / "draft" / "dflash-draft-3.6-q8_0.gguf"
_LOCAL_DRAFT_ROOT = ROOT / "models" / "draft"
DRAFT = None
TEST_DFLASH = os.environ.get("DFLASH_BIN", str(ROOT / "build" / f"test_dflash{BIN_SUFFIX}"))
TEST_GENERATE = os.environ.get("DFLASH_BIN_AR", str(ROOT / "build" / f"test_generate{BIN_SUFFIX}"))
TOKENIZER = os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.5-27B")
TMPDIR = Path(tempfile.gettempdir()) / "dflash_bench"
TMPDIR.mkdir(parents=True, exist_ok=True)

FIX_DIR = ROOT / "scripts" / "fixtures"
SWE_PARQUET = FIX_DIR / "swe_bench" / "swe_bench_verified.parquet"
SYS_PROMPT_SMALL = FIX_DIR / "agent_prompts" / "codex_gpt52_codex.md"   # ~1694 tok
SYS_PROMPT_LARGE = FIX_DIR / "agent_prompts" / "codex_gpt52.md"         # ~4756 tok

N_GEN = 256
BUDGET = 22

# Prompt buckets — target token counts (hit within ±20%).
BUCKETS = {
    "2k":  {"target": 2048,  "sys": SYS_PROMPT_SMALL},
    "8k":  {"target": 8192,  "sys": SYS_PROMPT_LARGE},
    "24k": {"target": 24576, "sys": SYS_PROMPT_LARGE},
}


# ── shared with bench_llm.py (intentionally duplicated to keep it standalone) ──
def _find_draft_model(root: Path):
    if root.is_file():
        return str(root)
    if not root.is_dir():
        return None
    for pattern in ("dflash-draft-*.gguf", "*.gguf", "model.safetensors"):
        matches = sorted(root.rglob(pattern))
        if matches:
            return str(matches[0])
    return None


def _resolve_draft() -> str:
    env = os.environ.get("DFLASH_DRAFT")
    if env:
        found = _find_draft_model(Path(env))
        if found:
            return found
        raise FileNotFoundError(f"DFLASH_DRAFT does not point to a draft GGUF/safetensors: {env}")
    for c in (_LOCAL_DRAFT_FILE, _LOCAL_DRAFT_ROOT):
        found = _find_draft_model(c)
        if found:
            return found
    raise FileNotFoundError(
        f"DFlash draft not found. Set DFLASH_DRAFT or place a file under {_LOCAL_DRAFT_ROOT}"
    )


def _require_file(path: str, label: str):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _run_timed(cmd, timeout: int, label: str):
    """Run a subprocess, return (CompletedProcess, wall_seconds)."""
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    wall_s = time.perf_counter() - t0
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "<no output>").strip()[-2000:]
        raise RuntimeError(f"{label} exited {r.returncode}: {tail}")
    return r, wall_s


def tokenize_to_file(tok, text: str, path: Path) -> int:
    ids = tok.encode(text, add_special_tokens=False)
    with open(path, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", int(t)))
    return len(ids)


def _auto_max_ctx(n_prompt: int, n_gen: int) -> int:
    pad = 64
    return ((n_prompt + n_gen + pad + 255) // 256) * 256


# ── parsing ────────────────────────────────────────────────────────────────
_RE_DECODE_TPS = re.compile(r"(\d+(?:\.\d+)?)\s+tok/s")
_RE_DF_PREFILL = re.compile(r"\[prefill\]\s+(\d+)\s+tokens\s+in\s+(\d+(?:\.\d+)?)\s*s")
_RE_AL = re.compile(r"avg commit/step=(\d+(?:\.\d+)?)")
_RE_TIMING_LINE = re.compile(r"^\s+([a-z_]+)\s+(\d+(?:\.\d+)?)\s*$")


def _parse_dflash(stdout: str) -> dict:
    """Parse test_dflash stdout into {prefill_s, decode_tps, al, stages{}}."""
    m_pf = _RE_DF_PREFILL.search(stdout)
    m_al = _RE_AL.search(stdout)
    # decode tps line is the LAST tok/s in the file ("[dflash] generated ... -> X tok/s")
    matches = list(_RE_DECODE_TPS.finditer(stdout))
    if not (m_pf and m_al and matches):
        raise RuntimeError(f"test_dflash parse failed:\n{stdout[-1500:]}")
    decode_tps = float(matches[-1].group(1))

    # Per-stage timing block (lines like "  draft_compute  14.70")
    stages = {}
    in_block = False
    for line in stdout.splitlines():
        if line.startswith("[timing]"):
            in_block = True
            continue
        if in_block:
            if line.startswith("[") or line.startswith("---"):
                # "  ----- sum     132.20" ends block; "[dflash] generated…" too
                if "----- sum" in line:
                    m = re.search(r"sum\s+(\d+(?:\.\d+)?)", line)
                    if m:
                        stages["sum"] = float(m.group(1))
                    in_block = False
                    continue
                if line.startswith("["):
                    in_block = False
                    continue
            m = _RE_TIMING_LINE.match(line)
            if m:
                stages[m.group(1)] = float(m.group(2))

    return {
        "prefill_s": float(m_pf.group(2)),
        "n_prompt_seen": int(m_pf.group(1)),
        "decode_tps": decode_tps,
        "al": float(m_al.group(1)),
        "stages": stages,
    }


def _parse_ar(stdout: str) -> dict:
    """Parse test_generate stdout → {decode_tps, n_gen}."""
    m = re.search(r"\[gen\]\s+(\d+)\s+new tokens in\s+(\d+(?:\.\d+)?)\s*s\s*->\s*(\d+(?:\.\d+)?)\s+tok/s",
                  stdout)
    if not m:
        raise RuntimeError(f"test_generate parse failed:\n{stdout[-1500:]}")
    return {
        "decode_tps": float(m.group(3)),
        "decode_s": float(m.group(2)),
        "n_gen": int(m.group(1)),
    }


# ── runners ────────────────────────────────────────────────────────────────
def run_ar(path: Path, n_gen: int):
    out_bin = TMPDIR / "ar_out.bin"
    r, wall_s = _run_timed(
        [TEST_GENERATE, TARGET, str(path), str(n_gen), str(out_bin)],
        timeout=600, label="test_generate",
    )
    p = _parse_ar(r.stdout)
    p["wall_s"] = wall_s
    return p


def run_df(path: Path, n_prompt: int, n_gen: int, budget: int = None):
    if budget is None:
        budget = BUDGET
    max_ctx = _auto_max_ctx(n_prompt, n_gen)
    out_bin = TMPDIR / "df_out.bin"
    r, wall_s = _run_timed(
        [
            TEST_DFLASH, TARGET, DRAFT, str(path), str(n_gen), str(out_bin),
            "--fast-rollback", "--ddtree",
            f"--ddtree-budget={budget}", f"--max-ctx={max_ctx}",
        ],
        timeout=900, label="test_dflash",
    )
    p = _parse_dflash(r.stdout)
    p["wall_s"] = wall_s
    p["max_ctx"] = max_ctx
    return p


# ── prompt construction ────────────────────────────────────────────────────
def _load_swe_rows():
    import pyarrow.parquet as pq
    t = pq.read_table(str(SWE_PARQUET))
    return t.to_pandas()


def _agent_user_message(row: dict, file_blocks_chars: int) -> str:
    """Synthesise a Codex/Claude-Code style user turn.

    Structure mimics what an agent client actually sends after a few tool
    calls have run (read_file results pasted into history).
    """
    repo = row["repo"]
    iid = row["instance_id"]
    problem = row["problem_statement"] or ""
    patch = row["patch"] or ""
    test_patch = row["test_patch"] or ""
    hints = row["hints_text"] or ""

    # Build a pool of "file content" — real code from the repo's patch +
    # test_patch, repeated if needed to hit the target. This is the same
    # shape Codex would have after a few read_file calls.
    pool = "\n\n".join(p for p in (patch, test_patch, hints) if p)
    if not pool:
        pool = problem
    # Repeat to reach target byte count. Padding is real code from the same
    # repo, just chunked into multiple <tool_result> blocks so it looks
    # like several read_file calls.
    chunks = []
    chunk_size = max(2000, file_blocks_chars // 6)
    cur = 0
    idx = 1
    while cur < file_blocks_chars:
        seg = pool[(cur % max(1, len(pool))) : (cur % max(1, len(pool))) + chunk_size]
        if not seg:
            seg = pool[:chunk_size]
        chunks.append(
            f"<tool_result tool=\"read_file\" path=\"{repo}/_ctx_{idx}.py\">\n{seg}\n</tool_result>"
        )
        cur += len(seg)
        idx += 1

    file_blocks = "\n\n".join(chunks)
    return (
        f"Repository: {repo}\n"
        f"Instance: {iid}\n\n"
        f"## Issue\n{problem}\n\n"
        f"## Context I gathered\n"
        f"I ran `read_file` on the relevant modules. Their contents are:\n\n"
        f"{file_blocks}\n\n"
        f"## Task\n"
        f"Investigate the bug and reply with a single tool call to `apply_patch` "
        f"that fixes it. Keep the patch minimal."
    )


def build_prompt(tok, sys_prompt_path: Path, row, target_tokens: int) -> tuple:
    """Build a chat-templated prompt that hits ``target_tokens`` ±20%.

    Returns (text, n_tokens).
    """
    sys_text = sys_prompt_path.read_text(encoding="utf-8")
    # Iteratively grow file_blocks_chars until we hit target.
    # Empirically ~3.5 chars/token for Qwen on code.
    sys_tokens = len(tok.encode(sys_text, add_special_tokens=False))
    overhead = 200  # chat template + scaffolding
    target_user_tokens = max(256, target_tokens - sys_tokens - overhead)
    chars = max(1024, target_user_tokens * 4)

    for _ in range(6):
        user_text = _agent_user_message(row, chars)
        msgs = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_text},
        ]
        # Try chat template; fall back to plain concat if tokenizer has none.
        try:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            text = sys_text + "\n\n" + user_text + "\n\n"
        n = len(tok.encode(text, add_special_tokens=False))
        if abs(n - target_tokens) / target_tokens < 0.20:
            return text, n
        # binary search-ish: scale chars by ratio
        chars = max(512, int(chars * (target_tokens / max(1, n))))
    return text, n


def select_rows_for_bucket(df, target_tokens, n_sample, seed=42):
    """Pick rows whose problem_statement is small enough that we can grow to
    target without truncating the issue itself."""
    # Just shuffle — _agent_user_message handles padding to any size.
    return df.sample(n=n_sample, random_state=seed).to_dict("records")


# ── main bench loop ────────────────────────────────────────────────────────
def main():
    global DRAFT, BUDGET

    p = argparse.ArgumentParser(description="DFlash agentic-workload benchmark")
    p.add_argument("--budget", type=int, default=BUDGET)
    p.add_argument("--n-sample", type=int, default=5,
                   help="prompts per bucket (default 5)")
    p.add_argument("--bucket", choices=list(BUCKETS) + ["all"], default="all")
    p.add_argument("--n-gen", type=int, default=N_GEN)
    p.add_argument("--out", type=str, default=str(TMPDIR / "bench_agent_results.json"))
    p.add_argument("--skip-ar", action="store_true",
                   help="skip the AR baseline (useful for budget sweeps)")
    args = p.parse_args()
    BUDGET = args.budget

    DRAFT = _resolve_draft()
    _require_file(TARGET, "target GGUF")
    _require_file(TEST_DFLASH, "test_dflash binary")
    if not args.skip_ar:
        _require_file(TEST_GENERATE, "test_generate binary")
    _require_file(str(SWE_PARQUET), "SWE-bench Verified parquet")
    _require_file(str(SYS_PROMPT_SMALL), "small Codex system prompt fixture")
    _require_file(str(SYS_PROMPT_LARGE), "large Codex system prompt fixture")

    print(f"[bench-agent] target    = {TARGET}", flush=True)
    print(f"[bench-agent] draft     = {DRAFT}", flush=True)
    print(f"[bench-agent] tokenizer = {TOKENIZER}", flush=True)
    print(f"[bench-agent] budget    = {BUDGET}  n_gen = {args.n_gen}  n_sample = {args.n_sample}",
          flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

    df = _load_swe_rows()
    print(f"[bench-agent] loaded {len(df)} SWE-bench Verified rows", flush=True)

    bucket_keys = list(BUCKETS) if args.bucket == "all" else [args.bucket]
    results = {}
    load_s_estimate = None  # calibrated from first DFlash run

    for bk in bucket_keys:
        cfg = BUCKETS[bk]
        target = cfg["target"]
        sys_path = cfg["sys"]
        rows = select_rows_for_bucket(df, target, args.n_sample, seed=42)

        print(f"\n[bench-agent] === bucket {bk} (target ~{target} tok, n={len(rows)}) ===",
              flush=True)
        per_prompt = []
        for i, row in enumerate(rows):
            text, n = build_prompt(tok, sys_path, row, target)
            path = TMPDIR / f"agent_{bk}_{i:02d}.bin"
            tokenize_to_file(tok, text, path)

            try:
                df_res = run_df(path, n, args.n_gen)
            except Exception as e:
                print(f"  [{i+1:02d}/{len(rows)}] n={n:5d}  DFlash FAILED: {e}",
                      flush=True)
                continue

            # calibrate load_s from the first DFlash run if not done yet
            if load_s_estimate is None:
                load_s_estimate = max(
                    0.0,
                    df_res["wall_s"] - df_res["prefill_s"]
                    - args.n_gen / max(1e-6, df_res["decode_tps"])
                )
                print(f"  [calibration] estimated model_load_s = {load_s_estimate:.2f}",
                      flush=True)

            ar_res = None
            if not args.skip_ar:
                try:
                    ar_res = run_ar(path, args.n_gen)
                except Exception as e:
                    print(f"  [{i+1:02d}/{len(rows)}] n={n:5d}  AR FAILED: {e}",
                          flush=True)

            entry = {
                "bucket": bk, "i": i, "instance_id": row["instance_id"],
                "n_prompt": n, "n_gen": args.n_gen,
                "df": df_res, "ar": ar_res, "load_s_est": load_s_estimate,
            }

            # Derived numbers (per-prompt, AR vs DFlash)
            df_decode_s = args.n_gen / max(1e-6, df_res["decode_tps"])
            df_total_s = df_res["prefill_s"] + df_decode_s
            df_ttft_s = df_res["prefill_s"] + 1.0 / max(1e-6, df_res["decode_tps"])
            entry["df_total_s"] = df_total_s
            entry["df_ttft_s"] = df_ttft_s

            line = (
                f"  [{i+1:02d}/{len(rows)}] n={n:5d}  "
                f"DF prefill={df_res['prefill_s']:6.2f}s ({n/df_res['prefill_s']:6.1f} tok/s)  "
                f"decode={df_res['decode_tps']:6.2f} tok/s  "
                f"AL={df_res['al']:5.2f}  "
                f"TTFT={df_ttft_s:6.2f}s  total={df_total_s:6.2f}s"
            )
            if ar_res is not None:
                ar_decode_s = args.n_gen / max(1e-6, ar_res["decode_tps"])
                ar_prefill_s = max(0.0, ar_res["wall_s"] - load_s_estimate - ar_decode_s)
                ar_ttft_s = ar_prefill_s + 1.0 / max(1e-6, ar_res["decode_tps"])
                ar_total_s = ar_prefill_s + ar_decode_s
                entry["ar_prefill_s_est"] = ar_prefill_s
                entry["ar_total_s"] = ar_total_s
                entry["ar_ttft_s"] = ar_ttft_s
                entry["speedup_decode"] = df_res["decode_tps"] / max(1e-6, ar_res["decode_tps"])
                entry["speedup_total"] = ar_total_s / max(1e-6, df_total_s)
                line += (
                    f"  ||  AR prefill≈{ar_prefill_s:6.2f}s  "
                    f"decode={ar_res['decode_tps']:5.2f} tok/s  "
                    f"total={ar_total_s:6.2f}s  "
                    f"speedup decode={entry['speedup_decode']:.2f}x "
                    f"total={entry['speedup_total']:.2f}x"
                )
            print(line, flush=True)
            per_prompt.append(entry)

        # bucket aggregates
        if per_prompt:
            def _mean(xs): return sum(xs) / len(xs) if xs else 0.0
            agg = {
                "n_samples": len(per_prompt),
                "n_prompt_mean": _mean([e["n_prompt"] for e in per_prompt]),
                "df_prefill_s_mean": _mean([e["df"]["prefill_s"] for e in per_prompt]),
                "df_decode_tps_mean": _mean([e["df"]["decode_tps"] for e in per_prompt]),
                "df_al_mean": _mean([e["df"]["al"] for e in per_prompt]),
                "df_ttft_s_mean": _mean([e["df_ttft_s"] for e in per_prompt]),
                "df_total_s_mean": _mean([e["df_total_s"] for e in per_prompt]),
                "stages_mean": {
                    k: _mean([e["df"]["stages"].get(k, 0.0) for e in per_prompt])
                    for k in sorted({k for e in per_prompt for k in e["df"]["stages"]})
                },
            }
            ar_entries = [e for e in per_prompt if e.get("ar") is not None]
            if ar_entries:
                agg["ar_prefill_s_est_mean"] = _mean([e["ar_prefill_s_est"] for e in ar_entries])
                agg["ar_decode_tps_mean"] = _mean([e["ar"]["decode_tps"] for e in ar_entries])
                agg["ar_ttft_s_mean"] = _mean([e["ar_ttft_s"] for e in ar_entries])
                agg["ar_total_s_mean"] = _mean([e["ar_total_s"] for e in ar_entries])
                agg["speedup_decode_mean"] = _mean([e["speedup_decode"] for e in ar_entries])
                agg["speedup_total_mean"] = _mean([e["speedup_total"] for e in ar_entries])
            results[bk] = {"per_prompt": per_prompt, "agg": agg}

            print(f"\n  [{bk}] mean: n={agg['n_prompt_mean']:.0f}  "
                  f"DF prefill={agg['df_prefill_s_mean']:.2f}s  "
                  f"decode={agg['df_decode_tps_mean']:.2f} tok/s  "
                  f"AL={agg['df_al_mean']:.2f}  "
                  f"TTFT={agg['df_ttft_s_mean']:.2f}s  "
                  f"total={agg['df_total_s_mean']:.2f}s",
                  flush=True)
            if ar_entries:
                print(f"  [{bk}] mean: AR prefill≈{agg['ar_prefill_s_est_mean']:.2f}s  "
                      f"decode={agg['ar_decode_tps_mean']:.2f} tok/s  "
                      f"total={agg['ar_total_s_mean']:.2f}s  "
                      f"|| speedup decode={agg['speedup_decode_mean']:.2f}x  "
                      f"TOTAL={agg['speedup_total_mean']:.2f}x",
                      flush=True)

    # ── final comparison vs RESULTS.md headline ─────────────────────────
    print("\n[bench-agent] === COMPARISON vs RESULTS.md HumanEval headline ===")
    print(f"{'Bucket':>8s}  {'n_tok':>6s}  {'AR tps':>7s}  {'DF tps':>7s}  "
          f"{'AL':>5s}  {'TTFT':>7s}  {'Total':>7s}  {'sp_dec':>7s}  {'sp_tot':>7s}")
    print(f"{'HumanEv':>8s}  {' ~120':>6s}  {' 37.78':>7s}  {'129.52':>7s}  "
          f"{' 8.31':>5s}  {'   --':>7s}  {'   --':>7s}  {' 3.43x':>7s}  {'   --':>7s}  "
          f"(from RESULTS.md)")
    for bk, r in results.items():
        a = r["agg"]
        sp_d = a.get("speedup_decode_mean", 0.0)
        sp_t = a.get("speedup_total_mean", 0.0)
        ar_tps = a.get("ar_decode_tps_mean", 0.0)
        print(f"{bk:>8s}  {a['n_prompt_mean']:6.0f}  {ar_tps:7.2f}  "
              f"{a['df_decode_tps_mean']:7.2f}  {a['df_al_mean']:5.2f}  "
              f"{a['df_ttft_s_mean']:6.2f}s  {a['df_total_s_mean']:6.2f}s  "
              f"{sp_d:6.2f}x  {sp_t:6.2f}x")

    # write JSON (per-prompt and agg)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[bench-agent] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
