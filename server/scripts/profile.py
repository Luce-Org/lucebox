#!/usr/bin/env python3
"""
Lucebox speed profiler (MVP)
============================
Measures forward-pass speed of the Lucebox inference binaries on one GPU and emits
a machine-readable profile.json + a human-readable profile.md (the report a reviewer
reads on a PR).

What it captures
----------------
  * prefill time, model-side TTFT estimate, decode tok/s, ms/token
  * speculative-decoding acceptance length (AL) and accept %
  * per-step phase breakdown (draft_* / verify_* ...) from the engine's own timers
  * fine-grained kernel view via Nsight Systems (nsys):
        - top CUDA kernels by GPU time
        - kernel launches per generated token   -> kernel-FUSION signal
        - host<->device memcpy time per token   -> CPU/GPU-OVERLAP signal
        - sync-heavy CUDA APIs                   -> CPU-stall signal
  * optional losslessness gate: greedy spec-decode must equal greedy AR token-for-token
  * optional delta vs a stored baseline (regression view)

Design choices
--------------
  * Wraps the binaries directly (no HTTP) so the numbers reflect compute, not server noise.
  * The nsys pass is SEPARATE from the timing pass: nsys adds overhead, so tok/s is
    measured on a clean pass and the kernel breakdown on its own short pass.
  * Built to run on the self-hosted RTX 3090 CI runner; keep --n-gen small in CI.

Usage
-----
  python3 profile.py \
      --target /opt/models/Qwen3.5-27B-Q4_K_M.gguf \
      --draft  /opt/models/draft/qwen35_dflash/model.safetensors \
      --n-gen 48 --budget 22 --reps 3 --nsys \
      --check-lossless \
      --out-json profile.json --out-md profile.md
"""
from __future__ import annotations
import argparse, csv, datetime, json, os, re, statistics, struct, subprocess, sys
from pathlib import Path

# --------------------------------------------------------------------------------------
# Canonical prompts (HumanEval-style completion). Override with --prompts <file.jsonl>
# where each line is {"name": "...", "text": "..."}.
# --------------------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    {"name": "has_close_elements",
     "text": ("from typing import List\n\n"
              "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
              '    """Check if any two numbers are closer than the threshold.\n'
              "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n"
              '    """\n    for')},
    {"name": "rolling_max",
     "text": ("from typing import List\n\n"
              "def rolling_max(numbers: List[int]) -> List[int]:\n"
              '    """Generate a list of rolling maximum elements seen so far.\n'
              "    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n    [1, 2, 3, 3, 3, 4, 4]\n"
              '    """\n    result = []\n    running_max = None\n    for n in numbers:\n        if running_max is')},
    {"name": "sum_product",
     "text": ("from typing import List, Tuple\n\n"
              "def sum_product(numbers: List[int]) -> Tuple[int, int]:\n"
              '    """Return a tuple of (sum, product) of all integers. Empty -> (0, 1).\n'
              "    >>> sum_product([1, 2, 3, 4])\n    (10, 24)\n"
              '    """\n    s = 0\n    p = 1\n    for n in')},
]

# --------------------------------------------------------------------------------------
# stdout parsers (matched to the exact print formats in test_dflash.cpp / test_generate.cpp)
# --------------------------------------------------------------------------------------
RE_PREFILL      = re.compile(r"\[prefill\]\s+(\d+)\s+tokens?\s+in\s+([\d.]+)\s*s")
RE_DECODE       = re.compile(r"\[dflash\]\s+generated\s+(\d+)\s+tokens\s+in\s+([\d.]+)\s*s\s*->\s*([\d.]+)\s*tok/s")
RE_STEPS        = re.compile(r"\[dflash\]\s+(\d+)\s+draft steps,\s+accepted=(\d+)/(\d+)\s+\(([\d.]+)%[^)]*\),\s+avg commit/step=([\d.]+)")
RE_AR           = re.compile(r"\[gen\]\s+(\d+)\s+new tokens in\s+([\d.]+)\s*s\s*->\s*([\d.]+)\s*tok/s")
RE_PHASE_LINE   = re.compile(r"^\s+([a-z_]+)\s+([\d.]+)\s*$")
RE_PHASE_SUM    = re.compile(r"-----\s+sum\s+([\d.]+)")
RE_TIMING_START = re.compile(r"\[timing\] per-step averages")


def parse_dflash(text: str) -> dict:
    out: dict = {"phases": {}}
    if m := RE_PREFILL.search(text):
        out["prefill_s"] = float(m.group(2)); out["prompt_tokens"] = int(m.group(1))
    if m := RE_DECODE.search(text):
        out["decode_tokens"] = int(m.group(1)); out["decode_s"] = float(m.group(2)); out["decode_tok_s"] = float(m.group(3))
    if m := RE_STEPS.search(text):
        out["steps"] = int(m.group(1)); out["accepted"] = int(m.group(2))
        out["draft_positions"] = int(m.group(3)); out["accept_pct"] = float(m.group(4)); out["al"] = float(m.group(5))
    # phase block
    in_block = False
    for line in text.splitlines():
        if RE_TIMING_START.search(line):
            in_block = True; continue
        if in_block:
            if m := RE_PHASE_SUM.search(line):
                out["phase_sum_ms"] = float(m.group(1)); in_block = False; continue
            if m := RE_PHASE_LINE.match(line):
                out["phases"][m.group(1)] = float(m.group(2))
    # derived
    if out.get("decode_tok_s"):
        out["ms_per_token"] = 1000.0 / out["decode_tok_s"]
    if "prefill_s" in out and "phase_sum_ms" in out:
        out["ttft_est_ms"] = out["prefill_s"] * 1000.0 + out["phase_sum_ms"]
    return out


def parse_ar(text: str) -> dict:
    if m := RE_AR.search(text):
        return {"ar_tokens": int(m.group(1)), "ar_s": float(m.group(2)), "ar_tok_s": float(m.group(3))}
    return {}


# --------------------------------------------------------------------------------------
# binary runners
# --------------------------------------------------------------------------------------
def tokenize(prompt: str, tok, out_path: Path) -> int:
    ids = tok.encode(prompt, add_special_tokens=False)
    out_path.write_bytes(struct.pack(f"<{len(ids)}i", *ids))
    return len(ids)


def run(cmd: list[str], timeout: int = 1200) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        sys.stderr.write(f"\n[profile] command failed ({r.returncode}): {' '.join(cmd)}\n")
        sys.stderr.write((r.stderr or r.stdout)[-2000:] + "\n")
        raise RuntimeError("binary failed")
    return r.stdout


def dflash_cmd(cfg, prompt_bin: Path, out_bin: Path, n_gen: int) -> list[str]:
    return [cfg.df_bin, cfg.target, cfg.draft, str(prompt_bin), str(n_gen), str(out_bin),
            "--fast-rollback", "--ddtree", f"--ddtree-budget={cfg.budget}"]


def ar_cmd(cfg, prompt_bin: Path, out_bin: Path, n_gen: int) -> list[str]:
    return [cfg.ar_bin, cfg.target, str(prompt_bin), str(n_gen), str(out_bin)]


# --------------------------------------------------------------------------------------
# nsys (Nsight Systems) — the fine-grained kernel layer
# --------------------------------------------------------------------------------------
def have_nsys() -> bool:
    from shutil import which
    return which("nsys") is not None


def nsys_profile(cmd: list[str], rep_out: Path) -> bool:
    full = ["nsys", "profile", "-t", "cuda", "--force-overwrite", "true",
            "-o", str(rep_out.with_suffix("")), *cmd]
    try:
        subprocess.run(full, capture_output=True, text=True, timeout=1800, check=True)
        return True
    except Exception as e:  # noqa
        sys.stderr.write(f"[profile] nsys profile failed: {e}\n")
        return False


def _num(row: dict, *names) -> float:
    for n in names:
        if n in row and str(row[n]).strip() not in ("", "-"):
            try:
                return float(str(row[n]).replace(",", ""))
            except ValueError:
                pass
    return 0.0


def _str(row: dict, *names) -> str:
    for n in names:
        if n in row and row[n]:
            return str(row[n])
    return ""


def nsys_report(rep: Path, report: str) -> list[dict]:
    try:
        r = subprocess.run(["nsys", "stats", "--report", report, "--format", "csv",
                            "--force-export=true", str(rep)],
                           capture_output=True, text=True, timeout=600)
    except Exception:
        return []
    lines = r.stdout.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if "Total Time (ns)" in l or "Time (%)" in l), None)
    if start is None:
        return []
    return list(csv.DictReader(lines[start:]))


def summarize_nsys(rep: Path, tokens: int) -> dict:
    kern = nsys_report(rep, "cuda_gpu_kern_sum")
    mem  = nsys_report(rep, "cuda_gpu_mem_time_sum")
    api  = nsys_report(rep, "cuda_api_sum")

    total_kern_ns = sum(_num(r, "Total Time (ns)") for r in kern)
    total_insts   = sum(_num(r, "Instances", "Count") for r in kern)
    top = sorted(kern, key=lambda r: _num(r, "Total Time (ns)"), reverse=True)[:8]
    top_kernels = [{"name": _str(r, "Name")[:70],
                    "total_ms": round(_num(r, "Total Time (ns)") / 1e6, 2),
                    "instances": int(_num(r, "Instances", "Count")),
                    "avg_us": round(_num(r, "Avg (ns)") / 1e3, 2)} for r in top]

    mem_ns = sum(_num(r, "Total Time (ns)") for r in mem)
    mem_breakdown = [{"op": _str(r, "Operation", "Name"),
                      "total_ms": round(_num(r, "Total Time (ns)") / 1e6, 2),
                      "count": int(_num(r, "Count", "Instances"))} for r in mem]

    sync_apis = [r for r in api if any(k in _str(r, "Name")
                 for k in ("Synchronize", "cudaMemcpy", "cudaLaunchKernel", "cudaStreamSync"))]
    api_rows = [{"name": _str(r, "Name"),
                 "total_ms": round(_num(r, "Total Time (ns)") / 1e6, 2),
                 "calls": int(_num(r, "Num Calls", "Count"))} for r in sync_apis]

    tk = max(1, tokens)
    return {
        "tokens_profiled": tokens,
        "gpu_kernel_total_ms": round(total_kern_ns / 1e6, 2),
        "kernel_launches": int(total_insts),
        "launches_per_token": round(total_insts / tk, 1),
        "memcpy_total_ms": round(mem_ns / 1e6, 2),
        "memcpy_ms_per_token": round(mem_ns / 1e6 / tk, 3),
        "top_kernels": top_kernels,
        "memcpy_breakdown": mem_breakdown,
        "sync_apis": sorted(api_rows, key=lambda x: x["total_ms"], reverse=True)[:8],
    }


# --------------------------------------------------------------------------------------
# losslessness gate
# --------------------------------------------------------------------------------------
def read_i32(p: Path) -> list[int]:
    b = p.read_bytes()
    return list(struct.unpack(f"<{len(b)//4}i", b))


def check_lossless(prompt_len: int, ar_bin: Path, df_bin: Path) -> dict:
    ar = read_i32(ar_bin)[prompt_len:]
    df = read_i32(df_bin)[prompt_len:]
    n = min(len(ar), len(df))
    first = next((i for i in range(n) if ar[i] != df[i]), None)
    return {"lossless": first is None,
            "compared_tokens": n,
            "first_divergence": first,
            "ar_len": len(ar), "df_len": len(df)}


# --------------------------------------------------------------------------------------
# heuristic flags — turn raw numbers into "where is the margin"
# --------------------------------------------------------------------------------------
def optimization_flags(summary: dict) -> list[str]:
    flags = []
    phases = summary.get("phases_mean", {})
    step = summary.get("phase_sum_ms_mean", 0) or 1
    for name, ms in sorted(phases.items(), key=lambda kv: kv[1], reverse=True)[:2]:
        if ms / step > 0.30:
            flags.append(f"`{name}` is {ms/step*100:.0f}% of the step ({ms:.1f} ms) — primary target.")
    n = summary.get("nsys")
    if n:
        if n["launches_per_token"] > 50:
            flags.append(f"{n['launches_per_token']} kernel launches/token — kernel-fusion candidate "
                         f"(launch overhead dominates many tiny kernels).")
        if n["memcpy_ms_per_token"] > 0.5:
            flags.append(f"{n['memcpy_ms_per_token']:.2f} ms/token in host<->device copies — "
                         f"CPU/GPU-overlap candidate (data shuttling off the critical path).")
    return flags


# --------------------------------------------------------------------------------------
# markdown report
# --------------------------------------------------------------------------------------
def render_md(doc: dict) -> str:
    c, s = doc["config"], doc["summary"]
    L = []
    L.append("# Lucebox speed profile\n")
    L.append(f"- created: `{doc['created']}`")
    L.append(f"- commit: `{doc.get('commit','?')}`")
    L.append(f"- gpu: `{c.get('gpu','?')}`  driver: `{c.get('driver','?')}`  power: `{c.get('power','?')}`")
    L.append(f"- target: `{Path(c['target']).name}`  draft: `{Path(c['draft']).name}`")
    L.append(f"- n_gen: `{c['n_gen']}`  budget: `{c['budget']}`  reps: `{c['reps']}`  nsys: `{c['nsys']}`\n")

    L.append("## Headline\n")
    L.append("| metric | value |")
    L.append("|---|---:|")
    if "ar_tok_s_mean" in s:    L.append(f"| AR decode (tok/s) | {s['ar_tok_s_mean']:.2f} |")
    L.append(f"| DFlash decode (tok/s) | {s['decode_tok_s_mean']:.2f} |")
    if "speedup" in s:          L.append(f"| speedup vs AR | {s['speedup']:.2f}x |")
    L.append(f"| ms / token | {s['ms_per_token_mean']:.2f} |")
    L.append(f"| TTFT estimate (ms) | {s['ttft_est_ms_mean']:.1f} |")
    L.append(f"| prefill (ms) | {s['prefill_ms_mean']:.1f} |")
    L.append(f"| acceptance length (AL) | {s['al_mean']:.2f} |")
    L.append(f"| accept % / step | {s['accept_pct_mean']:.1f} |")
    L.append(f"| decode tok/s spread (min–max) | {s['decode_tok_s_min']:.1f}–{s['decode_tok_s_max']:.1f} |\n")

    if "lossless" in doc:
        ll = doc["lossless"]
        verdict = "PASS — bit-identical to greedy AR" if ll["lossless"] else \
                  f"FAIL — diverges at generated token #{ll['first_divergence']}"
        L.append("## Correctness (losslessness gate)\n")
        L.append(f"- **{verdict}** (compared {ll['compared_tokens']} tokens)\n")

    L.append("## Per-step phase breakdown (engine timers, ms/step)\n")
    L.append("| phase | ms/step | % of step |")
    L.append("|---|---:|---:|")
    step = s.get("phase_sum_ms_mean", 0) or 1
    for name, ms in sorted(s.get("phases_mean", {}).items(), key=lambda kv: kv[1], reverse=True):
        L.append(f"| `{name}` | {ms:.2f} | {ms/step*100:.0f}% |")
    L.append(f"| **sum** | **{step:.2f}** | 100% |\n")

    if s.get("nsys"):
        n = s["nsys"]
        L.append("## Kernel-level (nsys)\n")
        L.append(f"- GPU kernel time: `{n['gpu_kernel_total_ms']} ms`  over `{n['tokens_profiled']}` tokens")
        L.append(f"- kernel launches/token: `{n['launches_per_token']}`  (fusion signal)")
        L.append(f"- host<->device copy: `{n['memcpy_total_ms']} ms` total, "
                 f"`{n['memcpy_ms_per_token']} ms/token` (overlap signal)\n")
        L.append("**Top kernels by GPU time**\n")
        L.append("| kernel | total ms | launches | avg µs |")
        L.append("|---|---:|---:|---:|")
        for k in n["top_kernels"]:
            L.append(f"| `{k['name']}` | {k['total_ms']} | {k['instances']} | {k['avg_us']} |")
        L.append("")
        if n["sync_apis"]:
            L.append("**Sync / launch / copy CUDA APIs** (CPU-stall signal)\n")
            L.append("| api | total ms | calls |")
            L.append("|---|---:|---:|")
            for a in n["sync_apis"]:
                L.append(f"| `{a['name']}` | {a['total_ms']} | {a['calls']} |")
            L.append("")

    flags = optimization_flags(s)
    if flags:
        L.append("## Where the margin is\n")
        for f in flags:
            L.append(f"- {f}")
        L.append("")

    if doc.get("baseline_delta"):
        L.append("## Delta vs baseline\n")
        L.append("| metric | baseline | now | Δ |")
        L.append("|---|---:|---:|---:|")
        for k, (b, now, d) in doc["baseline_delta"].items():
            L.append(f"| {k} | {b:.2f} | {now:.2f} | {d:+.2f} |")
        L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def gpu_info() -> dict:
    try:
        out = subprocess.run(["nvidia-smi",
                              "--query-gpu=name,driver_version,power.limit",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
        name, driver, power = (x.strip() for x in out.split(",")[:3])
        return {"gpu": name, "driver": driver, "power": power}
    except Exception:
        return {}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "?"


def median(xs):  # tolerant
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--df-bin", default=os.environ.get("DFLASH_BIN", "build/test_dflash"))
    ap.add_argument("--ar-bin", default=os.environ.get("DFLASH_BIN_AR", "build/test_generate"))
    ap.add_argument("--tokenizer", default=os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--n-gen", type=int, default=48)
    ap.add_argument("--budget", type=int, default=22)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--nsys", action="store_true")
    ap.add_argument("--check-lossless", action="store_true")
    ap.add_argument("--prompts", default=None, help="optional .jsonl of {name,text}")
    ap.add_argument("--baseline", default=None, help="optional baseline profile.json for delta")
    ap.add_argument("--out-json", default="profile.json")
    ap.add_argument("--out-md", default="profile.md")
    cfg = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer, trust_remote_code=True)

    prompts = DEFAULT_PROMPTS
    if cfg.prompts:
        prompts = [json.loads(l) for l in Path(cfg.prompts).read_text().splitlines() if l.strip()]

    tmp = Path("/tmp/lucebox_profile"); tmp.mkdir(exist_ok=True)
    runs = []
    for p in prompts:
        pbin = tmp / f"{p['name']}.bin"
        plen = tokenize(p["text"], tok, pbin)

        # clean timing passes (median over reps)
        reps = []
        for _ in range(cfg.reps):
            txt = run(dflash_cmd(cfg, pbin, tmp / "df.bin", cfg.n_gen))
            reps.append(parse_dflash(txt))
        agg = dict(reps[-1])  # keep phases from last
        agg["decode_tok_s"] = median([r.get("decode_tok_s") for r in reps])
        agg["al"]           = median([r.get("al") for r in reps])
        agg["prompt"] = p["name"]; agg["prompt_len"] = plen

        # AR baseline (single pass — it is deterministic)
        ar_txt = run(ar_cmd(cfg, pbin, tmp / "ar.bin", cfg.n_gen))
        agg.update(parse_ar(ar_txt))

        # losslessness (uses the AR + a fresh greedy DFlash output we just wrote)
        if cfg.check_lossless:
            run(dflash_cmd(cfg, pbin, tmp / "df_ll.bin", cfg.n_gen))
            run(ar_cmd(cfg, pbin, tmp / "ar_ll.bin", cfg.n_gen))
            agg["lossless"] = check_lossless(plen, tmp / "ar_ll.bin", tmp / "df_ll.bin")

        runs.append(agg)

    # nsys on the first prompt only (one short profiled pass)
    nsys = None
    if cfg.nsys:
        if not have_nsys():
            sys.stderr.write("[profile] nsys not found on PATH — skipping kernel layer.\n")
        else:
            pbin = tmp / f"{prompts[0]['name']}.bin"
            rep = tmp / "profile.nsys-rep"
            if nsys_profile(dflash_cmd(cfg, pbin, tmp / "df_nsys.bin", cfg.n_gen), rep):
                nsys = summarize_nsys(rep, cfg.n_gen)

    # ---- aggregate summary across prompts ----
    def col(k): return [r[k] for r in runs if k in r]
    summary = {
        "decode_tok_s_mean": statistics.mean(col("decode_tok_s")),
        "decode_tok_s_min": min(col("decode_tok_s")),
        "decode_tok_s_max": max(col("decode_tok_s")),
        "ms_per_token_mean": statistics.mean([1000.0 / x for x in col("decode_tok_s")]),
        "al_mean": statistics.mean(col("al")),
        "accept_pct_mean": statistics.mean(col("accept_pct")) if col("accept_pct") else 0.0,
        "prefill_ms_mean": statistics.mean([x * 1000 for x in col("prefill_s")]) if col("prefill_s") else 0.0,
        "ttft_est_ms_mean": statistics.mean(col("ttft_est_ms")) if col("ttft_est_ms") else 0.0,
        "phase_sum_ms_mean": statistics.mean(col("phase_sum_ms")) if col("phase_sum_ms") else 0.0,
    }
    if col("ar_tok_s"):
        summary["ar_tok_s_mean"] = statistics.mean(col("ar_tok_s"))
        summary["speedup"] = summary["decode_tok_s_mean"] / summary["ar_tok_s_mean"]
    # phase means
    pkeys = set().union(*[set(r.get("phases", {})) for r in runs]) if runs else set()
    summary["phases_mean"] = {k: statistics.mean([r["phases"][k] for r in runs if k in r.get("phases", {})])
                              for k in pkeys}
    if nsys:
        summary["nsys"] = nsys

    doc = {
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": git_commit(),
        "config": {**vars(cfg), **gpu_info()},
        "runs": runs,
        "summary": summary,
    }
    if cfg.check_lossless and runs and "lossless" in runs[0]:
        doc["lossless"] = runs[0]["lossless"]  # representative

    # baseline delta
    if cfg.baseline and Path(cfg.baseline).exists():
        base = json.loads(Path(cfg.baseline).read_text())["summary"]
        keys = ["decode_tok_s_mean", "al_mean", "ttft_est_ms_mean", "ms_per_token_mean"]
        doc["baseline_delta"] = {k: (base[k], summary[k], summary[k] - base[k])
                                 for k in keys if k in base and k in summary}

    Path(cfg.out_json).write_text(json.dumps(doc, indent=2))
    Path(cfg.out_md).write_text(render_md(doc))
    print(render_md(doc))
    print(f"\n[profile] wrote {cfg.out_json} and {cfg.out_md}")


if __name__ == "__main__":
    main()
