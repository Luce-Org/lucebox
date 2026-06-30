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
  * repeated-run mean/stddev and report-only noise warnings
  * optional delta vs a stored baseline (regression view)

Design choices
--------------
  * Wraps the binaries directly (no HTTP) so the numbers reflect compute, not server noise.
  * The nsys pass is SEPARATE from the timing pass: nsys adds overhead, so tok/s is
    measured on a clean pass and the kernel breakdown on its own short pass.
  * Built to run on the self-hosted RTX 3090 CI runner; --n-gen controls the
    requested generated tokens per prompt, so keep it representative but bounded.

Usage
-----
  python3 profile.py \
      --target /opt/models/Qwen3.6-27B-Q4_K_M.gguf \
      --draft  /opt/models/draft/dflash-draft-3.6-q4_k_m.gguf \
      --n-gen 128 --budget 22 --reps 5 --nsys \
      --check-lossless \
      --baseline scripts/speed-baseline.json --regress-pct 0.10 \
      --out-json profile.json --out-md profile.md
"""
from __future__ import annotations
import argparse, atexit, csv, datetime, json, os, re, shutil, statistics, struct, subprocess, sys, tempfile
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


# nsys renamed its canned reports across versions (e.g. modern `cuda_gpu_kern_sum`
# vs legacy `gpukernsum`). Trying both is what keeps the kernel layer working
# whatever nsys ships on the runner — a name mismatch is the usual reason the nsys
# section comes out all-zero with an empty table.
NSYS_REPORTS = {
    "kern": ["cuda_gpu_kern_sum", "gpukernsum"],
    "mem":  ["cuda_gpu_mem_time_sum", "gpumemtimesum"],
    "api":  ["cuda_api_sum", "cudaapisum"],
}


def nsys_version() -> str:
    try:
        out = subprocess.run(["nsys", "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
        return out.splitlines()[0] if out else "?"
    except Exception:
        return "?"


def _looks_like_header(line: str) -> bool:
    # nsys CSV section header, tolerant to per-version column naming.
    return "," in line and ("Total Time (ns)" in line or "Total Time" in line
                            or ("Time (%)" in line and "Name" in line))


def nsys_report(rep: Path, names: list[str]) -> tuple[list[dict], str]:
    """Run `nsys stats` for the first report-name alias that yields a table.

    Returns (rows, diag). diag is "" on success, else a one-line reason the
    section is empty so the report can say WHY the kernel layer is missing
    instead of silently printing zeros."""
    diag = ""
    for name in names:
        try:
            r = subprocess.run(["nsys", "stats", "--report", name, "--format", "csv",
                                "--force-export=true", str(rep)],
                               capture_output=True, text=True, timeout=600)
        except Exception as e:  # noqa
            diag = f"{name}: {e}"
            continue
        lines = r.stdout.splitlines()
        start = next((i for i, l in enumerate(lines) if _looks_like_header(l)), None)
        if start is not None:
            rows = list(csv.DictReader(lines[start:]))
            if rows:
                return rows, ""
        err = (r.stderr or r.stdout).strip()
        diag = f"{name}: {err.splitlines()[-1] if err else 'no table in output'}"
    return [], diag


def summarize_nsys(rep: Path, tokens: int) -> dict:
    kern, dk = nsys_report(rep, NSYS_REPORTS["kern"])
    mem,  dm = nsys_report(rep, NSYS_REPORTS["mem"])
    api,  da = nsys_report(rep, NSYS_REPORTS["api"])

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
    out = {
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
    diagnostics = {
        "kern": dk if not kern and dk else "",
        "mem": dm if not mem and dm else "",
        "api": da if not api and da else "",
    }
    missing = {section: diag for section, diag in diagnostics.items() if diag}

    # If the critical kernel report failed to parse, the launch/fusion metrics are
    # not measurements. Treat that as an nsys error even when secondary reports
    # (memcpy/API) produced rows, instead of rendering zero kernel metrics as real.
    if diagnostics["kern"]:
        out["error"] = f"kernel report unavailable: {diagnostics['kern']}"
        if any((dm, da)):
            out["warnings"] = [f"{section}: {diag}" for section, diag in missing.items()]
        out["nsys_version"] = nsys_version()
    elif missing:
        # Non-kernel sections are useful but not required for the fusion signal.
        # Surface their diagnostics while still rendering the valid kernel data.
        out["warnings"] = [f"{section}: {diag}" for section, diag in missing.items()]
        out["nsys_version"] = nsys_version()
    return out


# --------------------------------------------------------------------------------------
# losslessness gate
# --------------------------------------------------------------------------------------
def read_i32(p: Path) -> list[int]:
    b = p.read_bytes()
    return list(struct.unpack(f"<{len(b)//4}i", b))


def _first_divergence(a: list[int], b: list[int]) -> tuple[int | None, int]:
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), None), n


def check_lossless(prompt_len: int, ar_bin: Path, df_bin: Path,
                   ar_ctrl_bin: Path | None = None) -> dict:
    """Compare greedy spec-decode (df) against greedy autoregressive (ar).

    A raw bit-exact compare is too strict on its own: the target sees draft
    tokens as a *batch* in the verify step but one-at-a-time in AR decode, and
    different GEMM shapes reduce in a different order. IEEE FP is non-associative,
    so when the top-2 logits are within epsilon the argmax tie can flip — one
    token diverges and everything after it follows. The engine itself documents
    this (qwen35_backend.cpp: "different GPU batch sizes -> FP-nondeterministic
    state divergence -> different greedy output").

    To tell that intrinsic noise apart from a real spec-decode bug we run a
    second, identical-config AR pass as a determinism control (same idea as the
    eval harness's baseline_2). If AR disagrees with itself on a prompt, that
    prompt is intrinsically nondeterministic on this engine and a df-vs-ar
    mismatch there proves nothing — so we mark it inconclusive rather than FAIL."""
    ar = read_i32(ar_bin)[prompt_len:]
    df = read_i32(df_bin)[prompt_len:]
    first, n = _first_divergence(ar, df)
    out = {"lossless": first is None,
           "compared_tokens": n,
           "first_divergence": first,
           "ar_len": len(ar), "df_len": len(df)}
    if ar_ctrl_bin is not None:
        ar2 = read_i32(ar_ctrl_bin)[prompt_len:]
        ar_div, _ = _first_divergence(ar, ar2)
        out["ar_deterministic"] = ar_div is None   # control: AR == AR on rerun?
        out["ar_self_divergence"] = ar_div
    return out


# --------------------------------------------------------------------------------------
# baseline regression check
# --------------------------------------------------------------------------------------
# Which way is "better" for each headline metric: +1 = higher is better (throughput),
# -1 = lower is better (latency). A regression is a move the WRONG way by more than the
# threshold; a move the right way is just an improvement, never flagged.
REGRESS_DIRECTION = {
    "decode_tok_s_mean": +1,
    "al_mean":           +1,
    "ttft_est_ms_mean":  -1,
    "ms_per_token_mean": -1,
}

REGRESS_STD_KEYS = {
    "decode_tok_s_mean": "decode_tok_s_stddev",
    "al_mean": "al_stddev",
    "ttft_est_ms_mean": "ttft_est_ms_stddev",
    "ms_per_token_mean": "ms_per_token_stddev",
}


def _overlap_1sigma(base_mean: float, base_std: float, now_mean: float, now_std: float) -> bool:
    return max(base_mean - base_std, now_mean - now_std) <= min(base_mean + base_std, now_mean + now_std)


def regression_delta(base: dict, now: dict, thr: float) -> dict:
    """Per-metric (baseline, now, delta, pct, regressed?) + an overall verdict."""
    rows = {}
    for k, direction in REGRESS_DIRECTION.items():
        if k not in base or k not in now:
            continue
        b, n = base[k], now[k]
        pct = (n - b) / b if b else 0.0
        regressed = (direction > 0 and pct < -thr) or (direction < 0 and pct > thr)
        std_key = REGRESS_STD_KEYS.get(k)
        base_std = float(base.get(std_key, 0.0)) if std_key else 0.0
        now_std = float(now.get(std_key, 0.0)) if std_key else 0.0
        overlap = _overlap_1sigma(b, base_std, n, now_std) if (base_std or now_std) else False
        below_detection = overlap and abs(pct) < thr
        rows[k] = {
            "baseline": b, "now": n, "delta": n - b, "pct": pct, "regressed": regressed,
            "baseline_stddev": base_std, "now_stddev": now_std,
            "overlap_1sigma": overlap, "below_detection_threshold": below_detection,
        }
    return rows


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
    if n and not n.get("error"):   # don't reason from a failed/zeroed nsys pass
        if n.get("launches_per_token", 0) > 50:
            flags.append(f"{n['launches_per_token']} kernel launches/token — kernel-fusion candidate "
                         f"(launch overhead dominates many tiny kernels).")
        if n.get("memcpy_ms_per_token", 0) > 0.5:
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
    L.append(f"| DFlash decode (tok/s) | {s['decode_tok_s_mean']:.2f} ± {s.get('decode_tok_s_stddev', 0.0):.2f} |")
    if "speedup" in s:          L.append(f"| speedup vs AR | {s['speedup']:.2f}x |")
    L.append(f"| ms / token | {s['ms_per_token_mean']:.2f} ± {s.get('ms_per_token_stddev', 0.0):.2f} |")
    L.append(f"| TTFT estimate (ms) | {s['ttft_est_ms_mean']:.1f} ± {s.get('ttft_est_ms_stddev', 0.0):.1f} |")
    L.append(f"| prefill (ms) | {s['prefill_ms_mean']:.1f} |")
    L.append(f"| acceptance length (AL) | {s['al_mean']:.2f} ± {s.get('al_stddev', 0.0):.2f} |")
    L.append(f"| accept % / step | {s['accept_pct_mean']:.1f} |")
    L.append(f"| decode tok/s spread (min–max) | {s['decode_tok_s_min']:.1f}–{s['decode_tok_s_max']:.1f} |\n")

    noise = s.get("noise", {})
    if noise:
        threshold = noise.get("threshold_rsd", 0.0)
        if noise.get("noisy"):
            L.append("## Noise / detection threshold\n")
            L.append(f"- ⚠️ **NOISY** — relative stddev exceeded `{threshold*100:.1f}%` for "
                     f"{', '.join(noise.get('metrics', []))}. Treat small deltas as below detection threshold.")
            L.append("- This profiler remains report-only: noise warnings do not fail the run.\n")
        else:
            L.append("## Noise / detection threshold\n")
            L.append(f"- ✅ **stable** — all tracked relative stddevs are at or below `{threshold*100:.1f}%`.\n")


    if doc.get("runs"):
        L.append("## Repeated-run samples by prompt\n")
        L.append("| prompt | decode tok/s | ms/token | AL | TTFT ms |")
        L.append("|---|---:|---:|---:|---:|")
        for r in doc["runs"]:
            L.append(f"| `{r.get('prompt', '?')}` | "
                     f"{r.get('decode_tok_s', 0.0):.2f} ± {r.get('decode_tok_s_stddev', 0.0):.2f} | "
                     f"{r.get('ms_per_token', 0.0):.2f} ± {r.get('ms_per_token_stddev', 0.0):.2f} | "
                     f"{r.get('al', 0.0):.2f} ± {r.get('al_stddev', 0.0):.2f} | "
                     f"{r.get('ttft_est_ms', 0.0):.1f} ± {r.get('ttft_est_ms_stddev', 0.0):.1f} |")
        L.append("")

    if "lossless" in doc:
        ll = doc["lossless"]
        checked = ll["prompts_checked"]
        inconc = ll.get("prompts_inconclusive", [])
        if not ll["lossless"]:
            verdict = (f"FAIL — spec-decode output differs from greedy AR on {len(ll['prompts_failed'])}/{checked} "
                       f"prompts ({', '.join(ll['prompts_failed'])}), first at token #{ll['first_divergence']}. "
                       f"AR is self-deterministic, so this is NOT run-to-run noise — but not proven a logic bug "
                       f"either: it can be batched-verify FP (verify scores draft tokens as a batch vs AR "
                       f"one-at-a-time). Classify via the logit gap at the divergence: near-tie = FP, clear gap = bug.")
        elif inconc:
            verdict = (f"PASS — {len(ll['prompts_passed'])}/{checked} bit-identical to greedy AR; "
                       f"{len(inconc)} inconclusive ({', '.join(inconc)}) — the engine is intrinsically "
                       f"nondeterministic on those (AR disagreed with itself), so spec-decode can't be judged.")
        else:
            verdict = f"PASS — all {checked} prompts bit-identical to greedy AR"
        L.append("## Correctness (losslessness gate)\n")
        L.append(f"- **{verdict}**")
        L.append("- FAIL = output changed and it is not run-to-run noise (AR agreed with itself); "
                 "inconclusive prompts (AR non-deterministic) are excluded. FP-vs-bug needs the logit "
                 "gap at the first mismatch (a follow-up the binaries don't emit yet).\n")

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
        for warning in n.get("warnings", []):
            L.append(f"- ⚠️ nsys stats warning: `{warning}` (nsys `{n.get('nsys_version','?')}`).")
        if n.get("warnings"):
            L.append("")
        if n.get("error"):
            # Trace was captured but stats wouldn't parse — say why instead of
            # printing 0.0 everywhere with an empty table.
            L.append(f"- ⚠️ nsys trace captured but kernel stats could not be parsed: "
                     f"`{n['error']}` (nsys `{n.get('nsys_version','?')}`).")
            L.append("- The `.nsys-rep` is uploaded as an artifact — open it in Nsight Systems, or "
                     "check which report names this nsys exposes with `nsys stats --help-reports`.\n")
        else:
            for warning in n.get("warnings", []):
                L.append(f"- ⚠️ nsys stats warning: `{warning}` (nsys `{n.get('nsys_version','?')}`).")
            if n.get("warnings"):
                L.append("")
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
        reg = doc.get("regression", {})
        thr = reg.get("threshold_pct", 0.0)
        verdict = "⚠️ REGRESSION" if reg.get("regressed") else "✅ within threshold"
        L.append("## Delta vs baseline\n")
        L.append(f"- baseline commit: `{reg.get('baseline_commit','?')}`  "
                 f"threshold: `±{thr*100:.0f}%`  verdict: **{verdict}**\n")
        L.append("| metric | baseline | now | Δ | Δ% | |")
        L.append("|---|---:|---:|---:|---:|:--|")
        for k, r in doc["baseline_delta"].items():
            if r.get("below_detection_threshold"):
                mark = "⚠️ noisy / overlap"
            else:
                mark = "⚠️" if r["regressed"] else ""
            L.append(f"| {k} | {r['baseline']:.2f} ± {r.get('baseline_stddev', 0.0):.2f} | "
                     f"{r['now']:.2f} ± {r.get('now_stddev', 0.0):.2f} | {r['delta']:+.2f} | "
                     f"{r['pct']*100:+.1f}% | {mark} |")
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


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else 0.0


def stddev(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def rel_stddev(mean_value: float, stddev_value: float) -> float:
    return abs(stddev_value / mean_value) if mean_value else 0.0


def pooled_within_stddev(runs, key):
    """Run-to-run (within-prompt) stddev, pooled across prompts.

    The headline ± must reflect measurement *noise* — how much a single prompt's
    number wobbles between identical reps — NOT how much different prompts differ
    from each other. Pooling every raw sample across prompts would fold the
    (large, real, repeatable) prompt-to-prompt spread into the stddev and badly
    overstate the noise. We instead pool only the per-prompt deviations:

        sqrt( Σ_prompt Σ_rep (x - prompt_mean)^2 / Σ_prompt (n_rep - 1) )

    i.e. the standard pooled within-group standard deviation. Prompt-to-prompt
    spread stays visible via the min–max line and the per-prompt table.
    """
    num, den = 0.0, 0
    for r in runs:
        s = [x for x in r.get("samples", {}).get(key, []) if x is not None]
        if len(s) > 1:
            m = statistics.mean(s)
            num += sum((x - m) ** 2 for x in s)
            den += len(s) - 1
    return (num / den) ** 0.5 if den > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--df-bin", default=os.environ.get("DFLASH_BIN", "build/test_dflash"))
    ap.add_argument("--ar-bin", default=os.environ.get("DFLASH_BIN_AR", "build/test_generate"))
    ap.add_argument("--tokenizer", default=os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.6-27B"))
    ap.add_argument("--n-gen", type=int, default=128,
                    help="requested generated tokens per benchmark prompt")
    ap.add_argument("--budget", type=int, default=22)
    ap.add_argument("--reps", type=int, default=5,
                    help="timing repetitions per prompt; use 3-5+ to expose run-to-run jitter")
    ap.add_argument("--noise-rsd-pct", type=float, default=0.05,
                    help="report-only noisy-run threshold as relative stddev (0.05 = 5%%)")
    ap.add_argument("--nsys", action="store_true")
    ap.add_argument("--check-lossless", action="store_true")
    ap.add_argument("--prompts", default=None, help="optional .jsonl of {name,text}")
    ap.add_argument("--baseline", default=None, help="optional baseline profile.json for delta")
    ap.add_argument("--regress-pct", type=float, default=0.10,
                    help="fractional move that counts as a regression vs --baseline "
                         "(0.10 = 10%%; wide enough to absorb run-to-run GPU variance)")
    ap.add_argument("--out-json", default="profile.json")
    ap.add_argument("--out-md", default="profile.md")
    cfg = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer, trust_remote_code=True)

    prompts = DEFAULT_PROMPTS
    if cfg.prompts:
        prompts = [json.loads(l) for l in Path(cfg.prompts).read_text().splitlines() if l.strip()]

    tmp = Path(tempfile.mkdtemp(prefix="lucebox_profile_"))  # unique per run (no concurrent-run collisions)
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)  # clean up on exit (success or crash) -> no /tmp growth
    runs = []
    for p in prompts:
        pbin = tmp / f"{p['name']}.bin"
        plen = tokenize(p["text"], tok, pbin)

        # clean timing passes (mean/stddev over reps)
        reps = []
        for _ in range(cfg.reps):
            txt = run(dflash_cmd(cfg, pbin, tmp / "df.bin", cfg.n_gen))
            reps.append(parse_dflash(txt))
        agg = dict(reps[-1])  # keep phases from last
        decode_samples = [r.get("decode_tok_s") for r in reps if r.get("decode_tok_s") is not None]
        al_samples = [r.get("al") for r in reps if r.get("al") is not None]
        mspt_samples = [1000.0 / x for x in decode_samples if x]
        ttft_samples = [r.get("ttft_est_ms") for r in reps if r.get("ttft_est_ms") is not None]
        agg["samples"] = {
            "decode_tok_s": decode_samples,
            "ms_per_token": mspt_samples,
            "al": al_samples,
            "ttft_est_ms": ttft_samples,
        }
        agg["decode_tok_s"] = mean(decode_samples)
        agg["decode_tok_s_stddev"] = stddev(decode_samples)
        agg["ms_per_token"] = mean(mspt_samples)
        agg["ms_per_token_stddev"] = stddev(mspt_samples)
        agg["al"] = mean(al_samples)
        agg["al_stddev"] = stddev(al_samples)
        agg["ttft_est_ms"] = mean(ttft_samples)
        agg["ttft_est_ms_stddev"] = stddev(ttft_samples)
        agg["prompt"] = p["name"]; agg["prompt_len"] = plen

        # AR baseline (single pass — it is deterministic)
        ar_txt = run(ar_cmd(cfg, pbin, tmp / "ar.bin", cfg.n_gen))
        agg.update(parse_ar(ar_txt))

        # losslessness: greedy spec-decode vs greedy AR, with a second identical
        # AR pass as a determinism control. ar.bin (the baseline AR run above) is
        # reused as that control, so the gate costs one extra df + ar pass, same
        # as before — no extra GPU time for the control.
        if cfg.check_lossless:
            run(dflash_cmd(cfg, pbin, tmp / "df_ll.bin", cfg.n_gen))
            run(ar_cmd(cfg, pbin, tmp / "ar_ll.bin", cfg.n_gen))
            agg["lossless"] = check_lossless(plen, tmp / "ar_ll.bin", tmp / "df_ll.bin",
                                             ar_ctrl_bin=tmp / "ar.bin")

        runs.append(agg)

    # nsys on the first prompt only (one short profiled pass)
    nsys = None
    if cfg.nsys:
        if not have_nsys():
            sys.stderr.write("[profile] nsys not found on PATH — skipping kernel layer.\n")
        else:
            pbin = tmp / f"{prompts[0]['name']}.bin"
            rep = tmp / "profile.nsys-rep"
            nbin = tmp / "df_nsys.bin"
            if nsys_profile(dflash_cmd(cfg, pbin, nbin, cfg.n_gen), rep):
                # normalize per-token metrics by ACTUAL generated tokens, not requested
                # n_gen (generation can stop early at EOS), then summarize.
                actual_gen = max(1, len(read_i32(nbin)) - len(read_i32(pbin)))
                nsys = summarize_nsys(rep, actual_gen)
                # copy the trace next to the JSON report so CI can upload a deterministic path
                try:
                    shutil.copy(rep, Path(cfg.out_json).with_suffix(".nsys-rep"))
                except Exception:
                    pass

    # ---- aggregate summary across prompts ----
    def col(k): return [r[k] for r in runs if k in r]
    # Headline ± is the run-to-run (within-prompt) stddev, pooled across prompts.
    # Prompt-to-prompt spread is reported separately (min–max + per-prompt table)
    # so it never masquerades as measurement noise. See pooled_within_stddev().
    decode_mean = statistics.mean(col("decode_tok_s"))
    decode_sd = pooled_within_stddev(runs, "decode_tok_s")
    mspt_mean = statistics.mean(col("ms_per_token"))
    mspt_sd = pooled_within_stddev(runs, "ms_per_token")
    al_mean = statistics.mean(col("al"))
    al_sd = pooled_within_stddev(runs, "al")
    ttft_mean = statistics.mean(col("ttft_est_ms")) if col("ttft_est_ms") else 0.0
    ttft_sd = pooled_within_stddev(runs, "ttft_est_ms")
    summary = {
        "decode_tok_s_mean": decode_mean,
        "decode_tok_s_min": min(col("decode_tok_s")),
        "decode_tok_s_max": max(col("decode_tok_s")),
        "decode_tok_s_stddev": decode_sd,
        "decode_tok_s_rsd": rel_stddev(decode_mean, decode_sd),
        "ms_per_token_mean": mspt_mean,
        "ms_per_token_stddev": mspt_sd,
        "ms_per_token_rsd": rel_stddev(mspt_mean, mspt_sd),
        "al_mean": al_mean,
        "al_stddev": al_sd,
        "al_rsd": rel_stddev(al_mean, al_sd),
        "accept_pct_mean": statistics.mean(col("accept_pct")) if col("accept_pct") else 0.0,
        "prefill_ms_mean": statistics.mean([x * 1000 for x in col("prefill_s")]) if col("prefill_s") else 0.0,
        "ttft_est_ms_mean": ttft_mean,
        "ttft_est_ms_stddev": ttft_sd,
        "ttft_est_ms_rsd": rel_stddev(ttft_mean, ttft_sd),
        "phase_sum_ms_mean": statistics.mean(col("phase_sum_ms")) if col("phase_sum_ms") else 0.0,
    }
    if col("ar_tok_s"):
        summary["ar_tok_s_mean"] = statistics.mean(col("ar_tok_s"))
        summary["speedup"] = summary["decode_tok_s_mean"] / summary["ar_tok_s_mean"]
    # phase means
    pkeys = set().union(*[set(r.get("phases", {})) for r in runs]) if runs else set()
    summary["phases_mean"] = {k: statistics.mean([r["phases"][k] for r in runs if k in r.get("phases", {})])
                              for k in pkeys}
    per_prompt_rsd = {}
    for k in ("decode_tok_s", "ms_per_token", "al", "ttft_est_ms"):
        values = []
        for r in runs:
            values.append(rel_stddev(r.get(k, 0.0), r.get(f"{k}_stddev", 0.0)))
        per_prompt_rsd[k] = max(values) if values else 0.0
    noisy_metrics = [k for k, rsd in per_prompt_rsd.items() if rsd > cfg.noise_rsd_pct]
    summary["noise"] = {
        "threshold_rsd": cfg.noise_rsd_pct,
        "noisy": bool(noisy_metrics),
        "metrics": noisy_metrics,
        "per_prompt_max_rsd": per_prompt_rsd,
        "status": "noisy" if noisy_metrics else "stable",
    }
    if nsys:
        summary["nsys"] = nsys

    doc = {
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": git_commit(),
        "config": {**vars(cfg), **gpu_info()},
        "runs": runs,
        "summary": summary,
    }
    if cfg.check_lossless:
        ll_runs = [(r["prompt"], r["lossless"]) for r in runs if "lossless" in r]
        if ll_runs:
            # A prompt is a REAL losslessness failure only if AR is self-deterministic
            # (control passed) AND spec-decode still diverges from it. If AR already
            # disagrees with itself, the prompt is intrinsically nondeterministic on
            # this engine -> inconclusive, not a spec-decode bug.
            passed, real_fail, inconclusive = [], [], []
            for name, ll in ll_runs:
                if ll["lossless"]:
                    passed.append(name)
                elif ll.get("ar_deterministic", True):
                    real_fail.append((name, ll))
                else:
                    inconclusive.append(name)
            doc["lossless"] = {
                "lossless": not real_fail,                    # FAIL only on a REAL divergence
                "prompts_checked": len(ll_runs),
                "prompts_passed": passed,
                "prompts_failed": [name for name, _ in real_fail],
                "prompts_inconclusive": inconclusive,
                "first_divergence": real_fail[0][1]["first_divergence"] if real_fail else None,
                "per_prompt": {name: ll for name, ll in ll_runs},
            }

    # baseline delta + regression verdict
    if cfg.baseline and Path(cfg.baseline).exists():
        base_doc = json.loads(Path(cfg.baseline).read_text())
        rows = regression_delta(base_doc["summary"], summary, cfg.regress_pct)
        if rows:
            doc["baseline_delta"] = rows
            regressed = [k for k, r in rows.items() if r["regressed"]]
            doc["regression"] = {
                "threshold_pct": cfg.regress_pct,
                "regressed": bool(regressed),
                "metrics": regressed,
                "baseline_commit": base_doc.get("commit", "?"),
            }

    Path(cfg.out_json).write_text(json.dumps(doc, indent=2))
    Path(cfg.out_md).write_text(render_md(doc))
    print(render_md(doc))
    print(f"\n[profile] wrote {cfg.out_json} and {cfg.out_md}")


if __name__ == "__main__":
    main()
