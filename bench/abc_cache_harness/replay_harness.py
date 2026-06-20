#!/usr/bin/env python3
"""Multi-turn agentic composition harness with prefix-cache awareness.

Usage:
    python3 replay_harness.py --arm A_baseline --trace traces/goldgate_fix.jsonl
    python3 replay_harness.py --arm A_baseline --trace traces/goldgate_fix.jsonl --smoke
    python3 replay_harness.py --arm C_always   --trace traces/goldgate_fix.jsonl
    python3 replay_harness.py --arm C_threshold --trace traces/goldgate_fix.jsonl

Smoke gate (--smoke): runs first 3 turns only, N=1.
Full run: all turns, N=3.

Arms:
  A_baseline   : dFlash decode + prefix cache, NO pFlash, NO stochastic
  A_stochastic : dFlash decode + prefix cache + DFLASH_STOCHASTIC, NO pFlash
  C_always     : pFlash always + dFlash decode  (ATTN_PRIMARY, stochastic)
  C_threshold  : pFlash auto/32K threshold + dFlash decode

Port: 19099 (NEVER 18099)
Max-ctx: 65536
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, mean
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCH_DIR = Path(__file__).parent
REPO_DIR = BENCH_DIR.parent.parent

SERVER_BIN = REPO_DIR / "server/build/dflash_server"
TGT = Path("/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf")
DD  = Path("/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-bf16-reconverted.gguf")
PD  = Path("/home/peppi/models/Qwen3-0.6B-BF16.gguf")
PD_Q8 = Path("/home/peppi/models/Qwen3-0.6B-Q8_0.gguf")  # Q8 compress drafter (no-park arm)
TMPL = Path("/home/peppi/models/qwen3-coder-chat-template.jinja")

HOST = "127.0.0.1"
PORT = 19099
MAX_CTX = 65536
TEMP = 0.7

FORBIDDEN_PORT = 18099  # user's live server — never touch

# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------

ARMS: dict[str, dict] = {
    "A_baseline": {
        "description": "dFlash decode + prefix cache; NO pFlash",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
        ],
        "env": {},
    },
    "A_stochastic": {
        "description": "dFlash decode + prefix cache + DFLASH_STOCHASTIC; NO pFlash",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
        ],
        "env": {
            "DFLASH_STOCHASTIC": "1",
        },
    },
    "C_always": {
        "description": "pFlash always (ATTN_PRIMARY+stochastic) + dFlash decode",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.5",
        ],
        "env": {
            "PFLASH_ATTN_PRIMARY": "1",
            "DFLASH_STOCHASTIC": "1",
        },
    },
    "C_threshold": {
        "description": "pFlash auto/32K threshold + dFlash decode",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "auto",
            "--prefill-threshold", "32000",
            "--prefill-keep-ratio", "0.5",
        ],
        "env": {
            "PFLASH_ATTN_PRIMARY": "1",
            "DFLASH_STOCHASTIC": "1",
        },
    },
    # ── FlowKV validation arms ──────────────────────────────────────────
    # A_baseline run with the new FlowKV binary but no pFlash — proves no-op.
    "FLOWKV_OFF": {
        "description": "FlowKV binary, NO pFlash, PFLASH_FREEZE_HISTORY unset (no-op gate)",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
        ],
        "env": {},
    },
    # pFlash-always + freeze-history=1, hot_window=2.
    # Aged turns (indices 1..N-hot_window-1) compressed once + cached.
    # System (turn 0) + hot tail (last 2 turns) verbatim.
    "FLOWKV_ON": {
        "description": "pFlash-always + PFLASH_FREEZE_HISTORY=1 hot_window=2 per-msg threshold=1000 (FlowKV active)",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
        ],
        "env": {
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
        },
    },
    # ── #364 disk-prefix-cache composition arms ─────────────────────────
    # DISK364: #364 scoped disk cache ON, NO pFlash, NO freeze.
    #   The #364-alone baseline (momus-required). dFlash decode draft only.
    #   Fresh, empty --kv-cache-dir; rm -rf'd before the run, persists across 7 turns.
    "DISK364": {
        "description": "#364 scoped disk-prefix-cache (auto), dFlash decode; NO pFlash, NO freeze",
        "kv_cache_dir": "/tmp/compose_kv_disk364",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--disk-prefix-cache", "auto",
            "--kv-cache-dir", "/tmp/compose_kv_disk364",
        ],
        "env": {},
    },
    # DISK364_KV8: DISK364 with q8_0 KV. llama-bench 2026-06-12: tq3_0 KV costs 2x
    #   decode at 63K (14.4 vs 29 tok/s); q8_0 matches f16. Fit: q8 ~= +1.7GB over tq3.
    "DISK364_KV8": {
        "description": "DISK364 + q8_0 KV cache (tq3_0 decode-tax escape test)",
        "kv_cache_dir": "/tmp/compose_kv_disk364_kv8",
        "kv_cache_types": ("q8_0", "q8_0"),
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--disk-prefix-cache", "auto",
            "--kv-cache-dir", "/tmp/compose_kv_disk364_kv8",
        ],
        "env": {},
    },
    # COMPOSE_FLOWKV: unified-gate validation arm. pFlash ALWAYS so should_compress=true,
    #   continuations route to FlowKV-freeze, turn-1 verbatim. #364 disk cache w/ compress.
    #   ee7 cheap drafter (EARLY_EXIT_N=7 + SCORE_LAYERS=7). Fresh tmpdir per run.
    "COMPOSE_FLOWKV": {
        "description": "unified-gate: pFlash always + FlowKV-freeze + #364 disk(compress) + ee7",
        "kv_cache_dir": "/tmp/compose_kv_flowkv_unified",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
            "--disk-prefix-cache", "auto",
            "--disk-prefix-cache-compress",
            "--kv-cache-dir", "/tmp/compose_kv_flowkv_unified",
        ],
        "env": {
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
            "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
            "PFLASH_DRAFTER_SCORE_LAYERS": "7",
        },
    },
    # COMPOSE_FLOWKV_Q8NOPARK: COMPOSE_FLOWKV with the Q8_0 0.6B compress drafter
    #   AND --prefill-skip-park (target 27B stays RESIDENT during compress; no
    #   park/unpark of the 27B). Drafter also persists via drafter_loaded_.
    #   GGML_CUDA_NO_VMM=1 (set by launch_server) is the 24GB skip-park VMM workaround.
    #   Tests whether eliminating park kills the turn-4 BF16+park prefill spike (374.6s).
    "COMPOSE_FLOWKV_Q8NOPARK": {
        "description": "COMPOSE_FLOWKV + Q8_0 drafter + --prefill-skip-park (27B resident, no park)",
        "kv_cache_dir": "/tmp/compose_kv_flowkv_q8nopark",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD_Q8),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
            "--prefill-skip-park",
            "--disk-prefix-cache", "auto",
            "--disk-prefix-cache-compress",
            "--kv-cache-dir", "/tmp/compose_kv_flowkv_q8nopark",
        ],
        "env": {
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
            "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
            "PFLASH_DRAFTER_SCORE_LAYERS": "7",
        },
    },
    # COMPOSE_FLOWKV_NOPARK: COMPOSE_FLOWKV (BF16 drafter that ACTUALLY loads) +
    #   --prefill-skip-park. Isolates the park variable: compression still happens
    #   (BF16 loads fine), only park/unpark of the 27B is eliminated. This is the
    #   valid apples-to-apples vs the BF16+park COMPOSE_FLOWKV baseline (turn-4 374.6s).
    "COMPOSE_FLOWKV_NOPARK": {
        "description": "COMPOSE_FLOWKV + BF16 drafter + --prefill-skip-park (27B resident, compression ON)",
        "kv_cache_dir": "/tmp/compose_kv_flowkv_nopark",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
            "--prefill-skip-park",
            "--disk-prefix-cache", "auto",
            "--disk-prefix-cache-compress",
            "--kv-cache-dir", "/tmp/compose_kv_flowkv_nopark",
        ],
        "env": {
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
            "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
            "PFLASH_DRAFTER_SCORE_LAYERS": "7",
        },
    },
    # COMPOSE_KV8: COMPOSE_FLOWKV_NOPARK (PR372 shipped config, auto residency)
    #   with q8_0 KV. Pairs with DISK364_KV8 for the q8-config A/B of the PR372 win.
    "COMPOSE_KV8": {
        "description": "COMPOSE_FLOWKV_NOPARK + q8_0 KV (PR372 win re-check on q8 config)",
        "kv_cache_dir": "/tmp/compose_kv_flowkv_kv8",
        "kv_cache_types": ("q8_0", "q8_0"),
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
            "--prefill-skip-park",
            "--disk-prefix-cache", "auto",
            "--disk-prefix-cache-compress",
            "--kv-cache-dir", "/tmp/compose_kv_flowkv_kv8",
        ],
        "env": {
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
            "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
            "PFLASH_DRAFTER_SCORE_LAYERS": "7",
        },
    },
    # COMPOSE_FLOWKV_NOPARK_RS: NOPARK + --draft-residency request-scoped.
    #   Discriminator for the turn-4 prefill-rate collapse: frees the pflash
    #   drafter (~1.5-2GB) after compress scoring, before target prefill.
    "COMPOSE_FLOWKV_NOPARK_RS": {
        "description": "COMPOSE_FLOWKV_NOPARK + --draft-residency request-scoped (free drafter before target prefill)",
        "kv_cache_dir": "/tmp/compose_kv_flowkv_nopark_rs",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
            "--prefill-skip-park",
            "--draft-residency", "request-scoped",
            "--disk-prefix-cache", "auto",
            "--disk-prefix-cache-compress",
            "--kv-cache-dir", "/tmp/compose_kv_flowkv_nopark_rs",
        ],
        "env": {
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
            "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
            "PFLASH_DRAFTER_SCORE_LAYERS": "7",
        },
    },
    # DISK364_NODRAFT_F16: decode-tax discriminator. No draft models, f16 KV.
    #   If decode @63K ~= mainline llama-bench (29 tok/s) -> tax is tq3_0+residency.
    #   If still ~12 -> our serving loop (per-token graph rebuild) is the tax.
    "DISK364_NODRAFT_F16": {
        "description": "#364 disk cache, NO draft, f16 KV (decode-tax discriminator)",
        "kv_cache_dir": "/tmp/disk364_nodraft_f16",
        "kv_cache_types": ("f16", "f16"),
        "extra_args": [
            "--disk-prefix-cache", "auto",
            "--kv-cache-dir", "/tmp/disk364_nodraft_f16",
        ],
        "env": {},
    },
    # COMPOSE_FLOWKV_NOPARK_KV8: NOPARK with q8_0 KV instead of tq3_0.
    #   One-cell sweep: trade freed VRAM (drafter release) for cheaper attention
    #   dequant on decode reads.
    "COMPOSE_FLOWKV_NOPARK_KV8": {
        "description": "COMPOSE_FLOWKV_NOPARK + q8_0 KV cache (dequant-cost sweep)",
        "kv_cache_dir": "/tmp/compose_kv_flowkv_nopark_kv8",
        "kv_cache_types": ("q8_0", "q8_0"),
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
            "--prefill-skip-park",
            "--disk-prefix-cache", "auto",
            "--disk-prefix-cache-compress",
            "--kv-cache-dir", "/tmp/compose_kv_flowkv_nopark_kv8",
        ],
        "env": {
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
            "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
            "PFLASH_DRAFTER_SCORE_LAYERS": "7",
        },
    },
    # COMPOSE: FlowKV (pFlash auto-gated) + #364 disk cache w/ compression.
    #   pFlash fires only on turns >= 65K tokens (same gate as standalone pFlash).
    #   FLOWKV_ON pFlash flags + dFlash decode + disk-prefix-cache-compress.
    #   Fresh, empty --kv-cache-dir; rm -rf'd before the run, persists across 7 turns.
    "COMPOSE": {
        "description": "FlowKV (pFlash auto >=65K + freeze) + #364 disk cache (compress); dFlash decode",
        "kv_cache_dir": "/tmp/compose_kv_compose",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "auto",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "65000",
            "--disk-prefix-cache", "auto",
            "--disk-prefix-cache-compress",
            "--kv-cache-dir", "/tmp/compose_kv_compose",
        ],
        "env": {
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
        },
    },
    # COMPOSE_KVQK: arm B of the 372xKVFlash composition probe.
    #   COMPOSE_FLOWKV_NOPARK (PR372 shipped config) + KVFlash bounded residency
    #   pool=max(4096, max_ctx/10)=6553 with the target-QK scorer (env-only policy).
    "COMPOSE_KVQK": {
        "description": "PR372 shipped (FlowKV+disk364+NOPARK) + --kvflash 6553 + QK policy",
        "kv_cache_dir": "/tmp/compose_kv_kvqk",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
            "--prefill-skip-park",
            "--disk-prefix-cache", "auto",
            "--disk-prefix-cache-compress",
            "--kv-cache-dir", "/tmp/compose_kv_kvqk",
            "--kvflash", "6553",
        ],
        "env": {
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
            "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
            "PFLASH_DRAFTER_SCORE_LAYERS": "7",
            "DFLASH_KVFLASH_POLICY": "qk",
        },
    },
    # KVQK_ONLY: follow-up cell — KVFlash+QK WITHOUT FlowKV/pFlash (lossless lane).
    #   Disk364-style baseline + bounded residency; isolates lossy-vs-lossless retrieval.
    "KVQK_ONLY": {
        "description": "#364 disk cache + dFlash decode + --kvflash 6553 QK; NO pFlash/FlowKV",
        "kv_cache_dir": "/tmp/compose_kv_kvqk_only",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--disk-prefix-cache", "auto",
            "--kv-cache-dir", "/tmp/compose_kv_kvqk_only",
            "--kvflash", "6553",
        ],
        "env": {
            "DFLASH_KVFLASH_POLICY": "qk",
        },
    },
}

# ---------------------------------------------------------------------------
# Log parsing regexes
# These match the real dflash_server log format confirmed from arm1_baseline logs
# ---------------------------------------------------------------------------

# [server] chat CACHE ... restore=true|false slot=N prefix_len=N effective_prompt=N ...
CACHE_RE = re.compile(
    r'\[server\] chat CACHE\s+\S+\s+'
    r'restore=(\w+)\s+slot=(-?\d+)\s+prefix_len=(\d+)\s+effective_prompt=(\d+)'
    r'(?:\s+pflash=(\w+))?'
)

# [server] chat DONE ... in=N effective_in=N out=N ... prefix_len=N prefill=Xs decode=Xs(Ytok/s) ...
DONE_RE = re.compile(
    r'\[server\] chat DONE\s+\S+\s+ok=\w+\s+'
    r'in=(\d+)\s+effective_in=(\d+)\s+out=(\d+)\s+'
    r'[\d.]+s\s+[\d.]+\s+tok/s\s+finish=(\S+)\s+'
    r'restore=(\w+)\s+slot=(-?\d+)\s+prefix_len=(\d+)\s+'
    r'prefill=([\d.]+)s\s+decode=([\d.]+)s\(([\d.]+)tok/s\)'
)

# [spec-decode] tokens=N time=Xs speed=Ytok/s steps=N accepted=A/T (P%) avg_commit=C
SPEC_RE = re.compile(
    r'\[spec-decode\] tokens=(\d+) time=([\d.]+) s speed=([\d.]+) tok/s '
    r'steps=\d+ accepted=\d+/\d+ \(([\d.]+)%\) avg_commit=([\d.]+)'
)

# [ar-decode] tokens=N time=Xs speed=Ytok/s
AR_RE = re.compile(
    r'\[ar-decode\] tokens=(\d+) time=([\d.]+) s speed=([\d.]+) tok/s'
)

# [pflash] N -> M -> K tokens (P% kept)
PFLASH_KEPT_RE = re.compile(
    r'\[pflash\] (\d+) -> \d+ -> (\d+) tokens \(([\d.]+)% kept\)'
)

# [pflash] query survival: A/B (P%)
SURVIVAL_RE = re.compile(
    r'\[pflash\] query survival: (\d+)/(\d+) \(([\d.]+)%\)'
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(repo: Path) -> dict:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return {"branch": branch, "commit": commit}
    except Exception:
        return {"branch": "unknown", "commit": "unknown"}


def nvidia_smi_vram() -> tuple[Optional[int], Optional[int]]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip()
        parts = out.split(",")
        return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        return None, None


def wait_for_server(port: int, timeout: int = 360) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://{HOST}:{port}/health", timeout=3
            ) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def launch_server(arm_name: str, arm_cfg: dict, log_path: Path) -> tuple:
    """Start server for given arm. Returns (proc, log_file_handle)."""
    env = dict(os.environ)
    env["GGML_CUDA_NO_VMM"] = "1"
    # DFlash decode optimizations (from dFlash drafter integration session):
    # - block_size=16 (model card sweet spot for Qwen3.6 DFlash)
    # - feature ring cap=16384 (default 4096 collapses acceptance at the ring boundary)
    env["DFLASH_DRAFT_BLOCK_SIZE"] = "16"
    env["DFLASH_FEAT_RING_CAP"] = "16384"
    env.update(arm_cfg["env"])

    ctk, ctv = arm_cfg.get("kv_cache_types", ("tq3_0", "tq3_0"))
    cmd = [
        str(SERVER_BIN),
        str(TGT),
        "--host", HOST,
        "--port", str(PORT),
        "--max-ctx", str(MAX_CTX),
        "--cache-type-k", ctk,
        "--cache-type-v", ctv,
        "--chat-template-file", str(TMPL),
        "--model-name", "luce-dflash",
    ] + arm_cfg["extra_args"]

    print(f"[server] Launching {arm_name}: {' '.join(cmd)}")
    log_f = log_path.open("w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)
    print(f"[server] PID={proc.pid}  log={log_path}")
    return proc, log_f


def kill_server(proc, log_f, wait_s: int = 10) -> None:
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    if log_f:
        log_f.close()


# ---------------------------------------------------------------------------
# Log parsing: extract per-request metrics from server log
# ---------------------------------------------------------------------------

def parse_log_for_request(
    log_text: str,
    cache_offset: int,
    done_offset: int,
    spec_offset: int,
    ar_offset: int,
    pflash_offset: int,
    survival_offset: int,
) -> tuple[dict, int, int, int, int, int, int]:
    """Parse the NEXT request's metrics from log_text starting at given offsets.

    Returns (metrics_dict, new_cache_offset, new_done_offset, new_spec_offset,
             new_ar_offset, new_pflash_offset, new_survival_offset).
    """
    all_cache    = CACHE_RE.findall(log_text)
    all_done     = DONE_RE.findall(log_text)
    all_spec     = SPEC_RE.findall(log_text)
    all_ar       = AR_RE.findall(log_text)
    all_pflash   = PFLASH_KEPT_RE.findall(log_text)
    all_survival = SURVIVAL_RE.findall(log_text)

    new_cache    = all_cache[cache_offset:]
    new_done     = all_done[done_offset:]
    new_spec     = all_spec[spec_offset:]
    new_ar       = all_ar[ar_offset:]
    new_pflash   = all_pflash[pflash_offset:]
    new_survival = all_survival[survival_offset:]

    m: dict = {}

    # --- chat CACHE line ---
    if new_cache:
        c = new_cache[0]
        m["restore"]          = c[0] == "true"
        m["cache_slot"]       = int(c[1])
        m["prefix_len"]       = int(c[2])
        m["effective_prompt"] = int(c[3])
        if c[4]:
            m["pflash_active"] = c[4] == "true"

    # --- chat DONE line ---
    if new_done:
        d = new_done[0]
        m["prompt_tokens"]   = int(d[0])
        m["effective_in"]    = int(d[1])
        m["out_tokens"]      = int(d[2])
        m["finish_reason"]   = d[3]
        m["restore_done"]    = d[4] == "true"
        m["prefix_len_done"] = int(d[6])
        m["prefill_s"]       = float(d[7])
        m["decode_s"]        = float(d[8])
        m["decode_tps"]      = float(d[9])
        # prefix_len from DONE is more reliable; override if we got it
        if "prefix_len" not in m:
            m["prefix_len"] = m["prefix_len_done"]
        # prefill TPS
        pt = m["prompt_tokens"]
        pfs = m["prefill_s"]
        if pt > 0 and pfs > 0:
            m["prefill_tps"] = round(pt / pfs, 1)
        # fresh prefill = effective tokens not served from prefix cache
        eff = m.get("effective_in", m.get("effective_prompt", pt))
        pl  = m.get("prefix_len", 0)
        m["fresh_prefill"] = max(0, eff - pl)
        # cache hit ratio
        if pt > 0:
            m["cache_hit_ratio"] = round(pl / pt, 4)

    # --- spec or ar decode ---
    if new_spec:
        s = new_spec[0]
        m["decode_mode"]  = "spec"
        m["accept_rate"]  = float(s[3])
        m["avg_commit"]   = float(s[4])
        if "decode_tps" not in m:
            m["decode_tps"] = float(s[2])
    elif new_ar:
        a = new_ar[0]
        m["decode_mode"] = "ar"
        if "decode_tps" not in m:
            m["decode_tps"] = float(a[2])

    # --- pflash compression info ---
    if new_pflash:
        p = new_pflash[0]
        m["pflash_in_tokens"]   = int(p[0])
        m["pflash_kept_tokens"] = int(p[1])
        m["pflash_kept_pct"]    = float(p[2])
    if new_survival:
        sv = new_survival[0]
        m["query_survival_pct"] = float(sv[2])

    return (
        m,
        cache_offset    + (1 if new_cache    else 0),
        done_offset     + (1 if new_done     else 0),
        spec_offset     + (1 if new_spec     else 0),
        ar_offset       + (1 if new_ar       else 0),
        pflash_offset   + (1 if new_pflash   else 0),
        survival_offset + (1 if new_survival else 0),
    )


# ---------------------------------------------------------------------------
# Check tool_use in response
# ---------------------------------------------------------------------------

def check_tool_call(body: dict) -> tuple[bool, str]:
    """Return (valid, detail) — did the response contain a well-formed tool_use?"""
    try:
        content = body.get("content", [])
        if isinstance(content, str):
            # Shouldn't happen with Anthropic API but guard anyway
            return False, "content is string not list"
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name", "")
                inp  = block.get("input", {})
                if name and isinstance(inp, dict):
                    return True, f"tool={name}"
                return False, f"malformed tool_use block: {block}"
        # Also check OpenAI-compat format (tool_calls)
        choices = body.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            tcs = msg.get("tool_calls", [])
            if tcs:
                return True, f"tool={tcs[0].get('function',{}).get('name','?')}"
    except Exception as e:
        return False, f"exception: {e}"
    return False, "no tool_use block"


# ---------------------------------------------------------------------------
# Send one request
# ---------------------------------------------------------------------------

def send_request(req_body: dict, port: int) -> tuple[dict, float, Optional[str]]:
    """POST to /v1/messages. Returns (response_body, wall_s, error_str)."""
    payload = json.dumps(req_body, ensure_ascii=False).encode("utf-8")
    http_req = urllib.request.Request(
        f"http://{HOST}:{port}/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(http_req, timeout=900) as r:
            body = json.loads(r.read())
        return body, time.time() - t0, None
    except Exception as ex:
        return {}, time.time() - t0, str(ex)


# ---------------------------------------------------------------------------
# Run one repeat of the trace
# ---------------------------------------------------------------------------

def run_trace_repeat(
    turns: list[dict],
    port: int,
    log_path: Path,
    arm_name: str,
    smoke: bool = False,
    repeat_idx: int = 0,
) -> list[dict]:
    """Run all turns in the trace; parse server log per turn.

    Returns list of per-turn metric dicts.
    """
    turns_to_run = turns[:3] if smoke else turns

    # Wait for log file to exist and server to be settled
    deadline = time.time() + 300
    while not log_path.exists() and time.time() < deadline:
        time.sleep(1)

    cache_off = done_off = spec_off = ar_off = pflash_off = survival_off = 0

    results = []
    for turn_idx, req_body in enumerate(turns_to_run):
        turn_num = turn_idx + 1
        print(f"  [{arm_name} repeat={repeat_idx+1}] turn {turn_num}/{len(turns_to_run)} "
              f"(~{len(json.dumps(req_body))//3:,} est_tok) ...", end=" ", flush=True)

        t_send = time.time()
        resp_body, wall_s, err = send_request(req_body, port)
        t_done = time.time()

        if err:
            print(f"ERROR: {err}")
            results.append({
                "turn": turn_num,
                "repeat": repeat_idx + 1,
                "error": err,
                "wall_s": wall_s,
            })
            continue

        # Give the server a moment to flush the log lines
        time.sleep(0.3)

        # Parse log
        try:
            log_text = log_path.read_text(errors="replace")
        except Exception:
            log_text = ""

        metrics, cache_off, done_off, spec_off, ar_off, pflash_off, survival_off = \
            parse_log_for_request(
                log_text, cache_off, done_off, spec_off, ar_off, pflash_off, survival_off
            )

        # Check tool_call validity
        tool_valid, tool_detail = check_tool_call(resp_body)

        rec = {
            "turn": turn_num,
            "repeat": repeat_idx + 1,
            "wall_s": round(wall_s, 2),
            **metrics,
            "tool_call_valid": tool_valid,
            "tool_detail": tool_detail,
        }
        results.append(rec)

        # Print one-liner
        pt     = metrics.get("prompt_tokens", "?")
        eff    = metrics.get("effective_in", "?")
        pl     = metrics.get("prefix_len", "?")
        fp     = metrics.get("fresh_prefill", "?")
        hr     = metrics.get("cache_hit_ratio")
        pfs    = metrics.get("prefill_s")
        dec    = metrics.get("decode_tps")
        mode   = metrics.get("decode_mode", "?")
        acc    = metrics.get("accept_rate")
        pflkpt = metrics.get("pflash_kept_pct")

        hr_str  = f"{hr:.1%}" if hr is not None else "?"
        pfs_str = f"{pfs:.2f}s" if pfs else "?"
        dec_str = f"{dec:.1f}" if dec else "?"
        acc_str = f"{acc:.1f}%" if acc is not None else "AR"
        pfl_str = f"pflash={pflkpt:.1f}%kept" if pflkpt is not None else ""

        print(
            f"wall={wall_s:.1f}s  pt={pt}  eff={eff}  "
            f"prefix_len={pl} ({hr_str} hit)  "
            f"fresh={fp}  prefill={pfs_str}  "
            f"decode={dec_str}tok/s[{mode}] {acc_str}  "
            f"{pfl_str}  tool={tool_valid}"
        )

    return results


# ---------------------------------------------------------------------------
# Aggregate results across repeats
# ---------------------------------------------------------------------------

def _vals(records: list[dict], key: str) -> list:
    return [r[key] for r in records if key in r and r.get("error") is None]


def aggregate_turns(all_records: list[dict], n_turns: int) -> list[dict]:
    """Return per-turn median metrics aggregated across repeats."""
    agg = []
    for t in range(1, n_turns + 1):
        recs = [r for r in all_records if r.get("turn") == t and "error" not in r]
        if not recs:
            agg.append({"turn": t, "n_repeats": 0})
            continue

        def med(key):
            vs = _vals(recs, key)
            return round(median(vs), 3) if vs else None

        # disk_hit_rate: fraction of repeats where the disk prefix cache was hit.
        # Only meaningful in --restart-per-turn mode (always None in single-session).
        disk_hits = [r for r in recs if "disk_hit" in r]
        disk_hit_rate = (
            round(sum(1 for r in disk_hits if r["disk_hit"]) / len(disk_hits), 4)
            if disk_hits else None
        )

        agg.append({
            "turn": t,
            "n_repeats": len(recs),
            "wall_s": med("wall_s"),
            "prompt_tokens": med("prompt_tokens"),
            "effective_in": med("effective_in"),
            "prefix_len": med("prefix_len"),
            "fresh_prefill": med("fresh_prefill"),
            "cache_hit_ratio": med("cache_hit_ratio"),
            "prefill_s": med("prefill_s"),
            "prefill_tps": med("prefill_tps"),
            "decode_tps": med("decode_tps"),
            "decode_mode": recs[0].get("decode_mode"),
            "accept_rate": med("accept_rate"),
            "avg_commit": med("avg_commit"),
            "pflash_kept_pct": med("pflash_kept_pct"),
            "disk_hit_rate": disk_hit_rate,
            "tool_call_valid_rate": sum(1 for r in recs if r.get("tool_call_valid")) / len(recs),
        })
    return agg


def aggregate_arm(per_turn: list[dict]) -> dict:
    """Compute arm-level aggregates from per-turn medians."""
    valid = [t for t in per_turn if t.get("n_repeats", 0) > 0]
    if not valid:
        return {}

    def safe_mean(key):
        vs = [t[key] for t in valid if t.get(key) is not None]
        return round(mean(vs), 3) if vs else None

    def safe_sum(key):
        vs = [t[key] for t in valid if t.get(key) is not None]
        return round(sum(vs), 1) if vs else None

    spec_turns = [t for t in valid if t.get("decode_mode") == "spec"]

    # disk_hit_rate: only non-None when --restart-per-turn was used.
    disk_hit_vals = [t["disk_hit_rate"] for t in valid if t.get("disk_hit_rate") is not None]
    mean_disk_hit_rate = round(mean(disk_hit_vals), 4) if disk_hit_vals else None

    return {
        "total_wall_s": safe_sum("wall_s"),
        "sum_fresh_prefill_tokens": safe_sum("fresh_prefill"),
        "mean_cache_hit_ratio": safe_mean("cache_hit_ratio"),
        "mean_prefill_tps": safe_mean("prefill_tps"),
        "mean_decode_tps": safe_mean("decode_tps"),
        "spec_engagement_rate": round(len(spec_turns) / len(valid), 3) if valid else None,
        "mean_accept_rate": safe_mean("accept_rate"),
        "mean_disk_hit_rate": mean_disk_hit_rate,
        "tool_call_valid_rate": safe_mean("tool_call_valid_rate"),
    }


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_report(
    arm_name: str,
    per_turn: list[dict],
    arm_agg: dict,
    provenance: dict,
    out_path: Path,
    smoke: bool,
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append(f"# ABC Cache Harness — {arm_name}")
    lines.append(f"Generated: {ts}")
    lines.append(f"Mode: {'SMOKE (first 3 turns, N=1)' if smoke else 'FULL (all turns, N=3)'}")
    lines.append("")
    lines.append("## Provenance")
    lines.append("```")
    lines.append(json.dumps(provenance, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Per-Turn Cache Trace")
    has_disk_hit = any(t.get("disk_hit_rate") is not None for t in per_turn)
    header = (
        f"{'turn':>4} {'pt':>7} {'eff_in':>7} {'prefix_len':>11} "
        f"{'fresh_pf':>9} {'hr':>6} {'pf_s':>6} {'pf_tps':>7} "
        f"{'dec_tps':>8} {'mode':>5} {'accept':>7} {'pflash%':>8} "
        + (f"{'disk_hit':>9} " if has_disk_hit else "")
        + f"{'tool':>5} {'wall_s':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for t in per_turn:
        if t.get("n_repeats", 0) == 0:
            lines.append(f"  turn {t['turn']}: NO DATA")
            continue
        hr = t.get("cache_hit_ratio")
        acc = t.get("accept_rate")
        pfl = t.get("pflash_kept_pct")
        dhr = t.get("disk_hit_rate")
        tool_rate = t.get("tool_call_valid_rate")
        pfs_val = t.get('prefill_s')
        ptps_val = t.get('prefill_tps')
        dtps_val = t.get('decode_tps')
        pfs_str2  = f"{pfs_val:.2f}" if pfs_val else "?"
        ptps_str2 = f"{ptps_val:.0f}" if ptps_val else "?"
        dtps_str2 = f"{dtps_val:.1f}" if dtps_val else "?"
        dhr_str   = f"{dhr:.1%}" if dhr is not None else "-"
        lines.append(
            f"{t['turn']:>4} "
            f"{str(t.get('prompt_tokens','?')):>7} "
            f"{str(t.get('effective_in','?')):>7} "
            f"{str(t.get('prefix_len','?')):>11} "
            f"{str(t.get('fresh_prefill','?')):>9} "
            f"{f'{hr:.1%}' if hr is not None else '?':>6} "
            f"{pfs_str2:>6} "
            f"{ptps_str2:>7} "
            f"{dtps_str2:>8} "
            f"{str(t.get('decode_mode','?')):>5} "
            f"{f'{acc:.1f}%' if acc is not None else 'AR':>7} "
            f"{f'{pfl:.1f}%' if pfl is not None else '-':>8} "
            + (f"{dhr_str:>9} " if has_disk_hit else "")
            + f"{'Y' if tool_rate and tool_rate > 0 else 'N':>5} "
            f"{str(t.get('wall_s','?')):>7}"
        )
    lines.append("")
    lines.append("## Arm Aggregate")
    lines.append("```")
    lines.append(json.dumps(arm_agg, indent=2))
    lines.append("```")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] Written to {out_path}")


# ---------------------------------------------------------------------------
# Per-turn restart mode
# ---------------------------------------------------------------------------

def run_trace_restart_per_turn(
    turns: list[dict],
    port: int,
    arm_name: str,
    arm_cfg: dict,
    log_dir: Path,
    smoke: bool = False,
    repeat_idx: int = 0,
) -> list[dict]:
    """Restart-per-turn mode: each turn gets a fresh server process.

    Sequence per turn:
      1. launch_server() with arm_cfg (same flags including --kv-cache-dir if present)
      2. wait_for_server()
      3. send_request() for this single turn
      4. parse_log_for_request() with fresh offsets (0,0,0,0,0,0)
      5. kill_server()
      6. annotate metrics with disk_hit = metrics.get("restore", False)

    The --kv-cache-dir is NOT wiped between turns (wiped once at arm start by caller).
    This makes the on-disk cache the ONLY cross-turn reuse path.
    """
    turns_to_run = turns[:3] if smoke else turns
    results = []

    for turn_idx, req_body in enumerate(turns_to_run):
        turn_num = turn_idx + 1
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = log_dir / f"srv_{arm_name}_rep{repeat_idx+1}_t{turn_num:02d}_{ts_str}.log"

        print(f"  [{arm_name} repeat={repeat_idx+1} restart-per-turn] "
              f"turn {turn_num}/{len(turns_to_run)} "
              f"(~{len(json.dumps(req_body))//3:,} est_tok) — launching server...")

        # Step 1: launch fresh server
        proc, log_f = launch_server(arm_name, arm_cfg, log_path)

        try:
            # Step 2: wait for ready
            ok = wait_for_server(port, timeout=360)
            if not ok:
                print(f"  ERROR: server did not come up for turn {turn_num}. "
                      f"Check {log_path}", file=sys.stderr)
                results.append({
                    "turn": turn_num,
                    "repeat": repeat_idx + 1,
                    "error": "server_timeout",
                    "wall_s": 0.0,
                    "disk_hit": False,
                })
                continue

            print(f"  READY — sending request...", end=" ", flush=True)

            # Step 3: send the single request
            resp_body, wall_s, err = send_request(req_body, port)

            if err:
                print(f"ERROR: {err}")
                results.append({
                    "turn": turn_num,
                    "repeat": repeat_idx + 1,
                    "error": err,
                    "wall_s": wall_s,
                    "disk_hit": False,
                })
                continue

            # Give server a moment to flush log lines
            time.sleep(0.3)

            # Step 4: parse fresh log (offsets all 0 — new server, new log file)
            try:
                log_text = log_path.read_text(errors="replace")
            except Exception:
                log_text = ""

            metrics, _, _, _, _, _, _ = parse_log_for_request(
                log_text, 0, 0, 0, 0, 0, 0
            )

            # disk_hit: in restart-per-turn mode, restore=true can ONLY come from the
            # on-disk prefix cache (no in-RAM cache survives a fresh process).
            disk_hit = bool(metrics.get("restore", False))

            # Check tool_call validity
            tool_valid, tool_detail = check_tool_call(resp_body)

            rec = {
                "turn": turn_num,
                "repeat": repeat_idx + 1,
                "wall_s": round(wall_s, 2),
                "disk_hit": disk_hit,
                **metrics,
                "tool_call_valid": tool_valid,
                "tool_detail": tool_detail,
            }
            results.append(rec)

            # Print one-liner
            pt     = metrics.get("prompt_tokens", "?")
            eff    = metrics.get("effective_in", "?")
            pl     = metrics.get("prefix_len", "?")
            fp     = metrics.get("fresh_prefill", "?")
            hr     = metrics.get("cache_hit_ratio")
            pfs    = metrics.get("prefill_s")
            dec    = metrics.get("decode_tps")
            mode   = metrics.get("decode_mode", "?")
            acc    = metrics.get("accept_rate")
            pflkpt = metrics.get("pflash_kept_pct")

            hr_str  = f"{hr:.1%}" if hr is not None else "?"
            pfs_str = f"{pfs:.2f}s" if pfs else "?"
            dec_str = f"{dec:.1f}" if dec else "?"
            acc_str = f"{acc:.1f}%" if acc is not None else "AR"
            pfl_str = f"pflash={pflkpt:.1f}%kept" if pflkpt is not None else ""

            print(
                f"wall={wall_s:.1f}s  pt={pt}  eff={eff}  "
                f"prefix_len={pl} ({hr_str} hit)  "
                f"fresh={fp}  prefill={pfs_str}  "
                f"decode={dec_str}tok/s[{mode}] {acc_str}  "
                f"{pfl_str}  disk_hit={disk_hit}  tool={tool_valid}"
            )

        finally:
            # Step 5: kill server before next turn
            print(f"  Killing server PID={proc.pid} after turn {turn_num}...")
            kill_server(proc, log_f, wait_s=15)
            # Brief pause to let the server fully release GPU resources and flush the
            # disk KV cache to --kv-cache-dir before the next turn's server starts.
            time.sleep(3)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="ABC cache composition harness")
    ap.add_argument("--arm", required=True, choices=sorted(ARMS.keys()),
                    help="Which arm to run")
    ap.add_argument("--trace", default=str(BENCH_DIR / "traces" / "goldgate_fix.jsonl"),
                    help="Path to trace JSONL")
    ap.add_argument("--n", type=int, default=3, help="Number of full-trace repeats")
    ap.add_argument("--seed", type=int, default=42,
                    help="Sampling seed (noted in provenance; server may not support)")
    ap.add_argument("--port", type=int, default=PORT, help="Server port (default 19099)")
    ap.add_argument("--binary", default=None,
                    help="Explicit dflash_server binary path (overrides default build-dir path)")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke gate: first 3 turns only, N=1")
    ap.add_argument("--restart-per-turn", action="store_true",
                    help=(
                        "Per-turn server restart mode: for each turn, launch a fresh "
                        "server, send that one turn's request, record metrics, then kill "
                        "before the next turn.  The --kv-cache-dir (if set in the arm) "
                        "persists across restarts within this run so the on-disk prefix "
                        "cache is the ONLY cross-turn reuse path.  Measures the "
                        "claude-code-reconnects-each-turn / cross-session disk-hit scenario."
                    ))
    args = ap.parse_args()

    if args.port == FORBIDDEN_PORT:
        print(f"ERROR: port {FORBIDDEN_PORT} is the user's live server. Forbidden.", file=sys.stderr)
        sys.exit(1)

    # Override server binary if explicitly provided (preserved/external build).
    if args.binary:
        global SERVER_BIN
        SERVER_BIN = Path(args.binary)
        print(f"[binary] override: {SERVER_BIN}")

    arm_cfg = ARMS[args.arm]
    n_repeats = 1 if args.smoke else args.n
    port = args.port
    restart_per_turn = args.restart_per_turn

    # Verify binary
    if not SERVER_BIN.exists():
        print(f"ERROR: server binary not found: {SERVER_BIN}", file=sys.stderr)
        sys.exit(1)
    bin_sha = sha256_file(SERVER_BIN)
    KNOWN_SHAS = {"92ee2985", "eef74aa0", "47212487", "5c659610", "fc27fff4", "ab9af0a7", "bab0c1dd"}
    if not any(bin_sha.startswith(s) for s in KNOWN_SHAS):
        print(f"WARNING: binary sha {bin_sha[:16]}... not a known PR274 head SHA")

    # Load trace
    trace_path = Path(args.trace)
    turns = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(turns)} turns from {trace_path}")

    # Provenance
    git = git_info(REPO_DIR)
    provenance = {
        "binary": str(SERVER_BIN),
        "binary_sha256": bin_sha,
        "git_branch": git["branch"],
        "git_commit": git["commit"],
        "arm": args.arm,
        "arm_description": arm_cfg["description"],
        "arm_extra_args": arm_cfg["extra_args"],
        "arm_env": arm_cfg["env"],
        "model_target": str(TGT),
        "model_draft_decode": str(DD),
        "model_draft_prefill": str(PD),
        "chat_template": str(TMPL),
        "max_ctx": MAX_CTX,
        "cache_type_k": "tq3_0",
        "cache_type_v": "tq3_0",
        "temperature": TEMP,
        "seed_requested": args.seed,
        "seed_pluggable": "UNKNOWN - dflash_server may not support --seed; noted in provenance only",
        "n_repeats": n_repeats,
        "smoke": args.smoke,
        "restart_per_turn": restart_per_turn,
        "port": port,
        "trace": str(trace_path),
        "n_turns_in_trace": len(turns),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    print("=" * 72)
    print(f"ARM: {args.arm} — {arm_cfg['description']}")
    print(f"Binary SHA: {bin_sha[:16]}...")
    print(f"Branch: {git['branch']} @ {git['commit'][:12]}")
    mode_str = "SMOKE (turns 1-3, N=1)" if args.smoke else f"FULL ({n_repeats} repeats)"
    if restart_per_turn:
        mode_str += " [restart-per-turn: disk-cache cross-session mode]"
    print(f"Mode: {mode_str}")
    print("=" * 72)

    # Check no server already on port
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/health", timeout=2) as r:
            if r.status == 200:
                print(f"ERROR: Something already running on port {port}. Refusing to start.", file=sys.stderr)
                sys.exit(1)
    except Exception:
        pass  # expected: nothing running

    vram_pre, vram_total = nvidia_smi_vram()
    print(f"VRAM before load: {vram_pre}/{vram_total} MiB")

    # Fresh, empty disk KV cache dir per arm (rm -rf before run; persists across the
    # 7 turns WITHIN this arm — that's the disk-reuse point we're measuring).
    kv_dir = arm_cfg.get("kv_cache_dir")
    if kv_dir:
        import shutil
        shutil.rmtree(kv_dir, ignore_errors=True)
        os.makedirs(kv_dir, exist_ok=True)
        print(f"[kv-cache-dir] fresh empty dir: {kv_dir}")
        provenance["kv_cache_dir"] = kv_dir

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_records: list[dict] = []

    if not restart_per_turn:
        # ── single-session mode (default) ────────────────────────────────────
        # One server process handles all turns; in-RAM KV cache serves reuse.
        log_path = BENCH_DIR / f"srv_{args.arm}_{ts_str}.log"
        proc, log_f = launch_server(args.arm, arm_cfg, log_path)

        try:
            print(f"Waiting for server on port {port}...", end=" ", flush=True)
            ok = wait_for_server(port, timeout=360)
            if not ok:
                print("FAILED")
                print(f"Server did not come up. Check {log_path}", file=sys.stderr)
                try:
                    lines = log_path.read_text().splitlines()
                    print("Last 30 log lines:", file=sys.stderr)
                    for ln in lines[-30:]:
                        print(f"  {ln}", file=sys.stderr)
                except Exception:
                    pass
                sys.exit(1)
            print("READY")

            vram_post, _ = nvidia_smi_vram()
            print(f"VRAM after load: {vram_post}/{vram_total} MiB")

            for rep in range(n_repeats):
                print(f"\n--- Repeat {rep+1}/{n_repeats} ---")
                recs = run_trace_repeat(
                    turns, port, log_path, args.arm,
                    smoke=args.smoke, repeat_idx=rep,
                )
                all_records.extend(recs)

        finally:
            print(f"\nKilling server PID={proc.pid}...")
            kill_server(proc, log_f, wait_s=15)
            time.sleep(5)  # brief VRAM drain
            vram_after, _ = nvidia_smi_vram()
            print(f"VRAM after kill: {vram_after}/{vram_total} MiB")

    else:
        # ── restart-per-turn mode ─────────────────────────────────────────────
        # Fresh server per turn; the --kv-cache-dir (already wiped above) is the
        # ONLY cross-turn reuse path.  Measures cross-session disk-cache value.
        log_dir = BENCH_DIR / f"srv_{args.arm}_{ts_str}_turns"
        log_dir.mkdir(exist_ok=True)
        print(f"[restart-per-turn] Per-turn logs dir: {log_dir}")

        for rep in range(n_repeats):
            print(f"\n--- Repeat {rep+1}/{n_repeats} (restart-per-turn) ---")
            recs = run_trace_restart_per_turn(
                turns, port, args.arm, arm_cfg, log_dir,
                smoke=args.smoke, repeat_idx=rep,
            )
            all_records.extend(recs)

        # Final VRAM reading after all per-turn servers have been killed.
        vram_after, _ = nvidia_smi_vram()
        print(f"VRAM after all per-turn servers killed: {vram_after}/{vram_total} MiB")

    # Aggregate
    n_turns_ran = min(3, len(turns)) if args.smoke else len(turns)
    per_turn = aggregate_turns(all_records, n_turns_ran)
    arm_agg  = aggregate_arm(per_turn)

    # Write results
    BENCH_DIR.joinpath("results").mkdir(exist_ok=True)
    suffix = "smoke" if args.smoke else "full"
    report_path = BENCH_DIR / "results" / f"{args.arm}_{ts_str}_{suffix}.md"
    raw_path    = BENCH_DIR / "results" / f"{args.arm}_{ts_str}_{suffix}_raw.json"

    raw_path.write_text(json.dumps({
        "provenance": provenance,
        "per_turn_median": per_turn,
        "arm_aggregate": arm_agg,
        "all_records": all_records,
    }, indent=2, default=str))

    write_report(args.arm, per_turn, arm_agg, provenance, report_path, smoke=args.smoke)

    # Print summary
    print("\n" + "=" * 72)
    print(f"RESULTS: {args.arm}")
    print("=" * 72)
    print(f"total_wall_s              : {arm_agg.get('total_wall_s')}")
    print(f"sum_fresh_prefill_tokens  : {arm_agg.get('sum_fresh_prefill_tokens')}")
    print(f"mean_cache_hit_ratio      : {arm_agg.get('mean_cache_hit_ratio')}")
    print(f"mean_decode_tps           : {arm_agg.get('mean_decode_tps')}")
    print(f"spec_engagement_rate      : {arm_agg.get('spec_engagement_rate')}")
    print(f"mean_accept_rate          : {arm_agg.get('mean_accept_rate')}")
    if arm_agg.get('mean_disk_hit_rate') is not None:
        print(f"mean_disk_hit_rate        : {arm_agg.get('mean_disk_hit_rate')}")
    print(f"tool_call_valid_rate      : {arm_agg.get('tool_call_valid_rate')}")
    print(f"\nRaw:    {raw_path}")
    print(f"Report: {report_path}")
    print(f"Log:    {log_path}")

    # Smoke-specific check
    if args.smoke:
        print("\n=== SMOKE GATE CHECKS ===")
        # Check prefix_len is non-zero on turn 2 (cache reuse visible on warm turn)
        # Turn 1 is always cold (prefix_len=0). Turn 2 should hit the turn-1 snapshot.
        t1 = next((t for t in per_turn if t["turn"] == 1), {})
        t2 = next((t for t in per_turn if t["turn"] == 2), {})
        pl1 = t1.get("prefix_len", 0)
        pl2 = t2.get("prefix_len")
        pl_grows = pl1 == 0 and pl2 is not None and pl2 > 0
        print(f"prefix_len cold->warm t1->t2: {pl1} -> {pl2}  {'PASS' if pl_grows else 'FAIL (cache reuse not visible)'}")

        # Check decode_tps non-null
        dec_ok = any(t.get("decode_tps") is not None for t in per_turn)
        print(f"decode_tps non-null:     {'PASS' if dec_ok else 'FAIL (no decode metrics)'}")

        # Check parser not returning all nulls
        any_metrics = any(t.get("prompt_tokens") is not None for t in per_turn)
        print(f"parser extracting data:  {'PASS' if any_metrics else 'FAIL (all nulls)'}")

        if not pl_grows or not dec_ok or not any_metrics:
            print("\nSMOKE FAILED — do NOT proceed to full run.")
            print("Debug info:")
            for t in per_turn:
                print(f"  turn {t['turn']}: {json.dumps(t, default=str)}")
            print(f"\nLast 50 server log lines from {log_path}:")
            try:
                lines = log_path.read_text().splitlines()
                for ln in lines[-50:]:
                    print(f"  {ln}")
            except Exception:
                pass
            sys.exit(2)
        else:
            print("\nSMOKE PASSED — safe to run full sweep.")


if __name__ == "__main__":
    main()
