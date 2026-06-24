#!/usr/bin/env python3
"""Multi-turn agentic composition harness with prefix-cache awareness.

Usage:
    python3 replay_harness.py --arm A_baseline --trace traces/goldgate_fix.jsonl
    python3 replay_harness.py --arm A_baseline --trace traces/goldgate_fix.jsonl --smoke
    python3 replay_harness.py --arm C_always   --trace traces/goldgate_fix.jsonl
    python3 replay_harness.py --arm C_threshold --trace traces/goldgate_fix.jsonl
    python3 replay_harness.py --selftest

Smoke gate (--smoke): runs first 3 turns only, N=1.
Full run: all turns, N=3.

Arms:
  A_baseline      : dFlash decode + prefix cache, NO pFlash, NO stochastic
  A_stochastic    : dFlash decode + prefix cache + DFLASH_STOCHASTIC, NO pFlash
  C_always        : pFlash always + dFlash decode  (ATTN_PRIMARY, stochastic)
  C_threshold     : pFlash auto/32K threshold + dFlash decode
  lucebox         : dFlash 35B-A3B Q4_K_M + dFlash drafter + spec-gate
  llama_cpp_mtp   : llama.cpp b9781 + MTP bundled GGUF + --spec-type draft-mtp
  llama_cpp_ar    : llama.cpp b9781 + autoregressive (no speculation)

Port: 19099 (NEVER 18099)
Max-ctx: 65536
"""

from __future__ import annotations

import argparse
import fcntl
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
MAX_CTX = 40960
TEMP = 0.7

FORBIDDEN_PORT = 18099  # user's live server — never touch

# 3-way H2H model paths
TGT_35B   = Path("/home/peppi/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DD_35B    = Path("/home/peppi/models/qwen3.6-35b-a3b-dflash-new/qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf")
MTP_GGUF  = Path("/home/peppi/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
LLAMA_BIN     = Path("/home/peppi/llama.cpp/build-cuda/bin/llama-server")
LLAMA_CUDA_LIB = Path("/home/peppi/llama.cpp/build-cuda/bin")

GPU_LOCK_FILE = "/tmp/lucebox_gpu.lock"

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

    # ── Context-size bench arms (ring-cap=65536, gated dFlash, temp=0) ────
    # BENCH_35B: 35B-A3B Q3_K_XL all-hot + Modal drafter.
    #   Past-best config with DFLASH_FEAT_RING_CAP=65536 to avoid ring-wrap at 34K.
    #   DFLASH_SPEC_GATE=1 explicit (it's the default-on; belt-and-suspenders).
    #   fa_window=0 is the server default (full attention); no --fa-window needed.
    #   No --spark/--kvflash: all-hot MoE.
    "BENCH_35B": {
        "description": "35B-A3B Q3_K_XL all-hot + Modal BF16 drafter; ring=65536 gated spec-decode temp=0",
        "model_target":  Path("/home/peppi/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"),
        "model_draft":   Path("/home/peppi/models/qwen3.6-35b-a3b-dflash-new/qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf"),
        "temperature": 0.0,
        "kv_cache_types": ("f16", "f16"),
        "extra_args": [
            "--draft", str(Path("/home/peppi/models/qwen3.6-35b-a3b-dflash-new/qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf")),
            "--draft-swa", "2048",
        ],
        "env": {
            "DFLASH_FEAT_RING_CAP": "65536",
            "DFLASH_SPEC_GATE": "1",
            "DFLASH_DRAFT_CTX_MAX": "2048",
        },
    },

    # BENCH_27B: 27B dense Q4_K_M + Lucebox BF16 drafter.
    #   ring=16384: 27B uses fc_in=25600; ring=65536 would allocate 6.7 GB and OOM.
    #   16384*25600*4 = 1.7 GB (same cap as A_baseline default, proven-working config).
    #   gate=1, temp=0.
    "BENCH_27B": {
        "description": "27B Q4_K_M dense + Lucebox drafter; q4_0 KV, cap=2048 cliff fix, ring=4096, gated temp=0",
        "model_target":  Path("/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf"),
        "model_draft":   Path("/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-bf16-reconverted.gguf"),
        "temperature": 0.0,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--draft", str(Path("/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-bf16-reconverted.gguf")),
            "--draft-swa", "2048",
        ],
        "env": {
            # draft_ctx capped at 2048 (cliff fix) -> ring only needs ~4096, not max_ctx.
            # 4096*25600*2(bf16) = 0.2 GB (was 16384 -> 1.7 GB); dodges the 27B ring-OOM.
            "DFLASH_FEAT_RING_CAP": "4096",
            "DFLASH_SPEC_GATE": "1",
            "DFLASH_DRAFT_CTX_MAX": "2048",
        },
    },

    # ── 3-way H2H arms ──────────────────────────────────────────────────────
    # lucebox: dFlash server + 35B-A3B Q4_K_M (trunk-only) target + dFlash drafter.
    #   Validated 192117 config: restore-consume snapshot win + ddtree speculation.
    #   GGML_CUDA_NO_VMM in env is idempotent with the dflash launch-path default.
    "lucebox": {
        "description": "dFlash 35B-A3B Q4_K_M + dFlash drafter; spec-gate, kvflash=8192, ring=131072, ddtree",
        "server": "dflash",
        "model_target": TGT_35B,
        "model_draft":  DD_35B,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--draft", str(DD_35B),
            "--draft-swa", "2048",
            "--kvflash", "8192",
            "--ddtree", "--ddtree-budget", "16",
        ],
        "env": {
            "GGML_CUDA_NO_VMM": "1",
            "KVFLASH_RESTORE_CONSUME": "1",
            "DFLASH_SPEC_GATE": "1",
            "DFLASH_FEAT_RING_CAP": "131072",
            "DFLASH_DRAFT_CTX_MAX": "2048",
        },
    },

    # lucebox_ddtree22: identical to lucebox EXCEPT --ddtree-budget is 22 (sweep).
    #   Controlled ddtree-budget sweep: 16 vs 22, everything else identical.
    "lucebox_ddtree22": {
        "description": "dFlash 35B-A3B Q4_K_M + dFlash drafter; spec-gate, kvflash=8192, ring=131072, ddtree budget=22",
        "server": "dflash",
        "model_target": TGT_35B,
        "model_draft":  DD_35B,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--draft", str(DD_35B),
            "--draft-swa", "2048",
            "--kvflash", "8192",
            "--ddtree", "--ddtree-budget", "22",
        ],
        "env": {
            "GGML_CUDA_NO_VMM": "1",
            "KVFLASH_RESTORE_CONSUME": "1",
            "DFLASH_SPEC_GATE": "1",
            "DFLASH_FEAT_RING_CAP": "131072",
            "DFLASH_DRAFT_CTX_MAX": "2048",
        },
    },

    # lucebox_no_ddtree: identical to lucebox EXCEPT ddtree is removed.
    #   Controlled ddtree A/B: same target, drafter, kvflash=8192,
    #   KVFLASH_RESTORE_CONSUME=1, DFLASH_SPEC_GATE=1, ring=131072, q4_0 KV.
    #   Differs ONLY by absence of --ddtree --ddtree-budget 16.
    "lucebox_no_ddtree": {
        "description": "dFlash 35B-A3B Q4_K_M + dFlash drafter; spec-gate, kvflash=8192, ring=131072, NO ddtree",
        "server": "dflash",
        "model_target": TGT_35B,
        "model_draft":  DD_35B,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--draft", str(DD_35B),
            "--draft-swa", "2048",
            "--kvflash", "8192",
        ],
        "env": {
            "GGML_CUDA_NO_VMM": "1",
            "KVFLASH_RESTORE_CONSUME": "1",
            "DFLASH_SPEC_GATE": "1",
            "DFLASH_FEAT_RING_CAP": "131072",
            "DFLASH_DRAFT_CTX_MAX": "2048",
        },
    },

    # llama_cpp_mtp: llama.cpp b9781 + MTP bundled GGUF (NextN head embedded).
    #   Speculation via --spec-type draft-mtp, draft-n-max 2.
    "llama_cpp_mtp": {
        "description": "llama.cpp b9781 MTP spec-decode (draft-mtp, n-max=2); q4_0 KV",
        "server": "llama_cpp",
        "binary": str(LLAMA_BIN),
        "model":  str(MTP_GGUF),
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "2",
        ],
        "env": {},
    },

    # llama_cpp_ar: llama.cpp b9781 plain autoregressive baseline (no speculation).
    #   MTP GGUF used so weight match is exact vs llama_cpp_mtp; NextN head unused.
    "llama_cpp_ar": {
        "description": "llama.cpp b9781 autoregressive baseline (no speculation); q4_0 KV",
        "server": "llama_cpp",
        "binary": str(LLAMA_BIN),
        "model":  str(MTP_GGUF),
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [],
        "env": {},
    },

    # lucebox_ar: dflash_server pure autoregressive baseline.
    #   Same model+KV as the lucebox family (35B-A3B Q4_K_M, q4_0 KV) but with
    #   NO drafter, NO kvflash, NO ddtree, NO spec env vars.
    #   env={} — the launcher unconditionally injects GGML_CUDA_NO_VMM=1; all
    #   DFLASH_SPEC_GATE / FEAT_RING_CAP / DRAFT_CTX_MAX / RESTORE_CONSUME dropped.
    #   Isolates dflash_server's raw AR decode speed on the same model+KV+ctx
    #   so we can compare directly to llama_cpp_ar (139.8 tok/s on stock b9781).
    "lucebox_ar": {
        "description": "dflash_server 35B-A3B Q4_K_M pure AR (no draft/spec/kvflash); q4_0 KV",
        "server": "dflash",
        "model_target": TGT_35B,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [],
        "env": {},
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

# llama.cpp MTP per-slot acceptance line (stderr):
#   I slot print_timing: id N | task N | draft acceptance = 0.91176 (   31 accepted /    34 generated), ...
LLAMA_MTP_RE = re.compile(
    r'slot print_timing:.*?draft acceptance\s*=\s*([\d.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\)'
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


def build_llama_cpp_cmd(arm_cfg: dict, port: int, max_ctx: int) -> list[str]:
    """Build the CLI for a llama_cpp arm. Pure function — testable without GPU."""
    binary = arm_cfg.get("binary", str(LLAMA_BIN))
    model  = arm_cfg.get("model",  str(MTP_GGUF))
    ctk, ctv = arm_cfg.get("kv_cache_types", ("q4_0", "q4_0"))
    return [
        binary,
        "-m", model,
        "--host", HOST,
        "--port", str(port),
        "-c", str(max_ctx),
        "-ngl", "999",
        "--cache-type-k", ctk,
        "--cache-type-v", ctv,
        "--chat-template-file", str(TMPL),
        "--jinja",
    ] + arm_cfg.get("extra_args", [])


def parse_llama_cpp_timings(timings: dict) -> dict:
    """Extract prefill/decode metrics from llama.cpp timings dict. Pure function."""
    m: dict = {}
    pn  = timings.get("prompt_n", 0)
    pms = timings.get("prompt_ms", 0.0)
    dn  = timings.get("predicted_n", 0)
    dms = timings.get("predicted_ms", 0.0)
    if pms > 0:
        m["prefill_s"]   = pms / 1000.0
        m["prompt_tokens"] = pn
        if pn > 0:
            m["prefill_tps"] = round(pn / (pms / 1000.0), 1)
    if dms > 0:
        m["decode_s"]   = dms / 1000.0
        m["out_tokens"] = dn
        if dn > 0:
            m["decode_tps"] = round(dn / (dms / 1000.0), 1)
    return m


def parse_llama_cpp_accept(
    log_text: str,
    mtp_offset: int,
    extra_args: list,
) -> tuple[dict, int]:
    """Parse llama.cpp MTP draft-acceptance from server stderr log.

    Returns (metrics_dict, new_mtp_offset).

    If --spec-type draft-mtp is present in extra_args and the log contains a
    matching line, returns accept_rate (0-100 float), spec_engaged=True, and
    decode_mode='spec'.  If spec-type args present but no log line yet found,
    returns spec_engaged=True with accept_rate=None (not 0.0 — avoids false neg).
    If spec-type args absent, returns {} (AR arm, no speculation).
    """
    has_mtp = "--spec-type" in extra_args and "draft-mtp" in extra_args
    if not has_mtp:
        return {}, mtp_offset

    all_matches = LLAMA_MTP_RE.findall(log_text)
    new_matches = all_matches[mtp_offset:]

    if not new_matches:
        # MTP arm but log line not yet found — mark as spec_engaged but rate unknown
        return {"spec_engaged": True, "decode_mode": "spec"}, mtp_offset

    m_frac, m_acc, m_gen = new_matches[0]
    accept_rate = round(float(m_frac) * 100.0, 2)
    accepted    = int(m_acc)
    generated   = int(m_gen)
    return {
        "spec_engaged": True,
        "decode_mode": "spec",
        "accept_rate": accept_rate,
        "draft_accepted": accepted,
        "draft_generated": generated,
    }, mtp_offset + 1


_gpu_lock_fd: Optional[int] = None


def acquire_gpu_lock() -> None:
    global _gpu_lock_fd
    _gpu_lock_fd = open(GPU_LOCK_FILE, "w")
    fcntl.flock(_gpu_lock_fd, fcntl.LOCK_EX)
    print("[gpu-lock] acquired GPU lock")


def release_gpu_lock() -> None:
    global _gpu_lock_fd
    if _gpu_lock_fd is not None:
        fcntl.flock(_gpu_lock_fd, fcntl.LOCK_UN)
        _gpu_lock_fd.close()
        _gpu_lock_fd = None
        print("[gpu-lock] released GPU lock")


def launch_server(arm_name: str, arm_cfg: dict, log_path: Path, max_ctx: int = MAX_CTX) -> tuple:
    """Start server for given arm. Returns (proc, log_file_handle).

    Branches on arm_cfg.get('server', 'dflash'):
      'dflash'    — existing dflash_server path (unchanged).
      'llama_cpp' — llama.cpp llama-server, OpenAI-compat API.
    """
    server_type = arm_cfg.get("server", "dflash")

    if server_type == "llama_cpp":
        cmd = build_llama_cpp_cmd(arm_cfg, PORT, max_ctx)
        env = dict(os.environ)
        # Inject CUDA .so directory so the binary finds libggml-cuda.so etc.
        cuda_lib_dir = str(LLAMA_CUDA_LIB)
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{cuda_lib_dir}:{existing_ld}" if existing_ld else cuda_lib_dir
        # No DFLASH_* / GGML_CUDA_NO_VMM for llama_cpp arms
        print(f"[server] Launching {arm_name} (llama_cpp): {' '.join(cmd)}")
        print(f"[server] LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}")
        log_f = log_path.open("w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)
        print(f"[server] PID={proc.pid}  log={log_path}")
        return proc, log_f

    # --- dflash path (unchanged) ---
    env = dict(os.environ)
    env["GGML_CUDA_NO_VMM"] = "1"
    # DFlash decode optimizations (from dFlash drafter integration session):
    # - block_size=16 (model card sweet spot for Qwen3.6 DFlash)
    # - feature ring cap=16384 (default 4096 collapses acceptance at the ring boundary)
    env["DFLASH_DRAFT_BLOCK_SIZE"] = "16"
    env["DFLASH_FEAT_RING_CAP"] = "16384"
    env.update(arm_cfg["env"])

    # Per-arm model override (used by BENCH_35B / BENCH_27B / lucebox arms)
    target_model = arm_cfg.get("model_target", TGT)

    ctk, ctv = arm_cfg.get("kv_cache_types", ("tq3_0", "tq3_0"))
    cmd = [
        str(SERVER_BIN),
        str(target_model),
        "--host", HOST,
        "--port", str(PORT),
        "--max-ctx", str(max_ctx),
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

def _anthropic_to_openai_tools(tools: list) -> list:
    """Best-effort remap Anthropic tool defs to OpenAI function format."""
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn: dict = {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {}),
        }
        out.append({"type": "function", "function": fn})
    return out


def send_request(
    req_body: dict,
    port: int,
    server_type: str = "dflash",
) -> tuple[dict, float, Optional[str]]:
    """POST request to server. Returns (response_body, wall_s, error_str).

    server_type='dflash'    -> POST /v1/messages  (Anthropic format)
    server_type='llama_cpp' -> POST /v1/chat/completions  (OpenAI format)
    """
    if server_type == "llama_cpp":
        # Convert Anthropic trace format -> OpenAI messages format
        sys_prompt = req_body.get("system")
        messages: list = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        for msg in req_body.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, list):
                # flatten content blocks to text (tool_use blocks skipped for S1)
                text_parts = [
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                content = " ".join(text_parts)
            messages.append({"role": msg["role"], "content": content})

        body_out: dict = {
            "messages": messages,
            "temperature": req_body.get("temperature", 0),
            "max_tokens": req_body.get("max_tokens", 1024),
            "stream": False,
        }
        raw_tools = req_body.get("tools")
        if raw_tools:
            try:
                body_out["tools"] = _anthropic_to_openai_tools(raw_tools)
            except Exception:
                pass  # best-effort; S1 trace has no tools

        url = f"http://{HOST}:{port}/v1/chat/completions"
        payload = json.dumps(body_out, ensure_ascii=False).encode("utf-8")
        http_req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(http_req, timeout=900) as r:
                resp = json.loads(r.read())
            return resp, time.time() - t0, None
        except Exception as ex:
            return {}, time.time() - t0, str(ex)

    # --- dflash / Anthropic path (unchanged) ---
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
    arm_cfg: Optional[dict] = None,
) -> list[dict]:
    """Run all turns in the trace; parse server log per turn.

    Returns list of per-turn metric dicts.
    """
    turns_to_run = turns[:3] if smoke else turns

    arm_temp    = arm_cfg.get("temperature") if arm_cfg else None
    server_type = arm_cfg.get("server", "dflash") if arm_cfg else "dflash"
    is_dflash   = server_type == "dflash"

    # Wait for log file to exist and server to be settled (dflash only)
    if is_dflash:
        deadline = time.time() + 300
        while not log_path.exists() and time.time() < deadline:
            time.sleep(1)

    cache_off = done_off = spec_off = ar_off = pflash_off = survival_off = 0
    llama_mtp_off = 0

    results = []
    for turn_idx, req_body in enumerate(turns_to_run):
        turn_num = turn_idx + 1
        print(f"  [{arm_name} repeat={repeat_idx+1}] turn {turn_num}/{len(turns_to_run)} "
              f"(~{len(json.dumps(req_body))//3:,} est_tok) ...", end=" ", flush=True)

        if arm_temp is not None:
            req_body = dict(req_body)
            req_body["temperature"] = arm_temp

        resp_body, wall_s, err = send_request(req_body, port, server_type=server_type)

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

        if is_dflash:
            try:
                log_text = log_path.read_text(errors="replace")
            except Exception:
                log_text = ""

            metrics, cache_off, done_off, spec_off, ar_off, pflash_off, survival_off = \
                parse_log_for_request(
                    log_text, cache_off, done_off, spec_off, ar_off, pflash_off, survival_off
                )
        else:
            # llama_cpp: parse timings from response body
            timings = resp_body.get("timings")
            if timings:
                metrics = parse_llama_cpp_timings(timings)
            else:
                metrics = {}
                print("[warn] no timings in llama_cpp response", end=" ")

            # Parse MTP draft-acceptance from server log (guarded: llama_cpp arms only)
            extra_args = arm_cfg.get("extra_args", []) if arm_cfg else []
            try:
                log_text = log_path.read_text(errors="replace")
            except Exception:
                log_text = ""
            accept_m, llama_mtp_off = parse_llama_cpp_accept(log_text, llama_mtp_off, extra_args)
            metrics.update(accept_m)

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

def run_selftest() -> None:
    """No-GPU self-check. Verifies pure functions and trace file correctness."""
    print("=== SELFTEST ===")
    failures = []

    # (a) build_llama_cpp_cmd for llama_cpp_mtp
    arm = ARMS["llama_cpp_mtp"]
    cmd = build_llama_cpp_cmd(arm, PORT, MAX_CTX)
    assert "--spec-type" in cmd and "draft-mtp" in cmd, "missing --spec-type draft-mtp"
    assert "-m" in cmd, "missing -m"
    m_idx = cmd.index("-m")
    assert str(MTP_GGUF) in cmd[m_idx + 1], f"wrong model: {cmd[m_idx+1]}"
    assert "--cache-type-k" in cmd, "missing --cache-type-k"
    k_idx = cmd.index("--cache-type-k")
    assert cmd[k_idx + 1] == "q4_0", f"expected q4_0 cache-type-k, got {cmd[k_idx+1]}"
    print(f"  (a) build_llama_cpp_cmd: PASS  cmd={cmd}")

    # (b) parse_llama_cpp_timings
    sample = {"prompt_n": 100, "prompt_ms": 500.0, "predicted_n": 40, "predicted_ms": 800.0}
    t = parse_llama_cpp_timings(sample)
    assert t["prefill_tps"] == 200.0, f"prefill_tps expected 200.0 got {t['prefill_tps']}"
    assert t["decode_tps"]  ==  50.0, f"decode_tps expected 50.0 got {t['decode_tps']}"
    print(f"  (b) parse_llama_cpp_timings: PASS  prefill_tps={t['prefill_tps']} decode_tps={t['decode_tps']}")

    # (b2) parse_llama_cpp_accept — MTP log line extraction
    mtp_log_line = (
        "0.38.538.542 I slot print_timing: id  3 | task 0 | "
        "draft acceptance = 0.91176 (   31 accepted /    34 generated), "
        "mean acceptance length =  2.82, acceptance rate per position = (1.000, 0.824)"
    )
    mtp_args = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    am, new_off = parse_llama_cpp_accept(mtp_log_line, 0, mtp_args)
    assert am.get("spec_engaged") is True, f"expected spec_engaged=True, got {am}"
    assert am.get("decode_mode") == "spec", f"expected decode_mode=spec, got {am}"
    assert abs(am.get("accept_rate", 0) - 91.18) < 0.1, f"accept_rate wrong: {am}"
    assert am.get("draft_accepted") == 31, f"draft_accepted wrong: {am}"
    assert am.get("draft_generated") == 34, f"draft_generated wrong: {am}"
    assert new_off == 1, f"offset should advance to 1, got {new_off}"
    # AR arm (no --spec-type): should return empty dict
    ar_m, ar_off = parse_llama_cpp_accept("some log", 0, [])
    assert ar_m == {}, f"AR arm should return empty dict, got {ar_m}"
    # MTP arm with no log line yet: spec_engaged=True, accept_rate absent
    no_log_m, no_log_off = parse_llama_cpp_accept("no matching line here", 0, mtp_args)
    assert no_log_m.get("spec_engaged") is True, f"expected spec_engaged=True even without log line"
    assert "accept_rate" not in no_log_m, f"accept_rate should be absent when log line not found"
    assert no_log_off == 0, f"offset should not advance when no log line"
    print(f"  (b2) parse_llama_cpp_accept: PASS  {am}")

    # (b3) LLAMA_BIN points to build-cuda path
    assert "build-cuda" in str(LLAMA_BIN), f"LLAMA_BIN must point to build-cuda, got {LLAMA_BIN}"
    assert "build-cuda" in str(LLAMA_CUDA_LIB), f"LLAMA_CUDA_LIB must point to build-cuda, got {LLAMA_CUDA_LIB}"
    print(f"  (b3) LLAMA_BIN CUDA path: PASS  {LLAMA_BIN}")

    # (b4) lucebox family: lucebox / lucebox_ddtree22 / lucebox_no_ddtree
    assert "lucebox_no_ddtree" in ARMS, "lucebox_no_ddtree arm missing from ARMS"
    assert "lucebox_ddtree22" in ARMS, "lucebox_ddtree22 arm missing from ARMS"
    lb      = ARMS["lucebox"]
    lb_d22  = ARMS["lucebox_ddtree22"]
    lb_nod  = ARMS["lucebox_no_ddtree"]
    # ddtree flags
    assert "--ddtree" in lb["extra_args"], "lucebox must have --ddtree in extra_args"
    assert "--ddtree-budget" in lb["extra_args"], "lucebox must have --ddtree-budget"
    bud_idx = lb["extra_args"].index("--ddtree-budget")
    assert lb["extra_args"][bud_idx + 1] == "16", f"lucebox ddtree-budget must be 16, got {lb['extra_args'][bud_idx+1]}"
    assert "--ddtree" in lb_d22["extra_args"], "lucebox_ddtree22 must have --ddtree"
    assert "--ddtree-budget" in lb_d22["extra_args"], "lucebox_ddtree22 must have --ddtree-budget"
    bud22_idx = lb_d22["extra_args"].index("--ddtree-budget")
    assert lb_d22["extra_args"][bud22_idx + 1] == "22", f"lucebox_ddtree22 budget must be 22, got {lb_d22['extra_args'][bud22_idx+1]}"
    assert "22" in lb_d22["extra_args"], "lucebox_ddtree22 extra_args must contain '22'"
    assert "--ddtree" not in lb_nod["extra_args"], "lucebox_no_ddtree must NOT have --ddtree"
    # All three share the same env keys and kv type
    assert lb["env"].keys() == lb_d22["env"].keys(), "lucebox_ddtree22 env keys must match lucebox"
    assert lb["env"].keys() == lb_nod["env"].keys(), "lucebox_no_ddtree env keys must match lucebox"
    assert lb.get("kv_cache_types") == lb_d22.get("kv_cache_types"), "lucebox_ddtree22 kv_cache_types must match lucebox"
    assert lb.get("kv_cache_types") == lb_nod.get("kv_cache_types"), "lucebox_no_ddtree kv_cache_types must match lucebox"
    import json as _json
    print(f"  (b4) lucebox arm dict:")
    print(f"       {_json.dumps(lb, indent=6, default=str)}")
    print(f"  (b4) lucebox_ddtree22 arm dict:")
    print(f"       {_json.dumps(lb_d22, indent=6, default=str)}")
    print(f"  (b4) lucebox_no_ddtree arm dict:")
    print(f"       {_json.dumps(lb_nod, indent=6, default=str)}")
    print(f"  (b4) ddtree family (16/22/no) arm check: PASS")

    # (b5) lucebox_ar: pure AR dflash arm — no draft, no spec env vars
    assert "lucebox_ar" in ARMS, "lucebox_ar arm missing from ARMS"
    lb_ar = ARMS["lucebox_ar"]
    assert lb_ar.get("server") == "dflash", "lucebox_ar must use dflash server"
    assert lb_ar.get("model_target") == TGT_35B, "lucebox_ar model_target must be TGT_35B"
    assert lb_ar.get("kv_cache_types") == ("q4_0", "q4_0"), "lucebox_ar must use q4_0 KV"
    assert lb_ar.get("extra_args") == [], "lucebox_ar extra_args must be empty (no --draft)"
    assert lb_ar.get("env") == {}, "lucebox_ar env must be empty (no spec env vars)"
    # Confirm no spec-triggering flags leak into the arm
    for forbidden in ("--draft", "--kvflash", "--ddtree", "--prefill-drafter"):
        assert forbidden not in lb_ar.get("extra_args", []), \
            f"lucebox_ar must NOT have {forbidden} in extra_args"
    for forbidden_env in ("DFLASH_SPEC_GATE", "DFLASH_FEAT_RING_CAP",
                          "DFLASH_DRAFT_CTX_MAX", "KVFLASH_RESTORE_CONSUME"):
        assert forbidden_env not in lb_ar.get("env", {}), \
            f"lucebox_ar env must NOT contain {forbidden_env}"
    print(f"  (b5) lucebox_ar arm dict:")
    print(f"       {_json.dumps(lb_ar, indent=6, default=str)}")
    print(f"  (b5) lucebox_ar pure-AR check: PASS")

    # (c) chat_simple.jsonl — 5 lines, valid JSON each
    trace = BENCH_DIR / "traces" / "chat_simple.jsonl"
    if not trace.exists():
        failures.append(f"chat_simple.jsonl missing: {trace}")
        print(f"  (c) chat_simple.jsonl: FAIL — file not found")
    else:
        lines = [l for l in trace.read_text().splitlines() if l.strip()]
        assert len(lines) == 5, f"expected 5 lines, got {len(lines)}"
        for i, ln in enumerate(lines):
            obj = json.loads(ln)
            assert "messages" in obj, f"line {i} missing messages"
        print(f"  (c) chat_simple.jsonl: PASS  ({len(lines)} lines)")

    if failures:
        print("\nSELFTEST FAILED:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("=== SELFTEST PASS ===")


def main() -> None:
    ap = argparse.ArgumentParser(description="ABC cache composition harness")
    ap.add_argument("--arm", required=False, choices=sorted(ARMS.keys()),
                    help="Which arm to run")
    ap.add_argument("--trace", default=str(BENCH_DIR / "traces" / "goldgate_fix.jsonl"),
                    help="Path to trace JSONL")
    ap.add_argument("--n", type=int, default=3, help="Number of full-trace repeats")
    ap.add_argument("--seed", type=int, default=42,
                    help="Sampling seed (noted in provenance; server may not support)")
    ap.add_argument("--port", type=int, default=PORT, help="Server port (default 19099)")
    ap.add_argument("--max-ctx", type=int, default=MAX_CTX,
                    help=f"Context ceiling for server (default {MAX_CTX})")
    ap.add_argument("--binary", default=None,
                    help="Explicit dflash_server binary path (overrides default build-dir path)")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke gate: first 3 turns only, N=1")
    ap.add_argument("--selftest", action="store_true",
                    help="Run no-GPU self-check and exit")
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

    if args.selftest:
        run_selftest()
        return

    if not args.arm:
        ap.error("--arm is required (or use --selftest)")

    if args.port == FORBIDDEN_PORT:
        print(f"ERROR: port {FORBIDDEN_PORT} is the user's live server. Forbidden.", file=sys.stderr)
        sys.exit(1)

    # Override server binary if explicitly provided (preserved/external build).
    if args.binary:
        global SERVER_BIN
        SERVER_BIN = Path(args.binary)
        print(f"[binary] override: {SERVER_BIN}")

    arm_cfg = ARMS[args.arm]
    server_type = arm_cfg.get("server", "dflash")
    n_repeats = 1 if args.smoke else args.n
    port = args.port
    max_ctx = args.max_ctx
    restart_per_turn = args.restart_per_turn

    # Verify binary (dflash arms only; llama_cpp arms use arm_cfg["binary"])
    if server_type == "dflash":
        if not SERVER_BIN.exists():
            print(f"ERROR: server binary not found: {SERVER_BIN}", file=sys.stderr)
            sys.exit(1)
        bin_sha = sha256_file(SERVER_BIN)
        KNOWN_SHAS = {"92ee2985", "eef74aa0", "47212487", "5c659610", "fc27fff4", "ab9af0a7", "bab0c1dd"}
        if not any(bin_sha.startswith(s) for s in KNOWN_SHAS):
            print(f"WARNING: binary sha {bin_sha[:16]}... not a known PR274 head SHA")
    else:
        # llama_cpp arm: verify the arm's binary exists
        llama_bin = Path(arm_cfg.get("binary", str(LLAMA_BIN)))
        if not llama_bin.exists():
            print(f"ERROR: llama-server binary not found: {llama_bin}", file=sys.stderr)
            sys.exit(1)
        bin_sha = sha256_file(llama_bin)
        print(f"[binary] llama_cpp binary sha: {bin_sha[:16]}...")

    # Load trace
    trace_path = Path(args.trace)
    turns = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(turns)} turns from {trace_path}")

    # Provenance
    git = git_info(REPO_DIR)
    _arm_tgt  = arm_cfg.get("model_target",  TGT)
    _arm_dd   = arm_cfg.get("model_draft",   DD)
    _arm_temp = arm_cfg.get("temperature",   TEMP)

    provenance = {
        "binary": str(SERVER_BIN) if server_type == "dflash" else arm_cfg.get("binary", str(LLAMA_BIN)),
        "binary_sha256": bin_sha,
        "git_branch": git["branch"],
        "git_commit": git["commit"],
        "arm": args.arm,
        "arm_description": arm_cfg["description"],
        "arm_extra_args": arm_cfg["extra_args"],
        "arm_env": arm_cfg.get("env", {}),
        "server_type": server_type,
        "model_target": str(arm_cfg.get("model", _arm_tgt)),
        "model_draft_decode": str(_arm_dd),
        "model_draft_prefill": str(PD),
        "chat_template": str(TMPL),
        "max_ctx": max_ctx,
        "cache_type_k": arm_cfg.get("kv_cache_types", ("tq3_0",))[0],
        "cache_type_v": arm_cfg.get("kv_cache_types", ("tq3_0", "tq3_0"))[1],
        "temperature": _arm_temp,
        "seed_requested": args.seed,
        "n_repeats": n_repeats,
        "smoke": args.smoke,
        "restart_per_turn": restart_per_turn,
        "port": port,
        "trace": str(trace_path),
        "n_turns_in_trace": len(turns),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    print("=" * 72)
    print(f"ARM: {args.arm} — {arm_cfg['description']}  [{server_type}]")
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

    kv_dir = arm_cfg.get("kv_cache_dir")
    if kv_dir:
        import shutil
        shutil.rmtree(kv_dir, ignore_errors=True)
        os.makedirs(kv_dir, exist_ok=True)
        print(f"[kv-cache-dir] fresh empty dir: {kv_dir}")
        provenance["kv_cache_dir"] = kv_dir

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_records: list[dict] = []

    acquire_gpu_lock()
    try:
      if not restart_per_turn:
        # ── single-session mode (default) ────────────────────────────────────
        log_path = BENCH_DIR / f"srv_{args.arm}_{ts_str}.log"
        proc, log_f = launch_server(args.arm, arm_cfg, log_path, max_ctx=max_ctx)

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
                    arm_cfg=arm_cfg,
                )
                all_records.extend(recs)

        finally:
            print(f"\nKilling server PID={proc.pid}...")
            kill_server(proc, log_f, wait_s=15)
            time.sleep(5)
            vram_after, _ = nvidia_smi_vram()
            print(f"VRAM after kill: {vram_after}/{vram_total} MiB")

      else:
        # ── restart-per-turn mode ─────────────────────────────────────────────
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

        vram_after, _ = nvidia_smi_vram()
        print(f"VRAM after all per-turn servers killed: {vram_after}/{vram_total} MiB")

    finally:
        release_gpu_lock()

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
