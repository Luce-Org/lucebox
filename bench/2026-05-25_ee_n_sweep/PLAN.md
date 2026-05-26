# ee N-sweep Plan: baseline / ee3 / ee5 / ee7

Scope: NIAH @ 32K/64K/128K + 5-client multi-turn bandit session, all four conditions.
Goal: find the smallest N where quality and accept_rate hold vs ee7.

## Trigger

User says "GPU is free, run the ee_n sweep."

Pre-flight GPU check:

    flock -n /tmp/lucebox-gpu.lock echo ok

Exit 0 means GPU is free. Exit 1 means another process holds the lock — wait or
ask the user.

## Pre-flight: generate NIAH case files (once, CPU only)

Run once before the NIAH sweep. Requires `transformers` and `Qwen3-0.6B` tokenizer.

    python3 pflash/tests/niah_gen.py --context 32768  --n 3 -o /tmp/niah_32768.jsonl
    python3 pflash/tests/niah_gen.py --context 65536  --n 3 -o /tmp/niah_65536.jsonl
    python3 pflash/tests/niah_gen.py --context 131072 --n 3 -o /tmp/niah_131072.jsonl

Check niah_gen.py's flags first — --context / --n / -o are the expected interface
but verify before running if in doubt:

    python3 pflash/tests/niah_gen.py --help

## Commands (literal copy-paste, run from worktree root)

    cd /home/peppi/Dev/lucebox-hub/.claude/worktrees/drafter-fastpath

    # Step 1: NIAH sweep (~50 min total, serialized)
    dflash/bench/run_ee_n_sweep.sh

    # Step 2: multi-client (~140 min total, serialized)
    dflash/bench/run_ee_n_multiclient.sh

Both scripts accept an optional output_dir as $1.

## Expected output layout

    dflash/bench/results/2026-05-25_ee_n_sweep/
        raw_results.json
        SUMMARY.md                          # written by the sweep script
        baseline_32768_case{0,1,2}_server.log
        baseline_65536_case{0,1,2}_server.log
        baseline_131072_case{0,1,2}_server.log
        ee3_*.log  ee5_*.log  ee7_*.log

    .claude/worktrees/drafter-fastpath/bench/results/2026-05-25_ee_n_multiclient/
        baseline/
            claude_code.csv  claude_code.log  claude_code_server.log
            hermes.csv       hermes.log       hermes_server.log
            opencode.csv     opencode.log     opencode_server.log
            pi.csv           pi.log           pi_server.log
            codex.csv        codex.log        codex_server.log
        ee3/   ee5/   ee7/   (same structure)

## Decision gate

Smallest N where ALL hold vs ee7:

1. NIAH @ 32K and 64K: within +-1 needle (e.g. 2/3 or 3/3 both acceptable)
2. accept_rate across 5 clients: mean within +-2 pp of ee7
3. drafter wall: <= ee7 wall at each context (smaller N must be faster)
4. No crashes: zero ggml_view_3d asserts, zero server OOM

Outcome mapping:
- ee3 passes all -> propose ee3 as new production default (follow-up PR after #274 merges)
- ee5 passes, ee3 fails -> ee5
- neither passes -> ee7 stays default, close the N-reduction investigation

## Estimated cost

- GPU wall: ~3 hr serialized (NIAH 50 min + multi-client 140 min)
- Disk: ~50 MB
- Compute cost: $0 (local RTX 3090)

## Hard stops

- Any condition raises ggml_view_3d assert -> STOP, Bug #42 regressed, file issue
- flock wait exceeds 30 min -> STOP, something holds the GPU lock unexpectedly
- A condition's NIAH drops to 0/3 at 32K or 64K -> STOP, do not run further
- Server OOM on baseline (no early-exit) at 128K -> expected if VRAM too tight;
  note it and continue with ee conditions

## Out of scope (deferred)

- 1K-16K NIAH: already 3/3 on ee7 per 2026-05-21_ee7_broad results; ee3/ee5 speedup
  at short context is negligible, gate is long-context quality
- ee10, ee14, ee2: user explicitly scoped to baseline + ee3 + ee5 + ee7
- Cross-family drafter (SmolLM2): loader-ready but kernel not generalized yet

## Flag mismatch notes (for next session)

### NIAH script (run_niah_ee7_longctx.py)

The existing `run_niah_ee7_longctx.py` only understands conditions "baseline",
"ee14", "ee7" — it hardcodes env var logic for those three. It does NOT accept
"ee3" or "ee5".

Resolution: the new `run_ee_n_sweep_niah.py` script was written for this sweep.
It accepts arbitrary N via CONDITION_SPECS dict and handles env var injection
directly. Do NOT call run_niah_ee7_longctx.py for the N-sweep.

### bandit-session required flags

`client_test_runner.py bandit-session` requires --target, --draft, --bin (no
defaults). The multi-client script passes all three hardcoded to:
  --target /home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf
  --draft  /home/peppi/models/Qwen3-0.6B-BF16.gguf
  --bin    dflash/build/dflash_server

If model paths have changed, edit CLIENTS_TARGET / DRAFTER / BIN vars at the
top of run_ee_n_multiclient.sh before running.

### niah_gen.py interface

Verify niah_gen.py flags before generating cases:
  python3 pflash/tests/niah_gen.py --help

The expected interface is --context / --n / -o. If the flags differ, adjust
the generate commands above.

### env prefix for baseline in multi-client script

The shell uses `env $env_prefix python3 ...`. When env_prefix is empty (baseline),
this collapses to `env python3 ...` which is valid — env with no assignments is
a no-op passthrough.
