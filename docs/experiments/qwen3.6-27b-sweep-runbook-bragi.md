# Qwen 3.6 27B — coding-agent-loop sweep runbook for bragi

A step-by-step runbook for repeating the gemma-4-26b sweep
(`gemma4-26b-coding-agent-loop-sweep-2026-05-30.md`) against
qwen3.6-27b on **bragi** (RTX 5090 Laptop / MaxQ, ~24 GB).

This is the operational counterpart to
`autotune-profile-sweep-protocol.md` — that doc explains the
machinery; this one is the literal sequence to run.

## Why qwen needs its own sweep

The gemma sweep on sindri produced a heuristic update for the 24 GB
WSL tier (`max_ctx 65536 → 98304`). That update applies to **gemma4**
on a 24 GB box, but qwen3.6 has a different KV-quant story per the
auto-memory:

> **gemma4 KV is hardcoded F16** — `cache_type_k/v` and the
> `DFLASH27B_KV_*` envs are no-ops on the gemma4 backend
> (Qwen35-path only). Don't sweep KV-quant for gemma.

Qwen35's loader *does* honor `cache_type_k/v`, so the qwen bracket
adds that axis. The arch-aware bracket builder
(`_coding_agent_loop_qwen_bracket` in `autotune.py`) already does
this — you don't need to write code, just run the right invocation
under the right preset.

## Preflight on bragi

```sh
# 1. Working tree on the same branch as sindri's sweep
cd <worktree-on-bragi>
git fetch easel feat/lucebox-docker
git checkout feat/lucebox-docker
git pull easel feat/lucebox-docker

# 2. Submodules (one-time per worktree)
git submodule update --init --recursive

# 3. Image build for bragi's arch (sm_120 for RTX 5090)
DFLASH_CUDA_ARCHES=120 scripts/build_image.sh --load
# Verify the new entrypoint is in the image
docker run --rm --entrypoint cat lucebox-hub:cuda12 \
    /opt/lucebox-hub/server/scripts/entrypoint.sh \
  | grep -c DFLASH_FA_WINDOW   # must be >= 1

# 4. Switch the lucebox service to the local image
lucebox config set image=lucebox-hub
# Don't forget the variant if you've changed it:
# lucebox config set variant=cuda12

# 5. Populate host facts in config.toml (so the sweep's fallback works)
lucebox check     # exits 0 + prints host probe; updates [host] in config.toml
lucebox config get host.vram_gb    # must be > 0

# 6. Activate qwen3.6-27b. If not on disk yet:
lucebox models download qwen3.6-27b --activate
# Otherwise:
lucebox config set model.preset=qwen3.6-27b

# 7. Restart + verify
systemctl --user restart lucebox.service
sleep 12
curl -s http://localhost:8080/health     # 200
journalctl --user -u lucebox.service -n 30 --no-pager | grep arch=
# Expect: [backend_factory] detected arch=qwen3.5  (or arch=qwen36, depending on tip)
```

## Multi-turn fixture

The fixture from sindri's run is checked into the repo at
`luce-bench/src/lucebench/fixtures/agent_recorded/multi_turn_cases.json`
— it's already on bragi after the pull. It covers 8K/16K/32K/64K/100K/
128K buckets (approx, char/4) from one Claude Code session.

The `pick_multi_turn_case_for_budget` helper applies a 0.7
`safety_factor` to the prompt budget so cases that would tokenize
over a cell's effort-tier ceiling get excluded. This was calibrated
against gemma's tokenizer (1.39× expansion vs chars/4). If the qwen
tokenizer expansion is different and you see HTTP 400 / "context too
long" failures across multiple cells, drop `safety_factor` further.

You can re-harvest from a bragi-local session if you want a
different fixture — but per the
`feedback_iterate_with_one_trace` convention, the existing one is
fine unless something breaks.

## The sweep

```sh
cd <worktree>
uv run --project lucebox python -m lucebox autotune \
    --sweep --profile coding-agent-loop --yes 2>&1 \
  | tee /tmp/qwen-sweep-bragi.log
```

The qwen 22-31 GB bracket is `max_ctx × {tq3_0, q8_0} × budget` per
`_coding_agent_loop_qwen_bracket` — `2 × 2 × 3 = 12 cells` if your
bragi VRAM tier matches sindri's 24 GB band. On RTX 5090 the
expanded-VRAM tier (32-47 GB) gives `1 max_ctx × 3 KV × 2 budget = 6
cells` per the bracket — bragi's VRAM count after WSL/loaderoverhead
determines which branch fires. Check
`lucebox config get host.vram_gb` and read the corresponding branch in
`autotune.py::_coding_agent_loop_qwen_bracket` to predict the cell
count.

Each cell takes ~2–3 min on a 90K-real-token prompt; whole sweep
runs ~30–45 min wall.

## What to expect

Three things to verify in the result table:

1. **At least one cell passes.** If every cell fails with HTTP 400,
   the picker's safety_factor is still too lax for qwen's tokenizer.
   Try `--safety-factor 0.5` (you'd need to plumb a CLI flag; for
   now hot-patch the call site or shrink fixture buckets).

2. **The KV-quant axis shows a real signal.** Expect tq3_0 to enable
   larger max_ctx than q8_0 on the same VRAM (the whole reason
   qwen's bracket includes this axis). If the bracket lists both KV
   choices at the same max_ctx and tq3_0 is faster than q8_0, that
   confirms qwen prefers the smaller KV.

3. **The winner is `pass` not `partial`.** The agent_replay scorer
   is binary (pass / fail) — there's no partial. Multi-cell ties
   resolve to larger max_ctx, then larger fa_window, then lower
   budget. Sanity-check the persistent state with `lucebox config
   get` after the sweep.

## Documenting findings

Write up the run as `qwen3.6-27b-coding-agent-loop-sweep-bragi-
<date>.md` under `docs/experiments/`, modeled on the 2026-05-30
gemma doc. Include:

- Hardware + image SHA + commit SHA used
- The full bracket cells + their pass/fail + tok/s
- Specifically: what the KV-quant axis showed (whether tq3_0 won)
- Specifically: real tokenizer expansion ratio for qwen
  (`prompt_tokens` from a passing cell's server log vs the case's
  `context_tokens_approx`)
- Any heuristic update implication for `runtime_from_host` — the
  qwen tier almost certainly wants `cache_type_k=cache_type_v=tq3_0`
  on a 24 GB box if the sweep proves it

Then commit the doc + the autotune.py heuristic update in one commit,
mirroring `cefa0f5` from the gemma run.

## Safe rollback

If the sweep persists a winner that breaks normal usage:

```sh
# Restore from the sweep backup (created automatically; deleted only
# on a fully-successful sweep):
ls ~/.lucebox/config.toml.sweep-backup       # if present, restore
cp ~/.lucebox/config.toml.sweep-backup ~/.lucebox/config.toml
systemctl --user restart lucebox.service

# Or rewind individual keys:
lucebox config set dflash.max_ctx=<safe-value>
lucebox config set dflash.cache_type_k=
lucebox config set dflash.cache_type_v=
lucebox config set dflash.fa_window=0
systemctl --user restart lucebox.service
```

## Cross-host comparison

After bragi's run lands, the sindri (gemma) and bragi (qwen)
experiments together cover the two main 24-ish GB workloads. If
either run surfaces a heuristic update that contradicts the other,
write a third doc reconciling them — the per-arch branches in
`_coding_agent_loop_<arch>_bracket` make it possible for the two
presets to live with different ceilings without conflict, but the
*shared* `runtime_from_host` tiers should stay coherent across
presets at a given VRAM band.
