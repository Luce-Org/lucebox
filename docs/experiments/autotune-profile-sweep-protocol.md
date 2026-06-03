# Autotune profile sweep — developer protocol

How to run the `lucebox autotune --sweep` machinery against a live
server and persist the winning DflashRuntime. This doc is the
operator-facing companion to the code in
`lucebox/src/lucebox/autotune.py` (Profile dataclass, brackets) and
`lucebox/src/lucebox/sweep.py` (driver, scorers).

The motivating run is the gemma-4-26b sweep on sindri 2026-05-30 —
see `gemma4-26b-coding-agent-loop-sweep-2026-05-30.md` for the
findings. That doc reports a result; this doc reports the *procedure*
so a different host (e.g. bragi for qwen3.6-27b) can reproduce it
without re-deriving the steps.

## What a profile is

A `Profile` is a triple of `(name, candidate_configs, scorer)` registered
in `autotune.PROFILES`. Two ship today:

| Profile name        | Bracket builder                | Scorer                       |
|---------------------|--------------------------------|------------------------------|
| `heuristic`         | `_heuristic_candidates` (preset-agnostic; `budget × KV-quant` per VRAM tier) | `decode_tps_snapshot` — mean `decode_tokens_per_sec` averaged across `luce-bench snapshot --level level1` areas |
| `coding-agent-loop` | `_coding_agent_loop_candidates` (arch-aware: gemma4 sweeps `max_ctx × fa_window × budget × pflash`; qwen3.6/laguna add the KV-quant axis since the qwen35 backend respects it) | `agent_replay_pass_rate` — POST the largest fitting multi-turn replay case to `/v1/chat/completions`, pass iff non-empty response within wall budget. Speed metric is `completion_tokens / wall_seconds` because longctx-area snapshots ship empty `decode_tokens_per_sec`. |

Adding a profile is intentionally lightweight: append an entry to
`PROFILES`, write a `_<name>_candidates` builder (returns
`list[DflashRuntime]` from `HostFacts + preset`), and pick an existing
scorer key (`decode_tps_snapshot` or `agent_replay_pass_rate`). New
scorers require a new branch in `sweep.run_sweep`; profiles that reuse
an existing scorer do not.

## Preconditions for a sweep

1. **The server is up.** `lucebox status` → active; `curl -s
   http://localhost:8080/health` → 200.
2. **Config has the target preset.** `lucebox config get model.preset`
   shows the preset you want to sweep (e.g. `gemma-4-26b`,
   `qwen3.6-27b`). The bracket builder dispatches per-arch based on
   this string.
3. **The lucebox image is current.** If you've touched
   `entrypoint.sh`, `DflashRuntime` fields, or the sweep code itself,
   rebuild and switch the service to your local image — the in-image
   entrypoint is what reads `DFLASH_*` envs.

   ```sh
   git submodule update --init --recursive   # one-time, per worktree
   DFLASH_CUDA_ARCHES=<sm_NN> scripts/build_image.sh --load
   lucebox config set image=lucebox-hub
   systemctl --user restart lucebox.service
   docker exec lucebox grep -c DFLASH_FA_WINDOW \
       /opt/lucebox-hub/server/scripts/entrypoint.sh  # must print > 0
   ```

4. **Persisted host facts exist.** `lucebox config get host.vram_gb`
   must be > 0. If empty, run `lucebox check` once via the wrapper —
   that populates `[host]` in `config.toml`. The sweep's host-facts
   fallback (sweep.py: `from_env()` → `cfg.host` when env is empty)
   reads this when invoked outside the lucebox.sh wrapper.

5. **For `coding-agent-loop`: a multi-turn fixture is on disk.**
   `luce-bench/src/lucebench/fixtures/agent_recorded/multi_turn_cases.json`
   must exist with cases bucketed across the max_ctx values the
   bracket will exercise. Harvest one if missing:

   ```sh
   python3 scripts/extract-agentic-fixture.py --multi-turn \
       <path/to/long-session.jsonl> \
       --out luce-bench/src/lucebench/fixtures/agent_recorded/multi_turn_cases.json
   ```

   Per repo convention (see CLAUDE memory
   `feedback_iterate_with_one_trace.md`), ONE long session is enough
   to cycle with until something breaks — don't pre-curate a balanced
   set across length buckets.

## Running the sweep

The recommended invocation is host-side via `uv` because the sweep
restarts the lucebox.service per cell (which would kill any container
the wrapper exec'd into):

```sh
cd <worktree>          # must be the lucebox workspace root
uv run --project lucebox python -m lucebox autotune \
    --sweep --profile coding-agent-loop --yes
```

The `--yes` skips the interactive confirmation. The sweep writes a
backup of `config.toml` to `~/.lucebox/config.toml.sweep-backup`
before the first cell, restores it on Ctrl-C / signal / total
failure, and deletes it on a successful winner-apply.

For each cell the sweep:

1. Writes the cell's dflash.* fields to `config.toml` (the allowlist
   in `sweep.DFLASH_ALLOWLIST` — bumped 2026-05-30 to include
   `fa_window`).
2. `systemctl --user restart lucebox.service`.
3. Polls `http://localhost:8080/v1/models` for readiness (60 s budget).
4. Invokes the profile's scorer.
5. Records `(passed, speed_metric, error)` per cell.

After all cells: ranks via `_pick_winner(results, scorer)`, writes the
winner's dflash.* values, restarts onto the winner, removes the
backup.

## Reading the result table

The console output ends in a Rich table. Columns vary by scorer:

- `decode_tps_snapshot` columns: `#`, `budget`, `max_ctx`, `kv`, `tps`,
  `status`. Higher tps wins; ties → lower max_ctx, then lower budget.
- `agent_replay_pass_rate` columns: `#`, `budget`, `max_ctx`, `fa_win`,
  `kv`, `pflash`, `case_tok`, `tok/s`, `pass`, `status`. Only passing
  cells qualify; among those, higher `tok/s` wins; ties → larger
  max_ctx, larger fa_window, then lower budget.

`status` for losing cells carries a one-line reason (HTTP code,
timeout, wall budget). For winning cells it shows `← winner`.

## Known gotchas

- **chars/4 underestimates real tokenizer counts by ~40%** on gemma.
  The multi-turn picker applies a 0.7 `safety_factor` to the prompt
  budget so cases that would tokenize over the server's effort-tier
  ceiling get excluded. If your arch has a different tokenizer
  efficiency, tune `safety_factor` when calling
  `pick_multi_turn_case_for_budget`. Long-term fix: re-tokenize cases
  with the server's actual tokenize endpoint at extraction time and
  store the real count.

- **Wrapper dispatch can't reach localhost.** `lucebox autotune
  --sweep` via the `lucebox` wrapper spawns a one-off container via
  `cmd_in_container` *without* `--network host`, so the agent_replay
  scorer's `urllib.request.urlopen('http://localhost:8080/...')`
  doesn't reach the host's server port. Run via `uv` from the
  worktree as shown above. There's an open follow-up (task #9 from
  the May 2026 work) to teach the wrapper to `docker exec` into the
  running container for steady-state CLI calls.

- **The `lucebox autotune --sweep` writes dflash.fa_window
  unconditionally.** Even if your image's entrypoint doesn't honor
  `DFLASH_FA_WINDOW` (pre-2026-05-30 builds), the field lands in
  config.toml — but the server ignores it. Rebuild + re-tag before
  sweeping a profile whose bracket varies fa_window.

- **The DFLASH allowlist is duplicated.** `sweep.DFLASH_ALLOWLIST` and
  the strict 11-field set referenced in `cli.DFLASH_ALLOWLIST` need to
  stay in lockstep. The sweep's copy is intentionally longer (it adds
  `fa_window` for the bracket axis). When you add a new field, touch
  both.

- **fa_window=0 is the server's stock value.** docker_run.py only
  emits `DFLASH_FA_WINDOW` when nonzero, and entrypoint.sh only
  appends `--fa-window N` when `$DFLASH_FA_WINDOW > 0`. Both layers
  reproduce the server's default when the env is unset.

## How to repeat this for a different preset

The bracket builder dispatches per-preset inside
`_coding_agent_loop_candidates`. To run the sweep for a new
arch/preset:

1. Confirm the preset is in `lucebox.download.PRESETS` and that the
   target + draft GGUFs are on disk under `$LUCEBOX_MODELS`. If not:
   `lucebox models download <preset> --activate`.
2. Activate the preset: `lucebox config set model.preset=<name>` (and
   `model.target_file` / `model.draft_file` if the registry default
   isn't what you want).
3. Restart the service; verify the new arch boots:
   `journalctl --user -u lucebox.service -n 30 | grep arch=`
4. Confirm the bracket builder has a branch for your preset. If not,
   add one in `autotune.py` modeled on `_coding_agent_loop_gemma_bracket`
   or `_coding_agent_loop_qwen_bracket`. The key axis decisions:
   - Does the arch backend respect `cache_type_k/v`? gemma4 does not
     (hardcoded F16 in the loader); qwen35 does. Include the KV-quant
     axis only when the backend respects it.
   - What's the proven max_ctx ceiling on your VRAM tier? Use VRAM
     math from architecture metadata (`n_layer`, `n_head_kv`,
     `head_dim`, sliding-window pattern) — see the gemma4 32-31 GB
     tier comment in `_coding_agent_loop_gemma_bracket` for the
     reasoning.
   - Does PFlash need a separate drafter file you've configured?
     Leave pflash=off in the bracket if not, and add it later when
     `prefill_drafter` is wired up.
5. If the bracket adds a new axis (e.g. `--reasoning-effort`):
   - Add the field to `DflashRuntime` in `types.py`
   - Plumb it through `docker_run.py` → `entrypoint.sh` → server CLI
   - Add it to `DFLASH_ALLOWLIST` in `sweep.py` and the registry in
     `config.py`
   - Add a regression test mirroring the existing
     `test_fa_window_in_dflash_allowlist` test
6. Run as above.

## Output artifacts

- Console: the Rich table summarizing all cells + the winner.
- `config.toml`: the winner's dflash.* fields, persisted.
- `~/.local/share/lucebox/profile-snapshots/sweep/cell-NN-<hash>/` —
  per-cell snapshot dirs for the heuristic profile (the
  coding-agent-loop scorer doesn't write snapshots; it scores
  in-process).
- A short experiment doc under `docs/experiments/<host>-<preset>-...
  -sweep-<date>.md` summarizing findings + heuristic deltas, modeled
  on the 2026-05-30 gemma doc.

## When to update the heuristic

`runtime_from_host` in `autotune.py` ships the default DflashRuntime
per VRAM tier. When a sweep persists a winner that the heuristic
didn't predict, update the heuristic so first-run installations don't
pay the empirical-cost. Comment the update with the date + experiment
doc reference so future readers can audit the choice — see the
2026-05-30 WSL 24 GB bump for the convention.

Don't push a heuristic update without a corresponding sweep
experiment doc. The whole point of the heuristic is that it
approximates the empirical winner; an untraced change is harder to
debug when later sweeps disagree.
