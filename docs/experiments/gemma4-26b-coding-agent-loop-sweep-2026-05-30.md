# Gemma 4 26B-A4B-it — coding-agent-loop autotune sweep — 2026-05-30

First end-to-end run of the `coding-agent-loop` autotune profile against
the live gemma-4-26b server on sindri.

* **Host**: sindri (RTX 3090 Ti, 24 GB, WSL2)
* **Image**: locally-built `lucebox-hub:cuda12` from
  `feat/lucebox-docker` @ `cb58edb` (sm_86 only; includes the new
  entrypoint with `DFLASH_FA_WINDOW` plumbing)
* **Fixture**: one 6-bucket multi-turn replay case from
  `luce-bench/src/lucebench/fixtures/agent_recorded/multi_turn_cases.json`
  (single Claude Code session sliced at 8K/16K/32K/64K/100K/128K
  approx-token buckets per `extract-agentic-fixture.py --multi-turn`)
* **Profile**: `coding-agent-loop`, gemma bracket =
  `max_ctx × fa_window × budget × pflash` = `{98304, 131072} ×
  {0, 2048} × {16, 22, 32} × {off}` = 12 cells

## Bracket + outcome

| # | budget | max_ctx | fa_win | pflash | case_tok* | tok/s | pass |
|---|---|---|---|---|---|---|---|
| 1 | 16 | 98304  | 0    | off | 65205 → 90799 | **3.5** | ✓ winner |
| 2 | 22 | 98304  | 0    | off | 65205 → 90799 | 3.4 | ✓ |
| 3 | 32 | 98304  | 0    | off | 65205 → 90799 | 3.2 | ✓ |
| 4 | 16 | 98304  | 2048 | off | 65205 → 90799 | 3.3 | ✓ |
| 5 | 22 | 98304  | 2048 | off | 65205 → 90799 | 2.8 | ✓ |
| 6 | 32 | 98304  | 2048 | off | 65205 → 90799 | 3.0 | ✓ |
| 7 | 16 | 131072 | 0    | off | 102397 → ?    | —   | ✗ HTTP 400 in 0.2s |
| 8 | 22 | 131072 | 0    | off | 102397 → ?    | —   | ✗ HTTP 400 in 0.2s |
| 9 | 32 | 131072 | 0    | off | 102397 → ?    | —   | ✗ HTTP 400 in 0.2s |
| 10 | 16 | 131072 | 2048 | off | 102397 → ?    | —   | ✗ HTTP 400 in 0.2s |
| 11 | 22 | 131072 | 2048 | off | 102397 → ?    | —   | ✗ HTTP 400 in 0.2s |
| 12 | 32 | 131072 | 2048 | off | 102397 → ?    | —   | ✗ HTTP 400 in 0.2s |

\*`case_tok` is the picker's `context_tokens_approx` (`chars / 4`) →
the server's actual `prompt_tokens` after tokenization + chat template
wrapping. Real gemma tokenization expands by ~1.39× relative to chars/4
on this fixture.

## Verification: 131K serves the level2 suite on sindri (2026-05-30 evening)

After bragi's sweep showed 131K viable on a 23 GB Laptop, sindri was
bumped to `max_ctx=131072, budget=22, fa_window=0` and re-ran the
level2 area set. Drop-in works: no quality regression, longctx still
100%.

| area | 98K rate | 131K rate | delta |
|---|---|---|---|
| smoke | 100% (3/3) | 100% (3/3) | = |
| code | 10% (1/10) | 10% (1/10) | = |
| gsm8k | 91% (91/100) | 91% (91/100) | = |
| truthfulqa-mc1 | 80% (80/100) | 76% (76/100) | −4 pp (stochastic) |
| hellaswag | 70% (70/100) | 75% (75/100) | +5 pp (stochastic) |
| agent | 50% (2/4) | 50% (2/4) | = |
| longctx | 100% (6/6) | 100% (6/6) | = |

VRAM at boot on 131K: 21.1 / 24.6 GiB used; ~3 GiB headroom. The
longctx-64k cell prefilled 66,853 tokens in 45.9 s (~1450 tok/s
prefill) and decoded 61 tokens in 955 ms (~64 tok/s decode).
Snapshot: `…-gemma-131k-verify-2026-05-30-67f4`.

## Correction (added 2026-05-30 after bragi sweep)

The 131K failures below were a **fixture-picker artifact, not a VRAM limit**.
After `safety_factor` was updated to 0.7, the picker selects the 64K case
for 131K cells instead of the 100K case, and 131K cells pass on both sindri
and bragi. See
`docs/experiments/gemma4-26b-coding-agent-loop-sweep-bragi-2026-05-30.md`
for the full analysis. Finding 1 below describes what happened mechanically;
the conclusion "98K is the ceiling" no longer holds.

## Findings

1. **131K cells failed due to fixture selection, not VRAM.** All six
   98K cells passed; all six 131K cells failed fast with HTTP 400
   *before* any prefill. The failure mode is request-validation, not
   OOM — the server's "effort-tier ceiling = max_ctx(131072) − 4096 =
   126976" rejects requests whose `prompt_tokens` exceed the ceiling.

2. **The picker's `chars/4` token estimate undercounts on real gemma
   tokenization by ~40%.** The 65K-bucket case (`context_tokens_approx
   = 65205`) tokenizes to **90799** real tokens. The 102K-bucket case
   (`context_tokens_approx = 102397`) likely tokenizes to ~130K+ real
   tokens — over the 126976 ceiling at max_ctx=131072. The picker
   selected it for the 131K cells, the server rejected it, every
   131K cell failed identically.

3. **`fa_window` doesn't help at this prompt size on gemma4-26b.**
   `fa_window=0` (full attention, server default) beat `fa_window=2048`
   in every (budget, max_ctx) cell. The differences are small (~3-7%)
   but consistent. fa_window's sparse-decode optimization is wasted
   compute on a 26B-A4B-MoE model where decode bandwidth isn't the
   bottleneck at 90K tokens.

4. **`budget` axis is nearly flat at 90K prompt size.** 16/22/32 produce
   3.5/3.4/3.2 tok/s — small enough margin that noise dominates. The
   heuristic default of `budget=22` is fine; the sweep's preference for
   `budget=16` is within run-to-run variance.

5. **Decode throughput at 90K prompt: ~3.5 tok/s.** Mostly prefill cost:
   wall=72s, ~256 completion tokens, so decode-phase is ~30s for 256
   tokens (~8.5 tok/s decode-only). Prefill of 90K tokens takes ~40s
   on a 3090 Ti — about 2250 tok/s prefill rate.

## Heuristic update (gemma4 24 GB WSL)

Bump `runtime_from_host()` for the 22-31 GB / WSL tier from
`max_ctx=65536` to `max_ctx=98304`. Empirical evidence that 98K serves
real agentic traces with reasonable headroom (90K real prompts pass
with ~3 GB VRAM unused). Keep `budget=16` and the existing defaults.

131K remains plausible as a manual operator setting (proven to boot
2026-05-29; serves short prompts) but not as a default — the sweep
fixture overshoots its prompt budget, and we lack a long-prompt case
sized for the real 126976-token ceiling. Future work:

* Fix the picker's safety factor (use ~0.7× the approximate budget)
  or re-tokenize fixtures with the real gemma tokenizer at extraction
  time.
* Re-run the 131K cells with a properly-sized case (~110K real tokens)
  to confirm 131K serves agentic workloads, not just short prompts.

## Reproducing

```sh
# From the worktree, with LUCEBOX_HOST_* env unset (sweep falls back
# to the persisted [host] block in config.toml):
cd /home/erik/Projects/lucebox-hub-285
uv run --project lucebox python -m lucebox autotune \
    --sweep --profile coding-agent-loop --yes
```

Raw output captured at
`/tmp/sweep-gemma-coding-agent-loop.log` during the 2026-05-30 run
(local-only; not checked into the repo because the per-cell server
restarts produce ~MB of progress noise).
