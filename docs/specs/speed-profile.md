# Lucebox speed profiler (MVP)

A small, CI-runnable profiler for the inference engine. It measures forward-pass
speed on one GPU and produces a report that shows **where the time goes**, so a
reviewer can see at a glance whether a PR moved the needle and where the next
optimization margin is.

It runs the engine **binaries directly** — `test_dflash` (the speculative / DFlash
decode path) and `test_generate` (the plain autoregressive baseline) — with **no
HTTP**, so the numbers reflect compute, not server/network noise. Both are CMake
build targets under `server/build/` and can be overridden with `--df-bin` /
`--ar-bin` (or the `DFLASH_BIN` / `DFLASH_BIN_AR` environment variables).

## What it reports

Three layers, coarse to fine:

1. **Headline latency/throughput** — prefill time, model-side TTFT estimate, decode
   tok/s, ms/token, plus the speculative-decoding **acceptance length (AL)** and
   accept %. Repeated timing passes report **mean ± stddev**, not a single-point
   estimate, so reviewers can tell whether a small delta is real or just jitter.
   (`AL` is how many tokens the target commits per draft+verify step; decode
   throughput ≈ `AL / step_time`.)
2. **Per-step phase breakdown** — `draft_compute`, `draft_logits`, `verify_compute`,
   … from the engine's own timers. Tells you *which phase* dominates a step.
3. **Kernel-level (nsys)** — top CUDA kernels by GPU time, **kernel launches per
   token** (kernel-fusion signal), **host↔device copy time per token** (CPU/GPU-overlap
   signal), and sync-heavy CUDA APIs (CPU-stall signal). Tells you *why* a phase is slow.

## Parameters (and why they are fixed defaults)

The defaults mirror the **shipping config** so the numbers are production-representative,
and they stay fixed so every run is comparable over time:

- **`--budget 22`** — DDTree speculation budget = how many draft positions are verified
  per target pass. 22 is the `dflash_server` default. Bigger = a bigger bet (higher
  potential acceptance length) but more draft+verify cost; tuning it is a separate sweep,
  not the CI job.
- **`--n-gen 128`** — requested generated tokens per prompt. This is long enough
  to amortize startup costs and reduce very-short-generation bias, while still
  keeping the shared 3090 CI queue bounded. The argument is passed as `<n_gen>`
  to both `test_dflash` and `test_generate`, where it controls the response length
  generated for each benchmark prompt.
- **`--reps 5`** — repeats each prompt enough times to expose run-to-run GPU
  variance (thermals, clock boosting, scheduler jitter), then reports mean and
  stddev for the headline metrics. Use `--reps 3` for a faster smoke profile when
  needed, but PR comparisons should prefer 5+.
- **`--noise-rsd-pct 0.05`** — noise threshold. If any tracked
  headline metric has relative stddev above 5%, the markdown calls the profile
  **NOISY** and tells reviewers to treat small deltas as below the profiler
  detection threshold. Automatic PR profiles report this as a warning; a manual
  promotion profile fails because noisy measurements are not valid release evidence.

**Rule:** keep these consistent. A delta vs the baseline is only a valid regression
signal if both runs used the same config — if you ever change a parameter, re-seed the
baseline (you cannot compare across configs). When baseline and current 1σ intervals
overlap and the delta is smaller than `--regress-pct`, the report marks that row as
**noisy / overlap** instead of inviting reviewers to chase a ghost regression. These
states are warnings on automatic PR profiles. A manual dispatch is the promotion check:
it fails on a missing runner model, missing baseline/report, noisy result, or a
performance regression outside the configured threshold.

## Losslessness gate (and why a bit-exact compare is too strict on its own)

The gate checks that greedy speculative decode produces the same token stream as
greedy autoregressive (AR) decode — a lossless-spec-decode claim should never change
the model's output.

A naive bit-exact compare **flags false failures**, and the engine itself explains
why: the target sees draft tokens as a *batch* in the verify step but one-at-a-time in
AR decode, and different GEMM shapes reduce in a different order. IEEE FP is
non-associative, so when the top-2 logits sit within epsilon the argmax tie can flip —
one token diverges and everything after it follows. See
`server/src/qwen35/qwen35_backend.cpp` ("different GPU batch sizes → FP-nondeterministic
state divergence → different greedy output") and `server/eval/README.md`, which runs an
identical `baseline_2` config precisely because "cache-induced divergence and intrinsic
noise are indistinguishable."

So the gate runs a **determinism control**: a second, identical-config AR pass (reusing
the AR baseline run — no extra GPU cost). For each prompt:

| AR vs AR (control) | DFlash vs AR | verdict |
|---|---|---|
| identical | identical | **PASS** |
| identical | diverges | **FAIL** — output changed and it is not run-to-run noise (triage needed) |
| diverges | (either) | **inconclusive** — engine is intrinsically nondeterministic here, can't judge |

The gate fails **only** on the middle row, which answers "real bug or too-strict check?":
it no longer flags run-to-run noise (that becomes *inconclusive*). A FAIL means the fast
path genuinely changed the output — but that is still not proven a logic bug: it can be
the batched-verify FP effect above (verify scores draft tokens as a batch vs AR
one-at-a-time). Classifying a FAIL as bug-vs-FP needs the **logit gap** at the first
mismatch (near-tie = FP, clear gap = bug) — a follow-up the binaries don't emit yet. CI
surfaces a FAIL as a non-blocking `::warning::` for triage, including on manual
promotion profiles. It cannot safely become a blocking signal until the binaries expose
the logit gap at the first mismatch and the profiler can classify FP near-ties.

## CI settings

The `Speed Profile` workflow uses the same profiler defaults as the local recipe:
`--n-gen 128`, `--reps 5`, and `--noise-rsd-pct 0.05`. Runner owners can
temporarily override those values with repo variables `LUCEBOX_SPEED_N_GEN`,
`LUCEBOX_SPEED_REPS`, and `LUCEBOX_SPEED_NOISE_RSD_PCT`, but PR-to-PR comparisons
should keep them fixed.

Automatic pull-request runs are advisory and never block a merge. A manual dispatch is
strict enough to use as promotion evidence: it requires the staged target/draft models,
an existing baseline at `LUCEBOX_SPEED_BASELINE`, a complete JSON report, stable
measurements, and no regression beyond `LUCEBOX_SPEED_REGRESS_PCT` (10% by default).
Correctness divergences remain explicit warnings for the reason described above; use the
normal GPU correctness suite as the blocking correctness gate.

## Run it locally

```bash
cd server
python3 scripts/profile.py \
  --target /opt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft  /opt/models/draft/dflash-draft-3.6-q4_k_m.gguf \
  --tokenizer Qwen/Qwen3.6-27B \
  --n-gen 128 --budget 22 --reps 5 --noise-rsd-pct 0.05 \
  --nsys --check-lossless \
  --baseline scripts/speed-baseline.json --regress-pct 0.10 \
  --out-json profile.json --out-md profile.md
```

## Comparing against a baseline

The profiler can diff the current run against a saved profile so reviewers see a single
regression table instead of two separate reports. The delta is advisory on PR events and
blocking on manual promotion runs.
The comparison is a JSON round-trip:

1. **Capture a baseline.** Run the profiler on the reference commit and keep its
   JSON output:

   ```bash
   python3 scripts/profile.py ... --out-json scripts/speed-baseline.json
   ```

   For CI, stage that file outside the checkout on the self-hosted runner and set the
   repository variable `LUCEBOX_SPEED_BASELINE` to its absolute path. Keeping the
   promotion baseline outside the PR checkout prevents a candidate from weakening its
   own comparison. `scripts/speed-baseline.json` remains the local fallback. Re-seed the
   runner baseline from a reviewed `main` run whenever you intentionally change a
   profiler parameter (`--budget`, `--n-gen`, `--reps`, …): configs cannot be compared.

2. **Compare a later run.** Point `--baseline` at that file and set the regression
   threshold:

   ```bash
   python3 scripts/profile.py ... \
     --baseline scripts/speed-baseline.json --regress-pct 0.10
   ```

3. **Read the delta.** The report adds a **"Delta vs baseline"** table with, per
   headline metric, `baseline ± σ`, `now ± σ`, the absolute Δ and Δ%. A row is
   flagged as a regression only when the move exceeds `--regress-pct` **and** the
   baseline/current 1σ intervals do **not** overlap — a delta inside the noise band
   is marked **noisy / overlap** instead, so reviewers don't chase jitter.

Both runs must use the same parameters for the delta to be a valid signal (see the
**Rule** under *Parameters* above).
