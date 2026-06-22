# Lucebox speed profiler (MVP)

A small, CI-runnable profiler for the inference engine. It measures forward-pass
speed on one GPU and produces a report that shows **where the time goes**, so a
reviewer can see at a glance whether a PR moved the needle and where the next
optimization margin is.

It runs the engine **binaries directly** (no HTTP) so the numbers reflect compute,
not server/network noise.

## What it reports

Three layers, coarse to fine:

1. **Headline latency/throughput** — prefill time, model-side TTFT estimate, decode
   tok/s, ms/token, plus the speculative-decoding **acceptance length (AL)** and
   accept %. (`AL` is how many tokens the target commits per draft+verify step; decode
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
- **`--n-gen 48`** — tokens generated per prompt. Long enough to be past warmup, short
  enough to keep the shared 3090 fast. (Very short n_gen under-counts spec-decode because
  fixed startup costs aren't amortized.)
- **`--reps 3`** — repeats the decode and takes the **median** tok/s to absorb run-to-run
  GPU variance (thermals, clock boosting, scheduler jitter).

**Rule:** keep these consistent. A delta vs the baseline is only a valid regression
signal if both runs used the same config — if you ever change a parameter, re-seed the
baseline (you cannot compare across configs).

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
surfaces a FAIL as a non-blocking `::warning::` for triage; it stays report-only.

## Run it locally

```bash
cd server
python3 scripts/profile.py \
  --target /opt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft  /opt/models/draft/dflash-draft-3.6-q4_k_m.gguf \
  --tokenizer Qwen/Qwen3.6-27B \
  --n-gen 48 --budget 22 --reps 3 \
  --nsys --check-lossless \
  --baseline scripts/speed-baseline.json --regress-pct 0.10 \
  --out-json profile.json --out-md profile.md
