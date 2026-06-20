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

It also includes an optional **losslessness gate**: greedy speculative decode must
produce the exact same token stream as greedy autoregressive decode. A mismatch means
the "fast" path is changing the model's output, which a lossless-spec-decode claim
should never do.

## Run it locally

```bash
cd server
python3 scripts/profile.py \
  --target /opt/models/Qwen3.5-27B-Q4_K_M.gguf \
  --draft  /opt/models/draft/qwen35_dflash/model.safetensors \
  --n-gen 48 --budget 22 --reps 3 \
  --nsys --check-lossless \
  --out-json profile.json --out-md profile.md
```

Requirements: `transformers` (tokenizer), built `test_dflash` + `test_generate`,
and `nsys` on PATH for the kernel layer (the profiler degrades gracefully without it).

Regression view: pass `--baseline path/to/previous/profile.json` to print the delta
vs a stored run (e.g. the latest `main`).

## CI

`.github/workflows/speed-profile.yml` runs on the self-hosted RTX 3090 for PRs that
touch `server/**` or `optimizations/**`. It is **report-only** (`continue-on-error`)
— it publishes the report to the Actions run summary and uploads `profile.json`,
`profile.md`, and the `.nsys-rep` trace as artifacts. It never blocks a merge.

To wire it into the existing `ci.yml` instead of a standalone workflow, drop the
`speed-profile` job into that file (it already has the matching `[self-hosted, gpu,
sm86]` runner and `lucebox3-gpu-runner` concurrency group).

## Scope (MVP)

Starts with one path — Qwen3.5-27B Q4_K_M + DFlash/DDTree. The runner is parametrized
by `--target/--draft/--budget`, so the same tool extends to Qwen3.6-27B, the MoE target,
and the pflash / kvflash paths by changing flags. Per-PR runs can be gated to only the
path the PR touches.
