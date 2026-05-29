# Lucebox Benchmark Snapshot Spec

This document describes the stable report sections emitted by
`lucebox profile --export-snapshot`. The format is plain `key=value` blocks so
a snapshot can be diffed, archived, or used to seed another machine's tuning
run. Profile results are append-only under `models/.lucebox/profile/results`;
snapshot export computes the current fingerprint for every registered step and
uses the newest matching result. Missing, stale, failed, and unavailable steps
are emitted instead of hidden.

## Sections

- `[snapshot]`: schema, capture timestamp, git revision, selected base URL, and
  overall completeness status.
- `[profile.step.N]`: one registered step per section, including the current
  fingerprint hash, freshness status, result status, report hashes, and log
  tails for remote debugging.
- `[machine]`: observed Linux, CPU, GPU, memory, Python, and accelerator runtime
  facts at capture time.
- `[docker]`: Docker client/server/runtime facts.
- `[image]`: container image tag, digest, and creation time when available.
- `[runtime.config]`: port, container name, model directory, and autotune source.
- `[benchmark.autotune_latest]`: latest optimizer report when present.
- `[benchmark.frontiers]` and `[benchmark.frontiers.row.N]`: long-context HTTP
  frontier timing rows.
- `[quality.capability_smoke]`: hard-gated smoke capability results.
- `[quality.ds4_eval]`: score-only HTTP port of every antirez/ds4 `ds4-eval`
  case, preserving the original `source/id` case names for comparison. This
  uses the upstream-sized 16k-token generation budget with thinking enabled.
- `[quality.capability_long]`: hard-gated long-context capability results.
- `[quality.agentic_tools]`: single-turn OpenAI tool-call reliability results.
- `[benchmark.agentic_session]`: multi-turn tool-use session benchmark results.

## Agentic Session

`agentic-session` is a deterministic multi-turn coding-agent benchmark inspired
by the club-3090 agentic benchmark and shaped from a captured Claude Code
session against Lucebox. It complements `agentic-tools`:

- `agentic-tools` proves a short, single-turn request emits the requested
  OpenAI-format tool call.
- `agentic-session` uses Anthropic Messages, matching Claude Code's streamed
  `tool_use` plus replayed `tool_result` blocks. It records how history size,
  request size, first-content latency, wall time, decode speed, and tool-call
  reliability change as context accumulates. The server currently emits
  aggregated content blocks after decoding, so `first_content_ms` is the first
  streamed assistant block, not daemon token-level TTFT.

The benchmark writes `bench-agentic-session.json` and the snapshot exporter
normalizes it into:

- `[benchmark.agentic_session]`: fixture identity, pass rate, max prompt tokens,
  and final wall-time growth versus turn 1.
- `[benchmark.agentic_session.turn.N]`: per-turn aggregate prompt tokens,
  first-content latency, wall time, request/history size, decode TPS, tool
  calls, and tool-result size.
- `[benchmark.agentic_session.row.N]`: raw per-session/per-turn rows.

`lucebox profile` has a single canonical registry. It keeps multiple versions
of each result keyed by code/image/model/hardware/tunables/props/command/schema
hash. `--force-refresh` writes another result for the same current hash;
snapshot export uses only the newest matching result while preserving older
results for variance and historical percent comparisons.
