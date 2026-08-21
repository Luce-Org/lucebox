# Speculative tool execution

Lossless acceleration of tool-calling requests. The model stays authoritative:
speculative work is exposed only on an exact canonical match, and every failure
path falls back to the normal request flow.

Three engine flags, designed to be enabled together:

- `--early-dispatch` — every allowlisted tool call is launched through the
  isolated executor lane the moment its call block
  (`<function_call>`, `<tool_call>`, `<function=>`) closes in the token
  stream. N calls per response run concurrently; each commits independently
  against the parsed authoritative calls. Results are returned as
  `dflash_early_dispatch` (JSON body) / inside the
  `dflash_tool_speculation` SSE extension event. Calls whose arguments
  reference earlier results with `"$k.path"` placeholders are resolved and
  executed server-side, wave by wave.
- `--end-turn-snapshot` — after a tool-enabled generation the server
  snapshots prompt+output into the inline prefix cache, so the next turn of
  the conversation restores everything instead of a stale boundary.
- `--prefetch-prefill` — after a turn whose early-dispatched results all
  committed, the next request is fully determined (conversation + assistant
  output + canonical tool messages). The server renders it, prefills it and
  caches the KV before the client's next request arrives; a client that
  echoes each hit's `tool_message_content` string starts its next turn at
  decode speed. Any mismatch silently takes the normal path.

Clients can also supply a concrete prediction up front via the
`tool_speculation` request extension (unchanged). Automatic early dispatch is
off by default and requires an explicit per-request opt-in with
`"automatic_tool_speculation": true`.

## Safety

- Automatic early dispatch is fail-closed per request: omitting
  `automatic_tool_speculation` leaves it disabled. The server never infers
  whether a tool is safe; the operator supplies exact names with
  `--tool-spec-allow`.
- Only tools named by `--tool-spec-allow` ever run speculatively; the
  executor is spawned without a shell, with a minimal environment, closed
  descriptors, an optional pinned CPU lane disjoint from the model
  (`--tool-spec-cpu-affinity`), launch-based lifetime caps and a
  commit-based result deadline (`--tool-spec-timeout-ms`). Executors can also
  attach `_speculation_fresh_until_unix_ms`; expired live-data results are
  rejected immediately before commit and run again through the normal path.
- Allow immediate execution only for read-only tools. A side-effecting
  executor must stage the change until the engine sends `commit`, or provide
  transactional rollback; idempotency alone does not make wrong arguments
  safe.
- `--tool-spec-profile` (a measured interference profile) is required for
  every client-supplied prediction. Only calls already emitted by the model
  and launched through early dispatch are authoritative without a profile;
  a client cannot obtain that status by reporting confidence 1.

## Measured (Lucebox5, DeepSeek-V4-0731 + DSpark, live public APIs)

10 multi-step tasks over keyless real APIs (geocoding, weather, World Bank,
Wikipedia, FX; 0.1-0.9 s latency), control vs speculative arm on the same
server (`realapi/rest_bench.py`, historical performance traces in
`realapi/results/`):

| | control | speculative |
|---|---|---|
| tool calls hidden | 0/17 | **17/17** |
| client tool wait | 3.0 s | **0.0 s** |
| 10-task wall (end-of-turn snapshots) | 174 s | **156 s** (-10%, up to -27% on 3-4-turn tasks) |

With `--prefetch-prefill` on top (full stack, `realapi/results/rest_full_stack_10tasks.json`):
paired speedup **x1.44 aggregate / x1.30 median**, 17/17 calls hidden, 21.4 s
of client tool wait removed, and **72.3 s of prefill moved off the critical
path across 15 prefetched turns**.

Those committed traces predate the current result-derived correctness gate and
are latency evidence only. Current `rest_bench.py` and `dag_bench.py` require
successful tool execution plus values derived from every answer-producing tool
result; rerun them before making accuracy claims. The earlier corrupted DAG
trace was removed rather than presented as correctness evidence.

Gains scale with tool latency and call count: on benchmarks with
millisecond in-memory tools there is nothing to hide, while each second of
real API latency is a second saved per call. An earlier revision's
pre-generation predictor (Qwen3-0.6B) was measured at 3/17 hits on these
APIs and 0/75 on terminal-bench against ~0.5 s serial cost per turn, and
was removed; the history preserves it.

## Reproduce

```bash
# server (wraps the qualified 0731 launcher's binary selection)
TBSPEC_EXECUTOR=$PWD/realapi/rest_tool_executor.py \
  TBSPEC_ALLOW=geocode_city,get_weather,country_info,wikipedia_summary,exchange_rate \
  ./realapi/launch_server.sh

# paired benchmark (control vs speculative, alternating order, primed)
python3 realapi/rest_bench.py run1 --tasks 10
python3 realapi/rest_summary.py run1

# many-call / dependency benchmark
python3 realapi/dag_bench.py run2 --small --tasks 4
```
