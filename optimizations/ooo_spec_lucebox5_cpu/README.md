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
`tool_speculation` request extension (unchanged), and opt out per request
with `"automatic_tool_speculation": false`.

## Safety

- Only tools named by `--tool-spec-allow` ever run speculatively; the
  executor is spawned without a shell, with a minimal environment, closed
  descriptors, an optional pinned CPU lane disjoint from the model
  (`--tool-spec-cpu-affinity`), launch-based lifetime caps and a
  commit-based result deadline (`--tool-spec-timeout-ms`).
- `--tool-spec-profile` (a measured interference profile) is required only
  for client-supplied predictions with confidence < 1; authoritative
  early-dispatch launches need no profile.

## Measured (Lucebox5, DeepSeek-V4-0731 + DSpark, live public APIs)

10 multi-step tasks over keyless real APIs (geocoding, weather, World Bank,
Wikipedia, FX; 0.1-0.9 s latency), control vs speculative arm on the same
server, answers identical (`realapi/rest_bench.py`, artifacts in
`realapi/results/`):

| | control | speculative |
|---|---|---|
| tool calls hidden | 0/17 | **17/17** |
| client tool wait | 3.0 s | **0.0 s** |
| 10-task wall (end-of-turn snapshots) | 174 s | **156 s** (-10%, up to -27% on 3-4-turn tasks) |

Many-call tasks (4-8 calls each, `realapi/dag_bench.py`): parallel calls +
early dispatch complete in 11 model turns vs 32 for one-call-per-turn,
20/20 calls hidden, 4/4 vs 2/4 tasks correct, 1.28x end to end.

Gains scale with tool latency and call count: on benchmarks with
millisecond in-memory tools there is nothing to hide, while each second of
real API latency is a second saved per call. An earlier revision's
pre-generation predictor (Qwen3-0.6B) was measured at 3/17 hits on these
APIs and 0/75 on terminal-bench against ~0.5 s serial cost per turn, and
was removed; the history preserves it.

## Reproduce

```bash
# server (wraps the qualified 0731 launcher's binary selection)
BUILD_DIR=$PWD/realapi TOOL_SPEC_EXECUTOR=$PWD/realapi/rest_tool_executor.py \
  TOOL_SPEC_ALLOW=geocode_city,get_weather,country_info,wikipedia_summary,exchange_rate \
  ./realapi/launch_server.sh

# paired benchmark (control vs speculative, alternating order, primed)
python3 realapi/rest_bench.py run1 --tasks 10
python3 realapi/rest_summary.py run1

# many-call / dependency benchmark
python3 realapi/dag_bench.py run2 --small --tasks 4
```

