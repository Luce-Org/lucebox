# Agent Turn Cache for coding agents

Coding agents repeatedly resend a growing transcript. After the model emits a
tool call, the next request contains the same prompt, that generated assistant
turn, and a comparatively small tool result. A normal prompt cache can reuse
the old input, but it does not own the model state created while generating the
tool call.

`--agent-turn-cache` checkpoints normal prompt-end state before decode. After a
real tool call is parsed, Lucebox restores that checkpoint and prefills the
canonical completed-turn suffix into a cache slot. On the next request it
restores the completed turn and prefills only the appended tool result. No
tool is executed by the server, and the client tool loop remains unchanged.

The implementation renders and tokenizes the completed turn exactly as the
next stateless API request will. That tokenization must extend the checkpointed
prompt without changing any prior token; a BPE-boundary change fails closed and
the next request uses normal prefill. Lucebox never rewrites caller input to the
token IDs sampled during decode. Canonical prefill also avoids trusting live
post-generation state, which can be numerically different from ordinary
transcript prefill. A hit is reported only after the backend confirms that it
restored the snapshot; any checkpoint or replay failure falls back to normal
prefill.

The canonical suffix replay starts after response delivery and can overlap the
client's tool execution. A fast tool can return before replay finishes, so the
benchmark measures both backend prefill and the full follow-up request.

## Start the server

Add the feature to a normal server command:

```bash
./server/build/dflash_server model.gguf \
  --agent-turn-cache \
  [normal model options]
```

The reference wrapper adds that flag for you:

```bash
optimizations/ooo_spec_lucebox5_cpu/agentic_coding/launch_server.sh \
  ./server/build/dflash_server model.gguf [normal model options]
```

When the server flag is present, agent-turn caching applies by default to
requests that produce a parsed tool call. Clients can run a request-scoped A/B
test without restarting the model:

```json
{
  "model": "deepseek-v4-flash",
  "messages": ["..."],
  "tools": ["..."],
  "agent_turn_cache": false
}
```

Set the field to `true` for the cache arm. Asking for `true` on a server that
was not started with `--agent-turn-cache` returns HTTP 400 instead of silently
running a different configuration. Control-origin ordinary snapshots are not
eligible in the cache arm, and the benchmark gives each pair a shared opaque
prompt ID derived from a recorded run-unique nonce so earlier pairs and prior
benchmark invocations cannot warm either measurement. The initial prompts are
identical, exact normalized assistant/tool transcript parity (including
reasoning) and equal served token counts gate every follow-up, and arm order
alternates. The runner records `/props`; the publication gate requires the
separate full-prompt cache to be disabled so it cannot warm the second arm
outside the request-scoped inline-cache isolation.

Every response exposes backend-confirmed work under `usage.timings`:

```json
{
  "agent_turn_cache_hit": true,
  "cache_hit": true,
  "cached_prefix_tokens": 958,
  "prefilled_tokens": 95,
  "effective_prompt_tokens": 1053,
  "prefill_ms": 3960.1
}
```

`agent_turn_cache_hit` is narrower than `cache_hit`: it is true only when the
restored entry contains a generated assistant tool-call turn. The token counts
come from the backend's actual restore, not from the cache candidate selected
by the HTTP layer.

## Agentic-coding benchmark

`agentic_coding/` contains a workspace-confined implementation of
`read_file`, `search_code`, and `list_files`, a deterministic repository
fixture, and a paired `control/cache` benchmark. Fixture prompts name the files
to inspect so the smoke test holds model planning variance constant. Both arms
execute tools on the client. Their only engine difference is the
`agent_turn_cache` request field.

```bash
cd optimizations/ooo_spec_lucebox5_cpu/agentic_coding
python3 -m unittest -v test_coding_tools.py

DFLASH_BENCH_HARDWARE='host/GPU/CPU description' \
DFLASH_SERVER_BUILD_ID='git commit or immutable image digest' \
  python3 coding_bench.py run1 --repetitions 5
python3 coding_summary.py run1
```

For an additional smoke over the real server tree:

```bash
python3 coding_bench.py lucebox-smoke \
  --workspace ../../../server \
  --tasks-file tasks_lucebox.json \
  --repetitions 1
```

## Engineering smoke (not a headline benchmark)

The final six-task fixture run on Lucebox5 used DeepSeek-V4-Flash-0731 on a
Radeon AI PRO R9700 plus Ryzen AI Max 395/8060S with ROCm 7.2.4. It measured
the clean engine commit `da8850f94d21ce713f07cdabc8694abec9bfe83f` and the
immutable server binary
`sha256:b04f77e78266301768d5faa43a42674c7ba68bea1aa0ee70c643466cbdcd639c`.
The server capability snapshot records Agent Turn Cache enabled and the
separate full-prompt cache disabled.

| Metric | Aggregate speedup | Paired bootstrap 95% interval |
|---|---:|---:|
| Eligible follow-up prefill | 2.49x | 1.95-2.97x |
| Eligible follow-up request | 1.70x | 1.27-2.14x |
| Whole coding-agent loop | 1.32x | 1.13-1.59x |

All 6/6 pairs were correct with identical normalized assistant/tool
transcripts, including reasoning and final answers. All 13/13 eligible cache
turns were backend-confirmed hits; control had none. The five-turn
`path_callers` task shows the intended long-loop behavior: eligible prefill
fell from 108.4 s to 32.1 s (3.38x), and the whole loop fell from 141.8 s to
73.2 s (1.94x). The short `parallel_reads` task cut prefill from 7.86 s to
6.15 s but had no measurable whole-loop gain, because generation dominated.

The raw artifact is
[`results/coding_deterministic-six-task-final.json`](agentic_coding/results/coding_deterministic-six-task-final.json)
with SHA-256
`db38f46ea503845a240c5fb2fbc1fdccd2c38e6bafb23ea9c11c129ce9c90aea`.
Every correctness, isolation, provenance, hit, token-work, and confidence gate
passed. The minimum-pair gate failed: 6 pairs is below the required 30. These
numbers are therefore engineering-smoke evidence, not a publishable headline.

A separate run read the real Lucebox server tree and correctly recovered the
Agent Turn Cache staging slots, hit telemetry, deprecated CLI alias, and
startup cache guard. It passed 3/3 exact pairs and 3/3 eligible hits. Its raw
artifact is
[`results/coding_lucebox-real-smoke-final.json`](agentic_coding/results/coding_lucebox-real-smoke-final.json)
with SHA-256
`7ea291da1bb59773f910d5029d4ea8eab97f3ac38ed17d8ca77c31a0b622daea`.
This second run is correctness evidence only; three pairs cannot support a
performance claim.

The summary rejects a publication claim unless all of these hold:

- every expected pair exists and both arms pass the evidence-derived answer
  checks;
- control and cache produce the same normalized assistant/tool transcript,
  including reasoning, the canonical sequence of tool names, arguments,
  byte-identical results, and final answer;
- corresponding follow-up prompts have equal token counts;
- every eligible cache follow-up reports a real agent-turn-cache hit and the
  control arm reports none;
- cached turns prefill fewer tokens;
- follow-up prefill time, full follow-up request time, and whole agent-loop
  time each have a positive bootstrap confidence interval;
- the engine, workload, hardware, inputs, and artifact are reproducible;
- the recorded server capability snapshot confirms Agent Turn Cache was on and
  the full-prompt cache was off for pair isolation.

The artifact records the run-isolation nonce. Pass `--run-id` (or
`DFLASH_BENCH_RUN_ID`) when an external harness needs a predetermined value;
otherwise the runner generates a fresh one so even an `--overwrite` rerun
cannot inherit exact prompt entries from an earlier invocation.

The fixture is an integration smoke test, not a headline workload. Publish a
performance number only from representative coding-agent tasks, enough paired
runs, and the complete artifact. There is intentionally no headline speedup in
this directory until those gates pass.

## Optional early read-only tools

Early tool dispatch is separate from Agent Turn Cache. It can overlap a slow,
operator-approved read-only tool with the tail of tool-call generation, but it
requires a confined executor and a CPU lane disjoint from model inference. It
is not required for turn-cache gains and is disabled in the cache benchmark.

To enable the reference executor as an additional experiment:

```bash
DFLASH_ENABLE_EARLY_DISPATCH=1 \
DFLASH_TOOL_WORKSPACE=/absolute/path/to/repository \
DFLASH_TOOL_CPU_AFFINITY=14-15 \
  ./agentic_coding/launch_server.sh \
  ../../server/build/dflash_server model.gguf [normal model options]
```

Do not declare shell commands, writes, package installation, tests with side
effects, or deployment actions as read-only. The complete executor protocol
and fallback contract are documented in
[`server/docs/TOOL_SPECULATION.md`](../../server/docs/TOOL_SPECULATION.md).
