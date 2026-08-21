# Agent-turn caching and optional early tools

`--agent-turn-cache` accelerates the repeated-prompt pattern used by coding
agents. During a tool-enabled request, Lucebox checkpoints the ordinary
prompt-end prefill state. If the model emits a real parsed tool call, the server
restores that checkpoint after delivering the response and prefills the
canonical completed-turn suffix into a normal cache slot. The next request
restores that state and prefills only the appended tool result. It works with
read-only and mutating client tools because the server does not execute the
tool.

Automatic early dispatch is a separate, optional mechanism. It can start an
operator-approved read-only tool as soon as the model emits a complete call,
then keep the private result until the final parsed call matches. It is useful
only when the tool is slow enough to overlap with the remaining generation.

For early dispatch, "verified" has a narrow meaning: Lucebox exposes a result only when the final
tool name and canonical JSON arguments exactly match the call that ran. It does
not prove that external data stayed unchanged during generation, and it does
not undo side effects. Use it only where reading a turn-consistent repository
snapshot is acceptable, or make the executor attach a conservative
`_speculation_fresh_until_unix_ms` deadline. The server therefore accepts only
tools the operator explicitly declares read-only. Writes, arbitrary shell
commands, tests with side effects, and deployment actions are outside this
interface's contract.

## Start the server

Agent-turn caching needs one option on a normal `dflash_server` command:

```text
--agent-turn-cache
```

Early dispatch additionally needs:

```text
--early-dispatch
--prefetch-prefill
--tool-spec-executor /absolute/path/to/coding_tool_executor.py
--tool-spec-read-only read_file
--tool-spec-read-only search_code
--tool-spec-read-only list_files
--tool-spec-cpu-affinity 14-15
--tool-spec-timeout-ms 5000
--tool-spec-max-executors 8
```

The reference adapter and wrapper are in
`optimizations/ooo_spec_lucebox5_cpu/agentic_coding/`. For example:

```bash
optimizations/ooo_spec_lucebox5_cpu/agentic_coding/launch_server.sh \
  ./server/build/dflash_server model.gguf [your normal model options]
```

To add the optional early executor, set
`DFLASH_ENABLE_EARLY_DISPATCH=1`, `DFLASH_TOOL_WORKSPACE`, and
`DFLASH_TOOL_CPU_AFFINITY` as shown in the optimization README.

Automatic early dispatch is Linux-only and requires a non-empty CPU affinity.
Pin the model process first, then set `DFLASH_TOOL_CPU_AFFINITY` to a disjoint
set. Startup compares the masks and fails closed on overlap.

## Agent-turn cache request and telemetry

Starting the server with `--agent-turn-cache` enables the feature for tool-using
requests without requiring client changes. Set the top-level
`"agent_turn_cache": false` to form a control arm, or `true` to state the cache
arm explicitly. A true value is rejected when the server capability is off.
Generated-turn slots are invisible to a control request, while ordinary
snapshots produced by a control request are invisible to the cache arm. This
prevents request-scoped inline-cache measurements from being warmed by the
other arm. Disable the separate full-prompt cache for a paired benchmark; the
included runner records `/props` and its publication gate verifies that
configuration.

Normal stateless follow-ups work across all three API shapes: Chat Completions
uses `assistant.tool_calls` plus a `tool` message, Responses uses
`function_call` plus `function_call_output`, and Anthropic Messages uses
`tool_use` plus `tool_result`. Lucebox normalizes each shape back into the model
transcript. Structured follow-ups replay the exact raw generated call by call
ID while the process is running and use a canonical reconstruction after a
restart; clients never need to send sampled token IDs or a server-owned
session handle.

The server commits a continuation snapshot only after the final response
parser finds at least one real tool call. It renders and tokenizes that
completed turn exactly as the next stateless API request will. The rendering
must begin with the checkpointed prompt token-for-token; if re-tokenization
changes a BPE boundary, the server fails closed and does not cache the turn.
It never rewrites a later caller prompt to the token IDs sampled during decode.
It also deliberately does not publish live post-decode state: restoring the
prompt checkpoint and prefilling the canonical suffix through the ordinary
path keeps the cached continuation equivalent to recomputing the transcript.
If checkpointing or canonical replay fails, the request succeeds normally and
the next turn is a cache miss.

Canonical replay starts after the response has been written, so it can overlap
client-side tool execution. If a very fast tool returns first, its next request
waits behind the short replay. Measure the full follow-up request and the whole
agent loop—not just backend prefill—to include that cost.

Non-streaming responses and final streaming usage expose:

```json
{
  "usage": {
    "timings": {
      "agent_turn_cache_hit": true,
      "cache_hit": true,
      "cached_prefix_tokens": 958,
      "prefilled_tokens": 95,
      "effective_prompt_tokens": 1053,
      "prefill_ms": 3960.1
    }
  }
}
```

`cached_prefix_tokens` derives from the backend's confirmed restored-token
count; a cache candidate that falls back to a full prefill reports zero and no
hit. `agent_turn_cache_hit` is true only for a restored snapshot that includes
generated assistant-turn state.

## Opt in to early read-only execution

Declare the same tools the executor implements and add the top-level boolean:

```json
{
  "model": "deepseek-v4-flash",
  "messages": [
    {"role": "user", "content": "Find the cache expiry test before editing."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_code",
        "description": "Find a literal string in repository files.",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {"type": "string"},
            "path": {"type": "string"}
          },
          "required": ["query"],
          "additionalProperties": false
        }
      }
    }
  ],
  "tool_choice": "auto",
  "automatic_tool_speculation": true,
  "stream": false
}
```

Automatic execution is off when the field is absent or false. Do not send
`tool_speculation` in the same request: that caller-prediction entry point and
`automatic_tool_speculation: true` are mutually exclusive, and the server
returns HTTP 400 if both are enabled.

The field works on Chat Completions, Anthropic Messages, and the Responses API.
SDKs that support an `extra_body` option normally merge this field into the
top-level JSON request.

## Consume a hit

A non-streaming response contains one entry per emitted call:

```json
{
  "dflash_early_dispatch": [
    {
      "name": "search_code",
      "arguments": {"query": "CachePolicy", "path": "src"},
      "call_id": "call_abc",
      "status": "hit",
      "result": {"ok": true, "value": {"matches": []}},
      "tool_message_content": "{\"matches\":[]}",
      "executor_wall_ms": 18.4,
      "commit_wait_ms": 0.0
    }
  ]
}
```

Match by `call_id`, not by array position. On `status: "hit"`, use
`tool_message_content` verbatim as that call's tool message. This lets a
completed prefetch match the next request exactly. On every other status, run
the tool normally and send its normal result. A client therefore needs only
this branch:

```python
item = early_dispatch_by_call_id.get(tool_call.id)
if item and item.get("status") == "hit":
    content = item["tool_message_content"]
else:
    content = run_tool_normally(tool_call)
messages.append({"role": "tool", "tool_call_id": tool_call.id,
                 "content": content})
```

Streaming responses deliver the same array in the
`dflash_tool_speculation.early_dispatch` extension immediately before the
normal terminal event:

- Chat Completions: a `data:` chunk with `dflash_tool_speculation`.
- Anthropic: `event: dflash_tool_speculation`.
- Responses API: `event: response.dflash_tool_speculation`.

Clients that ignore the extension remain correct and execute tools normally.

## Executor contract

Lucebox starts the configured executable directly, without a shell, with
`--dflash-tool-spec-v1`. It writes one newline-delimited request to stdin:

```json
{
  "protocol": "dflash.tool-speculation.v1",
  "request_id": "chatcmpl_...",
  "mode": "speculative",
  "resource_percentage": 100,
  "cpu_affinity": [14, 15],
  "call": {"name": "read_file", "arguments": {"path": "src/cache.py"}}
}
```

The executor writes one envelope to stdout:

```json
{"ok":true,"result":{"ok":true,"value":{"path":"src/cache.py"}}}
```

If the result object contains an integer
`_speculation_fresh_until_unix_ms`, Lucebox discards it after that Unix-time
deadline. This bounds reuse for indexes or workspaces that can change while a
model turn is running; it is not a substitute for a versioned repository
snapshot when strict point-in-time consistency is required.

Model-generated arguments are untrusted input. The executor must validate its
JSON shape, confine paths to a workspace, reject symlink escapes, limit input
and output sizes, verify its non-empty CPU mask, and expose only fixed read-only
operations. The reference coding adapter demonstrates those checks. Lucebox
also closes inherited file descriptors, passes a minimal environment, bounds
request writes, enforces an active wall-time lifetime cap, limits result size,
and kills the complete child process group on timeout or cancellation.

The optional `tool_speculation` request object is for a caller that knows one
concrete call before model generation. It requires a qualified
`--tool-spec-profile`; caller confidence never makes a result authoritative.
Most coding-agent integrations should use automatic early dispatch instead.

## What improves coding-agent latency

- **Agent Turn Cache:** prefills the stateless request's canonical completed-turn
  suffix from a prompt-end checkpoint, so the next request restores the
  completed turn and prefills only the new tool result. It can help with fast
  local tools when repeated-context work dominates, and it requires no
  server-side executor.
- **Early dispatch:** independent repository reads or index queries can overlap
  the rest of tool-call generation and one another.
- **Opportunistic next-turn prefill:** Lucebox prepares the exact tool-result
  turn only while the worker is idle. If a request arrives first, the prefetch
  is discarded and the real request takes priority. Logs count a prefetch only
  when it completed before any request arrived.

Fast local reads may show little early-dispatch gain, while long repeated
prompts can still benefit from Agent Turn Cache. The included benchmark disables
early execution, runs repeated `control/cache` pairs, requires identical
assistant/tool transcripts, tool names, arguments, and byte-identical result
traces (including assistant reasoning) plus backend-confirmed cache hits,
isolates earlier pairs and prior runs with a recorded run-unique opaque ID
shared by both arms, records the server capability snapshot, and reports
follow-up prefill, full follow-up request, and whole-loop confidence intervals:

```bash
cd optimizations/ooo_spec_lucebox5_cpu/agentic_coding
DFLASH_BENCH_HARDWARE='host/GPU/CPU description' \
DFLASH_SERVER_BUILD_ID='git commit or immutable image digest' \
  python3 coding_bench.py run1 --repetitions 5
python3 coding_summary.py run1
```

The default fixture names the files to inspect so it tests engine behavior
rather than an unstable planning policy. It is a smoke test, not publication
evidence. `tasks_lucebox.json` can be run against `../../../server` for a
separate real-codebase smoke; a headline still requires a larger representative
coding-agent workload and every summary gate.

## Status and fallback reasons

`hit` means the private result matched and is safe to consume. `miss`,
`skipped`, `deferred`, `failed`, and `cancelled` all mean the client must run
the authoritative tool normally. Inspect `reason` and `detail` for diagnosis;
common values include `tool_not_declared_read_only`,
`resource_isolation_required`, `executor_saturated`, `invocation_mismatch`,
`executor_timeout`, and `executor_lifetime_exceeded`.
