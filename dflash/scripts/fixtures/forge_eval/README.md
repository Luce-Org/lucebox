# forge_eval — vendored forge-guardrails eval harness + runtime

This directory hosts a vendored copy of the
[antoinezambelli/forge-guardrails](https://github.com/antoinezambelli/forge)
project, version **0.7.1**, used by `bench_http_capability.py --area forge`.

## Row schema (post per-iteration capture)

`bench_http_capability.py --area forge` emits one row per scenario in
`payload["rows"]`. Each row now matches the depth of the `ds4-eval` rows
(http_status, finish_reason, prompt_tokens, completion_tokens, timings,
prompt, output, …) AND carries an `iterations[]` array with one entry
per `client.send()` call forge made inside the scenario.

Scenario-row schema:

```python
{
    "area":              "forge",
    "source":            "forge-guardrails@0.7.1-vendored",
    "id":                "<scenario name>",
    "name":              "forge/<scenario name>",
    "kind":              "tool-calling",

    # Aggregate verdict
    "status":            "passed" | "failed" | "error",
    "ok":                bool,
    "graded_pass":       bool,
    "strict_pass":       bool,
    "format_pass":       bool,
    "semantic_hint":     False,  # not meaningful for forge
    "semantic_pass":     False,  # not meaningful for forge
    "given":             "PASS" | "FAIL" | "<error_type>",
    "correct":           ["PASS"],

    # Aggregate timing / tokens (summed across iterations)
    "wall_s":            float,        # total scenario wall
    "prompt_tokens":     int | None,   # sum of per-iteration input_tokens
    "completion_tokens": int | None,   # sum of per-iteration output_tokens
    "thinking_tokens":   0,            # forge runs without --think
    "content_tokens":    int | None,   # = completion_tokens - thinking_tokens
    "iterations_used":   int,          # number of model calls
    "timings": {                       # aggregated (when backend emits)
        "prefill_ms":             float,   # SUM across iterations
        "decode_ms":              float,
        "decode_tokens_per_sec":  float,   # weighted by decode_ms
        "prefill_tokens_per_sec": float,   # weighted by prefill_ms
    } | None,
    "timed_out":         bool,
    "finish_reason":     "stop" | "tool_calls" | "length" | None,  # of the final iteration
    "close_kind":        None,         # forge does not use thinking-budget close
    "http_status":       200 | <error code> | None,  # of the final iteration
    "reasoning_content": "",           # concat of per-iteration reasoning text

    # Final-iteration content
    "prompt":  "<final-iteration request body messages, json-serialised>",
    "output":  "<final-iteration model text>",

    # Per-iteration detail
    "iterations": [
        {
            "i":                 1,
            "wall_s":            2.711,
            "http_status":       200,
            "finish_reason":     "tool_calls",
            "stop_reason":       "tool_use",   # raw anthropic
            "prompt_tokens":     415,
            "completion_tokens": 127,
            "tool_calls": [
                {"id": "call_…", "name": "get_country_info",
                 "arguments": {"country": "France"}},
            ],
            "timings": {...} | None,
            "reasoning_content": "<text emitted alongside tool_use>",
            "prompt": "<json-serialised request messages>",
            "output": "<model text when no tool_use was emitted>",
            "error": None | "<APIError class>: <msg>",
        },
        ...
    ],
}
```

`payload["forge_results"]` is preserved for backward compatibility with
older consumers but is now redundant — every field it surfaces is also
present in the scenario row.

### Consuming `iterations[]`

To replay a scenario's request/response chain:

```python
import json
data = json.load(open("forge-run.json"))
for row in data["rows"]:
    if row["area"] != "forge":
        continue
    print(row["name"], row["status"], row["iterations_used"], "iters")
    for it in row["iterations"]:
        # `prompt` is the json-serialised messages forge sent on this call,
        # so the assistant tool_call / tool_result history accumulates as
        # `i` advances.
        request = json.loads(it["prompt"])
        names = [tc["name"] for tc in it["tool_calls"]]
        print(f"  iter {it['i']}: pt={it['prompt_tokens']} "
              f"ct={it['completion_tokens']} "
              f"finish={it['finish_reason']} tools={names}")
```

When the backend is the dflash cpp-server (post-#37), each iteration's
`timings` sub-block carries `prefill_ms` / `decode_ms` /
`decode_tokens_per_sec`; OR and Anthropic backends omit it (the field
is `None`). The scenario-level `timings` aggregate sums `prefill_ms`
and `decode_ms` and weights `decode_tokens_per_sec` by `decode_ms`.

The hook itself lives in `bench_http_capability.run_forge_area` —
specifically the `_RecordingAnthropicClient` subclass that overrides
`AnthropicClient.send` to grab the raw SDK `Message` (with
`stop_reason`, `usage`, `usage.timings`, content blocks) before
`_parse_response` collapses it into forge's `LLMResponse` shape.

## Why vendor?

The forge-guardrails PyPI wheel only ships the runtime under `src/forge/`.
The **eval harness** — the scenarios under `tests/eval/scenarios/`, the
ablation presets in `tests/eval/ablation.py`, and the `run_eval` driver in
`tests/eval/eval_runner.py` — is not packaged. To run the forge tool-calling
scenarios against our self-hosted server, we had to vendor those files.

Originally we vendored only the eval harness and depended on
`forge-guardrails` on PyPI for the runtime. We then inlined the runtime
itself (under [`_forge/`](_forge)) so the bench has zero forge-on-PyPI
dependencies — only the `anthropic` SDK needs to be installed (via
`dflash[eval]`).

## Layout

```
forge_eval/
  __init__.py                # docstring + provenance notes
  ablation.py                # ablation presets (vendored from tests/eval/ablation.py)
  eval_runner.py             # run_eval driver (vendored from tests/eval/eval_runner.py)
  scenarios/                 # vendored from tests/eval/scenarios/
    __init__.py
    _base.py
    _plumbing.py
    _model_quality.py
    _model_reasoning.py
    _compaction.py
    _compaction_chain.py
    _stateful_plumbing.py
    _stateful_model_quality.py
    _stateful_model_reasoning.py
    _stateful_relevance.py
  _forge/                    # vendored runtime (src/forge/) subset
    LICENSE                  # upstream MIT
    __init__.py              # version banner
    errors.py
    server.py                # ServerManager / BudgetMode
    clients/
      __init__.py            # re-exports base only
      base.py
      anthropic.py           # AnthropicClient used by --area forge
    context/
      __init__.py
      hardware.py
      manager.py
      strategies.py
    core/
      __init__.py
      messages.py
      workflow.py
      runner.py              # WorkflowRunner
      inference.py
      steps.py
    guardrails/
      __init__.py
      error_tracker.py
      response_validator.py
      step_enforcer.py
      nudge.py
      guardrails.py
    prompts/
      __init__.py
      nudges.py
      templates.py
```

## What is NOT vendored

The runtime subset under `_forge/` omits upstream modules our bench does
not exercise:

- `forge.proxy.*` — Anthropic-OpenAI proxy server; we drive the server
  directly via the Anthropic SDK.
- `forge.clients.{llamafile,ollama,sampling_defaults}` — only the
  Anthropic client is used; `llamafile`/`ollama` would also drag in
  `httpx` configuration the bench doesn't need.
- `forge.core.slot_worker` — concurrent slot worker, unused by `run_eval`.
- `forge.tools.*` — built-in tools (`respond`); the scenarios define
  their own tools.

## Bump path

When upstream ships 0.8 with breaking changes, re-sync deliberately:

1. Install the new release in a scratch venv (`pip install
   forge-guardrails==X.Y.Z`).
2. Copy the affected `src/forge/*` files into `_forge/`, prepending the
   NOTICE header and rewriting `from forge.X` imports to relative form.
3. Copy the corresponding `tests/eval/*` files into the parent directory
   (`forge_eval/`), rewriting `from forge.X` imports to address the
   vendored runtime at `..._forge.X` and re-stubbing `_compute_cost` if
   that pricing helper is still used.
4. Re-run `bench_http_capability.py --area forge --help` to smoke-test
   the imports, then run a real scenario pass against a local server.

The original eval harness re-sync notes are also captured at
[`__init__.py`](__init__.py).
