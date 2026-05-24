# forge_eval — vendored forge-guardrails eval harness + runtime

This directory hosts a vendored copy of the
[antoinezambelli/forge-guardrails](https://github.com/antoinezambelli/forge)
project, version **0.7.1**, used by `bench_http_capability.py --area forge`.

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
