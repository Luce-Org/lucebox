"""Vendored copy of forge-guardrails eval harness + runtime.

The forge-guardrails PyPI wheel ships ``src/forge/`` (the runtime) but
NOT the eval harness (scenarios, ablation presets, ``run_eval`` driver)
which lives under ``tests/eval/``. Originally we vendored only the eval
harness and depended on ``forge-guardrails`` via pip for the runtime.
We now inline the runtime as well, under ``_forge/``, so the bench can
run with only the ``anthropic`` SDK installed.

Sources:
 - Eval harness: https://github.com/antoinezambelli/forge/tree/main/tests/eval
 - Runtime: https://github.com/antoinezambelli/forge/tree/main/src/forge
Vendored from forge-guardrails 0.7.1. Upstream MIT license preserved
at ``_forge/LICENSE``.

Local modifications:
 - ``tests.eval.ablation`` imports rewritten to relative
   (``from .ablation import ...``).
 - ``tests.eval.scenarios`` imports rewritten to relative
   (``from .scenarios import ...``).
 - All ``from forge.X import Y`` imports rewritten to address the
   vendored runtime at ``forge_eval._forge.*`` via relative imports.
 - The inline ``from tests.eval.batch_eval import _compute_cost`` import
   inside ``run_eval`` is replaced with a stub that returns 0.0; the
   real pricing table is Anthropic-API-only and not meaningful for a
   self-hosted dflash bench.
 - ``main()`` (forge's standalone CLI) is dropped — bench_http_capability
   owns the entrypoint and only needs ``run_eval`` / ``RunResult`` /
   ``EvalConfig``.
 - The runtime subset under ``_forge/`` excludes upstream modules the
   eval harness does not exercise: ``forge.proxy.*``,
   ``forge.clients.{llamafile,ollama,sampling_defaults}``,
   ``forge.core.slot_worker``, and ``forge.tools.*``.

Bump path: when upstream ships 0.8 with breaking changes, the vendored
copy may diverge; re-sync deliberately by re-running the vendor process
documented in this directory's ``README.md``.
"""
