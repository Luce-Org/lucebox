# Local formal-contract registry

`registry.toml` is the source of truth for deterministic local selection of
the approved minimal formal boundaries. It coexists with `../manifest.toml`,
which preserves compatibility with the verifier's legacy local mode.

Each target records the exact production symbol and signature, approved
template, execution bounds, mutable implementation paths, immutable contract
paths, and paired native regression. Templates use only literal `{{ID}}`,
`{{SYMBOL}}`, `{{SIGNATURE}}`, and optional declared variables. The planner
substitutes those tokens deterministically; it does not generate contracts.

This PR does not install a workflow or alter branch protection. The targets'
`policy = "required"` values are planner metadata for a possible future
integration, not repository enforcement.

## Registered boundaries

The registry contains two complementary prefix-cache capsules:

- `prefix-cache-inline` maps to the prepare, confirm, and exact-lookup
  harness.
- `prefix-cache-abort-hole` checks the bounded scalar free-slot selector and
  pairs it with the native regression for the production
  prepare/abort/prepare integration point.

Both templates call production code rather than duplicate it. The component
depends on the production state extraction and regression from the preceding
prefix-cache correctness PR.

`[[critical_paths]]` also describes narrow state-machine areas that deserve
review when changed without a matching target. An unmatched critical path is
reported as an advisory coverage gap, never as verified. `watch_paths` and
`include_roots` are routing hints only; they do not expand formal coverage.

## Local validation

```bash
python3 scripts/formal_plan.py validate
python3 scripts/formal_plan.py plan \
  --changed-path server/src/server/prefix_cache_state.h
python3 -m unittest formal/contracts/tests/test_formal_plan.py -v
```

The `emit` command renders selected templates into an output directory and
records their hashes. It does not invoke ESBMC or modify the manifest lane.

## Boundary for a later CI change

If maintainers later approve CI integration, that change must load the
registry and selected template blobs from the accepted merge base (or an
equivalent protected artifact), verify the exact proposed source revision, and
record the immutable inputs in its plan. A contract-changing PR must not be
allowed to redefine the policy used to judge itself.

Those trust and enforcement mechanics are intentionally absent here. They
belong in a separate CI/security review and rollout.
