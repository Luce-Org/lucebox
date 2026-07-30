# Approved per-PR formal contracts

`registry.toml` is the source of truth for the minimal formal boundaries that
may be selected automatically for a pull request.  It is intentionally not a
replacement for `../manifest.toml` yet: the manifest continues to drive the
existing deterministic capsule workflow during the dual-run migration.

Each target records the exact production symbol and signature, approved
template, execution bounds, mutable implementation paths, immutable contract
paths, and paired native regression.  Templates use only literal
`{{ID}}`, `{{SYMBOL}}`, `{{SIGNATURE}}`, and optional declared variables.  The
planner substitutes those tokens deterministically; it never asks a model to
write a required contract.

## Trust and migration rules

For a PR, the planner must read this registry and the selected template blobs
from the merge base (or a protected artifact identified by that base), then
record their hashes in its plan.  It must not trust a registry or template that
the PR itself changed.  A contract-change PR therefore runs the old approved
contract as the gate and reports its proposed new contract separately until
formal CODEOWNERS approve promotion.

The promoted entries cover two complementary inline prefix-cache capsules:

- `prefix-cache-inline` maps to the existing prepare/confirm/lookup harness.
- `prefix-cache-abort-hole` exhaustively checks the bounded scalar free-slot
  selector, then runs the immutable base regression against the exact head to
  guard the real `prepare`/`abort`/`prepare` integration point.

Three first-wave entries are registered as advisory while proof latency and
mutation sensitivity are established:

- `prefix-cache-full-lifecycle`
- `spec-commit-exactness`
- `kvflash-residency-map`

Their generated wrappers may include shared harness bodies. Every such body is
declared in `contract_paths`. Once accepted into protected base policy, the
candidate companion image materializes those authenticated base-revision blobs
into an isolated include tree before the PR/head production include paths. A
PR therefore cannot weaken a transitive harness include while retaining a
green result.

The templates intentionally call production code; they are not duplicate
implementations.  The legacy harness runs the same transition during the
planner/verifier dual-run so comparison remains meaningful.

`[[critical_paths]]` lists the cache, snapshot, streaming, and tool-hint state
machines.  A changed path in one of those areas that matches no approved target
is an **advisory coverage gap**, never a pass.  Paths outside the list are
reported as not applicable by the CI planner.

An area may also declare advisory `watch_paths` and bounded `include_roots`.
In CI, the companion planner uses watch matches to route suspicious new files
for review, but a match never means that a file is formally covered.  The
companion records exact boundary, trigger, include-adjacency, and unmodeled
relationships separately; only a verifier result for a declared target may
use the word `verified`.

## Local validation

```bash
python3 scripts/formal_plan.py validate
python3 scripts/formal_plan.py plan \
  --changed-path server/src/server/prefix_cache_state.h
python3 -m unittest formal/contracts/tests/test_formal_plan.py -v
```

`emit` is a local fixture aid: it renders selected protected templates into an
output directory and records their hashes.  It does not invoke ESBMC or modify
the existing manifest lane.
