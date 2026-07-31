# Local formal-contract pilot

This directory defines two deterministic proof capsules and the local planning
tools that select them. This change is deliberately inert at repository level:
it adds no GitHub Actions workflow, required status check, branch-protection
rule, credentials, or AI integration. CI integration is a separate, later
change for maintainers to review.

The capsules run against the production prefix-cache transition boundary
introduced by the preceding prefix-cache correctness change. On a standalone
checkout of this component branch, registry and planner checks are meaningful,
but compiling or verifying the capsules requires that production change.

## Current capsules

`prefix-cache-inline` checks the production
`InlinePrefixCacheState` prepare, confirm, and exact-lookup path for symbolic
cache capacity, branch, and prefix depth. Its checked properties and exclusions
are documented in
[`prefix_cache/PROPERTIES.md`](prefix_cache/PROPERTIES.md).

`prefix-cache-abort-hole` checks the scalar free-slot selector. The complete
local plan also compiles and runs the paired native regression against the
selected source revision, covering the production
prepare/confirm/prepare/abort/prepare sequence. The formal and native
guarantees are separated in
[`prefix_cache/ABORT_HOLE_PROPERTIES.md`](prefix_cache/ABORT_HOLE_PROPERTIES.md).

Only behavior named in those property documents is claimed as checked.

## Policy files

[`manifest.toml`](manifest.toml) is the compatibility description consumed by
the verifier's legacy local mode.

[`contracts/registry.toml`](contracts/registry.toml) records the same two
capsules as deterministic templates, along with their source triggers, bounds,
mutable implementation paths, immutable contract paths, and critical-path
routing metadata. [`contracts/README.md`](contracts/README.md) describes the
registry and planner in detail.

The targets retain `policy = "required"` as their intended planner
classification. In this PR that value is data used only by local tools: nothing
invokes the planner automatically and it has no effect on merges. Any future
CI change must define and review the trust boundary separately before relying
on this metadata.

## Local validation

Registry and planner validation do not require Docker:

```bash
python3 scripts/formal_plan.py validate
python3 scripts/formal_plan.py plan \
  --changed-path server/src/server/prefix_cache_state.h
python3 -m unittest formal/contracts/tests/test_formal_plan.py -v
```

The full local verifier uses the immutable verifier image declared in the
registry. It requires Docker and a committed checkout containing the preceding
prefix-cache production change:

```bash
./scripts/formal.sh --all
./scripts/formal.sh --nightly
./scripts/formal.sh --all --legacy
./scripts/formal_mutation_test.sh
```

`--base-sha REVISION` is available for local comparison only after that
revision contains the registry; it is not a bootstrap command for this first
registry change.

Results are written to `.formal-results/`. Set
`LUCEBOX_FORMAL_IMAGE` only when deliberately testing a different companion
image.

Local verifier containers run without network access or Linux capabilities,
with a read-only repository mount and writable temporary plan/result
directories. Immutable image digests make local runs reproducible; accepting
the companion image's source, ownership, and release process remains an
explicit maintainer decision before CI integration.

## Adding a contract

1. Extract a dependency-light production boundary; do not verify a duplicate
   implementation.
2. Write a deterministic template and a property document that separates
   checked properties from exclusions.
3. Add a deterministic native regression for integration behavior that is not
   captured by the scalar contract.
4. Declare exact symbols, triggers, PR/nightly bounds, mutable paths, and
   contract paths in the registry.
5. Keep the compatibility manifest and registry execution settings aligned.
6. Run both bounds and demonstrate mutation sensitivity.

A later CI proposal may build on these files, but it must independently review
base-locked policy loading, exact-head verification, failure behavior,
artifact handling, and branch-enforcement rollout. No such integration is part
of this PR.
