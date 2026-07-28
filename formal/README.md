# Lucebox formal-verification pilot

This directory defines proof capsules for the
[`dusterbloom/lucebox-esbmc-ai`](https://github.com/dusterbloom/lucebox-esbmc-ai)
companion runner.

## Trust boundaries

The `Formal Verification / verify` check is deterministic:

- its verifier image and ESBMC release are pinned by SHA-256 digest;
- the Lucebox checkout is mounted read-only;
- the results directory is the only writable bind mount;
- the container has no network, Linux capabilities, or writable root;
- no model SDK or repository secret is present.

The `Formal AI Candidate` workflow is separate and advisory:

- it ignores fork failures and infrastructure failures;
- a secretless job first confirms that the deterministic report contains a
  real counterexample;
- the `formal-ai` environment requires approval before model access;
- archive paths, sizes, immutable contract hashes, and mutable patch paths are
  validated;
- the credential-bearing proposer never executes generated code;
- generated code is applied, compiled, run, and reverified only in a second
  networkless container after the proposer has exited;
- the lane can upload only a diagnosis and candidate patch. It cannot commit,
  comment, push, or open a pull request.

## Current capsules

`prefix-cache-inline` checks the production
`InlinePrefixCacheState` one-entry prepare/confirm/exact-lookup path for
symbolic cache capacity, branch, and prefix depth. Its precise guarantees and
exclusions are in
[`prefix_cache/PROPERTIES.md`](prefix_cache/PROPERTIES.md).

`prefix-cache-abort-hole` checks every bounded cursor and occupancy pattern for
the production free-slot selector. It was added from a current regression in
which an aborted reservation left a free slot, but the next reservation
selected and invalidated a still-committed snapshot. Its contract is in
[`prefix_cache/ABORT_HOLE_PROPERTIES.md`](prefix_cache/ABORT_HOLE_PROPERTIES.md).

The dependency-free native test covers the wider transition matrix:
abort, cancellation, stale lookup, invalid confirmation, clear, full-capacity
slot reuse, and prefix-aware eviction. Only behavior explicitly named by a
capsule contract is described as model checked.

## Local use

Docker will pull the immutable verifier declared in `manifest.toml`.

```bash
./scripts/formal.sh --all
./scripts/formal.sh --nightly
./scripts/formal.sh --base-sha origin/main
```

Set `LUCEBOX_FORMAL_IMAGE` only when deliberately testing a new companion image.
Results are written to `.formal-results/`.

## Adding a capsule

1. Extract a dependency-light production transition boundary; do not verify a
   toy reimplementation.
2. Write the bounded harness and a property document that distinguishes checked
   properties from exclusions.
3. Add a deterministic native regression test.
4. Declare trigger, mutable, and immutable contract paths in `manifest.toml`.
5. Run both pull-request and nightly bounds locally.
6. Treat timeout and tool errors as failures, never as passes.

## Promotion after the proving period

Leave the check non-required for at least 14 days. Promote
`Formal Verification / verify` to a required fork branch-protection check only
after it has:

- no unexplained counterexamples or false passes;
- no recurring timeout/tool-error failures;
- stable pull-request latency;
- useful, reproducible artifacts for every failure.

Promotion in the fork is a separate decision. Upstream contribution should
follow only after the fork data supports the stated value proposition.
