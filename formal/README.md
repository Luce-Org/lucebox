# Lucebox formal-verification pilot

This directory defines proof capsules for the
[`dusterbloom/lucebox-esbmc-ai`](https://github.com/dusterbloom/lucebox-esbmc-ai)
companion runner.

## Trust boundaries

The `Formal Verification / verify` check is deterministic:

- its workflow, registry, templates, bounds, and verifier arguments come from
  the exact target-branch base commit, not from the PR being judged;
- it checks out the exact PR head separately and refuses a SHA mismatch;
- its verifier image and ESBMC release are pinned by SHA-256 digest;
- the Lucebox checkout is mounted read-only;
- only the generated-plan and results directories are writable;
- the container has no network, Linux capabilities, or writable root;
- no model SDK or repository secret is present.

The `Formal AI Candidate` workflow is separate and advisory:

- automatic model access is limited to same-repository revisions; a maintainer
  may explicitly dispatch a reviewed fork run;
- a secretless job authenticates the deterministic run, source SHA, evidence,
  and either a real counterexample or a critical coverage gap;
- the `formal-ai` environment requires approval before model access;
- archive paths, sizes, immutable contract hashes, and mutable patch paths are
  validated;
- the credential-bearing proposer never executes generated code;
- generated code is applied, compiled, run, and reverified only in a second
  networkless container after the proposer has exited;
- a missing-contract proposal is compiled and checked by ESBMC without
  credentials, but remains advisory until reviewed and merged into base policy;
- the lane can only upload artifacts. It cannot commit, comment, push, or open
  a pull request.

Every C/C++ capsule requests ESBMC's native self-contained HTML report. A
counterexample publishes it below `counterexamples/<capsule>/` in the formal
results artifact; CI never renders untrusted report HTML in the job summary.

## Current capsules

`prefix-cache-inline` checks the production
`InlinePrefixCacheState` one-entry prepare/confirm/exact-lookup path for
symbolic cache capacity, branch, and prefix depth. Its precise guarantees and
exclusions are in
[`prefix_cache/PROPERTIES.md`](prefix_cache/PROPERTIES.md).

The abort-hole capsule formally checks the scalar free-slot selector. Its
deterministic plan then compiles and runs the immutable base regression against
the exact head, covering the production prepare/confirm/prepare/abort/prepare
sequence. This catches both a broken selector and a `prepare` call site that
bypasses a correct selector. The formal and native guarantees are separated in
[`prefix_cache/ABORT_HOLE_PROPERTIES.md`](prefix_cache/ABORT_HOLE_PROPERTIES.md).

The dependency-free native test covers the wider transition matrix:
abort, cancellation, stale lookup, invalid confirmation, clear, full-capacity
slot reuse, and prefix-aware eviction. Only behavior explicitly named by a
capsule contract is described as model checked.

## Local use

Docker will pull the immutable verifier declared in
`contracts/registry.toml`. Local planning requires a committed checkout.

```bash
./scripts/formal.sh --all
./scripts/formal.sh --nightly
./scripts/formal.sh --base-sha origin/main
./scripts/formal.sh --all --legacy
./scripts/formal_mutation_test.sh
```

Set `LUCEBOX_FORMAL_IMAGE` only when deliberately testing a new companion image.
Results are written to `.formal-results/`.
The mutation test recreates the historical `prepare` call-site bypass in a
temporary standalone clone and requires the base-approved regression to reject
it while the scalar selector contract remains green.

## Adding an approved contract

1. Extract a dependency-light production transition boundary; do not verify a
   toy reimplementation.
2. Write a deterministic template and property document that distinguish
   checked properties from exclusions.
3. Add a deterministic native regression test.
4. Declare the exact symbol/signature, triggers, PR/nightly bounds, mutable
   paths, and immutable contract paths in `contracts/registry.toml`.
5. Run both pull-request and nightly bounds locally.
6. Demonstrate mutation sensitivity before marking the entry `required`.
7. Treat invalid contracts, exhausted bounds, timeout, and tool errors as
   failures, never as passes.

## Per-PR contract registry migration

The current capsules have also been recorded in
[`contracts/registry.toml`](contracts/registry.toml) with deterministic source
templates. This is a dual-run migration: the registry drives the base-locked
plan while `manifest.toml` remains an advisory comparison interface. See
[`contracts/README.md`](contracts/README.md) for the base-branch trust rule,
coverage-gap policy, and local registry validation commands.

The first registry PR cannot use itself as required base policy. Validate that
bootstrap revision with an `all` workflow dispatch; base-locked PR planning
starts after the registry exists on the protected target branch.

## AI environment

Create an approval-protected GitHub environment named `formal-ai`:

```text
Secret:
  ZAI_API_KEY=<key>

Variables:
  FORMAL_AI_MODEL=openai:glm-5
  FORMAL_AI_BASE_URL=https://api.z.ai/api/paas/v4/
```

`OPENAI_API_KEY` may be used instead when the base URL and model are configured
for OpenAI. Model credentials are passed only to proposal containers.

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
