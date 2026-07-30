# First-wave verification evidence

This file records the local promotion evidence for the first advisory wave. It
does not promote any target or claim proof of the complete Lucebox system.

## 2026-07-30 PR-bound replay

The five legacy capsules were run together with ESBMC 8.4.0 using the
repository's pinned verifier image:

`ghcr.io/dusterbloom/lucebox-esbmc-ai-verifier@sha256:cb38011acd8dbe5aeb702b6ba981607cf81656667d79461fc74e66d552cabd07`

| Capsule | Result | Bound and solver | Time |
|---|---:|---|---:|
| `prefix-cache-inline` | passed | capacity 4, Z3, unwind 5 | 7.37 s |
| `prefix-cache-abort-hole` | passed | capacity 4, Z3, unwind 17 | 0.81 s |
| `prefix-cache-full-lifecycle` | passed | capacity 2, Boolector, unwind 5 | 45.35 s |
| `spec-commit-exactness` | passed | width 4, Z3, unwind 17 | 1.02 s |
| `kvflash-residency-map` | passed | four blocks, Boolector, unwind 14 | 89.19 s |

The new targets' exact properties and exclusions are documented in:

- [`prefix_cache/FULL_LIFECYCLE_PROPERTIES.md`](prefix_cache/FULL_LIFECYCLE_PROPERTIES.md)
- [`spec_commit/PROPERTIES.md`](spec_commit/PROPERTIES.md)
- [`kvflash/RESIDENCY_MAP_PROPERTIES.md`](kvflash/RESIDENCY_MAP_PROPERTIES.md)

The broader exact-head native regressions remain part of each deterministic
plan. They cover integration behavior that is intentionally not described as
model checked.

## 2026-07-30 mutation campaign

The complete checked-in mutation suite passed against exact source commit
`a0e476bd`. Each case ran in an isolated clone and required the protected
target to conclude with a counterexample:

| Mutation | Target result | Intended evidence |
|---|---:|---|
| inline free-slot selector bypass | counterexample | exact-head native regression; inline formal control verified |
| full-cache free-slot selector bypass | counterexample | exact-head lifecycle regression; inline and abort-hole controls verified |
| speculative prefix comparison inversion | counterexample in 2.26 s | ESBMC and native regression |
| KVFlash capacity-boundary off-by-one | counterexample in 0.93 s | ESBMC and native regression |

This establishes sensitivity for the checked-in representative defects. It
does not imply sensitivity to every possible defect in the same production
areas.

## 2026-07-30 companion 0.3.0 candidate validation and proposed pin

Companion commit `971fda75b4b6f2da9ff26d1c06bf248472bf96b6` was published
and smoke-tested in
[run 30541259801](https://github.com/dusterbloom/lucebox-esbmc-ai/actions/runs/30541259801).
This branch proposes pinning the resulting post-smoke OCI digests:

- verifier:
  `ghcr.io/dusterbloom/lucebox-esbmc-ai-verifier@sha256:fd1a08aec947df86f3a441504f65d44ebb3fc05c299e98cdf3249a724446f62e`;
- repair:
  `ghcr.io/dusterbloom/lucebox-esbmc-ai-repair@sha256:2d77a93c2467641696b1080e00866b22d5f7ab6f4456eeceec35862aa01bdd46`.

Before the proposed pin transition, the new verifier was run locally as an
explicit candidate override against exact Lucebox head
`84ad08b22e34814f116628a3ea396b7c4efa6bf0`. The authenticated generated plan
and all declared native regressions verified:

```text
LUCEBOX_FORMAL_IMAGE=ghcr.io/dusterbloom/lucebox-esbmc-ai-verifier@sha256:fd1a08aec947df86f3a441504f65d44ebb3fc05c299e98cdf3249a724446f62e \
LUCEBOX_FORMAL_RESULTS=/tmp/lucebox-formal-new-image-candidate \
./scripts/formal.sh --all
```

Mode `all` used base=head=`84ad08b22e34814f116628a3ea396b7c4efa6bf0`.
The resulting `report.json` SHA-256 was
`8613f96727f73739e1072242f4a7e0ea6f98b92172d83061b3eb370268e69ea7`.
The generated plan directory was ephemeral and was not retained.

## 2026-07-30 separate KVFlash nightly timeout

Companion commit `8a7e1aae040a9f17a7e7513e7bb29c9495afcedb` adds an authenticated
`nightly_timeout_seconds` target setting. The PR timeout remains 180 seconds;
the KVFlash five-block nightly expansion is set to 240 seconds. The companion
CI suite passed all 49 tests, and the updated images were published and
smoke-tested in [run 30560639772](https://github.com/dusterbloom/lucebox-esbmc-ai/actions/runs/30560639772):

- verifier:
  `ghcr.io/dusterbloom/lucebox-esbmc-ai-verifier@sha256:9321a4aa371efec1b55e33dff0c1cefbc10778aa7f38b07cb2b84cff1c9ed39e`;
- repair:
  `ghcr.io/dusterbloom/lucebox-esbmc-ai-repair@sha256:e33712eac4207a74562fe9d9fda9a9456ce41c175456dda3dd54e7ba7286c597`.

This timeout change provides operational headroom for the nightly proof; it
does not promote KVFlash or relax the per-PR bound.

The exact-head nightly replay at Lucebox commit `2fe23811868701ec430325817972b1efd0872fbc`
passed both generated and legacy lanes in
[run 30561396096](https://github.com/dusterbloom/lucebox-hub/actions/runs/30561396096).
The generated KVFlash proof took 167.92 seconds and the legacy proof took
181.48 seconds against the 240-second nightly timeout, leaving approximately
58.5 seconds of margin in the slower lane. The artifact hashes are:

- plan: `d50d74575fae0871a6ac0eb94b2a36946edd4dcf4e6f7ca3f919ded5f5fec6f9`;
- generated report: `f045a24c6c8619091d375c488ae9b7295f5cc1ed197f795c23723fcc606b302e`;
- legacy report: `23a526ceb8f28279be848ef3548919d9247d9e185f715d9b10f515b22e7f1744`.

| Capsule | Result | Time |
|---|---:|---:|
| `prefix-cache-inline` | verified | 7.27 s |
| `prefix-cache-abort-hole` | verified | 0.96 s |
| `prefix-cache-full-lifecycle` | verified | 45.34 s |
| `spec-commit-exactness` | verified | 1.19 s |
| `kvflash-residency-map` | verified | 87.89 s |

At proposed-pin commit `b37dae25ea3f4104785ee2163174390fbbb9e0be`,
the normal pinned-image invocation also verified all five capsules with
base=head and mode `all`:

```text
LUCEBOX_FORMAL_RESULTS=/tmp/lucebox-formal-pin-head \
./scripts/formal.sh --all
```

The resulting `report.json` SHA-256 was
`eb2d633fa0108a94aa82364091714da2f8bbe842c1ff72c588079172512f8052`.
Capsule times were 7.02 s, 0.98 s, 45.21 s, 1.15 s, and 87.34 s in registry
order. The complete `./scripts/formal_mutation_test.sh` campaign then passed
at the same commit: each of the four checked-in defects produced its required
counterexample, while the applicable prefix-cache control capsules remained
verified.

The companion's
[49-test CI run](https://github.com/dusterbloom/lucebox-esbmc-ai/actions/runs/30538756694)
includes an adversarial regression checking that a head-edited transitive
contract body cannot shadow the authenticated base snapshot while production
headers still resolve from the exact head.

## 2026-07-30 nightly-bound calibration

The first hosted nightly-mode bootstrap at exact base=head
`a88ba49251fa657efe1d13938c9280a211d380e6` was intentionally treated as a
calibration run:
[run 30543897953](https://github.com/dusterbloom/lucebox-hub/actions/runs/30543897953).
Both the generated-plan and legacy lanes verified the two inline prefix-cache
targets, four-slot full prefix-cache lifecycle, and width-eight speculative
commit target. The six-block KVFlash target reached its exact 180-second
timeout in both lanes and was reported as inconclusive, not as a
counterexample.

The generated KVFlash run spent 145.39 seconds in symbolic execution and
produced 83,245 verification conditions before timing out in Boolector. Because
the four-block PR bound already verifies well within the shared timeout, the
advisory nightly bound is calibrated to five blocks rather than loosening the
PR timeout. Five remains a strict coverage expansion and adds a
non-power-of-two pool. It must pass generated, legacy, native, and mutation
checks before it counts as clean soak evidence.

## Promotion blockers

The three new targets remain advisory until all of the following are complete:

1. The companion and Lucebox pin transitions are reviewed and merged into
   their protected base branches.
2. The accepted base policy completes a clean PR/nightly soak on the new
   digests.
3. Maintainers approve each target's protected contract, bounds, latency, and
   ownership in a separate promotion change.
