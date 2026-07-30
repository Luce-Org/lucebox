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

## Promotion blockers

The three new targets remain advisory until all of the following are complete:

1. The checked-in mutations reliably produce target counterexamples while an
   unrelated control remains green where one is declared.
2. The companion verifier's authenticated transitive-contract snapshot fix is
   reviewed, merged, and published in a new immutable verifier image.
3. Lucebox pins that reviewed image digest in the registry and workflows.
4. Maintainers approve each target's protected contract, bounds, latency, and
   ownership.

The current pinned image is sufficient for the legacy replay above. It does
not contain the pending transitive-contract snapshot fix and therefore is not
the image to use when promoting the generated per-PR plan lane.
