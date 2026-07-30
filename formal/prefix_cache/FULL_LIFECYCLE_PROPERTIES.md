# Full prefix-cache lifecycle contract

`prefix-cache-full-lifecycle` checks the production
`dflash::common::FullPrefixCacheState` used by `PrefixCache` for exact
full-prompt snapshots. Pull-request proofs use capacity two and nightly proofs
use capacity four. ESBMC model-checks the fresh
prepare/invalid-confirm/confirm/lookup/clear sequence for every symbolic
snapshot boundary from one through four. The immutable native regression
covers the wider capacities, abort, hole-reuse, and LRU lifecycle against the
exact head.

## Model-checked guarantees for the named traces

- configuration accepts only a non-empty, non-overflowing absolute slot range
  that excludes the disk staging slot;
- the fresh reservation owns one in-range non-staging slot;
- prepare binds the effective-prompt snapshot boundary independently of the
  raw prompt used as the lookup key;
- confirm succeeds only for the matching key and slot, a still-valid victim,
  a positive raw prompt length, and a saved position inside the prepared
  effective-prompt boundary;
- a successful fresh confirm installs the requested key and clears the
  reservation;
- invalid confirmation leaves the pending reservation and committed ownership
  unchanged;
- exact lookup returns the confirmed slot and saved position;
- clear removes every entry and reservation and resets the allocation cursor.

The exact-head native regression also checks the production sequences for the
capacity-two abort hole, LRU touch and victim selection, victim abort, invalid
reservation inputs, staging-slot exclusion, and independent committed-slot
invalidation.

## Deliberate exclusions

This capsule does not prove:

- hashing collision resistance;
- GPU snapshot creation, restoration, or byte identity;
- the atomic public size mirror or HTTP request orchestration;
- disk-cache staging I/O;
- concurrent access (the production state is daemon-thread owned).

The `PrefixCache` adapter remains responsible for hashing prompts, invoking
backend snapshot operations, synchronizing its atomic size mirror, and logging.
