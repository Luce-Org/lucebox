# KVFlash residency-map contract

`kvflash-residency-map` checks the dependency-free production
`dflash::common::KvFlashResidencyMap` used by `KvFlashPager`. Pull-request
proofs use four one-token physical blocks; nightly proofs use five, adding a
non-power-of-two pool. ESBMC checks the named identity-fill, failed-eviction,
protected-LRU-eviction, mask, and logical-bound sequences. The exact-head
native regression covers the broader chunk offsets, recall, explicit page-out,
block order, reset, and scored paths.

The initial six-block nightly bound reached the declared 180-second timeout in
both generated and legacy lanes on GitHub-hosted runners. Five blocks retains a
strict nightly expansion without weakening the shared PR timeout. Six blocks
remains a future harness/solver optimization target rather than claimed proof
coverage.

## Model-checked guarantees for the named sequences

- invalid chunk sizes, pool sizes, protection counts, partial blocks, and
  eviction-deadlocking pools are rejected, and logical positions are bounded
  before vector growth;
- the filled and post-eviction maps satisfy the runtime partial-bijection and
  resident/free-complement invariant;
- `acquire` and `slot_of` agree on block and in-block token offsets;
- initial allocation follows identity placement and
  `identity_prefix_covers` requires every intersecting prefix chunk to be
  resident in its identity block;
- automatic eviction excludes configured sink chunks and the protected append
  tail;
- the page-out callback succeeds before ownership transfers to a new chunk;
- a rejected page-out callback preserves the map's logical extent, ownership,
  free blocks, append-head policy, and epoch;
- every successful residency change advances the epoch;
- slot-position and slot-validity arrays agree with resident ownership.

The exact-head native regression additionally covers malformed block orders,
explicit page-out failure, score-independent LRU selection, large and negative
positions, mask values, and runtime representation invariants.

## Deliberate exclusions

This capsule does not prove:

- CUDA/HIP allocation, asynchronous ordering, DMA success, or copied bytes;
- KV tensor shape, stride, numerical values, or attention output;
- host-backing allocation size and lifetime;
- score quality or floating-point ordering in lookahead reselect;
- concurrent access.

The pager adapter owns device and host storage. Its page-out callback is an
abstract success/failure boundary in this proof; the map cannot roll back
external side effects from a callback that returns false. The production pager
therefore returns false only before DMA, zeroing, or statistic changes. GPU
parity remains a separate native/nightly responsibility.
