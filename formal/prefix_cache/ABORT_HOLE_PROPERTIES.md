# Inline prefix-cache abort-hole allocation contract

The capsule enforces a native ESBMC function contract around the scalar
`select_inline_free_slot` helper for every bounded cursor and occupancy
pattern. The deterministic plan also compiles and runs an immutable
base-revision regression against the exact PR head. That regression drives the
production `InlinePrefixCacheState` through the historical defect sequence:
commit slot zero, reserve and abort slot one, then call `prepare` again while
the round-robin cursor points at occupied slot zero.

## Checked properties

- Scalar selection returns an in-range, unoccupied slot whenever one exists.
- The scalar selector cannot mutate program state.
- ESBMC's default pointer and bounds checks plus integer overflow checks remain
  enabled.

The paired native regression checks that aborting preserves every committed
entry and the replacement `prepare` call reuses the aborted slot. Those
integration assertions are regression-tested rather than described as
model-checked properties.

## Bounded operating envelope

The formal capsule covers capacities 1–4 in pull requests and 1–16 in extended
runs. The native regression fixes capacity at two, the smallest state that
reproduces the production call-site defect. `PrefixCache` clamps inline
capacity to 64 slots; the formal capsule does not claim coverage beyond its
declared bound.

On a contract violation, the deterministic lane publishes ESBMC's native,
self-contained HTML counterexample report alongside the textual trace and
immutable repair bundle.
