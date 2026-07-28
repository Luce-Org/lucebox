# Inline prefix-cache abort-hole allocation capsule

This capsule model-checks the scalar slot-selection decision used by the
production `InlinePrefixCacheState::prepare` transition.

## Checked properties

- Selection returns a slot in the configured capacity.
- When the cache is below capacity, selection never returns a slot owned by a
  committed entry.
- The selection works for every round-robin cursor and occupancy pattern in
  the bounded domain.
- ESBMC's default pointer and bounds checks plus integer overflow checks remain
  enabled.

## Bounded operating envelope

Pull requests cover capacities 1–4 and every occupancy bit pattern in that
range. Extended runs widen the capacity domain to 1–16. Production
`PrefixCache` clamps inline capacity to 64 slots; this capsule does not claim
coverage beyond its declared bound.

The immutable native regression constructs the real serialized transition:
commit slot zero, reserve and abort slot one, then require the replacement
reservation to reuse the free slot without invalidating the committed snapshot.
