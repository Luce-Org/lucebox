# Inline prefix-cache verification capsule

This capsule model-checks the production `InlinePrefixCacheState` transition
core used by `PrefixCache`.

## Checked properties

- A fresh reservation returns an in-range slot and the requested depth.
- A valid confirmation creates exactly one committed entry.
- Exact lookup returns the committed slot and prefix length.
- The number of committed entries never exceeds capacity.
- Every committed entry owns one in-range slot.
- The committed token prefix is non-empty and has the confirmed length.
- A fresh reservation does not spuriously create a pending eviction.
- ESBMC's pointer, bounds, memory-leak, and integer checks remain enabled.

## Bounded operating envelope

Pull requests cover a symbolic fresh-cache prepare/confirm/lookup path for
capacities 1–4. Extended runs widen the capacity domain to 1–16. Each generated
prefix has one to three tokens and belongs to one of three branches.

Abort, cancellation, stale lookup, invalid confirmation, clearing, slot reuse,
and prefix-aware eviction are exercised by the immutable native regression
test. They are intentionally not described as model-checked properties in this
first capsule.

The harness models prefix hashes as collision-free identifiers derived from the
bounded prefix family. It does not prove tokenizer correctness, SHA-1 collision
resistance, cache performance, backend snapshot correctness, or whole-server
behavior. A successful result is a bounded proof for this declared envelope,
not a claim that all Lucebox behavior is formally verified.
