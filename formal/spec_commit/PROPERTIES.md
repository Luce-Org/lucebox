# Speculative-commit exactness contract

`spec-commit-exactness` checks the dependency-free production decision core
`dflash::common::SpecCommitDecision`.  The pull-request bound uses verification
widths and commit budgets up to four; the nightly bound raises both to eight.

## Model-checked guarantees

For greedy verification:

- the accepted count is the longest draft prefix whose tokens are approved by
  the preceding target rows;
- the seed token is always accepted and the accepted count stays in
  `[1, verify_count]`;
- a target bonus is available exactly when the accepted prefix ends before the
  verification width, while `commits_bonus` is true only when the remaining
  generation budget can emit it;
- the commit count is non-negative, does not exceed the budget, and is at most
  the accepted prefix plus one target bonus;
- every committed token selected by `token_at` is either an accepted draft
  token at the same index or the one target token at the first mismatch;
- a negative mismatch target is treated as the no-bonus sentinel, so the
  already-verified prefix remains committable; a negative accepted draft token
  still cannot be emitted;
- clipped and out-of-range token requests fail instead of reading outside the
  draft prefix or returning a unavailable bonus.

For a model-specific verifier that supplies a precomputed accepted prefix:

- accepted counts outside `[1, verify_count]` fail closed;
- a strict prefix may declare one non-negative target-selected mismatch token
  or omit it when the model-specific verifier reports the negative no-token
  sentinel;
- a fully accepted width must not declare a bonus;
- the same budget, bonus, and safe token-selection rules apply as in the
  greedy path.

The immutable native regression is also compiled against the exact PR head.
It exercises greedy match and mismatch decisions, budget clipping, the sampled
verification finalizer, malformed input, and safe materialization.

## Deliberate exclusions

This capsule does not prove:

- target logits, sampling probabilities, or RNG history;
- model-family rollback, KV restore/replay, or feature-cache updates;
- EOS stopping, client cancellation, counters, or emitted network bytes;
- correctness of the draft or target model;
- tree speculative decoding.

Those effects remain owned by model-family native and GPU tests.  Later
capsules can abstract rollback and emission only after their production
orchestration is shared.
