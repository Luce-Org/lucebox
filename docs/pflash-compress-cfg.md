# pflash compression knobs

All PFLASH_COMPRESS_* and DFLASH_COMPRESS_* env vars are read once per
request in `compress_cfg_from_env(n_chunks, n_keep)` in qwen3_drafter.cpp.

## anchor_radius adaptive ladder

Prevents the 64K NIAH cliff: at long context the needle text is more likely
to straddle multiple chunks, and a fixed radius=2 window (5 chunks / ~160
tokens) loses the back half of the needle.

Default ladder (override via PFLASH_COMPRESS_ANCHOR_RADIUS):

| n_chunks   | anchor_radius |
|------------|---------------|
| < 1024     | 2             |
| 1024-2047  | 4             |
| >= 2048    | 8             |

## max_anchor_hits adaptive ladder

Same breakpoints as anchor_radius. At long context anchors are sparser, so
more hits per query token are affordable.

| n_chunks   | max_anchor_hits |
|------------|-----------------|
| < 1024     | 8               |
| 1024-2047  | 16              |
| >= 2048    | 32              |

## anchor_transitive

On by default. Gated rare-token bridge expands the query pool with tokens
from newly-forced chunks and re-runs anchor scan to fixed point.
Improves multi-hop F1 on LongBench HotpotQA (empirically; F1=0.628 ceiling
at ee7+anchor-transitive on RTX 3090 — see bench/2026-05-25_longbench_hotpotqa/).
Control via PFLASH_COMPRESS_ANCHOR_TRANSITIVE=0 to disable.

## head/tail chunk forcing

Head and tail chunks are force-included before top-K scoring fills the
remainder. The counts scale with n_keep so top-K always gets at least one
slot even when head_raw + tail_raw >= n_keep.

Defaults: head=8, tail=24 (override via DFLASH_COMPRESS_HEAD_CHUNKS /
DFLASH_COMPRESS_TAIL_CHUNKS).
