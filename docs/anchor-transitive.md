# anchor transitive scan

`scan_and_force_transitive` (anchor_scan.cpp) expands the query pool with
tokens from newly-forced chunks and re-runs `scan_and_force` until fixed
point or max_iters (default 3) is reached.

Improves multi-hop retrieval: enables discovery of intermediate context
chunks whose tokens do not appear in the original query but connect
query-to-needle via shared rare tokens.

Empirical result: F1=0.628 on LongBench HotpotQA at ee7 + keep=0.15
(vs uncompressed F1=0.697). This is the ceiling for attention-score-based
prefill compression on this task; see bench/2026-05-25_longbench_hotpotqa/.

On by default. Disable via PFLASH_COMPRESS_ANCHOR_TRANSITIVE=0.
