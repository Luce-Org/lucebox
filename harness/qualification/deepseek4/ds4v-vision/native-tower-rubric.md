# Native tower comparison

Base ef64f62 has passed 12 remote tests, all 267 original-byte comparisons, and native gguf.cpp parsing. Independent exporter review passed.

Score each candidate from 0 to 5 on these criteria.

1. Reproduces source patch projection, 32 blocks, half-split 2D rotary layout, full attention, RMSNorm, channel-first padded aligner, exact GELU, and BF16 operation boundaries.
2. Runs both original-image CPU fixtures with finite outputs, exact dimensions, reported numerical differences, and focused stage evidence for any discrepancy.
3. Owns weights and scratch safely while borrowing the text backend, rejects malformed contracts before allocation, and leaves no graph allocation after destruction.
4. Keeps a narrow reusable API and avoids new kernels, HTTP changes, or duplicate decoder logic.
5. Demonstrates bounded scratch memory and reports CPU encode time for both images without retaining all attention intermediates.

Candidate A uses a fresh inherited-model agent because the configured feature and hardest-task seats are unavailable or have returned capacity errors. Candidate B attempts the configured OpenAI hardest-task seat. Other model families are not callable here. Record any dropout and do not treat it as agreement. Native runtime work uses CPU only until text load proof releases the GPU pair.
