# Stream progress and GPU pool pressure

The legacy GPU pool now retries an allocation after releasing its unused cached blocks on a memory-allocation error. If the 5% padded allocation still fails, it retries the exact requested size. Live allocations are not reclaimed; other GPU errors and genuine exhaustion still fail. Pressure logs record requested, padded, and reclaimed bytes. HIP and MUSA aliases preserve the CUDA error-name abstraction.

The server emits `:lucebox-progress {"id":"...","tokens":64}` SSE comments every 64 generated tokens, including while tool syntax is buffered. The gateway accepts only a strictly increasing integer counter for the same upstream response. Generic heartbeats, repeated counts and changed IDs do not reset the token watchdog. Comments do not expose incomplete tool arguments to clients. The overall request deadline remains in force.

OpenAI chat streams ending without a finish reason (or explicit error event) now produce a backend failure instead of silently completing. The gateway logs the failure type and backend stop duration. Admission logs record checkpoint residency, free memory, graph scratch, requested output, capture size, and planned evictions/moves before execution; this preserves evidence even if a request crashes before its final trace.

## Validation

Run from the repository root:

```
python3 server/test/test_pool_pressure.py
PYTHONPATH=. python3 -m unittest discover -s server/test -p 'test_router*.py' -v
```

The pool test extracts the actual allocator class and substitutes an 8 MiB artificial capacity. It verifies reclamation, live-allocation preservation, exact-size fallback, and genuine exhaustion. On linuxmacan, `python3 server/test/test_pool_pressure.py --hip` uses real hipMalloc/hipFree under the same small artificial limit; it does not exhaust the GPU. The Python gateway suite requires aiohttp.

## Limits

These changes do not replace the fixed 4 GiB working allowance with a measured phase-specific peak model. That remains required before making a broad OOM-prevention claim. Full-model long-output/forced-thinking-close validation is separate from allocator tests. This commit is source-only; deployment and a production restart are separate operations.
