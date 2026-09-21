# GPU exhaustion and false-stall investigation

## Conclusions

There are two confirmed server failures. The first is an allocator OOM immediately after the forced thinking-close event. The second is a progress watchdog terminating a backend that was actively generating buffered tool-call text. Neither is explained by Pi failing to send messages.

Checkpoint eviction and scratch reclamation were operating, but they were insufficient. Admission checks a fixed transient allowance, not the worst execution phase of the admitted request. The GPU allocator has no cached-buffer reclamation fallback on allocation failure. There is no verified guarantee against another OOM in the deployed version.

## Timeline (UTC)

- 21:18:02: failed request begins, 16,951 input tokens, 32,768 requested output capacity, medium reasoning.
- Backend reaches `spec-decode close at committed=31287/18432`. The first value is absolute context position; the second is a generation budget. This log mixes coordinate systems and must not be interpreted as 31K generated tokens or proof of budget overrun.
- Immediately afterward: `ROCm error: out of memory` in legacy pool allocation, `ggml_cuda_device_malloc(...look_ahead_size...)`, line 517 of the preserved GPU source.
- 21:23:57: Pi records incomplete stream; gateway records connection completion.
- 21:23:59: next request initiates load; 21:24:05: ready. Reload is about six seconds, about eight seconds after the failure was observed. These journal timestamps have one-second resolution.
- Retried request: 17.890 seconds prefill, 317.603 seconds decode, 14,392 output tokens. Completes with a tool call at 21:29:40. The minutes spent waiting were mostly real generation.
- Subsequent tool request: 32,196 input, 29,005 restored prefix, 6.581 seconds prefill. GPU generates 8,559 tokens in 120.986 seconds. Gateway times out, closes upstream, and the server records client disconnect. Pi records the explicit no-progress error at 21:33:07.
- Second recovery stops the backend; loading is deferred until another request. This is not an automatic completed restart.

## Why reclamation did not prevent OOM

### 1. Different memory lifetimes are being treated too similarly

Weights, active attention/recurrent state, immutable checkpoints, graph allocator storage, and cached operator temporaries have different lifetimes. The checkpoint manager can evict optional snapshots or move inactive protected snapshots to RAM. It cannot reclaim working buffers owned by a currently executing kernel.

The backend does release graph scratch and trim the operator pool between requests. Logs immediately before the failing request show scratch trimming and optional checkpoint evictions. Thus “nothing was being recycled” is disproved. Idle free memory after process termination also shows this was process-owned memory, not evidence of a persistent driver leak.

### 2. The legacy GPU pool cannot reclaim cached blocks on demand

The actual allocator uses best-fit lookup over free blocks. If none is large enough, it allocates a new block with 5% extra capacity. The pool retains up to 8,192 free blocks, a count limit rather than a byte budget. If allocation fails, `CUDA_CHECK` aborts without trimming those free blocks or retrying without the optional 5% padding.

A long request or changed graph shape can therefore fail an allocation even if enough of its own cached memory could be released. Between-request cleanup cannot solve that within-request failure.

**Verified distinguishing experiment:** compiled the actual extracted allocator class with simulated GPU allocation calls. With an 8 GiB capacity, allocate/release a 4 GiB request (4.2 GiB retained after padding), then request 5 GiB. Original implementation fails; trimming cached blocks lets it succeed. No real GPU allocation or server disruption was needed.

### 3. Admission is not a reservation of the request's true peak

`hybrid_http.inc` uses `work=4*GiB`, with the policy's additional 2 GiB headroom. It checks GPU free bytes and simulates snapshots/metadata, but does not derive the transient peak from output length, decode graph shape, or the speculative-to-ordinary decode transition. Context-plus-output validation is a token-limit check, not a complete memory estimate.

`cache_reclaimable_scratch_bytes()` reports two graph allocator buffers; it does not report the legacy pool's cached free bytes. That omission is conservative for admission but demonstrates that the manager lacks a complete picture of working memory.

A source checkpoint and replacement snapshot may coexist while a generation runs. This is correct transactional behavior but needs to be included in the peak. A 9 GiB checkpoint budget alone cannot guarantee that weights, live state, drafter, and scratch fit alongside it.

### 4. The transition is suspicious but not fully reconstructed

The OOM follows the thinking-close marker. Source execution next restores/replays the overridden token prefix and may switch from speculative verification to ordinary decoding, changing allocation shapes. The existing trace lacks failed allocation size, free bytes, cached free bytes, live pool bytes, and a full worker backtrace. Consequently the exact kernel/allocation and the fraction attributable to checkpoints versus scratch cannot be established retrospectively.

Do not present the simulated allocator experiment as a reproduction of the entire production crash.

## Separate false-stall defect

The SSE emitter buffers tool-call syntax and parses it on completion. The gateway advances its progress deadline only for substantive streamed deltas and deliberately ignores heartbeat comments. A legitimate long tool call can generate thousands of tokens without such a delta, triggering the 120-second timer. The server's 70.74 tokens/second and eventual client-disconnect record distinguish this from a hung GPU.

Increasing the timeout merely moves the failure threshold. The watchdog needs a request-matched monotonic backend progress signal, with a bounded polling deadline and the existing overall request deadline. Heartbeats alone must not count as progress. Complete tool arguments must still be validated before execution.

## Staged correction and validation

Local-only allocator patch: `work/stream-failure-20260915/allocator/ggml-cuda.cu.staged`.

- On memory-allocation failure only, release this pool's free blocks and retry.
- If padded allocation still fails, retry exactly the requested size.
- Report requested, padded, and reclaimed bytes.
- Preserve failure behavior for non-memory GPU errors and genuine exhaustion.

Host simulation passes: reclaimable-cache case, preservation of live blocks, exact-size fallback, and true exhaustion. Original allocator fails the reclaimable-cache case. This has not been built or validated against ROCm, and is not installed.

## Work needed before claiming the failure is addressed

1. Validate allocator fallback on ROCm with a small bounded artificial-pressure test; verify runtime error handling and synchronization. Do not intentionally crash the production process.
2. Add measurements at admission, snapshot capture, decode start, thinking-close replay, ordinary-decode entry, and allocation pressure: device free/total, snapshot GPU/RAM bytes, graph allocations, and cached/live pool bytes. Log decision and failed allocation identity before abort.
3. Replace fixed working allowance with phase-specific bounds that include requested output capacity and capture coexistence. Where the bound is unknown, decline admission or choose a known-feasible exact path; do not treat unknown as zero.
4. Reclaim unused scratch and optional cache entries at supported serialized safe points. Move protected inactive state transactionally to RAM if necessary; never evict another chat's protection or free live tensors. Verify runtime growth against the admitted envelope.
5. Correct the gateway watchdog and reject incomplete SSE termination explicitly. Log timeout/recycle reason, process exit code, stop duration, load duration, and total unavailable time.
6. Regression tests must combine retained checkpoint state with long output and forced thinking-close, plus buffered tool calls that outlast a short test timeout. Prior 32K prefill speed tests and three small owner tests did not exercise this combination. Large 80K stress runs are unnecessary.

## Evidence and limits

Preserved backend/gateway logs: `work/stream-failure-20260915/`. Source snapshot and host allocator tests are in its `allocator/` directory. Live configuration confirms 120-second token timeout, 1,800-second prefill timeout, 3,600-second total deadline, and five-second cancellation/stop grace periods.

The deployed server was not modified or restarted in this investigation. Pi received extensive thinking text; whether its specific frontend hid that text remains a separate unverified display issue. No promise that OOM can never recur is justified until the missing admission and pressure-path validation is completed.
