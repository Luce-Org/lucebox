# Qwen4Exp concurrent serving P3 — 2026-09-25

No commit. P3 runtime gates are **not complete**; no quality or throughput
claim is made.

## Preflight and launch attempt

The box was reachable for the pre-run guard. It reported no resident
`luce_server`/`dflash_server`, GTT 18,636,800 bytes, and 60 GiB available RAM.
The build in `/tmp/qwen4exp-p1-src/server/build-hip/luce_server` had
`GGML_HIP_GRAPHS=OFF`; the committed P2 engine/backend sources matched the
local `20664ee6` sources by SHA-256.

An exact-environment server launch was attempted under
`flock /tmp/qwen-perf/gpu.lock` with the requested IQ4_NL model, HIP target,
ctx32768, concurrency4, chunk16384, and delivered env. The command returned
without captured startup output; immediately afterward both direct-IP and
MagicDNS SSH timed out. Six retry cycles across both routes also timed out.
Local `tailscale status` reports that the local tailscaled daemon is not
running. It was not restarted. Because the box is now unreachable, server
residency, lock ownership, startup result, and post-attempt GTT/RAM are
unverified. **Do not launch another model or rebuild on that box until a fresh
remote idle guard succeeds.**

## Static startup blocker found

Inspection showed `server_main.cpp` unconditionally sets
`bargs.paged_attention=true` for every `--max-concurrency > 1` before calling
`prepare_backend()`. The Qwen4Exp full-cache exception in `feature_gate.cpp`
requires its opt-in flag pair while `paged_attention` is false; the generic
paged gate rejects Qwen4Exp (`model_capabilities.h` says `kNever`). This makes
the committed launch path internally inconsistent. I made a narrow working
tree change so the auto-paging assignment is skipped only when both
`LUCE_QWEN4EXP_SEQ_ENGINE=1` and `QWEN4EXP_BATCHED_DECODE=1` are set and
`QWEN4EXP_UPSTREAM` is not active. Other architectures and default Qwen4Exp
launches retain the prior behavior. This patch has **not been built or runtime
verified**, because SSH failed after the guard/launch attempt.

## Gate disposition

| Gate | Result |
|---|---|
| Concurrent server startup and `/proc/<pid>/environ` assertion | **Blocked / unverified** after SSH route failure; no selected env snapshot was collected |
| Four simultaneous distinct HTTP requests, repeated waves | **Not run** |
| Decisive concurrent HE/GSM/Math/recall scores vs HE10/GSM10/Math9/recall2 | **Not run** |
| N=1 vs N=4 aggregate goodput, per-request rate and TTFT | **Not run** |
| Fifth request, over-context isolation, disconnect isolation | **Not run** |
| `QWEN4EXP_UPSTREAM=1` unchanged behavior | Existing gate not rerun in this P3 attempt |
| Power profile / clocks during P3 arms | Only preflight was observed; platform was not captured before the disconnect |

The stock `client_test_runner.py` is serial. For a real quality-under-batch
gate, `concurrent_quality.py` uses its prompt loader, request implementation,
and scoring functions in four-request waves. `http_probes.py` contains the
planned HTTP, capacity, context, cancellation, and N=1/N=4 goodput probes.
Both scripts passed `py_compile`; neither ran against the model.

## Decision

Do **not** flip Qwen4Exp's concurrency capability or either default-off flag.
P3's decisive quality and goodput evidence is missing, and the launch-admission
path needs the unbuilt fix validated first. Keep the current opt-in contained
to qwen4exp, ctx32768, and at most four slots until the box is reachable and
all gates run under a fresh lock/idle guard.
