# Qwen 3.6 production qualification

Lucebox provides an operator-dispatched qualification for the supported Qwen
3.6 27B Q4 autoregressive configuration on Radeon AI PRO R9700 and Strix Halo.
It lives under `harness/qualification/qwen36` with the other hardware-specific
qualification tools.

The normal CI job runs model-free tests for profile parsing, command construction,
evidence validation, reporting, timeouts, and cleanup. It does not download the
model or occupy a GPU. The `Production quality` workflow is manual and runs the
model-backed stages on the `lucebox3` self-hosted runner, one hardware profile at
a time.

## Run it

Dispatch the `Production quality` workflow, or run a profile on the target host:

```bash
harness/qualification/qwen36/qualify.sh qwen36-27b-q4-r9700-ar-c4
harness/qualification/qwen36/qualify.sh qwen36-27b-q4-strix-halo-ar-c4
```

The model directory defaults to `/opt/models` and can be changed with `MODELS`.
Each run creates an evidence directory containing the profile snapshot, build and
server identity, generation reports, telemetry, a SHA-256 evidence manifest, and
the final JSON and Markdown reports.

| Stage | Check |
| --- | --- |
| R0 | Artifact checksum, selected architecture, Release build, and server readiness |
| R1 | Canonical autoregressive correctness |
| R2 | Bounded generation quality corpus |
| R3 | Byte determinism at profile concurrency |
| R4 | Bounded concurrency sweep |
| R5 | Repeated c1 and c4 performance evidence capture |
| R6 | Health, AMD telemetry, memory, and control-plane latency drift |
| R7 | Evidence cross-checks and final report |

R5 records samples but does not currently compare them with a checked-in
baseline. Baseline calibration and performance pass/fail thresholds are deferred
until the remote GPU machines are available. A completed run therefore validates
correctness, quality, determinism, concurrency, and drift while preserving the
performance data needed for later calibration.

Run the model-free checks locally with:

```bash
uv run --frozen --extra dev python -m pytest -q \
  harness/tests/test_qwen36_production_gates.py \
  harness/tests/test_qwen36_qualify.py
```
