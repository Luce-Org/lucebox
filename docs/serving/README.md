# Bounded model serving

This fork adds native host-memory admission and a single-GPU HTTP gateway to
[Lucebox](https://github.com/Luce-Org/lucebox). It retains the upstream Apache-2.0
license and development history. It is an independent fork, not an official
upstream release.

The native worker reclaims idle prefix/full snapshots in LRU order within each
pool before rejecting a request. If cache-friendly execution still does not fit,
it tries an uncached working-memory budget and skips optional snapshots. CPU
snapshot allocation independently checks actual tensor sizes, available RAM, and
an optional per-snapshot byte cap. Admission runs before SSE headers or GPU work.

The gateway provides configurable model aliases and automatic model switching,
one active GPU request, a bounded queue, disconnect cancellation, progress and
wall-clock deadlines, and backend recycling. Failed generations are not
transparently replayed: clients decide whether to retry partial output/tool calls.

## Quick start

Python 3.11+ is required for the standalone gateway; it has its own small set of
dependencies and does not change the upstream Python workspace requirements.

```sh
python3 -m venv .venv-serving
.venv-serving/bin/pip install -r requirements-serving.txt
cp models.example.json models.json
# Edit models.json: model/draft paths, backend executable and library location.
.venv-serving/bin/python model_router.py
```

`models.json` is deliberately ignored by Git. Example paths and aliases are
placeholders: obtain compatible model and DFlash draft weights separately.
The gateway defaults to loopback (`127.0.0.1:8216`). For LAN access, explicitly
set `listen_host` to the desired interface or `0.0.0.0`. Authentication is not
implemented by this gateway; put an authenticated reverse proxy in front of an
endpoint exposed beyond a trusted network.

`LUCEBOX_ROOT`, `LUCEBOX_CONFIG`, and `LUCEBOX_LOG_DIR` override the checkout,
configuration file, and backend-log directory. The managed backend URL must be
loopback HTTP. Set `backend.executable`, `backend.library_path`, and optional
`backend.extra_args` to match the installation. Relative executable paths are
resolved against the checkout. The inherited library search path is preserved.

For a HIP build, supply the GPU architecture and ROCm installation:

```sh
ROCM_PATH=/opt/rocm HIP_ARCHITECTURES=gfx1100 scripts/build_hip.sh
```

For repeatable local settings, put shell assignments for `ROCM_PATH`,
`HIP_ARCHITECTURES`, and `LUCEBOX_PYTHON` in `.lucebox/build.env` (ignored by Git).
Both build/test scripts source this optional file; `LUCEBOX_BUILD_ENV` overrides
its location.

The architecture above is an example, not autodetection. Set it for the actual
GPU. CUDA or other supported builds can be selected through `backend.executable`.
The gateway's defaults use DFlash block size 16, two prefix slots, and Q8 KV;
verify model/backend support or supply appropriate backend arguments.

## Boot service

`deploy/lucebox.service` is a systemd template for a checkout at `/opt/lucebox` and
a dedicated `lucebox` system account. Create the account, install the gateway
venv at `/opt/lucebox/.venv-serving`, and give the account read access to the
checkout/config/model files and the necessary GPU device access. On systems
using device groups, membership in `render` and/or `video` may be required.
Adapt the template if the installation layout or service account differs.

```sh
sudo useradd --system --home-dir /opt/lucebox --shell /usr/sbin/nologin lucebox
# Configure the account's GPU permissions for this host before starting.
sudo install -m 644 deploy/lucebox.service /etc/systemd/system/lucebox.service
sudo systemctl daemon-reload
sudo systemctl enable --now lucebox.service
curl --fail http://127.0.0.1:8216/health
```

Skip account creation if it already exists. `LogsDirectory=lucebox` creates the
writable log directory. Add any required host/container GPU-initialization
services as systemd dependencies through a local drop-in. Container or VM
hypervisor autostart is a separate host configuration.

`deploy/lucebox.logrotate` is a matching optional rotation policy. Install it under
`/etc/logrotate.d/` after configuring the account and log path. Existing services
can use the configurable gateway without adopting the example service account.
`scripts/start.sh` starts the installed service; `LUCEBOX_SERVICE` overrides its name.

## Admission, errors, and limits

| Condition | Behavior |
|---|---|
| Reusable caches consume needed RAM | Reclaim idle snapshots and recheck; recompute from request messages. |
| Optional caching still too expensive | Attempt uncached generation. |
| Uncached request still cannot fit | HTTP 503 `insufficient_memory`, `Retry-After: 5`. A smaller fresh request may succeed. |
| Prompt + requested output exceeds context | Native HTTP 400 context-limit error; clients such as Pi can compact. |
| Queue full or queue wait expired | HTTP 429 with retry guidance. |
| Client leaves | Close upstream; await native cancellation, then recycle if necessary. |
| Backend connection fails | HTTP 502, or an SSE error if streaming has begun; reload on next request. |
| Stream stops producing substantive deltas | Timeout and recycle; heartbeat comments do not reset progress. |

The example configuration contains conservative **example profiles**, not a
universal memory estimator. Calibrate before deployment for the model, KV type,
and caching policy. Current resident memory is reflected in Linux MemAvailable
and cgroup-v2 limits; swap does not count as usable headroom. Inactive file cache
is treated as reclaimable. Profiles are enabled only when configured.

| Setting | Example value |
|---|---:|
| Cache-friendly incremental RAM per input/output token | 96 KiB |
| Uncached incremental RAM per input/output token | 8 KiB |
| Fixed incremental RAM allowance | 1 GiB |
| Host-RAM safety reserve | 2 GiB |
| Maximum optional CPU snapshot | 4 GiB |
| Active requests / waiting requests | 1 / 4 |
| Request body / buffered nonstream response | 8 MiB / 8 MiB |
| Queue wait / upload deadline | 120 s / 15 s |
| First substantive streaming delta / subsequent progress | 600 s / 120 s |
| Total generation / model load deadline | 3600 s / 180 s |
| Cancellation / termination grace | 5 s / 5 s |

Memory profile fields map to `LUCEBOX_RAM_BYTES_PER_TOKEN`,
`LUCEBOX_RAM_WORK_BYTES_PER_TOKEN`, `LUCEBOX_RAM_FIXED_BYTES`,
`LUCEBOX_RAM_RESERVE_BYTES`, and `LUCEBOX_SNAPSHOT_MAX_BYTES` in the native process.
The serving deadlines and queue policy are configured under `serving`.

## Observability and tests

- `/livez`: gateway liveness.
- `/readyz` and `/health`: backend availability, busy/pending state and counters.
- `/memory`: RAM headroom, reserve, native eviction/uncached/rejection counters.
- `/status/json`: native generation state and performance history.

Counters reset on process restart. Readiness alone cannot prove GPU progress;
streaming progress deadlines are independent. Logs and deployment configurations
are excluded from version control.

```sh
LUCEBOX_PYTHON="$PWD/.venv-serving/bin/python" ROCM_PATH=/opt/rocm \
  scripts/test_serving_resilience.sh
```

`LUCEBOX_BUILD_DIR` selects an alternate CMake build tree. The standalone memory
tests need a C++17 compiler. Gateway tests use local fake backends and do not load
weights or interrupt a live service. The native server suite covers cache-index
correctness, including reuse of holes left by LRU eviction.

Validation before publication included native component tests, gateway tests,
and controlled live checks: memory rejection followed by successful fresh
context without restart, eviction followed by successful uncached generation,
and a terminated test backend returning 502 followed by successful reload.
Memory-pressure injection used inflated estimates instead of exhausting RAM.

## Design scope

The resource lifecycle follows [vLLM's scheduling principles](https://docs.vllm.ai/en/stable/configuration/optimization/)
and [idle prefix-cache eviction](https://docs.vllm.ai/en/stable/design/prefix_caching/).
This implementation serves a single active request; it does not implement
vLLM's paged KV representation, continuous batching, or active-sequence preemption.
The new native admission path applies to Lucebox's serialized worker. Its existing
GPU KV allocation, chunked prefill, and scratch release remain in use.

Host-memory profiles are estimates, not OS reservations; external allocations
can race checks. GPU-memory profiling and driver-hang recovery remain separate
concerns. A fault that prevents model reload requires operator intervention.
