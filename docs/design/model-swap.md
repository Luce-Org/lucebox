# Swapping models on a shared GPU in one luce_server process

Status: implemented for qwen35 (cold and warm eviction) and DeepSeek V4
(partial and cold eviction), both driven by swap residency. User-facing flags are documented in
[MODEL_LOAD_BALANCING.md](../../server/docs/MODEL_LOAD_BALANCING.md#swap-residency).

## Goal

One process serves several model blocks (`--model-routing name`) that share
GPUs but do not fit on them together. Only models whose device sets do not
overlap are resident at a time. A request for an evicted model swaps it in:

1. drain the conflicting resident models;
2. release their device memory while keeping their host-RAM prefix snapshots;
3. bring the requested model back with the device layout it had at load.

Compared with restarting a process per model, an in-process swap keeps every
model's prefix snapshots, so a long session that comes back after a swap
restores its prefix instead of prefilling it again. That is most of the value;
the weight upload itself can only approach the PCIe floor.

Out of scope: choosing which model serves a request (the client names it),
co-residence of two models' working sets on one GPU, and GPU-to-GPU
migration.

## Measured constraints (R9700, PCIe Gen3 x4)

- About 3.5 GB/s each way, pinned or pageable alike. The weight upload floor
  is about 4.1 s for Qwen 3.8 27B + DFlash2 and 1.8 s for Qwen3.5-9B + DFlash.
- A process restart takes 6.2 s and 2.9 s for the two, and loses every prefix
  snapshot.
- Snapshot restore at 52k tokens takes 2.2 s (27B) and 1.0 s (9B); a cold
  prefill at 52k takes 77 s and 29.5 s.

Measured with this change: cold reinstatement 5.1 s (27B) and 2.1-2.35 s (9B),
warm reinstatement 5-9 ms, a 52.7k-token session back in 2.5 s instead of about
80 s, greedy output identical after a snapshot restore, and 26.9 GB peak VRAM
with both models' weights kept plus a 52k-token prefill at 64k context.

## Backend contract

`ModelBackend` gains, separate from `park()` (which a backend uses for its own
transient memory juggling):

- `supports_eviction(reason)`: whether this configuration can be evicted.
- `evict(level)`: releases device state on the model's worker while it is
  idle. `rejected` means nothing was torn down and the model is unchanged.
  - `cold`: every device allocation (weights, draft, KV/recurrent caches,
    paged pools, graphs, scratch, backend contexts and their pools).
  - `warm`: keeps the weights on the device and releases the rest, so a
    reinstatement only rebuilds caches.
  - `partial`: releases the primary device only and keeps a secondary
    device's expert tier (DeepSeek V4 with in-process experts).
- `reinstate(error)`: rebuilds exactly the layout frozen at init (context,
  KV types, paged-pool size, draft configuration), never re-deriving it from
  the free memory of the moment, so snapshots taken before the eviction
  restore afterwards. On failure the model stays evicted with its partial
  state released.
- `is_evicted()`, `weights_resident()`, `weight_device_bytes()` for the
  manager and `/status/json`.

While evicted, generation, hidden-state readouts and snapshot saves fail with
an explicit error, and size and memory queries answer without touching the
freed tensors.

### qwen35

Supported for the single-device serial worker with the draft on the target
device; `supports_eviction()` refuses `--max-concurrency` > 1, paged
attention, KVFlash, a remote draft, tensor parallelism, vision and the other
Qwen3.5 variants.

`init()` is split so the device-allocation steps run both at first init and in
`reinstate()`. Eviction materializes any deferred live snapshot to host, then
releases consumers before what they reference: the DFlash adapter, drafter,
step graphs, DFlash2 selector graphs and the worker's thread-local device
scratch (DFlash2, SpecLA, GPU sampler and draft top-k, BSA persistent
buffers), the draft KV and feature mirror, the weights, the target cache, and
finally the backend contexts: on the HIP VMM pool only `ggml_backend_free`
returns pool memory. KV caches are zeroed on the device at reinstatement
instead of uploading host zeros.

Warm eviction keeps the weight buffers (they belong to the device's buffer
type, not to the backend context) and points the rebuilt model at them, which
makes a reinstatement a cache rebuild of a few milliseconds.

### DeepSeek V4

Supported for the classic serial worker with in-process experts on two GPUs
of one runtime (the `ds41-lucebox` profile); `supports_eviction()` names what
else refuses (paged serving, vision, PFlash, remote experts, spare primary
cache slots, mixed-qtype experts).

| State | Primary device | Secondary expert tier | Host snapshots and model state |
| --- | --- | --- | --- |
| resident | present | present, running | present |
| partially evicted | released | present, suspended | present |
| fully evicted | released | released | present |

- Settings the backend used to reread from the environment (expert
  parallelism, draft placement, fused modes) are resolved once at init into
  `DeepSeek4ResolvedSettings`, so a reinstated model comes back with the
  devices and modes it was loaded with.
- The expert placement is frozen at init (`FrozenPlacement`): ordered hot and
  cold expert ids, physical and decode owner maps, streamed count, spill
  sizes and the selection bias after routing adjustments. A reinstatement
  never places experts again, since a placement recomputed from current free
  memory would move rows under the retained secondary stack and cache, and
  verifies the rebuilt storage against the record.
- Router-bias and protected-expert inputs are parsed once and applied once per
  set of weights.
- Partial eviction order: finish mailbox-dependent launches while the
  resolver and loaders still run; suspend the streamed expert cache (no new
  jobs, issued copies finish, slots and LRU kept); synchronize every context
  and stream; destroy graph consumers before tensors (fused graphs, storage
  graph caches and prefill arenas, worker-local runtime, DSpark head scratch,
  direct HC scratch, stream engine); release the primary payloads and backend
  context.
- Partial reinstatement recreates the primary backend, reloads the dense
  tensors, rebuilds the primary expert stacks from the recorded shapes and
  ordered ids, reapplies the routing adjustments, recreates the cache with the
  frozen geometry and resumes the expert cache without resizing it.
- Primitives used: `MoeHybridStorage::release_primary_experts()` /
  `rebuild_primary_experts()` / `rebind_cold_backend()` /
  `release_prefill_allocators()`, expert cache `suspend()` / `resume()` /
  `release_graphs()`, and per-device release of the DSpark head and HC
  scratch.

## Control operations on the worker

The worker's job queue carries control operations (evict warm or cold,
reinstate) with a completion the manager waits on. The serial worker runs
them between jobs, when nothing else touches the GPU, and publishes a fresh
memory report afterwards. Models served by the concurrent scheduler
(`--max-concurrency` > 1) refuse control operations for now, so swap
residency needs the serial worker.

## Swap manager

The multi-model front keeps, per model block, the device set (target, layer
split GPUs, draft device, expert device), a residency state
(`resident | evicted | failed`), waiter counts and swap statistics.

- Startup: cheap checks first (explicit target devices, and draft devices
  for blocks with a draft), then every later
  block whose command-line device set overlaps a model that stays resident
  loads and is evicted on its own, before any resident block loads. The
  eviction is partial when the block has an expert device that no resident
  block and no other retained tier uses, otherwise cold. The first block
  stays resident, as does any block on devices of its own. Keep-weights
  applies to runtime swaps.
  Overlapping resolved device sets of resident models are rejected.
- A pinned request for an evicted model registers as a waiter and drives the
  swap: one swap at a time; conflicting residents first serve the requests
  that were waiting when the swap was requested, then close admission and
  drain; each is evicted (partial when its secondary devices do not overlap
  the target's, warm when listed in `--swap-keep-weights` and both models are
  single-device, otherwise cold); the target is reinstated.
- A partially evicted model keeps its secondary devices as
  `retained_devices`. A model that needs one of them, or a warm-held model
  whose devices a multi-device target needs whole, is evicted cold before
  the target reinstates.
- If a reinstatement fails while other models keep weights warm on the same
  devices, those are evicted cold and the reinstatement retried once.
- A failed eviction or reinstatement reinstates what this swap evicted, marks
  the target failed (no retry for 30 s) and answers its waiters 503.
- Unnamed and `auto` requests only use resident models and never swap; when
  no model is resident and no swap runs they get 503 instead of waiting.
- A requester whose model became resident admits at once.
- Weight residency and kept-weight bytes for `/status/json` are read on the
  worker when a transition finishes and published under the manager's lock,
  so no HTTP thread queries a backend that may be mid-swap.

## Remaining work

- Launch profiles that set environment variables are refused in a
  multi-model server, so a mixed server sets the `ds41-lucebox` environment
  and flags itself, and Qwen runs under it (including
  `ROCBLAS_USE_HIPBLASLT=0`). Qwen performance under that environment is
  not qualified beyond the 27B decode check below.
- Control operations through the concurrent scheduler, so models served with
  `--max-concurrency` > 1 can swap.

Measured on an R9700 (PCIe Gen5 x16) with a Strix Halo (no carve, 125 GiB)
serving Qwen 3.8 27B + DFlash2 and DeepSeek V4.1 Flash (`ds41-lucebox`
placement, Q2K-Q4K DSpark drafter): DeepSeek V4.1 starts partially evicted
with 9761 experts (90 GiB) held on the Strix Halo. Its partial eviction takes
2-58 ms and its partial reinstatement 9-14 s (dense weights, primary experts
and drafter reread from SSD); the Strix Halo tier is never reloaded. The 27B
reinstates cold in 7.6-8.4 s and decodes at the same speed after a swap.
Across ten partial cycles the R9700 returned to the same free memory, and
greedy outputs and restored snapshots matched the first run.

GPU graphs: multi-model processes disable graph capture because another
worker's blocking calls break a capture in flight. With swap residency and one
resident model per device this hazard may disappear; left for later.
