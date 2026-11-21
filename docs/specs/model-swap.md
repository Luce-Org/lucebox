# Fast model switching on R9700 + Strix Halo

Status: implemented for Qwen3.8 Flash Next (`qwen4exp`), dense Qwen 3.5/3.8
(`qwen35`) and DeepSeek V4/V4.1 (`deepseek4`, one request at a time), all
measured on R9700 + Strix Halo.

## Problem

A Lucebox has an R9700 (32 GB, PCIe Gen3 x4 on the measured box) and a
Strix Halo whose iGPU (`gfx1151`) uses the system's 128 GB through GTT.
Models that share the R9700 cannot all be resident at once:

| Model | R9700 (`hip:0`) | Strix Halo iGPU (`hip:1`) |
| --- | --- | --- |
| Qwen 3.8 27B + DFlash2, 262K context | all of it: 28.4 GB in use | nothing |
| Qwen 3.8 Flash Next, split | 7.08 GiB dense weights, caches, graphs | 71.73 GiB routed experts, 7 of 12 attention layers' KV |
| DeepSeek V4.1 Flash, `ds41-lucebox` | dense weights, hot experts, drafter | the routed-expert tier |

Before this change, switching from one to the other meant stopping one
`luce_server` and starting the other. Measured on the box (2026-11-12, short
prompt):

| Switch | Time to the first reply |
| --- | --- |
| Qwen 3.8 27B to Flash Next | 101 s (Flash Next start 99.7 s) |
| Flash Next to Qwen 3.8 27B | 16 s (Flash Next stop 5 s, Qwen start 10 s) |

A restart also drops every prefix snapshot of the model it stops. A long
conversation that comes back to that model prefills from scratch (52K tokens
on the 27B: about 80 s).

### Where Flash Next's start goes

The loader maps the shards and calls `ggml_backend_tensor_set` once per
tensor (`qwen4exp_loader.cpp`, the upload loop). Flash Next turns off the
managed-memory path (`ggml_cuda_device_use_uma`). Each tensor therefore goes
through `ggml_cuda_staged_h2d`: a memcpy on one thread from the mapping into
a 16 MiB pinned buffer, then a DMA. Page faults on that single thread bound
the load:

| Copy (measured on the box, 2026-11-13) | Rate | 78.8 GiB at that rate |
| --- | --- | --- |
| mmap + memcpy, 1 thread, file not cached (today's pattern) | 0.52 GB/s | about 163 s |
| mmap + memcpy, 1 thread, file cached | 9.1 GB/s | about 9 s |
| mmap + memcpy, 16 threads, file not cached | 4.3 GB/s | about 20 s |
| mmap + memcpy, 16 threads, file cached | 31.3 GB/s | about 3 s |
| SSD, O_DIRECT, 4-16 threads | 4.5-4.8 GB/s | about 18 s |
| pinned host to iGPU, 1 / 4 streams | 18.6 / 35.7 GB/s | about 4.5 / 2.4 s |
| host to R9700 (PCIe Gen3 x4) | about 3.5 GB/s | 7.08 GiB dense part: about 2.2 s |

The 100 s measured falls between the cached and uncached rates of today's
pattern.

## Goals

- Switching between Flash Next (split) and Qwen 3.8 27B takes a few seconds
  each way, and each model keeps its prefix snapshots.
- DeepSeek V4.1 (`ds41-lucebox`) and Qwen 3.8 27B switch the same way.
- One mechanism for every backend and every pair. A backend says what it
  holds on each device. A switch releases only the memory that conflicts.
- Clients change nothing. A request to a model that is not resident brings
  it in.

## Design

### 1. Residency in the backend

`ModelBackend` gains a residency contract, separate from `park()` (which
backends use for their own drafter juggling):

- `residency()`: whether the backend supports it (and why not), the GPUs it
  uses, the ones it holds nothing on, the ones it can keep while releasing
  the others, and the bytes it holds on each.
- `release_gpus(gpus)`: frees the model's weights, caches, graphs and scratch
  on the named GPUs and trims the device pools. It keeps what lives on other
  GPUs and the prefix snapshots and checkpoints in system memory. Releasing
  every GPU is a full release.
- `reinstate()`: rebuilds the layout fixed at first load: context, slots, KV
  types and placement (including the attention layers moved to the expert
  device) and drafter. It never re-derives that layout from the memory free
  at that moment, so snapshots taken before the release restore afterwards.

| Backend | Release of the R9700 only | Full release |
| --- | --- | --- |
| `qwen4exp` split (Flash Next) | keeps the 71.73 GiB expert stacks, each cache's iGPU-side K/V and the host token embeddings | yes |
| `qwen35` (Qwen 3.8 27B), one request at a time, drafter on the target | same as a full release (everything is on the R9700) | yes |
| `deepseek4` split (`ds41-halo`), one request at a time | keeps the cold expert stacks (with their locked host memory), the streamed expert cache and the expert placement | yes, keeping the prefix snapshots |

DeepSeek V4.1's split layout puts the dense weights, the hot experts, the KV
cache and the drafter on the target, and the cold expert stacks on the expert
device's own backend, with the experts in neither streamed from the SSD into
a cache beside them. A release of the target frees the first group and the
hot stacks' graph caches, and keeps the cold stacks, the streamed expert
cache, the expert maps and the file mapping both read. A reinstate loads the
dense weights and the hot stacks on the kept placement (read from the file on
several threads, in one pass over all layers), registers the mixed-quant
decode tables again and rebuilds the cache and the drafter. The expert device
is then checked against the serving headroom, not the load one: on an
integrated GPU its free memory is the system's, which the model that ran
meanwhile may have used. A release that names the expert device frees
everything; a cluster rank always does, since its placement is agreed with
its peers at load.

### 2. Faster weight upload

Loaders upload their tensors in 64 MiB pieces from a pool of threads
(`common/parallel_upload.h`), one pool per destination GPU, all at once: 16
threads for an integrated GPU and 4 for a discrete one, with the kernel's
read-ahead started for the source bytes. The direct-read loader
(`TensorFileReader`) lets four tensor writes run at once. Both serve the
first load and `reinstate()`.

### 3. Draining under concurrent serving

A release or reinstate runs only while the backend is idle. Admission closes,
running requests finish (every slot, with `--max-concurrency` > 1), the
operation runs, and admission reopens. A request that arrives in between waits
for the swap; it is not dropped.

### 4. Swap groups

Servers started with `--swap-group <dir>` share a registry: one JSON file
per server (pid, port, model, bytes per device, state) and a lock file.

1. A request to a server that is not resident takes the group lock.
2. It asks every resident peer that holds memory on a device it needs to
   release that device (`POST /v1/residency`).
3. The peer drains, releases and answers.
4. The requester reinstates itself, releases the lock and serves.

Only one swap runs at a time. A server joining the group takes the lock
before it loads and asks the peers on its GPUs to release them, so a model
can start beside a peer that holds them. While a server waits for the lock
it still answers its peers' release requests. If its reinstate fails, the
request is answered 503 with the reason; a peer that released comes back on
its own next request.

- `GET /v1/residency`: state, bytes per GPU, last release and reinstate.
- `POST /v1/residency` with `{"state": "released", "gpus": [0]}` or
  `{"state": "resident"}`.
- `--swap-start released`: once loaded and once its serving loop has
  started, the server releases the GPUs a peer may need and keeps the rest
  (Flash Next keeps its expert stacks).

Swap groups use `flock`, `kill(pid, 0)` and POSIX sockets, so a Windows
build refuses `--swap-group` at start-up; `/v1/residency` still works there.

Why one process per model rather than several models in one process:

- each model keeps its own flags and environment (DeepSeek V4 and Qwen want
  different `ROCBLAS_USE_HIPBLASLT` values);
- a fault in one model does not take the other down;
- request handling stays as it is.

## Memory on a 128 GB box

- **Flash Next released from the R9700, Qwen 27B resident:** the iGPU holds
  the 71.73 GiB expert tier and Flash Next's iGPU-side KV. The R9700 is free
  for Qwen. That leaves about 40 GB of system memory for the system, both
  processes' host state, Qwen's prefix snapshots and the page cache.
- **DeepSeek V4.1 (`ds41-halo`) released from the R9700, Qwen 27B resident:**
  the Strix Halo keeps DeepSeek's cold expert stacks, which fill its carve and
  continue in locked host memory, and the streamed expert cache in what the
  carve has left. Locked memory cannot be reclaimed, so the room left for
  Qwen's process, both prefix caches and the page cache is what DeepSeek's
  start-up left (it keeps 4 GiB of host memory free beside the locked tier).
- **Qwen 27B released, Flash Next resident:** Qwen keeps only its host
  snapshots. Its weights come back from the page cache when they are still
  there, otherwise from the SSD; either way the R9700's
  link sets the pace.

## Prefix caches

- **Each model keeps its own.** Both processes stay alive, so each model's
  RAM prefix cache and checkpoints live through any number of switches, and
  a conversation returning to a model restores instead of prefilling again.
- **The SSD session cache keeps working** where the engine has it (one
  request at a time, as Qwen 3.8 27B serves): RAM evictions are written out
  and restored within the TTL, across switches and restarts. Warming pinned
  prefixes waits while a model is released.
- **RAM budgets follow the group.** An automatic budget is sized at start-up
  to a quarter of the memory free then. A peer that loads later takes system
  memory through the integrated GPU (Flash Next keeps 72-90 GiB there), so a
  swap-group member resizes its automatic budget after every swap to a
  quarter of what is available now (counting its own cache), never above its
  start-up size, and evicts down to it (to the SSD when that cache is on).
- **Not covered:** concurrent serving (`--max-concurrency` > 1, as Flash Next
  runs) has no SSD session cache in the engine, so its evicted checkpoints
  are dropped; and members sharing a disk cache directory each apply their
  own disk budget.

## Usage

Start each model as its own server with the same `--swap-group` directory.
`--swap-start released` makes a server give its shared GPUs back once
loaded, keeping only what never conflicts:

```sh
luce_server Qwen3.8-27B-UD-IQ4_XS.gguf --draft qwen38-dflash2-q8_0.gguf \
  --target-device hip:0 --max-ctx 262144 --port 8216 --swap-group /run/user/$UID/luce-swap
luce_server Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --target-device hip:0 --expert-device hip:1 --max-ctx auto --max-concurrency 4 \
  --port 8218 --swap-group /run/user/$UID/luce-swap --swap-start released
```

A request to either port brings that model in. `GET /v1/residency` reports
what a model holds per GPU and the last release and reinstate times;
`POST /v1/residency` with `{"state": "released", "gpus": [0]}` or
`{"state": "resident"}` does it by hand.

Load tuning: `LUCE_LOAD_THREADS` (uploads to an integrated GPU, default 16),
`LUCE_LOAD_THREADS_DISCRETE` (uploads to a discrete GPU, default 4),
`LUCE_LOAD_WRITERS` (concurrent writes of the direct-read loader, default 4).

## Measured

R9700 on PCIe Gen3 x4 + Strix Halo, 128 GB, two servers in one swap group
with the flags above (2026-11-13):

| Switch | Restarting the server | Swap group |
| --- | --- | --- |
| Qwen 3.8 27B to Flash Next | 101 s | 3.8-4.7 s reinstate (2.2 s of it the 7.08 GiB upload at 3.4 GB/s) |
| Flash Next to Qwen 3.8 27B | 16 s | 5.3-7.0 s reinstate (14.3 GiB at 2.6-3.2 GB/s) |
| Releasing Flash Next's R9700 part | process stop, 5 s | 0.03 s |
| Flash Next first start, files not cached | about 100 s | 39-48 s |

| 29.5K-token conversation | First visit | Next turn after switching away and back |
| --- | --- | --- |
| Flash Next | 26.4 s | 5.6 s (checkpoint restored in 0.25 s) |
| Qwen 3.8 27B | 44.5 s | 8.1 s (29.5K tokens restored) |

Greedy output is identical on both models across switch cycles.

DeepSeek V4.1 Flash (`ds41-halo`, one request at a time, DSpark drafter,
204,800-token context) and Qwen 3.8 27B in one group, same box (2026-11-15):

| Switch | Releasing every GPU | Keeping the expert tier |
| --- | --- | --- |
| Qwen 3.8 27B to DeepSeek V4.1 | 66-101 s reinstate | 12.2-16.6 s reinstate (dense 5-9 s, hot experts 3.7-4.5 s, drafter 3.1-3.5 s), first reply 17-25 s |
| DeepSeek V4.1 to Qwen 3.8 27B | | 7.4-8.1 s reinstate |
| Releasing DeepSeek's R9700 part | 4.4-4.8 s | 0.05-0.11 s |
| DeepSeek first start in the group | 80-104 s | |

While released, DeepSeek keeps 104-107 GiB on the Strix Halo (its carve and
13.6-16.6 GiB of locked host memory). A 29.5K-token conversation's next turn
after switching away and back restores its prefix on both models (Qwen 8.6 s
instead of 43 s). Greedy output is identical across switch cycles; a release
of every GPU sizes the expert placement again from the memory free then, so
its output can differ from before.

What sets these times:

- **The R9700's link.** One to four writers of staged copies reach 3.45 GB/s;
  eight or more interleave and drop to 2.35 GB/s. Uploads therefore pick
  their threads per destination: four for a discrete GPU, sixteen for the
  integrated one, whose copies are host memcpys.
- **What a release keeps.** Flash Next keeps its expert stacks, each cache's
  expert-device K/V buffer and its host copy of the token embeddings. Freeing
  and reallocating tens of GiB of shared memory took about a second each way,
  and the GPU driver keeps freed pages in its pool rather than returning them
  as available memory.
- **Start-up order.** `--swap-start released` runs on the serving thread
  after the scheduler has sized its prefix checkpoints from the live caches.

## Remaining work

- Keep what fits on a shared GPU while the other model runs: about 7.4 GiB of
  the R9700 is free while Flash Next runs (enough for Qwen's 1.9 GiB drafter)
  and about 5 GiB while Qwen runs (part of Flash Next's dense weights). Needs
  members to agree on how much each may keep, and weight buffers split by
  layer.
- `deepseek4`: its reinstate moves 26 GB over the R9700's link (dense weights,
  hot experts, drafter, KV cache), at best about 8 s; the three uploads run one
  after another today (about 12 s). Beside Qwen 3.8 27B on a 128 GB box only
  about 10 GiB of system memory stays available, and the box swaps.
- Swapping with `--max-concurrency` > 1 on `qwen35`, and paged serving.
- An SSD session cache for concurrent serving, and one disk budget shared by
  the members of a group.
