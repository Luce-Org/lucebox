# CUDA/HIP mixed backend placement

This guide covers the bench/runtime harness paths for placing PFlash and
DFlash work across separate CUDA and HIP builds. The mixed-backend boundary is
kept at host-data or process boundaries:

- PFlash phase split passes compressed token/text data from `pflash_daemon` to
  the target run.
- DFlash draft split can run the draft model in a separate backend process and
  feed a target process through host IPC.
- Target layer split can run across two backends: `dflash_server` runs the
  head layer group on its compiled backend and hands the boundary activation
  to a remote `backend_ipc_daemon` (other backend) for the tail group. DFlash
  speculative decode works across this boundary.

## Build CUDA and HIP binaries

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DDFLASH27B_GPU_BACKEND=cuda
cmake --build build-cuda --target pflash_daemon test_dflash backend_ipc_daemon -j

cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES=<your-gfx-arch>
cmake --build build-hip --target pflash_daemon test_dflash backend_ipc_daemon -j
```

## PFlash phase split

PFlash targets the prefill side of long-context requests. The phase-split
harness keeps the PFlash drafter resident in `pflash_daemon`, then can launch a
separate target generation pass on another backend.

- `--pflash-backend cuda|hip` and `--pflash-visible-devices` select the
  backend/device set used by `pflash_daemon`.
- `--run-target` launches `test_dflash` after compression.
- `--target-backend cuda|hip` and `--target-visible-devices` select the target
  backend/device environment.
- Reports include compressed token/text output, PFlash timing, target timing,
  target return code, and GPU resource summaries for both sides.

Example: HIP PFlash drafter followed by CUDA target layer split:

```bash
python scripts/phase_split_dual_gpu.py bench-niah \
  --build-dir build-hip \
  --pflash-backend hip \
  --pflash-visible-devices 0 \
  --run-target \
  --target-bin build-cuda/test_dflash \
  --target-backend cuda \
  --target-visible-devices 0,1 \
  --target-gpus 0,1 \
  --target-layer-split 1,1 \
  --target-gen-tokens 8 \
  --contexts 4096 \
  --local-files-only \
  --report-dir reports/pflash_hybrid_hip_drafter_cuda_target
```

Compress a real prompt without running target generation:

```bash
python scripts/phase_split_dual_gpu.py run-prompt \
  --build-dir build-cuda \
  --prompt-file /path/to/prompt.txt \
  --local-files-only \
  --report-dir reports/pflash_phase_split_prompt
```

## DFlash draft split

For DFlash, the target process can launch a separate backend IPC daemon from a
different backend build. The target process keeps target execution and any
target layer split inside its own backend binary.

Use these `test_dflash` options for the target process:

- `--draft-ipc-bin <path>` points to the other backend's `backend_ipc_daemon`
  binary.
- `--draft-ipc-gpu <id>` selects the draft daemon device in that backend's
  visible-device namespace.
- `--draft-ipc-work-dir <path>` selects where temporary IPC payload files are
  written.
- `--draft-ipc-ring-cap <n>` controls the remote draft feature-ring capacity.

`scripts/bench_he.py` passes these options through for HumanEval-style
validation runs.


## Mixed-backend target layer split

`dflash_server` can split the target across two backends in a single decode:
the local process (this binary's compiled backend) runs the first contiguous
layer group, then hands the boundary activation to a remote `backend_ipc_daemon`
(built for the other backend) that runs the remaining layers, the final norm,
and the LM-head projection. DFlash verify, target feature capture, KV
snapshot/restore, and draft-token projection all work across this boundary.

Use these `dflash_server` flags:

| Flag | Purpose |
|---|---|
| `--target-devices <list>` | Comma-separated per-shard devices; mixed backends allowed, e.g. `cuda:0,hip:0`. The first group must be this binary's compiled backend (the local group); the remainder forms one remote backend group. |
| `--target-layer-split <weights>` | Proportional layer split across the shards, e.g. `0.5,0.5`. |
| `--target-shard-ipc-bin <path>` | Remote `backend_ipc_daemon` built for the remote backend. Required for mixed target split. |
| `--target-shard-ipc-work-dir <path>` | Scratch directory for the remote target-shard IPC payload files. |

Only one backend boundary is supported: one local group followed by one remote
group (e.g. `cuda:0` local + `hip:0,hip:1` remote).

Example: a CUDA-built server running the head layers on an RTX GPU, with the
tail layers + LM head on a HIP `backend_ipc_daemon` (Strix Halo iGPU):

```bash
./build-cuda/dflash_server models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft models/draft/dflash-draft-3.6-q4_k_m.gguf \
  --target-devices cuda:0,hip:0 \
  --target-layer-split 0.5,0.5 \
  --target-shard-ipc-bin build-hip/backend_ipc_daemon \
  --target-shard-ipc-work-dir /tmp/target_shard \
  --host 127.0.0.1 --port 18080
```
