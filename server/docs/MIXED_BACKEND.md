# CUDA/HIP mixed backend placement

This guide covers the bench/runtime harness paths for placing PFlash and
DFlash work across separate CUDA and HIP builds. The mixed-backend boundary is
kept at host-data or process boundaries:

- PFlash phase split passes compressed token/text data from `pflash_daemon` to
  the target run.
- DFlash draft split can run the draft model in a separate backend process and
  feed a target process through host IPC.
- Target layer split remains inside one backend binary; cross-backend target
  layer split is intentionally out of scope.

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
