# Vendored llama.cpp/ggml snapshot

This directory contains the ggml-only subset used by Lucebox Hub.

- Source repository: https://github.com/Luce-Org/lucebox-ggml
- Source base branch: `luce-dflash`
- Source base commit: `6fbe72d67069136bbd370be703e1d4f441b5e942`
- Included merged PR: `#35` (`0fe65d9354b7c5da52a7741d2e37ba85f0d0c925`)
- Included test PR: `#37` (`0699be81480428f01b9b7ac49a09a2d51c77f8df`)
- Included upstream backport: `llama.cpp #22298` (`9725a313be0528214c4a02fed906ddaf7b3f712e`)
- Included tensor-parallel source PR: `Luce-Org/lucebox-ggml#39` (`a6eb14b8d678c23f111b7acfcfe6b51b2ea95c46`)
- Reconstruction: `luce-dflash@6fbe72d67069136bbd370be703e1d4f441b5e942` plus cherry-picked PRs `#35`, `#37`, upstream `llama.cpp #22298`, and the GGML delta from `Luce-Org/lucebox-ggml#39`
- Vendored paths: `LICENSE`, `common/jinja`, `common/log.h`, `common/unicode.*`, `ggml`, `gguf-py`

Open ggml feature PRs are not included unless they are explicitly listed above as a vendored source.

## Hub-local heterogeneous execution patches

PR `Luce-Org/lucebox#505` carries a temporary, reviewable patch set on top of
the snapshot above for AMD heterogeneous MoE execution:

- scheduler support for late cross-backend joins and deferred peer copies;
- ROCmFP quantized MMVQ/MMID kernels and grouped expert execution;
- fused MoE owner/combine operations and DeepSeek V4 HC/indexer kernels; and
- thread-local CUDA/HIP graph policy overrides used by fixed-topology graphs.

These changes are limited to `ggml/`. Keep their public declarations in
`ggml/include`, avoid DeepSeek-specific policy in generic kernels, and update
this provenance when the patch set is moved to `lucebox-ggml`.

## Hub-local DS4V HIP fused-bias linear

The local `GGML_OP_MUL_MAT_BIAS_BF16` operation is appended after `PAGED_ATTN`;
all existing op numeric values are preserved, while `GGML_OP_COUNT` grows from
105 to 106; the existing RPC header contract advances protocol patch 5 to 6.
The RPC registry does not expose the HIP capability, so vision never sends this
new operation through RPC; RPC supports_op also rejects it locally. Protocol
patch mismatches only warn, so this does not rely on a version handshake to
reject older peers. Rebuild ggml-base, CPU/HIP backends, and consumers together. Do not
mix old shared libraries with this header or serialize the new operation for
an older reader. This is an inference-only extension; CPU compute/backward
reject it, and HIP alone advertises the explicit registry capability. Other
backends are never selected by a generic unknown-op supports default.

Only DS4V biased linears opt in. Ordinary text MUL_MAT/ADD fusion, unbiased
vision linears, and CPU/NVIDIA vision graph construction are unchanged. The
HIP-only CMake dependency is official hipBLASLt (`roc::hipblaslt`). The Lt
configuration follows the frozen PyTorch revision
`3d3aa833db84eed6b7f5595cb5f162c2f78300a4`: BF16 W/X/bias/output, F32 compute and
scalars, T/N, alpha=1, beta=0, bias epilogue, C=D, one first heuristic with a
76 MiB workspace maximum. There is no algorithm sweep or arithmetic fallback.

One exact-size 76 MiB workspace and one Lt handle belong to each HIP backend
context that actually uses the operation. An event orders shared workspace
reuse on the actual execution stream; destruction waits for its last use.
The workspace is retained outside the ggml arena and is conservatively
included in VisionRuntime's scratch reservation/report even after arena
release. Graphs containing this operation are capture-ineligible; no global
text graph policy changes. Descriptors are per invocation, not cached.
Qualification is scoped to the Radeon RX 7900 XT and pinned ROCm/PyTorch
reference; availability on other HIP devices is not a qualification claim.
