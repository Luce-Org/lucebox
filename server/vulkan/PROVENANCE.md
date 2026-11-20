# Source and license provenance

Base: https://github.com/Luce-Org/lucebox at
2a8ea2f48f6c1014184e5d606bbf6f0788958b1a (Apache-2.0, root LICENSE).
Upstream source files retain their history and authorship. The shared router is
`server/src/common/moe_router_graph.h`, used directly, not copied or relabeled.
The existing HTTP server is changed only at backend-scoped policy/BOS hooks.

Vulkan ggml: https://github.com/ggml-org/llama.cpp at
c061df19838ff60970faf54fd7e414953590125d. Public archive:
https://codeload.github.com/ggml-org/llama.cpp/tar.gz/c061df19838ff60970faf54fd7e414953590125d
SHA256 d345a35f541d8f23ca519df580ed9f1ffcdd3a6f46a54ac97c80659fad8694d3.
It carries MIT, Copyright (c) 2023-2026 The ggml authors. LICENSE.ggml is the
verbatim license; CMake preserves the archive's original notices. Graph
semantics were ported with reference to `src/models/lfm2moe.cpp`, the recurrent
short-convolution graph and ggml operators in that public tree. No authorship
of that upstream work is claimed here. The local graph/backend/launcher glue
is an AI-assisted contribution prepared by Astra under Drew's direction, not
an upstream ggml implementation or an endorsement by its authors.

The only dependency delta is `../cmake/patches/vulkan-w9100-rows.patch`, an
opt-in AMD device 0x67a0 K-quant rows/workgroup override. It is separately
reviewable from the native backend. The later pin supplies required Vulkan
operations/APIs without changing the default CUDA/HIP vendor. Do not replace
`server/deps/llama.cpp/ggml` wholesale: it contains Lucebox-specific changes.

Native Qwen2 Vulkan source and finite-only greedy selector originate in the
previous isolated, locally accepted Astra Qwen implementation. Native LFM
source originates in the separately accepted Astra LFM implementation, including
its convolution-layout and final-chunk/final-four correctness fixes. This
integration adapts those contributions to current upstream and keeps Qwen2 as
an additive architecture selection. New files follow Lucebox's Apache-2.0 terms,
with the incorporated/reference-derived ggml portions retaining MIT notices.
No weights, generated shaders, build binaries or earlier evidence are vendored.

Third-party JSON, Jinja/tokenizer and remaining Lucebox dependencies are reused
unchanged from upstream or the documented system packages. Their licenses and
notices are not rewritten. Model weights are external and are not licensed or
distributed by this source patch. No publication has occurred.
