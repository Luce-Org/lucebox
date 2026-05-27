#pragma once
// Chunked gated delta-net graph builder.
// See src/delta_net_chunked.cpp for the algorithm and history.

#include <ggml.h>

namespace dflash::common {

struct DeltaNetChunkedResult {
    ggml_tensor * output;     // [S_v, H_v, n_tokens, n_seqs]
    ggml_tensor * new_state;  // [S_v, S_v, H_v, n_seqs]
};

// Chain-only, no-capture, no-tree variant. Caller passes q/k/v/g/b/s in the
// same shape as ggml_gated_delta_net expects. Returns the per-token output
// and the final recurrent state as two separate tensors (unlike the fused
// kernel which packs them into one dst tensor).
DeltaNetChunkedResult build_delta_net_chunked(
        ggml_context * ctx0,
        ggml_tensor  * q,
        ggml_tensor  * k,
        ggml_tensor  * v,
        ggml_tensor  * g,
        ggml_tensor  * b,
        ggml_tensor  * s);

} // namespace dflash::common
