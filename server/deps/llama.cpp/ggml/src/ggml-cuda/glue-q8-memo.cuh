#pragma once

// Q8 memo, the parts ggml-cuda.cu calls: whether it is on (always, unless a
// test turned it off), and the release of every live entry at the end of an evaluation
// (and wherever the pools may go away). The memo itself is in mmq.cu; which
// matmuls may share an entry is decided by glue-q8-memo.h.

#include "common.cuh"

// True unless a test turned it off (ggml_backend_cuda_set_q8_memo_override).
bool ggml_cuda_glue_q8_memo_on();

// Free every live entry of ctx.luce_glue_q8 and forget the current node.
void ggml_cuda_glue_q8_memo_clear(ggml_backend_cuda_context & ctx);
