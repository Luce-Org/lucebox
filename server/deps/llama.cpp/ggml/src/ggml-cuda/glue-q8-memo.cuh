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

// Whether ggml_cuda_mul_mat runs a MUL_MAT node on its plain MMQ branch.
typedef bool (*ggml_cuda_glue_q8_takes_mmq_t)(const ggml_backend_cuda_context & ctx, const ggml_tensor * mm);

// The producer at node `at` of `graph` (an RMS norm or a SWIGLU) writes the
// q8_1 MMQ rows of its F32 output itself. Where the memo would serve a matmul reading that
// output (the conditions of luce_glue_q8_acquire) and the first reader that
// runs MMQ (glue_q8_first_reader with `takes_mmq`) is found, this registers
// the memo entry that reader would have made, created at `at` so that the
// reader reuses it too, and returns its buffer: block_q8_1_mmq rows in the
// reader's layout (*layout, an mmq_q8_1_ds_layout). Null and *layout = -1
// otherwise: the producer then writes only its F32 output, as before.
char * ggml_cuda_glue_q8_memo_produce(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph, int at,
                                      ggml_cuda_glue_q8_takes_mmq_t takes_mmq, int * layout);
