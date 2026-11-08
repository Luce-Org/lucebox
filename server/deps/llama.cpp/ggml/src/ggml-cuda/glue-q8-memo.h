#pragma once

// Q8 memo: which later matmuls of one graph may
// reuse the q8_1 MMQ activation a matmul quantizes. Pure host code over ggml
// graph nodes: mmq.cu applies it per MMQ call (luce_glue_q8_acquire), and
// test_glue_fusions checks it without a GPU.
//
// MMQ quantizes its F32 src1 into a pool buffer before every launch
// (quantize_mmq_q8_1_cuda). What that buffer holds depends only on the bytes
// src1 covers, its shape and strides, and the q8_1 layout of the weight type
// (mmq_get_q8_1_ds_layout: D4, DS4 or D2S6). Several matmuls of a layer read
// the same activation (attn_qkv|gate and ssm_beta|alpha, q/k/v, ffn_gate and
// ffn_up), so a later one of the same layout can read the earlier buffer
// instead of quantizing again, as long as nothing wrote those bytes between
// the two.
//
// The memo is keyed by the reshape root of src1, its address and its ne,
// for an F32 src1 that is contiguous and lives in a buffer; with ne and
// contiguity equal, the same bytes are read in the same order. Before an
// entry is created, glue_q8_scan walks the graph forward from the quantizing
// node and stops at the first node that may write those bytes:
//   - its own output overlaps them (an in-place op, a copy into them, or a
//     node gallocr placed over them after their last reader), or
//   - it reads them and is not a matmul (any op that writes a source in
//     place, such as a fused state update, counts as a writer).
// Only matmuls before that node can reuse the entry (clean_until), and the
// last of them that reads the same key with the same layout bounds its life
// (last_reader). Nodes the CUDA loop never launches (views, reshapes, nodes
// without the compute flag) write nothing.
//
// Bitwise: a reused buffer holds what quantize_mmq_q8_1 wrote for the same
// bytes, shape and layout, and that kernel is deterministic (fixed shuffle
// reductions per block, no atomics). The tail past the quantized blocks
// (get_mmq_x_max_host blocks of padding, the same size for every MMQ call on
// a device) is never written by the quantizer on either path and only feeds
// output columns past ncols, which MMQ does not store.

#include "ggml.h"

#include <cstddef>
#include <cstdint>

struct glue_q8_plan {
    int last_reader = -1;  // last later node predicted to reuse the entry; -1: none
    int clean_until = -1;  // last node index before the bytes may be rewritten
};

// Nodes ggml_cuda_graph_evaluate_and_capture never launches
// (ggml_cuda_node_is_noop): they write nothing.
inline bool glue_q8_node_launches_nothing(const ggml_tensor * n) {
    return ggml_is_empty(n) || n->op == GGML_OP_NONE || n->op == GGML_OP_VIEW ||
           n->op == GGML_OP_RESHAPE || n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE ||
           (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0;
}

// src[0], not view_src: view_src is the allocation root and would skip an
// in-place node between the reshape and that root (as the MMVQ memo keys it).
inline const ggml_tensor * glue_q8_reshape_root(const ggml_tensor * t) {
    while (t->op == GGML_OP_RESHAPE && t->src[0] != nullptr) {
        t = t->src[0];
    }
    return t;
}

// A src1 the memo can key: F32, contiguous, in a real buffer (synthetic
// tensors built inside a fused operator have none).
inline bool glue_q8_keyable(const ggml_tensor * t) {
    return t != nullptr && t->type == GGML_TYPE_F32 && t->buffer != nullptr && t->data != nullptr &&
           ggml_is_contiguous(t);
}

// Two src1 tensors that read the same bytes in the same order.
inline bool glue_q8_same_input(const ggml_tensor * a, const ggml_tensor * b) {
    if (!glue_q8_keyable(a) || !glue_q8_keyable(b) || a->data != b->data) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
    }
    return glue_q8_reshape_root(a) == glue_q8_reshape_root(b);
}

// Integer addresses: relational comparison of pointers into different
// allocations is unspecified in C++.
inline bool glue_q8_overlap(const void * a, size_t an, const void * b, size_t bn) {
    const uintptr_t pa = (uintptr_t) a;
    const uintptr_t pb = (uintptr_t) b;
    return an > 0 && bn > 0 && pa < pb + bn && pb < pa + an;
}

inline bool glue_q8_tensor_overlaps(const ggml_tensor * t, const void * data, size_t nbytes) {
    return t != nullptr && t->data != nullptr && glue_q8_overlap(t->data, ggml_nbytes(t), data, nbytes);
}

// Whether node `n` (one the loop launches) may write [data, data + nbytes).
inline bool glue_q8_may_write(const ggml_tensor * n, const void * data, size_t nbytes) {
    if (glue_q8_tensor_overlaps(n, data, nbytes)) {
        return true;
    }
    if (n->op == GGML_OP_MUL_MAT || n->op == GGML_OP_MUL_MAT_ID) {
        return false;  // a matmul writes only its own output
    }
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        if (glue_q8_tensor_overlaps(n->src[s], data, nbytes)) {
            return true;
        }
    }
    return false;
}

// From the matmul at nodes[at], which quantizes `src1` to q8_1 layout
// `layout`: the later matmuls (within `window` nodes) that read the same
// input with the same layout before anything may write it. layout_of(type)
// is the MMQ q8_1 layout of a weight type, negative where MMQ has none.
template <typename LayoutOf>
inline glue_q8_plan glue_q8_scan(ggml_tensor * const * nodes, int n_nodes, int at, const ggml_tensor * src1,
                                 int layout, LayoutOf layout_of, int window) {
    glue_q8_plan plan;
    plan.clean_until = at;
    if (!glue_q8_keyable(src1) || at < 0 || at >= n_nodes) {
        return plan;
    }
    const void * data = src1->data;
    const size_t nbytes = ggml_nbytes(src1);
    const int end = (window > 0 && n_nodes - at - 1 > window) ? at + 1 + window : n_nodes;
    for (int j = at + 1; j < end; ++j) {
        const ggml_tensor * n = nodes[j];
        if (glue_q8_node_launches_nothing(n)) {
            plan.clean_until = j;
            continue;
        }
        if (glue_q8_may_write(n, data, nbytes)) {
            return plan;  // clean_until stays at j - 1
        }
        plan.clean_until = j;
        if (n->op == GGML_OP_MUL_MAT && n->src[0] != nullptr && ggml_is_quantized(n->src[0]->type) &&
            glue_q8_same_input(n->src[1], src1) && layout_of(n->src[0]->type) == layout) {
            plan.last_reader = j;
        }
    }
    return plan;
}
