#pragma once

#include "ggml-backend.h"

// Axes are checked by the caller: paired tensors can store the same expert
// partition on different axes. Require identical layout representations, not
// just equal per-device totals, so local element ordering cannot differ.
inline bool ggml_backend_meta_split_layout_equal(
        const ggml_backend_meta_split_state & a,
        const ggml_backend_meta_split_state & b,
        size_t n_devices) {
    if (n_devices == 0 || n_devices > GGML_BACKEND_META_MAX_DEVICES ||
        a.n_segments == 0 || a.n_segments > sizeof(a.nr) / sizeof(a.nr[0]) ||
        a.n_segments != b.n_segments) {
        return false;
    }
    for (size_t s = 0; s < a.n_segments; ++s) {
        if (a.nr[s] != b.nr[s]) {
            return false;
        }
        for (size_t j = 0; j < n_devices; ++j) {
            if (a.ne[s*n_devices + j] != b.ne[s*n_devices + j]) {
                return false;
            }
        }
    }
    return true;
}
