#pragma once

struct ds4_ratio4_visibility {
    int raw_first;
    int raw_last;
    int comp_first;
    int comp_last;
};

// Shared by the device bounds kernel and its host-side brute-force oracle.
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
// `ratio` is the layer's compression ratio: a compressed row becomes
// visible once its `ratio` source tokens are complete (V4: 4; V4.1: 1, 2).
static inline ds4_ratio4_visibility ds4_ratio4_causal_visibility(
        int token, int n_tokens, int raw_rows, int n_comp_rows,
        int raw_window, int kv_start, int ratio = 4) {
    const int prior_rows = raw_rows - n_tokens;
    const int first = prior_rows + token - raw_window + 1;
    const int completed = (kv_start + token + 1) / ratio;
    const int visible = n_comp_rows < completed ? n_comp_rows : completed;
    return {first > 0 ? first : 0, prior_rows + token,
            visible > 0 ? raw_rows : raw_rows + n_comp_rows,
            visible > 0 ? raw_rows + visible - 1 : -1};
}
