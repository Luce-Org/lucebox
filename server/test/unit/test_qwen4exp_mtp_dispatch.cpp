// Host dispatch regression: links the CUDA/HIP backend, but needs no GPU/model.
#include "ggml.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>

// Internal host-only query from mmvq.cuh; avoid importing CUDA device headers.
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

int main() {
    // HIP defaults to the ungrouped path. Disable the CUDA default as well so
    // this synthetic architecture test never queries the physical device.
#ifdef _WIN32
    _putenv_s("LUCE_MMID_GROUPED", "0");
#else
    setenv("LUCE_MMID_GROUPED", "0", 1);
#endif
    constexpr int gfx1151 = 0x1000000 + 0x1151; // common.cuh AMD offset + gfx id
    const struct { ggml_type type; int regular; int invariant; } cases[] = {
        {GGML_TYPE_Q4_K, 4, 8},
        {GGML_TYPE_Q5_K, 4, 8},
        {GGML_TYPE_Q6_K, 4, 8},
        {GGML_TYPE_IQ4_NL, 6, 8},
        {GGML_TYPE_Q8_0, 8, 8},
        // Unsupported MMVQ types must remain on their original route.
        {GGML_TYPE_Q3_1_ROCMFP3_MIX, 0, 0},
        {GGML_TYPE_Q2_1_ROCMFP2_MIX, 0, 0},
    };
    int failures = 0;
    const bool prior = ggml_backend_cuda_set_mmvq_batch_invariant(false);
    // Recheck ordinary admission after leaving verification: no global leak.
    const bool modes[] = {false, true, false};
    for (const bool invariant : modes) {
        ggml_backend_cuda_set_mmvq_batch_invariant(invariant);
        for (const auto & c : cases) {
            const int actual = get_mmvq_mmid_max_batch(c.type, gfx1151);
            const int expected = invariant ? c.invariant : c.regular;
            if (actual != expected) {
                std::fprintf(stderr, "gfx1151 %s invariant=%d: batch limit %d != %d\n",
                             ggml_type_name(c.type), (int) invariant, actual, expected);
                ++failures;
            }
        }
    }
    ggml_backend_cuda_set_mmvq_batch_invariant(prior);
    std::printf("qwen4exp MTP dispatch: 21 cases, %d failures\n", failures);
    return failures != 0;
}
