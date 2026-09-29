// The Qwen3.5 drafter's in-graph causal mask against the CPU loop it
// replaced, byte for byte, on the ggml CPU backend.

#include "CppUnitTestFramework.hpp"

#include "pflash/qwen35_drafter.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Qwen35CausalMaskFixture {};

int align32(int x) { return ((x + 31) / 32) * 32; }

// The per-ubatch mask the drafter used to build on the host and upload.
std::vector<uint16_t> host_causal_mask(int kv_len, int n_tokens, int kv_start) {
    const int kv_pad = align32(kv_len);
    const int q_pad = align32(n_tokens);
    std::vector<uint16_t> out((size_t)kv_pad * q_pad, 0xFC00);
    for (int q = 0; q < n_tokens; ++q) {
        const int visible = std::min(kv_len, kv_start + q + 1);
        if (visible > 0) {
            std::memset(out.data() + (size_t)q * kv_pad, 0, (size_t)visible * sizeof(uint16_t));
        }
    }
    return out;
}

// Empty when the masks match, else where they first differ.
std::string compare_mask(ggml_backend_t backend, int kv_start, int n_tokens) {
    ggml_init_params ip{};
    ip.mem_size = 64 * ggml_tensor_overhead() + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * mask = luce::common::build_qwen35_causal_mask(ctx, gf, kv_start, n_tokens);
    ggml_set_output(mask);
    ggml_build_forward_expand(gf, mask);

    const std::string where = "kv_start=" + std::to_string(kv_start) +
                              " n=" + std::to_string(n_tokens);
    const std::vector<uint16_t> want = host_causal_mask(kv_start + n_tokens, n_tokens, kv_start);
    std::string error;
    if (mask->type != GGML_TYPE_F16 || !ggml_is_contiguous(mask) ||
        mask->ne[0] != align32(kv_start + n_tokens) || mask->ne[1] != align32(n_tokens) ||
        mask->ne[2] != 1 || mask->ne[3] != 1) {
        error = where + ": wrong mask type or shape";
    }
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (error.empty() && !ggml_gallocr_alloc_graph(alloc, gf)) {
        error = where + ": graph allocation failed";
    }
    if (error.empty() && ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        error = where + ": graph compute failed";
    }
    if (error.empty()) {
        std::vector<uint16_t> got(want.size());
        ggml_backend_tensor_get(mask, got.data(), 0, got.size() * sizeof(uint16_t));
        const auto diff = std::mismatch(got.begin(), got.end(), want.begin());
        if (diff.first != got.end()) {
            const size_t at = (size_t)(diff.first - got.begin());
            const size_t row = at / (size_t)mask->ne[0];
            error = where + ": differs at row " + std::to_string(row) + " key " +
                    std::to_string(at - row * (size_t)mask->ne[0]);
        }
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return error;
}

} // namespace

TEST_CASE(Qwen35CausalMaskFixture, device_mask_matches_host_mask_bytes) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);
    // Aligned ubatch starts, a resume and a checkpoint that are not, and
    // contexts to about 20K; n = 7 and 64 leave padded query rows.
    const int starts[] = {0, 1, 777, 1024, 5120, 18976, 19937};
    const int lengths[] = {1, 7, 64, 1000, 1024};
    for (const int kv_start : starts) {
        for (const int n : lengths) {
            const std::string error = compare_mask(backend, kv_start, n);
            if (!error.empty()) std::fprintf(stderr, "%s\n", error.c_str());
            CHECK(error.empty());
        }
    }
    ggml_backend_free(backend);
}
