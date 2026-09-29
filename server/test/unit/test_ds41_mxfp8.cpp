// MXFP8 dense weights (native DS4.1 E4M3 + E8M0 per 32): decode must be exact on every backend,
// the GEMV must not quantize activations, and a column's result must not depend on the batch width.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static constexpr int QK = 256, BLOCK = 264;

static double native_value(uint8_t code, uint8_t e) {
    const int exponent = (code >> 3) & 15, mantissa = code & 7;
    const double v = exponent ? std::ldexp(1.0 + mantissa / 8.0, exponent - 7) : std::ldexp((double) mantissa, -9);
    return std::ldexp(code & 128 ? -v : v, int(e) - 127);
}

// Random valid MXFP8 rows: codes avoid the two NaN encodings, scales in a realistic band.
static std::vector<uint8_t> random_rows(std::mt19937 & rng, int cols, int rows, int e_lo, int e_hi) {
    std::vector<uint8_t> data(size_t(rows) * (cols / QK) * BLOCK);
    std::uniform_int_distribution<int> code(0, 255), scale(e_lo, e_hi);
    for (size_t b = 0; b < data.size() / BLOCK; ++b) {
        uint8_t * p = data.data() + b * BLOCK;
        for (int g = 0; g < 8; ++g) p[g] = uint8_t(scale(rng));
        for (int j = 0; j < QK; ++j) {
            int c;
            do { c = code(rng); } while ((c & 0x7f) == 0x7f);
            p[8 + j] = uint8_t(c);
        }
    }
    return data;
}

static double weight(const std::vector<uint8_t> & w, int cols, int row, int col) {
    const uint8_t * b = w.data() + (size_t(row) * (cols / QK) + col / QK) * BLOCK;
    return native_value(b[8 + col % QK], b[(col % QK) / 32]);
}

struct Result { std::vector<float> out; bool ok; };

// dst[channel][token][row] = sum_col W[channel/ratio][row][col] * X[channel][token][col]
static Result run(ggml_backend_t backend, const std::vector<uint8_t> & w, const std::vector<float> & x,
                  int cols, int rows, int tokens, int wchannels, int channels) {
    ggml_init_params params{};
    params.mem_size = 16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false);
    params.no_alloc = true;
    auto ctx = ggml_init(params);
    auto a = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP8, cols, rows, wchannels);
    auto b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cols, tokens, channels);
    auto y = ggml_mul_mat(ctx, a, b);
    Result r{{}, ggml_backend_supports_op(backend, y)};
    if (!r.ok) { std::printf("  %s: mul_mat not supported\n", ggml_backend_name(backend)); ggml_free(ctx); return r; }
    auto graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, y);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(a, w.data(), 0, w.size());
    ggml_backend_tensor_set(b, x.data(), 0, x.size() * sizeof(float));
    r.ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (!r.ok) std::printf("  %s: graph compute failed\n", ggml_backend_name(backend));
    ggml_backend_synchronize(backend);
    r.out.resize(size_t(rows) * tokens * channels);
    ggml_backend_tensor_get(y, r.out.data(), 0, r.out.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return r;
}

// Every code under several scales, read back through one-hot columns: each output is one exact
// product, so any decode error shows. Tokens > 8 routes through the BF16 GEMM, <= 8 the GEMV.
static bool exact_decode(ggml_backend_t backend, int tokens) {
    const int cols = QK, rows = 4;
    std::vector<uint8_t> w(size_t(rows) * BLOCK);
    const uint8_t scales[4] = {100, 115, 127, 140};
    for (int r = 0; r < rows; ++r) {
        uint8_t * p = w.data() + r * BLOCK;
        for (int g = 0; g < 8; ++g) p[g] = scales[(r + g) % 4];
        for (int j = 0; j < QK; ++j) p[8 + j] = uint8_t((j & 0x7f) == 0x7f ? 0 : j);   // NaN codes -> 0
    }
    bool ok = true;
    for (int base = 0; base < cols && ok; base += tokens) {
        std::vector<float> x(size_t(cols) * tokens, 0.0f);
        for (int t = 0; t < tokens && base + t < cols; ++t) x[size_t(t) * cols + base + t] = 1.0f;
        auto res = run(backend, w, x, cols, rows, tokens, 1, 1);
        if (!res.ok) return false;
        for (int t = 0; t < tokens && base + t < cols; ++t)
            for (int r = 0; r < rows; ++r) {
                const double want = weight(w, cols, r, base + t);
                const float got = res.out[size_t(t) * rows + r];
                if (double(got) != want) {
                    std::printf("  decode mismatch col=%d row=%d got=%.9g want=%.9g\n", base + t, r, got, want);
                    ok = false;
                }
            }
    }
    std::printf("backend=%s exact_decode tokens=%d %s\n", ggml_backend_name(backend), tokens, ok ? "PASS" : "FAIL");
    return ok;
}

static bool accuracy(ggml_backend_t backend, std::mt19937 & rng, int cols, int rows, int tokens, int wchannels, int channels) {
    auto w = random_rows(rng, cols, rows * wchannels, 105, 125);
    std::normal_distribution<float> n(0.0f, 1.0f);
    std::vector<float> x(size_t(cols) * tokens * channels);
    for (auto & v : x) v = n(rng);
    auto res = run(backend, w, x, cols, rows, tokens, wchannels, channels);
    if (!res.ok) { std::printf("  unsupported or failed\n"); return false; }
    // GEMV: F32 activations, F32 accumulation. Wider: BF16 activations (relative 2^-9 each).
    const bool gemv = tokens <= 8;
    double worst = 0;
    bool ok = true;
    const int ratio = channels / wchannels;
    for (int c = 0; c < channels; ++c)
        for (int t = 0; t < tokens; ++t)
            for (int r = 0; r < rows; ++r) {
                double sum = 0, mag = 0;
                for (int k = 0; k < cols; ++k) {
                    const double wv = weight(w, cols, (c / ratio) * rows + r, k);
                    const double xv = x[(size_t(c) * tokens + t) * cols + k];
                    sum += wv * xv; mag += std::fabs(wv * xv);
                }
                const double got = res.out[(size_t(c) * tokens + t) * rows + r];
                const double err = std::fabs(got - sum) / std::max(mag, 1e-30);
                worst = std::max(worst, err);
                if (!std::isfinite(got) || err > (gemv ? 2e-6 : 8e-3)) ok = false;
            }
    std::printf("backend=%s cols=%d rows=%d tokens=%d wchannels=%d channels=%d path=%s max_rel=%.3g %s\n",
                ggml_backend_name(backend), cols, rows, tokens, wchannels, channels, gemv ? "gemv" : "bf16-gemm",
                worst, ok ? "PASS" : "FAIL");
    return ok;
}

// Column j of an n-column GEMV must equal the 1-column result bit for bit.
static bool batch_invariant(ggml_backend_t backend, std::mt19937 & rng) {
    const int cols = 5120, rows = 96;
    auto w = random_rows(rng, cols, rows, 105, 125);
    std::normal_distribution<float> n(0.0f, 1.0f);
    std::vector<float> x(size_t(cols) * 8);
    for (auto & v : x) v = n(rng);
    bool ok = true;
    for (int width = 2; width <= 8; ++width) {
        auto wide = run(backend, w, std::vector<float>(x.begin(), x.begin() + size_t(cols) * width), cols, rows, width, 1, 1);
        for (int t = 0; t < width; ++t) {
            auto one = run(backend, w, std::vector<float>(x.begin() + size_t(cols) * t, x.begin() + size_t(cols) * (t + 1)), cols, rows, 1, 1, 1);
            if (std::memcmp(one.out.data(), wide.out.data() + size_t(t) * rows, rows * sizeof(float))) ok = false;
        }
    }
    std::printf("backend=%s batch_invariant %s\n", ggml_backend_name(backend), ok ? "PASS" : "FAIL");
    return ok;
}

static bool host_checks() {
    std::mt19937 rng(7);
    auto w = random_rows(rng, QK, 3, 100, 130);
    const auto * traits = ggml_get_type_traits(GGML_TYPE_MXFP8);
    std::vector<float> f(3 * QK), g(3 * QK);
    traits->to_float(w.data(), f.data(), 3 * QK);
    bool ok = true;
    for (int r = 0; r < 3; ++r)
        for (int k = 0; k < QK; ++k)
            if (double(f[r * QK + k]) != weight(w, QK, r, k)) ok = false;
    std::vector<uint8_t> q(w.size());
    ggml_quantize_chunk(GGML_TYPE_MXFP8, f.data(), q.data(), 0, 3, QK, nullptr);
    traits->to_float(q.data(), g.data(), 3 * QK);
    if (std::memcmp(f.data(), g.data(), f.size() * sizeof(float))) ok = false;   // values survive requantization
    if (!ggml_validate_row_data(GGML_TYPE_MXFP8, w.data(), w.size())) ok = false;
    auto bad = w; bad[8 + 5] = 0x7f;
    if (ggml_validate_row_data(GGML_TYPE_MXFP8, bad.data(), bad.size())) ok = false;
    bad = w; bad[3] = 247;
    if (ggml_validate_row_data(GGML_TYPE_MXFP8, bad.data(), bad.size())) ok = false;
    std::printf("host decode/requantize/validate %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char ** argv) {
    const bool cpu = argc > 1 && !std::strcmp(argv[1], "cpu");
    if (!host_checks()) return 3;
    const int count = cpu ? 1 : ggml_backend_cuda_get_device_count();
    if (!count) return 77;
    for (int i = 0; i < count; ++i) {
        auto backend = cpu ? ggml_backend_cpu_init() : ggml_backend_cuda_init(i);
        if (!backend) return 2;
        std::mt19937 rng(1234 + i);
        bool ok = exact_decode(backend, 1) && exact_decode(backend, 8) && exact_decode(backend, 16);
        const std::vector<std::vector<int>> shapes = {
            {256, 1, 1, 1, 1}, {1280, 37, 1, 1, 1}, {1280, 37, 3, 1, 1}, {5120, 512, 1, 1, 1}, {5120, 64, 8, 1, 1},
            {2304, 128, 5, 1, 1}, {4096, 64, 1, 8, 8}, {4096, 64, 3, 8, 8}, {1280, 32, 2, 1, 4},
            {5120, 64, 9, 1, 1}, {2304, 128, 32, 1, 1}, {4096, 64, 12, 8, 8}};
        for (const auto & s : shapes) ok = ok && accuracy(backend, rng, s[0], s[1], s[2], s[3], s[4]);
        if (!cpu) ok = ok && batch_invariant(backend, rng);
        ggml_backend_free(backend);
        if (!ok) return 3;
    }
    return 0;
}
