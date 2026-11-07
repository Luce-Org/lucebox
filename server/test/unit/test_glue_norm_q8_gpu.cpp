// The q8_1 producers on the GPU: the RMS norms (ADD + RMS_NORM + MUL,
// RMS_NORM + MUL) and SWIGLU that also write the q8_1 MMQ rows of their
// output, registered in the Q8 memo so that the MMQ reading it next skips its
// quantize. SKIP (77) without a GPU.
//
// Every arm runs in a child process (this binary with --child; the parent
// passes the arm in LUCE_TEST_NORM_Q8*, test-only messages that the child
// turns into the test overrides), and the parent compares:
//   kernel     the kernels alone, on raw device pointers
//   off        the graphs, producers off
//              (ggml_backend_cuda_set_glue_q8_producers_override): the
//              readers quantize, the path before the producers
//   on         the graphs, the defaults
//   on_nomemo  the graphs with the Q8 memo turned off by its test override:
//              there is no entry to write into, so no producer writes
//
// Kernel arm: for row lengths 96..17408 (with and without row padding to
// MATRIX_ROW_PADDING) and 1..64 rows, each producer with its q8_1 MMQ rows
// for a D4 (Q8_0), DS4 (Q4_K) and D2S6 (Q2_K) reader must write, byte for
// byte, what quantize_mmq_q8_1 writes for the producer's own F32 output (both
// buffers prefilled alike, compared past the rows too, so nothing more is
// written), and the same F32 output (and sum) as the kernel without them.
// Every row holds one all-zero 32-value block (amax 0, as in row padding).
//
// Graph arms: one target-like graph at 1, 5 and 16 columns (K = 1024):
//   s  = x + r;            n1 = rms_norm(s) * g1       ADD + RMS_NORM + MUL
//     read by Q4_K, Q5_K (DS4), then Q8_0, IQ4_XS (D4)
//   n2 = rms_norm(m_a) * g2                             RMS_NORM + MUL
//     read by Q8_0, Q6_K (D4)
//   n3 = scale_inplace(rms_norm(m_a) * g3) read by IQ4_XS: rewritten before
//     its reader, so no norm write
//   s2 = s + m_a;          n4 = rms_norm(s2) * g4       read by Q2_K, Q2_K (D2S6)
//   s3 = s2 + n2;          n5 = rms_norm(s3) * g5       read by one Q4_K (alone,
//     as the stacked ffn_gate|ffn_up GEMM reads post-attention norm)
//   n6 = swiglu(m_b, m_c)                               SWIGLU, read by Q8_0 (D4)
//     at K = 256, rows padded to MATRIX_ROW_PADDING
// Every arm must hash-equal off on every output (the matmuls and the
// producers' F32 outputs) with the same MMQ calls. On a legacy pool at MMQ
// widths (5, 16) off makes kBaseHits memo hits; on writes kProducerWrites
// producers' q8_1 rows (n1, n2, n4, n5, n6, not n3) and hits once more per
// write (the first reader's quantize); on_nomemo writes none and hits none.
// At 1 column the readers run MMVQ: no writes, no hits.
//
//   test_glue_norm_q8_gpu            all arms, compare
//   test_glue_norm_q8_gpu --child    one arm, the environment decides

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

extern "C" char ** environ;

// Parity-test entry points (ggml-cuda norm.cu, quantize.cu).
extern "C" int ggml_cuda_test_norm_q8_mmq(int kind, const float * a, const float * b, float * sum_out,
                                          float * dst, const float * w, int ncols, int nrows, float eps,
                                          void * q8, int q8_type, void * stream);
extern "C" int ggml_cuda_test_quantize_mmq_q8_1(const float * x, void * q8, int q8_type, int ncols, int nrows,
                                                void * stream);

namespace {

uint64_t fnv1a(const void * data, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto * p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

struct Rng {
    uint64_t s;
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    float uniform() { return (float) ((next() >> 40) & 0xffffff) / 8388608.0f - 1.0f; }
};

// ── kernel arm ──────────────────────────────────────────────────────────────

constexpr int kMaxCols = 17408;
constexpr int kMaxRows = 64;
constexpr size_t kGuard = 4096;   // bytes compared past the q8_1 rows

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

// block_q8_1_mmq rows of [ncols, nrows]: 144 bytes per 128 padded values.
size_t q8_bytes(int ncols, int nrows) { return (size_t) nrows * (size_t) (pad512(ncols) / 128) * 144; }

const char * type_name(int t) { return ggml_type_name((ggml_type) t); }

int run_kernel_arm() {
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (gpu == nullptr) {
        std::printf("skip no GPU\n");
        return 77;
    }
    ggml_init_params ip{ggml_tensor_overhead() * 16, nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    const int64_t nf = (int64_t) kMaxCols * kMaxRows;
    const int64_t nq = (int64_t) (q8_bytes(kMaxCols, kMaxRows) + kGuard);
    ggml_tensor * ta    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nf);
    ggml_tensor * tb    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nf);
    ggml_tensor * tsum0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nf);
    ggml_tensor * tdst0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nf);
    ggml_tensor * tsum  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nf);
    ggml_tensor * tdst  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nf);
    ggml_tensor * tw    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kMaxCols);
    ggml_tensor * tqp   = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, nq);
    ggml_tensor * tqr   = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, nq);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    if (buf == nullptr) {
        std::printf("error kernel: device allocation failed\n");
        ggml_free(ctx);
        ggml_backend_free(gpu);
        return 1;
    }
    {
        std::vector<float> w((size_t) kMaxCols);
        Rng rng{424242};
        for (float & v : w) v = 1.0f + 0.5f * rng.uniform();
        ggml_backend_tensor_set(tw, w.data(), 0, w.size() * sizeof(float));
    }

    const int lengths[] = {96, 256, 1024, 1056, 2080, 5120, 6144, 17408};
    const int rows[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 33, 47, 48, 63, 64};
    const int types[] = {GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q2_K};   // D4, DS4, D2S6
    const float eps = 1e-6f;

    int cases = 0, q8_fail = 0, f32_fail = 0, api_fail = 0, printed = 0;
    auto report = [&](const std::string & what) {
        if (printed++ < 24) std::printf("error kernel %s\n", what.c_str());
    };
    std::vector<float> a, b, dst0, sum0, dst, sum;
    std::vector<uint8_t> qp, qr;
    for (int kind = 0; kind < 2; ++kind) {
        for (const int L : lengths) {
            for (const int R : rows) {
                const size_t n = (size_t) L * R;
                a.resize(n);
                b.resize(n);
                Rng rng{(uint64_t) (1000003 * (kind + 1) + 7919 * L + R)};
                for (int r = 0; r < R; ++r) {
                    const float mag = std::ldexp(1.0f, r % 5 - 2);   // rows of different magnitude
                    for (int c = 0; c < L; ++c) {
                        a[(size_t) r * L + c] = mag * rng.uniform();
                        b[(size_t) r * L + c] = mag * rng.uniform();
                    }
                    const int z = (r * 7 + 3) % (L / 32) * 32;   // one all-zero 32-value block per row
                    for (int c = z; c < z + 32; ++c) {
                        a[(size_t) r * L + c] = 0.0f;
                        b[(size_t) r * L + c] = 0.0f;
                    }
                }
                ggml_backend_tensor_set(ta, a.data(), 0, n * sizeof(float));
                ggml_backend_tensor_set(tb, b.data(), 0, n * sizeof(float));
                const std::string where = "kind=" + std::to_string(kind) + " ncols=" + std::to_string(L) +
                                          " rows=" + std::to_string(R);
                if (!ggml_cuda_test_norm_q8_mmq(kind, (const float *) ta->data, (const float *) tb->data,
                                                (float *) tsum0->data, (float *) tdst0->data,
                                                (const float *) tw->data, L, R, eps, nullptr, 0, nullptr)) {
                    ++api_fail;
                    report(where + ": plain launch failed");
                    continue;
                }
                dst0.resize(n);
                ggml_backend_tensor_get(tdst0, dst0.data(), 0, n * sizeof(float));
                if (kind == 0) {
                    sum0.resize(n);
                    ggml_backend_tensor_get(tsum0, sum0.data(), 0, n * sizeof(float));
                }
                const size_t nb = q8_bytes(L, R) + kGuard;
                for (const int type : types) {
                    ++cases;
                    const std::string at = where + " " + type_name(type);
                    ggml_backend_tensor_memset(tqp, 0x5A, 0, nb);
                    ggml_backend_tensor_memset(tqr, 0x5A, 0, nb);
                    if (!ggml_cuda_test_norm_q8_mmq(kind, (const float *) ta->data, (const float *) tb->data,
                                                    (float *) tsum->data, (float *) tdst->data,
                                                    (const float *) tw->data, L, R, eps, tqp->data, type, nullptr) ||
                        !ggml_cuda_test_quantize_mmq_q8_1((const float *) tdst->data, tqr->data, type, L, R,
                                                          nullptr)) {
                        ++api_fail;
                        report(at + ": q8 launch failed");
                        continue;
                    }
                    qp.resize(nb);
                    qr.resize(nb);
                    ggml_backend_tensor_get(tqp, qp.data(), 0, nb);
                    ggml_backend_tensor_get(tqr, qr.data(), 0, nb);
                    if (std::memcmp(qp.data(), qr.data(), nb) != 0) {
                        ++q8_fail;
                        size_t off = 0;
                        while (off < nb && qp[off] == qr[off]) ++off;
                        const size_t rows_end = q8_bytes(L, R);
                        const size_t blk = off / 144, in = off % 144;
                        report(at + ": q8_1 rows differ from quantize_mmq_q8_1 at byte " + std::to_string(off) +
                               (off >= rows_end ? std::string(" (past the rows)")
                                                : " (block " + std::to_string(blk) + " = k-block " +
                                                      std::to_string(blk / R) + " of row " +
                                                      std::to_string(blk % R) +
                                                      (in < 16 ? ", scale/sum slot " + std::to_string(in / 4)
                                                               : ", quant " + std::to_string(in - 16)) +
                                                      ")"));
                    }
                    dst.resize(n);
                    ggml_backend_tensor_get(tdst, dst.data(), 0, n * sizeof(float));
                    bool same = std::memcmp(dst.data(), dst0.data(), n * sizeof(float)) == 0;
                    if (kind == 0) {
                        sum.resize(n);
                        ggml_backend_tensor_get(tsum, sum.data(), 0, n * sizeof(float));
                        same = same && std::memcmp(sum.data(), sum0.data(), n * sizeof(float)) == 0;
                    }
                    if (!same) {
                        ++f32_fail;
                        report(at + ": the F32 output differs from the kernel without q8_1 rows");
                    }
                }
            }
        }
    }
    std::printf("kernel cases=%d q8_fail=%d f32_fail=%d api_fail=%d\n", cases, q8_fail, f32_fail, api_fail);
    std::fflush(stdout);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(gpu);
    return 0;
}

// ── graph arms ──────────────────────────────────────────────────────────────

// Byte offsets of the fp16 scale fields in one block (ggml-common.h).
bool block_scales(ggml_type type, size_t & block_bytes, std::vector<size_t> & half_offsets) {
    switch (type) {
        case GGML_TYPE_Q2_K:   block_bytes = 84;  half_offsets = {80, 82}; break;
        case GGML_TYPE_IQ4_XS: block_bytes = 136; half_offsets = {0};      break;
        case GGML_TYPE_Q4_K:   block_bytes = 144; half_offsets = {0, 2};   break;
        case GGML_TYPE_Q5_K:   block_bytes = 176; half_offsets = {0, 2};   break;
        case GGML_TYPE_Q6_K:   block_bytes = 210; half_offsets = {208};    break;
        case GGML_TYPE_Q8_0:   block_bytes = 34;  half_offsets = {0};      break;
        default: return false;
    }
    return block_bytes == ggml_type_size(type);
}

// Random blocks with every fp16 scale finite: magnitude 2^-12..2^-8, either sign.
bool make_weights(ggml_type type, int64_t k, int64_t m, uint64_t seed, std::vector<uint8_t> & w) {
    size_t block_bytes = 0;
    std::vector<size_t> half_offsets;
    if (!block_scales(type, block_bytes, half_offsets)) {
        return false;
    }
    w.resize(ggml_row_size(type, k) * (size_t) m);
    Rng rng{seed * 0x9E3779B97F4A7C15ull + 1};
    for (size_t i = 0; i < w.size(); ++i) {
        w[i] = (uint8_t) rng.next();
    }
    for (size_t b = 0; b < w.size(); b += block_bytes) {
        for (size_t off : half_offsets) {
            const uint64_t r = rng.next();
            const uint16_t sign = (uint16_t) ((r & 1) << 15);
            const uint16_t exp  = (uint16_t) (3 + (r >> 1) % 5);
            const uint16_t mant = (uint16_t) ((r >> 8) & 0x3ff);
            const uint16_t h = sign | (uint16_t) (exp << 10) | mant;
            std::memcpy(w.data() + b + off, &h, 2);
        }
    }
    return ggml_validate_row_data(type, w.data(), w.size());
}

// Weights in their own buffer, filled after allocation.
struct Weights {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> fill;
    bool ok = true;

    explicit Weights(int n) {
        ggml_init_params ip{ggml_tensor_overhead() * (size_t) n, nullptr, true};
        ctx = ggml_init(ip);
    }
    ~Weights() {
        if (buf) ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_tensor * quant(ggml_type type, int64_t k, int64_t m, uint64_t seed) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, type, k, m);
        std::vector<uint8_t> bytes;
        ok = ok && make_weights(type, k, m, seed, bytes);
        fill.emplace_back(t, std::move(bytes));
        return t;
    }
    ggml_tensor * f32(int64_t n, uint64_t seed) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        std::vector<float> v((size_t) n);
        Rng rng{seed};
        for (float & x : v) x = 1.0f + 0.25f * rng.uniform();
        std::vector<uint8_t> bytes(v.size() * sizeof(float));
        std::memcpy(bytes.data(), v.data(), bytes.size());
        fill.emplace_back(t, std::move(bytes));
        return t;
    }
    bool upload(ggml_backend_t gpu) {
        if (!ok) return false;
        buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
        if (!buf) return false;
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto & [t, bytes] : fill) {
            ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
        }
        return true;
    }
};

struct GraphRun {
    std::map<std::string, uint64_t> hashes;
    size_t mmq = 0;
    size_t hits = 0;
    size_t norm_q8 = 0;
    bool finite = true;
};

std::vector<float> random_input(int64_t n, uint64_t seed) {
    std::vector<float> v((size_t) n);
    Rng rng{seed};
    for (float & x : v) x = rng.uniform();
    return v;
}

constexpr int64_t K = 1024;   // row length (a multiple of 512: no row padding)
// Legacy pool, MMQ widths: memo hits without the norm writes (n1: Q5_K reuses
// Q4_K's DS4 entry and IQ4_XS Q8_0's D4 entry; n2: Q6_K reuses Q8_0's; n4:
// the second Q2_K reuses the first's), and the producers that write q8_1
// rows (n1, n2, n4, n5, n6).
constexpr size_t kBaseHits = 4;
constexpr size_t kProducerWrites = 5;

bool run_norm_graph(ggml_backend_t gpu, int64_t N, GraphRun & run) {
    Weights w(32);
    ggml_tensor * g1 = w.f32(K, 1);
    ggml_tensor * g2 = w.f32(K, 2);
    ggml_tensor * g3 = w.f32(K, 3);
    ggml_tensor * g4 = w.f32(K, 4);
    ggml_tensor * g5 = w.f32(K, 5);
    ggml_tensor * w_a  = w.quant(GGML_TYPE_Q4_K,   K, K,   11);   // DS4; its output feeds n2, n3, s2
    ggml_tensor * w_b  = w.quant(GGML_TYPE_Q5_K,   K, 256, 12);   // DS4
    ggml_tensor * w_c  = w.quant(GGML_TYPE_Q8_0,   K, 256, 13);   // D4
    ggml_tensor * w_d  = w.quant(GGML_TYPE_IQ4_XS, K, 256, 14);   // D4
    ggml_tensor * w_e  = w.quant(GGML_TYPE_Q8_0,   K, 256, 15);   // D4
    ggml_tensor * w_f  = w.quant(GGML_TYPE_Q6_K,   K, 256, 16);   // D4
    ggml_tensor * w_g  = w.quant(GGML_TYPE_IQ4_XS, K, 256, 17);   // D4
    ggml_tensor * w_h1 = w.quant(GGML_TYPE_Q2_K,   K, 256, 18);   // D2S6
    ggml_tensor * w_h2 = w.quant(GGML_TYPE_Q2_K,   K, 256, 19);   // D2S6
    ggml_tensor * w_i  = w.quant(GGML_TYPE_Q4_K,   K, 512, 20);   // DS4
    ggml_tensor * w_j  = w.quant(GGML_TYPE_Q8_0,   256, 256, 21); // D4, reads the SWIGLU
    if (!w.upload(gpu)) {
        return false;
    }

    ggml_init_params ip{ggml_tensor_overhead() * 96 + ggml_graph_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    const float eps = 1e-6f;
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * r = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_set_input(x);
    ggml_set_input(r);

    std::vector<std::pair<std::string, ggml_tensor *>> outs;
    auto out = [&](const char * name, ggml_tensor * t) {
        ggml_set_output(t);
        outs.emplace_back(name, t);
        return t;
    };
    auto mm = [&](const char * name, ggml_tensor * wt, ggml_tensor * in) {
        return out(name, ggml_mul_mat(ctx, wt, in));
    };
    ggml_tensor * s  = out("s", ggml_add(ctx, x, r));
    ggml_tensor * n1 = out("n1", ggml_mul(ctx, ggml_rms_norm(ctx, s, eps), g1));
    ggml_tensor * m_a = mm("m_a", w_a, n1);
    ggml_tensor * m_b = mm("m_b", w_b, n1);
    ggml_tensor * m_c = mm("m_c", w_c, n1);
    ggml_tensor * m_d = mm("m_d", w_d, n1);
    ggml_tensor * n2 = out("n2", ggml_mul(ctx, ggml_rms_norm(ctx, m_a, eps), g2));
    ggml_tensor * m_e = mm("m_e", w_e, n2);
    ggml_tensor * m_f = mm("m_f", w_f, n2);
    ggml_tensor * n3 = out("n3", ggml_scale_inplace(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, m_a, eps), g3), 0.5f));
    ggml_tensor * m_g = mm("m_g", w_g, n3);
    ggml_tensor * s2 = out("s2", ggml_add(ctx, s, m_a));
    ggml_tensor * n4 = out("n4", ggml_mul(ctx, ggml_rms_norm(ctx, s2, eps), g4));
    ggml_tensor * m_h1 = mm("m_h1", w_h1, n4);
    ggml_tensor * m_h2 = mm("m_h2", w_h2, n4);
    ggml_tensor * s3 = out("s3", ggml_add(ctx, s2, n2));
    ggml_tensor * n5 = out("n5", ggml_mul(ctx, ggml_rms_norm(ctx, s3, eps), g5));
    ggml_tensor * m_i = mm("m_i", w_i, n5);
    ggml_tensor * n6 = out("n6", ggml_swiglu_split(ctx, m_b, m_c));
    ggml_tensor * m_j = mm("m_j", w_j, n6);

    // Expanded reader by reader, so each norm's nodes sit together
    // (ADD, RMS_NORM, MUL or RMS_NORM, MUL) right before its readers.
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (ggml_tensor * t : {m_a, m_b, m_c, m_d, m_e, m_f, m_g, m_h1, m_h2, m_i, m_j}) {
        ggml_build_forward_expand(gf, t);
    }

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
    bool ok = ggml_gallocr_alloc_graph(ga, gf);
    if (ok) {
        const std::vector<float> xv = random_input(K * N, 101 + (uint64_t) N);
        const std::vector<float> rv = random_input(K * N, 202 + (uint64_t) N);
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
        ggml_backend_tensor_set(r, rv.data(), 0, rv.size() * sizeof(float));
        const size_t mmq0 = ggml_backend_cuda_get_mmq_launch_count();
        const size_t hit0 = ggml_backend_cuda_get_glue_q8_memo_hit_count();
        const size_t nq0  = ggml_backend_cuda_get_glue_q8_producer_count();
        ok = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(gpu);
        run.mmq = ggml_backend_cuda_get_mmq_launch_count() - mmq0;
        run.hits = ggml_backend_cuda_get_glue_q8_memo_hit_count() - hit0;
        run.norm_q8 = ggml_backend_cuda_get_glue_q8_producer_count() - nq0;
        for (const auto & [name, t] : outs) {
            std::vector<float> host((size_t) ggml_nelements(t));
            ggml_backend_tensor_get(t, host.data(), 0, host.size() * sizeof(float));
            for (float v : host) run.finite = run.finite && std::isfinite(v);
            run.hashes[name] = fnv1a(host.data(), host.size() * sizeof(float));
        }
    }
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return ok;
}

const int64_t kWidths[] = {1, 5, 16};

int run_graph_arm() {
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (gpu == nullptr) {
        std::printf("skip no GPU\n");
        return 77;
    }
    ggml_backend_cuda_set_graphs_disabled_override(true);
    if (const char * memo = std::getenv("LUCE_TEST_NORM_Q8_MEMO"); memo != nullptr) {
        ggml_backend_cuda_set_q8_memo_override(std::strcmp(memo, "0") != 0);
    }
    if (const char * prod = std::getenv("LUCE_TEST_NORM_Q8_PRODUCERS"); prod != nullptr) {
        ggml_backend_cuda_set_glue_q8_producers_override(std::strcmp(prod, "0") != 0);
    }
    int rc = 0;
    for (const int64_t N : kWidths) {
        GraphRun run;
        const std::string g = "n" + std::to_string(N);
        if (!run_norm_graph(gpu, N, run)) {
            std::printf("error graph %s: build or compute failed\n", g.c_str());
            rc = 1;
        }
        std::printf("graph %s mmq=%zu hits=%zu normq8=%zu finite=%d\n", g.c_str(), run.mmq, run.hits, run.norm_q8,
                    run.finite ? 1 : 0);
        for (const auto & [name, h] : run.hashes) {
            std::printf("out %s/%s hash=%016llx\n", g.c_str(), name.c_str(), (unsigned long long) h);
        }
    }
    // The CUDA pools are created on first use: ask after the graphs ran, or the answer is always 0.
    std::printf("pool legacy=%d\n", ggml_backend_cuda_has_legacy_pool(gpu) ? 1 : 0);
    std::fflush(stdout);
    ggml_backend_free(gpu);
    return rc;
}

int child_main() {
    const char * arm = std::getenv("LUCE_TEST_NORM_Q8");
    if (arm != nullptr && std::strcmp(arm, "kernel") == 0) {
        return run_kernel_arm();
    }
    return run_graph_arm();
}

// ── parent ──────────────────────────────────────────────────────────────────

struct Arm {
    const char * name;
    std::vector<std::string> env;
};

struct GraphStats {
    size_t mmq = 0;
    size_t hits = 0;
    size_t norm_q8 = 0;
    bool finite = false;
};

struct ArmRun {
    int rc = -1;
    bool legacy_pool = false;
    int kernel_cases = -1, q8_fail = -1, f32_fail = -1, api_fail = -1;
    std::map<std::string, uint64_t> hashes;
    std::map<std::string, GraphStats> graphs;
    std::vector<std::string> errors;
};

bool run_child(const Arm & arm, ArmRun & run) {
    std::vector<std::string> env_store;
    for (char ** e = environ; e != nullptr && *e != nullptr; ++e) {
        if (std::strncmp(*e, "LUCE_TEST_NORM_Q8", 17) != 0) {
            env_store.emplace_back(*e);
        }
    }
    for (const std::string & e : arm.env) {
        env_store.push_back(e);
    }
    std::vector<char *> envp;
    for (std::string & e : env_store) {
        envp.push_back(e.data());
    }
    envp.push_back(nullptr);
    char self[] = "/proc/self/exe";
    char flag[] = "--child";
    char * argv[] = {self, flag, nullptr};

    int fds[2];
    if (pipe(fds) != 0) {
        return false;
    }
    // stdout and stderr both.
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&fa, fds[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&fa, fds[0]);
    pid_t pid = 0;
    const int err = posix_spawn(&pid, self, &fa, nullptr, argv, envp.data());
    posix_spawn_file_actions_destroy(&fa);
    close(fds[1]);
    if (err != 0) {
        close(fds[0]);
        return false;
    }
    std::string text;
    char buf[4096];
    ssize_t got;
    while ((got = read(fds[0], buf, sizeof buf)) > 0) {
        text.append(buf, (size_t) got);
    }
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    run.rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;

    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        char name[128];
        unsigned long long hash = 0;
        GraphStats gs;
        int finite = 0;
        int legacy = 0;
        if (std::sscanf(line.c_str(), "pool legacy=%d", &legacy) == 1) {
            run.legacy_pool = legacy != 0;
        } else if (std::sscanf(line.c_str(), "kernel cases=%d q8_fail=%d f32_fail=%d api_fail=%d", &run.kernel_cases,
                               &run.q8_fail, &run.f32_fail, &run.api_fail) == 4) {
        } else if (std::sscanf(line.c_str(), "graph %127s mmq=%zu hits=%zu normq8=%zu finite=%d", name, &gs.mmq,
                               &gs.hits, &gs.norm_q8, &finite) == 5) {
            gs.finite = finite != 0;
            run.graphs[name] = gs;
        } else if (std::sscanf(line.c_str(), "out %127s hash=%llx", name, &hash) == 2) {
            run.hashes[name] = hash;
        } else if (line.rfind("error ", 0) == 0) {
            run.errors.push_back(line);
        }
    }
    return true;
}

int parent_main() {
    if (ggml_backend_cuda_get_device_count() < 1) {
        std::printf("[glue-norm-q8] SKIP: no GPU\n");
        return 77;
    }
    const std::vector<Arm> arms = {
        {"kernel",    {"LUCE_TEST_NORM_Q8=kernel"}},
        {"off",       {"LUCE_TEST_NORM_Q8=graph", "LUCE_TEST_NORM_Q8_PRODUCERS=0"}},
        {"on",        {"LUCE_TEST_NORM_Q8=graph"}},
        {"on_nomemo", {"LUCE_TEST_NORM_Q8=graph", "LUCE_TEST_NORM_Q8_MEMO=0"}},
    };
    std::vector<ArmRun> runs(arms.size());
    for (size_t i = 0; i < arms.size(); ++i) {
        if (!run_child(arms[i], runs[i])) {
            std::printf("[glue-norm-q8] FAIL: cannot spawn the %s arm\n", arms[i].name);
            return 1;
        }
        if (runs[i].rc == 77) {
            std::printf("[glue-norm-q8] SKIP: the %s arm found no GPU\n", arms[i].name);
            return 77;
        }
    }

    bool ok = true;
    auto fail = [&](const std::string & what) {
        std::printf("[glue-norm-q8] FAIL %s\n", what.c_str());
        ok = false;
    };
    for (size_t i = 0; i < arms.size(); ++i) {
        for (const std::string & e : runs[i].errors) fail(std::string(arms[i].name) + ": " + e);
        if (runs[i].rc != 0) fail(std::string(arms[i].name) + ": child exit " + std::to_string(runs[i].rc));
    }

    // The kernels: every case byte-identical to quantize_mmq_q8_1, F32 untouched.
    const ArmRun & kern = runs[0];
    if (kern.kernel_cases <= 0 || kern.q8_fail != 0 || kern.f32_fail != 0 || kern.api_fail != 0) {
        fail("kernel: cases=" + std::to_string(kern.kernel_cases) + " q8_fail=" + std::to_string(kern.q8_fail) +
             " f32_fail=" + std::to_string(kern.f32_fail) + " api_fail=" + std::to_string(kern.api_fail));
    }

    // The graphs: every arm bit-identical to off, with the same MMQ calls.
    const ArmRun & base = runs[1];
    if (base.hashes.empty() || base.graphs.size() != sizeof(kWidths) / sizeof(kWidths[0])) {
        fail("off: " + std::to_string(base.graphs.size()) + " graphs, " + std::to_string(base.hashes.size()) +
             " outputs");
    }
    for (size_t i = 1; i < arms.size(); ++i) {
        const ArmRun & r = runs[i];
        const std::string arm = arms[i].name;
        const bool producers_on = arm != "off";
        const bool memo_on = arm != "on_nomemo";
        if (r.hashes.size() != base.hashes.size()) {
            fail(arm + ": " + std::to_string(r.hashes.size()) + " outputs, off has " +
                 std::to_string(base.hashes.size()));
        }
        for (const auto & [name, h] : base.hashes) {
            const auto it = r.hashes.find(name);
            if (it == r.hashes.end()) {
                fail(arm + " " + name + ": missing");
            } else if (it->second != h) {
                fail(arm + " " + name + ": hash differs from off");
            }
        }
        for (const int64_t N : kWidths) {
            const std::string g = "n" + std::to_string(N);
            const auto it = r.graphs.find(g);
            const auto bt = base.graphs.find(g);
            if (it == r.graphs.end() || bt == base.graphs.end()) {
                fail(arm + " graph " + g + ": missing");
                continue;
            }
            const GraphStats & gs = it->second;
            if (!gs.finite) fail(arm + " graph " + g + ": non-finite output");
            if (gs.mmq != bt->second.mmq) {
                fail(arm + " graph " + g + ": " + std::to_string(gs.mmq) + " MMQ calls, off " +
                     std::to_string(bt->second.mmq));
            }
            // MMQ readers at 5 and 16 columns; MMVQ at 1. The memo (and so a
            // norm write) only on a legacy pool.
            const bool mmq_width = N >= 4 && r.legacy_pool;
            const size_t want_base = memo_on && mmq_width ? kBaseHits : 0;
            const size_t want_writes = producers_on && memo_on && mmq_width ? kProducerWrites : 0;
            if (gs.norm_q8 != want_writes) {
                fail(arm + " graph " + g + ": " + std::to_string(gs.norm_q8) + " producer q8_1 writes, want " +
                     std::to_string(want_writes));
            }
            // Each producer write saves its first reader's quantize: one more hit.
            if (gs.hits != want_base + want_writes) {
                fail(arm + " graph " + g + ": " + std::to_string(gs.hits) + " memo hits, want " +
                     std::to_string(want_base) + " + " + std::to_string(want_writes));
            }
        }
    }
    if (!ok) {
        return 1;
    }
    std::printf("[glue-norm-q8] PASS: %d kernel cases byte-identical to quantize_mmq_q8_1; %zu graph outputs "
                "bit-identical across %zu arms (legacy pool %d; at 16 columns memo hits %zu -> %zu, producer "
                "q8_1 writes %zu)\n",
                kern.kernel_cases, base.hashes.size(), arms.size() - 1, runs[2].legacy_pool ? 1 : 0,
                base.graphs.count("n16") ? base.graphs.at("n16").hits : 0,
                runs[2].graphs.count("n16") ? runs[2].graphs.at("n16").hits : 0,
                runs[2].graphs.count("n16") ? runs[2].graphs.at("n16").norm_q8 : 0);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--child") == 0) {
        return child_main();
    }
    return parent_main();
}
