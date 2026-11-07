// Glue fusions on the GPU: each must leave every output bit-identical to the
// unfused path. SKIP (77) without a GPU.
//
// The loader lays out the gate|up stack once, before the first graph, so each
// arm runs in a child process (this binary with --child) that picks its
// fusions with the test-only overrides (the parent passes the arm in
// LUCE_TEST_GLUE):
//   base     all three off (the unfused path)
//   q8       only the Q8 memo on
//   strided  only the strided gates on
//   gateup   only the gate|up stack on
//   all      the defaults (all three on)
// Every child computes the same graphs on the same deterministic weights and
// inputs and prints one line per output: the FNV-1a hash of its bytes, plus
// per graph the MMQ launches, the q8_1 memo hits and (for the tiny target's
// verify graph) the CONT and MUL_MAT node counts. The parent requires every
// arm's hashes to equal base's, its MMQ counts too (less the stacked FFN
// GEMMs of the gateup arms), base to make no memo hit, an
// arm with the memo on to hit exactly where glue-q8-memo.h allows it (on a
// legacy-pool device; a VMM pool keeps the memo inert), plus one hit per q8_1
// producer write (an RMS norm or SWIGLU that wrote its first MMQ reader's
// rows: that reader's quantize becomes a hit; the tiny target must have
// some), the strided arms to
// drop exactly the three copies of the tiny target's DeltaNet layer, and the
// gateup arms exactly one GEMM per FFN of its 16-token verify graph.
//
// Graphs:
//   q8      a verify-width (16 columns) layer input read by matmuls of the
//           three q8_1 layouts (D4: IQ4_XS, Q8_0, Q6_K; DS4: Q4_K, Q5_K; D2S6:
//           Q2_K), a reshape of it, a half-width view of it, and a second
//           activation that a copy rewrites between two of its readers (the
//           second must quantize the new bytes, never reuse the old ones).
//   gdn/*   the chain GDN op on strided beta/alpha views of a stacked [2H, T]
//           projection (ggml_gated_delta_net_shared_gates) beside the parent
//           op on contiguous copies, in one graph: both must write the same
//           bytes (result, state, F16 checkpoints), for the grouped-cols
//           (S_v = 128) and scalar (S_v = 64) kernels, verify (persistent
//           checkpoints) and in-place prefill form. These run in every arm.
//   z       the gated norm reading z as the strided row slice of a stacked
//           projection (fused silu * mul) beside the parent's copied z.
//   stack/* one MUL_MAT over two weights stacked by rows plus a SwiGLU on its
//           row views, beside the two separate MUL_MATs and their SwiGLU, at
//           16 columns on IQ4_XS, IQ3_S, Q4_K, Q6_K and Q8_0: the halves of
//           the stacked product and the SwiGLU must equal byte for byte
//           wherever ggml_backend_cuda_row_stack_inexact vouches for it.
//   tiny/*  a two-layer synthetic Qwen3.8 target (one DeltaNet and one
//           attention layer, quantized Q8_0/Q4_K projections, stacked
//           attn_gate|qkv, ssm_beta|alpha and ffn_gate|ffn_up) driven through
//           Qwen35DFlashTarget::verify_batch as the server drives it: an
//           8-token prompt, a 16-token capture verify, rollback_to, a second
//           16-token capture verify. Its logits, committed states and
//           checkpoints are hashed; this is where the graph builder's fusion
//           wiring runs.
//   tree    the same target's single-request tree verify graph
//           (build_target_step_tree, 16 nodes), built but not computed: not
//           the dense chain, so STRIDED_GDN must keep its beta, alpha and z
//           copies (every arm has base's CONT count there).
//
//   test_glue_fusions_gpu            all arms, compare
//   test_glue_fusions_gpu --child    one arm, the environment decides

#include "internal.h"
#include "qwen35_dflash_target.h"
#include "graph_builders.h"
#include "step_graph.h"
#include "attn_masks.h"
#include "glue_fusions.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

extern "C" char ** environ;

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

// Byte offsets of the fp16 scale fields in one block (ggml-common.h).
bool block_scales(ggml_type type, size_t & block_bytes, std::vector<size_t> & half_offsets) {
    switch (type) {
        case GGML_TYPE_Q2_K:   block_bytes = 84;  half_offsets = {80, 82}; break;
        case GGML_TYPE_IQ3_S:  block_bytes = 110; half_offsets = {0};      break;
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
        if (!make_weights(type, k, m, seed, bytes)) {
            return nullptr;
        }
        fill.emplace_back(t, std::move(bytes));
        return t;
    }
    ggml_tensor * f32(int64_t n, uint64_t seed) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        std::vector<float> v((size_t) n);
        Rng rng{seed};
        for (float & x : v) x = 0.5f + 0.5f * rng.uniform();
        std::vector<uint8_t> bytes(v.size() * sizeof(float));
        std::memcpy(bytes.data(), v.data(), bytes.size());
        fill.emplace_back(t, std::move(bytes));
        return t;
    }
    bool upload(ggml_backend_t gpu) {
        buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
        if (!buf) return false;
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto & [t, bytes] : fill) {
            ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
        }
        return true;
    }
};

struct Output {
    std::string name;
    ggml_tensor * t;
};

struct GraphRun {
    std::map<std::string, uint64_t> hashes;
    std::map<std::string, bool> checks;   // in-process pair comparisons
    size_t mmq = 0;
    size_t hits = 0;
    size_t writes = 0;   // q8_1 producer writes (ggml_backend_cuda_get_glue_q8_producer_count)
    bool finite = true;
    int n_cont = 0;      // tiny target: CONT nodes of the first 16-token verify graph
    int n_mul_mat = 0;   // tiny target: MUL_MAT nodes of that graph
};

// Compute `outs` (already expanded into gf) with gallocr, as the server does.
bool compute(ggml_backend_t gpu, ggml_cgraph * gf, const std::vector<std::pair<ggml_tensor *, std::vector<float>>> & inputs,
             const std::vector<Output> & outs, GraphRun & run) {
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
    if (!ggml_gallocr_alloc_graph(ga, gf)) {
        ggml_gallocr_free(ga);
        return false;
    }
    for (const auto & [t, v] : inputs) {
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
    }
    const size_t mmq0 = ggml_backend_cuda_get_mmq_launch_count();
    const size_t hit0 = ggml_backend_cuda_get_glue_q8_memo_hit_count();
    const size_t wr0 = ggml_backend_cuda_get_glue_q8_producer_count();
    const bool ok = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(gpu);
    run.mmq += ggml_backend_cuda_get_mmq_launch_count() - mmq0;
    run.hits += ggml_backend_cuda_get_glue_q8_memo_hit_count() - hit0;
    run.writes += ggml_backend_cuda_get_glue_q8_producer_count() - wr0;
    for (const Output & o : outs) {
        std::vector<float> host((size_t) ggml_nelements(o.t));
        ggml_backend_tensor_get(o.t, host.data(), 0, host.size() * sizeof(float));
        for (float v : host) run.finite = run.finite && std::isfinite(v);
        run.hashes[o.name] = fnv1a(host.data(), host.size() * sizeof(float));
    }
    ggml_gallocr_free(ga);
    return ok;
}

std::vector<float> random_input(int64_t n, uint64_t seed) {
    std::vector<float> v((size_t) n);
    Rng rng{seed};
    for (float & x : v) x = rng.uniform();
    return v;
}

constexpr int64_t K = 1024;  // activation row length (a multiple of 512: no row padding)
constexpr int64_t N = 16;    // verify width

// Readers of the q8 graph: their name, and whether the memo serves them.
// Hits: m_q8_0 and m_q6_k reuse m_iq4_xs's D4 entry, m_q5_k reuses m_q4_k's
// DS4 entry, and m_reshape (a reshape of the same input) reuses the D4 entry.
constexpr size_t kQ8MemoHits = 4;

// Tiny target, per 8- or 16-token forward: attn_gate|qkv shares its input
// with ssm_beta|alpha (D4, D4), attn_q with attn_k (D4, D4; attn_v is DS4
// and alone), and in each layer ffn_gate with ffn_up (DS4, DS4). The prompt
// and the two capture verifies are three forwards.
constexpr size_t kTinyMemoHits = 3 * 4;
// With the gate|up stack the two 16-token verifies run one FFN
// GEMM per layer, so their gate/up pairs no longer share a quantize; the
// 8-token prompt is at MMVQ-or-less width for the stack and keeps two.
constexpr size_t kTinyMemoHitsStacked = 4 + 2 * 2;

bool run_q8_memo_graph(ggml_backend_t gpu, GraphRun & run) {
    Weights w(16);
    ggml_tensor * norm_w = w.f32(K, 7);
    ggml_tensor * w_iq4  = w.quant(GGML_TYPE_IQ4_XS, K, 512, 11);
    ggml_tensor * w_q8   = w.quant(GGML_TYPE_Q8_0,   K, 128, 12);
    ggml_tensor * w_q4k  = w.quant(GGML_TYPE_Q4_K,   K, 512, 13);
    ggml_tensor * w_q6k  = w.quant(GGML_TYPE_Q6_K,   K, 256, 14);
    ggml_tensor * w_q5k  = w.quant(GGML_TYPE_Q5_K,   K, 256, 15);
    ggml_tensor * w_q2k  = w.quant(GGML_TYPE_Q2_K,   K, 256, 16);
    ggml_tensor * w_half = w.quant(GGML_TYPE_IQ4_XS, K, 256, 17);
    ggml_tensor * w_resh = w.quant(GGML_TYPE_IQ4_XS, K, 256, 18);
    ggml_tensor * w_y1   = w.quant(GGML_TYPE_IQ4_XS, K, 256, 19);
    ggml_tensor * w_y2   = w.quant(GGML_TYPE_IQ4_XS, K, 256, 20);
    if (!w_iq4 || !w_q8 || !w_q4k || !w_q6k || !w_q5k || !w_q2k || !w_half || !w_resh || !w_y1 || !w_y2 ||
        !w.upload(gpu)) {
        return false;
    }

    ggml_init_params ip{ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_set_input(x);
    ggml_tensor * n = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), norm_w);

    std::vector<Output> outs;
    auto mm = [&](const char * name, ggml_tensor * wt, ggml_tensor * in) {
        ggml_tensor * o = ggml_mul_mat(ctx, wt, in);
        ggml_set_output(o);
        outs.push_back({name, o});
        return o;
    };
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, mm("m_iq4_xs", w_iq4, n));
    ggml_build_forward_expand(gf, mm("m_q8_0",   w_q8,  n));
    ggml_build_forward_expand(gf, mm("m_q4_k",   w_q4k, n));
    ggml_build_forward_expand(gf, mm("m_q6_k",   w_q6k, n));
    ggml_build_forward_expand(gf, mm("m_q5_k",   w_q5k, n));
    ggml_build_forward_expand(gf, mm("m_q2_k",   w_q2k, n));
    // Half the columns of the same bytes: another shape, so another key.
    ggml_build_forward_expand(gf, mm("m_half", w_half, ggml_view_2d(ctx, n, K, N / 2, n->nb[1], 0)));
    // y is read, rewritten by a copy, then read again.
    ggml_tensor * y = ggml_scale(ctx, x, 0.25f);
    ggml_build_forward_expand(gf, mm("m_y_before", w_y1, y));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_scale(ctx, x, -1.5f), y));
    ggml_build_forward_expand(gf, mm("m_y_after", w_y2, y));
    // The same bytes and shape through a reshape: the D4 entry serves it.
    ggml_build_forward_expand(gf, mm("m_reshape", w_resh, ggml_reshape_3d(ctx, n, K, N, 1)));

    const bool ok = compute(gpu, gf, {{x, random_input(K * N, 99)}}, outs, run);
    ggml_free(ctx);
    return ok;
}

// ── the strided gates, kernel level ────────────────────────────

void upload_uniform(ggml_tensor * t, uint64_t seed, float lo, float hi) {
    std::vector<float> v((size_t) ggml_nelements(t));
    Rng rng{seed};
    for (float & x : v) x = lo + (hi - lo) * 0.5f * (rng.uniform() + 1.0f);
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
}

std::vector<uint8_t> read_all(const ggml_tensor * t) {
    std::vector<uint8_t> b(ggml_nbytes(t));
    ggml_backend_tensor_get(t, b.data(), 0, b.size());
    return b;
}

// Parent: contiguous copies of the beta/alpha row slices of a stacked [2H, T]
// projection. Fused: the slices as [1, H, T, 1] views with one layout.
// Raw gates, skip-intermediate, and in the verify form the persistent F16
// checkpoints in src[7], as build_delta_net_block builds the chain op.
bool check_gdn_shared_gates(ggml_backend_t gpu, int S, int H, int T, bool inplace, GraphRun & run,
                            const std::string & name) {
    ggml_init_params ip{ggml_tensor_overhead() * 48 + ggml_graph_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, 1);
    ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, 1);
    ggml_tensor * v  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, 1);
    ggml_tensor * ba = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * H, T);
    ggml_tensor * gate_ba = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * H);
    ggml_tensor * st_a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, 1);
    ggml_tensor * st_b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, 1);
    ggml_tensor * ck_a = inplace ? nullptr : ggml_new_tensor_4d(ctx, GGML_TYPE_F16, S, S, H, T);
    ggml_tensor * ck_b = inplace ? nullptr : ggml_new_tensor_4d(ctx, GGML_TYPE_F16, S, S, H, T);

    const size_t e = sizeof(float);
    ggml_tensor * beta_c  = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, ba, H, T, ba->nb[1], 0)), 1, H, T, 1);
    ggml_tensor * alpha_c = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, ba, H, T, ba->nb[1], (size_t) H * e)),
                                            1, H, T, 1);
    ggml_tensor * beta_v  = ggml_view_4d(ctx, ba, 1, H, T, 1, e, ba->nb[1], ba->nb[1] * T, 0);
    ggml_tensor * alpha_v = ggml_view_4d(ctx, ba, 1, H, T, 1, e, ba->nb[1], ba->nb[1] * T, (size_t) H * e);

    ggml_tensor * r_a = inplace ? ggml_gated_delta_net_inplace(ctx, q, k, v, alpha_c, beta_c, st_a)
                                : ggml_gated_delta_net(ctx, q, k, v, alpha_c, beta_c, st_a);
    ggml_tensor * r_b = ggml_gated_delta_net_shared_gates(ctx, q, k, v, alpha_v, beta_v, st_b, inplace);
    // As build_delta_net_block: the capture verify writes its checkpoints to
    // src[7] and keeps the packed result whole; the prefill form compacts it.
    if (!inplace) {
        r_a->src[7] = ck_a;
        r_b->src[7] = ck_b;
    }
    for (ggml_tensor * r : {r_a, r_b}) {
        ggml_gated_delta_net_set_raw_gates(r, gate_ba);
        if (inplace) ggml_gated_delta_net_set_skip_intermediate(r, true);
    }
    ggml_set_output(r_a);
    ggml_set_output(r_b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r_a);
    ggml_build_forward_expand(gf, r_b);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    const float qk = 1.0f / std::sqrt((float) S);
    upload_uniform(q, 101, -qk, qk);
    upload_uniform(k, 102, -qk, qk);
    upload_uniform(v, 103, -1.0f, 1.0f);
    upload_uniform(ba, 104, -2.0f, 2.0f);
    upload_uniform(st_a, 105, -0.5f, 0.5f);
    ggml_backend_tensor_copy(st_a, st_b);
    std::vector<float> gb((size_t) 2 * H);
    for (int h = 0; h < H; ++h) {
        gb[(size_t) h] = 0.1f * (float) h;                 // dt_bias
        gb[(size_t) (H + h)] = -1.0f - 0.05f * (float) h;  // A
    }
    ggml_backend_tensor_set(gate_ba, gb.data(), 0, gb.size() * sizeof(float));
    const bool ok = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(gpu);
    // The written part of the packed result: the attention output, then (not
    // in place) the final state. With src[7] set, the result's embedded
    // checkpoint region stays unwritten on both ops, so it is not compared.
    const size_t written = sizeof(float) * (size_t) S * H * T + (inplace ? 0 : sizeof(float) * (size_t) S * S * H);
    auto prefix = [&](const ggml_tensor * t) {
        std::vector<uint8_t> b(std::min(written, ggml_nbytes(t)));
        ggml_backend_tensor_get(t, b.data(), 0, b.size());
        return b;
    };
    const bool same = ggml_nbytes(r_a) == ggml_nbytes(r_b) && ggml_nbytes(r_a) >= written &&
                      prefix(r_a) == prefix(r_b) && read_all(st_a) == read_all(st_b) &&
                      (inplace || read_all(ck_a) == read_all(ck_b));
    const std::vector<uint8_t> out = prefix(r_b);
    run.hashes[name] = fnv1a(out.data(), out.size());
    run.checks[name] = ok && same;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// Parent: z copied out of the stacked [n_z + C, T] projection, reshaped to
// [hv, H, T]; fused: the strided [n_z, T] slice against the norm viewed as
// [n_z, T]. Both fuse silu + mul (ggml_cuda_op_unary_mul).
bool check_strided_z(ggml_backend_t gpu, int hv, int H, int C, int T, GraphRun & run) {
    const int64_t n_z = (int64_t) hv * H;
    ggml_init_params ip{ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * o    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hv, H, T, 1);
    ggml_tensor * w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hv);
    ggml_tensor * qkvz = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_z + C, T);
    ggml_set_input(o);
    ggml_set_input(w);
    ggml_set_input(qkvz);
    ggml_tensor * z = ggml_view_2d(ctx, qkvz, n_z, T, qkvz->nb[1], 0);
    ggml_tensor * on_a = ggml_mul(ctx, ggml_rms_norm(ctx, o, 1e-6f), w);
    ggml_tensor * out_a = ggml_mul(ctx, on_a, ggml_silu(ctx, ggml_reshape_4d(ctx, ggml_cont(ctx, z), hv, H, T, 1)));
    ggml_tensor * on_b = ggml_reshape_2d(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, o, 1e-6f), w), n_z, T);
    ggml_tensor * out_b = ggml_mul(ctx, on_b, ggml_silu(ctx, z));
    ggml_set_output(out_a);
    ggml_set_output(out_b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_a);
    ggml_build_forward_expand(gf, out_b);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
    if (!ggml_gallocr_alloc_graph(ga, gf)) {
        ggml_gallocr_free(ga);
        ggml_free(ctx);
        return false;
    }
    upload_uniform(o, 201, -1.0f, 1.0f);
    upload_uniform(w, 202, 0.5f, 1.5f);
    upload_uniform(qkvz, 203, -4.0f, 4.0f);
    const bool ok = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(gpu);
    const std::vector<uint8_t> a = read_all(out_a);
    run.hashes["z"] = fnv1a(a.data(), a.size());
    run.checks["z"] = ok && a == read_all(out_b);
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return ok;
}

// ── the gate|up stack, kernel level ───────────────────────────

// Parent: gate = W_gate x, up = W_up x, swiglu(gate, up), W_gate and W_up the
// two row halves of one weight (adjacent, as the loader places them). Fused:
// one MUL_MAT over the whole weight and swiglu on its row views. Compared
// only where the backend says the stacked rows are exact; elsewhere the
// check records a pass and a note line.
bool check_stacked_gemm(ggml_backend_t gpu, ggml_type type, GraphRun & run) {
    constexpr int64_t SK = 1024, SM = 512, SN = 16;
    const std::string name = std::string("stack_") + ggml_type_name(type);
    Weights w(2);
    ggml_tensor * stack = w.quant(type, SK, 2 * SM, 31 + (uint64_t) type);
    if (!stack || !w.upload(gpu)) {
        return false;
    }
    const char * inexact = ggml_backend_cuda_row_stack_inexact(gpu, stack, SM, SN);
    ggml_init_params ip{ggml_tensor_overhead() * 24 + ggml_graph_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, SK, SN);
    ggml_set_input(x);
    ggml_tensor * w_gate = ggml_view_2d(ctx, stack, SK, SM, stack->nb[1], 0);
    ggml_tensor * w_up   = ggml_view_2d(ctx, stack, SK, SM, stack->nb[1], (size_t) SM * stack->nb[1]);
    ggml_tensor * gate = ggml_mul_mat(ctx, w_gate, x);
    ggml_tensor * up   = ggml_mul_mat(ctx, w_up, x);
    ggml_tensor * glu_a = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * all  = ggml_mul_mat(ctx, stack, x);
    const size_t e = sizeof(float);
    ggml_tensor * glu_b = ggml_swiglu_split(ctx, ggml_view_2d(ctx, all, SM, SN, all->nb[1], 0),
                                            ggml_view_2d(ctx, all, SM, SN, all->nb[1], (size_t) SM * e));
    for (ggml_tensor * t : {gate, up, glu_a, all, glu_b}) ggml_set_output(t);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (ggml_tensor * t : {glu_a, glu_b}) ggml_build_forward_expand(gf, t);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
    if (!ggml_gallocr_alloc_graph(ga, gf)) {
        ggml_gallocr_free(ga);
        ggml_free(ctx);
        return false;
    }
    const std::vector<float> xin = random_input(SK * SN, 77);
    ggml_backend_tensor_set(x, xin.data(), 0, xin.size() * sizeof(float));
    const bool ok = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(gpu);
    bool same = ok;
    if (ok && inexact == nullptr) {
        std::vector<float> g((size_t) SM * SN), u((size_t) SM * SN), a((size_t) 2 * SM * SN);
        ggml_backend_tensor_get(gate, g.data(), 0, g.size() * sizeof(float));
        ggml_backend_tensor_get(up, u.data(), 0, u.size() * sizeof(float));
        ggml_backend_tensor_get(all, a.data(), 0, a.size() * sizeof(float));
        for (int64_t j = 0; j < SN && same; ++j) {
            same = std::memcmp(&a[(size_t) (j * 2 * SM)], &g[(size_t) (j * SM)], SM * sizeof(float)) == 0 &&
                   std::memcmp(&a[(size_t) (j * 2 * SM + SM)], &u[(size_t) (j * SM)], SM * sizeof(float)) == 0;
        }
        same = same && read_all(glu_a) == read_all(glu_b);
    } else if (ok) {
        std::printf("note %s: stacked rows not vouched for here (%s); not compared\n", name.c_str(), inexact);
    }
    const std::vector<uint8_t> out = read_all(glu_a);
    run.hashes[name] = fnv1a(out.data(), out.size());
    run.checks[name] = same;
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return ok;
}

// ── A two-layer synthetic Qwen3.8 target ──────────────────────────────────

// Layer 0 Gated DeltaNet, layer 1 full attention, dense SwiGLU FFNs. The
// projections are quantized (Q8_0, and Q4_K for ffn_gate/ffn_up and attn_v)
// so that 8- and 16-token forwards run on MMQ; attn_gate|qkv,
// ssm_beta|alpha and ffn_gate|ffn_up are allocated as stacks whose members
// are row views, as the loader's zero-copy aliases are. F16 host embedding,
// single-sequence cache (prefill-only cache migrated to the verify layout).
struct TinyGlueTarget {
    static constexpr int kHidden   = 256;
    static constexpr int kState    = 128;   // head_k_dim == head_v_dim
    static constexpr int kVHeads   = 2;
    static constexpr int kHeadDim  = 128;   // attention head
    static constexpr int kHeads    = 2;
    static constexpr int kHeadsKv  = 1;
    static constexpr int kFf       = 256;
    static constexpr int kVocab    = 64;
    static constexpr int kMaxCtx   = 256;
    static constexpr int kMaxVerify = 16;

    ggml_backend_t backend = nullptr;   // not owned
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    luce::common::TargetWeights w;
    luce::common::TargetCache cache;
    std::vector<std::pair<ggml_tensor *, ggml_type>> quantized;
    bool ok = false;

    explicit TinyGlueTarget(ggml_backend_t be) : backend(be) {
        using luce::common::TargetLayer;
        w.n_layer = 2;
        w.full_attention_interval = 2;
        w.n_embd = kHidden;
        w.n_embd_head_k = kHeadDim;
        w.n_embd_head_v = kHeadDim;
        w.n_head = kHeads;
        w.n_head_kv = kHeadsKv;
        w.n_ff = kFf;
        w.n_vocab = kVocab;
        w.rope_dimension_count = 64;
        w.rope_sections[0] = 11; w.rope_sections[1] = 11;
        w.rope_sections[2] = 10; w.rope_sections[3] = 0;
        w.n_capture_layers = 0;
        w.is_moe = false;
        w.ssm_d_state = kState;
        w.ssm_n_group = 1;
        w.ssm_dt_rank = kVHeads;
        w.ssm_d_inner = kState * kVHeads;
        w.ssm_d_conv = 4;
        w.eos_id = -1;
        w.eos_chat_id = -1;
        w.layers.resize(2);

        ggml_init_params ip{};
        ip.mem_size = 96 * ggml_tensor_overhead();
        ip.no_alloc = true;
        wctx = ggml_init(ip);
        if (!wctx) return;

        const int conv_ch = w.ssm_d_inner + 2 * w.ssm_n_group * w.ssm_d_state;
        const int n_z = w.ssm_d_inner;
        const int q_dim = kHeadDim * kHeads;
        const int kv_dim = kHeadDim * kHeadsKv;
        auto f32_1d = [&](int a) { return ggml_new_tensor_1d(wctx, GGML_TYPE_F32, a); };
        auto f32_2d = [&](int a, int b) { return ggml_new_tensor_2d(wctx, GGML_TYPE_F32, a, b); };
        auto quant = [&](ggml_type t, int a, int b) {
            ggml_tensor * x = ggml_new_tensor_2d(wctx, t, a, b);
            quantized.emplace_back(x, t);
            return x;
        };
        auto rows = [&](ggml_tensor * stack, int first, int n) {
            return ggml_view_2d(wctx, stack, stack->ne[0], n, stack->nb[1], (size_t) first * stack->nb[1]);
        };
        for (int il = 0; il < 2; ++il) {
            TargetLayer & L = w.layers[(size_t) il];
            L.attn_norm      = f32_1d(kHidden);
            L.attn_post_norm = f32_1d(kHidden);
            ggml_tensor * gu = quant(GGML_TYPE_Q4_K, kHidden, 2 * kFf);
            L.ffn_gate_up = gu;   // read only under the gate|up stack
            L.w_gate = rows(gu, 0, kFf);
            L.w_up   = rows(gu, kFf, kFf);
            L.w_down = quant(GGML_TYPE_Q8_0, kFf, kHidden);
        }
        TargetLayer & G = w.layers[0];
        G.wqkv_z      = quant(GGML_TYPE_Q8_0, kHidden, n_z + conv_ch);
        G.wqkv_gate   = rows(G.wqkv_z, 0, n_z);
        G.wqkv        = rows(G.wqkv_z, n_z, conv_ch);
        G.ssm_ba      = quant(GGML_TYPE_Q8_0, kHidden, 2 * kVHeads);
        G.ssm_beta    = rows(G.ssm_ba, 0, kVHeads);
        G.ssm_alpha   = rows(G.ssm_ba, kVHeads, kVHeads);
        G.ssm_conv1d  = f32_2d(w.ssm_d_conv, conv_ch);
        G.ssm_dt_bias = f32_1d(kVHeads);
        G.ssm_a       = f32_1d(kVHeads);
        G.ssm_norm    = f32_1d(kState);
        G.ssm_out     = quant(GGML_TYPE_Q8_0, w.ssm_d_inner, kHidden);
        G.ssm_gate_ba = f32_1d(2 * kVHeads);
        TargetLayer & A = w.layers[1];
        A.wq     = quant(GGML_TYPE_Q8_0, kHidden, 2 * q_dim);
        A.wk     = quant(GGML_TYPE_Q8_0, kHidden, kv_dim);
        A.wv     = quant(GGML_TYPE_Q4_K, kHidden, kv_dim);
        A.wo     = quant(GGML_TYPE_Q8_0, q_dim, kHidden);
        A.q_norm = f32_1d(kHeadDim);
        A.k_norm = f32_1d(kHeadDim);
        w.out_norm = f32_1d(kHidden);
        w.output   = f32_2d(kHidden, kVocab);

        wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
        if (!wbuf) return;
        ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        std::mt19937 rng(20261105u);
        std::uniform_real_distribution<float> small(-0.08f, 0.08f);
        for (const auto & [t, type] : quantized) {
            std::vector<float> f((size_t) ggml_nelements(t));
            for (float & x : f) x = small(rng);
            std::vector<uint8_t> qb(ggml_nbytes(t));
            ggml_quantize_chunk(type, f.data(), qb.data(), 0, t->ne[1], t->ne[0], nullptr);
            ggml_backend_tensor_set(t, qb.data(), 0, qb.size());
        }
        const ggml_tensor * norms[] = {A.q_norm, A.k_norm, G.ssm_norm, w.out_norm,
                                       w.layers[0].attn_norm, w.layers[0].attn_post_norm,
                                       w.layers[1].attn_norm, w.layers[1].attn_post_norm};
        for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            if (t->type != GGML_TYPE_F32 || t->view_src != nullptr || t == G.ssm_gate_ba) continue;
            bool norm = false;
            for (const ggml_tensor * n : norms) norm = norm || n == t;
            std::vector<float> v((size_t) ggml_nelements(t));
            for (float & x : v) x = norm ? 1.0f + small(rng) : small(rng);
            if (t == G.ssm_a) {
                for (float & x : v) x = -std::exp(0.5f * small(rng) * 10.0f);
            }
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        }
        std::vector<float> gate_ba((size_t) 2 * kVHeads);
        ggml_backend_tensor_get(G.ssm_dt_bias, gate_ba.data(), 0, sizeof(float) * kVHeads);
        ggml_backend_tensor_get(G.ssm_a, gate_ba.data() + kVHeads, 0, sizeof(float) * kVHeads);
        ggml_backend_tensor_set(G.ssm_gate_ba, gate_ba.data(), 0, sizeof(float) * gate_ba.size());

        w.embedder.n_embd = kHidden;
        w.embedder.n_vocab = kVocab;
        w.embedder.tok_embd_type = GGML_TYPE_F16;
        w.embedder.row_bytes = (size_t) kHidden * sizeof(ggml_fp16_t);
        w.embedder.tok_embd_owned.resize(w.embedder.row_bytes * kVocab);
        auto * erows = reinterpret_cast<ggml_fp16_t *>(w.embedder.tok_embd_owned.data());
        for (int i = 0; i < kHidden * kVocab; ++i) erows[i] = ggml_fp32_to_fp16(4.0f * small(rng));
        w.embedder.tok_embd_bytes = w.embedder.tok_embd_owned.data();

        ok = luce::common::create_target_cache_partial(
            w, kMaxCtx, kMaxVerify, backend, cache,
            /*prefill_only=*/true, /*layer_begin=*/0, /*layer_end=*/2,
            /*allocate_target_feat=*/false, /*ctx_alloc=*/0,
            /*f32_ssm_intermediates=*/false, /*paged_attention=*/false,
            /*n_seq_slots=*/1, /*concurrent_tree=*/false,
            GGML_TYPE_F16, GGML_TYPE_F16);
        if (ok) {
            ok = luce::common::migrate_prefill_cache(w, kMaxCtx, kMaxVerify, backend, cache,
                                                     /*enable_specla=*/false);
        }
        if (!ok) std::printf("error tiny target cache: %s\n", luce_last_error());
    }

    ~TinyGlueTarget() {
        luce::common::free_target_cache(cache);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
    }

    TinyGlueTarget(const TinyGlueTarget &) = delete;
    TinyGlueTarget & operator=(const TinyGlueTarget &) = delete;
};

std::vector<int32_t> tiny_tokens(int n, int salt) {
    std::vector<int32_t> out((size_t) n);
    for (int i = 0; i < n; ++i) out[(size_t) i] = (int32_t) ((salt * 37 + i * 13 + 11) % TinyGlueTarget::kVocab);
    return out;
}

void hash_into(GraphRun & run, const std::string & name, const std::vector<uint8_t> & bytes) {
    run.hashes[name] = fnv1a(bytes.data(), bytes.size());
}

// The server's call sequence on the tiny target. Returns false on an API
// failure; hashes, counters and the verify graph's node counts go to `run`.
bool run_tiny_target(ggml_backend_t gpu, GraphRun & run) {
    TinyGlueTarget t(gpu);
    if (!t.ok) return false;
    luce::common::StepGraph sg;
    bool ok = true;
    {
        luce::common::Qwen35DFlashTarget target(t.w, t.cache, gpu, sg, luce::common::KQ_MASK_PAD, /*fa_window=*/0);
        target.set_fast_rollback(true);
        const size_t mmq0 = ggml_backend_cuda_get_mmq_launch_count();
        const size_t hit0 = ggml_backend_cuda_get_glue_q8_memo_hit_count();
        const size_t wr0 = ggml_backend_cuda_get_glue_q8_producer_count();
        int last = -1;
        std::vector<int32_t> argmax;
        std::vector<float> logits;
        // A failed or short read fails the arm: hashing nothing in every arm
        // would compare equal without having checked a single logit.
        auto logits_bytes = [&](int n) {
            std::vector<uint8_t> b;
            if (!target.read_verify_logits(n, logits) ||
                logits.size() != (size_t) n * TinyGlueTarget::kVocab) {
                std::fprintf(stderr, "tiny target: reading %d rows of verify logits failed\n", n);
                ok = false;
                return b;
            }
            b.resize(logits.size() * sizeof(float));
            std::memcpy(b.data(), logits.data(), b.size());
            for (float x : logits) run.finite = run.finite && std::isfinite(x);
            return b;
        };
        ok = ok && target.verify_batch(tiny_tokens(8, 1), 0, last, &argmax, /*capture=*/false);
        hash_into(run, "tiny/prompt_logits", logits_bytes(8));
        ok = ok && target.verify_batch(tiny_tokens(16, 2), 8, last, &argmax, /*capture=*/true);
        hash_into(run, "tiny/verify1_logits", logits_bytes(16));
        if (t.cache.ssm_intermediate.size() > 0 && t.cache.ssm_intermediate[0]) {
            hash_into(run, "tiny/verify1_checkpoints", read_all(t.cache.ssm_intermediate[0]));
        }
        for (int i = 0; ok && i < ggml_graph_n_nodes(sg.gf); ++i) {
            const ggml_tensor * n = ggml_graph_node(sg.gf, i);
            if (n->op == GGML_OP_CONT) ++run.n_cont;
            if (n->op == GGML_OP_MUL_MAT) ++run.n_mul_mat;
        }
        ok = ok && target.rollback_to(8, 7);
        hash_into(run, "tiny/commit_ssm", read_all(t.cache.ssm_state[0]));
        hash_into(run, "tiny/commit_conv", read_all(t.cache.conv_state[0]));
        ok = ok && target.verify_batch(tiny_tokens(16, 3), 15, last, &argmax, /*capture=*/true);
        hash_into(run, "tiny/verify2_logits", logits_bytes(16));
        ggml_backend_synchronize(gpu);
        run.mmq += ggml_backend_cuda_get_mmq_launch_count() - mmq0;
        run.hits += ggml_backend_cuda_get_glue_q8_memo_hit_count() - hit0;
        run.writes += ggml_backend_cuda_get_glue_q8_producer_count() - wr0;
    }
    luce::common::step_graph_destroy(sg);
    return ok;
}

// The tiny target's single-request tree verify graph, as verify_tree builds
// it. Only its node counts matter (no inputs, no compute).
bool count_tiny_tree_graph(ggml_backend_t gpu, GraphRun & run) {
    TinyGlueTarget t(gpu);
    if (!t.ok) return false;
    luce::common::StepGraph sg;
    const bool ok = luce::common::build_target_step_tree(sg, t.w, t.cache, gpu, /*kv_start=*/8,
                                                         TinyGlueTarget::kMaxVerify);
    for (int i = 0; ok && i < ggml_graph_n_nodes(sg.gf); ++i) {
        const ggml_tensor * n = ggml_graph_node(sg.gf, i);
        if (n->op == GGML_OP_CONT) ++run.n_cont;
        if (n->op == GGML_OP_MUL_MAT) ++run.n_mul_mat;
    }
    luce::common::step_graph_destroy(sg);
    return ok;
}

void print_graph(const char * name, const GraphRun & r) {
    std::printf("graph %s mmq=%zu hits=%zu writes=%zu finite=%d cont=%d mul_mat=%d\n", name, r.mmq, r.hits,
                r.writes, r.finite ? 1 : 0, r.n_cont, r.n_mul_mat);
    for (const auto & [out, h] : r.hashes) {
        std::printf("out %s/%s hash=%016llx\n", name, out.c_str(), (unsigned long long) h);
    }
    for (const auto & [check, pass] : r.checks) {
        std::printf("check %s/%s ok=%d\n", name, check.c_str(), pass ? 1 : 0);
    }
}

int child_main() {
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (gpu == nullptr) {
        std::printf("skip no GPU\n");
        return 77;
    }
    ggml_backend_cuda_set_graphs_disabled_override(true);
    // The arm, as the parent passed it: "<q8 memo><strided gates><gate|up stack>".
    if (const char * glue = std::getenv("LUCE_TEST_GLUE"); glue != nullptr && std::strlen(glue) == 3) {
        ggml_backend_cuda_set_q8_memo_override(glue[0] == '1');
        luce::qwen35::glue::Fusions f;
        f.strided_gates = glue[1] == '1';
        f.gate_up_stack = glue[2] == '1';
        luce::qwen35::glue::set_fusions_for_test(f);
    }
    int rc = 0;
    GraphRun q8;
    if (!run_q8_memo_graph(gpu, q8)) {
        std::printf("error q8 graph: compute failed\n");
        rc = 1;
    }
    std::printf("pool legacy=%d\n", ggml_backend_cuda_has_legacy_pool(gpu) ? 1 : 0);
    {
        // Whether this device vouches for the tiny target's Q4_K 256|256
        // ffn_gate|ffn_up stack at 16 columns (the gateup arms' MUL_MAT drop).
        ggml_init_params ip{ggml_tensor_overhead() * 2, nullptr, true};
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * gu = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, TinyGlueTarget::kHidden, 2 * TinyGlueTarget::kFf);
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(ctx, gpu);
        const char * why = b ? ggml_backend_cuda_row_stack_inexact(gpu, gu, TinyGlueTarget::kFf, 16) : "alloc";
        std::printf("stack exact=%d\n", why == nullptr ? 1 : 0);
        if (why) std::printf("note tiny stack: %s\n", why);
        if (b) ggml_backend_buffer_free(b);
        ggml_free(ctx);
    }
    print_graph("q8", q8);

    GraphRun gdn;
    struct GdnCase { int S; bool inplace; const char * name; };
    for (const GdnCase & c : {GdnCase{128, false, "s128_verify"}, GdnCase{128, true, "s128_prefill"},
                              GdnCase{64, false, "s64_verify"}, GdnCase{64, true, "s64_prefill"}}) {
        if (!check_gdn_shared_gates(gpu, c.S, 4, 16, c.inplace, gdn, c.name)) {
            std::printf("error gdn %s: compute failed\n", c.name);
            rc = 1;
        }
    }
    print_graph("gdn", gdn);

    GraphRun z;
    if (!check_strided_z(gpu, 128, 4, 1024, 16, z)) {
        std::printf("error z graph: compute failed\n");
        rc = 1;
    }
    print_graph("z", z);

    GraphRun stack;
    for (ggml_type t : {GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_S, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) {
        if (!check_stacked_gemm(gpu, t, stack)) {
            std::printf("error stack %s: compute failed\n", ggml_type_name(t));
            rc = 1;
        }
    }
    print_graph("stack", stack);

    GraphRun tiny;
    if (!run_tiny_target(gpu, tiny)) {
        std::printf("error tiny target: a verify, rollback or cache call failed\n");
        rc = 1;
    }
    print_graph("tiny", tiny);

    GraphRun tree;
    if (!count_tiny_tree_graph(gpu, tree)) {
        std::printf("error tree graph: build_target_step_tree failed\n");
        rc = 1;
    }
    print_graph("tree", tree);
    std::fflush(stdout);
    ggml_backend_free(gpu);
    return rc;
}

struct Arm {
    const char * name;
    std::vector<std::string> env;  // LUCE_TEST_GLUE value of this arm
};

struct GraphStats {
    size_t mmq = 0;
    size_t hits = 0;
    size_t writes = 0;
    bool finite = false;
    int n_cont = 0;
    int n_mul_mat = 0;
};

struct ArmRun {
    int rc = -1;
    bool legacy_pool = false;
    bool stack_exact = false;
    std::map<std::string, uint64_t> hashes;
    std::map<std::string, GraphStats> graphs;
    std::map<std::string, bool> checks;
    std::vector<std::string> errors;
};

bool run_child(const Arm & arm, ArmRun & run) {
    std::vector<std::string> env_store;
    for (char ** e = environ; e != nullptr && *e != nullptr; ++e) {
        if (std::strncmp(*e, "LUCE_TEST_GLUE=", 15) != 0) {
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
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[1], STDOUT_FILENO);
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
        int exact = 0;
        if (std::sscanf(line.c_str(), "pool legacy=%d", &legacy) == 1) {
            run.legacy_pool = legacy != 0;
        } else if (std::sscanf(line.c_str(), "stack exact=%d", &exact) == 1) {
            run.stack_exact = exact != 0;
        } else if (std::sscanf(line.c_str(), "graph %127s mmq=%zu hits=%zu writes=%zu finite=%d cont=%d mul_mat=%d",
                               name, &gs.mmq, &gs.hits, &gs.writes, &finite, &gs.n_cont, &gs.n_mul_mat) == 7) {
            gs.finite = finite != 0;
            run.graphs[name] = gs;
        } else if (std::sscanf(line.c_str(), "check %127s ok=%d", name, &finite) == 2) {
            run.checks[name] = finite != 0;
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
        std::printf("[glue-fusions] SKIP: no GPU\n");
        return 77;
    }
    const std::vector<Arm> arms = {
        {"base",    {"LUCE_TEST_GLUE=000"}},
        {"q8",      {"LUCE_TEST_GLUE=100"}},
        {"strided", {"LUCE_TEST_GLUE=010"}},
        {"gateup",  {"LUCE_TEST_GLUE=001"}},
        {"all",     {"LUCE_TEST_GLUE=111"}},
    };
    std::vector<ArmRun> runs(arms.size());
    for (size_t i = 0; i < arms.size(); ++i) {
        if (!run_child(arms[i], runs[i])) {
            std::printf("[glue-fusions] FAIL: cannot spawn the %s arm\n", arms[i].name);
            return 1;
        }
        if (runs[i].rc == 77) {
            std::printf("[glue-fusions] SKIP: the %s arm found no GPU\n", arms[i].name);
            return 77;
        }
    }

    bool ok = true;
    auto fail = [&](const std::string & what) {
        std::printf("[glue-fusions] FAIL %s\n", what.c_str());
        ok = false;
    };
    const ArmRun & base = runs[0];
    for (size_t i = 0; i < arms.size(); ++i) {
        const ArmRun & r = runs[i];
        const std::string arm = arms[i].name;
        for (const std::string & e : r.errors) fail(arm + ": " + e);
        if (r.rc != 0) fail(arm + ": child exit " + std::to_string(r.rc));
        if (r.hashes.size() != base.hashes.size() || base.hashes.empty()) {
            fail(arm + ": " + std::to_string(r.hashes.size()) + " outputs, base has " +
                 std::to_string(base.hashes.size()));
        }
        for (const auto & [name, h] : base.hashes) {
            const auto it = r.hashes.find(name);
            if (it == r.hashes.end()) {
                fail(arm + " " + name + ": missing");
            } else if (it->second != h) {
                fail(arm + " " + name + ": hash differs from base");
            }
        }
        for (const auto & [name, pass] : r.checks) {
            if (!pass) fail(arm + " " + name + ": the two forms differ");
        }
        if (r.checks.size() != base.checks.size() || base.checks.empty()) {
            fail(arm + ": " + std::to_string(r.checks.size()) + " pair checks, base has " +
                 std::to_string(base.checks.size()));
        }
        for (const auto & [name, gs] : base.graphs) {
            const auto it = r.graphs.find(name);
            if (it == r.graphs.end()) {
                fail(arm + " graph " + name + ": missing");
                continue;
            }
            if (!it->second.finite) fail(arm + " graph " + name + ": non-finite output");
            // The same MMQ calls, except that GATEUP_STACK runs one per FFN
            // (two layers) in each of the tiny target's two 16-token verifies.
            const bool gateup_on = arm == "gateup" || arm == "all";
            const size_t fewer = name == "tiny" && gateup_on && r.stack_exact ? 2 * 2 : 0;
            if (it->second.mmq + fewer != gs.mmq) {
                fail(arm + " graph " + name + ": " + std::to_string(it->second.mmq) + " MMQ calls, base " +
                     std::to_string(gs.mmq) + " less " + std::to_string(fewer));
            }
        }
    }
    // Memo hits: none with the memo off; with it on a legacy pool, exactly
    // kQ8MemoHits in the q8 graph and kTinyMemoHits in the tiny target, plus
    // one per producer write (none without the memo; some in the tiny
    // target, whose norms and SWIGLU feed MMQ); none on a VMM pool (the memo
    // stays inert there).
    // Node counts of the tiny verify graph: STRIDED_GDN drops the beta, alpha
    // and z copies of its one DeltaNet layer.
    const GraphStats & tiny_base = runs[0].graphs["tiny"];
    // ...and keeps every copy of the tree verify graph (not the dense chain).
    const GraphStats & tree_base = runs[0].graphs["tree"];
    if (tree_base.n_cont < 3) {
        fail("base tree verify graph: " + std::to_string(tree_base.n_cont) +
             " CONT nodes, want at least the beta, alpha and z copies");
    }
    for (size_t i = 0; i < arms.size(); ++i) {
        const std::string arm = arms[i].name;
        const bool memo_on = arm == "q8" || arm == "all";
        const bool strided_on = arm == "strided" || arm == "all";
        const bool gateup_on = arm == "gateup" || arm == "all";
        for (const auto & [name, gs] : runs[i].graphs) {
            size_t want = 0;
            if (memo_on && runs[i].legacy_pool) {
                want = name == "q8" ? kQ8MemoHits
                     : name == "tiny" ? (gateup_on && runs[i].stack_exact ? kTinyMemoHitsStacked : kTinyMemoHits)
                     : 0;
            }
            if (!(memo_on && runs[i].legacy_pool) && gs.writes != 0) {
                fail(arm + " graph " + name + ": " + std::to_string(gs.writes) +
                     " q8_1 producer writes without the memo");
            }
            if (memo_on && runs[i].legacy_pool && name == "tiny" && gs.writes == 0) {
                fail(arm + " graph tiny: no q8_1 producer write");
            }
            if (gs.hits != want + gs.writes) {
                fail(arm + " graph " + name + ": " + std::to_string(gs.hits) + " memo hits, want " +
                     std::to_string(want) + " + " + std::to_string(gs.writes) + " producer writes");
            }
        }
        const GraphStats & tiny = runs[i].graphs["tiny"];
        const int want_cont = tiny_base.n_cont - (strided_on ? 3 : 0);
        if (tiny.n_cont != want_cont) {
            fail(arm + " tiny verify graph: " + std::to_string(tiny.n_cont) + " CONT nodes, want " +
                 std::to_string(want_cont));
        }
        const GraphStats & tree = runs[i].graphs["tree"];
        if (tree.n_cont != tree_base.n_cont) {
            fail(arm + " tree verify graph: " + std::to_string(tree.n_cont) + " CONT nodes, base has " +
                 std::to_string(tree_base.n_cont));
        }
        // ...and GATEUP_STACK one GEMM per FFN (two layers) on RDNA/legacy
        // MMQ devices; elsewhere the stack stays inert.
        const int want_mm = tiny_base.n_mul_mat - (gateup_on && runs[i].stack_exact ? 2 : 0);
        if (tiny.n_mul_mat != want_mm) {
            fail(arm + " tiny verify graph: " + std::to_string(tiny.n_mul_mat) + " MUL_MAT nodes, want " +
                 std::to_string(want_mm));
        }
    }
    if (!ok) {
        return 1;
    }
    std::printf("[glue-fusions] PASS: %zu outputs and %zu pair checks bit-identical across %zu arms "
                "(q8 memo hits %zu + %zu, legacy pool %d; tiny verify CONT %d -> %d, MUL_MAT %d -> %d; "
                "tree verify CONT %d kept)\n",
                base.hashes.size(), base.checks.size(), arms.size(), runs[1].graphs["q8"].hits,
                runs[1].graphs["tiny"].hits, runs[1].legacy_pool ? 1 : 0, tiny_base.n_cont,
                runs[4].graphs["tiny"].n_cont, tiny_base.n_mul_mat, runs[4].graphs["tiny"].n_mul_mat,
                tree_base.n_cont);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--child") == 0) {
        return child_main();
    }
    return parent_main();
}
