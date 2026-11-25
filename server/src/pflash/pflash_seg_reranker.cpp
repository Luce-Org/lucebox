// Bidirectional Segment Transformer Reranker implementation (ggml).
// See pflash_seg_reranker.h.

#include "pflash_seg_reranker.h"
#include "pflash_drafter.h"

#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

namespace luce::common {

void free_pflash_seg_reranker(PFlashSegRerankerState & st) {
    if (st.buf) {
        ggml_backend_buffer_free(st.buf);
        st.buf = nullptr;
    }
    if (st.ctx) {
        ggml_free(st.ctx);
        st.ctx = nullptr;
    }
    st.layers.clear();
    st.k_in_w = st.k_in_b = st.q_in_w = st.q_in_b = nullptr;
    st.s_in_w = st.s_in_b = st.int_in_w = st.in_norm_w = nullptr;
    st.out_norm_w = st.score_head_w = st.score_head_b = nullptr;
    st.pool_gate_w = st.pool_gate_b = st.budget_head_w = st.budget_head_b = nullptr;
    st.delta_min = -1e9f;
    st.delta_max = 1e9f;
    st.loaded_file_identity.clear();
    st.loaded = false;
}

namespace {

static bool kv_str(gguf_context * g, const char * key, std::string & out) {
    const int id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) return false;
    out = gguf_get_val_str(g, id);
    return true;
}

static bool kv_u32(gguf_context * g, const char * key, int & out) {
    const int id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) return false;
    out = (int) gguf_get_val_u32(g, id);
    return true;
}

static bool kv_f32(gguf_context * g, const char * key, float & out) {
    const int id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) return false;
    out = gguf_get_val_f32(g, id);
    return true;
}

static void zscore_clamped(std::vector<float> & v, float lo, float hi) {
    const size_t n = v.size();
    if (n == 0) return;
    double mean = 0.0;
    for (float x : v) mean += (double) x;
    mean /= (double) n;
    double var = 0.0;
    for (float x : v) {
        const double d = (double) x - mean;
        var += d * d;
    }
    var /= (double) n;
    const double inv_std = 1.0 / std::max(1e-5, std::sqrt(var));
    for (float & x : v) {
        double z = ((double) x - mean) * inv_std;
        if (z < lo) z = lo;
        if (z > hi) z = hi;
        x = (float) z;
    }
}

static constexpr int kMaxRerankerInDim  = 16384;
static constexpr int kMaxRerankerDModel = 4096;
static constexpr int kMaxRerankerHeads  = 128;
static constexpr int kMaxRerankerLayers = 32;
static constexpr int kMaxRerankerDff    = 16384;

struct RerankerMeta {
    std::string feature_mode;
    int in_dim = 0;
    int n_scalars = 0;
    int d_model = 0;
    int n_heads = 0;
    int n_layers = 0;
    int d_ff = 0;
    float rope_theta = 0.0f;
    float norm_eps = 0.0f;
    float delta_min = -1e9f;
    float delta_max = 1e9f;
};

struct RerankerTensorSpec {
    std::string name;
    int n_dims;
    int64_t ne0;
    int64_t ne1;
};

static bool parse_and_validate_reranker_header(
        gguf_context * g,
        ggml_context * data_ctx,
        uint64_t file_size,
        const std::string & expected_drafter_sha256,
        const std::string & expected_head_sha256,
        int expected_k_dim,
        RerankerMeta & meta,
        std::vector<RerankerTensorSpec> & specs,
        std::string & error) {
    std::string arch, schema, mode, drafter_sha;
    if (!kv_str(g, "general.architecture", arch) || arch != "pflash_seg_reranker" ||
        !kv_str(g, "reranker.schema", schema) || schema != "pflash_seg_reranker_v1" ||
        !kv_str(g, "reranker.feature_mode", mode) ||
        !kv_str(g, "reranker.runtime_gguf_sha256", drafter_sha)) {
        error = "seg reranker metadata contract mismatch";
        return false;
    }
    if (!expected_drafter_sha256.empty() && drafter_sha != expected_drafter_sha256) {
        error = "seg reranker drafter sha256 mismatch";
        return false;
    }
    // Its inputs come from the scoring head (Stage-1 mass, head-projected keys and query),
    // so a reranker that names its head serves only beside that head.
    std::string head_sha;
    if (kv_str(g, "reranker.scoring_head_sha256", head_sha) && !expected_head_sha256.empty() &&
        head_sha != expected_head_sha256) {
        error = "seg reranker scoring head sha256 mismatch";
        return false;
    }
    if (mode != "kbar" && mode != "kbar_katt") {
        error = "seg reranker feature mode " + mode + " is not kbar or kbar_katt";
        return false;
    }
    if (!kv_u32(g, "reranker.in_dim", meta.in_dim) ||
        !kv_u32(g, "reranker.n_scalars", meta.n_scalars) ||
        !kv_u32(g, "reranker.d_model", meta.d_model) ||
        !kv_u32(g, "reranker.n_heads", meta.n_heads) ||
        !kv_u32(g, "reranker.n_layers", meta.n_layers) ||
        !kv_u32(g, "reranker.d_ff", meta.d_ff) ||
        !kv_f32(g, "reranker.rope_theta", meta.rope_theta) ||
        !kv_f32(g, "reranker.norm_eps", meta.norm_eps)) {
        error = "seg reranker dimension metadata missing";
        return false;
    }
    if (meta.in_dim <= 0 || meta.in_dim > kMaxRerankerInDim ||
        meta.n_scalars != 12 ||
        meta.d_model <= 0 || meta.d_model > kMaxRerankerDModel ||
        meta.n_heads <= 0 || meta.n_heads > kMaxRerankerHeads ||
        (meta.d_model % meta.n_heads) != 0 ||
        ((meta.d_model / meta.n_heads) % 2) != 0 ||
        meta.n_layers <= 0 || meta.n_layers > kMaxRerankerLayers ||
        meta.d_ff <= 0 || meta.d_ff > kMaxRerankerDff) {
        error = "seg reranker invalid dimensions";
        return false;
    }
    if (!std::isfinite(meta.rope_theta) || !(meta.rope_theta > 0.0f) ||
        !std::isfinite(meta.norm_eps) || !(meta.norm_eps > 0.0f)) {
        error = "seg reranker invalid rope_theta or norm_eps";
        return false;
    }
    if (expected_k_dim > 0) {
        const int want_in = (mode == "kbar_katt") ? (2 * expected_k_dim) : expected_k_dim;
        if (meta.in_dim != want_in) {
            error = "in_dim " + std::to_string(meta.in_dim) + " does not match the drafter's " +
                    std::to_string(expected_k_dim) + "-wide keys for feature mode " + mode;
            return false;
        }
    }
    meta.feature_mode = mode;
    meta.delta_min = -1e9f;
    meta.delta_max = 1e9f;
    {
        float dmin = meta.delta_min;
        float dmax = meta.delta_max;
        if (kv_f32(g, "reranker.delta_min", dmin) && std::isfinite(dmin)) {
            meta.delta_min = dmin;
        }
        if (kv_f32(g, "reranker.delta_max", dmax) && std::isfinite(dmax)) {
            meta.delta_max = dmax;
        }
        if (meta.delta_min > meta.delta_max) {
            error = "seg reranker invalid delta bounds";
            return false;
        }
    }

    const int64_t I = meta.in_dim;
    const int64_t S = meta.n_scalars;
    const int64_t D = meta.d_model;
    const int64_t F = meta.d_ff;

    specs = {
        {"reranker.k_in.weight", 2, I, D},
        {"reranker.k_in.bias", 1, D, 1},
        {"reranker.q_in.weight", 2, I, D},
        {"reranker.q_in.bias", 1, D, 1},
        {"reranker.s_in.weight", 2, S, D},
        {"reranker.s_in.bias", 1, D, 1},
        {"reranker.int_in.weight", 2, D, D},
        {"reranker.in_norm.weight", 1, D, 1},
        {"reranker.out_norm.weight", 1, D, 1},
        {"reranker.score_head.weight", 2, D, 1},
        {"reranker.score_head.bias", 1, 1, 1},
        {"reranker.pool_gate.weight", 2, D, 1},
        {"reranker.pool_gate.bias", 1, 1, 1},
        {"reranker.budget_head.weight", 2, D, 1},
        {"reranker.budget_head.bias", 1, 1, 1},
    };
    const uint64_t data_offset = (uint64_t) gguf_get_data_offset(g);
    const auto check_tensor = [&](const RerankerTensorSpec & r) -> bool {
        ggml_tensor * src = ggml_get_tensor(data_ctx, r.name.c_str());
        // 2D tensors only feed ggml_mul_mat as src0, so they may be F16; 1D norms and biases
        // feed ggml_add/ggml_mul and stay F32.
        if (!src ||
            !(src->type == GGML_TYPE_F32 || (r.n_dims == 2 && src->type == GGML_TYPE_F16)) ||
            src->ne[0] != r.ne0 || src->ne[1] != r.ne1 ||
            src->ne[2] != 1 || src->ne[3] != 1) {
            error = "seg reranker tensor mismatch: " + r.name;
            return false;
        }
        const int64_t tid = gguf_find_tensor(g, r.name.c_str());
        if (tid < 0) {
            error = "seg reranker tensor missing from header: " + r.name;
            return false;
        }
        const uint64_t t_off = (uint64_t) gguf_get_tensor_offset(g, tid);
        const uint64_t t_sz = (uint64_t) gguf_get_tensor_size(g, tid);
        const uint64_t expected_sz = (uint64_t) ggml_nbytes(src);
        if (t_sz != expected_sz ||
            data_offset > file_size ||
            t_off > file_size - data_offset ||
            t_sz > file_size - data_offset - t_off) {
            error = "seg reranker truncated tensor payload: " + r.name;
            return false;
        }
        return true;
    };
    for (const auto & r : specs) {
        if (!check_tensor(r)) return false;
    }
    const int64_t expected_tensors = (int64_t) specs.size() + 9 * (int64_t) meta.n_layers;
    if (gguf_get_n_tensors(g) < expected_tensors) {
        error = "seg reranker tensor count too small for n_layers";
        return false;
    }
    specs.reserve((size_t) expected_tensors);
    for (int l = 0; l < meta.n_layers; ++l) {
        const std::string p = "reranker.blocks." + std::to_string(l) + ".";
        const RerankerTensorSpec layer_specs[9] = {
            {p + "norm1.weight", 1, D, 1},
            {p + "q_proj.weight", 2, D, D},
            {p + "k_proj.weight", 2, D, D},
            {p + "v_proj.weight", 2, D, D},
            {p + "o_proj.weight", 2, D, D},
            {p + "norm2.weight", 1, D, 1},
            {p + "gate_proj.weight", 2, D, F},
            {p + "up_proj.weight", 2, D, F},
            {p + "down_proj.weight", 2, F, D},
        };
        for (const auto & r : layer_specs) {
            if (!check_tensor(r)) return false;
            specs.push_back(r);
        }
    }
    // Validate the full aligned GGUF data section across every tensor in the file.
    {
        const uint64_t alignment = (uint64_t) gguf_get_alignment(g);
        const int64_t n_all = gguf_get_n_tensors(g);
        uint64_t total_data = 0;
        for (int64_t i = 0; i < n_all; ++i) {
            const uint64_t off = (uint64_t) gguf_get_tensor_offset(g, i);
            const uint64_t sz = (uint64_t) gguf_get_tensor_size(g, i);
            const uint64_t rem = alignment > 1 ? (sz % alignment) : 0;
            const uint64_t pad = rem ? (alignment - rem) : 0;
            if (sz > UINT64_MAX - pad || off > UINT64_MAX - (sz + pad)) {
                error = "seg reranker tensor size/offset overflow";
                return false;
            }
            const uint64_t end = off + sz + pad;
            if (end > total_data) total_data = end;
        }
        if (data_offset > file_size || total_data > file_size - data_offset) {
            error = "seg reranker truncated data section";
            return false;
        }
    }
    return true;
}

} // namespace

bool validate_pflash_seg_reranker_gguf(
        const std::string & path,
        const std::string & expected_drafter_sha256,
        const std::string & expected_head_sha256,
        int expected_k_dim,
        std::string & error) {
    error.clear();
    FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        error = "seg reranker GGUF could not be opened: " + path;
        return false;
    }
    if (std::fseek(fp, 0, SEEK_END) != 0) {
        std::fclose(fp);
        error = "seg reranker GGUF seek failed: " + path;
        return false;
    }
    const long fsz_signed = std::ftell(fp);
    if (fsz_signed < 0) {
        std::fclose(fp);
        error = "seg reranker GGUF size check failed: " + path;
        return false;
    }
    const uint64_t file_size = (uint64_t) fsz_signed;
    std::rewind(fp);
    ggml_context * data_ctx = nullptr;
    gguf_init_params params{ /*no_alloc=*/ true, /*ctx=*/ &data_ctx };
    gguf_context * g = gguf_init_from_file_ptr(fp, params);
    std::fclose(fp);
    if (!g || !data_ctx) {
        if (g) gguf_free(g);
        if (data_ctx) ggml_free(data_ctx);
        error = "seg reranker GGUF could not be opened: " + path;
        return false;
    }
    RerankerMeta meta;
    std::vector<RerankerTensorSpec> specs;
    const bool ok = parse_and_validate_reranker_header(
        g, data_ctx, file_size, expected_drafter_sha256, expected_head_sha256, expected_k_dim, meta, specs, error);
    gguf_free(g);
    ggml_free(data_ctx);
    return ok;
}

bool load_pflash_seg_reranker(
        const std::string & path,
        ggml_backend_t backend,
        const std::string & expected_drafter_sha256,
        const std::string & expected_head_sha256,
        PFlashSegRerankerState & out,
        std::string & error) {
    free_pflash_seg_reranker(out);
    if (!backend) {
        error = "seg reranker requires a valid ggml backend";
        return false;
    }
    FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        error = "seg reranker GGUF could not be opened: " + path;
        return false;
    }
    if (std::fseek(fp, 0, SEEK_END) != 0) {
        std::fclose(fp);
        error = "seg reranker GGUF seek failed: " + path;
        return false;
    }
    const long fsz_signed = std::ftell(fp);
    if (fsz_signed < 0) {
        std::fclose(fp);
        error = "seg reranker GGUF size check failed: " + path;
        return false;
    }
    const uint64_t file_size = (uint64_t) fsz_signed;
    std::rewind(fp);
    const std::string loaded_id = pflash_file_identity_from_fd(path, fileno(fp));
    ggml_context * data_ctx = nullptr;
    // Parse header and read payload from the exact same open handle `fp`, with
    // no_alloc=true so malformed headers never allocate giant host buffers
    // before dimension, tensor-count, and payload-extent validation runs.
    gguf_init_params params{ /*no_alloc=*/ true, /*ctx=*/ &data_ctx };
    gguf_context * g = gguf_init_from_file_ptr(fp, params);
    if (!g || !data_ctx) {
        std::fclose(fp);
        if (g) gguf_free(g);
        if (data_ctx) ggml_free(data_ctx);
        error = "seg reranker GGUF could not be opened: " + path;
        return false;
    }
    auto fail = [&](const std::string & msg) {
        free_pflash_seg_reranker(out);
        std::fclose(fp);
        gguf_free(g);
        ggml_free(data_ctx);
        error = msg;
        return false;
    };

    RerankerMeta meta;
    std::vector<RerankerTensorSpec> specs;
    std::string header_err;
    if (!parse_and_validate_reranker_header(
            g, data_ctx, file_size, expected_drafter_sha256, expected_head_sha256, /*expected_k_dim=*/0,
            meta, specs, header_err)) {
        return fail(header_err);
    }
    out.loaded_file_identity = loaded_id;
    out.feature_mode = meta.feature_mode;
    out.in_dim = meta.in_dim;
    out.n_scalars = meta.n_scalars;
    out.d_model = meta.d_model;
    out.n_heads = meta.n_heads;
    out.n_layers = meta.n_layers;
    out.d_ff = meta.d_ff;
    out.rope_theta = meta.rope_theta;
    out.norm_eps = meta.norm_eps;
    out.delta_min = meta.delta_min;
    out.delta_max = meta.delta_max;

    std::vector<ggml_tensor **> dsts;
    dsts.reserve(specs.size());
    dsts = {
        &out.k_in_w, &out.k_in_b, &out.q_in_w, &out.q_in_b,
        &out.s_in_w, &out.s_in_b, &out.int_in_w, &out.in_norm_w,
        &out.out_norm_w, &out.score_head_w, &out.score_head_b,
        &out.pool_gate_w, &out.pool_gate_b, &out.budget_head_w, &out.budget_head_b,
    };
    out.layers.resize((size_t) out.n_layers);
    for (int l = 0; l < out.n_layers; ++l) {
        auto & L = out.layers[(size_t) l];
        dsts.push_back(&L.norm1_w);
        dsts.push_back(&L.q_proj_w);
        dsts.push_back(&L.k_proj_w);
        dsts.push_back(&L.v_proj_w);
        dsts.push_back(&L.o_proj_w);
        dsts.push_back(&L.norm2_w);
        dsts.push_back(&L.gate_proj_w);
        dsts.push_back(&L.up_proj_w);
        dsts.push_back(&L.down_proj_w);
    }

    ggml_init_params ip{};
    ip.mem_size = (specs.size() + 8) * ggml_tensor_overhead() + 4096;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return fail("seg reranker context alloc failed");

    int n_matmul = 0;
    int n_matmul_f16 = 0;
    for (size_t idx = 0; idx < specs.size(); ++idx) {
        const auto & r = specs[idx];
        ggml_tensor * src = ggml_get_tensor(data_ctx, r.name.c_str());
        *dsts[idx] = r.n_dims == 1
            ? ggml_new_tensor_1d(out.ctx, src->type, r.ne0)
            : ggml_new_tensor_2d(out.ctx, src->type, r.ne0, r.ne1);
        if (ggml_nbytes(src) != ggml_nbytes(*dsts[idx])) {
            return fail("seg reranker tensor byte size mismatch: " + r.name);
        }
        ggml_set_name(*dsts[idx], r.name.c_str());
        if (r.n_dims == 2) {
            ++n_matmul;
            if (src->type == GGML_TYPE_F16) ++n_matmul_f16;
        }
    }
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) return fail("seg reranker backend buffer alloc failed");

    const uint64_t data_offset = (uint64_t) gguf_get_data_offset(g);
    std::vector<uint8_t> scratch;
    for (size_t idx = 0; idx < specs.size(); ++idx) {
        const auto & r = specs[idx];
        const int64_t tid = gguf_find_tensor(g, r.name.c_str());
        const uint64_t offset = data_offset + (uint64_t) gguf_get_tensor_offset(g, tid);
        const size_t nbytes = ggml_nbytes(*dsts[idx]);
        scratch.resize(nbytes);
        if (std::fseek(fp, (long) offset, SEEK_SET) != 0 ||
            std::fread(scratch.data(), 1, nbytes, fp) != nbytes) {
            return fail("seg reranker failed to read tensor payload: " + r.name);
        }
        ggml_backend_tensor_set(*dsts[idx], scratch.data(), 0, nbytes);
    }

    std::fclose(fp);
    gguf_free(g);
    ggml_free(data_ctx);
    out.loaded = true;
    if (out.delta_min > -1e8f || out.delta_max < 1e8f) {
        std::fprintf(stderr,
            "[pflash-reranker] loaded %s (mode=%s in=%d d=%d H=%d L=%d ff=%d f16_weights=%d/%d delta=[%.3g,%.3g])\n",
            path.c_str(), out.feature_mode.c_str(), out.in_dim, out.d_model,
            out.n_heads, out.n_layers, out.d_ff, n_matmul_f16, n_matmul,
            out.delta_min, out.delta_max);
    } else {
        std::fprintf(stderr,
            "[pflash-reranker] loaded %s (mode=%s in=%d d=%d H=%d L=%d ff=%d f16_weights=%d/%d)\n",
            path.c_str(), out.feature_mode.c_str(), out.in_dim, out.d_model,
            out.n_heads, out.n_layers, out.d_ff, n_matmul_f16, n_matmul);
    }
    std::fflush(stderr);
    return true;
}

void pflash_reranker_build_scalars(
        int input_tokens,
        const std::vector<float> & token_mass,
        const std::vector<PFlashTokenSpan> & segments,
        const std::vector<float> * container_probs,
        const std::vector<float> * adj_kcos,
        std::vector<float> & out_scalars,
        std::vector<float> & out_log_density) {
    const size_t N = segments.size();
    out_scalars.assign(N * 12, 0.0f);
    out_log_density.assign(N, 0.0f);
    if (N == 0) return;

    constexpr double kEps = 1e-12;
    std::vector<float> log_dens(N, 0.0f);
    std::vector<float> log_mass(N, 0.0f);
    std::vector<float> log_max(N, 0.0f);
    std::vector<float> log_top4(N, 0.0f);
    std::vector<float> log_peak_ratio(N, 0.0f);
    std::vector<float> log_len(N, 0.0f);
    std::vector<float> rel_pos(N, 0.0f);
    std::vector<float> cont_left(N, 0.0f);
    std::vector<float> cont_right(N, 1.0f);
    std::vector<float> kcos_left(N, 0.0f);
    std::vector<float> kcos_right(N, 0.0f);

    float max_log_dens = -1e30f;
    const int tok_sz = (int) token_mass.size();
    const double inv_S = 1.0 / (double) std::max(1, input_tokens);

    for (size_t i = 0; i < N; ++i) {
        const int s0 = std::max(0, segments[i].begin);
        const int e0 = std::min(tok_sz, segments[i].end);
        const int L = std::max(1, e0 - s0);
        double sum_m = 0.0;
        float top4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int n_top = 0;
        for (int t = s0; t < e0; ++t) {
            const float v = token_mass[(size_t) t];
            sum_m += (double) v;
            if (n_top < 4) {
                top4[n_top++] = v;
                for (int k = n_top - 1; k > 0 && top4[k] > top4[k - 1]; --k) {
                    std::swap(top4[k], top4[k - 1]);
                }
            } else if (v > top4[3]) {
                top4[3] = v;
                for (int k = 3; k > 0 && top4[k] > top4[k - 1]; --k) {
                    std::swap(top4[k], top4[k - 1]);
                }
            }
        }
        const double max_v = n_top > 0 ? (double) top4[0] : 0.0;
        double top4_mean = 0.0;
        for (int k = 0; k < n_top; ++k) top4_mean += (double) top4[k];
        if (n_top > 0) top4_mean /= (double) n_top;

        const double dens = std::max(kEps, sum_m / (double) L);
        const float ld = (float) std::log(dens);
        out_log_density[i] = ld;
        log_dens[i] = ld;
        if (ld > max_log_dens) max_log_dens = ld;

        log_mass[i] = (float) std::log(std::max(kEps, sum_m));
        log_max[i]  = (float) std::log(std::max(kEps, max_v));
        log_top4[i] = (float) std::log(std::max(kEps, top4_mean));
        log_peak_ratio[i] = std::clamp((float) std::log(std::max(kEps, max_v) / dens), -2.0f, 8.0f);
        log_len[i] = std::clamp((float) std::log((double) L / 64.0), -4.0f, 4.0f);
        rel_pos[i] = (float) (0.5 * (double) (s0 + e0) * inv_S);

        if (container_probs && !container_probs->empty()) {
            const int cp_sz = (int) container_probs->size();
            cont_left[i] = (s0 >= 0 && s0 < cp_sz) ? (*container_probs)[(size_t) s0] : 0.0f;
            if (i + 1 < N) {
                const int nxt = segments[i + 1].begin;
                cont_right[i] = (nxt >= 0 && nxt < cp_sz) ? (*container_probs)[(size_t) nxt] : 1.0f;
            } else {
                cont_right[i] = 1.0f;
            }
        }
        if (adj_kcos && adj_kcos->size() == N) {
            kcos_left[i] = (*adj_kcos)[i];
            kcos_right[i] = (i + 1 < N) ? (*adj_kcos)[i + 1] : 0.0f;
        }
    }

    std::vector<float> log_dens_z = log_dens;
    zscore_clamped(log_dens_z, -6.0f, 6.0f);
    zscore_clamped(log_mass, -6.0f, 6.0f);
    zscore_clamped(log_max, -6.0f, 6.0f);
    zscore_clamped(log_top4, -6.0f, 6.0f);

    for (size_t i = 0; i < N; ++i) {
        float * row = out_scalars.data() + i * 12;
        row[0]  = log_dens_z[i];
        row[1]  = std::clamp(log_dens[i] - max_log_dens, -12.0f, 0.0f);
        row[2]  = log_mass[i];
        row[3]  = log_max[i];
        row[4]  = log_top4[i];
        row[5]  = log_peak_ratio[i];
        row[6]  = log_len[i];
        row[7]  = rel_pos[i];
        row[8]  = cont_left[i];
        row[9]  = cont_right[i];
        row[10] = kcos_left[i];
        row[11] = kcos_right[i];
    }
}

bool pflash_seg_reranker_run(
        const PFlashSegRerankerState & st,
        ggml_backend_t backend,
        int n_seg,
        const float * seg_vec,
        const float * q_vec,
        const float * scalars,
        const float * log_density,
        std::vector<float> & out_scores,
        std::vector<float> * out_delta,
        float * out_m_star,
        std::string * error) {
    if (!st.loaded || !backend || n_seg <= 0 || !seg_vec || !q_vec || !scalars || !log_density) {
        if (error) *error = "invalid seg reranker input";
        return false;
    }

    const int I = st.in_dim;
    const int S = st.n_scalars;
    const int D = st.d_model;
    const int H = st.n_heads;
    const int Dh = D / H;
    const int L = st.n_layers;

    // 1. Allocate input tensors on backend
    ggml_init_params lip{};
    lip.mem_size = 8 * ggml_tensor_overhead() + 4096;
    lip.no_alloc = true;
    ggml_context * in_ctx = ggml_init(lip);
    if (!in_ctx) {
        if (error) *error = "seg reranker input ctx alloc failed";
        return false;
    }
    ggml_tensor * t_seg = ggml_new_tensor_2d(in_ctx, GGML_TYPE_F32, I, n_seg);
    ggml_tensor * t_q   = ggml_new_tensor_2d(in_ctx, GGML_TYPE_F32, I, 1);
    ggml_tensor * t_sc  = ggml_new_tensor_2d(in_ctx, GGML_TYPE_F32, S, n_seg);
    ggml_tensor * t_pos = ggml_new_tensor_1d(in_ctx, GGML_TYPE_I32, n_seg);
    ggml_backend_buffer_t in_buf = ggml_backend_alloc_ctx_tensors(in_ctx, backend);
    if (!in_buf) {
        ggml_free(in_ctx);
        if (error) *error = "seg reranker input buf alloc failed";
        return false;
    }
    ggml_backend_tensor_set(t_seg, seg_vec, 0, (size_t) n_seg * (size_t) I * sizeof(float));
    ggml_backend_tensor_set(t_q, q_vec, 0, (size_t) I * sizeof(float));
    ggml_backend_tensor_set(t_sc, scalars, 0, (size_t) n_seg * (size_t) S * sizeof(float));
    std::vector<int32_t> pos_h((size_t) n_seg);
    std::iota(pos_h.begin(), pos_h.end(), 0);
    ggml_backend_tensor_set(t_pos, pos_h.data(), 0, (size_t) n_seg * sizeof(int32_t));

    // 2. Build bidirectional encoder compute graph
    const size_t graph_nodes = std::max<size_t>(4096, 256 + 96 * (size_t) L);
    ggml_init_params gip{};
    gip.mem_size = (size_t) (128 + 64 * L) * ggml_tensor_overhead() +
                   ggml_graph_overhead_custom(graph_nodes, false) + 64 * 1024;
    gip.no_alloc = true;
    ggml_context * gctx = ggml_init(gip);
    if (!gctx) {
        ggml_backend_buffer_free(in_buf);
        ggml_free(in_ctx);
        if (error) *error = "seg reranker graph ctx alloc failed";
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, graph_nodes, false);
    if (!gf) {
        ggml_free(gctx);
        ggml_backend_buffer_free(in_buf);
        ggml_free(in_ctx);
        if (error) *error = "seg reranker graph alloc failed";
        return false;
    }

    // Input projection:
    //   k_d = seg_vec @ k_in_w^T + k_in_b       [D, n_seg]
    //   q_d = q_vec   @ q_in_w^T + q_in_b       [D, 1]
    //   s_d = scalars @ s_in_w^T + s_in_b       [D, n_seg]
    //   inter = (k_d * q_d) @ int_in_w^T        [D, n_seg]
    //   x = RMSNorm(k_d + q_d + s_d + inter)    [D, n_seg]
    ggml_tensor * k_d = ggml_add(gctx, ggml_mul_mat(gctx, st.k_in_w, t_seg), st.k_in_b);
    ggml_tensor * q_d = ggml_add(gctx, ggml_mul_mat(gctx, st.q_in_w, t_q), st.q_in_b);
    ggml_tensor * s_d = ggml_add(gctx, ggml_mul_mat(gctx, st.s_in_w, t_sc), st.s_in_b);
    ggml_tensor * inter = ggml_mul_mat(gctx, st.int_in_w, ggml_mul(gctx, k_d, q_d));
    ggml_tensor * x = ggml_add(gctx, ggml_add(gctx, ggml_add(gctx, k_d, s_d), inter), q_d);
    x = ggml_mul(gctx, ggml_rms_norm(gctx, x, st.norm_eps), st.in_norm_w);

    const float attn_scale = 1.0f / std::sqrt((float) Dh);

    for (int l = 0; l < L; ++l) {
        const auto & blk = st.layers[(size_t) l];
        ggml_tensor * h = ggml_mul(gctx, ggml_rms_norm(gctx, x, st.norm_eps), blk.norm1_w);

        ggml_tensor * Q = ggml_reshape_3d(gctx, ggml_mul_mat(gctx, blk.q_proj_w, h), Dh, H, n_seg);
        ggml_tensor * K = ggml_reshape_3d(gctx, ggml_mul_mat(gctx, blk.k_proj_w, h), Dh, H, n_seg);
        ggml_tensor * V = ggml_reshape_3d(gctx, ggml_mul_mat(gctx, blk.v_proj_w, h), Dh, H, n_seg);

        // 1D segment-index NeoX RoPE matching PyTorch apply_1d_rope bit-for-bit!
        Q = ggml_rope_ext(gctx, Q, t_pos, nullptr, Dh, GGML_ROPE_TYPE_NEOX, 0,
                          st.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(gctx, K, t_pos, nullptr, Dh, GGML_ROPE_TYPE_NEOX, 0,
                          st.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Permute to [Dh, n_seg, H] for Q and K; [n_seg, Dh, H] for V
        ggml_tensor * Qp = ggml_cont(gctx, ggml_permute(gctx, Q, 0, 2, 1, 3));
        ggml_tensor * Kp = ggml_cont(gctx, ggml_permute(gctx, K, 0, 2, 1, 3));
        ggml_tensor * Vp = ggml_cont(gctx, ggml_permute(gctx, V, 1, 2, 0, 3));
        // Note: ggml_permute(V [Dh, H, n_seg], 1, 2, 0, 3):
        //   old dim0 (Dh) -> new dim1
        //   old dim1 (H)  -> new dim2
        //   old dim2 (N)  -> new dim0
        // So Vp has shape [n_seg, Dh, H]!

        // Full bidirectional attention (NO causal mask!):
        // KQ = mul_mat(Kp [Dh, N, H], Qp [Dh, N, H]) -> [N_key, N_query, H]
        ggml_tensor * KQ = ggml_mul_mat(gctx, Kp, Qp);
        ggml_tensor * probs = ggml_soft_max_ext(gctx, KQ, nullptr, attn_scale, 0.0f);
        // KQV = mul_mat(Vp [N_key, Dh, H], probs [N_key, N_query, H]) -> [Dh, N_query, H]
        ggml_tensor * KQV = ggml_mul_mat(gctx, Vp, probs);
        // Back to [Dh, H, N_query] -> [D, N_query]
        ggml_tensor * attn_out = ggml_reshape_2d(
            gctx, ggml_cont(gctx, ggml_permute(gctx, KQV, 0, 2, 1, 3)), D, n_seg);
        x = ggml_add(gctx, x, ggml_mul_mat(gctx, blk.o_proj_w, attn_out));

        // SwiGLU FFN
        ggml_tensor * h2 = ggml_mul(gctx, ggml_rms_norm(gctx, x, st.norm_eps), blk.norm2_w);
        ggml_tensor * gate = ggml_silu(gctx, ggml_mul_mat(gctx, blk.gate_proj_w, h2));
        ggml_tensor * up   = ggml_mul_mat(gctx, blk.up_proj_w, h2);
        ggml_tensor * ff   = ggml_mul_mat(gctx, blk.down_proj_w, ggml_mul(gctx, gate, up));
        x = ggml_add(gctx, x, ff);
    }

    ggml_tensor * h_out = ggml_mul(gctx, ggml_rms_norm(gctx, x, st.norm_eps), st.out_norm_w);
    // delta: [1, n_seg]
    ggml_tensor * delta_t = ggml_add(gctx, ggml_mul_mat(gctx, st.score_head_w, h_out), st.score_head_b);
    ggml_set_output(delta_t);
    ggml_build_forward_expand(gf, delta_t);

    ggml_tensor * m_logit_t = nullptr;
    if (out_m_star) {
        // pool_logits: [1, n_seg] -> transpose [n_seg, 1] -> softmax -> [n_seg, 1]
        ggml_tensor * pl = ggml_cont(gctx, ggml_transpose(
            gctx, ggml_add(gctx, ggml_mul_mat(gctx, st.pool_gate_w, h_out), st.pool_gate_b)));
        ggml_tensor * pw = ggml_soft_max(gctx, pl); // [n_seg, 1]
        // z_doc = mul_mat(transpose(h_out) [n_seg, D], pw [n_seg, 1]) -> [D, 1]
        ggml_tensor * z_doc = ggml_mul_mat(gctx, ggml_cont(gctx, ggml_transpose(gctx, h_out)), pw);
        m_logit_t = ggml_add(gctx, ggml_mul_mat(gctx, st.budget_head_w, z_doc), st.budget_head_b);
        ggml_set_output(m_logit_t);
        ggml_build_forward_expand(gf, m_logit_t);
    }

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    const bool ok = alloc != nullptr &&
                    ggml_gallocr_alloc_graph(alloc, gf) &&
                    ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;

    std::vector<float> delta_h((size_t) n_seg, 0.0f);
    float m_logit = 0.0f;
    if (ok) {
        ggml_backend_tensor_get(delta_t, delta_h.data(), 0, (size_t) n_seg * sizeof(float));
        if (m_logit_t) {
            ggml_backend_tensor_get(m_logit_t, &m_logit, 0, sizeof(float));
        }
    }

    if (alloc) ggml_gallocr_free(alloc);
    ggml_free(gctx);
    ggml_backend_buffer_free(in_buf);
    ggml_free(in_ctx);

    if (!ok) {
        if (error) *error = "seg reranker graph compute failed";
        return false;
    }

    out_scores.resize((size_t) n_seg);
    for (int i = 0; i < n_seg; ++i) {
        const float raw_d = delta_h[(size_t) i];
        if (!std::isfinite(raw_d) || !std::isfinite(log_density[i])) {
            out_scores.clear();
            if (error) *error = "seg reranker produced non-finite output";
            return false;
        }
        const float d = std::clamp(raw_d, st.delta_min, st.delta_max);
        delta_h[(size_t) i] = d;
        out_scores[(size_t) i] = log_density[i] + d;
    }
    if (out_m_star) {
        if (!std::isfinite(m_logit)) {
            out_scores.clear();
            if (error) *error = "seg reranker produced non-finite budget logit";
            return false;
        }
        const float sig = 1.0f / (1.0f + std::exp(-m_logit));
        *out_m_star = sig * 0.55f + 0.05f;
    }
    if (out_delta) *out_delta = std::move(delta_h);
    return true;
}

PFlashAdaptiveBudget pflash_reranker_adaptive_budget(
        const std::vector<PFlashTokenSpan> & segments,
        const std::vector<float> & rerank_scores,
        const std::vector<uint8_t> & mandatory_mask,
        const std::vector<double> * stage1_masses) {
    PFlashAdaptiveBudget out;
    const size_t n = segments.size();
    if (n == 0 || rerank_scores.size() != n || mandatory_mask.size() != n) {
        return out;
    }
    const bool has_masses = (stage1_masses != nullptr && stage1_masses->size() == n);

    // Collect all non-mandatory segments with valid finite scores.
    std::vector<int> opt_lens;
    std::vector<double> opt_scores;
    std::vector<double> opt_masses;
    opt_lens.reserve(n);
    opt_scores.reserve(n);
    if (has_masses) opt_masses.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (mandatory_mask[i]) continue;
        const int len = segments[i].end - segments[i].begin;
        const float sc = rerank_scores[i];
        if (len <= 0 || !std::isfinite(sc)) continue;
        opt_lens.push_back(len);
        opt_scores.push_back((double) sc);
        if (has_masses) {
            const double m_i = (*stage1_masses)[i];
            opt_masses.push_back((std::isfinite(m_i) && m_i > 0.0) ? m_i : 0.0);
        }
    }
    // Exclude the leading optional segment (BOS / document-start attention sink
    // immediately after chat template headers like `<|im_start|>user\n`) when
    // enough optional segments remain to measure the body distribution cleanly.
    const size_t start = (opt_lens.size() > 4) ? 1 : 0;
    const size_t m = opt_lens.size() - start;
    if (m < 2) {
        return out;
    }

    std::vector<double> sorted_scores(opt_scores.begin() + (ptrdiff_t) start, opt_scores.end());
    std::sort(sorted_scores.begin(), sorted_scores.end());
    const double med = (m % 2 == 1)
        ? sorted_scores[m / 2]
        : 0.5 * (sorted_scores[m / 2 - 1] + sorted_scores[m / 2]);

    constexpr double kCapAboveMed = 4.0;
    constexpr double kTemperature = 0.85;
    constexpr double kWidenEff = kPFlashAdaptiveWidenEff;
    constexpr double kBypassEff = kPFlashAdaptiveBypassEff;
    constexpr double kNucleusMassTarget = 0.90;

    double total_opt_tokens = 0.0;
    double max_capped = -1e300;
    for (size_t i = start; i < opt_lens.size(); ++i) {
        const double sc = std::min(opt_scores[i], med + kCapAboveMed);
        if (sc > max_capped) max_capped = sc;
        total_opt_tokens += (double) opt_lens[i];
    }
    if (!(total_opt_tokens > 0.0)) return out;

    std::vector<double> weights(m, 0.0);
    double sum_w = 0.0;
    for (size_t k = 0; k < m; ++k) {
        const double sc = std::min(opt_scores[start + k], med + kCapAboveMed);
        const double z = std::max(-60.0, (sc - max_capped) / kTemperature);
        const double w = (double) opt_lens[start + k] * std::exp(z);
        weights[k] = w;
        sum_w += w;
    }
    if (!(sum_w > 0.0) || !std::isfinite(sum_w)) return out;

    double h_tok = 0.0;
    for (size_t k = 0; k < m; ++k) {
        const double p = weights[k] / sum_w;
        if (p > 1e-30) {
            h_tok -= p * std::log(std::max(1e-30, p / (double) opt_lens[start + k]));
        }
    }
    const double eff = std::clamp(std::exp(h_tok) / total_opt_tokens, 0.0, 1.0);
    if (!std::isfinite(eff)) return out;

    // Nucleus / Top-p Stage-1 mass floor: walk all optional segments in descending
    // reranked order until 90% of Stage-1 attention mass is retained (capped at 0.58).
    double nucleus_r = 0.0;
    if (has_masses) {
        double tot_mass = 0.0;
        double all_opt_toks = 0.0;
        for (size_t i = 0; i < opt_lens.size(); ++i) {
            tot_mass += opt_masses[i];
            all_opt_toks += (double) opt_lens[i];
        }
        if (tot_mass > 1e-30 && all_opt_toks > 0.0) {
            std::vector<size_t> order(opt_lens.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                if (opt_scores[a] != opt_scores[b]) return opt_scores[a] > opt_scores[b];
                return a < b;
            });
            double cum_mass = 0.0;
            double nuc_toks = 0.0;
            const double mass_goal = kNucleusMassTarget * tot_mass;
            for (const size_t idx : order) {
                if (cum_mass >= mass_goal) break;
                cum_mass += opt_masses[idx];
                nuc_toks += (double) opt_lens[idx];
            }
            nucleus_r = std::clamp(nuc_toks / all_opt_toks, 0.0, kBypassEff);
        }
    }

    out.valid = true;
    out.eff = (float) eff;
    out.nucleus_ratio = (float) nucleus_r;
    if (eff >= kBypassEff) {
        out.bypass_to_full = true;
        out.keep_ratio = 1.0f;
    } else {
        const double eff_widen = (eff >= kWidenEff) ? eff : 0.0;
        out.bypass_to_full = false;
        out.keep_ratio = (float) std::max(eff_widen, nucleus_r);
    }
    return out;
}

} // namespace luce::common
