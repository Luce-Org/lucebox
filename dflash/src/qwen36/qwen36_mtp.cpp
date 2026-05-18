// qwen36_mtp.cpp — see qwen36_mtp.h for contract.
//
// PR 2d-bis (Shape B): implements the full DeepSeek-V3 NextN per-head forward
// per Eq 21-23: RMSNorm → eh_proj([hnorm(h_prev); enorm(embed)]) → head-owned
// TRMBlock (Q/K/V w/ QK-norm + RoPE + GQA → attn_output residual → SwiGLU FFN
// residual) → shared_head_norm → shared LM head projection.
//
// Implementation: Path A — CPU host floats. All per-head tensors are
// dequantized to float once via tensor_to_floats() and matvec / RMSNorm /
// SiLU / RoPE / softmax are hand-rolled on the host. Per-step cost is a few
// hundred ms for n_embd=5120, n_ffn=17408 (acceptable for proof-of-correctness;
// GPU migration is a follow-up PR).
//
// The MTP head's attn_q tensor is packed Q+gate (same convention as backbone's
// full-attention blocks): first q_dim elements = Q, last q_dim = gate. The gate
// is passed through sigmoid and multiplied into the attention output before the
// attn_output projection. This matches the backbone forward at blk.63.
//
// RoPE: standard rotary at rope_dimension_count=64 out of head_dim=256, using
// rope_theta=1e7 (qwen35.rope.freq_base from GGUF). For γ_max=1 the draft
// position is base_pos + 0 = base_pos.
//
// Phase A fallback: define MTP_PHASE_A_FALLBACK to bypass the TRMBlock and
// use only eh_proj+shared_head_norm+lm_head (useful if a smaller/synthetic GGUF
// lacks the transformer-block tensors). The default path requires all attn/ffn
// tensors to be non-null.

#include "qwen36_mtp.h"
#include "qwen36_mtp_graph.h"

#include "common/dflash_target.h"
#include "common/gguf_mmap.h"
#include "qwen35/qwen35_dflash_target.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dflash27b::mtp {

#ifdef DFLASH_MTP_PROFILE
// Per-iter profiler for the step_chain_gpu_ loop.  Enabled by -DDFLASH_MTP_PROFILE=1.
// Uses host-side wall-clock (same latency as cudaEvents because every ggml
// backend call internally calls cudaStreamSynchronize).
namespace {
inline bool mtp_profile_enabled() {
    static const bool on = (std::getenv("DFLASH_MTP_PROFILE") != nullptr);
    return on;
}

using prof_clock = std::chrono::steady_clock;
inline double prof_ms_since(prof_clock::time_point t0) {
    auto t1 = prof_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
}  // namespace
#endif  // DFLASH_MTP_PROFILE


// ── Internal helpers ──────────────────────────────────────────────────────

namespace {

constexpr size_t kMtpHeadKvSnapshotMaxBytes = 500ull * 1024ull * 1024ull;

// RMSNorm: out[i] = x[i] / rms(x) * weight[i]
// All operations in-place on a separate output buffer.
static void rmsnorm_cpu(const float * x,
                        const float * weight,
                        float * out,
                        int n,
                        float eps = 1e-6f) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    const float rms_inv = 1.0f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; i++) {
        // ggml weight tensors store as F32 rows; safe to cast for small dims
        out[i] = x[i] * rms_inv * weight[i];
    }
}

// Read a ggml_tensor's data as a flat float array. Returns false if the
// tensor is null or its type is not GGML_TYPE_F32.
static bool tensor_to_floats(const ggml_tensor * t,
                              std::vector<float> & out) {
    if (!t) return false;
    const size_t n = ggml_nelements(t);
    out.resize(n);

    // Stage the tensor's raw bytes to a host buffer.  Tensors backed by a
    // backend buffer (CPU or CUDA) require ggml_backend_tensor_get to copy
    // host-side; bare tensors created by tests (no buffer, raw host pointer
    // assigned to t->data) can be read directly.
    const size_t total_bytes = ggml_nbytes(t);
    std::vector<uint8_t> staging(total_bytes);
    const uint8_t * src_bytes = nullptr;
    if (t->buffer) {
        ggml_backend_tensor_get(const_cast<ggml_tensor *>(t),
                                 staging.data(), 0, total_bytes);
        src_bytes = staging.data();
    } else if (t->data) {
        src_bytes = static_cast<const uint8_t *>(t->data);
    } else {
        return false;
    }

    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), src_bytes, n * sizeof(float));
        return true;
    }

    const ggml_type_traits * tr = ggml_get_type_traits(t->type);
    if (!tr || !tr->to_float) return false;
    const int64_t row_len = t->ne[0];
    if (row_len <= 0 || n % (size_t)row_len != 0) return false;
    const int64_t n_rows = (int64_t)n / row_len;
    const size_t row_bytes = ggml_row_size(t->type, row_len);
    for (int64_t r = 0; r < n_rows; r++) {
        tr->to_float(src_bytes + (size_t)r * row_bytes,
                     out.data() + (size_t)r * row_len,
                     row_len);
    }
    return true;
}

// Matrix-vector multiply: y = A @ x
// A is [rows x cols] stored row-major (rows = out_dim, cols = in_dim).
static void matvec_cpu(const float * A,
                       const float * x,
                       float * y,
                       int rows,
                       int cols) {
    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        const float * row = A + (size_t)r * cols;
        for (int c = 0; c < cols; c++) acc += row[c] * x[c];
        y[r] = acc;
    }
}

// Argmax over a float vector; returns index of max element.
static int32_t argmax(const float * logits, int n) {
    int32_t best = 0;
    for (int i = 1; i < n; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

// Fill StepOutput.topk_logprobs / topk_ids with the K highest log-softmax
// entries from `logits` (length n_vocab), sorted DESCENDING by logprob.
// Uses partial_sort over (logprob, id) pairs — O(n_vocab + K log K) is
// trivial vs the per-head TRMBlock forward and avoids a full sort.
static void emit_topk_logprobs(const float * logits, int n_vocab, int K,
                               mtp::StepOutput & out) {
    K = std::min(K, n_vocab);
    if (K <= 0) return;

    // log-softmax: stable via max-shift + logsumexp.
    float max_l = logits[0];
    for (int i = 1; i < n_vocab; i++) if (logits[i] > max_l) max_l = logits[i];
    double denom = 0.0;
    for (int i = 0; i < n_vocab; i++) denom += std::exp((double)(logits[i] - max_l));
    const float log_denom = (float)std::log(denom) + max_l;

    // Pair (logprob, id); partial_sort top-K descending.
    std::vector<std::pair<float, int32_t>> scratch;
    scratch.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        scratch.emplace_back(logits[i] - log_denom, (int32_t)i);
    }
    std::partial_sort(scratch.begin(), scratch.begin() + K, scratch.end(),
                      [](const auto & a, const auto & b) {
                          return a.first > b.first;
                      });

    out.topk_logprobs.resize(K);
    out.topk_ids.resize(K);
    for (int i = 0; i < K; i++) {
        out.topk_logprobs[i] = scratch[i].first;
        out.topk_ids[i]      = scratch[i].second;
    }
}

// Per-head RMSNorm: apply rmsnorm_cpu to each n_per_head-element slice of `x`
// (total dim = n_heads_total * n_per_head) using corresponding weight slice.
// Weight tensor `w` has shape [n_per_head] (same weight shared across all heads
// when called for Q-norm / K-norm).
static void per_head_rmsnorm(float * x,
                             const float * w,
                             int n_heads_total,
                             int n_per_head,
                             float eps = 1e-6f) {
    for (int h = 0; h < n_heads_total; h++) {
        float * slice = x + (size_t)h * n_per_head;
        float ss = 0.0f;
        for (int i = 0; i < n_per_head; i++) ss += slice[i] * slice[i];
        const float inv = 1.0f / std::sqrt(ss / n_per_head + eps);
        for (int i = 0; i < n_per_head; i++) {
            slice[i] = slice[i] * inv * w[i];
        }
    }
}

// SiLU: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
static inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// Apply standard rotary position embedding (RoPE) in-place to a flat
// [n_heads * head_dim] buffer. Only the first `n_rot` elements of each head
// are rotated (n_rot <= head_dim). The remaining elements pass through.
// freq_base: rope theta (e.g. 1e7). position: absolute sequence position.
static void rope_cpu(float * x,
                     int n_heads,
                     int head_dim,
                     int n_rot,        // number of dims to rotate (e.g. 64)
                     int position,
                     float freq_base) {
    // n_rot must be even (pairs of dims rotated together).
    const int half = n_rot / 2;
    for (int h = 0; h < n_heads; h++) {
        float * head = x + (size_t)h * head_dim;
        for (int i = 0; i < half; i++) {
            const float theta = (float)position /
                std::pow(freq_base, (float)(2 * i) / (float)n_rot);
            const float cos_t = std::cos(theta);
            const float sin_t = std::sin(theta);
            const float x0 = head[i];
            const float x1 = head[i + half];
            head[i]        = x0 * cos_t - x1 * sin_t;
            head[i + half] = x0 * sin_t + x1 * cos_t;
        }
    }
}

// Multi-slot scaled dot-product attention with causal masking and GQA.
//
// q       : [n_head * head_dim]                        — queries at the current draft position
// k_cache : [n_slots * n_head_kv * head_dim]           — K cache, slot-major
// v_cache : [n_slots * n_head_kv * head_dim]           — V cache, slot-major
// out     : [n_head * head_dim]                        — attention output (GQA-expanded)
//
// Attends over slots [0, n_slots).  No explicit causal mask is needed because
// the caller passes only the slots representing positions <= the current
// draft position.  Score = (Q · K) / sqrt(head_dim).
static void range_attention(const float * q,
                            const float * k_cache,
                            const float * v_cache,
                            float * out,
                            int n_head,
                            int n_head_kv,
                            int head_dim,
                            int n_slots) {
    if (n_slots <= 0) {
        std::memset(out, 0, sizeof(float) * (size_t)n_head * head_dim);
        return;
    }
    const int   group  = n_head / n_head_kv;
    const float scale  = 1.0f / std::sqrt((float)head_dim);
    std::vector<float> scores(n_slots);
    for (int qh = 0; qh < n_head; qh++) {
        const int kvh        = qh / group;
        const float * qhead  = q + (size_t)qh * head_dim;
        for (int s = 0; s < n_slots; s++) {
            const float * khead =
                k_cache + ((size_t)s * n_head_kv + kvh) * head_dim;
            float acc = 0.0f;
            for (int i = 0; i < head_dim; i++) acc += qhead[i] * khead[i];
            scores[s] = acc * scale;
        }
        // Stable softmax.
        float max_s = scores[0];
        for (int s = 1; s < n_slots; s++) if (scores[s] > max_s) max_s = scores[s];
        float denom = 0.0f;
        for (int s = 0; s < n_slots; s++) {
            scores[s] = std::exp(scores[s] - max_s);
            denom += scores[s];
        }
        const float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
        float * ohead = out + (size_t)qh * head_dim;
        std::memset(ohead, 0, sizeof(float) * head_dim);
        for (int s = 0; s < n_slots; s++) {
            const float w = scores[s] * inv_denom;
            const float * vhead =
                v_cache + ((size_t)s * n_head_kv + kvh) * head_dim;
            for (int i = 0; i < head_dim; i++) ohead[i] += w * vhead[i];
        }
    }
}

static bool append_tensor(std::vector<ggml_tensor *> & tensors,
                          ggml_tensor * t) {
    if (!t || t->data) return true;
    if (std::find(tensors.begin(), tensors.end(), t) == tensors.end()) {
        tensors.push_back(t);
    }
    return true;
}

static bool materialize_mtp_tensors(const std::string & gguf_path,
                                    const Qwen36MtpWeights & weights,
                                    ggml_backend_buffer_type_t target_buft,
                                    ggml_backend_buffer_t & out_buf,
                                    std::string & out_error) {
    std::vector<ggml_tensor *> tensors;
    for (const auto & h : weights.heads) {
        // NextN-specific tensors
        append_tensor(tensors, h.eh_proj);
        append_tensor(tensors, h.enorm);
        append_tensor(tensors, h.hnorm);
        append_tensor(tensors, h.shared_head_norm);
        // shared_head_head can be vocab-sized; leave it unmaterialized so
        // production uses the target's GPU lm_head projection fallback.

        // Shape B: head-owned transformer-block tensors (all 11 required).
        append_tensor(tensors, h.attn_norm);
        append_tensor(tensors, h.attn_q);
        append_tensor(tensors, h.attn_q_norm);
        append_tensor(tensors, h.attn_k);
        append_tensor(tensors, h.attn_k_norm);
        append_tensor(tensors, h.attn_v);
        append_tensor(tensors, h.attn_output);
        append_tensor(tensors, h.post_attention_norm);
        append_tensor(tensors, h.ffn_gate);
        append_tensor(tensors, h.ffn_up);
        append_tensor(tensors, h.ffn_down);
    }
    if (tensors.empty()) return true;

    ggml_backend_buffer_type_t buft = target_buft
        ? target_buft : ggml_backend_cpu_buffer_type();
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t total = 0;
    std::vector<size_t> offsets;
    offsets.reserve(tensors.size());
    for (ggml_tensor * t : tensors) {
        const size_t r = total % alignment;
        if (r != 0) total += alignment - r;
        offsets.push_back(total);
        total += ggml_backend_buft_get_alloc_size(buft, t);
    }

    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, total);
    if (!buf) {
        out_error = "qwen36_mtp: failed to allocate CPU buffer for MTP tensors";
        return false;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    char * base = static_cast<char *>(ggml_backend_buffer_get_base(buf));
    for (size_t i = 0; i < tensors.size(); i++) {
        if (ggml_backend_tensor_alloc(buf, tensors[i], base + offsets[i]) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            out_error = "qwen36_mtp: ggml_backend_tensor_alloc failed";
            return false;
        }
    }

    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx = nullptr;
    gguf_context * gguf = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!gguf) {
        ggml_backend_buffer_free(buf);
        out_error = "qwen36_mtp: gguf_init_from_file failed for " + gguf_path;
        return false;
    }

    dflash::common::GgufMmap mm;
    std::string mmap_error;
    if (!mm.open(gguf_path, mmap_error)) {
        gguf_free(gguf);
        ggml_backend_buffer_free(buf);
        out_error = mmap_error;
        return false;
    }

    const size_t data_start = gguf_get_data_offset(gguf);
    for (ggml_tensor * t : tensors) {
        const char * name = ggml_get_name(t);
        const int64_t tid = gguf_find_tensor(gguf, name);
        if (tid < 0) {
            gguf_free(gguf);
            ggml_backend_buffer_free(buf);
            out_error = std::string("qwen36_mtp: tensor missing from GGUF: ") + name;
            return false;
        }
        const size_t off = data_start + gguf_get_tensor_offset(gguf, tid);
        const size_t sz = gguf_get_tensor_size(gguf, tid);
        if (off + sz > mm.size()) {
            gguf_free(gguf);
            ggml_backend_buffer_free(buf);
            out_error = std::string("qwen36_mtp: tensor overflows GGUF: ") + name;
            return false;
        }
        ggml_backend_tensor_set(t,
                                static_cast<const uint8_t *>(mm.data()) + off,
                                0, sz);
    }

    gguf_free(gguf);
    out_buf = buf;
    return true;
}

static ggml_context * load_gguf_tensor_context(const std::string & gguf_path,
                                               std::string & out_error) {
    ggml_context * ctx = nullptr;
    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx = &ctx;
    gguf_context * gguf = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!gguf || !ctx) {
        if (gguf) gguf_free(gguf);
        out_error = "qwen36_mtp: failed to create GGUF tensor context for " + gguf_path;
        return nullptr;
    }
    gguf_free(gguf);
    return ctx;
}

}  // anonymous namespace

// ── State ─────────────────────────────────────────────────────────────────

// Per-head KV cache buffer.
//   GPU mode: ggml_tensors on the backbone backend with layout
//             [head_dim, n_ctx, n_head_kv] (matches backbone cache_k / cache_v
//             so flash_attn views slice cleanly along dim 1).
//   CPU mode (T1 stubs / no-backend fallback): fp32 host vectors with
//             layout [n_slot * n_head_kv * head_dim] (slot-major).
struct HeadKvBuffer {
    ggml_tensor * k_cache = nullptr;
    ggml_tensor * v_cache = nullptr;
    std::vector<float> k;
    std::vector<float> v;
};

struct Qwen36MtpModule::State {
    Qwen36MtpWeights  weights;
    DFlashTarget *    target    = nullptr;
    ggml_context *    owned_ctx = nullptr;
    ggml_backend_buffer_t mtp_buf = nullptr;
    bool              loaded    = false;
    bool              attached  = false;

    // Phase A bootstrap: last_hidden is zeroed on init / reset_chain.
    // Once Qwen35Backend integration (Phase B) is complete this will
    // carry the backbone's post-norm hidden from the last committed step.
    std::vector<float> last_hidden;   // length == weights.n_embd when loaded

    // set_initial_hidden state: stash pointer + dim from the backbone caller.
    // Consumed by the Shape B TRMBlock forward in PR 2d-bis.
    // Pointer is NOT owned; it must remain valid for the duration of step_batch.
    const float *      initial_hidden_ptr = nullptr;
    int                initial_hidden_dim = 0;

    // Per-head KV cache buffers, sized n_ctx slots × n_head_kv × key/val_length.
    // GPU path: backed by ggml tensors on backbone backend (kv_ctx + kv_buf).
    // warm_head_kv() fills slots [1, n_prompt] post-prefill; each step_batch()
    // call writes its draft K/V at slot (base_pos + h) inside the cgraph.
    std::vector<HeadKvBuffer> head_kv;
    std::vector<int>          head_kv_pos;
    int                       n_ctx     = 0;   // allocated slots per head
    int                       n_ctx_max = 0;   // requested ceiling for lazy-grow

    // GPU-side head_kv tensor lifetimes.
    ggml_context *            kv_ctx = nullptr;
    ggml_backend_buffer_t     kv_buf = nullptr;

    // Bug #5 fix: graphs are shape-only.  Slot write + FA read slots/mask
    // are runtime inputs, so a single graph services every draft_pos for
    // a given (head_idx, fa_window, fused_lm_head, topk_k).  Cap at 4
    // entries — head_kv is single-head in production and target config
    // is stable, so the LRU collapses to 1 entry in practice.
    struct StepGraphKey {
        int  head_idx     = -1;
        int  fa_window    = 0;
        bool fused_lm_head = false;
        int  topk_k       = 0;
    };
    std::array<std::pair<StepGraphKey, std::unique_ptr<Qwen36MtpStepGraph>>, 4> step_sg_cache{};
    // Single scratch graph for the deprecated path (callers that pass the
    // legacy build_qwen36_mtp_step_graph signature with no caching).
    Qwen36MtpStepGraph        step_sg;
    // Cached warmup graph (rebuilt per call because n_tokens varies).
    Qwen36MtpWarmGraph        warm_sg;

    // Per-head top-K logprob emission. K=1 means argmax-only (legacy ABI:
    // StepOutput.topk_* stays empty). K>1 populates the topk surface in
    // every emitted StepOutput. Configured once via set_draft_topk();
    // persists across reset_chain() since the bench/harness toggles K at
    // setup time, not per chain.
    int                       draft_topk = 1;
};

// ── Lazy head_kv capacity growth ──────────────────────────────────────────
//
// At max_ctx=65536 the GPU head_kv tensors are ~256 MiB constant residency
// even for tiny prompts. We instead allocate small at init (default 8192
// slots) and grow on first warm_head_kv when the prompt's true size is
// known. Growth invalidates the step-graph cache and the warm graph
// (their tensor pointers are stale post-realloc); both rebuild lazily on
// the next call. We never shrink — once a daemon serves a long prompt,
// subsequent requests benefit from the warmed-up allocation.
bool Qwen36MtpModule::ensure_head_kv_capacity_(int required_slots) {
    if (required_slots <= state_->n_ctx) return true;
    if (!state_->target) return false;
    ggml_backend_t backend = state_->target->backend();
    if (!backend) return true;  // CPU stub uses host vectors — no grow

    if (required_slots > state_->n_ctx_max) {
        std::fprintf(stderr,
            "[qwen36_mtp] ensure_head_kv_capacity: required=%d > n_ctx_max=%d\n",
            required_slots, state_->n_ctx_max);
        return false;
    }

    // Round up to 1024-slot quantum, cap at n_ctx_max.
    int new_ctx = std::min(((required_slots + 1023) / 1024) * 1024,
                            state_->n_ctx_max);
    if (new_ctx <= state_->n_ctx) return true;

    // Tear down stale graphs first — their cached tensor pointers will be
    // invalid after we free the backing buffer.
    qwen36_mtp_warm_graph_free(state_->warm_sg);
    state_->warm_sg = {};
    for (auto & e : state_->step_sg_cache) {
        e.first  = {};
        e.second.reset();
    }
    if (state_->kv_buf) {
        ggml_backend_buffer_free(state_->kv_buf);
        state_->kv_buf = nullptr;
    }
    if (state_->kv_ctx) {
        ggml_free(state_->kv_ctx);
        state_->kv_ctx = nullptr;
    }
    for (auto & kv : state_->head_kv) {
        kv.k_cache = nullptr;
        kv.v_cache = nullptr;
    }

    const int n_head_kv = state_->weights.n_head_kv;
    const int key_len   = state_->weights.n_key_length;
    const int val_len   = state_->weights.n_value_length;
    const int gamma_max = state_->weights.n_heads;
    const int rb_tensors = 2 * gamma_max;

    ggml_init_params kp{};
    kp.mem_size   = (size_t)(rb_tensors + 16) * ggml_tensor_overhead();
    kp.mem_buffer = nullptr;
    kp.no_alloc   = true;
    state_->kv_ctx = ggml_init(kp);
    if (!state_->kv_ctx) return false;

    for (int h = 0; h < gamma_max; h++) {
        ggml_tensor * k_t = ggml_new_tensor_3d(state_->kv_ctx,
            GGML_TYPE_F16, key_len, new_ctx, n_head_kv);
        ggml_tensor * v_t = ggml_new_tensor_3d(state_->kv_ctx,
            GGML_TYPE_F16, val_len, new_ctx, n_head_kv);
        char nm[64];
        std::snprintf(nm, sizeof(nm), "mtp_head_%d_k", h);
        ggml_set_name(k_t, nm);
        std::snprintf(nm, sizeof(nm), "mtp_head_%d_v", h);
        ggml_set_name(v_t, nm);
        state_->head_kv[h].k_cache = k_t;
        state_->head_kv[h].v_cache = v_t;
    }
    state_->kv_buf = ggml_backend_alloc_ctx_tensors(state_->kv_ctx, backend);
    if (!state_->kv_buf) {
        ggml_free(state_->kv_ctx);
        state_->kv_ctx = nullptr;
        for (auto & kv : state_->head_kv) {
            kv.k_cache = nullptr;
            kv.v_cache = nullptr;
        }
        return false;
    }
    ggml_backend_buffer_clear(state_->kv_buf, 0);
    state_->n_ctx = new_ctx;
    for (auto & p : state_->head_kv_pos) p = 0;

    std::fprintf(stderr,
        "[qwen36_mtp] head_kv grew n_ctx=%d (max=%d)\n",
        new_ctx, state_->n_ctx_max);
    return true;
}

// ── Lifecycle ─────────────────────────────────────────────────────────────

Qwen36MtpModule::Qwen36MtpModule() : state_(std::make_unique<State>()) {}
Qwen36MtpModule::~Qwen36MtpModule() { shutdown(); }

bool Qwen36MtpModule::init(const std::string & gguf_path,
                           DFlashTarget * target,
                           std::string & out_error,
                           int n_ctx_request) {
    shutdown();
    ggml_context * ctx = load_gguf_tensor_context(gguf_path, out_error);
    if (!ctx) {
        return false;
    }
    state_->owned_ctx = ctx;
    if (!init(gguf_path, ctx, target, out_error, n_ctx_request)) {
        shutdown();
        return false;
    }
    return true;
}

bool Qwen36MtpModule::init(const std::string & gguf_path,
                           ggml_context * ctx,
                           DFlashTarget * target,
                           std::string & out_error,
                           int n_ctx_request) {
    if (!target) {
        out_error = "Qwen36MtpModule::init: target is null";
        return false;
    }
    if (!ctx) {
        out_error = "Qwen36MtpModule::init: ctx is null";
        return false;
    }
    if (!state_->owned_ctx) {
        shutdown();
    }
    const bool ok = load_qwen36_mtp_weights(
        gguf_path, ctx,
        /*expected_n_embd=*/target->hidden_size(),
        /*expected_n_vocab=*/0,
        state_->weights, out_error);

    if (!ok) return false;
    ggml_backend_t tgt_backend = target ? target->backend() : nullptr;
    ggml_backend_buffer_type_t tgt_buft = tgt_backend
        ? ggml_backend_get_default_buffer_type(tgt_backend)
        : ggml_backend_cpu_buffer_type();
    if (!materialize_mtp_tensors(gguf_path, state_->weights, tgt_buft,
                                  state_->mtp_buf, out_error)) {
        state_->weights = {};
        return false;
    }

    // Hard contract — current GPU paths (warm_head_kv, step_batch_gpu_,
    // step_chain) hardcode h=0 and chain only against the first NextN head.
    // A GGUF with n_heads>1 (e.g. DeepSeek-V3 NextN can ship up to 3) would
    // silently produce wrong drafts: head_kv is allocated for all heads but
    // only head 0 is warmed/written.  Fail loudly until the multi-head path
    // is built and tested.  Per momus review, "the one thing nobody checked".
    if (state_->weights.n_heads > 1) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "Qwen36MtpModule::init: n_heads=%d (>1) is not supported. "
            "warm_head_kv only initializes head 0 and step_chain reuses "
            "head 0 across iters; multi-head GGUFs would silently produce "
            "wrong drafts. See momus review.",
            state_->weights.n_heads);
        out_error = buf;
        state_->weights = {};
        return false;
    }

    // Per-head KV cache allocation.  Two modes:
    //   - GPU mode (default when target has a backend): allocate ggml tensors
    //     [head_dim, n_ctx, n_head_kv] on backbone backend so the step cgraph
    //     can write/read KV in-place via ggml_cpy and ggml_view_3d.
    //   - CPU mode (no backend, e.g. T1 tests): fp32 host vectors only.
    {
        const int gamma_max = state_->weights.n_heads;
        const int n_head_kv = state_->weights.n_head_kv;
        const int key_len   = state_->weights.n_key_length;
        const int val_len   = state_->weights.n_value_length;
        // Two knobs:
        //   n_ctx_max — hard ceiling we'll ever grow to. Defaults to
        //               n_ctx_request (= backbone max_ctx). The chain runner
        //               can NEVER index past this.
        //   n_ctx     — initial allocation. Defaults to min(8192, n_ctx_max)
        //               so a daemon configured for --max-ctx 65536 doesn't
        //               eat 256 MiB of VRAM up front when short prompts are
        //               the norm. Env DFLASH27B_MTP_INITIAL_CTX overrides.
        int n_ctx_max = 8192;
        if (const char * s = std::getenv("DFLASH27B_MTP_CTX")) {
            const int v = std::atoi(s);
            if (v > 0) n_ctx_max = v;
        }
        if (n_ctx_request > 0) n_ctx_max = n_ctx_request;

        int n_ctx = std::min(8192, n_ctx_max);
        if (const char * s = std::getenv("DFLASH27B_MTP_INITIAL_CTX")) {
            const int v = std::atoi(s);
            if (v > 0) n_ctx = std::min(v, n_ctx_max);
        }
        state_->n_ctx     = n_ctx;
        state_->n_ctx_max = n_ctx_max;

        if (n_head_kv > 0 && key_len > 0 && val_len > 0 && gamma_max > 0) {
            state_->head_kv.resize(gamma_max);
            state_->head_kv_pos.assign(gamma_max, 0);
            // The CPU forward path (T1 stub tests, no backend) reads/writes
            // the host vectors; the GPU path reads/writes the backend tensors.
            // Allocate only the side we will actually use.
            if (!tgt_backend) {
                for (int h = 0; h < gamma_max; h++) {
                    state_->head_kv[h].k.assign(
                        (size_t)n_ctx * n_head_kv * key_len, 0.0f);
                    state_->head_kv[h].v.assign(
                        (size_t)n_ctx * n_head_kv * val_len, 0.0f);
                }
            } else {
                const int rb_tensors = 2 * gamma_max;
                ggml_init_params kp{};
                kp.mem_size   = (size_t)(rb_tensors + 16) * ggml_tensor_overhead();
                kp.mem_buffer = nullptr;
                kp.no_alloc   = true;
                state_->kv_ctx = ggml_init(kp);
                if (!state_->kv_ctx) {
                    out_error = "qwen36_mtp: head_kv ggml_init failed";
                    return false;
                }
                for (int h = 0; h < gamma_max; h++) {
                    // Head KV stored as F16 on device (Phase B+).  CUDA
                    // ggml_flash_attn_ext takes F16 K/V natively (fattn.cu
                    // accepts F16/BF16/quant K/V; F32 K/V are auto-cast
                    // up-front) and ggml_cpy F32 -> F16 is supported inside
                    // the step graph for the per-step K/V write.  Saves 50%
                    // of the per-head KV footprint and matches the backbone
                    // cache_k/cache_v dtype.
                    ggml_tensor * k_t = ggml_new_tensor_3d(state_->kv_ctx,
                        GGML_TYPE_F16, key_len, n_ctx, n_head_kv);
                    ggml_tensor * v_t = ggml_new_tensor_3d(state_->kv_ctx,
                        GGML_TYPE_F16, val_len, n_ctx, n_head_kv);
                    char name[64];
                    std::snprintf(name, sizeof(name), "mtp_head_%d_k", h);
                    ggml_set_name(k_t, name);
                    std::snprintf(name, sizeof(name), "mtp_head_%d_v", h);
                    ggml_set_name(v_t, name);
                    state_->head_kv[h].k_cache = k_t;
                    state_->head_kv[h].v_cache = v_t;
                }
                state_->kv_buf = ggml_backend_alloc_ctx_tensors(
                    state_->kv_ctx, tgt_backend);
                if (!state_->kv_buf) {
                    ggml_free(state_->kv_ctx);
                    state_->kv_ctx = nullptr;
                    state_->head_kv.clear();
                    out_error = "qwen36_mtp: ggml_backend_alloc_ctx_tensors for head_kv failed";
                    return false;
                }
                ggml_backend_buffer_clear(state_->kv_buf, 0);
            }
        }
    }

    state_->loaded = true;
    // Zero the Phase A bootstrap hidden.
    state_->last_hidden.assign(state_->weights.n_embd, 0.0f);
    // Clear initial_hidden state for this init.
    state_->initial_hidden_ptr = nullptr;
    state_->initial_hidden_dim = 0;
    return attach(target);
}

int  Qwen36MtpModule::max_gamma()   const {
    // Post-Phase-A semantics: max_gamma is the autoregressive CHAIN depth ceiling,
    // not the physical NextN head count. We re-feed the single head's own
    // post-shared_head_norm hidden as h_prev to extend the chain to arbitrary depth
    // (oracle blocker 5.6 analysis). Capped at 8 to match Unsloth's --spec-draft-n-max
    // ceiling and keep the head_kv slot writes within n_ctx=8192. Returns 0 pre-init
    // so the basic contract test (max_gamma()==0 before init) still holds.
    if (!state_->loaded) return 0;
    return 8;
}
int  Qwen36MtpModule::hidden_size() const { return state_->weights.n_embd; }
int  Qwen36MtpModule::num_heads()   const { return state_->weights.n_heads; }

bool Qwen36MtpModule::attach(DFlashTarget * target) {
    if (!target) return false;
    if (state_->loaded && target->hidden_size() != state_->weights.n_embd) {
        std::fprintf(stderr,
            "[qwen36_mtp] hidden_size mismatch (target=%d, mtp=%d)\n",
            target->hidden_size(), state_->weights.n_embd);
        return false;
    }
    state_->target   = target;
    state_->attached = true;
    // The MTP forward needs per-position post-norm hiddens for warmup +
    // chain-iter h_prev lookup; signal that to the target so it captures
    // them.  Non-MTP-bound targets pay nothing.
    if (auto * t = dynamic_cast<Qwen35DFlashTarget *>(target)) {
        t->enable_hidden_seq_capture(true);
    }
    return true;
}

void Qwen36MtpModule::reset_chain() {
    if (state_->loaded) {
        std::fill(state_->last_hidden.begin(), state_->last_hidden.end(), 0.0f);
    }
    state_->initial_hidden_ptr = nullptr;
    state_->initial_hidden_dim = 0;
}

void Qwen36MtpModule::set_draft_topk(int k) {
    state_->draft_topk = (k >= 1) ? k : 1;
}

void Qwen36MtpModule::shutdown() {
    for (auto & e : state_->step_sg_cache) {
        if (e.second) qwen36_mtp_step_graph_free(*e.second);
        e.second.reset();
        e.first = State::StepGraphKey{};
    }
    qwen36_mtp_step_graph_free(state_->step_sg);
    qwen36_mtp_warm_graph_free(state_->warm_sg);
    if (state_->kv_buf) {
        ggml_backend_buffer_free(state_->kv_buf);
        state_->kv_buf = nullptr;
    }
    if (state_->kv_ctx) {
        ggml_free(state_->kv_ctx);
        state_->kv_ctx = nullptr;
    }
    // Per-head KV CPU mirrors are std::vector<float> — destructors free them.
    state_->head_kv.clear();
    state_->head_kv_pos.clear();
    state_->n_ctx = 0;

    if (state_->mtp_buf) {
        ggml_backend_buffer_free(state_->mtp_buf);
        state_->mtp_buf = nullptr;
    }
    if (state_->owned_ctx) {
        ggml_free(state_->owned_ctx);
        state_->owned_ctx = nullptr;
    }
    state_->target             = nullptr;
    state_->attached           = false;
    state_->loaded             = false;
    state_->weights            = {};
    state_->last_hidden.clear();
    state_->initial_hidden_ptr = nullptr;
    state_->initial_hidden_dim = 0;
}

// ── Shape B forward ───────────────────────────────────────────────────────
//
// For each MTP head h ∈ [0, n_heads):
//
//   Step A: h_prev = initial_hidden (k=0) or last_hidden (k>0)
//   Step B: embed cur/drafted token
//   Step C: Eq 21 — x = eh_proj @ [hnorm(h_prev); enorm(embed)]
//   Step D: Eq 22 — TRMBlock_k (head-owned attn + FFN) applied to x
//   Step E: Eq 23 — shared_head_norm + shared LM head → draft token
//
// TRMBlock_k uses head-owned tensors (NOT backbone). attn_q is packed Q+gate
// [n_embd, 2*(head_count*key_length)]; gate is sigmoid-multiplied into attn out.
// RoPE: standard (not M-RoPE) at n_rot=rope_dimension_count, theta=1e7.
// KV: single-slot, attending only to the new K/V (trivial for γ_max=1).
//
// When MTP_PHASE_A_FALLBACK is defined, skips the TRMBlock (Phase A path).

bool Qwen36MtpModule::step_batch(int32_t current_token,
                                 int base_pos,
                                 std::vector<StepOutput> & out) {
    // Guard: module must be loaded and attached.
    if (!state_->loaded || !state_->attached) {
        out.clear();
        return false;
    }

    // GPU path runs whenever a CUDA backend is bound; the CPU forward below
    // is reserved for T1 stub tests (attach_weights_for_test, no backend).
    if (state_->kv_ctx && state_->target && state_->target->backend()) {
        return step_batch_gpu_(current_token, base_pos, out);
    }

    const int n_embd     = state_->weights.n_embd;
    const int n_vocab    = state_->weights.n_vocab;
    const int n_heads    = state_->weights.n_heads;
    const int n_head     = state_->weights.n_head_count;
    const int n_head_kv  = state_->weights.n_head_kv;
    const int key_len    = state_->weights.n_key_length;
    const int val_len    = state_->weights.n_value_length;
    const int ffn_len    = state_->weights.n_ffn_length;

    // Packed Q layout: attn_q has 2*q_dim columns (Q || gate).
    const int q_dim  = n_head * key_len;    // 24*256 = 6144 for 27B
    const int kv_dim = n_head_kv * key_len; // 4*256  = 1024
    const int v_total = n_head_kv * val_len;

    // RoPE params (Qwen3.6-27B constants from GGUF metadata).
    // For the 27B GGUF: rope_dimension_count=64, rope_theta=1e7.
    // We use these constants directly since they're part of the verified GGUF
    // contract (qwen36_mtp_redesign.md §Verified GGUF Constants).
    // For text-mode MROPE, the 3 active axes share the same position, so
    // it reduces to NeoX RoPE with n_rot=rope.dimension_count=64.  The CPU
    // fallback path uses plain rope_cpu(); the GPU graph calls ggml_rope_multi
    // with the real sections so it stays correct for multi-axis modes.
    const int   rope_n_rot   = std::min(64, key_len);
    const float rope_theta   = 1e7f;

    out.clear();
    out.reserve(n_heads);

    // Working buffers.
    std::vector<float> embed_buf(n_embd);
    std::vector<float> e_in(n_embd);
    std::vector<float> h_in(n_embd);
    std::vector<float> concat_buf(2 * n_embd);
    std::vector<float> x(n_embd);
    std::vector<float> x_normed(n_embd);

    // Per-head tensor float caches (dequantized from ggml tensors).
    std::vector<float> enorm_data;
    std::vector<float> hnorm_data;
    std::vector<float> eh_proj_data;
    std::vector<float> shared_head_norm_data;
    std::vector<float> shared_head_head_data;

#ifndef MTP_PHASE_A_FALLBACK
    std::vector<float> attn_norm_data;
    std::vector<float> attn_q_data;    // packed: [(Q||gate) x n_embd]
    std::vector<float> attn_q_norm_data;
    std::vector<float> attn_k_data;
    std::vector<float> attn_k_norm_data;
    std::vector<float> attn_v_data;
    std::vector<float> attn_output_data;
    std::vector<float> post_attn_norm_data;
    std::vector<float> ffn_gate_data;
    std::vector<float> ffn_up_data;
    std::vector<float> ffn_down_data;

    // TRMBlock working buffers (sized for worst case; reused across heads).
    std::vector<float> q_buf(2 * q_dim);  // packed Q+gate from projection
    std::vector<float> k_buf(kv_dim);
    std::vector<float> v_buf(v_total);
    std::vector<float> attn_out_buf(n_head * val_len);
    std::vector<float> proj_buf(n_embd);
    std::vector<float> ffn_gate_buf(ffn_len);
    std::vector<float> ffn_up_buf(ffn_len);
    std::vector<float> ffn_in_buf(ffn_len);
    std::vector<float> ffn_out(n_embd);
#endif  // MTP_PHASE_A_FALLBACK

    std::vector<float> logits_buf;
    int32_t cur_token = current_token;

    for (int h = 0; h < n_heads; h++) {
        const auto & head = state_->weights.heads[h];

        // h=0 reads h_{base_pos-1} from the target; h>0 chains use the
        // previous head's un-normed output stashed in state_->last_hidden.
        const float * h_prev = state_->last_hidden.data();
        if (h == 0) {
            bool found_hidden = false;
            if (state_->target) {
                if (base_pos >= 1) {
                    // Prefer pre-output-norm (PR #22673 t_h_pre_norm) so
                    // the head's hnorm doesn't double-normalise.  Fall
                    // back to post-norm if the adapter did not capture
                    // the pre-norm sequence this verify_batch.
                    const float * tgt_h =
                        state_->target->hidden_at_pos_pre_norm(base_pos - 1);
                    if (!tgt_h) {
                        tgt_h = state_->target->hidden_at_pos(base_pos - 1);
                    }
                    if (tgt_h) {
                        h_prev = tgt_h;
                        found_hidden = true;
                    }
                }
                if (!found_hidden) {
                    const float * tgt_h = state_->target->last_hidden();
                    if (tgt_h) {
                        h_prev = tgt_h;
                        found_hidden = true;
                    }
                }
            }
            if (!found_hidden && state_->initial_hidden_ptr &&
                state_->initial_hidden_dim == n_embd) {
                h_prev = state_->initial_hidden_ptr;
                found_hidden = true;
            }
            if (!found_hidden) {
                std::fprintf(stderr,
                    "[qwen36_mtp] step_batch: no live hidden available at base_pos=%d; "
                    "using zero vector for h_prev\n", base_pos);
            }
        }

        // ── Step B: embed current/drafted token ────────────────────────
        const int32_t tok_ids[1] = { cur_token };
        if (!state_->target->embed_tokens(tok_ids, 1, embed_buf.data())) {
            std::fprintf(stderr,
                "[qwen36_mtp] step_batch: embed_tokens failed for head %d\n", h);
            out.clear();
            return false;
        }

        // ── Step C: Eq 21 — eh_proj([hnorm(h_prev); enorm(embed)]) ───

        // Load enorm and hnorm
        if (!tensor_to_floats(head.enorm, enorm_data) ||
            (int)enorm_data.size() != n_embd) {
            std::fprintf(stderr,
                "[qwen36_mtp] step_batch: invalid enorm at head %d\n", h);
            out.clear();
            return false;
        }
        if (!tensor_to_floats(head.hnorm, hnorm_data) ||
            (int)hnorm_data.size() != n_embd) {
            std::fprintf(stderr,
                "[qwen36_mtp] step_batch: invalid hnorm at head %d\n", h);
            out.clear();
            return false;
        }

        rmsnorm_cpu(embed_buf.data(), enorm_data.data(), e_in.data(), n_embd);
        rmsnorm_cpu(h_prev,           hnorm_data.data(), h_in.data(), n_embd);

        // Concat order [e_in; h_in] (embed first, hidden second) matches the
        // reference llama.cpp PR #22673 graph_mtp:
        //   `ggml_concat(ctx0, e_norm, h_norm, /*dim=*/0)`.
        // The earlier "hidden first" claim in qwen36_mtp_redesign.md was wrong.
        std::copy(e_in.begin(), e_in.end(), concat_buf.begin());
        std::copy(h_in.begin(), h_in.end(), concat_buf.begin() + n_embd);

        // Project: x = eh_proj @ concat, shape [n_embd, 2*n_embd] × [2*n_embd] → [n_embd]
        if (!tensor_to_floats(head.eh_proj, eh_proj_data) ||
            (int)eh_proj_data.size() != n_embd * 2 * n_embd) {
            std::fprintf(stderr,
                "[qwen36_mtp] step_batch: invalid eh_proj at head %d "
                "(got %zu, expected %d)\n",
                h, eh_proj_data.size(), n_embd * 2 * n_embd);
            out.clear();
            return false;
        }
        matvec_cpu(eh_proj_data.data(), concat_buf.data(), x.data(),
                   n_embd, 2 * n_embd);

#ifndef MTP_PHASE_A_FALLBACK
        // ── Step D: Eq 22 — TRMBlock_k (head-owned attn + FFN) ────────
        // All required tensors must be non-null (validated by loader).

        // Pre-attn norm: cur = RMSNorm(x, attn_norm)
        if (!tensor_to_floats(head.attn_norm, attn_norm_data) ||
            (int)attn_norm_data.size() != n_embd) {
            std::fprintf(stderr,
                "[qwen36_mtp] step_batch: invalid attn_norm at head %d\n", h);
            out.clear();
            return false;
        }
        // We'll use x_normed as the pre-attn cur.
        rmsnorm_cpu(x.data(), attn_norm_data.data(), x_normed.data(), n_embd);

        // Q projection (packed Q||gate): [2*q_dim, n_embd] × [n_embd] → [2*q_dim]
        // ggml convention: tensor shape [cols, rows], stored row-major (rows × cols).
        // attn_q is [n_embd, 2*q_dim] in ggml's ne[] → rows=2*q_dim, cols=n_embd.
        if (!tensor_to_floats(head.attn_q, attn_q_data) ||
            (int)attn_q_data.size() != 2 * q_dim * n_embd) {
            std::fprintf(stderr,
                "[qwen36_mtp] step_batch: invalid attn_q at head %d "
                "(got %zu, expected %d)\n",
                h, attn_q_data.size(), 2 * q_dim * n_embd);
            out.clear();
            return false;
        }
        // attn_q output is laid out [Q_head_0 | gate_head_0 | Q_head_1 | gate_head_1 | ...]
        // per the backbone graph at qwen35_target_graph.cpp:468-486: QG is
        // reshaped to [head_dim*2, n_head] and viewed as Q (offset 0) + gate
        // (offset head_dim) with stride head_dim*2 between heads.  We must
        // de-interleave into contiguous Q and gate buffers before QK-norm /
        // RoPE / attention (which all assume [n_head, head_dim] layout).
        std::vector<float> q_raw(2 * q_dim);
        matvec_cpu(attn_q_data.data(), x_normed.data(), q_raw.data(),
                   2 * q_dim, n_embd);
        std::vector<float> gate_data(q_dim);
        q_buf.resize(q_dim);
        for (int hd = 0; hd < n_head; hd++) {
            const float * src = q_raw.data() + (size_t)hd * 2 * key_len;
            std::memcpy(q_buf.data() + (size_t)hd * key_len, src,
                        sizeof(float) * key_len);
            std::memcpy(gate_data.data() + (size_t)hd * key_len,
                        src + key_len, sizeof(float) * key_len);
        }
        float * q_ptr    = q_buf.data();
        float * gate_ptr = gate_data.data();

        // K projection: [kv_dim, n_embd] × [n_embd] → [kv_dim]
        if (!tensor_to_floats(head.attn_k, attn_k_data) ||
            (int)attn_k_data.size() != kv_dim * n_embd) {
            std::fprintf(stderr,
                "[qwen36_mtp] step_batch: invalid attn_k at head %d\n", h);
            out.clear();
            return false;
        }
        k_buf.resize(kv_dim);
        matvec_cpu(attn_k_data.data(), x_normed.data(), k_buf.data(),
                   kv_dim, n_embd);

        // V projection: [v_total, n_embd] × [n_embd] → [v_total]
        if (!tensor_to_floats(head.attn_v, attn_v_data) ||
            (int)attn_v_data.size() != v_total * n_embd) {
            std::fprintf(stderr,
                "[qwen36_mtp] step_batch: invalid attn_v at head %d\n", h);
            out.clear();
            return false;
        }
        v_buf.resize(v_total);
        matvec_cpu(attn_v_data.data(), x_normed.data(), v_buf.data(),
                   v_total, n_embd);

        // QK-norm (per-head RMSNorm on Q and K)
        if (!tensor_to_floats(head.attn_q_norm, attn_q_norm_data) ||
            (int)attn_q_norm_data.size() != key_len) {
            std::fprintf(stderr,
                "[qwen36_mtp] step_batch: invalid attn_q_norm at head %d\n", h);
            out.clear();
            return false;
        }
        if (!tensor_to_floats(head.attn_k_norm, attn_k_norm_data) ||
            (int)attn_k_norm_data.size() != key_len) {
            std::fprintf(stderr,
                "[qwen36_mtp] step_batch: invalid attn_k_norm at head %d\n", h);
            out.clear();
            return false;
        }
        per_head_rmsnorm(q_ptr, attn_q_norm_data.data(), n_head,    key_len);
        per_head_rmsnorm(k_buf.data(), attn_k_norm_data.data(), n_head_kv, key_len);

        const int draft_pos = base_pos + h;
        rope_cpu(q_ptr,      n_head,    key_len, rope_n_rot, draft_pos, rope_theta);
        rope_cpu(k_buf.data(), n_head_kv, key_len, rope_n_rot, draft_pos, rope_theta);
        if (draft_pos >= state_->n_ctx ||
            (int)state_->head_kv.size() <= h) {
            std::fprintf(stderr,
                "[qwen36_mtp] step_batch: draft_pos %d out of head_kv range "
                "(n_ctx=%d, head=%d, head_kv_size=%zu)\n",
                draft_pos, state_->n_ctx, h, state_->head_kv.size());
            out.clear();
            return false;
        }
        {
            auto & kv = state_->head_kv[h];
            const size_t k_slot_off = (size_t)draft_pos * n_head_kv * key_len;
            const size_t v_slot_off = (size_t)draft_pos * n_head_kv * val_len;
            std::memcpy(kv.k.data() + k_slot_off, k_buf.data(),
                        sizeof(float) * (size_t)n_head_kv * key_len);
            std::memcpy(kv.v.data() + v_slot_off, v_buf.data(),
                        sizeof(float) * (size_t)n_head_kv * val_len);
        }
        if ((int)state_->head_kv_pos.size() > h) {
            state_->head_kv_pos[h] = std::max(state_->head_kv_pos[h], draft_pos);
        }

        // Range attention over slots [0, draft_pos] of head_kv[h] (causal).
        attn_out_buf.resize(n_head * val_len);
        range_attention(q_ptr,
                        state_->head_kv[h].k.data(),
                        state_->head_kv[h].v.data(),
                        attn_out_buf.data(),
                        n_head, n_head_kv, val_len,
                        /*n_slots=*/draft_pos + 1);

        // Reshape attn output: [n_head * val_len] = [q_dim] (val_len == key_len).
        // Apply sigmoid gate from the packed Q.
        for (int i = 0; i < q_dim; i++) {
            const float g = 1.0f / (1.0f + std::exp(-gate_ptr[i]));
            attn_out_buf[i] *= g;
        }

        // attn_output projection: [n_embd, q_dim] × [q_dim] → [n_embd]
        // attn_output tensor is [head_count*value_length, n_embd] = [q_dim, n_embd]
        // in ggml convention: ne[0]=q_dim, ne[1]=n_embd → rows=n_embd, cols=q_dim.
        if (!tensor_to_floats(head.attn_output, attn_output_data) ||
            (int)attn_output_data.size() != n_embd * q_dim) {
            std::fprintf(stderr,
                "[qwen36_mtp] step_batch: invalid attn_output at head %d "
                "(got %zu, expected %d)\n",
                h, attn_output_data.size(), n_embd * q_dim);
            out.clear();
            return false;
        }
        proj_buf.resize(n_embd);
        matvec_cpu(attn_output_data.data(), attn_out_buf.data(), proj_buf.data(),
                   n_embd, q_dim);

        // Residual: x = x + attn_proj
        for (int i = 0; i < n_embd; i++) x[i] += proj_buf[i];

        // Pre-FFN norm: cur = RMSNorm(x, post_attention_norm)
        if (!tensor_to_floats(head.post_attention_norm, post_attn_norm_data) ||
            (int)post_attn_norm_data.size() != n_embd) {
            std::fprintf(stderr,
                "[qwen36_mtp] step_batch: invalid post_attention_norm at head %d\n", h);
            out.clear();
            return false;
        }
        rmsnorm_cpu(x.data(), post_attn_norm_data.data(), x_normed.data(), n_embd);

        // SwiGLU FFN: ffn_out = ffn_down @ (silu(ffn_gate @ x_n) * (ffn_up @ x_n))
        if (!tensor_to_floats(head.ffn_gate, ffn_gate_data) ||
            (int)ffn_gate_data.size() != ffn_len * n_embd) {
            std::fprintf(stderr,
                "[qwen36_mtp] step_batch: invalid ffn_gate at head %d\n", h);
            out.clear();
            return false;
        }
        if (!tensor_to_floats(head.ffn_up, ffn_up_data) ||
            (int)ffn_up_data.size() != ffn_len * n_embd) {
            std::fprintf(stderr,
                "[qwen36_mtp] step_batch: invalid ffn_up at head %d\n", h);
            out.clear();
            return false;
        }
        if (!tensor_to_floats(head.ffn_down, ffn_down_data) ||
            (int)ffn_down_data.size() != n_embd * ffn_len) {
            std::fprintf(stderr,
                "[qwen36_mtp] step_batch: invalid ffn_down at head %d\n", h);
            out.clear();
            return false;
        }

        ffn_gate_buf.resize(ffn_len);
        ffn_up_buf.resize(ffn_len);
        ffn_in_buf.resize(ffn_len);
        matvec_cpu(ffn_gate_data.data(), x_normed.data(), ffn_gate_buf.data(),
                   ffn_len, n_embd);
        matvec_cpu(ffn_up_data.data(), x_normed.data(), ffn_up_buf.data(),
                   ffn_len, n_embd);
        for (int i = 0; i < ffn_len; i++) {
            ffn_in_buf[i] = silu(ffn_gate_buf[i]) * ffn_up_buf[i];
        }
        ffn_out.resize(n_embd);
        matvec_cpu(ffn_down_data.data(), ffn_in_buf.data(), ffn_out.data(),
                   n_embd, ffn_len);

        // Residual: x = x + ffn_out
        for (int i = 0; i < n_embd; i++) x[i] += ffn_out[i];
#endif  // MTP_PHASE_A_FALLBACK

        // ── Step E: Eq 23 — shared_head_norm + shared LM head ─────────

        if (head.shared_head_norm &&
            tensor_to_floats(head.shared_head_norm, shared_head_norm_data) &&
            (int)shared_head_norm_data.size() == n_embd) {
            rmsnorm_cpu(x.data(), shared_head_norm_data.data(), x_normed.data(), n_embd);
        } else {
            // Fallback: treat as unit weights (norm without scale).
            float ss = 0.0f;
            for (int i = 0; i < n_embd; i++) ss += x[i] * x[i];
            const float rms_inv = 1.0f / std::sqrt(ss / n_embd + 1e-6f);
            for (int i = 0; i < n_embd; i++) x_normed[i] = x[i] * rms_inv;
        }

        // LM head projection → draft token
        StepOutput step_out;
        if (head.shared_head_head &&
            tensor_to_floats(head.shared_head_head, shared_head_head_data) &&
            (int)shared_head_head_data.size() == n_vocab * n_embd) {
            // Explicit per-head LM head (absent in 27B GGUF): [n_vocab x n_embd]
            logits_buf.resize(n_vocab);
            matvec_cpu(shared_head_head_data.data(), x_normed.data(),
                       logits_buf.data(), n_vocab, n_embd);
            step_out.draft_token = argmax(logits_buf.data(), n_vocab);
            step_out.draft_logit = logits_buf[step_out.draft_token];
            if (state_->draft_topk > 1) {
                emit_topk_logprobs(logits_buf.data(), n_vocab,
                                   state_->draft_topk, step_out);
            }
        } else {
            // Standard path: shared LM head via target's project_hidden_to_*.
            // For top-K (draft_topk > 1) prefer project_hidden_to_logits so we
            // can populate the top-K logprob surface; fall back to the argmax
            // path for K=1 or when the target lacks the logits virtual.
            if (state_->draft_topk > 1) {
                std::vector<float> logits_buf_t;
                int vocab = 0;
                if (state_->target->project_hidden_to_logits(x_normed.data(), 1,
                                                              logits_buf_t, vocab)
                    && vocab > 0) {
                    step_out.draft_token = argmax(logits_buf_t.data(), vocab);
                    step_out.draft_logit = logits_buf_t[step_out.draft_token];
                    emit_topk_logprobs(logits_buf_t.data(), vocab,
                                       state_->draft_topk, step_out);
                } else {
                    std::vector<int32_t> tok_out;
                    if (!state_->target->project_hidden_to_tokens(x_normed.data(), 1, tok_out)
                        || tok_out.empty()) {
                        std::fprintf(stderr,
                            "[qwen36_mtp] step_batch: project_hidden_to_tokens failed at head %d\n", h);
                        out.clear();
                        return false;
                    }
                    step_out.draft_token = tok_out[0];
                    step_out.draft_logit = 0.0f;
                    static bool warned = false;
                    if (!warned) {
                        std::fprintf(stderr,
                            "[qwen36_mtp] step_batch: draft_topk=%d requested but target "
                            "lacks project_hidden_to_logits; emitting argmax only.\n",
                            state_->draft_topk);
                        warned = true;
                    }
                }
            } else {
                std::vector<int32_t> tok_out;
                if (!state_->target->project_hidden_to_tokens(x_normed.data(), 1, tok_out)
                    || tok_out.empty()) {
                    std::fprintf(stderr,
                        "[qwen36_mtp] step_batch: project_hidden_to_tokens failed at head %d\n", h);
                    out.clear();
                    return false;
                }
                step_out.draft_token = tok_out[0];
                step_out.draft_logit = 0.0f;
            }
        }

        out.push_back(std::move(step_out));

        // ── Update chain state for next head ──────────────────────────
        // Stash post-residual, pre-shared_head_norm hidden (Eq 22's output).
        state_->last_hidden = x;
        cur_token = out.back().draft_token;
    }

    return true;
}

const Qwen36MtpWeights & Qwen36MtpModule::weights() const {
    return state_->weights;
}

void Qwen36MtpModule::attach_weights_for_test(const Qwen36MtpWeights & w) {
    state_->weights     = w;
    state_->loaded      = true;
    state_->last_hidden.assign(w.n_embd, 0.0f);
    state_->initial_hidden_ptr = nullptr;
    state_->initial_hidden_dim = 0;
    // Allocate head_kv so step_batch's range attention can read/write slots.
    // Tests typically exercise only a few positions; n_ctx=64 is plenty.
    const int n_ctx     = 64;
    const int n_head_kv = w.n_head_kv;
    const int key_len   = w.n_key_length;
    const int val_len   = w.n_value_length;
    state_->n_ctx       = n_ctx;
    state_->head_kv.clear();
    state_->head_kv_pos.clear();
    if (n_head_kv > 0 && key_len > 0 && val_len > 0 && w.n_heads > 0) {
        state_->head_kv.resize(w.n_heads);
        state_->head_kv_pos.assign(w.n_heads, 0);
        for (int h = 0; h < w.n_heads; h++) {
            state_->head_kv[h].k.assign(
                (size_t)n_ctx * n_head_kv * key_len, 0.0f);
            state_->head_kv[h].v.assign(
                (size_t)n_ctx * n_head_kv * val_len, 0.0f);
        }
    }
}

void Qwen36MtpModule::set_initial_hidden(const float * h_prev, int dim) {
    // Stash caller's pointer + dim. The pointer must remain valid for the
    // duration of the next step_batch() call. In PR 2c-bis this is a no-op
    // store; the Shape B TRMBlock forward in PR 2d-bis reads it.
    state_->initial_hidden_ptr = h_prev;
    state_->initial_hidden_dim = dim;
}

// R2 thin snapshot — GPU path captures only [0..head_kv_pos+1] slots instead
// of the full [key_len, n_ctx, n_head_kv] tensor. At max_ctx=65536 this drops
// per-snapshot payload from ~256 MiB to ~(N+1) × n_head_kv × (key+val) × 2 B,
// typically <20 MiB for a 4K-token cut. Layout per MTP head:
//
//   [int32 slots]            // = head_kv_pos[h] + 1 (slot 0 included)
//   [int32 n_head_kv]        // shape header for restore-side validation
//   [int32 key_len]
//   [int32 val_len]
//   [K block]                // n_head_kv × slots × key_len × F16, ordered
//                            // by ggml dim-2 (the "head_kv group" axis)
//   [V block]                // n_head_kv × slots × val_len × F16
//
// CPU-stub path (tests, kv.k_cache == nullptr) keeps the legacy full byte
// copy so the existing test_prefix_cache_mtp round-trips byte-equal.
namespace {

constexpr size_t kThinSnapHeaderBytes = 4 * sizeof(int32_t);

size_t thin_per_head_bytes(int slots, int n_head_kv, int key_len, int val_len) {
    const size_t k = (size_t)n_head_kv * (size_t)slots * (size_t)key_len * sizeof(uint16_t);
    const size_t v = (size_t)n_head_kv * (size_t)slots * (size_t)val_len * sizeof(uint16_t);
    return kThinSnapHeaderBytes + k + v;
}

}  // namespace

bool Qwen36MtpModule::snapshot_head_kv(std::vector<std::vector<uint8_t>> & out_buf,
                                       std::vector<int> & out_pos) const {
    out_buf.clear();
    out_pos.clear();
    if (!state_->loaded || state_->head_kv.empty() ||
        state_->head_kv_pos.size() != state_->head_kv.size()) {
        return false;
    }

    const int n_head_kv = state_->weights.n_head_kv;
    const int key_len   = state_->weights.n_key_length;
    const int val_len   = state_->weights.n_value_length;
    const int n_ctx     = state_->n_ctx;

    // R4 dtype guard.
    for (const auto & kv : state_->head_kv) {
        if (kv.k_cache && kv.k_cache->type != GGML_TYPE_F16) return false;
        if (kv.v_cache && kv.v_cache->type != GGML_TYPE_F16) return false;
    }

    // Compute total payload up-front against the size cap.
    size_t total = 0;
    for (size_t h = 0; h < state_->head_kv.size(); ++h) {
        const auto & kv = state_->head_kv[h];
        if (kv.k_cache) {
            const int slots = std::min(state_->head_kv_pos[h] + 1, n_ctx);
            if (slots <= 0) return false;
            total += thin_per_head_bytes(slots, n_head_kv, key_len, val_len);
        } else {
            const size_t k_bytes = kv.k.size() * sizeof(float);
            const size_t v_bytes = kv.v.size() * sizeof(float);
            if (k_bytes == 0 || v_bytes == 0) return false;
            total += k_bytes + v_bytes;
        }
    }
    if (total > kMtpHeadKvSnapshotMaxBytes) {
        std::fprintf(stderr,
            "[qwen36_mtp] snapshot_head_kv: refusing %.1f MiB snapshot\n",
            (double)total / (1024.0 * 1024.0));
        return false;
    }

    out_buf.resize(state_->head_kv.size());
    out_pos = state_->head_kv_pos;

    for (size_t h = 0; h < state_->head_kv.size(); ++h) {
        const auto & kv = state_->head_kv[h];
        auto & dst = out_buf[h];

        if (kv.k_cache) {
            // Thin GPU path.
            const int slots = std::min(state_->head_kv_pos[h] + 1, n_ctx);
            const size_t k_slice_bytes = (size_t)slots * key_len * sizeof(uint16_t);
            const size_t v_slice_bytes = (size_t)slots * val_len * sizeof(uint16_t);
            const size_t k_stride_g    = (size_t)key_len * n_ctx * sizeof(uint16_t);
            const size_t v_stride_g    = (size_t)val_len * n_ctx * sizeof(uint16_t);

            dst.resize(thin_per_head_bytes(slots, n_head_kv, key_len, val_len));
            int32_t hdr[4] = { slots, n_head_kv, key_len, val_len };
            std::memcpy(dst.data(), hdr, sizeof(hdr));

            uint8_t * k_out = dst.data() + kThinSnapHeaderBytes;
            uint8_t * v_out = k_out + (size_t)n_head_kv * k_slice_bytes;
            for (int g = 0; g < n_head_kv; ++g) {
                ggml_backend_tensor_get(kv.k_cache,
                    k_out + (size_t)g * k_slice_bytes,
                    (size_t)g * k_stride_g, k_slice_bytes);
                ggml_backend_tensor_get(kv.v_cache,
                    v_out + (size_t)g * v_slice_bytes,
                    (size_t)g * v_stride_g, v_slice_bytes);
            }
        } else {
            // CPU stub: legacy full F32 byte copy (kept stable for tests).
            const size_t k_bytes = kv.k.size() * sizeof(float);
            const size_t v_bytes = kv.v.size() * sizeof(float);
            dst.resize(k_bytes + v_bytes);
            std::memcpy(dst.data(), kv.k.data(), k_bytes);
            std::memcpy(dst.data() + k_bytes, kv.v.data(), v_bytes);
        }
    }
    return true;
}

bool Qwen36MtpModule::restore_head_kv(const std::vector<std::vector<uint8_t>> & buf,
                                      const std::vector<int> & pos) {
    if (!state_->loaded || buf.size() != state_->head_kv.size() ||
        pos.size() != state_->head_kv.size()) {
        return false;
    }

    const int n_head_kv = state_->weights.n_head_kv;
    const int key_len   = state_->weights.n_key_length;
    const int val_len   = state_->weights.n_value_length;
    const int n_ctx     = state_->n_ctx;

    // Validate every head before mutating any state — R5 spirit.
    for (size_t h = 0; h < state_->head_kv.size(); ++h) {
        auto & kv = state_->head_kv[h];
        if (kv.k_cache) {
            if (kv.k_cache->type != GGML_TYPE_F16) return false;
            if (kv.v_cache->type != GGML_TYPE_F16) return false;
            if (buf[h].size() < kThinSnapHeaderBytes) return false;
            int32_t hdr[4];
            std::memcpy(hdr, buf[h].data(), sizeof(hdr));
            const int slots = hdr[0];
            if (slots <= 0 || slots > n_ctx) return false;
            if (hdr[1] != n_head_kv || hdr[2] != key_len || hdr[3] != val_len) {
                return false;
            }
            if (buf[h].size() != thin_per_head_bytes(slots, n_head_kv, key_len, val_len)) {
                return false;
            }
            if (pos[h] < 0 || pos[h] >= n_ctx) return false;
        } else {
            const size_t k_bytes = kv.k.size() * sizeof(float);
            const size_t v_bytes = kv.v.size() * sizeof(float);
            if (k_bytes == 0 || v_bytes == 0 ||
                buf[h].size() != k_bytes + v_bytes ||
                pos[h] < 0 || pos[h] >= n_ctx) {
                return false;
            }
        }
    }

    for (size_t h = 0; h < state_->head_kv.size(); ++h) {
        auto & kv = state_->head_kv[h];
        const auto & src = buf[h];

        if (kv.k_cache) {
            int32_t hdr[4];
            std::memcpy(hdr, src.data(), sizeof(hdr));
            const int slots = hdr[0];
            const size_t k_slice_bytes = (size_t)slots * key_len * sizeof(uint16_t);
            const size_t v_slice_bytes = (size_t)slots * val_len * sizeof(uint16_t);
            const size_t k_stride_g    = (size_t)key_len * n_ctx * sizeof(uint16_t);
            const size_t v_stride_g    = (size_t)val_len * n_ctx * sizeof(uint16_t);

            const uint8_t * k_in = src.data() + kThinSnapHeaderBytes;
            const uint8_t * v_in = k_in + (size_t)n_head_kv * k_slice_bytes;
            for (int g = 0; g < n_head_kv; ++g) {
                ggml_backend_tensor_set(kv.k_cache,
                    k_in + (size_t)g * k_slice_bytes,
                    (size_t)g * k_stride_g, k_slice_bytes);
                ggml_backend_tensor_set(kv.v_cache,
                    v_in + (size_t)g * v_slice_bytes,
                    (size_t)g * v_stride_g, v_slice_bytes);
            }
        } else {
            const size_t k_bytes = kv.k.size() * sizeof(float);
            const size_t v_bytes = kv.v.size() * sizeof(float);
            std::memcpy(kv.k.data(), src.data(), k_bytes);
            std::memcpy(kv.v.data(), src.data() + k_bytes, v_bytes);
        }
    }
    state_->head_kv_pos = pos;
    return true;
}

bool Qwen36MtpModule::warm_head_kv(const int32_t * prompt_tokens,
                                   int             n_prompt,
                                   int32_t         prefill_next,
                                   const float *   hiddens) {
    if (!state_->loaded || !state_->attached || !state_->target) {
        std::fprintf(stderr,
            "[qwen36_mtp] warm_head_kv: module not loaded/attached\n");
        return false;
    }
    if (n_prompt <= 0 || !prompt_tokens || !hiddens) return true;

    const int n_embd     = state_->weights.n_embd;
    const int n_heads    = state_->weights.n_heads;
    const int n_head_kv  = state_->weights.n_head_kv;
    const int key_len    = state_->weights.n_key_length;
    const int val_len    = state_->weights.n_value_length;
    const int kv_dim     = n_head_kv * key_len;
    const int v_total    = n_head_kv * val_len;
    const int rope_n_rot = std::min(64, key_len);
    const float rope_theta = 1e7f;

    // Lazy grow: head_kv was allocated small at init; size it now to fit the
    // actual prompt + a decode-horizon margin (env override). On steady-state
    // agentic loops the same daemon serves prompts of similar size, so we
    // pay the realloc once at startup. Failure here is a real OOM/limit hit.
    int decode_margin = 1024;
    if (const char * s = std::getenv("DFLASH27B_MTP_DECODE_MARGIN")) {
        const int v = std::atoi(s);
        if (v > 0) decode_margin = v;
    }
    // warm fills slots [1..n_prompt]; the chain runner writes slot n_prompt
    // and beyond on subsequent iters. The +1 accommodates slot n_prompt.
    if (!ensure_head_kv_capacity_(n_prompt + 1 + decode_margin)) {
        std::fprintf(stderr,
            "[qwen36_mtp] warm_head_kv: cannot ensure capacity for n_prompt=%d (max=%d)\n",
            n_prompt, state_->n_ctx_max);
        return false;
    }
    if (n_prompt > state_->n_ctx) {
        // Should not happen — ensure_head_kv_capacity_ either grew or returned
        // false. Keep the guard as belt-and-braces in case of future refactor.
        std::fprintf(stderr,
            "[qwen36_mtp] warm_head_kv: n_prompt=%d exceeds head_kv capacity n_ctx=%d\n",
            n_prompt, state_->n_ctx);
        return false;
    }
    if ((int)state_->head_kv.size() < n_heads) {
        std::fprintf(stderr,
            "[qwen36_mtp] warm_head_kv: head_kv not allocated (size=%zu, expected=%d)\n",
            state_->head_kv.size(), n_heads);
        return false;
    }

    // GPU path: when the backbone backend is available, batch-process all
    // n_prompt positions in a single cgraph.  Replaces the host-side per-slot
    // dequant+matvec+upload loop (~2 s on Qwen3.6-27B with 69-token prompt)
    // with one backend pass (~tens of ms).
    if (state_->kv_ctx && state_->target->backend()) {
        const int h = 0;  // GGUF has n_heads=1; multi-head warmup would loop
        const auto & head = state_->weights.heads[h];
        int rope_sections[4] = { 11, 11, 10, 0 };
        const int slot_start = 1;  // slot 0 unused (no h_{-1} input)
        ggml_backend_t backend = state_->target->backend();

        // Pre-embed all input tokens on host. input_tok_seq[i] is the token at
        // sequence position (i+1): prompt_tokens[i+1] for i+1 < n_prompt,
        // prefill_next at slot n_prompt.
        std::vector<int32_t> input_tok_seq(n_prompt);
        for (int i = 0; i < n_prompt; i++) {
            const int p = i + 1;
            input_tok_seq[i] = (p < n_prompt) ? prompt_tokens[p] : prefill_next;
        }
        std::vector<float> embed_seq((size_t)n_prompt * n_embd);
        if (!state_->target->embed_tokens(input_tok_seq.data(), n_prompt,
                                           embed_seq.data())) {
            std::fprintf(stderr, "[qwen36_mtp gpu-warm] embed_tokens failed\n");
            return false;
        }

        // MROPE positions, interleaved (4 axes per token) matching backbone.
        std::vector<int32_t> pos_seq(4 * n_prompt);
        for (int i = 0; i < n_prompt; i++) {
            const int p = i + 1;
            pos_seq[4 * i + 0] = p;
            pos_seq[4 * i + 1] = p;
            pos_seq[4 * i + 2] = p;
            pos_seq[4 * i + 3] = 0;
        }

        if (!build_qwen36_mtp_warm_graph(state_->warm_sg, head,
                                          state_->head_kv[h].k_cache,
                                          state_->head_kv[h].v_cache,
                                          backend,
                                          n_embd, n_head_kv, key_len, val_len,
                                          rope_n_rot, rope_sections,
                                          rope_theta, 1e-6f,
                                          slot_start, n_prompt)) {
            return false;
        }

        ggml_backend_tensor_set(state_->warm_sg.inp_embed_seq, embed_seq.data(),
            0, sizeof(float) * embed_seq.size());
        ggml_backend_tensor_set(state_->warm_sg.inp_h_seq, hiddens,
            0, sizeof(float) * (size_t)n_prompt * n_embd);
        ggml_backend_tensor_set(state_->warm_sg.inp_pos, pos_seq.data(),
            0, sizeof(int32_t) * pos_seq.size());

        auto st = ggml_backend_graph_compute(backend, state_->warm_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr,
                "[qwen36_mtp gpu-warm] graph_compute status=%d\n", (int)st);
            return false;
        }
        if ((int)state_->head_kv_pos.size() > h) {
            state_->head_kv_pos[h] = n_prompt;
        }
        return true;
    }

    // We only warm head 0's K/V (the only head on this 27B GGUF).  Multi-head
    // GGUFs would warm each head; their h_prev would be the previous head's
    // un-normed output, which we don't have during prefill.  The handoff and
    // redesign doc both pin γ_max=1 for this GGUF.
    const int h = 0;
    const auto & head = state_->weights.heads[h];

    // Working buffers (sized once, reused across positions).
    std::vector<float> embed_buf(n_embd);
    std::vector<float> e_in(n_embd);
    std::vector<float> h_in(n_embd);
    std::vector<float> concat_buf(2 * n_embd);
    std::vector<float> x(n_embd);
    std::vector<float> x_normed(n_embd);
    std::vector<float> k_buf(kv_dim);
    std::vector<float> v_buf(v_total);

    // Dequantize head's per-position-invariant tensors once.
    std::vector<float> enorm_data, hnorm_data, eh_proj_data;
    std::vector<float> attn_norm_data, attn_k_data, attn_k_norm_data, attn_v_data;
    if (!tensor_to_floats(head.enorm, enorm_data) ||
        (int)enorm_data.size() != n_embd ||
        !tensor_to_floats(head.hnorm, hnorm_data) ||
        (int)hnorm_data.size() != n_embd ||
        !tensor_to_floats(head.eh_proj, eh_proj_data) ||
        (int)eh_proj_data.size() != n_embd * 2 * n_embd ||
        !tensor_to_floats(head.attn_norm, attn_norm_data) ||
        (int)attn_norm_data.size() != n_embd ||
        !tensor_to_floats(head.attn_k, attn_k_data) ||
        (int)attn_k_data.size() != kv_dim * n_embd ||
        !tensor_to_floats(head.attn_v, attn_v_data) ||
        (int)attn_v_data.size() != v_total * n_embd ||
        !tensor_to_floats(head.attn_k_norm, attn_k_norm_data) ||
        (int)attn_k_norm_data.size() != key_len) {
        std::fprintf(stderr,
            "[qwen36_mtp] warm_head_kv: failed to dequant head tensors\n");
        return false;
    }

    auto & kv = state_->head_kv[h];
    // Slot p of head_kv represents sequence position p (matching backbone KV
    // slot p, matching RoPE position p, matching step_batch's draft_pos = base_pos+h).
    // The head's K/V at slot p use inputs (h_{p-1}, t_p): backbone post-norm
    // hidden at the PREVIOUS position and the input token AT position p.
    // Slot 0 has no h_{-1}, so it stays zero (Q at later slots will see slot 0
    // as a near-zero contribution; softmax shifts mass to populated slots).
    // For p in [1, n_prompt-1]: input_tok = prompt_tokens[p].
    // For p = n_prompt:         input_tok = prefill_next, h_{p-1} = h_{n_prompt-1}
    //                            (the last prefill hidden).
    const int last_slot = n_prompt;  // inclusive
    if (last_slot >= state_->n_ctx) {
        std::fprintf(stderr,
            "[qwen36_mtp] warm_head_kv: last_slot=%d exceeds n_ctx=%d\n",
            last_slot, state_->n_ctx);
        return false;
    }
    for (int p = 1; p <= last_slot; p++) {
        const int32_t input_tok =
            (p < n_prompt) ? prompt_tokens[p] : prefill_next;

        if (!state_->target->embed_tokens(&input_tok, 1, embed_buf.data())) {
            std::fprintf(stderr,
                "[qwen36_mtp] warm_head_kv: embed_tokens failed at p=%d\n", p);
            return false;
        }
        rmsnorm_cpu(embed_buf.data(), enorm_data.data(), e_in.data(), n_embd);

        // h_{p-1}: backbone post-norm hidden at the PREVIOUS sequence position.
        const float * h_prev_p = hiddens + (size_t)(p - 1) * n_embd;
        rmsnorm_cpu(h_prev_p, hnorm_data.data(), h_in.data(), n_embd);

        // Concat order [e_in; h_in] matches llama.cpp PR #22673 graph_mtp.
        std::copy(e_in.begin(), e_in.end(), concat_buf.begin());
        std::copy(h_in.begin(), h_in.end(), concat_buf.begin() + n_embd);
        matvec_cpu(eh_proj_data.data(), concat_buf.data(), x.data(),
                   n_embd, 2 * n_embd);

        rmsnorm_cpu(x.data(), attn_norm_data.data(), x_normed.data(), n_embd);

        matvec_cpu(attn_k_data.data(), x_normed.data(), k_buf.data(),
                   kv_dim, n_embd);
        matvec_cpu(attn_v_data.data(), x_normed.data(), v_buf.data(),
                   v_total, n_embd);

        per_head_rmsnorm(k_buf.data(), attn_k_norm_data.data(), n_head_kv, key_len);
        rope_cpu(k_buf.data(), n_head_kv, key_len, rope_n_rot, p, rope_theta);

        const size_t k_slot_off = (size_t)p * n_head_kv * key_len;
        const size_t v_slot_off = (size_t)p * n_head_kv * val_len;
        std::memcpy(kv.k.data() + k_slot_off, k_buf.data(),
                    sizeof(float) * (size_t)n_head_kv * key_len);
        std::memcpy(kv.v.data() + v_slot_off, v_buf.data(),
                    sizeof(float) * (size_t)n_head_kv * val_len);
    }
    if ((int)state_->head_kv_pos.size() > h) {
        state_->head_kv_pos[h] = last_slot;
    }
    // Prefill is done.  From this point on, every verify_batch is a decode
    // step whose ONLY hidden-sequence consumer is hidden_at_pos(base_pos-1)
    // (the chain's iter-0 h_prev seed).  Tell the target to download only
    // that single row from all_norm_hidden / all_h_pre_norm instead of the
    // full [n_tokens, n_embd] tensor — collapses the 2x per-verify ~80 KB
    // device->host sync to a 2x ~20 KB sync (hidden_dim=5120, D+1=4 tokens
    // baseline) and erases the WSL2 cudaStreamSynchronize scheduler tax
    // that dominated decode-side verify_batch in the verify_prof traces.
    if (auto * t = dynamic_cast<Qwen35DFlashTarget *>(state_->target)) {
        t->set_hidden_capture_mode(
            Qwen35DFlashTarget::VerifyCaptureMode::LAST_ROW_ONLY);
    }
    return true;
}

// Range-warm a contiguous slot window. Same graph as warm_head_kv but with
// caller-controlled slot_start and slot-to-prompt mapping so partial-WARM
// restore can fill slots [snap_pos..prompt_len] after a head_kv restore
// covered [1..snap_pos]. Row i in `hiddens` is h_{start_slot+i-1}.
bool Qwen36MtpModule::warm_head_kv_range(const int32_t * prompt_tokens,
                                          int             n_prompt,
                                          int             start_slot,
                                          int             n_chunk,
                                          int32_t         prefill_next,
                                          const float *   hiddens) {
    if (!state_->loaded || !state_->attached || !state_->target) {
        std::fprintf(stderr,
            "[qwen36_mtp] warm_head_kv_range: module not loaded/attached\n");
        return false;
    }
    if (n_chunk <= 0 || start_slot < 1 || !prompt_tokens || !hiddens) {
        return false;
    }

    const int n_embd     = state_->weights.n_embd;
    const int n_heads    = state_->weights.n_heads;
    const int n_head_kv  = state_->weights.n_head_kv;
    const int key_len    = state_->weights.n_key_length;
    const int val_len    = state_->weights.n_value_length;
    const int rope_n_rot = std::min(64, key_len);
    const float rope_theta = 1e7f;

    int decode_margin = 1024;
    if (const char * s = std::getenv("DFLASH27B_MTP_DECODE_MARGIN")) {
        const int v = std::atoi(s);
        if (v > 0) decode_margin = v;
    }
    const int end_slot = start_slot + n_chunk;
    if (!ensure_head_kv_capacity_(end_slot + decode_margin)) {
        std::fprintf(stderr,
            "[qwen36_mtp] warm_head_kv_range: cannot ensure capacity for end_slot=%d (max=%d)\n",
            end_slot, state_->n_ctx_max);
        return false;
    }
    if (end_slot > state_->n_ctx) {
        std::fprintf(stderr,
            "[qwen36_mtp] warm_head_kv_range: end_slot=%d exceeds n_ctx=%d\n",
            end_slot, state_->n_ctx);
        return false;
    }
    if ((int)state_->head_kv.size() < n_heads) {
        std::fprintf(stderr,
            "[qwen36_mtp] warm_head_kv_range: head_kv not allocated (size=%zu, expected=%d)\n",
            state_->head_kv.size(), n_heads);
        return false;
    }
    if (!state_->kv_ctx || !state_->target->backend()) {
        // CPU stub path is not used in production; range-warm only supports GPU.
        return false;
    }

    const int h = 0;
    const auto & head = state_->weights.heads[h];
    int rope_sections[4] = { 11, 11, 10, 0 };
    ggml_backend_t backend = state_->target->backend();

    // Input mapping: slot p uses input t_p (= prompt[p] if p < n_prompt,
    // else prefill_next). Row i corresponds to slot start_slot+i.
    std::vector<int32_t> input_tok_seq(n_chunk);
    for (int i = 0; i < n_chunk; i++) {
        const int p = start_slot + i;
        input_tok_seq[i] = (p < n_prompt) ? prompt_tokens[p] : prefill_next;
    }
    std::vector<float> embed_seq((size_t)n_chunk * n_embd);
    if (!state_->target->embed_tokens(input_tok_seq.data(), n_chunk,
                                       embed_seq.data())) {
        std::fprintf(stderr, "[qwen36_mtp range-warm] embed_tokens failed\n");
        return false;
    }

    std::vector<int32_t> pos_seq(4 * n_chunk);
    for (int i = 0; i < n_chunk; i++) {
        const int p = start_slot + i;
        pos_seq[4 * i + 0] = p;
        pos_seq[4 * i + 1] = p;
        pos_seq[4 * i + 2] = p;
        pos_seq[4 * i + 3] = 0;
    }

    if (!build_qwen36_mtp_warm_graph(state_->warm_sg, head,
                                      state_->head_kv[h].k_cache,
                                      state_->head_kv[h].v_cache,
                                      backend,
                                      n_embd, n_head_kv, key_len, val_len,
                                      rope_n_rot, rope_sections,
                                      rope_theta, 1e-6f,
                                      start_slot, n_chunk)) {
        return false;
    }

    ggml_backend_tensor_set(state_->warm_sg.inp_embed_seq, embed_seq.data(),
        0, sizeof(float) * embed_seq.size());
    ggml_backend_tensor_set(state_->warm_sg.inp_h_seq, hiddens,
        0, sizeof(float) * (size_t)n_chunk * n_embd);
    ggml_backend_tensor_set(state_->warm_sg.inp_pos, pos_seq.data(),
        0, sizeof(int32_t) * pos_seq.size());

    auto st = ggml_backend_graph_compute(backend, state_->warm_sg.gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
            "[qwen36_mtp range-warm] graph_compute status=%d\n", (int)st);
        return false;
    }
    if ((int)state_->head_kv_pos.size() > h) {
        // Extend head_kv_pos to the last slot we wrote (do not regress on a
        // shorter range; the chain runner's first read is at pos = base_pos).
        state_->head_kv_pos[h] = std::max(state_->head_kv_pos[h], end_slot - 1);
    }
    return true;
}

const float * Qwen36MtpModule::test_initial_hidden_ptr() const {
    return state_->initial_hidden_ptr;
}
int Qwen36MtpModule::test_initial_hidden_dim() const {
    return state_->initial_hidden_dim;
}

// Bug #5 fix: shape-only graph cached per (head_idx, fa_window, fused, topk).
// Per-call slot routing is uploaded via push_kv_slot_inputs_() below.

// Push the runtime KV routing inputs (write slot, read idxs, mask) for the
// current draft_pos / kv_len.  fa_max is baked into the graph at build time.
static void push_kv_slot_inputs_(Qwen36MtpStepGraph * sg,
                                 int draft_pos, int kv_len,
                                 int n_head_kv) {
    const int fa_max  = sg->fa_max;
    const int fa_win  = sg->fa_window;
    const int fa_kv_lo = (fa_win > 0 && kv_len > fa_win) ? (kv_len - fa_win) : 0;
    const int fa_kv_n  = std::min(kv_len - fa_kv_lo, fa_max);

    const int64_t widx = (int64_t)draft_pos;
    ggml_backend_tensor_set(sg->inp_kv_idx_write, &widx, 0, sizeof(int64_t));

    std::vector<int32_t> ridx((size_t)fa_max * n_head_kv);
    for (int h = 0; h < n_head_kv; ++h) {
        int32_t * row = ridx.data() + (size_t)h * fa_max;
        for (int i = 0; i < fa_max; ++i) {
            row[i] = (i < fa_kv_n) ? (fa_kv_lo + i) : 0;  // unused rows masked to -INF
        }
    }
    ggml_backend_tensor_set(sg->inp_kv_idxs_read, ridx.data(), 0,
        sizeof(int32_t) * ridx.size());

    std::vector<uint16_t> mask((size_t)fa_max, 0);
    const uint16_t neg_inf_f16 = 0xFC00;
    for (int i = fa_kv_n; i < fa_max; ++i) mask[i] = neg_inf_f16;
    ggml_backend_tensor_set(sg->inp_kv_mask, mask.data(), 0,
        sizeof(uint16_t) * mask.size());
}

Qwen36MtpStepGraph * Qwen36MtpModule::get_or_build_step_graph_(int head_idx) {
    if (head_idx < 0 || head_idx >= (int)state_->weights.heads.size()) {
        return nullptr;
    }
    if (head_idx >= (int)state_->head_kv.size())     return nullptr;

    const int    n_embd    = state_->weights.n_embd;
    const int    n_head    = state_->weights.n_head_count;
    const int    n_head_kv = state_->weights.n_head_kv;
    const int    key_len   = state_->weights.n_key_length;
    const int    val_len   = state_->weights.n_value_length;
    const int    ffn_len   = state_->weights.n_ffn_length;
    const int    n_rot     = std::min(64, key_len);
    const float  rope_freq_base = 1e7f;
    const float  rms_eps        = 1e-6f;
    int          rope_sections[4] = { 11, 11, 10, 0 };

    const int    fa_win   = state_->target ? state_->target->fa_window() : 0;
    ggml_tensor * lm_head = state_->target ? state_->target->lm_head_weight() : nullptr;
    const bool   fused    = (lm_head != nullptr);
    const int    topk_k   = (fused && state_->draft_topk > 1) ? state_->draft_topk : 0;

    // Find matching cached entry; else pick first empty / oldest slot.
    int hit = -1, free_slot = -1;
    for (int i = 0; i < (int)state_->step_sg_cache.size(); ++i) {
        const auto & k = state_->step_sg_cache[i].first;
        if (state_->step_sg_cache[i].second &&
            k.head_idx == head_idx && k.fa_window == fa_win &&
            k.fused_lm_head == fused && k.topk_k == topk_k) {
            hit = i; break;
        }
        if (!state_->step_sg_cache[i].second && free_slot < 0) free_slot = i;
    }
    if (hit >= 0) return state_->step_sg_cache[hit].second.get();
    if (free_slot < 0) free_slot = 0;  // evict slot 0 (FIFO is fine; cap=4)

    auto & slot = state_->step_sg_cache[free_slot];
    if (slot.second) qwen36_mtp_step_graph_free(*slot.second);
    else             slot.second.reset(new Qwen36MtpStepGraph());

    const auto & head = state_->weights.heads[head_idx];
    if (!build_qwen36_mtp_step_graph(*slot.second, head,
                                      state_->head_kv[head_idx].k_cache,
                                      state_->head_kv[head_idx].v_cache,
                                      state_->target->backend(),
                                      n_embd, n_head, n_head_kv,
                                      key_len, val_len, ffn_len,
                                      n_rot, rope_sections,
                                      rope_freq_base, rms_eps,
                                      state_->n_ctx,
                                      fa_win, lm_head, topk_k)) {
        std::fprintf(stderr,
            "[qwen36_mtp] get_or_build_step_graph_: build failed head=%d\n",
            head_idx);
        slot.second.reset();
        slot.first = State::StepGraphKey{};
        return nullptr;
    }
    slot.first = State::StepGraphKey{head_idx, fa_win, fused, topk_k};
    return slot.second.get();
}

// Per-call cgraph on the backbone backend; cached per (head_idx, draft_pos)
// in state_->step_sg_cache (Phase B+).  When the bound target exposes its
// lm_head_weight() the graph also fuses the LM-head matmul + argmax so we
// skip the hidden -> host -> separate-cgraph round trip per call.
bool Qwen36MtpModule::step_batch_gpu_(int32_t current_token,
                                       int base_pos,
                                       std::vector<StepOutput> & out) {
    const int n_embd    = state_->weights.n_embd;
    const int n_heads   = state_->weights.n_heads;

    out.clear();
    out.reserve(n_heads);

    ggml_backend_t backend = state_->target->backend();
    int32_t cur_token = current_token;
    std::vector<float> embed_buf(n_embd);

    for (int h = 0; h < n_heads; h++) {
        const int draft_pos = base_pos + h;
        if (draft_pos >= state_->n_ctx) {
            std::fprintf(stderr,
                "[qwen36_mtp gpu] draft_pos=%d exceeds n_ctx=%d\n",
                draft_pos, state_->n_ctx);
            out.clear();
            return false;
        }
        const int kv_len = draft_pos + 1;

        // ── Select h_prev (same priority as CPU path) ───────────────────
        // Pre-output-norm preferred (PR #22673 t_h_pre_norm) — head's
        // hnorm normalises h_prev internally and post-norm seed double-
        // normalises.  Fall back to post-norm if the adapter did not
        // capture the pre-norm sequence this verify_batch.
        const float * h_prev = nullptr;
        if (h == 0) {
            if (base_pos >= 1) {
                h_prev = state_->target->hidden_at_pos_pre_norm(base_pos - 1);
                if (!h_prev) {
                    h_prev = state_->target->hidden_at_pos(base_pos - 1);
                }
            }
            if (!h_prev) h_prev = state_->target->last_hidden();
            if (!h_prev && state_->initial_hidden_ptr &&
                state_->initial_hidden_dim == n_embd) {
                h_prev = state_->initial_hidden_ptr;
            }
            if (!h_prev) {
                std::fprintf(stderr,
                    "[qwen36_mtp gpu] no hidden available at base_pos=%d\n",
                    base_pos);
                out.clear();
                return false;
            }
        } else {
            // h>0 chain: use the head's own previous output (kept on host as
            // last_hidden).  Only matters when n_heads > 1; the 27B GGUF has
            // n_heads=1 so this branch is rarely exercised.
            h_prev = state_->last_hidden.data();
        }
        // Embed cur_token via target (already on host).
        if (!state_->target->embed_tokens(&cur_token, 1, embed_buf.data())) {
            std::fprintf(stderr, "[qwen36_mtp gpu] embed_tokens failed h=%d\n", h);
            out.clear();
            return false;
        }

        // Get-or-build the shape-only step graph.
        Qwen36MtpStepGraph * sg = get_or_build_step_graph_(h);
        if (!sg) { out.clear(); return false; }

        // Upload inputs.  Pass h_prev directly (no scratch memcpy — task E).
        ggml_backend_tensor_set(sg->inp_embed,
            embed_buf.data(), 0, sizeof(float) * n_embd);
        ggml_backend_tensor_set(sg->inp_h_prev,
            h_prev, 0, sizeof(float) * n_embd);
        // MROPE positions: text-only mode (axes 0..2 = position, axis 3 = 0).
        const int32_t pos[4] = { draft_pos, draft_pos, draft_pos, 0 };
        ggml_backend_tensor_set(sg->inp_pos, pos, 0, sizeof(pos));
        push_kv_slot_inputs_(sg, draft_pos, kv_len, state_->weights.n_head_kv);

        auto st = ggml_backend_graph_compute(backend, sg->gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[qwen36_mtp gpu] graph_compute status=%d\n", (int)st);
            out.clear();
            return false;
        }
        if ((int)state_->head_kv_pos.size() > h) {
            state_->head_kv_pos[h] = std::max(state_->head_kv_pos[h], draft_pos);
        }

        StepOutput so;
        if (sg->fused_lm_head && sg->out_argmax_token) {
            // Fused path: read the argmax (and optional full logits for
            // top-K) directly from the cached graph's outputs — no separate
            // projection cgraph, no hidden -> host round trip.
            int32_t tok = 0;
            ggml_backend_tensor_get(sg->out_argmax_token, &tok, 0, sizeof(int32_t));
            so.draft_token = tok;
            so.draft_logit = 0.0f;
            if (sg->out_logits && state_->draft_topk > 1) {
                const int vocab = (int)sg->out_logits->ne[0];
                std::vector<float> logits_buf((size_t)vocab);
                ggml_backend_tensor_get(sg->out_logits, logits_buf.data(),
                    0, sizeof(float) * vocab);
                so.draft_logit = logits_buf[tok];
                emit_topk_logprobs(logits_buf.data(), vocab,
                                   state_->draft_topk, so);
            }
            // Chain state: when n_heads > 1 the next head's h_prev needs the
            // hidden.  Skip the download otherwise to keep the fused path
            // zero-host-roundtrip.  Read PRE-shared_head_norm hidden — feeding
            // post-norm here causes the next iter's `hnorm` to double-
            // normalise (see qwen36_mtp_graph.cpp pre-norm output comment;
            // mirrors llama.cpp PR #22673 `t_h_pre_norm` design).
            if (n_heads > 1) {
                std::vector<float> h_pre(n_embd);
                ggml_backend_tensor_get(sg->out_h_pre_norm, h_pre.data(),
                    0, sizeof(float) * n_embd);
                state_->last_hidden = std::move(h_pre);
            }
        } else {
            // Unfused fallback: read hidden, project on host via target.
            std::vector<float> x_normed(n_embd);
            ggml_backend_tensor_get(sg->out_x_normed,
                x_normed.data(), 0, sizeof(float) * n_embd);
            if (state_->draft_topk > 1) {
                std::vector<float> logits_buf;
                int vocab = 0;
                if (state_->target->project_hidden_to_logits(x_normed.data(), 1,
                                                              logits_buf, vocab) &&
                    vocab > 0) {
                    so.draft_token = argmax(logits_buf.data(), vocab);
                    so.draft_logit = logits_buf[so.draft_token];
                    emit_topk_logprobs(logits_buf.data(), vocab,
                                       state_->draft_topk, so);
                } else {
                    std::vector<int32_t> tok_out;
                    if (!state_->target->project_hidden_to_tokens(x_normed.data(), 1,
                                                                   tok_out) ||
                        tok_out.empty()) {
                        std::fprintf(stderr,
                            "[qwen36_mtp gpu] project_hidden_to_tokens failed h=%d\n", h);
                        out.clear();
                        return false;
                    }
                    so.draft_token = tok_out[0];
                    so.draft_logit = 0.0f;
                    static bool warned = false;
                    if (!warned) {
                        std::fprintf(stderr,
                            "[qwen36_mtp gpu] draft_topk=%d requested but target "
                            "lacks project_hidden_to_logits; emitting argmax only.\n",
                            state_->draft_topk);
                        warned = true;
                    }
                }
            } else {
                std::vector<int32_t> tok_out;
                if (!state_->target->project_hidden_to_tokens(x_normed.data(), 1,
                                                               tok_out) ||
                    tok_out.empty()) {
                    std::fprintf(stderr,
                        "[qwen36_mtp gpu] project_hidden_to_tokens failed h=%d\n", h);
                    out.clear();
                    return false;
                }
                so.draft_token = tok_out[0];
                so.draft_logit = 0.0f;
            }
            // last_hidden must be PRE-shared_head_norm (chain h_prev contract;
            // see qwen36_mtp_graph.cpp pre-norm output comment and
            // llama.cpp PR #22673 `t_h_pre_norm`).  `x_normed` is consumed
            // above for the LM-head projection only.
            if (n_heads > 1) {
                std::vector<float> h_pre(n_embd);
                ggml_backend_tensor_get(sg->out_h_pre_norm, h_pre.data(),
                    0, sizeof(float) * n_embd);
                state_->last_hidden = std::move(h_pre);
            }
        }
        out.push_back(so);
        cur_token = so.draft_token;
    }
    return true;
}

// Phase A autoregressive chain draft.  Reuses head 0 (the only NextN head on
// the Qwen3.6-27B GGUF) `chain_depth` times, feeding the head's own
// post-shared_head_norm hidden back as h_prev for the next iteration.  The
// per-iter graph is rebuilt on each call — Phase B (R1) will cache it.
//
// CPU stub path (no backend / no kv_ctx): degrade gracefully to the default
// `step_batch`+clamp.  Unit tests exercising the CPU forward at depth=1
// remain byte-identical to the old behaviour.
bool Qwen36MtpModule::step_chain(int32_t current_token,
                                 int base_pos,
                                 int chain_depth,
                                 std::vector<StepOutput> & out) {
    out.clear();
    if (!state_->loaded || !state_->attached) return false;
    if (chain_depth <= 0) chain_depth = 1;

    // Single-iter fast path: byte-identical to the established step_batch
    // contract.  Avoids the additional state_->last_hidden plumbing in the
    // depth>1 loop below.
    if (chain_depth == 1) {
        std::vector<StepOutput> tmp;
        if (!step_batch(current_token, base_pos, tmp)) return false;
        if (!tmp.empty()) out.push_back(std::move(tmp.front()));
        return true;
    }

    // CPU stub fallback for depth>1 — not exercised by production today
    // (the only depth>1 caller goes through the GPU path); degrade to one
    // step_batch call so tests that swap out the backend still link.
    const bool gpu_ready = (state_->kv_ctx && state_->target &&
                            state_->target->backend());
    if (!gpu_ready) {
        std::vector<StepOutput> tmp;
        if (!step_batch(current_token, base_pos, tmp)) return false;
        if (!tmp.empty()) out.push_back(std::move(tmp.front()));
        return true;
    }

    // ── GPU multi-iter chain ─────────────────────────────────────────
    // Phase B+: per-iter step graphs are pulled from state_->step_sg_cache
    // via get_or_build_step_graph_().  First-pass at each draft_pos is a
    // build; subsequent calls are a pure tensor_set + compute.  When the
    // bound target exposes lm_head_weight() the fused argmax output is read
    // from the graph directly so we skip the projection-cgraph round trip.
    out.reserve(chain_depth);

    ggml_backend_t backend = state_->target->backend();

    // The Qwen3.6-27B GGUF has n_heads=1; chain depth replays that single
    // head against successive draft positions.  If a future GGUF lands with
    // multiple physical NextN heads, this implementation would still chain
    // on head 0 only — multi-head + chain interaction is a Phase >A concern.
    const int h = 0;
    if ((int)state_->weights.heads.size() <= h ||
        (int)state_->head_kv.size() <= h) {
        std::fprintf(stderr,
            "[qwen36_mtp gpu chain] head 0 missing (heads=%zu head_kv=%zu)\n",
            state_->weights.heads.size(), state_->head_kv.size());
        return false;
    }

    const int n_embd = state_->weights.n_embd;
    int32_t cur_token = current_token;
    std::vector<float> embed_buf(n_embd);

#ifdef DFLASH_MTP_PROFILE
    // Profiling accumulators (DFLASH_MTP_PROFILE=1).  All in ms.
    const bool prof_on = mtp_profile_enabled();
    double prof_sum_embed = 0.0, prof_sum_set = 0.0, prof_sum_compute = 0.0;
    double prof_sum_get_x = 0.0, prof_sum_get_h = 0.0, prof_sum_get_argmax = 0.0;
    double prof_sum_total = 0.0;
    int    prof_iters = 0;
#endif  // DFLASH_MTP_PROFILE

    for (int it = 0; it < chain_depth; it++) {
        const int draft_pos = base_pos + it;
        if (draft_pos >= state_->n_ctx) {
            std::fprintf(stderr,
                "[qwen36_mtp gpu chain] draft_pos=%d exceeds n_ctx=%d (iter=%d)\n",
                draft_pos, state_->n_ctx, it);
            return false;
        }
        const int kv_len = draft_pos + 1;

        // h_prev resolution mirrors step_batch_gpu_:
        //   iter 0: pull from target (hidden_at_pos_pre_norm > hidden_at_pos
        //           > last_hidden > initial).  Pre-norm preferred to mirror
        //           llama.cpp PR #22673 `t_h_pre_norm` — the head's hnorm
        //           normalises h_prev internally so the post-output-norm
        //           tensor double-normalises and crushes D>=2 accept.
        //   iter h>0: use state_->last_hidden written by the previous iter
        //             from out_h_pre_norm (the pre-shared_head_norm hidden;
        //             commit 9850ec9).
        const float * h_prev = nullptr;
        if (it == 0) {
            if (base_pos >= 1) {
                h_prev = state_->target->hidden_at_pos_pre_norm(base_pos - 1);
                if (!h_prev) {
                    // Fallback: adapter did not capture the pre-norm
                    // sequence this verify_batch.  Post-norm degrades
                    // D>=2 accept (commit 9850ec9 fixed the intra-iter
                    // re-feed; this is the OUTER seed and silently
                    // returns to the pre-9850ec9 regime when it fires).
                    // Warn once per process so production knows.
                    static bool warned_post_norm = false;
                    if (!warned_post_norm) {
                        std::fprintf(stderr,
                            "[qwen36_mtp gpu chain] WARN: hidden_at_pos_pre_norm "
                            "returned null at base_pos=%d, falling back to "
                            "post-norm hidden. This silently undoes the "
                            "PR #22673 t_h_pre_norm fix at iter 0 and crushes "
                            "D>=2 accept. Caller should call "
                            "enable_hidden_seq_capture(true) on the target "
                            "BEFORE the prefill verify_batch. (warning fires once)\n",
                            base_pos);
                        warned_post_norm = true;
                    }
                    h_prev = state_->target->hidden_at_pos(base_pos - 1);
                }
            }
            if (!h_prev) h_prev = state_->target->last_hidden();
            if (!h_prev && state_->initial_hidden_ptr &&
                state_->initial_hidden_dim == n_embd) {
                h_prev = state_->initial_hidden_ptr;
            }
            if (!h_prev) {
                std::fprintf(stderr,
                    "[qwen36_mtp gpu chain] no hidden available at base_pos=%d\n",
                    base_pos);
                return false;
            }
        } else {
            if ((int)state_->last_hidden.size() != n_embd) {
                std::fprintf(stderr,
                    "[qwen36_mtp gpu chain] last_hidden missing for iter %d\n", it);
                return false;
            }
            h_prev = state_->last_hidden.data();
        }

#ifdef DFLASH_MTP_PROFILE
        prof_clock::time_point t_iter0 = prof_on ? prof_clock::now() : prof_clock::time_point{};
        prof_clock::time_point t0 = prof_on ? prof_clock::now() : prof_clock::time_point{};
#endif  // DFLASH_MTP_PROFILE
        if (!state_->target->embed_tokens(&cur_token, 1, embed_buf.data())) {
            std::fprintf(stderr,
                "[qwen36_mtp gpu chain] embed_tokens failed iter=%d\n", it);
            return false;
        }
#ifdef DFLASH_MTP_PROFILE
        const double t_embed = prof_on ? prof_ms_since(t0) : 0.0;
#endif  // DFLASH_MTP_PROFILE

        Qwen36MtpStepGraph * sg = get_or_build_step_graph_(h);
        if (!sg) return false;

#ifdef DFLASH_MTP_PROFILE
        t0 = prof_on ? prof_clock::now() : prof_clock::time_point{};
#endif
        ggml_backend_tensor_set(sg->inp_embed,
            embed_buf.data(), 0, sizeof(float) * n_embd);
#ifdef DFLASH_MTP_PROFILE
        const double t_set_embed = prof_on ? prof_ms_since(t0) : 0.0;
        t0 = prof_on ? prof_clock::now() : prof_clock::time_point{};
#endif
        // Pass h_prev directly (no scratch memcpy — task E).
        ggml_backend_tensor_set(sg->inp_h_prev,
            h_prev, 0, sizeof(float) * n_embd);
#ifdef DFLASH_MTP_PROFILE
        const double t_set_hprev = prof_on ? prof_ms_since(t0) : 0.0;
        t0 = prof_on ? prof_clock::now() : prof_clock::time_point{};
#endif
        const int32_t pos[4] = { draft_pos, draft_pos, draft_pos, 0 };
        ggml_backend_tensor_set(sg->inp_pos, pos, 0, sizeof(pos));
        push_kv_slot_inputs_(sg, draft_pos, kv_len, state_->weights.n_head_kv);
#ifdef DFLASH_MTP_PROFILE
        const double t_set_pos = prof_on ? prof_ms_since(t0) : 0.0;
        t0 = prof_on ? prof_clock::now() : prof_clock::time_point{};
#endif
        auto st = ggml_backend_graph_compute(backend, sg->gf);
#ifdef DFLASH_MTP_PROFILE
        const double t_compute = prof_on ? prof_ms_since(t0) : 0.0;
#endif
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr,
                "[qwen36_mtp gpu chain] graph_compute status=%d iter=%d\n",
                (int)st, it);
            return false;
        }
        if ((int)state_->head_kv_pos.size() > h) {
            state_->head_kv_pos[h] = std::max(state_->head_kv_pos[h], draft_pos);
        }

        StepOutput so;
        // h_prev for the next iter must be PRE-shared_head_norm (not post-norm);
        // feeding post-norm double-normalises against head's hnorm (see PR #22673).
        // On the fused-LM-head greedy path neither x_normed nor h_pre is needed —
        // the argmax is already on-device; skip both downloads.
        std::vector<float> x_normed;
        const bool need_x_normed =
            !(sg->fused_lm_head && sg->out_argmax_token);
#ifdef DFLASH_MTP_PROFILE
        double t_get_x = 0.0;
#endif
        if (need_x_normed) {
            x_normed.resize(n_embd);
#ifdef DFLASH_MTP_PROFILE
            t0 = prof_on ? prof_clock::now() : prof_clock::time_point{};
#endif
            ggml_backend_tensor_get(sg->out_x_normed,
                x_normed.data(), 0, sizeof(float) * n_embd);
#ifdef DFLASH_MTP_PROFILE
            t_get_x = prof_on ? prof_ms_since(t0) : 0.0;
#endif
        }
        // Only download h_pre when ANOTHER iteration follows that will
        // consume it as h_prev.  On the last chain step the value is
        // discarded — skip the device->host transfer entirely.
        const bool need_h_pre = (it + 1 < chain_depth);
        std::vector<float> h_pre;
#ifdef DFLASH_MTP_PROFILE
        double t_get_h = 0.0;
#endif
        if (need_h_pre) {
            h_pre.resize(n_embd);
#ifdef DFLASH_MTP_PROFILE
            t0 = prof_on ? prof_clock::now() : prof_clock::time_point{};
#endif
            ggml_backend_tensor_get(sg->out_h_pre_norm,
                h_pre.data(), 0, sizeof(float) * n_embd);
#ifdef DFLASH_MTP_PROFILE
            t_get_h = prof_on ? prof_ms_since(t0) : 0.0;
#endif
        }

#ifdef DFLASH_MTP_PROFILE
        double t_get_argmax = 0.0;
#endif
        if (sg->fused_lm_head && sg->out_argmax_token) {
            int32_t tok = 0;
#ifdef DFLASH_MTP_PROFILE
            t0 = prof_on ? prof_clock::now() : prof_clock::time_point{};
#endif
            ggml_backend_tensor_get(sg->out_argmax_token, &tok, 0, sizeof(int32_t));
#ifdef DFLASH_MTP_PROFILE
            t_get_argmax = prof_on ? prof_ms_since(t0) : 0.0;
#endif
            so.draft_token = tok;
            so.draft_logit = 0.0f;
            if (sg->out_logits && state_->draft_topk > 1) {
                const int vocab = (int)sg->out_logits->ne[0];
                std::vector<float> logits_buf((size_t)vocab);
                ggml_backend_tensor_get(sg->out_logits, logits_buf.data(),
                    0, sizeof(float) * vocab);
                so.draft_logit = logits_buf[tok];
                emit_topk_logprobs(logits_buf.data(), vocab,
                                   state_->draft_topk, so);
            }
        } else if (state_->draft_topk > 1) {
            std::vector<float> logits_buf;
            int vocab = 0;
            if (state_->target->project_hidden_to_logits(x_normed.data(), 1,
                                                          logits_buf, vocab) &&
                vocab > 0) {
                so.draft_token = argmax(logits_buf.data(), vocab);
                so.draft_logit = logits_buf[so.draft_token];
                emit_topk_logprobs(logits_buf.data(), vocab,
                                   state_->draft_topk, so);
            } else {
                std::vector<int32_t> tok_out;
                if (!state_->target->project_hidden_to_tokens(x_normed.data(), 1,
                                                               tok_out) ||
                    tok_out.empty()) {
                    std::fprintf(stderr,
                        "[qwen36_mtp gpu chain] project_hidden_to_tokens failed iter=%d\n", it);
                    return false;
                }
                so.draft_token = tok_out[0];
                so.draft_logit = 0.0f;
            }
        } else {
            std::vector<int32_t> tok_out;
            if (!state_->target->project_hidden_to_tokens(x_normed.data(), 1,
                                                           tok_out) ||
                tok_out.empty()) {
                std::fprintf(stderr,
                    "[qwen36_mtp gpu chain] project_hidden_to_tokens failed iter=%d\n", it);
                return false;
            }
            so.draft_token = tok_out[0];
            so.draft_logit = 0.0f;
        }
        out.push_back(so);

        // Stash PRE-shared_head_norm hidden for next iter's h_prev and
        // advance cur_token to the freshly drafted token.  Post-norm here
        // (the previous behaviour) compounded a distribution drift per
        // depth — see the pre-norm-hidden comment above.  Skipped on the
        // final iter (h_pre stays empty when need_h_pre=false).
        if (need_h_pre) {
            state_->last_hidden = std::move(h_pre);
        }
        cur_token = so.draft_token;

#ifdef DFLASH_MTP_PROFILE
        if (prof_on) {
            const double t_total = prof_ms_since(t_iter0);
            std::fprintf(stderr,
                "[mtp_prof iter=%d] embed=%.3f set=%.3f.%.3f.%.3f "
                "compute=%.3f get=%.3f.%.3f argmax=%.3f total=%.3f (ms)\n",
                it, t_embed, t_set_embed, t_set_hprev, t_set_pos,
                t_compute, t_get_x, t_get_h, t_get_argmax, t_total);
            prof_sum_embed   += t_embed;
            prof_sum_set     += t_set_embed + t_set_hprev + t_set_pos;
            prof_sum_compute += t_compute;
            prof_sum_get_x   += t_get_x;
            prof_sum_get_h   += t_get_h;
            prof_sum_get_argmax += t_get_argmax;
            prof_sum_total   += t_total;
            ++prof_iters;
        }
#endif  // DFLASH_MTP_PROFILE
    }

#ifdef DFLASH_MTP_PROFILE
    if (prof_on && prof_iters > 0) {
        const double sum_get = prof_sum_get_x + prof_sum_get_h + prof_sum_get_argmax;
        const double denom   = prof_sum_total > 0.0 ? prof_sum_total : 1.0;
        const double pct_embed   = 100.0 * prof_sum_embed   / denom;
        const double pct_set     = 100.0 * prof_sum_set     / denom;
        const double pct_compute = 100.0 * prof_sum_compute / denom;
        const double pct_get     = 100.0 * sum_get          / denom;
        std::fprintf(stderr,
            "[mtp_prof chain depth=%d iters=%d avg_iter=%.3f ms "
            "breakdown_pct: embed=%.1f%% set=%.1f%% compute=%.1f%% get=%.1f%% "
            "(get_x=%.3f get_h=%.3f get_argmax=%.3f ms total)]\n",
            chain_depth, prof_iters, prof_sum_total / prof_iters,
            pct_embed, pct_set, pct_compute, pct_get,
            prof_sum_get_x, prof_sum_get_h, prof_sum_get_argmax);
    }
#endif  // DFLASH_MTP_PROFILE
    return true;
}

}  // namespace dflash27b::mtp
