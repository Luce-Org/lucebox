#include "qwen4exp_backend.h"
#include "qwen4exp_chunk.h"
#include "qwen4exp_graph.h"

#include "common/moe_hybrid_routing_stats.h"
#include "common/peer_access.h"
#include "common/sampler.h"
#include "common/snapshot_backend.h"
#include "pflash/pflash_window.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace luce::common {

// Prompt lookup (common/prompt_lookup.h) beside the adaptive MTP, in the single-slot loop and the multi-slot engine.
// Burn-in kill switch: LUCE_QWEN4EXP_LOOKUP=0.
static bool prompt_lookup_on() {
    static const bool on = [] {
        const char * e = std::getenv("LUCE_QWEN4EXP_LOOKUP");
        return e == nullptr || std::strcmp(e, "0") != 0;
    }();
    return on;
}

// The expert device's read-ahead of each split layer's picks (qwen4exp_graph.cpp, kReadAheadPicks): per split layer,
// the router rows of the experts the target's slots do not hold at start, as Q8_0 [n_embd, n_cold], and their expert
// ids, on the expert device. A layer without them (or a failed allocation) only loses the read-ahead: the output does
// not depend on it.
static void load_qwen4exp_read_ahead(Qwen4ExpWeights & w) {
    const int64_t qk = ggml_blck_size(GGML_TYPE_Q8_0);
    std::vector<std::vector<int32_t>> cold_ids((size_t) w.n_layer);
    for (int il = 0; il < w.n_layer; ++il) {
        const Qwen4ExpLayer & L = w.layers[(size_t) il];
        const ggml_tensor * R = L.ffn_gate_inp;
        if (!L.ffn_gate_exps || L.ffn_gate_exps->buffer != w.expert_buf || !R || R->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(R) || R->ne[0] != w.n_embd || R->ne[1] != w.n_expert || w.n_embd % qk != 0) continue;
        std::vector<uint8_t> held((size_t) w.n_expert, 0);
        if (w.hot) for (int32_t e : w.hot->initial[(size_t) il]) held[(size_t) e] = 1;
        for (int e = 0; e < w.n_expert; ++e) if (!held[(size_t) e]) cold_ids[(size_t) il].push_back(e);
    }
    w.ahead_router.assign((size_t) w.n_layer, nullptr);
    w.ahead_ids.assign((size_t) w.n_layer, nullptr);
    ggml_context * ctx = w.derived.open(2 * (size_t) w.n_layer);
    for (int il = 0; ctx && il < w.n_layer; ++il) {
        const int64_t n_cold = (int64_t) cold_ids[(size_t) il].size();
        if (n_cold == 0) continue;
        w.ahead_router[(size_t) il] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, w.n_embd, n_cold);
        w.ahead_ids[(size_t) il] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_cold);
    }
    if (!ctx || !w.derived.commit(w.expert_backend)) {
        w.ahead_router.clear();
        w.ahead_ids.clear();
        std::fprintf(stderr, "[qwen4exp] read-ahead: allocation failed, off\n");
        return;
    }
    std::vector<float> all((size_t) w.n_embd * (size_t) w.n_expert), rows;
    std::vector<uint8_t> q;
    size_t bytes = 0;
    int layers = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        if (!w.ahead_router[(size_t) il]) continue;
        const std::vector<int32_t> & ids = cold_ids[(size_t) il];
        ggml_backend_tensor_get(w.layers[(size_t) il].ffn_gate_inp, all.data(), 0, all.size() * sizeof(float));
        rows.resize(ids.size() * (size_t) w.n_embd);
        for (size_t j = 0; j < ids.size(); ++j) {
            std::memcpy(rows.data() + j * (size_t) w.n_embd, all.data() + (size_t) ids[j] * (size_t) w.n_embd,
                        (size_t) w.n_embd * sizeof(float));
        }
        q.resize(ggml_nbytes(w.ahead_router[(size_t) il]));
        ggml_quantize_chunk(GGML_TYPE_Q8_0, rows.data(), q.data(), 0, (int64_t) ids.size(), w.n_embd, nullptr);
        ggml_backend_tensor_set(w.ahead_router[(size_t) il], q.data(), 0, q.size());
        ggml_backend_tensor_set(w.ahead_ids[(size_t) il], ids.data(), 0, ids.size() * sizeof(int32_t));
        bytes += q.size() + ids.size() * sizeof(int32_t);
        ++layers;
    }
    std::fprintf(stderr, "[qwen4exp] read-ahead: routers of %d layers on the expert device (%.1f MiB)\n", layers,
                 bytes / 1048576.0);
}

// Split mode: the LM head's last rows on the expert device, which computes their logits while the target computes
// the others (qwen4exp_graph.cpp, build_lm_head). The verify head (248320 Q8_0 rows) is the round's largest matrix
// and the expert device idles meanwhile. Both GPUs run those products on the RDNA4 Q8_0 matvec
// (ggml_backend_cuda_mul_mat_rdna4_rows), so every logit keeps the bits of the target's single product; without
// that kernel on both, or without room for the rows, the target computes them all.
// The expert device's share. The Strix streams Q8_0 rows at ~230 GB/s against the R9700's ~640, which alone would
// balance at ~26%, but its part also pays the hop in, its wake and the logits back. Measured on an R9700 + Ryzen AI
// Max+ PRO 495 (split decode, same text at every share): 10% +0.14%, 15% +0.39%, 20% +0.54%, 25% +0.05%, 30% -0.6%.
static constexpr int kLmHeadTailPercent = 20;

static void load_qwen4exp_output_tail(Qwen4ExpWeights & w, ggml_backend_t target) {
    const ggml_tensor * out = w.output;
    if (!w.expert_backend || !out || out->type != GGML_TYPE_Q8_0 || !ggml_is_contiguous(out) || out->ne[2] != 1 ||
        out->ne[3] != 1 || !ggml_backend_cuda_rdna4_rows_device(ggml_backend_cuda_get_device_id(target)) ||
        !ggml_backend_cuda_rdna4_rows_device(ggml_backend_cuda_get_device_id(w.expert_backend))) return;
    const int64_t rows = out->ne[1], tail = rows * kLmHeadTailPercent / 100;
    if (tail <= 0 || tail >= rows) return;
    ggml_context * ctx = w.derived.open(1);
    ggml_tensor * t = ctx ? ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, out->ne[0], tail) : nullptr;
    if (!t || !w.derived.commit(w.expert_backend)) {
        std::fprintf(stderr, "[qwen4exp] LM head: no room for its last rows on the expert device\n");
        return;
    }
    std::vector<uint8_t> bytes(ggml_nbytes(t));   // through the host: the two GPUs need not reach each other
    ggml_backend_tensor_get(out, bytes.data(), (size_t) (rows - tail) * out->nb[1], bytes.size());
    ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    w.output_tail = t;
    std::fprintf(stderr, "[qwen4exp] LM head: rows %lld..%lld of %lld on the expert device (%.1f MiB)\n",
                 (long long) (rows - tail), (long long) rows - 1, (long long) rows, bytes.size() / 1048576.0);
}

// Split mode: hot expert slots for `budget` bytes on the target (Qwen4ExpHotSet), started from the routing-stats CSV
// when one is given, and the expert device's read-ahead.
static bool load_qwen4exp_hot_experts(ggml_backend_t backend, Qwen4ExpWeights & w, const std::string & csv,
                                      uint64_t budget) {
    MoeHybridRoutingStats stats;
    if (!csv.empty()) {
        std::string err;
        if (!MoeHybridRoutingStats::load_csv(csv.c_str(), stats, &err) ||
            !stats.matches(w.n_layer, w.n_expert, w.n_expert_used)) {
            std::fprintf(stderr, "[qwen4exp] hot experts: %s: %s\n", csv.c_str(), err.empty() ? "shape mismatch" : err.c_str());
            return false;
        }
    }
    if (budget > 0) {
        auto hot = std::make_unique<Qwen4ExpHotSet>();
        if (hot->init(backend, w, budget, csv.empty() ? nullptr : &stats)) {
            w.hot = std::move(hot);
        } else {
            std::fprintf(stderr, "[qwen4exp] hot experts: no room for a slot per layer (%.2f GiB)\n",
                         (double) budget / (1ull << 30));
        }
    }
    load_qwen4exp_read_ahead(w);
    return true;
}

// Split mode: the last `layers` full-attention layers run their attention on the expert device, beside their cache
// rows (Qwen4ExpCache::remote_layers), so their projection, norm and indexer weights move there. Layers moved before
// stay; the target's copies stay in its weights buffer.
static bool move_qwen4exp_attention(Qwen4ExpWeights & w, int layers) {
    const ggml_backend_buffer_type_t expert = ggml_backend_get_default_buffer_type(w.expert_backend);
    std::vector<ggml_tensor **> fields;
    for (int il = w.n_layer - 1; il >= 0 && layers > 0; --il) {
        Qwen4ExpLayer & L = w.layers[(size_t) il];
        if (!L.is_full_attention) continue;
        --layers;
        for (ggml_tensor ** f : {&L.wq, &L.wk, &L.wv, &L.wo, &L.q_norm, &L.k_norm,
                                 &L.indexer_q_proj, &L.indexer_k_proj, &L.indexer_q_norm, &L.indexer_k_norm}) {
            if (*f && ggml_backend_buffer_get_type((*f)->buffer) != expert) fields.push_back(f);
        }
    }
    if (fields.empty()) return true;
    ggml_context * ctx = w.derived.open(fields.size());
    std::vector<ggml_tensor *> moved;
    for (ggml_tensor ** f : fields) moved.push_back(ctx ? ggml_dup_tensor(ctx, *f) : nullptr);
    if (!ctx || !w.derived.commit(w.expert_backend)) return false;
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < fields.size(); ++i) {   // through the host: the two GPUs need not reach each other
        bytes.resize(ggml_nbytes(*fields[i]));
        ggml_backend_tensor_get(*fields[i], bytes.data(), 0, bytes.size());
        ggml_backend_tensor_set(moved[i], bytes.data(), 0, bytes.size());
        *fields[i] = moved[i];
    }
    return true;
}

// Free memory on a device. The generic UMA query substitutes MemAvailable; retain the HIP/GTT limit too.
static size_t qwen4exp_free_memory(ggml_backend_t backend) {
    size_t free_device = 0, total = 0, free_host = 0;
    ggml_backend_cuda_get_device_memory(ggml_backend_cuda_get_device_id(backend), &free_device, &total);
    ggml_backend_dev_memory(ggml_backend_get_device(backend), &free_host, &total);
    return std::min(free_device, free_host);
}

int qwen4exp_select_chunk(ggml_backend_t backend, const Qwen4ExpWeights & w,
        Qwen4ExpCache & cache, int slots, int resident_slots, size_t * snapshot_budget, Qwen4ExpChunkPlan * plan,
        size_t snapshot_keep, size_t reserved, int restore_spacing) {
    if (slots < 1 || resident_slots < 1 || resident_slots > slots) return 0;
    const size_t free_now = qwen4exp_free_memory(backend);
    const size_t available = free_now > reserved ? free_now - reserved : 0;
    size_t shadow = 0, shadow_tmp = 0;
    if (w.gfx1151) {
        auto count_shadow = [&](ggml_context * ctx) {
            for (auto * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
                if (t->buffer && t->ne[2] == 1 && t->ne[3] == 1 && t->ne[1] <= 32768 &&
                    (t->type == GGML_TYPE_IQ4_NL || t->type == GGML_TYPE_Q6_K || t->type == GGML_TYPE_Q5_K)) {
                    const size_t bytes = (size_t) ggml_nelements(t) * 2;
                    shadow += bytes;
                    if (t->type == GGML_TYPE_Q5_K) shadow_tmp = std::max(shadow_tmp, bytes);
                }
            }
        };
        count_shadow(w.ctx);
        for (auto * ctx : w.extra_meta_ctxs) count_shadow(ctx);
    }
    const size_t state = ggml_backend_buffer_get_size(cache.buf);
    // The other concurrency slots' caches carry the MTP draft layer and the rollback rows of a batched verify
    // (kQwen4ExpBatchedVerifyRows), so each reserves the trunk part of this cache's buffer, the draft K/V and that
    // many of its rollback rows.
    size_t draft_kv = 0, rollback = 0;
    for (const ggml_tensor * t : {cache.mtp_k, cache.mtp_v, cache.mtp_prev_hidden, cache.mtp_chain_hidden,
                                  cache.mtp_chain_ids}) {
        if (t) draft_kv += ggml_nbytes(t);
    }
    if (cache.spec_ple) rollback += ggml_nbytes(cache.spec_ple);
    for (const ggml_tensor * t : cache.spec_conv) rollback += ggml_nbytes(t);
    const size_t mtp_state = draft_kv + rollback;
    const bool mtp = qwen4exp_verify_supported(cache);
    const int verify_rows = std::min(kQwen4ExpBatchedVerifyRows, cache.mtp_draft + 1);
    const size_t slot_state = (state > mtp_state ? state - mtp_state : 0) +
        (mtp ? draft_kv + rollback / (size_t) (cache.mtp_draft + 1) * (size_t) verify_rows : 0);
    const int ratio = w.compress_ratios.empty() ? 1 : std::max(1, *std::max_element(w.compress_ratios.begin(), w.compress_ratios.end()));
    const int dense_end = std::min(cache.max_ctx, w.indexer_top_k + ratio - 1);
    size_t decode = 0, verify = 0, draft = 0, runtime_scratch = 0, runtime_host = 0;
    // Each verify width retains its own graph allocation; their metadata arenas stay mostly untouched.
    // So does each MTP draft graph: rank 0 per catch-up width, and every later rank (sized as width 1).
    std::array<size_t, QWEN4EXP_MTP_MAX_VERIFY + 1> verify_width{}, draft_width{};
    size_t verify_arena = 0;
    auto retain = [&](const Qwen4ExpGraphMemory & m, size_t & resident) {
        if (m.graph == SIZE_MAX) return false;
        resident = std::max(resident, m.graph + m.metadata);
        runtime_scratch = std::max(runtime_scratch, m.scratch);
        runtime_host = std::max(runtime_host, m.host);
        return true;
    };
    // Resident caches (including MTP KV + rollback snapshots) and weights are
    // already deducted from free memory. Reserve all retained graph allocators,
    // including the hidden-output trunk and every admitted catch-up/verify width.
    for (const int end : {dense_end, std::min(cache.max_ctx, dense_end + cache.mtp_draft + 1), cache.max_ctx}) {
        if (!retain(qwen4exp_graph_memory(backend, w, cache, 1, end - 1), decode)) return 0;
        if (mtp) for (int n = 1; n <= cache.mtp_draft + 1 && n <= end; ++n) {
            const Qwen4ExpGraphMemory d = qwen4exp_mtp_graph_memory(backend, w, cache, n, end - n);
            size_t resident = 0;
            if (!retain(d, resident)) return 0;
            draft_width[n] = std::max(draft_width[n], d.graph + d.metadata);
            if (n > 1) {
                const Qwen4ExpGraphMemory m = qwen4exp_graph_memory(backend, w, cache, n, end - n, true);
                if (!retain(m, resident)) return 0;
                verify_width[n] = std::max(verify_width[n], m.graph);
                verify_arena = std::max(verify_arena, m.metadata);
            }
        }
    }
    verify = verify_arena;
    for (const size_t bytes : verify_width) verify += bytes;
    for (const size_t bytes : draft_width) draft += bytes;
    draft += (size_t) cache.mtp_draft * draft_width[1];
    // Every other slot's own draft graphs: rank 0 per catch-up width up to a batched verify's, and its later ranks.
    size_t slot_draft = 0;
    if (mtp) {
        for (int n = 1; n <= verify_rows; ++n) slot_draft += draft_width[n];
        slot_draft += (size_t) (verify_rows - 1) * draft_width[1];
    }
    // A multi-slot engine also keeps its batched decode graph resident: plan its widest
    // shapes, every slot at the end of the dense span, and at the end of the context.
    size_t batched = 0;
    if (slots > 1) {
        const bool qsa_rows = w.qsa && !cache.kvflash;   // a KVFlash pool attends densely
        const int dense_last = (qsa_rows ? dense_end : cache.max_ctx) - 1;
        for (const bool qsa : {false, true}) {
            if (qsa && !qsa_rows) continue;
            const Qwen4ExpGraphMemory m = qwen4exp_batched_graph_memory(
                backend, w, cache, slots, qsa ? cache.max_ctx - 1 : dense_last, qsa);
            size_t resident = 0;
            if (!retain(m, resident)) return 0;
            batched = std::max(batched, resident);
        }
        // And the batched verify graph, at the same two shapes.
        size_t batched_verify = 0;
        for (const bool qsa : {false, true}) {
            if (!mtp || (qsa && !qsa_rows)) continue;
            const int end = qsa ? cache.max_ctx : dense_last + 1;
            const Qwen4ExpGraphMemory m = qwen4exp_batched_verify_graph_memory(
                backend, w, cache, slots, verify_rows, std::max(0, end - verify_rows));
            size_t resident = 0;
            if (!retain(m, resident)) return 0;
            batched_verify = std::max(batched_verify, resident);
        }
        batched += batched_verify;
        // And the batched draft's later ranks (qwen4exp_mtp_draft_slots: ranks 1 .. verify_rows - 2), each a graph of
        // every slot's row, attending over the widest draft K/V span.
        if (mtp && verify_rows > 2) {
            const Qwen4ExpGraphMemory m = qwen4exp_mtp_rank_slots_graph_memory(backend, w, cache, slots,
                                                                                cache.max_ctx - 1);
            size_t resident = 0;
            if (!retain(m, resident)) return 0;
            batched += (size_t) (verify_rows - 2) * resident;
        }
    }
    // The decode workspaces stay resident while the next prompt prefills.
    const size_t fixed = (slots - resident_slots) * slot_state + (slots - 1) * slot_draft + shadow + shadow_tmp +
                         slots * decode + verify + draft + batched;
    std::fprintf(stderr, "[qwen4exp] chunk-runtime mtp=%d decode=%zu verify=%zu draft=%zu batched=%zu scratch=%zu host=%zu\n",
        (int) mtp, decode, verify, draft, batched, runtime_scratch, runtime_host);
    // The largest chunk. Concurrent serving prefills in fixed granules (see Qwen4ExpSeqEngine): on one GPU a
    // 2048-row granule halves a live stream's stall behind a long prompt for about 3% of prompt throughput; with the
    // experts on a second GPU it would cost 12%, so that placement keeps one pipeline stream's rows. Split mode: the
    // expert device's MoE id helper bounds a chunk (hot experts only take the memory the largest chunk leaves).
    const int max_rows = std::min(qwen4exp_max_forward_rows(cache),
        slots > 1 ? (w.expert_backend ? kQwen4ExpPipelineStreamRows : kQwen4ExpConcurrentGranule)
                  : !w.expert_backend ? kQwen4ExpMaxChunk : kQwen4ExpSplitMaxChunk);
    struct Plan { size_t graph = 0, ring = 0, host = 0, scratch = 0; };
    std::vector<std::pair<int, Plan>> plans;   // one per probed chunk size
    auto chunk_workspace = [&](int n) {
        size_t graph = 0, inputs = 0, mask = 0, host = runtime_host, scratch = runtime_scratch;
        // End-of-context QSA workspace, and the largest dense span before QSA.
        // Include an unaligned context tail, which can fall back to dense FA.
        auto measure = [&](int rows, int end) {
            if (rows <= 0 || end <= 0) return true;
            rows = std::min(rows, end);
            const auto m = qwen4exp_graph_memory(backend, w, cache, rows, end - rows);
            if (m.graph == SIZE_MAX) return false;
            graph = std::max(graph, m.graph + m.metadata);
            inputs = std::max(inputs, m.inputs);
            mask = std::max(mask, m.mask);
            // m.host: two chunks' host inputs, the running chunk's and its lookahead's. With concurrent slots the
            // seq engine can also hold every other slot's finished lookahead (Qwen4ExpSeqEngine::prompt_inputs).
            host = std::max(host, slots > 1 ? m.host / 2 * (size_t) (slots + 1) : m.host);
            scratch = std::max(scratch, m.scratch);
            return true;
        };
        for (const int end : {n, dense_end, std::min(cache.max_ctx, dense_end + n),
                             cache.max_ctx - cache.max_ctx % ratio, cache.max_ctx}) {
            if (!measure(n, end)) return SIZE_MAX;
        }
        // Partial final chunks can cross the MMB (512) or packed QSA (128)
        // dispatch floors. Their F32 buffers/masks can outgrow the full chunk.
        for (const int tail : {511, 127, 1}) {
            if (tail < n && !measure(tail, cache.max_ctx)) return SIZE_MAX;
        }
        const size_t workspace = graph + slots * (inputs + mask) + host + scratch;
        std::fprintf(stderr, "[qwen4exp] chunk-plan rows=%d graph=%zu ring=%zu host=%zu scratch=%zu required=%zu available=%zu\n",
            n, graph, slots * (inputs + mask), host, scratch, fixed + workspace, available);
        return workspace;
    };
    const int floor_rows = w.expert_backend ? kQwen4ExpSplitChunkFloor : 4096;
    if (plan) {
        const size_t floor = chunk_workspace(std::min({floor_rows, max_rows, cache.max_ctx}));
        plan->available = available;
        plan->required = floor == SIZE_MAX ? SIZE_MAX : fixed + floor + available / 10;
        // Every slot's attention and indexer rows on the target; only this cache adds the MTP draft layer's.
        size_t rows = 0, layer = 0;
        for (const auto * group : {&cache.attn_k, &cache.attn_v, &cache.indexer_raw, &cache.indexer_k}) {
            for (const ggml_tensor * t : *group) if (t && t->buffer == cache.buf) rows += ggml_nbytes(t);
            if (!group->empty() && group->front()) layer += ggml_nbytes(group->front());
        }
        const size_t mtp_rows = (cache.mtp_k ? ggml_nbytes(cache.mtp_k) : 0) + (cache.mtp_v ? ggml_nbytes(cache.mtp_v) : 0);
        plan->position_bytes = ((size_t) slots * rows + mtp_rows) / cache.max_ctx + 1;
        plan->layer_bytes = (size_t) slots * layer;
        if (plan->required > available) return 0;   // the context fit moves on without searching smaller chunks
    }
    const auto restore_point_spacing = [restore_spacing] { return restore_spacing; };
    const size_t min_snapshot = snapshot_budget
        ? qwen4exp_snapshot_bytes(
              backend, cache,
              std::min(cache.max_ctx, restore_point_spacing()))
        : 0;
    const int reserved_rows = !snapshot_budget ? floor_rows
        : qwen4exp_reserved_rows(cache.max_ctx, available, fixed, chunk_workspace, floor_rows,
                                 std::min(kQwen4ExpPreferredChunk, max_rows), std::min(*snapshot_budget, snapshot_keep));
    const int chunk = qwen4exp_fit_chunk(cache.max_ctx, available, fixed, chunk_workspace, snapshot_budget,
                                         reserved_rows, max_rows, min_snapshot);
    if (plan && chunk > 0) {
        const size_t workspace = chunk_workspace(plan->rows > 0 ? plan->rows : chunk);
        plan->chunk_bytes = workspace == SIZE_MAX ? SIZE_MAX
                          : fixed + (snapshot_budget ? *snapshot_budget : 0) + workspace;
    }
    std::fprintf(stderr, "[qwen4exp] chunk-auto ctx=%d slots=%d resident=%d chunk=%d state=%zu fixed=%zu available=%zu headroom=%zu\n",
        cache.max_ctx, slots, resident_slots, chunk, state,
        fixed + (snapshot_budget ? *snapshot_budget : 0), available, available / 10);
    return chunk;
}

Qwen4ExpBackend::Qwen4ExpBackend(Qwen4ExpBackendConfig cfg)
    : cfg_(std::move(cfg)) {}

Qwen4ExpBackend::~Qwen4ExpBackend() {
    shutdown();
}

bool Qwen4ExpBackend::init() {
    if (cfg_.verify_width < 0 || cfg_.verify_width > QWEN4EXP_MTP_MAX_VERIFY) {
        std::fprintf(stderr, "[qwen4exp] --verify-width must be 0..%d\n", QWEN4EXP_MTP_MAX_VERIFY);
        return false;
    }
    if (cfg_.max_concurrency < 1 || cfg_.max_concurrency > 4) {
        std::fprintf(stderr, "[qwen4exp] --max-concurrency must be between 1 and 4\n");
        return false;
    }
    // MTP stays on with concurrent slots: the first slot's cache drafts and
    // verifies whenever its request decodes alone (Qwen4ExpSeqEngine).
    if (cfg_.device.is_layer_split()) {
        std::fprintf(stderr, "[qwen4exp] layer split is not supported yet\n");
        return false;
    }
    backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!backend_) {
        std::fprintf(stderr, "[qwen4exp] backend init failed for GPU %d\n",
                     cfg_.device.gpu);
        return false;
    }
    // Prefix snapshots stay in the gfx1151 iGPU's memory, which is the system's. Beside a GPU with its own memory
    // they go to system memory (create_snapshot_backend), which leaves that memory to the KV caches.
    snap_backend_ = ggml_backend_cuda_qwen4exp_supported(backend_) ? backend_ : create_snapshot_backend(backend_);
    if (!snap_backend_) return false;
    if (!cfg_.expert_placement_path.empty() && !cfg_.expert_device) {
        std::fprintf(stderr, "[qwen4exp] --expert-placement needs --expert-device\n");
        return false;
    }
    // Split mode: routed experts on a second GPU, everything else on the target.
    if (cfg_.expert_device) {
        const int expert_gpu = cfg_.expert_device->gpu;   // the plan checked it differs from the target
        weights_.expert_backend = ggml_backend_cuda_init(expert_gpu);
        if (!weights_.expert_backend) {
            std::fprintf(stderr, "[qwen4exp] expert backend init failed for GPU %d\n", expert_gpu);
            return false;
        }
        weights_.expert_gfx1151 = ggml_backend_cuda_qwen4exp_supported(weights_.expert_backend);
        // Direct device-to-device copies for the activations crossing between the two GPUs.
        if (!enable_peer_access_pair(cfg_.device.gpu, expert_gpu)) {
            std::fprintf(stderr, "[qwen4exp] peer access between GPU %d and %d unavailable; copies stage through the host\n",
                         cfg_.device.gpu, expert_gpu);
        }
    }
    if (cfg_.max_concurrency > 1 && !cfg_.expert_placement_path.empty())
        std::fprintf(stderr, "[qwen4exp] --expert-placement unused with --max-concurrency > 1: every pick stays on the expert device\n");
    return load_target();
}

bool Qwen4ExpBackend::load_target() {
    if (!load_qwen4exp_gguf(cfg_.model_path, backend_, weights_,
                           cfg_.verify_width == 1 ? "0" : cfg_.draft_path.value_or(""))) {
        std::fprintf(stderr, "[qwen4exp] model load failed: %s\n",
                     luce_last_error());
        return false;
    }
    load_qwen4exp_output_tail(weights_, backend_);
    size_kvflash_pool();
    if (!create_main_cache()) {
        std::fprintf(stderr, "[qwen4exp] cache creation failed\n");
        return false;
    }
    if (remote_layers_ && !move_qwen4exp_attention(weights_, remote_layers_)) {
        std::fprintf(stderr, "[qwen4exp] moving attention to the expert device failed\n");
        return false;
    }
    // --max-ctx auto: the trained context, or the largest that fits (Qwen4ExpContextFit) with every slot's cache, the
    // resident graphs, a floor-sized prompt chunk and, with snapshots on the target, one full snapshot per slot. In
    // split mode a miss first moves the fewest full-attention layers' cache rows and attention to the expert device
    // that cover it, while that device holds them; the context shrinks once none can move. The layout found stays:
    // an unpark sizes itself for it.
    for (Qwen4ExpContextFit search;;) {
        Qwen4ExpChunkPlan plan;
        chunk_ = plan_chunk(cfg_.device.fit_ctx ? &plan : nullptr);
        if (!cfg_.device.fit_ctx) break;
        const int ctx = cache_.max_ctx;
        const size_t snapshots = snap_backend_ == backend_
            ? (size_t) cfg_.max_concurrency * snapshot_bytes_estimate(ctx) : 0;
        // The planner kept the PFlash drafter's room free; it grows with the context too.
        const size_t position = plan.position_bytes + snapshots / ctx +
                                (pflash_reserve(backend_) ? cfg_.pflash_bytes_per_token : 0);
        const bool fit = plan.required <= plan.available && snapshots <= plan.available - plan.required;
        const size_t missing = fit ? 0 : plan.required == SIZE_MAX ? (size_t) ctx / 4 * position
                                                                    : plan.required + snapshots - plan.available;
        int next = 0;
        if (fit) {
            search.fits = ctx;
            search.spare = plan.available - plan.required - snapshots;
        } else if (const int remote = remote_layers_for(plan, missing); remote > remote_layers_) {
            remote_layers_ = remote;
            if (!move_qwen4exp_attention(weights_, remote_layers_)) {
                std::fprintf(stderr, "[qwen4exp] moving attention to the expert device failed\n");
                return false;
            }
            next = ctx;   // plan the same context with the rows moved
        } else {
            search.misses = ctx;
            search.missing = missing;
        }
        if (!next) next = search.next(position);
        if (!next && !search.fits) {
            std::fprintf(stderr, "[qwen4exp] no context fits beside the weights\n");
            return false;
        }
        if (!next) {
            cfg_.device.fit_ctx = false;
            std::fprintf(stderr, "[qwen4exp] context fitted to the devices: %d tokens, %d at once, %d of %zu "
                         "attention layers on the expert device\n", search.fits, cfg_.max_concurrency, remote_layers_,
                         cache_.full_layer_ids.size());
            if (fit) break;
            next = search.fits;   // the last plan missed: plan the fit again
        }
        free_qwen4exp_cache(cache_);
        cache_ = {};
        cfg_.device.max_ctx = next;
        if (!create_main_cache()) {
            std::fprintf(stderr, "[qwen4exp] cache creation failed\n");
            return false;
        }
    }
    if (!load_hot_experts()) return false;
    if (snapshot_budget_ == 0)
        std::fprintf(stderr, "[qwen4exp] prefix snapshots disabled: memory allowance insufficient for one checkpoint\n");
    else if (snapshot_budget_ != SIZE_MAX)
        std::fprintf(stderr, "[qwen4exp] prefix snapshot allowance=%zu bytes\n", snapshot_budget_);
    else if (snap_backend_ != backend_)
        std::fprintf(stderr, "[qwen4exp] prefix snapshots in system memory\n");
    if (chunk_ <= 0) {
        std::fprintf(stderr, "[qwen4exp] insufficient prefill memory at the configured context\n");
        return false;
    }
    if (!start_seq_engine()) {
        std::fprintf(stderr, "[qwen4exp] full-cache slot allocation failed\n");
        return false;
    }
    if (cache_.kvflash) {
        std::printf("[kvflash] resident pool %d tokens over %zu full-attn layers per slot (logical max_ctx %d, %d "
                    "slots, prompt forwards <= %d rows), policy=lru%s\n",
                    cache_.kvflash->pool_tokens(), cache_.full_layer_ids.size(), cache_.max_ctx,
                    cfg_.max_concurrency, qwen4exp_max_forward_rows(cache_),
                    kvflash_policy_is_lru() ? "" : " (qwen4exp has no residency scorer yet)");
    } else if (kvflash_pool_ > 0) {
        std::printf("[kvflash] pool %d holds the whole %d-token context: one row per position\n",
                    kvflash_pool_, cache_.max_ctx);
    }
    std::fflush(stdout);
    return true;
}

// Split mode, one sequence, with --expert-placement: hot expert slots on the target for decode and verify steps (a
// prompt chunk keeps every pick on the expert device), started from the placement and then following the served
// picks (Qwen4ExpHotSet). They take the memory the fitted context and the prompt chunk leave, at most
// LUCE_EXPERT_BUDGET_MB, so they shrink neither. Concurrent serving keeps every pick on the expert device, so a
// request computes the same alone and beside other requests: no hot copies. The expert device's read-ahead (no
// effect on the output) is built either way.
bool Qwen4ExpBackend::load_hot_experts() {
    if (!weights_.expert_backend || chunk_ <= 0)
        return true;
    if (cfg_.expert_placement_path.empty() || cfg_.max_concurrency > 1)
        return load_qwen4exp_hot_experts(backend_, weights_, std::string(), 0);
    Qwen4ExpChunkPlan room;
    room.rows = chunk_;
    plan_chunk(&room);
    const size_t needed = room.chunk_bytes == 0 || room.chunk_bytes == SIZE_MAX ? SIZE_MAX
                        : room.chunk_bytes + room.chunk_bytes / 9 + kQwen4ExpHotExpertMargin;   // the plan's 10%
    const uint64_t spare = needed < room.available ? room.available - needed : 0;
    const uint64_t budget = std::min<uint64_t>(expert_budget_bytes_from_env(UINT64_MAX), spare);
    std::fprintf(stderr, "[qwen4exp] hot experts: %.2f GiB beside a %d-row prompt chunk (available %.2f GiB)\n",
                 (double) budget / (1ull << 30), chunk_, (double) room.available / (1ull << 30));
    if (!load_qwen4exp_hot_experts(backend_, weights_, cfg_.expert_placement_path, budget)) return false;
    if (!weights_.hot) return true;
    chunk_ = plan_chunk();   // the decode and verify graphs grew with the hot experts' lanes
    return chunk_ > 0;
}

// Memory the PFlash drafter needs on `device` when it scores there: its weights
// and a workspace that grows with the prompt, which can fill the context.
size_t Qwen4ExpBackend::pflash_reserve(ggml_backend_t device, int ctx) const {
    if (cfg_.pflash_gpu < 0 || !device || ggml_backend_cuda_get_device_id(device) != cfg_.pflash_gpu) return 0;
    return cfg_.pflash_bytes + (size_t) std::max(0, ctx < 0 ? cache_.max_ctx : ctx) * cfg_.pflash_bytes_per_token;
}

// The prompt chunk, measured against free memory unless --chunk sets it (concurrent serving measures either way,
// to size the prefix allowance; so does a context fit, which reads `plan`).
int Qwen4ExpBackend::plan_chunk(Qwen4ExpChunkPlan * plan) {
    snapshot_budget_ = SIZE_MAX;
    // Snapshots in the target's memory get an allowance the planner sizes and captures respect. In system memory
    // the server's resident limit bounds them.
    auto allowance = [this](size_t snapshots) -> size_t * {
        if (snap_backend_ != backend_) return nullptr;
        snapshot_budget_ = snapshots * snapshot_bytes_estimate(cache_.max_ctx);
        return &snapshot_budget_;
    };
    const int explicit_chunk = std::min({cfg_.chunk, qwen4exp_max_forward_rows(cache_),
                                         weights_.expert_backend ? kQwen4ExpSplitMaxChunk : cfg_.chunk});
    if (cfg_.chunk > 0 && cfg_.max_concurrency == 1 && !plan) return explicit_chunk;
    // Prefix checkpoints with concurrent slots: each slot's restore point and capture in flight plus one shared
    // head, the server's concurrent prefix budget. One sequence keeps three.
    const int slots = cfg_.max_concurrency;
    const int fit = qwen4exp_select_chunk(backend_, weights_, cache_, slots, 1,
                                          allowance(slots > 1 ? 2 * (size_t) slots + 1 : 3), plan,
                                          snapshot_bytes_estimate(cache_.max_ctx),
                                          pflash_reserve(backend_),
                                          restore_point_spacing());
    return cfg_.chunk > 0 && fit > 0 ? explicit_chunk : fit;
}

void Qwen4ExpBackend::release_target() {
    for (int i = 0; i < kMaxSlots; ++i) snapshot_free(i);
    tokens_.clear(); logits_.clear();
    seq_engine_.reset();
    for (Qwen4ExpCache & cache : seq_caches_) free_qwen4exp_cache(cache);
    seq_caches_.clear();
    free_qwen4exp_cache(cache_);
    free_qwen4exp_slot_states(slot_states_);
    free_qwen4exp_weights(weights_);
    ggml_backend_cuda_trim_pool(backend_); // also frees the bf16 weight shadows keyed by the freed weights' addresses
}

// Slot 0 of the concurrency engine; with concurrent slots its recurrent state is
// the first slab of the slot states the multi-slot decode graph batches over.
bool Qwen4ExpBackend::create_main_cache() {
    // The slot states do not depend on the context, so a context fit's rebuilt cache keeps them.
    if (cfg_.max_concurrency > 1 && !slot_states_.buf &&
        !create_qwen4exp_slot_states(backend_, weights_, cfg_.max_concurrency, slot_states_)) return false;
    return create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx, cache_, /*mtp=*/true,
        cfg_.verify_width == 0 ? QWEN4EXP_MTP_MAX_DRAFT : std::max(1, cfg_.verify_width - 1),
        slot_states_.ctx ? &slot_states_ : nullptr, 0, remote_layers_, kvflash_pool_for(cfg_.device.max_ctx));
}

// --kvflash: each slot's pool, sized with the weights resident and no cache yet ("auto" takes a share of the free
// memory). It holds the full-attention layers' F16 K/V; the MTP draft layer keeps one row per position.
void Qwen4ExpBackend::size_kvflash_pool() {
    const Qwen4ExpWeights & w = weights_;
    int64_t n_full = 0;
    for (int il = 0; il < w.n_layer; ++il) n_full += w.layers[il].is_full_attention;
    KvFlashAutoBudget budget;
    budget.free_bytes = (int64_t) qwen4exp_free_memory(backend_);
    // The decode graphs and a prompt chunk, which the planner sizes after, and a PFlash drafter on this GPU.
    // No cache yet: the drafter's per-token room is counted at the configured context.
    budget.reserve_bytes = (2ll << 30) + (int64_t) pflash_reserve(backend_, cfg_.device.max_ctx);
    budget.bytes_per_token = cfg_.max_concurrency * n_full * w.n_head_kv *
        (int64_t) (ggml_row_size(GGML_TYPE_F16, w.n_embd_head_k) + ggml_row_size(GGML_TYPE_F16, w.n_embd_head_v));
    kvflash_pool_ = kvflash_pool_from_env(cfg_.device.max_ctx, KvFlashConfig{}, /*scorer_expected=*/false, budget);
}

// A pool that holds the whole context pages nothing: such a context keeps one row per position (and QSA).
int Qwen4ExpBackend::kvflash_pool_for(int ctx) const {
    return kvflash_pool_ < ctx ? kvflash_pool_ : 0;
}

// --max-ctx auto in split mode: the full-attention layers to keep on the expert device so a plan that lacks
// `missing` bytes fits, when that device holds every slot's rows of them beside its own headroom (else the layers
// already there).
int Qwen4ExpBackend::remote_layers_for(const Qwen4ExpChunkPlan & plan, size_t missing) const {
    const int n_full = (int) cache_.full_layer_ids.size();
    // A KVFlash pool pages one device's rows; it is small enough to stay on the target.
    if (!weights_.expert_backend || !plan.layer_bytes || plan.required == SIZE_MAX || remote_layers_ >= n_full ||
        cache_.kvflash)
        return remote_layers_;
    const int remote = std::min<int>(n_full, remote_layers_ + (int) ((missing + plan.layer_bytes - 1) / plan.layer_bytes));
    const size_t reserved = pflash_reserve(weights_.expert_backend);
    const size_t free_now = qwen4exp_free_memory(weights_.expert_backend) +
                            (cache_.remote_buf ? ggml_backend_buffer_get_size(cache_.remote_buf) : 0);
    const size_t available = free_now > reserved ? free_now - reserved : 0;
    return (size_t) remote * plan.layer_bytes <= available - available / 10 ? remote : remote_layers_;
}

bool Qwen4ExpBackend::start_seq_engine() {
    if (cfg_.max_concurrency <= 1) return true;
    seq_caches_.resize((size_t)cfg_.max_concurrency - 1);
    std::vector<Qwen4ExpCache *> caches{&cache_};
    for (Qwen4ExpCache & cache : seq_caches_) {
        // With MTP, each slot drafts and verifies too (Qwen4ExpSeqEngine): the rollback rows of a batched verify.
        const bool mtp = cfg_.verify_width != 1 && qwen4exp_verify_supported(cache_);
        if (!create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx, cache, mtp,
                                   mtp ? std::min(kQwen4ExpBatchedVerifyRows, cache_.mtp_draft + 1) - 1 : 1,
                                   &slot_states_, (int) caches.size(), remote_layers_,
                                   kvflash_pool_for(cfg_.device.max_ctx))) return false;
        cache.split_owner = &cache_;
        caches.push_back(&cache);
    }
    seq_engine_ = std::make_unique<Qwen4ExpSeqEngine>(
        backend_, weights_, std::move(caches), cfg_.device.max_ctx, chunk_, snapshot_budget_,
        cfg_.verify_width, &mtp_width_, &mtp_costs_, snap_backend_, prompt_lookup_on() ? &lookup_policy_ : nullptr);
    std::fprintf(stderr,
        "[qwen4exp-seq] independent-slot engine enabled: %d full F16 caches, ctx=%d, chunk=%d\n",
        cfg_.max_concurrency, cfg_.device.max_ctx, chunk_);
    return true;
}

void Qwen4ExpBackend::print_ready_banner() const {
    const Qwen4ExpWeights & w = weights_;
    std::printf(
        "[qwen4exp-daemon] ready layers=%d linear=%d full=%d experts=%d/%d "
        "hc=%d/%d ple=%zu ngram=%d ctx=%d\n",
        w.n_layer,
        w.n_layer - w.n_layer / w.full_attention_interval,
        w.n_layer / w.full_attention_interval,
        w.n_expert_used, w.n_expert, w.n_hc, w.hc_lowrank,
        w.ple_layer_ids.size(), w.ple_ngram_size, cfg_.device.max_ctx);
    std::fflush(stdout);
}

bool Qwen4ExpBackend::park(ParkTarget target) {
    if (target != ParkTarget::TargetModel && target != ParkTarget::All) {
        return false;
    }
    if (parked_) return true;
    release_target();
    parked_ = true;
    std::printf("[qwen4exp] target parked\n");
    std::fflush(stdout);
    return true;
}

bool Qwen4ExpBackend::unpark(ParkTarget target) {
    if (target != ParkTarget::TargetModel && target != ParkTarget::All) {
        return false;
    }
    if (!parked_) return true;
    // The same bring-up as init (the resolved serving policy is kept), sized for the memory free now.
    if (!load_target()) {
        std::fprintf(stderr, "[qwen4exp] unpark failed; nothing stays resident\n");
        release_target();
        return false;
    }
    parked_ = false;
    std::printf("[qwen4exp] target unparked\n");
    std::fflush(stdout);
    return true;
}

GenerateResult Qwen4ExpBackend::generate_impl(const GenerateRequest & req,
                                              const DaemonIO & io) {
    snapshot_flush_deferred();
    tokens_.clear(); logits_.clear();
    if (!parked_) reset_qwen4exp_state(backend_, cache_);
    return run(req, io, 0);
}

GenerateResult Qwen4ExpBackend::run(const GenerateRequest & req, const DaemonIO & raw_io, int restored, int restore_slot) {
    GenerateResult result;
    result.restored_prefix_tokens = restored;
    const DaemonIO io = raw_io.with_token_callback(req.on_token);
    // A failed graph may have changed recurrent state. Never publish it.
    struct Guard {
        Qwen4ExpBackend & b;
        bool complete = false;
        ~Guard() { if (!complete) { b.tokens_.clear(); b.logits_.clear(); } }
    } guard{*this};
    if (parked_) {
        result.fail(GenerateErrorCode::ModelParked, "qwen4exp target is parked");
        return result;
    }
    if (req.prompt.empty()) {
        result.fail(GenerateErrorCode::PrefillFailed, "empty prompt");
        return result;
    }
    if (req.prompt.size() > (size_t) cache_.max_ctx ||
        (int64_t) req.prompt.size() + std::max(0, req.n_gen - 1) > cache_.max_ctx) {
        result.fail(GenerateErrorCode::ContextOverflow, "qwen4exp request exceeds max_ctx");
        return result;
    }

    std::vector<float> logits = logits_;
    const int chunk = chunk_;
    int pos = restored;

    // MTP speculation (sidecar loaded, default graph, a budget that leaves room for a draft): the draft head
    // predicts x_{p+2} from the pair (h_p, x_{p+1}), h_p being the trunk's final HC residual at p. The pairs not yet
    // run through the draft layer are pending (a pair's token arrives with the next forward).
    const bool mtp = qwen4exp_verify_supported(cache_);
    const bool spec = !req.force_ar_decode && req.n_gen > 2 && mtp;
    const size_t hd = (size_t) weights_.n_embd * weights_.n_hc;
    std::vector<float> hidden;
    Qwen4ExpMtpPending mtp_pending;
    mtp_pending.pos = std::max(0, pos - 1);
    if (mtp && pos > 0) {
        mtp_pending.h.resize(hd);
        ggml_backend_tensor_get(cache_.mtp_prev_hidden, mtp_pending.h.data(), 0, hd * sizeof(float));
    }

    // Each restore point starts a chunk in both cold and resumed requests.
    // Capture cuts are restore points too, including a one-token interval.
    std::vector<int> cuts = req.restore_points;
    if (req.snap_slot >= 0 && req.snap_pos > 0) cuts.push_back(req.snap_pos);
    std::sort(cuts.begin(), cuts.end());
    auto capture = [&]() {
        if (req.snap_slot >= 0 && req.snap_pos == pos) snapshot_save_replacing(req.snap_slot, restore_slot);
    };

    const auto t_pre0 = std::chrono::steady_clock::now();
    auto chunk_end = [&](int offset) {
        int end = offset + std::min(chunk, (int) req.prompt.size() - offset);
        const auto cut = std::upper_bound(cuts.begin(), cuts.end(), offset);
        if (cut != cuts.end()) end = std::min(end, *cut);
        return end;
    };
    // Exactly one host-only gather ahead. A future joins before returning on
    // failure/cancellation, so neither the prompt nor the reader can be freed
    // while a worker is using it. No GPU or cache mutation on that thread.
    auto prepare = [&](int offset) {
        const int start = std::max(0, offset - std::max(0, weights_.ple_ngram_size - 1));
        std::vector<int32_t> prev(req.prompt.begin() + start, req.prompt.begin() + offset);
        return qwen4exp_prepare_inputs(weights_, req.prompt.data() + offset,
            chunk_end(offset) - offset, prev);
    };
    capture();
    auto inputs = pos < (int) req.prompt.size() ? prepare(pos) : Qwen4ExpInputs{};
    while (pos < (int) req.prompt.size()) {
        if (io.is_cancelled()) {
            result.fail(GenerateErrorCode::Cancelled, "cancelled during prefill");
            return result;
        }
        const int end = chunk_end(pos);
        const int n = end - pos;
        std::future<Qwen4ExpInputs> pending;
        if (end < (int) req.prompt.size()) pending = std::async(std::launch::async, prepare, end);
        const bool mtp_prefill = mtp && n > 1;
        const Qwen4ExpForwardResult r = qwen4exp_forward(
            backend_, weights_, cache_, req.prompt.data() + pos, n, pos, logits, mtp ? &hidden : nullptr,
            false, mtp_prefill, &inputs);
        if (!r.ok) {
            result.fail(GenerateErrorCode::PrefillFailed, "qwen4exp prefill forward failed");
            return result;
        }
        if (mtp_prefill) {
            mtp_pending.h.swap(hidden);
            mtp_pending.pos = pos + n - 1;
        } else if (mtp) {   // single-row chunk: retain the original catch-up path
            mtp_pending.tok.assign(req.prompt.begin() + pos + (pos == 0 ? 1 : 0), req.prompt.begin() + pos + n);
            mtp_pending.h.insert(mtp_pending.h.end(), hidden.begin(), hidden.end());
            if (!qwen4exp_mtp_catch_up(backend_, weights_, cache_, mtp_pending, 0)) {
                result.fail(GenerateErrorCode::PrefillFailed, "qwen4exp MTP catch-up failed");
                return result;
            }
            ggml_backend_tensor_set(cache_.mtp_prev_hidden, mtp_pending.h.data(), 0, hd * sizeof(float));
            cache_.mtp_prev_pos = pos;
        }
        pos += n;
        tokens_.insert(tokens_.end(), req.prompt.begin() + pos - n, req.prompt.begin() + pos);
        logits_ = logits;
        capture();
        if (io.is_cancelled()) {
            result.fail(GenerateErrorCode::Cancelled, "cancelled during prefill");
            return result;
        }
        if (pending.valid()) inputs = pending.get();
    }
    const auto t_pre1 = std::chrono::steady_clock::now();
    result.prefill_s = std::chrono::duration<double>(t_pre1 - t_pre0).count();
    if (req.n_gen <= 0) {
        guard.complete = true;
        result.succeed();
        return result;
    }

    std::mt19937_64 rng(request_rng_seed(req.sampler));
    std::vector<int32_t> history = req.prompt;
    auto sample = [&](const float * row) {
        return (int32_t) sample_logits(row, weights_.n_vocab, req.sampler, history, rng);
    };
    ThinkingBudget budget(req.budget_hook, req.n_gen);
    bool cancelled = false;
    // MTP compares this substituted token with its draft before retaining the
    // next row, so matching forced drafts are safe and mismatches cut the block.
    // False ends generation. The same controller also owns the prefill seed.
    auto commit = [&](int32_t & tok) {
        tok = budget.apply(tok).token;
        result.budget_forced_close = budget.forced_close();
        result.tokens.push_back(tok);
        io.emit(tok);
        if (io.is_cancelled()) {
            cancelled = true;
            return false;
        }
        return tok != weights_.eos_id && tok != weights_.eos_chat_id && (int) result.tokens.size() < req.n_gen;
    };

    long long drafts = 0, accepted = 0, steps = 0;
    std::array<long long, QWEN4EXP_MTP_MAX_VERIFY> width_steps{};
    const int decode_ctx = pos;
    const bool adaptive = spec && cfg_.verify_width == 0;
    if (adaptive) {
        mtp_costs_.load(mtp_width_, decode_ctx);
        mtp_width_.carry_acceptance(QWEN4EXP_MTP_CARRIED_TRIALS);
    }
    const bool use_lookup = adaptive && prompt_lookup_on();
    long long lookup_rounds = 0, lookup_drafts = 0, lookup_accepted = 0;
    double draft_s = 0.0;
    auto verify_graphs = [&](uint64_t Qwen4ExpDecodeWorkspace::*count) {
        uint64_t n = 0;
        for (const auto & ws : cache_.verify_workspace) n += ws.*count;
        return n;
    };
    auto draft_graphs = [&](uint64_t Qwen4ExpDecodeWorkspace::*count) {
        uint64_t n = 0;
        for (const auto & ws : cache_.mtp_catchup_workspace) n += ws.*count;
        for (const auto & ws : cache_.mtp_rank_workspace) n += ws.*count;
        return n;
    };
    const uint64_t builds0 = verify_graphs(&Qwen4ExpDecodeWorkspace::builds);
    const uint64_t replays0 = verify_graphs(&Qwen4ExpDecodeWorkspace::replays);
    const uint64_t draft_builds0 = draft_graphs(&Qwen4ExpDecodeWorkspace::builds);
    const uint64_t draft_replays0 = draft_graphs(&Qwen4ExpDecodeWorkspace::replays);
    const auto t_dec0 = std::chrono::steady_clock::now();
    int32_t next = sample(logits.data());
    bool more = req.n_gen > 0 && commit(next);
    if (mtp) mtp_pending.tok.assign(1, next);
    if (use_lookup) {   // the sequence so far: the prompt and the next input
        lookup_.reset();
        lookup_.append(history.data(), history.size());
        lookup_.append(&next, 1);
    }
    // Each trunk row sampled (lazily, in qwen4exp_mtp_step) updates history and applies the budget hook exactly
    // once per emitted token. No RNG draws for unvisited verify rows, and no reads of them: a greedy request takes
    // each row's pick from the device (tok already holds it, the host scan's pick); a top_k draw first tries the
    // row's top candidates, the same draw sample_logits makes from the whole row, which it reads only when the
    // candidates cannot settle the cut. Kill switch: LUCE_GPU_ARGMAX=0 reads every row back.
    const Qwen4ExpRows rows = qwen4exp_rows_for(req.sampler, decode_check_ != nullptr);
    const int top_n = rows == Qwen4ExpRows::lazy ? sample_logits_top_n(req.sampler, weights_.n_vocab) : 0;
    int top_rows = 0, top_misses = 0;
    auto sample_row = [&](int32_t fed, const Qwen4ExpRow & row, int32_t & tok) {
        history.push_back(fed);
        if (!row.picked) {
            const int t = row.top_n > 0 ? sample_logits_top(row.top_vals, row.top_ids, row.top_n, weights_.n_vocab,
                                                            req.sampler, history, rng) : -1;
            top_rows += row.top_n > 0;
            top_misses += row.top_n > 0 && t < 0;
            tok = t >= 0 ? t : sample(row.logits());
        }
        return commit(tok);
    };
    auto on_drafts = [&](std::vector<int32_t> & draft_tokens) {
        if (decode_check_) decode_check_(true, draft_tokens, mtp_pending.h);
    };
    while (more) {
        const int max_k = spec ? std::min(req.n_gen - (int) result.tokens.size() - 1, cache_.max_ctx - pos - 1) : 0;
        Qwen4ExpMtpStep step;
        std::array<int32_t, QWEN4EXP_MTP_MAX_DRAFT> look{};
        int look_match = 0;
        PromptLookupPolicy::Pick pick;
        if (use_lookup && max_k > 0) {
            const int look_k = lookup_.propose(std::min(max_k, cache_.mtp_draft), look.data(), look_match);
            if (look_k > 0) {
                pick = lookup_policy_.choose(mtp_width_.next_width_cost_aware({}, cache_.mtp_draft + 1), look_k,
                                             look_match);
            }
        }
        if (!qwen4exp_mtp_step(backend_, weights_, cache_, pos, next, max_k, adaptive ? &mtp_width_ : nullptr,
                               mtp ? &mtp_pending : nullptr, sample_row, logits, step, on_drafts,
                               pick.lookup ? look.data() : nullptr, pick.lookup ? pick.width - 1 : 0,
                               rows, top_n)) {
            result.fail(GenerateErrorCode::DecodeFailed, step.error);
            return result;
        }
        if (use_lookup && step.k > 0) {
            lookup_policy_.observe(step.lookup, step.k + 1, step.decision.n_accepted, look_match, step.round_ms);
            if (step.lookup) {
                ++lookup_rounds;
                lookup_drafts += step.k;
                lookup_accepted += step.decision.n_accepted;
            }
        }
        if (use_lookup) lookup_.append(step.decision.emitted.data(), (size_t) step.decision.n_emitted);
        more = step.more;
        ++steps;
        ++width_steps[step.k];
        drafts += step.k;
        accepted += step.decision.n_accepted;
        draft_s += step.draft_s;
        const int retained = step.decision.n_emitted;
        next = step.decision.emitted[retained - 1];
        if (rows == Qwen4ExpRows::all) {
            logits_.assign(logits.begin() + (size_t) (retained - 1) * weights_.n_vocab,
                           logits.begin() + (size_t) retained * weights_.n_vocab);
        } else if (!step.more) {
            logits_.swap(logits);   // the last emitted row
        }
        pos += retained;
        if (decode_check_) decode_check_(false, mtp_pending.tok, logits_);
    }
    if (adaptive) mtp_costs_.store(mtp_width_, decode_ctx);
    if (cancelled) {
        result.fail(GenerateErrorCode::Cancelled, "cancelled during decode");
        return result;
    }
    // Replace speculative MTP rows with committed trunk pairs. Keep exactly
    // one pending hidden row: x[cur_pos] is still unknown to the live cache.
    // An AR retry runs no drafts and leaves one pair per decoded token, so the
    // catch-up goes in prompt-chunk-sized forwards. It only serves later drafts
    // and snapshots: if it fails, the text stands, and the MTP state is marked
    // incomplete so no snapshot of this sequence is published.
    if (mtp) {
        const int pairs = (int) mtp_pending.tok.size() - 1;
        if (qwen4exp_mtp_catch_up(backend_, weights_, cache_, mtp_pending, 1, chunk)) {
            ggml_backend_tensor_set(cache_.mtp_prev_hidden, mtp_pending.h.data(), 0, hd * sizeof(float));
            cache_.mtp_prev_pos = pos - 1;
        } else {
            std::fprintf(stderr, "[qwen4exp-mtp] checkpoint catch-up of %d pairs failed; this sequence keeps "
                                 "its text but publishes no snapshot\n", pairs);
            cache_.mtp_prev_pos = -1;
        }
    }
    tokens_ = std::move(history);
    const auto t_dec1 = std::chrono::steady_clock::now();
    result.decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    result.spec_decode_ran = drafts > 0;
    result.accept_rate = drafts > 0 ? (float) accepted / (float) drafts : 0.0f;
    if (drafts > 0) {
        const double decoded = (double) result.tokens.size() - 1.0;   // the first token came from the prefill
        std::string width_costs;   // the measured step cost of each timed width, ms
        for (int w = 2; w <= QWEN4EXP_MTP_MAX_VERIFY; ++w) {
            const float ms = mtp_width_.measured_costs()[(size_t) w];
            if (!std::isfinite(ms)) continue;
            char item[24];
            std::snprintf(item, sizeof(item), "%s%d:%.1f", width_costs.empty() ? "" : ",", w, ms);
            width_costs += item;
        }
        std::fprintf(stderr,
            "[qwen4exp-mtp] k=%d drafts=%lld accepted=%lld rate=%.3f tokens_per_step=%.3f draft_ms=%.2f decode=%.2f tok/s "
            "adaptive=%d steps_k1=%lld steps_k2=%lld steps_k3=%lld steps_k4=%lld steps_k5=%lld steps_k6=%lld steps_k7=%lld "
            "verify_builds=%llu verify_replays=%llu draft_builds=%llu draft_replays=%llu width_ms=%s "
            "lookup=%d lookup_rounds=%lld lookup_drafts=%lld lookup_accepted=%lld top_n=%d top_rows=%d top_misses=%d\n",
            cache_.mtp_draft, drafts, accepted, (double) accepted / (double) drafts, steps > 0 ? decoded / (double) steps : 0.0,
            1e3 * draft_s / (double) drafts, result.decode_s > 0.0 ? decoded / result.decode_s : 0.0,
            (int) adaptive, width_steps[1], width_steps[2], width_steps[3], width_steps[4],
            width_steps[5], width_steps[6], width_steps[7],
            (unsigned long long) (verify_graphs(&Qwen4ExpDecodeWorkspace::builds) - builds0),
            (unsigned long long) (verify_graphs(&Qwen4ExpDecodeWorkspace::replays) - replays0),
            (unsigned long long) (draft_graphs(&Qwen4ExpDecodeWorkspace::builds) - draft_builds0),
            (unsigned long long) (draft_graphs(&Qwen4ExpDecodeWorkspace::replays) - draft_replays0),
            width_costs.c_str(), (int) use_lookup, lookup_rounds, lookup_drafts, lookup_accepted, top_n, top_rows,
            top_misses);
    }

    guard.complete = true;
    result.succeed();
    return result;
}

bool Qwen4ExpBackend::snapshot_save(int slot) {
    return snapshot_save_replacing(slot, -1);
}

bool Qwen4ExpBackend::snapshot_save_replacing(int slot, int source) {
    if (slot < 0 || slot >= kMaxSlots || tokens_.empty() ||
        tokens_.size() != (size_t) cache_.cur_pos || logits_.empty() ||
        (cache_.mtp_k && cache_.mtp_prev_pos != cache_.cur_pos - 1)) return false;
    snapshot_free(slot);
    // Only replace the restored ancestor, and only under pressure. Plan the
    // replacement before freeing it; an oversized or unrelated capture must
    // not destroy the checkpoint we already have.
    if (!snapshot_fits(slot) && source >= 0 && snapshot_used(source) &&
        snapshots_[source].tokens.size() < tokens_.size() &&
        std::equal(snapshots_[source].tokens.begin(), snapshots_[source].tokens.end(), tokens_.begin()) &&
        snapshot_fits(slot, source)) {
        std::fprintf(stderr, "[qwen4exp-snap] replace ancestor slot=%d pos=%d with slot=%d pos=%d\n",
                     source, snapshot_cur_pos(source), slot, cache_.cur_pos);
        snapshot_free(source);
    }
    if (!snapshot_fits(slot)) return false;
    auto & s = snapshots_[slot];
    if (!save_qwen4exp_snapshot(backend_, snap_backend_, cache_, s)) return false;
    s.tokens = tokens_;
    s.logits = logits_;
    std::fprintf(stderr, "[qwen4exp-snap] slot=%d pos=%d bytes=%zu\n",
                 slot, s.cur_pos, ggml_backend_buffer_get_size(s.buf));
    return true;
}

bool Qwen4ExpBackend::snapshot_save_deferred(int slot) {
    if (slot < 0 || slot >= kMaxSlots || tokens_.empty() ||
        tokens_.size() != (size_t) cache_.cur_pos || logits_.empty() ||
        (cache_.mtp_k && cache_.mtp_prev_pos != cache_.cur_pos - 1)) return false;
    snapshot_flush_deferred();
    snapshot_free(slot);
    if (!snapshot_fits(slot)) return false;
    live_slot_ = slot;
    return true;
}

void Qwen4ExpBackend::snapshot_flush_deferred() {
    const int slot = live_slot_;
    live_slot_ = -1;
    if (slot >= 0) snapshot_save(slot);
}

void Qwen4ExpBackend::snapshot_free(int slot) {
    if (slot < 0 || slot >= kMaxSlots) return;
    if (slot == live_slot_) live_slot_ = -1;
    free_qwen4exp_snapshot(snapshots_[slot]);
}

bool Qwen4ExpBackend::snapshot_used(int slot) const {
    return slot >= 0 && slot < kMaxSlots && (slot == live_slot_ || snapshots_[slot].buf);
}

int Qwen4ExpBackend::snapshot_cur_pos(int slot) const {
    if (!snapshot_used(slot)) return 0;
    return slot == live_slot_ ? cache_.cur_pos : snapshots_[slot].cur_pos;
}

size_t Qwen4ExpBackend::snapshot_bytes_estimate(int tokens) const {
    return snapshot_bytes(tokens, /*rows=*/true);
}

size_t Qwen4ExpBackend::snapshot_state_bytes_estimate(int tokens) const {
    // A KVFlash pool's rows are its slots: a snapshot exists only while they map positions one to one (a context
    // within the pool), where a backup saves little, so its snapshots stay whole. Split mode keeps one row per
    // position on either device, and a strip copy reaches both.
    return cache_.kvflash ? 0 : snapshot_bytes(tokens, /*rows=*/false);
}

size_t Qwen4ExpBackend::snapshot_bytes(int tokens, bool rows) const {
    if (!snap_backend_ || !cache_.buf) return 0;
    const int n = std::clamp(tokens, 0, cache_.max_ctx);
    if (!n) return 0;
    size_t host = 0;
    const size_t copies = qwen4exp_snapshot_bytes(snap_backend_, cache_, n, &host, rows);
    return copies + host + (size_t(n) + weights_.n_vocab +
        std::max(0, weights_.ple_ngram_size - 1)) * sizeof(int32_t);
}

bool Qwen4ExpBackend::snapshot_drop_kv(int slot) {
    if (!snapshot_state_bytes_estimate(1) || !snapshot_used(slot)) return false;
    // A deferred snapshot is the live cache: copy it out first.
    if (slot == live_slot_) snapshot_flush_deferred();
    return snapshots_[slot].buf && drop_qwen4exp_snapshot_rows(cache_, snapshots_[slot]);
}

bool Qwen4ExpBackend::snapshot_state_only(int slot) const {
    return slot >= 0 && slot < kMaxSlots && snapshots_[slot].state_only;
}

bool Qwen4ExpBackend::snapshot_fits(int slot, int replaced) const {
    size_t remaining = snapshot_budget_;
    for (int i = 0; i < kMaxSlots; ++i) {
        if (i == slot || i == replaced || !snapshot_used(i)) continue;
        // A state-only backup holds its point state alone.
        const size_t bytes = snapshot_state_only(i) ? snapshot_state_bytes_estimate(snapshot_cur_pos(i))
                                                    : snapshot_bytes_estimate(snapshot_cur_pos(i));
        if (bytes > remaining) return false;
        remaining -= bytes;
    }
    const size_t bytes = snapshot_bytes_estimate(cache_.cur_pos);
    if (bytes <= remaining) return true;
    std::fprintf(stderr, "[qwen4exp-snap] pressure slot=%d bytes=%zu remaining=%zu allowance=%zu\n",
                 slot, bytes, remaining, snapshot_budget_);
    return false;
}

GenerateResult Qwen4ExpBackend::restore_and_generate_impl(
        int slot, const GenerateRequest & req, const DaemonIO & io) {
    if (parked_ || !snapshot_used(slot)) {
        GenerateResult result;
        result.fail(parked_ ? GenerateErrorCode::ModelParked : GenerateErrorCode::InvalidSnapshotSlot);
        return result;
    }
    const auto & prefix = slot == live_slot_ ? tokens_ : snapshots_[slot].tokens;
    // Recurrent state cannot be shortened to an LCP. The server chooses a
    // shallower checkpoint; stale/direct callers safely fall back to cold.
    if (prefix.size() > req.prompt.size() || !std::equal(prefix.begin(), prefix.end(), req.prompt.begin())) {
        return generate_impl(req, io);
    }
    const int pos = snapshot_cur_pos(slot);
    if (slot == live_slot_) {
        live_slot_ = -1; // consume the zero-copy checkpoint before mutation
        std::fprintf(stderr, "[qwen4exp-snap] resident slot=%d pos=%d\n", slot, pos);
    } else {
        snapshot_flush_deferred();   // a deferred donor too: the restore below overwrites the live cache
        // A state-only backup takes its rows from the deeper snapshot the request names; without it, recompute.
        const int kv = req.restore_kv_slot;
        const Qwen4ExpSnapshot * rows = kv >= 0 && kv < kMaxSlots && kv != slot ? &snapshots_[kv] : nullptr;
        if (!restore_qwen4exp_snapshot(backend_, snapshots_[slot], cache_, rows)) return generate_impl(req, io);
        tokens_ = snapshots_[slot].tokens;
        logits_ = snapshots_[slot].logits;
    }
    return run(req, io, pos, slot);
}

ModelBackend::CompressResult Qwen4ExpBackend::compress(const CompressRequest & req) {
    const auto results = compress_batch({req});
    return results.empty() ? CompressResult{} : results.front();
}

std::vector<ModelBackend::CompressResult> Qwen4ExpBackend::compress_batch(
        const std::vector<CompressRequest> & requests) {
    bool was_parked = false;
    PFlashWindow window;
    window.tag = "qwen4exp-pflash";
    // Parking releases the target's and the expert device's buffers: drain
    // their queued work first.
    window.drain = [this] {
        if (backend_) ggml_backend_synchronize(backend_);
        if (weights_.expert_backend) ggml_backend_synchronize(weights_.expert_backend);
    };
    window.park = [&] {
        was_parked = parked_;
        return park(ParkTarget::TargetModel);
    };
    // unpark sizes the target for the memory free then, so a drafter that
    // stays loaded is accounted for.
    window.unpark = [&] { return was_parked || unpark(ParkTarget::TargetModel); };
    window.release = [this] { free_drafter(); };
    window.shared_backends = {backend_, weights_.expert_backend};
    window.engine_max_ctx = cfg_.device.max_ctx;
    return pflash_compress_batch(drafter_ctx_, drafter_loaded_, requests, window);
}

bool Qwen4ExpBackend::handle_compress(const std::string & line,
                                      const DaemonIO & io) {
    CompressRequest req;
    if (!pflash_parse_compress_line(line, req) || req.drafter_path.empty()) {
        std::fprintf(stderr, "[qwen4exp-pflash] bad compress arguments\n");
        io.emit(-1);
        return false;
    }
    const CompressResult result = compress(req);
    for (int32_t token : result.compressed_ids) io.emit(token);
    io.emit(-1);
    return result.ok;
}

void Qwen4ExpBackend::free_drafter() {
    if (!drafter_loaded_ && !drafter_ctx_.backend) return;
    luce::common::free_drafter(drafter_ctx_);
    drafter_loaded_ = false;
}

void Qwen4ExpBackend::shutdown() {
    free_drafter();
    if (backend_) release_target();
    free_snapshot_backend(snap_backend_, backend_);
    snap_backend_ = nullptr;
    if (weights_.expert_backend) {
        ggml_backend_free(weights_.expert_backend);
        weights_.expert_backend = nullptr;
    }
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

}  // namespace luce::common
