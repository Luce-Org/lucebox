#include "qwen4exp_backend.h"
#include "qwen4exp_chunk.h"
#include "qwen4exp_graph.h"

#include "common/peer_access.h"
#include "common/sampler.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <random>
#include <utility>
#include <vector>

namespace luce::common {

// Split mode, research: copy the most-routed experts (the routing CSV in
// LUCE_QWEN4EXP_HOT_EXPERTS, LUCE_QWEN4EXP_HOT_GIB of them, default 8) to the
// target, plus the per-layer lookup tables that send each pick to one device.
static bool load_qwen4exp_hot_experts(ggml_backend_t backend, Qwen4ExpWeights & w) {
    const char * path = std::getenv("LUCE_QWEN4EXP_HOT_EXPERTS");
    if (!path || !*path || !w.expert_backend) return true;
    std::string err;
    MoeHybridRoutingStats stats;
    if (!MoeHybridRoutingStats::load_csv(path, stats, &err) ||
        !stats.matches(w.n_layer, w.n_expert, w.n_expert_used)) {
        std::fprintf(stderr, "[qwen4exp] hot experts: %s: %s\n", path, err.empty() ? "shape mismatch" : err.c_str());
        return false;
    }
    std::vector<MoeLayerDesc> descs((size_t) w.n_layer);
    std::vector<uint64_t> expert_bytes((size_t) w.n_layer);
    for (int il = 0; il < w.n_layer; ++il) {
        const Qwen4ExpLayer & L = w.layers[(size_t) il];
        descs[(size_t) il].ffn_gate_exps = L.ffn_gate_exps;
        descs[(size_t) il].ffn_up_exps   = L.ffn_up_exps;
        descs[(size_t) il].ffn_down_exps = L.ffn_down_exps;
        expert_bytes[(size_t) il] = L.ffn_gate_exps->nb[2] + L.ffn_up_exps->nb[2] + L.ffn_down_exps->nb[2];
    }
    const char * gib = std::getenv("LUCE_QWEN4EXP_HOT_GIB");
    const uint64_t budget = (uint64_t) ((gib ? std::atof(gib) : 8.0) * (double) (1ull << 30));
    MoeHybridPlacement placement;
    if (!MoeHybridPlacement::build_from_stats_with_layer_bytes(stats, expert_bytes, budget, 0, placement, &err)) {
        std::fprintf(stderr, "[qwen4exp] hot experts: placement failed: %s\n", err.c_str());
        return false;
    }
    MoeHybridConfig cfg;
    cfg.n_embd = w.n_embd;
    cfg.n_expert = w.n_expert;
    cfg.n_expert_used = w.n_expert_used;
    cfg.n_ff_exp = (int) w.layers[0].ffn_gate_exps->ne[1];
    cfg.n_layer = w.n_layer;
    cfg.cold_expert_backend = MoeHybridColdBackend::None;   // the full stacks stay on the expert device
    cfg.materialize_cold_experts = false;
    w.hot = std::make_unique<MoeHybridStorage>();
    if (!build_moe_hybrid_storage(cfg, backend, placement, descs, *w.hot, &err)) {
        std::fprintf(stderr, "[qwen4exp] hot experts: %s\n", err.c_str());
        return false;
    }
    const ggml_init_params ip = { 2 * (size_t) w.n_layer * ggml_tensor_overhead(), nullptr, true };
    w.lut_ctx = ggml_init(ip);
    w.hot_lut.assign((size_t) w.n_layer, nullptr);
    w.cold_lut.assign((size_t) w.n_layer, nullptr);
    for (int il = 0; il < w.n_layer; ++il) {
        if (w.hot->layers[(size_t) il].hot_expert_ids.empty()) continue;
        w.hot_lut[(size_t) il]  = ggml_new_tensor_2d(w.lut_ctx, GGML_TYPE_I32, 1, w.n_expert);
        w.cold_lut[(size_t) il] = ggml_new_tensor_2d(w.lut_ctx, GGML_TYPE_I32, 1, w.n_expert);
    }
    w.lut_buf = ggml_backend_alloc_ctx_tensors(w.lut_ctx, backend);
    uint64_t hot_routes = 0, routes = 0;
    double hot_bytes = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        const std::vector<int32_t> & hot = w.hot->layers[(size_t) il].hot_local_by_global;
        hot_bytes += (double) placement.hot_counts[(size_t) il] * (double) expert_bytes[(size_t) il];
        for (int e = 0; e < w.n_expert; ++e) {
            routes += stats.count(il, e);
            if (w.hot_lut[(size_t) il] && hot[(size_t) e] >= 0) hot_routes += stats.count(il, e);
        }
        if (!w.hot_lut[(size_t) il]) continue;
        std::vector<int32_t> cold((size_t) w.n_expert);
        for (int e = 0; e < w.n_expert; ++e) cold[(size_t) e] = hot[(size_t) e] >= 0 ? -1 : e;
        ggml_backend_tensor_set(w.hot_lut[(size_t) il], hot.data(), 0, hot.size() * sizeof(int32_t));
        ggml_backend_tensor_set(w.cold_lut[(size_t) il], cold.data(), 0, cold.size() * sizeof(int32_t));
    }
    std::fprintf(stderr, "[qwen4exp] hot experts: %d of %d (%.2f GiB) on the target, %.1f%% of the profiled picks\n",
                 placement.total_hot, w.n_layer * w.n_expert, hot_bytes / (1ull << 30),
                 routes ? 100.0 * (double) hot_routes / (double) routes : 0.0);
    return true;
}

int qwen4exp_select_chunk(ggml_backend_t backend, const Qwen4ExpWeights & w,
        Qwen4ExpCache & cache, int slots, int resident_slots, size_t * snapshot_budget) {
    if (slots < 1 || resident_slots < 1 || resident_slots > slots) return 0;
    size_t free_device = 0, total = 0, free_host = 0;
    ggml_backend_cuda_get_device_memory(ggml_backend_cuda_get_device_id(backend), &free_device, &total);
    ggml_backend_dev_memory(ggml_backend_get_device(backend), &free_host, &total);
    // The generic UMA query substitutes MemAvailable; retain the HIP/GTT limit too.
    const size_t available = std::min(free_device, free_host);
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
    const bool mtp = qwen4exp_verify_supported(cache);
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
    // The decode workspaces stay resident while the next prompt prefills.
    const size_t fixed = (slots - resident_slots) * state + shadow + shadow_tmp +
                         slots * (decode + verify + draft);
    std::fprintf(stderr, "[qwen4exp] chunk-runtime mtp=%d decode=%zu verify=%zu draft=%zu scratch=%zu host=%zu\n",
        (int) mtp, decode, verify, draft, runtime_scratch, runtime_host);
    struct Plan { size_t graph = 0, ring = 0, host = 0, scratch = 0; };
    std::vector<std::pair<int, Plan>> plans;   // one per probed chunk size
    const int chunk = qwen4exp_fit_chunk(cache.max_ctx, available, fixed, [&](int n) {
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
            host = std::max(host, m.host);
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
    }, snapshot_budget);
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
    // ponytail: concurrent slots decode without MTP or prefix snapshots (the
    // seq engine replaces the serial loop that owns both); per-slot MTP is follow-up.
    if (cfg_.max_concurrency > 1 && cfg_.verify_width != 1) {
        std::fprintf(stderr, "[qwen4exp] --max-concurrency %d: MTP off\n", cfg_.max_concurrency);
        cfg_.verify_width = 1;
    }
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
    // Research: routed experts on a second GPU, everything else on the target.
    if (const char * expert_gpu = std::getenv("LUCE_QWEN4EXP_EXPERT_GPU")) {
        weights_.expert_backend = ggml_backend_cuda_init(std::atoi(expert_gpu));
        if (!weights_.expert_backend) {
            std::fprintf(stderr, "[qwen4exp] expert backend init failed for GPU %s\n", expert_gpu);
            return false;
        }
        weights_.expert_gfx1151 = ggml_backend_cuda_qwen4exp_supported(weights_.expert_backend);
        // Direct device-to-device copies for the activations crossing between the two GPUs.
        if (!enable_peer_access_pair(cfg_.device.gpu, std::atoi(expert_gpu))) {
            std::fprintf(stderr, "[qwen4exp] peer access between GPU %d and %s unavailable; copies stage through the host\n",
                         cfg_.device.gpu, expert_gpu);
        }
    }
    if (!load_qwen4exp_gguf(cfg_.model_path, backend_, weights_,
                           cfg_.verify_width == 1 ? "0" : cfg_.draft_path.value_or(""))) {
        std::fprintf(stderr, "[qwen4exp] model load failed: %s\n",
                     luce_last_error());
        return false;
    }
    if (!create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx, cache_, /*mtp=*/true,
                               cfg_.verify_width == 0 ? QWEN4EXP_MTP_MAX_DRAFT : std::max(1, cfg_.verify_width - 1))) {
        std::fprintf(stderr, "[qwen4exp] cache creation failed\n");
        return false;
    }
    if (!load_qwen4exp_hot_experts(backend_, weights_)) return false;
    if (const char * stats_path = std::getenv("LUCE_QWEN4EXP_ROUTING_STATS_OUT")) {
        weights_.route_recorder = std::make_unique<Qwen4ExpRouteRecorder>();
        weights_.route_recorder->stats.init(weights_.n_layer, weights_.n_expert, weights_.n_expert_used);
        weights_.route_recorder->path = stats_path;
    }
    if (cfg_.chunk > 0) {
        chunk_ = cfg_.chunk;
    } else if (cfg_.max_concurrency > 1) {
        // Cap at the single-slot speed target: a larger chunk only holds the
        // other slots' decode for longer (and filled 93 GB at 4x32K: 12800 rows).
        chunk_ = std::min(4096, qwen4exp_select_chunk(backend_, weights_, cache_, cfg_.max_concurrency, 1));
    } else {
        snapshot_budget_ = 3 * snapshot_bytes_estimate(cache_.max_ctx);
        chunk_ = qwen4exp_select_chunk(backend_, weights_, cache_, 1, 1, &snapshot_budget_);
        std::fprintf(stderr, "[qwen4exp] prefix snapshot allowance=%zu bytes\n", snapshot_budget_);
    }
    if (weights_.expert_backend) chunk_ = std::min(chunk_, 16384);   // the generic MoE id helper's limit
    // The planner does not see the target's matmul scratch pool, which grows
    // over a long prompt; with hot experts the target has no slack left for it.
    // 8192 rows still give the prompt pipeline two 4096-row streams.
    if (weights_.hot && cfg_.chunk <= 0) chunk_ = std::min(chunk_, 8192);
    if (chunk_ <= 0) {
        std::fprintf(stderr, "[qwen4exp] insufficient prefill memory at the configured context\n");
        return false;
    }
    if (!start_seq_engine()) {
        std::fprintf(stderr, "[qwen4exp] full-cache slot allocation failed\n");
        return false;
    }
    return true;
}

bool Qwen4ExpBackend::start_seq_engine() {
    if (cfg_.max_concurrency <= 1) return true;
    seq_caches_.resize((size_t)cfg_.max_concurrency - 1);
    std::vector<Qwen4ExpCache *> caches{&cache_};
    for (Qwen4ExpCache & cache : seq_caches_) {
        if (!create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx, cache)) return false;
        caches.push_back(&cache);
    }
    seq_engine_ = std::make_unique<Qwen4ExpSeqEngine>(
        backend_, weights_, std::move(caches), cfg_.device.max_ctx, chunk_);
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
    for (int i = 0; i < kMaxSlots; ++i) snapshot_free(i);
    tokens_.clear(); logits_.clear();
    seq_engine_.reset();
    for (Qwen4ExpCache & cache : seq_caches_) free_qwen4exp_cache(cache);
    seq_caches_.clear();
    free_qwen4exp_cache(cache_);
    free_qwen4exp_weights(weights_);
    ggml_backend_cuda_trim_pool(backend_); // also frees the bf16 weight shadows keyed by the freed weights' addresses
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
    if (!load_qwen4exp_gguf(cfg_.model_path, backend_, weights_,
                           cfg_.verify_width == 1 ? "0" : cfg_.draft_path.value_or(""))) {
        std::fprintf(stderr, "[qwen4exp] unpark reload failed: %s\n",
                     luce_last_error());
        return false;
    }
    if (!create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx, cache_, /*mtp=*/true,
                               cfg_.verify_width == 0 ? QWEN4EXP_MTP_MAX_DRAFT : std::max(1, cfg_.verify_width - 1))) {
        std::fprintf(stderr, "[qwen4exp] unpark cache creation failed\n");
        free_qwen4exp_weights(weights_);
        return false;
    }
    // Keep the resolved serving policy across park/unpark (and /props stable).
    if (!start_seq_engine()) {
        std::fprintf(stderr, "[qwen4exp] unpark slot allocation failed\n");
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
    // predicts x_{p+2} from the pair (h_p, x_{p+1}), h_p being the trunk's final HC residual at p. Pairs at positions
    // [mtp_pos, ...) not yet run through the draft layer: mtp_h holds their hidden rows, mtp_tok the tokens known so
    // far (a pair's token arrives with the next forward).
    const bool mtp = qwen4exp_verify_supported(cache_);
    const bool spec = !req.force_ar_decode && req.n_gen > 2 && mtp;
    const size_t hd = (size_t) weights_.n_embd * weights_.n_hc;
    std::vector<float> hidden, mtp_h, mtp_logits;
    std::vector<int32_t> mtp_tok;
    int mtp_pos = std::max(0, pos - 1);
    if (mtp && pos > 0) {
        mtp_h.resize(hd);
        ggml_backend_tensor_get(cache_.mtp_prev_hidden, mtp_h.data(), 0, hd * sizeof(float));
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
            mtp_h.swap(hidden);
            mtp_pos = pos + n - 1;
        } else if (mtp) {   // single-row chunk: retain the original catch-up path
            mtp_tok.assign(req.prompt.begin() + pos + (pos == 0 ? 1 : 0), req.prompt.begin() + pos + n);
            mtp_h.insert(mtp_h.end(), hidden.begin(), hidden.end());
            const int n_pairs = (int) mtp_tok.size();
            if (n_pairs > 0 && !qwen4exp_mtp_forward(backend_, weights_, cache_, mtp_tok.data(), mtp_h.data(),
                                                     n_pairs, mtp_pos, mtp_logits, nullptr, /*kv_only=*/true)) {
                result.fail(GenerateErrorCode::PrefillFailed, "qwen4exp MTP catch-up failed");
                return result;
            }
            mtp_h.erase(mtp_h.begin(), mtp_h.begin() + (std::ptrdiff_t) ((size_t) n_pairs * hd));
            mtp_pos += n_pairs;
            ggml_backend_tensor_set(cache_.mtp_prev_hidden, mtp_h.data(), 0, hd * sizeof(float));
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

    std::mt19937_64 rng(req.sampler.seed != 0 ? req.sampler.seed
                                              : std::random_device{}());
    std::vector<int32_t> history = req.prompt;
    auto sample = [&](const float * row) {
        return (int32_t) sample_logits(row, weights_.n_vocab, req.sampler, history, rng);
    };
    BudgetHookState budget;   // thinking force-close: keeps the reply reserve of the budget for the answer
    bool cancelled = false;
    // Commits a sampled token, after the budget hook's substitution; false once generation ends.
    auto commit = [&](int32_t & tok) {
        if (budget.apply(req.budget_hook, (int) result.tokens.size(), req.n_gen, tok)) result.budget_forced_close = true;
        result.tokens.push_back(tok);
        io.emit(tok);
        if (io.is_cancelled()) {
            cancelled = true;
            return false;
        }
        return tok != weights_.eos_id && tok != weights_.eos_chat_id && (int) result.tokens.size() < req.n_gen;
    };

    // Sample trunk rows lazily, updating history and applying the budget hook
    // exactly once per emitted token. No RNG draws for unvisited verify rows.
    long long drafts = 0, accepted = 0, steps = 0;
    std::array<long long, QWEN4EXP_MTP_MAX_VERIFY> width_steps{};
    auto width_policy = qwen4exp_mtp_width_policy(cache_.mtp_draft,
        spec && cfg_.verify_width == 0, (int) req.prompt.size(), mtp_cost_learned_ ? &mtp_cost_ : nullptr);
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
    auto built = [&] { return verify_graphs(&Qwen4ExpDecodeWorkspace::builds) + draft_graphs(&Qwen4ExpDecodeWorkspace::builds); };
    const uint64_t builds0 = verify_graphs(&Qwen4ExpDecodeWorkspace::builds);
    const uint64_t replays0 = verify_graphs(&Qwen4ExpDecodeWorkspace::replays);
    const uint64_t draft_builds0 = draft_graphs(&Qwen4ExpDecodeWorkspace::builds);
    const uint64_t draft_replays0 = draft_graphs(&Qwen4ExpDecodeWorkspace::replays);
    const auto t_dec0 = std::chrono::steady_clock::now();
    int32_t next = sample(logits.data());
    bool more = req.n_gen > 0 && commit(next);
    if (mtp) mtp_tok.assign(1, next);
    std::vector<int32_t> draft_tokens;
    while (more) {
        const int k = spec ? std::max(0, std::min({qwen4exp_mtp_next_width(width_policy) - 1,
            req.n_gen - (int) result.tokens.size() - 1, cache_.max_ctx - pos - 1})) : 0;
        const bool verify = k > 0;
        const auto step_start = verify ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const uint64_t step_builds = verify ? built() : 0;
        if (verify) {
            const auto td0 = std::chrono::steady_clock::now();
            if (!qwen4exp_mtp_draft(backend_, weights_, cache_, mtp_tok.data(), mtp_h.data(), (int) mtp_tok.size(),
                                    mtp_pos, k, draft_tokens)) {
                result.fail(GenerateErrorCode::DecodeFailed, "qwen4exp MTP draft failed");
                return result;
            }
            draft_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();
            if (decode_check_) decode_check_(true, draft_tokens, mtp_h);
        }
        std::array<int32_t, QWEN4EXP_MTP_MAX_VERIFY> in{}, samples{};
        in[0] = next;
        if (verify) std::copy(draft_tokens.begin(), draft_tokens.end(), in.begin() + 1);
        const Qwen4ExpForwardResult r = qwen4exp_forward(
            backend_, weights_, cache_, in.data(), k + 1, pos, logits, mtp ? &hidden : nullptr, verify);
        if (!r.ok) {
            result.fail(GenerateErrorCode::DecodeFailed, "qwen4exp decode forward failed");
            return result;
        }
        ++steps;
        ++width_steps[k];
        drafts += k;
        Qwen4ExpMtpAcceptance decision;
        for (int i = 0; i <= k; ++i) {
            history.push_back(in[i]);
            int32_t tok = sample(logits.data() + (size_t) i * weights_.n_vocab);
            more = commit(tok);
            samples[i] = tok;
            decision = qwen4exp_mtp_accept(draft_tokens.data(), k, samples.data(), i + 1);
            if (!more || decision.n_accepted != i + 1) break;
        }
        const int retained = decision.n_emitted;
        next = decision.emitted[retained - 1];
        accepted += decision.n_accepted;
        if (verify) {
            // Replace every predicted MTP hidden/KV row with the corresponding
            // trunk pair on next catch-up, including after a partial accept.
            mtp_h.assign(hidden.begin(), hidden.begin() + (std::ptrdiff_t) ((size_t) retained * hd));
            mtp_tok.assign(decision.emitted.begin(), decision.emitted.begin() + retained);
            mtp_pos = pos;
            if (!qwen4exp_verify_rollback(backend_, weights_, cache_, pos, retained)) {
                result.fail(GenerateErrorCode::DecodeFailed, "qwen4exp verify rollback failed");
                return result;
            }
        } else if (mtp) {
            mtp_h.insert(mtp_h.end(), hidden.begin(), hidden.end());
            mtp_tok.push_back(next);
        }
        logits_.assign(logits.begin() + (size_t) (retained - 1) * weights_.n_vocab,
                       logits.begin() + (size_t) retained * weights_.n_vocab);
        pos += retained;
        if (decode_check_) decode_check_(false, mtp_tok, logits_);
        // Only a step that replayed its verify and draft graphs times the width; one
        // that built a graph is slower and counts for acceptance alone.
        if (verify) width_policy.observe(decision.n_accepted + 1, k + 1,
            built() != step_builds ? -1.0f :
            (float) std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - step_start).count());
    }
    if (width_policy.enabled()) {
        mtp_cost_ = width_policy.cost();
        mtp_cost_learned_ = true;
    }
    if (cancelled) {
        result.fail(GenerateErrorCode::Cancelled, "cancelled during decode");
        return result;
    }
    // Replace speculative MTP rows with committed trunk pairs. Keep exactly
    // one pending hidden row: x[cur_pos] is still unknown to the live cache.
    if (mtp) {
        const int pairs = (int) mtp_tok.size() - 1;
        if (pairs > 0 && !qwen4exp_mtp_forward(backend_, weights_, cache_, mtp_tok.data(), mtp_h.data(),
                                               pairs, mtp_pos, mtp_logits, nullptr, true)) {
            result.fail(GenerateErrorCode::DecodeFailed, "qwen4exp MTP checkpoint catch-up failed");
            return result;
        }
        ggml_backend_tensor_set(cache_.mtp_prev_hidden, mtp_h.data() + (size_t) pairs * hd, 0, hd * sizeof(float));
        cache_.mtp_prev_pos = pos - 1;
    }
    tokens_ = std::move(history);
    const auto t_dec1 = std::chrono::steady_clock::now();
    result.decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    result.spec_decode_ran = drafts > 0;
    result.accept_rate = drafts > 0 ? (float) accepted / (float) drafts : 0.0f;
    if (drafts > 0) {
        const double decoded = (double) result.tokens.size() - 1.0;   // the first token came from the prefill
        std::fprintf(stderr,
            "[qwen4exp-mtp] k=%d drafts=%lld accepted=%lld rate=%.3f tokens_per_step=%.3f draft_ms=%.2f decode=%.2f tok/s "
            "adaptive=%d steps_k1=%lld steps_k2=%lld steps_k3=%lld steps_k4=%lld steps_k5=%lld steps_k6=%lld steps_k7=%lld "
            "verify_builds=%llu verify_replays=%llu draft_builds=%llu draft_replays=%llu cycle_ms=%.1f+%.2fk\n",
            cache_.mtp_draft, drafts, accepted, (double) accepted / (double) drafts, steps > 0 ? decoded / (double) steps : 0.0,
            1e3 * draft_s / (double) drafts, result.decode_s > 0.0 ? decoded / result.decode_s : 0.0,
            (int) width_policy.enabled(), width_steps[1], width_steps[2], width_steps[3], width_steps[4],
            width_steps[5], width_steps[6], width_steps[7],
            (unsigned long long) (verify_graphs(&Qwen4ExpDecodeWorkspace::builds) - builds0),
            (unsigned long long) (verify_graphs(&Qwen4ExpDecodeWorkspace::replays) - replays0),
            (unsigned long long) (draft_graphs(&Qwen4ExpDecodeWorkspace::builds) - draft_builds0),
            (unsigned long long) (draft_graphs(&Qwen4ExpDecodeWorkspace::replays) - draft_replays0),
            width_policy.cost().base, width_policy.cost().slope);
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
    if (!save_qwen4exp_snapshot(backend_, cache_, s)) return false;
    s.tokens = tokens_;
    s.logits = logits_;
    std::fprintf(stderr, "[qwen4exp-snap] slot=%d pos=%d device_bytes=%zu\n",
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
    if (!backend_ || !cache_.buf) return 0;
    const int n = std::clamp(tokens, 0, cache_.max_ctx);
    if (!n) return 0;
    size_t host = 0;
    const size_t device = qwen4exp_snapshot_bytes(backend_, cache_, n, &host);
    return device + host + (size_t(n) + weights_.n_vocab +
        std::max(0, weights_.ple_ngram_size - 1)) * sizeof(int32_t);
}

bool Qwen4ExpBackend::snapshot_fits(int slot, int replaced) const {
    size_t remaining = snapshot_budget_;
    for (int i = 0; i < kMaxSlots; ++i) {
        if (i == slot || i == replaced || !snapshot_used(i)) continue;
        const size_t bytes = snapshot_bytes_estimate(snapshot_cur_pos(i));
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
        snapshot_flush_deferred();
        restore_qwen4exp_snapshot(backend_, snapshots_[slot], cache_);
        tokens_ = snapshots_[slot].tokens;
        logits_ = snapshots_[slot].logits;
    }
    return run(req, io, pos, slot);
}

bool Qwen4ExpBackend::handle_compress(const std::string & line,
                                      const DaemonIO & io) {
    (void) line;
    (void) io;
    return false;
}

void Qwen4ExpBackend::free_drafter() {}

void Qwen4ExpBackend::shutdown() {
    if (weights_.route_recorder) weights_.route_recorder->stats.save_csv(weights_.route_recorder->path);
    for (int i = 0; i < kMaxSlots; ++i) snapshot_free(i);
    tokens_.clear(); logits_.clear();
    seq_engine_.reset();
    for (Qwen4ExpCache & cache : seq_caches_) free_qwen4exp_cache(cache);
    seq_caches_.clear();
    free_qwen4exp_cache(cache_);
    free_qwen4exp_weights(weights_);
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
