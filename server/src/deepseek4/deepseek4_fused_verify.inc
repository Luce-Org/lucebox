// ─── Fused whole-model graph (n_tokens = q in [1,5]) ────────────────────
// One whole-model graph per (q, flush, padded-comp) shape: batched attention
// (causal mask input) + batched MoE, per-token fused HC chains, in-graph
// optional drafter-feature capture and q-wide logits. The spec loop guarantees a
// ratio-4 boundary can only sit at the batch's LAST token (dynamic q), so the
// compressor pools at most once per step, exactly like an AR step at the last
// position. It also serves opt-in q=1 hybrid decode, because the older q=1
// fused graph is single-backend. Enable with LUCE_DS4_FUSED_VERIFY=1 and,
// for q=1 hybrid decode, LUCE_DS4_FUSED_HYBRID_DECODE=1.

// tensor_set that tolerates inputs the graph never consumed (gallocr leaves
// them unallocated, e.g. comp-emission bundles in non-flush graphs).
static void ds4_fv_set(ggml_tensor * t, const void * data, size_t nbytes) {
    if (t && t->buffer) ggml_backend_tensor_set(t, data, 0, nbytes);
}

// Attention lanes of a verify batch. A tokenwise verify (the staggered pre-mix
// model, whose compressors may emit a row at every token) runs each token as
// its own lane, exactly the single-token decode attention at its position;
// otherwise the batch is one lane.
static int ds4_fused_attention_lanes(const DeepSeek4Weights & w, int q) {
    return w.hc_staggered_pre && q > 1 ? q : 1;
}

// Compressed rows a lane's attention spans: the rows visible at its last
// token, rounded up to the replay stride.
static int ds4_fused_lane_padded(const DeepSeek4Weights & w, const DeepSeek4LayerCache & clc,
                                 int il, int lane_token_pos) {
    const int ratio = (int) w.compress_ratios[il];
    if (ratio <= 0 || !clc.comp_kv) return 0;
    const int n_comp = ds4_comp_rows_used(clc.comp_kv, clc.n_comp, ratio, lane_token_pos);
    return ds4_padded_comp_rows(n_comp, (int) clc.comp_kv->ne[1]);
}

struct Ds4MixedMoePolicy {
    bool owner_local_reduction = false;
    bool direct_device_join = false;
    bool schedule_branches = false;
    bool native_route_width = false;
    bool fused_hc_join = false;
    bool late_join_split = false;
    bool deferred_join_split = false;
    bool batch_peer_copies = false;
    bool report_split_count = false;
    bool scheduler_trace = false;
    bool pin_route_weights = false;
    bool preserve_routes_for_diagnostics = false;
};

static const Ds4MixedMoePolicy & ds4_mixed_moe_policy() {
    static const Ds4MixedMoePolicy policy = [] {
        Ds4MixedMoePolicy result;
        result.owner_local_reduction =
            ds4_env_flag("LUCE_DS4_CROSS_VENDOR_OWNER_SUMS");
        result.direct_device_join =
            ds4_env_flag("LUCE_DS4_TP_PEER_FENCE") ||
            moe_hybrid_graph_policy().device_join;
        result.schedule_branches =
            ds4_env_flag("LUCE_DS4_TP_SCHEDULE_BRANCHES");
        result.native_route_width =
            ds4_env_flag("LUCE_DS4_TP_NATIVE_ROUTE_WIDTH");
        result.fused_hc_join =
            ds4_env_flag("LUCE_DS4_TP_FUSED_HC_JOIN");
        result.late_join_split =
            ds4_env_flag("LUCE_DS4_TP_LATE_JOIN_SPLIT");
        result.deferred_join_split =
            ds4_env_flag("LUCE_DS4_TP_DEVICE_JOIN_SPLIT");
        result.batch_peer_copies =
            ds4_env_flag("LUCE_DS4_TP_BATCH_SPLIT_COPIES") ||
            ds4_env_flag("GGML_BATCH_PEER_COPIES") ||
            ds4_env_flag("GGML_CUDA_BATCH_PEER_COPIES");
        result.report_split_count =
            ds4_env_flag("LUCE_DS4_TP_SPLIT_COUNT");
        result.scheduler_trace =
            ds4_env_flag("LUCE_DS4_TP_SCHED_TRACE");
        result.pin_route_weights =
            ds4_env_flag("LUCE_DS4_TP_MAIN_ROUTE_WEIGHTS");
        result.preserve_routes_for_diagnostics =
            ds4_env_flag("LUCE_DS4_TP_ROUTE_STATS") ||
            ds4_env_flag("LUCE_DS4_ROUTING_STATS_OUT") ||
            ds4_env_flag("LUCE_DS4_TP_CACHE_AUDIT");
        return result;
    }();
    return policy;
}

template <typename T>
static void ds4_fv_set_repeated_rows(
        ggml_tensor * tensor,
        const std::vector<T> & row) {
    if (!tensor || row.empty()) return;
    const size_t elements = (size_t) ggml_nelements(tensor);
    GGML_ASSERT(elements % row.size() == 0);
    std::vector<T> repeated(elements);
    for (size_t offset = 0; offset < elements; offset += row.size()) {
        std::copy(row.begin(), row.end(), repeated.begin() + offset);
    }
    ds4_fv_set(tensor, repeated.data(), repeated.size() * sizeof(T));
}
static bool ds4_fused_verify_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = ds4_env_flag("LUCE_DS4_FUSED_VERIFY") ? 1 : 0;
    }
    return enabled == 1;
}

static bool ds4_fused_verify_trace_node(
        ggml_tensor * tensor, bool ask, void *) {
    static const bool trace_all =
        ds4_env_flag("LUCE_DS4_TP_SCHED_TRACE_ALL");
    const bool routed = tensor &&
        (tensor->op == GGML_OP_MUL_MAT_ID ||
         tensor->op == GGML_OP_GET_ROWS);
    if (!tensor || (!trace_all && !routed)) return false;
    const ggml_tensor * src0 = tensor->src[0];
    const ggml_tensor * ids = routed
        ? (tensor->op == GGML_OP_MUL_MAT_ID ? tensor->src[2] : tensor->src[1])
        : nullptr;
    const char * buffer = src0 && src0->buffer
        ? ggml_backend_buffer_name(src0->buffer) : "unallocated";
    std::fprintf(stderr,
                 "[ds4-tp-trace] %s op=%s name='%s' src0=%s "
                 "ne=[%lld,%lld,%lld] ids=[%lld,%lld] buffer=%s "
                 "ids_buffer=%s ids_ptr=%p src1_ptr=%p\n",
                 ask ? "begin" : "done",
                 ggml_op_desc(tensor), tensor->name,
                 src0 ? ggml_type_name(src0->type) : "none",
                 src0 ? (long long) src0->ne[0] : 0LL,
                 src0 ? (long long) src0->ne[1] : 0LL,
                 src0 ? (long long) src0->ne[2] : 0LL,
                 ids ? (long long) ids->ne[0] : 0LL,
                 ids ? (long long) ids->ne[1] : 0LL,
                 buffer,
                 ids && ids->buffer ? ggml_backend_buffer_name(ids->buffer)
                                    : "unallocated",
                 ids ? ids->data : nullptr,
                 tensor->src[1] ? tensor->src[1]->data : nullptr);
    if (((ask && tensor->op == GGML_OP_MUL_MAT_ID) ||
         (!ask && tensor->op == GGML_OP_GET_ROWS)) &&
        ids && ids->buffer &&
        ids->type == GGML_TYPE_I32 &&
        ds4_env_flag("LUCE_DS4_TP_SCHED_TRACE_IDS")) {
        const size_t count = std::min<size_t>(
            (size_t) ggml_nelements(ids), 32u);
        std::array<int32_t, 32> values{};
        // TRACE_ALL synchronizes the preceding node, so the peer-local remap
        // has completed before this diagnostic read.
        ggml_backend_tensor_get(ids, values.data(), 0,
                                count * sizeof(values[0]));
        std::fprintf(stderr, "[ds4-tp-trace] ids=");
        for (size_t i = 0; i < count; ++i) {
            std::fprintf(stderr, "%s%d", i ? "," : "", values[i]);
        }
        std::fprintf(stderr, "\n");
        if (!ask && tensor->op == GGML_OP_GET_ROWS &&
            tensor->buffer && tensor->type == GGML_TYPE_I32) {
            const size_t out_count = std::min<size_t>(
                (size_t) ggml_nelements(tensor), 32u);
            values.fill(0);
            ggml_backend_tensor_get(tensor, values.data(), 0,
                                    out_count * sizeof(values[0]));
            std::fprintf(stderr, "[ds4-tp-trace] get_rows_out=");
            for (size_t i = 0; i < out_count; ++i) {
                std::fprintf(stderr, "%s%d", i ? "," : "", values[i]);
            }
            std::fprintf(stderr, "\n");
        }
    }
    return true;
}

// Populate a cached verifier graph's LUT inputs from its fixed expert
// placement after allocation. Mask any hot/cold duplication defensively so
// every selected route is evaluated exactly once. Mutable LRU placement is
// rejected before this graph path is entered.
// Local expert id of a route an owner does not hold: -1 masks it exactly in
// the MUL_MAT_ID kernels; LUCE_DS4_TP_MASKED_ROUTES=0 restores the legacy
// zero-weight route to local expert 0.
static int32_t ds4_fused_invalid_route() {
    const char * masked_routes =
        std::getenv("LUCE_DS4_TP_MASKED_ROUTES");
    return !masked_routes || !*masked_routes || std::strcmp(masked_routes, "0") != 0
        ? -1
        : 0;
}

static void ds4_fused_verify_refresh_hybrid_luts(
        const DeepSeek4Weights & w,
        MoeHybridStorage * hybrid,
        DeepSeek4FusedDecodeGraph & fg,
        bool force_masked_routes = false) {
    if (!hybrid) return;
    // A route this owner does not hold is either sent to local slot 0 and
    // multiplied by a zero weight (correct, but it pays the full expert
    // matmul) or given id -1, which the MMVQ kernels retire early with an
    // exact zero. A cluster rank owns only its shard, so masking is where the
    // sharding actually saves work: it is not optional there.
    const int32_t invalid_route = force_masked_routes ? -1 : ds4_fused_invalid_route();
    for (size_t hi = 0; hi < fg.hybrid_inputs.size(); ++hi) {
        const int il = (int) hi;
        const MoeHybridLayerStorage & layer = hybrid->layers[(size_t) il];
        const MoeHybridGraphInputs & inputs = fg.hybrid_inputs[hi];
        const bool dynamic_route_balance = inputs.dynamic_route_balance;
        const MoeHybridOwnerMapView owner_maps =
            moe_hybrid_owner_maps(layer, dynamic_route_balance);
        const std::vector<int32_t> & decode_hot = *owner_maps.main;
        const std::vector<int32_t> & decode_cold = *owner_maps.peer;
        std::vector<int32_t> hot_lut((size_t) w.n_expert, invalid_route);
        std::vector<int32_t> cold_lut((size_t) w.n_expert, invalid_route);
        std::vector<float> hot_valid((size_t) w.n_expert, 0.0f);
        std::vector<float> cold_valid((size_t) w.n_expert, 0.0f);
        for (int ie = 0; ie < w.n_expert; ++ie) {
            int32_t hot = decode_hot[(size_t) ie];
            const int32_t cold = decode_cold[(size_t) ie];
            // A phase-specific placement may move a resident expert to the
            // peer only when the peer owns a copy. Otherwise retain its
            // physical main-owner route so no selected expert is dropped.
            if (!dynamic_route_balance && hot < 0 && cold < 0) {
                hot = layer.hot_local_by_global[(size_t) ie];
            }
            if (hot >= 0) {
                hot_lut[(size_t) ie] = hot;
                hot_valid[(size_t) ie] = 1.0f;
            }
            if (cold >= 0) {
                cold_lut[(size_t) ie] = cold;
                // In dynamic mode this LUT is the main-residency candidate
                // set, copied independently to peer. The peer selector keeps
                // its exact complement after the per-token main cap. Static
                // placement retains the ordinary cold-validity mask.
                cold_valid[(size_t) ie] = dynamic_route_balance
                    ? (hot >= 0 ? 1.0f : 0.0f)
                    : (hot < 0 ? 1.0f : 0.0f);
            }
        }
        ds4_fv_set_repeated_rows(inputs.hot_local_lut, hot_lut);
        ds4_fv_set_repeated_rows(inputs.hot_valid_lut, hot_valid);
        ds4_fv_set_repeated_rows(inputs.cold_local_lut, cold_lut);
        ds4_fv_set_repeated_rows(inputs.cold_valid_lut, cold_valid);
    }
}

// Streamed experts of a fused graph run as a third owner over the device
// cache's slot stacks. Each such layer posts its routes to the cache's host
// mailbox in the graph and waits there for their slots; the mailbox resolver
// pins and loads them meanwhile. Pins are released after the graph.
static std::vector<MoeStreamedMailbox::Job> ds4_fused_stream_jobs(
        const DeepSeek4FusedDecodeGraph & fg) {
    std::vector<MoeStreamedMailbox::Job> jobs;
    for (const auto & route : fg.authoritative_routes) {
        if (!fg.hybrid_inputs[(size_t) route.layer].stream_post) continue;
        jobs.push_back({route.layer, route.width * route.n_tokens, route.n_tokens});
    }
    return jobs;
}

// Runs a fused graph, answering its streamed layers from the cache's
// mailbox while it runs. A layer that could not be answered exactly fails
// the launch.
static enum ggml_status ds4_fused_graph_compute(
        const DeepSeek4Weights & w, DeepSeek4FusedDecodeGraph & fg, ggml_backend_t backend,
        MoeHybridStorage * hybrid, const char * who,
        const std::function<void()> & while_running = {}) {
    MoeStreamedMailbox * mailbox =
        hybrid && hybrid->expert_cache ? hybrid->expert_cache->mailbox() : nullptr;
    const std::vector<MoeStreamedMailbox::Job> jobs =
        mailbox ? ds4_fused_stream_jobs(fg) : std::vector<MoeStreamedMailbox::Job>{};
    if (!jobs.empty()) mailbox->begin(jobs, ds4_fused_invalid_route());
    enum ggml_status status;
    if (fg.sched && jobs.empty() && while_running) {
        // Host work that needs no device result (building the next shape)
        // runs while the GPUs execute this graph.
        status = ggml_backend_sched_graph_compute_async(fg.sched, fg.sg.gf);
        if (status == GGML_STATUS_SUCCESS) while_running();
        ggml_backend_sched_synchronize(fg.sched);
    } else {
        status = fg.sched
            ? ggml_backend_sched_graph_compute(fg.sched, fg.sg.gf)
            : ggml_backend_graph_compute(backend, fg.sg.gf);
    }
    if (!jobs.empty()) {
        std::string error;
        if (!mailbox->end(&error)) {
            std::fprintf(stderr, "[%s] streamed experts: %s\n", who, error.c_str());
            status = GGML_STATUS_FAILED;
        }
    }
    return status;
}

template <typename T>
static bool ds4_fused_read_route_matrix(
        ggml_tensor * tensor,
        ggml_type expected_type,
        int width,
        int n_tokens,
        std::vector<T> & out) {
    if (!tensor || tensor->type != expected_type || width <= 0 ||
        n_tokens <= 0 || tensor->ne[0] < width ||
        tensor->ne[1] < n_tokens || tensor->nb[0] != sizeof(T)) {
        return false;
    }
    const size_t row_bytes = sizeof(T) * (size_t) width;
    const size_t tensor_bytes = ggml_nbytes(tensor);
    out.resize((size_t) width * n_tokens);
    for (int token = 0; token < n_tokens; ++token) {
        const size_t offset = (size_t) token * tensor->nb[1];
        if (offset > tensor_bytes || row_bytes > tensor_bytes - offset) {
            out.clear();
            return false;
        }
        ggml_backend_tensor_get(
            tensor, out.data() + (size_t) token * width,
            offset, row_bytes);
    }
    return true;
}

static void ds4_fused_consume_route_diagnostics(
        DeepSeek4FusedDecodeGraph & fg,
        const MoeHybridStorage * hybrid,
        MoeHybridRoutingStats * routing_stats,
        const int32_t * active_slots = nullptr) {
    const bool cache_audit =
        ds4_env_flag("LUCE_DS4_TP_CACHE_AUDIT");
    if (!routing_stats && !cache_audit) return;

    uint64_t cache_hot = 0;
    uint64_t cache_total = 0;
    for (const auto & route : fg.authoritative_routes) {
        std::vector<int32_t> ids;
        std::vector<float> weights;
        if (!ds4_fused_read_route_matrix(
                route.selected, GGML_TYPE_I32, route.width,
                route.n_tokens, ids) ||
            !ds4_fused_read_route_matrix(
                route.weights, GGML_TYPE_F32, route.width,
                route.n_tokens, weights)) {
            std::fprintf(stderr,
                "[ds4-route-audit] failed to read layer=%d lane=%d\n",
                route.layer, route.lane_start);
            continue;
        }
        for (int token = 0; token < route.n_tokens; ++token) {
            if (active_slots && active_slots[route.lane_start + token] < 0) continue;
            const int32_t * token_ids = ids.data() +
                (size_t) token * route.width;
            const float * token_weights = weights.data() +
                (size_t) token * route.width;
            observe_active_routing(
                routing_stats, route.layer, token_ids, token_weights,
                route.width);
            if (cache_audit && hybrid && route.layer >= 0 &&
                route.layer < (int) hybrid->layers.size()) {
                const MoeHybridLayerStorage & layer =
                    hybrid->layers[(size_t) route.layer];
                const std::vector<int32_t> & decode_hot =
                    layer.decode_hot_local_by_global.empty()
                        ? layer.hot_local_by_global
                        : layer.decode_hot_local_by_global;
                for (int i = 0; i < route.width; ++i) {
                    const int32_t id = token_ids[i];
                    if (token_weights[i] == 0.0f || id < 0 ||
                        id >= (int32_t) decode_hot.size()) {
                        continue;
                    }
                    ++cache_total;
                    if (decode_hot[(size_t) id] >= 0) {
                        ++cache_hot;
                    }
                }
            }
        }
    }

    if (cache_audit && cache_total > 0) {
        std::fprintf(stderr,
            "[ds4-route-cache-audit] hot=%llu total=%llu hit=%.3f\n",
            (unsigned long long) cache_hot,
            (unsigned long long) cache_total,
            (double) cache_hot / (double) cache_total);
    }
}

// Each hybrid cache slot owns a multi-backend scheduler, and each scheduler's
// gallocr retains its per-backend scratch buffers until the slot is destroyed,
// so the hybrid profile caps how many of the kSlotCount verifier shapes stay
// alive. LUCE_DS4_TP_FUSED_CACHE_SLOTS lowers or raises that cap without
// affecting the single-backend verifier cache.
static size_t ds4_fused_verify_hybrid_slot_limit() {
    static const size_t limit = []() {
        const char * raw = std::getenv("LUCE_DS4_TP_FUSED_CACHE_SLOTS");
        // Every miss is a full rebuild of the whole-model graph, 5.4 ms of an
        // 88.9 ms step doing no arithmetic, while the slots themselves are
        // nearly free: the graphs share their scratch through the allocator,
        // so the whole cache measured 303 MiB at two slots and 351 MiB at
        // 64, about 0.8 MiB per slot.
        //
        // q<=4: the shape set needs four slots and an adaptive width moving
        // between q=2, 3 and 4 needs more; eight stay under the nine that
        // were model-qualified at 28.5 GiB peak on the 31.9 GiB R9700.
        // q5 (adaptive q2..q5): four widths across four ratio-4 phases plus
        // the boundary shapes, i.e. every one of the kSlotCount resident
        // shapes stays warm. Device scratch is shared (about 0.8 MiB per
        // slot); each built slot also keeps its 32 MiB host metadata arena,
        // so a fully populated q5 cache holds up to 768 MiB of host memory.
        const long default_slots = ds4_env_flag("LUCE_DS4_Q5_VERIFY")
            ? (long) Ds4FusedVerifyCache::kSlotCount : 8;
        const long requested = raw ? std::strtol(raw, nullptr, 10)
                                   : default_slots;
        return (size_t) std::max<long>(
            1, std::min<long>(requested,
                              (long) Ds4FusedVerifyCache::kSlotCount));
    }();
    return limit;
}

static size_t ds4_fused_capture_hybrid_slot_limit() {
    static const size_t limit = []() {
        const char * raw = std::getenv("LUCE_DS4_TP_CAPTURE_CACHE_SLOTS");
        const long requested = raw ? std::strtol(raw, nullptr, 10) : 4;
        return (size_t) std::max<long>(
            1, std::min<long>(requested,
                              (long) Ds4FusedVerifyCache::kSlotCount));
    }();
    return limit;
}

static ggml_tensor * ds4_fused_weights_for_selected(
        ggml_context * ctx,
        ggml_tensor * cur,
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L,
        ggml_tensor * selected,
        int n_tokens) {
    ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, cur);
    ggml_tensor * probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));
    ggml_tensor * probs_3d = ggml_reshape_3d(
        ctx, probs, 1, w.n_expert, n_tokens);
    ggml_tensor * weights = ggml_get_rows(ctx, probs_3d, selected);
    const int n_used = (int) selected->ne[0];
    weights = ggml_reshape_2d(ctx, weights, n_used, n_tokens);
    ggml_tensor * sum = ggml_clamp(
        ctx, ggml_sum_rows(ctx, weights), 6.103515625e-5f, INFINITY);
    weights = ggml_div(ctx, weights, sum);
    if (w.expert_weight_scale != 1.0f) {
        weights = ggml_scale(ctx, weights, w.expert_weight_scale);
    }
    return weights;
}

static bool ds4_build_fused_verify_graph(
        DeepSeek4FusedDecodeCache & mc,          // fn mirrors (shared with AR fused)
        DeepSeek4FusedDecodeGraph & fg,
        Ds4FusedVerifyCache::Extra & ex,
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        DeepSeek4Cache * cache,
        const std::vector<HcLayerWeightsCpu> & hc_weights,
        const HcWeightsCpu & hc_out_weights,
        const std::vector<HashRoutingTableCpu> & hash_tables,
        int kv_start,
        int q,
        bool have_token_ids,
        const std::vector<int> & capture_ids,
        MoeHybridStorage * hybrid,
        std::vector<int64_t> && shape_key,
        DeepSeek4PagedCache * paged_cache = nullptr,
        const std::vector<std::vector<DeepSeek4GatheredLaneRows>> * paged_rows = nullptr,
        bool paged_history_buckets = false) {
    const bool report_build_timing =
        ds4_env_flag("LUCE_DS4_VERIFY_BUILD_TIMING");
    const auto build_t0 = Ds4TimingClock::now();
    const bool paged_mode = paged_cache && paged_rows;
    if (!paged_mode && !cache) return false;
    if (paged_mode) {
        if (q < 1 || q > 16 ||
            paged_rows->size() != (size_t) w.n_layer) {
            return false;
        }
        for (const auto & layer_rows : *paged_rows) {
            if (layer_rows.size() != (size_t) q) return false;
        }
    }
    // A step that packs chronological prompt rows keeps the established
    // per-column arithmetic: dense projections run in parts sized by weight
    // type (ROCmFP4 sixteen columns on the gfx1151 weight-reuse kernel,
    // narrow F16 eight on MMVF, four otherwise), all bit-identical per column
    // to the single-lane kernels, and routed experts run in parts of at most
    // eight rows on the multi-token expert kernel. A round with one row per
    // sequence keeps its unpartitioned graph.
    int projection_columns = 0;
    int moe_columns = 0;
    if (paged_mode && !hybrid) {
        std::vector<int32_t> active_slots;
        int active_rows = 0;
        for (const auto & row : (*paged_rows)[0]) {
            if (row.slot < 0) continue;
            ++active_rows;
            if (std::find(active_slots.begin(), active_slots.end(), row.slot) == active_slots.end())
                active_slots.push_back(row.slot);
        }
        if (active_rows > (int)active_slots.size()) {
            projection_columns = 4;
            // The multi-token expert kernel gives every token its own warp
            // with a fixed per-row K traversal, so a part's arithmetic does
            // not depend on how many tokens share the launch. The paged
            // compute scope keeps these eight-row parts on FP2/FP3 MMV;
            // the default five-row cutoff would instead select MMQ. A
            // single live sequence keeps one-row parts: its decode rounds use
            // the single-token expert kernel, whose arithmetic differs from
            // the multi-token kernel's.
            moe_columns = active_slots.size() == 1
                ? 1 : std::min(DEEPSEEK4_MAX_MOE_ROWS_PER_PART, active_rows);
        }
    }
    const size_t graph_capacity = q > 8 ? 131072u : 65536u;
    std::array<ggml_backend_t, 3> sched_backends{};
    if (hybrid) {
        sched_backends = {backend, hybrid->cold_backend, hybrid->cpu_backend};
    }
    const bool reuse_sched =
        hybrid && fg.sched_reusable(sched_backends, graph_capacity);
    if (fg.sched && !reuse_sched) {
        ggml_backend_sched_free(fg.sched);
        fg.sched = nullptr;
        fg.sched_capacity = 0;
        fg.sched_backends = {};
    }
    step_graph_free(fg.sg);
    fg.reset_nodes();
    { const bool keep_vocab_split = ex.vocab_split; ex.reset(); ex.vocab_split = keep_vocab_split; }
    fg.hash_ids.assign((size_t) w.n_layer, nullptr);
    ex.boundary_checkpoint.layers.resize((size_t) w.n_layer);
    if (hybrid) {
        fg.hybrid_inputs.assign((size_t) w.n_layer, {});
    }
    const int n_embd = w.n_embd;
    const int n_hc = w.n_hc;
    const BackendPairCapabilities pair_capabilities = hybrid
        ? backend_pair_capabilities(backend, hybrid->cold_backend)
        : BackendPairCapabilities{true, true};
    const bool same_gpu_runtime = pair_capabilities.same_runtime;
    // Expert-parallel cluster (from maikzz32's lucebox-halo-cluster): the
    // hybrid owners hold this rank's shard of the routed experts (hot on the
    // primary GPU, cold on the secondary), routes to other ranks' experts are
    // masked in the owner LUTs, the joined routed partial is summed across the
    // ranks by an in-graph all-reduce on the primary's stream, and the
    // shared expert is added to that sum.
    Ds4ClusterRuntime * cluster_rt = cache ? cache->cluster_rt : nullptr;
    const bool cluster_hybrid = hybrid && cluster_rt && cluster_rt->size() > 1;
    std::vector<ggml_tensor *> cluster_nodes;
    // Attention head parallelism: this rank's query heads, in
    // whole output groups; 0 = attention replicated.
    const int attn_head_begin = cluster_hybrid ? cluster_rt->attn_head_begin : 0;
    const int attn_head_count = cluster_hybrid ? cluster_rt->attn_head_count : 0;
    const int attn_n_head = attn_head_count > 0 ? attn_head_count : w.n_head;

    // The 43-layer q<=5 verifier currently consumes at most ~6.5 MiB of
    // metadata.  Keeping 32 MiB leaves nearly 5x headroom while avoiding the
    // cost (and retained UMA footprint) of zero-filling 256 MiB for every
    // compressor-boundary shape in the cache.
    const size_t arena_size = 32u * 1024 * 1024;
    if (fg.sg.meta_arena.size() < arena_size) fg.sg.meta_arena.resize(arena_size);
    const auto arena_t1 = Ds4TimingClock::now();
    ggml_init_params params{};
    params.mem_size = fg.sg.meta_arena.size();
    params.mem_buffer = fg.sg.meta_arena.data();
    params.no_alloc = true;
    fg.sg.ctx = ggml_init(params);
    if (!fg.sg.ctx) return false;
    ggml_context * ctx = fg.sg.ctx;
    // Gathered paged attention builds one q=1 MLA lane per concurrent
    // sequence. Above eight lanes the resulting whole-model graph exceeds
    // the verifier-era 64K scheduler hash table even though the metadata
    // arena still has ample room.
    fg.sg.gf = ggml_new_graph_custom(ctx, graph_capacity, false);
    ggml_cgraph * gf = fg.sg.gf;

    fg.inp_embed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, q);
    ggml_set_input(fg.inp_embed);
    const DeepSeek4EngramRuntime * engram = w.engram_runtime.get();
    if (engram) {
        ex.engram_keys = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, (int64_t) engram->key_floats(), q,
                                            engram->n_layers());
        ggml_set_input(ex.engram_keys);
    }
    // Engram tables live on rank 0; its decoded rows reach the other ranks.
    ggml_tensor * engram_keys_all = ex.engram_keys;
    if (cluster_hybrid && engram_keys_all) {
        engram_keys_all = ds4_cluster_broadcast_node(ctx, ex.engram_keys, *cluster_rt);
        if (!engram_keys_all) return false;
        cluster_nodes.push_back(engram_keys_all);
    }
    ex.q = q;
    ex.pos_q = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, q);  ggml_set_input(ex.pos_q);
    ex.neg_q = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, q);  ggml_set_input(ex.neg_q);
    ex.rawrows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, q); ggml_set_input(ex.rawrows);
    ex.saved_rawrows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, q);
    ggml_set_input(ex.saved_rawrows);
    if (ds4_lane_batch_comp_enabled() && !paged_mode && q > 1) {
        ex.pos_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, q);
        ggml_set_input(ex.pos_rows);
    }
    for (uint32_t ratio : w.compress_ratios) {
        if (ratio == 0 || ex.rows_for((int) ratio)) continue;
        auto & rows = ex.ratio_rows.emplace_back();
        rows.ratio = (int) ratio;
        rows.ape = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, q);   ggml_set_input(rows.ape);
        rows.state = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, q); ggml_set_input(rows.state);
    }
    // Up to two compressor emissions per layer and attention lane. q<=4 uses
    // at most one; q=5 can touch two ratio-4 boundaries when it starts at
    // position mod 4 == 3; a tokenwise lane emits at most one.
    const int n_lanes = paged_mode ? 1 : ds4_fused_attention_lanes(w, q);
    const int attn_lane_q = q / n_lanes;
    // Tokenwise lanes keep the six routed experts in one owner graph (see the
    // routed FFN below); LUCE_DS4_TP_HEAD4_TAIL2=1 restores the 4 + 2
    // decomposition for bisection.
    static const bool legacy_head4_tail2 = ds4_env_flag("LUCE_DS4_TP_HEAD4_TAIL2");
    const bool whole_routes_for_lanes = n_lanes > 1 && !legacy_head4_tail2;
    const int64_t comp_slots = 2 * (int64_t) w.n_layer * n_lanes;
    fg.i32_bundle = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, comp_slots);
    ggml_set_input(fg.i32_bundle);
    fg.i64_bundle = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, comp_slots);
    ggml_set_input(fg.i64_bundle);

    // Mask bundle: q>1 preserves the q ring rows that the batch overwrites,
    // so its attention span is [n_swa + padded + q].  q=1 reads the current
    // row through the in-graph ring update and has no preserved-row suffix,
    // matching the established single-token fused-decode topology.
    int64_t mask_total = 0;
    if (paged_mode && paged_history_buckets) {
        for (int il = 0; il < w.n_layer; ++il) {
            const int ratio = (int) w.compress_ratios[il];
            for (int t = 0; t < q; ++t) {
                const auto & rows = (*paged_rows)[(size_t) il][(size_t) t];
                const int emitted =
                    ratio > 0 && rows.compressed_emitted ? 1 : 0;
                mask_total += (int64_t) rows.raw_history.size() + 1 +
                              rows.compressed_history.size() + emitted;
            }
        }
    } else if (!paged_mode) {
        for (int il = 0; il < w.n_layer; ++il) {
            const DeepSeek4LayerCache & clc = ds4_comp_cache(*cache, w, il);
            for (int lane = 0; lane < n_lanes; ++lane) {
                const int padded = ds4_fused_lane_padded(
                    w, clc, il, kv_start + (lane + 1) * attn_lane_q - 1);
                const int preserved_rows = attn_lane_q > 1 ? attn_lane_q : 0;
                mask_total += (int64_t) (w.n_swa + padded + preserved_rows) * attn_lane_q;
            }
        }
    }
    fg.mask_bundle = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
                                        std::max<int64_t>(mask_total, 1));
    ggml_set_input(fg.mask_bundle);
    // LUCE_DS4_MASK_F16_ONCE=1: one F16 copy of the whole mask per step for
    // the flash lanes instead of one cast per lane and layer.
    fg.mask_bundle_f16 = !paged_mode && ds4_env_flag("LUCE_DS4_MASK_F16_ONCE")
        ? ggml_cast(ctx, fg.mask_bundle, GGML_TYPE_F16) : nullptr;
    // LUCE_DS4_EXPLICIT_SPLIT=1: the identity row list that sends
    // short-context lanes to the split-KV flash kernel.
    fg.identity_rows = nullptr;
    if (!paged_mode && ds4_env_flag("LUCE_DS4_EXPLICIT_SPLIT") && w.n_indexer_top_k > 1) {
        fg.identity_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, w.n_indexer_top_k);
        ggml_set_input(fg.identity_rows);
    }
    int64_t mask_off = 0;

    // Back every gathered lane's scalar and index inputs with shared
    // tensors. Their sizes are fixed by the prepared history shape key.
    int64_t paged_gather_total = 0;
    if (paged_mode) {
        for (int il = 0; il < w.n_layer; ++il) {
            const int layer_ratio = (int) w.compress_ratios[il];
            for (int t = 0; t < q; ++t) {
                const auto & rows = (*paged_rows)[(size_t) il][(size_t) t];
                paged_gather_total +=
                    (int64_t) rows.raw_history.size() + 1;
                if (layer_ratio > 0) {
                    paged_gather_total += (int64_t) std::max<size_t>(
                        rows.compressed_history.size(), 1);
                }
            }
        }
        ex.paged_i32_n = (int64_t) 5 * w.n_layer * q;
        ex.paged_i64_n = (int64_t) 8 * w.n_layer * q;
        ex.paged_gather_n = std::max<int64_t>(paged_gather_total, 1);
        ex.paged_i32 = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I32, ex.paged_i32_n);
        ggml_set_input(ex.paged_i32);
        ex.paged_i64 = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I64, ex.paged_i64_n);
        ggml_set_input(ex.paged_i64);
        ex.paged_gather = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I32, ex.paged_gather_n);
        ggml_set_input(ex.paged_gather);
    }
    int64_t paged_gather_off = 0;

    // Keep every speculative or gathered lane in one HC tensor. HC kernels
    // already support a token dimension, so issue one batched op per boundary.
    ggml_tensor * embed_3d = ggml_reshape_3d(
        ctx, fg.inp_embed, n_embd, 1, q);
    ggml_tensor * hc_repeated = ggml_repeat_4d(
        ctx, embed_3d, n_embd, n_hc, q, 1);
    ggml_tensor * hc_cur = ggml_reshape_2d(
        ctx, hc_repeated, (int64_t) n_embd * n_hc, q);

    std::vector<ggml_tensor *> capture_layers(capture_ids.size(), nullptr);
    std::vector<ggml_tensor *> attention_contexts;
    // Tokenwise lanes record every token's compressor window row so a
    // rejected suffix can be rolled back (Ds4VerifyWindowRows).
    std::vector<ggml_tensor *> window_kv_rows, window_score_rows;
    if (n_lanes > 1) {
        for (int il = 0; il < w.n_layer; ++il) {
            if (deepseek4_is_window_state(cache->layers[(size_t) il].attn_compressor,
                                          (int) w.compress_ratios[il])) {
                ex.window_layers.push_back(il);
            }
        }
    }
    // Staggered pre-mix: the pre coefficients of the previous sub-block.
    ggml_tensor * carried_pre = nullptr;
    // The latest index source's top-k of each attention lane (paged: of each
    // row), for the layers after it (w.shared_index_topk).
    std::vector<ggml_tensor *> lane_selection((size_t) (paged_mode ? q : n_lanes), nullptr);
    // Lane batch: the batched ops' selections, shared across layers, and the
    // latest index source's selection for all lanes (lane_selection's twin).
    Ds4LaneBatchSelections lane_batch_selections;
    ggml_tensor * lane_batch_carrier = nullptr;

    // LUCE_DS4_ENGRAM_PRECOMPUTE=<rows> (hybrid exchange): an Engram layer's
    // projection engram_wkv @ keys depends only on the step's tokens, so it is
    // computed in row chunks of that many rows, one after each of the
    // preceding layers' FFN exports, where the dGPU waits for the iGPU's sum;
    // the Engram layer computes whatever rows remain and joins them. Each row
    // is the same MMVF product as in the whole projection.
    static const int engram_chunk_rows = [] {
        const char * v = std::getenv("LUCE_DS4_ENGRAM_PRECOMPUTE");
        return v && *v ? std::max(0, std::atoi(v)) : 0;
    }();
    std::vector<std::vector<ggml_tensor *>> engram_kv_parts(engram ? (size_t) engram->n_layers() : 0u);
    std::vector<int64_t> engram_kv_done(engram_kv_parts.size(), 0);
    // The chunk (Engram layer index, first row) queued after layer il's export.
    auto engram_chunk_for = [&](int il, int & e_out, int64_t & r0_out) -> bool {
        if (!engram || engram_chunk_rows <= 0 || !engram_keys_all) return false;
        for (int e = 0; e < engram->n_layers(); ++e) {
            const int il_e = engram->layer_id(e);
            const int64_t rows = w.layers[(size_t) il_e].engram_wkv
                ? w.layers[(size_t) il_e].engram_wkv->ne[1] : 0;
            const int n_chunks = (int) ((rows + engram_chunk_rows - 1) / engram_chunk_rows);
            // Chunks fill the layers just before il_e, the last chunk last.
            const int first = std::max(0, il_e - n_chunks);
            if (il < first || il >= il_e) continue;
            const int64_t r0 = (int64_t) (il - (il_e - n_chunks)) * engram_chunk_rows;
            if (r0 < 0 || r0 >= rows || r0 != engram_kv_done[(size_t) e]) continue;
            e_out = e; r0_out = r0;
            return true;
        }
        return false;
    };

    for (int il = 0; il < w.n_layer; ++il) {
        const DeepSeek4Layer & L = w.layers[(size_t) il];
        DeepSeek4LayerCache * lc =
            paged_mode ? nullptr : &cache->layers[(size_t) il];
        DeepSeek4LayerCache * comp_lc =
            paged_mode ? nullptr : &ds4_comp_cache(*cache, w, il);
        const HcLayerWeightsCpu & hlw = hc_weights[(size_t) il];
        const int ratio = (int) w.compress_ratios[il];

        const int lane_q = q;
        const int lane_start = 0;
        const int lane_kv_start = kv_start + lane_start;
        const size_t hybrid_idx = (size_t) il;

        // ── Engram: the n-gram memory enters the residual copies ──
        if (const int e = engram ? engram->layer_index(il) : -1; e >= 0) {
            ggml_tensor * keys = ggml_view_2d(ctx, engram_keys_all, engram_keys_all->ne[0], q,
                                              engram_keys_all->nb[1], (size_t) e * engram_keys_all->nb[2]);
            ggml_tensor * h3 = ggml_reshape_3d(ctx, ggml_cont(ctx, hc_cur), n_embd, n_hc, q);
            ggml_tensor * kv_pre = nullptr;
            if (!engram_kv_parts[(size_t) e].empty()) {
                const int64_t rows = L.engram_wkv->ne[1];
                const int64_t done = engram_kv_done[(size_t) e];
                for (ggml_tensor * part : engram_kv_parts[(size_t) e]) {
                    kv_pre = kv_pre ? ggml_concat(ctx, kv_pre, part, 0) : part;
                }
                if (done < rows) {
                    ggml_tensor * rest = ggml_mul_mat(ctx,
                        ggml_view_2d(ctx, L.engram_wkv, L.engram_wkv->ne[0], rows - done,
                                     L.engram_wkv->nb[1], (size_t) done * L.engram_wkv->nb[1]),
                        keys);
                    ggml_mul_mat_set_prec(rest, GGML_PREC_F32);
                    kv_pre = ggml_concat(ctx, kv_pre, rest, 0);
                }
            }
            ggml_tensor * updated = deepseek4_build_engram_apply(ctx, h3, keys, L, n_embd, n_hc, w.rms_eps,
                                                                 nullptr, kv_pre);
            if (!updated) return false;
            hc_cur = ggml_reshape_2d(ctx, updated, (int64_t) n_embd * n_hc, q);
        }

        // ── Batched HC pre (attention) ──
        ggml_tensor * hc_flat = hc_cur;
        ggml_tensor * split_attn = nullptr;
        ggml_tensor * attn_working = ds4_build_fused_hc_pre(
            ctx, w, hc_flat, mc.fn_attn_f16[(size_t) il], L.hc_attn_base,
            hlw.attn, &split_attn, projection_columns);
        if (!attn_working) return false;
        // HC-pre stores working and split values together per token. Materialize
        // the strided working view once, replacing q kernels and q-1 concats.
        // The staggered pre-mix collapses with the previous sub-block's pre.
        ggml_tensor * attn_in = w.hc_staggered_pre
            ? ds4_build_hc_collapse(ctx, hc_flat, carried_pre, n_embd, n_hc)
            : ggml_cont(ctx, attn_working);
        if (w.hc_staggered_pre) carried_pre = ds4_hc_split_pre(ctx, split_attn, n_hc);

        // ── Attention ──
        ggml_tensor * attn_out = nullptr;
        if (paged_mode) {
            // q denotes independent decode lanes here, not a causal sequence.
            // Gather each lane's immutable chronological history and run the
            // established MLA lane core at q=1; all surrounding HC/MoE/output
            // machinery remains q-wide and unchanged.
            DeepSeek4PagedLayerCache & plc = paged_cache->layers[(size_t) il];
            // Compressed rows and compressor state live at the kv source.
            DeepSeek4PagedLayerCache & cplc = ds4_comp_cache(*paged_cache, w, il);
            ggml_tensor * raw_flat = ggml_reshape_2d(
                ctx, plc.raw_kv, w.head_dim,
                (int64_t) DS4_PAGE_TOKENS * paged_cache->plan.slots);
            ggml_tensor * attn_normed =
                build_rms_norm(ctx, attn_in, L.attn_norm, w.rms_eps);
            DeepSeek4PreparedProjectedLane batched_proj =
                build_mla_qkv_projection(ctx, attn_normed, w, L, q, projection_columns);
            // Monolithic paged serving uses gfx1151. Its ROCmFP4 MMVQ
            // preserves each column's arithmetic through sixteen rows.
            // Only projections are shared: same-slot compressor mutations
            // remain chronological inside each lane below.
            if (!hybrid && q > 1 && ratio > 0 && L.attn_compressor_kv &&
                L.attn_compressor_gate &&
                L.attn_compressor_kv->type == GGML_TYPE_Q4_0_ROCMFP4_FAST &&
                L.attn_compressor_gate->type == GGML_TYPE_Q4_0_ROCMFP4_FAST) {
                batched_proj.compressor_kv = ds4_mul_mat_columns(
                    ctx, L.attn_compressor_kv, attn_normed, projection_columns);
                batched_proj.compressor_score = ds4_mul_mat_columns(
                    ctx, L.attn_compressor_gate, attn_normed, projection_columns);
            }
            build_mla_qkv_rope(
                ctx, batched_proj, w, ds4_rope_params(w, ratio), q,
                ex.pos_q, /*fuse_q_rope=*/false);
            ggml_tensor * contexts = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, w.head_dim * w.n_head, q);
            if (hybrid) attention_contexts.push_back(contexts);
            bool independent = true;
            int64_t raw_count = 0, comp_count = 0;
            for (int t = 0; t < q; ++t) {
                const auto & rows = (*paged_rows)[(size_t) il][(size_t) t];
                for (int prior = 0; prior < t; ++prior) {
                    if (rows.slot == (*paged_rows)[(size_t) il][(size_t) prior].slot)
                        independent = false;
                }
            }
            for (int t = 0; t < q; ++t) {
                const auto & rows = (*paged_rows)[(size_t) il][(size_t) t];
                raw_count += independent ? rows.raw_history.size() + 1
                    : std::max<size_t>(rows.raw_history.size(), 1);
                if (ratio > 0) comp_count += std::max<size_t>(rows.compressed_history.size(), 1);
            }
            const int64_t raw_begin = paged_gather_off;
            const int64_t comp_begin = raw_begin + raw_count;
            int64_t raw_offset = raw_begin, comp_offset = comp_begin;
            for (int t = 0; t < q; ++t) {
                const auto & rows = (*paged_rows)[(size_t) il][(size_t) t];
                auto & px = ex.paged.emplace_back();
                const int64_t lane_idx = (int64_t) il * q + t;
                px.i32_base = lane_idx * 5;
                px.i64_base = lane_idx * 7;
                auto scalar_i32 = [&](int slot) {
                    return ggml_view_1d(
                        ctx, ex.paged_i32, 1,
                        (size_t) (px.i32_base + slot) * sizeof(int32_t));
                };
                auto scalar_i64 = [&](int slot) {
                    return ggml_view_1d(
                        ctx, ex.paged_i64, 1,
                        (size_t) (px.i64_base + slot) * sizeof(int64_t));
                };
                px.pos = scalar_i32(0);
                px.neg_pos = scalar_i32(1);
                px.raw_n = independent
                    ? (int64_t) rows.raw_history.size() + 1
                    : (int64_t) std::max<size_t>(rows.raw_history.size(), 1);
                px.raw_off = raw_offset;
                px.raw_gather = ggml_view_1d(
                    ctx, ex.paged_gather, px.raw_n,
                    (size_t) px.raw_off * sizeof(int32_t));
                raw_offset += px.raw_n;
                if (ratio > 0) {
                    px.comp_n = (int64_t) std::max<size_t>(
                        rows.compressed_history.size(), 1);
                    px.comp_off = comp_offset;
                    px.comp_gather = ggml_view_1d(
                        ctx, ex.paged_gather, px.comp_n,
                        (size_t) px.comp_off * sizeof(int32_t));
                    comp_offset += px.comp_n;
                    if (ratio == 4) {
                        // The indexer and attention compressors address the
                        // same chronological compressed rows. Explicit mode
                        // has no indexer consumer, so share the gather and do
                        // not emit the otherwise-dead index-history read.
                        px.index_gather = px.comp_gather;
                    }
                }
                px.raw_write = scalar_i64(0);
                if (ratio > 0) {
                    px.comp_write = scalar_i64(1);
                    px.comp_read = scalar_i32(2);
                    px.ape = scalar_i32(3);
                    px.state_row = scalar_i64(2);
                    px.comp_pos = scalar_i32(4);
                }
            }
            paged_gather_off = comp_offset;
            ggml_tensor * raw_stage = nullptr;
            ggml_tensor * comp_stage = nullptr;
            if (independent) {
                ggml_tensor * indices = ggml_view_1d(ctx, ex.paged_gather, raw_count,
                    (size_t) raw_begin * sizeof(int32_t));
                raw_stage = ggml_get_rows(ctx, raw_flat, indices);
                ggml_tensor * current_rows = ggml_view_1d(ctx, ex.paged_i64, q,
                    (size_t) (7 * w.n_layer * q + il * q) * sizeof(int64_t));
                raw_stage = ggml_set_rows(ctx, raw_stage, ggml_cont(ctx, batched_proj.kv), current_rows);
                ggml_build_forward_expand(gf, raw_stage);
                if (comp_count > 0) {
                    indices = ggml_view_1d(ctx, ex.paged_gather, comp_count,
                        (size_t) comp_begin * sizeof(int32_t));
                    comp_stage = ggml_get_rows(ctx, cplc.comp_kv, indices);
                    ggml_build_forward_expand(gf, comp_stage);
                }
            }
            for (int t = 0; t < q; ++t) {
                const auto & rows = (*paged_rows)[(size_t) il][(size_t) t];
                auto & px = ex.paged[(size_t) il * q + t];
                ggml_tensor * raw_history = rows.raw_history.empty() ? nullptr
                    : raw_stage ? ggml_view_2d(ctx, raw_stage, w.head_dim,
                        rows.raw_history.size(), raw_stage->nb[1],
                        (size_t) (px.raw_off - raw_begin) * raw_stage->nb[1])
                    : ggml_get_rows(ctx, raw_flat, px.raw_gather);
                ggml_tensor * comp_history = rows.compressed_history.empty() ? nullptr
                    : comp_stage ? ggml_view_2d(ctx, comp_stage, w.head_dim,
                        rows.compressed_history.size(), comp_stage->nb[1],
                        (size_t) (px.comp_off - comp_begin) * comp_stage->nb[1])
                    : ggml_get_rows(ctx, cplc.comp_kv, px.comp_gather);
                // V4.1 index sources score the index keys of the same rows.
                ggml_tensor * index_history = w.shared_index_topk && cplc.index_comp_kv &&
                        !rows.compressed_history.empty()
                    ? ggml_get_rows(ctx, cplc.index_comp_kv, px.comp_gather) : nullptr;
                // Stateless (ratio-1) sources carry no pooling state to view.
                DeepSeek4CompressorState attn_state{}, index_state{};
                if (ratio > 0 && rows.slot >= 0 && cplc.attn_compressor.state_kv) {
                    const size_t off = (size_t) rows.slot * cplc.attn_compressor.state_kv->nb[2];
                    attn_state.state_kv = ggml_view_2d(ctx, cplc.attn_compressor.state_kv,
                        cplc.attn_compressor.state_kv->ne[0], cplc.attn_compressor.state_kv->ne[1],
                        cplc.attn_compressor.state_kv->nb[1], off);
                    attn_state.state_score = ggml_view_2d(ctx, cplc.attn_compressor.state_score,
                        cplc.attn_compressor.state_score->ne[0], cplc.attn_compressor.state_score->ne[1],
                        cplc.attn_compressor.state_score->nb[1], off);
                    if (ratio == 4) {
                        const size_t io = (size_t) rows.slot * cplc.indexer_compressor.state_kv->nb[2];
                        index_state.state_kv = ggml_view_2d(ctx, cplc.indexer_compressor.state_kv,
                            cplc.indexer_compressor.state_kv->ne[0], cplc.indexer_compressor.state_kv->ne[1],
                            cplc.indexer_compressor.state_kv->nb[1], io);
                        index_state.state_score = ggml_view_2d(ctx, cplc.indexer_compressor.state_score,
                            cplc.indexer_compressor.state_score->ne[0], cplc.indexer_compressor.state_score->ne[1],
                            cplc.indexer_compressor.state_score->nb[1], io);
                    }
                }
                DeepSeek4MlaLaneBindings lane{};
                lane.history_mode = DeepSeek4MlaLaneBindings::HistoryMode::ChronologicalGathered;
                lane.prepared_raw_attention = raw_stage ? ggml_view_2d(
                    ctx, raw_stage, w.head_dim, rows.raw_history.size() + 1,
                    raw_stage->nb[1], (size_t) (px.raw_off - raw_begin) * raw_stage->nb[1]) : nullptr;
                lane.raw_history = raw_history; lane.n_raw_history = (int) rows.raw_history.size();
                lane.comp_history = comp_history; lane.n_comp_history = (int) rows.compressed_history.size();
                lane.index_comp_history = index_history;
                lane.n_index_comp_history = (int) rows.compressed_history.size();
                // Writes use the same flattened physical-row geometry as the
                // gather indices (not the slot-local 128-row second axis).
                lane.raw_kv = raw_flat; lane.comp_kv = cplc.comp_kv;
                lane.index_comp_kv = cplc.index_comp_kv;
                lane.raw_write_rows = px.raw_write; lane.comp_write_rows = px.comp_write;
                lane.index_comp_write_rows = px.comp_write;
                lane.comp_read_rows = px.comp_read;
                lane.index_comp_read_rows = px.comp_read;
                lane.write_raw = rows.slot >= 0;
                lane.write_comp = lane.write_raw && deepseek4_is_kv_source(w, il);
                lane.attn_compressor = &attn_state; lane.indexer_compressor = &index_state;
                lane.index_selection = &lane_selection[(size_t) t];
                DeepSeek4AttentionGraphInputs ci{};
                ci.rope_pos = px.pos; ci.neg_pos = px.neg_pos;
                if (ratio == 4) {
                    ci.flush_rows = ggml_view_1d(ctx, ex.paged_i64, 4,
                        (size_t) (px.i64_base + 3) * sizeof(int64_t));
                }
                ci.attn_ape_row = px.ape; ci.attn_state_rows = px.state_row;
                ci.attn_comp_pos = px.comp_pos; ci.index_ape_row = px.ape;
                ci.index_state_rows = px.state_row; ci.index_comp_pos = px.comp_pos;
                if (paged_history_buckets) {
                    px.mask_off = mask_off;
                    px.mask_n = (int64_t) rows.raw_history.size() + 1 +
                                rows.compressed_history.size() +
                                (ratio > 0 && rows.compressed_emitted ? 1 : 0);
                    ci.attn_row_mask = ggml_view_2d(
                        ctx, fg.mask_bundle, px.mask_n, 1,
                        (size_t) px.mask_n * sizeof(float),
                        (size_t) px.mask_off * sizeof(float));
                    ci.padded_comp = (int) rows.compressed_history.size() +
                        (ratio > 0 && rows.compressed_emitted ? 1 : 0);
                    mask_off += px.mask_n;
                }
                std::vector<DeepSeek4I32InputBinding> ib;
                std::vector<DeepSeek4I32ArrayBinding> iab;
                std::vector<DeepSeek4I64ArrayBinding> lab;
                ggml_tensor * col = ggml_view_2d(
                    ctx, attn_normed, n_embd, 1, attn_normed->nb[1],
                    (size_t) t * attn_normed->nb[1]);
                const DeepSeek4PreparedProjectedLane lane_proj =
                    ds4_slice_projected_lane(ctx, batched_proj, w, t, px.pos);
                ggml_tensor * lane_context = nullptr;
                ggml_tensor * one = build_mla_attention_lane_core(
                    ctx, gf, col, w, L, lane, il, (int) rows.position, 1,
                    &ci, ib, iab, lab, nullptr,
                    DeepSeek4AttentionImpl::Explicit, &lane_proj,
                    &lane_context);
                if (!one || !lane_context || !ib.empty() || !iab.empty() ||
                    !lab.empty()) {
                    std::fprintf(stderr,
                        "[deepseek4-paged] layer %d lane %d attention build "
                        "failed (graph=%d i32=%zu arrays=%zu i64=%zu, "
                        "first_array=%d)\n",
                        il, t, one != nullptr, ib.size(), iab.size(), lab.size(),
                        iab.empty() || iab[0].values.empty()
                            ? INT32_MIN : iab[0].values[0]);
                    return false;
                }
                // Complete the chronological read before a later prompt row
                // mutates this slot's raw ring or compressor state.
                ggml_build_forward_expand(gf, lane_context);
                ggml_tensor * destination = ggml_view_2d(
                    ctx, contexts, w.head_dim * w.n_head, 1, contexts->nb[1],
                    (size_t) t * contexts->nb[1]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, lane_context, destination));
            }
            const auto rope = ds4_rope_params(w, ratio);
            attn_out = build_tail_rope_3d(
                ctx, ggml_reshape_3d(ctx, contexts, w.head_dim, w.n_head, q),
                ex.neg_q, w.n_rot, w.head_dim, w.n_head, q,
                rope.freq, rope.scale, rope.ext, rope.attn,
                w.rope_yarn_beta_fast, w.rope_yarn_beta_slow, rope.n_ctx_orig);
            attn_out = ggml_reshape_2d(ctx, attn_out, w.head_dim * w.n_head, q);
            attn_out = build_mla_output_projection(
                ctx, attn_out, w, L, q, /*allow_grouped=*/false, projection_columns);
        } else {
        // ── Batched speculative attention ──
        // One lane for the batch, or one per token (ds4_fused_attention_lanes):
        // each lane is the attention core of a decode step at its position,
        // built in position order, so its ring write, compressor step and
        // attention see exactly the rows that decode step sees. Per-token
        // lanes share the batched q/kv projections with their RoPE, and the
        // inverse RoPE with the output projection (batch-invariant products),
        // as the host verify does (ds4_run_verify_attention).
        ggml_tensor * normed_all = build_rms_norm(ctx, attn_in, L.attn_norm, w.rms_eps);
        DeepSeek4PreparedProjectedLane batched_lanes{};
        ggml_tensor * lane_contexts = nullptr;
        if (n_lanes > 1) {
            batched_lanes = build_mla_qkv_projection(ctx, normed_all, w, L, q, 0,
                                                     attn_head_begin, attn_head_count);
            build_mla_qkv_rope(ctx, batched_lanes, w, ds4_rope_params(w, ratio), q, ex.pos_q,
                               /*fuse_q_rope=*/false);
        }
        const int window_idx = [&] {
            const auto it = std::find(ex.window_layers.begin(), ex.window_layers.end(), il);
            return it == ex.window_layers.end() ? -1 : (int) (it - ex.window_layers.begin());
        }();
        // LUCE_DS4_LANE_BATCH: single-token lanes over one padded compressed
        // span hand their attention to one op (build_mla_lane_batch_attention).
        bool lane_batch = false;
        int lane_batch_padded = 0;
        if (n_lanes > 1 && attn_lane_q == 1 && ds4_lane_batch_enabled()) {
            lane_batch_padded = ds4_fused_lane_padded(w, *comp_lc, il, kv_start);
            bool uniform = true;
            for (int lane = 1; lane < n_lanes && uniform; ++lane) {
                uniform = ds4_fused_lane_padded(w, *comp_lc, il, kv_start + lane) ==
                          lane_batch_padded;
            }
            lane_batch = uniform &&
                ds4_lane_batch_layer_ok(w, il, *lc, *comp_lc, lane_batch_padded, fg.identity_rows);
        }
        // Past top_k compressed rows the lanes attend over V4.1's shared
        // top-k: the index source selects for all lanes in one chain and the
        // layers after it reuse that selection (LUCE_DS4_LANE_BATCH_SELECT=0
        // keeps one chain per lane and concatenates their selections).
        static const bool lane_batch_select = [] {
            const char * value = std::getenv("LUCE_DS4_LANE_BATCH_SELECT");
            return !value || std::strcmp(value, "0") != 0;
        }();
        const bool external_selection = lane_batch && lane_batch_select &&
            lane_batch_padded > w.n_indexer_top_k;
        std::vector<DeepSeek4LaneDeferral> deferrals(lane_batch ? (size_t) n_lanes : 0u);
        for (DeepSeek4LaneDeferral & deferral : deferrals) {
            deferral.external_selection = external_selection;
        }
        const int64_t lane_batch_mask_off = mask_off;
        // Lane batch over a ratio-1 kv source: one width-q compressor pass
        // writes every lane's compressed row and index key ahead of the lanes,
        // which then only read them (the batched attention masks each lane's
        // later rows, as it does for the per-lane writes).
        ggml_tensor * batched_comp_source = nullptr;
        ggml_tensor * batched_index_source = nullptr;
        // Lane batch over a stateful kv source (ratio > 1): the compressor's kv
        // and gate projections are per-token products, so they are evaluated
        // at width q and each lane takes its column (prepared_kv/score); the
        // state writes and pooling still run per lane, in lane order. F16
        // MMVF picks its block size from K alone at these shapes, so every
        // column is the product the lane would have computed.
        if (lane_batch && ratio > 1 && ds4_lane_batch_comp_enabled() && attn_lane_q == 1 &&
            L.attn_compressor_kv && L.attn_compressor_gate &&
            L.attn_compressor_kv->type == GGML_TYPE_F16 &&
            L.attn_compressor_gate->type == GGML_TYPE_F16 &&
            deepseek4_is_kv_source(w, il) && !batched_lanes.compressor_kv) {
            batched_lanes.compressor_kv = ggml_mul_mat(ctx, L.attn_compressor_kv, normed_all);
            batched_lanes.compressor_score = ggml_mul_mat(ctx, L.attn_compressor_gate, normed_all);
        }
        if (lane_batch && ratio == 1 && ex.pos_rows && attn_lane_q == 1 &&
            L.attn_compressor_kv && deepseek4_is_kv_source(w, il)) {
            DeepSeek4MlaLaneBindings b0 = deepseek4_contiguous_lane_bindings(
                w, il, *lc, *comp_lc, kv_start + q - 1);
            const DeepSeek4Layer * latent_index =
                (L.indexer_k && b0.index_comp_kv) ? &L : nullptr;
            std::vector<DeepSeek4I32ArrayBinding> bi32;
            std::vector<DeepSeek4I64ArrayBinding> bi64;
            ggml_tensor * last_row = ggml_view_2d(ctx, normed_all, n_embd, 1, normed_all->nb[1],
                                                  (size_t) (q - 1) * normed_all->nb[1]);
            build_compressor_step(ctx, gf, last_row,
                                  L.attn_compressor_ape, L.attn_compressor_kv,
                                  L.attn_compressor_gate, L.attn_compressor_norm,
                                  *b0.attn_compressor, b0.comp_kv, ratio, w.head_dim,
                                  kv_start + q - 1, w.n_rot, w.rms_eps,
                                  w.compress_rope_freq_base, w.rope_scale_factor,
                                  w.rope_yarn_beta_fast, w.rope_yarn_beta_slow,
                                  (int) w.rope_orig_ctx,
                                  nullptr, nullptr, ex.pos_rows, ex.pos_q,
                                  bi64, bi32, &batched_comp_source, nullptr,
                                  normed_all, q, kv_start, false, nullptr, false,
                                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                  latent_index, b0.index_comp_kv, &batched_index_source);
            if (!bi32.empty() || !bi64.empty()) {
                std::fprintf(stderr,
                    "[ds4-fused-verify] layer %d: lane batch compressor took dynamic inputs\n", il);
                return false;
            }
            if (batched_comp_source == b0.comp_kv ||
                (latent_index && (!batched_index_source || batched_index_source == b0.index_comp_kv))) {
                batched_comp_source = nullptr;   // not written: the lanes run it
                batched_index_source = nullptr;
            }
        }
        for (int lane = 0; lane < n_lanes; ++lane) {
        const int aq = attn_lane_q;
        const int a_start = lane * aq;
        const int a_kv_start = kv_start + a_start;
        const int a_token_pos = a_kv_start + aq - 1;
        const auto lane_rows = [&](ggml_tensor * t) {
            return n_lanes == 1 ? t : ggml_view_1d(ctx, t, aq, (size_t) a_start * t->nb[0]);
        };
        const size_t comp_slot = ((size_t) il * n_lanes + (size_t) lane) * 2;
        DeepSeek4AttentionGraphInputs ain{};
        ain.rope_pos = lane_rows(ex.pos_q);
        ain.neg_pos = lane_rows(ex.neg_q);
        ain.raw_kv_rows = lane_rows(ex.rawrows);
        ain.preserved_raw_rows = lane_rows(ex.saved_rawrows);
        if (ratio > 0) {
            int n_flush = 0;
            for (int t = 0; t < aq; ++t) {
                if (((a_kv_start + t + 1) % ratio) == 0) ++n_flush;
            }
            const int n_comp_inputs = std::max(1, n_flush);
            ain.attn_ape_row = lane_rows(ex.rows_for(ratio)->ape);
            ain.attn_state_rows = lane_rows(ex.rows_for(ratio)->state);
            ain.attn_comp_pos = ggml_view_1d(
                ctx, fg.i32_bundle, n_comp_inputs, comp_slot * sizeof(int32_t));
            ain.attn_comp_rows = ggml_view_1d(
                ctx, fg.i64_bundle, n_comp_inputs, comp_slot * sizeof(int64_t));
            if (ratio == 4) {
                ain.index_ape_row = ain.attn_ape_row;
                ain.index_state_rows = ain.attn_state_rows;
                ain.index_comp_pos = ain.attn_comp_pos;
                ain.index_comp_rows = ain.attn_comp_rows;
            }
        }
        const int padded = ds4_fused_lane_padded(w, *comp_lc, il, a_token_pos);
        const int64_t n_attn = (int64_t) w.n_swa + padded + (aq > 1 ? aq : 0);
        ain.attn_row_mask = ggml_view_2d(ctx, fg.mask_bundle, n_attn, aq,
                                         n_attn * sizeof(float), (size_t) mask_off * sizeof(float));
        ain.padded_comp = padded;
        ain.identity_rows = fg.identity_rows;
        if (fg.mask_bundle_f16) {
            ain.attn_row_mask_f16 = ggml_view_2d(ctx, fg.mask_bundle_f16, n_attn, aq,
                                                 n_attn * sizeof(ggml_fp16_t),
                                                 (size_t) mask_off * sizeof(ggml_fp16_t));
        }
        mask_off += n_attn * aq;

        std::vector<DeepSeek4I32InputBinding> i32b;
        std::vector<DeepSeek4I32ArrayBinding> i32ab;
        std::vector<DeepSeek4I64ArrayBinding> i64ab;
        std::vector<DeepSeek4F32ArrayBinding> f32ab;
        // Sparse attention pays a fixed indexer/compaction cost. Keep the
        // established explicit kernel while history is small, then switch per
        // layer once sparse work can remove more than half of the compressed
        // rows. This also avoids a transient identity binding at the exact
        // top-k boundary, preserving fused-graph replay.
        const bool sparse_attention =
            ds4_env_flag("LUCE_DS4_SPARSE_DECODE_FLASH") && ratio > 0 &&
            // The exact indexer carries saved raw rows for batched lanes.
            // Coarse block selection has no saved-row contract; retain the
            // explicit verifier there instead of pruning raw history.
            (aq == 1 || ratio == 4) &&
            padded > 2 * w.n_indexer_top_k;
        const DeepSeek4AttentionImpl attention_impl = sparse_attention
            ? DeepSeek4AttentionImpl::SparseFlash
            : DeepSeek4AttentionImpl::Explicit;
        ggml_tensor * normed = n_lanes == 1 ? normed_all
            : ggml_view_2d(ctx, normed_all, n_embd, aq, normed_all->nb[1],
                           (size_t) a_start * normed_all->nb[1]);
        ggml_tensor * lane_out = nullptr;
        if (n_lanes == 1) {
            lane_out = build_mla_attention(ctx, gf, normed, w, L, *lc, *comp_lc, il,
                                           a_kv_start, aq, &ain, i32b, i32ab, i64ab,
                                           &f32ab, attention_impl,
                                           &ex.boundary_checkpoint.layers[(size_t) il], {},
                                           &lane_selection[(size_t) lane],
                                           attn_head_begin, attn_head_count);
        } else {
            const DeepSeek4PreparedProjectedLane lane_proj =
                ds4_slice_projected_lane(ctx, batched_lanes, w, a_start, ain.rope_pos);
            DeepSeek4MlaLaneBindings bindings = deepseek4_contiguous_lane_bindings(
                w, il, *lc, *comp_lc, a_token_pos);
            bindings.index_selection = &lane_selection[(size_t) lane];
            bindings.head_begin = attn_head_begin;
            bindings.head_count = attn_head_count;
            if (batched_comp_source) {
                bindings.write_comp = false;
                bindings.comp_kv = batched_comp_source;
                if (batched_index_source) bindings.index_comp_kv = batched_index_source;
            }
            if (!build_mla_attention_lane_core(
                    ctx, gf, normed, w, L, bindings, il, a_kv_start, aq, &ain,
                    i32b, i32ab, i64ab, &f32ab, attention_impl, &lane_proj, &lane_out,
                    nullptr, {}, lane_batch ? &deferrals[(size_t) lane] : nullptr)) {
                lane_out = nullptr;
            }
        }
        if (lane_batch ? !deferrals[(size_t) lane].deferred : !lane_out) return false;
        if (!i32b.empty() || !i32ab.empty() || !i64ab.empty() ||
            !f32ab.empty()) {
            std::fprintf(stderr,
                "[ds4-fused-verify] layer %d dynamic bindings "
                "(%zu/%zu/%zu/%zu); cannot fuse\n",
                il, i32b.size(), i32ab.size(), i64ab.size(), f32ab.size());
            return false;
        }
        if (window_idx >= 0) {
            // The window row this token wrote, before the next lane's write.
            const DeepSeek4CompressorState & st = lc->attn_compressor;
            const size_t row_off = (size_t) ds4_compressor_state_row(ratio, a_kv_start) * st.state_kv->nb[1];
            window_kv_rows.push_back(ggml_cont(ctx, ggml_view_2d(
                ctx, st.state_kv, st.state_kv->ne[0], 1, st.state_kv->nb[1], row_off)));
            window_score_rows.push_back(ggml_cont(ctx, ggml_view_2d(
                ctx, st.state_score, st.state_score->ne[0], 1, st.state_score->nb[1], row_off)));
            // Outputs: the rollback copies them back on the device.
            ggml_set_output(window_kv_rows.back());
            ggml_set_output(window_score_rows.back());
            ggml_build_forward_expand(gf, window_kv_rows.back());
            ggml_build_forward_expand(gf, window_score_rows.back());
        }
        if (n_lanes == 1) {
            attn_out = lane_out;
        } else if (!lane_batch) {
            ggml_build_forward_expand(gf, lane_out);
            lane_contexts = lane_contexts ? ggml_concat(ctx, lane_contexts, lane_out, 1) : lane_out;
        }
        }
        if (lane_batch) {
            const DeepSeek4LaneDeferral & last = deferrals.back();
            const int64_t n_attn = (int64_t) w.n_swa + last.n_comp_attn;
            GGML_ASSERT(lane_batch_mask_off + n_attn * n_lanes == mask_off);
            ggml_tensor * selection = nullptr;
            if (external_selection) {
                ggml_tensor * visibility = deepseek4_is_index_source(w, il)
                    ? ggml_cont(ctx, ggml_view_2d(
                          ctx, fg.mask_bundle, lane_batch_padded, n_lanes,
                          n_attn * sizeof(float),
                          (size_t) (lane_batch_mask_off + w.n_swa) * sizeof(float)))
                    : nullptr;
                selection = ds4_lane_batch_shared_selection(
                    ctx, w, L, il, batched_lanes.normalized_q_lora, normed_all,
                    last.index_comp_source, lane_batch_padded, kv_start, ex.pos_q,
                    visibility, &lane_batch_carrier);
                if (!selection) {
                    std::fprintf(stderr,
                        "[ds4-fused-verify] layer %d: lane batch selection failed\n", il);
                    return false;
                }
                // The per-lane carriers predate this selection.
                if (deepseek4_is_index_source(w, il)) {
                    std::fill(lane_selection.begin(), lane_selection.end(), nullptr);
                }
            } else {
                selection = ds4_lane_batch_selection(
                    ctx, deferrals, fg.identity_rows, lane_batch_selections);
            }
            ggml_tensor * mask_f16 = fg.mask_bundle_f16
                ? ggml_view_2d(ctx, fg.mask_bundle_f16, n_attn, n_lanes,
                               n_attn * sizeof(ggml_fp16_t),
                               (size_t) lane_batch_mask_off * sizeof(ggml_fp16_t))
                : ggml_cast(ctx, ggml_view_2d(ctx, fg.mask_bundle, n_attn, n_lanes,
                                              n_attn * sizeof(float),
                                              (size_t) lane_batch_mask_off * sizeof(float)),
                            GGML_TYPE_F16);
            lane_contexts = build_mla_lane_batch_attention(
                ctx, gf, w, L, lc->raw_kv, batched_lanes.q, batched_lanes.kv,
                ex.saved_rawrows, ex.rawrows, mask_f16, selection,
                last.comp_source, last.n_comp_attn, attn_head_begin, attn_head_count);
            if (!lane_contexts) return false;
        }
        if (!external_selection && w.shared_index_topk && ratio > 0 &&
            deepseek4_is_index_source(w, il)) {
            lane_batch_carrier = nullptr;  // the lanes selected one by one
        }
        if (n_lanes > 1) {
            const Ds4RopeParams rope = ds4_rope_params(w, ratio);
            attn_out = build_tail_rope_3d(
                ctx, ggml_reshape_3d(ctx, lane_contexts, w.head_dim, attn_n_head, q), ex.neg_q,
                w.n_rot, w.head_dim, attn_n_head, q, rope.freq, rope.scale, rope.ext, rope.attn,
                w.rope_yarn_beta_fast, w.rope_yarn_beta_slow, rope.n_ctx_orig);
            // The single-token decode graph never takes the grouped output kernel.
            attn_out = build_mla_output_projection(
                ctx, ggml_reshape_2d(ctx, attn_out, (int64_t) w.head_dim * attn_n_head, q),
                w, L, q, /*allow_grouped=*/false, 0, attn_head_begin, attn_head_count);
        }
        }

        // Head split: the attention output is a partial sum over this rank's
        // output groups; one in-graph all-reduce makes it whole.
        if (cluster_hybrid && attn_head_count > 0) {
            // The partial is a fresh product only this node reads: sum it in
            // place, without a device copy (a contiguous one skips the cont).
            attn_out = ds4_cluster_allreduce_node(
                ctx, ggml_is_contiguous(attn_out) ? attn_out : ggml_cont(ctx, attn_out),
                *cluster_rt, /*inplace=*/true);
            if (!attn_out) return false;
            cluster_nodes.push_back(attn_out);
        }

        // ── Batched HC post (attention) + HC pre (FFN) ──
        ggml_tensor * attn_batch = ggml_is_contiguous(attn_out)
            ? attn_out : ggml_cont(ctx, attn_out);
        hc_cur = ggml_ds4_hc_post(
            ctx, hc_flat, attn_batch, split_attn, n_hc);
        hc_flat = hc_cur;
        ggml_tensor * split_ffn = nullptr;
        ggml_tensor * ffn_working = ds4_build_fused_hc_pre(
            ctx, w, hc_flat, mc.fn_ffn_f16[(size_t) il], L.hc_ffn_base,
            hlw.ffn, &split_ffn, projection_columns);
        if (!ffn_working) return false;
        ggml_tensor * ffn_in = w.hc_staggered_pre
            ? ds4_build_hc_collapse(ctx, hc_flat, carried_pre, n_embd, n_hc)
            : ggml_cont(ctx, ffn_working);
        if (w.hc_staggered_pre) carried_pre = ds4_hc_split_pre(ctx, split_ffn, n_hc);

        // ── Batched FFN ──
        ggml_tensor * ffn_normed = build_rms_norm(ctx, ffn_in, L.ffn_norm, w.rms_eps);
        ggml_tensor * ffn_out = nullptr;
        MoeHybridGraphInputs * hyb_ex_inputs = nullptr;
        const MoeHybridGraphInputs * fused_hc_join_inputs = nullptr;
        const bool hash_routed = il < w.n_hash_layer && L.ffn_gate_tid2eid &&
                                 have_token_ids && hash_tables[(size_t) il].loaded;
        ggml_tensor * selected = nullptr;
        ggml_tensor * router_weights = nullptr;
        std::vector<ggml_tensor *> router_nodes;
        if (hash_routed) {
            const int route_width = ds4_effective_expert_count(w);
            // Keep one q-wide input for host filling.
            ggml_tensor * hids_all = fg.hash_ids[(size_t) il];
            if (!hids_all) {
                hids_all = ggml_new_tensor_2d(
                    ctx, GGML_TYPE_I32, route_width, q);
                ggml_set_input(hids_all);
                fg.hash_ids[(size_t) il] = hids_all;
            }
            selected = hids_all;
            if (hybrid) {
                router_weights = ds4_fused_weights_for_selected(
                    ctx, ffn_normed, w, L, selected, lane_q);
            } else {
                for (int t = 0; t < lane_q; ++t) {
                    ggml_tensor * fcol = ggml_view_2d(ctx, ffn_normed, n_embd, 1,
                                                      ffn_normed->nb[1], (size_t) t * ffn_normed->nb[1]);
                    ggml_tensor * hcol = ggml_view_2d(
                        ctx, selected, route_width, 1,
                        selected->nb[1], (size_t) t * selected->nb[1]);
                    ggml_tensor * fo = ds4_build_hash_routed_ffn(
                        ctx, w, L, ggml_cont(ctx, fcol), hcol, 1);
                    if (!fo) { ffn_out = nullptr; break; }
                    ffn_out = ffn_out ? ggml_concat(ctx, ffn_out, fo, 1) : fo;
                }
            }
        } else if (hybrid) {
            Ds4MoeRouting routing =
                build_moe_routing(ctx, ffn_normed, w, L, lane_q);
            selected = routing.selected;
            router_weights = routing.weights;
            router_nodes = routing.nodes;
            // Predict the next layer's routes from this FFN input and post them
            // ahead of this layer's own, so the streamed-expert resolver
            // starts loading the next layer's misses one layer early.
            MoeStreamedMailbox * mailbox =
                hybrid->expert_cache ? hybrid->expert_cache->mailbox() : nullptr;
            if (mailbox && il + 1 < w.n_layer && il + 1 >= w.n_hash_layer &&
                hybrid->layers[(size_t) il + 1].n_streamed > 0) {
                const DeepSeek4Layer & N = w.layers[(size_t) il + 1];
                const MoeStreamedMailbox::Channel * ch = mailbox->channel(il + 1);
                Ds4MoeRouting next = build_moe_routing(
                    ctx, build_rms_norm(ctx, ffn_in, N.ffn_norm, w.rms_eps), w, N, lane_q);
                ggml_tensor * post = ggml_host_mailbox_post(
                    ctx, ggml_cont(ctx, next.selected), ch->step, ch->predicted,
                    ch->predicted_ids);
                fg.stream_predict_posts.push_back(post);
                ggml_build_forward_expand(gf, post);
            }
        } else {
            const int columns = moe_columns > 0 ? moe_columns : lane_q;
            for (int first = 0; first < lane_q; first += columns) {
                const int count = std::min(columns, lane_q - first);
                ggml_tensor * part = moe_columns > 0
                    ? ggml_cont(ctx, ggml_view_2d(ctx, ffn_normed, n_embd, count,
                        ffn_normed->nb[1], (size_t)first * ffn_normed->nb[1]))
                    : ffn_normed;
                ggml_tensor * one = build_moe_ffn(ctx, part, w, L, il, count);
                ffn_out = ffn_out ? ggml_concat(ctx, ffn_out, one, 1) : one;
            }
        }

        if (hybrid) {
            DeepSeek4FusedDecodeGraph::AuthoritativeRouteOutput route;
            route.layer = il;
            route.lane_start = lane_start;
            route.n_tokens = lane_q;
            route.width = (int) selected->ne[0];
            route.selected = selected;
            route.weights = router_weights;
            fg.authoritative_routes.push_back(route);
        }

        if (hybrid) {
            const Ds4MixedMoePolicy & mixed_policy =
                ds4_mixed_moe_policy();
            const bool cross_vendor_owner_sums =
                !same_gpu_runtime &&
                mixed_policy.owner_local_reduction;
            const bool allow_fused_combine =
                same_gpu_runtime || cross_vendor_owner_sums;
            const bool allow_direct_device_join =
                pair_capabilities.native_gpu_handoff &&
                mixed_policy.direct_device_join;
            const bool schedule_host_staged_branches =
                !same_gpu_runtime &&
                mixed_policy.schedule_branches;
            ggml_cgraph * hybrid_schedule_graph =
                (allow_direct_device_join || schedule_host_staged_branches)
                    ? gf : nullptr;
            // Cross-vendor staging normally preserves canonical route order
            // so it matches the single-owner reduction as closely as
            // possible.  For performance qualification, allow each owner to
            // reduce its routed experts locally and stage one [n_embd, q]
            // partial instead of the full [n_embd, n_used, q] route tensor.
            // This is the same join shape used by the qualified same-runtime
            // heterogeneous path, but remains opt-in until exact-output and
            // mixed-vendor burn-in have passed.
            const MoeHybridJoinMode join_mode =
                same_gpu_runtime || cross_vendor_owner_sums
                    ? MoeHybridJoinMode::OwnerPartialSums
                    : MoeHybridJoinMode::CanonicalRouteOrder;
            MoeHybridConfig hybrid_cfg = make_ds4_moe_hybrid_config(w);
            hybrid_cfg.n_expert_used = (int) selected->ne[0];
            MoeLayerDesc desc = make_ds4_moe_layer_desc(L);
            MoeHybridGraphInputs & inputs = fg.hybrid_inputs[hybrid_idx];
            // Experts neither owner holds come from the streamed-expert
            // cache, whose slots must live beside the secondary owner.
            MoeStreamedOwner streamed_owner;
            const MoeStreamedOwner * streamed = nullptr;
            if (hybrid->layers[(size_t) il].n_streamed > 0) {
                MoeStreamedExpertCache * cache = hybrid->expert_cache;
                if (!cache || !cache->ready() || !cache->mailbox() ||
                    cache->device() != ggml_backend_cuda_get_device_id(hybrid->cold_backend)) {
                    return false;
                }
                const MoeStreamedMailbox * mailbox = cache->mailbox();
                if (!mailbox) return false;
                const MoeStreamedExpertCache::Stacks stacks = cache->stacks(il);
                streamed_owner = {stacks.gate, stacks.up, stacks.down, stacks.gate_up,
                                  mailbox->channel(il)};
                streamed = &streamed_owner;
            }
            inputs.router_nodes = router_nodes;
            const MoeHybridLayerStorage & hybrid_layer =
                hybrid->layers[(size_t) il];
            const bool has_hot_routed =
                (hybrid_layer.gate_up_hot ||
                 (hybrid_layer.gate_hot && hybrid_layer.up_hot)) &&
                hybrid_layer.down_hot;
            // Legacy q>1 graphs represent six routes as 4 + 2. Native q1
            // decode already uses width six directly. Keep the decomposition
            // as a bisectable fallback while validating the dedicated q4 MoE
            // kernel's native route width on both GPUs.
            // The dedicated multi-token MUL_MAT_ID kernel is indexed by token
            // width, not route width.  Once that kernel is enabled, keep all
            // six routes in one owner graph. Until then, retain the existing
            // head-plus-tail reduction order without padding the two-route
            // tail with duplicate zero-weight expert paths.
            // A streamed layer posts its routes once, so it keeps them whole.
            // Tokenwise verify lanes keep them whole as well: the q=1 graph
            // sums all six routes in one combine, and the 4 + 2 decomposition
            // sums them in another order, so a batched verify built with it
            // differs from decode by rounding (H200, DS4.1-Flash: the first
            // differing cache bytes appear at layer 1 of the first verify;
            // with whole routes the q=3 verify and q=1 decode are bit-identical).
            // LUCE_DS4_TP_HEAD4_TAIL2=1 restores the decomposition for bisection.
            const bool native_route_width =
                cluster_hybrid || !same_gpu_runtime ||
                mixed_policy.native_route_width || streamed ||
                whole_routes_for_lanes;
            if (lane_q > 1 && hybrid_cfg.n_expert_used > 4 && has_hot_routed &&
                !native_route_width) {
                const int route_width = hybrid_cfg.n_expert_used;
                if (route_width != 6) return false;
                hybrid_cfg.n_expert_used = 4;

                DeepSeek4Head4Tail2Routes routes;
                if (!build_deepseek4_head4_tail2_routes(
                        ctx, selected, router_weights, lane_q, routes)) {
                    return false;
                }
                if (!build_moe_hybrid_ffn_graph(
                        ctx,
                        hybrid_schedule_graph,
                        hybrid_cfg, desc,
                        hybrid->layers[(size_t) il], ffn_normed,
                        routes.head_ids, routes.head_weights,
                        lane_q, inputs, true,
                        allow_fused_combine, join_mode,
                        MoeHybridRouteBalance::Disabled, streamed)) {
                    return false;
                }
                ffn_out = inputs.output;

                hybrid_cfg.n_expert_used = 2;
                if (!build_moe_hybrid_ffn_graph(
                        ctx,
                        hybrid_schedule_graph,
                        hybrid_cfg, desc,
                        hybrid->layers[(size_t) il], ffn_normed,
                        routes.tail_ids, routes.tail_weights,
                        lane_q, inputs, false,
                        allow_fused_combine, join_mode,
                        MoeHybridRouteBalance::Disabled, streamed)) {
                    return false;
                }
                ffn_out = ggml_add(ctx, ffn_out, inputs.output);
            } else {
                // Cluster hybrid exchange: keep the owners' partials apart;
                // the reduction below adds them on the secondary owner.
                inputs.external_join = cluster_hybrid && cluster_rt &&
                    ds4_cluster_hybrid_exchange_enabled(*cluster_rt) &&
                    L.ffn_gate_shexp && L.ffn_up_shexp && L.ffn_down_shexp &&
                    cluster_rt->shexp_ff_count > 0 && !streamed;
                if (!build_moe_hybrid_ffn_graph(
                        ctx,
                        hybrid_schedule_graph,
                        hybrid_cfg, desc,
                        hybrid->layers[(size_t) il],
                        ffn_normed, selected, router_weights,
                        lane_q, inputs, /*include_shared=*/!cluster_hybrid,
                        allow_fused_combine,
                        join_mode,
                        (streamed || cluster_hybrid) ? MoeHybridRouteBalance::Disabled
                                                     : MoeHybridRouteBalance::Allowed,
                        streamed)) {
                    return false;
                }
                ffn_out = inputs.output;
                if (inputs.external_join && inputs.peer_raw && inputs.main_output) {
                    hyb_ex_inputs = &inputs;
                }
                // The staggered pre-mix model joins like single-token decode
                // (verify lanes and gathered sequences match it exactly).
                if (lane_q > 1 && !w.hc_staggered_pre && native_route_width &&
                    !cluster_hybrid && mixed_policy.fused_hc_join &&
                    inputs.main_output && inputs.peer_output) {
                    fused_hc_join_inputs = &inputs;
                }
            }
        }
        if (!ffn_out) {
            std::fprintf(stderr,
                "[deepseek4-paged] layer %d FFN graph build failed\n", il);
            return false;
        }

        // Cluster: sum the ranks' routed partials as a graph node,
        // then add this rank's copy of the replicated shared expert. Doing it
        // in this order is what keeps the result identical to a single node:
        // the shared expert is added exactly once, and the reduction sees
        // only routed contributions.
        if (cluster_hybrid) {
            const bool have_shexp =
                L.ffn_gate_shexp && L.ffn_up_shexp && L.ffn_down_shexp;
            const int shexp_ff_count = cluster_rt->shexp_ff_count;
            if (have_shexp && shexp_ff_count > 0) {
                // Sharded: this rank's slice is a partial sum over the down
                // projection's contraction axis, so it is added to the routed
                // partial and the two ride in ONE reduction.
                ggml_tensor * shared = build_shared_ffn_slice(
                    ctx, ffn_normed, w, L,
                    cluster_rt->shexp_ff_begin, shexp_ff_count);
                if (!shared) return false;
                // Queue the shared slice ahead of the owner join: it needs only
                // the FFN input, so its kernels run while the secondary owner
                // is still computing instead of after the join's wait.
                ggml_build_forward_expand(gf, shared);
                if (hyb_ex_inputs) {
                    // dGPU exports hot + shared, the iGPU adds its cold
                    // partial in the same order and exchanges, the dGPU
                    // imports the sum (no device join, one link crossing
                    // each way on the dGPU).
                    ggml_tensor * combine = nullptr;
                    ggml_tensor * export_hot = nullptr;
                    ggml_tensor * export_shared = nullptr;
                    ggml_tensor * sum = ds4_cluster_hybrid_exchange_nodes(
                        ctx, hyb_ex_inputs->main_output, shared, hyb_ex_inputs->peer_raw,
                        *cluster_rt, &combine, &export_hot, &export_shared);
                    if (!sum) return false;
                    ggml_build_forward_expand(gf, export_hot);
                    ggml_build_forward_expand(gf, export_shared);
                    cluster_nodes.push_back(export_shared);
                    ggml_set_output(combine);
                    ggml_build_forward_expand(gf, combine);
                    hyb_ex_inputs->cold_nodes.push_back(combine);
                    cluster_nodes.push_back(export_hot);
                    // An upcoming Engram layer's projection chunk, in the wait
                    // for the iGPU's sum (ahead of the import in the stream).
                    int engram_e = -1;
                    int64_t engram_r0 = 0;
                    if (engram_chunk_for(il, engram_e, engram_r0)) {
                        const DeepSeek4Layer & Le = w.layers[(size_t) engram->layer_id(engram_e)];
                        const int64_t rows = std::min<int64_t>(engram_chunk_rows, Le.engram_wkv->ne[1] - engram_r0);
                        ggml_tensor * keys_e = ggml_view_2d(ctx, engram_keys_all, engram_keys_all->ne[0], q,
                                                            engram_keys_all->nb[1],
                                                            (size_t) engram_e * engram_keys_all->nb[2]);
                        ggml_tensor * part = ggml_mul_mat(ctx,
                            ggml_view_2d(ctx, Le.engram_wkv, Le.engram_wkv->ne[0], rows,
                                         Le.engram_wkv->nb[1], (size_t) engram_r0 * Le.engram_wkv->nb[1]),
                            keys_e);
                        ggml_mul_mat_set_prec(part, GGML_PREC_F32);
                        ggml_build_forward_expand(gf, part);
                        cluster_nodes.push_back(part);
                        engram_kv_parts[(size_t) engram_e].push_back(part);
                        engram_kv_done[(size_t) engram_e] = engram_r0 + rows;
                    }
                    cluster_nodes.push_back(sum);
                    ffn_out = sum;
                } else {
                    ffn_out = ggml_add(ctx, ffn_out, shared);
                    ffn_out = ds4_cluster_allreduce_node(ctx, ffn_out, *cluster_rt, /*inplace=*/true);
                    if (!ffn_out) return false;
                    cluster_nodes.push_back(ffn_out);
                }
            } else {
                ffn_out = ds4_cluster_allreduce_node(ctx, ggml_cont(ctx, ffn_out), *cluster_rt, /*inplace=*/true);
                if (!ffn_out) return false;
                cluster_nodes.push_back(ffn_out);
                if (have_shexp) {
                    ggml_tensor * shared = build_shared_ffn(ctx, ffn_normed, w, L);
                    if (!shared) return false;
                    ffn_out = ggml_add(ctx, ffn_out, shared);
                }
            }
        }

        // ── Batched HC post (FFN); capture at drafter layers ──
        if (fused_hc_join_inputs) {
            hc_cur = ggml_ds4_hc_post_split(
                ctx, hc_flat, fused_hc_join_inputs->main_output,
                fused_hc_join_inputs->peer_output, split_ffn, n_hc);
        } else {
            ggml_tensor * ffn_batch = ggml_is_contiguous(ffn_out)
                ? ffn_out : ggml_cont(ctx, ffn_out);
            hc_cur = ggml_ds4_hc_post(
                ctx, hc_flat, ffn_batch, split_ffn, n_hc);
        }
        for (size_t ci = 0; ci < capture_ids.size(); ++ci) {
            if (capture_ids[ci] != il) continue;
            ggml_tensor * hs = ggml_reshape_3d(
                ctx, hc_cur, n_embd, n_hc, lane_q);
            ggml_tensor * hs_t = ggml_cont(
                ctx, ggml_transpose(ctx, hs));                    // [n_hc,n_embd,q]
            ggml_tensor * summed = ggml_sum_rows(ctx, hs_t);      // [1,n_embd,q]
            ggml_tensor * mean = ggml_scale(
                ctx, summed, 1.0f / (float) n_hc);
            capture_layers[ci] = ggml_reshape_2d(
                ctx, mean, n_embd, lane_q);                       // [n_embd,q]
        }
    }
    GGML_ASSERT(mask_off == mask_total);

    // ── Batched output HC merge → out_norm + lm_head ──
    ggml_tensor * final_all = nullptr;
    if (w.hc_staggered_pre) {
        final_all = ds4_build_hc_collapse(ctx, hc_cur, carried_pre, n_embd, n_hc);
    } else {
        ggml_tensor * onorm = ggml_rms_norm(ctx, hc_cur, w.hc_eps);
        ggml_tensor * omix = ds4_mul_mat_columns(ctx, mc.fn_out_f16, onorm, projection_columns);
        omix = ggml_reshape_2d(ctx, omix, ggml_nelements(omix) / q, q);
        ggml_tensor * obase = ds4_fused_hc_base_f32(ctx, w.output_hc_base);
        if (!obase || hc_out_weights.scale_data.empty()) return false;
        final_all = ggml_ds4_hc_out(
            ctx, omix, obase, hc_cur, n_hc, hc_out_weights.scale_data[0]);
    }
    ggml_tensor * out_normed = build_rms_norm(ctx, final_all, w.out_norm, w.rms_eps);
    if (ex.vocab_split && cluster_rt && cluster_rt->size() == 2) {
        // LUCE_CLUSTER_VOCAB_SPLIT: each rank projects its half of the
        // vocabulary (the Q8_0 rows are the same products as in the full
        // head) and the ranks exchange their column maxima as one slot each.
        const int rank = cluster_rt->rank();
        const int64_t V = w.output->ne[1];
        const int64_t half = ((V / 2) + 255) / 256 * 256;
        const int64_t r0 = rank == 0 ? 0 : half;
        const int64_t nr = rank == 0 ? half : V - half;
        ggml_tensor * out_rows = ggml_view_2d(ctx, w.output, w.output->ne[0], nr, w.output->nb[1],
                                              (size_t) r0 * w.output->nb[1]);
        fg.logits = ds4_mul_mat_columns(ctx, out_rows, out_normed, projection_columns);   // [nr, q]
        ggml_tensor * pair = ggml_ds4_argmax_pair(ctx, fg.logits, (int) r0, rank, 2);
        pair = ds4_cluster_allreduce_node(ctx, pair, *cluster_rt, /*inplace=*/true);
        if (!pair) return false;
        cluster_nodes.push_back(pair);
        ex.argmax = ggml_ds4_argmax_pick(ctx, pair, 2);
        ggml_set_output(ex.argmax);
        ggml_build_forward_expand(gf, ex.argmax);
    } else {
    fg.logits = ds4_mul_mat_columns(ctx, w.output, out_normed, projection_columns);   // [n_vocab, q]
    ggml_set_output(fg.logits);
    ggml_build_forward_expand(gf, fg.logits);
    }
    if (!ex.vocab_split && (paged_mode || ds4_env_flag("LUCE_DS4_GPU_ARGMAX_VERIFY"))) {
        ex.argmax = ggml_argmax(ctx, fg.logits);
        ggml_set_output(ex.argmax);
        ggml_build_forward_expand(gf, ex.argmax);
    }

    if (!capture_layers.empty()) {
        for (ggml_tensor * layer : capture_layers) {
            if (!layer) {
                std::fprintf(stderr, "[ds4-fused-verify] capture layer id is invalid\n");
                return false;
            }
        }
        // Concatenating layer matrices on dim 0 produces the token-major
        // [n_capture*n_embd,q] layout consumed by the drafter directly.
        ggml_tensor * cap = capture_layers[0];
        for (size_t i = 1; i < capture_layers.size(); ++i) {
            cap = ggml_concat(ctx, cap, capture_layers[i], 0);
        }
        ex.capture = cap;
        ggml_set_output(ex.capture);
        ggml_build_forward_expand(gf, ex.capture);
    }
    // The window rows stay where the lanes wrote them (no concat, no
    // readback): a rejected suffix copies the few it needs back on the device.
    ex.window_kv_rows = window_kv_rows;
    ex.window_score_rows = window_score_rows;
    ex.boundary_checkpoint.available = std::any_of(
        ex.boundary_checkpoint.layers.begin(),
        ex.boundary_checkpoint.layers.end(),
        [](const DeepSeek4SpecBoundaryCheckpointLayer & layer) {
            return layer.attn_kv_src || layer.index_kv_src;
        });
    const auto topology_t1 = Ds4TimingClock::now();

    // One-shot graph inventory for profiling residual single-token FP4 work.
    // This is deliberately build-time only: it does not install a scheduler
    // callback or synchronize kernels, so enabling it cannot distort the
    // execution timeline being diagnosed.
    if (q == 4 && ds4_env_flag("LUCE_DS4_TP_Q4_INVENTORY")) {
        const int n_nodes = ggml_graph_n_nodes(fg.sg.gf);
        std::fprintf(stderr,
            "[ds4-q4-inventory] graph_nodes=%d q=%d\n", n_nodes, q);
        for (int ni = 0; ni < n_nodes; ++ni) {
            ggml_tensor * node = ggml_graph_node(fg.sg.gf, ni);
            if (!node ||
                (node->op != GGML_OP_MUL_MAT &&
                 node->op != GGML_OP_MUL_MAT_ID) ||
                !node->src[0] || !node->src[1] ||
                node->src[0]->type != GGML_TYPE_Q4_0_ROCMFP4_FAST) {
                continue;
            }
            const ggml_tensor * weight = node->src[0];
            const ggml_tensor * rhs = node->src[1];
            std::fprintf(stderr,
                "[ds4-q4-inventory] node=%d op=%s weight='%s' "
                "w=[%lld,%lld,%lld] rhs=[%lld,%lld,%lld] out=[%lld,%lld,%lld]\n",
                ni, ggml_op_desc(node), weight->name,
                (long long) weight->ne[0], (long long) weight->ne[1],
                (long long) weight->ne[2], (long long) rhs->ne[0],
                (long long) rhs->ne[1], (long long) rhs->ne[2],
                (long long) node->ne[0], (long long) node->ne[1],
                (long long) node->ne[2]);
        }
    }

    if (hybrid) {
        ggml_backend_t peer = hybrid->cold_backend;
        if (!peer || peer == backend) {
            std::fprintf(stderr,
                         "[ds4-fused-verify] hybrid peer backend unavailable\n");
            return false;
        }
        if (!hybrid->cpu_backend) {
            std::fprintf(stderr,
                         "[ds4-fused-verify] scheduler CPU fallback unavailable\n");
            return false;
        }
        const Ds4MixedMoePolicy & mixed_policy =
            ds4_mixed_moe_policy();
        if (!same_gpu_runtime && mixed_policy.direct_device_join) {
            static bool warned_cross_vendor_join = false;
            if (!warned_cross_vendor_join) {
                std::fprintf(stderr,
                    "[ds4-fused-verify] direct peer join disabled across GPU "
                    "vendors; using scheduler host staging\n");
                warned_cross_vendor_join = true;
            }
        }
        if (reuse_sched) {
            // Drop prior tensor assignments and deferred events while keeping
            // the scheduler-owned pinned staging allocations alive.
            ggml_backend_sched_reset(fg.sched);
        } else {
            ggml_backend_t backends[3] = {
                backend, peer, hybrid->cpu_backend};
            fg.sched = ggml_backend_sched_new(
                backends, nullptr, 3, graph_capacity, false, true);
            if (!fg.sched) {
                std::fprintf(stderr,
                             "[ds4-fused-verify] scheduler creation failed\n");
                return false;
            }
            fg.sched_capacity = graph_capacity;
            fg.sched_backends = sched_backends;
        }
        const bool late_join_split = mixed_policy.late_join_split;
        const MoeHybridGraphPolicy & moe_policy =
            moe_hybrid_graph_policy();
        const bool targeted_join_split =
            moe_policy.targeted_join_split;
        const bool device_join_split = mixed_policy.deferred_join_split;
        const bool batch_split_copies = mixed_policy.batch_peer_copies;
        const bool report_split_count = mixed_policy.report_split_count;
        ggml_backend_sched_set_late_cross_input_split(
            fg.sched, late_join_split);
        ggml_backend_sched_set_deferred_peer_copy_split(
            fg.sched, device_join_split);
        ggml_backend_sched_set_batch_split_copies(
            fg.sched, batch_split_copies);
        // LUCE_DS4_SCHED_DEDICATED_COPIES=1: the verifier's split inputs own
        // their storage (a few MiB), so the host never waits for a device
        // to drain before queueing the next split.
        static const bool dedicated_copies = [] {
            const char * v = std::getenv("LUCE_DS4_SCHED_DEDICATED_COPIES");
            return v && *v && std::strcmp(v, "0") != 0;
        }();
        ggml_backend_sched_set_dedicated_copies(fg.sched, dedicated_copies);
        if (mixed_policy.scheduler_trace) {
            ggml_backend_sched_set_eval_callback(
                fg.sched, ds4_fused_verify_trace_node, nullptr);
        }
        auto pin_main = [&](ggml_tensor * tensor) {
            if (tensor) {
                ggml_backend_sched_set_tensor_backend(fg.sched, tensor, backend);
            }
        };
        auto pin_peer = [&](ggml_tensor * tensor) {
            if (tensor) {
                ggml_backend_sched_set_tensor_backend(fg.sched, tensor, peer);
            }
        };
        for (ggml_tensor * context : attention_contexts) pin_main(context);
        // The communicator lives on the primary GPU (ROCm 7.2.4's RCCL has
        // no gfx1151 kernels): every collective node runs on its stream.
        for (ggml_tensor * node : cluster_nodes) pin_main(node);
        pin_main(fg.inp_embed);
        pin_main(ex.engram_keys);
        pin_main(ex.pos_q);
        pin_main(ex.neg_q);
        pin_main(ex.rawrows);
        pin_main(ex.saved_rawrows);
        pin_main(ex.pos_rows);
        for (const auto & rows : ex.ratio_rows) {
            pin_main(rows.ape);
            pin_main(rows.state);
        }
        pin_main(fg.i32_bundle);
        pin_main(fg.i64_bundle);
        pin_main(fg.mask_bundle);
        if (fg.identity_rows) pin_main(fg.identity_rows);
        if (fg.mask_bundle_f16) pin_main(fg.mask_bundle_f16);
        pin_main(ex.paged_i32);
        pin_main(ex.paged_i64);
        pin_main(ex.paged_gather);
        for (const auto & px : ex.paged) {
            pin_main(px.pos); pin_main(px.neg_pos); pin_main(px.raw_gather); pin_main(px.comp_gather);
            pin_main(px.index_gather);
            pin_main(px.raw_write); pin_main(px.comp_write); pin_main(px.comp_read); pin_main(px.ape);
            pin_main(px.state_row); pin_main(px.comp_pos);
        }
        for (ggml_tensor * hids : fg.hash_ids) pin_main(hids);
        for (ggml_tensor * post : fg.stream_predict_posts) pin_peer(post);
        for (const MoeHybridGraphInputs & inputs : fg.hybrid_inputs) {
            if (mixed_policy.pin_route_weights) {
                for (ggml_tensor * node : inputs.router_nodes) {
                    pin_main(node);
                }
                pin_main(inputs.router_weights);
            }
            if (moe_policy.route_prefork) {
                for (ggml_tensor * node : inputs.route_prefork_nodes) {
                    pin_main(node);
                }
            }
            pin_main(inputs.hot_local_lut);
            pin_main(inputs.hot_valid_lut);
            // The LUT is already q-batched and consumed only by the cold
            // remap, so keep it with that owner for every verifier width.
            pin_peer(inputs.cold_local_lut);
            pin_peer(inputs.cold_valid_lut);
            pin_peer(inputs.stream_local_lut);
            pin_peer(inputs.stream_valid_lut);
            for (ggml_tensor * node : inputs.hot_remap_nodes) {
                pin_main(node);
            }
            for (ggml_tensor * node : inputs.cold_remap_nodes) {
                pin_peer(node);
            }
            for (ggml_tensor * node : inputs.hot_nodes) {
                pin_main(node);
            }
            for (ggml_tensor * node : inputs.cold_nodes) {
                pin_peer(node);
            }
            for (ggml_tensor * node : inputs.join_nodes) {
                pin_main(node);
                if (targeted_join_split) {
                    ggml_backend_sched_add_late_cross_input_split_node(
                        fg.sched, node);
                }
            }
            for (ggml_tensor * node : inputs.deferred_peer_copy_nodes) {
                pin_main(node);
                ggml_backend_sched_add_deferred_peer_copy_node(
                    fg.sched, node);
            }
            pin_main(inputs.main_output);
            pin_main(inputs.peer_output);
            pin_main(inputs.output);
        }
        for (const auto & route : fg.authoritative_routes) {
            pin_main(route.selected);
            pin_main(route.weights);
        }
        if (mixed_policy.preserve_routes_for_diagnostics) {
            // Preserve the unsplit authoritative route and weight matrices.
            // Reading the pre-fork list is insufficient in the legacy 4+2
            // lowering because its second build overwrites the per-layer
            // bookkeeping with padded IDs.
            for (const auto & route : fg.authoritative_routes) {
                ggml_set_output(route.selected);
                ggml_set_output(route.weights);
            }
        }
        pin_main(fg.logits);
        pin_main(ex.argmax);
        pin_main(ex.capture);
        for (const DeepSeek4SpecBoundaryCheckpointLayer & checkpoint :
             ex.boundary_checkpoint.layers) {
            pin_main(checkpoint.attn_kv_src);
            pin_main(checkpoint.attn_kv_dst);
            pin_main(checkpoint.attn_score_src);
            pin_main(checkpoint.attn_score_dst);
            pin_main(checkpoint.index_kv_src);
            pin_main(checkpoint.index_kv_dst);
            pin_main(checkpoint.index_score_src);
            pin_main(checkpoint.index_score_dst);
        }

        if (!ggml_backend_sched_alloc_graph(fg.sched, fg.sg.gf)) {
            std::fprintf(stderr,
                         "[ds4-fused-verify] scheduler graph alloc failed\n");
            return false;
        }
        if (late_join_split || targeted_join_split || report_split_count) {
            const char * split_mode = targeted_join_split
                ? "targeted-join"
                : (late_join_split ? "late-join" : "diagnostic");
            std::fprintf(stderr,
                         "[ds4-fused-verify] %s scheduler splits=%d q=%d\n",
                         split_mode,
                         ggml_backend_sched_get_n_splits(fg.sched), q);
        }

        ds4_fused_verify_refresh_hybrid_luts(w, hybrid, fg,
                                             /*force_masked_routes=*/cluster_hybrid);
    } else {
        if (!fg.sg.alloc) {
            fg.sg.alloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend));
        }
        if (!fg.sg.alloc || !ggml_gallocr_alloc_graph(fg.sg.alloc, fg.sg.gf)) {
            std::fprintf(stderr, "[ds4-fused-verify] graph alloc failed\n");
            return false;
        }
    }
    const auto alloc_t1 = Ds4TimingClock::now();
    if (report_build_timing) {
        std::fprintf(
            stderr,
            "[ds4-fused-verify-build] q=%d kv_start=%d nodes=%d "
            "meta_used=%.2fMiB meta_capacity=%.2fMiB arena=%.1fms "
            "topology=%.1fms alloc=%.1fms total=%.1fms\n",
            q, kv_start, ggml_graph_n_nodes(fg.sg.gf),
            (double) ggml_used_mem(ctx) / (1024.0 * 1024.0),
            (double) fg.sg.meta_arena.size() / (1024.0 * 1024.0),
            (double) ds4_elapsed_us(build_t0, arena_t1) / 1000.0,
            (double) ds4_elapsed_us(arena_t1, topology_t1) / 1000.0,
            (double) ds4_elapsed_us(topology_t1, alloc_t1) / 1000.0,
            (double) ds4_elapsed_us(build_t0, alloc_t1) / 1000.0);
    }
    fg.shape_key = std::move(shape_key);
    return true;
}

// Each built slot keeps its scheduler's device buffers, and at a long
// context those hold an attention's compressed rows as F32 (about 0.5 GiB a
// slot at 128K, so eight slots took 4.4 GiB of the R9700). Before a miss
// builds another slot, retire least-recently-used slots while the device
// has less free than the largest slot plus half its load-time headroom
// (DeepSeek4Backend's fit check): the cache shrinks with the context
// instead of running the device out of memory. A hybrid slot also keeps
// buffers on the peer device, which gets the same check.
static void ds4_fused_cache_make_room(Ds4FusedVerifyCache & vc, size_t slot_limit,
                                      ggml_backend_t backend) {
    for (ggml_backend_t b : {backend, vc.peer_backend}) {
        if (!b) continue;
        size_t largest = 0;
        for (size_t i = 0; i < slot_limit; ++i) {
            ggml_backend_sched_t sched = vc.slots[i].built() ? vc.slots[i].sched : nullptr;
            for (int k = 0; sched && k < ggml_backend_sched_get_n_backends(sched); ++k) {
                if (ggml_backend_sched_get_backend(sched, k) == b) {
                    largest = std::max(largest, ggml_backend_sched_get_buffer_size(sched, b));
                }
            }
        }
        ggml_backend_dev_t dev = largest ? ggml_backend_get_device(b) : nullptr;
        if (!dev) continue;
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(dev, &props);
        const size_t keep = props.type == GGML_BACKEND_DEVICE_TYPE_IGPU
            ? ((size_t) 4 << 30) / 2 : ((size_t) 3 << 29) / 2;
        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_memory(dev, &free_b, &total_b);
        while (free_b < largest + keep) {
            size_t lru = slot_limit;
            for (size_t i = 0; i < slot_limit; ++i) {
                if (vc.slots[i].built() &&
                    (lru == slot_limit || vc.slots[i].last_use < vc.slots[lru].last_use)) {
                    lru = i;
                }
            }
            if (lru == slot_limit) break;
            vc.slots[lru].destroy(vc.backend, vc.peer_backend);
            vc.extra[lru].reset();
            ggml_backend_dev_memory(dev, &free_b, &total_b);
        }
    }
}

// Upper bound of a step's pinned traffic: every per-step input of the slot,
// and the readbacks.
static size_t ds4_fused_stage_bytes(const DeepSeek4Weights & w,
                                    const DeepSeek4FusedDecodeGraph & fg,
                                    const Ds4FusedVerifyCache::Extra & ex,
                                    int q, const Ds4VerifyHooks * hooks) {
    size_t up = 0;
    const auto add = [&](const ggml_tensor * t) { if (t) up += ggml_nbytes(t) + 64; };
    add(fg.inp_embed); add(ex.engram_keys); add(ex.pos_q); add(ex.neg_q);
    add(ex.rawrows); add(ex.saved_rawrows); add(ex.pos_rows);
    for (const auto & rows : ex.ratio_rows) { add(rows.ape); add(rows.state); }
    add(fg.i32_bundle); add(fg.i64_bundle); add(fg.identity_rows); add(fg.mask_bundle);
    for (const ggml_tensor * hids : fg.hash_ids) add(hids);
    size_t down = sizeof(int32_t) * (size_t) q + 256;
    if (hooks && hooks->capture_layer_ids) {
        down += sizeof(float) * hooks->capture_layer_ids->size() * (size_t) w.n_embd * (size_t) q + 64;
    }
    // The sparse mask path uploads ranges; give them room beside the rest.
    return std::max(up, down) + up / 4;
}

static int ds4_try_fused_verify_step(
        Ds4FusedVerifyCache & vc,
        bool capture_cache,
        DeepSeek4FusedDecodeCache & mc,
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        DeepSeek4Cache & cache,
        const std::vector<HcLayerWeightsCpu> & hc_weights,
        const HcWeightsCpu & hc_out_weights,
        const std::vector<HashRoutingTableCpu> & hash_tables,
        std::vector<int32_t> & hash_scratch,
        const float * embed,
        int n_tokens,
        int kv_start,
        std::vector<float> & out_logits,
        const int32_t * token_ids,
        Ds4VerifyHooks * hooks,
        DeepSeek4StepTelemetry * telemetry,
        MoeHybridStorage * hybrid,
        MoeHybridRoutingStats * routing_stats) {
    if (vc.disabled || mc.disabled) return 0;
    if (!hooks || !hooks->capture_layer_ids) return 0;
    if (n_tokens < 1 ||
        n_tokens > GGML_CUDA_DS4_MIX_MMV_MAX_TOKENS) return 0;
    if (!w.hc_staggered_pre &&
        (!hc_out_weights.loaded || hc_out_weights.scale_data.empty() ||
         !w.output_hc_fn || !w.output_hc_base)) { vc.disabled = true; return 0; }
    for (int il = 0; il < w.n_layer; ++il) {
        const HcLayerWeightsCpu & hlw = hc_weights[(size_t) il];
        const DeepSeek4Layer & L = w.layers[(size_t) il];
        if (!hlw.attn.loaded || hlw.attn.scale_data.size() < 3 ||
            !hlw.ffn.loaded || hlw.ffn.scale_data.size() < 3 ||
            !L.hc_attn_fn || !L.hc_ffn_fn || !L.hc_attn_base || !L.hc_ffn_base) {
            vc.disabled = true;
            return 0;
        }
    }
    ggml_backend_t peer_backend = hybrid ? hybrid->cold_backend : nullptr;
    if (vc.owner_ctx != w.ctx || vc.backend != backend ||
        vc.peer_backend != peer_backend) {
        vc.destroy();
        vc.owner_ctx = w.ctx;
        vc.backend = backend;
        vc.peer_backend = peer_backend;
    }
    if (mc.owner_ctx != w.ctx || mc.backend != backend) {
        mc.destroy();
        mc.owner_ctx = w.ctx;
        mc.backend = backend;
    }
    if (!ds4_fused_ensure_fn_mirrors(mc, backend, w, hc_weights, hc_out_weights)) {
        vc.disabled = true;
        return 0;
    }

    const int q = n_tokens;
    std::vector<int64_t> key;
    key.reserve((size_t) w.n_layer + hooks->capture_layer_ids->size() + 5);
    key.push_back(w.n_swa);
    key.push_back(q);
    key.push_back(token_ids ? 1 : 0);
    key.push_back(hybrid ? 1 : 0);
    key.push_back((int64_t) hooks->capture_layer_ids->size());
    for (int layer : *hooks->capture_layer_ids) key.push_back(layer);
    static const bool vocab_split_env = ds4_env_flag("LUCE_CLUSTER_VOCAB_SPLIT") &&
                                        ds4_env_flag("LUCE_DS4_GPU_ARGMAX_VERIFY");
    const bool vocab_split = vocab_split_env && hybrid && hooks->prefer_argmax_only && hooks->argmax_out &&
                             cache.cluster_rt && cache.cluster_rt->size() == 2;
    key.push_back(vocab_split ? 1 : 0);
    static bool vocab_split_reported = false;
    if (vocab_split_env && !vocab_split_reported) {
        vocab_split_reported = true;
        std::fprintf(stderr, "[ds4-fused-verify] vocab split %d (hybrid %d argmax_only %d argmax_out %d cluster %d)\n",
                     vocab_split ? 1 : 0, hybrid ? 1 : 0, hooks->prefer_argmax_only ? 1 : 0,
                     hooks->argmax_out ? 1 : 0, cache.cluster_rt ? cache.cluster_rt->size() : 0);
    }
    const int n_lanes = ds4_fused_attention_lanes(w, q);
    const int attn_lane_q = q / n_lanes;
    for (int il = 0; il < w.n_layer; ++il) {
        const int ratio = (int) w.compress_ratios[il];
        DeepSeek4LayerCache & lc = ds4_comp_cache(cache, w, il);
        for (int lane = 0; lane < n_lanes; ++lane) {
            const int lane_kv_start = kv_start + lane * attn_lane_q;
            const int padded = ds4_fused_lane_padded(
                w, lc, il, lane_kv_start + attn_lane_q - 1);
            int b_idx = -1;
            if (ratio > 0) {
                for (int t = 0; t < attn_lane_q; ++t) {
                    if (((lane_kv_start + t + 1) % ratio) == 0) { b_idx = t; break; }
                }
            }
            key.push_back(
                ((int64_t) padded << 4) | (int64_t) (b_idx + 1));
        }
    }
    // The key a step at (kv, qq) would look up: the rules above.
    auto key_for = [&](int kv, int qq) {
        std::vector<int64_t> k;
        k.reserve(key.size());
        k.push_back(w.n_swa);
        k.push_back(qq);
        k.push_back(token_ids ? 1 : 0);
        k.push_back(hybrid ? 1 : 0);
        k.push_back((int64_t) hooks->capture_layer_ids->size());
        for (int layer : *hooks->capture_layer_ids) k.push_back(layer);
        k.push_back(vocab_split ? 1 : 0);
        const int nl = ds4_fused_attention_lanes(w, qq);
        const int alq = qq / nl;
        for (int il = 0; il < w.n_layer; ++il) {
            const int ratio = (int) w.compress_ratios[il];
            DeepSeek4LayerCache & lc = ds4_comp_cache(cache, w, il);
            for (int lane = 0; lane < nl; ++lane) {
                const int lks = kv + lane * alq;
                const int padded = ds4_fused_lane_padded(w, lc, il, lks + alq - 1);
                int b_idx = -1;
                if (ratio > 0) {
                    for (int t = 0; t < alq; ++t) {
                        if (((lks + t + 1) % ratio) == 0) { b_idx = t; break; }
                    }
                }
                k.push_back(((int64_t) padded << 4) | (int64_t) (b_idx + 1));
            }
        }
        return k;
    };

    vc.counter++;
    const size_t slot_limit = hybrid
        ? (capture_cache ? ds4_fused_capture_hybrid_slot_limit()
                         : ds4_fused_verify_hybrid_slot_limit())
        : vc.slots.size();
    static bool reported_verify_slot_limit = false;
    static bool reported_capture_slot_limit = false;
    bool & reported_slot_limit = capture_cache
        ? reported_capture_slot_limit : reported_verify_slot_limit;
    if (hybrid && !reported_slot_limit) {
        std::fprintf(stderr,
                     "[ds4-fused-verify] hybrid %s cache slots=%zu\n",
                     capture_cache ? "capture" : "verify", slot_limit);
        reported_slot_limit = true;
    }
    DeepSeek4FusedDecodeGraph * fg = nullptr;
    Ds4FusedVerifyCache::Extra * ex = nullptr;
    for (size_t i = 0; i < slot_limit; ++i) {
        if (vc.slots[i].built() && vc.slots[i].shape_key == key) {
            fg = &vc.slots[i]; ex = &vc.extra[i]; break;
        }
    }
    if (!fg) {
        ds4_fused_cache_make_room(vc, slot_limit, backend);
        size_t pick = 0;
        for (size_t i = 0; i < slot_limit; ++i) {
            if (!vc.slots[i].built()) { pick = i; break; }
            if (vc.slots[i].last_use < vc.slots[pick].last_use) pick = i;
        }
        fg = &vc.slots[pick]; ex = &vc.extra[pick];

        // A cache miss may select an occupied LRU slot.  Tear the previous
        // graph down before rebuilding it: its scheduler owns backend scratch
        // buffers and HIP graph replay state still refers to those addresses.
        // Reinitializing the ggml context over the same metadata arena while
        // that state is live can replay kernels against stale allocations.
        fg->destroy(vc.backend, vc.peer_backend);
        ex->reset();
        ex->vocab_split = vocab_split;

        const auto build_t0 = Ds4TimingClock::now();
        if (!ds4_build_fused_verify_graph(mc, *fg, *ex, backend, w, &cache,
                                          hc_weights, hc_out_weights, hash_tables,
                                          kv_start, q, token_ids != nullptr,
                                          *hooks->capture_layer_ids, hybrid,
                                          std::move(key))) {
            fg->destroy(vc.backend, vc.peer_backend);
            ex->reset();
            vc.disabled = true;
            return 0;
        }
        if (telemetry) telemetry->full_graph_build_us += ds4_elapsed_us(build_t0, Ds4TimingClock::now());
    }
    fg->last_use = vc.counter;

    // ── Fill inputs ──
    // With pinned step I/O the mask bundle and the readbacks go through pinned
    // slices (one wait for all of them, below) and the Engram rows are read on
    // a worker while the graph already runs.
    const bool pinned_io = ds4_pinned_step_io_enabled() &&
        vc.stage.reserve(backend, ds4_fused_stage_bytes(w, *fg, *ex, q, hooks));
    vc.stage.reset();
    // The async read needs a place in the stream right before its consumer:
    // the cluster's Engram broadcast callback (after layer 0's attention).
    Ds4ClusterRuntime * engram_rt = cache.cluster_rt && cache.cluster_rt->size() > 1
        ? cache.cluster_rt : nullptr;
    bool engram_async = false;
    if (ex->engram_keys) {
        const DeepSeek4EngramRuntime * engram = w.engram_runtime.get();
        if (pinned_io && engram_rt && engram && token_ids && ds4_engram_async_enabled() &&
            ex->engram_keys->buffer) {
            if (!vc.engram_async) vc.engram_async = std::make_unique<Ds4EngramAsync>();
            engram_async = vc.engram_async->start(
                backend, w, cache.engram_tokens, token_ids, kv_start, q,
                (size_t) engram->n_layers() * (size_t) q * engram->key_floats());
        }
        if (!engram_async) {
            std::vector<float> keys;
            if (!ds4_engram_read_keys(w, cache.engram_tokens, token_ids, kv_start, q, keys, telemetry)) {
                return -1;
            }
            ds4_fv_set(ex->engram_keys, keys.data(), sizeof(float) * keys.size());
        }
    }
    // A started read must be waited for on every exit, or the worker could
    // still write the context the caller rolls back.
    struct EngramJoin {
        Ds4EngramAsync * reader;
        DeepSeek4StepTelemetry * telemetry;
        bool ok = true;
        bool finish() {
            if (!reader) return ok;
            uint64_t us = 0;
            ok = reader->finish(&us);
            if (telemetry) telemetry->engram_read_us += us;
            reader = nullptr;
            return ok;
        }
        ~EngramJoin() { finish(); }
    } engram_join{engram_async ? vc.engram_async.get() : nullptr, telemetry};
    const auto set_t0 = Ds4TimingClock::now();
    // LUCE_DS4_STAGE_BATCH=1: the step's pinned uploads leave as one copy
    // kernel before the single wait below, instead of a dozen copies of
    // which the larger ones wait for a copy engine to start.
    static const bool stage_batch_env = [] {
        const char * v = std::getenv("LUCE_DS4_STAGE_BATCH");
        return v && *v && std::strcmp(v, "0") != 0;
    }();
    const bool stage_batch = stage_batch_env && pinned_io && ggml_backend_is_cuda(backend);
    if (stage_batch) vc.stage.begin_batch();
    ds4_fv_set(fg->inp_embed, embed, sizeof(float) * (size_t) w.n_embd * q);
    std::vector<int32_t> iv(q);
    std::vector<int64_t> lv(q);
    for (int i = 0; i < q; ++i) iv[(size_t) i] = kv_start + i;
    ds4_fv_set(ex->pos_q, iv.data(), sizeof(int32_t) * q);
    for (int i = 0; i < q; ++i) iv[(size_t) i] = -(kv_start + i);
    ds4_fv_set(ex->neg_q, iv.data(), sizeof(int32_t) * q);
    for (int i = 0; i < q; ++i) lv[(size_t) i] = (kv_start + i) % w.n_swa;
    ds4_fv_set(ex->rawrows, lv.data(), sizeof(int64_t) * q);
    for (int i = 0; i < q; ++i) iv[(size_t) i] = (int32_t) lv[(size_t) i];
    ds4_fv_set(ex->saved_rawrows, iv.data(), sizeof(int32_t) * q);
    if (ex->pos_rows && ex->pos_rows->buffer) {
        for (int i = 0; i < q; ++i) lv[(size_t) i] = kv_start + i;
        ds4_fv_set(ex->pos_rows, lv.data(), sizeof(int64_t) * q);
    }
    for (const auto & rows : ex->ratio_rows) {
        for (int i = 0; i < q; ++i) iv[(size_t) i] = (kv_start + i) % rows.ratio;
        ds4_fv_set(rows.ape, iv.data(), sizeof(int32_t) * q);
        for (int i = 0; i < q; ++i) {
            lv[(size_t) i] = ds4_compressor_state_row(rows.ratio, kv_start + i);
        }
        ds4_fv_set(rows.state, lv.data(), sizeof(int64_t) * q);
    }

    std::vector<int32_t> i32v((size_t) w.n_layer * n_lanes * 2, 0);
    std::vector<int64_t> i64v((size_t) w.n_layer * n_lanes * 2, 0);
    for (int il = 0; il < w.n_layer; ++il) {
        const int ratio = (int) w.compress_ratios[il];
        if (ratio <= 0) continue;
        for (int lane = 0; lane < n_lanes; ++lane) {
            const size_t slot = ((size_t) il * n_lanes + (size_t) lane) * 2;
            const int lane_kv_start = kv_start + lane * attn_lane_q;
            const int lane_token_pos = lane_kv_start + attn_lane_q - 1;
            int n_flush = 0;
            for (int t = 0; t < attn_lane_q; ++t) {
                if (((lane_kv_start + t + 1) % ratio) != 0) continue;
                GGML_ASSERT(n_flush < 2);
                const int pos_b = lane_kv_start + t;
                i32v[slot + (size_t) n_flush] = pos_b + 1 - ratio;
                i64v[slot + (size_t) n_flush] = pos_b / ratio;
                ++n_flush;
            }
            if (n_flush == 0) {
                // The graph does not consume these values in a no-flush shape,
                // but initialize them deterministically for diagnostics.
                i32v[slot] = lane_token_pos + 1 - ratio;
                i64v[slot] = lane_token_pos / ratio;
            }
        }
    }
    ds4_fv_set(fg->i32_bundle, i32v.data(), sizeof(int32_t) * i32v.size());
    if (fg->identity_rows) {
        static thread_local std::vector<int32_t> identity;
        if (identity.size() != (size_t) fg->identity_rows->ne[0]) {
            identity.resize((size_t) fg->identity_rows->ne[0]);
            for (size_t i = 0; i < identity.size(); ++i) identity[i] = (int32_t) i;
        }
        ds4_fv_set(fg->identity_rows, identity.data(), sizeof(int32_t) * identity.size());
    }
    ds4_fv_set(fg->i64_bundle, i64v.data(), sizeof(int64_t) * i64v.size());

    static const bool set_timing = std::getenv("LUCE_DS4_SET_TIMING") != nullptr;
    static double st_inputs = 0, st_mask = 0, st_hash = 0, st_sync = 0; static int st_n = 0;
    auto st_now = [] { return std::chrono::steady_clock::now(); };
    auto st_us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
    const auto st_t1 = st_now();
    st_inputs += st_us(set_t0, st_t1);
    // Causal mask values.  At long context this tensor is multi-megabyte,
    // while only the ring slots overwritten by this q-wide batch and the
    // padded compressed tail change between verifier steps.  Build those
    // negative spans directly.  Large masks are zeroed on the GPU, then only
    // those negative spans cross the host/backend boundary.
    {
        using MaskRange = std::pair<size_t, size_t>; // [begin, end)
        const size_t mask_count = (size_t) ggml_nelements(fg->mask_bundle);
        std::vector<float> & maskv = ex->mask_values;
        std::vector<MaskRange> next_negative;
        next_negative.reserve((size_t) w.n_layer * (size_t) q * 4u);
        const auto add_range = [&](size_t begin, size_t end) {
            if (begin < end) next_negative.emplace_back(begin, end);
        };
        size_t off = 0;
        for (int il = 0; il < w.n_layer; ++il) {
          const int ratio = (int) w.compress_ratios[il];
          DeepSeek4LayerCache & lc = ds4_comp_cache(cache, w, il);
          for (int lane = 0; lane < n_lanes; ++lane) {
            const int lane_q = attn_lane_q;
            const int lane_start = lane * attn_lane_q;
            const int lane_kv_start = kv_start + lane_start;
            const int lane_token_pos = lane_kv_start + lane_q - 1;
            const int padded = ds4_fused_lane_padded(w, lc, il, lane_token_pos);
            const size_t n_attn = (size_t) w.n_swa + padded +
                                  (lane_q > 1 ? (size_t) lane_q : 0u);
            for (int i = 0; i < lane_q; ++i) {
                const size_t col = off + (size_t) i * n_attn;
                const int pos_i = lane_kv_start + i;

                DeepSeek4RawRingSpan raw_spans[2];
                const int n_raw_spans = deepseek4_verify_raw_mask_spans(
                    lane_kv_start, w.n_swa, lane_q, i, raw_spans);
                for (int span = 0; span < n_raw_spans; ++span) {
                    const auto & range = raw_spans[span];
                    add_range(col + (size_t) range.row,
                              col + (size_t) (range.row + range.count));
                }

                const int vis = (ratio > 0 && lc.comp_kv)
                              ? ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio, pos_i) : 0;
                add_range(col + (size_t) w.n_swa + (size_t) vis,
                          col + (size_t) w.n_swa + (size_t) padded);

                if (lane_q > 1) {
                    const int first_visible = std::max(
                        i + 1, w.n_swa - lane_kv_start);
                    const int masked = std::clamp(first_visible, 0, lane_q);
                    add_range(col + (size_t) w.n_swa + (size_t) padded,
                              col + (size_t) w.n_swa + (size_t) padded +
                                  (size_t) masked);
                }
            }
            off += n_attn * lane_q;
          }
        }
        GGML_ASSERT(off == mask_count);

        const auto normalize_ranges = [](std::vector<MaskRange> & ranges) {
            std::sort(ranges.begin(), ranges.end());
            size_t out = 0;
            for (const MaskRange & range : ranges) {
                if (out > 0 && range.first <= ranges[out - 1].second) {
                    ranges[out - 1].second = std::max(
                        ranges[out - 1].second, range.second);
                } else {
                    ranges[out++] = range;
                }
            }
            ranges.resize(out);
        };
        normalize_ranges(next_negative);

        static const bool sparse_mask_update = [] {
            const char * value = std::getenv(
                "LUCE_DS4_INCREMENTAL_VERIFY_MASK");
            return !value || std::atoi(value) != 0;
        }();
        static const size_t sparse_mask_min_bytes = [] {
            constexpr size_t default_bytes = 4u * 1024u * 1024u;
            const char * value = std::getenv(
                "LUCE_DS4_INCREMENTAL_VERIFY_MASK_MIN_BYTES");
            if (!value || !*value) return default_bytes;
            const unsigned long long parsed = std::strtoull(value, nullptr, 10);
            return parsed > 0 ? (size_t) parsed : default_bytes;
        }();
        const bool use_sparse_mask_update = sparse_mask_update &&
            mask_count * sizeof(float) >= sparse_mask_min_bytes;
        if (!use_sparse_mask_update) {
            maskv.assign(mask_count, 0.0f);
            for (const MaskRange & range : next_negative) {
                std::fill(maskv.begin() + range.first,
                          maskv.begin() + range.second, -1.0e30f);
            }
            ds4_fv_set(fg->mask_bundle, maskv.data(),
                       sizeof(float) * maskv.size());
        } else {
            if (maskv.size() != mask_count) maskv.resize(mask_count);
            ggml_backend_tensor_memset(
                fg->mask_bundle, 0, 0, mask_count * sizeof(float));
            for (const MaskRange & range : next_negative) {
                std::fill(maskv.begin() + range.first,
                          maskv.begin() + range.second, -1.0e30f);
                if (pinned_io) {
                    vc.stage.set(backend, fg->mask_bundle, maskv.data() + range.first,
                                 range.first * sizeof(float),
                                 (range.second - range.first) * sizeof(float));
                } else {
                    ggml_backend_tensor_set(
                        fg->mask_bundle, maskv.data() + range.first,
                        range.first * sizeof(float),
                        (range.second - range.first) * sizeof(float));
                }
            }
        }
    }

    const auto st_t2 = st_now();
    st_mask += st_us(st_t1, st_t2);
    if (token_ids) {
        for (int il = 0; il < w.n_layer; ++il) {
            ggml_tensor * hids = fg->hash_ids[(size_t) il];
            if (!hids) continue;
            const HashRoutingTableCpu & table = hash_tables[(size_t) il];
            const int n_used = (int) hids->ne[0];
            hash_scratch.resize((size_t) n_used * q);
            for (int i = 0; i < q; ++i) {
                const int32_t * row = hash_routing_row(
                    table, token_ids[i], w.n_expert_used);
                if (!row) {
                    std::fprintf(stderr,
                                 "[ds4-fused-verify] token id %d outside hash table for layer %d\n",
                                 token_ids[i], il);
                    return -1;
                }
                std::memcpy(hash_scratch.data() + (size_t) i * n_used,
                            row,
                            (size_t) n_used * sizeof(int32_t));
            }
            ds4_fv_set(hids, hash_scratch.data(), sizeof(int32_t) * (size_t) n_used * q);
        }
    }
    // One wait for every pinned upload: the scheduler copies some inputs to
    // the peer device with a blocking copy that is not ordered behind this
    // stream. The Engram keys are queued after it: their upload waits on the
    // GPU for the worker, not on the host.
    const auto st_t3 = st_now();
    st_hash += st_us(st_t2, st_t3);
    if (stage_batch) {
        const auto & pending = vc.stage.batch();
        if (!pending.empty()) {
            std::vector<ggml_cuda_copy_desc> descs;
            descs.reserve(pending.size());
            for (const auto & p : pending) descs.push_back({p.src, p.dst, p.bytes});
            ggml_backend_cuda_copy_batch_async(backend, descs.data(), (int) descs.size());
        }
        vc.stage.end_batch();
    }
    if (pinned_io) ggml_backend_synchronize(backend);
    st_sync += st_us(st_t3, st_now());
    if (set_timing && ++st_n % 64 == 0) {
        std::fprintf(stderr, "[ds4-set-t] per verify over %d: inputs %.0f mask %.0f hash %.0f sync %.0f us (pinned %d)\n",
                     st_n, st_inputs / st_n, st_mask / st_n, st_hash / st_n, st_sync / st_n, pinned_io ? 1 : 0);
    }
    // The root's broadcast sends the keys: its callback queues the upload
    // behind the worker's word, so layer 0 runs while the rows are read.
    // The other ranks' keys are overwritten by the broadcast, their worker
    // only records the tokens in the n-gram context.
    struct BroadcastHook {
        Ds4ClusterRuntime * rt;
        ~BroadcastHook() { if (rt) rt->before_broadcast = nullptr; }
    } broadcast_hook{nullptr};
    if (engram_async) {
        Ds4EngramAsync * reader = vc.engram_async.get();
        const int rank = engram_rt->rank();
        const size_t key_bytes = ggml_nbytes(ex->engram_keys);
        engram_rt->before_broadcast = [reader, rank, key_bytes](void * data, size_t n, void * stream) {
            if (rank != 0) return true;
            return n * sizeof(float) == key_bytes && reader->enqueue_raw(stream, data, key_bytes);
        };
        broadcast_hook.rt = engram_rt;
    }
    if (telemetry) {
        telemetry->full_graph_set_us += ds4_elapsed_us(set_t0, Ds4TimingClock::now());
    }

    // ── Compute + read back ──
    const auto compute_t0 = Ds4TimingClock::now();

    // A q=1 feature-capture graph contains HC width-4 matmuls even though it
    // advances only one token. On gfx1151, selecting MMVQ for those width-4
    // nodes is correct in eager mode but its HIP graph replay faults against
    // unified-memory expert mappings. Keep the proven width-3 ceiling only
    // while q=1 capture graphs are evaluated/captured. q>=2 verification uses
    // the process ceiling (LUCE_MMVQ_MAX_NCOLS=5 for q=5 verification).
    const bool capture_eager = capture_cache &&
        ds4_env_flag("LUCE_DS4_TP_CAPTURE_EAGER");
    // q>=2 verifier graph slots own fixed tensor storage. Position, mask,
    // routing-map, and embedding values are refreshed through stable input
    // buffers above; the graph topology and all allocation addresses remain
    // immutable until the slot is destroyed. The generic CUDA/HIP property
    // scan can nevertheless observe backend-mutated metadata and reset replay
    // at request boundaries. Mirror the qualified draft fast path behind an
    // independent opt-in while retaining normal warmup/capture semantics. The
    // backend additionally checks the scheduler graph uid, so a rebuilt slot
    // can never bypass validation using the previous generation's executable.
    const bool force_verify_graph_replay = q > 1 &&
        ds4_env_flag("LUCE_DS4_VERIFY_FORCE_GRAPH_REPLAY");
    // Tokenwise lanes match single-token decode only with batch-invariant
    // products in the batched projections, FFN and output head.
    ScopedCudaGraphOverrides graph_override_scope(
        /*disable_graphs=*/capture_eager,
        /*mmvq_max_ncols=*/capture_cache && q == 1 ? 3 : 0,
        /*skip_property_check=*/force_verify_graph_replay,
        /*ds4_mix_mmv_max_tokens=*/0,
        /*mmvq_batch_invariant=*/n_lanes > 1);
    // Experimental, off by default: on the two-box code run it filled the slot
    // cache and the graphs thrashed through HIP-graph warmup (verify 42 -> 82 ms).
    // LUCE_DS4_VERIFY_PREBUILD=<n> (hybrid): while this graph runs, build up
    // to n slots for the shapes the next step will most likely look up (this
    // width, the next start after a full or one-short acceptance) when a free
    // slot is left. A miss costs a ~15 ms rebuild on the critical path; here
    // the host would only wait for the GPUs.
    static const int prebuild_max = [] {
        const char * v = std::getenv("LUCE_DS4_VERIFY_PREBUILD");
        return v && *v ? std::max(0, std::atoi(v)) : 0;
    }();
    std::function<void()> prebuild;
    if (prebuild_max > 0 && hybrid && q > 1) {
        prebuild = [&]() {
            int built = 0;
            for (int a = q; a >= 1 && built < prebuild_max; --a) {
                std::vector<int64_t> k = key_for(kv_start + a, q);
                bool have = false;
                for (size_t i = 0; i < slot_limit && !have; ++i) {
                    have = vc.slots[i].built() && vc.slots[i].shape_key == k;
                }
                if (have) continue;
                size_t free_slot = slot_limit;
                for (size_t i = 0; i < slot_limit; ++i) {
                    if (!vc.slots[i].built() && &vc.slots[i] != fg) { free_slot = i; break; }
                }
                if (free_slot == slot_limit) return;   // never evict under a running graph
                DeepSeek4FusedDecodeGraph & pf = vc.slots[free_slot];
                Ds4FusedVerifyCache::Extra & pe = vc.extra[free_slot];
                pf.destroy(vc.backend, vc.peer_backend);
                pe.reset();
                pe.vocab_split = vocab_split;
                const auto pb_t0 = Ds4TimingClock::now();
                if (!ds4_build_fused_verify_graph(mc, pf, pe, backend, w, &cache,
                                                  hc_weights, hc_out_weights, hash_tables,
                                                  kv_start + a, q, token_ids != nullptr,
                                                  *hooks->capture_layer_ids, hybrid,
                                                  std::move(k))) {
                    pf.destroy(vc.backend, vc.peer_backend);
                    pe.reset();
                    return;
                }
                pf.last_use = 0;   // unused prebuilt slots are the first to go
                ++built;
                static int reported = 0;
                if (reported++ < 32) {
                    std::fprintf(stderr, "[ds4-fused-verify] prebuilt q=%d kv=%d in %.1f ms\n", q,
                                 kv_start + a, ds4_elapsed_us(pb_t0, Ds4TimingClock::now()) / 1000.0);
                }
            }
        };
    }
    const enum ggml_status compute_status =
        ds4_fused_graph_compute(w, *fg, backend, hybrid, "ds4-fused-verify", prebuild);
    if (broadcast_hook.rt) {
        broadcast_hook.rt->before_broadcast = nullptr;
        broadcast_hook.rt = nullptr;
    }
    if (!engram_join.finish()) return -1;
    if (compute_status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[ds4-fused-verify] compute failed\n");
        return -1;
    }
    if (telemetry) {
        telemetry->full_graph_compute_us += ds4_elapsed_us(compute_t0, Ds4TimingClock::now());
    }

    const auto read_t0 = Ds4TimingClock::now();
    ds4_fused_consume_route_diagnostics(*fg, hybrid, routing_stats);
    const int ncap = (int) hooks->capture_layer_ids->size();
    const bool argmax_only =
        hooks->prefer_argmax_only && hooks->argmax_out && ex->argmax;
    // The small readbacks go out together: queue every copy into pinned
    // slices, then one wait.
    const void * argmax_src = nullptr;
    const void * capture_src = nullptr;
    const bool want_capture = hooks->capture_out && ex->capture && ncap > 0;
    const bool want_window = hooks->window_rows && !ex->window_layers.empty() &&
        !ex->window_kv_rows.empty() &&
        ex->window_kv_rows.size() == ex->window_score_rows.size() &&
        ex->window_kv_rows.size() % ex->window_layers.size() == 0;
    if (pinned_io) {
        vc.stage.reset();
        if (argmax_only) {
            argmax_src = vc.stage.get(backend, ex->argmax, 0, sizeof(int32_t) * (size_t) q);
        }
        if (want_capture) {
            capture_src = vc.stage.get(backend, ex->capture, 0,
                                       sizeof(float) * (size_t) ncap * w.n_embd * q);
        }
        ggml_backend_synchronize(backend);
    }
    if (argmax_only) {
        hooks->argmax_out->resize((size_t) q);
        if (argmax_src) {
            std::memcpy(hooks->argmax_out->data(), argmax_src, sizeof(int32_t) * (size_t) q);
        } else {
            ggml_backend_tensor_get(ex->argmax, hooks->argmax_out->data(), 0,
                                    sizeof(int32_t) * (size_t) q);
        }
    } else if (hooks->all_logits_out) {
        hooks->all_logits_out->resize((size_t) w.n_vocab * q);
        ggml_backend_tensor_get(fg->logits, hooks->all_logits_out->data(), 0,
                                sizeof(float) * (size_t) w.n_vocab * q);
    }
    if (!argmax_only) {
        out_logits.resize((size_t) w.n_vocab);
        ggml_backend_tensor_get(fg->logits, out_logits.data(),
                                (size_t) (q - 1) * (size_t) w.n_vocab * sizeof(float),
                                sizeof(float) * (size_t) w.n_vocab);
    } else {
        out_logits.clear();
    }
    if (want_capture) {
        hooks->capture_out->resize((size_t) ncap * w.n_embd * q);
        if (capture_src) {
            std::memcpy(hooks->capture_out->data(), capture_src,
                        sizeof(float) * hooks->capture_out->size());
        } else {
            ggml_backend_tensor_get(
                ex->capture, hooks->capture_out->data(), 0,
                sizeof(float) * hooks->capture_out->size());
        }
    }
    if (hooks->boundary_checkpoint_out) {
        *hooks->boundary_checkpoint_out = ex->boundary_checkpoint;
    }
    if (want_window) {
        // Layer-major rows -> per layer, one row tensor per batch token.
        Ds4VerifyWindowRows & rows = *hooks->window_rows;
        rows.kv.assign((size_t) w.n_layer, {});
        rows.score.assign((size_t) w.n_layer, {});
        rows.kv_dev.assign((size_t) w.n_layer, {});
        rows.score_dev.assign((size_t) w.n_layer, {});
        const size_t per_layer = ex->window_kv_rows.size() / ex->window_layers.size();
        for (size_t i = 0; i < ex->window_layers.size(); ++i) {
            const size_t il = (size_t) ex->window_layers[i];
            rows.kv_dev[il].assign(ex->window_kv_rows.begin() + i * per_layer,
                                   ex->window_kv_rows.begin() + (i + 1) * per_layer);
            rows.score_dev[il].assign(ex->window_score_rows.begin() + i * per_layer,
                                      ex->window_score_rows.begin() + (i + 1) * per_layer);
        }
    }
    if (telemetry) {
        telemetry->full_graph_read_us += ds4_elapsed_us(read_t0, Ds4TimingClock::now());
    }
    return 1;
}
