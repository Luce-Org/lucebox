// ─── Fused whole-model graph (n_tokens = q in [1,4]) ────────────────────
// One whole-model graph per (q, flush, padded-comp) shape: batched attention
// (causal mask input) + batched MoE, per-token fused HC chains, in-graph
// optional drafter-feature capture and q-wide logits. The spec loop guarantees a
// ratio-4 boundary can only sit at the batch's LAST token (dynamic q), so the
// compressor pools at most once per step, exactly like an AR step at the last
// position. It also serves opt-in q=1 hybrid decode, because the older q=1
// fused graph is single-backend. Enable with DFLASH_DS4_FUSED_VERIFY=1 and,
// for q=1 hybrid decode, DFLASH_DS4_FUSED_HYBRID_DECODE=1.

// tensor_set that tolerates inputs the graph never consumed (gallocr leaves
// them unallocated, e.g. comp-emission bundles in non-flush graphs).
static void ds4_fv_set(ggml_tensor * t, const void * data, size_t nbytes) {
    if (t && t->buffer) ggml_backend_tensor_set(t, data, 0, nbytes);
}
static bool ds4_fused_verify_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = ds4_env_flag("DFLASH_DS4_FUSED_VERIFY") ? 1 : 0;
    }
    return enabled == 1;
}

// Zero-output-change audit for Luce Spark's proposed layer-ahead cache.  A
// positive value asks the graph to predict routes that many layers ahead.
// The widest candidate list is retained once; post-compute accounting reports
// recall at the authoritative route width and at wider prefixes.
static int ds4_tp_route_predict_audit_lookahead() {
    static const int value = [] {
        const char * raw = std::getenv("DFLASH_DS4_TP_PREDICT_AUDIT");
        return raw ? std::max(0, std::min(4, std::atoi(raw))) : 0;
    }();
    return value;
}

static int ds4_tp_route_predict_audit_width() {
    static const int value = [] {
        const char * raw = std::getenv("DFLASH_DS4_TP_PREDICT_WIDTH");
        return raw ? std::max(1, std::min(32, std::atoi(raw))) : 12;
    }();
    return value;
}

// Opt-in heterogeneous q4 pipeline.  The ordinary verifier batches all four
// tokens through attention and then places a hot+cold expert join at every
// transformer layer.  On R9700 + Strix this exposes the tail of the slower
// owner once per transformer layer.  The wavefront keeps two q2 lanes in
// flight instead:
// lane B's attention/FFN prefix is appended before lane A's deferred join,
// then lane A advances to the next layer before lane B joins.  Both lanes
// remain exact causal target work; this is not drafting or prediction.
static bool ds4_tp_q2_wavefront_requested() {
    static const bool enabled =
        ds4_env_flag("DFLASH_DS4_TP_Q2_WAVEFRONT");
    return enabled;
}

static bool ds4_tp_q2_wavefront_active(
        int q, int, const MoeHybridStorage * hybrid) {
    // build_compressor_step already supports one boundary at any index inside
    // a causal batch.  Exactly one q2 lane owns that boundary; its span-A
    // write, pool/rotation, and optional span-B write preserve q4 semantics.
    return ds4_tp_q2_wavefront_requested() && hybrid && q == 4;
}

static bool ds4_fused_verify_trace_node(
        ggml_tensor * tensor, bool ask, void *) {
    static const bool trace_all =
        ds4_env_flag("DFLASH_DS4_TP_SCHED_TRACE_ALL");
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
        ds4_env_flag("DFLASH_DS4_TP_SCHED_TRACE_IDS")) {
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

struct Ds4FusedVerifyCache {
    const ggml_context * owner_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_t peer_backend = nullptr;
    bool disabled = false;
    uint64_t counter = 0;
    std::array<DeepSeek4FusedDecodeGraph, 12> slots;
    // verify-specific inputs/outputs per slot (parallel arrays, same index)
    struct Extra {
        ggml_tensor * pos_q = nullptr;    // i32 [q]
        ggml_tensor * neg_q = nullptr;    // i32 [q]
        ggml_tensor * rawrows = nullptr;  // i64 [1,q]
        ggml_tensor * ape4 = nullptr;     // i32 [q]
        ggml_tensor * ape128 = nullptr;   // i32 [q]
        ggml_tensor * st4 = nullptr;      // i64 [1,q]
        ggml_tensor * st128 = nullptr;    // i64 [1,q]
        ggml_tensor * capture = nullptr;  // f32 [n_embd*ncap*q], order [ci][t]
        ggml_tensor * argmax = nullptr;   // i32 [q], optional greedy verify output
        int q = 0;
        void reset() { *this = Extra{}; }
    };
    std::array<Extra, 12> extra;
    void destroy() {
        for (auto & s : slots) s.destroy();
        for (auto & e : extra) e.reset();
        owner_ctx = nullptr;
        backend = nullptr;
        peer_backend = nullptr;
        disabled = false;
        counter = 0;
    }
};

static Ds4FusedVerifyCache fused_verify_graph_cache;
// q=1 target-feature capture walks many prompt-position shapes.  Keep those
// graphs out of the q>=2 speculative verifier LRU or every fresh prefill
// evicts the expensive warm q=3 verifier working set.
static Ds4FusedVerifyCache fused_capture_graph_cache;

void reset_deepseek4_graph_runtime_caches() {
    fused_verify_graph_cache.destroy();
    fused_capture_graph_cache.destroy();
    if (g_deepseek4_fused_decode_runtime_cache) {
        g_deepseek4_fused_decode_runtime_cache->destroy();
        g_deepseek4_fused_decode_runtime_cache = nullptr;
    }
}

// Cached verifier graphs retain their input tensors, while a request-adaptive
// placement changes only the global->local expert maps.  Refresh those small
// LUT inputs without rebuilding or reallocating the graph.  Mask any hot/cold
// duplication defensively so every selected route is evaluated exactly once.
static void ds4_fused_verify_refresh_hybrid_luts(
        const DeepSeek4Weights & w,
        MoeHybridStorage * hybrid,
        DeepSeek4FusedDecodeGraph & fg,
        bool q2_wavefront) {
    if (!hybrid) return;
    const int32_t invalid_route =
        ds4_env_flag("DFLASH_DS4_TP_MASKED_ROUTES") ? -1 : 0;
    for (size_t hi = 0; hi < fg.hybrid_inputs.size(); ++hi) {
        const int il = q2_wavefront ? (int) (hi / 2u) : (int) hi;
        const MoeHybridLayerStorage & layer = hybrid->layers[(size_t) il];
        const MoeHybridGraphInputs & inputs = fg.hybrid_inputs[hi];
        std::vector<int32_t> hot_lut((size_t) w.n_expert, invalid_route);
        std::vector<int32_t> cold_lut((size_t) w.n_expert, invalid_route);
        std::vector<float> hot_valid((size_t) w.n_expert, 0.0f);
        std::vector<float> cold_valid((size_t) w.n_expert, 0.0f);
        for (int ie = 0; ie < w.n_expert; ++ie) {
            const int32_t hot = layer.hot_local_by_global[(size_t) ie];
            const int32_t cold = layer.cold_local_by_global[(size_t) ie];
            if (hot >= 0) {
                hot_lut[(size_t) ie] = hot;
                hot_valid[(size_t) ie] = 1.0f;
            }
            if (cold >= 0 && hot < 0) {
                cold_lut[(size_t) ie] = cold;
                cold_valid[(size_t) ie] = 1.0f;
            }
        }
        ds4_fv_set(inputs.hot_local_lut, hot_lut.data(),
                   sizeof(int32_t) * hot_lut.size());
        ds4_fv_set(inputs.hot_valid_lut, hot_valid.data(),
                   sizeof(float) * hot_valid.size());
        ds4_fv_set(inputs.cold_local_lut, cold_lut.data(),
                   sizeof(int32_t) * cold_lut.size());
        ds4_fv_set(inputs.cold_valid_lut, cold_valid.data(),
                   sizeof(float) * cold_valid.size());
        ds4_fv_set(inputs.shard_local_lut, cold_lut.data(),
                   sizeof(int32_t) * cold_lut.size());
        ds4_fv_set(inputs.shard_valid_lut, cold_valid.data(),
                   sizeof(float) * cold_valid.size());
    }
}

struct Ds4AuthoritativeRouteSnapshot {
    const DeepSeek4FusedDecodeGraph::AuthoritativeRouteOutput * route = nullptr;
    std::vector<int32_t> ids;
    std::vector<float> weights;
};

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
        MoeHybridRoutingStats * routing_stats) {
    const bool predict_audit =
        ds4_tp_route_predict_audit_lookahead() > 0;
    const bool cache_audit =
        ds4_env_flag("DFLASH_DS4_TP_CACHE_AUDIT");
    if (!routing_stats && !predict_audit && !cache_audit) return;

    std::vector<Ds4AuthoritativeRouteSnapshot> snapshots;
    snapshots.reserve(fg.authoritative_routes.size());
    uint64_t cache_hot = 0;
    uint64_t cache_total = 0;
    for (const auto & route : fg.authoritative_routes) {
        Ds4AuthoritativeRouteSnapshot snapshot;
        snapshot.route = &route;
        if (!ds4_fused_read_route_matrix(
                route.selected, GGML_TYPE_I32, route.width,
                route.n_tokens, snapshot.ids) ||
            !ds4_fused_read_route_matrix(
                route.weights, GGML_TYPE_F32, route.width,
                route.n_tokens, snapshot.weights)) {
            std::fprintf(stderr,
                "[ds4-route-audit] failed to read layer=%d lane=%d\n",
                route.layer, route.lane_start);
            continue;
        }
        for (int token = 0; token < route.n_tokens; ++token) {
            const int32_t * ids = snapshot.ids.data() +
                (size_t) token * route.width;
            const float * weights = snapshot.weights.data() +
                (size_t) token * route.width;
            observe_active_routing(
                routing_stats, route.layer, ids, weights, route.width);
            if (cache_audit && hybrid && route.layer >= 0 &&
                route.layer < (int) hybrid->layers.size()) {
                const MoeHybridLayerStorage & layer =
                    hybrid->layers[(size_t) route.layer];
                for (int i = 0; i < route.width; ++i) {
                    const int32_t id = ids[i];
                    if (weights[i] == 0.0f || id < 0 ||
                        id >= (int32_t) layer.hot_local_by_global.size()) {
                        continue;
                    }
                    ++cache_total;
                    if (layer.hot_local_by_global[(size_t) id] >= 0) {
                        ++cache_hot;
                    }
                }
            }
        }
        snapshots.push_back(std::move(snapshot));
    }

    if (cache_audit && cache_total > 0) {
        std::fprintf(stderr,
            "[ds4-route-cache-audit] hot=%llu total=%llu hit=%.3f\n",
            (unsigned long long) cache_hot,
            (unsigned long long) cache_total,
            (double) cache_hot / (double) cache_total);
    }
    if (!predict_audit) return;

    uint64_t active_total = 0;
    uint64_t base_hits = 0;
    uint64_t wide_hits = 0;
    uint64_t comparisons = 0;
    int base_width = 0;
    int wide_width = 0;
    for (const auto & prediction : fg.route_predictions) {
        const Ds4AuthoritativeRouteSnapshot * target = nullptr;
        for (const auto & snapshot : snapshots) {
            if (snapshot.route->layer == prediction.target_layer &&
                snapshot.route->lane_start == prediction.lane_start &&
                snapshot.route->n_tokens == prediction.n_tokens) {
                target = &snapshot;
                break;
            }
        }
        if (!target) continue;
        std::vector<int32_t> predicted;
        if (!ds4_fused_read_route_matrix(
                prediction.selected, GGML_TYPE_I32, prediction.width,
                prediction.n_tokens, predicted)) {
            std::fprintf(stderr,
                "[ds4-route-predict-audit] failed to read source=%d target=%d\n",
                prediction.source_layer, prediction.target_layer);
            continue;
        }
        const int primary = std::min(
            target->route->width, prediction.width);
        base_width = std::max(base_width, primary);
        wide_width = std::max(wide_width, prediction.width);
        ++comparisons;
        for (int token = 0; token < prediction.n_tokens; ++token) {
            const int32_t * actual = target->ids.data() +
                (size_t) token * target->route->width;
            const float * weights = target->weights.data() +
                (size_t) token * target->route->width;
            const int32_t * candidate = predicted.data() +
                (size_t) token * prediction.width;
            for (int i = 0; i < target->route->width; ++i) {
                if (weights[i] == 0.0f || actual[i] < 0) continue;
                ++active_total;
                bool in_base = false;
                bool in_wide = false;
                for (int j = 0; j < prediction.width; ++j) {
                    if (candidate[j] != actual[i]) continue;
                    in_wide = true;
                    if (j < primary) in_base = true;
                    break;
                }
                base_hits += in_base ? 1u : 0u;
                wide_hits += in_wide ? 1u : 0u;
            }
        }
    }
    if (active_total > 0) {
        std::fprintf(stderr,
            "[ds4-route-predict-audit] comparisons=%llu active=%llu "
            "recall@%d=%.3f recall@%d=%.3f\n",
            (unsigned long long) comparisons,
            (unsigned long long) active_total,
            base_width, (double) base_hits / (double) active_total,
            wide_width, (double) wide_hits / (double) active_total);
    }
}

// Each hybrid cache slot owns a multi-backend scheduler, and each scheduler's
// gallocr retains its per-backend scratch buffers until the slot is destroyed.
// Keeping all 12 shape variants alive can therefore exhaust the 32 GiB target
// GPU after only a few distinct speculative shapes.  Two variants fit on the
// R9700 while preserving the common q=3/q=4 reuse; allow experiments to lower
// this to one (or raise it deliberately) without affecting the single-backend
// verifier cache.
static size_t ds4_fused_verify_hybrid_slot_limit() {
    static const size_t limit = []() {
        const char * raw = std::getenv("DFLASH_DS4_TP_FUSED_CACHE_SLOTS");
        const long requested = raw ? std::strtol(raw, nullptr, 10) : 2;
        return (size_t) std::max<long>(
            1, std::min<long>(requested,
                              (long) fused_verify_graph_cache.slots.size()));
    }();
    return limit;
}

static size_t ds4_fused_capture_hybrid_slot_limit() {
    static const size_t limit = []() {
        const char * raw = std::getenv("DFLASH_DS4_TP_CAPTURE_CACHE_SLOTS");
        const long requested = raw ? std::strtol(raw, nullptr, 10) : 4;
        return (size_t) std::max<long>(
            1, std::min<long>(requested,
                              (long) fused_capture_graph_cache.slots.size()));
    }();
    return limit;
}

static bool ds4_tp_whole_step_graphs_enabled() {
    static const bool enabled =
        ds4_env_flag("DFLASH_DS4_TP_WHOLE_STEP_GRAPHS");
    return enabled;
}

// Replace the host's per-split launch sequence with one executable graph per
// GPU after two ordinary warmups.  All kernels, copies, and reduction order
// remain unchanged; only their submission boundary moves to the device.  The
// capture is rejected if the scheduler assigned any split to a non-CUDA
// backend because such work could not be replayed by the two device graphs.
static ggml_status ds4_fused_verify_compute_whole_step(
        DeepSeek4FusedDecodeGraph & fg, int q) {
    if (!fg.sched || q != 4 || !ds4_tp_whole_step_graphs_enabled()) {
        return fg.sched
            ? ggml_backend_sched_graph_compute(fg.sched, fg.sg.gf)
            : GGML_STATUS_FAILED;
    }

    std::array<ggml_backend_t, 2> gpu_backends{};
    int gpu_count = 0;
    int gpu_splits = 0;
    const int n_backends = ggml_backend_sched_get_n_backends(fg.sched);
    for (int i = 0; i < n_backends; ++i) {
        ggml_backend_t candidate =
            ggml_backend_sched_get_backend(fg.sched, i);
        const int splits = ggml_backend_sched_get_n_splits_for_backend(
            fg.sched, candidate);
        if (ggml_backend_is_cuda(candidate)) {
            if (gpu_count < (int) gpu_backends.size()) {
                gpu_backends[(size_t) gpu_count++] = candidate;
            }
            gpu_splits += splits;
        } else if (splits != 0) {
            static bool logged_cpu_split = false;
            if (!logged_cpu_split) {
                std::fprintf(stderr,
                    "[ds4-whole-graph] disabled: non-GPU backend has %d splits\n",
                    splits);
                logged_cpu_split = true;
            }
            return ggml_backend_sched_graph_compute(fg.sched, fg.sg.gf);
        }
    }
    if (gpu_count != 2 ||
        gpu_splits != ggml_backend_sched_get_n_splits(fg.sched)) {
        return ggml_backend_sched_graph_compute(fg.sched, fg.sg.gf);
    }

    if (fg.whole_step_graph_count == 2) {
        // Waits are captured compute kernels on this ROCm pair, not hardware
        // semaphore packets. Enqueue the R9700 publisher first so MES never
        // starts with an unsignaled Strix spin queue. R9700 dense work leaves
        // ample time to enqueue Strix before the first reverse dependency.
        for (int i = 0; i < 2; ++i) {
            if (!ggml_backend_cuda_whole_graph_launch(
                    fg.whole_step_backends[(size_t) i],
                    fg.whole_step_graphs[(size_t) i])) {
                return GGML_STATUS_FAILED;
            }
        }
        ggml_backend_sched_synchronize(fg.sched);
        return GGML_STATUS_SUCCESS;
    }

    if (fg.whole_step_warmups++ < 2) {
        return ggml_backend_sched_graph_compute(fg.sched, fg.sg.gf);
    }

    // Both streams must be capturing before the scheduler emits the first
    // cross-device event. Begin with the cold owner to match replay order.
    if (!ggml_backend_cuda_whole_graph_capture_prepare(
            gpu_backends[0], gpu_backends[1])) {
        std::fprintf(stderr,
            "[ds4-whole-graph] doorbell preparation failed; using scheduler\n");
        return ggml_backend_sched_graph_compute(fg.sched, fg.sg.gf);
    }
    std::fprintf(stderr, "[ds4-whole-graph] capture begin peer\n");
    const bool began_peer =
        ggml_backend_cuda_whole_graph_capture_begin(gpu_backends[1]);
    std::fprintf(stderr, "[ds4-whole-graph] capture begin main peer=%d\n",
                 (int) began_peer);
    const bool began_main = began_peer &&
        ggml_backend_cuda_whole_graph_capture_begin(gpu_backends[0]);
    if (!began_main) {
        if (began_peer) {
            auto abandoned =
                ggml_backend_cuda_whole_graph_capture_end(gpu_backends[1]);
            ggml_backend_cuda_whole_graph_free(
                gpu_backends[1], abandoned);
        }
        return ggml_backend_sched_graph_compute(fg.sched, fg.sg.gf);
    }

    ggml_backend_sched_set_whole_graph_capture(fg.sched, true);
    std::fprintf(stderr, "[ds4-whole-graph] scheduler capture submit\n");
    const ggml_status captured_status =
        ggml_backend_sched_graph_compute_async(fg.sched, fg.sg.gf);
    std::fprintf(stderr, "[ds4-whole-graph] scheduler capture status=%d\n",
                 (int) captured_status);
    ggml_backend_sched_set_whole_graph_capture(fg.sched, false);
    std::fprintf(stderr, "[ds4-whole-graph] capture end peer\n");
    auto peer_graph =
        ggml_backend_cuda_whole_graph_capture_end(gpu_backends[1]);
    std::fprintf(stderr, "[ds4-whole-graph] capture end main peer_graph=%p\n",
                 peer_graph);
    auto main_graph =
        ggml_backend_cuda_whole_graph_capture_end(gpu_backends[0]);
    if (captured_status != GGML_STATUS_SUCCESS ||
        !main_graph || !peer_graph) {
        ggml_backend_cuda_whole_graph_free(gpu_backends[0], main_graph);
        ggml_backend_cuda_whole_graph_free(gpu_backends[1], peer_graph);
        std::fprintf(stderr,
            "[ds4-whole-graph] capture failed status=%d main=%p peer=%p\n",
            (int) captured_status, main_graph, peer_graph);
        return captured_status == GGML_STATUS_SUCCESS
            ? GGML_STATUS_FAILED : captured_status;
    }

    fg.whole_step_backends = gpu_backends;
    fg.whole_step_graphs = {main_graph, peer_graph};
    fg.whole_step_graph_count = 2;
    std::fprintf(stderr,
        "[ds4-whole-graph] active q=4 splits=%d\n",
        ggml_backend_sched_get_n_splits(fg.sched));
    for (int i = 0; i < 2; ++i) {
        if (!ggml_backend_cuda_whole_graph_launch(
                fg.whole_step_backends[(size_t) i],
                fg.whole_step_graphs[(size_t) i])) {
            return GGML_STATUS_FAILED;
        }
    }
    ggml_backend_sched_synchronize(fg.sched);
    return GGML_STATUS_SUCCESS;
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
        DeepSeek4Cache & cache,
        const std::vector<HcLayerWeightsCpu> & hc_weights,
        const HcWeightsCpu & hc_out_weights,
        const std::vector<HashRoutingTableCpu> & hash_tables,
        int kv_start,
        int q,
        bool have_token_ids,
        const std::vector<int> & capture_ids,
        MoeHybridStorage * hybrid,
        std::vector<int64_t> && shape_key) {
    fg.clear_whole_step_graphs();
    if (fg.sched) {
        ggml_backend_sched_free(fg.sched);
        fg.sched = nullptr;
    }
    step_graph_free(fg.sg);
    fg.reset_nodes();
    ex.reset();
    const bool q2_wavefront =
        ds4_tp_q2_wavefront_active(q, kv_start, hybrid);
    fg.hash_ids.assign((size_t) w.n_layer, nullptr);
    if (hybrid) {
        fg.hybrid_inputs.assign(
            (size_t) w.n_layer * (q2_wavefront ? 2u : 1u), {});
    }
    const int n_embd = w.n_embd;
    const int n_hc = w.n_hc;

    const size_t arena_size =
        (q2_wavefront ? 512u : 256u) * 1024 * 1024;
    if (fg.sg.meta_arena.size() < arena_size) fg.sg.meta_arena.resize(arena_size);
    ggml_init_params params{};
    params.mem_size = fg.sg.meta_arena.size();
    params.mem_buffer = fg.sg.meta_arena.data();
    params.no_alloc = true;
    fg.sg.ctx = ggml_init(params);
    if (!fg.sg.ctx) return false;
    ggml_context * ctx = fg.sg.ctx;
    const size_t graph_capacity = q2_wavefront ? 131072u : 65536u;
    fg.sg.gf = ggml_new_graph_custom(ctx, graph_capacity, false);
    ggml_cgraph * gf = fg.sg.gf;

    fg.inp_embed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, q);
    ggml_set_input(fg.inp_embed);
    ex.q = q;
    ex.pos_q = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, q);  ggml_set_input(ex.pos_q);
    ex.neg_q = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, q);  ggml_set_input(ex.neg_q);
    ex.rawrows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, q); ggml_set_input(ex.rawrows);
    ex.ape4 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, q);   ggml_set_input(ex.ape4);
    ex.ape128 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, q); ggml_set_input(ex.ape128);
    ex.st4 = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, q); ggml_set_input(ex.st4);
    ex.st128 = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, q); ggml_set_input(ex.st128);
    // comp emission scalars per layer: i32 {ape_row(unused batched), comp_pos},
    // i64 {comp_row}; reuse the decode-style bundles (2 i32 + 1 i64 per layer).
    fg.i32_bundle = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2 * (int64_t) w.n_layer);
    ggml_set_input(fg.i32_bundle);
    fg.i64_bundle = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1 * (int64_t) w.n_layer);
    ggml_set_input(fg.i64_bundle);

    // Mask bundle: q>1 preserves the q ring rows that the batch overwrites,
    // so its attention span is [n_swa + padded + q].  q=1 reads the current
    // row through the in-graph ring update and has no preserved-row suffix,
    // matching the established single-token fused-decode topology.
    int64_t mask_total = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        const int ratio = (int) w.compress_ratios[il];
        const int n_lanes = q2_wavefront ? 2 : 1;
        for (int lane = 0; lane < n_lanes; ++lane) {
            const int lane_q = q2_wavefront ? 2 : q;
            const int lane_start = q2_wavefront ? 2 * lane : 0;
            const int lane_token_pos = kv_start + lane_start + lane_q - 1;
            int padded = 0;
            if (ratio > 0 && cache.layers[(size_t) il].comp_kv) {
                const int n_comp = ds4_comp_rows_used(
                    cache.layers[(size_t) il].comp_kv,
                    cache.layers[(size_t) il].n_comp, ratio,
                    lane_token_pos);
                padded = ds4_padded_comp_rows(
                    n_comp,
                    (int) cache.layers[(size_t) il].comp_kv->ne[1]);
            }
            const int preserved_rows = lane_q > 1 ? lane_q : 0;
            mask_total +=
                (int64_t) (w.n_swa + padded + preserved_rows) * lane_q;
        }
    }
    fg.mask_bundle = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mask_total);
    ggml_set_input(fg.mask_bundle);
    int64_t mask_off = 0;

    // per-token HC streams
    std::vector<ggml_tensor *> hc_cur(q);
    for (int t = 0; t < q; ++t) {
        ggml_tensor * col = ggml_view_2d(ctx, fg.inp_embed, n_embd, 1,
                                         fg.inp_embed->nb[1], (size_t) t * fg.inp_embed->nb[1]);
        hc_cur[(size_t) t] = ggml_repeat_4d(ctx, col, n_embd, n_hc, 1, 1);
    }

    std::vector<ggml_tensor *> capture_pieces(
        capture_ids.size() * (size_t) q, nullptr);   // order [ci][t]

    for (int il = 0; il < w.n_layer; ++il) {
        const DeepSeek4Layer & L = w.layers[(size_t) il];
        DeepSeek4LayerCache & lc = cache.layers[(size_t) il];
        const HcLayerWeightsCpu & hlw = hc_weights[(size_t) il];
        const int ratio = (int) w.compress_ratios[il];

        // In wavefront mode the two lane builds deliberately stay inside the
        // same layer loop.  The hybrid builder expands each lane's cold and
        // hot prefixes but leaves its deferred-copy join unexpanded.  Building
        // lane B therefore appends useful main-GPU work before lane A's join;
        // at the next outer iteration lane A's dependency pulls that join in
        // before layer il+1.  This produces the diagonal A/B pipeline without
        // changing transformer dependencies.
        const int n_lanes = q2_wavefront ? 2 : 1;
        for (int lane = 0; lane < n_lanes; ++lane) {
        const int lane_q = q2_wavefront ? 2 : q;
        const int lane_start = q2_wavefront ? 2 * lane : 0;
        const int lane_kv_start = kv_start + lane_start;
        const int lane_token_pos = lane_kv_start + lane_q - 1;
        const size_t hybrid_idx = q2_wavefront
            ? (size_t) il * 2 + (size_t) lane
            : (size_t) il;

        // ── HC pre (attention), per token ──
        std::vector<ggml_tensor *> hc_flat(lane_q), split_attn(lane_q);
        ggml_tensor * attn_in = nullptr;
        for (int t = 0; t < lane_q; ++t) {
            const size_t token_idx = (size_t) lane_start + (size_t) t;
            hc_flat[(size_t) t] = ggml_reshape_1d(
                ctx, hc_cur[token_idx], (int64_t) n_embd * n_hc);
            ggml_tensor * working = ds4_build_fused_hc_pre(ctx, w, hc_flat[(size_t) t],
                                                           mc.fn_attn_f16[(size_t) il], L.hc_attn_base,
                                                           hlw.attn, &split_attn[(size_t) t]);
            if (!working) return false;
            ggml_tensor * w2 = ggml_reshape_2d(ctx, working, n_embd, 1);
            attn_in = attn_in ? ggml_concat(ctx, attn_in, w2, 1) : w2;
        }

        // ── Batched attention ──
        DeepSeek4AttentionGraphInputs ain{};
        const size_t i32_lane_off =
            (size_t) lane_start * sizeof(int32_t);
        const size_t i64_lane_off =
            (size_t) lane_start * sizeof(int64_t);
        ain.rope_pos = q2_wavefront
            ? ggml_view_1d(ctx, ex.pos_q, lane_q, i32_lane_off)
            : ex.pos_q;
        ain.neg_pos = q2_wavefront
            ? ggml_view_1d(ctx, ex.neg_q, lane_q, i32_lane_off)
            : ex.neg_q;
        ain.raw_kv_rows = q2_wavefront
            ? ggml_view_1d(ctx, ex.rawrows, lane_q, i64_lane_off)
            : ex.rawrows;
        if (ratio > 0) {
            ggml_tensor * ape_all = (ratio == 4) ? ex.ape4 : ex.ape128;
            ggml_tensor * state_all = (ratio == 4) ? ex.st4 : ex.st128;
            ain.attn_ape_row = q2_wavefront
                ? ggml_view_1d(ctx, ape_all, lane_q, i32_lane_off)
                : ape_all;
            ain.attn_state_rows = q2_wavefront
                ? ggml_view_1d(ctx, state_all, lane_q, i64_lane_off)
                : state_all;
            ain.attn_comp_pos = ggml_view_1d(ctx, fg.i32_bundle, 1, ((size_t) il * 2 + 1) * sizeof(int32_t));
            ain.attn_comp_rows = ggml_view_2d(ctx, fg.i64_bundle, 1, 1, sizeof(int64_t),
                                              (size_t) il * sizeof(int64_t));
        }
        if (ratio == 4) {
            ain.index_ape_row = q2_wavefront
                ? ggml_view_1d(ctx, ex.ape4, lane_q, i32_lane_off)
                : ex.ape4;
            ain.index_state_rows = q2_wavefront
                ? ggml_view_1d(ctx, ex.st4, lane_q, i64_lane_off)
                : ex.st4;
            ain.index_comp_pos = ggml_view_1d(ctx, fg.i32_bundle, 1, ((size_t) il * 2 + 1) * sizeof(int32_t));
            ain.index_comp_rows = ggml_view_2d(ctx, fg.i64_bundle, 1, 1, sizeof(int64_t),
                                               (size_t) il * sizeof(int64_t));
        }
        int padded = 0;
        if (ratio > 0 && lc.comp_kv) {
            const int n_comp = ds4_comp_rows_used(
                lc.comp_kv, lc.n_comp, ratio, lane_token_pos);
            padded = ds4_padded_comp_rows(n_comp, (int) lc.comp_kv->ne[1]);
        }
        const int64_t n_attn = (int64_t) w.n_swa + padded +
                               (lane_q > 1 ? lane_q : 0);
        ain.attn_row_mask = ggml_view_2d(ctx, fg.mask_bundle, n_attn, lane_q,
                                         n_attn * sizeof(float), (size_t) mask_off * sizeof(float));
        ain.padded_comp = padded;
        mask_off += n_attn * lane_q;

        std::vector<DeepSeek4I32InputBinding> i32b;
        std::vector<DeepSeek4I32ArrayBinding> i32ab;
        std::vector<DeepSeek4I64ArrayBinding> i64ab;
        ggml_tensor * normed = build_rms_norm(ctx, attn_in, L.attn_norm, w.rms_eps);
        ggml_tensor * attn_out = build_mla_attention(ctx, gf, normed, w, L, lc, il,
                                                     lane_kv_start, lane_q, &ain,
                                                     i32b, i32ab, i64ab);
        if (!attn_out) return false;
        if (!i32b.empty() || !i32ab.empty() || !i64ab.empty()) {
            std::fprintf(stderr, "[ds4-fused-verify] layer %d dynamic bindings; cannot fuse\n", il);
            return false;
        }

        // ── HC post (attention) + HC pre (FFN), per token ──
        ggml_tensor * ffn_in = nullptr;
        std::vector<ggml_tensor *> split_ffn(lane_q);
        for (int t = 0; t < lane_q; ++t) {
            const size_t token_idx = (size_t) lane_start + (size_t) t;
            ggml_tensor * ao = ggml_view_2d(ctx, attn_out, n_embd, 1,
                                            attn_out->nb[1], (size_t) t * attn_out->nb[1]);
            ggml_tensor * ao_flat = ggml_reshape_1d(ctx, ggml_cont(ctx, ao), n_embd);
            ggml_tensor * hc_next = ggml_ds4_hc_post(ctx, hc_flat[(size_t) t], ao_flat,
                                                     split_attn[(size_t) t], n_hc);
            hc_cur[token_idx] = ggml_reshape_2d(
                ctx, hc_next, n_embd, n_hc);
            hc_flat[(size_t) t] = ggml_reshape_1d(
                ctx, hc_cur[token_idx], (int64_t) n_embd * n_hc);
            ggml_tensor * fworking = ds4_build_fused_hc_pre(ctx, w, hc_flat[(size_t) t],
                                                            mc.fn_ffn_f16[(size_t) il], L.hc_ffn_base,
                                                            hlw.ffn, &split_ffn[(size_t) t]);
            if (!fworking) return false;
            ggml_tensor * f2 = ggml_reshape_2d(ctx, fworking, n_embd, 1);
            ffn_in = ffn_in ? ggml_concat(ctx, ffn_in, f2, 1) : f2;
        }

        // ── Batched FFN ──
        ggml_tensor * ffn_normed = build_rms_norm(ctx, ffn_in, L.ffn_norm, w.rms_eps);
        ggml_tensor * ffn_out = nullptr;
        const MoeHybridGraphInputs * fused_hc_join_inputs = nullptr;
        const bool hash_routed = il < w.n_hash_layer && L.ffn_gate_tid2eid &&
                                 have_token_ids && hash_tables[(size_t) il].loaded;
        ggml_tensor * selected = nullptr;
        ggml_tensor * router_weights = nullptr;
        std::vector<ggml_tensor *> router_nodes;
        if (hash_routed) {
            const int route_width = ds4_effective_expert_count(w);
            // Keep one q-wide input for host filling and give each wavefront
            // lane a zero-copy two-column view.
            ggml_tensor * hids_all = fg.hash_ids[(size_t) il];
            if (!hids_all) {
                hids_all = ggml_new_tensor_2d(
                    ctx, GGML_TYPE_I32, route_width, q);
                ggml_set_input(hids_all);
                fg.hash_ids[(size_t) il] = hids_all;
            }
            selected = q2_wavefront
                ? ggml_view_2d(
                    ctx, hids_all, route_width, lane_q,
                    hids_all->nb[1],
                    (size_t) lane_start * hids_all->nb[1])
                : hids_all;
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
                    ggml_tensor * fo = ds4_build_hash_routed_ffn(ctx, w, L, ggml_cont(ctx, fcol), hcol);
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
        } else {
            ffn_out = build_moe_ffn(
                ctx, ffn_normed, w, L, il, lane_q);
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

        // Diagnostic branch only: predict future score-routed expert IDs from
        // the current normalized FFN input.  The authoritative routing above
        // remains the sole input to the model FFN, so this cannot affect
        // logits.  q2 wavefront is excluded to keep the audit's source/target
        // correspondence unambiguous.
        const int predict_ahead = ds4_tp_route_predict_audit_lookahead();
        if (hybrid && q > 1 && predict_ahead > 0 && !q2_wavefront) {
            for (int ahead = 1; ahead <= predict_ahead; ++ahead) {
                const int target_layer = il + ahead;
                if (target_layer >= w.n_layer ||
                    target_layer < w.n_hash_layer) {
                    continue;
                }
                const DeepSeek4Layer & future =
                    w.layers[(size_t) target_layer];
                if (!future.ffn_gate_inp) continue;
                Ds4MoeRoutePrediction prediction =
                    build_moe_route_prediction(
                        ctx, ffn_normed, w, future,
                        ds4_tp_route_predict_audit_width());
                if (!prediction.selected) return false;
                ggml_set_output(prediction.selected);
                ggml_build_forward_expand(gf, prediction.selected);
                DeepSeek4FusedDecodeGraph::RoutePredictionOutput saved;
                saved.source_layer = il;
                saved.target_layer = target_layer;
                saved.lane_start = lane_start;
                saved.n_tokens = lane_q;
                saved.width = (int) prediction.selected->ne[0];
                saved.selected = prediction.selected;
                saved.nodes = std::move(prediction.nodes);
                fg.route_predictions.push_back(std::move(saved));
            }
        }
        if (hybrid) {
            const bool allow_fused_combine =
                ggml_backend_is_cuda(backend) &&
                ggml_backend_is_cuda(hybrid->cold_backend);
            MoeHybridConfig hybrid_cfg = make_ds4_moe_hybrid_config(w);
            hybrid_cfg.n_expert_used = (int) selected->ne[0];
            MoeLayerDesc desc = make_ds4_moe_layer_desc(L);
            MoeHybridGraphInputs & inputs = fg.hybrid_inputs[hybrid_idx];
            inputs.router_nodes = router_nodes;
            const MoeHybridLayerStorage & hybrid_layer =
                hybrid->layers[(size_t) il];
            const bool has_hot_routed =
                (hybrid_layer.gate_up_hot ||
                 (hybrid_layer.gate_hot && hybrid_layer.up_hot)) &&
                hybrid_layer.down_hot;
            // Legacy q>1 graphs represented six routes as 4 + (2 padded to
            // 4). Native q1 decode already uses width six directly. Keep the
            // decomposition as a bisectable fallback while validating the
            // dedicated q4 MoE kernel's native route width on both GPUs.
            // The dedicated multi-token MUL_MAT_ID kernel is indexed by token
            // width, not route width.  Once that kernel is enabled, keep all
            // six routes in one owner graph: the legacy 4 + padded-2 split
            // computes two duplicate expert paths whose weights are zero.
            const bool native_route_width =
                ds4_env_flag("DFLASH_DS4_TP_NATIVE_ROUTE_WIDTH");
            if (lane_q > 1 && hybrid_cfg.n_expert_used > 4 && has_hot_routed &&
                !native_route_width) {
                const int route_width = hybrid_cfg.n_expert_used;
                if (route_width != 6) return false;
                hybrid_cfg.n_expert_used = 4;

                ggml_tensor * first_ids = nullptr;
                ggml_tensor * first_weights = nullptr;
                ggml_tensor * padded_ids = nullptr;
                ggml_tensor * padded_weights = nullptr;
                for (int t = 0; t < lane_q; ++t) {
                    const size_t ids_row = (size_t) t * selected->nb[1];
                    const size_t weights_row = (size_t) t * router_weights->nb[1];

                    ggml_tensor * ids_head = ggml_cont(ctx, ggml_view_1d(
                        ctx, selected, 4, ids_row));
                    ggml_tensor * weights_head = ggml_cont(ctx, ggml_view_1d(
                        ctx, router_weights, 4, weights_row));
                    first_ids = first_ids
                        ? ggml_concat(ctx, first_ids, ids_head, 1)
                        : ids_head;
                    first_weights = first_weights
                        ? ggml_concat(ctx, first_weights, weights_head, 1)
                        : weights_head;

                    ggml_tensor * ids_tail = ggml_cont(ctx, ggml_view_1d(
                        ctx, selected, 2, ids_row + 4 * selected->nb[0]));
                    ggml_tensor * weights_tail = ggml_cont(ctx, ggml_view_1d(
                        ctx, router_weights, 2,
                        weights_row + 4 * router_weights->nb[0]));
                    ggml_tensor * ids_tail_padded = ggml_concat(
                        ctx, ids_tail, ids_tail, 0);
                    ggml_tensor * weights_tail_padded = ggml_concat(
                        ctx, weights_tail, ggml_scale(ctx, weights_tail, 0.0f), 0);
                    padded_ids = padded_ids
                        ? ggml_concat(ctx, padded_ids, ids_tail_padded, 1)
                        : ids_tail_padded;
                    padded_weights = padded_weights
                        ? ggml_concat(ctx, padded_weights, weights_tail_padded, 1)
                        : weights_tail_padded;
                }
                if (!build_moe_hybrid_ffn_graph(
                        ctx,
                        (ds4_env_flag("DFLASH_DS4_TP_PEER_FENCE") ||
                         ds4_env_flag("DFLASH_DS4_TP_DEVICE_JOIN")) ? gf : nullptr,
                        hybrid_cfg, desc,
                        hybrid->layers[(size_t) il], ffn_normed,
                        first_ids, first_weights, lane_q, inputs, true,
                        allow_fused_combine)) {
                    return false;
                }
                ffn_out = inputs.output;

                if (!build_moe_hybrid_ffn_graph(
                        ctx,
                        (ds4_env_flag("DFLASH_DS4_TP_PEER_FENCE") ||
                         ds4_env_flag("DFLASH_DS4_TP_DEVICE_JOIN")) ? gf : nullptr,
                        hybrid_cfg, desc,
                        hybrid->layers[(size_t) il], ffn_normed,
                        padded_ids, padded_weights, lane_q, inputs, false,
                        allow_fused_combine)) {
                    return false;
                }
                ffn_out = ggml_add(ctx, ffn_out, inputs.output);
            } else {
                if (!build_moe_hybrid_ffn_graph(
                        ctx,
                        (ds4_env_flag("DFLASH_DS4_TP_PEER_FENCE") ||
                         ds4_env_flag("DFLASH_DS4_TP_DEVICE_JOIN")) ? gf : nullptr,
                        hybrid_cfg, desc,
                        hybrid->layers[(size_t) il],
                        ffn_normed, selected, router_weights,
                        lane_q, inputs, true, allow_fused_combine)) {
                    return false;
                }
                ffn_out = inputs.output;
                if (lane_q > 1 && native_route_width &&
                    ds4_env_flag("DFLASH_DS4_TP_FUSED_HC_JOIN") &&
                    inputs.main_output && inputs.peer_output) {
                    fused_hc_join_inputs = &inputs;
                }
            }
        }
        if (!ffn_out) return false;

        // ── HC post (FFN), per token; capture at drafter layers ──
        for (int t = 0; t < lane_q; ++t) {
            const size_t token_idx = (size_t) lane_start + (size_t) t;
            ggml_tensor * hc_next = nullptr;
            if (fused_hc_join_inputs) {
                ggml_tensor * main_part = ggml_view_2d(
                    ctx, fused_hc_join_inputs->main_output, n_embd, 1,
                    fused_hc_join_inputs->main_output->nb[1],
                    (size_t) t * fused_hc_join_inputs->main_output->nb[1]);
                ggml_tensor * peer_part = ggml_view_2d(
                    ctx, fused_hc_join_inputs->peer_output, n_embd, 1,
                    fused_hc_join_inputs->peer_output->nb[1],
                    (size_t) t * fused_hc_join_inputs->peer_output->nb[1]);
                hc_next = ggml_ds4_hc_post_split(
                    ctx, hc_flat[(size_t) t], main_part, peer_part,
                    split_ffn[(size_t) t], n_hc);
            } else {
                ggml_tensor * fo = ggml_view_2d(
                    ctx, ffn_out, n_embd, 1, ffn_out->nb[1],
                    (size_t) t * ffn_out->nb[1]);
                ggml_tensor * fo_flat = ggml_reshape_1d(
                    ctx, ggml_cont(ctx, fo), n_embd);
                hc_next = ggml_ds4_hc_post(
                    ctx, hc_flat[(size_t) t], fo_flat,
                    split_ffn[(size_t) t], n_hc);
            }
            hc_cur[token_idx] = ggml_reshape_2d(
                ctx, hc_next, n_embd, n_hc);
        }
        for (size_t ci = 0; ci < capture_ids.size(); ++ci) {
            if (capture_ids[ci] != il) continue;
            for (int t = 0; t < lane_q; ++t) {
                const size_t token_idx =
                    (size_t) lane_start + (size_t) t;
                ggml_tensor * hs = hc_cur[token_idx];             // [n_embd, n_hc]
                ggml_tensor * hsT = ggml_cont(ctx, ggml_transpose(ctx, hs)); // [n_hc, n_embd]
                ggml_tensor * summed = ggml_sum_rows(ctx, hsT);   // [1, n_embd]
                ggml_tensor * mean = ggml_scale(ctx, summed, 1.0f / (float) n_hc);
                capture_pieces[ci * (size_t) q + (size_t) t] =
                    ggml_reshape_1d(ctx, ggml_cont(ctx, mean), n_embd);
            }
        }
        } // lane
    }

    // ── Output: per-token HC merge → batched out_norm + lm_head ──
    ggml_tensor * final_all = nullptr;
    for (int t = 0; t < q; ++t) {
        ggml_tensor * hc_flat = ggml_reshape_1d(ctx, hc_cur[(size_t) t], (int64_t) n_embd * n_hc);
        ggml_tensor * onorm = ggml_rms_norm(ctx, hc_flat, w.hc_eps);
        ggml_tensor * omix = ggml_mul_mat(ctx, mc.fn_out_f16, onorm);
        omix = ggml_reshape_1d(ctx, omix, ggml_nelements(omix));
        ggml_tensor * obase = ds4_fused_hc_base_f32(ctx, w.output_hc_base);
        if (!obase || hc_out_weights.scale_data.empty()) return false;
        ggml_tensor * fe = ggml_ds4_hc_out(ctx, omix, obase, hc_flat, n_hc,
                                           hc_out_weights.scale_data[0]);
        ggml_tensor * fe2 = ggml_reshape_2d(ctx, fe, n_embd, 1);
        final_all = final_all ? ggml_concat(ctx, final_all, fe2, 1) : fe2;
    }
    ggml_tensor * out_normed = build_rms_norm(ctx, final_all, w.out_norm, w.rms_eps);
    fg.logits = ggml_mul_mat(ctx, w.output, out_normed);   // [n_vocab, q]
    ggml_set_output(fg.logits);
    ggml_build_forward_expand(gf, fg.logits);
    if (ds4_env_flag("DFLASH_DS4_GPU_ARGMAX_VERIFY")) {
        ex.argmax = ggml_argmax(ctx, fg.logits);
        ggml_set_output(ex.argmax);
        ggml_build_forward_expand(gf, ex.argmax);
    }

    if (!capture_pieces.empty()) {
        for (ggml_tensor * piece : capture_pieces) {
            if (!piece) {
                std::fprintf(stderr, "[ds4-fused-verify] capture layer id is invalid\n");
                return false;
            }
        }
        ggml_tensor * cap = capture_pieces[0];
        for (size_t i = 1; i < capture_pieces.size(); ++i) {
            cap = ggml_concat(ctx, cap, capture_pieces[i], 0);
        }
        ex.capture = cap;
        ggml_set_output(ex.capture);
        ggml_build_forward_expand(gf, ex.capture);
    }

    // One-shot graph inventory for profiling residual single-token FP4 work.
    // This is deliberately build-time only: it does not install a scheduler
    // callback or synchronize kernels, so enabling it cannot distort the
    // execution timeline being diagnosed.
    if (q == 4 && ds4_env_flag("DFLASH_DS4_TP_Q4_INVENTORY")) {
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
        ggml_backend_t backends[3] = {
            backend, peer, hybrid->cpu_backend};
        fg.sched = ggml_backend_sched_new(
            backends, nullptr, 3, graph_capacity, false, true);
        if (!fg.sched) {
            std::fprintf(stderr,
                         "[ds4-fused-verify] scheduler creation failed\n");
            return false;
        }
        const bool late_join_split =
            ds4_env_flag("DFLASH_DS4_TP_LATE_JOIN_SPLIT");
        const bool targeted_join_split =
            ds4_env_flag("DFLASH_DS4_TP_TARGETED_JOIN_SPLIT");
        const bool gpu_i32_repeat =
            ds4_env_flag("LUCE_CUDA_I32_REPEAT");
        const bool report_split_count =
            ds4_env_flag("DFLASH_DS4_TP_SPLIT_COUNT");
        ggml_backend_sched_set_late_cross_input_split(
            fg.sched, late_join_split);
        if (ds4_env_flag("DFLASH_DS4_TP_SCHED_TRACE")) {
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
        pin_main(fg.inp_embed);
        pin_main(ex.pos_q);
        pin_main(ex.neg_q);
        pin_main(ex.rawrows);
        pin_main(ex.ape4);
        pin_main(ex.ape128);
        pin_main(ex.st4);
        pin_main(ex.st128);
        pin_main(fg.i32_bundle);
        pin_main(fg.i64_bundle);
        pin_main(fg.mask_bundle);
        for (ggml_tensor * hids : fg.hash_ids) pin_main(hids);
        for (const MoeHybridGraphInputs & inputs : fg.hybrid_inputs) {
            if (ds4_env_flag("DFLASH_DS4_TP_MAIN_ROUTE_WEIGHTS")) {
                for (ggml_tensor * node : inputs.router_nodes) {
                    pin_main(node);
                }
                pin_main(inputs.router_weights);
            }
            if (ds4_env_flag("DFLASH_DS4_TP_ROUTE_PREFORK")) {
                for (ggml_tensor * node : inputs.route_prefork_nodes) {
                    pin_main(node);
                }
            }
            pin_main(inputs.hot_local_lut);
            pin_main(inputs.hot_valid_lut);
            pin_main(inputs.shard_local_lut);
            pin_main(inputs.shard_valid_lut);
            // q1 has no repeat.  The opt-in exact I32 GPU repeat lets q>1 keep
            // both the LUT and its repeated output on the cold owner as well.
            if (q == 1 || gpu_i32_repeat) {
                pin_peer(inputs.cold_local_lut);
            } else {
                pin_main(inputs.cold_local_lut);
            }
            pin_peer(inputs.cold_valid_lut);
            for (ggml_tensor * node : inputs.hot_remap_nodes) {
                pin_main(node);
            }
            for (ggml_tensor * node : inputs.cold_remap_nodes) {
                pin_peer(node);
            }
            for (ggml_tensor * node : inputs.shard_remap_nodes) {
                pin_main(node);
            }
            for (ggml_tensor * node : inputs.hot_nodes) {
                pin_main(node);
            }
            for (ggml_tensor * node : inputs.cold_nodes) {
                pin_peer(node);
            }
            for (ggml_tensor * node : inputs.shard_nodes) {
                pin_main(node);
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
        for (const auto & prediction : fg.route_predictions) {
            for (ggml_tensor * node : prediction.nodes) pin_main(node);
            pin_main(prediction.selected);
        }
        for (const auto & route : fg.authoritative_routes) {
            pin_main(route.selected);
            pin_main(route.weights);
        }
        if (ds4_env_flag("DFLASH_DS4_TP_ROUTE_STATS") ||
            ds4_env_flag("DFLASH_DS4_ROUTING_STATS_OUT") ||
            ds4_env_flag("DFLASH_DS4_TP_DYNAMIC_HOTSET") ||
            (q > 1 && ds4_tp_route_predict_audit_lookahead() > 0) ||
            ds4_env_flag("DFLASH_DS4_TP_CACHE_AUDIT")) {
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

        ds4_fused_verify_refresh_hybrid_luts(
            w, hybrid, fg, q2_wavefront);
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
    fg.shape_key = std::move(shape_key);
    return true;
}

static int ds4_try_fused_verify_step(
        Ds4FusedVerifyCache & vc,
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
    if (!hc_out_weights.loaded || hc_out_weights.scale_data.empty() ||
        !w.output_hc_fn || !w.output_hc_base) { vc.disabled = true; return 0; }
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
    const int token_pos = kv_start + q - 1;
    const bool q2_wavefront =
        ds4_tp_q2_wavefront_active(q, kv_start, hybrid);
    std::vector<int64_t> key;
    key.reserve((size_t) w.n_layer + hooks->capture_layer_ids->size() + 6);
    key.push_back(w.n_swa);
    key.push_back(q);
    key.push_back(token_ids ? 1 : 0);
    key.push_back(hybrid ? 1 : 0);
    key.push_back(q2_wavefront ? 1 : 0);
    key.push_back((int64_t) hooks->capture_layer_ids->size());
    for (int layer : *hooks->capture_layer_ids) key.push_back(layer);
    for (int il = 0; il < w.n_layer; ++il) {
        const int ratio = (int) w.compress_ratios[il];
        DeepSeek4LayerCache & lc = cache.layers[(size_t) il];
        int padded = 0;
        if (ratio > 0 && lc.comp_kv) {
            const int n_comp = ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio, token_pos);
            padded = ds4_padded_comp_rows(n_comp, (int) lc.comp_kv->ne[1]);
        }
        int b_idx = -1;
        if (ratio > 0) {
            for (int t = 0; t < q; ++t) {
                if (((kv_start + t + 1) % ratio) == 0) { b_idx = t; break; }
            }
        }
        if (q2_wavefront) {
            int padded_a = 0;
            if (ratio > 0 && lc.comp_kv) {
                const int n_comp_a = ds4_comp_rows_used(
                    lc.comp_kv, lc.n_comp, ratio, kv_start + 1);
                padded_a = ds4_padded_comp_rows(
                    n_comp_a, (int) lc.comp_kv->ne[1]);
            }
            key.push_back(
                ((int64_t) padded_a << 32) |
                ((int64_t) padded << 8) |
                (int64_t) (b_idx + 1));
        } else {
            key.push_back(
                ((int64_t) padded << 4) | (int64_t) (b_idx + 1));
        }
    }

    vc.counter++;
    const bool capture_cache = &vc == &fused_capture_graph_cache;
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
        fg->destroy();
        ex->reset();

        const auto build_t0 = Ds4TimingClock::now();
        if (!ds4_build_fused_verify_graph(mc, *fg, *ex, backend, w, cache,
                                          hc_weights, hc_out_weights, hash_tables,
                                          kv_start, q, token_ids != nullptr,
                                          *hooks->capture_layer_ids, hybrid,
                                          std::move(key))) {
            fg->destroy();
            ex->reset();
            vc.disabled = true;
            return 0;
        }
        if (telemetry) telemetry->full_graph_build_us += ds4_elapsed_us(build_t0, Ds4TimingClock::now());
    }
    fg->last_use = vc.counter;

    // A fresh request clears KV/HC/compressor storage in place. Kernel
    // addresses remain stable, but a whole-step executable also retains its
    // captured cross-device dependency state. Never carry that executable
    // across the cache-clear boundary; recapture after the normal two q4
    // warmups and continue replaying for the rest of this request.
    if (q == 4 && ds4_tp_whole_step_graphs_enabled() &&
        fg->whole_step_sequence_id != cache.sequence_id) {
        if (fg->whole_step_graph_count != 0) {
            std::fprintf(stderr,
                "[ds4-whole-graph] request boundary old=%llu new=%llu; recapturing\n",
                (unsigned long long) fg->whole_step_sequence_id,
                (unsigned long long) cache.sequence_id);
        }
        fg->clear_whole_step_graphs();
        fg->whole_step_sequence_id = cache.sequence_id;
    }

    // ── Fill inputs ──
    const auto set_t0 = Ds4TimingClock::now();
    ds4_fv_set(fg->inp_embed, embed, sizeof(float) * (size_t) w.n_embd * q);
    std::vector<int32_t> iv(q);
    std::vector<int64_t> lv(q);
    for (int i = 0; i < q; ++i) iv[(size_t) i] = kv_start + i;
    ds4_fv_set(ex->pos_q, iv.data(), sizeof(int32_t) * q);
    for (int i = 0; i < q; ++i) iv[(size_t) i] = -(kv_start + i);
    ds4_fv_set(ex->neg_q, iv.data(), sizeof(int32_t) * q);
    for (int i = 0; i < q; ++i) lv[(size_t) i] = (kv_start + i) % w.n_swa;
    ds4_fv_set(ex->rawrows, lv.data(), sizeof(int64_t) * q);
    for (int i = 0; i < q; ++i) iv[(size_t) i] = (kv_start + i) % 4;
    ds4_fv_set(ex->ape4, iv.data(), sizeof(int32_t) * q);
    for (int i = 0; i < q; ++i) lv[(size_t) i] = 4 + (kv_start + i) % 4;
    ds4_fv_set(ex->st4, lv.data(), sizeof(int64_t) * q);
    for (int i = 0; i < q; ++i) iv[(size_t) i] = (kv_start + i) % 128;
    ds4_fv_set(ex->ape128, iv.data(), sizeof(int32_t) * q);
    for (int i = 0; i < q; ++i) lv[(size_t) i] = (kv_start + i) % 128;
    ds4_fv_set(ex->st128, lv.data(), sizeof(int64_t) * q);

    std::vector<int32_t> i32v((size_t) w.n_layer * 2, 0);
    std::vector<int64_t> i64v((size_t) w.n_layer * 1, 0);
    for (int il = 0; il < w.n_layer; ++il) {
        const int ratio = (int) w.compress_ratios[il];
        if (ratio > 0) {
            int pos_b = -1;   // boundary position inside the batch (if any)
            for (int t = 0; t < q; ++t) {
                if (((kv_start + t + 1) % ratio) == 0) { pos_b = kv_start + t; break; }
            }
            i32v[(size_t) il * 2 + 0] = token_pos % ratio;
            i32v[(size_t) il * 2 + 1] = (pos_b >= 0 ? pos_b : token_pos) + 1 - ratio;
            i64v[(size_t) il] = (pos_b >= 0 ? pos_b : token_pos) / ratio;
        }
    }
    ds4_fv_set(fg->i32_bundle, i32v.data(), sizeof(int32_t) * i32v.size());
    ds4_fv_set(fg->i64_bundle, i64v.data(), sizeof(int64_t) * i64v.size());

    // causal mask values
    {
        std::vector<float> maskv((size_t) ggml_nelements(fg->mask_bundle), 0.0f);
        size_t off = 0;
        for (int il = 0; il < w.n_layer; ++il) {
            const int ratio = (int) w.compress_ratios[il];
            DeepSeek4LayerCache & lc = cache.layers[(size_t) il];
            const int n_lanes = q2_wavefront ? 2 : 1;
            for (int lane = 0; lane < n_lanes; ++lane) {
            const int lane_q = q2_wavefront ? 2 : q;
            const int lane_start = q2_wavefront ? 2 * lane : 0;
            const int lane_kv_start = kv_start + lane_start;
            const int lane_token_pos = lane_kv_start + lane_q - 1;
            // At lane A's attention only its two writes have happened.  At
            // lane B, e covers all four writes.  This makes the physical-ring
            // mask and the preserved overwritten-row suffix equivalent to the
            // original q4 causal graph.
            const int e = lane_kv_start + lane_q;
            int padded = 0;
            if (ratio > 0 && lc.comp_kv) {
                const int n_comp = ds4_comp_rows_used(
                    lc.comp_kv, lc.n_comp, ratio, lane_token_pos);
                padded = ds4_padded_comp_rows(n_comp, (int) lc.comp_kv->ne[1]);
            }
            const size_t n_attn = (size_t) w.n_swa + padded +
                                  (lane_q > 1 ? (size_t) lane_q : 0u);
            for (int i = 0; i < lane_q; ++i) {
                float * col = maskv.data() + off + (size_t) i * n_attn;
                const int pos_i = lane_kv_start + i;
                for (int r = 0; r < w.n_swa; ++r) {
                    // position held by slot r AFTER this batch's ring writes
                    const int p_r = (e <= w.n_swa) ? (r < e ? r : -1)
                                  : (e - 1) - ((e - 1 - r) % w.n_swa);
                    if (p_r < 0 || p_r > pos_i) col[r] = -1.0e30f;
                }
                const int vis = (ratio > 0 && lc.comp_kv)
                              ? ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio, pos_i) : 0;
                for (int c = 0; c < padded; ++c) {
                    if (c >= vis) col[(size_t) w.n_swa + c] = -1.0e30f;
                }
                if (lane_q > 1) {
                    for (int j = 0; j < lane_q; ++j) {
                        const bool visible =
                            (j > i) &&
                            (lane_kv_start + j >= w.n_swa);
                        if (!visible) {
                            col[(size_t) w.n_swa + padded + j] = -1.0e30f;
                        }
                    }
                }
            }
            off += n_attn * lane_q;
            } // lane
        }
        GGML_ASSERT(off == maskv.size());
        ds4_fv_set(fg->mask_bundle, maskv.data(), sizeof(float) * maskv.size());
    }

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
            ggml_backend_tensor_set(hids, hash_scratch.data(), 0,
                                    sizeof(int32_t) * (size_t) n_used * q);
        }
    }
    if (telemetry) {
        telemetry->full_graph_set_us += ds4_elapsed_us(set_t0, Ds4TimingClock::now());
    }

    // A graph can be reused across requests after the fixed hot tensor stacks
    // have been repopulated.  The weights keep stable addresses; only these
    // tiny map inputs must follow the new request-adaptive placement.
    if (hybrid && ds4_env_flag("DFLASH_DS4_TP_DYNAMIC_HOTSET")) {
        ds4_fused_verify_refresh_hybrid_luts(
            w, hybrid, *fg, ds4_tp_q2_wavefront_active(q, kv_start, hybrid));
    }

    // ── Compute + read back ──
    const auto compute_t0 = Ds4TimingClock::now();

    // A q=1 feature-capture graph contains HC width-4 matmuls even though it
    // advances only one token. On gfx1151, selecting MMVQ for those width-4
    // nodes is correct in eager mode but its HIP graph replay faults against
    // unified-memory expert mappings. Keep the proven width-3 ceiling only
    // while q=1 capture graphs are evaluated/captured. q>=2 verification uses
    // the process ceiling (LUCE_MMVQ_MAX_NCOLS=4 in the optimized launch).
    const bool capture_eager = capture_cache &&
        ds4_env_flag("DFLASH_DS4_TP_CAPTURE_EAGER");
    if (capture_cache) {
        ggml_backend_cuda_set_mmvq_max_ncols_override(3);
    }
    if (capture_eager) {
        ggml_backend_cuda_set_graphs_disabled_override(true);
    }
    // q>=2 verifier graph slots own fixed tensor storage. Position, mask,
    // routing-map, and embedding values are refreshed through stable input
    // buffers above; the graph topology and all allocation addresses remain
    // immutable until the slot is destroyed. The generic CUDA/HIP property
    // scan can nevertheless observe backend-mutated metadata and reset replay
    // at request boundaries. Mirror the qualified draft fast path behind an
    // independent opt-in while retaining normal warmup/capture semantics.
    const bool force_verify_graph_replay = q > 1 &&
        ds4_env_flag("DFLASH_DS4_VERIFY_FORCE_GRAPH_REPLAY");
    if (force_verify_graph_replay) {
        ggml_backend_cuda_set_skip_props_check(true);
    }
    const enum ggml_status compute_status = fg->sched
        ? ds4_fused_verify_compute_whole_step(*fg, q)
        : ggml_backend_graph_compute(backend, fg->sg.gf);
    if (force_verify_graph_replay) {
        ggml_backend_cuda_set_skip_props_check(false);
    }
    if (capture_eager) {
        ggml_backend_cuda_set_graphs_disabled_override(false);
    }
    if (capture_cache) {
        ggml_backend_cuda_set_mmvq_max_ncols_override(0);
    }
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
    if (argmax_only) {
        hooks->argmax_out->resize((size_t) q);
        ggml_backend_tensor_get(ex->argmax, hooks->argmax_out->data(), 0,
                                sizeof(int32_t) * (size_t) q);
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
    if (hooks->capture_out && ex->capture && ncap > 0) {
        std::vector<float> flat((size_t) w.n_embd * ncap * q);
        ggml_backend_tensor_get(ex->capture, flat.data(), 0, sizeof(float) * flat.size());
        hooks->capture_out->assign((size_t) ncap * w.n_embd * q, 0.0f);
        for (int ci = 0; ci < ncap; ++ci) {
            for (int t = 0; t < q; ++t) {
                const float * src = flat.data() + ((size_t) ci * q + t) * w.n_embd;
                float * dst = hooks->capture_out->data() +
                              (size_t) t * ncap * w.n_embd + (size_t) ci * w.n_embd;
                std::memcpy(dst, src, sizeof(float) * (size_t) w.n_embd);
            }
        }
    }
    if (telemetry) {
        telemetry->full_graph_read_us += ds4_elapsed_us(read_t0, Ds4TimingClock::now());
    }
    return 1;
}
