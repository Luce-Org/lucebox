#include "qwen35moe_backend.h"

#include "../common/moe_hybrid_placement.h"
#include "../common/moe_hybrid_stream.h"
#include "../common/moe_hybrid_types.h"
#include "../common/moe_hybrid_types_impl.h"
#include "../common/kvflash_placement.h"
#include "common/ggml_graph_precision.h"
#include "common/sampler.h"
#include "common/dflash_spec_decode.h"
#include "dflash_draft_graph.h"
#include "dflash_feature_ring.h"
#include "graph_builders.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "common/gguf_mmap.h"

namespace dflash::common {

namespace {

using HybridClock = std::chrono::steady_clock;

static uint64_t elapsed_us(HybridClock::time_point start, HybridClock::time_point end) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

} // namespace

Qwen35MoeBackend::Qwen35MoeBackend(const Qwen35Config & cfg)
    : Qwen35Backend(cfg) {}

bool Qwen35MoeBackend::load_target_model(ggml_backend_t backend, TargetWeights & out) {
    // Phase 1: Load core model (non-expert tensors) to GPU.
    // Expert tensors get metadata descriptors but are NOT allocated on GPU.
    TargetLoadPlan plan;
    plan.skip_expert_tensors = true;
    if (!load_target_gguf_partial(cfg_.target_path, backend, plan, out)) {
        return false;
    }

    if (const char * stats_path = std::getenv("DFLASH_QWEN35MOE_RUNTIME_STATS_OUT")) {
        routing_stats_ = std::make_shared<MoeHybridRoutingStats>();
        if (!routing_stats_->init(out.n_layer, out.n_expert, out.n_expert_used)) {
            set_last_error("qwen35moe runtime stats init failed");
            return false;
        }
        routing_stats_out_path_ = stats_path;
    }

    // Phase 2: Compute dynamic placement based on VRAM budget.
    // Expert tensor metadata (ne/nb) is valid even without GPU allocation.
    MoeHybridPlacement placement;
    std::string placement_source;
    std::string err;

    const char * hotness_path = std::getenv("DFLASH_QWEN35MOE_HOTNESS");

    if (!load_dynamic_placement(hotness_path, backend, out, placement, &err)) {
        set_last_error(std::string("qwen35moe dynamic placement failed: ") + err);
        return false;
    }
    placement_source = hotness_path && hotness_path[0]
        ? std::string("hotness:") + hotness_path
        : std::string("uniform");

    // If all experts fit on GPU, reload with experts included
    if (placement.total_hot >= out.n_layer * out.n_expert) {
        std::printf("[qwen35moe] all experts fit in VRAM, loading fully to GPU\n");
        std::fflush(stdout);
        // Record the placement result so post_kvflash_init_gate() can disable
        // the KVFlash pool (moe_hybrid is null on this all-hot path).
        placement_all_hot_ = true;
        free_target_weights(out);
        return load_target_gguf(cfg_.target_path, backend, out);
    }

    if (const char * telemetry = std::getenv("DFLASH_QWEN35MOE_TELEMETRY")) {
        hybrid_telemetry_ = std::atoi(telemetry) != 0;
    }

    // Phase 3: Load expert data from GGUF mmap directly into split hot/cold buffers.
    // Open GGUF again to get tensor file offsets and mmap the data.
    {
        ggml_context * expert_meta = nullptr;
        gguf_init_params gip{};
        gip.no_alloc = true;
        gip.ctx = &expert_meta;
        gguf_context * gctx = gguf_init_from_file(cfg_.target_path, gip);
        if (!gctx) {
            set_last_error("failed to re-open GGUF for expert loading");
            return false;
        }

        // Mmap the file
        GgufMmap _mf;
        std::string _mferr;
        if (!_mf.open(cfg_.target_path, _mferr)) {
            set_last_error("mmap failed on GGUF: " + _mferr);
            gguf_free(gctx);
            return false;
        }
        const size_t file_size = _mf.size();
        // Transfer mmap ownership out of the RAII wrapper: the hybrid storage
        // keeps the mapping alive for streaming prefill and unmaps it in
        // ~MoeHybridStorage. On POSIX the fd can be closed now (the mapping
        // stays valid); on Windows release() already closed the mapping handle.
        GgufMmap::OwnedRegion _region = _mf.release();
        const void * mmap_addr = _region.data;
#if !defined(_WIN32)
        if (_region.fd >= 0) ::close(_region.fd);
#endif

        const size_t data_start = gguf_get_data_offset(gctx);
        const auto * file_bytes = (const uint8_t *)mmap_addr;

        // Build per-layer expert file data
        std::vector<LayerExpertFileData> layer_file_data((size_t)out.n_layer);
        for (int il = 0; il < out.n_layer; ++il) {
            char name[128];
            auto find_tensor_data = [&](const char * suffix) -> ExpertTensorFileData {
                std::snprintf(name, sizeof(name), "blk.%d.%s.weight", il, suffix);
                int64_t tid = gguf_find_tensor(gctx, name);
                if (tid < 0) return {};
                size_t off = data_start + gguf_get_tensor_offset(gctx, tid);
                size_t sz = gguf_get_tensor_size(gctx, tid);
                if (off + sz > file_size) return {};
                return { file_bytes + off, sz };
            };

            layer_file_data[(size_t)il].gate_exps    = find_tensor_data("ffn_gate_exps");
            layer_file_data[(size_t)il].up_exps      = find_tensor_data("ffn_up_exps");
            layer_file_data[(size_t)il].down_exps    = find_tensor_data("ffn_down_exps");
            layer_file_data[(size_t)il].gate_up_exps = find_tensor_data("ffn_gate_up_exps");
        }

        auto hybrid = std::make_shared<MoeHybridStorage>();
        MoeHybridConfig hybrid_cfg = make_moe_hybrid_config(out);
        if (hybrid_cfg.mmq_safe_full_batch) {
            std::fprintf(stderr, "[qwen35moe] GPU sm_%d: MMQ full-batch enabled (no sub-batch workaround)\n",
                         query_gpu_compute_sm());
        }
        std::vector<MoeLayerDesc> layer_descs((size_t)out.n_layer);
        for (int il = 0; il < out.n_layer; ++il) {
            layer_descs[(size_t)il] = make_moe_layer_desc(out.layers[(size_t)il]);
        }
        int cache_slots = 0;
    if (const char * cs = std::getenv("DFLASH_QWEN35MOE_CACHE_SLOTS")) cache_slots = std::max(0, std::atoi(cs));
    else if (cache_slots_ >= 0) cache_slots = cache_slots_;
    if (!build_moe_hybrid_storage_from_file_with_mmap(hybrid_cfg, backend, placement, layer_descs, layer_file_data, mmap_addr, file_size, *hybrid, &err, cache_slots)) {
#if defined(_WIN32)
            UnmapViewOfFile(const_cast<void *>(mmap_addr));
#else
            ::munmap(const_cast<void *>(mmap_addr), file_size);
#endif
            gguf_free(gctx);
            set_last_error(std::string("qwen35moe hybrid storage build failed: ") + err);
            return false;
        }

        // Keep mmap open for streaming prefill — do NOT munmap here.
        // The mmap_data/mmap_size are stored in hybrid storage for lifetime management.
        gguf_free(gctx);

        out.moe_hybrid = std::move(hybrid);
    }

    // Initialize streaming engine for prefill (if cold experts exist and mmap is available)
    if (out.moe_hybrid && out.moe_hybrid->has_mmap() && !out.moe_hybrid->layers.empty()) {
        // Compute max expert size across all layers
        size_t max_expert_bytes = 0;
        for (const auto & layer : out.moe_hybrid->layers) {
            size_t per_expert = layer.fused_gate_up
                ? layer.gate_up_expert_bytes + layer.down_expert_bytes
                : layer.gate_expert_bytes + layer.up_expert_bytes + layer.down_expert_bytes;
            max_expert_bytes = std::max(max_expert_bytes, per_expert);
        }
        if (max_expert_bytes > 0) {
            std::string stream_err;
            if (stream_engine_.init(backend, max_expert_bytes, &stream_err)) {
                std::printf("[qwen35moe] streaming prefill engine ready (pinned=%.1f MiB, scratch=%.1f MiB)\n",
                            stream_engine_.pinned_bytes() / 1024.0 / 1024.0,
                            stream_engine_.scratch_bytes() / 1024.0 / 1024.0);
            } else {
                std::fprintf(stderr, "[qwen35moe] warning: streaming engine init failed: %s (prefill will use fallback)\n",
                             stream_err.c_str());
            }
        }
    }

    int total_cold = 0;
    uint64_t hot_bytes = 0;
    uint64_t cold_bytes = 0;
    for (const auto & layer : out.moe_hybrid->layers) {
        total_cold += (int)layer.cold_expert_ids.size();
        const uint64_t per_expert_bytes = layer.fused_gate_up
            ? (uint64_t)layer.gate_up_expert_bytes + (uint64_t)layer.down_expert_bytes
            : (uint64_t)layer.gate_expert_bytes + (uint64_t)layer.up_expert_bytes + (uint64_t)layer.down_expert_bytes;
        hot_bytes  += per_expert_bytes * (uint64_t)layer.hot_expert_ids.size();
        cold_bytes += per_expert_bytes * (uint64_t)layer.cold_expert_ids.size();
    }
    std::printf("[qwen35moe] hybrid storage ready: total_hot=%d (%.2f GiB VRAM) total_cold=%d (%.2f GiB RAM) source=%s\n",
                out.moe_hybrid->placement.total_hot,
                hot_bytes / 1024.0 / 1024.0 / 1024.0,
                total_cold,
                cold_bytes / 1024.0 / 1024.0 / 1024.0,
                placement_source.c_str());
    std::printf("[qwen35moe] pipelined decode path active (hot=%d cold=%d)\n",
                out.moe_hybrid->placement.total_hot, total_cold);
    if (const char * out_path = std::getenv("DFLASH_QWEN35MOE_NEXT_PLACEMENT_OUT")) {
        placement_out_path_ = out_path;
    }
    if (const char * swap_max = std::getenv("DFLASH_QWEN35MOE_SWAP_MAX")) {
        swap_policy_.max_swaps_total = std::max(0, std::atoi(swap_max));
    }
    if (const char * swap_gain = std::getenv("DFLASH_QWEN35MOE_SWAP_MIN_GAIN")) {
        swap_policy_.min_promote_gain = (uint64_t)std::max(1, std::atoi(swap_gain));
    }
    return true;
}

void Qwen35MoeBackend::after_target_compute(StepGraph & sg, int, int) {
    if (!routing_stats_) return;
    if (sg.moe_selected.empty()) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr, "[qwen35moe] warning: moe_selected is empty in after_target_compute\n");
            warned = true;
        }
        return;
    }
    std::string err;
    for (int il = 0; il < target_weights().n_layer; ++il) {
        ggml_tensor * selected = (il < (int)sg.moe_selected.size()) ? sg.moe_selected[(size_t)il] : nullptr;
        if (!selected) continue;
        if (!routing_stats_->observe_selected_tensor(target_backend(), il, selected, &err)) {
            std::fprintf(stderr, "[qwen35moe] routing-stats observe failed at layer %d: %s\n",
                         il, err.c_str());
            break;
        }
    }
}

bool Qwen35MoeBackend::spark_wants_bootstrap() const {
    return routing_stats_ && !layer_expert_bytes_.empty() && spark_expert_budget_ > 0;
}

// Re-mmap the GGUF and rebuild the hot/cold storage for a new placement. Used by
// the Spark bootstrap to apply the calibrated placement in-process.
bool Qwen35MoeBackend::rebuild_hybrid_from_placement(const MoeHybridPlacement & placement,
                                                     std::string & err) {
    TargetWeights & out = target_weights();
    ggml_backend_t backend = target_backend();

    gguf_init_params gip{};
    gguf_context * gctx = gguf_init_from_file(cfg_.target_path, gip);
    if (!gctx) { err = "gguf reinit failed"; return false; }
    GgufMmap _mf;
    std::string _mferr;
    if (!_mf.open(cfg_.target_path, _mferr)) {
        gguf_free(gctx);
        err = "mmap failed: " + _mferr;
        return false;
    }
    const size_t file_size = _mf.size();
    const void * mmap_addr = _mf.data();

    const size_t data_start = gguf_get_data_offset(gctx);
    const auto * file_bytes = (const uint8_t *)mmap_addr;
    std::vector<LayerExpertFileData> layer_file_data((size_t)out.n_layer);
    for (int il = 0; il < out.n_layer; ++il) {
        char name[128];
        auto find_tensor_data = [&](const char * suffix) -> ExpertTensorFileData {
            std::snprintf(name, sizeof(name), "blk.%d.%s.weight", il, suffix);
            int64_t tid = gguf_find_tensor(gctx, name);
            if (tid < 0) return {};
            size_t off = data_start + gguf_get_tensor_offset(gctx, tid);
            size_t sz = gguf_get_tensor_size(gctx, tid);
            if (off + sz > file_size) return {};
            return { file_bytes + off, sz };
        };
        layer_file_data[(size_t)il].gate_exps    = find_tensor_data("ffn_gate_exps");
        layer_file_data[(size_t)il].up_exps      = find_tensor_data("ffn_up_exps");
        layer_file_data[(size_t)il].down_exps    = find_tensor_data("ffn_down_exps");
        layer_file_data[(size_t)il].gate_up_exps = find_tensor_data("ffn_gate_up_exps");
    }

    // Free the current hot/cold buffers before allocating the new ones so the
    // rebuild fits in VRAM (no transient 2x). Safe: bootstrap runs at startup
    // with no in-flight requests, and the budget is unchanged so the build that
    // succeeded at init succeeds again.
    out.moe_hybrid.reset();

    auto hybrid = std::make_shared<MoeHybridStorage>();
    MoeHybridConfig hybrid_cfg = make_moe_hybrid_config(out);
    std::vector<MoeLayerDesc> layer_descs((size_t)out.n_layer);
    for (int il = 0; il < out.n_layer; ++il)
        layer_descs[(size_t)il] = make_moe_layer_desc(out.layers[(size_t)il]);
    const int cache_slots = cache_slots_ >= 0 ? cache_slots_ : 0;

    const bool ok = build_moe_hybrid_storage_from_file(hybrid_cfg, backend, placement, layer_descs,
                                                       layer_file_data, *hybrid, &err, cache_slots);
    gguf_free(gctx);
    if (!ok) return false;
    out.moe_hybrid = std::move(hybrid);
    return true;
}

bool Qwen35MoeBackend::spark_bootstrap_finalize(const std::string & profile_path) {
    if (!spark_wants_bootstrap()) return false;
    std::string err;
    routing_stats_->save_csv(profile_path, &err);  // persist the observed routing
    const TargetWeights & w = target_weights();
    MoeHybridPlacement placement;
    if (!MoeHybridPlacement::build_from_stats_with_layer_bytes(
            *routing_stats_, layer_expert_bytes_, spark_expert_budget_,
            std::min(w.n_expert_used, w.n_expert), placement, &err)) {
        std::fprintf(stderr, "[spark] bootstrap placement build failed: %s\n", err.c_str());
        return false;
    }
    if (!rebuild_hybrid_from_placement(placement, err)) {
        std::fprintf(stderr, "[spark] bootstrap storage rebuild failed: %s\n", err.c_str());
        return false;
    }
    return true;
}

bool Qwen35MoeBackend::post_kvflash_init_gate() {
    // Gate: disable the KVFlash pool when dynamic placement confirmed all experts
    // fit hot even with the FULL max_ctx KV reservation — the pool then reserves
    // nothing useful (pure slot-map overhead).  placement_all_hot_full_kv_ is set
    // in load_dynamic_placement().  When the pool is what KEEPS experts hot
    // (placement_all_hot_ true but _full_kv_ false), we must NOT disable it.
    if (!kvflash_active()) return true;

    bool should_disable = false;
    if (placement_all_hot_full_kv_) {
        should_disable = true;
    } else if (target_weights().moe_hybrid) {
        int total_cold = 0;
        for (const auto & ls : target_weights().moe_hybrid->layers) {
            total_cold += (int)ls.cold_expert_ids.size();
        }
        if (total_cold == 0) should_disable = true;  // hybrid built but 0 cold
    }

    if (should_disable) {
        std::printf("[kvflash] disabled: placement all-hot at max_ctx %d, pool not needed\n",
                    cfg_.device.max_ctx);
        std::fflush(stdout);
        kvflash_tokens_ = 0;
        kvflash_tau_    = 64;
        kvflash_drafter_path_.clear();
    }
    return true;
}

void Qwen35MoeBackend::maybe_post_request_swap() {
    if (!routing_stats_) return;

    if (!routing_stats_out_path_.empty()) {
        std::string err;
        if (!routing_stats_->save_csv(routing_stats_out_path_, &err)) {
            std::fprintf(stderr, "[qwen35moe] failed to save runtime stats: %s\n", err.c_str());
        }
    }

    if (!target_weights().moe_hybrid || swap_policy_.max_swaps_total <= 0) return;

    MoeHybridSwapPlan plan;
    std::string err;
    if (!build_moe_hybrid_swap_plan(target_weights().moe_hybrid->placement, *routing_stats_,
                                   swap_policy_, plan, &err)) {
        std::fprintf(stderr, "[qwen35moe] swap plan failed: %s\n", err.c_str());
        return;
    }
    if (plan.actions.empty()) return;

    auto rebuilt = std::make_shared<MoeHybridStorage>();
    MoeHybridConfig swap_cfg = make_moe_hybrid_config(target_weights());
    std::vector<MoeLayerDesc> swap_descs((size_t)target_weights().n_layer);
    for (int il = 0; il < target_weights().n_layer; ++il) {
        swap_descs[(size_t)il] = make_moe_layer_desc(target_weights().layers[(size_t)il]);
    }
    if (!build_moe_hybrid_storage(swap_cfg, target_backend(),
                                        plan.next_placement, swap_descs, *rebuilt, &err)) {
        std::fprintf(stderr, "[qwen35moe] swap rebuild failed: %s\n", err.c_str());
        return;
    }
    target_weights().moe_hybrid = std::move(rebuilt);
    if (!placement_out_path_.empty()) {
        if (!plan.next_placement.save_json(placement_out_path_, "qwen35moe", &err)) {
            std::fprintf(stderr, "[qwen35moe] failed to save next placement: %s\n", err.c_str());
        }
    }
    std::printf("[qwen35moe] applied %zu swap actions at request boundary\n", plan.actions.size());
}

bool Qwen35MoeBackend::run_ar_decode_path(int committed, int n_gen,
                                          std::vector<int32_t> & out_tokens,
                                          const DaemonIO & io) {
    if (!target_weights().moe_hybrid) {
        return Qwen35Backend::run_ar_decode_path(committed, n_gen, out_tokens, io);
    }
    if (n_gen <= 0) return true;

    return run_pipelined_decode_path(committed, n_gen, out_tokens, io);
}

// ─── Pipelined decode: cached DeltaNet graphs + optimized FFN loop ───────────

bool Qwen35MoeBackend::ensure_pipe_state(int kv_start) {
    if (pipe_state_ && pipe_state_->valid()) return true;
    pipe_state_ = std::make_unique<PipelinedDecodeState>();
    if (!init_pipelined_decode_state(*pipe_state_, target_backend(), target_weights(),
                                     target_cache(), *target_weights().moe_hybrid,
                                     kv_start, cfg_.kq_stride_pad)) {
        pipe_state_.reset();
        return false;
    }
    pipe_state_->routing_collector = routing_collector_;
    return true;
}

bool Qwen35MoeBackend::run_pipelined_decode_path(int committed, int n_gen,
                                                  std::vector<int32_t> & out_tokens,
                                                  const DaemonIO & io) {
    const int hidden = target_weights().n_embd;
    const int vocab  = target_weights().n_vocab;
    std::vector<float> logits_buf((size_t)vocab);
    std::vector<float> act_cur((size_t)hidden);

    // Telemetry accumulators for the full decode loop
    using DecodeClock = std::chrono::steady_clock;
    uint64_t tel_embed_us = 0;
    uint64_t tel_layers_us = 0;
    uint64_t tel_logits_us = 0;
    uint64_t tel_sample_us = 0;
    PipelinedDecodeTelemetry tel_layers_accum{};
    int tel_n_tokens = 0;

    // Persistent logits graph (built once, reused per token).
    // Precision policy (#310): keep the rms-norm input and out_norm in f32.
    StepGraph logits_sg;
    auto project_logits = [&]() -> bool {
        if (!logits_sg.ctx) {
            ggml_init_params ip{};
            ip.mem_size = 64 * 1024 * 1024;
            ip.no_alloc = true;
            logits_sg.ctx = ggml_init(ip);
            if (!logits_sg.ctx) return false;
            logits_sg.hidden_input = ggml_new_tensor_3d(logits_sg.ctx, GGML_TYPE_F32, hidden, 1, 1);
            ggml_set_input(logits_sg.hidden_input);
            logits_sg.gf = ggml_new_graph_custom(logits_sg.ctx, 1024, false);
            ggml_tensor * normed = ggml_rms_norm(
                logits_sg.ctx,
                rms_norm_input_f32(logits_sg.ctx, logits_sg.hidden_input),
                target_weights().rms_eps);
            normed = ggml_mul(
                logits_sg.ctx, normed,
                graph_tensor_f32(logits_sg.ctx, target_weights().out_norm));
            logits_sg.logits = ggml_mul_mat(logits_sg.ctx, target_weights().output, normed);
            ggml_set_output(logits_sg.logits);
            ggml_build_forward_expand(logits_sg.gf, logits_sg.logits);
            logits_sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend()));
            if (!ggml_gallocr_alloc_graph(logits_sg.alloc, logits_sg.gf)) {
                step_graph_destroy(logits_sg);
                return false;
            }
        }
        // GPU→GPU: pipe act_cur directly into logits graph (no host bounce)
        ggml_backend_tensor_copy_async(target_backend(), target_backend(),
                                       pipe_state_->gpu_state.act_cur, logits_sg.hidden_input);
        auto st = ggml_backend_graph_compute(target_backend(), logits_sg.gf);
        if (st != GGML_STATUS_SUCCESS) return false;
        ggml_backend_tensor_get(logits_sg.logits, logits_buf.data(), 0, sizeof(float) * (size_t)vocab);
        return true;
    };

    // ── First token: sample from prefill logits ──
    {
        int32_t first_tok;
        if (sampler_config().temp > 0) {
            if (!prefill_logits_valid()) return false;
            ggml_backend_tensor_get(target_step_graph().logits, logits_buf.data(),
                                    prefill_logits_offset(), sizeof(float) * (size_t)vocab);
            first_tok = sample_logits(logits_buf.data(), vocab, sampler_config(),
                                     out_tokens, sampler_rng_engine());
        } else {
            first_tok = target_cache().last_tok;
        }
        out_tokens.push_back(first_tok);
        io.emit(first_tok);
        if (is_eos_tok(first_tok, target_weights())) return true;
        committed++;
        target_cache().cur_pos = committed;
        if (kvflash_active()) kvflash_history_.push_back(first_tok);
    }

    // ── Ensure persistent pipelined state (built once, reused) ──
    if (!ensure_pipe_state(committed)) {
        return false;
    }

    for (int step = 1; step < n_gen; ++step) {
        const auto tok_t0 = DecodeClock::now();

        int32_t tok = out_tokens.back();
        if (!target_weights().embedder.embed(&tok, 1, act_cur.data())) {
            return false;
        }
        ggml_backend_tensor_set_async(target_backend(), pipe_state_->gpu_state.act_cur,
                                      act_cur.data(), 0, sizeof(float) * (size_t)hidden);
        const auto embed_done = DecodeClock::now();

        // kvflash: physical pool slot for this token's KV rows (may evict).
        int kv_slot = -1;
        if (kvflash_active()) {
            kv_slot = kvflash_pager_.slot_for(committed);
            if (kv_slot < 0) {
                std::fprintf(stderr, "[kvflash] pipelined decode: no slot at pos %d\n",
                             committed);
                return false;
            }
        }

        PipelinedDecodeTelemetry tel;
        if (!pipelined_decode_one_token(*pipe_state_, target_backend(), target_weights(),
                                       target_cache(), *target_weights().moe_hybrid,
                                       committed, cfg_.kq_stride_pad,
                                       hybrid_telemetry_ ? &tel : nullptr,
                                       kv_slot)) {
            return false;
        }
        const auto layers_done = DecodeClock::now();

        // act_cur stays on GPU — project_logits reads it via GPU→GPU copy
        if (!project_logits()) {
            step_graph_destroy(logits_sg);
            return false;
        }
        const auto logits_done = DecodeClock::now();

        int32_t next_tok;
        if (sampler_config().temp > 0) {
            next_tok = sample_logits(logits_buf.data(), vocab, sampler_config(),
                                    out_tokens, sampler_rng_engine());
        } else {
            next_tok = 0;
            float best = logits_buf[0];
            for (int j = 1; j < vocab; ++j) {
                if (logits_buf[(size_t)j] > best) {
                    best = logits_buf[(size_t)j];
                    next_tok = j;
                }
            }
        }
        const auto sample_done = DecodeClock::now();

        if (hybrid_telemetry_) {
            auto us = [](DecodeClock::time_point a, DecodeClock::time_point b) -> uint64_t {
                return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
            };
            tel_embed_us += us(tok_t0, embed_done);
            tel_layers_us += us(embed_done, layers_done);
            tel_logits_us += us(layers_done, logits_done);
            tel_sample_us += us(logits_done, sample_done);
            tel_n_tokens++;
            // Accumulate per-layer telemetry
            tel_layers_accum.total_us += tel.total_us;
            tel_layers_accum.prefn_graph_build_us += tel.prefn_graph_build_us;
            tel_layers_accum.prefn_compute_us += tel.prefn_compute_us;
            tel_layers_accum.routing_readback_us += tel.routing_readback_us;
            tel_layers_accum.ffn_us += tel.ffn_us;
            tel_layers_accum.ffn_allhot_us += tel.ffn_allhot_us;
            tel_layers_accum.ffn_mixed_us += tel.ffn_mixed_us;
            tel_layers_accum.gpu_idle_us += tel.gpu_idle_us;
            tel_layers_accum.tensor_io_us += tel.tensor_io_us;
            tel_layers_accum.combine_overhead_us += tel.combine_overhead_us;
            tel_layers_accum.cold_cpu_us += tel.cold_cpu_us;
            tel_layers_accum.cold_compute_us += tel.cold_compute_us;
            tel_layers_accum.hot_graph_build_us += tel.hot_graph_build_us;
            tel_layers_accum.ffn_post_get_us += tel.ffn_post_get_us;
            tel_layers_accum.sync_wait_us += tel.sync_wait_us;
            tel_layers_accum.allhot_layers += tel.allhot_layers;
            tel_layers_accum.mixed_layers += tel.mixed_layers;
            tel_layers_accum.total_layers += tel.total_layers;
            tel_layers_accum.hot_graph_rebuilds += tel.hot_graph_rebuilds;
            tel_layers_accum.routed_ffn_layers += tel.routed_ffn_layers;
            tel_layers_accum.routed_prefn_us += tel.routed_prefn_us;
            tel_layers_accum.routed_sync_us += tel.routed_sync_us;
            tel_layers_accum.routed_readback_us += tel.routed_readback_us;
            tel_layers_accum.routed_cpu_remap_us += tel.routed_cpu_remap_us;
            tel_layers_accum.routed_ffn_dispatch_us += tel.routed_ffn_dispatch_us;
            tel_layers_accum.routed_final_sync_us += tel.routed_final_sync_us;
            tel_layers_accum.routed_cold_expert_hits += tel.routed_cold_expert_hits;
            tel_layers_accum.routed_total_expert_slots += tel.routed_total_expert_slots;
        }

        out_tokens.push_back(next_tok);
        io.emit(next_tok);
        committed++;
        target_cache().cur_pos = committed;
        if (kvflash_active()) {
            kvflash_history_.push_back(next_tok);
            kvflash_maybe_reselect((int)out_tokens.size());
        }
        if (io.cancelled) break;
        if (is_eos_tok(next_tok, target_weights())) break;
    }

    // ── Print decode telemetry ──
    if (hybrid_telemetry_ && tel_n_tokens > 0) {
        const double total_us = (double)(tel_embed_us + tel_layers_us + tel_logits_us + tel_sample_us);
        std::printf("[qwen35moe-ar] === AR DECODE TELEMETRY (n_tokens=%d, %.1f tok/s) ===\n",
                    tel_n_tokens, tel_n_tokens / (total_us / 1e6));
        std::printf("  per-token breakdown:\n");
        std::printf("    embed=%.2fms  layers=%.2fms  logits=%.2fms  sample=%.2fms\n",
                    tel_embed_us / 1000.0 / tel_n_tokens,
                    tel_layers_us / 1000.0 / tel_n_tokens,
                    tel_logits_us / 1000.0 / tel_n_tokens,
                    tel_sample_us / 1000.0 / tel_n_tokens);
        std::printf("  time budget: embed=%.1f%% layers=%.1f%% logits=%.1f%% sample=%.1f%%\n",
                    100.0 * tel_embed_us / total_us,
                    100.0 * tel_layers_us / total_us,
                    100.0 * tel_logits_us / total_us,
                    100.0 * tel_sample_us / total_us);
        // Routed path breakdown (the dominant path)
        if (tel_layers_accum.routed_ffn_layers > 0) {
            const int rl = tel_layers_accum.routed_ffn_layers;
            std::printf("  routed FFN path (%d layer-evals, %d cold_hits / %d slots = %.1f%% cold):\n",
                        rl,
                        tel_layers_accum.routed_cold_expert_hits,
                        tel_layers_accum.routed_total_expert_slots,
                        tel_layers_accum.routed_total_expert_slots > 0
                            ? 100.0 * tel_layers_accum.routed_cold_expert_hits / tel_layers_accum.routed_total_expert_slots
                            : 0.0);
            std::printf("    per-layer avg: prefn_dispatch=%.1fus sync_stall=%.1fus readback=%.1fus remap=%.1fus ffn_dispatch=%.1fus\n",
                        (double)tel_layers_accum.routed_prefn_us / rl,
                        (double)tel_layers_accum.routed_sync_us / rl,
                        (double)tel_layers_accum.routed_readback_us / rl,
                        (double)tel_layers_accum.routed_cpu_remap_us / rl,
                        (double)tel_layers_accum.routed_ffn_dispatch_us / rl);
            std::printf("    total: sync_stall=%.1fms (%.1f%% of layers time)\n",
                        tel_layers_accum.routed_sync_us / 1000.0,
                        100.0 * tel_layers_accum.routed_sync_us / (double)tel_layers_us);
            std::printf("    final_sync=%.1fms (%.1f%% of layers time)\n",
                        tel_layers_accum.routed_final_sync_us / 1000.0,
                        100.0 * tel_layers_accum.routed_final_sync_us / (double)tel_layers_us);
        }
        // Split path stats (if any)
        if (tel_layers_accum.mixed_layers > 0) {
            std::printf("  split path: mixed=%d layers, cold_cpu=%.1fms, ffn_mixed=%.1fms\n",
                        tel_layers_accum.mixed_layers,
                        tel_layers_accum.cold_cpu_us / 1000.0,
                        tel_layers_accum.ffn_mixed_us / 1000.0);
        }
        std::printf("  split path allhot=%d layers, hot_graph_rebuilds=%d\n",
                    tel_layers_accum.allhot_layers - tel_layers_accum.routed_ffn_layers,
                    tel_layers_accum.hot_graph_rebuilds);
        std::fflush(stdout);
    }

    step_graph_destroy(logits_sg);
    return true;
}

GenerateResult Qwen35MoeBackend::generate_impl(const GenerateRequest & req,
                                          const DaemonIO & io) {
    if (!target_weights().moe_hybrid) {
        auto result = Qwen35Backend::generate_impl(req, io);
        if (result.ok) maybe_post_request_swap();
        return result;
    }

    // Hybrid generate: integrated prefill + decode using per-token hybrid FFN
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    sampler_config() = req.sampler;
    if (req.do_sample && sampler_config().seed != 0) {
        sampler_rng_engine().seed(sampler_config().seed);
    }

    reset_recurrent_state(target_cache());

    // Invalidate cached pipelined decode state between requests.
    // The cached DeltaNet graphs reference conv_state/ssm_state tensors that were
    // just zeroed; rebuilding is cheap (~30 graphs) and avoids stale-pointer crashes.
    pipe_state_.reset();

    // Free per-layer cached FFN graphs from previous decode to release GPU memory
    // before prefill allocates its own graph buffers.
    if (target_weights().moe_hybrid) {
        for (auto & layer : target_weights().moe_hybrid->layers) {
            layer.hot_graph.free();
            layer.cold_graph.free();
        }
    }

    const int hidden = target_weights().n_embd;
    const int vocab  = target_weights().n_vocab;
    std::vector<float> act_cur((size_t)hidden);
    std::vector<float> logits_buf((size_t)vocab);
    std::vector<int32_t> selected((size_t)target_weights().n_expert_used);
    std::vector<float> weights_buf((size_t)target_weights().n_expert_used);
    ggml_backend_t cpu_be = target_weights().moe_hybrid->cpu_backend;

    const int n_layer = target_weights().n_layer;
    uint64_t build_us_total = 0, compute_us_total = 0, readback_us_total = 0, ffn_us_total = 0;
    int hot_only_layers = 0, total_ffn_layers = 0;
    MoeHybridFfnTelemetry ffn_tel_accum{};

    StepGraph logits_sg;  // Persistent logits graph (used by spec-decode branch)

    auto cleanup_graphs = [&]() {
        step_graph_destroy(logits_sg);
    };

    // Helper: compute logits from act_cur (persistent graph, built once)
    auto compute_logits = [&](ggml_tensor* gpu_src = nullptr) -> bool {
        if (!logits_sg.ctx) {
            // First call: build the logits graph
            ggml_init_params ip{};
            ip.mem_size   = 64 * 1024 * 1024;
            ip.mem_buffer = nullptr;
            ip.no_alloc   = true;
            logits_sg.ctx = ggml_init(ip);
            if (!logits_sg.ctx) return false;
            logits_sg.hidden_input = ggml_new_tensor_3d(logits_sg.ctx, GGML_TYPE_F32, hidden, 1, 1);
            ggml_set_input(logits_sg.hidden_input);
            logits_sg.gf = ggml_new_graph_custom(logits_sg.ctx, 1024, false);
            ggml_tensor * normed = ggml_rms_norm(
                logits_sg.ctx,
                rms_norm_input_f32(logits_sg.ctx, logits_sg.hidden_input),
                target_weights().rms_eps);
            normed = ggml_mul(
                logits_sg.ctx, normed,
                graph_tensor_f32(logits_sg.ctx, target_weights().out_norm));
            logits_sg.logits = ggml_mul_mat(logits_sg.ctx, target_weights().output, normed);
            ggml_set_output(logits_sg.logits);
            ggml_build_forward_expand(logits_sg.gf, logits_sg.logits);
            logits_sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend()));
            if (!ggml_gallocr_alloc_graph(logits_sg.alloc, logits_sg.gf)) {
                step_graph_destroy(logits_sg);
                return false;
            }
        }
        if (gpu_src) {
            // GPU→GPU: pipe act_cur directly without host bounce
            ggml_backend_tensor_copy_async(target_backend(), target_backend(),
                                            gpu_src, logits_sg.hidden_input);
        } else {
            ggml_backend_tensor_set(logits_sg.hidden_input, act_cur.data(), 0, sizeof(float) * (size_t)hidden);
        }
        auto st = ggml_backend_graph_compute(target_backend(), logits_sg.gf);
        if (st != GGML_STATUS_SUCCESS) return false;
        ggml_backend_tensor_get(logits_sg.logits, logits_buf.data(), 0, sizeof(float) * (size_t)vocab);
        return true;
    };

    // ── Hybrid Prefill: chunked batched pre-FFN per layer, batched FFN ──
    auto t_prefill_start = std::chrono::steady_clock::now();
    const int prompt_len = (int)req.prompt.size();
    const int prefill_chunk = std::min(128, prompt_len); // batch size per GPU compute

    // kvflash: hybrid prefill writes rows identity-mapped, so the prompt must
    // fit the pool with one chunk of decode headroom (same contract as the
    // base do_prefill).
    if (kvflash_active() &&
        prompt_len > kvflash_tokens_ - kvflash_pager_.chunk_tokens()) {
        std::fprintf(stderr,
            "[kvflash] hybrid prompt (%d) exceeds pool %d; raise --kvflash "
            "or enable pflash compression\n", prompt_len, kvflash_tokens_);
        result.error = "kvflash: prompt exceeds resident pool";
        cleanup_graphs();
        return result;
    }

    // Embed all prompt tokens
    const int n_expert_used = target_weights().n_expert_used;
    std::vector<float> embed_all((size_t)prompt_len * (size_t)hidden);
    for (int i = 0; i < prompt_len; ++i) {
        int32_t tok = req.prompt[(size_t)i];
        if (!target_weights().embedder.embed(&tok, 1, embed_all.data() + (size_t)i * (size_t)hidden)) {
            result.error = "prefill_embed";
            cleanup_graphs();
            return result;
        }
    }

    // Process layer by layer, chunked within each layer
    StepGraph prefill_sg;  // persistent across layers to reuse GPU buffer
    ggml_gallocr_t ffn_hot_alloc = nullptr;
    ggml_gallocr_t ffn_cold_alloc = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        auto & storage = target_weights().moe_hybrid->layers[(size_t)il];

        for (int chunk_start = 0; chunk_start < prompt_len; chunk_start += prefill_chunk) {
            const int chunk_len = std::min(prefill_chunk, prompt_len - chunk_start);
            const auto t0 = HybridClock::now();

            const bool with_mask = (cfg_.kq_stride_pad > KQ_MASK_PAD) || (chunk_len > 1);

            // Build pre-FFN graph for this chunk
            step_graph_free(prefill_sg);  // reset ctx/graph but keep gallocr buffer
            if (!build_layer_prefn_step(prefill_sg, target_weights(), target_cache(), target_backend(),
                                        il, /*kv_start=*/chunk_start, /*n_tokens=*/chunk_len,
                                        with_mask, /*fa_window=*/0, cfg_.kq_stride_pad)) {
                result.error = "prefill_build";
                step_graph_destroy(prefill_sg);
                if (ffn_hot_alloc) ggml_gallocr_free(ffn_hot_alloc);
                if (ffn_cold_alloc) ggml_gallocr_free(ffn_cold_alloc);
                cleanup_graphs();
                return result;
            }

            // Set input embeddings for this chunk
            ggml_backend_tensor_set(prefill_sg.inp_embed,
                                    embed_all.data() + (size_t)chunk_start * (size_t)hidden, 0,
                                    sizeof(float) * (size_t)chunk_len * (size_t)hidden);

            // Set positions if attention layer
            if (prefill_sg.positions) {
                std::vector<int32_t> pos_data((size_t)chunk_len * 4);
                for (int i = 0; i < chunk_len; ++i) {
                    pos_data[(size_t)i * 4 + 0] = chunk_start + i;
                    pos_data[(size_t)i * 4 + 1] = chunk_start + i;
                    pos_data[(size_t)i * 4 + 2] = chunk_start + i;
                    pos_data[(size_t)i * 4 + 3] = 0;
                }
                ggml_backend_tensor_set(prefill_sg.positions, pos_data.data(), 0, sizeof(int32_t) * pos_data.size());
            }

            // Set causal attention mask if needed
            if (prefill_sg.attn_mask) {
                const int kv_len = chunk_start + chunk_len;
                const int kv_pad_override = (int)prefill_sg.attn_mask->ne[0];
                std::vector<uint16_t> mask_buf;
                build_causal_mask(mask_buf, kv_len, chunk_len, /*kv_start=*/chunk_start,
                                  cfg_.kq_stride_pad, /*win_start=*/0, kv_pad_override);
                ggml_backend_tensor_set(prefill_sg.attn_mask, mask_buf.data(), 0,
                                        sizeof(uint16_t) * mask_buf.size());
            }

            const auto t1 = HybridClock::now();
            build_us_total += elapsed_us(t0, t1);

            // Compute batched pre-FFN
            auto st = ggml_backend_graph_compute(target_backend(), prefill_sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                result.error = "prefill_compute";
                step_graph_destroy(prefill_sg);
                if (ffn_hot_alloc) ggml_gallocr_free(ffn_hot_alloc);
                if (ffn_cold_alloc) ggml_gallocr_free(ffn_cold_alloc);
                cleanup_graphs();
                return result;
            }
            const auto t2 = HybridClock::now();
            compute_us_total += elapsed_us(t1, t2);

            // Read back chunk results
            std::vector<float> chunk_residuals((size_t)chunk_len * (size_t)hidden);
            std::vector<float> chunk_post((size_t)chunk_len * (size_t)hidden);
            std::vector<int32_t> chunk_selected((size_t)chunk_len * (size_t)n_expert_used);
            std::vector<float> chunk_weights((size_t)chunk_len * (size_t)n_expert_used);

            ggml_backend_tensor_get(prefill_sg.ffn_residual, chunk_residuals.data(), 0,
                                    sizeof(float) * chunk_residuals.size());
            ggml_backend_tensor_get(prefill_sg.ffn_post, chunk_post.data(), 0,
                                    sizeof(float) * chunk_post.size());
            ggml_tensor * layer_selected = (!prefill_sg.moe_selected.empty() && (size_t)il < prefill_sg.moe_selected.size())
                ? prefill_sg.moe_selected[(size_t)il]
                : nullptr;
            if (!layer_selected || !prefill_sg.moe_weights) {
                result.error = "prefill_router_outputs";
                step_graph_destroy(prefill_sg);
                cleanup_graphs();
                return result;
            }
            ggml_backend_tensor_get(layer_selected, chunk_selected.data(), 0,
                                    sizeof(int32_t) * chunk_selected.size());
            ggml_backend_tensor_get(prefill_sg.moe_weights, chunk_weights.data(), 0,
                                    sizeof(float) * chunk_weights.size());
            const auto t3 = HybridClock::now();
            readback_us_total += elapsed_us(t2, t3);

            // Observe routing stats
            if (routing_stats_) {
                for (int i = 0; i < chunk_len; ++i) {
                    routing_stats_->observe(il, chunk_selected.data() + (size_t)i * (size_t)n_expert_used, n_expert_used);
                }
            }

            // Collect routing data for predictor training (prefill path)
            if (routing_collector_) {
                for (int i = 0; i < chunk_len; ++i) {
                    routing_collector_->record(
                        il,
                        chunk_post.data() + (size_t)i * (size_t)hidden,
                        hidden,
                        chunk_selected.data() + (size_t)i * (size_t)n_expert_used,
                        n_expert_used);
                }
            }

            // Hybrid FFN — skip batched path when cold experts exist (CUDA mul_mat_id bug on sm_75)
            MoeHybridConfig chunk_cfg = make_moe_hybrid_config(target_weights());
            MoeLayerDesc chunk_desc = make_moe_layer_desc(target_weights().layers[(size_t)il]);
            std::vector<float> ffn_batch_out;
            bool ffn_ok = false;
            ++total_ffn_layers;
            if (storage.cold_expert_ids.empty()) {
                // All experts hot — safe to use batched path
                ++hot_only_layers;
                ffn_ok = eval_moe_hybrid_ffn_batched(
                        target_backend(), cpu_be, chunk_cfg, chunk_desc, storage,
                        chunk_post.data(),
                        chunk_selected.data(),
                        chunk_weights.data(),
                        chunk_len, ffn_batch_out, &result.error,
                        &ffn_hot_alloc, &ffn_cold_alloc);
            } else if (storage.all_routed_are_hot(chunk_selected.data(),
                                                   chunk_len * n_expert_used)) {
                // All selected experts happen to be in VRAM — pure GPU, no CPU
                ++hot_only_layers;
                ffn_ok = eval_moe_hot_only_batched(
                        target_backend(), chunk_cfg, chunk_desc, storage,
                        chunk_post.data(),
                        chunk_selected.data(),
                        chunk_weights.data(),
                        chunk_len, ffn_batch_out, &result.error,
                        &ffn_hot_alloc);
            } else if (target_weights().moe_hybrid->has_mmap() &&
                       !target_weights().moe_hybrid->layer_regions.empty() &&
                       stream_engine_.is_ready() && chunk_len >= 16 &&
                       !storage.cold_expert_ids.empty()) {
                // Streaming prefill: batched eval handles hot on GPU + cold on CPU.
                // The streaming engine's mmap keeps data paged in via madvise.
                auto * hybrid = target_weights().moe_hybrid.get();
                const auto & regions = hybrid->layer_regions[(size_t)il];
                // Prefetch cold expert data from mmap for upcoming layers
                std::vector<int32_t> cold_ids_copy(storage.cold_expert_ids.begin(),
                                                   storage.cold_expert_ids.end());
                stream_engine_.prefetch_cold_experts(hybrid->mmap_data, hybrid->mmap_size,
                                                    regions, cold_ids_copy.data(),
                                                    (int)cold_ids_copy.size());
                ffn_ok = eval_moe_hybrid_ffn_batched(
                        target_backend(), cpu_be, chunk_cfg, chunk_desc, storage,
                        chunk_post.data(),
                        chunk_selected.data(),
                        chunk_weights.data(),
                        chunk_len, ffn_batch_out, &result.error,
                        &ffn_hot_alloc, &ffn_cold_alloc);
            }
            if (!ffn_ok) {
                // Per-token fallback (avoids sm_75 mul_mat_id assertion with cold experts)
                result.error.clear();
                ffn_batch_out.assign((size_t)hidden * (size_t)chunk_len, 0.0f);
                std::vector<float> single_out;
                for (int ti = 0; ti < chunk_len; ++ti) {
                    const float * tok_post = chunk_post.data() + (size_t)ti * (size_t)hidden;
                    const int32_t * tok_sel = chunk_selected.data() + (size_t)ti * (size_t)n_expert_used;
                    const float * tok_wts = chunk_weights.data() + (size_t)ti * (size_t)n_expert_used;
                    if (!eval_moe_hybrid_ffn_single(
                            target_backend(), chunk_cfg, chunk_desc, storage, cpu_be,
                            tok_post, tok_sel, tok_wts, n_expert_used, single_out)) {
                        result.error = "prefill_ffn_single";
                        step_graph_destroy(prefill_sg);
                        if (ffn_hot_alloc) ggml_gallocr_free(ffn_hot_alloc);
                        if (ffn_cold_alloc) ggml_gallocr_free(ffn_cold_alloc);
                        cleanup_graphs();
                        return result;
                    }
                    std::memcpy(ffn_batch_out.data() + (size_t)ti * (size_t)hidden,
                                single_out.data(), sizeof(float) * (size_t)hidden);
                }
            }

            // Combine FFN output + residual → embed_all for next layer
            for (int i = 0; i < chunk_len; ++i) {
                const float * ffn = ffn_batch_out.data() + (size_t)i * (size_t)hidden;
                const float * tok_res = chunk_residuals.data() + (size_t)i * (size_t)hidden;
                float * tok_embed = embed_all.data() + (size_t)(chunk_start + i) * (size_t)hidden;
                for (int j = 0; j < hidden; ++j) {
                    tok_embed[j] = ffn[j] + tok_res[j];
                }

                // Feature capture at capture layers for spec-decode
                if (target_cache().target_feat && cfg_.draft_path) {
                    int capture_idx = -1;
                    for (int k = 0; k < target_weights().n_capture_layers; k++) {
                        if (target_weights().capture_layer_ids[k] == il) {
                            capture_idx = k;
                            break;
                        }
                    }
                    if (capture_idx >= 0) {
                        const int token_pos = chunk_start + i;
                        const int cap = target_cache().target_feat_cap;
                        const int slot = token_pos % cap;
                        const size_t elt = ggml_element_size(target_cache().target_feat);
                        const size_t col_stride = target_cache().target_feat->nb[1];
                        const size_t offset = (size_t)slot * col_stride +
                                              (size_t)capture_idx * (size_t)hidden * elt;
                        std::vector<ggml_bf16_t> bf16_tmp((size_t)hidden);
                        ggml_fp32_to_bf16_row(tok_embed, bf16_tmp.data(), hidden);
                        ggml_backend_tensor_set(target_cache().target_feat, bf16_tmp.data(),
                                                 offset, (size_t)hidden * elt);
                    }
                }
            }
            const auto t4 = HybridClock::now();
            ffn_us_total += elapsed_us(t3, t4);
        }
    }
    step_graph_destroy(prefill_sg);
    if (ffn_hot_alloc) ggml_gallocr_free(ffn_hot_alloc);
    if (ffn_cold_alloc) ggml_gallocr_free(ffn_cold_alloc);

    // Copy last token's output to act_cur for decode
    std::memcpy(act_cur.data(), embed_all.data() + (size_t)(prompt_len - 1) * (size_t)hidden,
                sizeof(float) * (size_t)hidden);

    int committed = prompt_len;
    target_cache().cur_pos = committed;
    if (kvflash_active()) {
        kvflash_sync_prefill(committed, req.prompt, /*kv_offset=*/0);
    }
    auto t_prefill_end = std::chrono::steady_clock::now();
    result.prefill_s = std::chrono::duration<double>(t_prefill_end - t_prefill_start).count();

    if (req.snap_slot >= 0 && req.snap_pos > 0 && req.snap_pos == committed) {
        if (compute_logits()) {
            int32_t last_tok = 0;
            float best = logits_buf[0];
            for (int j = 1; j < vocab; ++j) {
                if (logits_buf[(size_t)j] > best) {
                    best = logits_buf[(size_t)j];
                    last_tok = j;
                }
            }
            target_cache().last_tok = last_tok;
            if (snapshot_save(req.snap_slot)) {
                std::printf("[snap] hybrid boundary slot=%d cur_pos=%d\n",
                            req.snap_slot, committed);
                std::fflush(stdout);
            }
        } else {
            std::fprintf(stderr, "[snap] hybrid boundary logits failed at cur_pos=%d\n",
                         committed);
        }
    } else if (req.snap_slot >= 0 && req.snap_pos > 0) {
        std::fprintf(stderr,
                     "[snap] hybrid skip unsafe boundary slot=%d req_snap_pos=%d cur_pos=%d\n",
                     req.snap_slot, req.snap_pos, committed);
    }

    // ── Hybrid Decode ──
    if (req.n_gen > 0) {
        auto t_decode_start = std::chrono::steady_clock::now();

        // Hybrid spec-decode runs on the pool: hybrid_forward_batch is
        // slot-mapped (verify and replay both route through it) and the
        // recurrent-state rollback is ssm snapshot/restore (pool-neutral).
        const bool can_hybrid_spec = !req.force_ar_decode
            && cfg_.draft_path
            && !is_draft_parked()
            && feature_mirror().target_feat
            && sampler_config().temp == 0.0f
            && draft_weights().block_size > 0;

        if (can_hybrid_spec) {
            result.spec_decode_ran = true;
            // Sync prefill features to mirror before spec-decode
            if (target_cache().target_feat) {
                draft_feature_mirror_sync_range(target_cache().target_feat,
                                                target_cache().target_feat_cap,
                                                feature_mirror(), 0, committed);
            }

            // Get argmax from last prefill position for last_tok
            if (!compute_logits()) {
                result.error = "decode_logits";
                cleanup_graphs();
                return result;
            }
            int32_t first_tok = 0;
            float best = logits_buf[0];
            for (int j = 1; j < vocab; ++j) {
                if (logits_buf[(size_t)j] > best) { best = logits_buf[(size_t)j]; first_tok = j; }
            }
            target_cache().last_tok = first_tok;

            cleanup_graphs();
            if (!do_hybrid_spec_decode(committed, req.n_gen, result.tokens, out_io,
                                       &result.accept_rate)) {
                result.error = "hybrid_spec_decode";
                return result;
            }
        } else {
            // AR fallback decode — use pipelined path (cached DeltaNet + GPU-resident FFN)
            if (!ensure_pipe_state(committed)) {
                result.error = "pipe_state_init";
                cleanup_graphs();
                return result;
            }

            // Get logits from last prefill token (reuses persistent logits graph)
            if (!compute_logits()) {
                result.error = "decode_logits";
                cleanup_graphs();
                return result;
            }

            // Sample first token
            int32_t first_tok;
            if (sampler_config().temp > 0) {
                first_tok = sample_logits(logits_buf.data(), vocab, sampler_config(),
                                         result.tokens, sampler_rng_engine());
            } else {
                first_tok = 0;
                float best = logits_buf[0];
                for (int j = 1; j < vocab; ++j) {
                    if (logits_buf[(size_t)j] > best) { best = logits_buf[(size_t)j]; first_tok = j; }
                }
            }
            result.tokens.push_back(first_tok);
            out_io.emit(first_tok);
            if (!is_eos_tok(first_tok, target_weights())) {
                committed++;
                target_cache().cur_pos = committed;
                if (kvflash_active()) kvflash_history_.push_back(first_tok);

                // Pipelined decode loop
                PipelinedDecodeTelemetry decode_tel_accum{};
                for (int step = 1; step < req.n_gen; ++step) {
                    int32_t tok = result.tokens.back();
                    if (!target_weights().embedder.embed(&tok, 1, act_cur.data())) {
                        result.error = "decode_embed";
                        cleanup_graphs();
                        return result;
                    }
                    // Upload embedding async on compute stream
                    ggml_backend_tensor_set_async(target_backend(), pipe_state_->gpu_state.act_cur,
                                                  act_cur.data(), 0, sizeof(float) * (size_t)hidden);

                    // kvflash: pool slot for this token's KV rows (may evict)
                    int kv_slot = -1;
                    if (kvflash_active()) {
                        kv_slot = kvflash_pager_.slot_for(committed);
                        if (kv_slot < 0) {
                            result.error = "kvflash_slot";
                            cleanup_graphs();
                            return result;
                        }
                    }

                    PipelinedDecodeTelemetry tel;
                    if (!pipelined_decode_one_token(*pipe_state_, target_backend(), target_weights(),
                                                    target_cache(), *target_weights().moe_hybrid,
                                                    committed, cfg_.kq_stride_pad,
                                                    hybrid_telemetry_ ? &tel : nullptr,
                                                    kv_slot)) {
                        result.error = "decode";
                        cleanup_graphs();
                        return result;
                    }
                    if (hybrid_telemetry_) {
                        decode_tel_accum.total_us += tel.total_us;
                        decode_tel_accum.prefn_graph_build_us += tel.prefn_graph_build_us;
                        decode_tel_accum.prefn_compute_us += tel.prefn_compute_us;
                        decode_tel_accum.routing_readback_us += tel.routing_readback_us;
                        decode_tel_accum.ffn_us += tel.ffn_us;
                        decode_tel_accum.ffn_allhot_us += tel.ffn_allhot_us;
                        decode_tel_accum.ffn_mixed_us += tel.ffn_mixed_us;
                        decode_tel_accum.gpu_idle_us += tel.gpu_idle_us;
                        decode_tel_accum.tensor_io_us += tel.tensor_io_us;
                        decode_tel_accum.combine_overhead_us += tel.combine_overhead_us;
                        decode_tel_accum.cold_cpu_us += tel.cold_cpu_us;
                        decode_tel_accum.cold_compute_us += tel.cold_compute_us;
                        decode_tel_accum.hot_graph_build_us += tel.hot_graph_build_us;
                        decode_tel_accum.ffn_post_get_us += tel.ffn_post_get_us;
                        decode_tel_accum.sync_wait_us += tel.sync_wait_us;
                        decode_tel_accum.allhot_layers += tel.allhot_layers;
                        decode_tel_accum.mixed_layers += tel.mixed_layers;
                        decode_tel_accum.total_layers += tel.total_layers;
                        decode_tel_accum.hot_graph_rebuilds += tel.hot_graph_rebuilds;
                        decode_tel_accum.routed_ffn_layers += tel.routed_ffn_layers;
                        decode_tel_accum.routed_prefn_us += tel.routed_prefn_us;
                        decode_tel_accum.routed_sync_us += tel.routed_sync_us;
                        decode_tel_accum.routed_readback_us += tel.routed_readback_us;
                        decode_tel_accum.routed_cpu_remap_us += tel.routed_cpu_remap_us;
                        decode_tel_accum.routed_ffn_dispatch_us += tel.routed_ffn_dispatch_us;
                        decode_tel_accum.routed_final_sync_us += tel.routed_final_sync_us;
                        decode_tel_accum.routed_cold_expert_hits += tel.routed_cold_expert_hits;
                        decode_tel_accum.routed_total_expert_slots += tel.routed_total_expert_slots;
                    }

                    // act_cur stays on GPU — compute_logits reads it via GPU→GPU copy
                    if (!compute_logits(pipe_state_->gpu_state.act_cur)) {
                        result.error = "decode_logits";
                        cleanup_graphs();
                        return result;
                    }

                    int32_t next_tok;
                    if (sampler_config().temp > 0) {
                        next_tok = sample_logits(logits_buf.data(), vocab, sampler_config(),
                                                 result.tokens, sampler_rng_engine());
                    } else {
                        next_tok = 0;
                        float best = logits_buf[0];
                        for (int j = 1; j < vocab; ++j) {
                            if (logits_buf[(size_t)j] > best) { best = logits_buf[(size_t)j]; next_tok = j; }
                        }
                    }
                    result.tokens.push_back(next_tok);
                    out_io.emit(next_tok);
                    committed++;
                    target_cache().cur_pos = committed;
                    if (kvflash_active()) {
                        kvflash_history_.push_back(next_tok);
                        kvflash_maybe_reselect((int)result.tokens.size());
                    }
                    if (out_io.cancelled) break;
                    if (is_eos_tok(next_tok, target_weights())) break;
                }
                if (hybrid_telemetry_) {
                    const int n_dec = (int)result.tokens.size() - 1;
                    std::printf("[qwen35moe] === DECODE BREAKDOWN (n_tokens=%d) ===\n", n_dec);
                    std::printf("  prefn_build=%.1fms prefn_compute=%.1fms routing_readback=%.1fms ffn=%.1fms\n",
                                decode_tel_accum.prefn_graph_build_us / 1000.0,
                                decode_tel_accum.prefn_compute_us / 1000.0,
                                decode_tel_accum.routing_readback_us / 1000.0,
                                decode_tel_accum.ffn_us / 1000.0);
                    std::printf("  ffn_allhot=%.1fms ffn_mixed=%.1fms allhot_layers=%d mixed_layers=%d\n",
                                decode_tel_accum.ffn_allhot_us / 1000.0,
                                decode_tel_accum.ffn_mixed_us / 1000.0,
                                decode_tel_accum.allhot_layers,
                                decode_tel_accum.mixed_layers);
                    std::printf("  GPU IDLE: tensor_io=%.1fms combine=%.1fms sync_wait=%.1fms\n",
                                decode_tel_accum.tensor_io_us / 1000.0,
                                decode_tel_accum.combine_overhead_us / 1000.0,
                                decode_tel_accum.sync_wait_us / 1000.0);
                    std::printf("  CPU TIME: cold_total=%.1fms cold_compute=%.1fms hot_graph_build=%.1fms ffn_post_get=%.1fms\n",
                                decode_tel_accum.cold_cpu_us / 1000.0,
                                decode_tel_accum.cold_compute_us / 1000.0,
                                decode_tel_accum.hot_graph_build_us / 1000.0,
                                decode_tel_accum.ffn_post_get_us / 1000.0);
                    std::printf("  hot_graph_rebuilds=%d routed_ffn_layers=%d\n",
                                decode_tel_accum.hot_graph_rebuilds,
                                decode_tel_accum.routed_ffn_layers);
                    // Routed path breakdown
                    if (decode_tel_accum.routed_ffn_layers > 0) {
                        const int rl = decode_tel_accum.routed_ffn_layers;
                        std::printf("  ROUTED PATH (%d layer-evals, %d cold / %d slots = %.1f%% cold):\n",
                                    rl, decode_tel_accum.routed_cold_expert_hits,
                                    decode_tel_accum.routed_total_expert_slots,
                                    decode_tel_accum.routed_total_expert_slots > 0
                                        ? 100.0 * decode_tel_accum.routed_cold_expert_hits / decode_tel_accum.routed_total_expert_slots
                                        : 0.0);
                        std::printf("    per-layer: prefn=%.1fus sync=%.1fus readback=%.1fus remap=%.1fus ffn_dispatch=%.1fus\n",
                                    (double)decode_tel_accum.routed_prefn_us / rl,
                                    (double)decode_tel_accum.routed_sync_us / rl,
                                    (double)decode_tel_accum.routed_readback_us / rl,
                                    (double)decode_tel_accum.routed_cpu_remap_us / rl,
                                    (double)decode_tel_accum.routed_ffn_dispatch_us / rl);
                        std::printf("    totals: sync_stall=%.1fms final_sync=%.1fms\n",
                                    decode_tel_accum.routed_sync_us / 1000.0,
                                    decode_tel_accum.routed_final_sync_us / 1000.0);
                    }
                    if (n_dec > 0 && decode_tel_accum.total_us > 0) {
                        const double gpu_compute_us = (double)(decode_tel_accum.prefn_compute_us + decode_tel_accum.ffn_us - decode_tel_accum.cold_cpu_us);
                        const double gpu_util_pct = 100.0 * gpu_compute_us / (double)decode_tel_accum.total_us;
                        std::printf("  per-token avg: prefn_build=%.2fms prefn_compute=%.2fms readback=%.2fms ffn=%.2fms\n",
                                    decode_tel_accum.prefn_graph_build_us / 1000.0 / n_dec,
                                    decode_tel_accum.prefn_compute_us / 1000.0 / n_dec,
                                    decode_tel_accum.routing_readback_us / 1000.0 / n_dec,
                                    decode_tel_accum.ffn_us / 1000.0 / n_dec);
                        std::printf("  per-token avg: tensor_io=%.2fms combine=%.2fms cold_cpu=%.2fms cold_compute=%.2fms\n",
                                    decode_tel_accum.tensor_io_us / 1000.0 / n_dec,
                                    decode_tel_accum.combine_overhead_us / 1000.0 / n_dec,
                                    decode_tel_accum.cold_cpu_us / 1000.0 / n_dec,
                                    decode_tel_accum.cold_compute_us / 1000.0 / n_dec);
                        std::printf("  estimated GPU utilization: %.1f%%\n", gpu_util_pct);
                    }
                    std::fflush(stdout);
                }
            }
            cleanup_graphs();
        }

        result.decode_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_decode_start).count();
    } else {
        cleanup_graphs();
    }
    if (hybrid_telemetry_) {
        const int n_decode_toks = (int)result.tokens.size();
        std::printf("[qwen35moe] === PREFILL ANALYSIS (prompt_len=%d, chunk=%d) ===\n", prompt_len, prefill_chunk);
        std::printf("  prefill_total=%.1fms (%.1f tok/s)\n",
                    result.prefill_s * 1000.0, prompt_len / result.prefill_s);
        std::printf("  build=%.1fms compute=%.1fms readback=%.1fms ffn=%.1fms\n",
                    build_us_total / 1000.0, compute_us_total / 1000.0,
                    readback_us_total / 1000.0, ffn_us_total / 1000.0);
        if (total_ffn_layers > 0) {
            std::printf("  hot_only_layers=%d/%d (%.1f%% skip CPU)\n",
                        hot_only_layers, total_ffn_layers,
                        100.0 * hot_only_layers / total_ffn_layers);
        }
        const double prefill_total_us = (double)(build_us_total + compute_us_total + readback_us_total + ffn_us_total);
        if (prefill_total_us > 0) {
            std::printf("  pct: build=%.1f%% compute=%.1f%% readback=%.1f%% ffn=%.1f%%\n",
                        100.0 * build_us_total / prefill_total_us,
                        100.0 * compute_us_total / prefill_total_us,
                        100.0 * readback_us_total / prefill_total_us,
                        100.0 * ffn_us_total / prefill_total_us);
        }
        std::printf("[qwen35moe] === DECODE (pipelined, n_tokens=%d) ===\n", n_decode_toks);
        if (result.decode_s > 0) {
            std::printf("  decode_total=%.1fms (%.2f tok/s)\n",
                        result.decode_s * 1000.0, n_decode_toks / result.decode_s);
        }
        std::printf("[qwen35moe] === FFN BREAKDOWN (prefill) ===\n");
        std::printf("  hot_gpu=%.1fms cold_cpu=%.1fms partition=%.1fms combine=%.1fms\n",
                    ffn_tel_accum.hot_us / 1000.0, ffn_tel_accum.cold_us / 1000.0,
                    ffn_tel_accum.partition_us / 1000.0, ffn_tel_accum.combine_us / 1000.0);
        std::printf("  hot_expert_selections=%d cold_expert_selections=%d\n",
                    ffn_tel_accum.hot_selected, ffn_tel_accum.cold_selected);
        std::fflush(stdout);
    }
    result.ok = true;
    if (result.ok) maybe_post_request_swap();
    return result;
}

GenerateResult Qwen35MoeBackend::restore_and_generate_impl(int slot,
                                                      const GenerateRequest & req,
                                                      const DaemonIO & io) {
    if (!target_weights().moe_hybrid) {
        auto result = Qwen35Backend::restore_and_generate_impl(slot, req, io);
        if (result.ok) maybe_post_request_swap();
        return result;
    }
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    if (slot < 0 || !snapshot_used(slot)) {
        result.error = "bad slot";
        out_io.emit(-1);
        return result;
    }

    sampler_config() = req.sampler;
    if (req.do_sample && sampler_config().seed != 0) {
        sampler_rng_engine().seed(sampler_config().seed);
    }

    if (req.n_gen > 0 && sampler_config().temp > 0) {
        std::fprintf(stderr,
                     "[qwen35moe] hybrid snapshot restore falls back to full generate for temp sampling\n");
        return generate_impl(req, io);
    }

    pipe_state_.reset();
    for (auto & layer : target_weights().moe_hybrid->layers) {
        layer.hot_graph.free();
        layer.cold_graph.free();
    }

    if (!restore_target_cache_from_snapshot(slot)) {
        result.error = "restore";
        out_io.emit(-1);
        return result;
    }

    const int snap_pos = snapshot_cur_pos(slot);
    int committed = snap_pos;
    target_cache().cur_pos = committed;

    const int prompt_len = (int)req.prompt.size();
    if (prompt_len > 0 && prompt_len < snap_pos) {
        result.error = "snapshot_longer_than_prompt";
        out_io.emit(-1);
        return result;
    }

    // kvflash: the restored prefix + delta prefill land identity-mapped, so
    // the full prompt must fit the pool (snapshots past the pool are never
    // saved, but the delta can still overflow it).
    if (kvflash_active() &&
        prompt_len > kvflash_tokens_ - kvflash_pager_.chunk_tokens()) {
        std::fprintf(stderr,
            "[kvflash] hybrid restore prompt (%d) exceeds pool %d; raise "
            "--kvflash\n", prompt_len, kvflash_tokens_);
        result.error = "kvflash: prompt exceeds resident pool";
        out_io.emit(-1);
        return result;
    }

    // kvflash: the delta prefill below runs the maskless pipelined forward
    // over the padded pool span; map the restored prefix identity-style and
    // zero stale free slots BEFORE any forward reads them.
    if (kvflash_active()) {
        kvflash_pager_.reset();
        if (!kvflash_pager_.alloc_span(0, snap_pos)) {
            result.error = "kvflash_slot";
            out_io.emit(-1);
            return result;
        }
        kvflash_pager_.zero_free_blocks();
    }

    const int hidden = target_weights().n_embd;
    std::vector<float> act_cur((size_t)hidden);
    if (prompt_len > snap_pos) {
        auto t_prefill_start = std::chrono::steady_clock::now();
        for (int i = snap_pos; i < prompt_len; ++i) {
            int32_t argmax = -1;
            if (!hybrid_forward_one_token(req.prompt[(size_t)i], committed,
                                          act_cur, argmax)) {
                result.error = "prefill_delta";
                return result;
            }
            committed++;
            target_cache().cur_pos = committed;
            target_cache().last_tok = argmax;
        }
        result.prefill_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_prefill_start).count();
    }

    if (kvflash_active()) {
        // Rebuild the pager mapping over the identity-mapped [0, committed).
        // With the full prompt available the history carries real ids;
        // restore-only generates keep an unknown-prefix history.
        if (prompt_len == committed) {
            kvflash_sync_prefill(committed, req.prompt, /*kv_offset=*/0);
        } else {
            kvflash_sync_prefill(committed, {}, /*kv_offset=*/committed);
        }
    }

    if (req.n_gen > 0) {
        if (target_cache().last_tok < 0) {
            std::fprintf(stderr,
                         "[qwen35moe] hybrid snapshot has no last token; falling back to full generate\n");
            return generate_impl(req, io);
        }
        auto t_decode_start = std::chrono::steady_clock::now();
        if (!run_pipelined_decode_path(committed, req.n_gen, result.tokens, out_io)) {
            result.error = "decode";
            return result;
        }
        result.decode_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_decode_start).count();
    }

    result.ok = true;
    if (result.ok) maybe_post_request_swap();
    return result;
}

// ── Hybrid spec-decode: draft → verify via hybrid forward → accept ──────────

bool Qwen35MoeBackend::hybrid_forward_one_token(int32_t tok, int kv_pos,
                                                 std::vector<float> & act_cur,
                                                 int32_t & argmax_out,
                                                 std::vector<float> * logits_out) {
    const int hidden = target_weights().n_embd;

    // Embed the token
    if (!target_weights().embedder.embed(&tok, 1, act_cur.data())) return false;

    // Ensure pipelined state
    if (!ensure_pipe_state(kv_pos)) return false;

    // Upload to GPU-resident act_cur (async — compute stream ordering guarantees correctness)
    ggml_backend_tensor_set_async(target_backend(), pipe_state_->gpu_state.act_cur, act_cur.data(), 0,
                            sizeof(float) * (size_t)hidden);

    // Run pipelined decode (all 40 layers with cached DeltaNet + hot/cold FFN)
    if (!pipelined_decode_one_token(*pipe_state_, target_backend(), target_weights(),
                                    target_cache(), *target_weights().moe_hybrid,
                                    kv_pos, cfg_.kq_stride_pad, nullptr)) {
        return false;
    }

    // Read back act_cur for feature capture + logits
    ggml_backend_tensor_get(pipe_state_->gpu_state.act_cur, act_cur.data(), 0,
                            sizeof(float) * (size_t)hidden);

    // Feature capture: write act_cur (F32) → cache_.target_feat (BF16)
    if (target_cache().target_feat) {
        const int cap = target_cache().target_feat_cap;
        const int slot = kv_pos % cap;
        const size_t elt = ggml_element_size(target_cache().target_feat);
        const size_t col_stride = target_cache().target_feat->nb[1];
        // Convert once — all capture layers store the same final hidden state
        std::vector<ggml_bf16_t> bf16_buf((size_t)hidden);
        ggml_fp32_to_bf16_row(act_cur.data(), bf16_buf.data(), hidden);
        for (int k = 0; k < target_weights().n_capture_layers; k++) {
            const size_t offset = (size_t)slot * col_stride +
                                  (size_t)k * (size_t)hidden * elt;
            ggml_backend_tensor_set(target_cache().target_feat, bf16_buf.data(),
                                     offset, (size_t)hidden * elt);
        }
    }

    // Project to logits and get argmax
    const int vocab = target_weights().n_vocab;
    StepGraph proj_sg;
    ggml_init_params ip{};
    ip.mem_size = 64 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    proj_sg.ctx = ggml_init(ip);
    if (!proj_sg.ctx) return false;
    proj_sg.hidden_input = ggml_new_tensor_3d(proj_sg.ctx, GGML_TYPE_F32, hidden, 1, 1);
    ggml_set_input(proj_sg.hidden_input);
    proj_sg.gf = ggml_new_graph_custom(proj_sg.ctx, 1024, false);
    ggml_tensor * normed = ggml_rms_norm(
        proj_sg.ctx,
        rms_norm_input_f32(proj_sg.ctx, proj_sg.hidden_input),
        target_weights().rms_eps);
    normed = ggml_mul(
        proj_sg.ctx, normed,
        graph_tensor_f32(proj_sg.ctx, target_weights().out_norm));
    proj_sg.logits = ggml_mul_mat(proj_sg.ctx, target_weights().output, normed);
    ggml_set_output(proj_sg.logits);
    ggml_build_forward_expand(proj_sg.gf, proj_sg.logits);
    proj_sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend()));
    if (!ggml_gallocr_alloc_graph(proj_sg.alloc, proj_sg.gf)) {
        step_graph_destroy(proj_sg);
        return false;
    }
    ggml_backend_tensor_set(proj_sg.hidden_input, act_cur.data(), 0, sizeof(float) * (size_t)hidden);
    auto proj_st = ggml_backend_graph_compute(target_backend(), proj_sg.gf);
    if (proj_st != GGML_STATUS_SUCCESS) {
        step_graph_destroy(proj_sg);
        return false;
    }
    std::vector<float> logits_buf((size_t)vocab);
    ggml_backend_tensor_get(proj_sg.logits, logits_buf.data(), 0, sizeof(float) * (size_t)vocab);
    step_graph_destroy(proj_sg);
    if (logits_out) {
        *logits_out = logits_buf;
    }

    // Argmax
    argmax_out = 0;
    float best = logits_buf[0];
    for (int j = 1; j < vocab; ++j) {
        if (logits_buf[(size_t)j] > best) {
            best = logits_buf[(size_t)j];
            argmax_out = j;
        }
    }
    return true;
}

// ── Batched hybrid forward ─────────────────────────────────────────────────
// Processes all tokens layer-by-layer using the same approach as prefill:
//   per layer: build_layer_prefn_step (DeltaNet + router) → MoE FFN (batched)
// Then project all tokens to logits and take argmax.
// This is ~22× fewer dispatches than sequential hybrid_forward_one_token.

bool Qwen35MoeBackend::hybrid_forward_batch(
    const int32_t * tokens, int n_tokens, int base_pos,
    std::vector<float> & act_cur,
    std::vector<int32_t> & argmax_out,
    bool capture_features) {

    const int hidden = target_weights().n_embd;
    const int n_layer = target_weights().n_layer;
    const int n_expert_used = target_weights().n_expert_used;

    // Embed all tokens
    std::vector<float> embed_all((size_t)n_tokens * (size_t)hidden);
    for (int i = 0; i < n_tokens; ++i) {
        if (!target_weights().embedder.embed(&tokens[i], 1,
                embed_all.data() + (size_t)i * (size_t)hidden)) {
            return false;
        }
    }

    // kvflash: allocate the block's slots up front (may evict) and build
    // the slot-mapped write rows + slot-space mask once; every layer's
    // graph gets the same fills (verify and replay both land here, so all
    // hybrid-spec KV writes are pool-routed).
    const bool kvf = kvflash_active();
    std::vector<int64_t> kvf_rows;
    std::vector<uint16_t> kvf_mask;
    std::vector<int> kvf_slots;
    if (kvf) {
        if (!kvflash_pager_.alloc_span(base_pos, n_tokens)) return false;
        kvf_slots.resize((size_t)n_tokens);
        for (int i = 0; i < n_tokens; ++i) {
            kvf_slots[(size_t)i] = kvflash_pager_.slot_of(base_pos + i);
        }
        // [n_tokens, n_head_kv] ne0-major (see verify_batch).
        kvf_rows.resize((size_t)n_tokens * target_weights().n_head_kv);
        for (int h = 0; h < target_weights().n_head_kv; ++h) {
            for (int i = 0; i < n_tokens; ++i) {
                kvf_rows[(size_t)h * n_tokens + i] = kvf_slots[(size_t)i];
            }
        }
    }

    // Process layer-by-layer (same as prefill)
    StepGraph prefn_sg;
    ggml_gallocr_t ffn_hot_alloc = nullptr;

    MoeHybridConfig chunk_cfg = make_moe_hybrid_config(target_weights());

    for (int il = 0; il < n_layer; ++il) {
        auto & storage = target_weights().moe_hybrid->layers[(size_t)il];

        const bool with_mask = kvf ||
            (cfg_.kq_stride_pad > KQ_MASK_PAD) || (n_tokens > 1);

        // Build pre-FFN graph (DeltaNet/attention + router) for all tokens
        step_graph_free(prefn_sg);
        if (!build_layer_prefn_step(prefn_sg, target_weights(), target_cache(), target_backend(),
                                    il, /*kv_start=*/base_pos, n_tokens,
                                    with_mask, /*fa_window=*/0, cfg_.kq_stride_pad,
                                    /*kvflash=*/kvf)) {
            step_graph_destroy(prefn_sg);
            if (ffn_hot_alloc) ggml_gallocr_free(ffn_hot_alloc);
            return false;
        }
        if (prefn_sg.kv_write_rows) {
            ggml_backend_tensor_set(prefn_sg.kv_write_rows, kvf_rows.data(), 0,
                                    sizeof(int64_t) * kvf_rows.size());
        }

        // Upload embeddings
        ggml_backend_tensor_set(prefn_sg.inp_embed, embed_all.data(), 0,
                                sizeof(float) * (size_t)n_tokens * (size_t)hidden);

        // Set positions for attention layers
        if (prefn_sg.positions) {
            std::vector<int32_t> pos_data((size_t)n_tokens * 4);
            for (int i = 0; i < n_tokens; ++i) {
                pos_data[(size_t)i * 4 + 0] = base_pos + i;
                pos_data[(size_t)i * 4 + 1] = base_pos + i;
                pos_data[(size_t)i * 4 + 2] = base_pos + i;
                pos_data[(size_t)i * 4 + 3] = 0;
            }
            ggml_backend_tensor_set(prefn_sg.positions, pos_data.data(), 0,
                                    sizeof(int32_t) * pos_data.size());
        }

        // Set causal mask
        if (prefn_sg.attn_mask && kvf) {
            // Slot-space mask (verify_batch recipe): committed resident
            // positions (< base_pos) plus this block's own slots, causal.
            // Built once, reused for every layer's graph.
            if (kvf_mask.empty()) {
                constexpr uint16_t F16_ZERO = 0x0000, F16_NEG_INF = 0xFC00;
                const size_t kvd = (size_t)prefn_sg.attn_mask->ne[0];
                const int q_pad = (int)prefn_sg.attn_mask->ne[1];
                kvf_mask.assign(kvd * q_pad, F16_NEG_INF);
                const int ct = kvflash_pager_.chunk_tokens();
                for (int c = 0; c < kvflash_pager_.n_chunks(); c++) {
                    const int blk = kvflash_pager_.block_of(c);
                    if (blk < 0) continue;
                    for (int i = 0; i < ct; i++) {
                        if ((int64_t)c * ct + i >= base_pos) break;
                        kvf_mask[(size_t)blk * ct + i] = F16_ZERO;
                    }
                }
                for (int q = 1; q < n_tokens; q++) {
                    std::memcpy(kvf_mask.data() + (size_t)q * kvd, kvf_mask.data(), kvd * 2);
                }
                for (int q = 0; q < n_tokens; q++) {
                    for (int i = 0; i <= q; i++) {
                        kvf_mask[(size_t)q * kvd + kvf_slots[(size_t)i]] = F16_ZERO;
                    }
                }
            }
            ggml_backend_tensor_set(prefn_sg.attn_mask, kvf_mask.data(), 0,
                                    sizeof(uint16_t) * kvf_mask.size());
        } else if (prefn_sg.attn_mask) {
            const int kv_len = base_pos + n_tokens;
            const int kv_pad_override = (int)prefn_sg.attn_mask->ne[0];
            std::vector<uint16_t> mask_buf;
            build_causal_mask(mask_buf, kv_len, n_tokens, /*kv_start=*/base_pos,
                              cfg_.kq_stride_pad, /*win_start=*/0, kv_pad_override);
            ggml_backend_tensor_set(prefn_sg.attn_mask, mask_buf.data(), 0,
                                    sizeof(uint16_t) * mask_buf.size());
        }

        // Compute pre-FFN (DeltaNet + router for all tokens in one dispatch)
        auto st = ggml_backend_graph_compute(target_backend(), prefn_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            step_graph_destroy(prefn_sg);
            if (ffn_hot_alloc) ggml_gallocr_free(ffn_hot_alloc);
            return false;
        }

        // Readback results
        std::vector<float> chunk_residuals((size_t)n_tokens * (size_t)hidden);
        std::vector<float> chunk_post((size_t)n_tokens * (size_t)hidden);
        std::vector<int32_t> chunk_selected((size_t)n_tokens * (size_t)n_expert_used);
        std::vector<float> chunk_weights((size_t)n_tokens * (size_t)n_expert_used);

        ggml_backend_tensor_get(prefn_sg.ffn_residual, chunk_residuals.data(), 0,
                                sizeof(float) * chunk_residuals.size());
        ggml_backend_tensor_get(prefn_sg.ffn_post, chunk_post.data(), 0,
                                sizeof(float) * chunk_post.size());

        ggml_tensor * layer_selected = (!prefn_sg.moe_selected.empty() && (size_t)il < prefn_sg.moe_selected.size())
            ? prefn_sg.moe_selected[(size_t)il] : nullptr;
        if (!layer_selected || !prefn_sg.moe_weights) {
            step_graph_destroy(prefn_sg);
            if (ffn_hot_alloc) ggml_gallocr_free(ffn_hot_alloc);
            return false;
        }
        ggml_backend_tensor_get(layer_selected, chunk_selected.data(), 0,
                                sizeof(int32_t) * chunk_selected.size());
        ggml_backend_tensor_get(prefn_sg.moe_weights, chunk_weights.data(), 0,
                                sizeof(float) * chunk_weights.size());

        // MoE FFN — batched
        MoeLayerDesc chunk_desc = make_moe_layer_desc(target_weights().layers[(size_t)il]);
        std::vector<float> ffn_batch_out;
        bool ffn_ok = false;

        // Spark expert cache: pull the verify batch's selected cold experts into
        // spare GPU slots (LRU) so the batched FFN serves them on-die — the SAME
        // residency mechanism the AR pipelined path uses. Without this the verify
        // re-evaluated cold experts on the CPU every step, which dominated its FFN
        // time (the spec-decode-with-offloading inefficiency). After warmup the
        // working set is resident and the CPU cold path is rarely taken.
        const int n_route_slots = n_tokens * n_expert_used;
        if (storage.cache_slots > 0 && !storage.cold_expert_ids.empty()) {
            for (int i = 0; i < n_route_slots; ++i)
                dflash::common::moe_hybrid_cache_swap_in(storage, chunk_selected[(size_t)i], target_backend());
        }
        const bool routed_all_hot = storage.cold_expert_ids.empty()
            || storage.all_routed_are_hot(chunk_selected.data(), n_route_slots);
        if (routed_all_hot) {
            // All routed experts resident on GPU: fast batched hot-only path.
            ffn_ok = eval_moe_hot_only_batched(
                target_backend(), chunk_cfg, chunk_desc, storage,
                chunk_post.data(), chunk_selected.data(), chunk_weights.data(),
                n_tokens, ffn_batch_out, nullptr, &ffn_hot_alloc);
        } else {
            // Cache full / residue still cold: hybrid path (remaining cold on CPU).
            ffn_ok = eval_moe_hybrid_ffn_batched(
                target_backend(), target_weights().moe_hybrid->cpu_backend,
                chunk_cfg, chunk_desc, storage,
                chunk_post.data(), chunk_selected.data(), chunk_weights.data(),
                n_tokens, ffn_batch_out, nullptr, &ffn_hot_alloc, nullptr);
        }

        if (!ffn_ok) {
            // Per-token fallback
            ffn_batch_out.assign((size_t)hidden * (size_t)n_tokens, 0.0f);
            std::vector<float> single_out;
            for (int ti = 0; ti < n_tokens; ++ti) {
                if (!eval_moe_hybrid_ffn_single(
                        target_backend(), chunk_cfg, chunk_desc, storage,
                        target_weights().moe_hybrid->cpu_backend,
                        chunk_post.data() + (size_t)ti * (size_t)hidden,
                        chunk_selected.data() + (size_t)ti * (size_t)n_expert_used,
                        chunk_weights.data() + (size_t)ti * (size_t)n_expert_used,
                        n_expert_used, single_out, nullptr)) {
                    step_graph_destroy(prefn_sg);
                    if (ffn_hot_alloc) ggml_gallocr_free(ffn_hot_alloc);
                    return false;
                }
                std::memcpy(ffn_batch_out.data() + (size_t)ti * (size_t)hidden,
                            single_out.data(), sizeof(float) * (size_t)hidden);
            }
        }

        // Combine FFN + residual → embed_all for next layer
        for (int i = 0; i < n_tokens; ++i) {
            const float * ffn = ffn_batch_out.data() + (size_t)i * (size_t)hidden;
            const float * res = chunk_residuals.data() + (size_t)i * (size_t)hidden;
            float * emb = embed_all.data() + (size_t)i * (size_t)hidden;
            for (int j = 0; j < hidden; ++j) {
                emb[j] = ffn[j] + res[j];
            }

            // Feature capture at capture layers
            if (capture_features && target_cache().target_feat && cfg_.draft_path) {
                int capture_idx = -1;
                for (int k = 0; k < target_weights().n_capture_layers; k++) {
                    if (target_weights().capture_layer_ids[k] == il) {
                        capture_idx = k;
                        break;
                    }
                }
                if (capture_idx >= 0) {
                    const int token_pos = base_pos + i;
                    const int cap = target_cache().target_feat_cap;
                    const int slot = token_pos % cap;
                    const size_t elt = ggml_element_size(target_cache().target_feat);
                    const size_t col_stride = target_cache().target_feat->nb[1];
                    const size_t offset = (size_t)slot * col_stride +
                                          (size_t)capture_idx * (size_t)hidden * elt;
                    std::vector<ggml_bf16_t> bf16_tmp((size_t)hidden);
                    ggml_fp32_to_bf16_row(emb, bf16_tmp.data(), hidden);
                    ggml_backend_tensor_set(target_cache().target_feat, bf16_tmp.data(),
                                            offset, (size_t)hidden * elt);
                }
            }
        }
    }
    step_graph_destroy(prefn_sg);
    if (ffn_hot_alloc) ggml_gallocr_free(ffn_hot_alloc);

    // Store last token hidden state in act_cur
    act_cur.assign(embed_all.data() + (size_t)(n_tokens - 1) * (size_t)hidden,
                   embed_all.data() + (size_t)n_tokens * (size_t)hidden);

    // Project ALL tokens to logits and argmax ON THE GPU, reading back only
    // n_tokens token ids instead of vocab*n_tokens floats. The host logits
    // readback + host argmax was a large per-step D2H cost in the verify and
    // replay forwards (vocab ~152k x n_tokens x 4B, twice per spec step).
    argmax_out.resize(n_tokens);
    StepGraph proj_sg;
    if (!build_lm_head_projection_step(proj_sg, target_weights(), target_backend(), n_tokens)) {
        return false;
    }
    ggml_backend_tensor_set(proj_sg.hidden_input, embed_all.data(), 0,
                            sizeof(float) * (size_t)n_tokens * (size_t)hidden);
    auto proj_st = ggml_backend_graph_compute(target_backend(), proj_sg.gf);
    if (proj_st != GGML_STATUS_SUCCESS) {
        step_graph_destroy(proj_sg);
        return false;
    }
    ggml_backend_tensor_get(proj_sg.argmax_tokens, argmax_out.data(), 0,
                            sizeof(int32_t) * (size_t)n_tokens);
    step_graph_destroy(proj_sg);
    return true;
}

bool Qwen35MoeBackend::do_hybrid_spec_decode(int committed, int n_gen,
                                              std::vector<int32_t> & out_tokens,
                                              const DaemonIO & io,
                                              float * accept_rate_out) {
    const int hidden = target_weights().n_embd;
    const int q_len = draft_weights().block_size;
    if (q_len <= 0) return false;

    // Verify width: cap how many draft tokens we actually verify. The batched
    // verify's cost is dominated by the distinct experts its tokens touch
    // (especially under --spark expert offload, where extra tokens stream extra
    // cold experts). Tokens past the realized accept length are wasted, so
    // capping the verify to a width above the typical accept length cuts that
    // waste at no acceptance cost. Default = full draft block; tune via env.
    // Verify-width control (see note above). DFLASH_VERIFY_WIDTH pins a fixed
    // width; otherwise the width adapts to the realized accept length so chain
    // decoding (low AL) verifies just a few tokens (cheap, especially under
    // expert offload) while a high-AL draft still gets enough width.
    const int forced_verify_width = [&]{
        const char * e = std::getenv("DFLASH_VERIFY_WIDTH");
        return e ? std::max(1, std::min(q_len, std::atoi(e))) : 0;
    }();
    int observed_max_accept = 1;

    int32_t last_tok = target_cache().last_tok;
    std::vector<float> act_cur((size_t)hidden);

    StepGraph draft_sg;
    std::vector<float>   noise_embed((size_t)hidden * q_len);
    std::vector<int32_t> noise_ids(q_len);
    std::vector<int32_t> draft_tok(q_len);
    std::vector<int32_t> target_tok(q_len);
    std::vector<int32_t> pos_q(q_len);
    std::vector<int32_t> pos_k;
    std::vector<float>   local_hidden;

    int n_generated = 0;
    int n_draft_steps = 0;
    int n_accept_sum = 0;

    // Allocate DeltaNet rollback snapshot tensors (no-op if already present).
    // Without these, snapshot_ssm_state/restore_ssm_state silently do nothing
    // and rejected draft tokens leak into the recurrent state, collapsing output.
    if (!ensure_ssm_snapshot(target_cache(), target_backend())) {
        std::fprintf(stderr, "[hybrid-spec] ensure_ssm_snapshot failed\n");
        step_graph_destroy(draft_sg);
        return false;
    }

    auto t_dec0 = std::chrono::steady_clock::now();

    while (n_generated < n_gen) {
        const int need_commit_budget = n_gen - n_generated;
        const int verify_width = forced_verify_width > 0
            ? forced_verify_width
            : std::min(q_len, std::max(6, observed_max_accept + 2));

        // 1. Build noise input for draft
        noise_ids[0] = last_tok;
        for (int i = 1; i < q_len; i++) noise_ids[i] = target_weights().mask_token_id;
        if (!target_weights().embedder.embed(noise_ids.data(), q_len, noise_embed.data())) {
            std::fprintf(stderr, "[hybrid-spec] noise embed failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }

        // 2. Draft compute
        constexpr int DRAFT_CTX_MAX_DEFAULT = 2048;
        const int ring_cap = feature_mirror().cap;
        const int draft_ctx = std::min(committed,
            std::min(ring_cap, std::max(DRAFT_CTX_MAX_DEFAULT, cfg_.draft_ctx_max)));
        const int draft_start = committed - draft_ctx;
        int mirror_slot0 = 0;
        const bool use_mirror_view =
            draft_feature_mirror_can_view(feature_mirror(), committed, draft_ctx, mirror_slot0);

        if (!build_draft_step(draft_sg, draft_weights(), /*lm_head=*/nullptr, draft_backend(),
                              draft_ctx, use_mirror_view ? &feature_mirror() : nullptr,
                              committed,
                              std::min(ring_cap, std::max(DRAFT_CTX_MAX_DEFAULT, cfg_.draft_ctx_max)))) {
            std::fprintf(stderr, "[hybrid-spec] draft build failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }
        if (!use_mirror_view &&
            !copy_feature_ring_range_to_tensor(feature_mirror(), draft_sg.target_hidden_cat,
                                               draft_start, draft_ctx)) {
            std::fprintf(stderr, "[hybrid-spec] feature copy failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }
        ggml_backend_tensor_set(draft_sg.inp_embed, noise_embed.data(), 0,
                                 sizeof(float) * noise_embed.size());
        pos_k.resize((size_t)draft_ctx + q_len);
        for (int i = 0; i < q_len; i++) pos_q[i] = draft_ctx + i;
        for (int i = 0; i < draft_ctx + q_len; i++) pos_k[i] = i;
        ggml_backend_tensor_set(draft_sg.positions, pos_q.data(), 0,
                                 sizeof(int32_t) * pos_q.size());
        ggml_backend_tensor_set(draft_sg.positions_k, pos_k.data(), 0,
                                 sizeof(int32_t) * pos_k.size());

        auto st = ggml_backend_graph_compute(draft_backend(), draft_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[hybrid-spec] draft compute failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }

        // Read draft hidden states
        local_hidden.resize((size_t)hidden * q_len);
        ggml_backend_tensor_get(draft_sg.hidden_states, local_hidden.data(), 0,
                                 sizeof(float) * local_hidden.size());

        // 3. Project draft hidden → token IDs via target LM head
        // Use a simple LM head projection graph
        {
            StepGraph proj_sg;
            if (!build_lm_head_projection_step(proj_sg, target_weights(), target_backend(), q_len)) {
                std::fprintf(stderr, "[hybrid-spec] projection build failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }
            ggml_backend_tensor_set(proj_sg.hidden_input, local_hidden.data(), 0,
                                     sizeof(float) * local_hidden.size());
            auto ps = ggml_backend_graph_compute(target_backend(), proj_sg.gf);
            if (ps != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[hybrid-spec] projection compute failed\n");
                step_graph_destroy(proj_sg);
                step_graph_destroy(draft_sg);
                return false;
            }
            draft_tok.resize(q_len);
            ggml_backend_tensor_get(proj_sg.argmax_tokens, draft_tok.data(), 0,
                                     sizeof(int32_t) * q_len);
            step_graph_destroy(proj_sg);
        }
        draft_tok[0] = last_tok;

        // 4. Verify: snapshot recurrent state, then run ALL draft tokens batched
        snapshot_ssm_state(target_cache());

        target_tok.resize(verify_width);
        bool verify_ok = hybrid_forward_batch(
            draft_tok.data(), verify_width, committed,
            act_cur, target_tok, /*capture_features=*/false);
        if (!verify_ok) {
            std::fprintf(stderr, "[hybrid-spec] verify failed\n");
            restore_ssm_state(target_cache());
            step_graph_destroy(draft_sg);
            return false;
        }

        // 5. Acceptance: longest matching prefix
        int accept_n = 1;
        for (int i = 0; i < verify_width - 1; i++) {
            if (draft_tok[i + 1] == target_tok[i]) accept_n++;
            else break;
        }
        int bonus_tok = (accept_n < verify_width) ? target_tok[accept_n - 1] : -1;
        observed_max_accept = std::max(observed_max_accept, accept_n);
        int commit_n = accept_n + (bonus_tok >= 0 ? 1 : 0);
        if (commit_n > need_commit_budget) {
            commit_n = need_commit_budget;
            if (commit_n <= accept_n) bonus_tok = -1;
        }

        // 6. Restore and replay accepted tokens
        restore_ssm_state(target_cache());

        std::vector<int32_t> replay_tok((size_t)commit_n);
        for (int i = 0; i < commit_n; i++) {
            replay_tok[i] = (i < accept_n) ? draft_tok[i] : bonus_tok;
        }

        // Replay tokens through batched hybrid forward (captures features for next draft step)
        std::vector<int32_t> replay_argmax;
        if (!hybrid_forward_batch(replay_tok.data(), commit_n, committed,
                                  act_cur, replay_argmax, /*capture_features=*/true)) {
            std::fprintf(stderr, "[hybrid-spec] replay failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }
        last_tok = replay_argmax[commit_n - 1];

        // 7. Sync features to mirror for next draft step
        if (feature_mirror().target_feat && target_cache().target_feat) {
            draft_feature_mirror_sync_range(target_cache().target_feat,
                                             target_cache().target_feat_cap,
                                             feature_mirror(), committed, commit_n);
        }

        // 8. Emit committed tokens
        bool hit_eos = false;
        int emitted = 0;
        for (int i = 0; i < commit_n; i++) {
            out_tokens.push_back(replay_tok[i]);
            io.emit(replay_tok[i]);
            emitted++;
            if (io.cancelled) break;
            if (is_eos_tok(replay_tok[i], target_weights())) { hit_eos = true; break; }
        }
        committed += emitted;
        target_cache().cur_pos = committed;
        // Keep the kvflash pool history + scorer reselection in sync during spec
        // decode (the AR paths do this); otherwise the pool evicts on stale,
        // prompt-only history under --spark --draft --kvflash.
        if (kvflash_active()) {
            for (int i = 0; i < emitted; i++) kvflash_history_.push_back(replay_tok[(size_t)i]);
            kvflash_maybe_reselect((int)out_tokens.size());
        }
        n_generated += emitted;
        n_accept_sum += std::min(accept_n, emitted);
        n_draft_steps++;
        if (io.cancelled) break;
        if (hit_eos) break;
    }

    step_graph_destroy(draft_sg);

    auto t_dec1 = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    const int total_draft_pos = std::max(1, n_draft_steps * q_len);
    const double accept_pct = 100.0 * (double)n_accept_sum / (double)total_draft_pos;
    if (accept_rate_out) {
        *accept_rate_out = total_draft_pos > 0
            ? (float)((double)n_accept_sum / (double)total_draft_pos) : 0.0f;
    }
    std::fprintf(stderr, "[hybrid-spec] tokens=%d time=%.3f s speed=%.2f tok/s "
                 "steps=%d accepted=%d/%d (%.1f%%) avg_commit=%.2f AL=%.2f\n",
                 n_generated, decode_s,
                 n_generated > 0 ? n_generated / decode_s : 0.0,
                 n_draft_steps, n_accept_sum, total_draft_pos, accept_pct,
                 n_draft_steps > 0 ? (double)n_generated / (double)n_draft_steps : 0.0,
                 n_draft_steps > 0 ? (double)n_accept_sum / (double)n_draft_steps : 0.0);

    io.emit(-1);
    return true;
}

bool Qwen35MoeBackend::load_dynamic_placement(const char * hotness_path,
                                               ggml_backend_t backend,
                                               const TargetWeights & w,
                                               MoeHybridPlacement & out,
                                               std::string * err) {
    // Load hotness table or assume uniform hotness
    MoeHybridRoutingStats hotness;
    if (hotness_path && hotness_path[0]) {
        if (!MoeHybridRoutingStats::load_csv(std::string(hotness_path), hotness, err)) {
            return false;
        }
        if (hotness.n_layer != w.n_layer || hotness.n_expert != w.n_expert) {
            if (err) *err = "hotness table dimensions don't match model (n_layer=" +
                std::to_string(hotness.n_layer) + " vs " + std::to_string(w.n_layer) +
                ", n_expert=" + std::to_string(hotness.n_expert) + " vs " + std::to_string(w.n_expert) + ")";
            return false;
        }
    } else {
        // No hotness file: assume uniform activation (all experts equally hot)
        hotness.n_layer = w.n_layer;
        hotness.n_expert = w.n_expert;
        hotness.n_expert_used = w.n_expert_used;
        hotness.counts.assign((size_t)w.n_layer * (size_t)w.n_expert, 1);
        hotness.layer_totals.assign((size_t)w.n_layer, (uint64_t)w.n_expert);
    }

    // Spark: seed the live accumulator from the loaded profile so calibration
    // accumulates across restarts instead of resetting to zero each boot.
    if (routing_stats_ && hotness_path && hotness_path[0] &&
        hotness.counts.size() == (size_t)w.n_layer * (size_t)w.n_expert) {
        routing_stats_->counts = hotness.counts;
        routing_stats_->layer_totals.assign((size_t)w.n_layer, 0);
        for (int il = 0; il < w.n_layer; ++il)
            for (int ie = 0; ie < w.n_expert; ++ie)
                routing_stats_->layer_totals[(size_t)il] +=
                    hotness.counts[(size_t)il * (size_t)w.n_expert + ie];
    }

    // Query GPU memory
    size_t gpu_free = 0, gpu_total = 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev) {
        ggml_backend_dev_memory(dev, &gpu_free, &gpu_total);
    }
    if (gpu_total == 0) {
        if (err) *err = "could not query GPU memory";
        return false;
    }

    // Compute per-layer expert size in bytes
    std::vector<uint64_t> layer_expert_bytes((size_t)w.n_layer);
    for (int il = 0; il < w.n_layer; ++il) {
        const TargetLayer & L = w.layers[(size_t)il];
        uint64_t bytes = 0;
        if (L.ffn_gate_up_exps) {
            bytes += ggml_nbytes(L.ffn_gate_up_exps) / (uint64_t)w.n_expert;
        } else {
            if (L.ffn_gate_exps) bytes += ggml_nbytes(L.ffn_gate_exps) / (uint64_t)w.n_expert;
            if (L.ffn_up_exps)   bytes += ggml_nbytes(L.ffn_up_exps) / (uint64_t)w.n_expert;
        }
        if (L.ffn_down_exps) bytes += ggml_nbytes(L.ffn_down_exps) / (uint64_t)w.n_expert;
        layer_expert_bytes[(size_t)il] = bytes;
    }

    // Compute available VRAM for experts
    // With split loading, only core model is on GPU at this point.
    // gpu_total - gpu_free = core model usage (no expert tensors loaded).
    uint64_t total_expert_bytes = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        total_expert_bytes += layer_expert_bytes[(size_t)il] * (uint64_t)w.n_expert;
    }

    // KV cache size estimate — use config max_ctx (from --max-ctx flag),
    // env var DFLASH_MAX_CONTEXT as override, fallback to DevicePlacement default.
    const char * ctx_env = std::getenv("DFLASH_MAX_CONTEXT");
    int max_context = ctx_env ? std::atoi(ctx_env) : cfg_.device.max_ctx;
    if (max_context <= 0) max_context = 8192;

    // KV cache: n_layer × 2 (K+V) × n_head_kv × head_dim × sizeof(fp16) × max_context
    const uint64_t kv_bytes_per_tok = (uint64_t)w.n_layer * 2 *
        (uint64_t)w.n_head_kv * (uint64_t)w.n_embd_head_k * 2;
    // Size the reservation with the SAME inputs runtime uses (scorer policy +
    // VRAM budget); the bare-max_context call took the no-budget fallback
    // (max_ctx/2) and over-reserved KV, starving experts of hot placement.
    const int      kvf_pool      = kvflash_pool_from_env(
        max_context, KvFlashConfig{}, kvflash_scorer_expected(),
        make_kvflash_budget(w, (int64_t)gpu_free));

    const uint64_t warm_cache_bytes = 200ULL * 1024 * 1024;  // 200 MB warm/staging
    uint64_t safety_bytes = 512ULL * 1024 * 1024;      // 512 MB safety margin

    // When draft model is configured, reserve VRAM for it by reducing expert budget.
    // Draft model (~0.9 GiB) + its KV cache + scratch needs space.
    uint64_t draft_reserve_bytes = 0;
    if (cfg_.draft_path) {
        draft_reserve_bytes = 1200ULL * 1024 * 1024;  // ~1.2 GiB for draft model + buffers
        safety_bytes = 256ULL * 1024 * 1024;  // can reduce safety when draft accounts for it
    }

    // Core model bytes = what's already used on GPU (non-expert tensors).
    const uint64_t core_bytes =
        moe_hybrid_core_bytes_from_memory("qwen35moe", gpu_free, gpu_total);

    // KVFlash reserves a fixed resident pool, not max_ctx, of KV.  When active
    // and the full reservation would force experts cold, reserve for the pool so
    // experts stay hot (decouples max_ctx from the expert-placement cliff).  The
    // rule is centralised in kvflash_placement_decision() so future MoE backends
    // (DeepSeek-V4, ...) inherit it instead of re-deriving the byte math.
    const auto kvf_dec = dflash::common::kvflash_placement_decision(
        kv_bytes_per_tok, max_context, kvf_pool,
        gpu_total, core_bytes, total_expert_bytes,
        warm_cache_bytes, safety_bytes, draft_reserve_bytes);
    const uint64_t kv_total   = kvf_dec.kv_total;
    const int      kv_ctx_log = kvf_dec.kv_ctx;
    placement_all_hot_full_kv_ = kvf_dec.all_hot_full_kv;
    if (kvf_dec.pool_reduced) {
        std::printf("[kvflash] placement reserves pool KV (%d tokens, not max_ctx %d) "
                    "-> experts stay hot\n", kvf_pool, max_context);
        std::fflush(stdout);
    }

    uint64_t expert_budget = 0;
    if (gpu_total > core_bytes + kv_total + warm_cache_bytes + safety_bytes + draft_reserve_bytes) {
        expert_budget = gpu_total - core_bytes - kv_total - warm_cache_bytes - safety_bytes - draft_reserve_bytes;
    }

    // Clamp budget to total expert size
    if (expert_budget > total_expert_bytes) {
        expert_budget = total_expert_bytes;
    }

    // Allow manual budget cap via env var (for profiling/testing hybrid mode)
    if (const char * cap_env = std::getenv("DFLASH_EXPERT_BUDGET_MB")) {
        uint64_t cap_bytes = (uint64_t)std::atoi(cap_env) * 1024ULL * 1024ULL;
        if (cap_bytes > 0 && cap_bytes < expert_budget) {
            std::printf("[qwen35moe] capping expert budget from %.2f GiB to %d MB (DFLASH_EXPERT_BUDGET_MB)\n",
                        expert_budget / 1024.0 / 1024.0 / 1024.0, std::atoi(cap_env));
            expert_budget = cap_bytes;
        }
    }

    // Spark: clamp experts to the --spark-vram target and auto-size the cache ring.
    if (std::getenv("DFLASH_SPARK")) {
        uint64_t target = 0;
        if (const char * t = std::getenv("DFLASH_SPARK_VRAM_MB")) target = (uint64_t)std::atoll(t) << 20;
        auto sb = dflash::common::spark_budget_split(expert_budget, total_expert_bytes, w.n_expert,
                                                     core_bytes + kv_total + safety_bytes, target);
        expert_budget = sb.hot_bytes;
        cache_slots_ = sb.cache_slots;
        std::printf("[spark] vram=%s, hot=%.2f GiB, cache=%d slots/layer\n",
                    target ? "target" : "auto(card)", expert_budget / 1073741824.0, cache_slots_);
    }

    std::printf("[qwen35moe] dynamic placement: gpu_total=%.2f GiB, core=%.2f GiB, "
                "kv_cache=%.2f GiB (ctx=%d), warm=%.0f MB, safety=%.0f MB, "
                "expert_budget=%.2f GiB (of %.2f GiB total experts)\n",
                gpu_total / 1024.0 / 1024.0 / 1024.0,
                core_bytes / 1024.0 / 1024.0 / 1024.0,
                kv_total / 1024.0 / 1024.0 / 1024.0,
                kv_ctx_log,
                warm_cache_bytes / 1024.0 / 1024.0,
                safety_bytes / 1024.0 / 1024.0,
                expert_budget / 1024.0 / 1024.0 / 1024.0,
                total_expert_bytes / 1024.0 / 1024.0 / 1024.0);

    if (expert_budget == 0) {
        if (err) *err = "no VRAM budget available for experts (GPU too small or context too long)";
        return false;
    }

    // Stash for the Spark bootstrap rebuild (same budget + cache as this init).
    spark_expert_budget_ = expert_budget;
    layer_expert_bytes_  = layer_expert_bytes;

    // Build placement using greedy knapsack with byte budget
    if (!MoeHybridPlacement::build_from_stats_with_layer_bytes(
            hotness, layer_expert_bytes, expert_budget,
            /*min_hot_per_layer=*/std::min(w.n_expert_used, w.n_expert),
            out, err)) {
        return false;
    }

    std::printf("[qwen35moe] dynamic placement result: %d hot experts, %d cold experts\n",
                out.total_hot, w.n_layer * w.n_expert - out.total_hot);
    std::fflush(stdout);
    return true;
}

}  // namespace dflash::common
