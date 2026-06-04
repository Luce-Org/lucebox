#include "qwen35moe_backend.h"

#include "../common/moe_hybrid_placement.h"
#include "../common/moe_hybrid_types.h"
#include "../common/moe_hybrid_types_impl.h"
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
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

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
        int fd = ::open(cfg_.target_path, O_RDONLY);
        if (fd < 0) {
            set_last_error("failed to open GGUF file for mmap");
            gguf_free(gctx);
            return false;
        }
        struct stat st;
        if (::fstat(fd, &st) < 0) {
            ::close(fd);
            set_last_error("fstat failed on GGUF");
            gguf_free(gctx);
            return false;
        }
        const size_t file_size = (size_t)st.st_size;
        void * mmap_addr = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (mmap_addr == MAP_FAILED) {
            set_last_error("mmap failed on GGUF");
            gguf_free(gctx);
            return false;
        }

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
        std::vector<MoeLayerDesc> layer_descs((size_t)out.n_layer);
        for (int il = 0; il < out.n_layer; ++il) {
            layer_descs[(size_t)il] = make_moe_layer_desc(out.layers[(size_t)il]);
        }
        if (!build_moe_hybrid_storage_from_file(hybrid_cfg, backend, placement, layer_descs, layer_file_data, *hybrid, &err)) {
            ::munmap(mmap_addr, file_size);
            gguf_free(gctx);
            set_last_error(std::string("qwen35moe hybrid storage build failed: ") + err);
            return false;
        }

        ::munmap(mmap_addr, file_size);
        gguf_free(gctx);

        out.moe_hybrid = std::move(hybrid);
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

    // Persistent logits graph (built once, reused per token)
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
            ggml_tensor * normed = ggml_rms_norm(logits_sg.ctx, logits_sg.hidden_input, target_weights().rms_eps);
            normed = ggml_mul(logits_sg.ctx, normed, target_weights().out_norm);
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

        PipelinedDecodeTelemetry tel;
        if (!pipelined_decode_one_token(*pipe_state_, target_backend(), target_weights(),
                                       target_cache(), *target_weights().moe_hybrid,
                                       committed, cfg_.kq_stride_pad,
                                       hybrid_telemetry_ ? &tel : nullptr)) {
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
            ggml_tensor * normed = ggml_rms_norm(logits_sg.ctx, logits_sg.hidden_input, target_weights().rms_eps);
            normed = ggml_mul(logits_sg.ctx, normed, target_weights().out_norm);
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

            // Hybrid FFN — skip batched path when cold experts exist (CUDA mul_mat_id bug on sm_75)
            MoeHybridConfig chunk_cfg = make_moe_hybrid_config(target_weights());
            MoeLayerDesc chunk_desc = make_moe_layer_desc(target_weights().layers[(size_t)il]);
            std::vector<float> ffn_batch_out;
            bool ffn_ok = false;
            if (storage.cold_expert_ids.empty()) {
                // All experts hot — safe to use batched path
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
    auto t_prefill_end = std::chrono::steady_clock::now();
    result.prefill_s = std::chrono::duration<double>(t_prefill_end - t_prefill_start).count();

    // ── Hybrid Decode ──
    if (req.n_gen > 0) {
        auto t_decode_start = std::chrono::steady_clock::now();

        // Check if hybrid spec-decode is available
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
            if (!do_hybrid_spec_decode(committed, req.n_gen, result.tokens, out_io)) {
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

                    PipelinedDecodeTelemetry tel;
                    if (!pipelined_decode_one_token(*pipe_state_, target_backend(), target_weights(),
                                                    target_cache(), *target_weights().moe_hybrid,
                                                    committed, cfg_.kq_stride_pad,
                                                    hybrid_telemetry_ ? &tel : nullptr)) {
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
    // Snapshot restore not supported in hybrid split-load mode.
    // Fall back to full generate (ignores snapshot).
    return generate_impl(req, io);
}

// ── Hybrid spec-decode: draft → verify via hybrid forward → accept ──────────

bool Qwen35MoeBackend::hybrid_forward_one_token(int32_t tok, int kv_pos,
                                                 std::vector<float> & act_cur,
                                                 int32_t & argmax_out) {
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
    ggml_tensor * normed = ggml_rms_norm(proj_sg.ctx, proj_sg.hidden_input, target_weights().rms_eps);
    normed = ggml_mul(proj_sg.ctx, normed, target_weights().out_norm);
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

bool Qwen35MoeBackend::do_hybrid_spec_decode(int committed, int n_gen,
                                              std::vector<int32_t> & out_tokens,
                                              const DaemonIO & io) {
    const int hidden = target_weights().n_embd;
    const int q_len = draft_weights().block_size;
    if (q_len <= 0) return false;

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

    auto t_dec0 = std::chrono::steady_clock::now();

    while (n_generated < n_gen) {
        const int need_commit_budget = n_gen - n_generated;

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

        // 4. Verify: snapshot recurrent state, then run each draft token through hybrid forward
        snapshot_ssm_state(target_cache());

        target_tok.resize(q_len);
        bool verify_ok = true;
        for (int i = 0; i < q_len; i++) {
            int32_t argmax = -1;
            if (!hybrid_forward_one_token(draft_tok[i], committed + i, act_cur, argmax)) {
                verify_ok = false;
                break;
            }
            target_tok[i] = argmax;
        }
        if (!verify_ok) {
            std::fprintf(stderr, "[hybrid-spec] verify failed\n");
            restore_ssm_state(target_cache());
            step_graph_destroy(draft_sg);
            return false;
        }

        // 5. Acceptance: longest matching prefix
        int accept_n = 1;
        for (int i = 0; i < q_len - 1; i++) {
            if (draft_tok[i + 1] == target_tok[i]) accept_n++;
            else break;
        }
        int bonus_tok = (accept_n < q_len) ? target_tok[accept_n - 1] : -1;
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

        // Replay tokens through hybrid forward (captures features for next draft step)
        int32_t replay_last = -1;
        for (int i = 0; i < commit_n; i++) {
            int32_t argmax = -1;
            if (!hybrid_forward_one_token(replay_tok[i], committed + i, act_cur, argmax)) {
                std::fprintf(stderr, "[hybrid-spec] replay failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }
            replay_last = argmax;
        }
        last_tok = replay_last;

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
    const uint64_t kv_total = kv_bytes_per_tok * (uint64_t)max_context;

    const uint64_t warm_cache_bytes = 200ULL * 1024 * 1024;  // 200 MB warm/staging
    uint64_t safety_bytes = 512ULL * 1024 * 1024;      // 512 MB safety margin

    // When draft model is configured, reserve VRAM for it by reducing expert budget.
    // Draft model (~0.9 GiB) + its KV cache + scratch needs space.
    uint64_t draft_reserve_bytes = 0;
    if (cfg_.draft_path) {
        draft_reserve_bytes = 1200ULL * 1024 * 1024;  // ~1.2 GiB for draft model + buffers
        safety_bytes = 256ULL * 1024 * 1024;  // can reduce safety when draft accounts for it
    }

    // Core model bytes = what's already used on GPU (non-expert tensors)
    const uint64_t core_bytes = gpu_total - gpu_free;

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

    std::printf("[qwen35moe] dynamic placement: gpu_total=%.2f GiB, core=%.2f GiB, "
                "kv_cache=%.2f GiB (ctx=%d), warm=%.0f MB, safety=%.0f MB, "
                "expert_budget=%.2f GiB (of %.2f GiB total experts)\n",
                gpu_total / 1024.0 / 1024.0 / 1024.0,
                core_bytes / 1024.0 / 1024.0 / 1024.0,
                kv_total / 1024.0 / 1024.0 / 1024.0,
                max_context,
                warm_cache_bytes / 1024.0 / 1024.0,
                safety_bytes / 1024.0 / 1024.0,
                expert_budget / 1024.0 / 1024.0 / 1024.0,
                total_expert_bytes / 1024.0 / 1024.0 / 1024.0);

    if (expert_budget == 0) {
        if (err) *err = "no VRAM budget available for experts (GPU too small or context too long)";
        return false;
    }

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
