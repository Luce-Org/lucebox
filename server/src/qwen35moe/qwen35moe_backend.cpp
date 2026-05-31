#include "qwen35moe_backend.h"

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
        routing_stats_ = std::make_shared<Qwen35MoeRoutingStats>();
        if (!routing_stats_->init_from_weights(out)) {
            set_last_error("qwen35moe runtime stats init failed");
            return false;
        }
        routing_stats_out_path_ = stats_path;
    }

    // Phase 2: Compute dynamic placement based on VRAM budget.
    // Expert tensor metadata (ne/nb) is valid even without GPU allocation.
    Qwen35MoeExpertPlacement placement;
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

        auto hybrid = std::make_shared<Qwen35MoeHybridStorage>();
        if (!build_qwen35moe_hybrid_storage_from_file(out, backend, placement, layer_file_data, *hybrid, &err)) {
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
    if (total_cold > 0) {
        hybrid_mode_ = true;
        // Keep cfg_.draft_path set — hybrid spec-decode uses it for drafting
        // while target verification runs through the hybrid forward path.
        std::printf("[qwen35moe] hybrid decode path active (%d cold experts)\n", total_cold);
    } else {
        std::printf("[qwen35moe] all experts hot — using fused all-GPU decode path\n");
    }
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

    if (!hybrid_mode_ || !target_weights().moe_hybrid || swap_policy_.max_swaps_total <= 0) return;

    Qwen35MoeSwapPlan plan;
    std::string err;
    if (!build_qwen35moe_swap_plan(target_weights().moe_hybrid->placement, *routing_stats_,
                                   swap_policy_, plan, &err)) {
        std::fprintf(stderr, "[qwen35moe] swap plan failed: %s\n", err.c_str());
        return;
    }
    if (plan.actions.empty()) return;

    auto rebuilt = std::make_shared<Qwen35MoeHybridStorage>();
    if (!build_qwen35moe_hybrid_storage(target_weights(), target_backend(),
                                        plan.next_placement, *rebuilt, &err)) {
        std::fprintf(stderr, "[qwen35moe] swap rebuild failed: %s\n", err.c_str());
        return;
    }
    target_weights().moe_hybrid = std::move(rebuilt);
    if (!placement_out_path_.empty()) {
        if (!plan.next_placement.save_json(placement_out_path_, &err)) {
            std::fprintf(stderr, "[qwen35moe] failed to save next placement: %s\n", err.c_str());
        }
    }
    std::printf("[qwen35moe] applied %zu swap actions at request boundary\n", plan.actions.size());
}

bool Qwen35MoeBackend::run_ar_decode_path(int committed, int n_gen,
                                          std::vector<int32_t> & out_tokens,
                                          const DaemonIO & io) {
    if (!hybrid_mode_ || !target_weights().moe_hybrid) {
        return Qwen35Backend::run_ar_decode_path(committed, n_gen, out_tokens, io);
    }
    if (n_gen <= 0) return true;

    const int hidden = target_weights().n_embd;
    const int vocab  = target_weights().n_vocab;
    std::vector<float> logits_buf((size_t)vocab);
    std::vector<float> act_cur((size_t)hidden);
    uint64_t hot_selected_total = 0;
    uint64_t cold_selected_total = 0;
    uint64_t decode_prefn_us = 0;
    uint64_t decode_logits_us = 0;
    uint64_t cold_layer_calls = 0;
    uint64_t layer_calls = 0;
    const auto decode_t0 = HybridClock::now();

    auto project_logits = [&](const float * hidden_host) -> bool {
        StepGraph proj_sg;
        ggml_init_params ip{};
        ip.mem_size   = 64 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
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
        ggml_backend_tensor_set(proj_sg.hidden_input, hidden_host, 0, sizeof(float) * (size_t)hidden);
        auto st = ggml_backend_graph_compute(target_backend(), proj_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            step_graph_destroy(proj_sg);
            return false;
        }
        ggml_backend_tensor_get(proj_sg.logits, logits_buf.data(), 0, sizeof(float) * (size_t)vocab);
        step_graph_destroy(proj_sg);
        return true;
    };

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

    StepGraph layer_sg;
    std::vector<int32_t> selected((size_t)target_weights().n_expert_used);
    std::vector<float> weights_buf((size_t)target_weights().n_expert_used);
    ggml_backend_t cpu_be = target_weights().moe_hybrid->cpu_backend;

    // Initialize GPU-resident state: persistent act_cur + combine graph on GPU
    GpuResidentState gpu_state;
    if (!init_gpu_resident_state(gpu_state, target_backend(), hidden)) {
        return false;
    }

    for (int step = 1; step < n_gen; ++step) {
        int32_t tok = out_tokens.back();
        // Embed token to host, then upload to GPU-resident act_cur once
        if (!target_weights().embedder.embed(&tok, 1, act_cur.data())) {
            gpu_state.destroy();
            return false;
        }
        ggml_backend_tensor_set(gpu_state.act_cur, act_cur.data(), 0, sizeof(float) * (size_t)hidden);

        for (int il = 0; il < target_weights().n_layer; ++il) {
            const auto prefn_t0 = HybridClock::now();
            // Build pre-FFN graph (attention/DeltaNet + router only, no MoE FFN)
            if (!build_layer_prefn_step(layer_sg, target_weights(), target_cache(), target_backend(),
                                        il, committed, /*n_tokens=*/1,
                                        /*with_mask=*/false, /*fa_window=*/0, cfg_.kq_stride_pad)) {
                step_graph_destroy(layer_sg);
                gpu_state.destroy();
                return false;
            }
            // GPU→GPU: copy persistent act_cur to pre-FFN graph input (no PCIe!)
            ggml_backend_tensor_copy(gpu_state.act_cur, layer_sg.inp_embed);
            if (layer_sg.positions) {
                int32_t pos4[4] = {committed, committed, committed, 0};
                ggml_backend_tensor_set(layer_sg.positions, pos4, 0, sizeof(pos4));
            }
            auto st = ggml_backend_graph_compute(target_backend(), layer_sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                step_graph_destroy(layer_sg);
                gpu_state.destroy();
                return false;
            }

            // Only read router decisions to CPU (64 bytes total — unavoidable)
            ggml_tensor * layer_selected = (!layer_sg.moe_selected.empty() && (size_t)il < layer_sg.moe_selected.size())
                ? layer_sg.moe_selected[(size_t)il]
                : nullptr;
            if (!layer_selected || !layer_sg.moe_weights) {
                step_graph_destroy(layer_sg);
                gpu_state.destroy();
                return false;
            }
            ggml_backend_tensor_get(layer_selected, selected.data(), 0,
                                    sizeof(int32_t) * selected.size());
            ggml_backend_tensor_get(layer_sg.moe_weights, weights_buf.data(), 0,
                                    sizeof(float) * weights_buf.size());
            if (routing_stats_) {
                routing_stats_->observe(il, selected.data(), (int)selected.size());
            }
            const auto prefn_t1 = HybridClock::now();
            decode_prefn_us += elapsed_us(prefn_t0, prefn_t1);

            // GPU-resident hybrid FFN: hot on GPU, cold on CPU, combine on GPU
            auto & storage = target_weights().moe_hybrid->layers[(size_t)il];
            const auto & L = target_weights().layers[(size_t)il];
            if (!eval_qwen35moe_hybrid_ffn_gpu_resident(
                    target_backend(), target_weights(), L, storage, cpu_be,
                    layer_sg.ffn_post, layer_sg.ffn_residual,
                    gpu_state,
                    selected.data(), weights_buf.data(),
                    (int)selected.size())) {
                step_graph_destroy(layer_sg);
                gpu_state.destroy();
                return false;
            }
            // gpu_state.act_cur now holds the layer output on GPU

            // Track routing stats
            if (hybrid_telemetry_) {
                layer_calls++;
                for (int32_t expert : selected) {
                    if (expert >= 0 && expert < (int32_t)storage.hot_local_by_global.size()) {
                        if (storage.hot_local_by_global[(size_t)expert] >= 0) {
                            hot_selected_total++;
                        } else {
                            cold_selected_total++;
                            cold_layer_calls++;
                        }
                    }
                }
            } else {
                for (int32_t expert : selected) {
                    if (expert >= 0 && expert < (int32_t)storage.hot_local_by_global.size()) {
                        if (storage.hot_local_by_global[(size_t)expert] >= 0) {
                            hot_selected_total++;
                        } else {
                            cold_selected_total++;
                        }
                    }
                }
            }
        }

        // Read final act_cur from GPU for logits projection (single 10KB read)
        ggml_backend_tensor_get(gpu_state.act_cur, act_cur.data(), 0, sizeof(float) * (size_t)hidden);
        const auto logits_t0 = HybridClock::now();
        if (!project_logits(act_cur.data())) {
            step_graph_destroy(layer_sg);
            gpu_state.destroy();
            return false;
        }
        const auto logits_t1 = HybridClock::now();
        decode_logits_us += elapsed_us(logits_t0, logits_t1);
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
        out_tokens.push_back(next_tok);
        io.emit(next_tok);
        committed++;
        target_cache().cur_pos = committed;
        if (io.cancelled) break;
        if (is_eos_tok(next_tok, target_weights())) break;
    }
    step_graph_destroy(layer_sg);
    gpu_state.destroy();
    last_hot_selected_ = hot_selected_total;
    last_cold_selected_ = cold_selected_total;
    std::printf("[qwen35moe] hybrid decode stats: hot_selected=%llu cold_selected=%llu\n",
                (unsigned long long)last_hot_selected_,
                (unsigned long long)last_cold_selected_);
    if (hybrid_telemetry_) {
        const uint64_t decode_us = elapsed_us(decode_t0, HybridClock::now());
        std::printf("[qwen35moe] hybrid telemetry: decode_ms=%.2f layer_ms=%.2f logits_ms=%.2f "
                    "layer_calls=%llu cold_layer_calls=%llu\n",
                    decode_us / 1000.0,
                    decode_prefn_us / 1000.0,
                    decode_logits_us / 1000.0,
                    (unsigned long long)layer_calls,
                    (unsigned long long)cold_layer_calls);
    }
    return true;
}

GenerateResult Qwen35MoeBackend::generate_impl(const GenerateRequest & req,
                                               const DaemonIO & io) {
    if (!hybrid_mode_ || !target_weights().moe_hybrid) {
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

    const int hidden = target_weights().n_embd;
    const int vocab  = target_weights().n_vocab;
    std::vector<float> act_cur((size_t)hidden);
    std::vector<float> residual_buf((size_t)hidden);
    std::vector<float> post_buf((size_t)hidden);
    std::vector<float> ffn_out((size_t)hidden);
    std::vector<float> logits_buf((size_t)vocab);
    std::vector<int32_t> selected((size_t)target_weights().n_expert_used);
    std::vector<float> weights_buf((size_t)target_weights().n_expert_used);
    ggml_backend_t cpu_be = target_weights().moe_hybrid->cpu_backend;

    StepGraph layer_sg;

    // Cached pre-FFN graphs for DeltaNet layers (position-independent, reusable)
    const int n_layer = target_weights().n_layer;
    std::vector<StepGraph> cached_prefn((size_t)n_layer);
    std::vector<bool> prefn_built((size_t)n_layer, false);
    uint64_t build_us_total = 0, compute_us_total = 0, readback_us_total = 0, ffn_us_total = 0;
    uint64_t prefill_build_us = 0, prefill_compute_us = 0, prefill_readback_us = 0, prefill_ffn_us = 0;
    uint64_t decode_build_us = 0, decode_compute_us = 0, decode_readback_us = 0, decode_ffn_us = 0;
    uint64_t embed_us_total = 0, logits_us_total = 0;
    Qwen35MoeHybridFfnTelemetry ffn_tel_accum{};

    StepGraph logits_sg;  // Persistent logits graph (built once, reused every token)

    auto cleanup_graphs = [&]() {
        step_graph_destroy(layer_sg);
        step_graph_destroy(logits_sg);
        for (auto & sg : cached_prefn) step_graph_destroy(sg);
    };

    // Helper: process one token through all layers (host-based with cached graphs)
    auto process_one_token = [&](int kv_pos) -> bool {
        for (int il = 0; il < n_layer; ++il) {
            const bool is_attn = (((il + 1) % target_weights().full_attention_interval) == 0);
            const auto t0 = HybridClock::now();

            StepGraph * sg_ptr;
            if (!is_attn && prefn_built[(size_t)il]) {
                sg_ptr = &cached_prefn[(size_t)il];
            } else {
                StepGraph & sg = is_attn ? layer_sg : cached_prefn[(size_t)il];
                if (!build_layer_prefn_step(sg, target_weights(), target_cache(), target_backend(),
                                            il, kv_pos, /*n_tokens=*/1,
                                            /*with_mask=*/false, /*fa_window=*/0, cfg_.kq_stride_pad)) {
                    return false;
                }
                if (!is_attn) prefn_built[(size_t)il] = true;
                sg_ptr = &sg;
            }

            // Upload act_cur from host → GPU (standard path)
            ggml_backend_tensor_set(sg_ptr->inp_embed, act_cur.data(), 0, sizeof(float) * (size_t)hidden);
            if (sg_ptr->positions) {
                int32_t pos4[4] = {kv_pos, kv_pos, kv_pos, 0};
                ggml_backend_tensor_set(sg_ptr->positions, pos4, 0, sizeof(pos4));
            }
            const auto t1 = HybridClock::now();
            build_us_total += elapsed_us(t0, t1);

            auto st = ggml_backend_graph_compute(target_backend(), sg_ptr->gf);
            if (st != GGML_STATUS_SUCCESS) return false;
            const auto t2 = HybridClock::now();
            compute_us_total += elapsed_us(t1, t2);

            // Read back pre-FFN outputs
            ggml_backend_tensor_get(sg_ptr->ffn_residual, residual_buf.data(), 0, sizeof(float) * (size_t)hidden);
            ggml_backend_tensor_get(sg_ptr->ffn_post, post_buf.data(), 0, sizeof(float) * (size_t)hidden);
            ggml_tensor * layer_selected = (!sg_ptr->moe_selected.empty() && (size_t)il < sg_ptr->moe_selected.size())
                ? sg_ptr->moe_selected[(size_t)il]
                : nullptr;
            if (!layer_selected || !sg_ptr->moe_weights) return false;
            ggml_backend_tensor_get(layer_selected, selected.data(), 0,
                                    sizeof(int32_t) * selected.size());
            ggml_backend_tensor_get(sg_ptr->moe_weights, weights_buf.data(), 0,
                                    sizeof(float) * weights_buf.size());
            if (routing_stats_) {
                routing_stats_->observe(il, selected.data(), (int)selected.size());
            }
            const auto t3 = HybridClock::now();
            readback_us_total += elapsed_us(t2, t3);

            // Hybrid FFN: hot on GPU, cold on CPU
            auto & storage = target_weights().moe_hybrid->layers[(size_t)il];
            const auto & L = target_weights().layers[(size_t)il];
            if (!eval_qwen35moe_hybrid_ffn_single(
                    target_backend(), target_weights(), L, storage, cpu_be,
                    post_buf.data(), selected.data(), weights_buf.data(),
                    (int)selected.size(), ffn_out, nullptr, nullptr)) {
                return false;
            }

            // Layer output = FFN output + residual
            for (int i = 0; i < hidden; ++i) {
                act_cur[(size_t)i] = ffn_out[(size_t)i] + residual_buf[(size_t)i];
            }

            if (hybrid_telemetry_) {
                for (int32_t expert : selected) {
                    if (expert >= 0 && expert < (int32_t)storage.hot_local_by_global.size()) {
                        if (storage.hot_local_by_global[(size_t)expert] >= 0) {
                            ffn_tel_accum.hot_selected++;
                        } else {
                            ffn_tel_accum.cold_selected++;
                        }
                    }
                }
            }

            const auto t4 = HybridClock::now();
            ffn_us_total += elapsed_us(t3, t4);
        }
        return true;
    };

    // Helper: compute logits from act_cur (persistent graph, built once)
    auto compute_logits = [&]() -> bool {
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
        ggml_backend_tensor_set(logits_sg.hidden_input, act_cur.data(), 0, sizeof(float) * (size_t)hidden);
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
    for (int il = 0; il < n_layer; ++il) {
        auto & storage = target_weights().moe_hybrid->layers[(size_t)il];
        const auto & L = target_weights().layers[(size_t)il];

        for (int chunk_start = 0; chunk_start < prompt_len; chunk_start += prefill_chunk) {
            const int chunk_len = std::min(prefill_chunk, prompt_len - chunk_start);
            const auto t0 = HybridClock::now();

            const bool with_mask = (cfg_.kq_stride_pad > KQ_MASK_PAD) || (chunk_len > 1);

            // Build pre-FFN graph for this chunk
            StepGraph prefill_sg;
            if (!build_layer_prefn_step(prefill_sg, target_weights(), target_cache(), target_backend(),
                                        il, /*kv_start=*/chunk_start, /*n_tokens=*/chunk_len,
                                        with_mask, /*fa_window=*/0, cfg_.kq_stride_pad)) {
                result.error = "prefill_build";
                step_graph_destroy(prefill_sg);
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

            // Batched hybrid FFN for this chunk
            std::vector<float> ffn_batch_out;
            if (!eval_qwen35moe_hybrid_ffn_batched(
                    target_backend(), cpu_be, target_weights(), L, storage,
                    chunk_post.data(), chunk_selected.data(), chunk_weights.data(),
                    chunk_len, ffn_batch_out, &result.error)) {
                step_graph_destroy(prefill_sg);
                cleanup_graphs();
                return result;
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

            step_graph_destroy(prefill_sg);
        }
    }

    // Copy last token's output to act_cur for decode
    std::memcpy(act_cur.data(), embed_all.data() + (size_t)(prompt_len - 1) * (size_t)hidden,
                sizeof(float) * (size_t)hidden);

    // Save prefill counters
    prefill_build_us = build_us_total;
    prefill_compute_us = compute_us_total;
    prefill_readback_us = readback_us_total;
    prefill_ffn_us = ffn_us_total;
    build_us_total = 0; compute_us_total = 0; readback_us_total = 0; ffn_us_total = 0;

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
            // AR fallback decode
            // Get logits from last prefill token
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

                // Generate remaining tokens
                for (int step = 1; step < req.n_gen; ++step) {
                    const auto t_emb0 = HybridClock::now();
                    int32_t tok = result.tokens.back();
                    if (!target_weights().embedder.embed(&tok, 1, act_cur.data())) {
                        result.error = "decode_embed";
                        cleanup_graphs();
                        return result;
                    }
                    embed_us_total += elapsed_us(t_emb0, HybridClock::now());
                    if (!process_one_token(committed)) {
                        result.error = "decode";
                        cleanup_graphs();
                        return result;
                    }
                    const auto t_log0 = HybridClock::now();
                    if (!compute_logits()) {
                        result.error = "decode_logits";
                        cleanup_graphs();
                        return result;
                    }
                    logits_us_total += elapsed_us(t_log0, HybridClock::now());
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
            }
            cleanup_graphs();
        }

        result.decode_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_decode_start).count();
    } else {
        cleanup_graphs();
    }
    if (hybrid_telemetry_) {
        decode_build_us = build_us_total;
        decode_compute_us = compute_us_total;
        decode_readback_us = readback_us_total;
        decode_ffn_us = ffn_us_total;
        const int n_decode_toks = (int)result.tokens.size();
        std::printf("[qwen35moe] === PREFILL ANALYSIS (prompt_len=%d, chunk=%d) ===\n", prompt_len, prefill_chunk);
        std::printf("  prefill_total=%.1fms (%.1f tok/s)\n",
                    result.prefill_s * 1000.0, prompt_len / result.prefill_s);
        std::printf("  build=%.1fms compute=%.1fms readback=%.1fms ffn=%.1fms\n",
                    prefill_build_us / 1000.0, prefill_compute_us / 1000.0,
                    prefill_readback_us / 1000.0, prefill_ffn_us / 1000.0);
        const double prefill_total_us = (double)(prefill_build_us + prefill_compute_us + prefill_readback_us + prefill_ffn_us);
        if (prefill_total_us > 0) {
            std::printf("  pct: build=%.1f%% compute=%.1f%% readback=%.1f%% ffn=%.1f%%\n",
                        100.0 * prefill_build_us / prefill_total_us,
                        100.0 * prefill_compute_us / prefill_total_us,
                        100.0 * prefill_readback_us / prefill_total_us,
                        100.0 * prefill_ffn_us / prefill_total_us);
        }
        std::printf("[qwen35moe] === DECODE ANALYSIS (n_tokens=%d) ===\n", n_decode_toks);
        if (result.decode_s > 0) {
            std::printf("  decode_total=%.1fms (%.2f tok/s)\n",
                        result.decode_s * 1000.0, n_decode_toks / result.decode_s);
        }
        std::printf("  build=%.1fms compute=%.1fms readback=%.1fms ffn=%.1fms\n",
                    decode_build_us / 1000.0, decode_compute_us / 1000.0,
                    decode_readback_us / 1000.0, decode_ffn_us / 1000.0);
        std::printf("  embed=%.1fms logits=%.1fms\n",
                    embed_us_total / 1000.0, logits_us_total / 1000.0);
        const double decode_total_us = (double)(decode_build_us + decode_compute_us + decode_readback_us + decode_ffn_us + embed_us_total + logits_us_total);
        if (decode_total_us > 0 && n_decode_toks > 0) {
            std::printf("  per-token: build=%.2fms compute=%.2fms readback=%.2fms ffn=%.2fms embed=%.2fms logits=%.2fms\n",
                        decode_build_us / 1000.0 / n_decode_toks,
                        decode_compute_us / 1000.0 / n_decode_toks,
                        decode_readback_us / 1000.0 / n_decode_toks,
                        decode_ffn_us / 1000.0 / n_decode_toks,
                        embed_us_total / 1000.0 / n_decode_toks,
                        logits_us_total / 1000.0 / n_decode_toks);
        }
        std::printf("[qwen35moe] === FFN BREAKDOWN (total calls) ===\n");
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
    if (!hybrid_mode_ || !target_weights().moe_hybrid) {
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
    const int n_layer = target_weights().n_layer;
    const int n_expert_used = target_weights().n_expert_used;

    // Embed the token
    if (!target_weights().embedder.embed(&tok, 1, act_cur.data())) return false;

    std::vector<float> residual_buf((size_t)hidden);
    std::vector<float> post_buf((size_t)hidden);
    std::vector<float> ffn_out((size_t)hidden);
    std::vector<int32_t> selected((size_t)n_expert_used);
    std::vector<float> weights_buf((size_t)n_expert_used);
    ggml_backend_t cpu_be = target_weights().moe_hybrid->cpu_backend;

    StepGraph layer_sg;

    for (int il = 0; il < n_layer; ++il) {
        // Pre-FFN: attention/DeltaNet + router
        if (!build_layer_prefn_step(layer_sg, target_weights(), target_cache(), target_backend(),
                                     il, kv_pos, /*n_tokens=*/1,
                                     /*with_mask=*/false, /*fa_window=*/0, cfg_.kq_stride_pad)) {
            step_graph_destroy(layer_sg);
            return false;
        }
        ggml_backend_tensor_set(layer_sg.inp_embed, act_cur.data(), 0, sizeof(float) * (size_t)hidden);
        if (layer_sg.positions) {
            int32_t pos4[4] = {kv_pos, kv_pos, kv_pos, 0};
            ggml_backend_tensor_set(layer_sg.positions, pos4, 0, sizeof(pos4));
        }
        auto st = ggml_backend_graph_compute(target_backend(), layer_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            step_graph_destroy(layer_sg);
            return false;
        }

        // Read pre-FFN outputs
        ggml_backend_tensor_get(layer_sg.ffn_residual, residual_buf.data(), 0, sizeof(float) * (size_t)hidden);
        ggml_backend_tensor_get(layer_sg.ffn_post, post_buf.data(), 0, sizeof(float) * (size_t)hidden);
        ggml_tensor * layer_selected = (!layer_sg.moe_selected.empty() && (size_t)il < layer_sg.moe_selected.size())
            ? layer_sg.moe_selected[(size_t)il]
            : nullptr;
        if (!layer_selected || !layer_sg.moe_weights) {
            step_graph_destroy(layer_sg);
            return false;
        }
        ggml_backend_tensor_get(layer_selected, selected.data(), 0,
                                 sizeof(int32_t) * selected.size());
        ggml_backend_tensor_get(layer_sg.moe_weights, weights_buf.data(), 0,
                                 sizeof(float) * weights_buf.size());
        step_graph_destroy(layer_sg);

        // Hybrid FFN
        auto & storage = target_weights().moe_hybrid->layers[(size_t)il];
        const auto & L = target_weights().layers[(size_t)il];
        if (!eval_qwen35moe_hybrid_ffn_single(
                target_backend(), target_weights(), L, storage, cpu_be,
                post_buf.data(), selected.data(), weights_buf.data(),
                n_expert_used, ffn_out, nullptr, nullptr)) {
            return false;
        }

        // Layer output = FFN + residual
        for (int i = 0; i < hidden; ++i) {
            act_cur[(size_t)i] = ffn_out[(size_t)i] + residual_buf[(size_t)i];
        }

        // Feature capture: write act_cur (F32) → cache_.target_feat (BF16)
        if (target_cache().target_feat) {
            int capture_idx = -1;
            for (int k = 0; k < target_weights().n_capture_layers; k++) {
                if (target_weights().capture_layer_ids[k] == il) {
                    capture_idx = k;
                    break;
                }
            }
            if (capture_idx >= 0) {
                const int cap = target_cache().target_feat_cap;
                const int slot = kv_pos % cap;
                const size_t elt = ggml_element_size(target_cache().target_feat);
                const size_t col_stride = target_cache().target_feat->nb[1];
                const size_t offset = (size_t)slot * col_stride +
                                      (size_t)capture_idx * (size_t)hidden * elt;

                // Convert F32 → BF16 on host
                std::vector<ggml_bf16_t> bf16_buf((size_t)hidden);
                ggml_fp32_to_bf16_row(act_cur.data(), bf16_buf.data(), hidden);

                // Write to GPU target_feat tensor
                ggml_backend_tensor_set(target_cache().target_feat, bf16_buf.data(),
                                         offset, (size_t)hidden * elt);
            }
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
                                               Qwen35MoeExpertPlacement & out,
                                               std::string * err) {
    // Load hotness table or assume uniform hotness
    Qwen35MoeRoutingStats hotness;
    if (hotness_path && hotness_path[0]) {
        if (!Qwen35MoeRoutingStats::load_csv(std::string(hotness_path), hotness, err)) {
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
    if (!Qwen35MoeExpertPlacement::build_from_stats_with_layer_bytes(
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
