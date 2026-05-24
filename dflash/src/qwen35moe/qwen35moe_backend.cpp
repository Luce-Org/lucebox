#include "qwen35moe_backend.h"

#include "common/sampler.h"

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
        if (expert_meta) ggml_free(expert_meta);

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
        cfg_.draft_path = nullptr;
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
    std::vector<float> residual_buf((size_t)hidden);
    std::vector<float> post_buf((size_t)hidden);
    std::vector<float> ffn_out((size_t)hidden);
    std::vector<int32_t> selected((size_t)target_weights().n_expert_used);
    std::vector<float> weights_buf((size_t)target_weights().n_expert_used);
    ggml_backend_t cpu_be = target_weights().moe_hybrid->cpu_backend;

    for (int step = 1; step < n_gen; ++step) {
        int32_t tok = out_tokens.back();
        if (!target_weights().embedder.embed(&tok, 1, act_cur.data())) return false;

        for (int il = 0; il < target_weights().n_layer; ++il) {
            const auto prefn_t0 = HybridClock::now();
            // Build pre-FFN graph (attention/DeltaNet + router only, no MoE FFN)
            if (!build_layer_prefn_step(layer_sg, target_weights(), target_cache(), target_backend(),
                                        il, committed, /*n_tokens=*/1,
                                        /*with_mask=*/false, /*fa_window=*/0, cfg_.kq_stride_pad)) {
                step_graph_destroy(layer_sg);
                return false;
            }
            ggml_backend_tensor_set(layer_sg.inp_embed, act_cur.data(), 0, sizeof(float) * (size_t)hidden);
            if (layer_sg.positions) {
                int32_t pos4[4] = {committed, committed, committed, 0};
                ggml_backend_tensor_set(layer_sg.positions, pos4, 0, sizeof(pos4));
            }
            auto st = ggml_backend_graph_compute(target_backend(), layer_sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                step_graph_destroy(layer_sg);
                return false;
            }

            // Read pre-FFN outputs: residual, post-norm (FFN input), router selections
            ggml_backend_tensor_get(layer_sg.ffn_residual, residual_buf.data(), 0, sizeof(float) * (size_t)hidden);
            ggml_backend_tensor_get(layer_sg.ffn_post, post_buf.data(), 0, sizeof(float) * (size_t)hidden);

            if (!layer_sg.moe_selected.empty() && layer_sg.moe_selected[(size_t)il]) {
                ggml_backend_tensor_get(layer_sg.moe_selected[(size_t)il], selected.data(), 0,
                                        sizeof(int32_t) * selected.size());
            }
            if (layer_sg.moe_weights) {
                ggml_backend_tensor_get(layer_sg.moe_weights, weights_buf.data(), 0,
                                        sizeof(float) * weights_buf.size());
            }
            if (routing_stats_) {
                routing_stats_->observe(il, selected.data(), (int)selected.size());
            }
            const auto prefn_t1 = HybridClock::now();
            decode_prefn_us += elapsed_us(prefn_t0, prefn_t1);

            // Hybrid FFN: hot experts on GPU, cold on CPU
            auto & storage = target_weights().moe_hybrid->layers[(size_t)il];
            const auto & L = target_weights().layers[(size_t)il];
            if (!eval_qwen35moe_hybrid_ffn_single(
                    target_backend(), target_weights(), L, storage, cpu_be,
                    post_buf.data(), selected.data(), weights_buf.data(),
                    (int)selected.size(), ffn_out, nullptr, nullptr)) {
                step_graph_destroy(layer_sg);
                return false;
            }

            // Layer output = FFN output + residual
            for (int i = 0; i < hidden; ++i) {
                act_cur[(size_t)i] = ffn_out[(size_t)i] + residual_buf[(size_t)i];
            }

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

        const auto logits_t0 = HybridClock::now();
        if (!project_logits(act_cur.data())) {
            step_graph_destroy(layer_sg);
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

GenerateResult Qwen35MoeBackend::generate(const GenerateRequest & req,
                                          const DaemonIO & io) {
    if (!hybrid_mode_ || !target_weights().moe_hybrid) {
        auto result = Qwen35Backend::generate(req, io);
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
    uint64_t build_us_total = 0, compute_us_total = 0, readback_us_total = 0, ffn_us_total = 0;

    // Helper: process one token through all layers (hybrid: pre-FFN on GPU, FFN split)
    auto process_one_token = [&](int kv_pos) -> bool {
        for (int il = 0; il < target_weights().n_layer; ++il) {
            const auto t0 = HybridClock::now();
            if (!build_layer_prefn_step(layer_sg, target_weights(), target_cache(), target_backend(),
                                        il, kv_pos, /*n_tokens=*/1,
                                        /*with_mask=*/false, /*fa_window=*/0, cfg_.kq_stride_pad)) {
                return false;
            }
            ggml_backend_tensor_set(layer_sg.inp_embed, act_cur.data(), 0, sizeof(float) * (size_t)hidden);
            if (layer_sg.positions) {
                int32_t pos4[4] = {kv_pos, kv_pos, kv_pos, 0};
                ggml_backend_tensor_set(layer_sg.positions, pos4, 0, sizeof(pos4));
            }
            const auto t1 = HybridClock::now();
            build_us_total += elapsed_us(t0, t1);

            auto st = ggml_backend_graph_compute(target_backend(), layer_sg.gf);
            if (st != GGML_STATUS_SUCCESS) return false;
            const auto t2 = HybridClock::now();
            compute_us_total += elapsed_us(t1, t2);

            ggml_backend_tensor_get(layer_sg.ffn_residual, residual_buf.data(), 0, sizeof(float) * (size_t)hidden);
            ggml_backend_tensor_get(layer_sg.ffn_post, post_buf.data(), 0, sizeof(float) * (size_t)hidden);
            if (!layer_sg.moe_selected.empty() && layer_sg.moe_selected[(size_t)il]) {
                ggml_backend_tensor_get(layer_sg.moe_selected[(size_t)il], selected.data(), 0,
                                        sizeof(int32_t) * selected.size());
            }
            if (layer_sg.moe_weights) {
                ggml_backend_tensor_get(layer_sg.moe_weights, weights_buf.data(), 0,
                                        sizeof(float) * weights_buf.size());
            }
            const auto t3 = HybridClock::now();
            readback_us_total += elapsed_us(t2, t3);

            auto & storage = target_weights().moe_hybrid->layers[(size_t)il];
            const auto & L = target_weights().layers[(size_t)il];
            if (!eval_qwen35moe_hybrid_ffn_single(
                    target_backend(), target_weights(), L, storage, cpu_be,
                    post_buf.data(), selected.data(), weights_buf.data(),
                    (int)selected.size(), ffn_out, nullptr, nullptr)) {
                return false;
            }
            const auto t4 = HybridClock::now();
            ffn_us_total += elapsed_us(t3, t4);

            for (int i = 0; i < hidden; ++i) {
                act_cur[(size_t)i] = ffn_out[(size_t)i] + residual_buf[(size_t)i];
            }
        }
        return true;
    };

    // Helper: compute logits from act_cur
    auto compute_logits = [&]() -> bool {
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
        ggml_backend_tensor_set(proj_sg.hidden_input, act_cur.data(), 0, sizeof(float) * (size_t)hidden);
        auto st = ggml_backend_graph_compute(target_backend(), proj_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            step_graph_destroy(proj_sg);
            return false;
        }
        ggml_backend_tensor_get(proj_sg.logits, logits_buf.data(), 0, sizeof(float) * (size_t)vocab);
        step_graph_destroy(proj_sg);
        return true;
    };

    // ── Hybrid Prefill: process prompt tokens one at a time ──
    auto t_prefill_start = std::chrono::steady_clock::now();
    const int prompt_len = (int)req.prompt.size();
    for (int i = 0; i < prompt_len; ++i) {
        int32_t tok = req.prompt[(size_t)i];
        if (!target_weights().embedder.embed(&tok, 1, act_cur.data())) {
            result.error = "prefill_embed";
            step_graph_destroy(layer_sg);
            return result;
        }
        if (!process_one_token(i)) {
            result.error = "prefill";
            step_graph_destroy(layer_sg);
            return result;
        }
    }
    int committed = prompt_len;
    target_cache().cur_pos = committed;
    auto t_prefill_end = std::chrono::steady_clock::now();
    result.prefill_s = std::chrono::duration<double>(t_prefill_end - t_prefill_start).count();

    // ── Hybrid Decode ──
    if (req.n_gen > 0) {
        auto t_decode_start = std::chrono::steady_clock::now();

        // Get logits from last prefill token
        if (!compute_logits()) {
            result.error = "decode_logits";
            step_graph_destroy(layer_sg);
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
                int32_t tok = result.tokens.back();
                if (!target_weights().embedder.embed(&tok, 1, act_cur.data())) {
                    result.error = "decode_embed";
                    step_graph_destroy(layer_sg);
                    return result;
                }
                if (!process_one_token(committed)) {
                    result.error = "decode";
                    step_graph_destroy(layer_sg);
                    return result;
                }
                if (!compute_logits()) {
                    result.error = "decode_logits";
                    step_graph_destroy(layer_sg);
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
        }

        result.decode_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_decode_start).count();
    }

    step_graph_destroy(layer_sg);
    if (hybrid_telemetry_) {
        std::printf("[qwen35moe] perf breakdown: build=%.1fms compute=%.1fms readback=%.1fms ffn=%.1fms\n",
                    build_us_total / 1000.0, compute_us_total / 1000.0,
                    readback_us_total / 1000.0, ffn_us_total / 1000.0);
    }
    result.ok = true;
    if (result.ok) maybe_post_request_swap();
    return result;
}

GenerateResult Qwen35MoeBackend::restore_and_generate(int slot,
                                                      const GenerateRequest & req,
                                                      const DaemonIO & io) {
    if (!hybrid_mode_ || !target_weights().moe_hybrid) {
        auto result = Qwen35Backend::restore_and_generate(slot, req, io);
        if (result.ok) maybe_post_request_swap();
        return result;
    }
    // Snapshot restore not supported in hybrid split-load mode.
    // Fall back to full generate (ignores snapshot).
    return generate(req, io);
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

    // KV cache size estimate
    const char * ctx_env = std::getenv("DFLASH_MAX_CONTEXT");
    int max_context = ctx_env ? std::atoi(ctx_env) : 8192;
    if (max_context <= 0) max_context = 8192;

    // KV cache: n_layer × 2 (K+V) × n_head_kv × head_dim × sizeof(fp16) × max_context
    const uint64_t kv_bytes_per_tok = (uint64_t)w.n_layer * 2 *
        (uint64_t)w.n_head_kv * (uint64_t)w.n_embd_head_k * 2;
    const uint64_t kv_total = kv_bytes_per_tok * (uint64_t)max_context;

    const uint64_t warm_cache_bytes = 200ULL * 1024 * 1024;  // 200 MB warm/staging
    const uint64_t safety_bytes = 512ULL * 1024 * 1024;      // 512 MB safety margin

    // Core model bytes = what's already used on GPU (non-expert tensors)
    const uint64_t core_bytes = gpu_total - gpu_free;

    uint64_t expert_budget = 0;
    if (gpu_total > core_bytes + kv_total + warm_cache_bytes + safety_bytes) {
        expert_budget = gpu_total - core_bytes - kv_total - warm_cache_bytes - safety_bytes;
    }

    // Clamp budget to total expert size
    if (expert_budget > total_expert_bytes) {
        expert_budget = total_expert_bytes;
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
    return true;
}

}  // namespace dflash::common
