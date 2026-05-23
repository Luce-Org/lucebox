#include "qwen35moe_backend.h"

#include "common/sampler.h"

#include "ggml-alloc.h"

#include <cstdio>
#include <cstdlib>

namespace dflash::common {

Qwen35MoeBackend::Qwen35MoeBackend(const Qwen35Config & cfg)
    : Qwen35Backend(cfg) {}

bool Qwen35MoeBackend::load_target_model(ggml_backend_t backend, TargetWeights & out) {
    if (!load_target_gguf(cfg_.target_path, backend, out)) {
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

    const char * placement_path = std::getenv("DFLASH_QWEN35MOE_PLACEMENT");
    if (!placement_path || !placement_path[0]) {
        return true;
    }

    Qwen35MoeExpertPlacement placement;
    std::string err;
    if (!Qwen35MoeExpertPlacement::load_json(placement_path, placement, &err)) {
        set_last_error(std::string("qwen35moe placement load failed: ") + err);
        return false;
    }

    auto hybrid = std::make_shared<Qwen35MoeHybridStorage>();
    if (!build_qwen35moe_hybrid_storage(out, backend, placement, *hybrid, &err)) {
        set_last_error(std::string("qwen35moe hybrid storage build failed: ") + err);
        return false;
    }
    out.moe_hybrid = std::move(hybrid);
    hybrid_mode_ = true;
    cfg_.draft_path = nullptr;  // policy: hybrid mode falls back to AR-only until hybrid FFN lands
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
    std::printf("[qwen35moe] hybrid storage ready: total_hot=%d (%.2f GiB VRAM) total_cold=%d (%.2f GiB RAM) placement=%s (AR-only mode)\n",
                out.moe_hybrid->placement.total_hot,
                hot_bytes / 1024.0 / 1024.0 / 1024.0,
                total_cold,
                cold_bytes / 1024.0 / 1024.0 / 1024.0,
                placement_path);
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
        if (!routing_stats_->save_json(routing_stats_out_path_, &err)) {
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
    std::vector<float> act_next((size_t)hidden);
    std::vector<float> residual((size_t)hidden);
    std::vector<float> post((size_t)hidden);
    std::vector<float> ffn_out((size_t)hidden);
    std::string err;
    uint64_t hot_selected_total = 0;
    uint64_t cold_selected_total = 0;

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
    for (int step = 1; step < n_gen; ++step) {
        int32_t tok = out_tokens.back();
        if (!target_weights().embedder.embed(&tok, 1, act_cur.data())) return false;

        for (int il = 0; il < target_weights().n_layer; ++il) {
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

            ggml_backend_tensor_get(layer_sg.ffn_residual, residual.data(), 0, sizeof(float) * (size_t)hidden);
            ggml_backend_tensor_get(layer_sg.ffn_post, post.data(), 0, sizeof(float) * (size_t)hidden);

            const auto & storage = target_weights().moe_hybrid->layers[(size_t)il];
            std::vector<int32_t> selected((size_t)target_weights().n_expert_used);
            std::vector<float> weights((size_t)target_weights().n_expert_used);
            if (layer_sg.moe_selected.empty() || !layer_sg.moe_selected[(size_t)il] || !layer_sg.moe_weights) {
                step_graph_destroy(layer_sg);
                return false;
            }
            ggml_backend_tensor_get(layer_sg.moe_selected[(size_t)il], selected.data(), 0,
                                    sizeof(int32_t) * selected.size());
            ggml_backend_tensor_get(layer_sg.moe_weights, weights.data(), 0,
                                    sizeof(float) * weights.size());
            if (routing_stats_) {
                routing_stats_->observe(il, selected.data(), (int)selected.size());
            }

            if (!eval_qwen35moe_hybrid_ffn_single(target_backend(), target_weights(),
                                                  target_weights().layers[(size_t)il], storage,
                                                  target_weights().moe_hybrid->cpu_backend,
                                                  post.data(), selected.data(), weights.data(),
                                                  (int)selected.size(), ffn_out, &err)) {
                std::fprintf(stderr, "[qwen35moe] hybrid FFN eval failed layer=%d: %s\n",
                             il, err.c_str());
                step_graph_destroy(layer_sg);
                return false;
            }
            for (int i = 0; i < hidden; ++i) {
                act_next[(size_t)i] = residual[(size_t)i] + ffn_out[(size_t)i];
            }
            for (int32_t expert : selected) {
                if (expert >= 0 && expert < (int32_t)storage.hot_local_by_global.size()) {
                    if (storage.hot_local_by_global[(size_t)expert] >= 0) {
                        hot_selected_total++;
                    } else {
                        cold_selected_total++;
                    }
                }
            }
            act_cur.swap(act_next);
        }

        if (!project_logits(act_cur.data())) {
            step_graph_destroy(layer_sg);
            return false;
        }
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
    return true;
}

GenerateResult Qwen35MoeBackend::generate(const GenerateRequest & req,
                                          const DaemonIO & io) {
    auto result = Qwen35Backend::generate(req, io);
    if (result.ok) {
        maybe_post_request_swap();
    }
    return result;
}

GenerateResult Qwen35MoeBackend::restore_and_generate(int slot,
                                                      const GenerateRequest & req,
                                                      const DaemonIO & io) {
    auto result = Qwen35Backend::restore_and_generate(slot, req, io);
    if (result.ok) {
        maybe_post_request_swap();
    }
    return result;
}

}  // namespace dflash::common
