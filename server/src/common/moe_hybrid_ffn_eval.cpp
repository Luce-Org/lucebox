#include <cstdlib>
#include "moe_hybrid_ffn_eval.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>

namespace dflash::common {

// NVFP4 scale2: if weight has a per-tensor scale, multiply the matmul result
// by that scale. No-op when scale==1.0f (non-NVFP4 models).
inline ggml_tensor * apply_scale2(ggml_context * ctx, ggml_tensor * mm_result, float scale) {
    if (scale == 1.0f) return mm_result;
    return ggml_scale(ctx, mm_result, scale);
}

using HybridClock = std::chrono::steady_clock;

static uint64_t elapsed_us(HybridClock::time_point start, HybridClock::time_point end) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

struct MixedRoutingScratch {
    std::vector<int32_t> hot_sel;
    std::vector<float> hot_wts;
    std::vector<int32_t> cold_sel;
    std::vector<float> cold_wts;
    std::vector<int32_t> lut;
    std::vector<float> vlut;

    void ensure(int total_slots, int n_expert) {
        if (total_slots > 0) {
            hot_sel.resize((size_t)total_slots);
            hot_wts.resize((size_t)total_slots);
            cold_sel.resize((size_t)total_slots);
            cold_wts.resize((size_t)total_slots);
        }
        if (n_expert > 0) {
            lut.resize((size_t)n_expert);
            vlut.resize((size_t)n_expert);
        }
    }
};

static thread_local MixedRoutingScratch g_mixed_routing_scratch;

struct RemoteColdBatchScratch {
    std::vector<int> cold_counts;
    std::vector<int32_t> cold_sel;
    std::vector<float> cold_wts;
    std::vector<std::vector<int>> token_groups;
    std::vector<int> active_groups;
    std::vector<float> group_input;
    std::vector<float> group_output;
    std::vector<int32_t> group_ids;
    std::vector<float> group_wts;

    void ensure(int n_tokens, int n_used, int n_embd, int max_cold_selected) {
        if (n_tokens > 0) {
            cold_counts.resize((size_t)n_tokens);
        }
        if (n_tokens > 0 && n_used > 0) {
            cold_sel.resize((size_t)n_tokens * (size_t)n_used);
            cold_wts.resize((size_t)n_tokens * (size_t)n_used);
        }
        if (n_embd > 0 && n_tokens > 0) {
            group_input.reserve((size_t)n_embd * (size_t)n_tokens);
            group_output.reserve((size_t)n_embd * (size_t)n_tokens);
        }
        if (max_cold_selected > 0 && n_tokens > 0) {
            group_ids.reserve((size_t)n_tokens * (size_t)max_cold_selected);
            group_wts.reserve((size_t)n_tokens * (size_t)max_cold_selected);
        }
    }
};

static thread_local RemoteColdBatchScratch g_remote_cold_batch_scratch;

struct HybridSplitScratch {
    std::vector<float> cold_partial;
    std::vector<float> hot_out;
};

static thread_local HybridSplitScratch g_hybrid_split_scratch;

static bool prepare_unique_hot_local_slots(
    const MoeHybridLayerStorage & storage,
    const int32_t * selected_ids,
    const float * selected_weights,
    int n_tokens,
    int n_used,
    std::vector<int32_t> & out_ids,
    std::vector<float> & out_weights,
    bool * out_has_hot,
    bool * out_has_cold,
    std::string * err) {
    const int n_hot_init = std::max(1, storage.hot_active);
    std::vector<uint8_t> used((size_t) n_hot_init, 0);
    bool has_hot = false;
    bool has_cold = false;

    for (int t = 0; t < n_tokens; ++t) {
        std::fill(used.begin(), used.end(), 0);
        int next_dummy = 0;
        for (int i = 0; i < n_used; ++i) {
            const size_t idx = (size_t) t * (size_t) n_used + (size_t) i;
            out_ids[idx] = -1;
            out_weights[idx] = 0.0f;

            const int32_t gid = selected_ids[idx];
            if (gid < 0 || gid >= (int32_t) storage.hot_local_by_global.size()) {
                continue;
            }
            const int32_t hot_local = storage.hot_local_by_global[(size_t) gid];
            if (hot_local >= 0) {
                if (hot_local >= n_hot_init) {
                    if (err) *err = "hot local id exceeds initialized hot stack";
                    return false;
                }
                if (used[(size_t) hot_local]) {
                    if (err) *err = "duplicate hot local id in routed selection";
                    return false;
                }
                out_ids[idx] = hot_local;
                out_weights[idx] = selected_weights[idx];
                used[(size_t) hot_local] = 1;
                has_hot = true;
                continue;
            }
            if ((size_t) gid < storage.cold_local_by_global.size() &&
                storage.cold_local_by_global[(size_t) gid] >= 0) {
                has_cold = true;
            }
        }

        for (int i = 0; i < n_used; ++i) {
            const size_t idx = (size_t) t * (size_t) n_used + (size_t) i;
            if (out_ids[idx] >= 0) continue;
            while (next_dummy < n_hot_init && used[(size_t) next_dummy]) {
                ++next_dummy;
            }
            if (next_dummy >= n_hot_init) {
                if (err) *err = "insufficient initialized hot experts for unique padded routing";
                return false;
            }
            out_ids[idx] = next_dummy;
            used[(size_t) next_dummy] = 1;
            ++next_dummy;
        }
    }

    if (out_has_hot) *out_has_hot = has_hot;
    if (out_has_cold) *out_has_cold = has_cold;
    return true;
}

static bool prepare_unique_cold_local_slots(
    const MoeHybridLayerStorage & storage,
    const int32_t * selected_ids,
    const float * selected_weights,
    int n_tokens,
    int n_used,
    std::vector<int32_t> & out_ids,
    std::vector<float> & out_weights,
    bool * out_has_cold,
    std::string * err) {
    const int n_cold_stack =
        std::max(1, (int) (storage.down_cold ? storage.down_cold->ne[2] : 0));
    std::vector<uint8_t> used((size_t) n_cold_stack, 0);
    bool has_cold = false;

    for (int t = 0; t < n_tokens; ++t) {
        std::fill(used.begin(), used.end(), 0);
        int next_dummy = 0;
        for (int i = 0; i < n_used; ++i) {
            const size_t idx = (size_t) t * (size_t) n_used + (size_t) i;
            out_ids[idx] = -1;
            out_weights[idx] = 0.0f;

            const int32_t gid = selected_ids[idx];
            if (gid < 0 || gid >= (int32_t) storage.hot_local_by_global.size()) {
                continue;
            }
            if (storage.hot_local_by_global[(size_t) gid] >= 0) {
                continue;
            }
            const int32_t cold_local = storage.cold_local_by_global[(size_t) gid];
            if (cold_local < 0) {
                continue;
            }
            if (cold_local >= n_cold_stack) {
                if (err) *err = "cold local id exceeds initialized cold stack";
                return false;
            }
            if (used[(size_t) cold_local]) {
                if (err) *err = "duplicate cold local id in routed selection";
                return false;
            }
            out_ids[idx] = cold_local;
            out_weights[idx] = selected_weights[idx];
            used[(size_t) cold_local] = 1;
            has_cold = true;
        }

        for (int i = 0; i < n_used; ++i) {
            const size_t idx = (size_t) t * (size_t) n_used + (size_t) i;
            if (out_ids[idx] >= 0) continue;
            while (next_dummy < n_cold_stack && used[(size_t) next_dummy]) {
                ++next_dummy;
            }
            if (next_dummy >= n_cold_stack) {
                if (err) *err = "insufficient cold experts for unique padded routing";
                return false;
            }
            out_ids[idx] = next_dummy;
            used[(size_t) next_dummy] = 1;
            ++next_dummy;
        }
    }

    if (out_has_cold) *out_has_cold = has_cold;
    return true;
}

static bool prepare_unique_hot_global_slots(
    const MoeHybridLayerStorage & storage,
    const int32_t * selected_ids,
    const float * selected_weights,
    int n_tokens,
    int n_used,
    std::vector<int32_t> & out_global_ids,
    std::vector<float> & out_weights,
    std::string * err) {
    const int n_hot_init = std::max(1, storage.hot_active);
    std::vector<int32_t> hot_global_by_local((size_t) n_hot_init, -1);
    for (size_t global = 0; global < storage.hot_local_by_global.size(); ++global) {
        const int32_t local = storage.hot_local_by_global[global];
        if (local >= 0 && local < n_hot_init) {
            hot_global_by_local[(size_t) local] = (int32_t) global;
        }
    }

    std::vector<uint8_t> used((size_t) n_hot_init, 0);
    for (int t = 0; t < n_tokens; ++t) {
        std::fill(used.begin(), used.end(), 0);
        int next_dummy = 0;
        for (int i = 0; i < n_used; ++i) {
            const size_t idx = (size_t) t * (size_t) n_used + (size_t) i;
            out_global_ids[idx] = -1;
            out_weights[idx] = 0.0f;

            const int32_t gid = selected_ids[idx];
            if (gid < 0 || gid >= (int32_t) storage.hot_local_by_global.size()) {
                continue;
            }
            const int32_t hot_local = storage.hot_local_by_global[(size_t) gid];
            if (hot_local < 0) {
                continue;
            }
            if (hot_local >= n_hot_init) {
                if (err) *err = "hot local id exceeds initialized hot stack";
                return false;
            }
            if (used[(size_t) hot_local]) {
                if (err) *err = "duplicate hot local id in routed selection";
                return false;
            }
            out_global_ids[idx] = gid;
            out_weights[idx] = selected_weights[idx];
            used[(size_t) hot_local] = 1;
        }

        for (int i = 0; i < n_used; ++i) {
            const size_t idx = (size_t) t * (size_t) n_used + (size_t) i;
            if (out_global_ids[idx] >= 0) continue;
            while (next_dummy < n_hot_init &&
                   (used[(size_t) next_dummy] ||
                    hot_global_by_local[(size_t) next_dummy] < 0)) {
                ++next_dummy;
            }
            if (next_dummy >= n_hot_init) {
                if (err) *err = "insufficient initialized hot experts for unique remapped routing";
                return false;
            }
            out_global_ids[idx] = hot_global_by_local[(size_t) next_dummy];
            used[(size_t) next_dummy] = 1;
            ++next_dummy;
        }
    }

    return true;
}

class HybridColdWorker {
public:
    HybridColdWorker() = default;
    ~HybridColdWorker() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stopping_ = true;
        }
        task_cv_.notify_one();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    HybridColdWorker(const HybridColdWorker &) = delete;
    HybridColdWorker & operator=(const HybridColdWorker &) = delete;

    void submit(std::function<void()> task) {
        ensure_started();
        {
            std::unique_lock<std::mutex> lock(mu_);
            done_cv_.wait(lock, [this]() { return done_ && !has_task_; });
            task_ = std::move(task);
            has_task_ = true;
            done_ = false;
        }
        task_cv_.notify_one();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(mu_);
        done_cv_.wait(lock, [this]() { return done_; });
    }

private:
    void ensure_started() {
        if (worker_.joinable()) return;
        worker_ = std::thread([this]() { loop(); });
    }

    void loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mu_);
                task_cv_.wait(lock, [this]() { return stopping_ || has_task_; });
                if (stopping_ && !has_task_) {
                    return;
                }
                task = std::move(task_);
                has_task_ = false;
            }

            task();

            {
                std::lock_guard<std::mutex> lock(mu_);
                done_ = true;
            }
            done_cv_.notify_one();
        }
    }

    std::thread worker_;
    std::mutex mu_;
    std::condition_variable task_cv_;
    std::condition_variable done_cv_;
    std::function<void()> task_;
    bool has_task_ = false;
    bool done_ = true;
    bool stopping_ = false;
};

static thread_local HybridColdWorker g_hybrid_cold_worker;

static bool moe_hybrid_batched_profile_enabled() {
    const char * raw = std::getenv("DFLASH_MOE_BATCHED_PROFILE");
    return raw && *raw && std::strcmp(raw, "0") != 0 &&
        std::strcmp(raw, "false") != 0 && std::strcmp(raw, "off") != 0;
}

struct MoeHybridBatchedProfile {
    uint64_t calls = 0;
    uint64_t tokens = 0;
    uint64_t partition_us = 0;
    uint64_t remote_cold_us = 0;
    uint64_t overlap_us = 0;
    uint64_t hot_calls = 0;
    uint64_t hot_tokens = 0;
    uint64_t hot_us = 0;
    uint64_t merge_us = 0;
    uint64_t total_us = 0;

    void maybe_print(bool force) const {
        if (calls == 0) return;
        if (!force && calls % 64 != 0) return;
        std::fprintf(stderr,
            "[hybrid-ffn-batched] profile calls=%" PRIu64
            " tokens=%" PRIu64 " hot_calls=%" PRIu64
            " hot_tokens=%" PRIu64
            " partition_ms=%.3f remote_cold_ms=%.3f hot_ms=%.3f"
            " overlap_ms=%.3f merge_ms=%.3f total_ms=%.3f\n",
            calls, tokens, hot_calls, hot_tokens,
            partition_us / 1000.0, remote_cold_us / 1000.0,
            hot_us / 1000.0, overlap_us / 1000.0,
            merge_us / 1000.0, total_us / 1000.0);
    }
};

static MoeHybridBatchedProfile & moe_hybrid_batched_profile() {
    static MoeHybridBatchedProfile profile;
    return profile;
}

// Build the shared-expert FFN subgraph onto an existing ggml_context.
// Returns the output tensor (or nullptr if no shared expert is present).
static ggml_tensor * build_shared_expert_subgraph(
        ggml_context * ctx, const MoeLayerDesc & desc, ggml_tensor * inp) {
    if (!desc.ffn_up_shexp || !desc.ffn_gate_shexp || !desc.ffn_down_shexp)
        return nullptr;
    ggml_tensor * sh_gate = apply_scale2(ctx,
        ggml_mul_mat(ctx, desc.ffn_gate_shexp, inp), desc.ffn_gate_shexp_s);
    ggml_tensor * sh_up = apply_scale2(ctx,
        ggml_mul_mat(ctx, desc.ffn_up_shexp, inp), desc.ffn_up_shexp_s);
    ggml_tensor * sh_gu = ggml_swiglu_split(ctx, sh_gate, sh_up);
    ggml_tensor * shared = apply_scale2(ctx,
        ggml_mul_mat(ctx, desc.ffn_down_shexp, sh_gu), desc.ffn_down_shexp_s);
    if (desc.ffn_gate_inp_shexp) {
        // The shared-expert gate is a single-row weight (M=1): out[0,n] = sum_k W[k]*inp[k,n].
        // Computing it as ggml_mul_mat routes to cublas, and on the shipped CUDA 12.0
        // cublasLt the M=1 heuristic selects a gemv/split-K reduce algorithm whose kernel
        // is ABSENT from the library once N>1 (spec-decode verify/replay batches) — for
        // BOTH F32 (cublasSgemm SSS) and F16 (cublasGemmEx HHH splitKreduce). That poisons
        // the stream and surfaces as an illegal access in the next op. Compute the gate as
        // broadcast elementwise-mul + sum_rows instead: identical math, ggml kernels only,
        // no cublas. This is what unblocks single-pass full-batch verify.
        ggml_tensor * gate_prod = ggml_mul(ctx, inp, desc.ffn_gate_inp_shexp);
        ggml_tensor * shared_gate = apply_scale2(ctx,
            ggml_sum_rows(ctx, gate_prod), desc.ffn_gate_inp_shexp_s);
        shared_gate = ggml_sigmoid(ctx, shared_gate);
        shared = ggml_mul(ctx, shared, shared_gate);
    }
    return shared;
}

// Run routed expert subset on a given backend (GPU or CPU).
static bool run_routed_subset(ggml_backend_t backend,
                              ggml_tensor * gate_tensor,
                              ggml_tensor * up_tensor,
                              ggml_tensor * down_tensor,
                              ggml_tensor * gate_up_tensor,
                              float gate_scale,
                              float up_scale,
                              float down_scale,
                              float gate_up_scale,
                              int n_embd,
                              int n_ff_exp,
                              const float * cur_host,
                              const int32_t * selected_ids,
                              const float * selected_weights,
                              int n_selected,
                              std::vector<float> & out,
                              std::string * err) {
    out.assign((size_t)n_embd, 0.0f);
    if (n_selected <= 0) return true;

    ggml_init_params ip{};
    ip.mem_size = 32 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        if (err) *err = "ggml_init failed";
        return false;
    }

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_input(inp);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_selected, 1);
    ggml_set_input(ids);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_selected, 1);
    ggml_set_input(weights);

    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, inp, n_embd, 1, 1);
    auto validate_mm_id_tensor = [&](const char * name, ggml_tensor * t) -> bool {
        if (!t) return true;
        if (t->ne[3] != 1) {
            std::fprintf(stderr, "[hybrid-ffn] %s ptr=%p ne=[%lld,%lld,%lld,%lld]\n",
                         name, (void *)t,
                         (long long)t->ne[0], (long long)t->ne[1],
                         (long long)t->ne[2], (long long)t->ne[3]);
            if (err) {
                *err = std::string(name) + " has ne[3]=" + std::to_string((long long)t->ne[3]);
            }
            ggml_free(ctx);
            return false;
        }
        return true;
    };
    if (!validate_mm_id_tensor("gate_tensor", gate_tensor) ||
        !validate_mm_id_tensor("up_tensor", up_tensor) ||
        !validate_mm_id_tensor("down_tensor", down_tensor) ||
        !validate_mm_id_tensor("gate_up_tensor", gate_up_tensor)) {
        return false;
    }
    ggml_tensor * gu = nullptr;
    if (gate_up_tensor) {
        ggml_tensor * gate_up_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, gate_up_tensor, cur_3d, ids), gate_up_scale);
        ggml_tensor * gate_e = ggml_view_3d(ctx, gate_up_e,
            n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
            gate_up_e->nb[1], gate_up_e->nb[2], 0);
        ggml_tensor * up_e = ggml_view_3d(ctx, gate_up_e,
            n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
            gate_up_e->nb[1], gate_up_e->nb[2],
            (size_t)n_ff_exp * ggml_element_size(gate_up_e));
        gate_e = ggml_cont(ctx, gate_e);
        up_e = ggml_cont(ctx, up_e);
        gu = ggml_swiglu_split(ctx, gate_e, up_e);
    } else {
        ggml_tensor * gate_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, gate_tensor, cur_3d, ids), gate_scale);
        ggml_tensor * up_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, up_tensor, cur_3d, ids), up_scale);
        gu = ggml_swiglu_split(ctx, gate_e, up_e);
    }

    ggml_tensor * experts = apply_scale2(ctx,
        ggml_mul_mat_id(ctx, down_tensor, gu, ids), down_scale);
    ggml_tensor * w_view = ggml_reshape_3d(ctx, weights, 1, n_selected, 1);
    experts = ggml_mul(ctx, experts, w_view);

    ggml_tensor * routed = nullptr;
    for (int i = 0; i < n_selected; ++i) {
        ggml_tensor * slice = ggml_view_2d(ctx, experts, n_embd, 1, experts->nb[2],
                                           (size_t)i * experts->nb[1]);
        routed = (i == 0) ? slice : ggml_add(ctx, routed, slice);
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);
    ggml_set_output(routed);
    ggml_build_forward_expand(gf, routed);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        if (err) *err = "ggml_gallocr_alloc_graph failed";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(inp, cur_host, 0, sizeof(float) * (size_t)n_embd);
    ggml_backend_tensor_set(ids, selected_ids, 0, sizeof(int32_t) * (size_t)n_selected);
    ggml_backend_tensor_set(weights, selected_weights, 0, sizeof(float) * (size_t)n_selected);

    auto st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        if (err) *err = "ggml_backend_graph_compute failed";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(routed, out.data(), 0, sizeof(float) * (size_t)n_embd);
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

// Shared expert FFN on GPU.
static bool run_shared_ffn_gpu(ggml_backend_t backend,
                               const MoeLayerDesc & desc,
                               int n_embd,
                               const float * cur_host,
                               std::vector<float> & out,
                               std::string * err) {
    out.assign((size_t)n_embd, 0.0f);
    if (!desc.ffn_up_shexp || !desc.ffn_gate_shexp || !desc.ffn_down_shexp) {
        return true;
    }

    ggml_init_params ip{};
    ip.mem_size = 16 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        if (err) *err = "ggml_init failed";
        return false;
    }

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_input(inp);

    ggml_tensor * shared = build_shared_expert_subgraph(ctx, desc, inp);
    if (!shared) {
        ggml_free(ctx);
        return true;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 512, false);
    ggml_set_output(shared);
    ggml_build_forward_expand(gf, shared);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        if (err) *err = "ggml_gallocr_alloc_graph failed";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(inp, cur_host, 0, sizeof(float) * (size_t)n_embd);
    auto st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        if (err) *err = "ggml_backend_graph_compute failed";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_get(shared, out.data(), 0, sizeof(float) * (size_t)n_embd);
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

// Fused hot routed + shared FFN in a single GPU graph compute.
static bool run_hot_and_shared_ffn_gpu(
    ggml_backend_t backend,
    ggml_tensor * gate_tensor,
    ggml_tensor * up_tensor,
    ggml_tensor * down_tensor,
    ggml_tensor * gate_up_tensor,
    float gate_scale,
    float up_scale,
    float down_scale,
    float gate_up_scale,
    const MoeLayerDesc & desc,
    int n_embd,
    int n_ff_exp,
    const float * cur_host,
    const int32_t * hot_ids,
    const float * hot_weights,
    int n_hot,
    std::vector<float> & out,
    std::string * err) {

    out.assign((size_t)n_embd, 0.0f);

    const bool has_hot = (n_hot > 0);
    const bool has_shared = (desc.ffn_up_shexp && desc.ffn_gate_shexp && desc.ffn_down_shexp);
    if (!has_hot && !has_shared) return true;

    ggml_init_params ip{};
    ip.mem_size = 48 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        if (err) *err = "ggml_init failed";
        return false;
    }

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_input(inp);

    ggml_tensor * routed = nullptr;
    ggml_tensor * ids_tensor = nullptr;
    ggml_tensor * weights_tensor = nullptr;

    if (has_hot) {
        ids_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_hot, 1);
        ggml_set_input(ids_tensor);
        weights_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hot, 1);
        ggml_set_input(weights_tensor);

        ggml_tensor * cur_3d = ggml_reshape_3d(ctx, inp, n_embd, 1, 1);
        ggml_tensor * gu = nullptr;
        if (gate_up_tensor) {
            ggml_tensor * gate_up_e = apply_scale2(ctx,
                ggml_mul_mat_id(ctx, gate_up_tensor, cur_3d, ids_tensor), gate_up_scale);
            ggml_tensor * gate_e = ggml_view_3d(ctx, gate_up_e,
                n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
                gate_up_e->nb[1], gate_up_e->nb[2], 0);
            ggml_tensor * up_e = ggml_view_3d(ctx, gate_up_e,
                n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
                gate_up_e->nb[1], gate_up_e->nb[2],
                (size_t)n_ff_exp * ggml_element_size(gate_up_e));
            gate_e = ggml_cont(ctx, gate_e);
            up_e = ggml_cont(ctx, up_e);
            gu = ggml_swiglu_split(ctx, gate_e, up_e);
        } else {
            ggml_tensor * gate_e = apply_scale2(ctx,
                ggml_mul_mat_id(ctx, gate_tensor, cur_3d, ids_tensor), gate_scale);
            ggml_tensor * up_e = apply_scale2(ctx,
                ggml_mul_mat_id(ctx, up_tensor, cur_3d, ids_tensor), up_scale);
            gu = ggml_swiglu_split(ctx, gate_e, up_e);
        }

        ggml_tensor * experts = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, down_tensor, gu, ids_tensor), down_scale);
        ggml_tensor * w_view = ggml_reshape_3d(ctx, weights_tensor, 1, n_hot, 1);
        experts = ggml_mul(ctx, experts, w_view);

        for (int i = 0; i < n_hot; ++i) {
            ggml_tensor * slice = ggml_view_2d(ctx, experts, n_embd, 1, experts->nb[2],
                                               (size_t)i * experts->nb[1]);
            routed = (i == 0) ? slice : ggml_add(ctx, routed, slice);
        }
    }

    ggml_tensor * shared = build_shared_expert_subgraph(ctx, desc, inp);

    // Combine hot routed + shared into a single output tensor
    ggml_tensor * combined = nullptr;
    if (routed && shared) {
        combined = ggml_add(ctx, routed, shared);
    } else if (routed) {
        combined = routed;
    } else {
        combined = shared;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 2048, false);
    ggml_set_output(combined);
    ggml_build_forward_expand(gf, combined);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        if (err) *err = "fused hot+shared gallocr failed";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(inp, cur_host, 0, sizeof(float) * (size_t)n_embd);
    if (ids_tensor) {
        ggml_backend_tensor_set(ids_tensor, hot_ids, 0, sizeof(int32_t) * (size_t)n_hot);
    }
    if (weights_tensor) {
        ggml_backend_tensor_set(weights_tensor, hot_weights, 0, sizeof(float) * (size_t)n_hot);
    }

    auto st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        if (err) *err = "fused hot+shared compute failed";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(combined, out.data(), 0, sizeof(float) * (size_t)n_embd);
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

// Build batched routed graph helper for batched prefill.
static bool build_batched_routed_graph(
    ggml_context * ctx,
    ggml_tensor * gate_tensor,
    ggml_tensor * up_tensor,
    ggml_tensor * down_tensor,
    ggml_tensor * gate_up_tensor,
    float gate_scale,
    float up_scale,
    float down_scale,
    float gate_up_scale,
    ggml_tensor * inp,
    ggml_tensor * sel,
    ggml_tensor * wts,
    int n_embd, int n_ff_exp, int n_used, int n_tokens,
    ggml_tensor ** out_routed)
{
    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, inp, n_embd, 1, n_tokens);
    ggml_tensor * gu = nullptr;
    if (gate_up_tensor) {
        ggml_tensor * gate_up_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, gate_up_tensor, cur_3d, sel), gate_up_scale);
        ggml_tensor * gate_e = ggml_view_3d(ctx, gate_up_e,
            n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
            gate_up_e->nb[1], gate_up_e->nb[2], 0);
        ggml_tensor * up_e = ggml_view_3d(ctx, gate_up_e,
            n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
            gate_up_e->nb[1], gate_up_e->nb[2],
            (size_t)n_ff_exp * ggml_element_size(gate_up_e));
        gate_e = ggml_cont(ctx, gate_e);
        up_e = ggml_cont(ctx, up_e);
        gu = ggml_swiglu_split(ctx, gate_e, up_e);
    } else {
        ggml_tensor * gate_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, gate_tensor, cur_3d, sel), gate_scale);
        ggml_tensor * up_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, up_tensor, cur_3d, sel), up_scale);
        gu = ggml_swiglu_split(ctx, gate_e, up_e);
    }

    ggml_tensor * experts = apply_scale2(ctx,
        ggml_mul_mat_id(ctx, down_tensor, gu, sel), down_scale);

    // Weight and sum over experts: [n_embd, n_used, n_tokens] * [1, n_used, n_tokens]
    ggml_tensor * w_view = ggml_reshape_3d(ctx, wts, 1, n_used, n_tokens);
    experts = ggml_mul(ctx, experts, w_view);

    ggml_tensor * sum_shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * moe_sum = ggml_repeat_back(ctx, experts, sum_shape);
    *out_routed = ggml_reshape_2d(ctx, moe_sum, n_embd, n_tokens);
    return true;
}

// ── Public API ──────────────────────────────────────────────────────────────────

bool build_cached_hot_graph(
    CachedFfnGraph & out,
    ggml_backend_t backend,
    ggml_tensor * gate_tensor,
    ggml_tensor * up_tensor,
    ggml_tensor * down_tensor,
    ggml_tensor * gate_up_tensor,
    float gate_scale,
    float up_scale,
    float down_scale,
    float gate_up_scale,
    const MoeLayerDesc & desc,
    int n_embd,
    int n_ff_exp,
    int n_hot,
    bool gpu_remap,
    int n_expert) {

    out.free();
    out.n_hot = n_hot;

    ggml_init_params ip{};
    ip.mem_size = 48 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    out.inp = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_input(out.inp);

    ggml_tensor * routed = nullptr;
    if (n_hot > 0) {
        out.ids = ggml_new_tensor_2d(out.ctx, GGML_TYPE_I32, n_hot, 1);
        ggml_set_input(out.ids);
        out.weights = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_hot, 1);
        ggml_set_input(out.weights);
        if (gpu_remap && n_expert > 0) {
            out.global_ids = ggml_new_tensor_2d(out.ctx, GGML_TYPE_I32, n_hot, 1);
            ggml_set_input(out.global_ids);
            out.raw_weights = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_hot, 1);
            ggml_set_input(out.raw_weights);
            out.hot_local_lut = ggml_new_tensor_2d(out.ctx, GGML_TYPE_I32, 1, n_expert);
            ggml_set_input(out.hot_local_lut);
            out.valid_lut = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, 1, n_expert);
            ggml_set_input(out.valid_lut);
            out.residual_in = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_embd, 1);
            ggml_set_input(out.residual_in);
            ggml_tensor * lid = ggml_get_rows(out.ctx, out.hot_local_lut, out.global_ids);
            out.ids = ggml_cont(out.ctx, ggml_reshape_2d(out.ctx, lid, n_hot, 1));
            ggml_tensor * vm = ggml_get_rows(out.ctx, out.valid_lut, out.global_ids);
            vm = ggml_reshape_2d(out.ctx, vm, n_hot, 1);
            out.weights = ggml_mul(out.ctx, out.raw_weights, vm);
        }

        ggml_tensor * cur_3d = ggml_reshape_3d(out.ctx, out.inp, n_embd, 1, 1);
        ggml_tensor * gu = nullptr;
        if (gate_up_tensor) {
            ggml_tensor * gate_up_e = apply_scale2(out.ctx,
                ggml_mul_mat_id(out.ctx, gate_up_tensor, cur_3d, out.ids), gate_up_scale);
            ggml_tensor * gate_e = ggml_view_3d(out.ctx, gate_up_e,
                n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
                gate_up_e->nb[1], gate_up_e->nb[2], 0);
            ggml_tensor * up_e = ggml_view_3d(out.ctx, gate_up_e,
                n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
                gate_up_e->nb[1], gate_up_e->nb[2],
                (size_t)n_ff_exp * ggml_element_size(gate_up_e));
            gate_e = ggml_cont(out.ctx, gate_e);
            up_e = ggml_cont(out.ctx, up_e);
            gu = ggml_swiglu_split(out.ctx, gate_e, up_e);
        } else {
            ggml_tensor * gate_e = apply_scale2(out.ctx,
                ggml_mul_mat_id(out.ctx, gate_tensor, cur_3d, out.ids), gate_scale);
            ggml_tensor * up_e = apply_scale2(out.ctx,
                ggml_mul_mat_id(out.ctx, up_tensor, cur_3d, out.ids), up_scale);
            gu = ggml_swiglu_split(out.ctx, gate_e, up_e);
        }

        ggml_tensor * experts = apply_scale2(out.ctx,
            ggml_mul_mat_id(out.ctx, down_tensor, gu, out.ids), down_scale);
        ggml_tensor * w_view = ggml_reshape_3d(out.ctx, out.weights, 1, n_hot, 1);
        experts = ggml_mul(out.ctx, experts, w_view);

        for (int i = 0; i < n_hot; ++i) {
            ggml_tensor * slice = ggml_view_2d(out.ctx, experts, n_embd, 1, experts->nb[2],
                                               (size_t)i * experts->nb[1]);
            routed = (i == 0) ? slice : ggml_add(out.ctx, routed, slice);
        }
    }

    ggml_tensor * shared = build_shared_expert_subgraph(out.ctx, desc, out.inp);

    if (routed && shared) {
        out.output = ggml_add(out.ctx, routed, shared);
    } else if (routed) {
        out.output = routed;
    } else {
        out.output = shared;
    }
    if (!out.output) { out.free(); return false; }
    if (gpu_remap && out.residual_in) { out.output = ggml_cont(out.ctx, ggml_add(out.ctx, out.output, out.residual_in)); }

    out.gf = ggml_new_graph_custom(out.ctx, 2048, false);
    ggml_set_output(out.output);
    ggml_build_forward_expand(out.gf, out.output);
    out.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(out.alloc, out.gf)) {
        out.free();
        return false;
    }
    return true;
}

bool build_cached_cold_graph(
    CachedFfnGraph & out,
    ggml_backend_t cpu_backend,
    ggml_tensor * gate_tensor,
    ggml_tensor * up_tensor,
    ggml_tensor * down_tensor,
    ggml_tensor * gate_up_tensor,
    float gate_scale,
    float up_scale,
    float down_scale,
    float gate_up_scale,
    int n_embd,
    int n_ff_exp,
    int n_cold) {

    out.free();
    out.n_hot = n_cold;  // reuse field for "n experts in this graph"

    ggml_init_params ip{};
    ip.mem_size = 32 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    out.inp = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_input(out.inp);
    out.ids = ggml_new_tensor_2d(out.ctx, GGML_TYPE_I32, n_cold, 1);
    ggml_set_input(out.ids);
    out.weights = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_cold, 1);
    ggml_set_input(out.weights);

    ggml_tensor * cur_3d = ggml_reshape_3d(out.ctx, out.inp, n_embd, 1, 1);
    ggml_tensor * gu = nullptr;
    if (gate_up_tensor) {
        ggml_tensor * gate_up_e = apply_scale2(out.ctx,
            ggml_mul_mat_id(out.ctx, gate_up_tensor, cur_3d, out.ids), gate_up_scale);
        ggml_tensor * gate_e = ggml_view_3d(out.ctx, gate_up_e,
            n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
            gate_up_e->nb[1], gate_up_e->nb[2], 0);
        ggml_tensor * up_e = ggml_view_3d(out.ctx, gate_up_e,
            n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
            gate_up_e->nb[1], gate_up_e->nb[2],
            (size_t)n_ff_exp * ggml_element_size(gate_up_e));
        gate_e = ggml_cont(out.ctx, gate_e);
        up_e = ggml_cont(out.ctx, up_e);
        gu = ggml_swiglu_split(out.ctx, gate_e, up_e);
    } else {
        ggml_tensor * gate_e = apply_scale2(out.ctx,
            ggml_mul_mat_id(out.ctx, gate_tensor, cur_3d, out.ids), gate_scale);
        ggml_tensor * up_e = apply_scale2(out.ctx,
            ggml_mul_mat_id(out.ctx, up_tensor, cur_3d, out.ids), up_scale);
        gu = ggml_swiglu_split(out.ctx, gate_e, up_e);
    }

    ggml_tensor * experts = apply_scale2(out.ctx,
        ggml_mul_mat_id(out.ctx, down_tensor, gu, out.ids), down_scale);
    ggml_tensor * w_view = ggml_reshape_3d(out.ctx, out.weights, 1, n_cold, 1);
    experts = ggml_mul(out.ctx, experts, w_view);

    out.output = nullptr;
    for (int i = 0; i < n_cold; ++i) {
        ggml_tensor * slice = ggml_view_2d(out.ctx, experts, n_embd, 1, experts->nb[2],
                                           (size_t)i * experts->nb[1]);
        out.output = (i == 0) ? slice : ggml_add(out.ctx, out.output, slice);
    }
    if (!out.output) { out.free(); return false; }

    out.gf = ggml_new_graph_custom(out.ctx, 1024, false);
    ggml_set_output(out.output);
    ggml_build_forward_expand(out.gf, out.output);
    out.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu_backend));
    if (!ggml_gallocr_alloc_graph(out.alloc, out.gf)) {
        out.free();
        return false;
    }
    return true;
}

bool build_cached_hot_batched_graph(
    CachedHotBatchedGraph & out,
    ggml_backend_t gpu_backend,
    MoeHybridLayerStorage & storage,
    const MoeLayerDesc & desc,
    const MoeHybridConfig & cfg,
    int n_tokens,
    bool gpu_remap) {

    out.free();
    out.n_tokens = n_tokens;
    out.gpu_remap = gpu_remap;

    const int n_embd = cfg.n_embd;
    const int n_used = cfg.n_expert_used;
    const int n_ff_exp = cfg.n_ff_exp;

    ggml_init_params ip{};
    ip.mem_size = 128 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    out.inp = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(out.inp);
    out.sel = ggml_new_tensor_2d(out.ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_input(out.sel);
    out.wts = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_used, n_tokens);
    ggml_set_input(out.wts);

    if (gpu_remap) {
        const int total_slots = n_used * n_tokens;
        out.global_ids = ggml_new_tensor_2d(out.ctx, GGML_TYPE_I32, total_slots, 1);
        ggml_set_input(out.global_ids);
        out.raw_weights = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, total_slots, 1);
        ggml_set_input(out.raw_weights);
        out.hot_local_lut = ggml_new_tensor_2d(out.ctx, GGML_TYPE_I32, 1, cfg.n_expert);
        ggml_set_input(out.hot_local_lut);
        out.valid_lut = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, 1, cfg.n_expert);
        ggml_set_input(out.valid_lut);

        ggml_tensor * lid = ggml_get_rows(out.ctx, out.hot_local_lut, out.global_ids);
        out.sel = ggml_cont(out.ctx, ggml_reshape_2d(out.ctx, lid, n_used, n_tokens));
        ggml_tensor * vm = ggml_get_rows(out.ctx, out.valid_lut, out.global_ids);
        vm = ggml_reshape_2d(out.ctx, vm, n_used, n_tokens);
        ggml_tensor * raw_wts = ggml_reshape_2d(out.ctx, out.raw_weights,
                                                n_used, n_tokens);
        out.wts = ggml_mul(out.ctx, raw_wts, vm);
    }

    ggml_tensor * routed = nullptr;
    build_batched_routed_graph(out.ctx,
        storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
        desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
        out.inp, out.sel, out.wts, n_embd, n_ff_exp, n_used, n_tokens, &routed);

    // Shared expert (always on GPU)
    ggml_tensor * combined = routed;
    ggml_tensor * shared = build_shared_expert_subgraph(out.ctx, desc, out.inp);
    if (shared) {
        combined = combined ? ggml_add(out.ctx, combined, shared) : shared;
    }

    if (!combined) { out.free(); return false; }
    out.output = combined;

    out.gf = ggml_new_graph_custom(out.ctx, 4096, false);
    ggml_set_output(out.output);
    ggml_build_forward_expand(out.gf, out.output);
    out.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu_backend));
    if (!ggml_gallocr_alloc_graph(out.alloc, out.gf)) {
        out.free();
        return false;
    }
    return true;
}

// Cached batched COLD routed graph (CPU backend, no shared expert). Mirror of
// build_cached_hot_batched_graph for the cold expert stack; used by the mixed
// batched path so spec-decode verify/replay reuse the graph instead of
// rebuilding it every call.
static bool build_cached_cold_batched_graph(
    CachedHotBatchedGraph & out,
    ggml_backend_t cpu_backend,
    MoeHybridLayerStorage & storage,
    const MoeLayerDesc & desc,
    const MoeHybridConfig & cfg,
    int n_tokens) {

    out.free();
    out.n_tokens = n_tokens;
    const int n_embd = cfg.n_embd;
    const int n_used = cfg.n_expert_used;
    const int n_ff_exp = cfg.n_ff_exp;

    ggml_init_params ip{};
    ip.mem_size = 128 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    out.inp = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(out.inp);
    out.sel = ggml_new_tensor_2d(out.ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_input(out.sel);
    out.wts = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_used, n_tokens);
    ggml_set_input(out.wts);

    ggml_tensor * routed = nullptr;
    build_batched_routed_graph(out.ctx,
        storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
        desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
        out.inp, out.sel, out.wts, n_embd, n_ff_exp, n_used, n_tokens, &routed);
    if (!routed) { out.free(); return false; }
    out.output = routed;

    out.gf = ggml_new_graph_custom(out.ctx, 4096, false);
    ggml_set_output(out.output);
    ggml_build_forward_expand(out.gf, out.output);
    out.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu_backend));
    if (!ggml_gallocr_alloc_graph(out.alloc, out.gf)) {
        out.free();
        return false;
    }
    return true;
}

bool eval_moe_hybrid_ffn_single(
    ggml_backend_t                  gpu_backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    MoeHybridLayerStorage &         storage,
    ggml_backend_t                  cpu_backend,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_selected,
    std::vector<float> &            out,
    MoeHybridFfnTelemetry *         telemetry,
    std::string *                   err) {

    if (telemetry) *telemetry = {};
    const auto ffn_wall_t0 = HybridClock::now();
    const auto partition_t0 = HybridClock::now();

    std::vector<int32_t> hot_ids;
    std::vector<float> hot_weights;
    std::vector<int32_t> cold_ids;
    std::vector<float> cold_weights;
    for (int i = 0; i < n_selected; ++i) {
        const int32_t gid = selected_ids[i];
        if (gid < 0 || gid >= (int32_t)storage.hot_local_by_global.size()) {
            if (err) *err = "selected id out of range";
            return false;
        }
        const int32_t hot_local = storage.hot_local_by_global[(size_t)gid];
        if (hot_local >= 0) {
            hot_ids.push_back(hot_local);
            hot_weights.push_back(selected_weights[i]);
            continue;
        }
        const int32_t cold_local = storage.cold_local_by_global[(size_t)gid];
        if (cold_local >= 0) {
            cold_ids.push_back(cold_local);
            cold_weights.push_back(selected_weights[i]);
        }
    }
    const auto partition_t1 = HybridClock::now();
    if (telemetry) {
        telemetry->partition_us = elapsed_us(partition_t0, partition_t1);
        telemetry->hot_selected = (int)hot_ids.size();
        telemetry->cold_selected = (int)cold_ids.size();
    }

    std::vector<float> hot_and_shared, cold;

    const int n_hot = (int)hot_ids.size();
    const bool has_hot = (n_hot > 0);
    const bool has_shared = (desc.ffn_up_shexp && desc.ffn_gate_shexp && desc.ffn_down_shexp);
    const bool has_cold = !cold_ids.empty();
    const int n_cold = (int)cold_ids.size();

    // ── Hot + Shared path on GPU ──
    bool hot_async_launched = false;
    const auto hot_t0 = HybridClock::now();
    if (!has_hot && !has_shared) {
        hot_and_shared.assign((size_t)cfg.n_embd, 0.0f);
    } else {
        // Lazily build cached hot graph on first use
        if (!storage.hot_graph.valid() || storage.hot_graph.n_hot != n_hot) {
            build_cached_hot_graph(storage.hot_graph, gpu_backend,
                                   storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                                   desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                   desc, cfg.n_embd, cfg.n_ff_exp, n_hot);
        }
        if (storage.hot_graph.valid() && storage.hot_graph.n_hot == n_hot) {
            ggml_backend_tensor_set(storage.hot_graph.inp, cur_host, 0, sizeof(float) * (size_t)cfg.n_embd);
            if (storage.hot_graph.ids && has_hot) {
                ggml_backend_tensor_set(storage.hot_graph.ids, hot_ids.data(), 0, sizeof(int32_t) * (size_t)n_hot);
            }
            if (storage.hot_graph.weights && has_hot) {
                ggml_backend_tensor_set(storage.hot_graph.weights, hot_weights.data(), 0, sizeof(float) * (size_t)n_hot);
            }
            // Launch GPU async — kernel runs while we do cold on CPU
            ggml_backend_graph_compute_async(gpu_backend, storage.hot_graph.gf);
            hot_async_launched = true;
        } else {
            // Fallback: sync compute (no overlap)
            if (!run_hot_and_shared_ffn_gpu(gpu_backend,
                                            storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                                            desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                            desc, cfg.n_embd, cfg.n_ff_exp,
                                            cur_host,
                                            hot_ids.empty() ? nullptr : hot_ids.data(),
                                            hot_weights.empty() ? nullptr : hot_weights.data(),
                                            n_hot, hot_and_shared, err)) {
                return false;
            }
        }
    }

    // ── Cold path on CPU (overlaps with GPU kernels in flight) ──
    const auto cold_t0 = HybridClock::now();
    if (has_cold) {
        if (!storage.cold_graph.valid() || storage.cold_graph.n_hot != n_cold) {
            build_cached_cold_graph(storage.cold_graph, cpu_backend,
                                    storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
                                    desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                    cfg.n_embd, cfg.n_ff_exp, n_cold);
        }
        if (storage.cold_graph.valid() && storage.cold_graph.n_hot == n_cold) {
            ggml_backend_tensor_set(storage.cold_graph.inp, cur_host, 0, sizeof(float) * (size_t)cfg.n_embd);
            ggml_backend_tensor_set(storage.cold_graph.ids, cold_ids.data(), 0, sizeof(int32_t) * (size_t)n_cold);
            ggml_backend_tensor_set(storage.cold_graph.weights, cold_weights.data(), 0, sizeof(float) * (size_t)n_cold);
            auto st = ggml_backend_graph_compute(cpu_backend, storage.cold_graph.gf);
            if (st != GGML_STATUS_SUCCESS) {
                if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
                if (err) *err = "cached cold graph compute failed";
                return false;
            }
            cold.resize((size_t)cfg.n_embd);
            ggml_backend_tensor_get(storage.cold_graph.output, cold.data(), 0, sizeof(float) * (size_t)cfg.n_embd);
        } else {
            if (!run_routed_subset(cpu_backend,
                                   storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
                                   desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                   cfg.n_embd, cfg.n_ff_exp,
                                   cur_host, cold_ids.data(), cold_weights.data(), n_cold, cold, err)) {
                if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
                return false;
            }
        }
    } else {
        cold.assign((size_t)cfg.n_embd, 0.0f);
    }
    const auto cold_t1 = HybridClock::now();

    // ── Sync GPU and read result ──
    if ((has_hot || has_shared) && storage.hot_graph.valid() && storage.hot_graph.n_hot == n_hot) {
        ggml_backend_synchronize(gpu_backend);
        hot_and_shared.resize((size_t)cfg.n_embd);
        ggml_backend_tensor_get(storage.hot_graph.output, hot_and_shared.data(), 0, sizeof(float) * (size_t)cfg.n_embd);
    }
    const auto hot_t1 = HybridClock::now();

    if (telemetry) {
        telemetry->hot_us = elapsed_us(hot_t0, hot_t1);
        telemetry->cold_us = has_cold ? elapsed_us(cold_t0, cold_t1) : 0;
        telemetry->shared_us = 0;
    }

    const auto combine_t0 = HybridClock::now();
    out.assign((size_t)cfg.n_embd, 0.0f);
    for (int i = 0; i < cfg.n_embd; ++i) {
        out[(size_t)i] = hot_and_shared[(size_t)i] + cold[(size_t)i];
    }
    const auto combine_t1 = HybridClock::now();
    if (telemetry) {
        telemetry->combine_us = elapsed_us(combine_t0, combine_t1);
        telemetry->ffn_wall_us = elapsed_us(ffn_wall_t0, combine_t1);
    }
    return true;
}

bool eval_moe_batched_prefill_ffn(
    ggml_backend_t                  gpu_backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err) {

    const int n_embd = cfg.n_embd;
    const int n_used = cfg.n_expert_used;
    const int n_ff_exp = cfg.n_ff_exp;
    out.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
    if (n_tokens <= 0) return true;

    ggml_init_params ip{};
    ip.mem_size = 128 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        if (err) *err = "ggml_init failed";
        return false;
    }

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp);
    ggml_tensor * sel = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_input(sel);
    ggml_tensor * wts = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_used, n_tokens);
    ggml_set_input(wts);

    // Routed expert computation using full GPU expert tensors
    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, inp, n_embd, 1, n_tokens);
    ggml_tensor * gu = nullptr;
    if (desc.ffn_gate_up_exps) {
        ggml_tensor * gate_up_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, desc.ffn_gate_up_exps, cur_3d, sel), desc.ffn_gate_up_exps_s);
        ggml_tensor * gate_e = ggml_view_3d(ctx, gate_up_e,
            n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
            gate_up_e->nb[1], gate_up_e->nb[2], 0);
        ggml_tensor * up_e = ggml_view_3d(ctx, gate_up_e,
            n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
            gate_up_e->nb[1], gate_up_e->nb[2],
            (size_t)n_ff_exp * ggml_element_size(gate_up_e));
        gate_e = ggml_cont(ctx, gate_e);
        up_e = ggml_cont(ctx, up_e);
        gu = ggml_swiglu_split(ctx, gate_e, up_e);
    } else {
        ggml_tensor * gate_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, desc.ffn_gate_exps, cur_3d, sel), desc.ffn_gate_exps_s);
        ggml_tensor * up_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, desc.ffn_up_exps, cur_3d, sel), desc.ffn_up_exps_s);
        gu = ggml_swiglu_split(ctx, gate_e, up_e);
    }

    ggml_tensor * experts = apply_scale2(ctx,
        ggml_mul_mat_id(ctx, desc.ffn_down_exps, gu, sel), desc.ffn_down_exps_s);

    // Weight and sum over experts
    ggml_tensor * w_view = ggml_reshape_3d(ctx, wts, 1, n_used, n_tokens);
    experts = ggml_mul(ctx, experts, w_view);

    ggml_tensor * sum_shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * moe_sum = ggml_repeat_back(ctx, experts, sum_shape);
    ggml_tensor * routed = ggml_reshape_2d(ctx, moe_sum, n_embd, n_tokens);

    // Shared expert
    ggml_tensor * combined = routed;
    ggml_tensor * shared = build_shared_expert_subgraph(ctx, desc, inp);
    if (shared) {
        combined = combined ? ggml_add(ctx, combined, shared) : shared;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_set_output(combined);
    ggml_build_forward_expand(gf, combined);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu_backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        if (err) *err = "batched prefill gallocr failed";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(inp, cur_host, 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
    ggml_backend_tensor_set(sel, selected_ids, 0, sizeof(int32_t) * (size_t)n_used * (size_t)n_tokens);
    ggml_backend_tensor_set(wts, selected_weights, 0, sizeof(float) * (size_t)n_used * (size_t)n_tokens);

    auto st = ggml_backend_graph_compute(gpu_backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        if (err) *err = "batched prefill compute failed";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(combined, out.data(), 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

// MMQ full-batch mul_mat_id on a reduced hot stack is only stable for large
// batches. Small batches (spec verify/replay, <=~24 tokens) spread n_used*n_tokens
// slots over thousands of hot experts; that extreme imbalance hits an unbounded
// stream-k tile load in the MMQ kernel and faults (observed on sm_86, not just
// sm_75). Prefill chunks (>=64 tokens) are dense enough and run clean, so keep
// the sm_80+ fast path for them and route small batches through the proven
// <=4-token MMVQ sub-batch path.
static bool mmq_full_batch_ok(const MoeHybridConfig & cfg, int n_tokens) {
    static const int min_tokens = [](){
        const char * v = std::getenv("DFLASH_MMQ_FULL_BATCH_MIN");
        return v ? std::atoi(v) : 64;
    }();
    return cfg.mmq_safe_full_batch && n_tokens >= min_tokens;
}

// Sub-batch size for the reduced-hot-stack routed mul_mat_id. The MMQ path
// (n_tokens > 8) illegal-accesses on a REDUCED expert stack for sparse/
// imbalanced sub-64 batches (a genuine ggml-cuda MMQ mul_mat_id bug, observed
// on sm_86 + gfx1151); the MMVQ-mmid path is stable. Q4_K MMVQ-mmid handles up
// to 8 tokens on CUDA sm_80+ (MMVQ_MAX_BATCH_SIZE) and 4 on AMD. Earlier this
// had to be 1 because the F32 shared-expert gate (cublasSgemm, M=1) also faulted
// at N>1 on the shipped CUDA 12.0 cublasLt; that is now computed cublas-free
// (mul + sum_rows), so sub-batch=8 is safe and validated on sm_86. Default to 8
// on sm_80+ (CUDA), 1 elsewhere (proven single-token path on unvalidated archs);
// env override tunes per arch without a rebuild.
static int mmq_safe_sub_batch() {
    static const int v = [](){
        const char * e = std::getenv("DFLASH_MMQ_SUB_BATCH");
        if (e) return std::max(1, std::atoi(e));
#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
        return 4;
#else
        return (query_gpu_compute_sm() >= 80) ? 8 : 1;
#endif
    }();
    return v;
}

static bool eval_moe_hybrid_remote_cold_batched(
    const MoeHybridConfig &         cfg,
    const MoeHybridLayerStorage &   storage,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err,
    MoeExpertCompute *              expert_compute,
    const MoeExpertLayer *          expert_layer);

static bool eval_moe_hybrid_ffn_batched_core(
    ggml_backend_t                  gpu_backend,
    ggml_backend_t                  cpu_backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    MoeHybridLayerStorage &         storage,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err,
    ggml_gallocr_t *                p_hot_alloc,
    ggml_gallocr_t *                p_cold_alloc,
    MoeExpertCompute *              expert_compute,
    const MoeExpertLayer *          expert_layer) {

    const int n_embd = cfg.n_embd;
    const int n_used = cfg.n_expert_used;
    const int n_ff_exp = cfg.n_ff_exp;
    out.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
    if (n_tokens <= 0) return true;

    // ── Fast path: cached hot+cold batched graphs (spec-decode verify/replay) ──
    // Mixed layers used to rebuild+free their hot and cold ggml graphs on every
    // call; that graph churn (not the matmul) dominated the verify FFN time.
    // Reuse per-n_tokens cached graphs so steady-state rebuilds nothing. Large
    // prefill batches (n_tokens >= kMaxBatchedCache) fall through to the inline
    // path below.
    if (n_tokens > 0 && n_tokens < MoeHybridLayerStorage::kMaxBatchedCache) {
        const int total_slots = n_used * n_tokens;
        auto & scratch = g_mixed_routing_scratch;
        scratch.ensure(total_slots, cfg.n_expert);
        auto & hot_sel = scratch.hot_sel;
        auto & hot_wts = scratch.hot_wts;
        auto & cold_sel = scratch.cold_sel;
        auto & cold_wts = scratch.cold_wts;
        bool fp_has_cold = false;
        bool fp_has_hot = false;
        if (!prepare_unique_hot_local_slots(
                storage, selected_ids, selected_weights,
                n_tokens, n_used, hot_sel, hot_wts,
                &fp_has_hot, &fp_has_cold, err)) {
            return false;
        }
        if (fp_has_cold && !prepare_unique_cold_local_slots(
                               storage, selected_ids, selected_weights,
                               n_tokens, n_used, cold_sel, cold_wts,
                               nullptr, err)) {
            return false;
        }

        CachedHotBatchedGraph & hg = storage.hot_batched_mixed[n_tokens];
        const bool hg_ok = (hg.valid() && hg.n_tokens == n_tokens && !hg.gpu_remap)
            || build_cached_hot_batched_graph(hg, gpu_backend, storage, desc, cfg, n_tokens);
        const bool use_remote_cold = fp_has_cold && expert_compute && expert_layer;
        CachedHotBatchedGraph * cg = nullptr;
        bool cg_ok = true;
        if (fp_has_cold && !use_remote_cold) {
            cg = &storage.cold_batched_mixed[n_tokens];
            cg_ok = (cg->valid() && cg->n_tokens == n_tokens)
                || build_cached_cold_batched_graph(*cg, cpu_backend, storage, desc, cfg, n_tokens);
        }

        if (hg_ok && cg_ok) {
            // Hot (GPU, async): shared expert + routed hot (zero-weight dummy slots
            // keep an all-cold batch's shared-expert contribution).
            ggml_backend_tensor_set(hg.inp, cur_host, 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
            ggml_backend_tensor_set(hg.sel, hot_sel.data(), 0, sizeof(int32_t) * (size_t)total_slots);
            ggml_backend_tensor_set(hg.wts, hot_wts.data(), 0, sizeof(float) * (size_t)total_slots);
            ggml_backend_graph_compute_async(gpu_backend, hg.gf);

            std::vector<float> cold_partial;
            if (fp_has_cold) {
                cold_partial.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
                if (use_remote_cold) {
                    if (!eval_moe_hybrid_remote_cold_batched(
                            cfg, storage, cur_host, selected_ids,
                            selected_weights, n_tokens, cold_partial, err,
                            expert_compute, expert_layer)) {
                        ggml_backend_synchronize(gpu_backend);
                        if (err && err->empty()) *err = "batched remote cold compute failed";
                        return false;
                    }
                } else {
                    ggml_backend_tensor_set(cg->inp, cur_host, 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
                    ggml_backend_tensor_set(cg->sel, cold_sel.data(), 0, sizeof(int32_t) * (size_t)total_slots);
                    ggml_backend_tensor_set(cg->wts, cold_wts.data(), 0, sizeof(float) * (size_t)total_slots);
                    if (ggml_backend_graph_compute(cpu_backend, cg->gf) != GGML_STATUS_SUCCESS) {
                        ggml_backend_synchronize(gpu_backend);
                        if (err) *err = "batched cold cached compute failed";
                        return false;
                    }
                    ggml_backend_tensor_get(cg->output, cold_partial.data(), 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
                }
            }

            ggml_backend_synchronize(gpu_backend);
            ggml_backend_tensor_get(hg.output, out.data(), 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
            if (fp_has_cold) {
                const size_t ntot = (size_t)n_embd * (size_t)n_tokens;
                for (size_t i = 0; i < ntot; ++i) out[i] += cold_partial[i];
            }
            return true;
        }
        // build failed -> fall through to the inline rebuild path
    }

    // ── Step 1: Partition routing into hot and cold ──
    // Dummy slots use weight 0.0 and are distributed evenly across all experts
    // to avoid pathological routing imbalance that triggers OOB in MMQ stream-k.
    const int total_slots = n_used * n_tokens;
    auto & scratch = g_mixed_routing_scratch;
    scratch.ensure(total_slots, cfg.n_expert);
    auto & hot_sel = scratch.hot_sel;
    auto & hot_wts = scratch.hot_wts;
    auto & cold_sel = scratch.cold_sel;
    auto & cold_wts = scratch.cold_wts;
    bool has_hot = false, has_cold = false;
    if (!prepare_unique_hot_local_slots(
            storage, selected_ids, selected_weights,
            n_tokens, n_used, hot_sel, hot_wts,
            &has_hot, &has_cold, err)) {
        return false;
    }
    if (has_cold && !prepare_unique_cold_local_slots(
                        storage, selected_ids, selected_weights,
                        n_tokens, n_used, cold_sel, cold_wts,
                        nullptr, err)) {
        return false;
    }

    // ── Step 2: Build and run hot GPU graph (includes shared expert always) ──
    std::vector<float> hot_partial((size_t)n_embd * (size_t)n_tokens, 0.0f);
    bool hot_async_launched = false;

    ggml_context * hot_ctx = nullptr;
    ggml_cgraph * hot_gf = nullptr;
    ggml_gallocr_t hot_alloc = nullptr;
    ggml_tensor * hot_output = nullptr;

    const bool has_shared = (desc.ffn_up_shexp && desc.ffn_gate_shexp && desc.ffn_down_shexp);
    if (has_hot || has_shared) {
        ggml_init_params ip{};
        ip.mem_size = 128 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        hot_ctx = ggml_init(ip);
        if (!hot_ctx) { if (err) *err = "hot ggml_init failed"; return false; }

        ggml_tensor * inp = ggml_new_tensor_2d(hot_ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp);

        ggml_tensor * sel = nullptr;
        ggml_tensor * wts = nullptr;
        ggml_tensor * routed = nullptr;
        if (has_hot) {
            sel = ggml_new_tensor_2d(hot_ctx, GGML_TYPE_I32, n_used, n_tokens);
            ggml_set_input(sel);
            wts = ggml_new_tensor_2d(hot_ctx, GGML_TYPE_F32, n_used, n_tokens);
            ggml_set_input(wts);

            build_batched_routed_graph(hot_ctx,
                storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                inp, sel, wts, n_embd, n_ff_exp, n_used, n_tokens, &routed);
        }

        // Shared expert (always on GPU)
        ggml_tensor * combined = routed;
        ggml_tensor * shared = build_shared_expert_subgraph(hot_ctx, desc, inp);
        if (shared) {
            combined = combined ? ggml_add(hot_ctx, combined, shared) : shared;
        }
        hot_output = combined;

        hot_gf = ggml_new_graph_custom(hot_ctx, 4096, false);
        ggml_set_output(hot_output);
        ggml_build_forward_expand(hot_gf, hot_output);
        if (p_hot_alloc) {
            if (!*p_hot_alloc)
                *p_hot_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu_backend));
            hot_alloc = *p_hot_alloc;
        } else {
            hot_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu_backend));
        }
        if (!ggml_gallocr_alloc_graph(hot_alloc, hot_gf)) {
            if (err) *err = "hybrid batched hot gallocr failed";
            if (!p_hot_alloc) ggml_gallocr_free(hot_alloc);
            ggml_free(hot_ctx);
            return false;
        }

        ggml_backend_tensor_set(inp, cur_host, 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
        if (has_hot) {
            ggml_backend_tensor_set(sel, hot_sel.data(), 0, sizeof(int32_t) * (size_t)total_slots);
            ggml_backend_tensor_set(wts, hot_wts.data(), 0, sizeof(float) * (size_t)total_slots);
        }

        // Launch GPU async
        ggml_backend_graph_compute_async(gpu_backend, hot_gf);
        hot_async_launched = true;
    }

    // ── Step 3: Build and run cold CPU graph (overlaps with GPU) ──
    auto & split_scratch = g_hybrid_split_scratch;
    auto & cold_partial = split_scratch.cold_partial;
    cold_partial.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);

    if (has_cold) {
        if (expert_compute && expert_layer) {
            if (!eval_moe_hybrid_remote_cold_batched(
                    cfg, storage, cur_host, selected_ids, selected_weights,
                    n_tokens, cold_partial, err, expert_compute, expert_layer)) {
                if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
                if (!p_hot_alloc && hot_alloc) ggml_gallocr_free(hot_alloc);
                if (hot_ctx) ggml_free(hot_ctx);
                if (err && err->empty()) *err = "hybrid batched remote cold compute failed";
                return false;
            }
        } else {
            ggml_init_params ip{};
            ip.mem_size = 128 * 1024 * 1024;
            ip.mem_buffer = nullptr;
            ip.no_alloc = true;
            ggml_context * cold_ctx = ggml_init(ip);
            if (!cold_ctx) {
                if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
                if (!p_hot_alloc && hot_alloc) ggml_gallocr_free(hot_alloc);
                if (hot_ctx) ggml_free(hot_ctx);
                if (err) *err = "cold ggml_init failed";
                return false;
            }

            ggml_tensor * inp = ggml_new_tensor_2d(cold_ctx, GGML_TYPE_F32, n_embd, n_tokens);
            ggml_set_input(inp);
            ggml_tensor * sel = ggml_new_tensor_2d(cold_ctx, GGML_TYPE_I32, n_used, n_tokens);
            ggml_set_input(sel);
            ggml_tensor * wts = ggml_new_tensor_2d(cold_ctx, GGML_TYPE_F32, n_used, n_tokens);
            ggml_set_input(wts);

            ggml_tensor * cold_routed = nullptr;
            build_batched_routed_graph(cold_ctx,
                storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
                desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                inp, sel, wts, n_embd, n_ff_exp, n_used, n_tokens, &cold_routed);

            ggml_cgraph * cold_gf = ggml_new_graph_custom(cold_ctx, 4096, false);
            ggml_set_output(cold_routed);
            ggml_build_forward_expand(cold_gf, cold_routed);
            ggml_gallocr_t cold_alloc;
            if (p_cold_alloc) {
                if (!*p_cold_alloc)
                    *p_cold_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu_backend));
                cold_alloc = *p_cold_alloc;
            } else {
                cold_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu_backend));
            }
            if (!ggml_gallocr_alloc_graph(cold_alloc, cold_gf)) {
                if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
                if (!p_hot_alloc && hot_alloc) ggml_gallocr_free(hot_alloc);
                if (hot_ctx) ggml_free(hot_ctx);
                if (!p_cold_alloc) ggml_gallocr_free(cold_alloc);
                ggml_free(cold_ctx);
                if (err) *err = "hybrid batched cold gallocr failed";
                return false;
            }

            ggml_backend_tensor_set(inp, cur_host, 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
            ggml_backend_tensor_set(sel, cold_sel.data(), 0, sizeof(int32_t) * (size_t)total_slots);
            ggml_backend_tensor_set(wts, cold_wts.data(), 0, sizeof(float) * (size_t)total_slots);

            auto st = ggml_backend_graph_compute(cpu_backend, cold_gf);
            if (st != GGML_STATUS_SUCCESS) {
                if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
                if (!p_hot_alloc && hot_alloc) ggml_gallocr_free(hot_alloc);
                if (hot_ctx) ggml_free(hot_ctx);
                if (!p_cold_alloc) ggml_gallocr_free(cold_alloc);
                ggml_free(cold_ctx);
                if (err) *err = "hybrid batched cold compute failed";
                return false;
            }

            ggml_backend_tensor_get(cold_routed, cold_partial.data(), 0,
                sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
            if (!p_cold_alloc) ggml_gallocr_free(cold_alloc);
            ggml_free(cold_ctx);
        }
    }

    // ── Step 4: Sync GPU and read hot result ──
    if (hot_async_launched) {
        ggml_backend_synchronize(gpu_backend);
        ggml_backend_tensor_get(hot_output, hot_partial.data(), 0,
            sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
    }
    if (!p_hot_alloc && hot_alloc) ggml_gallocr_free(hot_alloc);
    if (hot_ctx) ggml_free(hot_ctx);

    // ── Step 5: Merge hot + cold ──
    const size_t total_floats = (size_t)n_embd * (size_t)n_tokens;
    for (size_t i = 0; i < total_floats; ++i) {
        out[i] = hot_partial[i] + cold_partial[i];
    }

    return true;
}

static bool eval_moe_hybrid_remote_cold_batched(
    const MoeHybridConfig &         cfg,
    const MoeHybridLayerStorage &   storage,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err,
    MoeExpertCompute *              expert_compute,
    const MoeExpertLayer *          expert_layer) {
    const int n_embd = cfg.n_embd;
    const int n_used = cfg.n_expert_used;
    const int n_ff_exp = cfg.n_ff_exp;
    out.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
    if (n_tokens <= 0) return true;
    if (!expert_compute || !expert_layer) return true;

    auto & scratch = g_remote_cold_batch_scratch;
    scratch.ensure(n_tokens, n_used, n_embd, n_used);
    auto & cold_counts = scratch.cold_counts;
    auto & cold_sel = scratch.cold_sel;
    auto & cold_wts = scratch.cold_wts;
    int max_cold_selected = 0;

    for (int t = 0; t < n_tokens; ++t) {
        int count = 0;
        for (int i = 0; i < n_used; ++i) {
            const size_t src = (size_t)t * (size_t)n_used + (size_t)i;
            const int32_t gid = selected_ids[src];
            if (gid < 0 || gid >= (int32_t)storage.hot_local_by_global.size()) {
                continue;
            }
            if (storage.hot_local_by_global[(size_t)gid] >= 0) {
                continue;
            }
            const int32_t cold_lid = storage.cold_local_by_global[(size_t)gid];
            if (cold_lid >= 0) {
                const size_t dst = (size_t)t * (size_t)n_used + (size_t)count;
                cold_sel[dst] = cold_lid;
                cold_wts[dst] = selected_weights[src];
                count++;
            }
        }
        cold_counts[(size_t)t] = count;
        max_cold_selected = std::max(max_cold_selected, count);
    }

    if (max_cold_selected == 0) return true;
    if (expert_layer->cold_global_by_local.empty()) {
        if (err) *err = "hybrid batched remote cold layer has no cold experts";
        return false;
    }
    auto & group_ids = scratch.group_ids;
    auto & group_wts = scratch.group_wts;
    if (group_ids.capacity() < (size_t)n_tokens * (size_t)max_cold_selected) {
        group_ids.reserve((size_t)n_tokens * (size_t)max_cold_selected);
    }
    if (group_wts.capacity() < (size_t)n_tokens * (size_t)max_cold_selected) {
        group_wts.reserve((size_t)n_tokens * (size_t)max_cold_selected);
    }

    const char * pack_env = std::getenv("DFLASH_MOE_REMOTE_COLD_PACK");
    const bool pack_env_set = pack_env && *pack_env;
    const bool padded_pack = pack_env_set
        ? (std::strcmp(pack_env, "padded") == 0 ||
           std::strcmp(pack_env, "single") == 0)
        : expert_compute->prefers_padded_batch();
    if (padded_pack) {
        const int cold_stack = (int)expert_layer->cold_global_by_local.size();
        if (cold_stack < max_cold_selected) {
            if (err) *err = "hybrid batched remote cold padded pack exceeds cold stack";
            return false;
        }
        group_ids.resize((size_t)n_tokens * (size_t)max_cold_selected);
        group_wts.resize((size_t)n_tokens * (size_t)max_cold_selected);
        std::vector<uint8_t> used((size_t)cold_stack, 0);
        for (int t = 0; t < n_tokens; ++t) {
            const int count = cold_counts[(size_t)t];
            std::fill(used.begin(), used.end(), 0);
            int next_dummy = 0;
            for (int i = 0; i < max_cold_selected; ++i) {
                const size_t dst = (size_t)t * (size_t)max_cold_selected + (size_t)i;
                if (i < count) {
                    const size_t src = (size_t)t * (size_t)n_used + (size_t)i;
                    const int32_t cold_local = cold_sel[src];
                    if (cold_local < 0 || cold_local >= cold_stack) {
                        if (err) *err = "hybrid batched remote cold local id out of range";
                        return false;
                    }
                    if (used[(size_t)cold_local]) {
                        if (err) *err = "hybrid batched remote cold duplicate local id";
                        return false;
                    }
                    group_ids[dst] = cold_local;
                    group_wts[dst] = cold_wts[src];
                    used[(size_t)cold_local] = 1;
                } else {
                    while (next_dummy < cold_stack && used[(size_t)next_dummy]) {
                        ++next_dummy;
                    }
                    if (next_dummy >= cold_stack) {
                        if (err) *err = "hybrid batched remote cold lacks unique padded ids";
                        return false;
                    }
                    group_ids[dst] = next_dummy;
                    group_wts[dst] = 0.0f;
                    used[(size_t)next_dummy] = 1;
                    ++next_dummy;
                }
            }
        }
        if (!expert_compute->compute_batch(*expert_layer,
                                           cur_host,
                                           group_ids.data(),
                                           group_wts.data(),
                                           n_tokens,
                                           max_cold_selected,
                                           n_embd, n_ff_exp,
                                           out.data())) {
            if (err) *err = "hybrid batched remote cold padded compute failed";
            return false;
        }
        return true;
    }

    auto & token_groups = scratch.token_groups;
    if (token_groups.size() < (size_t)max_cold_selected + 1) {
        token_groups.resize((size_t)max_cold_selected + 1);
    }
    auto & active_groups = scratch.active_groups;
    for (int n_cold : active_groups) {
        if (n_cold >= 0 && (size_t)n_cold < token_groups.size()) {
            token_groups[(size_t)n_cold].clear();
        }
    }
    active_groups.clear();
    for (int t = 0; t < n_tokens; ++t) {
        const int count = cold_counts[(size_t)t];
        if (count > 0) {
            auto & bucket = token_groups[(size_t)count];
            if (bucket.empty()) {
                active_groups.push_back(count);
            }
            bucket.push_back(t);
        }
    }

    auto & group_input = scratch.group_input;
    auto & group_output = scratch.group_output;
    if (group_input.capacity() < (size_t)n_embd * (size_t)n_tokens) {
        group_input.reserve((size_t)n_embd * (size_t)n_tokens);
    }
    if (group_output.capacity() < (size_t)n_embd * (size_t)n_tokens) {
        group_output.reserve((size_t)n_embd * (size_t)n_tokens);
    }
    for (int n_cold : active_groups) {
        const auto & token_group = token_groups[(size_t)n_cold];
        if (token_group.empty()) continue;
        const size_t group_limit = (size_t)std::max(
            1, std::min(
                   moe_expert_compute_batch_limit_from_env(),
                   (int)token_group.size()));
        for (size_t base = 0; base < token_group.size(); base += group_limit) {
            const int tc = (int)std::min(group_limit, token_group.size() - base);
            bool contiguous_tokens = true;
            for (int gi = 1; gi < tc; ++gi) {
                if (token_group[base + (size_t)gi] != token_group[base] + gi) {
                    contiguous_tokens = false;
                    break;
                }
            }

            const int first_token = token_group[base];
            const float * compute_input = contiguous_tokens
                ? cur_host + (size_t)first_token * (size_t)n_embd
                : nullptr;
            float * compute_output = contiguous_tokens
                ? out.data() + (size_t)first_token * (size_t)n_embd
                : nullptr;
            if (!contiguous_tokens) {
                group_input.resize((size_t)tc * (size_t)n_embd);
                group_output.resize((size_t)tc * (size_t)n_embd);
                compute_input = group_input.data();
                compute_output = group_output.data();
            }

            group_ids.resize((size_t)tc * (size_t)n_cold);
            group_wts.resize((size_t)tc * (size_t)n_cold);
            for (int gi = 0; gi < tc; ++gi) {
                const int t = token_group[base + (size_t)gi];
                if (!contiguous_tokens) {
                    std::memcpy(group_input.data() + (size_t)gi * (size_t)n_embd,
                                cur_host + (size_t)t * (size_t)n_embd,
                                sizeof(float) * (size_t)n_embd);
                }
                for (int i = 0; i < n_cold; ++i) {
                    const size_t src = (size_t)t * (size_t)n_used + (size_t)i;
                    const size_t dst = (size_t)gi * (size_t)n_cold + (size_t)i;
                    group_ids[dst] = cold_sel[src];
                    group_wts[dst] = cold_wts[src];
                }
            }

            if (!expert_compute->compute_batch(*expert_layer,
                                               compute_input,
                                               group_ids.data(),
                                               group_wts.data(),
                                               tc, n_cold,
                                               n_embd, n_ff_exp,
                                               compute_output)) {
                if (err) *err = "hybrid batched remote cold compute failed";
                return false;
            }

            if (!contiguous_tokens) {
                for (int gi = 0; gi < tc; ++gi) {
                    const int t = token_group[base + (size_t)gi];
                    std::memcpy(out.data() + (size_t)t * (size_t)n_embd,
                                group_output.data() + (size_t)gi * (size_t)n_embd,
                                sizeof(float) * (size_t)n_embd);
                }
            }
        }
        token_groups[(size_t)n_cold].clear();
    }
    return true;
}

static bool eval_moe_hot_remap_batched(
    ggml_backend_t                  gpu_backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    MoeHybridLayerStorage &         storage,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err) {
    const int n_embd = cfg.n_embd;
    const int n_used = cfg.n_expert_used;
    const int total_slots = n_used * n_tokens;
    out.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
    if (n_tokens <= 0) return true;
    if (!cur_host || !selected_ids || !selected_weights) return false;
    if ((int)storage.hot_local_by_global.size() < cfg.n_expert) {
        if (err) *err = "hot-remap batched graph lacks hot-local LUT";
        return false;
    }

    auto & cached = (n_tokens > 0 && n_tokens < MoeHybridLayerStorage::kMaxBatchedCache)
                  ? storage.hot_batched_mixed[n_tokens]
                  : storage.hot_batched_graph;
    if (!cached.valid() || cached.n_tokens != n_tokens || !cached.gpu_remap ||
        !cached.global_ids || !cached.raw_weights ||
        !cached.hot_local_lut || !cached.valid_lut) {
        if (!build_cached_hot_batched_graph(cached, gpu_backend, storage,
                                            desc, cfg, n_tokens,
                                            /*gpu_remap=*/true)) {
            if (err) *err = "hot-remap batched graph build failed";
            return false;
        }
    }

    auto & scratch = g_mixed_routing_scratch;
    scratch.ensure(total_slots, cfg.n_expert);
    auto & lut = scratch.lut;
    auto & vlut = scratch.vlut;
    for (int expert = 0; expert < cfg.n_expert; ++expert) {
        const int32_t local = storage.hot_local_by_global[(size_t)expert];
        if (local >= 0) {
            lut[(size_t)expert] = local;
            vlut[(size_t)expert] = 1.0f;
        } else {
            lut[(size_t)expert] = 0;
            vlut[(size_t)expert] = 0.0f;
        }
    }

    ggml_backend_tensor_set(cached.inp, cur_host, 0,
        sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
    ggml_backend_tensor_set(cached.hot_local_lut, lut.data(), 0,
        sizeof(int32_t) * (size_t)cfg.n_expert);
    ggml_backend_tensor_set(cached.valid_lut, vlut.data(), 0,
        sizeof(float) * (size_t)cfg.n_expert);
    ggml_backend_tensor_set(cached.global_ids, selected_ids, 0,
        sizeof(int32_t) * (size_t)total_slots);
    ggml_backend_tensor_set(cached.raw_weights, selected_weights, 0,
        sizeof(float) * (size_t)total_slots);

    if (ggml_backend_graph_compute(gpu_backend, cached.gf) != GGML_STATUS_SUCCESS) {
        if (err) *err = "hot-remap batched graph compute failed";
        return false;
    }
    ggml_backend_tensor_get(cached.output, out.data(), 0,
        sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
    return true;
}

static bool eval_moe_hybrid_ffn_split_hot_remote_cold(
    ggml_backend_t                  gpu_backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    MoeHybridLayerStorage &         storage,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    int                             hot_sub_batch,
    std::vector<float> &            out,
    std::string *                   err,
    ggml_gallocr_t *                p_hot_alloc,
    MoeExpertCompute *              expert_compute,
    const MoeExpertLayer *          expert_layer) {
    const int n_embd = cfg.n_embd;
    const int n_used = cfg.n_expert_used;
    std::fprintf(stderr,
                 "[hybrid-ffn] split_hot_remote_cold n_used=%d n_tokens=%d hot_sub_batch=%d\n",
                 n_used, n_tokens, hot_sub_batch);
    out.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
    const bool profile_enabled = moe_hybrid_batched_profile_enabled();
    const auto total_t0 = profile_enabled ? HybridClock::now() : HybridClock::time_point{};

    auto & split_scratch = g_hybrid_split_scratch;
    auto & cold_partial = split_scratch.cold_partial;
    cold_partial.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
    const auto cold_t0 = profile_enabled ? HybridClock::now() : HybridClock::time_point{};
    bool cold_ok = false;
    std::string cold_err;
    g_hybrid_cold_worker.submit([&]() {
        cold_ok = eval_moe_hybrid_remote_cold_batched(
            cfg, storage, cur_host, selected_ids, selected_weights,
            n_tokens, cold_partial, &cold_err, expert_compute, expert_layer);
    });

    auto & hot_out = split_scratch.hot_out;
    bool hot_ok = true;
    std::string hot_err;
    uint64_t local_hot_calls = 0;
    uint64_t local_hot_tokens = 0;
    uint64_t local_hot_us = 0;
    for (int t0 = 0; t0 < n_tokens; t0 += hot_sub_batch) {
        const int tc = std::min(hot_sub_batch, n_tokens - t0);
        const auto hot_t0 = profile_enabled ? HybridClock::now() : HybridClock::time_point{};
        if (!eval_moe_hot_remap_batched(
                gpu_backend, cfg, desc, storage,
                cur_host + (size_t)t0 * (size_t)n_embd,
                selected_ids + (size_t)t0 * (size_t)n_used,
                selected_weights + (size_t)t0 * (size_t)n_used,
                tc, hot_out, &hot_err)) {
            hot_ok = false;
            break;
        }
        const auto hot_t1 = profile_enabled ? HybridClock::now() : HybridClock::time_point{};
        if (profile_enabled) {
            local_hot_calls++;
            local_hot_tokens += (uint64_t)tc;
            local_hot_us += elapsed_us(hot_t0, hot_t1);
        }
        std::memcpy(out.data() + (size_t)t0 * (size_t)n_embd,
                    hot_out.data(),
                    sizeof(float) * (size_t)n_embd * (size_t)tc);
    }

    const auto overlap_wait_t0 = profile_enabled ? HybridClock::now() : HybridClock::time_point{};
    g_hybrid_cold_worker.wait();
    const auto overlap_wait_t1 = profile_enabled ? HybridClock::now() : HybridClock::time_point{};
    const auto cold_t1 = profile_enabled ? HybridClock::now() : HybridClock::time_point{};
    if (!hot_ok) {
        if (err) *err = hot_err.empty() ? "hybrid batched hot compute failed" : hot_err;
        return false;
    }
    if (!cold_ok) {
        if (err) *err = cold_err.empty() ? "hybrid batched remote cold compute failed" : cold_err;
        return false;
    }

    const auto merge_t0 = profile_enabled ? HybridClock::now() : HybridClock::time_point{};
    const size_t total_floats = (size_t)n_embd * (size_t)n_tokens;
    for (size_t i = 0; i < total_floats; ++i) {
        out[i] += cold_partial[i];
    }
    const auto merge_t1 = profile_enabled ? HybridClock::now() : HybridClock::time_point{};
    if (profile_enabled) {
        const auto total_t1 = HybridClock::now();
        auto & prof = moe_hybrid_batched_profile();
        prof.calls++;
        prof.tokens += (uint64_t)n_tokens;
        prof.remote_cold_us += elapsed_us(cold_t0, cold_t1);
        prof.hot_calls += local_hot_calls;
        prof.hot_tokens += local_hot_tokens;
        prof.hot_us += local_hot_us;
        prof.overlap_us += elapsed_us(overlap_wait_t0, overlap_wait_t1);
        prof.merge_us += elapsed_us(merge_t0, merge_t1);
        prof.total_us += elapsed_us(total_t0, total_t1);
        prof.maybe_print(false);
    }
    return true;
}

// ── Hot-Only Batched Prefill ──
// When all selected experts are in VRAM, skip cold entirely: no CPU graph,
// no partition into hot/cold, no merge loop. Pure GPU.

bool eval_moe_hot_only_batched(
    ggml_backend_t                  gpu_backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    MoeHybridLayerStorage &         storage,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err,
    ggml_gallocr_t *                p_hot_alloc) {

    const int n_embd = cfg.n_embd;
    const int n_used = cfg.n_expert_used;
    const int n_ff_exp = cfg.n_ff_exp;
    out.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
    if (n_tokens <= 0) return true;

    // The batched hot-only fast path has only been validated on the legacy
    // top-k<=8 shapes. Larger routed top-k values, such as Ornith's
    // n_expert_used=10, should stay on the hybrid path until their mul_mat_id
    // shapes are explicitly verified.
    if (n_used > 8) {
        std::fprintf(stderr,
                     "[hybrid-ffn] reject hot_only batched path n_used=%d n_tokens=%d\n",
                     n_used, n_tokens);
        if (err) *err = "hot_only batched fast path disabled for n_expert_used > 8";
        return false;
    }

    // Workaround for the ggml-cuda MMQ mul_mat_id stream-k fault on a REDUCED
    // hot stack (sm_75/gfx1151 AND sm_86): slice sub-64 batches to a size the
    // MMVQ-mmid path handles. See mmq_safe_sub_batch().
    const int n_hot_stack = storage.gate_up_hot ? (int)storage.gate_up_hot->ne[2]
                          : storage.gate_hot    ? (int)storage.gate_hot->ne[2]
                          : 0;
    const int MMQ_SAFE_SUB_BATCH = mmq_safe_sub_batch();
    if (!mmq_full_batch_ok(cfg, n_tokens)
        && n_hot_stack > 0 && n_tokens > MMQ_SAFE_SUB_BATCH) {
        std::vector<float> sub_out;
        for (int t0 = 0; t0 < n_tokens; t0 += MMQ_SAFE_SUB_BATCH) {
            const int tc = std::min(MMQ_SAFE_SUB_BATCH, n_tokens - t0);
            if (!eval_moe_hot_only_batched(
                    gpu_backend, cfg, desc, storage,
                    cur_host + (size_t)t0 * (size_t)n_embd,
                    selected_ids + (size_t)t0 * (size_t)n_used,
                    selected_weights + (size_t)t0 * (size_t)n_used,
                    tc, sub_out, err, p_hot_alloc)) {
                return false;
            }
            std::memcpy(out.data() + (size_t)t0 * (size_t)n_embd,
                        sub_out.data(),
                        sizeof(float) * (size_t)n_embd * (size_t)tc);
        }
        return true;
    }

    // ── Fast path: use cached graph (avoids rebuild + realloc) ──
    // Per-n_tokens cache: spec-decode alternates verify (verify_width) and
    // replay (commit_n) sizes; a single slot would rebuild all 40 layers' FFN
    // graphs on every flip. Index by n_tokens so each size keeps its own graph.
    auto & cached = (n_tokens > 0 && n_tokens < MoeHybridLayerStorage::kMaxBatchedCache)
                  ? storage.hot_batched_mixed[n_tokens]
                  : storage.hot_batched_graph;
    const bool use_remap = cached.valid() && cached.gpu_remap;
    if (cached.n_tokens == n_tokens && cached.valid() && use_remap) {
        const int total_slots = n_used * n_tokens;
        auto & scratch = g_mixed_routing_scratch;
        scratch.ensure(total_slots, cfg.n_expert);
        auto & lut = scratch.lut;
        auto & vlut = scratch.vlut;
        auto & hot_sel = scratch.hot_sel;
        auto & hot_wts = scratch.hot_wts;
        for (int expert = 0; expert < cfg.n_expert; ++expert) {
            const int32_t local = storage.hot_local_by_global[(size_t)expert];
            if (local >= 0) {
                lut[(size_t)expert] = local;
                vlut[(size_t)expert] = 1.0f;
            } else {
                lut[(size_t)expert] = 0;
                vlut[(size_t)expert] = 0.0f;
            }
        }
        ggml_backend_tensor_set(cached.inp, cur_host, 0,
            sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
        ggml_backend_tensor_set(cached.hot_local_lut, lut.data(), 0,
            sizeof(int32_t) * (size_t)cfg.n_expert);
        ggml_backend_tensor_set(cached.valid_lut, vlut.data(), 0,
            sizeof(float) * (size_t)cfg.n_expert);
        if (!prepare_unique_hot_global_slots(
                storage, selected_ids, selected_weights,
                n_tokens, n_used, hot_sel, hot_wts, err)) {
            return false;
        }
        ggml_backend_tensor_set(cached.global_ids, hot_sel.data(), 0,
            sizeof(int32_t) * (size_t)total_slots);
        ggml_backend_tensor_set(cached.raw_weights, hot_wts.data(), 0,
            sizeof(float) * (size_t)total_slots);
        auto st = ggml_backend_graph_compute(gpu_backend, cached.gf);
        if (st != GGML_STATUS_SUCCESS) {
            if (err) *err = "hot_only remap cached compute failed";
            return false;
        }
        ggml_backend_tensor_get(cached.output, out.data(), 0,
            sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
        return true;
    }
    if (cached.n_tokens == n_tokens && cached.valid() && !cached.gpu_remap) {
        // Reuse pre-built graph: just upload data and compute
        const int total_slots = n_used * n_tokens;
        auto & scratch = g_mixed_routing_scratch;
        scratch.ensure(total_slots, cfg.n_expert);
        auto & hot_sel = scratch.hot_sel;
        auto & hot_wts = scratch.hot_wts;
        bool has_hot = false;
        bool has_cold = false;
        if (!prepare_unique_hot_local_slots(
                storage, selected_ids, selected_weights,
                n_tokens, n_used, hot_sel, hot_wts,
                &has_hot, &has_cold, err)) {
            return false;
        }
        ggml_backend_tensor_set(cached.inp, cur_host, 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
        ggml_backend_tensor_set(cached.sel, hot_sel.data(), 0, sizeof(int32_t) * (size_t)total_slots);
        ggml_backend_tensor_set(cached.wts, hot_wts.data(), 0, sizeof(float) * (size_t)total_slots);

        auto st = ggml_backend_graph_compute(gpu_backend, cached.gf);
        if (st != GGML_STATUS_SUCCESS) {
            if (err) *err = "hot_only cached compute failed";
            return false;
        }
        ggml_backend_tensor_get(cached.output, out.data(), 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
        return true;
    }

    // ── Slow path: build graph (first call or size mismatch) ──
    // Try to build and cache for this n_tokens size.
    // Cache when: sub-batch size (legacy), full stack (all hot), or full-batch safe (sm_80+).
    if (mmq_full_batch_ok(cfg, n_tokens) || n_tokens <= MMQ_SAFE_SUB_BATCH
        || (n_hot_stack == 0 || n_hot_stack >= cfg.n_expert)) {
        if (build_cached_hot_batched_graph(cached, gpu_backend, storage, desc, cfg, n_tokens,
                                           /*gpu_remap=*/true)) {
            // Successfully cached — use it immediately
            return eval_moe_hot_remap_batched(gpu_backend, cfg, desc, storage,
                                              cur_host, selected_ids,
                                              selected_weights, n_tokens,
                                              out, err);
        }
        // Fall through to uncached path if build fails
    }

    // ── Uncached fallback (remainder sub-batches with n_tokens < MMQ_SAFE_SUB_BATCH) ──
    ggml_init_params ip{};
    ip.mem_size = 128 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { if (err) *err = "hot_only ggml_init failed"; return false; }

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp);
    ggml_tensor * sel = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_input(sel);
    ggml_tensor * wts = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_used, n_tokens);
    ggml_set_input(wts);

    ggml_tensor * routed = nullptr;
    build_batched_routed_graph(ctx,
        storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
        desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
        inp, sel, wts, n_embd, n_ff_exp, n_used, n_tokens, &routed);

    // Shared expert (always on GPU)
    ggml_tensor * combined = routed;
    ggml_tensor * shared = build_shared_expert_subgraph(ctx, desc, inp);
    if (shared) {
        combined = combined ? ggml_add(ctx, combined, shared) : shared;
    }

    if (!combined) {
        ggml_free(ctx);
        if (err) *err = "hot_only: no routed or shared output";
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_set_output(combined);
    ggml_build_forward_expand(gf, combined);

    ggml_gallocr_t alloc = nullptr;
    if (p_hot_alloc) {
        if (!*p_hot_alloc)
            *p_hot_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu_backend));
        alloc = *p_hot_alloc;
    } else {
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu_backend));
    }
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        if (err) *err = "hot_only gallocr failed";
        if (!p_hot_alloc) ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(inp, cur_host, 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
    if (!cached.gpu_remap) {
        const int total_slots = n_used * n_tokens;
        auto & scratch = g_mixed_routing_scratch;
        scratch.ensure(total_slots, cfg.n_expert);
        auto & hot_sel = scratch.hot_sel;
        auto & hot_wts = scratch.hot_wts;
        bool has_hot = false;
        bool has_cold = false;
        if (!prepare_unique_hot_local_slots(
                storage, selected_ids, selected_weights,
                n_tokens, n_used, hot_sel, hot_wts,
                &has_hot, &has_cold, err)) {
            if (!p_hot_alloc) ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return false;
        }
        ggml_backend_tensor_set(sel, hot_sel.data(), 0, sizeof(int32_t) * (size_t)total_slots);
        ggml_backend_tensor_set(wts, hot_wts.data(), 0, sizeof(float) * (size_t)total_slots);
    } else {
        if (err) *err = "hot_only remap cached graph should be handled before uncached fallback";
        if (!p_hot_alloc) ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    auto st = ggml_backend_graph_compute(gpu_backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        if (err) *err = "hot_only compute failed";
        if (!p_hot_alloc) ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(combined, out.data(), 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
    if (!p_hot_alloc) ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

// ── GPU-Resident Residual State ──

// Public entry. Workaround for a ggml-cuda/HIP defect: the MMQ mul_mat_id
// kernel illegal-accesses on gfx1151 when the per-layer hot expert stack is
// REDUCED (n_hot_stack < n_expert); the full-stack (all-hot) case is fine.
// MMVQ is used instead of MMQ only when the matmul batch dim (= n_tokens) is
// small (Q4_K AMD MMVQ-mmid cap is 4). So for reduced hot stacks we slice the
// prefill batch into <=4-token sub-batches, routing the routed mul_mat_id
// through the stable MMVQ path. Full stacks keep the fast single-shot MMQ.
bool eval_moe_hybrid_ffn_batched(
    ggml_backend_t                  gpu_backend,
    ggml_backend_t                  cpu_backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    MoeHybridLayerStorage &         storage,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err,
    ggml_gallocr_t *                p_hot_alloc,
    ggml_gallocr_t *                p_cold_alloc,
    MoeExpertCompute *              expert_compute,
    const MoeExpertLayer *          expert_layer) {
    if (cfg.n_expert_used > 8) {
        std::fprintf(stderr,
                     "[hybrid-ffn] disable batched path for n_used=%d n_tokens=%d\n",
                     cfg.n_expert_used, n_tokens);
        if (err) *err = "hybrid batched path disabled for n_expert_used > 8";
        return false;
    }
    const int n_hot_stack = storage.gate_up_hot ? (int)storage.gate_up_hot->ne[2]
                          : storage.gate_hot    ? (int)storage.gate_hot->ne[2]
                          : 0;
    const int MMQ_SAFE_SUB_BATCH = mmq_safe_sub_batch();
    std::fprintf(stderr,
                 "[hybrid-ffn] batched entry n_used=%d n_tokens=%d n_hot_stack=%d mmq_safe=%d remote_cold=%d\n",
                 cfg.n_expert_used, n_tokens, n_hot_stack, MMQ_SAFE_SUB_BATCH,
                 (expert_compute && expert_layer) ? 1 : 0);
    if (!mmq_full_batch_ok(cfg, n_tokens)
        && n_hot_stack > 0 && n_tokens > MMQ_SAFE_SUB_BATCH) {
        if (expert_compute && expert_layer) {
            return eval_moe_hybrid_ffn_split_hot_remote_cold(
                gpu_backend, cfg, desc, storage,
                cur_host, selected_ids, selected_weights,
                n_tokens, MMQ_SAFE_SUB_BATCH, out, err, p_hot_alloc,
                expert_compute, expert_layer);
        }
        const int n_embd = cfg.n_embd;
        const int n_used = cfg.n_expert_used;
        out.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
        std::vector<float> sub_out;
        for (int t0 = 0; t0 < n_tokens; t0 += MMQ_SAFE_SUB_BATCH) {
            const int tc = std::min(MMQ_SAFE_SUB_BATCH, n_tokens - t0);
            if (!eval_moe_hybrid_ffn_batched_core(
                    gpu_backend, cpu_backend, cfg, desc, storage,
                    cur_host + (size_t)t0 * (size_t)n_embd,
                    selected_ids + (size_t)t0 * (size_t)n_used,
                    selected_weights + (size_t)t0 * (size_t)n_used,
                    tc, sub_out, err, p_hot_alloc, p_cold_alloc,
                    expert_compute, expert_layer)) {
                return false;
            }
            std::memcpy(out.data() + (size_t)t0 * (size_t)n_embd,
                        sub_out.data(),
                        sizeof(float) * (size_t)n_embd * (size_t)tc);
        }
        return true;
    }
    return eval_moe_hybrid_ffn_batched_core(
        gpu_backend, cpu_backend, cfg, desc, storage,
        cur_host, selected_ids, selected_weights, n_tokens, out, err,
        p_hot_alloc, p_cold_alloc, expert_compute, expert_layer);
}

void ResidualCombineGraph::free() {
    if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
    if (ctx) { ggml_free(ctx); ctx = nullptr; }
    gf = nullptr;
    residual_in = nullptr;
    hot_in = nullptr;
    cold_in = nullptr;
    output = nullptr;
}

void ResidualCombineGraph::destroy() {
    free();
}

bool build_residual_combine_graph(ResidualCombineGraph & out, ggml_backend_t backend, int n_embd) {
    out.free();

    ggml_init_params ip{};
    ip.mem_size = 4 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    out.residual_in = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
    ggml_set_input(out.residual_in);
    out.hot_in = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
    ggml_set_input(out.hot_in);
    out.cold_in = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
    ggml_set_input(out.cold_in);

    // output = residual + hot + cold
    ggml_tensor * sum = ggml_add(out.ctx, out.residual_in, out.hot_in);
    out.output = ggml_add(out.ctx, sum, out.cold_in);
    ggml_set_output(out.output);

    out.gf = ggml_new_graph_custom(out.ctx, 64, false);
    ggml_build_forward_expand(out.gf, out.output);

    out.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(out.alloc, out.gf)) {
        out.free();
        return false;
    }
    return true;
}

void GpuResidentState::destroy() {
    combine.destroy();
    if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
    if (ctx) { ggml_free(ctx); ctx = nullptr; }
    act_cur = nullptr;
}

bool init_gpu_resident_state(GpuResidentState & out, ggml_backend_t backend, int n_embd) {
    out.destroy();

    ggml_init_params ip{};
    ip.mem_size = 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    out.act_cur = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, n_embd, 1, 1);
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        out.destroy();
        return false;
    }

    if (!build_residual_combine_graph(out.combine, backend, n_embd)) {
        out.destroy();
        return false;
    }

    std::vector<float> zeros((size_t)n_embd, 0.0f);
    ggml_backend_tensor_set(out.combine.cold_in, zeros.data(), 0, sizeof(float) * (size_t)n_embd);

    return true;
}

bool eval_moe_hybrid_ffn_gpu_resident(
    ggml_backend_t                  gpu_backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    MoeHybridLayerStorage &         storage,
    ggml_backend_t                  cpu_backend,
    ggml_tensor *                   ffn_post_gpu,
    ggml_tensor *                   ffn_residual_gpu,
    GpuResidentState &              gpu_state,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_selected) {

    const int n_embd = cfg.n_embd;

    // ── Partition into hot/cold ──
    std::vector<int32_t> hot_ids;
    std::vector<float> hot_weights;
    std::vector<int32_t> cold_ids;
    std::vector<float> cold_weights;
    hot_ids.reserve((size_t)n_selected);
    hot_weights.reserve((size_t)n_selected);

    for (int i = 0; i < n_selected; ++i) {
        const int32_t gid = selected_ids[i];
        if (gid < 0 || gid >= (int32_t)storage.hot_local_by_global.size()) return false;
        const int32_t hot_local = storage.hot_local_by_global[(size_t)gid];
        if (hot_local >= 0) {
            hot_ids.push_back(hot_local);
            hot_weights.push_back(selected_weights[i]);
        } else {
            const int32_t cold_local = storage.cold_local_by_global[(size_t)gid];
            if (cold_local >= 0) {
                cold_ids.push_back(cold_local);
                cold_weights.push_back(selected_weights[i]);
            }
        }
    }

    const int n_hot = (int)hot_ids.size();
    const bool has_hot = (n_hot > 0);
    const bool has_shared = (desc.ffn_up_shexp && desc.ffn_gate_shexp && desc.ffn_down_shexp);
    const bool has_cold = !cold_ids.empty();
    const int n_cold = (int)cold_ids.size();

    // ── GPU-remap fast path (laguna): fold residual + hot-routed + shared into a
    // single cached GPU graph that consumes the router's expert ids directly
    // (cold experts masked to 0 via valid_lut). Removes the separate per-layer
    // residual-combine graph_compute and the host hot/cold partition for the GPU
    // path. Cold experts (rare under realistic placement) are added on CPU after.
    // IEEE add is commutative, so this is bit-exact vs the split+combine path.
    static const bool kLagunaGpuRemap = (std::getenv("DFLASH_LAGUNA_GPU_REMAP") != nullptr);
    if (kLagunaGpuRemap) {
        // Reactive bounded expert cache: pull selected cold experts into spare
        // GPU slots (LRU evict) so the unified GPU FFN serves them on-die. After
        // warmup the working set is resident and the CPU cold path is rarely taken.
        static const bool kCache = (std::getenv("DFLASH_LAGUNA_EXPERT_CACHE") != nullptr);
        if (kCache && storage.cache_slots > 0) {
            for (int i = 0; i < n_selected; ++i)
                moe_hybrid_cache_swap_in(storage, selected_ids[i], gpu_backend);
        }
        // Cold residue after caching (equals the original cold set when disabled).
        std::vector<int32_t> cache_cold_ids; std::vector<float> cache_cold_w;
        for (int i = 0; i < n_selected; ++i) {
            const int32_t g = selected_ids[i];
            if (g < 0 || g >= (int)storage.hot_local_by_global.size()) continue;
            if (storage.hot_local_by_global[(size_t)g] < 0) {
                const int32_t cl = storage.cold_local_by_global[(size_t)g];
                if (cl >= 0) { cache_cold_ids.push_back(cl); cache_cold_w.push_back(selected_weights[i]); }
            }
        }
        const bool has_cold2 = !cache_cold_ids.empty();
        const int n_cold2 = (int)cache_cold_ids.size();
        if (!storage.hot_graph.valid() || storage.hot_graph.n_hot != n_selected ||
            !storage.hot_graph.global_ids) {
            build_cached_hot_graph(storage.hot_graph, gpu_backend,
                                   storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                                   desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                   desc, n_embd, cfg.n_ff_exp, n_selected,
                                   /*gpu_remap=*/true, cfg.n_expert);
        }
        if (!storage.hot_graph.valid() || !storage.hot_graph.global_ids ||
            !storage.hot_graph.hot_local_lut || !storage.hot_graph.valid_lut ||
            !storage.hot_graph.residual_in) {
            return false;
        }
        {
            auto & scratch = g_mixed_routing_scratch;
            scratch.ensure(n_selected, cfg.n_expert);
            auto & lut = scratch.lut;
            auto & vlut = scratch.vlut;
            for (int e = 0; e < cfg.n_expert; ++e) {
                const int32_t l = storage.hot_local_by_global[(size_t)e];
                lut[(size_t)e]  = (l >= 0) ? l : 0;
                vlut[(size_t)e] = (l >= 0) ? 1.0f : 0.0f;
            }
            ggml_backend_tensor_set(storage.hot_graph.hot_local_lut, lut.data(), 0,
                                    sizeof(int32_t) * (size_t)cfg.n_expert);
            ggml_backend_tensor_set(storage.hot_graph.valid_lut, vlut.data(), 0,
                                    sizeof(float) * (size_t)cfg.n_expert);
        }
        ggml_backend_tensor_copy(ffn_post_gpu, storage.hot_graph.inp);
        ggml_backend_tensor_copy(ffn_residual_gpu, storage.hot_graph.residual_in);
        ggml_backend_tensor_set(storage.hot_graph.global_ids, selected_ids, 0,
                                sizeof(int32_t) * (size_t)n_selected);
        ggml_backend_tensor_set(storage.hot_graph.raw_weights, selected_weights, 0,
                                sizeof(float) * (size_t)n_selected);
        if (ggml_backend_graph_compute(gpu_backend, storage.hot_graph.gf) != GGML_STATUS_SUCCESS) {
            return false;
        }
        if (!has_cold2) {
            ggml_backend_tensor_copy(storage.hot_graph.output, gpu_state.act_cur);
            return true;
        }
        std::vector<float> post_host((size_t)n_embd);
        ggml_backend_tensor_get(ffn_post_gpu, post_host.data(), 0, sizeof(float) * (size_t)n_embd);
        if (!storage.cold_graph.valid() || storage.cold_graph.n_hot != n_cold2) {
            build_cached_cold_graph(storage.cold_graph, cpu_backend,
                                    storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
                                    desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                    n_embd, cfg.n_ff_exp, n_cold2);
        }
        if (!storage.cold_graph.valid() || storage.cold_graph.n_hot != n_cold2) return false;
        ggml_backend_tensor_set(storage.cold_graph.inp, post_host.data(), 0, sizeof(float) * (size_t)n_embd);
        ggml_backend_tensor_set(storage.cold_graph.ids, cache_cold_ids.data(), 0, sizeof(int32_t) * (size_t)n_cold2);
        ggml_backend_tensor_set(storage.cold_graph.weights, cache_cold_w.data(), 0, sizeof(float) * (size_t)n_cold2);
        if (ggml_backend_graph_compute(cpu_backend, storage.cold_graph.gf) != GGML_STATUS_SUCCESS) return false;
        std::vector<float> cold_res((size_t)n_embd), hot_res((size_t)n_embd);
        ggml_backend_tensor_get(storage.cold_graph.output, cold_res.data(), 0, sizeof(float) * (size_t)n_embd);
        ggml_backend_tensor_get(storage.hot_graph.output, hot_res.data(), 0, sizeof(float) * (size_t)n_embd);
        for (int i = 0; i < n_embd; ++i) hot_res[(size_t)i] += cold_res[(size_t)i];
        ggml_backend_tensor_set(gpu_state.act_cur, hot_res.data(), 0, sizeof(float) * (size_t)n_embd);
        return true;
    }

    // ── GPU→GPU: copy residual to combine input ──
    ggml_backend_tensor_copy(ffn_residual_gpu, gpu_state.combine.residual_in);

    // ── Prepare hot graph input via GPU→GPU copy ──
    bool hot_async_launched = false;
    if (has_hot || has_shared) {
        if (!storage.hot_graph.valid() || storage.hot_graph.n_hot != n_hot) {
            build_cached_hot_graph(storage.hot_graph, gpu_backend,
                                   storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                                   desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                   desc, n_embd, cfg.n_ff_exp, n_hot);
        }
        if (storage.hot_graph.valid() && storage.hot_graph.n_hot == n_hot) {
            // GPU→GPU copy: ffn_post → hot_graph.inp (no PCIe!)
            ggml_backend_tensor_copy(ffn_post_gpu, storage.hot_graph.inp);
            if (storage.hot_graph.ids && has_hot) {
                ggml_backend_tensor_set(storage.hot_graph.ids, hot_ids.data(), 0,
                                        sizeof(int32_t) * (size_t)n_hot);
            }
            if (storage.hot_graph.weights && has_hot) {
                ggml_backend_tensor_set(storage.hot_graph.weights, hot_weights.data(), 0,
                                        sizeof(float) * (size_t)n_hot);
            }
        }
    }

    // ── If cold needed, read ffn_post to CPU BEFORE launching hot async ──
    std::vector<float> post_host;
    if (has_cold) {
        post_host.resize((size_t)n_embd);
        ggml_backend_tensor_get(ffn_post_gpu, post_host.data(), 0, sizeof(float) * (size_t)n_embd);
    }

    // ── Launch hot async (GPU kernels in flight) ──
    if ((has_hot || has_shared) && storage.hot_graph.valid() && storage.hot_graph.n_hot == n_hot) {
        ggml_backend_graph_compute_async(gpu_backend, storage.hot_graph.gf);
        hot_async_launched = true;
    }

    // ── Cold path on CPU (overlaps with hot GPU kernels) ──
    std::vector<float> cold_result;
    if (has_cold) {
        if (!storage.cold_graph.valid() || storage.cold_graph.n_hot != n_cold) {
            build_cached_cold_graph(storage.cold_graph, cpu_backend,
                                    storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
                                    desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                    n_embd, cfg.n_ff_exp, n_cold);
        }
        if (storage.cold_graph.valid() && storage.cold_graph.n_hot == n_cold) {
            ggml_backend_tensor_set(storage.cold_graph.inp, post_host.data(), 0,
                                    sizeof(float) * (size_t)n_embd);
            ggml_backend_tensor_set(storage.cold_graph.ids, cold_ids.data(), 0,
                                    sizeof(int32_t) * (size_t)n_cold);
            ggml_backend_tensor_set(storage.cold_graph.weights, cold_weights.data(), 0,
                                    sizeof(float) * (size_t)n_cold);
            auto st = ggml_backend_graph_compute(cpu_backend, storage.cold_graph.gf);
            if (st != GGML_STATUS_SUCCESS) {
                if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
                return false;
            }
            cold_result.resize((size_t)n_embd);
            ggml_backend_tensor_get(storage.cold_graph.output, cold_result.data(), 0,
                                    sizeof(float) * (size_t)n_embd);
        } else {
            if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
            return false;
        }
    }

    // ── Sync hot graph and copy output to combine.hot_in ──
    if (hot_async_launched) {
        ggml_backend_synchronize(gpu_backend);
        // GPU→GPU: hot output → combine.hot_in
        ggml_backend_tensor_copy(storage.hot_graph.output, gpu_state.combine.hot_in);
    } else {
        std::vector<float> zeros((size_t)n_embd, 0.0f);
        ggml_backend_tensor_set(gpu_state.combine.hot_in, zeros.data(), 0,
                                sizeof(float) * (size_t)n_embd);
    }

    // ── Upload cold result (or zeros) to combine.cold_in ──
    if (has_cold) {
        ggml_backend_tensor_set(gpu_state.combine.cold_in, cold_result.data(), 0,
                                sizeof(float) * (size_t)n_embd);
    } else {
        std::vector<float> zeros((size_t)n_embd, 0.0f);
        ggml_backend_tensor_set(gpu_state.combine.cold_in, zeros.data(), 0,
                                sizeof(float) * (size_t)n_embd);
    }

    // ── Compute residual combine on GPU: output = residual + hot + cold ──
    auto st = ggml_backend_graph_compute(gpu_backend, gpu_state.combine.gf);
    if (st != GGML_STATUS_SUCCESS) return false;

    // ── Copy combine output to persistent act_cur (GPU→GPU) ──
    ggml_backend_tensor_copy(gpu_state.combine.output, gpu_state.act_cur);

    return true;
}

}  // namespace dflash::common
