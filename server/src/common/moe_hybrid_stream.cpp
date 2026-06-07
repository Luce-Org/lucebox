// MoE hybrid prefill streaming engine — implementation.

#include "moe_hybrid_stream.h"
#include "gpu_runtime_compat.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace dflash::common {

MoeHybridStreamEngine::~MoeHybridStreamEngine() {
    destroy();
}

MoeHybridStreamEngine::MoeHybridStreamEngine(MoeHybridStreamEngine && o) noexcept
    : pinned_buf_(o.pinned_buf_), pinned_size_(o.pinned_size_),
      gpu_scratch_(o.gpu_scratch_), scratch_size_(o.scratch_size_),
      backend_(o.backend_),
      scratch_gate_(o.scratch_gate_), scratch_up_(o.scratch_up_),
      scratch_down_(o.scratch_down_),
      last_gate_bytes_(o.last_gate_bytes_), last_up_bytes_(o.last_up_bytes_),
      last_down_bytes_(o.last_down_bytes_) {
    o.pinned_buf_ = nullptr; o.pinned_size_ = 0;
    o.gpu_scratch_ = nullptr; o.scratch_size_ = 0;
    o.backend_ = nullptr;
    o.scratch_gate_ = nullptr; o.scratch_up_ = nullptr; o.scratch_down_ = nullptr;
    o.last_gate_bytes_ = 0; o.last_up_bytes_ = 0; o.last_down_bytes_ = 0;
}

MoeHybridStreamEngine & MoeHybridStreamEngine::operator=(MoeHybridStreamEngine && o) noexcept {
    if (this != &o) {
        destroy();
        pinned_buf_ = o.pinned_buf_; pinned_size_ = o.pinned_size_;
        gpu_scratch_ = o.gpu_scratch_; scratch_size_ = o.scratch_size_;
        backend_ = o.backend_;
        scratch_gate_ = o.scratch_gate_; scratch_up_ = o.scratch_up_;
        scratch_down_ = o.scratch_down_;
        last_gate_bytes_ = o.last_gate_bytes_; last_up_bytes_ = o.last_up_bytes_;
        last_down_bytes_ = o.last_down_bytes_;
        o.pinned_buf_ = nullptr; o.pinned_size_ = 0;
        o.gpu_scratch_ = nullptr; o.scratch_size_ = 0;
        o.backend_ = nullptr;
        o.scratch_gate_ = nullptr; o.scratch_up_ = nullptr; o.scratch_down_ = nullptr;
        o.last_gate_bytes_ = 0; o.last_up_bytes_ = 0; o.last_down_bytes_ = 0;
    }
    return *this;
}

bool MoeHybridStreamEngine::init(ggml_backend_t gpu_backend, size_t max_expert_bytes,
                                 std::string * err) {
    destroy();
    if (!gpu_backend || max_expert_bytes == 0) {
        if (err) *err = "invalid arguments to stream engine init";
        return false;
    }

    // Allocate pinned host staging buffer
    cudaError_t cuda_err = cudaMallocHost(&pinned_buf_, max_expert_bytes);
    if (cuda_err != cudaSuccess) {
        if (err) *err = std::string("cudaMallocHost failed: ") + cudaGetErrorString(cuda_err);
        return false;
    }
    pinned_size_ = max_expert_bytes;

    // Allocate GPU scratch buffer
    cuda_err = cudaMalloc(&gpu_scratch_, max_expert_bytes);
    if (cuda_err != cudaSuccess) {
        if (err) *err = std::string("cudaMalloc scratch failed: ") + cudaGetErrorString(cuda_err);
        cudaFreeHost(pinned_buf_);
        pinned_buf_ = nullptr;
        pinned_size_ = 0;
        return false;
    }
    scratch_size_ = max_expert_bytes;
    backend_ = gpu_backend;
    return true;
}

bool MoeHybridStreamEngine::is_ready() const {
    return pinned_buf_ && gpu_scratch_ && backend_;
}

void MoeHybridStreamEngine::destroy() {
    if (gpu_scratch_) {
        cudaFree(gpu_scratch_);
        gpu_scratch_ = nullptr;
    }
    if (pinned_buf_) {
        cudaFreeHost(pinned_buf_);
        pinned_buf_ = nullptr;
    }
    pinned_size_ = 0;
    scratch_size_ = 0;
    backend_ = nullptr;
    scratch_gate_ = nullptr;
    scratch_up_ = nullptr;
    scratch_down_ = nullptr;
    last_gate_bytes_ = 0;
    last_up_bytes_ = 0;
    last_down_bytes_ = 0;
}

void MoeHybridStreamEngine::prefetch_cold_experts(const void * mmap_data, size_t mmap_size,
                                                  const LayerExpertRegions & regions,
                                                  const int32_t * cold_expert_ids,
                                                  int n_cold) {
    if (!mmap_data || mmap_size == 0 || !cold_expert_ids || n_cold <= 0) return;

#if !defined(_WIN32)
    auto do_advise = [&](size_t offset, size_t length) {
        if (offset + length > mmap_size) return;
        const size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
        const size_t aligned_offset = (offset / page_size) * page_size;
        const size_t aligned_length = length + (offset - aligned_offset);
        ::madvise(const_cast<uint8_t *>(static_cast<const uint8_t *>(mmap_data)) + aligned_offset,
                  aligned_length, MADV_WILLNEED);
    };
#endif

    for (int i = 0; i < n_cold; ++i) {
        const int32_t eid = cold_expert_ids[i];
#if !defined(_WIN32)
        if (regions.fused_gate_up) {
            if (regions.gate_up_exps.size > 0) {
                do_advise(regions.gate_up_exps.offset + (size_t)eid * regions.expert_bytes_gate_up,
                          regions.expert_bytes_gate_up);
            }
        } else {
            if (regions.gate_exps.size > 0) {
                do_advise(regions.gate_exps.offset + (size_t)eid * regions.expert_bytes_gate,
                          regions.expert_bytes_gate);
            }
            if (regions.up_exps.size > 0) {
                do_advise(regions.up_exps.offset + (size_t)eid * regions.expert_bytes_up,
                          regions.expert_bytes_up);
            }
        }
        if (regions.down_exps.size > 0) {
            do_advise(regions.down_exps.offset + (size_t)eid * regions.expert_bytes_down,
                      regions.expert_bytes_down);
        }
#else
        (void)eid;
        (void)regions;
#endif
    }
}

bool MoeHybridStreamEngine::stream_expert_sync(const void * mmap_data, size_t mmap_size,
                                               const LayerExpertRegions & regions,
                                               int expert_id,
                                               ggml_backend_t gpu_backend,
                                               std::string * err) {
    if (!is_ready()) {
        if (err) *err = "stream engine not initialized";
        return false;
    }
    if (!mmap_data || mmap_size == 0) {
        if (err) *err = "mmap not available";
        return false;
    }

    const auto * file_base = static_cast<const uint8_t *>(mmap_data);
    size_t staging_offset = 0;

    // Validate expert_id against region size
    if (expert_id < 0) {
        if (err) *err = "expert_id is negative";
        return false;
    }

    // Copy gate (or fused gate_up) from mmap → pinned
    if (regions.fused_gate_up) {
        const size_t bytes = regions.expert_bytes_gate_up;
        const size_t file_off = regions.gate_up_exps.offset + (size_t)expert_id * bytes;
        if (file_off + bytes > mmap_size) {
            if (err) *err = "gate_up expert out of file bounds";
            return false;
        }
        std::memcpy(static_cast<uint8_t *>(pinned_buf_) + staging_offset,
                    file_base + file_off, bytes);
        last_gate_bytes_ = bytes;
        last_up_bytes_ = 0;
        staging_offset += bytes;
    } else {
        // gate
        {
            const size_t bytes = regions.expert_bytes_gate;
            const size_t file_off = regions.gate_exps.offset + (size_t)expert_id * bytes;
            if (file_off + bytes > mmap_size) {
                if (err) *err = "gate expert out of file bounds";
                return false;
            }
            std::memcpy(static_cast<uint8_t *>(pinned_buf_) + staging_offset,
                        file_base + file_off, bytes);
            last_gate_bytes_ = bytes;
            staging_offset += bytes;
        }
        // up
        {
            const size_t bytes = regions.expert_bytes_up;
            const size_t file_off = regions.up_exps.offset + (size_t)expert_id * bytes;
            if (file_off + bytes > mmap_size) {
                if (err) *err = "up expert out of file bounds";
                return false;
            }
            std::memcpy(static_cast<uint8_t *>(pinned_buf_) + staging_offset,
                        file_base + file_off, bytes);
            last_up_bytes_ = bytes;
            staging_offset += bytes;
        }
    }

    // down
    {
        const size_t bytes = regions.expert_bytes_down;
        const size_t file_off = regions.down_exps.offset + (size_t)expert_id * bytes;
        if (file_off + bytes > mmap_size) {
            if (err) *err = "down expert out of file bounds";
            return false;
        }
        std::memcpy(static_cast<uint8_t *>(pinned_buf_) + staging_offset,
                    file_base + file_off, bytes);
        last_down_bytes_ = bytes;
        staging_offset += bytes;
    }

    if (staging_offset > scratch_size_) {
        if (err) *err = "expert exceeds scratch buffer size";
        return false;
    }

    // DMA pinned → GPU scratch (synchronous for now; async pipeline in eval function)
    cudaError_t cuda_err = cudaMemcpy(gpu_scratch_, pinned_buf_, staging_offset,
                                      cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        if (err) *err = std::string("cudaMemcpy H2D failed: ") + cudaGetErrorString(cuda_err);
        return false;
    }

    // Set pointers into scratch
    auto * scratch_bytes = static_cast<uint8_t *>(gpu_scratch_);
    size_t off = 0;
    if (regions.fused_gate_up) {
        scratch_gate_ = scratch_bytes + off;
        off += last_gate_bytes_;
        scratch_up_ = nullptr;
    } else {
        scratch_gate_ = scratch_bytes + off;
        off += last_gate_bytes_;
        scratch_up_ = scratch_bytes + off;
        off += last_up_bytes_;
    }
    scratch_down_ = scratch_bytes + off;

    return true;
}

// ── Streaming prefill evaluation ────────────────────────────────────────────

bool eval_moe_cold_experts_streaming(
    MoeHybridStreamEngine &         engine,
    ggml_backend_t                  gpu_backend,
    const void *                    mmap_data,
    size_t                          mmap_size,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    const LayerExpertRegions &      regions,
    const MoeHybridLayerStorage &   storage,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err) {

    const int n_embd = cfg.n_embd;
    const int n_ff_exp = cfg.n_ff_exp;
    const int n_used = cfg.n_expert_used;
    const int total_slots = n_used * n_tokens;

    out.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
    if (!engine.is_ready()) {
        if (err) *err = "stream engine not ready";
        return false;
    }
    if (!mmap_data || mmap_size == 0) {
        if (err) *err = "mmap not available";
        return false;
    }

    // Identify unique cold experts needed across all tokens.
    std::vector<bool> cold_needed((size_t)cfg.n_expert, false);
    for (int i = 0; i < total_slots; ++i) {
        const int32_t gid = selected_ids[i];
        if (gid < 0 || gid >= cfg.n_expert) continue;
        if (storage.hot_local_by_global[(size_t)gid] < 0) {
            cold_needed[(size_t)gid] = true;
        }
    }

    std::vector<int32_t> unique_cold;
    for (int e = 0; e < cfg.n_expert; ++e) {
        if (cold_needed[(size_t)e]) unique_cold.push_back((int32_t)e);
    }

    if (unique_cold.empty()) return true;

    // Prefetch all cold experts via madvise
    engine.prefetch_cold_experts(mmap_data, mmap_size, regions, unique_cold.data(), (int)unique_cold.size());

    // For each unique cold expert: stream to GPU, compute ALL tokens that selected it
    // in a single batched matmul graph.
    for (int32_t cold_eid : unique_cold) {
        // Stream expert weights to GPU scratch
        if (!engine.stream_expert_sync(mmap_data, mmap_size, regions, cold_eid, gpu_backend, err)) {
            return false;
        }

        // Gather all tokens that selected this expert
        struct TokenHit { int ti; float weight; };
        std::vector<TokenHit> hits;
        hits.reserve((size_t)n_tokens);
        for (int ti = 0; ti < n_tokens; ++ti) {
            for (int k = 0; k < n_used; ++k) {
                const int slot = ti * n_used + k;
                if (selected_ids[slot] != cold_eid) continue;
                const float w = selected_weights[slot];
                if (w != 0.0f) hits.push_back({ti, w});
                break;  // each expert selected at most once per token
            }
        }
        if (hits.empty()) continue;

        const int batch = (int)hits.size();

        // Build batched input: [n_embd, batch]
        std::vector<float> batch_input((size_t)n_embd * (size_t)batch);
        for (int i = 0; i < batch; ++i) {
            const float * src = cur_host + (size_t)hits[(size_t)i].ti * (size_t)n_embd;
            std::memcpy(batch_input.data() + (size_t)i * (size_t)n_embd, src, sizeof(float) * (size_t)n_embd);
        }

        // Build single ggml graph for this expert with all tokens batched
        ggml_init_params ip{};
        ip.mem_size = 32 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            if (err) *err = "ggml_init failed in streaming eval";
            return false;
        }

        ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, batch);
        ggml_set_input(inp);

        // Weight tensors pointing into GPU scratch (same for all tokens in batch)
        ggml_tensor * gate_t = nullptr;
        ggml_tensor * up_t = nullptr;
        ggml_tensor * down_t = nullptr;
        ggml_tensor * gate_up_t = nullptr;

        if (regions.fused_gate_up) {
            gate_up_t = ggml_new_tensor_2d(ctx, desc.ffn_gate_up_exps->type, n_embd, 2 * n_ff_exp);
            ggml_set_input(gate_up_t);
            down_t = ggml_new_tensor_2d(ctx, desc.ffn_down_exps->type, n_ff_exp, n_embd);
            ggml_set_input(down_t);
        } else {
            gate_t = ggml_new_tensor_2d(ctx, desc.ffn_gate_exps->type, n_embd, n_ff_exp);
            ggml_set_input(gate_t);
            up_t = ggml_new_tensor_2d(ctx, desc.ffn_up_exps->type, n_embd, n_ff_exp);
            ggml_set_input(up_t);
            down_t = ggml_new_tensor_2d(ctx, desc.ffn_down_exps->type, n_ff_exp, n_embd);
            ggml_set_input(down_t);
        }

        // FFN graph: out = down(silu(gate(x)) * up(x))  — batched over all tokens
        ggml_tensor * gu = nullptr;
        if (gate_up_t) {
            ggml_tensor * gate_up_out = ggml_mul_mat(ctx, gate_up_t, inp);  // [2*n_ff, batch]
            if (desc.ffn_gate_up_exps_s != 1.0f)
                gate_up_out = ggml_scale(ctx, gate_up_out, desc.ffn_gate_up_exps_s);
            ggml_tensor * g_part = ggml_view_2d(ctx, gate_up_out, n_ff_exp, batch,
                                                gate_up_out->nb[1], 0);
            ggml_tensor * u_part = ggml_view_2d(ctx, gate_up_out, n_ff_exp, batch,
                                                gate_up_out->nb[1],
                                                (size_t)n_ff_exp * sizeof(float));
            g_part = ggml_cont(ctx, g_part);
            u_part = ggml_cont(ctx, u_part);
            gu = ggml_swiglu_split(ctx, g_part, u_part);
        } else {
            ggml_tensor * g = ggml_mul_mat(ctx, gate_t, inp);  // [n_ff, batch]
            if (desc.ffn_gate_exps_s != 1.0f)
                g = ggml_scale(ctx, g, desc.ffn_gate_exps_s);
            ggml_tensor * u = ggml_mul_mat(ctx, up_t, inp);    // [n_ff, batch]
            if (desc.ffn_up_exps_s != 1.0f)
                u = ggml_scale(ctx, u, desc.ffn_up_exps_s);
            gu = ggml_swiglu_split(ctx, g, u);
        }

        ggml_tensor * expert_out = ggml_mul_mat(ctx, down_t, gu);  // [n_embd, batch]
        if (desc.ffn_down_exps_s != 1.0f)
            expert_out = ggml_scale(ctx, expert_out, desc.ffn_down_exps_s);

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 512, false);
        ggml_set_output(expert_out);
        ggml_build_forward_expand(gf, expert_out);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu_backend));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            if (err) *err = "streaming eval gallocr failed";
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return false;
        }

        // Upload batched input (host → GPU via ggml_backend_tensor_set)
        ggml_backend_tensor_set(inp, batch_input.data(), 0, sizeof(float) * (size_t)n_embd * (size_t)batch);

        // Point weight tensors directly at GPU scratch (device-to-device, no copy needed).
        // The gallocr allocated these tensors on the same GPU, but we override their data
        // pointers to point at our pre-loaded scratch buffer.
        if (gate_up_t) {
            gate_up_t->data = const_cast<void *>(engine.scratch_gate_data());
            down_t->data = const_cast<void *>(engine.scratch_down_data());
        } else {
            gate_t->data = const_cast<void *>(engine.scratch_gate_data());
            up_t->data = const_cast<void *>(engine.scratch_up_data());
            down_t->data = const_cast<void *>(engine.scratch_down_data());
        }

        auto st = ggml_backend_graph_compute(gpu_backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            if (err) *err = "streaming eval compute failed";
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return false;
        }

        // Read batched result [n_embd, batch] and scatter-accumulate with weights
        std::vector<float> batch_result((size_t)n_embd * (size_t)batch);
        ggml_backend_tensor_get(expert_out, batch_result.data(), 0, sizeof(float) * (size_t)n_embd * (size_t)batch);

        for (int i = 0; i < batch; ++i) {
            const float w = hits[(size_t)i].weight;
            const int ti = hits[(size_t)i].ti;
            const float * res = batch_result.data() + (size_t)i * (size_t)n_embd;
            float * out_tok = out.data() + (size_t)ti * (size_t)n_embd;
            for (int j = 0; j < n_embd; ++j) {
                out_tok[j] += w * res[(size_t)j];
            }
        }

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }

    return true;
}

}  // namespace dflash::common
