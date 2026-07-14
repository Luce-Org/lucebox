#include "moe_hybrid_storage.h"
#include "moe_hybrid_types.h"

#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cstring>

#if defined(DFLASH27B_BACKEND_CUDA)
#include <cuda_runtime_api.h>
#endif

#if !defined(_WIN32)
#include <sys/mman.h>
#else
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace dflash::common {

int query_gpu_compute_sm() {
#if defined(DFLASH27B_BACKEND_CUDA)
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0) return 0;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 0;
    return prop.major * 10 + prop.minor;
#else
    // HIP/gfx1151 has the same MMQ bug — keep sub-batch workaround active.
    return 0;
#endif
}

void CachedFfnGraph::free() {
    if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
    if (ctx) { ggml_free(ctx); ctx = nullptr; }
    gf = nullptr;
    inp = nullptr;
    ids = nullptr;
    weights = nullptr;
    output = nullptr;
    global_ids = nullptr;
    raw_weights = nullptr;
    hot_local_lut = nullptr;
    valid_lut = nullptr;
    residual_in = nullptr;
    n_hot = 0;
    n_tokens = 1;
}

void CachedHotBatchedGraph::free() {
    if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
    if (ctx) { ggml_free(ctx); ctx = nullptr; }
    gf = nullptr;
    inp = nullptr;
    sel = nullptr;
    wts = nullptr;
    output = nullptr;
    n_tokens = 0;
}

namespace {

static bool read_expert_slices(ggml_backend_t backend,
                               ggml_tensor * tensor,
                               const std::vector<int32_t> & expert_ids,
                               size_t expert_bytes,
                               std::vector<uint8_t> & out,
                               std::string * err) {
    if (!tensor || expert_ids.empty() || expert_bytes == 0) {
        out.clear();
        return true;
    }
    out.resize(expert_bytes * expert_ids.size());
    for (size_t i = 0; i < expert_ids.size(); ++i) {
        const int32_t expert_id = expert_ids[i];
        const size_t offset = expert_bytes * (size_t)expert_id;
        ggml_backend_tensor_get(tensor, out.data() + expert_bytes * i, offset, expert_bytes);
    }
    (void)backend;
    (void)err;
    return true;
}

static bool read_expert_slices_from_mem(const uint8_t * tensor_data,
                                        size_t tensor_size,
                                        const std::vector<int32_t> & expert_ids,
                                        size_t expert_bytes,
                                        std::vector<uint8_t> & out,
                                        std::string * err) {
    if (!tensor_data || expert_ids.empty() || expert_bytes == 0) {
        out.clear();
        return true;
    }
    out.resize(expert_bytes * expert_ids.size());
    for (size_t i = 0; i < expert_ids.size(); ++i) {
        const int32_t expert_id = expert_ids[i];
        const size_t offset = expert_bytes * (size_t)expert_id;
        if (offset + expert_bytes > tensor_size) {
            if (err) *err = "expert slice out of bounds in file";
            return false;
        }
        std::memcpy(out.data() + expert_bytes * i, tensor_data + offset, expert_bytes);
    }
    return true;
}

static bool validate_expert_tensor(ggml_tensor * tensor, int n_expert, size_t * expert_bytes, std::string * err) {
    if (!tensor) {
        *expert_bytes = 0;
        return true;
    }
    if (tensor->ne[2] != n_expert) {
        if (err) *err = "tensor expert dimension mismatch";
        return false;
    }
    if ((int64_t)tensor->nb[2] <= 0) {
        if (err) *err = "tensor expert stride invalid";
        return false;
    }
    *expert_bytes = (size_t)tensor->nb[2];
    return true;
}

static ggml_tensor * new_like_with_expert_count(ggml_context * ctx, ggml_tensor * src, int hot_count) {
    if (!src || hot_count <= 0) return nullptr;
    const int64_t ne[4] = { src->ne[0], src->ne[1], hot_count, 1 };
    return ggml_new_tensor(ctx, src->type, 4, ne);
}

} // namespace

MoeHybridStorage::~MoeHybridStorage() {
    for (auto & layer : layers) {
        layer.hot_graph.free();
        layer.cold_graph.free();
        layer.hot_batched_graph.free();
        for (auto & g : layer.hot_batched_mixed) g.free();
        for (auto & g : layer.cold_batched_mixed) g.free();
        layer.shared_batched_graph.free();
        if (layer.hot_buf) {
            ggml_backend_buffer_free(layer.hot_buf);
            layer.hot_buf = nullptr;
        }
        if (layer.hot_ctx) {
            ggml_free(layer.hot_ctx);
            layer.hot_ctx = nullptr;
        }
        if (layer.cold_buf) {
            ggml_backend_buffer_free(layer.cold_buf);
            layer.cold_buf = nullptr;
        }
        if (layer.cold_ctx) {
            ggml_free(layer.cold_ctx);
            layer.cold_ctx = nullptr;
        }
        layer.gate_hot = nullptr;
        layer.up_hot = nullptr;
        layer.down_hot = nullptr;
        layer.gate_up_hot = nullptr;
        layer.gate_cold = nullptr;
        layer.up_cold = nullptr;
        layer.down_cold = nullptr;
        layer.gate_up_cold = nullptr;
    }
    if (cpu_backend) {
        ggml_backend_free(cpu_backend);
        cpu_backend = nullptr;
    }
    if (mmap_data) {
#if !defined(_WIN32)
        ::munmap(const_cast<void *>(mmap_data), mmap_size);
#else
        // On Windows, the mapping is unmapped when the view handle is closed.
        // mmap_data was mapped via MapViewOfFile; UnmapViewOfFile is the correct cleanup.
        ::UnmapViewOfFile(mmap_data);
#endif
        mmap_data = nullptr;
        mmap_size = 0;
    }
}

bool MoeHybridStorage::matches(const MoeHybridConfig & cfg) const {
    return placement.matches(cfg) &&
           (int)layers.size() == cfg.n_layer &&
           cold_backend_kind == cfg.cold_expert_backend &&
           materialized_hot_experts == cfg.materialize_hot_experts &&
           materialized_cold_experts == cfg.materialize_cold_experts;
}

bool MoeHybridStorage::empty() const {
    return layers.empty();
}

bool build_moe_hybrid_storage(const MoeHybridConfig & cfg,
                              ggml_backend_t gpu_backend,
                              const MoeHybridPlacement & placement,
                              const std::vector<MoeLayerDesc> & layer_descs,
                              MoeHybridStorage & out,
                              std::string * err) {
    if (!placement.matches(cfg)) {
        if (err) *err = "placement does not match config";
        return false;
    }
    if ((int)layer_descs.size() != cfg.n_layer) {
        if (err) *err = "layer_descs size does not match n_layer";
        return false;
    }

    out.placement = placement;
    out.layers.resize((size_t)cfg.n_layer);
    out.cpu_backend = ggml_backend_cpu_init();
    if (!out.cpu_backend) {
        if (err) *err = "failed to init cpu backend";
        return false;
    }
    ggml_backend_cpu_set_n_threads(out.cpu_backend, std::max(1, std::min(cfg.n_expert_used, 8)));
    out.cold_backend_kind = cfg.cold_expert_backend;
    out.materialized_hot_experts = cfg.materialize_hot_experts;
    out.materialized_cold_experts = cfg.materialize_cold_experts;
    out.cold_backend = (cfg.cold_expert_backend == MoeHybridColdBackend::Gpu) ? gpu_backend : out.cpu_backend;
    if (!out.cold_backend) {
        if (err) *err = "failed to select cold expert backend";
        return false;
    }

    for (int il = 0; il < cfg.n_layer; ++il) {
        const MoeLayerDesc & desc = layer_descs[(size_t)il];
        MoeHybridLayerStorage & dst = out.layers[(size_t)il];
        dst.cold_backend = out.cold_backend;
        dst.cold_backend_kind = out.cold_backend_kind;

        // Skip dense layers (no experts)
        if (!desc.ffn_gate_exps && !desc.ffn_up_exps && !desc.ffn_down_exps && !desc.ffn_gate_up_exps) {
            continue;
        }

        dst.hot_expert_ids = placement.hot_expert_ids[(size_t)il];
        dst.hot_local_by_global.assign((size_t)cfg.n_expert, -1);
        dst.cold_local_by_global.assign((size_t)cfg.n_expert, -1);

        std::vector<uint8_t> is_hot((size_t)cfg.n_expert, 0);
        for (size_t i = 0; i < dst.hot_expert_ids.size(); ++i) {
            const int32_t expert = dst.hot_expert_ids[i];
            if (expert < 0 || expert >= cfg.n_expert) {
                if (err) *err = "hot expert id out of range";
                return false;
            }
            dst.hot_local_by_global[(size_t)expert] = (int32_t)i;
            is_hot[(size_t)expert] = 1;
        }
        for (int expert = 0; expert < cfg.n_expert; ++expert) {
            if (!is_hot[(size_t)expert]) {
                dst.cold_local_by_global[(size_t)expert] = (int32_t)dst.cold_expert_ids.size();
                dst.cold_expert_ids.push_back((int32_t)expert);
            }
        }

        // Populate VRAM bitmask from hot expert IDs
        std::memset(dst.expert_vram_mask, 0, sizeof(dst.expert_vram_mask));
        for (int32_t eid : dst.hot_expert_ids) {
            if (eid >= 0 && eid < 256)
                dst.expert_vram_mask[eid >> 6] |= (1ULL << (eid & 63));
        }

        dst.fused_gate_up = desc.has_fused_gate_up();
        if (!validate_expert_tensor(desc.ffn_gate_exps, cfg.n_expert, &dst.gate_expert_bytes, err) ||
            !validate_expert_tensor(desc.ffn_up_exps, cfg.n_expert, &dst.up_expert_bytes, err) ||
            !validate_expert_tensor(desc.ffn_down_exps, cfg.n_expert, &dst.down_expert_bytes, err) ||
            !validate_expert_tensor(desc.ffn_gate_up_exps, cfg.n_expert, &dst.gate_up_expert_bytes, err)) {
            return false;
        }

        const int cold_count = (int)dst.cold_expert_ids.size();
        const int hot_count = (int)dst.hot_expert_ids.size();

        // Allocate hot expert tensors on GPU
        if (hot_count > 0 && cfg.materialize_hot_experts) {
            ggml_init_params ip{};
            ip.mem_size   = 16 * ggml_tensor_overhead();
            ip.mem_buffer = nullptr;
            ip.no_alloc   = true;
            dst.hot_ctx = ggml_init(ip);
            if (!dst.hot_ctx) {
                if (err) *err = "failed to init hot_ctx";
                return false;
            }
            if (dst.fused_gate_up) {
                dst.gate_up_hot = new_like_with_expert_count(dst.hot_ctx, desc.ffn_gate_up_exps, hot_count);
                dst.down_hot    = new_like_with_expert_count(dst.hot_ctx, desc.ffn_down_exps, hot_count);
            } else {
                dst.gate_hot = new_like_with_expert_count(dst.hot_ctx, desc.ffn_gate_exps, hot_count);
                dst.up_hot   = new_like_with_expert_count(dst.hot_ctx, desc.ffn_up_exps, hot_count);
                dst.down_hot = new_like_with_expert_count(dst.hot_ctx, desc.ffn_down_exps, hot_count);
            }
            dst.hot_buf = ggml_backend_alloc_ctx_tensors(dst.hot_ctx, gpu_backend);
            if (!dst.hot_buf) {
                if (err) *err = "failed to allocate hot expert buffer";
                return false;
            }

            std::vector<uint8_t> hot_bytes;
            if (dst.fused_gate_up) {
                if (!read_expert_slices(gpu_backend, desc.ffn_gate_up_exps, dst.hot_expert_ids,
                                        dst.gate_up_expert_bytes, hot_bytes, err))
                    return false;
                ggml_backend_tensor_set(dst.gate_up_hot, hot_bytes.data(), 0, hot_bytes.size());
                if (!read_expert_slices(gpu_backend, desc.ffn_down_exps, dst.hot_expert_ids,
                                        dst.down_expert_bytes, hot_bytes, err))
                    return false;
                ggml_backend_tensor_set(dst.down_hot, hot_bytes.data(), 0, hot_bytes.size());
            } else {
                if (!read_expert_slices(gpu_backend, desc.ffn_gate_exps, dst.hot_expert_ids,
                                        dst.gate_expert_bytes, hot_bytes, err))
                    return false;
                ggml_backend_tensor_set(dst.gate_hot, hot_bytes.data(), 0, hot_bytes.size());
                if (!read_expert_slices(gpu_backend, desc.ffn_up_exps, dst.hot_expert_ids,
                                        dst.up_expert_bytes, hot_bytes, err))
                    return false;
                ggml_backend_tensor_set(dst.up_hot, hot_bytes.data(), 0, hot_bytes.size());
                if (!read_expert_slices(gpu_backend, desc.ffn_down_exps, dst.hot_expert_ids,
                                        dst.down_expert_bytes, hot_bytes, err))
                    return false;
                ggml_backend_tensor_set(dst.down_hot, hot_bytes.data(), 0, hot_bytes.size());
            }
        }

        // Allocate cold expert tensors on the selected cold backend.
        if (cold_count > 0 && cfg.materialize_cold_experts) {
            ggml_init_params ip{};
            ip.mem_size   = 16 * ggml_tensor_overhead();
            ip.mem_buffer = nullptr;
            ip.no_alloc   = true;
            dst.cold_ctx = ggml_init(ip);
            if (!dst.cold_ctx) {
                if (err) *err = "failed to init cold_ctx";
                return false;
            }
            if (dst.fused_gate_up) {
                dst.gate_up_cold = new_like_with_expert_count(dst.cold_ctx, desc.ffn_gate_up_exps, cold_count);
                dst.down_cold    = new_like_with_expert_count(dst.cold_ctx, desc.ffn_down_exps, cold_count);
            } else {
                dst.gate_cold = new_like_with_expert_count(dst.cold_ctx, desc.ffn_gate_exps, cold_count);
                dst.up_cold   = new_like_with_expert_count(dst.cold_ctx, desc.ffn_up_exps, cold_count);
                dst.down_cold = new_like_with_expert_count(dst.cold_ctx, desc.ffn_down_exps, cold_count);
            }
            dst.cold_buf = ggml_backend_alloc_ctx_tensors(dst.cold_ctx, out.cold_backend);





            if (!dst.cold_buf) {
                if (err) {
                    *err = (out.cold_backend_kind == MoeHybridColdBackend::Gpu)
                        ? "failed to allocate cold expert GPU buffer"
                        : "failed to allocate cold expert CPU buffer";
                }
                return false;
            }

            std::vector<uint8_t> cold_bytes;
            if (dst.fused_gate_up) {
                if (!read_expert_slices(gpu_backend, desc.ffn_gate_up_exps, dst.cold_expert_ids,
                                        dst.gate_up_expert_bytes, cold_bytes, err))
                    return false;
                ggml_backend_tensor_set(dst.gate_up_cold, cold_bytes.data(), 0, cold_bytes.size());
                if (!read_expert_slices(gpu_backend, desc.ffn_down_exps, dst.cold_expert_ids,
                                        dst.down_expert_bytes, cold_bytes, err))
                    return false;
                ggml_backend_tensor_set(dst.down_cold, cold_bytes.data(), 0, cold_bytes.size());
            } else {
                if (!read_expert_slices(gpu_backend, desc.ffn_gate_exps, dst.cold_expert_ids,
                                        dst.gate_expert_bytes, cold_bytes, err))
                    return false;
                ggml_backend_tensor_set(dst.gate_cold, cold_bytes.data(), 0, cold_bytes.size());
                if (!read_expert_slices(gpu_backend, desc.ffn_up_exps, dst.cold_expert_ids,
                                        dst.up_expert_bytes, cold_bytes, err))
                    return false;
                ggml_backend_tensor_set(dst.up_cold, cold_bytes.data(), 0, cold_bytes.size());
                if (!read_expert_slices(gpu_backend, desc.ffn_down_exps, dst.cold_expert_ids,
                                        dst.down_expert_bytes, cold_bytes, err))
                    return false;
                ggml_backend_tensor_set(dst.down_cold, cold_bytes.data(), 0, cold_bytes.size());
            }
        }
    }

    return true;
}

bool build_moe_hybrid_storage_from_file(
    const MoeHybridConfig & cfg,
    ggml_backend_t gpu_backend,
    const MoeHybridPlacement & placement,
    const std::vector<MoeLayerDesc> & layer_descs,
    const std::vector<LayerExpertFileData> & file_data,
    MoeHybridStorage & out,
    std::string * err,
    int cache_slots,
    bool allocate_cold) {

    if (!placement.matches(cfg)) {
        if (err) *err = "placement does not match config";
        return false;
    }
    if ((int)layer_descs.size() != cfg.n_layer || (int)file_data.size() != cfg.n_layer) {
        if (err) *err = "layer_descs/file_data size does not match n_layer";
        return false;
    }

    out.placement = placement;
    out.layers.resize((size_t)cfg.n_layer);
    out.cpu_backend = ggml_backend_cpu_init();
    if (!out.cpu_backend) {
        if (err) *err = "failed to init cpu backend";
        return false;
    }
    ggml_backend_cpu_set_n_threads(out.cpu_backend, std::max(1, std::min(cfg.n_expert_used, 8)));
    out.cold_backend_kind = cfg.cold_expert_backend;
    out.materialized_hot_experts = cfg.materialize_hot_experts;
    out.materialized_cold_experts = cfg.materialize_cold_experts;
    out.cold_backend = (cfg.cold_expert_backend == MoeHybridColdBackend::Gpu) ? gpu_backend : out.cpu_backend;
    if (!out.cold_backend) {
        if (err) *err = "failed to select cold expert backend";
        return false;
    }

    for (int il = 0; il < cfg.n_layer; ++il) {
        const MoeLayerDesc & desc = layer_descs[(size_t)il];
        const LayerExpertFileData & fd = file_data[(size_t)il];
        MoeHybridLayerStorage & dst = out.layers[(size_t)il];
        dst.cold_backend = out.cold_backend;
        dst.cold_backend_kind = out.cold_backend_kind;

        // Skip dense layers (no experts)
        if (!desc.ffn_gate_exps && !desc.ffn_up_exps && !desc.ffn_down_exps && !desc.ffn_gate_up_exps) {
            continue;
        }

        dst.hot_expert_ids = placement.hot_expert_ids[(size_t)il];
        dst.hot_local_by_global.assign((size_t)cfg.n_expert, -1);
        dst.cold_local_by_global.assign((size_t)cfg.n_expert, -1);

        std::vector<uint8_t> is_hot((size_t)cfg.n_expert, 0);
        for (size_t i = 0; i < dst.hot_expert_ids.size(); ++i) {
            const int32_t expert = dst.hot_expert_ids[i];
            if (expert < 0 || expert >= cfg.n_expert) {
                if (err) *err = "hot expert id out of range";
                return false;
            }
            dst.hot_local_by_global[(size_t)expert] = (int32_t)i;
            is_hot[(size_t)expert] = 1;
        }
        if (allocate_cold) {
            for (int expert = 0; expert < cfg.n_expert; ++expert) {
                if (!is_hot[(size_t)expert]) {
                    dst.cold_local_by_global[(size_t)expert] = (int32_t)dst.cold_expert_ids.size();
                    dst.cold_expert_ids.push_back((int32_t)expert);
                }
            }
        }

        // Populate VRAM bitmask from hot expert IDs
        std::memset(dst.expert_vram_mask, 0, sizeof(dst.expert_vram_mask));
        for (int32_t eid : dst.hot_expert_ids) {
            if (eid >= 0 && eid < 256)
                dst.expert_vram_mask[eid >> 6] |= (1ULL << (eid & 63));
        }

        dst.fused_gate_up = desc.has_fused_gate_up();
        if (!validate_expert_tensor(desc.ffn_gate_exps, cfg.n_expert, &dst.gate_expert_bytes, err) ||
            !validate_expert_tensor(desc.ffn_up_exps, cfg.n_expert, &dst.up_expert_bytes, err) ||
            !validate_expert_tensor(desc.ffn_down_exps, cfg.n_expert, &dst.down_expert_bytes, err) ||
            !validate_expert_tensor(desc.ffn_gate_up_exps, cfg.n_expert, &dst.gate_up_expert_bytes, err)) {
            return false;
        }

        const int hot_count = (int)dst.hot_expert_ids.size();
        const int cold_count = (int)dst.cold_expert_ids.size();
        const int spare = (cold_count > 0 && cache_slots > 0)
                          ? std::min(cache_slots, cold_count) : 0;
        const int hot_alloc = hot_count + spare;
        dst.hot_active  = hot_count;
        dst.cache_slots = spare;
        dst.spare_global.assign((size_t)spare, -1);
        dst.spare_lru.assign((size_t)spare, 0);

        // Allocate hot expert tensors on GPU
        if (hot_count > 0 && cfg.materialize_hot_experts) {
            ggml_init_params ip{};
            ip.mem_size   = 16 * ggml_tensor_overhead();
            ip.mem_buffer = nullptr;
            ip.no_alloc   = true;
            dst.hot_ctx = ggml_init(ip);
            if (!dst.hot_ctx) {
                if (err) *err = "failed to init hot_ctx";
                return false;
            }
            if (dst.fused_gate_up) {
                dst.gate_up_hot = new_like_with_expert_count(dst.hot_ctx, desc.ffn_gate_up_exps, hot_alloc);
                dst.down_hot    = new_like_with_expert_count(dst.hot_ctx, desc.ffn_down_exps, hot_alloc);
            } else {
                dst.gate_hot = new_like_with_expert_count(dst.hot_ctx, desc.ffn_gate_exps, hot_alloc);
                dst.up_hot   = new_like_with_expert_count(dst.hot_ctx, desc.ffn_up_exps, hot_alloc);
                dst.down_hot = new_like_with_expert_count(dst.hot_ctx, desc.ffn_down_exps, hot_alloc);
            }
            dst.hot_buf = ggml_backend_alloc_ctx_tensors(dst.hot_ctx, gpu_backend);
            if (!dst.hot_buf) {
                char msg[128];
                std::snprintf(msg, sizeof(msg),
                    "failed to allocate hot expert GPU buffer (layer %d, %d hot experts)", il, hot_count);
                if (err) *err = msg;
                return false;
            }

            std::vector<uint8_t> slice_buf;
            if (dst.fused_gate_up) {
                if (!read_expert_slices_from_mem(fd.gate_up_exps.data, fd.gate_up_exps.size,
                                                 dst.hot_expert_ids, dst.gate_up_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.gate_up_hot, slice_buf.data(), 0, slice_buf.size());
                if (!read_expert_slices_from_mem(fd.down_exps.data, fd.down_exps.size,
                                                 dst.hot_expert_ids, dst.down_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.down_hot, slice_buf.data(), 0, slice_buf.size());
            } else {
                if (!read_expert_slices_from_mem(fd.gate_exps.data, fd.gate_exps.size,
                                                 dst.hot_expert_ids, dst.gate_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.gate_hot, slice_buf.data(), 0, slice_buf.size());
                if (!read_expert_slices_from_mem(fd.up_exps.data, fd.up_exps.size,
                                                 dst.hot_expert_ids, dst.up_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.up_hot, slice_buf.data(), 0, slice_buf.size());
                if (!read_expert_slices_from_mem(fd.down_exps.data, fd.down_exps.size,
                                                 dst.hot_expert_ids, dst.down_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.down_hot, slice_buf.data(), 0, slice_buf.size());
            }
        }

        // Allocate cold expert tensors on the selected cold backend.
        if (allocate_cold && cold_count > 0 && cfg.materialize_cold_experts) {
            ggml_init_params ip{};
            ip.mem_size   = 16 * ggml_tensor_overhead();
            ip.mem_buffer = nullptr;
            ip.no_alloc   = true;
            dst.cold_ctx = ggml_init(ip);
            if (!dst.cold_ctx) {
                if (err) *err = "failed to init cold_ctx";
                return false;
            }
            if (dst.fused_gate_up) {
                dst.gate_up_cold = new_like_with_expert_count(dst.cold_ctx, desc.ffn_gate_up_exps, cold_count);
                dst.down_cold    = new_like_with_expert_count(dst.cold_ctx, desc.ffn_down_exps, cold_count);
            } else {
                dst.gate_cold = new_like_with_expert_count(dst.cold_ctx, desc.ffn_gate_exps, cold_count);
                dst.up_cold   = new_like_with_expert_count(dst.cold_ctx, desc.ffn_up_exps, cold_count);
                dst.down_cold = new_like_with_expert_count(dst.cold_ctx, desc.ffn_down_exps, cold_count);
            }
            dst.cold_buf = ggml_backend_alloc_ctx_tensors(dst.cold_ctx, out.cold_backend);
            if (!dst.cold_buf) {
                if (err) {
                    *err = (out.cold_backend_kind == MoeHybridColdBackend::Gpu)
                        ? "failed to allocate cold expert GPU buffer"
                        : "failed to allocate cold expert CPU buffer";
                }
                return false;
            }

            std::vector<uint8_t> slice_buf;
            if (dst.fused_gate_up) {
                if (!read_expert_slices_from_mem(fd.gate_up_exps.data, fd.gate_up_exps.size,
                                                 dst.cold_expert_ids, dst.gate_up_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.gate_up_cold, slice_buf.data(), 0, slice_buf.size());
                if (!read_expert_slices_from_mem(fd.down_exps.data, fd.down_exps.size,
                                                 dst.cold_expert_ids, dst.down_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.down_cold, slice_buf.data(), 0, slice_buf.size());
            } else {
                if (!read_expert_slices_from_mem(fd.gate_exps.data, fd.gate_exps.size,
                                                 dst.cold_expert_ids, dst.gate_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.gate_cold, slice_buf.data(), 0, slice_buf.size());
                if (!read_expert_slices_from_mem(fd.up_exps.data, fd.up_exps.size,
                                                 dst.cold_expert_ids, dst.up_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.up_cold, slice_buf.data(), 0, slice_buf.size());
                if (!read_expert_slices_from_mem(fd.down_exps.data, fd.down_exps.size,
                                                 dst.cold_expert_ids, dst.down_expert_bytes, slice_buf, err))
                    return false;
                ggml_backend_tensor_set(dst.down_cold, slice_buf.data(), 0, slice_buf.size());
            }
        }
    }

    return true;
}


int moe_hybrid_cache_swap_in(MoeHybridLayerStorage & st, int global_expert,
                             ggml_backend_t gpu_backend) {
    if (global_expert < 0 || global_expert >= (int)st.hot_local_by_global.size()) return -1;
    const int existing = st.hot_local_by_global[(size_t)global_expert];
    if (existing >= 0) {  // already resident (pinned-hot or cached)
        if (existing >= st.hot_active && st.cache_slots > 0) {
            const int sl = existing - st.hot_active;
            if (sl >= 0 && sl < st.cache_slots) st.spare_lru[(size_t)sl] = ++st.lru_clock;
        }
        return existing;
    }
    if (st.cache_slots <= 0) return -1;  // no cache
    const int cold_local = st.cold_local_by_global[(size_t)global_expert];
    if (cold_local < 0) return -1;       // not a cold expert
    // Validate the tensors for whichever expert layout this model uses.
    if (st.fused_gate_up) {
        if (!st.gate_up_hot || !st.down_hot || !st.gate_up_cold || !st.down_cold) return -1;
    } else {
        if (!st.gate_hot || !st.up_hot || !st.down_hot ||
            !st.gate_cold || !st.up_cold || !st.down_cold) return -1;
    }

    // Pick a free spare slot, else evict the LRU one.
    int slot = -1; uint64_t best = (uint64_t)-1;
    for (int i = 0; i < st.cache_slots; ++i) {
        if (st.spare_global[(size_t)i] < 0) { slot = i; break; }
        if (st.spare_lru[(size_t)i] < best) { best = st.spare_lru[(size_t)i]; slot = i; }
    }
    if (slot < 0) return -1;
    const int evicted = st.spare_global[(size_t)slot];
    if (evicted >= 0) st.hot_local_by_global[(size_t)evicted] = -1;  // evicted -> served cold again
    if (evicted >= 0 && evicted < 256) st.expert_vram_mask[evicted >> 6] &= ~(1ULL << (evicted & 63));

    const int hslot = st.hot_active + slot;  // hot-local index of the spare slot
    auto copy_slice = [&](ggml_tensor * cold_t, ggml_tensor * hot_t, size_t ebytes) {
        const uint8_t * src = (const uint8_t *)cold_t->data + (size_t)cold_local * ebytes;
        // Pinned cold store + async H2D on cudaStreamPerThread -> overlaps the
        // compute stream. Pinned makes cudaMemcpyAsync truly asynchronous.
        ggml_backend_tensor_set_async(gpu_backend, hot_t, src, (size_t)hslot * ebytes, ebytes);
    };
    if (st.fused_gate_up) {
        copy_slice(st.gate_up_cold, st.gate_up_hot, st.gate_up_expert_bytes);
        copy_slice(st.down_cold,    st.down_hot,    st.down_expert_bytes);
    } else {
        copy_slice(st.gate_cold, st.gate_hot, st.gate_expert_bytes);
        copy_slice(st.up_cold,   st.up_hot,   st.up_expert_bytes);
        copy_slice(st.down_cold, st.down_hot, st.down_expert_bytes);
    }

    st.hot_local_by_global[(size_t)global_expert] = hslot;
    if (global_expert < 256) st.expert_vram_mask[global_expert >> 6] |= 1ULL << (global_expert & 63);
    st.spare_global[(size_t)slot] = global_expert;
    st.spare_lru[(size_t)slot] = ++st.lru_clock;
    return hslot;
}

MoeSparkBudget spark_budget_split(uint64_t expert_budget, uint64_t total_expert_bytes,
                                  int n_expert, uint64_t core_kv_safety,
                                  uint64_t target_bytes) {
    if (target_bytes > 0) {
        const uint64_t avail = target_bytes > core_kv_safety ? target_bytes - core_kv_safety : 0;
        if (avail < expert_budget) expert_budget = avail;
    }
    MoeSparkBudget r{expert_budget, 0};
    if (n_expert > 0 && total_expert_bytes > 0 && expert_budget > 0) {
        const uint64_t bytes_per_slot = total_expert_bytes / (uint64_t)n_expert;  // 1 expert, all layers
        if (bytes_per_slot > 0) {
            uint64_t reserve = expert_budget / 8;             // ~12% for the cache ring
            const uint64_t cap = 1536ULL * 1024 * 1024;       // capped at 1.5 GiB
            if (reserve > cap) reserve = cap;
            const int slots = (int)(reserve / bytes_per_slot);
            if (slots > 0) {
                r.cache_slots = slots;
                r.hot_bytes = expert_budget - (uint64_t)slots * bytes_per_slot;
            }
        }
    }
    return r;
}

bool build_moe_hybrid_storage_from_file_with_mmap(
    const MoeHybridConfig & cfg,
    ggml_backend_t gpu_backend,
    const MoeHybridPlacement & placement,
    const std::vector<MoeLayerDesc> & layer_descs,
    const std::vector<LayerExpertFileData> & file_data,
    const void * mmap_base,
    size_t mmap_total_size,
    MoeHybridStorage & out,
    std::string * err,
    int cache_slots) {

    // First build storage normally (hot GPU + cold CPU buffers).
    if (!build_moe_hybrid_storage_from_file(cfg, gpu_backend, placement, layer_descs, file_data, out, err, cache_slots)) {
        return false;
    }

    // Store mmap metadata for streaming prefill.
    out.mmap_data = mmap_base;
    out.mmap_size = mmap_total_size;

    // Compute per-layer expert file regions (offsets relative to mmap base).
    const auto * base = static_cast<const uint8_t *>(mmap_base);
    out.layer_regions.resize((size_t)cfg.n_layer);
    for (int il = 0; il < cfg.n_layer; ++il) {
        const auto & fd = file_data[(size_t)il];
        auto & reg = out.layer_regions[(size_t)il];

        if (fd.gate_exps.data && fd.gate_exps.size > 0) {
            reg.gate_exps.offset = (size_t)(fd.gate_exps.data - base);
            reg.gate_exps.size = fd.gate_exps.size;
        }
        if (fd.up_exps.data && fd.up_exps.size > 0) {
            reg.up_exps.offset = (size_t)(fd.up_exps.data - base);
            reg.up_exps.size = fd.up_exps.size;
        }
        if (fd.down_exps.data && fd.down_exps.size > 0) {
            reg.down_exps.offset = (size_t)(fd.down_exps.data - base);
            reg.down_exps.size = fd.down_exps.size;
        }
        if (fd.gate_up_exps.data && fd.gate_up_exps.size > 0) {
            reg.gate_up_exps.offset = (size_t)(fd.gate_up_exps.data - base);
            reg.gate_up_exps.size = fd.gate_up_exps.size;
        }

        // Copy per-expert byte sizes from layer storage (already computed)
        const auto & ls = out.layers[(size_t)il];
        reg.expert_bytes_gate    = ls.gate_expert_bytes;
        reg.expert_bytes_up      = ls.up_expert_bytes;
        reg.expert_bytes_down    = ls.down_expert_bytes;
        reg.expert_bytes_gate_up = ls.gate_up_expert_bytes;
        reg.fused_gate_up        = ls.fused_gate_up;
    }

    return true;
}

}  // namespace dflash::common
