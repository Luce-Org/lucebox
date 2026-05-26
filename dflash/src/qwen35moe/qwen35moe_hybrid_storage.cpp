#include "qwen35moe_hybrid_storage.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cstring>

namespace dflash::common {

void CachedFfnGraph::free() {
    if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
    if (ctx) { ggml_free(ctx); ctx = nullptr; }
    gf = nullptr;
    inp = nullptr;
    ids = nullptr;
    weights = nullptr;
    output = nullptr;
    n_hot = 0;
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

// Read expert slices from raw memory (e.g. mmap) instead of a GPU tensor.
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

Qwen35MoeHybridStorage::~Qwen35MoeHybridStorage() {
    for (auto & layer : layers) {
        layer.hot_graph.free();
        layer.cold_graph.free();
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
}

bool Qwen35MoeHybridStorage::matches(const TargetWeights & w) const {
    return placement.matches(w) && (int)layers.size() == w.n_layer;
}

bool Qwen35MoeHybridStorage::empty() const {
    return layers.empty();
}

bool build_qwen35moe_hybrid_storage(const TargetWeights & w,
                                    ggml_backend_t backend,
                                    const Qwen35MoeExpertPlacement & placement,
                                    Qwen35MoeHybridStorage & out,
                                    std::string * err) {
    if (!placement.matches(w)) {
        if (err) *err = "placement does not match model";
        return false;
    }
    if (!w.is_moe) {
        if (err) *err = "target is not qwen35moe";
        return false;
    }

    
    out.placement = placement;
    out.layers.resize((size_t)w.n_layer);
    out.cpu_backend = ggml_backend_cpu_init();
    if (!out.cpu_backend) {
        if (err) *err = "failed to init cpu backend";
        return false;
    }
    ggml_backend_cpu_set_n_threads(out.cpu_backend, std::max(1, std::min(w.n_expert_used, 8)));

    for (int il = 0; il < w.n_layer; ++il) {
        const TargetLayer & L = w.layers[(size_t)il];
        Qwen35MoeHybridLayerStorage & dst = out.layers[(size_t)il];
        dst.hot_expert_ids = placement.hot_expert_ids[(size_t)il];
        dst.hot_local_by_global.assign((size_t)w.n_expert, -1);
        dst.cold_local_by_global.assign((size_t)w.n_expert, -1);

        std::vector<uint8_t> is_hot((size_t)w.n_expert, 0);
        for (size_t i = 0; i < dst.hot_expert_ids.size(); ++i) {
            const int32_t expert = dst.hot_expert_ids[i];
            if (expert < 0 || expert >= w.n_expert) {
                if (err) *err = "hot expert id out of range";
                return false;
            }
            dst.hot_local_by_global[(size_t)expert] = (int32_t)i;
            is_hot[(size_t)expert] = 1;
        }
        for (int expert = 0; expert < w.n_expert; ++expert) {
            if (!is_hot[(size_t)expert]) {
                dst.cold_local_by_global[(size_t)expert] = (int32_t)dst.cold_expert_ids.size();
                dst.cold_expert_ids.push_back((int32_t)expert);
            }
        }

        dst.fused_gate_up = (L.ffn_gate_up_exps != nullptr);
        if (!validate_expert_tensor(L.ffn_gate_exps, w.n_expert, &dst.gate_expert_bytes, err) ||
            !validate_expert_tensor(L.ffn_up_exps, w.n_expert, &dst.up_expert_bytes, err) ||
            !validate_expert_tensor(L.ffn_down_exps, w.n_expert, &dst.down_expert_bytes, err) ||
            !validate_expert_tensor(L.ffn_gate_up_exps, w.n_expert, &dst.gate_up_expert_bytes, err)) {
            return false;
        }

        const int cold_count = (int)dst.cold_expert_ids.size();
        const int hot_count = (int)dst.hot_expert_ids.size();

        // Allocate hot expert tensors on GPU
        if (hot_count > 0) {
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
                dst.gate_up_hot = new_like_with_expert_count(dst.hot_ctx, L.ffn_gate_up_exps, hot_count);
                dst.down_hot    = new_like_with_expert_count(dst.hot_ctx, L.ffn_down_exps, hot_count);
            } else {
                dst.gate_hot = new_like_with_expert_count(dst.hot_ctx, L.ffn_gate_exps, hot_count);
                dst.up_hot   = new_like_with_expert_count(dst.hot_ctx, L.ffn_up_exps, hot_count);
                dst.down_hot = new_like_with_expert_count(dst.hot_ctx, L.ffn_down_exps, hot_count);
            }
            dst.hot_buf = ggml_backend_alloc_ctx_tensors(dst.hot_ctx, backend);
            if (!dst.hot_buf) {
                if (err) *err = "failed to allocate hot expert buffer";
                return false;
            }

            // Copy hot expert slices from full GPU tensors to hot_buf
            std::vector<uint8_t> hot_bytes;
            if (dst.fused_gate_up) {
                if (!read_expert_slices(backend, L.ffn_gate_up_exps, dst.hot_expert_ids,
                                        dst.gate_up_expert_bytes, hot_bytes, err)) {
                    return false;
                }
                ggml_backend_tensor_set(dst.gate_up_hot, hot_bytes.data(), 0, hot_bytes.size());
                if (!read_expert_slices(backend, L.ffn_down_exps, dst.hot_expert_ids,
                                        dst.down_expert_bytes, hot_bytes, err)) {
                    return false;
                }
                ggml_backend_tensor_set(dst.down_hot, hot_bytes.data(), 0, hot_bytes.size());
            } else {
                if (!read_expert_slices(backend, L.ffn_gate_exps, dst.hot_expert_ids,
                                        dst.gate_expert_bytes, hot_bytes, err)) {
                    return false;
                }
                ggml_backend_tensor_set(dst.gate_hot, hot_bytes.data(), 0, hot_bytes.size());
                if (!read_expert_slices(backend, L.ffn_up_exps, dst.hot_expert_ids,
                                        dst.up_expert_bytes, hot_bytes, err)) {
                    return false;
                }
                ggml_backend_tensor_set(dst.up_hot, hot_bytes.data(), 0, hot_bytes.size());
                if (!read_expert_slices(backend, L.ffn_down_exps, dst.hot_expert_ids,
                                        dst.down_expert_bytes, hot_bytes, err)) {
                    return false;
                }
                ggml_backend_tensor_set(dst.down_hot, hot_bytes.data(), 0, hot_bytes.size());
            }
        }

        // Allocate cold expert tensors on CPU
        if (cold_count > 0) {
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
                dst.gate_up_cold = new_like_with_expert_count(dst.cold_ctx, L.ffn_gate_up_exps, cold_count);
                dst.down_cold    = new_like_with_expert_count(dst.cold_ctx, L.ffn_down_exps, cold_count);
            } else {
                dst.gate_cold = new_like_with_expert_count(dst.cold_ctx, L.ffn_gate_exps, cold_count);
                dst.up_cold   = new_like_with_expert_count(dst.cold_ctx, L.ffn_up_exps, cold_count);
                dst.down_cold = new_like_with_expert_count(dst.cold_ctx, L.ffn_down_exps, cold_count);
            }
            dst.cold_buf = ggml_backend_alloc_ctx_tensors(dst.cold_ctx, out.cpu_backend);
            if (!dst.cold_buf) {
                if (err) *err = "failed to allocate cold expert buffer";
                return false;
            }
        }

        if (dst.fused_gate_up) {
            if (!read_expert_slices(backend, L.ffn_gate_up_exps, dst.cold_expert_ids,
                                    dst.gate_up_expert_bytes, dst.gate_up_cold_bytes, err)) {
                return false;
            }
        } else {
            if (!read_expert_slices(backend, L.ffn_gate_exps, dst.cold_expert_ids,
                                    dst.gate_expert_bytes, dst.gate_cold_bytes, err) ||
                !read_expert_slices(backend, L.ffn_up_exps, dst.cold_expert_ids,
                                    dst.up_expert_bytes, dst.up_cold_bytes, err)) {
                return false;
            }
        }
        if (!read_expert_slices(backend, L.ffn_down_exps, dst.cold_expert_ids,
                                dst.down_expert_bytes, dst.down_cold_bytes, err)) {
            return false;
        }

        if (dst.fused_gate_up) {
            if (dst.gate_up_cold && !dst.gate_up_cold_bytes.empty()) {
                ggml_backend_tensor_set(dst.gate_up_cold, dst.gate_up_cold_bytes.data(), 0, dst.gate_up_cold_bytes.size());
                dst.gate_up_cold_bytes.clear();
                dst.gate_up_cold_bytes.shrink_to_fit();
            }
            if (dst.down_cold && !dst.down_cold_bytes.empty()) {
                ggml_backend_tensor_set(dst.down_cold, dst.down_cold_bytes.data(), 0, dst.down_cold_bytes.size());
                dst.down_cold_bytes.clear();
                dst.down_cold_bytes.shrink_to_fit();
            }
        } else {
            if (dst.gate_cold && !dst.gate_cold_bytes.empty()) {
                ggml_backend_tensor_set(dst.gate_cold, dst.gate_cold_bytes.data(), 0, dst.gate_cold_bytes.size());
                dst.gate_cold_bytes.clear();
                dst.gate_cold_bytes.shrink_to_fit();
            }
            if (dst.up_cold && !dst.up_cold_bytes.empty()) {
                ggml_backend_tensor_set(dst.up_cold, dst.up_cold_bytes.data(), 0, dst.up_cold_bytes.size());
                dst.up_cold_bytes.clear();
                dst.up_cold_bytes.shrink_to_fit();
            }
            if (dst.down_cold && !dst.down_cold_bytes.empty()) {
                ggml_backend_tensor_set(dst.down_cold, dst.down_cold_bytes.data(), 0, dst.down_cold_bytes.size());
                dst.down_cold_bytes.clear();
                dst.down_cold_bytes.shrink_to_fit();
            }
        }
    }

    return true;
}

bool build_qwen35moe_hybrid_storage_from_file(
    const TargetWeights & w,
    ggml_backend_t gpu_backend,
    const Qwen35MoeExpertPlacement & placement,
    const std::vector<LayerExpertFileData> & file_data,
    Qwen35MoeHybridStorage & out,
    std::string * err) {

    if (!placement.matches(w)) {
        if (err) *err = "placement does not match model";
        return false;
    }
    if (!w.is_moe) {
        if (err) *err = "target is not qwen35moe";
        return false;
    }
    if ((int)file_data.size() != w.n_layer) {
        if (err) *err = "file_data size does not match n_layer";
        return false;
    }

    
    out.placement = placement;
    out.layers.resize((size_t)w.n_layer);
    out.cpu_backend = ggml_backend_cpu_init();
    if (!out.cpu_backend) {
        if (err) *err = "failed to init cpu backend";
        return false;
    }
    ggml_backend_cpu_set_n_threads(out.cpu_backend, std::max(1, std::min(w.n_expert_used, 8)));

    for (int il = 0; il < w.n_layer; ++il) {
        const TargetLayer & L = w.layers[(size_t)il];
        const LayerExpertFileData & fd = file_data[(size_t)il];
        Qwen35MoeHybridLayerStorage & dst = out.layers[(size_t)il];
        dst.hot_expert_ids = placement.hot_expert_ids[(size_t)il];
        dst.hot_local_by_global.assign((size_t)w.n_expert, -1);
        dst.cold_local_by_global.assign((size_t)w.n_expert, -1);

        std::vector<uint8_t> is_hot((size_t)w.n_expert, 0);
        for (size_t i = 0; i < dst.hot_expert_ids.size(); ++i) {
            const int32_t expert = dst.hot_expert_ids[i];
            if (expert < 0 || expert >= w.n_expert) {
                if (err) *err = "hot expert id out of range";
                return false;
            }
            dst.hot_local_by_global[(size_t)expert] = (int32_t)i;
            is_hot[(size_t)expert] = 1;
        }
        for (int expert = 0; expert < w.n_expert; ++expert) {
            if (!is_hot[(size_t)expert]) {
                dst.cold_local_by_global[(size_t)expert] = (int32_t)dst.cold_expert_ids.size();
                dst.cold_expert_ids.push_back((int32_t)expert);
            }
        }

        dst.fused_gate_up = (L.ffn_gate_up_exps != nullptr);
        if (!validate_expert_tensor(L.ffn_gate_exps, w.n_expert, &dst.gate_expert_bytes, err) ||
            !validate_expert_tensor(L.ffn_up_exps, w.n_expert, &dst.up_expert_bytes, err) ||
            !validate_expert_tensor(L.ffn_down_exps, w.n_expert, &dst.down_expert_bytes, err) ||
            !validate_expert_tensor(L.ffn_gate_up_exps, w.n_expert, &dst.gate_up_expert_bytes, err)) {
            return false;
        }

        const int hot_count = (int)dst.hot_expert_ids.size();
        const int cold_count = (int)dst.cold_expert_ids.size();

        // Allocate hot expert tensors on GPU
        if (hot_count > 0) {
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
                dst.gate_up_hot = new_like_with_expert_count(dst.hot_ctx, L.ffn_gate_up_exps, hot_count);
                dst.down_hot    = new_like_with_expert_count(dst.hot_ctx, L.ffn_down_exps, hot_count);
            } else {
                dst.gate_hot = new_like_with_expert_count(dst.hot_ctx, L.ffn_gate_exps, hot_count);
                dst.up_hot   = new_like_with_expert_count(dst.hot_ctx, L.ffn_up_exps, hot_count);
                dst.down_hot = new_like_with_expert_count(dst.hot_ctx, L.ffn_down_exps, hot_count);
            }
            dst.hot_buf = ggml_backend_alloc_ctx_tensors(dst.hot_ctx, gpu_backend);
            if (!dst.hot_buf) {
                if (err) *err = "failed to allocate hot expert GPU buffer";
                return false;
            }

            // Load hot expert slices from file
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

        // Allocate cold expert tensors on CPU
        if (cold_count > 0) {
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
                dst.gate_up_cold = new_like_with_expert_count(dst.cold_ctx, L.ffn_gate_up_exps, cold_count);
                dst.down_cold    = new_like_with_expert_count(dst.cold_ctx, L.ffn_down_exps, cold_count);
            } else {
                dst.gate_cold = new_like_with_expert_count(dst.cold_ctx, L.ffn_gate_exps, cold_count);
                dst.up_cold   = new_like_with_expert_count(dst.cold_ctx, L.ffn_up_exps, cold_count);
                dst.down_cold = new_like_with_expert_count(dst.cold_ctx, L.ffn_down_exps, cold_count);
            }
            dst.cold_buf = ggml_backend_alloc_ctx_tensors(dst.cold_ctx, out.cpu_backend);
            if (!dst.cold_buf) {
                if (err) *err = "failed to allocate cold expert CPU buffer";
                return false;
            }

            // Load cold expert slices from file directly to CPU tensors
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

}  // namespace dflash::common
