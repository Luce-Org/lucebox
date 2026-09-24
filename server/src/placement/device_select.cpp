#include "device_select.h"

#include "common/gguf_inspect.h"
#include "common/gpu_runtime_compat.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include "kv_quant.h"

#include <cstdio>
#include <filesystem>
#include <regex>
#include <utility>

namespace luce::common {

std::vector<GpuDeviceInfo> enumerate_gpu_devices(bool query_free) {
    std::vector<GpuDeviceInfo> devices;
    const int count = ggml_backend_cuda_get_device_count();
    for (int i = 0; i < count; ++i) {
        GpuDeviceInfo info;
        info.index = i;
        char description[256] = {};
        ggml_backend_cuda_get_device_description(i, description, sizeof(description));
        info.name = description;
        if (query_free) {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            ggml_backend_cuda_get_device_memory(i, &free_bytes, &total_bytes);
            info.free_bytes = free_bytes;
        }
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            info.total_bytes = prop.totalGlobalMem;
            info.integrated = prop.integrated != 0;
#if defined(LUCE_BACKEND_HIP) || defined(GGML_USE_HIP)
            info.arch = prop.gcnArchName;
            const size_t features = info.arch.find(':');
            if (features != std::string::npos) info.arch.resize(features);
#else
            info.arch = "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
#endif
        }
        devices.push_back(std::move(info));
    }
    return devices;
}

uint64_t gguf_model_bytes(const std::string & path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const uint64_t first = fs::file_size(path, ec);
    if (ec) return 0;

    // llama.cpp split naming: <stem>-00001-of-00003.gguf
    static const std::regex split_re(R"(^(.*)-(\d{5})-of-(\d{5})\.gguf$)");
    std::smatch match;
    const std::string file = fs::path(path).filename().string();
    if (!std::regex_match(file, match, split_re)) return first;

    const int parts = std::stoi(match[3].str());
    uint64_t total = 0;
    for (int part = 1; part <= parts; ++part) {
        char suffix[32];
        std::snprintf(suffix, sizeof(suffix), "-%05d-of-%05d.gguf", part, parts);
        const fs::path sibling =
            fs::path(path).parent_path() / (match[1].str() + suffix);
        const uint64_t size = fs::file_size(sibling, ec);
        if (ec) return 0;
        total += size;
    }
    return total;
}

namespace {

// A u32 header value, or 0 when the key is missing or has another type (some
// architectures store per-layer arrays; those are not estimated).
uint32_t gguf_u32(const gguf_context * gctx, const std::string & key) {
    const int64_t id = gguf_find_key(gctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(gctx, id) != GGUF_TYPE_UINT32) return 0;
    return gguf_get_val_u32(gctx, id);
}

}  // namespace

uint64_t gguf_kv_cache_bytes(const std::string & path, int max_ctx,
                             ggml_type cache_type_k, ggml_type cache_type_v) {
    if (max_ctx <= 0) return 0;
    gguf_init_params params{};
    params.no_alloc = true;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), params);
    if (!gctx) return 0;

    uint64_t bytes = 0;
    const int64_t arch_id = gguf_find_key(gctx, "general.architecture");
    const std::string arch = arch_id >= 0 ? gguf_get_val_str(gctx, arch_id) : "";
    if (arch == "qwen35" || arch == "qwen35moe") {
        const std::string pre = arch + ".";
        uint32_t n_layer = 0;
        std::string error;
        const uint32_t n_head_kv = gguf_u32(gctx, pre + "attention.head_count_kv");
        const uint32_t key_length = gguf_u32(gctx, pre + "attention.key_length");
        const uint32_t value_length = gguf_u32(gctx, pre + "attention.value_length");
        if (n_head_kv && key_length &&
            derive_effective_target_layer_count(
                arch, gguf_u32(gctx, pre + "block_count"),
                gguf_u32(gctx, pre + "nextn_predict_layers"), n_layer, error)) {
            ggml_type kv_k = GGML_TYPE_Q4_0, kv_v = GGML_TYPE_Q4_0;
            luce::resolve_kv_types(kv_k, kv_v, cache_type_k, cache_type_v);
            bytes = luce::kv_reservation_bytes_per_token(
                        (int) n_layer, (int) gguf_u32(gctx, pre + "full_attention_interval"),
                        (int) n_head_kv, kv_k, (int) key_length,
                        kv_v, (int) (value_length ? value_length : key_length)) *
                    (uint64_t) max_ctx;
        }
    }
    gguf_free(gctx);
    return bytes;
}

}  // namespace luce::common
