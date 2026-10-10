#include "device_select.h"

#include "common/gguf_inspect.h"
#include "common/gpu_runtime_compat.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include "kv_quant.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <regex>
#include <utility>

#if defined(__linux__) && (defined(LUCE_BACKEND_HIP) || defined(GGML_USE_HIP))
#include "kfd_topology.h"

#include <unistd.h>
#endif

namespace luce::common {

namespace {

#if defined(__linux__) && (defined(LUCE_BACKEND_HIP) || defined(GGML_USE_HIP))
std::string read_first_line(const std::filesystem::path & path) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    return line;
}

// Another process that holds a GPU, read from /sys/class/kfd/kfd/proc/<pid>:
// it has compute or copy queues there, or more than 64 MiB of device memory
// (vram_<gpu_id> also counts an APU's carve-out; it reads as a negative
// number when unused). Queues appear within a second of a server starting,
// before its weights load, and a split server holds every GPU it spans.
// Opening a runtime context alone does neither, so tools that only enumerate
// devices do not count, and neither does this process.
struct KfdGpuUser {
    std::string pid;
    std::string name;   // /proc/<pid>/comm
    std::string model;  // first .gguf on its command line
    std::vector<uint32_t> gpu_ids;
};

std::string first_gguf_argument(const std::string & pid) {
    std::ifstream in("/proc/" + pid + "/cmdline", std::ios::binary);
    std::string arg;
    while (std::getline(in, arg, '\0')) {
        if (arg.size() > 5 && arg.compare(arg.size() - 5, 5, ".gguf") == 0) {
            return std::filesystem::path(arg).filename().string();
        }
    }
    return {};
}

std::vector<KfdGpuUser> kfd_other_gpu_users(const std::vector<uint32_t> & gpu_ids) {
    namespace fs = std::filesystem;
    constexpr long long kMemoryThreshold = 64ll << 20;
    std::vector<KfdGpuUser> users;
    std::error_code ec;
    const std::string self = std::to_string(getpid());
    // Processes and queues can disappear while this snapshot is read.
    for (auto proc = fs::directory_iterator("/sys/class/kfd/kfd/proc", ec);
         !ec && proc != fs::directory_iterator(); proc.increment(ec)) {
        KfdGpuUser user;
        user.pid = proc->path().filename().string();
        if (user.pid == self) continue;
        std::vector<std::string> queue_gpus;
        std::error_code queue_ec;
        for (auto queue = fs::directory_iterator(proc->path() / "queues", queue_ec);
             !queue_ec && queue != fs::directory_iterator(); queue.increment(queue_ec)) {
            queue_gpus.push_back(read_first_line(queue->path() / "gpuid"));
        }
        for (uint32_t gpu_id : gpu_ids) {
            const std::string id = std::to_string(gpu_id);
            const bool queues =
                std::find(queue_gpus.begin(), queue_gpus.end(), id) != queue_gpus.end();
            const long long memory = (long long) std::strtoull(
                read_first_line(proc->path() / ("vram_" + id)).c_str(), nullptr, 10);
            if (gpu_id && (queues || memory > kMemoryThreshold)) user.gpu_ids.push_back(gpu_id);
        }
        if (user.gpu_ids.empty()) continue;
        user.name = read_first_line("/proc/" + user.pid + "/comm");
        user.model = first_gguf_argument(user.pid);
        users.push_back(std::move(user));
    }
    return users;
}

// Mark the devices other processes hold, naming each process, its model and
// every device it spans: "pid 42 luce_server ds4.gguf on hip:0+hip:1".
void mark_busy_devices(std::vector<GpuDeviceInfo> & devices,
                       const std::vector<uint32_t> & gpu_ids) {
    const std::vector<KfdGpuUser> users = kfd_other_gpu_users(gpu_ids);
    const std::string backend = placement_backend_name(compiled_placement_backend());
    for (size_t d = 0; d < devices.size(); ++d) {
        for (const KfdGpuUser & user : users) {
            if (!gpu_ids[d] || std::find(user.gpu_ids.begin(), user.gpu_ids.end(),
                                         gpu_ids[d]) == user.gpu_ids.end()) {
                continue;
            }
            std::string spans;
            for (size_t other = 0; other < devices.size(); ++other) {
                if (gpu_ids[other] && std::find(user.gpu_ids.begin(), user.gpu_ids.end(),
                                                gpu_ids[other]) != user.gpu_ids.end()) {
                    spans += (spans.empty() ? "" : "+") + backend + ":" +
                             std::to_string(devices[other].index);
                }
            }
            std::string & by = devices[d].busy_by;
            by += std::string(by.empty() ? "" : "; ") + "pid " + user.pid +
                  (user.name.empty() ? "" : " " + user.name) +
                  (user.model.empty() ? "" : " " + user.model) + " on " + spans;
            devices[d].busy = true;
        }
    }
}
#endif

}  // namespace

std::vector<GpuDeviceInfo> enumerate_gpu_devices(bool query_free) {
    std::vector<GpuDeviceInfo> devices;
    const int count = ggml_backend_cuda_get_device_count();
#if defined(__linux__) && (defined(LUCE_BACKEND_HIP) || defined(GGML_USE_HIP))
    std::vector<uint32_t> gpu_ids(query_free ? (size_t) std::max(count, 0) : 0, 0);
#endif
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
#if defined(__linux__)
            char pci_address[32] = {};
            if (query_free &&
                hipDeviceGetPCIBusId(pci_address, sizeof(pci_address), i) == hipSuccess) {
                gpu_ids[(size_t) i] = detail::kfd_gpu_id(
                    "/sys/class/kfd/kfd/topology/nodes", pci_address);
            }
#endif
#else
            info.arch = "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
#endif
        }
        devices.push_back(std::move(info));
    }
#if defined(__linux__) && (defined(LUCE_BACKEND_HIP) || defined(GGML_USE_HIP))
    if (query_free) mark_busy_devices(devices, gpu_ids);
#endif
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
