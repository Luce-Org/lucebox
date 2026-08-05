#include "dynamic_backend.h"

#include "ggml-cuda.h"

#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace dflash::common {
namespace {

namespace fs = std::filesystem;

static std::string executable_directory() {
#if defined(__linux__)
    std::vector<char> path(1024);
    for (;;) {
        const ssize_t size = ::readlink("/proc/self/exe", path.data(), path.size());
        if (size < 0) return {};
        if ((size_t)size < path.size()) {
            return fs::path(std::string(path.data(), (size_t)size))
                .parent_path().string();
        }
        path.resize(path.size() * 2);
    }
#else
    return {};
#endif
}

#if defined(DFLASH27B_BACKEND_MIXED)
static ggml_backend_reg_t load_peer_registry(PlacementBackend backend,
                                             std::string * error) {
    static std::mutex mutex;
    static ggml_backend_reg_t cuda_registry = nullptr;
    static ggml_backend_reg_t hip_registry = nullptr;
    std::lock_guard<std::mutex> lock(mutex);

    ggml_backend_reg_t & registry = backend == PlacementBackend::Cuda
        ? cuda_registry : hip_registry;
    if (registry) return registry;
    const char * registry_name = backend == PlacementBackend::Cuda
        ? "CUDA" : "ROCm";
    const char * path_variable = backend == PlacementBackend::Cuda
        ? "DFLASH_CUDA_BACKEND_PATH" : "DFLASH_HIP_BACKEND_PATH";
    const char * module_name = backend == PlacementBackend::Cuda
        ? "libggml-cuda.so" : "libggml-hip.so";

    registry = ggml_backend_reg_by_name(registry_name);
    if (registry) return registry;

    std::vector<fs::path> candidates;
    if (const char * explicit_path = std::getenv(path_variable)) {
        if (*explicit_path) candidates.emplace_back(explicit_path);
    }
    if (candidates.empty()) {
        const std::string exe_dir = executable_directory();
        if (!exe_dir.empty()) {
            candidates.emplace_back(fs::path(exe_dir) / module_name);
        }
    }

    std::string attempted;
    for (const fs::path & candidate : candidates) {
        std::error_code ec;
        if (!fs::is_regular_file(candidate, ec)) {
            if (!attempted.empty()) attempted += ", ";
            attempted += candidate.string();
            continue;
        }
        ggml_backend_reg_t loaded = ggml_backend_load(candidate.string().c_str());
        if (loaded &&
            std::string(ggml_backend_reg_name(loaded)) == registry_name) {
            registry = loaded;
            return registry;
        }
        if (loaded) ggml_backend_unload(loaded);
        if (!attempted.empty()) attempted += ", ";
        attempted += candidate.string();
    }

    if (error) {
        *error = "could not load the ";
        *error += registry_name;
        *error += " backend module";
        if (!attempted.empty()) *error += " (tried " + attempted + ")";
        *error += "; set ";
        *error += path_variable;
        *error += " to the full path of ";
        *error += module_name;
    }
    return nullptr;
}
#endif

}  // namespace

ggml_backend_t init_placement_backend(PlacementBackend backend,
                                      int device,
                                      std::string * error) {
    if (backend == PlacementBackend::Auto) backend = compiled_placement_backend();
    if (device < 0) {
        if (error) *error = "GPU device index must be non-negative";
        return nullptr;
    }

    if (backend == compiled_placement_backend()) {
        const int count = ggml_backend_cuda_get_device_count();
        if (device >= count) {
            if (error) {
                *error = std::string(placement_backend_name(backend)) +
                         " device " + std::to_string(device) +
                         " is out of range (found " +
                         std::to_string(count) + ")";
            }
            return nullptr;
        }
        ggml_backend_t result = ggml_backend_cuda_init(device);
        if (!result && error) {
            *error = "failed to initialize ";
            *error += placement_backend_name(backend);
            *error += " device " + std::to_string(device);
        }
        return result;
    }

#if defined(DFLASH27B_BACKEND_MIXED)
    if (backend == PlacementBackend::Cuda || backend == PlacementBackend::Hip) {
        ggml_backend_reg_t registry = load_peer_registry(backend, error);
        if (!registry) return nullptr;
        const size_t count = ggml_backend_reg_dev_count(registry);
        if ((size_t)device >= count) {
            if (error) {
                *error = std::string(placement_backend_name(backend)) +
                         " device " + std::to_string(device) +
                         " is out of range (found " +
                         std::to_string(count) + ")";
            }
            return nullptr;
        }
        ggml_backend_t result = ggml_backend_dev_init(
            ggml_backend_reg_dev_get(registry, (size_t)device), nullptr);
        if (!result && error) {
            *error = "failed to initialize ";
            *error += placement_backend_name(backend);
            *error += " device " + std::to_string(device);
        }
        return result;
    }
#endif

    if (error) {
        *error = "this binary does not contain the requested GPU backend";
    }
    return nullptr;
}

PlacementBackend placement_backend_of(ggml_backend_t backend) {
    if (!backend) return PlacementBackend::Auto;
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (!device) return PlacementBackend::Auto;
    ggml_backend_reg_t registry = ggml_backend_dev_backend_reg(device);
    if (!registry) return PlacementBackend::Auto;
    const char * name = ggml_backend_reg_name(registry);
    if (!name) return PlacementBackend::Auto;
    const std::string value(name);
    if (value == "CUDA") return PlacementBackend::Cuda;
    if (value == "ROCm") return PlacementBackend::Hip;
    return PlacementBackend::Auto;
}

BackendPairCapabilities backend_pair_capabilities(ggml_backend_t first,
                                                  ggml_backend_t second) {
    BackendPairCapabilities result;
    if (!first || !second) return result;

    result.same_runtime = ggml_guid_matches(
        ggml_backend_guid(first), ggml_backend_guid(second));
    if (!result.same_runtime) return result;

    const auto is_gpu = [](ggml_backend_t backend) {
        ggml_backend_dev_t device = ggml_backend_get_device(backend);
        if (!device) return false;
        const enum ggml_backend_dev_type type =
            ggml_backend_dev_type(device);
        return type == GGML_BACKEND_DEVICE_TYPE_GPU ||
               type == GGML_BACKEND_DEVICE_TYPE_IGPU;
    };
    result.native_gpu_handoff = is_gpu(first) && is_gpu(second);
    return result;
}

}  // namespace dflash::common
