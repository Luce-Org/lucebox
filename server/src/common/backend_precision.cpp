#include "backend_precision.h"

#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
#include "gpu_runtime_compat.h"
#endif

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace dflash::common {
namespace {

std::string backend_device_description(ggml_backend_t backend) {
    if (!backend) {
        return {};
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) {
        return {};
    }
    const char * desc = ggml_backend_dev_description(dev);
    if (desc && desc[0]) {
        return desc;
    }
    return {};
}

std::string backend_device_logical_name(ggml_backend_t backend) {
    if (!backend) {
        return {};
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) {
        return {};
    }
    const char * name = ggml_backend_dev_name(dev);
    return name ? std::string(name) : std::string{};
}

std::string backend_name(ggml_backend_t backend) {
    if (!backend) {
        return {};
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) {
        return {};
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char * name = reg ? ggml_backend_reg_name(reg) : nullptr;
    return name ? std::string(name) : std::string{};
}

int parse_backend_device_id(const std::string & logical_name) {
    if (logical_name.empty()) return -1;
    size_t end = logical_name.size();
    while (end > 0 && std::isspace((unsigned char)logical_name[end - 1])) {
        --end;
    }
    size_t begin = end;
    while (begin > 0 && std::isdigit((unsigned char)logical_name[begin - 1])) {
        --begin;
    }
    if (begin == end) return -1;
    return std::atoi(logical_name.substr(begin, end - begin).c_str());
}

int current_device_id() {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0) {
        return -1;
    }
    return device;
#else
    return -1;
#endif
}

int device_props_for(int device,
                     std::string * device_name,
                     std::string * arch_name) {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    if (device < 0) {
        return 0;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return 0;
    }
    if (device_name && prop.name[0]) {
        *device_name = prop.name;
    }
#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    if (arch_name && prop.gcnArchName[0]) {
        *arch_name = prop.gcnArchName;
    }
#endif
    return prop.major * 10 + prop.minor;
#else
    (void)device;
    (void)device_name;
    (void)arch_name;
    return 0;
#endif
}

} // namespace

const char * backend_precision_type_name(ggml_type type) {
    return ggml_type_name(type);
}

BackendPrecisionPolicy select_drafter_precision_policy(ggml_backend_t backend) {
    BackendPrecisionPolicy policy;
    policy.backend_name = backend_name(backend);
    const std::string logical_name = backend_device_logical_name(backend);
    policy.device_name = backend_device_description(backend);
    policy.device_id = parse_backend_device_id(logical_name);
    if (policy.device_id < 0) {
        policy.device_id = current_device_id();
    }

#if defined(DFLASH27B_BACKEND_CUDA)
    policy.cuda_sm = device_props_for(
        policy.device_id,
        policy.device_name.empty() ? &policy.device_name : nullptr,
        nullptr);
    if (policy.cuda_sm >= 80) {
        policy.weight_type  = GGML_TYPE_BF16;
        policy.compute_type = GGML_TYPE_BF16;
        policy.reason       = "CUDA sm80+ BF16 tensor-core path";
    } else if (policy.cuda_sm >= 70) {
        policy.weight_type  = GGML_TYPE_F16;
        policy.compute_type = GGML_TYPE_F16;
        policy.reason       = "CUDA sm70-sm79 F16 tensor-core path";
    } else if (policy.cuda_sm == 60) {
        policy.weight_type  = GGML_TYPE_F16;
        policy.compute_type = GGML_TYPE_F16;
        policy.reason       = "CUDA sm60 GP100 F16 path";
    } else {
        policy.weight_type  = GGML_TYPE_F32;
        policy.compute_type = GGML_TYPE_F32;
        policy.reason       = "CUDA legacy compatibility fallback without useful F16/BF16 acceleration";
    }
#elif defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    policy.cuda_sm = device_props_for(
        policy.device_id,
        policy.device_name.empty() ? &policy.device_name : nullptr,
        &policy.runtime_arch);
    policy.weight_type  = GGML_TYPE_BF16;
    policy.compute_type = GGML_TYPE_BF16;
    policy.reason       = "HIP ROCm/ggml BF16-compatible path";
#else
    policy.weight_type  = GGML_TYPE_F32;
    policy.compute_type = GGML_TYPE_F32;
    policy.reason       = "portable non-GPU fallback";
#endif

    if (policy.backend_name.empty()) {
        policy.backend_name = "unknown";
    }
    if (policy.device_name.empty()) {
        policy.device_name = "unknown";
    }
    return policy;
}

} // namespace dflash::common
