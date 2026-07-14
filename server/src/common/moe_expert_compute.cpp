#include "moe_expert_compute.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#if !defined(_WIN32)
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash::common {

namespace {

uint64_t hash_u64(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

const char * nonempty_env(const char * name) {
    const char * raw = std::getenv(name);
    return raw && *raw ? raw : nullptr;
}

int parse_nonnegative_env(const char * name, int fallback) {
    const char * raw = nonempty_env(name);
    if (!raw) return fallback;
    errno = 0;
    char * end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (errno == ERANGE || end == raw || *end != '\0' ||
        value < 0 || value > std::numeric_limits<int>::max()) {
        return fallback;
    }
    return (int)value;
}

std::string make_runtime_key(const MoeExpertComputeRuntimeConfig & cfg) {
    std::string key = cfg.target_path;
    key += "|n_layer=" + std::to_string(cfg.n_layer);
    key += "|n_expert=" + std::to_string(cfg.n_expert);
    key += "|n_used=" + std::to_string(cfg.n_expert_used);
    key += "|n_embd=" + std::to_string(cfg.n_embd);
    key += "|n_ff=" + std::to_string(cfg.n_ff_exp);
    if (const char * ipc_bin = nonempty_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_BIN")) {
        key += "|ipc_bin=";
        key += ipc_bin;
        key += "|ipc_gpu=" + std::to_string(
            parse_nonnegative_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_GPU", 0));
        key += "|ipc_work=";
        if (const char * work_dir = nonempty_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_WORK_DIR")) {
            key += work_dir;
        }
        key += "|ipc_required=" + std::to_string(
            parse_nonnegative_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_REQUIRED", 0));
    } else {
        key += "|cpu";
    }
    return key;
}

bool validate_executable_file(const char * path, std::string * err) {
#if defined(_WIN32)
    (void)path;
    (void)err;
    return true;
#else
    struct stat st {};
    if (::stat(path, &st) != 0) {
        if (err) *err = std::string("MoE expert compute IPC binary is not accessible: ") +
                        path + ": " + std::strerror(errno);
        return false;
    }
    if (!S_ISREG(st.st_mode)) {
        if (err) *err = std::string("MoE expert compute IPC binary is not a regular file: ") + path;
        return false;
    }
    if (::access(path, X_OK) != 0) {
        if (err) *err = std::string("MoE expert compute IPC binary is not executable: ") +
                        path + ": " + std::strerror(errno);
        return false;
    }
    return true;
#endif
}

}  // namespace

uint64_t moe_expert_placement_fingerprint(const MoeHybridStorage & hybrid,
                                         int n_layer,
                                         int n_expert,
                                         int n_expert_used) {
    uint64_t h = 1469598103934665603ULL;
    h = hash_u64(h, (uint64_t)n_layer);
    h = hash_u64(h, (uint64_t)n_expert);
    h = hash_u64(h, (uint64_t)n_expert_used);
    h = hash_u64(h, (uint64_t)hybrid.placement.total_hot);
    for (size_t il = 0; il < hybrid.placement.hot_expert_ids.size(); ++il) {
        h = hash_u64(h, (uint64_t)il);
        for (int32_t expert : hybrid.placement.hot_expert_ids[il]) {
            h = hash_u64(h, (uint64_t)(uint32_t)expert);
        }
    }
    return h;
}

std::vector<MoeExpertLayer> make_moe_expert_layers(
    const MoeHybridStorage & hybrid,
    const std::vector<MoeLayerDesc> & layer_descs) {
    std::vector<MoeExpertLayer> layers(hybrid.layers.size());
    for (size_t il = 0; il < hybrid.layers.size(); ++il) {
        const auto & storage = hybrid.layers[il];
        const MoeLayerDesc * desc =
            il < layer_descs.size() ? &layer_descs[il] : nullptr;
        auto & cl = layers[il];
        cl.layer_idx = (int)il;
        cl.cold_global_by_local = storage.cold_expert_ids;
        cl.fused_gate_up = (storage.gate_up_cold != nullptr);
        if (cl.fused_gate_up) {
            cl.gate_up_data =
                storage.gate_up_cold ? storage.gate_up_cold->data : nullptr;
            cl.gate_up_stride =
                storage.gate_up_cold ? storage.gate_up_cold->nb[2] : 0;
            cl.gate_up_type =
                storage.gate_up_cold ? storage.gate_up_cold->type : GGML_TYPE_Q4_K;
            cl.gate_up_scale = desc ? desc->ffn_gate_up_exps_s : 1.0f;
        } else {
            cl.gate_data = storage.gate_cold ? storage.gate_cold->data : nullptr;
            cl.up_data = storage.up_cold ? storage.up_cold->data : nullptr;
            cl.gate_stride = storage.gate_cold ? storage.gate_cold->nb[2] : 0;
            cl.up_stride = storage.up_cold ? storage.up_cold->nb[2] : 0;
            cl.gate_type =
                storage.gate_cold ? storage.gate_cold->type : GGML_TYPE_Q4_K;
            cl.up_type =
                storage.up_cold ? storage.up_cold->type : GGML_TYPE_Q4_K;
            cl.gate_scale = desc ? desc->ffn_gate_exps_s : 1.0f;
            cl.up_scale = desc ? desc->ffn_up_exps_s : 1.0f;
        }
        cl.down_data = storage.down_cold ? storage.down_cold->data : nullptr;
        cl.down_stride = storage.down_cold ? storage.down_cold->nb[2] : 0;
        cl.down_type =
            storage.down_cold ? storage.down_cold->type : GGML_TYPE_Q4_K;
        cl.down_scale = desc ? desc->ffn_down_exps_s : 1.0f;
    }
    return layers;
}

void MoeExpertComputeRuntime::reset() {
    compute.reset();
    layers.clear();
    target_path.clear();
    runtime_key.clear();
    placement_fingerprint = 0;
}

bool ensure_moe_expert_compute_runtime(
    MoeExpertComputeRuntime & runtime,
    const MoeExpertComputeRuntimeConfig & cfg,
    const MoeHybridStorage & hybrid,
    const std::vector<MoeLayerDesc> & layer_descs,
    std::string * err) {
    if (!cfg.enabled) {
        runtime.reset();
        return true;
    }
    if (cfg.n_layer <= 0 || cfg.n_expert <= 0 || cfg.n_expert_used <= 0 ||
        cfg.n_embd <= 0 || cfg.n_ff_exp <= 0) {
        if (err) *err = "invalid MoE expert compute runtime config";
        runtime.reset();
        return false;
    }

    const uint64_t fingerprint =
        moe_expert_placement_fingerprint(hybrid, cfg.n_layer, cfg.n_expert,
                                         cfg.n_expert_used);
    const std::string runtime_key = make_runtime_key(cfg);
    const bool can_reuse =
        runtime.compute &&
        runtime.compute->healthy() &&
        runtime.runtime_key == runtime_key &&
        runtime.placement_fingerprint == fingerprint;
    if (!can_reuse) {
        runtime.compute.reset();
        runtime.target_path.clear();
        runtime.runtime_key.clear();
        runtime.placement_fingerprint = 0;
    }

    if (!runtime.compute) {
        if (const char * ipc_bin = nonempty_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_BIN")) {
            if (!validate_executable_file(ipc_bin, err)) {
                std::fprintf(stderr, "%s %s\n", cfg.log_prefix ? cfg.log_prefix : "[moe-expert-compute]",
                             err ? err->c_str() : "invalid remote IPC binary");
                runtime.reset();
                return false;
            }
            const char * work_dir = nonempty_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_WORK_DIR");
            const int remote_gpu =
                parse_nonnegative_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_GPU", 0);
            const bool required =
                parse_nonnegative_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_REQUIRED", 0) != 0;
            MoeExpertComputeIpcStartResult remote = make_moe_expert_compute_ipc(
                ipc_bin, cfg.target_path, remote_gpu, hybrid.placement,
                cfg.n_embd, cfg.n_ff_exp, cfg.n_expert_used,
                work_dir ? work_dir : "", required);
            if (required && !remote.started_remote) {
                if (err) *err = "remote MoE expert compute IPC is required but did not start";
                std::fprintf(stderr, "%s %s\n", cfg.log_prefix ? cfg.log_prefix : "[moe-expert-compute]",
                             err ? err->c_str() : "remote IPC did not start");
                runtime.reset();
                return false;
            }
            runtime.compute = std::move(remote.compute);
        }
        if (!runtime.compute) {
            runtime.compute = make_cpu_moe_expert_compute(cfg.n_ff_exp);
        }
    }

    runtime.layers = make_moe_expert_layers(hybrid, layer_descs);
    runtime.target_path = cfg.target_path;
    runtime.runtime_key = runtime_key;
    runtime.placement_fingerprint = fingerprint;
    return true;
}

}  // namespace dflash::common
