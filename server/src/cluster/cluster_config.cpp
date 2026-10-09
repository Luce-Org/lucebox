// cluster_config.cpp - validation, parsing helpers and RCCL environment
// export for the caller-requested cluster configuration (cluster_config.h).
//
// Pure C++/POSIX: no ggml, HIP or RCCL dependency so the feature gate, the
// CLI parser and the GPU-free unit tests can link this unconditionally.

#include "cluster/cluster_config.h"

#include "common/platform_env.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>

namespace luce::cluster {

namespace {

// Strict decimal integer parse: whole string must be consumed.
bool parse_int_strict(const std::string & text, int & out) {
    if (text.empty()) return false;
    errno = 0;
    char * end = nullptr;
    const long v = std::strtol(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0') return false;
    if (v < -2147483647L - 1L || v > 2147483647L) return false;
    out = (int)v;
    return true;
}

void set_err(std::string * err, const std::string & msg) {
    if (err) *err = msg;
}

bool ends_with(const std::string & s, const char * suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// True when the variable is set to a non-empty value other than "0".
bool env_flag_set(const char * name) {
    const char * v = std::getenv(name);
    if (!v || v[0] == '\0') return false;
    return std::strcmp(v, "0") != 0;
}

// setenv only when absent; returns 1 when a variable was written.
int export_if_absent(const char * name, const std::string & value) {
    if (std::getenv(name) != nullptr) return 0;
    if (value.empty()) return 0;
    return common::set_environment_variable(name, value.c_str(), false) == 0 ? 1 : 0;
}

}  // namespace

// ─── ClusterConfig ──────────────────────────────────────────────────────

std::string ClusterConfig::validate() const {
    if (!enabled()) {
        return {};
    }
    if (size < kClusterMinSize || size > kClusterMaxSize) {
        return "--cluster-size must be in [" + std::to_string(kClusterMinSize) + ", " +
               std::to_string(kClusterMaxSize) + "], got " + std::to_string(size);
    }
    if (rank < 0 || rank >= size) {
        return "--cluster-rank must be in [0, " + std::to_string(size) + "), got " +
               std::to_string(rank);
    }
    if (head_host.empty()) {
        return "--cluster-head <host:port> is required in cluster mode";
    }
    if (head_port <= 0 || head_port > 65535) {
        return "--cluster-head port must be in [1, 65535], got " + std::to_string(head_port);
    }
    if (gid_index < -1) {
        return "--cluster-gid-index must be non-negative, got " + std::to_string(gid_index);
    }
    if (replicate_hot < 0) {
        return "--cluster-replicate-hot must be non-negative, got " + std::to_string(replicate_hot);
    }
    if (placement_source == PlacementSource::File && placement_file.empty()) {
        return "--cluster-expert-placement <file.json> requires a file path";
    }
    if (placement_source != PlacementSource::File && !placement_file.empty()) {
        return "internal: placement file set without a File placement source";
    }
    if (timeout_ms == 0) {
        return "--cluster-timeout-ms must be positive";
    }
    if (verify_hash_every < 0) {
        return "--cluster-verify-hash must be non-negative, got " + std::to_string(verify_hash_every);
    }
    return {};
}

// ─── Parsing helpers ────────────────────────────────────────────────────

bool parse_host_port(const std::string & text, std::string & host, int & port, std::string * err) {
    if (text.empty()) {
        set_err(err, "expected host:port, got an empty string");
        return false;
    }

    std::string host_part;
    std::string port_part;
    bool has_port = false;
    if (text[0] == '[') {
        // Bracketed IPv6 literal: [addr]:port or [addr]
        const size_t close = text.find(']');
        if (close == std::string::npos) {
            set_err(err, "unterminated '[' in host:port '" + text + "'");
            return false;
        }
        host_part = text.substr(1, close - 1);
        if (close + 1 < text.size()) {
            if (text[close + 1] != ':') {
                set_err(err, "expected ':' after ']' in '" + text + "'");
                return false;
            }
            port_part = text.substr(close + 2);
            has_port = true;
        }
    } else {
        const size_t colon = text.rfind(':');
        if (colon != std::string::npos && text.find(':') != colon) {
            // More than one ':' without brackets: a bare IPv6 address. Only
            // accepted when unambiguous (no port).
            host_part = text;
        } else if (colon != std::string::npos) {
            host_part = text.substr(0, colon);
            port_part = text.substr(colon + 1);
            has_port = true;
        } else {
            host_part = text;
        }
    }

    if (host_part.empty()) {
        set_err(err, "empty host in '" + text + "'");
        return false;
    }
    if (has_port && port_part.empty()) {
        set_err(err, "missing port after ':' in '" + text + "'");
        return false;
    }

    int p = kClusterDefaultPort;
    if (has_port) {
        if (!parse_int_strict(port_part, p) || p <= 0 || p > 65535) {
            set_err(err, "bad port '" + port_part + "' in '" + text + "' (expected 1..65535)");
            return false;
        }
    }

    host = host_part;
    port = p;
    return true;
}

bool parse_shared_expert_mode(const std::string & text, SharedExpertMode & out, std::string * err) {
    if (text == "replicate") {
        out = SharedExpertMode::Replicate;
        return true;
    }
    if (text == "shard") {
        out = SharedExpertMode::Shard;
        return true;
    }
    if (text == "rank0") {
        out = SharedExpertMode::Rank0;
        return true;
    }
    set_err(err, "--cluster-shared-expert expects replicate, shard, or rank0, got '" + text + "'");
    return false;
}

bool parse_allreduce_dtype(const std::string & text, AllreduceDType & out, std::string * err) {
    if (text == "f32") {
        out = AllreduceDType::F32;
        return true;
    }
    if (text == "bf16") {
        out = AllreduceDType::BF16;
        return true;
    }
    if (text == "auto") {
        out = AllreduceDType::Auto;
        return true;
    }
    set_err(err, "--cluster-allreduce-dtype expects f32, bf16, or auto, got '" + text + "'");
    return false;
}

bool parse_placement_source(const std::string & text, PlacementSource & out,
                            std::string & file, std::string * err) {
    if (text == "uniform") {
        out = PlacementSource::Uniform;
        file.clear();
        return true;
    }
    if (text == "balanced") {
        out = PlacementSource::Balanced;
        file.clear();
        return true;
    }
    if (ends_with(text, ".json") && text.size() > 5) {
        out = PlacementSource::File;
        file = text;
        return true;
    }
    set_err(err, "--cluster-expert-placement expects uniform, balanced, or a path ending in "
                 ".json, got '" + text + "'");
    return false;
}

// ─── Names ──────────────────────────────────────────────────────────────

const char * shared_expert_mode_name(SharedExpertMode mode) {
    switch (mode) {
        case SharedExpertMode::Replicate: return "replicate";
        case SharedExpertMode::Shard:     return "shard";
        case SharedExpertMode::Rank0:     return "rank0";
    }
    return "unknown";
}

const char * allreduce_dtype_name(AllreduceDType dtype) {
    switch (dtype) {
        case AllreduceDType::F32:  return "f32";
        case AllreduceDType::BF16: return "bf16";
        case AllreduceDType::Auto: return "auto";
    }
    return "unknown";
}

const char * placement_source_name(PlacementSource source) {
    switch (source) {
        case PlacementSource::Uniform:  return "uniform";
        case PlacementSource::Balanced: return "balanced";
        case PlacementSource::File:     return "file";
    }
    return "unknown";
}

// ─── Environment ────────────────────────────────────────────────────────

int export_rccl_environment(const ClusterConfig & cfg) {
    int n = 0;

    // gfx1151 has no GPUDirect RDMA: RCCL must bounce through host memory.
    n += export_if_absent("NCCL_NET_GDR_LEVEL", "0");

    // Operator-supplied transport hints.
    n += export_if_absent("NCCL_SOCKET_IFNAME", cfg.ifname);
    n += export_if_absent("NCCL_IB_HCA", cfg.ib_hca);
    if (cfg.gid_index >= 0) {
        n += export_if_absent("NCCL_IB_GID_INDEX", std::to_string(cfg.gid_index));
    }

    // Hardening defaults measured on the 4-node Strix Halo RoCE v2 cluster.
    n += export_if_absent("NCCL_IB_DISABLE", "0");
    n += export_if_absent("NCCL_ASYNC_ERROR_HANDLING", "1");
    n += export_if_absent("NCCL_IB_QPS_PER_CONNECTION", "2");
    n += export_if_absent("NCCL_IB_TIMEOUT", "22");
    n += export_if_absent("NCCL_IB_RETRY_CNT", "7");
    // On the R9700, RCCL runs reliably only with the Simple protocol.
    n += export_if_absent("NCCL_PROTO", "Simple");

    return n;
}

bool cluster_env_no_ingraph_allreduce() {
    return env_flag_set("LUCE_CLUSTER_NO_INGRAPH_ALLREDUCE");
}

bool cluster_env_no_graph_capture() {
    return env_flag_set("LUCE_CLUSTER_NO_GRAPH_CAPTURE");
}

bool cluster_env_trace() {
    return env_flag_set("LUCE_CLUSTER_TRACE");
}

}  // namespace luce::cluster
