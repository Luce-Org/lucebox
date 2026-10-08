// cluster_identity.h - the HelloMsg identity every rank presents.
//
// Head and workers must compute these fields with the same code, otherwise
// the head's accept_workers() refuses ranks that are in fact identical.
// Header-only so server_main.cpp (selftest), ClusterHeadBackend and
// run_cluster_worker share one definition without a link dependency.
//
//   build_sha      : LUCE_BUILD_SHA compile-time define (CMake sets it
//                    from `git describe` when LUCE_CLUSTER=ON), else
//                    "unknown" - then only model and placement are compared.
//   model_sha      : fnv1a64 hex of the model path. All ranks must load the
//                    same file from the same path (the launcher enforces the
//                    same mount layout); hashing the 98 GB file itself at
//                    startup is not an option.
//   placement_hash : ClusterExpertPlacement::hash() of the rank's built
//                    placement, or 0 before a placement exists (selftest).

#pragma once

#include "cluster/cluster_control.h"
#include "cluster/cluster_protocol.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace luce::cluster {

inline std::string cluster_build_sha() {
#ifdef LUCE_BUILD_SHA
    return LUCE_BUILD_SHA;
#else
    return "unknown";
#endif
}

inline std::string cluster_model_sha(const std::string & model_path) {
    char hex[32];
    std::snprintf(hex, sizeof(hex), "%016llx",
                  (unsigned long long) fnv1a64(model_path.data(), model_path.size()));
    return hex;
}

inline HelloMsg cluster_identity(int rank, int size, const std::string & model_path,
                                 uint64_t placement_hash) {
    HelloMsg h;
    h.rank = rank;
    h.size = size;
    h.protocol_version = kProtocolVersion;
    h.build_sha = cluster_build_sha();
    h.model_sha = cluster_model_sha(model_path);
    h.placement_hash = placement_hash;
    h.hostname = local_hostname(rank);
    return h;
}

}  // namespace luce::cluster
