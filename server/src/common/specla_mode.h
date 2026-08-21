// SpecLA runtime mode (docs/SPECLA.md, arXiv:2607.16673).
//
// DFLASH_SPECLA=1 switches the single-target qwen35 speculative verifier to
// the paper's state-resident path.  A heavy-light schedule runs adjacent tree
// nodes while the GDN and convolution state tiles are live, records raw
// per-node factors, and applies the previously accepted factor buffer at the
// beginning of the next verification.  Prefill and ordinary AR decode retain
// their normal state writebacks.
#pragma once

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>

namespace dflash::common {

inline bool specla_enabled() {
    static const bool on = []() {
        const char * v = std::getenv("DFLASH_SPECLA");
        return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
    }();
    return on;
}

// Section 6.2 assumes a tiny, trained EAGLE-style draft layer that can roll a
// beam out prefix by prefix.  Qwen's current DFlash draft is a substantially
// larger block-diffusion model; rerunning all five layers for every expanded
// node is useful for algorithm experiments but normally costs more than it
// saves.  Keep the exact branch-conditioned builder available without making
// it the production default for this model.
inline bool specla_conditional_draft_enabled() {
    static const bool on = []() {
        const char * v = std::getenv("DFLASH_SPECLA_CONDITIONAL_DRAFT");
        return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
    }();
    return on;
}

inline int specla_tree_topk() {
    static const int topk = []() {
        const char * v = std::getenv("DFLASH_SPECLA_TOPK");
        if (!v || v[0] == '\0') return 4;  // paper's end-to-end setting
        char * end = nullptr;
        errno = 0;
        const long parsed = std::strtol(v, &end, 10);
        if (errno == ERANGE || end == v || *end != '\0' ||
            parsed <= 0 || parsed > INT_MAX) {
            return 4;  // malformed/out-of-range → documented default
        }
        return static_cast<int>(parsed);
    }();
    return topk;
}

}  // namespace dflash::common
