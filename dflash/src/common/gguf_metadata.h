// common/gguf_metadata.h — Shared helpers for reading GGUF metadata.
//
// Provides typed "get or default" accessors and "require" accessors for
// gguf_context key-value pairs, plus an architecture validation helper.
// Use these in every loader; do not inline equivalent helpers per-arch.
//
// Include convention: #include "common/gguf_metadata.h"
// Never: ../common/gguf_metadata.h or absolute paths.

#pragma once

#include "gguf.h"

#include <cstdint>
#include <string>

namespace dflash::common {

// ── Read-with-default ─────────────────────────────────────────────────────
// Return the stored value when the key is present, default_val otherwise.

inline uint32_t gguf_get_u32_or(struct gguf_context * gguf, const char * key, uint32_t default_val) {
    int64_t id = gguf_find_key(gguf, key);
    return (id >= 0) ? gguf_get_val_u32(gguf, id) : default_val;
}

inline int32_t gguf_get_i32_or(struct gguf_context * gguf, const char * key, int32_t default_val) {
    int64_t id = gguf_find_key(gguf, key);
    return (id >= 0) ? gguf_get_val_i32(gguf, id) : default_val;
}

inline float gguf_get_f32_or(struct gguf_context * gguf, const char * key, float default_val) {
    int64_t id = gguf_find_key(gguf, key);
    return (id >= 0) ? gguf_get_val_f32(gguf, id) : default_val;
}

inline std::string gguf_get_str_or(struct gguf_context * gguf, const char * key, const std::string & default_val) {
    int64_t id = gguf_find_key(gguf, key);
    return (id >= 0) ? std::string(gguf_get_val_str(gguf, id)) : default_val;
}

// ── Required reads ────────────────────────────────────────────────────────
// Return false and write a descriptive error when the key is absent.

inline bool gguf_require_u32(struct gguf_context * gguf, const char * key,
                              uint32_t & out, std::string & out_error) {
    int64_t id = gguf_find_key(gguf, key);
    if (id < 0) {
        out_error = std::string("missing required GGUF key: ") + key;
        return false;
    }
    out = gguf_get_val_u32(gguf, id);
    return true;
}

inline bool gguf_require_str(struct gguf_context * gguf, const char * key,
                              std::string & out, std::string & out_error) {
    int64_t id = gguf_find_key(gguf, key);
    if (id < 0) {
        out_error = std::string("missing required GGUF key: ") + key;
        return false;
    }
    out = gguf_get_val_str(gguf, id);
    return true;
}

// ── Architecture validation ───────────────────────────────────────────────
// Return true when "general.architecture" equals expected_arch.
// On mismatch or absence, writes a descriptive error and returns false.

inline bool gguf_check_architecture(struct gguf_context * gguf,
                                    const char * expected_arch,
                                    std::string & out_error) {
    int64_t id = gguf_find_key(gguf, "general.architecture");
    if (id < 0) {
        out_error = "missing required GGUF key: general.architecture";
        return false;
    }
    const char * arch = gguf_get_val_str(gguf, id);
    if (std::string(arch) != expected_arch) {
        out_error = std::string("unexpected architecture: got '") + arch
                    + "', expected '" + expected_arch + "'";
        return false;
    }
    return true;
}

} // namespace dflash::common
