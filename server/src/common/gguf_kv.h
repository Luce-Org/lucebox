// common/gguf_kv.h — GGUF metadata reads with a fallback, and buffer alignment,
// shared by the native loaders (bailingmoe3, qwen4exp).
//
// Include convention: #include "common/gguf_kv.h"

#pragma once

#include "gguf.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace luce::common {

// A u32 key, or the first element of a UINT32 / INT32 array. Missing keys,
// empty or other-typed arrays and negative INT32 values return `fallback`.
inline uint32_t get_u32_or(const gguf_context * g, const std::string & key,
                           uint32_t fallback) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) return fallback;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(g, id) == 0) return fallback;
        const gguf_type type = gguf_get_arr_type(g, id);
        const void * data = gguf_get_arr_data(g, id);
        if (type == GGUF_TYPE_UINT32) return static_cast<const uint32_t *>(data)[0];
        if (type == GGUF_TYPE_INT32) {
            const int32_t value = static_cast<const int32_t *>(data)[0];
            return value < 0 ? fallback : static_cast<uint32_t>(value);
        }
        return fallback;
    }
    return gguf_get_val_u32(g, id);
}

// An f32 key, or the first element of a FLOAT32 array; `fallback` otherwise.
inline float get_f32_or(const gguf_context * g, const std::string & key,
                        float fallback) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) return fallback;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(g, id) == 0 ||
            gguf_get_arr_type(g, id) != GGUF_TYPE_FLOAT32) {
            return fallback;
        }
        return static_cast<const float *>(gguf_get_arr_data(g, id))[0];
    }
    return gguf_get_val_f32(g, id);
}

inline size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0) return value;
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

}  // namespace luce::common
