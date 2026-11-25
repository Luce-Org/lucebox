// Shared GPU cache snapshots for independent-slot/solo differential tests.
#pragma once
#include "qwen4exp_cache.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace qwen4exp_test {
using namespace luce::common;
struct SavedTensor { ggml_tensor * tensor; std::vector<uint8_t> bytes; };
struct SavedCache {
    std::vector<SavedTensor> tensors;
    std::vector<int32_t> ple_prev;
    int indexer_blocks = 0, cur_pos = 0;
};

inline std::vector<ggml_tensor *> state_tensors(Qwen4ExpCache & c) {
    std::vector<ggml_tensor *> out;
    auto add = [&](const std::vector<ggml_tensor *> & v) { out.insert(out.end(), v.begin(), v.end()); };
    add(c.attn_k); add(c.attn_v); add(c.indexer_k); add(c.indexer_raw); add(c.ssm_state);
    add(c.conv_state); add(c.ple_conv_state);
    out.erase(std::remove(out.begin(), out.end(), nullptr), out.end());
    return out;
}
inline SavedCache save_cache(Qwen4ExpCache & c) {
    SavedCache out; out.ple_prev = c.ple_prev; out.indexer_blocks = c.indexer_blocks; out.cur_pos = c.cur_pos;
    for (ggml_tensor * t : state_tensors(c)) {
        SavedTensor v{t, std::vector<uint8_t>(ggml_nbytes(t))};
        ggml_backend_tensor_get(t, v.bytes.data(), 0, v.bytes.size());
        out.tensors.push_back(std::move(v));
    }
    return out;
}
inline void restore_cache(Qwen4ExpCache & c, const SavedCache & in) {
    for (const SavedTensor & v : in.tensors)
        ggml_backend_tensor_set(v.tensor, v.bytes.data(), 0, v.bytes.size());
    c.ple_prev = in.ple_prev; c.indexer_blocks = in.indexer_blocks; c.cur_pos = in.cur_pos;
}
inline bool equal_cache(Qwen4ExpCache & c, const SavedCache & in) {
    if (c.ple_prev != in.ple_prev || c.indexer_blocks != in.indexer_blocks || c.cur_pos != in.cur_pos) return false;
    for (const SavedTensor & v : in.tensors) {
        std::vector<uint8_t> now(v.bytes.size());
        ggml_backend_tensor_get(v.tensor, now.data(), 0, now.size());
        if (now != v.bytes) return false;
    }
    return true;
}

// Poison raw and pooled capacity so the written-row mask is checked exactly,
// including after reset/reuse. Finite (0x3f3f3f3f = 0.747f): stable QSA decode
// scores the masked bucket suffix, which reused caches hold as stale finite keys.
constexpr uint8_t kIndexerPoison = 0x3f;
constexpr uint32_t kIndexerPoisonBits = 0x3f3f3f3fu;
inline void poison_indexer(Qwen4ExpCache & c) {
    for (const auto * tensors : {&c.indexer_raw, &c.indexer_k})
        for (auto * t : *tensors)
            if (t) ggml_backend_tensor_memset(t, kIndexerPoison, 0, ggml_nbytes(t));
}
struct SavedIndexer {
    int blocks;
    std::vector<std::vector<float>> rows;
};
inline SavedIndexer save_indexer(Qwen4ExpCache & c, int end) {
    SavedIndexer saved{c.indexer_blocks, {}};
    for (const auto * tensors : {&c.indexer_raw, &c.indexer_k}) {
        for (auto * t : *tensors) {
            if (!t) continue;
            const int count = tensors == &c.indexer_raw ? end : c.indexer_blocks;
            saved.rows.emplace_back((size_t) t->ne[0] * count);
            auto & row = saved.rows.back();
            if (!row.empty()) ggml_backend_tensor_get(t, row.data(), 0, row.size() * sizeof(float));
        }
    }
    return saved;
}
inline bool equal_indexer(Qwen4ExpCache & c, const SavedIndexer & expected, int end) {
    if (c.indexer_blocks != expected.blocks) {
        std::fprintf(stderr, "[indexer] blocks solo=%d actual=%d\n", expected.blocks, c.indexer_blocks);
        return false;
    }
    size_t tensor = 0;
    for (const auto * tensors : {&c.indexer_raw, &c.indexer_k}) {
        for (auto * t : *tensors) {
            if (!t) continue;
            if (tensor >= expected.rows.size()) return false;
            const auto & b = expected.rows[tensor++];
            const size_t count = tensors == &c.indexer_raw ? end : c.indexer_blocks;
            const size_t width = (size_t) t->ne[0];
            if (b.size() != count * width) return false;
            std::vector<float> a((size_t) ggml_nelements(t));
            ggml_backend_tensor_get(t, a.data(), 0, ggml_nbytes(t));
            for (size_t i = 0; i < b.size(); ++i) {
                if (!std::isfinite(a[i]) || !std::isfinite(b[i]) ||
                    std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
                    std::fprintf(stderr, "[indexer] tensor=%zu row=%zu col=%zu solo=%g actual=%g\n",
                                 tensor - 1, i / width, i % width, b[i], a[i]);
                    return false;
                }
            }
            // indexer_k's last row is the stable decode graph's permanent scratch row.
            const size_t checked = tensors == &c.indexer_k ? a.size() - width : a.size();
            for (size_t i = b.size(); i < checked; ++i) {
                uint32_t bits;
                std::memcpy(&bits, &a[i], sizeof(bits));
                if (bits != kIndexerPoisonBits) { // no writes outside the solo prefix
                    std::fprintf(stderr, "[indexer] tensor=%zu write outside prefix row=%zu (end=%d blocks=%d)\n",
                                 tensor - 1, i / width, end, c.indexer_blocks);
                    return false;
                }
            }
        }
    }
    std::printf("[indexer] end=%d blocks=%d exact=1\n", end, c.indexer_blocks);
    return tensor == expected.rows.size();
}
} // namespace qwen4exp_test
