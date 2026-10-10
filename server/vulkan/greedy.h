#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace luce::common {
// Exact finite-only greedy selection. Equal logits choose the lowest token ID,
// like std::max_element. No reassociation of model arithmetic or quantization.
inline int32_t finite_argmax(const float *x, int n) {
    if (n <= 0) throw std::runtime_error("empty logits");
    int i = 0, best_index = 0;
    float best = -std::numeric_limits<float>::infinity();
#if defined(__SSE2__)
    __m128 values = _mm_set1_ps(best);
    __m128i indices = _mm_setr_epi32(0, 1, 2, 3);
    __m128i current = indices;
    const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    const __m128 infinity = _mm_set1_ps(std::numeric_limits<float>::infinity());
    for (; i + 4 <= n; i += 4) {
        const __m128 v = _mm_loadu_ps(x + i);
        if (_mm_movemask_ps(_mm_cmplt_ps(_mm_and_ps(v, abs_mask), infinity)) != 15)
            throw std::runtime_error("nonfinite logits");
        const __m128 greater = _mm_cmpgt_ps(v, values);
        indices = _mm_or_si128(_mm_and_si128(_mm_castps_si128(greater), current),
                              _mm_andnot_si128(_mm_castps_si128(greater), indices));
        values = _mm_max_ps(values, v);
        current = _mm_add_epi32(current, _mm_set1_epi32(4));
    }
    alignas(16) float lane_values[4];
    alignas(16) int32_t lane_indices[4];
    _mm_store_ps(lane_values, values);
    _mm_store_si128(reinterpret_cast<__m128i *>(lane_indices), indices);
    for (int lane = 0; lane < 4; ++lane) {
        if (lane_values[lane] > best ||
            (lane_values[lane] == best && lane_indices[lane] < best_index)) {
            best = lane_values[lane]; best_index = lane_indices[lane];
        }
    }
#endif
    for (; i < n; ++i) {
        if (!std::isfinite(x[i])) throw std::runtime_error("nonfinite logits");
        if (x[i] > best) { best = x[i]; best_index = i; }
    }
    return best_index;
}
}
