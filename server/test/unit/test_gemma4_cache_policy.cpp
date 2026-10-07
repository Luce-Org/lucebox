#include "gemma4/gemma4_internal.h"
#include "host_check.h"

#include <cstdio>
#include <vector>

using namespace luce::common;

static int g_checks = 0;

static void check_capacity_and_bounds() {
    CHECK(gemma4_swa_capacity(1024, 9216, 16) == 1280);
    CHECK(gemma4_swa_capacity(1024, 9216, 512) == 1536);
    CHECK(gemma4_swa_capacity(1024, 1000, 512) == 1024);
    CHECK(gemma4_swa_capacity(1024, 4096, 0) == 0);
    CHECK(!gemma4_cache_span_fits(1024, 1536, 4096, 1500, 513));
    CHECK(!gemma4_cache_span_fits(1024, 1536, 4096, -1, 16));
    CHECK(!gemma4_cache_span_fits(1024, 1536, 4096, 4090, 16));
}

static void check_retained_history() {
    // Store absolute token positions instead of K/V values. Check that every
    // query and every rejection cut can still find the history it needs.
    for (int window : {4, 127, 256, 1024}) {
        for (int width : {1, 2, 16, 128, 512}) {
            const int max_ctx = 8192;
            const int capacity = gemma4_swa_capacity(window, max_ctx, width);
            for (int base : {0, window - 1, window, capacity - 8, capacity - 1,
                             capacity, capacity + 5, 2 * capacity - 8, 2 * capacity + 5}) {
                CHECK(gemma4_cache_span_fits(window, capacity, max_ctx, base, width));
                std::vector<int> rows(capacity, -1);
                for (int p = 0; p < base; ++p) rows[p % capacity] = p;
                for (int p = base; p < base + width; ++p) rows[p % capacity] = p;
                for (int q = base; q < base + width; ++q) {
                    for (int k = std::max(0, q - window + 1); k <= q; ++k) {
                        CHECK(rows[k % capacity] == k);
                    }
                }
                for (int accepted = 0; accepted <= width; ++accepted) {
                    const int next = base + accepted;
                    for (int k = std::max(0, next - window); k < next; ++k) {
                        CHECK(rows[k % capacity] == k);
                    }
                }
            }
        }
    }
}

static void check_layer_split_chunks() {
    // An override or IPC request can exceed the headroom of a particular shard.
    CHECK(gemma4_swa_chunk_size(1024, 1536, 8192, 1536, 1024) == 512);
    CHECK(gemma4_swa_chunk_size(1024, 2048, 8192, 2048, 1024) == 1024);
    CHECK(gemma4_swa_chunk_size(1024, 1536, 8192, 1528, 1024) == 8);
    CHECK(gemma4_swa_chunk_size(1024, 1024, 1000, 990, 512) == 10);
    CHECK(gemma4_swa_chunk_size(1024, 1536, 8192, -1, 512) == 0);
    CHECK(gemma4_swa_chunk_size(1024, 1536, 8192, 8192, 512) == 0);
    CHECK(gemma4_swa_chunk_size(1024, 1536, 8192, 0, 0) == 0);
    CHECK(gemma4_swa_chunk_size(1024, 1024, 8192, 0, 512) == 0);

    // Process a long request using smaller safe forwards. Check the history
    // needed by every query after each chunk's writes, including wraparound.
    const int window = 1024;
    const int max_ctx = 16384;
    for (int reserved : {128, 512, 1024}) {
        const int capacity = gemma4_swa_capacity(window, max_ctx, reserved);
        for (int requested : {16, 512, 1024, 2048}) {
            const int begin = capacity - 8;
            const int end = begin + 2 * capacity;
            std::vector<int> rows(capacity, -1);
            for (int p = 0; p < begin; ++p) rows[p % capacity] = p;
            for (int start = begin; start < end;) {
                const int count = gemma4_swa_chunk_size(
                    window, capacity, max_ctx, start, std::min(requested, end - start));
                CHECK(count > 0);
                CHECK(gemma4_cache_span_fits(window, capacity, max_ctx, start, count));
                for (int p = start; p < start + count; ++p) rows[p % capacity] = p;
                for (int q = start; q < start + count; ++q) {
                    for (int k = q - window + 1; k <= q; ++k) {
                        CHECK(rows[k % capacity] == k);
                    }
                }
                start += count;
            }
        }
    }
}

int main() {
    check_capacity_and_bounds();
    check_retained_history();
    check_layer_split_chunks();
    std::printf("Gemma4 cache policy tests passed: %d checks\n", g_checks);
}
