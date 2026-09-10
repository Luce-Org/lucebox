#pragma once
#include <algorithm>
#include <cstdint>
#include <exception>
#include <future>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

// Internal scheduling only: no source, calibration, codec or FILE state lives here.
namespace ds4_mix_detail {
using EncodedExpert = std::vector<uint8_t>;
inline size_t checked_encoded_size(uint64_t row_bytes, uint64_t rows, unsigned workers) {
    if (workers < 1 || workers > 16) throw std::runtime_error("encode threads must be in 1..16");
    if (!row_bytes || !rows || row_bytes > std::numeric_limits<uint64_t>::max()/rows)
        throw std::runtime_error("invalid or overflowing encoded expert size");
    const uint64_t bytes = row_bytes*rows;
    if (bytes > std::numeric_limits<uint64_t>::max()/workers ||
        bytes*workers > std::numeric_limits<size_t>::max())
        throw std::runtime_error("overflowing encoded batch size");
    return static_cast<size_t>(bytes);
}
struct AsyncExpert {
    template<class Task> auto operator()(Task task) const {
        return std::async(std::launch::async, std::move(task));
    }
};
// Launch is injectable only to test failure after some tasks have started.
// All launched futures are explicitly consumed before any captured owner can die.
template<class Encode, class Write, class Launch = AsyncExpert>
void ordered_expert_batches(uint32_t count, unsigned workers, size_t expert_bytes,
                            Encode encode, Write write, Launch launch = {}) {
    checked_encoded_size(expert_bytes, 1, workers);
    auto publish = [&](uint32_t expert, const EncodedExpert & bytes) {
        if (bytes.size() != expert_bytes) throw std::runtime_error("encoded expert byte count mismatch");
        write(expert, bytes);
    };
    if (workers == 1) {
        for (uint32_t expert = 0; expert < count; ++expert) publish(expert, encode(expert));
        return;
    }
    for (uint32_t first = 0; first < count;) {
        const unsigned batch = std::min<uint32_t>(workers, count - first);
        std::vector<std::future<EncodedExpert>> futures;
        futures.reserve(batch); // Allocation precedes all launches.
        std::exception_ptr failure;
        try {
            for (unsigned i = 0; i < batch; ++i) {
                const uint32_t expert = first + i;
                futures.push_back(launch([&encode, expert] { return encode(expert); }));
            }
        } catch (...) { failure = std::current_exception(); }
        for (unsigned i = 0; i < futures.size(); ++i) {
            try {
                EncodedExpert bytes = futures[i].get();
                if (!failure) publish(first + i, bytes);
            } catch (...) { if (!failure) failure = std::current_exception(); }
        }
        if (failure) std::rethrow_exception(failure);
        first += batch; // No next batch until every future was drained.
    }
}
} // namespace ds4_mix_detail
