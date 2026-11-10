// The two durations of a request the slot scheduler admits late: the wait for admission (queue), the prefill.
// Pure: timestamps in, seconds out (the scheduler stamps them; tests pass synthetic ones).
#pragma once

#include <chrono>

namespace luce::common {

struct AdmitTiming {
    double queue_s = 0.0;     // first admit attempt -> successful admit (Deferred retries)
    double prefill_s = 0.0;   // successful admit -> prefill complete
};

inline AdmitTiming admit_timing(std::chrono::steady_clock::time_point first_attempt,
                                std::chrono::steady_clock::time_point admitted,
                                std::chrono::steady_clock::time_point prefill_end) {
    using sec = std::chrono::duration<double>;
    AdmitTiming t;
    t.queue_s = sec(admitted - first_attempt).count();
    t.prefill_s = sec(prefill_end - admitted).count();
    return t;
}

// The prompt tokens an admitted request does not prefill: the prefix cache's restored tokens, or the point a parked session resumed at.
inline int admit_cached_tokens(int prefix_restored, int engine_resumed) {
    return prefix_restored > engine_resumed ? prefix_restored : engine_resumed;
}

}  // namespace luce::common
