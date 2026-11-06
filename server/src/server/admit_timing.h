// The three durations of a request the slot scheduler admits late: the wait for admission (queue), the prefill, the total.
// Pure: timestamps in, seconds out (the scheduler stamps them; tests pass synthetic ones).
#pragma once

#include <chrono>

namespace luce::common {

struct AdmitTiming {
    double queue_s = 0.0;     // first admit attempt -> successful admit (Deferred retries)
    double prefill_s = 0.0;   // successful admit -> prefill complete
    double total_s = 0.0;     // first admit attempt -> prefill complete
};

inline AdmitTiming admit_timing(std::chrono::steady_clock::time_point first_attempt,
                                std::chrono::steady_clock::time_point admitted,
                                std::chrono::steady_clock::time_point prefill_end) {
    using sec = std::chrono::duration<double>;
    AdmitTiming t;
    t.total_s = sec(prefill_end - first_attempt).count();
    t.queue_s = sec(admitted - first_attempt).count();
    t.prefill_s = sec(prefill_end - admitted).count();
    return t;
}

}  // namespace luce::common
