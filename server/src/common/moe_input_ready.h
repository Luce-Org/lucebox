#pragma once

#include <exception>
#include <future>
#include <mutex>
#include <utility>

namespace dflash::common {

class MoeInputReady {
public:
    explicit MoeInputReady(bool enabled)
        : enabled_(enabled),
          future_(enabled ? promise_.get_future() : std::future<void>()) {
    }

    void signal() {
        if (!enabled_) {
            return;
        }
        std::call_once(signaled_, [&]() { promise_.set_value(); });
    }

    void set_exception(std::exception_ptr exception) {
        if (!enabled_) {
            return;
        }
        std::call_once(signaled_, [&]() { promise_.set_exception(exception); });
    }

    void wait() {
        if (enabled_) {
            future_.get();
        }
    }

private:
    bool enabled_;
    std::promise<void> promise_;
    std::future<void> future_;
    std::once_flag signaled_;
};

template <typename Work>
std::future<bool> launch_moe_input_ready_worker(MoeInputReady & input_ready, Work work) {
    return std::async(
        std::launch::async,
        [&input_ready, work = std::move(work)]() mutable {
            try {
                const bool ok = work();
                input_ready.signal();
                return ok;
            } catch (...) {
                input_ready.set_exception(std::current_exception());
                throw;
            }
        });
}

}  // namespace dflash::common
