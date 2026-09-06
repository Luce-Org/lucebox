#include "common/moe_input_ready.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace dflash::common;

static constexpr auto kTimeout = std::chrono::seconds(2);

static bool wait_bounded(std::future<bool> & result, const char * scenario) {
    if (result.wait_for(kTimeout) != std::future_status::ready) {
        std::fprintf(stderr, "FAIL: %s timed out\n", scenario);
        std::_Exit(1);
    }
    return result.get();
}

static bool test_exception_before_ready() {
    MoeInputReady input_ready(true);
    auto cold = launch_moe_input_ready_worker(input_ready, []() -> bool {
        throw std::runtime_error("cold worker failed before input readiness");
    });
    auto caller = std::async(std::launch::async, [&]() {
        try {
            input_ready.wait();
        } catch (const std::runtime_error & exception) {
            return std::string(exception.what()) ==
                   "cold worker failed before input readiness";
        }
        return false;
    });

    const bool caller_received_exception =
        wait_bounded(caller, "exception-before-ready caller");
    bool worker_retained_exception = false;
    try {
        cold.get();
    } catch (const std::runtime_error & exception) {
        worker_retained_exception =
            std::string(exception.what()) ==
            "cold worker failed before input readiness";
    }
    return caller_received_exception && worker_retained_exception;
}

static bool test_success() {
    MoeInputReady input_ready(true);
    std::atomic<bool> cold_work_completed{false};
    auto cold = launch_moe_input_ready_worker(input_ready, [&]() {
        std::vector<std::thread> signalers;
        for (int i = 0; i < 8; ++i) {
            signalers.emplace_back([&]() { input_ready.signal(); });
        }
        for (std::thread & signaler : signalers) {
            signaler.join();
        }
        cold_work_completed.store(true);
        return true;
    });
    auto caller = std::async(std::launch::async, [&]() {
        input_ready.wait();
        return true;
    });

    const bool caller_released = wait_bounded(caller, "successful caller");
    const bool worker_completed = cold.get();
    return caller_released && worker_completed && cold_work_completed.load();
}

int main() {
    if (!test_exception_before_ready()) {
        std::fprintf(stderr, "FAIL: exception was not propagated on the pre-ready path\n");
        return 1;
    }
    if (!test_success()) {
        std::fprintf(stderr, "FAIL: successful readiness path did not complete\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
