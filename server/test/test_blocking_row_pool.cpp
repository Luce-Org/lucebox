#include "common/blocking_row_pool.h"
#include "CppUnitTestFramework.hpp"
#include <array>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

using dflash::common::BlockingRowPool;
using namespace CppUnitTestFramework;
struct BlockingRowPoolFixture : CommonFixture {
    using CommonFixture::CommonFixture;
};
static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static int iterations(int normal, int stress) {
    // CI may expose only two CPUs: oversubscribed spinning workers make a
    // 100k-job soak exceed its timeout. The deterministic generation test
    // below proves the race independently of iteration count. Keep the full
    // soak available for sanitizer runs and dedicated hosts.
    return std::getenv("DFLASH_ROW_POOL_STRESS") ? stress : normal;
}

static void mixed_widths(unsigned workers) {
    BlockingRowPool pool(workers);
    // Changing the active set exposes late inactive workers: they must not
    // execute a new callback under a stale generation or acknowledge it twice.
    for (int step = 0; step < iterations(256, 25000); ++step) {
        const int rows = step % 3 == 0 ? 4 : step % 3 == 1 ? 24 : 1;
        std::array<std::atomic<int>, 24> hits{};
        pool.run_custom(rows, [&](int row) {
            hits[(size_t) row].fetch_add(1, std::memory_order_relaxed);
        });
        for (int row = 0; row < 24; ++row) {
            check(hits[(size_t) row].load() == (row < rows ? 1 : 0),
                  "job returned before exactly-once completion");
        }
    }
}

static void concurrent_clients() {
    BlockingRowPool pool(8);
    std::atomic<int> errors{0};
    std::vector<std::thread> clients;
    for (int client = 0; client < 4; ++client) {
        clients.emplace_back([&, client] {
            for (int step = 0; step < iterations(128, 1000); ++step) {
                std::array<int, 24> output{};
                pool.run_chunks(24, [&](int begin, int end) {
                    for (int row = begin; row < end; ++row) output[row] = row + client + step;
                });
                for (int row = 0; row < 24; ++row) {
                    if (output[row] != row + client + step) ++errors;
                }
            }
        });
    }
    for (auto & client : clients) client.join();
    check(errors == 0, "concurrent clients mixed jobs");
}

static void boundaries_and_idle() {
    bool rejected = false;
    try { BlockingRowPool invalid(0); } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "zero workers accepted");
    BlockingRowPool pool(8);
    int calls = 0;
    pool.run_custom(0, [&](int) { ++calls; });
    pool.run_custom(-1, [&](int) { ++calls; });
    check(calls == 0, "empty job called callback");
    std::atomic<int64_t> total{0};
    pool.run_chunks(INT_MAX, [&](int begin, int end) {
        total.fetch_add((int64_t) end - begin);
    });
    check(total == INT_MAX, "large row partition overflowed");
    std::atomic<bool> wrong_row{false};
    for (int step = 0; step < 4; ++step) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        pool.run_custom(1, [&](int row) {
            if (row != 0) wrong_row.store(true, std::memory_order_relaxed);
            ++calls;
        });
    }
    check(!wrong_row.load(), "wrong row");
    check(calls == 4, "idle workers missed wakeup");
}

TEST_CASE(BlockingRowPoolFixture, mixed_widths_complete_exactly_once) {
    for (unsigned workers : {1u, 2u, 4u, 8u}) mixed_widths(workers);
}

TEST_CASE(BlockingRowPoolFixture, concurrent_clients_do_not_mix_jobs) {
    concurrent_clients();
}

TEST_CASE(BlockingRowPoolFixture, boundaries_and_idle_wakeup) {
    boundaries_and_idle();
}

TEST_CASE(BlockingRowPoolFixture, worker_count_preserves_unknown_hardware_fallback) {
    CHECK(BlockingRowPool::default_worker_count(0) == 4);
    CHECK(BlockingRowPool::default_worker_count(1) == 1);
    CHECK(BlockingRowPool::default_worker_count(4) == 4);
    CHECK(BlockingRowPool::default_worker_count(32) == 8);
}

TEST_CASE(BlockingRowPoolFixture, delayed_inactive_worker_holds_generation_open) {
    struct Gate {
        std::mutex mutex;
        std::condition_variable cv;
        unsigned acknowledged = 0;
        bool stalled = false;
        bool release = false;
    } gate;
    const auto hook = [](void * context, unsigned worker, bool after_ack) {
        auto & gate = *static_cast<Gate *>(context);
        std::unique_lock<std::mutex> lock(gate.mutex);
        if (after_ack) {
            ++gate.acknowledged;
            gate.cv.notify_all();
        } else if (worker == 7) {
            gate.stalled = true;
            gate.cv.notify_all();
            gate.cv.wait(lock, [&] { return gate.release; });
        }
    };
    BlockingRowPool pool(8, hook, &gate);
    std::atomic<int> rows{0};
    std::atomic<bool> returned{false};
    std::thread client([&] {
        pool.run_custom(1, [&](int) { ++rows; });
        returned = true;
    });
    bool reached = false, held = false;
    {
        std::unique_lock<std::mutex> lock(gate.mutex);
        reached = gate.cv.wait_for(lock, std::chrono::seconds(5), [&] {
            return gate.stalled && gate.acknowledged == 7;
        });
        // All other workers finished. This inactive worker has observed the
        // generation but has not read its borrowed job: completion is unsafe.
        held = pool.pending_workers_for_test() == 1 && !returned.load();
        gate.release = true;
    }
    gate.cv.notify_all();
    client.join();
    CHECK(reached);
    CHECK(held);
    CHECK(rows == 1);
    CHECK(returned);
}
