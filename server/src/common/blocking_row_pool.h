#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace dflash::common {

// Synchronous, non-reentrant row partitioning. Callbacks must not throw.
// The callback is borrowed only until run_chunks returns; no per-job allocation.
// Every worker acknowledges every generation, including workers without rows.
// Counting only active workers lets a delayed inactive worker observe a newer
// job under an older generation and execute it twice, outliving its caller.
class BlockingRowPool {
public:
#ifdef DFLASH_BLOCKING_ROW_POOL_TEST_HOOKS
    using TestHook = void (*)(void *, unsigned, bool);
    unsigned pending_workers_for_test() const {
        return remaining_.load(std::memory_order_acquire);
    }
#endif
    static constexpr unsigned default_worker_count(unsigned reported) {
        return reported ? std::min(8u, reported) : 4u;
    }
    explicit BlockingRowPool(unsigned count = default_worker_count(
            std::thread::hardware_concurrency())
#ifdef DFLASH_BLOCKING_ROW_POOL_TEST_HOOKS
            , TestHook hook = nullptr, void * hook_context = nullptr
#endif
            ) : nth_(count)
#ifdef DFLASH_BLOCKING_ROW_POOL_TEST_HOOKS
              , test_hook_(hook), test_hook_context_(hook_context)
#endif
    {
        if (count == 0) throw std::invalid_argument("row pool needs a worker");
        try {
            for (unsigned i = 0; i < count; ++i) {
                workers_.emplace_back([this, i] { worker(i); });
            }
        } catch (...) {
            shutdown();
            throw;
        }
    }
    ~BlockingRowPool() { shutdown(); }
    BlockingRowPool(const BlockingRowPool &) = delete;
    BlockingRowPool & operator=(const BlockingRowPool &) = delete;

    template<class Fn> void run_chunks(int rows, const Fn & fn) {
        if (rows <= 0) return;
        std::lock_guard<std::mutex> client(client_mu_);
        job_ = {&fn, [](const void * context, int begin, int end) {
                    (*static_cast<const Fn *>(context))(begin, end);
                }, rows, std::min(nth_, (unsigned) rows)};
        // Inactive workers still read the generation/job header. Wait for
        // them too before mutating job_ or returning its borrowed callback.
        remaining_.store(nth_, std::memory_order_release);
        {
            std::lock_guard<std::mutex> wake(wait_mu_);
            seq_.fetch_add(1, std::memory_order_release);
        }
        wait_cv_.notify_all();
        unsigned spins = 0;
        while (remaining_.load(std::memory_order_acquire) != 0) {
            if (++spins < 65536) relax();
            else { std::this_thread::yield(); spins = 0; }
        }
    }

    template<class Fn> void run_custom(int rows, const Fn & fn) {
        run_chunks(rows, [&fn](int begin, int end) {
            for (int row = begin; row < end; ++row) fn(row);
        });
    }

private:
    struct Job {
        const void * context;
        void (*invoke)(const void *, int, int);
        int rows;
        unsigned active;
    };
    static void relax() {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
        __builtin_ia32_pause();
#endif
    }
    void worker(unsigned index) {
        uint64_t last = 0;
        for (;;) {
            uint64_t generation = last;
            for (unsigned spins = 0; spins < 65536; ++spins) {
                generation = seq_.load(std::memory_order_acquire);
                if (generation != last || stop_.load(std::memory_order_relaxed)) break;
                relax();
            }
            if (generation == last && !stop_.load(std::memory_order_relaxed)) {
                std::unique_lock<std::mutex> wait(wait_mu_);
                wait_cv_.wait(wait, [&] {
                    return stop_.load(std::memory_order_relaxed) ||
                           seq_.load(std::memory_order_acquire) != last;
                });
                generation = seq_.load(std::memory_order_acquire);
            }
            if (stop_.load(std::memory_order_relaxed)) return;
            last = generation;
#ifdef DFLASH_BLOCKING_ROW_POOL_TEST_HOOKS
            if (test_hook_) test_hook_(test_hook_context_, index, false);
#endif
            const Job job = job_;
            if (index < job.active) {
                const int chunk = (job.rows - 1) / (int) job.active + 1;
                const int64_t begin = (int64_t) index * chunk;
                const int64_t end = std::min<int64_t>(job.rows, begin + chunk);
                if (begin < end) job.invoke(job.context, (int) begin, (int) end);
            }
            remaining_.fetch_sub(1, std::memory_order_acq_rel);
#ifdef DFLASH_BLOCKING_ROW_POOL_TEST_HOOKS
            if (test_hook_) test_hook_(test_hook_context_, index, true);
#endif
        }
    }
    void shutdown() {
        {
            std::lock_guard<std::mutex> wait(wait_mu_);
            stop_.store(true, std::memory_order_release);
        }
        wait_cv_.notify_all();
        for (auto & worker : workers_) worker.join();
    }
    const unsigned nth_;
#ifdef DFLASH_BLOCKING_ROW_POOL_TEST_HOOKS
    // Host test target only; production contains neither hooks nor branches.
    const TestHook test_hook_;
    void * const test_hook_context_;
#endif
    std::mutex client_mu_, wait_mu_;
    std::condition_variable wait_cv_;
    std::atomic<uint64_t> seq_{0};
    std::atomic<unsigned> remaining_{0};
    std::atomic<bool> stop_{false};
    Job job_{};
    std::vector<std::thread> workers_;
};

} // namespace dflash::common
