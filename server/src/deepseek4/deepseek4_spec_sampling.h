// Speculative sampling for DSpark's greedy drafts.
//
// The DSpark drafter proposes deterministic (greedy) candidates. For a
// deterministic draft q(x) = [x == d], the speculative-sampling rule reduces
// to: keep candidate d with the target's probability p(d); on rejection draw
// from p with d removed, renormalized (max(0, p - q) / (1 - p(d))). Each
// emitted token then follows the target sampler's distribution exactly, so a
// sampled request keeps its sampler contract while decoding speculatively.

#pragma once

#include "common/sampler.h"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <utility>
#include <vector>

namespace luce::common {

// Per-request state: the request's sampler chain, the token history its
// penalties read (prompt + every emitted token, seed included) and its RNG.
struct DSparkSpecSampling {
    SamplerCfg cfg;
    std::vector<int32_t> history;
    std::mt19937_64 * rng = nullptr;
};

// History a verify row's sampler reads: the tail of `history` that the
// penalties can see (rep_window tokens) followed by draft[1..i]. Empty when no
// penalty is active, since the chain then ignores history.
inline void dspark_row_history(const SamplerCfg & cfg, const std::vector<int32_t> & history,
                               const int32_t * draft, int i, std::vector<int32_t> & out) {
    out.clear();
    if (!(cfg.rep_pen > 1.0f || cfg.freq_pen != 0.0f || cfg.pres_pen != 0.0f)) return;
    const size_t keep = (size_t) std::max(0, cfg.rep_window);
    const size_t from = history.size() > keep ? history.size() - keep : 0;
    out.assign(history.begin() + (std::ptrdiff_t) from, history.end());
    for (int k = 1; k <= i; k++) out.push_back(draft[k]);
}

// Persistent workers for building verify rows in parallel. One pool lives for
// a request, so a long generation does not create threads per step. run(n, fn)
// calls fn(0) on the caller and fn(1..n-1) on the workers, and returns when
// all have finished.
class DSparkRowPool {
public:
    explicit DSparkRowPool(int n_workers) {
        for (int w = 0; w < n_workers; w++) threads_.emplace_back([this, w] { loop(w); });
    }
    ~DSparkRowPool() {
        { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
        cv_.notify_all();
        for (auto & t : threads_) t.join();
    }
    DSparkRowPool(const DSparkRowPool &) = delete;
    DSparkRowPool & operator=(const DSparkRowPool &) = delete;

    // A row that throws (here or on a worker) is rethrown once every worker
    // is done with `fn`; the first failure wins and the pool stays usable.
    void run(int n, const std::function<void(int)> & fn) {
        const int n_workers = std::max(0, std::min(n - 1, (int) threads_.size()));
        {
            std::lock_guard<std::mutex> lk(mu_);
            fn_ = &fn; n_jobs_ = n_workers; pending_ = n_workers; error_ = nullptr; gen_++;
        }
        cv_.notify_all();
        std::exception_ptr local;
        try {
            if (n > 0) fn(0);
            for (int i = n_workers + 1; i < n; i++) fn(i);   // more rows than workers
        } catch (...) {
            local = std::current_exception();
        }
        std::exception_ptr err;
        {
            std::unique_lock<std::mutex> lk(mu_);
            done_cv_.wait(lk, [this] { return pending_ == 0; });
            fn_ = nullptr;
            err = local ? local : error_;
            error_ = nullptr;
        }
        if (err) std::rethrow_exception(err);
    }

private:
    void loop(int w) {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(int)> * fn = nullptr;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
                if (w >= n_jobs_) continue;
                fn = fn_;
            }
            std::exception_ptr err;
            try {
                (*fn)(w + 1);
            } catch (...) {
                err = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (err && !error_) error_ = err;
                if (--pending_ == 0) done_cv_.notify_one();
            }
        }
    }

    std::vector<std::thread> threads_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    const std::function<void(int)> * fn_ = nullptr;
    std::exception_ptr error_;
    int n_jobs_ = 0, pending_ = 0;
    uint64_t gen_ = 0;
    bool stop_ = false;
};

struct DSparkSampleStep {
    int accept = 1;   // seed + kept candidates, as in the greedy loop
    int bonus = -1;   // token emitted after the last kept candidate
};

// rows[i] is the target sampler's distribution (sampler_distribution) after
// draft[0..i], i.e. with draft[1..i] appended to the history; draft[0] is the
// step's seed. Row i is only consulted when draft[1..i] were all kept, so the
// rows can be built up front. Rows are modified (a rejected candidate's
// weight is zeroed). Uniforms are drawn in walk order: one per candidate
// test, one for the final draw.
inline DSparkSampleStep dspark_spec_sample_accept(
        std::vector<std::vector<std::pair<float, int>>> & rows,
        const int32_t * draft, int q, std::mt19937_64 & rng) {
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    DSparkSampleStep step;
    for (int i = 0; i < q; i++) {
        auto & dist = rows[(size_t) i];
        if (i == q - 1) {                       // every candidate kept
            step.bonus = sampler_draw(dist, unif(rng));
            return step;
        }
        const int cand = draft[i + 1];
        float p_cand = 0.0f;
        for (auto & d : dist) {
            if (d.second == cand) { p_cand = d.first; d.first = 0.0f; break; }
        }
        if (unif(rng) < (double) p_cand) {
            step.accept++;
            continue;
        }
        step.bonus = sampler_draw(dist, unif(rng));   // p without the candidate
        return step;
    }
    return step;
}

}  // namespace luce::common
