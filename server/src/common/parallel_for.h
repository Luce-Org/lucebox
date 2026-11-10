// Spawn-and-join helpers: run_workers(n, fn(w)) runs fn on n threads (w = 0 on the caller), parallel_for(n, f)
// stripes f(0..n-1) over the cores.
#pragma once

#include <algorithm>
#include <exception>
#include <thread>
#include <vector>

namespace luce::common {

// n <= 0 runs nothing. Every worker finishes and is joined before the first exception (lowest worker index) is rethrown.
template <class W> inline void run_workers(int n, W worker) {
    if (n <= 0) return;
    std::vector<std::exception_ptr> err((size_t)n);
    auto run = [&](int w) { try { worker(w); } catch (...) { err[(size_t)w] = std::current_exception(); } };
    std::vector<std::thread> th;
    for (int w = 1; w < n; w++) th.emplace_back(run, w);
    run(0);
    for (auto & t : th) t.join();
    for (auto & e : err) if (e) std::rethrow_exception(e);
}

template <class F> inline void parallel_for(int n, F f) {
    const int nt = std::max(1, std::min<int>(n, (int)std::thread::hardware_concurrency()));
    run_workers(nt, [&](int w) { for (int i = w; i < n; i += nt) f(i); });
}

}  // namespace luce::common
