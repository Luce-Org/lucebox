// Spawn-and-join helpers: run_workers(n, fn(w)) runs fn on n threads (w = 0 on the caller), parallel_for(n, f)
// stripes f(0..n-1) over the cores.
#pragma once

#include <algorithm>
#include <thread>
#include <vector>

namespace luce::common {

template <class W> inline void run_workers(int n, W worker) {
    std::vector<std::thread> th;
    for (int w = 1; w < n; w++) th.emplace_back([&, w] { worker(w); });
    worker(0);
    for (auto & t : th) t.join();
}

template <class F> inline void parallel_for(int n, F f) {
    const int nt = std::max(1, std::min<int>(n, (int)std::thread::hardware_concurrency()));
    run_workers(nt, [&](int w) { for (int i = w; i < n; i += nt) f(i); });
}

}  // namespace luce::common
