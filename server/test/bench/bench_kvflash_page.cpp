// KVFlash page-event microbench (CUDA). Real-size device KV: 16 attention
// layers, K and V, n_head_kv=4, head_dim=256, q8_0, 64-token chunks. Times
// reselect cycles (recent -> picked -> recent) and reports ms per page event.
//   bench_kvflash_page [cycles=20] [pool_chunks=48] [ctx_chunks=96]
#include "../../src/common/kvflash_pager.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace luce::common;

int main(int argc, char ** argv) {
    const int cycles = argc > 1 ? std::atoi(argv[1]) : 20;
    const int pool_chunks = argc > 2 ? std::atoi(argv[2]) : 48;
    const int ctx_chunks = argc > 3 ? std::atoi(argv[3]) : 96;
    constexpr int kChunk = 64, kLayers = 16, kHeadDim = 256, kHeadsKv = 4;
    const int pool = pool_chunks * kChunk;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "no cuda\n"); return 2; }
    ggml_init_params ip{};
    ip.mem_size = (size_t)(kLayers * 2 + 8) * ggml_tensor_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> k, v;
    for (int l = 0; l < kLayers; l++) {
        k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, kHeadDim, pool, kHeadsKv));
        v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, kHeadDim, pool, kHeadsKv));
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { std::fprintf(stderr, "alloc failed\n"); return 2; }

    KvFlashConfig cfg;
    cfg.pool_tokens = pool; cfg.chunk_tokens = kChunk; cfg.sink_chunks = 1; cfg.tail_window_chunks = 4;
    KvFlashPager p;
    if (!p.attach(cfg, k, v)) { std::fprintf(stderr, "attach failed\n"); return 2; }
    if (!p.alloc_span(0, ctx_chunks * kChunk)) { std::fprintf(stderr, "fill failed\n"); return 2; }
    p.synchronize_paging();

    // "picked" = a band of old chunks the scorer wants back; the rest is recency.
    const int lo = 8;
    auto picked = [lo](int c) { return (c >= lo && c < lo + 24) ? 1000.0f + c : (float)c * 1e-3f; };
    std::vector<double> ms;
    int64_t events_total = 0;
    const KvFlashStats s0 = p.stats();
    for (int i = 0; i < cycles; i++) {
        auto t0 = std::chrono::steady_clock::now();
        p.score_hook = picked;
        int ev = p.reselect(0);
        p.score_hook = nullptr;
        ev += p.reselect_recent();
        p.synchronize_paging();
        auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / (ev > 0 ? ev : 1));
        events_total += ev;
    }
    const KvFlashStats s1 = p.stats();
    std::sort(ms.begin(), ms.end());
    // cycle 0 pays the first (dirty) page-outs; report median plus the steady (last half) mean
    double steady = 0; int n = 0;
    for (int i = cycles / 2; i < cycles; i++) { steady += ms[i]; n++; }
    std::printf("cycles=%d events=%lld page_outs=%lld page_ins=%lld moved=%.1f MiB\n",
                cycles, (long long)events_total, (long long)(s1.page_outs - s0.page_outs),
                (long long)(s1.page_ins - s0.page_ins), (s1.moved_bytes - s0.moved_bytes) / 1048576.0);
    std::printf("ms/event median=%.4f min=%.4f max=%.4f\n", ms[ms.size() / 2], ms.front(), ms.back());
    return 0;
}
