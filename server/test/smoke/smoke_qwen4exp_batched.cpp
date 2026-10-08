// Exact-width qwen4exp independent-slot decode probe.
// Usage: smoke_qwen4exp_batched <iq4nl-shard1.gguf> [prefill_tokens=16] [ctx=32768] [steps=3]
#include "qwen4exp_internal.h"
#include "qwen4exp_graph.h"
#include "qwen4exp_cache.h"
#include "ggml-cuda.h"
#include "qwen4exp_test_state.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace luce::common;
using namespace qwen4exp_test;
namespace {
constexpr int N = 4;
int argmax(const std::vector<float> & x) {
    return (int) std::distance(x.begin(), std::max_element(x.begin(), x.end()));
}
float max_delta(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return INFINITY;
    float e = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        e = std::max(e, std::abs(a[i] - b[i]));
    }
    return e;
}
uint64_t logits_hash(const std::vector<float> & x) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t * p = reinterpret_cast<const uint8_t *>(x.data());
    for (size_t i = 0; i < x.size() * sizeof(float); ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
bool same_bits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <shard1.gguf> [prefill=16] [ctx=32768] [steps=3]\n", argv[0]); return 2; }
    const int prompt_n = argc > 2 ? std::atoi(argv[2]) : 16;
    const int ctx = argc > 3 ? std::atoi(argv[3]) : 32768;
    const int steps = argc > 4 ? std::atoi(argv[4]) : 3;
    if (prompt_n <= 0 || steps <= 0 || prompt_n + steps + 1 >= ctx) return 2;
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "no GPU backend\n"); return 77; }
    Qwen4ExpWeights w;
    if (!load_qwen4exp_gguf(argv[1], backend, w)) { std::fprintf(stderr, "model load failed\n"); return 1; }
    Qwen4ExpCache caches[N]; Qwen4ExpCache * ptrs[N];
    for (int s = 0; s < N; ++s) {
        ptrs[s] = &caches[s];
        if (!create_qwen4exp_cache(backend, w, ctx, caches[s])) return 1;
    }
    const bool qsa = w.gfx1151;
    Qwen4ExpBatchedDecodeWorkspace workspace;
    int failures = 0;

    for (int phase = 0; phase < 2; ++phase) {
        const bool identical = phase == 0;
        for (int s = 0; s < N; ++s) {
            reset_qwen4exp_state(backend, caches[s]);
            if (qsa) poison_indexer(caches[s]);
            std::vector<int32_t> prompt((size_t) prompt_n);
            for (int i = 0; i < prompt_n; ++i)
                prompt[(size_t) i] = (int32_t) ((i * 7919 + 13 + (identical ? 0 : s * 104729)) % w.n_vocab);
            std::vector<float> logits;
            auto r = qwen4exp_forward(backend, w, caches[s], prompt.data(), prompt_n, 0, logits);
            if (!r.ok) { std::fprintf(stderr, "prefill failed slot=%d\n", s); return 1; }
        }
        int32_t feed[N], pos[N];
        for (int s = 0; s < N; ++s) {
            feed[s] = (int32_t) ((1237 + (identical ? 0 : s * 3571)) % w.n_vocab);
            pos[s] = prompt_n;
        }
        {
            SavedCache before = save_cache(caches[0]);
            std::vector<float> solo, via_api;
            auto sr = qwen4exp_forward(backend, w, caches[0], &feed[0], 1, pos[0], solo);
            SavedCache solo_after = save_cache(caches[0]);
            restore_cache(caches[0], before);
            Qwen4ExpCache * one_cache[1] = { &caches[0] };
            std::vector<std::vector<float>> one_logits;
            auto br = qwen4exp_forward_batched(backend, w, one_cache, &feed[0], &pos[0], 1,
                                                workspace, one_logits);
            if (!sr.ok || !br.ok || one_logits.size() != 1 ||
                !same_bits(solo, one_logits[0]) || !equal_cache(caches[0], solo_after)) {
                ++failures;
                std::printf("[batch-probe] N=1-single-path=FAIL\n");
            } else std::printf("[batch-probe] N=1-single-path=PASS\n");
            restore_cache(caches[0], before);
        }
        std::printf("[batch-probe] phase=%s N=%d ctx=%d prefill=%d\n", identical ? "identical" : "distinct", N, ctx, prompt_n);
        for (int step = 0; step < steps; ++step) {
            std::vector<SavedCache> snapshots; snapshots.reserve(N);
            for (int s = 0; s < N; ++s) snapshots.push_back(save_cache(caches[s]));
            std::vector<std::vector<float>> solo(N), batched;
            std::vector<SavedIndexer> expected;
            for (int s = 0; s < N; ++s) {
                auto r = qwen4exp_forward(backend, w, caches[s], &feed[s], 1, pos[s], solo[s]);
                expected.push_back(save_indexer(caches[s], qsa ? pos[s] + 1 : 0));
                restore_cache(caches[s], snapshots[s]);
                if (!r.ok) { std::fprintf(stderr, "solo failed slot=%d\n", s); return 1; }
            }
            auto r = qwen4exp_forward_batched(backend, w, ptrs, feed, pos, N, workspace, batched);
            if (!r.ok || batched.size() != N) { std::fprintf(stderr, "batched forward failed\n"); return 1; }
            float phase_eps = 0;
            for (int s = 0; s < N; ++s) {
                const float eps = max_delta(solo[s], batched[s]); phase_eps = std::max(phase_eps, eps);
                const int solo_id = argmax(solo[s]), batch_id = argmax(batched[s]);
                const bool allowed = std::isfinite(eps) && same_bits(solo[s], batched[s]);
                if (!allowed) ++failures;
                if (qsa && !equal_indexer(caches[s], expected[s], pos[s] + 1)) ++failures;
                std::printf("[batch-probe] phase=%s step=%d slot=%d max_abs_delta=%.9g solo=%d batch=%d exact=%s\n",
                    identical ? "identical" : "distinct", step, s, eps, solo_id, batch_id, allowed ? "pass" : "FAIL");
                if (identical && !same_bits(batched[0], batched[s])) {
                    ++failures; std::printf("[batch-probe] identical_logits=FAIL slot=%d\n", s);
                }
            }
            std::printf("[batch-probe] phase=%s step=%d epsilon_max=%.9g\n", identical ? "identical" : "distinct", step, phase_eps);
            if (identical) for (int s = 0; s < N; ++s) if (argmax(batched[s]) != argmax(batched[0])) ++failures;
            for (int s = 0; s < N; ++s) { feed[s] = argmax(batched[s]); pos[s]++; }
        }

        // Exercise non-contiguous active rows and prove untouched slots do not move.
        SavedCache untouched0 = save_cache(caches[0]), untouched2 = save_cache(caches[2]);
        Qwen4ExpCache * pair_a[2] = { &caches[3], &caches[1] };
        int32_t pair_tokens_a[2] = { feed[3], feed[1] }, pair_pos_a[2] = { pos[3], pos[1] };
        SavedCache active3 = save_cache(caches[3]), active1 = save_cache(caches[1]);
        std::vector<std::vector<float>> pair_logits_a;
        auto pr = qwen4exp_forward_batched(backend, w, pair_a, pair_tokens_a, pair_pos_a, 2, workspace, pair_logits_a);
        if (!pr.ok || !equal_cache(caches[0], untouched0) || !equal_cache(caches[2], untouched2)) {
            ++failures; std::printf("[batch-probe] untouched-slot-isolation=FAIL\n");
        } else std::printf("[batch-probe] untouched-slot-isolation=PASS\n");
        // Reset the active states and repeat with reversed row order.
        restore_cache(caches[3], active3); restore_cache(caches[1], active1);
        Qwen4ExpCache * pair_b[2] = { &caches[1], &caches[3] };
        int32_t pair_tokens_b[2] = { pair_tokens_a[1], pair_tokens_a[0] };
        int32_t pair_pos_b[2] = { pair_pos_a[1], pair_pos_a[0] };
        std::vector<std::vector<float>> pair_logits_b;
        pr = qwen4exp_forward_batched(backend, w, pair_b, pair_tokens_b, pair_pos_b, 2, workspace, pair_logits_b);
        if (!pr.ok || pair_logits_a.size() != 2 || pair_logits_b.size() != 2 ||
            !same_bits(pair_logits_a[0], pair_logits_b[1]) || !same_bits(pair_logits_a[1], pair_logits_b[0])) {
            ++failures; std::printf("[batch-probe] row-permutation=FAIL\n");
        } else std::printf("[batch-probe] row-permutation=PASS\n");

        // A reset/reuse must match a freshly allocated cache for the same prefix.
        Qwen4ExpCache fresh;
        if (!create_qwen4exp_cache(backend, w, ctx, fresh)) return 1;
        reset_qwen4exp_state(backend, caches[0]);
        reset_qwen4exp_state(backend, fresh);
        std::vector<int32_t> prompt((size_t) prompt_n);
        for (int i = 0; i < prompt_n; ++i)
            prompt[(size_t) i] = (int32_t) ((i * 7919 + 13) % w.n_vocab);
        std::vector<float> tmp, reused_logits, fresh_logits;
        qwen4exp_forward(backend, w, caches[0], prompt.data(), prompt_n, 0, tmp);
        qwen4exp_forward(backend, w, fresh, prompt.data(), prompt_n, 0, tmp);
        const int32_t reuse_token = 77 % w.n_vocab;
        auto rr = qwen4exp_forward(backend, w, caches[0], &reuse_token, 1, prompt_n, reused_logits);
        auto fr = qwen4exp_forward(backend, w, fresh, &reuse_token, 1, prompt_n, fresh_logits);
        if (!rr.ok || !fr.ok || !same_bits(reused_logits, fresh_logits)) {
            ++failures; std::printf("[batch-probe] cancel-reset-reuse=FAIL\n");
        } else std::printf("[batch-probe] cancel-reset-reuse=PASS reused_hash=%016llx fresh_cache_hash=%016llx\n",
            (unsigned long long) logits_hash(reused_logits),
            (unsigned long long) logits_hash(fresh_logits));
        free_qwen4exp_cache(fresh);
    }
    clear_qwen4exp_batched_decode_workspace(workspace);
    for (auto & c : caches) free_qwen4exp_cache(c);
    free_qwen4exp_weights(w);
    ggml_backend_free(backend);
    std::printf("[batch-probe] %s failures=%d\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
