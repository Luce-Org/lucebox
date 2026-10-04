// Smoke test for the Qwen3.8-Flash-Next (`qwen4exp`) forward path.
//
// Loads shard 1 (shard 2, the lazy PLE table, is discovered from the shard-1
// filename), builds the KV + delta-net cache, runs a prefill chunk and one
// decode step, and checks the logits are finite and the expected size. Mirrors
// smoke_qwen3_forward.cpp; no daemon involved.
//
// Usage:
//   smoke_qwen4exp_forward <shard1.gguf> [seq_len=16] [--token-file FILE] [--split N[:chunk]] [--reference] [--dump] [--chunk N] [--compare-chunk N] [--plan-ctx N --slots N]

#include "qwen4exp_internal.h"
#include "qwen4exp_graph.h"
#include "qwen4exp_cache.h"
#include "qwen4exp_chunk.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

using namespace luce::common;

namespace {

bool all_finite(const std::vector<float> & v) {
    for (const float f : v) {
        if (!std::isfinite(f)) {
            return false;
        }
    }
    return true;
}

int argmax(const std::vector<float> & v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > v[best]) {
            best = (int) i;
        }
    }
    return best;
}

bool same_bits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() &&
        std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

uint64_t logits_hash(const std::vector<float> & x) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t * p = reinterpret_cast<const uint8_t *>(x.data());
    for (size_t i = 0; i < x.size() * sizeof(float); ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <shard1.gguf> [seq_len=16] [--token-file FILE] [--split N[:chunk]] [--reference] [--dump] [--chunk N] [--compare-chunk N] [--plan-ctx N --slots N]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    int S = 16, N = 0, step = 1, chunk = 0, compare_chunk = 0, plan_ctx = 0, slots = 1;
    const char * token_file = nullptr;
    bool reference = false, dump = false;
    auto positive = [](const char * first, const char * last, int & n) {
        const auto r = std::from_chars(first, last, n);
        return r.ec == std::errc{} && r.ptr == last && n > 0;
    };
    int arg = 2;
    if (arg < argc && argv[arg][0] != '-') {
        if (!positive(argv[arg], argv[arg] + std::strlen(argv[arg]), S)) return 2;
        ++arg;
    }
    for (; arg < argc; ++arg) {
        const std::string opt = argv[arg];
        if (opt == "--reference") reference = true;
        else if (opt == "--dump") dump = true;
        else if ((opt == "--chunk" || opt == "--compare-chunk" || opt == "--plan-ctx" || opt == "--slots") && arg + 1 < argc) {
            const char * val = argv[++arg];
            int & dst = opt == "--chunk" ? chunk : opt == "--compare-chunk" ? compare_chunk : opt == "--plan-ctx" ? plan_ctx : slots;
            if (!positive(val, val + std::strlen(val), dst)) return 2;
        }
        else if (opt == "--token-file" && arg + 1 < argc) token_file = argv[++arg];
        else if (opt == "--split" && arg + 1 < argc) {
            const char * val = argv[++arg], * end = val + std::strlen(val), * colon = std::strchr(val, ':');
            if (!positive(val, colon ? colon : end, N) || N >= S ||
                (colon && !positive(colon + 1, end, step))) return 2;
        } else { std::fprintf(stderr, "invalid option: %s\n", argv[arg]); return 2; }
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "[smoke] no GPU backend available\n");
        return 77;
    }

    Qwen4ExpWeights w;
    auto t_load0 = std::chrono::steady_clock::now();
    if (!load_qwen4exp_gguf(path, backend, w, reference)) {
        std::fprintf(stderr, "[smoke] load_qwen4exp_gguf failed\n");
        ggml_backend_free(backend);
        return 1;
    }
    auto t_load1 = std::chrono::steady_clock::now();
    std::printf("[smoke] load %.2fs layers=%d vocab=%d shard2=%s\n",
        std::chrono::duration<double>(t_load1 - t_load0).count(),
        w.n_layer, w.n_vocab, w.ple_reader.available() ? "yes" : "no");

    Qwen4ExpCache cache;
    if (!create_qwen4exp_cache(backend, w, plan_ctx ? plan_ctx : S + (compare_chunk ? 132 : 4), GGML_TYPE_F16, cache, reference)) {
        std::fprintf(stderr, "[smoke] create_qwen4exp_cache failed\n");
        free_qwen4exp_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    if (plan_ctx) {
        const int selected = qwen4exp_select_chunk(backend, w, cache, slots);
        std::printf("[smoke] memory-plan ctx=%d slots=%d chunk=%d\n", plan_ctx, slots, selected);
        free_qwen4exp_cache(cache);
        free_qwen4exp_weights(w);
        ggml_backend_free(backend);
        return selected > 0 ? 0 : 1;
    }

    std::vector<int32_t> tokens((size_t) S);
    for (int i = 0; i < S; ++i) {
        tokens[(size_t) i] = (int32_t) ((i * 7919 + 13) % w.n_vocab);
    }

    if (token_file) {
        std::ifstream input(token_file);
        for (int i = 0; i < S; ++i) {
            if (!(input >> tokens[i]) || tokens[i] < 0 || tokens[i] >= w.n_vocab) {
                std::fprintf(stderr, "invalid --token-file\n");
                return 2;
            }
        }
        int extra;
        if (input >> extra) { std::fprintf(stderr, "too many input tokens\n"); return 2; }
    }
    auto prefill = [&](Qwen4ExpCache & state, int length, std::vector<float> & logits, bool summaries) {
        Qwen4ExpForwardResult r;
        const int stride = chunk > 0 ? chunk : length;
        for (int pos = 0; pos < length; pos += stride) {
            r = qwen4exp_forward(backend, w, state, tokens.data() + pos,
                                std::min(stride, length - pos), pos, logits, summaries);
            if (!r.ok) break;
        }
        return r;
    };
    if (compare_chunk) {
        // Both free-running greedy and teacher-forced distributions are needed:
        // KL after independently diverged tokens measures a different context.
        const int baseline_chunk = chunk > 0 ? chunk : 2048;
        constexpr int generate = 128;
        std::vector<int32_t> reference_tokens, candidate_tokens;
        std::vector<std::vector<float>> reference_logits;
        std::vector<float> logits;
        bool ok = true;
        auto begin = [&](int c) {
            chunk = c;
            reset_qwen4exp_state(backend, cache);
            return prefill(cache, S, logits, false).ok && all_finite(logits);
        };
        ok = begin(baseline_chunk);
        for (int i = 0; ok && i < generate; ++i) {
            reference_logits.push_back(logits);
            reference_tokens.push_back(argmax(logits));
            if (i + 1 < generate) ok = qwen4exp_forward(backend, w, cache,
                &reference_tokens.back(), 1, S + i, logits).ok && all_finite(logits);
        }
        auto kl = [](const std::vector<float> & a, const std::vector<float> & b) {
            const double ma = *std::max_element(a.begin(), a.end());
            const double mb = *std::max_element(b.begin(), b.end());
            double za = 0, zb = 0, d = 0;
            for (size_t i = 0; i < a.size(); ++i) { za += std::exp(a[i] - ma); zb += std::exp(b[i] - mb); }
            for (size_t i = 0; i < a.size(); ++i) {
                const double la = a[i] - ma - std::log(za), lb = b[i] - mb - std::log(zb);
                d += std::exp(la) * (la - lb);
            }
            return d;
        };
        double kl_sum = 0, kl_max = 0, initial_kl = 0;
        int forced_disagreements = 0, bits_differ = 0;
        ok = ok && begin(compare_chunk);
        for (int i = 0; ok && i < generate; ++i) {
            const double d = kl(reference_logits[i], logits);
            if (i == 0) initial_kl = d;
            kl_sum += d;
            kl_max = std::max(kl_max, d);
            forced_disagreements += argmax(logits) != reference_tokens[i];
            bits_differ += !same_bits(reference_logits[i], logits);
            if (i + 1 < generate) ok = qwen4exp_forward(backend, w, cache,
                &reference_tokens[i], 1, S + i, logits).ok && all_finite(logits);
        }
        ok = ok && begin(compare_chunk);
        int first = -1;
        for (int i = 0; ok && i < generate; ++i) {
            candidate_tokens.push_back(argmax(logits));
            if (first < 0 && candidate_tokens.back() != reference_tokens[i]) first = i;
            if (i + 1 < generate) ok = qwen4exp_forward(backend, w, cache,
                &candidate_tokens.back(), 1, S + i, logits).ok && all_finite(logits);
        }
        std::printf("[chunk-compare] {\"ok\":%s,\"baseline\":%d,\"candidate\":%d,\"prompt_tokens\":%d,"
            "\"generated\":%d,\"first_greedy_divergence\":%d,\"teacher_argmax_disagreements\":%d,"
            "\"logit_rows_differ\":%d,\"initial_kl\":%.9g,\"teacher_kl_mean\":%.9g,\"teacher_kl_max\":%.9g}\n",
            ok ? "true" : "false", baseline_chunk, compare_chunk, S, generate, first,
            forced_disagreements, bits_differ, initial_kl, kl_sum / generate, kl_max);
        for (const auto * sequence : {&reference_tokens, &candidate_tokens}) {
            std::printf("[chunk-tokens] %s", sequence == &reference_tokens ? "baseline" : "candidate");
            for (int32_t t : *sequence) std::printf(" %d", t);
            std::printf("\n");
        }
        free_qwen4exp_cache(cache);
        free_qwen4exp_weights(w);
        ggml_backend_free(backend);
        return ok ? 0 : 1; // A numerics change is reported, never silently blessed.
    }
    std::vector<float> logits;
    auto t0 = std::chrono::steady_clock::now();
    const Qwen4ExpForwardResult pre = prefill(cache, S, logits, dump);
    auto t1 = std::chrono::steady_clock::now();

    int rc = 0;
    if (!pre.ok || logits.size() != (size_t) w.n_vocab || !all_finite(logits)) {
        std::fprintf(stderr, "[smoke] prefill FAILED ok=%d logits=%zu finite=%d\n",
            (int) pre.ok, logits.size(), (int) all_finite(logits));
        rc = 1;
    } else {
        std::printf("[smoke] prefill OK %.3fs T=%d argmax=%d\n",
            std::chrono::duration<double>(t1 - t0).count(), S, argmax(logits));
    }

    if (rc == 0) {
        const int32_t next = (int32_t) argmax(logits);
        std::vector<float> logits2;
        auto d0 = std::chrono::steady_clock::now();
        const Qwen4ExpForwardResult dec = qwen4exp_forward(
            backend, w, cache, &next, 1, S, logits2, dump);
        auto d1 = std::chrono::steady_clock::now();
        if (!dec.ok || logits2.size() != (size_t) w.n_vocab || !all_finite(logits2)) {
            std::fprintf(stderr, "[smoke] decode FAILED ok=%d logits=%zu finite=%d\n",
                (int) dec.ok, logits2.size(), (int) all_finite(logits2));
            rc = 1;
        } else {
            std::printf("[smoke] decode OK %.3fs pos=%d argmax=%d\n",
                std::chrono::duration<double>(d1 - d0).count(), S, argmax(logits2));
        }
    }

    // --split N[:c]: compare one prefill with S-N prompt tokens followed by N
    // tokens in chunks of c (default 1: decode). These are numerical probes,
    // not a bitwise prefill/decode contract: short QSA and packed QSA use
    // different reductions. At S=6000 the established KLs for 100:1, 200:1,
    // 100:100, 256:128 are .082874/.118543/.652715/.734763 (the last flips
    // argmax 271 -> 17 and exits 1). Below the budget, T=1 deliberately keeps
    // dense decode while multi-row prompt prefill uses precise QSA.
    if (rc == 0 && N > 0) {
        std::vector<float> full, split;
        reset_qwen4exp_state(backend, cache);
        bool ok = N > 0 && N < S && prefill(cache, S, full, dump).ok;
        reset_qwen4exp_state(backend, cache);
        ok = ok && prefill(cache, S - N, split, dump).ok;
        for (int i = S - N; ok && i < S; i += step) {
            const int n = std::min(step, S - i);
            ok = qwen4exp_forward(backend, w, cache, &tokens[i], n, i, split, dump).ok;
        }
        ok = ok && full.size() == (size_t) w.n_vocab && split.size() == full.size() &&
             all_finite(full) && all_finite(split);
        float max_diff = 0.0f;
        double kl = 0.0;
        int overlap = 0;
        if (ok) {
            auto softmax = [](const std::vector<float> & v) {
                const float mx = *std::max_element(v.begin(), v.end());
                std::vector<double> p(v.size());
                double z = 0.0;
                for (size_t i = 0; i < v.size(); ++i) z += (p[i] = std::exp((double) v[i] - mx));
                for (double & x : p) x /= z;
                return p;
            };
            auto top10 = [](const std::vector<float> & v) {
                std::vector<int> idx(v.size());
                for (size_t i = 0; i < v.size(); ++i) idx[i] = (int) i;
                std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(), [&](int a, int b) { return v[a] > v[b]; });
                idx.resize(10);
                return idx;
            };
            const std::vector<double> pf = softmax(full), ps = softmax(split);
            for (size_t i = 0; i < full.size(); ++i) {
                max_diff = std::max(max_diff, std::fabs(full[i] - split[i]));
                if (pf[i] > 0.0) kl += pf[i] * std::log(pf[i] / std::max(ps[i], 1e-300));
            }
            const std::vector<int> tf = top10(full), ts = top10(split);
            for (int a : tf) overlap += (int) std::count(ts.begin(), ts.end(), a);
        }
        std::printf("[smoke] split S=%d N=%d chunk=%d ok=%d argmax %d vs %d max_abs_diff=%.4f kl=%.6f top10_overlap=%d\n",
            S, N, step, (int) ok, ok ? argmax(full) : -1, ok ? argmax(split) : -1, max_diff, kl, overlap);
        if (!ok || argmax(full) != argmax(split)) rc = 1;
    }

    // A reset/reuse must match a freshly allocated cache for the same prefix.
    if (rc == 0) {
        Qwen4ExpCache fresh;
        if (!create_qwen4exp_cache(backend, w, cache.max_ctx, GGML_TYPE_F16, fresh, reference)) {
            std::fprintf(stderr, "[smoke] fresh cache creation failed\n");
            rc = 1;
        } else {
            reset_qwen4exp_state(backend, cache);
            reset_qwen4exp_state(backend, fresh);
            std::vector<float> tmp, reused_logits, fresh_logits;
            const bool p1 = prefill(cache, S, tmp, dump).ok;
            const bool p2 = prefill(fresh, S, tmp, dump).ok;
            const int32_t reuse_token = 77 % w.n_vocab;
            const Qwen4ExpForwardResult rr = qwen4exp_forward(backend, w, cache, &reuse_token, 1, S, reused_logits, dump);
            const Qwen4ExpForwardResult fr = qwen4exp_forward(backend, w, fresh, &reuse_token, 1, S, fresh_logits, dump);
            if (!p1 || !p2 || !rr.ok || !fr.ok || !same_bits(reused_logits, fresh_logits)) {
                std::fprintf(stderr, "[smoke] cancel-reset-reuse FAILED\n");
                rc = 1;
            } else {
                std::printf("[smoke] cancel-reset-reuse OK reused_hash=%016llx fresh_cache_hash=%016llx\n",
                    (unsigned long long) logits_hash(reused_logits),
                    (unsigned long long) logits_hash(fresh_logits));
            }
            free_qwen4exp_cache(fresh);
        }
    }

    free_qwen4exp_cache(cache);
    free_qwen4exp_weights(w);
    ggml_backend_free(backend);

    std::printf("[smoke] %s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
