// Smoke test for the Qwen3.8-Flash-Next (`qwen4exp`) forward path.
//
// Loads shard 1 (shard 2, the lazy PLE table, is discovered from the shard-1
// filename), builds the KV + delta-net cache, runs a prefill chunk and one
// decode step, and checks the logits are finite and the expected size. Mirrors
// smoke_qwen3_forward.cpp; no daemon involved.
//
// Usage:
//   smoke_qwen4exp_forward <shard1.gguf> [seq_len=16] [--token-file FILE] [--split N[:chunk]] [--reference] [--dump]

#include "qwen4exp_internal.h"
#include "qwen4exp_graph.h"
#include "qwen4exp_cache.h"

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
        std::fprintf(stderr, "usage: %s <shard1.gguf> [seq_len=16] [--token-file FILE] [--split N[:chunk]] [--reference] [--dump]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    int S = 16, N = 0, step = 1;
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
    if (!create_qwen4exp_cache(backend, w, S + 4, GGML_TYPE_F16, cache, reference)) {
        std::fprintf(stderr, "[smoke] create_qwen4exp_cache failed\n");
        free_qwen4exp_weights(w);
        ggml_backend_free(backend);
        return 1;
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
    std::vector<float> logits;
    auto t0 = std::chrono::steady_clock::now();
    const Qwen4ExpForwardResult pre = qwen4exp_forward(backend, w, cache, tokens.data(), S, 0, logits, dump);
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

    // --split N[:c]: the same S tokens as one prefill vs a prefill of S-N plus the last N tokens in
    // chunks of c (default 1, i.e. decode) must give the same last-position distribution. QSA selection is
    // chunk-invariant, so past the block budget this checks decode (c=1) against prefill; c in 9..127 runs dense
    // attention there and shows how far a wrong selection drifts.
    if (rc == 0 && N > 0) {
        std::vector<float> full, split;
        reset_qwen4exp_state(backend, cache);
        bool ok = N > 0 && N < S && qwen4exp_forward(backend, w, cache, tokens.data(), S, 0, full, dump).ok;
        reset_qwen4exp_state(backend, cache);
        ok = ok && qwen4exp_forward(backend, w, cache, tokens.data(), S - N, 0, split, dump).ok;
        for (int i = S - N; ok && i < S; i += step) {
            const int n = std::min(step, S - i);
            ok = qwen4exp_forward(backend, w, cache, &tokens[i], n, i, split, dump).ok;
        }
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
        if (!create_qwen4exp_cache(backend, w, S + 4, GGML_TYPE_F16, fresh, reference)) {
            std::fprintf(stderr, "[smoke] fresh cache creation failed\n");
            rc = 1;
        } else {
            reset_qwen4exp_state(backend, cache);
            reset_qwen4exp_state(backend, fresh);
            std::vector<float> tmp, reused_logits, fresh_logits;
            const bool p1 = qwen4exp_forward(backend, w, cache, tokens.data(), S, 0, tmp, dump).ok;
            const bool p2 = qwen4exp_forward(backend, w, fresh, tokens.data(), S, 0, tmp, dump).ok;
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
