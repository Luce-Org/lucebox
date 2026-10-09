// Smoke test for the Qwen3.8-Flash-Next (`qwen4exp`) forward path.
//
// Loads shard 1 (shard 2, the lazy PLE table, is discovered from the shard-1
// filename), builds the KV + delta-net cache, runs a prefill chunk and one
// decode step, and checks the logits are finite and the expected size. Mirrors
// smoke_qwen3_forward.cpp; no daemon involved.
//
// Usage:
//   smoke_qwen4exp_forward <shard1.gguf> [seq_len=16]
//
// --mtp N (needs the MTP sidecar): greedy-decodes N tokens after the seq_len-token prompt with plain
// decode and with MTP speculation and fails unless both give the same tokens from bit-identical logits. Use a
// seq_len past 2052 to cover QSA decode, e.g.
//   smoke_qwen4exp_forward <shard1.gguf> 2200 --mtp 128 --mtp-draft 4
// Also forces every retained prefix at all four block alignments and compares
// cache bytes plus replacement-token logits, including QSA block recompletion.
//   smoke_qwen4exp_forward <shard1.gguf> [seq_len=16] [--token-file FILE] [--split N[:chunk]] [--chunk N] [--compare-chunk N]

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
#include <cstdlib>
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

// --mtp: greedy-decode n_gen tokens after `prompt` twice -- plain T=1 steps, then MTP speculation
// (k chained drafts, k+1-token verify, rollback on reject) -- and require the same tokens from bit-identical logits at every
// position. Prefill runs in chunks of --chunk (default: the whole prompt).
bool same_cache(const Qwen4ExpCache & a, const Qwen4ExpCache & b, int tokens, bool same_pooled_count);

std::vector<char> mtp_kv_bytes(const Qwen4ExpCache & c, int pos, int n) {
        std::vector<char> bytes;
        if (n == 0) return bytes;
        for (ggml_tensor * t : {c.mtp_k, c.mtp_v}) {
            const size_t size = (size_t) n * t->nb[1];
            for (int64_t h = 0; h < t->ne[2]; ++h) {
                const size_t start = bytes.size();
                bytes.resize(start + size);
                ggml_backend_tensor_get(t, bytes.data() + start, h * t->nb[2] + (size_t) pos * t->nb[1], size);
            }
        }
        return bytes;
    }

int run_mtp_check(ggml_backend_t backend, const Qwen4ExpWeights & w, const std::vector<int32_t> & prompt, int n_gen, int k, int chunk, bool adaptive = false, int window = 0) {
    Qwen4ExpCache cache;
    const int S = (int) prompt.size();
    if (!w.mtp_eh_proj || n_gen < 1 ||
        !create_qwen4exp_cache(backend, w, S + n_gen + 4, cache, /*mtp=*/true, k) ||
        !qwen4exp_verify_supported(cache)) {
        std::fprintf(stderr, "[smoke] mtp: needs a sidecar (--draft PATH or <repo>/MTP/mtp-*.gguf) and the default graph\n");
        free_qwen4exp_cache(cache);
        return 1;
    }
    Qwen4ExpCache folded;
    if (!create_qwen4exp_cache(backend, w, S + n_gen + 4, folded, /*mtp=*/true, k)) {
        free_qwen4exp_cache(cache);
        return 1;
    }
    cache.mtp_window = folded.mtp_window = window;
    if (chunk == 0) chunk = S;
    const size_t V = (size_t) w.n_vocab, hd = (size_t) w.n_embd * w.n_hc;
    auto top = [&](const float * row) { return (int32_t) (std::max_element(row, row + V) - row); };
    std::vector<float> logits, hidden, mtp_h, mtp_logits;
    std::vector<int32_t> mtp_tok;
    int mtp_pos = 0;
    bool prefill_kv_identical = true;

    // Full MTP is the oracle: compare the actual F16 K/V bytes for every prompt
    // pair, including the pair carried across a trunk chunk boundary.


    // Prefill; with `mtp` also the draft layer's catch-up over the prompt (as Qwen4ExpBackend::generate_impl).
    auto prefill = [&](bool mtp) {
        reset_qwen4exp_state(backend, cache);
        if (mtp) reset_qwen4exp_state(backend, folded);
        mtp_h.clear();
        mtp_pos = 0;
        for (int pos = 0; pos < S; pos += chunk) {
            const int n = std::min(chunk, S - pos);
            if (!qwen4exp_forward(backend, w, cache, prompt.data() + pos, n, pos, logits, mtp ? &hidden : nullptr).ok) {
                return false;
            }
            if (!mtp) continue;
            mtp_tok.assign(prompt.begin() + pos + (pos == 0 ? 1 : 0), prompt.begin() + pos + n);
            mtp_h.insert(mtp_h.end(), hidden.begin(), hidden.end());
            const int n_pairs = (int) mtp_tok.size();
            if (n_pairs > 0 && !qwen4exp_mtp_forward(backend, w, cache, mtp_tok.data(), mtp_h.data(), n_pairs, mtp_pos,
                                                     mtp_logits)) {
                return false;
            }
            if (n_pairs > 0) {
                const auto expected = mtp_kv_bytes(cache, mtp_pos, n_pairs);
                // Poison the destination so a missing/partial write cannot pass on stale oracle bytes.
                for (ggml_tensor * t : {cache.mtp_k, cache.mtp_v}) {
                    for (int64_t h = 0; h < t->ne[2]; ++h) {
                        ggml_backend_tensor_memset(t, 0xa5, h * t->nb[2] + (size_t) mtp_pos * t->nb[1],
                                                   (size_t) n_pairs * t->nb[1]);
                    }
                }
                if (!qwen4exp_mtp_forward(backend, w, cache, mtp_tok.data(), mtp_h.data(), n_pairs, mtp_pos,
                                          mtp_logits, nullptr, /*kv_only=*/true)) return false;
                prefill_kv_identical = expected == mtp_kv_bytes(cache, mtp_pos, n_pairs) && mtp_logits.empty();
                if (!prefill_kv_identical) return false;
            }
            // Independently advance the new server path. Poison its destination,
            // then compare against the unchanged standalone/full MTP oracle.
            for (ggml_tensor * t : {folded.mtp_k, folded.mtp_v}) {
                for (int64_t h = 0; h < t->ne[2] && n_pairs > 0; ++h) {
                    ggml_backend_tensor_memset(t, 0xa5, h * t->nb[2] + (size_t) mtp_pos * t->nb[1],
                                               (size_t) n_pairs * t->nb[1]);
                }
            }
            std::vector<float> folded_logits, pending;
            if (!qwen4exp_forward(backend, w, folded, prompt.data() + pos, n, pos,
                                  folded_logits, &pending, false, n > 1).ok) return false;
            if (n == 1 && n_pairs > 0 && !qwen4exp_mtp_forward(backend, w, folded, mtp_tok.data(), mtp_h.data(),
                                                              n_pairs, mtp_pos, mtp_logits, nullptr, true)) return false;
            const bool trunk_same = same_bits(logits, folded_logits) && same_cache(cache, folded, pos + n, true);
            const bool pending_same = pending.size() == hd &&
                std::memcmp(pending.data(), hidden.data() + hidden.size() - hd, hd * sizeof(float)) == 0;
            prefill_kv_identical = mtp_kv_bytes(cache, mtp_pos, n_pairs) == mtp_kv_bytes(folded, mtp_pos, n_pairs);
            if (!trunk_same || !pending_same || !prefill_kv_identical) {
                std::fprintf(stderr, "[smoke] folded prefill pos=%d n=%d trunk_identical=%d pending_identical=%d kv_identical=%d\n",
                    pos, n, (int) trunk_same, (int) pending_same, (int) prefill_kv_identical);
                return false;
            }
            mtp_h.erase(mtp_h.begin(), mtp_h.begin() + (std::ptrdiff_t) ((size_t) n_pairs * hd));
            mtp_pos += n_pairs;
        }
        if (mtp) std::swap(cache, folded); // decode/rollback must consume the new path's cache
        return true;
    };

    std::vector<int32_t> ref, out;
    std::vector<float> ref_rows;
    int first_logits = -1;
    bool ok = prefill(false);
    if (ok) {
        int32_t next = top(logits.data());
        ref.push_back(next);
        ref_rows.insert(ref_rows.end(), logits.begin(), logits.end());
        for (int pos = S; ok && (int) ref.size() < n_gen; ++pos) {
            ok = qwen4exp_forward(backend, w, cache, &next, 1, pos, logits).ok && all_finite(logits);
            next = top(logits.data());
            ref.push_back(next);
            ref_rows.insert(ref_rows.end(), logits.begin(), logits.end());
        }
    }

    long long drafts = 0, accepted = 0, steps = 0;
    const int configured_k = cache.mtp_draft;
    auto policy = qwen4exp_mtp_width_controller();
    auto emit = [&](const float * row) {
        const size_t index = out.size();
        out.push_back(top(row));
        if (first_logits < 0 && (ref_rows.size() < (index + 1) * V ||
            std::memcmp(row, ref_rows.data() + index * V, V * sizeof(float)) != 0)) first_logits = (int) index;
    };
    const auto t0 = std::chrono::steady_clock::now();
    ok = ok && prefill(true);
    if (ok) {
        int32_t next = top(logits.data());
        emit(logits.data());
        mtp_tok.assign(1, next);
        int pos = S;
        std::vector<int32_t> draft_tokens;
        while (ok && (int) out.size() < n_gen) {
            const int width = adaptive ? policy.next_width_cost_aware({}, configured_k + 1) : configured_k + 1;
            const int k = std::min(width - 1, n_gen - (int) out.size() - 1);
            const bool verify = k > 0;
            std::vector<char> catchup_kv;
            if (verify) {
                ok = qwen4exp_mtp_forward(backend, w, cache, mtp_tok.data(), mtp_h.data(),
                                           (int) mtp_tok.size(), mtp_pos, mtp_logits);
                catchup_kv = mtp_kv_bytes(cache, mtp_pos, (int) mtp_tok.size());
                for (ggml_tensor * t : {cache.mtp_k, cache.mtp_v}) for (int64_t h = 0; h < t->ne[2]; ++h)
                    ggml_backend_tensor_memset(t, 0xa5, h * t->nb[2] + (size_t) mtp_pos * t->nb[1],
                                              mtp_tok.size() * t->nb[1]);
                ok = ok && qwen4exp_mtp_draft(backend, w, cache, mtp_tok.data(), mtp_h.data(),
                                              (int) mtp_tok.size(), mtp_pos, k, draft_tokens);
                ok = ok && catchup_kv == mtp_kv_bytes(cache, mtp_pos, (int) mtp_tok.size());
                // Once per run, independently chain CPU embeddings/hidden with
                // the full head. Allow only tied argmax alternatives. Window
                // arms have a deliberately different draft oracle; target and
                // catch-up K/V still have the exact checks above and below.
                if (ok && steps == 0 && window == 0) {
                    std::vector<float> chain_h, chain_logits;
                    for (int rank = 0; ok && rank < k; ++rank) {
                        ok = qwen4exp_mtp_forward(backend, w, cache,
                            rank ? &draft_tokens[rank - 1] : mtp_tok.data(), rank ? chain_h.data() : mtp_h.data(),
                            rank ? 1 : (int) mtp_tok.size(), rank ? mtp_pos + (int) mtp_tok.size() + rank - 1 : mtp_pos,
                            chain_logits, &chain_h, false, true);
                        if (!ok || !all_finite(chain_logits)) { ok = false; break; }
                        const int32_t chosen = draft_tokens[rank];
                        const auto & ids = w.mtp_vocab_ids;
                        const auto best = *std::max_element(ids.begin(), ids.end(),
                            [&](int a, int b) { return chain_logits[a] < chain_logits[b]; });
                        ok = std::binary_search(ids.begin(), ids.end(), chosen) && chain_logits[chosen] == chain_logits[best];
                    }
                    std::vector<float> device_h(hd);
                    ggml_backend_tensor_get(cache.mtp_chain_hidden, device_h.data(), 0, hd * sizeof(float));
                    ok = ok && same_bits(chain_h, device_h);
                    std::printf("[smoke] mtp device S=%d k=%d ranks=%d chain_identical=%d\n", S, configured_k, k, (int) ok);
                }
            }
            std::array<int32_t, QWEN4EXP_MTP_MAX_VERIFY> in{}, samples{};
            in[0] = next;
            if (verify) std::copy(draft_tokens.begin(), draft_tokens.end(), in.begin() + 1);
            ok = ok && qwen4exp_forward(backend, w, cache, in.data(), k + 1, pos, logits,
                                        verify ? &hidden : nullptr, verify).ok;
            if (!ok || !all_finite(logits)) { ok = false; break; }
            ++steps;
            drafts += k;
            for (int i = 0; i <= k; ++i) samples[i] = top(logits.data() + (size_t) i * V);
            const auto decision = qwen4exp_mtp_accept(draft_tokens.data(), k, samples.data(), k + 1);
            accepted += decision.n_accepted;
            const int retained = decision.n_emitted;
            for (int i = 0; i < retained; ++i) emit(logits.data() + (size_t) i * V);
            next = decision.emitted[retained - 1];
            if (verify) {
                mtp_h.assign(hidden.begin(), hidden.begin() + (std::ptrdiff_t) ((size_t) retained * hd));
                mtp_tok.assign(decision.emitted.begin(), decision.emitted.begin() + retained);
                mtp_pos = pos;
                ok = qwen4exp_verify_rollback(backend, w, cache, pos, retained);
            }
            pos += retained;
            // Oracle work is deliberately much heavier than a serving cycle.
            if (verify) policy.observe(decision.n_accepted + 1, k + 1);
        }
    }
    const double spec_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (ok && configured_k == QWEN4EXP_MTP_MAX_DRAFT) ok = prefill(true); // reset/reuse must discard the preceding prompt's carried row
    std::printf("[smoke] mtp prefill S=%d chunk=%d kv_identical=%d\n", S, chunk, (int) (ok && prefill_kv_identical));
    free_qwen4exp_cache(cache);
    free_qwen4exp_cache(folded);

    int first_token = -1;
    for (size_t i = 0; ok && i < ref.size() && i < out.size(); ++i) {
        if (first_token < 0 && ref[i] != out[i]) first_token = (int) i;
    }
    const bool same = ok && ref.size() == out.size() && first_token < 0 && first_logits < 0;
    std::printf("[smoke] mtp S=%d k=%d n_gen=%d ok=%d tokens_identical=%d logits_identical=%d first_token_diff=%d "
                "first_logits_diff=%d drafts=%lld accepted=%lld rate=%.3f tokens_per_step=%.3f spec_s=%.2f adaptive=%d\n",
                S, configured_k, n_gen, (int) ok, (int) (ok && ref.size() == out.size() && first_token < 0),
                (int) (ok && ref.size() == out.size() && first_logits < 0), first_token,
                first_logits, drafts, accepted, drafts ? (double) accepted / (double) drafts : 0.0,
                steps ? (double) (out.size() - 1) / (double) steps : 0.0, spec_s, (int) adaptive);
    return same ? 0 : 1;
}

// Compare only authoritative cache rows; reset deliberately leaves unused K/V intact.
bool same_cache(const Qwen4ExpCache & a, const Qwen4ExpCache & b, int tokens, bool same_pooled_count = true) {
    if ((same_pooled_count && a.indexer_blocks != b.indexer_blocks) || a.ple_prev != b.ple_prev) return false;
    auto tensors_equal = [](const std::vector<ggml_tensor *> & x,
                            const std::vector<ggml_tensor *> & y, int rows) {
        if (x.size() != y.size()) return false;
        for (size_t i = 0; i < x.size(); ++i) {
            if (!x[i] || !y[i]) { if (x[i] != y[i]) return false; else continue; }
            const size_t bytes = rows < 0 ? ggml_nbytes(x[i]) : (size_t) rows * x[i]->nb[1];
            std::vector<char> xb(bytes), yb(bytes);
            for (int64_t h = 0; h < (rows < 0 ? 1 : x[i]->ne[2]); ++h) {
                ggml_backend_tensor_get(x[i], xb.data(), h * x[i]->nb[2], bytes);
                ggml_backend_tensor_get(y[i], yb.data(), h * y[i]->nb[2], bytes);
                if (xb != yb) return false;
            }
        }
        return true;
    };
    return tensors_equal(a.attn_k, b.attn_k, tokens) && tensors_equal(a.attn_v, b.attn_v, tokens) &&
        tensors_equal(a.indexer_raw, b.indexer_raw, tokens) &&
        tensors_equal(a.indexer_k, b.indexer_k, std::min(a.indexer_blocks, b.indexer_blocks)) &&
        tensors_equal(a.ssm_state, b.ssm_state, -1) && tensors_equal(a.conv_state, b.conv_state, -1) &&
        tensors_equal(a.ple_conv_state, b.ple_conv_state, -1);
}

// Exercise every retained prefix and every block alignment, independent of the
// model's natural acceptance rate. After truncation replace the first rejected
// input and finish its block; compare authoritative state and logits with AR.
int run_mtp_rollback_check(ggml_backend_t backend, const Qwen4ExpWeights & w,
                           const std::vector<int32_t> & prompt, int k, int chunk) {
    Qwen4ExpCache cache, reference;
    const int S = (int) prompt.size();
    const int capacity = S + 512;
    bool ok = create_qwen4exp_cache(backend, w, capacity, cache, true, k) &&
              create_qwen4exp_cache(backend, w, capacity, reference);
    std::vector<float> actual, expected, verified, hidden, draft_logits;
    for (auto * t : {cache.mtp_k, cache.mtp_v}) if (t) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
    // Honor --chunk: a monolithic long prompt's compute graph cannot be allocated (~25.6 GiB at S=70000).
    if (chunk <= 0) chunk = S;
    for (int p = 0; ok && p < S; p += chunk) {
        const int n = std::min(chunk, S - p);
        ok = qwen4exp_forward(backend, w, cache, prompt.data() + p, n, p, actual).ok &&
             qwen4exp_forward(backend, w, reference, prompt.data() + p, n, p, expected).ok;
    }
    if (!ok) std::fprintf(stderr, "[smoke] rollback setup failed S=%d chunk=%d (cache/prefill)\n", S, chunk);
    int pos = S, cases = 0;
    auto advance = [&](int32_t token) {
        const bool same = qwen4exp_forward(backend, w, cache, &token, 1, pos, actual).ok &&
                          qwen4exp_forward(backend, w, reference, &token, 1, pos, expected).ok &&
            actual.size() == expected.size() && all_finite(actual) &&
            std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0 &&
            cache.cur_pos == pos + 1 && cache.indexer_blocks <= (pos + 1) / 4 &&
            (cache.indexer_blocks == reference.indexer_blocks || reference.indexer_blocks == 0) &&
            same_cache(cache, reference, pos + 1, false);
        ++pos;
        return same;
    };
    for (int retained = 1; ok && retained <= k + 1; ++retained) {
        for (int alignment = 0; ok && alignment < 4; ++alignment) {
            while (ok && pos % 4 != alignment) ok = advance((pos * 7919 + 13) % w.n_vocab);
            std::array<int32_t, QWEN4EXP_MTP_MAX_VERIFY> in{};
            for (int i = 0; i <= k; ++i) in[i] = ((pos + i) * 7919 + 13) % w.n_vocab;
            ok = ok && qwen4exp_forward(backend, w, cache, in.data(), k + 1, pos, verified, &hidden, true).ok;
            if (ok) {
                ok = qwen4exp_mtp_forward(backend, w, cache, in.data(), hidden.data(), retained, pos, draft_logits);
                const auto kv = mtp_kv_bytes(cache, pos, retained);
                for (ggml_tensor * t : {cache.mtp_k, cache.mtp_v}) for (int64_t h = 0; h < t->ne[2]; ++h)
                    ggml_backend_tensor_memset(t, 0xa5, h * t->nb[2] + (size_t) pos * t->nb[1], retained * t->nb[1]);
                std::vector<int32_t> unused;
                ok = ok && qwen4exp_mtp_draft(backend, w, cache, in.data(), hidden.data(), retained, pos, 1, unused);
                ok = ok && kv == mtp_kv_bytes(cache, pos, retained);
            }
            for (int i = 0; ok && i < retained; ++i) {
                ok = qwen4exp_forward(backend, w, reference, &in[i], 1, pos + i, expected).ok &&
                     all_finite(expected) &&
                     std::memcmp(verified.data() + (size_t) i * w.n_vocab, expected.data(),
                                 expected.size() * sizeof(float)) == 0;
            }
            ok = ok && qwen4exp_verify_rollback(backend, w, cache, pos, retained) &&
                 cache.cur_pos == pos + retained && cache.indexer_blocks <= (pos + retained) / 4 &&
                 // Verify may bootstrap QSA ahead of a still-dense reference.
                 (cache.indexer_blocks == reference.indexer_blocks || reference.indexer_blocks == 0) &&
                 same_cache(cache, reference, pos + retained, false);
            pos += retained;
            // Four replacements guarantee that a rejected block completion is
            // recomputed, even when the retained position is block-aligned.
            for (int i = 0; ok && i < 4; ++i) ok = advance((pos * 7919 + 14) % w.n_vocab);
            if (ok) ++cases;
            else std::fprintf(stderr, "[smoke] rollback mismatch k=%d retained=%d alignment=%d pos=%d\n",
                              k, retained, alignment, pos);
        }
    }
    std::printf("[smoke] mtp rollback S=%d k=%d cases=%d/%d cache_and_logits_identical=%d\n",
                S, k, cases, 4 * (k + 1), (int) ok);
    free_qwen4exp_cache(cache);
    free_qwen4exp_cache(reference);
    std::printf("[smoke] mtp catchup S=%d k=%d cases=%d/%d kv_identical=%d\n", S, k, cases, 4 * (k + 1), (int) ok);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <shard1.gguf> [seq_len=16] [--token-file FILE] [--split N[:chunk]] [--draft PATH|0] [--mtp N] [--mtp-draft 1..7] [--mtp-vocab 40000|64000|106000] [--mtp-window 32768] [--mtp-all] [--chunk N] [--tg N] [--stable N] [--compare-chunk N] [--expert-gpu N]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    int S = 16, N = 0, step = 1;
    int n_gen = 1, mtp_gen = 0, mtp_draft = 1, chunk = 0, stable = 0;
    int mtp_vocab = QWEN4EXP_MTP_VOCAB, mtp_window = 0;
    int compare_chunk = 0;
    bool mtp_all = false;
    std::string draft;
    const char * token_file = nullptr;
    int expert_gpu = -1;
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
        if (opt == "--mtp-all") mtp_all = true;
        else if ((opt == "--mtp-vocab" || opt == "--mtp-window") && arg + 1 < argc) {
            const char * val = argv[++arg];
            int & value = opt == "--mtp-vocab" ? mtp_vocab : mtp_window;
            if (!positive(val, val + std::strlen(val), value)) return 2;
        }
        else if (opt == "--draft" && arg + 1 < argc) draft = argv[++arg];
        else if ((opt == "--tg" || opt == "--mtp" || opt == "--mtp-draft" || opt == "--chunk" || opt == "--stable") && arg + 1 < argc) {
            int & value = opt == "--tg" ? n_gen : opt == "--mtp" ? mtp_gen :
                          opt == "--mtp-draft" ? mtp_draft : opt == "--chunk" ? chunk : stable;
            const char * val = argv[++arg];
            if (!positive(val, val + std::strlen(val), value)) return 2;
        }
        else if (opt == "--compare-chunk" && arg + 1 < argc) {
            const char * val = argv[++arg];
            if (!positive(val, val + std::strlen(val), compare_chunk)) return 2;
        }
        else if (opt == "--token-file" && arg + 1 < argc) token_file = argv[++arg];
        else if (opt == "--expert-gpu" && arg + 1 < argc) expert_gpu = std::atoi(argv[++arg]);
        else if (opt == "--split" && arg + 1 < argc) {
            const char * val = argv[++arg], * end = val + std::strlen(val), * colon = std::strchr(val, ':');
            if (!positive(val, colon ? colon : end, N) || N >= S ||
                (colon && !positive(colon + 1, end, step))) return 2;
        } else { std::fprintf(stderr, "invalid option: %s\n", argv[arg]); return 2; }
    }
    if ((mtp_vocab != 40000 && mtp_vocab != 64000 && mtp_vocab != 106000) ||
        (mtp_window != 0 && mtp_window != 32768) ||
        mtp_draft > QWEN4EXP_MTP_MAX_DRAFT || stable >= S ||
        (mtp_all && (mtp_gen == 0 || S < 16))) return 2;
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "[smoke] no GPU backend available\n");
        return 77;
    }

    Qwen4ExpWeights w;
    // As the server's --expert-device: the routed experts on that device (split mode).
    if (expert_gpu >= 0) {
        w.expert_backend = ggml_backend_cuda_init(expert_gpu);
        if (!w.expert_backend) return 77;
        w.expert_gfx1151 = ggml_backend_cuda_qwen4exp_supported(w.expert_backend);
    }
    auto t_load0 = std::chrono::steady_clock::now();
    if (!load_qwen4exp_gguf(path, backend, w, draft, mtp_vocab)) {
        std::fprintf(stderr, "[smoke] load_qwen4exp_gguf failed\n");
        ggml_backend_free(backend);
        return 1;
    }
    auto t_load1 = std::chrono::steady_clock::now();
    std::printf("[smoke] load %.2fs layers=%d vocab=%d shard2=%s\n",
        std::chrono::duration<double>(t_load1 - t_load0).count(),
        w.n_layer, w.n_vocab, w.ple_reader.available() ? "yes" : "no");

    Qwen4ExpCache cache;
    if (!create_qwen4exp_cache(backend, w, S + std::max(compare_chunk ? 132 : 4, n_gen), cache)) {
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
    auto prefill = [&](Qwen4ExpCache & state, int length, std::vector<float> & logits) {
        Qwen4ExpForwardResult r;
        const int stride = chunk > 0 ? chunk : length;
        for (int pos = 0; pos < length; pos += stride) {
            r = qwen4exp_forward(backend, w, state, tokens.data() + pos,
                                std::min(stride, length - pos), pos, logits);
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
            return prefill(cache, S, logits).ok && all_finite(logits);
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
    const Qwen4ExpForwardResult pre = prefill(cache, S, logits);
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
        auto d0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n_gen; ++i) {
            const int32_t next = (int32_t) argmax(logits);
            const auto dec = qwen4exp_forward(backend, w, cache, &next, 1, S + i, logits);
            if (!dec.ok || logits.size() != (size_t) w.n_vocab || !all_finite(logits)) {
                std::fprintf(stderr, "[smoke] decode FAILED pos=%d\n", S + i);
                rc = 1;
                break;
            }
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - d0).count();
        std::printf("[smoke] tg%d ok=%d %.3fs %.3f tok/s argmax=%d\n",
            n_gen, rc == 0, seconds, n_gen / seconds, argmax(logits));
    }

    // Optional exact differential: identical prefill, then every T=1 step of the
    // replayed stable QSA graph vs one built fresh at that step (including cache bits).
    if (rc == 0 && stable > 0) {
        const int n = stable, start = S - n;
        Qwen4ExpCache reference;
        bool ok = n > 0 && n < S && create_qwen4exp_cache(backend, w, cache.max_ctx, reference);
        std::vector<float> expected, actual;
        reset_qwen4exp_state(backend, cache);
        ok = ok && qwen4exp_forward(backend, w, cache, tokens.data(), start, 0, actual).ok &&
                   qwen4exp_forward(backend, w, reference, tokens.data(), start, 0, expected).ok;
        int replays = 0;
        for (int p = start; ok && p < S; ++p) {
            clear_qwen4exp_decode_workspace(reference.decode_workspace);   // a fresh build every step
            ok = qwen4exp_forward(backend, w, reference, &tokens[p], 1, p, expected).ok;
            const auto & ws = cache.decode_workspace;
            const bool replay = ws.gf && ws.qsa_blocks > 0 && ws.next_pos == p &&
                                cache.indexer_blocks == p / 4 && p + 1 <= ws.kv_bucket;
            const uint64_t builds = ws.builds, prior_replays = ws.replays;
            ok = ok && qwen4exp_forward(backend, w, cache, &tokens[p], 1, p, actual).ok;
            ok = ok && actual.size() == expected.size() &&
                 std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0 &&
                 same_cache(cache, reference, p + 1) &&
                 (!replay || (ws.builds == builds && ws.replays == prior_replays + 1));
            replays += replay;
            if (!ok) std::fprintf(stderr, "[smoke] stable/rebuild mismatch at pos=%d\n", p);
        }
        free_qwen4exp_cache(reference);
        std::printf("[smoke] stable/rebuild bits ok=%d QSA replays=%d\n", (int) ok, replays);
        if (!ok || replays == 0) rc = 1;
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
        bool ok = N > 0 && N < S && prefill(cache, S, full).ok;
        reset_qwen4exp_state(backend, cache);
        ok = ok && prefill(cache, S - N, split).ok;
        for (int i = S - N; ok && i < S; i += step) {
            const int n = std::min(step, S - i);
            ok = qwen4exp_forward(backend, w, cache, &tokens[i], n, i, split).ok;
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

    if (rc == 0 && mtp_gen > 0) {
        // Load the model once for all widths and both depths.
        for (int n : mtp_all ? std::vector<int>{16, S} : std::vector<int>{S}) {
            const std::vector<int32_t> prompt(tokens.begin(), tokens.begin() + n);
            for (int k = mtp_all ? 1 : mtp_draft; k <= (mtp_all ? QWEN4EXP_MTP_MAX_DRAFT : mtp_draft); ++k) {
                const int decode_rc = run_mtp_check(backend, w, prompt, mtp_gen, k, chunk, false, mtp_window);
                // Independent caches: report rollback even when natural drafting differs.
                const int rollback_rc = run_mtp_rollback_check(backend, w, prompt, k, chunk);
                rc |= decode_rc || rollback_rc;
            }
            if (mtp_all) rc |= run_mtp_check(backend, w, prompt, mtp_gen, QWEN4EXP_MTP_MAX_DRAFT, chunk, true, mtp_window);
        }
        if (mtp_all && S > 2048) {
            // A 2048-row chunk followed by one row exercises the server fallback.
            rc |= run_mtp_check(backend, w, std::vector<int32_t>(tokens.begin(), tokens.begin() + 2049),
                                mtp_gen, QWEN4EXP_MTP_MAX_DRAFT, 2048, false, mtp_window);
        }
    }

    // A reset/reuse must match a freshly allocated cache for the same prefix.
    if (rc == 0) {
        Qwen4ExpCache fresh;
        if (!create_qwen4exp_cache(backend, w, cache.max_ctx, fresh)) {
            std::fprintf(stderr, "[smoke] fresh cache creation failed\n");
            rc = 1;
        } else {
            reset_qwen4exp_state(backend, cache);
            reset_qwen4exp_state(backend, fresh);
            std::vector<float> tmp, reused_logits, fresh_logits;
            const bool p1 = prefill(cache, S, tmp).ok;
            const bool p2 = prefill(fresh, S, tmp).ok;
            const int32_t reuse_token = 77 % w.n_vocab;
            const Qwen4ExpForwardResult rr = qwen4exp_forward(backend, w, cache, &reuse_token, 1, S, reused_logits);
            const Qwen4ExpForwardResult fr = qwen4exp_forward(backend, w, fresh, &reuse_token, 1, S, fresh_logits);
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
    if (w.expert_backend) ggml_backend_free(w.expert_backend);
    ggml_backend_free(backend);

    std::printf("[smoke] %s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
