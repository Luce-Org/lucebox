// Laguna daemon library implementation. Hosts the Poolside Laguna-XS.2
// target on a fresh CUDA backend and services stdin commands. See
// laguna_daemon.h for the protocol overview.
//
// Invoked from two places:
//   - test/test_dflash.cpp: arch dispatch — when the GGUF reports
//     `general.architecture == "laguna"`, main() builds a LagunaDaemonArgs
//     and hands off here, bypassing the qwen35 + DFlash + DDTree code path.
//   - test/test_laguna_daemon.cpp: legacy thin wrapper kept for the NIAH
//     driver (scripts/laguna_pflash_niah.py) which spawns the binary
//     directly with `--max-ctx`/`--kv`/`--chunk`/`--stream-fd` flags.

#include "laguna_daemon.h"

#include "laguna_internal.h"
#include "internal.h"
#include "dflash27b.h"
#include "sampler.h"          // shared CPU sampler chain (SamplerCfg /
                              // sample_logits / parse_sampler_token), defined
                              // once in src/sampler.cpp.
#include "qwen3_drafter.h"   // Qwen3-0.6B drafter — same one used by
                              // pflash_daemon and the qwen35 compress dance.
                              // Loaded lazily on the first `compress` command
                              // so a server that never asks for compression
                              // never pays the drafter weight load.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <array>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cuda.h"

#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#include <fcntl.h>
#define ssize_t long
#endif

namespace dflash27b {

namespace {

// laguna_serve.py + laguna_pflash_niah.py write prompts as a uint32 length
// prefix followed by N int32 token IDs. Used by the legacy `generate` path.
std::vector<int32_t> read_counted_i32(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    uint32_t n = 0;
    f.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (!f) return {};
    std::vector<int32_t> ids((size_t)n);
    if (n > 0) {
        f.read(reinterpret_cast<char *>(ids.data()), (std::streamsize)ids.size() * sizeof(int32_t));
        if (!f) return {};
    }
    return ids;
}

// scripts/server.py writes prompts as a raw int32 stream (no length prefix);
// the file size implies the token count. Used by the bare-prompt path so the
// daemon stays drop-in for the qwen35 server.py protocol.
std::vector<int32_t> read_uncounted_i32(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const auto sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<int32_t> ids(sz / sizeof(int32_t));
    if (!ids.empty()) {
        f.read(reinterpret_cast<char *>(ids.data()),
               (std::streamsize)ids.size() * sizeof(int32_t));
        if (!f) return {};
    }
    return ids;
}

bool write_counted_i32(const std::string & path, const std::vector<int32_t> & ids) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t n = (uint32_t)ids.size();
    f.write(reinterpret_cast<const char *>(&n), sizeof(n));
    if (n > 0) f.write(reinterpret_cast<const char *>(ids.data()), (std::streamsize)ids.size() * sizeof(int32_t));
    return (bool)f;
}

// laguna_step lives in src/laguna_target_graph.cpp as a public helper so the
// daemon, benches, and any future caller share one forward-step
// implementation. We just call it from the daemon loop below.

}  // namespace


int run_laguna_daemon(const LagunaDaemonArgs & args) {
    const bool no_mask = (std::getenv("DFLASH_NO_MASK") != nullptr);

    int stream_fd = args.stream_fd;
    auto emit_int32 = [&](int32_t v) {
        if (stream_fd < 0) return;
        const int32_t w = v;
        ssize_t n = ::write(stream_fd, &w, sizeof(w));
        (void)n;
    };

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    LagunaTargetWeights w;
    if (!load_target_gguf_laguna(args.target_path, backend, w)) {
        std::fprintf(stderr, "load failed: %s\n", dflash27b_last_error());
        ggml_backend_free(backend); return 1;
    }

    LagunaTargetCache cache;
    cache.kv_k_type = args.kv_type;
    cache.kv_v_type = args.kv_type;
    if (!create_laguna_target_cache(w, args.max_ctx, backend, cache)) {
        std::fprintf(stderr, "cache failed: %s\n", dflash27b_last_error());
        free_laguna_target_weights(w); ggml_backend_free(backend); return 1;
    }

    std::printf("[laguna-daemon] ready vocab=%lld eos=%d eot=%d max_ctx=%d kv=%s chunk=%d\n",
                (long long)w.embedder.n_vocab, w.eos_id, w.eos_chat_id, args.max_ctx,
                ggml_type_name(args.kv_type), args.chunk);
    std::fflush(stdout);

    std::mt19937_64 sampler_rng{std::random_device{}()};

    // PFlash compression state. The drafter (Qwen3-0.6B BF16 GGUF) is loaded
    // lazily on the first `compress` command and freed on `free drafter` /
    // shutdown. While the drafter is resident the target weights are parked
    // (released to host) so the two don'́t fight for VRAM on a 24 GB card:
    // target ~18.77 GiB + drafter ~1.2 GiB + scratch leaves no room for
    // activations. The lifecycle mirrors qwen35'́s daemon — server.py'́s
    // _prefill_hook drives `park target` -> `compress ...` -> `unpark target`
    // unchanged.
    DrafterContext drafter_ctx{};
    bool drafter_loaded = false;
    bool target_parked  = false;

    auto park_target = [&]() -> bool {
        if (target_parked) return true;
        free_laguna_target_cache(cache);
        free_laguna_target_weights(w);
        target_parked = true;
        std::printf("[park] target released\n"); std::fflush(stdout);
        return true;
    };
    auto unpark_target = [&]() -> bool {
        if (!target_parked) return true;
        if (!load_target_gguf_laguna(args.target_path, backend, w)) {
            std::fprintf(stderr, "[unpark] target: %s\n", dflash27b_last_error());
            return false;
        }
        cache.kv_k_type = args.kv_type;
        cache.kv_v_type = args.kv_type;
        if (!create_laguna_target_cache(w, args.max_ctx, backend, cache)) {
            std::fprintf(stderr, "[unpark] cache: %s\n", dflash27b_last_error());
            return false;
        }
        target_parked = false;
        std::printf("[unpark] target restored\n"); std::fflush(stdout);
        return true;
    };

    auto stream_emit = [&](int32_t v) {
        if (stream_fd < 0) return;
        ssize_t n = ::write(stream_fd, &v, sizeof(v));
        (void)n;
    };

    // Prefix-cache slots. Lazy-allocated on first SNAPSHOT into the slot;
    // server.py PrefixCache caps at DAEMON_MAX_SLOTS=8 so we mirror that.
    static constexpr int kMaxSlots = 8;
    std::array<LagunaCacheSnapshot, kMaxSlots> snapshots{};
    auto ensure_slot = [&](int slot) -> bool {
        if (slot < 0 || slot >= kMaxSlots) {
            std::fprintf(stderr, "[snap] slot=%d out of range\n", slot);
            return false;
        }
        if (target_parked) {
            std::fprintf(stderr, "[snap] target parked, cannot allocate snapshot\n");
            return false;
        }
        if (!snapshots[slot].ctx) {
            if (!laguna_snapshot_alloc(cache, backend, w.n_layer, args.max_ctx,
                                        w.n_head_kv, w.head_dim, snapshots[slot])) {
                std::fprintf(stderr, "[snap] alloc slot=%d: %s\n", slot,
                              dflash27b_last_error());
                return false;
            }
        }
        return true;
    };

    auto run_prompt = [&](const std::vector<int32_t> & prompt,
                          int n_gen,
                          const SamplerCfg & sampler,
                          bool do_sample,
                          bool stream,
                          double & pf_s_out,
                          double & g_s_out,
                          std::vector<int32_t> & generated_out) -> const char * {
        const int N = (int)prompt.size();
        if (N + n_gen > args.max_ctx) return "overflow";

        reset_laguna_target_cache(cache);

        std::vector<float> embed_pf((size_t)N * w.n_embd);
        if (!w.embedder.embed(prompt.data(), N, embed_pf.data())) return "embed_prefill";

        auto t_pf0 = std::chrono::steady_clock::now();
        std::vector<float> last_logits;
        bool ok = true;
        const int n_chunks = (N + args.chunk - 1) / args.chunk;
        for (int c = 0; c < n_chunks && ok; ++c) {
            const int kv_start = c * args.chunk;
            const int n_tok    = std::min(args.chunk, N - c * args.chunk);
            ok = laguna_step(backend, w, cache,
                              embed_pf.data() + (size_t)kv_start * w.n_embd,
                              n_tok, kv_start, no_mask, last_logits);
        }
        if (!ok) return "prefill";
        auto t_pf1 = std::chrono::steady_clock::now();
        pf_s_out = std::chrono::duration<double>(t_pf1 - t_pf0).count();

        auto argmax = [](const std::vector<float> & ll) {
            int best = 0; float bv = ll[0];
            for (size_t i = 1; i < ll.size(); ++i)
                if (ll[i] > bv) { bv = ll[i]; best = (int)i; }
            return best;
        };

        std::vector<int32_t> history;
        history.reserve((size_t)N + (size_t)n_gen);
        history.insert(history.end(), prompt.begin(), prompt.end());

        auto pick = [&](const std::vector<float> & ll) -> int {
            return do_sample
                ? sample_logits(ll.data(), (int)ll.size(), sampler, history, sampler_rng)
                : argmax(ll);
        };

        int next_tok = pick(last_logits);
        generated_out.clear();
        generated_out.reserve(n_gen);

        std::vector<float> embed_step((size_t)w.n_embd);
        auto t_g0 = std::chrono::steady_clock::now();
        for (int s = 0; s < n_gen; ++s) {
            if (next_tok == w.eos_id || next_tok == w.eos_chat_id) break;
            generated_out.push_back(next_tok);
            history.push_back(next_tok);
            if (stream) emit_int32(next_tok);
            if (!w.embedder.embed(&next_tok, 1, embed_step.data())) { ok = false; break; }
            std::vector<float> step_logits;
            if (!laguna_step(backend, w, cache, embed_step.data(), 1,
                              cache.cur_pos, no_mask, step_logits)) { ok = false; break; }
            next_tok = pick(step_logits);
        }
        auto t_g1 = std::chrono::steady_clock::now();
        g_s_out = std::chrono::duration<double>(t_g1 - t_g0).count();

        if (stream) emit_int32(-1);
        return ok ? nullptr : "decode";
    };

    auto looks_like_path = [](const std::string & s) {
        if (s.empty()) return false;
        if (s[0] == '/' || s[0] == '.') return true;
        return s.find('/') != std::string::npos;
    };

    auto starts_with = [](const std::string & s, const std::string & p) {
        return s.size() >= p.size() && std::memcmp(s.data(), p.data(), p.size()) == 0;
    };

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "quit" || line == "exit") break;

        // ---- Lifecycle commands (no sampler tail expected) -----------------
        // server.py'́s _prefill_hook calls `_send_and_ack(...)` which reads a
        // -1 sentinel off stream_fd after every park / unpark / compress /
        // free drafter. Each handler emits one when it is done.
        if (starts_with(line, "park")) {
            const bool want_target = (line == "park" || line == "park all" || line == "park target");
            // "park draft" is a no-op on the laguna path (no DFlash draft).
            if (want_target) park_target();
            stream_emit(-1);
            continue;
        }
        if (starts_with(line, "unpark")) {
            const bool want_target = (line == "unpark" || line == "unpark all" || line == "unpark target");
            if (want_target && !unpark_target()) { stream_emit(-1); continue; }
            stream_emit(-1);
            continue;
        }
        if (line == "LIST_SLOTS") {
            // Server-side prefix_cache.py:startup_sync() asks every freshly
            // spawned daemon for the list of slots already populated by a
            // prior run. We allocate snapshots lazily on demand so the list
            // is whatever is currently `used`. Reply protocol matches the
            // qwen35 daemon: a single line `[snap] slots=<csv>` (no -1
            // sentinel — server reads via stdout marker prefix).
            std::printf("[snap] slots=");
            bool first = true;
            for (int i = 0; i < kMaxSlots; ++i) {
                if (snapshots[i].used) {
                    std::printf("%s%d", first ? "" : ",", i);
                    first = false;
                }
            }
            std::printf("\n"); std::fflush(stdout);
            continue;
        }
        if (starts_with(line, "SNAPSHOT ")) {
            int slot = std::atoi(line.c_str() + 9);
            if (!ensure_slot(slot) ||
                !laguna_snapshot_save(cache, snapshots[slot])) {
                std::fprintf(stderr, "[snap] save slot=%d: %s\n",
                              slot, dflash27b_last_error());
            }
            // server-side _await_reply matches `[snap] inline slot=...`
            // (no stream_fd sentinel, mirrors qwen35).
            std::printf("[snap] inline slot=%d cur_pos=%d\n",
                         slot,
                         snapshots[slot].used ? snapshots[slot].cur_pos : -1);
            std::fflush(stdout);
            continue;
        }
        if (starts_with(line, "FREE_SNAPSHOT ")) {
            int slot = std::atoi(line.c_str() + 14);
            if (slot >= 0 && slot < kMaxSlots) {
                laguna_snapshot_free(snapshots[slot]);
            }
            // server-side _await_reply expects "[snap] freed slot=N" on
            // stdout (no stream_fd ack here — mirrors qwen35).
            std::printf("[snap] freed slot=%d\n", slot);
            std::fflush(stdout);
            continue;
        }

        if (line == "free drafter" || line == "drafter free") {
            if (drafter_loaded) {
                free_drafter(drafter_ctx);
                drafter_loaded = false;
                std::printf("[drafter] freed\n"); std::fflush(stdout);
            }
            stream_emit(-1);
            continue;
        }
        if (starts_with(line, "compress ")) {
            // Format mirrors qwen35'́s daemon so server.py'́s _prefill_hook
            // is byte-identical for both arches:
            //   compress <ids.bin> <keep_x1000> <drafter.gguf>
            // The driver may have already issued `park target` to make room
            // for the drafter — we still re-park here defensively if it
            // hasn'́t, and re-unpark on exit so the next generate works.
            char ppath[1024];
            int  keep_x1000 = 0;
            char drafter_path[1024];
            const int n = std::sscanf(line.c_str() + 9, "%1023s %d %1023s",
                                       ppath, &keep_x1000, drafter_path);
            if (n != 3) {
                std::fprintf(stderr,
                              "[compress] bad args, need: <bin> <keep_x1000> <drafter.gguf>\n");
                stream_emit(-1); continue;
            }
            auto src_ids = read_uncounted_i32(ppath);
            if (src_ids.empty()) {
                std::fprintf(stderr, "[compress] empty input\n");
                stream_emit(-1); continue;
            }

            const bool restore_target = !target_parked;
            if (restore_target && !park_target()) { stream_emit(-1); continue; }

            if (!drafter_loaded) {
                if (!load_drafter(drafter_path, /*gpu_layers=*/999, drafter_ctx)) {
                    std::fprintf(stderr, "[compress] load_drafter failed: %s\n",
                                  dflash27b_last_error());
                    stream_emit(-1); continue;
                }
                drafter_loaded = true;
                std::printf("[drafter] loaded %s vocab=%d\n",
                             drafter_path, drafter_ctx.weights.n_vocab);
                std::fflush(stdout);
            }

            const float keep = (float)keep_x1000 / 1000.0f;
            auto compressed = drafter_score_and_compress(drafter_ctx, src_ids, keep);
            std::printf("[compress] %zu -> %zu tokens (keep_ratio=%.3f)\n",
                         src_ids.size(), compressed.size(), keep);
            std::fflush(stdout);

            if (restore_target && !unpark_target()) { stream_emit(-1); continue; }

            for (int32_t t : compressed) stream_emit(t);
            stream_emit(-1);
            continue;
        }

        // ---- Generate / bare-prompt commands (sampler tail honored) -------
        SamplerCfg sampler{};
        const bool have_sampler = parse_sampler_token(line, sampler);
        if (have_sampler && sampler.seed != 0) sampler_rng.seed(sampler.seed);
        const bool do_sample = have_sampler && sampler.temp > 0.0f;
        if (target_parked) {
            std::fprintf(stderr,
                          "[laguna-daemon] target is parked; expected unpark before generate\n");
            std::printf("err target_parked\n"); std::fflush(stdout);
            stream_emit(-1);
            continue;
        }
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        if (cmd == "generate") {
            std::string in_path, out_path;
            int n_gen = 0;
            iss >> in_path >> n_gen >> out_path;
            if (in_path.empty() || out_path.empty() || n_gen <= 0) {
                std::fprintf(stderr, "[laguna-daemon] bad: %s\n", line.c_str());
                std::printf("err bad_args\n"); std::fflush(stdout);
                continue;
            }
            auto prompt = read_counted_i32(in_path);
            if (prompt.empty()) {
                std::printf("err empty_prompt\n"); std::fflush(stdout); continue;
            }
            double pf_s = 0.0, g_s = 0.0;
            std::vector<int32_t> generated;
            const char * err = run_prompt(prompt, n_gen, sampler, do_sample,
                                          /*stream=*/false, pf_s, g_s, generated);
            if (err) {
                std::printf("err %s\n", err); std::fflush(stdout);
                continue;
            }
            if (!write_counted_i32(out_path, generated)) {
                std::printf("err write_out\n"); std::fflush(stdout); continue;
            }
            std::printf("ok N=%d gen=%zu prefill_s=%.3f decode_s=%.3f decode_tok_s=%.1f out=%s\n",
                        (int)prompt.size(), generated.size(), pf_s, g_s,
                        generated.size() / std::max(1e-9, g_s), out_path.c_str());
            std::fflush(stdout);
            continue;
        }

        // ---- RESTORE: load slot KV, prefill prompt tail, decode ---------
        if (cmd == "RESTORE") {
            int slot = -1;
            std::string in_path;
            int n_gen = 0;
            iss >> slot >> in_path >> n_gen;
            if (slot < 0 || slot >= kMaxSlots || in_path.empty() || n_gen <= 0) {
                std::fprintf(stderr, "[snap] RESTORE bad args: %s\n", line.c_str());
                std::printf("err bad_args\n"); std::fflush(stdout);
                stream_emit(-1); continue;
            }
            if (!snapshots[slot].used) {
                std::fprintf(stderr, "[snap] RESTORE slot=%d not populated\n", slot);
                std::printf("err empty_slot\n"); std::fflush(stdout);
                stream_emit(-1); continue;
            }
            if (!laguna_snapshot_restore(snapshots[slot], cache)) {
                std::fprintf(stderr, "[snap] RESTORE slot=%d: %s\n",
                              slot, dflash27b_last_error());
                std::printf("err restore\n"); std::fflush(stdout);
                stream_emit(-1); continue;
            }
            const int prefix_len = cache.cur_pos;
            auto prompt_full = read_uncounted_i32(in_path);
            if ((int)prompt_full.size() < prefix_len) {
                std::fprintf(stderr, "[snap] RESTORE prompt shorter than cached prefix (%zu < %d)\n",
                              prompt_full.size(), prefix_len);
                std::printf("err prefix_shorter\n"); std::fflush(stdout);
                stream_emit(-1); continue;
            }
            // Run on the diff only (tokens past the cached prefix). When the
            // diff is empty (cache covers the whole prompt) we still need to
            // pull last-token logits, so synthesize a 1-token decode call by
            // re-running the very last cached token through laguna_step. To
            // keep things simple here we drop the last cached token and
            // re-prefill it together with the diff: cur_pos becomes
            // prefix_len-1 and we forward prompt_full[prefix_len-1:].
            int N = (int)prompt_full.size();
            if (prefix_len == N) {
                if (prefix_len <= 0) {
                    std::printf("err empty_diff\n"); std::fflush(stdout);
                    stream_emit(-1); continue;
                }
                cache.cur_pos = prefix_len - 1;
            }
            const int kv_start = cache.cur_pos;
            const int diff_n   = N - kv_start;
            std::vector<float> embed_diff((size_t)diff_n * w.n_embd);
            if (!w.embedder.embed(prompt_full.data() + kv_start, diff_n, embed_diff.data())) {
                std::printf("err embed_prefill\n"); std::fflush(stdout);
                stream_emit(-1); continue;
            }
            std::vector<float> last_logits;
            bool ok = true;
            const int n_chunks = (diff_n + args.chunk - 1) / args.chunk;
            for (int c = 0; c < n_chunks && ok; ++c) {
                const int off    = c * args.chunk;
                const int n_tok  = std::min(args.chunk, diff_n - off);
                const int starts = kv_start + off;
                ok = laguna_step(backend, w, cache,
                                  embed_diff.data() + (size_t)off * w.n_embd,
                                  n_tok, starts, no_mask, last_logits);
            }
            if (!ok) {
                std::printf("err prefill\n"); std::fflush(stdout);
                stream_emit(-1); continue;
            }
            std::vector<int32_t> history(prompt_full);
            auto argmax = [](const std::vector<float> & ll) {
                int best = 0; float bv = ll[0];
                for (size_t i = 1; i < ll.size(); ++i)
                    if (ll[i] > bv) { bv = ll[i]; best = (int)i; }
                return best;
            };
            auto pick = [&](const std::vector<float> & ll) {
                return do_sample
                    ? sample_logits(ll.data(), (int)ll.size(), sampler, history, sampler_rng)
                    : argmax(ll);
            };
            int next_tok = pick(last_logits);
            int generated = 0;
            std::vector<float> embed_step((size_t)w.n_embd);
            for (int s = 0; s < n_gen; ++s) {
                if (next_tok == w.eos_id || next_tok == w.eos_chat_id) break;
                history.push_back(next_tok);
                stream_emit(next_tok);
                ++generated;
                if (!w.embedder.embed(&next_tok, 1, embed_step.data())) { ok = false; break; }
                std::vector<float> step_logits;
                if (!laguna_step(backend, w, cache, embed_step.data(), 1,
                                  cache.cur_pos, no_mask, step_logits)) { ok = false; break; }
                next_tok = pick(step_logits);
            }
            stream_emit(-1);
            std::printf("ok N=%d gen=%d prefix_len=%d (RESTORE slot=%d) stream_fd=%d\n",
                         N, generated, prefix_len, slot, stream_fd);
            std::fflush(stdout);
            continue;
        }

        if (looks_like_path(cmd)) {
            const std::string & in_path = cmd;
            int n_gen = 0;
            iss >> n_gen;
            // Optional inline snap tail: ` snap=<L>:<slot>`. After the prompt
            // is fully prefilled, snapshot the cache at cur_pos=L into the
            // given slot. server.py'́s PrefixCache uses this to register a
            // mid-prompt prefix it can RESTORE from on later turns.
            int inline_snap_pos  = -1;
            int inline_snap_slot = -1;
            {
                size_t p = line.find(" snap=");
                if (p != std::string::npos) {
                    int L = -1, S = -1;
                    if (std::sscanf(line.c_str() + p + 6, "%d:%d", &L, &S) == 2 &&
                        L > 0 && S >= 0 && S < kMaxSlots) {
                        inline_snap_pos  = L;
                        inline_snap_slot = S;
                    }
                }
            }
            if (n_gen <= 0) {
                std::fprintf(stderr, "[laguna-daemon] bad: %s\n", line.c_str());
                std::printf("err bad_args\n"); std::fflush(stdout);
                emit_int32(-1);
                continue;
            }
            if (stream_fd < 0) {
                std::fprintf(stderr, "[laguna-daemon] bare-prompt requires --stream-fd\n");
                std::printf("err no_stream_fd\n"); std::fflush(stdout);
                continue;
            }
            auto prompt = read_uncounted_i32(in_path);
            if (prompt.empty()) {
                std::printf("err empty_prompt\n"); std::fflush(stdout);
                emit_int32(-1);
                continue;
            }
            double pf_s = 0.0, g_s = 0.0;
            std::vector<int32_t> generated;
            const char * err = run_prompt(prompt, n_gen, sampler, do_sample,
                                          /*stream=*/true, pf_s, g_s, generated);
            if (err) {
                emit_int32(-1);
                std::printf("err %s\n", err); std::fflush(stdout);
                continue;
            }
            std::printf("ok N=%d gen=%zu prefill_s=%.3f decode_s=%.3f decode_tok_s=%.1f stream_fd=%d\n",
                        (int)prompt.size(), generated.size(), pf_s, g_s,
                        generated.size() / std::max(1e-9, g_s), stream_fd);
            std::fflush(stdout);
            // Inline snapshot: register the first inline_snap_pos tokens of
            // the just-prefilled cache so a later request can RESTORE this
            // prefix. We snapshot the WHOLE buffer (one device-to-device
            // copy) but set snap.cur_pos = L; the surplus rows beyond L are
            // never read because RESTORE trims cur_pos back to L and any
            // subsequent forward overwrites them.
            if (inline_snap_slot >= 0 && inline_snap_pos > 0 &&
                inline_snap_pos <= (int)prompt.size()) {
                if (ensure_slot(inline_snap_slot) &&
                    laguna_snapshot_save(cache, snapshots[inline_snap_slot])) {
                    snapshots[inline_snap_slot].cur_pos = inline_snap_pos;
                    std::printf("[snap] inline slot=%d cur_pos=%d\n",
                                 inline_snap_slot, inline_snap_pos);
                    std::fflush(stdout);
                }
            }
            continue;
        }

        std::fprintf(stderr, "[laguna-daemon] unknown cmd: %s\n", line.c_str());
        std::printf("err unknown_command\n"); std::fflush(stdout);
        emit_int32(-1);
    }

    for (auto & snap : snapshots) laguna_snapshot_free(snap);
    if (drafter_loaded) free_drafter(drafter_ctx);
    if (!target_parked) {
        free_laguna_target_cache(cache);
        free_laguna_target_weights(w);
    }
    ggml_backend_free(backend);
    return 0;
}

}  // namespace dflash27b
