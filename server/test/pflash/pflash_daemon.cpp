// Persistent PFlash compressor daemon.
//
// Loads the Qwen3.5-0.8B PFlash drafter once, then accepts stdin commands:
//
//   compress <keep_x1000> <lookahead> <chunk> <pool> <counted_ids.bin>
//   segment ... / request ...   (see below)
//   quit
//
// Input token file format is little-endian u32 count followed by count int32
// token IDs in the drafter tokenizer. Compressed IDs are emitted as int32 LE
// values to --stream-fd=<fd>, terminated by -1. Logs go to stdout/stderr.

#include "luce.h"
#include "pflash/pflash_drafter.h"
#include "pflash/qwen35_drafter.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace luce::common;

static std::vector<int32_t> read_counted_i32_file(const std::string & path) {
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

static void stream_emit(int stream_fd, int32_t tok) {
    if (stream_fd < 0) return;
#ifdef _WIN32
    DWORD written = 0;
    WriteFile((HANDLE)(intptr_t)stream_fd, &tok, sizeof(tok), &written, nullptr);
#else
    ssize_t n = ::write(stream_fd, &tok, sizeof(tok));
    (void)n;
#endif
}

static void stream_ids(int stream_fd, const std::vector<int32_t> & ids) {
    for (int32_t t : ids) {
        stream_emit(stream_fd, t);
    }
    stream_emit(stream_fd, -1);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen3.5-0.8b.gguf> [--stream-fd=N]\n", argv[0]);
        return 2;
    }

    const std::string gguf = argv[1];
    int stream_fd = -1;
    for (int i = 2; i < argc; ++i) {
        if (std::strncmp(argv[i], "--stream-fd=", 12) == 0) {
            stream_fd = std::atoi(argv[i] + 12);
        }
    }

    DrafterContext ctx;
    auto t_load0 = std::chrono::steady_clock::now();
    if (!load_drafter(gguf, /*gpu_layers=*/-1, ctx)) {
        std::fprintf(stderr, "[pflash-daemon] load_drafter failed: %s\n", luce_last_error());
        return 1;
    }
    auto t_load1 = std::chrono::steady_clock::now();
    std::printf("[pflash-daemon] ready load=%.3fs vocab=%d\n",
                std::chrono::duration<double>(t_load1 - t_load0).count(),
                ctx.state->weights.n_vocab);
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "quit" || line == "exit") {
            break;
        }

        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        if (cmd == "request" || cmd == "request2") {
            // request <lookahead> <score_query_end> <suffix 0|1> <n_spans> [<b> <e>]... <n_human> [<b> <e>]... <path>
            // As segment, plus the human user messages' content spans (latest first) the request probe reads;
            // compressed ids go to --stream-fd like compress.
            // request2 <lookahead> <score_query_end> <suffix 0|1> <n_spans> [<b> <e>]... <n_messages>
            //          [<role> <b> <e>]... <path>
            // As request, with EVERY normalised message in order (role system|user|assistant|tool, content span):
            // the cand_v5 / stage-2 inputs the server maps (STAGE2-CONTRACT.md section 1); the human spans (the
            // latest two user messages, latest first) are derived from them.
            const bool all_messages = cmd == "request2";
            int lookahead = 0, query_end = -1, suffix = 0, n_spans = 0, n_human = 0;
            iss >> lookahead >> query_end >> suffix >> n_spans;
            std::vector<luce::common::PFlashTokenSpan> spans;
            bool ok = (bool) iss && lookahead > 0 && n_spans >= 0 && n_spans < 64;
            for (int i = 0; ok && i < n_spans; ++i) {
                int b = 0, e = 0;
                iss >> b >> e;
                ok = (bool) iss && 0 <= b && b < e;
                spans.push_back({b, e});
            }
            if (ok) iss >> n_human;
            ok = ok && (bool) iss && n_human >= 0 && (all_messages || n_human <= 2);
            luce::common::PFlashRequestMessages human;
            human.human_only = !all_messages;
            for (int i = 0; ok && i < n_human; ++i) {
                std::string role;
                if (all_messages) iss >> role;
                int b = 0, e = 0;
                iss >> b >> e;
                ok = (bool) iss && 0 <= b && b <= e;
                if (!all_messages) {
                    human.human.push_back({b, e});
                } else {
                    ok = ok && (role == "system" || role == "user" || role == "assistant" || role == "tool");
                    human.all_roles.push_back(role);
                    human.all_spans.push_back({b, e});
                }
            }
            if (all_messages) {
                for (size_t i = human.all_roles.size(); i-- > 0 && human.human.size() < 2;) {
                    if (human.all_roles[i] != "user") continue;
                    human.human.push_back(human.all_spans[i]);
                    human.human_index.push_back((int) i);
                }
            }
            std::string path;
            std::getline(iss, path);
            const size_t first = path.find_first_not_of(" \t");
            path = first == std::string::npos ? std::string() : path.substr(first);
            auto ids = ok && !path.empty() ? read_counted_i32_file(path) : std::vector<int32_t>{};
            if (!ok || ids.empty()) {
                std::fprintf(stderr, "[pflash-daemon] bad request command: %s\n", line.c_str());
                std::printf("[pflash-daemon] request failed\n");
                std::fflush(stdout);
                stream_emit(stream_fd, -1);
                continue;
            }
            auto t0 = std::chrono::steady_clock::now();
            // PFLASH_DAEMON_KEEP_X1000 (test tool only): the keep ratio, default 187.
            const char * keep_raw = std::getenv("PFLASH_DAEMON_KEEP_X1000");
            const float keep = (keep_raw && std::atoi(keep_raw) > 0 ? std::atoi(keep_raw) : 187.5f) / 1000.0f;
            std::vector<int32_t> out = drafter_score_and_compress(
                ctx, ids, keep, 256, lookahead, 1, query_end, spans, suffix != 0,
                {}, {-1, -1}, &human);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (out.empty()) {
                std::fprintf(stderr, "[pflash-daemon] request failed: %s\n", luce_last_error());
                std::printf("[pflash-daemon] request failed\n");
            } else {
                std::printf("[pflash-daemon] request done %.3fs in=%zu out=%zu\n", secs, ids.size(), out.size());
            }
            std::fflush(stdout);
            stream_ids(stream_fd, out);
            continue;
        }
        if (cmd == "segment") {
            // segment <lookahead> <score_query_end> <suffix_candidates 0|1> <n_instruction_spans> [<begin> <end>]... <path>
            // The served drafter path (score + probe segmentation + selection, PFlash trace written as for compress)
            // with explicit query end, suffix-candidate flag and required instruction spans, so callers can reproduce
            // a chat layout's forced cuts. Selection runs at keep 0.1875, chunk 256, pool 1 (segments do not depend on it).
            int lookahead = 0, query_end = -1, suffix = 0, n_spans = 0;
            iss >> lookahead >> query_end >> suffix >> n_spans;
            std::vector<luce::common::PFlashTokenSpan> spans;
            bool ok = (bool) iss && lookahead > 0 && n_spans >= 0 && n_spans < 64;
            for (int i = 0; ok && i < n_spans; ++i) {
                int b = 0, e = 0;
                iss >> b >> e;
                ok = (bool) iss && 0 <= b && b < e;
                spans.push_back({b, e});
            }
            std::string path;
            std::getline(iss, path);
            const size_t first = path.find_first_not_of(" \t");
            path = first == std::string::npos ? std::string() : path.substr(first);
            auto ids = ok && !path.empty() ? read_counted_i32_file(path) : std::vector<int32_t>{};
            if (!ok || ids.empty()) {
                std::fprintf(stderr, "[pflash-daemon] bad segment command: %s\n", line.c_str());
                std::printf("[pflash-daemon] segment failed\n");
                std::fflush(stdout);
                continue;
            }
            auto t0 = std::chrono::steady_clock::now();
            std::vector<int32_t> out = drafter_score_and_compress(
                ctx, ids, 0.1875f, 256, lookahead, 1, query_end, spans, suffix != 0);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (out.empty()) {
                std::fprintf(stderr, "[pflash-daemon] segment failed: %s\n", luce_last_error());
                std::printf("[pflash-daemon] segment failed\n");
            } else {
                std::printf("[pflash-daemon] segment done %.3fs in=%zu\n", secs, ids.size());
            }
            std::fflush(stdout);
            continue;
        }
        if (cmd != "compress") {
            std::fprintf(stderr, "[pflash-daemon] unknown command: %s\n", line.c_str());
            stream_emit(stream_fd, -1);
            continue;
        }

        int keep_x1000 = 0;
        int lookahead = 8;
        int chunk = 32;
        int pool = 13;
        iss >> keep_x1000 >> lookahead >> chunk >> pool;
        std::string path;
        std::getline(iss, path);
        const size_t first_non_space = path.find_first_not_of(" \t");
        if (first_non_space == std::string::npos) {
            path.clear();
        } else if (first_non_space > 0) {
            path.erase(0, first_non_space);
        }
        if (path.empty() || keep_x1000 <= 0) {
            std::fprintf(stderr, "[pflash-daemon] bad command, need: compress <keep_x1000> <lookahead> <chunk> <pool> <counted_ids.bin>\n");
            stream_emit(stream_fd, -1);
            continue;
        }

        auto ids = read_counted_i32_file(path);
        if (ids.empty()) {
            std::fprintf(stderr, "[pflash-daemon] empty input: %s\n", path.c_str());
            stream_emit(stream_fd, -1);
            continue;
        }

        const float keep_ratio = (float)keep_x1000 / 1000.0f;
        std::printf("[pflash-daemon] compress start tokens=%zu keep=%.3f lookahead=%d chunk=%d pool=%d path=%s\n",
                    ids.size(), keep_ratio, lookahead, chunk, pool, path.c_str());
        std::fflush(stdout);

        auto t0 = std::chrono::steady_clock::now();
        std::vector<int32_t> out = drafter_score_and_compress(ctx, ids, keep_ratio, chunk, lookahead, pool, (int)ids.size());
        auto t1 = std::chrono::steady_clock::now();

        const double secs = std::chrono::duration<double>(t1 - t0).count();
        if (out.empty()) {
            std::fprintf(stderr, "[pflash-daemon] compress failed: %s\n", luce_last_error());
            stream_emit(stream_fd, -1);
            continue;
        }

        std::printf("[pflash-daemon] compress done %.3fs in=%zu out=%zu ratio=%.4f\n",
                    secs, ids.size(), out.size(), (double)out.size() / (double)ids.size());
        std::fflush(stdout);
        stream_ids(stream_fd, out);
    }

    free_drafter(ctx);
    std::printf("[pflash-daemon] stopped\n");
    std::fflush(stdout);
    return 0;
}
