// Persistent PFlash compressor daemon.
//
// Loads the Qwen3-0.6B PFlash drafter once, then accepts stdin commands:
//
//   compress <keep_x1000> <lookahead> <chunk> <pool> <counted_ids.bin>
//   quit
//
// Input token file format is little-endian u32 count followed by count int32
// token IDs in the drafter tokenizer. Compressed IDs are emitted as int32 LE
// values to --stream-fd=<fd>, terminated by -1. Logs go to stdout/stderr.

#include "dflash27b.h"
#include "qwen3_drafter.h"

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

using namespace dflash::common;

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
        std::fprintf(stderr, "usage: %s <qwen3-0.6b.gguf> [--stream-fd=N]\n", argv[0]);
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
        std::fprintf(stderr, "[pflash-daemon] load_drafter failed: %s\n", dflash27b_last_error());
        return 1;
    }
    auto t_load1 = std::chrono::steady_clock::now();
    std::printf("[pflash-daemon] ready load=%.3fs vocab=%d\n",
                std::chrono::duration<double>(t_load1 - t_load0).count(),
                ctx.weights.n_vocab);
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "quit" || line == "exit") {
            break;
        }

        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
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
        std::vector<int32_t> out = drafter_score_and_compress(ctx, ids, keep_ratio, chunk, lookahead, pool);
        auto t1 = std::chrono::steady_clock::now();

        const double secs = std::chrono::duration<double>(t1 - t0).count();
        if (out.empty()) {
            std::fprintf(stderr, "[pflash-daemon] compress failed: %s\n", dflash27b_last_error());
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
