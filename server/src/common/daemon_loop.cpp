// Generic daemon command loop implementation.
//
// Handles stdin command parsing and protocol plumbing. All model-specific
// operations are dispatched through the ModelBackend interface.

#include "daemon_loop.h"

#include "sampler.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#include <fcntl.h>
#define ssize_t long
#endif

namespace dflash::common {

// ── DaemonIO ────────────────────────────────────────────────────────────

void DaemonIO::emit(int32_t v) const {
    // Call the token callback for non-sentinel tokens.
    if (on_token && v >= 0) {
        if (!on_token(v)) {
            cancelled = true;
            return;
        }
    }

    if (stream_fd < 0) return;
#ifndef _WIN32
    ssize_t n = ::write(stream_fd, &v, sizeof(v));
    (void)n;
#else
    _write(stream_fd, &v, sizeof(v));
#endif
}

DaemonIO DaemonIO::with_token_callback(const TokenCallback & cb) const {
    DaemonIO out = *this;
    if (!cb) return out;
    TokenCallback existing = out.on_token;
    out.on_token = [existing, cb](int32_t tok) -> bool {
        if (existing && !existing(tok)) return false;
        return cb(tok);
    };
    return out;
}

// Default typed compress: delegates to handle_compress via temp file + DaemonIO collector.
ModelBackend::CompressResult ModelBackend::compress(const CompressRequest & req) {
    CompressResult result;

    if (req.input_ids.empty()) return result;

    // Write input IDs to temp file (handle_compress reads from file)
    char tmp_path[] = "/tmp/pflash_XXXXXX.bin";
    int tmp_fd = mkstemps(tmp_path, 4);
    if (tmp_fd < 0) return result;
    const size_t to_write = req.input_ids.size() * sizeof(int32_t);
    const char *src = reinterpret_cast<const char *>(req.input_ids.data());
    size_t remaining = to_write;
    while (remaining > 0) {
        ssize_t n = ::write(tmp_fd, src, remaining);
        if (n <= 0) {
            ::close(tmp_fd);
            ::unlink(tmp_path);
            return result;
        }
        src += n;
        remaining -= (size_t)n;
    }
    ::close(tmp_fd);

    // Build collecting DaemonIO
    DaemonIO io;
    io.stream_fd = -1;
    io.on_token = [&](int32_t tok) -> bool {
        result.compressed_ids.push_back(tok);
        return true;
    };

    // Build command string for legacy handle_compress
    int keep_x1000 = (int)(req.keep_ratio * 1000.0f);
    std::string cmd = std::string("compress ") + tmp_path + " "
        + std::to_string(keep_x1000) + " " + req.drafter_path;
    if (req.skip_park) cmd += " nopark";

    result.ok = handle_compress(cmd, io) && !result.compressed_ids.empty();
    ::unlink(tmp_path);
    return result;
}

// ── Helpers ─────────────────────────────────────────────────────────────

static bool starts_with(const std::string & s, const std::string & p) {
    return s.size() >= p.size() && std::memcmp(s.data(), p.data(), p.size()) == 0;
}

static bool looks_like_path(const std::string & s) {
    if (s.empty()) return false;
    if (s[0] == '/' || s[0] == '.') return true;
    return s.find('/') != std::string::npos;
}

// Read a prompt file: raw int32 stream (file size implies token count).
static std::vector<int32_t> read_uncounted_i32(const std::string & path) {
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

// Read a prompt file with uint32 length prefix + N int32 token IDs.
static std::vector<int32_t> read_counted_i32(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    uint32_t n = 0;
    f.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (!f) return {};
    std::vector<int32_t> ids((size_t)n);
    if (n > 0) {
        f.read(reinterpret_cast<char *>(ids.data()),
               (std::streamsize)ids.size() * sizeof(int32_t));
        if (!f) return {};
    }
    return ids;
}

static bool write_counted_i32(const std::string & path,
                               const std::vector<int32_t> & ids) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t n = (uint32_t)ids.size();
    f.write(reinterpret_cast<const char *>(&n), sizeof(n));
    if (n > 0)
        f.write(reinterpret_cast<const char *>(ids.data()),
                (std::streamsize)ids.size() * sizeof(int32_t));
    return (bool)f;
}

// Parse optional inline-snap suffix: ` snap=<pos>:<slot>`.
static void parse_inline_snap(const std::string & line,
                               int & snap_pos, int & snap_slot) {
    snap_pos  = -1;
    snap_slot = -1;
    size_t p = line.find(" snap=");
    if (p != std::string::npos) {
        int L = -1, S = -1;
        if (std::sscanf(line.c_str() + p + 6, "%d:%d", &L, &S) == 2 &&
            L > 0 && S >= 0 && S < ModelBackend::kMaxSlots) {
            snap_pos  = L;
            snap_slot = S;
        }
    }
}

// ── Main loop ───────────────────────────────────────────────────────────

int run_daemon(ModelBackend & backend, const DaemonLoopArgs & args) {

    DaemonIO io;
    io.stream_fd = args.stream_fd;

    backend.print_ready_banner();
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "quit" || line == "exit") break;

        // ── Lifecycle commands (no sampler tail) ─────────────────────

        if (starts_with(line, "park")) {
            std::string what;
            if (line.size() > 5) what = line.substr(5);
            backend.park(what);
            io.emit(-1);
            continue;
        }
        if (starts_with(line, "unpark")) {
            std::string what;
            if (line.size() > 7) what = line.substr(7);
            backend.unpark(what);
            io.emit(-1);
            continue;
        }
        if (line == "free drafter" || line == "drafter free") {
            backend.free_drafter();
            io.emit(-1);
            continue;
        }
        if (starts_with(line, "compress ")) {
            backend.handle_compress(line, io);
            continue;
        }

        // ── Snapshot commands ────────────────────────────────────────

        if (line == "LIST_SLOTS") {
            std::printf("[snap] slots=");
            bool first = true;
            for (int i = 0; i < ModelBackend::kMaxSlots; ++i) {
                if (backend.snapshot_used(i)) {
                    std::printf("%s%d", first ? "" : ",", i);
                    first = false;
                }
            }
            std::printf("\n"); std::fflush(stdout);
            continue;
        }
        if (starts_with(line, "SNAPSHOT ")) {
            int slot = std::atoi(line.c_str() + 9);
            backend.snapshot_save(slot);
            std::printf("[snap] inline slot=%d cur_pos=%d\n",
                        slot,
                        backend.snapshot_used(slot) ? backend.snapshot_cur_pos(slot) : -1);
            std::fflush(stdout);
            continue;
        }
        if (starts_with(line, "FREE_SNAPSHOT ")) {
            int slot = std::atoi(line.c_str() + 14);
            if (slot >= 0 && slot < ModelBackend::kMaxSlots) {
                backend.snapshot_free(slot);
            }
            std::printf("[snap] freed slot=%d\n", slot);
            std::fflush(stdout);
            continue;
        }

        // ── Arch-specific command hook ───────────────────────────────
        // qwen35 uses this for SNAPSHOT_THIN, RESTORE_CHAIN, etc.
        if (backend.try_handle_command(line, io)) {
            continue;
        }

        // ── Generate commands (sampler tail honored) ─────────────────

        SamplerCfg sampler{};
        const bool have_sampler = parse_sampler_token(line, sampler);
        const bool do_sample    = have_sampler && sampler.needs_logit_processing();

        if (backend.is_target_parked()) {
            std::fprintf(stderr,
                "[daemon] target is parked; expected unpark before generate\n");
            std::printf("err target_parked\n"); std::fflush(stdout);
            io.emit(-1);
            continue;
        }

        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        // ── "generate <in> <n_gen> <out>" (file-based, legacy) ───────
        if (cmd == "generate") {
            std::string in_path, out_path;
            int n_gen = 0;
            iss >> in_path >> n_gen >> out_path;
            if (in_path.empty() || out_path.empty() || n_gen <= 0) {
                std::fprintf(stderr, "[daemon] bad: %s\n", line.c_str());
                std::printf("err bad_args\n"); std::fflush(stdout);
                continue;
            }
            auto prompt = read_counted_i32(in_path);
            if (prompt.empty()) {
                std::printf("err empty_prompt\n"); std::fflush(stdout);
                continue;
            }
            GenerateRequest req;
            req.prompt    = std::move(prompt);
            req.n_gen     = n_gen;
            req.sampler   = sampler;
            req.do_sample = do_sample;
            req.stream    = false;

            auto result = backend.generate(req, io);
            if (!result.ok) {
                std::printf("err %s\n", result.error.c_str());
                std::fflush(stdout);
                continue;
            }
            if (!write_counted_i32(out_path, result.tokens)) {
                std::printf("err write_out\n"); std::fflush(stdout);
                continue;
            }
            std::printf("ok N=%d gen=%zu prefill_s=%.3f decode_s=%.3f decode_tok_s=%.1f out=%s\n",
                        (int)req.prompt.size(), result.tokens.size(),
                        result.prefill_s, result.decode_s,
                        result.tokens.size() / std::max(1e-9, result.decode_s),
                        out_path.c_str());
            std::fflush(stdout);
            continue;
        }

        // ── RESTORE <slot> <prompt_path> <n_gen> ─────────────────────
        if (cmd == "RESTORE") {
            int slot = -1;
            std::string in_path;
            int n_gen = 0;
            iss >> slot >> in_path >> n_gen;
            if (slot < 0 || slot >= ModelBackend::kMaxSlots ||
                in_path.empty() || n_gen <= 0) {
                std::fprintf(stderr, "[snap] RESTORE bad args: %s\n", line.c_str());
                std::printf("err bad_args\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            if (!backend.snapshot_used(slot)) {
                std::fprintf(stderr, "[snap] RESTORE slot=%d not populated\n", slot);
                std::printf("err empty_slot\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            auto prompt = read_uncounted_i32(in_path);
            if (prompt.empty()) {
                std::printf("err empty_prompt\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }

            int snap_pos = -1, snap_slot = -1;
            parse_inline_snap(line, snap_pos, snap_slot);

            GenerateRequest req;
            req.prompt    = std::move(prompt);
            req.n_gen     = n_gen;
            req.sampler   = sampler;
            req.do_sample = do_sample;
            req.stream    = true;
            req.snap_pos  = snap_pos;
            req.snap_slot = snap_slot;

            auto result = backend.restore_and_generate(slot, req, io);
            if (!result.ok) {
                std::printf("err %s\n", result.error.c_str());
                std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            std::printf("ok N=%d gen=%zu prefix_len=%d (RESTORE slot=%d) stream_fd=%d\n",
                        (int)req.prompt.size(), result.tokens.size(),
                        backend.snapshot_cur_pos(slot), slot, io.stream_fd);
            std::fflush(stdout);
            continue;
        }

        // ── Bare prompt: "<path> <n_gen> [snap=L:S]" ─────────────────
        if (looks_like_path(cmd)) {
            const std::string & in_path = cmd;
            int n_gen = 0;
            iss >> n_gen;
            if (n_gen <= 0) {
                std::fprintf(stderr, "[daemon] bad: %s\n", line.c_str());
                std::printf("err bad_args\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            if (io.stream_fd < 0) {
                std::fprintf(stderr, "[daemon] bare-prompt requires --stream-fd\n");
                std::printf("err no_stream_fd\n"); std::fflush(stdout);
                continue;
            }
            auto prompt = read_uncounted_i32(in_path);
            if (prompt.empty()) {
                std::printf("err empty_prompt\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }

            int snap_pos = -1, snap_slot = -1;
            parse_inline_snap(line, snap_pos, snap_slot);

            GenerateRequest req;
            req.prompt    = std::move(prompt);
            req.n_gen     = n_gen;
            req.sampler   = sampler;
            req.do_sample = do_sample;
            req.stream    = true;
            req.snap_pos  = snap_pos;
            req.snap_slot = snap_slot;

            auto result = backend.generate(req, io);
            if (!result.ok) {
                io.emit(-1);
                std::printf("err %s\n", result.error.c_str());
                std::fflush(stdout);
                continue;
            }
            std::printf("ok N=%d gen=%zu prefill_s=%.3f decode_s=%.3f decode_tok_s=%.1f stream_fd=%d\n",
                        (int)req.prompt.size(), result.tokens.size(),
                        result.prefill_s, result.decode_s,
                        result.tokens.size() / std::max(1e-9, result.decode_s),
                        io.stream_fd);
            std::fflush(stdout);
            continue;
        }

        // ── Unknown command ──────────────────────────────────────────
        std::fprintf(stderr, "[daemon] unknown cmd: %s\n", line.c_str());
        std::printf("err unknown_command\n"); std::fflush(stdout);
        io.emit(-1);
    }

    backend.shutdown();
    return 0;
}

}  // namespace dflash::common
