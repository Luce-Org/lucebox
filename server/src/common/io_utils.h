// io_utils.h — small I/O helpers shared by daemon code and test harness.
// Header-only: all functions are static inline to avoid extra link targets.

#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <cerrno>
#  include <unistd.h>
#endif

namespace dflash::common {

// ── Binary file I/O ────────────────────────────────────────────────

static inline std::vector<int32_t> read_int32_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streampos pos = f.tellg();
    if (pos == std::streampos(-1)) return {};
    const std::streamoff bytes_off = pos - std::streampos(0);
    if (bytes_off < 0 || bytes_off % (std::streamoff)sizeof(int32_t) != 0) return {};
    const size_t bytes = (size_t)bytes_off;
    if (bytes > (size_t)std::numeric_limits<std::streamsize>::max()) return {};
    const size_t n = bytes / sizeof(int32_t);
    f.seekg(0, std::ios::beg);
    if (!f) return {};
    std::vector<int32_t> out(n);
    if (bytes > 0) {
        f.read((char *)out.data(), (std::streamsize)bytes);
        if (!f || f.gcount() != (std::streamsize)bytes) return {};
    }
    return out;
}

static inline bool write_int32_file(const std::string & path,
                                    const std::vector<int32_t> & v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write((const char *)v.data(), v.size() * sizeof(int32_t));
    return (bool)f;
}

static inline bool write_binary_file(const std::string & path,
                                     const void * data, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    if (bytes > 0) f.write((const char *)data, (std::streamsize)bytes);
    return (bool)f;
}

static inline bool read_binary_file_exact(const std::string & path,
                                          void * data, size_t bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    if (bytes > 0) f.read((char *)data, (std::streamsize)bytes);
    return (bool)f;
}

// ── String helpers ──────────────────────────────────────────────────

static inline std::string read_line_tail(std::istringstream & iss) {
    std::string tail;
    std::getline(iss, tail);
    const size_t first = tail.find_first_not_of(" \t");
    if (first == std::string::npos) return {};
    if (first > 0) tail.erase(0, first);
    return tail;
}

// ── Streaming / fd I/O ──────────────────────────────────────────────

static inline void stream_emit_fd(int stream_fd, int32_t tok) {
    if (stream_fd < 0) return;
#if defined(_WIN32)
    DWORD written = 0;
    const int32_t v = tok;
    WriteFile((HANDLE)(intptr_t)stream_fd, &v, sizeof(v), &written, nullptr);
#else
    const int32_t v = tok;
    (void)::write(stream_fd, &v, sizeof(v));
#endif
}

#if !defined(_WIN32)
static inline bool read_exact_fd(int fd, void * data, size_t bytes) {
    char * p = (char *)data;
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = ::read(fd, p + done, bytes - done);
        if (n == 0) return false;
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        done += (size_t)n;
    }
    return true;
}

static inline bool write_exact_fd(int fd, const void * data, size_t bytes) {
    const char * p = (const char *)data;
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = ::write(fd, p + done, bytes - done);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        done += (size_t)n;
    }
    return true;
}
#endif

// ── Numeric helpers ─────────────────────────────────────────────────

static inline int argmax_f32(const float * x, int n) {
    int best = 0;
    float bv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > bv) { bv = x[i]; best = i; }
    return best;
}

} // namespace dflash::common
