// common/gguf_mmap.h — RAII wrapper for platform-conditional mmap of GGUF files.
//
// Encapsulates POSIX mmap / Windows MapViewOfFile behind a single interface.
// Loaders that materialize tensor data from disk must use this class instead
// of inlining equivalent platform-conditional code.
//
// Include convention: #include "common/gguf_mmap.h"
// Never: ../common/gguf_mmap.h or absolute paths.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace dflash::common {

class GgufMmap {
public:
    GgufMmap() = default;
    ~GgufMmap();

    // Non-copyable.
    GgufMmap(const GgufMmap &) = delete;
    GgufMmap & operator=(const GgufMmap &) = delete;

    // Movable — transfers ownership, leaves source empty.
    GgufMmap(GgufMmap &&) noexcept;
    GgufMmap & operator=(GgufMmap &&) noexcept;

    // Open the file at path and mmap it read-only.
    // Returns true on success. On failure, writes a human-readable description
    // to out_error and leaves this object in the default (empty) state.
    bool open(const std::string & path, std::string & out_error);

    const void * data() const;   // nullptr when not open
    size_t       size() const;   // 0 when not open
    bool         is_open() const;

    // Advise the kernel to read ahead the given byte range into page cache.
    // No-op if not open or range is invalid. Safe to call from any thread.
    void advise_willneed(size_t offset, size_t length) const;

    // Transfer ownership of the mmap'd region to the caller.
    // After release() this object is empty (is_open() == false).
    // The caller is responsible for unmapping on POSIX or UnmapViewOfFile on
    // Windows, and for closing the fd on POSIX.
    struct OwnedRegion {
        const void * data;
        size_t       size;
        int          fd;    // POSIX fd; -1 on Windows (handle already closed)
    };
    OwnedRegion release();

private:
    const void * data_ = nullptr;
    size_t       size_ = 0;
#if defined(_WIN32)
    void *       handle_ = nullptr;  // HANDLE (Windows mapping object, reinterpret_cast'd)
#else
    int          fd_     = -1;
#endif
};

} // namespace dflash::common

// ── Implementation ────────────────────────────────────────────────────────
// Header-only: the platform-conditional code lives here rather than in a .cpp
// so that loaders can include a single file without adding a new translation unit.

#include <cstring>

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash::common {

inline GgufMmap::~GgufMmap() {
    if (!data_) return;
#if defined(_WIN32)
    UnmapViewOfFile(const_cast<void *>(data_));
    if (handle_) CloseHandle(reinterpret_cast<HANDLE>(handle_));
#else
    ::munmap(const_cast<void *>(data_), size_);
    if (fd_ >= 0) ::close(fd_);
#endif
    data_ = nullptr;
    size_ = 0;
}

inline GgufMmap::GgufMmap(GgufMmap && o) noexcept
    : data_(o.data_), size_(o.size_)
#if defined(_WIN32)
    , handle_(o.handle_)
#else
    , fd_(o.fd_)
#endif
{
    o.data_ = nullptr;
    o.size_ = 0;
#if defined(_WIN32)
    o.handle_ = nullptr;
#else
    o.fd_ = -1;
#endif
}

inline GgufMmap & GgufMmap::operator=(GgufMmap && o) noexcept {
    if (this != &o) {
        this->~GgufMmap();
        new (this) GgufMmap(std::move(o));
    }
    return *this;
}

inline bool GgufMmap::open(const std::string & path, std::string & out_error) {
    // Idempotency: if already open, release prior mapping before re-opening.
    // This prevents leaking the prior fd/mapping and ensures that, on failure,
    // the object is left in the default empty state (not half-overwritten).
    if (data_) { this->~GgufMmap(); new (this) GgufMmap(); }
#if defined(_WIN32)
    const int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (wlen <= 0) {
        out_error = "GgufMmap: MultiByteToWideChar failed for " + path;
        return false;
    }
    std::wstring wpath;
    wpath.resize(wlen - 1);
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);

    HANDLE hFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        out_error = "GgufMmap: CreateFileW failed for " + path;
        return false;
    }
    LARGE_INTEGER li;
    if (!GetFileSizeEx(hFile, &li)) {
        out_error = "GgufMmap: GetFileSizeEx failed for " + path;
        CloseHandle(hFile);
        return false;
    }
    size_t file_size = static_cast<size_t>(li.QuadPart);

    HANDLE hMapping = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(hFile);
    if (!hMapping) {
        out_error = "GgufMmap: CreateFileMappingA failed for " + path;
        return false;
    }
    void * addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!addr) {
        out_error = "GgufMmap: MapViewOfFile failed for " + path;
        CloseHandle(hMapping);
        return false;
    }
    data_   = addr;
    size_   = file_size;
    handle_ = reinterpret_cast<void *>(hMapping);
    return true;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        out_error = std::string("GgufMmap: open failed for ") + path + ": " + strerror(errno);
        return false;
    }
    struct stat st{};
    if (::fstat(fd, &st) < 0) {
        out_error = std::string("GgufMmap: fstat failed: ") + strerror(errno);
        ::close(fd);
        return false;
    }
    size_t file_size = static_cast<size_t>(st.st_size);
    void * addr = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        out_error = std::string("GgufMmap: mmap failed: ") + strerror(errno);
        ::close(fd);
        return false;
    }
    data_ = addr;
    size_ = file_size;
    fd_   = fd;
    return true;
#endif
}

inline const void * GgufMmap::data() const { return data_; }
inline size_t       GgufMmap::size() const { return size_; }
inline bool         GgufMmap::is_open() const { return data_ != nullptr; }

inline void GgufMmap::advise_willneed(size_t offset, size_t length) const {
    if (!data_ || offset >= size_) return;
    if (offset + length > size_) length = size_ - offset;
    if (length == 0) return;
#if defined(_WIN32)
    // PrefetchVirtualMemory (Windows 8+)
    WIN32_MEMORY_RANGE_ENTRY entry{};
    entry.VirtualAddress = const_cast<uint8_t *>(static_cast<const uint8_t *>(data_)) + offset;
    entry.NumberOfBytes  = length;
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
#else
    // Align to page boundary for madvise
    const size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
    const size_t aligned_offset = (offset / page_size) * page_size;
    const size_t aligned_length = length + (offset - aligned_offset);
    ::madvise(const_cast<uint8_t *>(static_cast<const uint8_t *>(data_)) + aligned_offset,
              aligned_length, MADV_WILLNEED);
#endif
}

inline GgufMmap::OwnedRegion GgufMmap::release() {
    OwnedRegion r{};
    r.data = data_;
    r.size = size_;
#if defined(_WIN32)
    // Close the mapping handle now.  Per MSDN, closing the handle does not
    // unmap the view — the view remains valid until UnmapViewOfFile is called.
    // OwnedRegion has no handle field; caller unmaps via UnmapViewOfFile(data).
    if (handle_) CloseHandle(reinterpret_cast<HANDLE>(handle_));
    r.fd = -1;
    handle_ = nullptr;
#else
    r.fd = fd_;
    fd_  = -1;
#endif
    data_ = nullptr;
    size_ = 0;
    return r;
}

} // namespace dflash::common
