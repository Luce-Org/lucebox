#include "platform_io.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <vector>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <intrin.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace luce::common {

ReadOnlyFile & ReadOnlyFile::operator=(ReadOnlyFile && o) noexcept {
    if (this != &o) {
        close();
        handle_ = o.handle_;
        size_ = o.size_;
        direct_ = o.direct_;
        o.handle_ = -1;
        o.size_ = 0;
        o.direct_ = false;
    }
    return *this;
}

bool ReadOnlyFile::read_at(uint64_t offset, void * buf, size_t size) const {
    const int64_t n = read_upto(offset, buf, size);
    if (n < 0) return false;
    if ((size_t) n < size) { errno = EIO; return false; }
    return true;
}

#if defined(_WIN32)

bool ReadOnlyFile::open(const std::string & path, bool direct, std::string * err) {
    close();
    if (direct) {
        if (err) *err = "direct reads are not supported on this platform";
        return false;
    }
    // Paths are UTF-8, as the model loader takes them.
    const int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath(wlen > 0 ? (size_t) wlen : 0, L'\0');
    if (wlen <= 0 || MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen) != wlen) {
        if (err) *err = "cannot open " + path;
        return false;
    }
    HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    LARGE_INTEGER sz{};
    if (h == INVALID_HANDLE_VALUE || !GetFileSizeEx(h, &sz)) {
        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
        if (err) *err = "cannot open " + path;
        return false;
    }
    handle_ = (intptr_t) h;
    size_ = (uint64_t) sz.QuadPart;
    return true;
}

void ReadOnlyFile::close() {
    if (handle_ >= 0) CloseHandle((HANDLE) handle_);
    handle_ = -1;
    size_ = 0;
    direct_ = false;
}

int64_t ReadOnlyFile::read_upto(uint64_t offset, void * buf, size_t size) const {
    uint8_t * out = static_cast<uint8_t *>(buf);
    size_t done = 0;
    while (done < size) {
        OVERLAPPED ov{};
        ov.Offset = (DWORD) ((offset + done) & 0xffffffffu);
        ov.OffsetHigh = (DWORD) ((offset + done) >> 32);
        const size_t want = size - done;
        DWORD got = 0;
        if (!ReadFile((HANDLE) handle_, out + done, (DWORD) (want > 0x40000000u ? 0x40000000u : want),
                      &got, &ov)) {
            if (GetLastError() == ERROR_HANDLE_EOF) break;
            errno = EIO;
            return -1;
        }
        if (got == 0) break;
        done += got;
    }
    return (int64_t) done;
}

void ReadOnlyFile::advise_willneed(uint64_t, size_t) const {}

void advise_mapped_willneed(const void *, size_t, uint64_t, size_t) {}

size_t mapped_resident_bytes(const void *, uint64_t, size_t) { return 0; }

uint64_t host_available_bytes() {
    MEMORYSTATUSEX st{};
    st.dwLength = sizeof(st);
    return GlobalMemoryStatusEx(&st) ? (uint64_t) st.ullAvailPhys : 0;
}

uint32_t host_word_load_acquire(const uint32_t * p) {
    const uint32_t v = *(const volatile uint32_t *) p;
    _ReadWriteBarrier();
    return v;
}

void host_word_store_release(uint32_t * p, uint32_t v) {
    _ReadWriteBarrier();
    *(volatile uint32_t *) p = v;
}

void cpu_relax() {
#if defined(_M_X64) || defined(_M_IX86)
    _mm_pause();
#endif
}

#else  // POSIX

bool ReadOnlyFile::open(const std::string & path, bool direct, std::string * err) {
    close();
    int flags = O_RDONLY | O_CLOEXEC;
#if defined(O_DIRECT)
    if (direct) flags |= O_DIRECT;
#else
    if (direct) {
        if (err) *err = "direct reads are not supported on this platform";
        return false;
    }
#endif
    const int fd = ::open(path.c_str(), flags);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        if (fd >= 0) ::close(fd);
        if (err) *err = "cannot open " + path;
        return false;
    }
    handle_ = fd;
    size_ = (uint64_t) st.st_size;
    direct_ = direct;
    return true;
}

void ReadOnlyFile::close() {
    if (handle_ >= 0) ::close((int) handle_);
    handle_ = -1;
    size_ = 0;
    direct_ = false;
}

int64_t ReadOnlyFile::read_upto(uint64_t offset, void * buf, size_t size) const {
    uint8_t * out = static_cast<uint8_t *>(buf);
    size_t done = 0;
    while (done < size) {
        const ssize_t n = pread((int) handle_, out + done, size - done, (off_t) (offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) return -1;
        if (n == 0) break;
        done += (size_t) n;
        // A direct read comes back short only at the end of the file, and
        // a retry from there would not be block-aligned.
        if (direct_ && done < size) break;
    }
    return (int64_t) done;
}

void ReadOnlyFile::advise_willneed(uint64_t offset, size_t size) const {
#if defined(POSIX_FADV_WILLNEED)
    if (handle_ >= 0) (void) posix_fadvise((int) handle_, (off_t) offset, (off_t) size, POSIX_FADV_WILLNEED);
#else
    (void) offset; (void) size;
#endif
}

void advise_mapped_willneed(const void * map, size_t map_size, uint64_t offset, size_t size) {
    if (!map || size == 0 || offset + size > map_size) return;
    static const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const size_t start = (size_t) offset / page * page;
    (void) madvise(const_cast<uint8_t *>(static_cast<const uint8_t *>(map)) + start,
                   size + ((size_t) offset - start), MADV_WILLNEED);
}

size_t mapped_resident_bytes(const void * map, uint64_t offset, size_t size) {
    if (!map || size == 0) return 0;
    static const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const uintptr_t start = ((uintptr_t) map + (uintptr_t) offset) / page * page;
    const uintptr_t end = (uintptr_t) map + (uintptr_t) offset + size;
    const size_t pages = (size_t) ((end - start + page - 1) / page);
#if defined(__APPLE__)
    std::vector<char> in(pages);
#else
    std::vector<unsigned char> in(pages);
#endif
    if (mincore(reinterpret_cast<void *>(start), end - start, in.data()) != 0) return 0;
    size_t resident = 0;
    for (size_t i = 0; i < pages; ++i) resident += (in[i] & 1) ? 1 : 0;
    return std::min(size, resident * page);
}

#if defined(__linux__)
// First unsigned number in a file; false when absent (e.g. memory.max "max").
static bool read_u64_file(const std::string & path, uint64_t & out) {
    FILE * f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    unsigned long long v = 0;
    const bool ok = std::fscanf(f, "%llu", &v) == 1;
    std::fclose(f);
    out = (uint64_t) v;
    return ok;
}
#endif

uint64_t host_available_bytes() {
#if defined(__linux__)
    FILE * f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kb = 0;
    uint64_t bytes = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) { bytes = (uint64_t) kb << 10; break; }
    }
    std::fclose(f);
    // A container sees the host's MemAvailable: bound it by the room left
    // under each cgroup v2 memory limit from this process's group upward.
    std::string group;
    if (FILE * cg = std::fopen("/proc/self/cgroup", "r")) {
        while (std::fgets(line, sizeof(line), cg)) {
            if (std::strncmp(line, "0::", 3) == 0) {
                group = line + 3;
                while (!group.empty() && (group.back() == '\n' || group.back() == '\r')) group.pop_back();
                break;
            }
        }
        std::fclose(cg);
    }
    while (!group.empty() && group != "/") {
        uint64_t max = 0, cur = 0;
        const std::string dir = "/sys/fs/cgroup" + group;
        if (read_u64_file(dir + "/memory.max", max) && read_u64_file(dir + "/memory.current", cur)) {
            const uint64_t room = max > cur ? max - cur : 0;
            if (bytes == 0 || room < bytes) bytes = room;
        }
        group.resize(group.rfind('/') == 0 ? 1 : group.rfind('/'));
    }
    return bytes;
#else
    return 0;
#endif
}

uint32_t host_word_load_acquire(const uint32_t * p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
void host_word_store_release(uint32_t * p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }

void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

#endif

}  // namespace luce::common
