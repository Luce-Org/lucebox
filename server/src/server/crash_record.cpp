// Crash record implementation; see crash_record.h.

#include "crash_record.h"

#if defined(__linux__)

#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace luce::common {

namespace {

constexpr int kSignals[] = {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT};
constexpr int kMaxFrames = 64;
constexpr size_t kAltStack = 64 * 1024;

int g_dirfd = -1;
char g_header[1024];
size_t g_header_len = 0;
char g_maps_buf[4096];

void put(int fd, const char * text, size_t n) {
    while (n > 0) {
        const ssize_t wrote = ::write(fd, text, n);
        if (wrote <= 0) return;
        text += wrote;
        n -= (size_t)wrote;
    }
}

void put(int fd, const char * text) { put(fd, text, std::strlen(text)); }

// Decimal and hexadecimal without printf: both are async-signal-safe.
size_t format_unsigned(char * out, uint64_t value, int base) {
    char tmp[24];
    size_t n = 0;
    do {
        const int digit = (int)(value % (uint64_t)base);
        tmp[n++] = (char)(digit < 10 ? '0' + digit : 'a' + digit - 10);
        value /= (uint64_t)base;
    } while (value != 0 && n < sizeof(tmp));
    for (size_t i = 0; i < n; ++i) out[i] = tmp[n - 1 - i];
    return n;
}

void put_hex(int fd, const void * pointer) {
    char text[24] = {'0', 'x'};
    const size_t n = format_unsigned(text + 2, (uint64_t)(uintptr_t)pointer, 16);
    text[2 + n] = '\n';
    put(fd, text, n + 3);
}

void put_number(int fd, const char * label, uint64_t value) {
    char text[32];
    const size_t n = format_unsigned(text, value, 10);
    put(fd, label);
    put(fd, text, n);
    put(fd, "\n");
}

const char * signal_name(int sig) {
    switch (sig) {
    case SIGSEGV: return "SIGSEGV";
    case SIGBUS:  return "SIGBUS";
    case SIGILL:  return "SIGILL";
    case SIGFPE:  return "SIGFPE";
    case SIGABRT: return "SIGABRT";
    default:      return "signal";
    }
}

void on_fatal(int sig, siginfo_t * info, void *) {
    char name[64] = "crash-";
    size_t n = 6;
    n += format_unsigned(name + n, (uint64_t)::getpid(), 10);
    name[n++] = '-';
    n += format_unsigned(name + n, (uint64_t)std::time(nullptr), 10);
    std::memcpy(name + n, ".txt", 5);
    const int fd = ::openat(g_dirfd, name, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0640);
    if (fd >= 0) {
        void * frames[kMaxFrames];
        const int n_frames = backtrace(frames, kMaxFrames);
        write_crash_record(fd, sig, info ? info->si_addr : nullptr, frames, n_frames);
        ::close(fd);
    }
    // SA_RESETHAND put the default action back: die of the same signal.
    ::raise(sig);
}

}  // namespace

void write_crash_record(int fd, int sig, const void * address, void * const * frames, int n_frames) {
    put(fd, g_header, g_header_len);
    put_number(fd, "signal ", (uint64_t)sig);
    put(fd, "name ");
    put(fd, signal_name(sig));
    put(fd, "\naddress ");
    put_hex(fd, address);
    put(fd, "frames\n");
    for (int i = 0; i < n_frames; ++i) put_hex(fd, frames[i]);
    put(fd, "maps\n");
    const int maps = ::open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
    if (maps >= 0) {
        ssize_t got;
        while ((got = ::read(maps, g_maps_buf, sizeof(g_maps_buf))) > 0) put(fd, g_maps_buf, (size_t)got);
        ::close(maps);
    }
    put(fd, "end\n");
}

bool install_crash_record(const char * dir, const std::string & about) {
    if (!dir || !*dir || g_dirfd >= 0) return false;
    const int dirfd = ::open(dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dirfd < 0) return false;
    g_dirfd = dirfd;
    std::string header = "luce crash record 1\n";
    char exe[512];
    const ssize_t exe_len = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (exe_len > 0) header += "binary " + std::string(exe, (size_t)exe_len) + "\n";
    if (!about.empty()) header += about.substr(0, 256) + "\n";
    char pid[32];
    header += "pid ";
    header.append(pid, format_unsigned(pid, (uint64_t)::getpid(), 10));
    header += "\nstarted ";
    header.append(pid, format_unsigned(pid, (uint64_t)std::time(nullptr), 10));
    header += "\n";
    g_header_len = std::min(header.size(), sizeof(g_header));
    std::memcpy(g_header, header.data(), g_header_len);
    // backtrace() loads its unwinder on first use, which can allocate: not in the handler.
    void * warm[2];
    backtrace(warm, 2);

    stack_t stack{};
    stack.ss_sp = std::malloc(kAltStack);
    stack.ss_size = kAltStack;
    if (stack.ss_sp) ::sigaltstack(&stack, nullptr);
    struct sigaction action{};
    action.sa_sigaction = on_fatal;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESETHAND;
    sigemptyset(&action.sa_mask);
    for (const int sig : kSignals) ::sigaction(sig, &action, nullptr);
    return true;
}

}  // namespace luce::common

#else

namespace luce::common {
bool install_crash_record(const char *, const std::string &) { return false; }
void write_crash_record(int, int, const void *, void * const *, int) {}
}  // namespace luce::common

#endif
