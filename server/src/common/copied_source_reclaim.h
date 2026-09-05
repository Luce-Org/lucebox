#pragma once

#include "moe_source_page_range.h"
#include <cerrno>
#include <cstdio>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace dflash::common {

struct CopiedSourceAdviceResult {
    size_t requested = 0;
    int range_error = 0;
    int madvise_error = 0;
    int fadvise_error = 0;
};

// Contract: a read-only file mapping starting at file offset zero, its original
// borrowed fd, and a fully copied source tensor. Never advise anonymous/GPU
// allocations. The caller retains mapping ownership; future reads refault the
// original file bytes. Return values report advice acceptance, not bytes freed.
inline CopiedSourceAdviceResult reclaim_copied_file_source(
        const void * mapping, size_t mapping_size, const void * source,
        size_t source_size, int fd, const char * label, int layer = -1) {
    CopiedSourceAdviceResult result;
#if defined(__linux__)
    const long page_size = ::sysconf(_SC_PAGESIZE);
    MoeSourcePageRange range;
    const uintptr_t base = reinterpret_cast<uintptr_t>(mapping);
    if (fd < 0 || page_size <= 0 || !moe_source_page_range(
            base, mapping_size, reinterpret_cast<uintptr_t>(source), source_size,
            static_cast<size_t>(page_size), range)) {
        result.range_error = EINVAL;
    } else if (range.size != 0) {
        const uintptr_t offset = range.address - base;
        const auto max_offset = static_cast<uintmax_t>(std::numeric_limits<off_t>::max());
        if (offset > max_offset || range.size > max_offset - offset) {
            result.range_error = EOVERFLOW;
        } else {
            result.requested = range.size;
            // Drop this mapping's PTE references before invalidating file cache.
            if (::madvise(reinterpret_cast<void *>(range.address), range.size, MADV_DONTNEED) != 0)
                result.madvise_error = errno;
            // posix_fadvise returns the error number itself, not -1/errno.
            result.fadvise_error = ::posix_fadvise(fd, static_cast<off_t>(offset),
                                                 static_cast<off_t>(range.size), POSIX_FADV_DONTNEED);
        }
    }
    if (result.requested || result.range_error) {
        std::fprintf(stderr, "[source-reclaim] %s layer=%d requested=%zu range_error=%d madvise_error=%d fadvise_error=%d\n",
                     label, layer, result.requested, result.range_error,
                     result.madvise_error, result.fadvise_error);
    }
#else
    (void) mapping; (void) mapping_size; (void) source; (void) source_size;
    (void) fd; (void) label; (void) layer;
#endif
    return result;
}

}  // namespace dflash::common
