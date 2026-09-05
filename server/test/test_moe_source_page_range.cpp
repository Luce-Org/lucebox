#include "../src/common/moe_source_page_range.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <vector>
#if defined(__linux__)
#include <cerrno>
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace dflash::common;

static void check(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

int main() {
    constexpr size_t page = 4096;
    MoeSourcePageRange range;
    check(moe_source_page_range(page, 5 * page, page + 17, 3 * page + 100, page, range), "unaligned tensor valid");
    check(range.address == 2 * page && range.size == 2 * page, "only tensor interior pages selected");
    check(moe_source_page_range(page, 2 * page, page, 2 * page, page, range) && range.size == 2 * page,
          "exact mapping boundaries");
    check(moe_source_page_range(page, page, page + 1, page - 2, page, range) && range.size == 0,
          "partial page never reclaimed");
    check(moe_source_page_range(page, page, 2 * page, 0, page, range) && range.size == 0, "empty end span");
    check(!moe_source_page_range(page, page, page - 1, page, page, range), "span before mapping rejected");
    check(!moe_source_page_range(page, page, page, page + 1, page, range), "span past mapping rejected");
    check(!moe_source_page_range(page, page, page, page, 0, range), "zero page size rejected");
    check(!moe_source_page_range(page, page, page, page, 3, range), "invalid page size rejected");
    const uintptr_t max = std::numeric_limits<uintptr_t>::max();
    check(!moe_source_page_range(max - page, page + 1, max - page, 1, page, range), "mapping overflow rejected");
    check(!moe_source_page_range(page, page, page, std::numeric_limits<size_t>::max(), page, range),
          "tensor overflow rejected");
    check(moe_source_page_range(max - 3 * page, 2 * page, max - 3 * page + 1, page, page, range),
          "valid near-address-limit span");
    for (unsigned mask = 0; mask < 16; ++mask) {
        check(moe_source_pageout_eligible(mask & 1, mask & 2, mask & 4, mask & 8) == (mask == 15),
              "CPU/unmaterialized/unallocated modes excluded");
    }
#if defined(__linux__) && defined(MADV_PAGEOUT)
    const long raw_page = ::sysconf(_SC_PAGESIZE);
    check(raw_page > 0, "native page size");
    const size_t p = static_cast<size_t>(raw_page), size = 5 * p;
    char name[] = "/tmp/ds4v-source-pageout-XXXXXX";
    const int fd = ::mkstemp(name);
    check(fd >= 0, "temporary backing file");
    ::unlink(name);
    check(::ftruncate(fd, static_cast<off_t>(size)) == 0, "size backing file");
    void * writable = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    check(writable != MAP_FAILED, "initialize file mapping");
    std::vector<uint8_t> expected(size);
    for (size_t i = 0; i < size; ++i) expected[i] = static_cast<uint8_t>((i * 37 + i / p) % 251);
    for (size_t i = 0; i < size; ++i) static_cast<uint8_t *>(writable)[i] = expected[i];
    check(::msync(writable, size, MS_SYNC) == 0, "clean backing pages");
    check(::munmap(writable, size) == 0, "close initialization mapping");
    void * mapped = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    check(mapped != MAP_FAILED, "read-only source mapping");
    ::close(fd);  // Match the production lifetime: mapping remains, FD is closed.
    const volatile uint8_t * bytes = static_cast<const volatile uint8_t *>(mapped);
    for (size_t i = 0; i < size; ++i) check(bytes[i] == expected[i], "source before advice");
    check(moe_source_page_range(reinterpret_cast<uintptr_t>(mapped), size,
          reinterpret_cast<uintptr_t>(mapped) + 17, 3 * p + 100, p, range), "native source range");
    std::vector<unsigned char> before(5), after(5);
    const int before_rc = ::mincore(mapped, size, before.data());
    errno = 0;
    const int rc = ::madvise(reinterpret_cast<void *>(range.address), range.size, MADV_PAGEOUT);
    const int advice_errno = rc == 0 ? 0 : errno;
    const int after_rc = ::mincore(mapped, size, after.data());
    unsigned before_count = 0, after_count = 0;
    for (size_t i = 0; i < 5; ++i) { before_count += before[i] & 1; after_count += after[i] & 1; }
    std::printf("MADV_PAGEOUT requested=%zu rc=%d errno=%d mincore_before_rc=%d pages=%u after_rc=%d pages=%u\n",
                range.size, rc, advice_errno, before_rc, before_count, after_rc, after_count);
    check(rc == 0 || advice_errno == EINVAL || advice_errno == ENOSYS || advice_errno == EOPNOTSUPP,
          "unexpected pageout failure");
    if (rc != 0) std::puts("SKIP: kernel does not support this advisory; bounds checks still passed");
    // Refault correctness is mandatory; eviction count is only diagnostic.
    for (size_t i = 0; i < size; ++i) check(bytes[i] == expected[i], "source/refault and edge bytes preserved");
    check(::munmap(mapped, size) == 0, "close source mapping");
#else
    std::puts("SKIP: Linux MADV_PAGEOUT unavailable; bounds and mode checks passed");
#endif
    std::puts("PASS: source-page bounds, mode exclusions and supported file-refault checks");
    return 0;
}
