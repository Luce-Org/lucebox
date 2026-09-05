#include "../src/common/moe_source_page_range.h"
#include "../src/common/copied_source_reclaim.h"
#include "../src/common/copied_source_upload.h"

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

static void check_staged_upload() {
    const size_t payload = 2 * COPIED_SOURCE_UPLOAD_CHUNK + 37;
    const size_t prefix = 17;
    std::vector<uint8_t> source(prefix + payload + 19), scratch, destination(payload);
    for (size_t i = 0; i < source.size(); ++i)
        source[i] = static_cast<uint8_t>((i * 37 + i / 4096) % 251);
    size_t next = 0, calls = 0;
    const uint8_t * scratch_base = nullptr;
    auto upload = [&](const uint8_t * bytes, size_t offset, size_t count) {
        check(offset == next, "staged destination offsets are contiguous and tensor-relative");
        check(count > 0 && count <= COPIED_SOURCE_UPLOAD_CHUNK && count <= payload - offset,
              "staged chunk bounds");
        check(bytes == scratch.data(), "backend receives heap scratch");
        const uintptr_t address = reinterpret_cast<uintptr_t>(bytes);
        const uintptr_t source_address = reinterpret_cast<uintptr_t>(source.data());
        check(address < source_address || address >= source_address + source.size(),
              "backend never receives original source bytes");
        if (!scratch_base) scratch_base = bytes;
        check(bytes == scratch_base, "one buffer reused for every chunk");
        std::memcpy(destination.data() + offset, bytes, count);
        next += count;
        ++calls;
    };
    check(upload_copied_file_chunks(source.data(), source.size(), prefix, payload, scratch, upload),
          "valid multichunk upload");
    check(next == payload && calls == 3, "two full chunks and short final chunk");
    check(std::memcmp(destination.data(), source.data() + prefix, payload) == 0,
          "all uploaded bytes preserved");
    auto unexpected = [&](const uint8_t *, size_t, size_t) { check(false, "invalid/empty upload callback"); };
    check(upload_copied_file_chunks(source.data(), source.size(), source.size(), 0, scratch, unexpected),
          "empty end span does not upload");
    check(!upload_copied_file_chunks(source.data(), source.size(), source.size() + 1, 0, scratch, unexpected),
          "source offset past mapping rejected");
    check(!upload_copied_file_chunks(source.data(), source.size(), prefix,
          std::numeric_limits<size_t>::max(), scratch, unexpected), "overflowing source size rejected");
    check(!upload_copied_file_chunks(nullptr, 0, 0, 0, scratch, unexpected), "null mapping rejected");
    check(!upload_copied_file_chunks(source.data(), std::numeric_limits<size_t>::max(),
          0, 1, scratch, unexpected), "mapping address overflow rejected");
    size_t small_calls = 0;
    check(upload_copied_file_chunks(source.data(), source.size(), prefix, 29, scratch,
        [&](const uint8_t * bytes, size_t offset, size_t count) {
            check(offset == 0 && count == 29 && bytes == scratch_base, "scratch reused across tensors");
            check(std::memcmp(bytes, source.data() + prefix, count) == 0, "short tensor bytes preserved");
            ++small_calls;
        }), "short next tensor upload");
    check(small_calls == 1, "short tensor uploaded once");
}

int main() {
    check_staged_upload();
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
#if defined(__linux__)
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
    // Keep the original descriptor until copied-source advice returns.
    const volatile uint8_t * bytes = static_cast<const volatile uint8_t *>(mapped);
    for (size_t i = 0; i < size; ++i) check(bytes[i] == expected[i], "source before advice");
    check(moe_source_page_range(reinterpret_cast<uintptr_t>(mapped), size,
          reinterpret_cast<uintptr_t>(mapped) + 17, 3 * p + 100, p, range), "native source range");
    std::vector<unsigned char> before(5), after(5);
    const int before_rc = ::mincore(mapped, size, before.data());
    errno = EBUSY;  // Successful advice must not report a stale errno.
    const auto advice = reclaim_copied_file_source(mapped, size,
        static_cast<const uint8_t *>(mapped) + 17, 3 * p + 100, fd, "test");
    const int after_rc = ::mincore(mapped, size, after.data());
    unsigned before_count = 0, after_count = 0;
    for (size_t i = 0; i < 5; ++i) { before_count += before[i] & 1; after_count += after[i] & 1; }
    std::printf("copied source requested=%zu madvise_error=%d fadvise_error=%d mincore_before_rc=%d pages=%u after_rc=%d pages=%u\n",
                advice.requested, advice.madvise_error, advice.fadvise_error,
                before_rc, before_count, after_rc, after_count);
    check(advice.range_error == 0 && advice.requested == 2 * p, "advice uses exact interior range");
    check(advice.madvise_error == 0, "read-only file MADV_DONTNEED accepted");
    check(advice.fadvise_error == 0 || advice.fadvise_error == ENOSYS || advice.fadvise_error == EOPNOTSUPP,
          "unexpected file cache advice failure");
    const auto partial = reclaim_copied_file_source(mapped, size,
        static_cast<const uint8_t *>(mapped) + 1, p - 2, fd, "partial-test");
    check(partial.requested == 0 && partial.range_error == 0, "empty interior causes no whole-file fadvise");
    const auto invalid = reclaim_copied_file_source(mapped, size, mapped, size, -1, "invalid-fd-test");
    check(invalid.requested == 0 && invalid.range_error == EINVAL, "invalid fd rejected before advice");
    check(::close(fd) == 0, "close borrowed fd after advice");
    // Refault correctness is mandatory; eviction count is only diagnostic.
    for (size_t i = 0; i < size; ++i) check(bytes[i] == expected[i], "source/refault and edge bytes preserved");
    check(::munmap(mapped, size) == 0, "close source mapping");
#else
    std::puts("SKIP: Linux copied-source advice unavailable; bounds and mode checks passed");
#endif
    std::puts("PASS: staged-copy bytes/bounds, source-page bounds, mode exclusions and supported file-refault checks");
    return 0;
}
