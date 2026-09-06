// Unit tests for dflash::common::GgufMmap (RAII platform mmap wrapper).
//
// T1: open + read first few bytes of a known file → ok, size > 0
// T2: open the same instance twice (idempotency) → no leak
// T3: open a non-existent path → returns false, object stays empty
// T4: explicit release() → object becomes empty, no crash
// T5: RAII destructor — scope exit, no crash

#include "CppUnitTestFramework.hpp"
#include "common/gguf_mmap.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#include <io.h>
#define MKSTEMP_FN(t) (_mktemp_s(t, sizeof(t)), _open(t, _O_CREAT | _O_WRONLY | _O_BINARY, 0600))
#define CLOSE_FN _close
#define WRITE_FN _write
#else
#include <unistd.h>
#define MKSTEMP_FN(t) mkstemp(t)
#define CLOSE_FN close
#define WRITE_FN write
#endif

// Use a local shim because helper functions below are free functions rather than
// fixture members, so the README's direct REQUIRE/CHECK macros are not in scope.
#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        throw std::runtime_error(std::string(__FILE__) + ":" + \
            std::to_string(__LINE__) + ": " + #cond); \
    } \
} while (0)
#undef assert
#define assert(cond) TEST_ASSERT(cond)

namespace {
struct GgufMmapFixture {};
}

// Create a small temp file with known content; returns its path.
static std::string make_temp_file(const std::string & payload = "GGUF_TEST_PAYLOAD_1234") {
#if defined(_WIN32)
    char tmpl[] = "gguf_mmap_test_XXXXXX";
    int fd = MKSTEMP_FN(tmpl);
#else
    char tmpl[] = "/tmp/gguf_mmap_test_XXXXXX";
    int fd = mkstemp(tmpl);
#endif
    assert(fd >= 0);
    const auto written = WRITE_FN(fd, payload.data(), static_cast<unsigned>(payload.size()));
    assert(written == static_cast<decltype(written)>(payload.size()));
    CLOSE_FN(fd);
    return std::string(tmpl);
}

// ─── T1: open + basic read ───────────────────────────────────────────────────

static void t1_open_and_read() {
    std::string path = make_temp_file();
    dflash::common::GgufMmap m;
    std::string err;

    assert(m.open(path, err));
    assert(m.is_open());
    assert(m.data() != nullptr);
    assert(m.size() > 0);

    // The first bytes should match the payload written above.
    const char expected[] = "GGUF_TEST_PAYLOAD";
    assert(m.size() >= sizeof(expected) - 1);
    assert(std::memcmp(m.data(), expected, sizeof(expected) - 1) == 0);

    std::remove(path.c_str());
    std::puts("T1 PASS");
}

// ─── T2: idempotent re-open ──────────────────────────────────────────────────

static void t2_idempotent_open() {
    std::string path1 = make_temp_file();
    const std::string replacement = "SECOND_FILE_DIFFERENT_CONTENT_AND_SIZE";
    std::string path2 = make_temp_file(replacement);

    dflash::common::GgufMmap m;
    std::string err;

    assert(m.open(path1, err));
    assert(m.is_open());

    // Second open on the same object must release the first mapping cleanly.
    assert(m.open(path2, err));
    assert(m.is_open());
    assert(m.size() == replacement.size());
    assert(std::memcmp(m.data(), replacement.data(), replacement.size()) == 0);
    assert(err.empty()); // no error expected after successful open

    std::remove(path1.c_str());
    std::remove(path2.c_str());
    std::puts("T2 PASS");
}

// ─── T3: missing file → returns false, object stays empty ───────────────────

static void t3_missing_file() {
    dflash::common::GgufMmap m;
    std::string err;

    bool ok = m.open("/tmp/gguf_mmap_does_not_exist_xyz987.bin", err);
    assert(!ok);
    assert(!m.is_open());
    assert(m.data() == nullptr);
    assert(m.size() == 0);
    assert(!err.empty());

    std::puts("T3 PASS");
}

// ─── T4: explicit release() → object becomes empty ──────────────────────────

static void t4_explicit_release() {
    std::string path = make_temp_file();
    dflash::common::GgufMmap m;
    std::string err;

    assert(m.open(path, err));
    assert(m.is_open());

    auto region = m.release();
    assert(region.data != nullptr);
    assert(region.size > 0);
    assert(!m.is_open());
    assert(m.data() == nullptr);
    assert(m.size() == 0);

    // Caller must clean up the released region.
#if defined(_WIN32)
    UnmapViewOfFile(const_cast<void *>(region.data));
#else
    ::munmap(const_cast<void *>(region.data), region.size);
    if (region.fd >= 0) ::close(region.fd);
#endif

    std::remove(path.c_str());
    std::puts("T4 PASS");
}

// ─── T5: RAII destructor — scope exit, no crash ─────────────────────────────

static void t5_raii_destructor() {
    std::string path = make_temp_file();
    {
        dflash::common::GgufMmap m;
        std::string err;
        assert(m.open(path, err));
        assert(m.is_open());
        // ~GgufMmap() fires here, must not crash or leak.
    }
    std::remove(path.c_str());
    std::puts("T5 PASS");
}

// ─── main ────────────────────────────────────────────────────────────────────

TEST_CASE(GgufMmapFixture, gguf_mmap_suite) {
    t1_open_and_read();
    t2_idempotent_open();
    t3_missing_file();
    t4_explicit_release();
    t5_raii_destructor();

    std::puts("ALL TESTS PASS");
}

TEST_CASE(GgufMmapFixture, failed_reopen_clears_previous_mapping) {
    const std::string path = make_temp_file();
    dflash::common::GgufMmap mapping;
    std::string error;
    REQUIRE(mapping.open(path, error));
    // A child of a regular file cannot exist; avoid a fixed supposedly-missing path.
    CHECK(!mapping.open(path + "/missing", error));
    CHECK(!mapping.is_open());
    CHECK(mapping.data() == nullptr);
    CHECK(mapping.size() == 0);
    CHECK(!error.empty());
    std::remove(path.c_str());
}

TEST_CASE(GgufMmapFixture, move_transfers_mapping_and_empties_source) {
    const std::string path = make_temp_file("moved mapping");
    const std::string old_path = make_temp_file("replaced mapping");
    {
        dflash::common::GgufMmap source, destination;
        std::string error;
        REQUIRE(source.open(path, error));
        const void * data = source.data();
        const size_t size = source.size();
        dflash::common::GgufMmap moved(std::move(source));
        CHECK(!source.is_open());
        CHECK(source.data() == nullptr);
        CHECK(source.size() == 0);
        REQUIRE(destination.open(old_path, error));
        destination = std::move(moved);
        CHECK(!moved.is_open());
        CHECK(moved.data() == nullptr);
        CHECK(moved.size() == 0);
        CHECK(destination.data() == data);
        CHECK(destination.size() == size);
        CHECK(std::memcmp(destination.data(), "moved mapping", size) == 0);
    }
    std::remove(path.c_str());
    std::remove(old_path.c_str());
}
