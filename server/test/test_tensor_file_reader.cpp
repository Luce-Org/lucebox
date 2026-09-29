// TensorFileReader: spans larger than one read, unaligned file offsets and
// writes at an offset inside a tensor land byte for byte; a span outside the
// file or its tensor is refused.

#include "CppUnitTestFramework.hpp"
#include "../src/common/tensor_file_reader.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <filesystem>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace luce::common;

namespace {
struct TensorFileReaderFixture {};

uint8_t pattern(size_t i) { return (uint8_t) (i * 131u + 7u); }

// A temporary file of `size` pattern bytes, removed with the object.
struct PatternFile {
    std::string path;
    explicit PatternFile(size_t size) {
        static int serial = 0;
        path = (std::filesystem::temp_directory_path() /
                ("tensor_file_reader_" + std::to_string((uintptr_t) this) + "_" +
                 std::to_string(serial++))).string();
        std::vector<uint8_t> bytes(size);
        for (size_t i = 0; i < size; ++i) bytes[i] = pattern(i);
        FILE * f = std::fopen(path.c_str(), "wb");
        if (f) {
            std::fwrite(bytes.data(), 1, bytes.size(), f);
            std::fclose(f);
        }
    }
    ~PatternFile() { std::remove(path.c_str()); }
};
}  // namespace

TEST_CASE(TensorFileReaderFixture, spans_land_byte_for_byte) {
    const size_t big = ((size_t) 40 << 20) + 123;  // more than one 32 MiB read
    PatternFile file(big + 8192);
    auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(
        ggml_backend_cpu_init(), ggml_backend_free);
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
        ggml_init({1u << 16, nullptr, true}), ggml_free);
    REQUIRE(backend && ctx);
    ggml_tensor * a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, (int64_t) big);
    ggml_tensor * b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, 4096);
    auto buf = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    REQUIRE(buf != nullptr);
    ggml_backend_buffer_clear(buf.get(), 0);

    TensorFileReader reader;
    std::string err;
    REQUIRE(reader.open(file.path, &err));
    // `a` from file offset 5 (unaligned); 1000 bytes of `b` at tensor offset
    // 100 from file offset 4097.
    const std::vector<TensorFileSpan> spans = {
        {a, 0, 5, big},
        {b, 100, 4097, 1000},
    };
    REQUIRE(reader.load(spans, &err));

    std::vector<uint8_t> got(big);
    ggml_backend_tensor_get(a, got.data(), 0, big);
    size_t bad = 0;
    for (size_t i = 0; i < big; ++i) bad += got[i] != pattern(5 + i);
    CHECK(bad == 0u);
    std::vector<uint8_t> got_b(4096);
    ggml_backend_tensor_get(b, got_b.data(), 0, got_b.size());
    bad = 0;
    for (size_t i = 0; i < got_b.size(); ++i) {
        const uint8_t want = i >= 100 && i < 1100 ? pattern(4097 + i - 100) : 0;
        bad += got_b[i] != want;
    }
    CHECK(bad == 0u);
}

TEST_CASE(TensorFileReaderFixture, refuses_spans_out_of_range) {
    PatternFile file(4096);
    auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(
        ggml_backend_cpu_init(), ggml_backend_free);
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
        ggml_init({1u << 16, nullptr, true}), ggml_free);
    REQUIRE(backend && ctx);
    ggml_tensor * t = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, 256);
    auto buf = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    REQUIRE(buf != nullptr);

    TensorFileReader reader;
    REQUIRE(reader.open(file.path));
    std::string err;
    CHECK(!reader.load({{t, 0, 4000, 200}}, &err));  // past the end of the file
    CHECK(!reader.load({{t, 100, 0, 200}}, &err));   // past the end of the tensor
    CHECK(reader.load({{t, 56, 0, 200}}, &err));
}
