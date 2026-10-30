// A tensor-parallel (meta) buffer fails like any other buffer when a device
// is out of memory: allocation returns null instead of aborting, so a load
// that does not fit (a managed buffer refused for lack of RAM on an
// integrated GPU, say) can be reported and the server keeps running.

#include "CppUnitTestFramework.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <memory>

namespace {
struct MetaBufferOom {};

using ContextPtr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

// More than any address space: no device can back it.
constexpr int64_t kImpossible = (int64_t) 1 << 60;
constexpr int64_t kSmall      = 4096;

ggml_backend_meta_split_state mirrored(const ggml_tensor *, void *) {
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, 1, {1}};
}

// Device 0 takes a small slice and device 1 one it cannot hold, so the first
// device's buffer is allocated before the second fails.
ggml_backend_meta_split_state second_device_too_big(const ggml_tensor *, void *) {
    return {GGML_BACKEND_SPLIT_AXIS_0, {kSmall, kImpossible}, 1, {1}};
}

ggml_backend_buffer_type_t meta_buft(size_t n_devs, ggml_backend_meta_get_split_state_t split) {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) return nullptr;
    ggml_backend_dev_t devs[2] = {cpu, cpu};
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, n_devs, split, nullptr);
    return meta ? ggml_backend_dev_buffer_type(meta) : nullptr;
}
}

TEST_CASE(MetaBufferOom, buffer_that_does_not_fit_is_null) {
    ggml_backend_buffer_type_t buft = meta_buft(1, mirrored);
    REQUIRE(buft != nullptr && ggml_backend_buft_is_meta(buft));
    CHECK(ggml_backend_buft_alloc_buffer(buft, (size_t) kImpossible) == nullptr);
}

TEST_CASE(MetaBufferOom, tensors_that_do_not_fit_stay_unallocated) {
    ggml_backend_buffer_type_t buft = meta_buft(2, second_device_too_big);
    REQUIRE(buft != nullptr && ggml_backend_buft_is_meta(buft));
    ContextPtr ctx(ggml_init({1u << 16, nullptr, true}), ggml_free);
    REQUIRE(ctx != nullptr);
    ggml_tensor * t = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, kSmall + kImpossible);

    CHECK(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft) == nullptr);
    // Unallocated again, so the caller can place the tensor elsewhere.
    CHECK(t->buffer == nullptr);
    CHECK(t->data == nullptr);
}
