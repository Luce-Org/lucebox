#pragma once
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <memory>
#include <type_traits>

namespace luce::common {
// Keep device ownership before its dependent members so it is released last
// during constructor unwinding. Deleters also cover post-init bad_alloc.
template<auto Free> struct GgmlDeleter {
    template<class T> void operator()(T * p) const noexcept { Free(p); }
};
using VulkanBackendOwner = std::unique_ptr<std::remove_pointer_t<ggml_backend_t>, GgmlDeleter<ggml_backend_free>>;
using GraphAllocatorOwner = std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlDeleter<ggml_gallocr_free>>;
} // namespace luce::common
