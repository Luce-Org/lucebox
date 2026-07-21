#include "internal.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstdio>
#include <vector>

using dflash::common::TargetCache;
using dflash::common::restore_ssm_state;
using dflash::common::snapshot_ssm_state;

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        failures++; \
    } \
} while (0)

static void set_tensor(ggml_tensor * tensor, const std::vector<float> & values) {
    CHECK(ggml_nelements(tensor) == (int64_t)values.size());
    ggml_backend_tensor_set(tensor, values.data(), 0,
                            values.size() * sizeof(float));
}

static std::vector<float> get_tensor(const ggml_tensor * tensor) {
    std::vector<float> values((size_t)ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0,
                            values.size() * sizeof(float));
    return values;
}

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr);
    if (!backend) return 1;

    ggml_init_params params{};
    params.mem_size = 8 * ggml_tensor_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);
    if (!ctx) {
        ggml_backend_free(backend);
        return 1;
    }

    ggml_tensor * ssm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_tensor * ssm_snap = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_tensor * conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 2);
    ggml_tensor * conv_snap = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 2);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buffer != nullptr);
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    TargetCache cache;
    cache.ssm_state = {ssm, nullptr};
    cache.ssm_state_snap = {ssm_snap, nullptr};
    cache.conv_state = {conv, nullptr};
    cache.conv_state_snap = {conv_snap, nullptr};

    const std::vector<float> ssm_original = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    };
    const std::vector<float> conv_original = {
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    };
    const std::vector<float> ssm_mutated(ssm_original.size(), -1.0f);
    const std::vector<float> conv_mutated(conv_original.size(), -2.0f);

    set_tensor(ssm, ssm_original);
    set_tensor(conv, conv_original);
    CHECK(snapshot_ssm_state(cache, backend));
    set_tensor(ssm, ssm_mutated);
    set_tensor(conv, conv_mutated);
    CHECK(restore_ssm_state(cache, backend));
    CHECK(get_tensor(ssm) == ssm_original);
    CHECK(get_tensor(conv) == conv_original);

    // A partial shard may omit a complete recurrent-state quartet, but an
    // asymmetric quartet must fail validation before any copy is queued.
    set_tensor(ssm, ssm_mutated);
    cache.conv_state_snap[0] = nullptr;
    CHECK(!snapshot_ssm_state(cache, backend));
    CHECK(get_tensor(ssm_snap) == ssm_original);
    cache.conv_state_snap[0] = conv_snap;

    CHECK(!snapshot_ssm_state(cache, nullptr));
    cache.ssm_state_snap.pop_back();
    CHECK(!restore_ssm_state(cache, backend));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (failures != 0) {
        std::fprintf(stderr, "%d recurrent snapshot test(s) failed\n", failures);
        return 1;
    }
    std::printf("recurrent snapshot tests passed\n");
    return 0;
}
