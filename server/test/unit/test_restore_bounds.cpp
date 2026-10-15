#include "CppUnitTestFramework.hpp"
#include "internal.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <vector>

using luce::common::PrefixSnapshot;
using luce::common::TargetCache;
using luce::common::restore_target_cache;

namespace {
struct RestoreBoundsFixture {};

// One full-attention layer: K and V tensors of [head_dim=4, rows, n_head_kv=2]
// on the CPU backend, as a bounded KVFlash pool allocates them (rows < max_ctx).
struct KvPair {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         k = nullptr;
    ggml_tensor *         v = nullptr;
    KvPair(ggml_backend_t be, int rows) {
        ggml_init_params p{};
        p.mem_size = 4 * ggml_tensor_overhead();
        p.no_alloc = true;
        ctx = ggml_init(p);
        k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, rows, 2);
        v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, rows, 2);
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_buffer_clear(buf, 0);
    }
    ~KvPair() { ggml_backend_buffer_free(buf); ggml_free(ctx); }
    float first_k() const {
        float f = -1.0f;
        ggml_backend_tensor_get(k, &f, 0, sizeof f);
        return f;
    }
};

void fill(const KvPair & s, float value) {
    const std::vector<float> data((size_t)ggml_nelements(s.k), value);
    ggml_backend_tensor_set(s.k, data.data(), 0, data.size() * sizeof(float));
    ggml_backend_tensor_set(s.v, data.data(), 0, data.size() * sizeof(float));
}

struct Restore {
    PrefixSnapshot snap;
    TargetCache    cache;
    Restore(const KvPair & pool, const KvPair & src, int pos) {
        cache.max_ctx   = 131072;  // logical context, far above the pool rows
        cache.kv_k_type = GGML_TYPE_F32;
        cache.attn_k    = {pool.k};
        cache.attn_v    = {pool.v};
        snap.layout     = PrefixSnapshot::Layout::dense;
        snap.max_ctx    = cache.max_ctx;
        snap.kv_k_type  = cache.kv_k_type;
        snap.cur_pos    = pos;
        snap.attn_k_snap = {src.k};
        snap.attn_v_snap = {src.v};
    }
};
} // namespace

// A dense snapshot read from the disk cache can be longer than the pool tensor
// (written by a server with no pool): restoring it must fail, not write past
// the end of the tensor.
TEST_CASE(RestoreBoundsFixture, dense_snapshot_longer_than_the_kv_rows_is_refused) {
    ggml_backend_t be = ggml_backend_cpu_init();
    CHECK(be != nullptr);
    KvPair pool(be, 8);
    KvPair src(be, 12);
    fill(src, 7.0f);
    Restore r(pool, src, 12);
    CHECK(!restore_target_cache(r.snap, r.cache));
    CHECK(pool.first_k() == 0.0f);  // untouched
    ggml_backend_free(be);
}

TEST_CASE(RestoreBoundsFixture, dense_snapshot_that_fits_the_kv_rows_restores) {
    ggml_backend_t be = ggml_backend_cpu_init();
    CHECK(be != nullptr);
    KvPair pool(be, 8);
    KvPair src(be, 8);
    fill(src, 7.0f);
    Restore r(pool, src, 8);  // exactly the pool rows
    CHECK(restore_target_cache(r.snap, r.cache));
    CHECK(pool.first_k() == 7.0f);
    CHECK(r.cache.cur_pos == 8);
    ggml_backend_free(be);
}
