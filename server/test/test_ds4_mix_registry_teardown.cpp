// free_deepseek4_weights must unregister EVERY mix-qtype tensor class, dense ones included.
//
// The bug: the teardown walked only ffn_down/gate/up, so the five dense attention classes the
// dmix sidecar registers were left behind -- their device codebooks leaked and their registry
// ranges kept resolving against released buffers, so a later allocation at the same address
// answered with stale side data.
//
// WHY THIS IS A SYNTHETIC FIXTURE, not an extension of test_ds4_load_transactional. Doing it
// through a real load would make the regression depend on a ~102 GB model AND on how that
// model happens to be packaged:
//
//   - a p4-only fixture has no dense mix tensors at all, so a "dense classes included"
//     assertion over its tensors passes VACUOUSLY -- present but proving nothing;
//   - an artifact with embedded codebooks cannot be made to fail by deleting a loose sidecar,
//     because the loader prefers the embedded copy, so the fixture's packaging silently
//     changes what the test exercises.
//
// Here the weights object is built by hand with a known dense population, so the assertion
// that dense entries were torn down cannot pass for the wrong reason. Runs in milliseconds
// and needs no model.
//
// The tensor `data` pointers are opaque keys: the registry only does pointer-range arithmetic
// on them and never dereferences, exactly as test_rocmfp3_mix_registry relies on.

#include "CppUnitTestFramework.hpp"
#include "deepseek4_internal.h"
#include "common/moe_hybrid_storage.h"
#include "ggml-cuda.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace CppUnitTestFramework;

using dflash::common::DeepSeek4Layer;
using dflash::common::DeepSeek4Weights;
using dflash::common::MoeHybridStorage;
using dflash::common::free_deepseek4_weights;

bool ggml_cuda_rocmfp3_mix_registered(const void * vx);
bool ggml_cuda_rocmfp2_mix_registered(const void * vx);

namespace {

// Distinct, aligned, non-overlapping stand-ins for device bases. Spaced well beyond the
// registered span (nb02 * n_experts) so no two ranges can be confused for one another.
struct Ds4MixRegistryTeardownFixture : CommonFixture {
    using CommonFixture::CommonFixture;
};

const void * fake_base(int i) {
    return (const void *) (uintptr_t) (0x100000000ull + (uintptr_t) i * 0x1000000ull);
}

// IN must be a multiple of 128: qtype-106 validates that at the registration chokepoint so
// its 16 B wide-load window stays in bounds on the final block, and registering with a
// smaller row aborts. That check is doing its job -- the fixture has to satisfy the real
// invariant rather than the check be relaxed for a test's convenience.
constexpr int    E    = 4;
constexpr int    OUT  = 32;
constexpr int    IN   = 128;
constexpr size_t NB02 = 4096;

bool register_as(bool is105, const void * base) {
    const int K = is105 ? 8 : 4;
    std::vector<uint16_t> books((size_t) E * 2 * (size_t) K, 0x3f80);  // bf16 ~1.0
    std::vector<uint8_t>  modes(E, 1);
    if (is105) {
        return ggml_cuda_rocmfp3_mix_register_host(
            base, NB02, E, OUT, IN, books.data(), modes.data());
    }
    return ggml_cuda_rocmfp2_mix_register_host(
        base, NB02, E, OUT, IN, books.data(), modes.data());
}

}  // namespace

TEST_CASE(Ds4MixRegistryTeardownFixture, unregisters_all_mix_tensor_classes) {
    // Tensors live in a no_alloc context: only type and data are read by the teardown, and
    // free_deepseek4_weights owns the context afterwards.
    struct ggml_init_params ip = { /*mem_size=*/ 32u * 1024u * 1024u,
                                  /*mem_buffer=*/ nullptr, /*no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        SKIP("ggml_init failed");
    }

    DeepSeek4Weights w;
    w.ctx = ctx;
    w.buf = nullptr;              // nothing device-allocated; teardown null-checks these
    w.dense_split_buf = nullptr;

    auto mk = [&](ggml_type t) {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, t, IN, OUT);
        return x;
    };

    // One layer carrying BOTH populations: five dense attention classes and the three expert
    // tensors. The dense half is what regressed; the expert half must keep working.
    DeepSeek4Layer & L = w.layers.emplace_back();
    int slot = 0;
    std::vector<const void *> dense105, dense106, expert105, expert106;
    auto skip_after_cleanup = [&](const char * reason) {
        free_deepseek4_weights(w);
        SKIP(reason);
    };

    // Dense: mix 105 and 106 across classes deliberately -- the sidecar records a qtype per
    // entry precisely so an artifact may do this, and the teardown must dispatch on the
    // TENSOR's type rather than on the class.
    ggml_tensor * const dense_t[5] = {
        (L.attn_q_a      = mk(GGML_TYPE_Q3_1_ROCMFP3_MIX)),
        (L.attn_q_b      = mk(GGML_TYPE_Q2_1_ROCMFP2_MIX)),
        (L.attn_kv       = mk(GGML_TYPE_Q3_1_ROCMFP3_MIX)),
        (L.attn_output_a = mk(GGML_TYPE_Q2_1_ROCMFP2_MIX)),
        (L.attn_output_b = mk(GGML_TYPE_Q3_1_ROCMFP3_MIX)),
    };
    for (ggml_tensor * t : dense_t) {
        if (!t) skip_after_cleanup("dense tensor allocation failed");
        t->data = (void *) fake_base(slot++);
        const bool is105 = (t->type == GGML_TYPE_Q3_1_ROCMFP3_MIX);
        CHECK(register_as(is105, t->data));
        (is105 ? dense105 : dense106).push_back(t->data);
    }

    // Experts, the population the original teardown did cover.
    L.ffn_down_exps = mk(GGML_TYPE_Q3_1_ROCMFP3_MIX);
    L.ffn_gate_exps = mk(GGML_TYPE_Q2_1_ROCMFP2_MIX);
    L.ffn_up_exps   = mk(GGML_TYPE_Q2_1_ROCMFP2_MIX);
    ggml_tensor * const exp_t[3] = { L.ffn_down_exps, L.ffn_gate_exps, L.ffn_up_exps };
    for (ggml_tensor * t : exp_t) {
        if (!t) skip_after_cleanup("expert tensor allocation failed");
        t->data = (void *) fake_base(slot++);
        const bool is105 = (t->type == GGML_TYPE_Q3_1_ROCMFP3_MIX);
        CHECK(register_as(is105, t->data));
        (is105 ? expert105 : expert106).push_back(t->data);
    }
    // The fixture must actually contain dense entries, or everything below is vacuous. This
    // is the assertion the review asked for, and it is why the fixture is synthetic.
    CHECK(!dense105.empty() && !dense106.empty());

    size_t live = 0;
    for (const void * b : dense105)  live += ggml_cuda_rocmfp3_mix_registered(b) ? 1 : 0;
    for (const void * b : expert105) live += ggml_cuda_rocmfp3_mix_registered(b) ? 1 : 0;
    for (const void * b : dense106)  live += ggml_cuda_rocmfp2_mix_registered(b) ? 1 : 0;
    for (const void * b : expert106) live += ggml_cuda_rocmfp2_mix_registered(b) ? 1 : 0;
    const size_t total = dense105.size() + dense106.size() +
                         expert105.size() + expert106.size();
    CHECK(live == total);

    free_deepseek4_weights(w);
    CHECK(w.ctx == nullptr);
    CHECK(w.layers.empty());

    // THE REGRESSION: dense entries must be gone, not just the expert ones.
    size_t stale_dense = 0, stale_expert = 0;
    for (const void * b : dense105)  stale_dense  += ggml_cuda_rocmfp3_mix_registered(b) ? 1 : 0;
    for (const void * b : dense106)  stale_dense  += ggml_cuda_rocmfp2_mix_registered(b) ? 1 : 0;
    for (const void * b : expert105) stale_expert += ggml_cuda_rocmfp3_mix_registered(b) ? 1 : 0;
    for (const void * b : expert106) stale_expert += ggml_cuda_rocmfp2_mix_registered(b) ? 1 : 0;
    std::fprintf(stderr, "note: after teardown %zu/%zu dense and %zu/%zu expert entries "
                 "still resolve\n", stale_dense, dense105.size() + dense106.size(),
                 stale_expert, expert105.size() + expert106.size());
    CHECK(stale_dense == 0);
    CHECK(stale_expert == 0);

    // Hybrid loading owns separate compact tensors on the primary and
    // secondary GPUs. MoeHybridStorage must unregister all of them before it
    // releases either owner's buffers.
    std::vector<const void *> hybrid105, hybrid106;
    {
        ggml_context * hybrid_ctx = ggml_init(ip);
        if (!hybrid_ctx) {
            SKIP("hybrid ggml_init failed");
        }

        MoeHybridStorage hybrid;
        hybrid.layers.resize(1);
        auto & layer = hybrid.layers[0];
        layer.hot_ctx = hybrid_ctx;
        layer.gate_hot  = ggml_new_tensor_2d(
            hybrid_ctx, GGML_TYPE_Q2_1_ROCMFP2_MIX, IN, OUT);
        layer.down_hot  = ggml_new_tensor_2d(
            hybrid_ctx, GGML_TYPE_Q3_1_ROCMFP3_MIX, IN, OUT);
        layer.gate_cold = ggml_new_tensor_2d(
            hybrid_ctx, GGML_TYPE_Q2_1_ROCMFP2_MIX, IN, OUT);
        layer.down_cold = ggml_new_tensor_2d(
            hybrid_ctx, GGML_TYPE_Q3_1_ROCMFP3_MIX, IN, OUT);

        ggml_tensor * compact[] = {
            layer.gate_hot, layer.down_hot, layer.gate_cold, layer.down_cold,
        };
        for (ggml_tensor * tensor : compact) {
            if (!tensor) {
                SKIP("hybrid tensor allocation failed");
            }
            tensor->data = (void *) fake_base(slot++);
            const bool is105 =
                tensor->type == GGML_TYPE_Q3_1_ROCMFP3_MIX;
            CHECK(register_as(is105, tensor->data));
            (is105 ? hybrid105 : hybrid106).push_back(tensor->data);
        }

        size_t live_hybrid = 0;
        for (const void * base : hybrid105) {
            live_hybrid += ggml_cuda_rocmfp3_mix_registered(base) ? 1 : 0;
        }
        for (const void * base : hybrid106) {
            live_hybrid += ggml_cuda_rocmfp2_mix_registered(base) ? 1 : 0;
        }
        CHECK(live_hybrid == hybrid105.size() + hybrid106.size());
    }

    size_t stale_hybrid = 0;
    for (const void * base : hybrid105) {
        stale_hybrid += ggml_cuda_rocmfp3_mix_registered(base) ? 1 : 0;
    }
    for (const void * base : hybrid106) {
        stale_hybrid += ggml_cuda_rocmfp2_mix_registered(base) ? 1 : 0;
    }
    CHECK(stale_hybrid == 0);
}
