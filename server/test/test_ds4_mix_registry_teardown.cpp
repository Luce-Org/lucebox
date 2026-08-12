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

#include "deepseek4_internal.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using dflash::common::DeepSeek4Layer;
using dflash::common::DeepSeek4Weights;
using dflash::common::free_deepseek4_weights;

extern "C" void ggml_cuda_rocmfp3_mix_register_host(
    const void * base, size_t nb02, int n_experts, int out, int in,
    const void * codebooks_bf16_host, const uint8_t * modes_host,
    const uint8_t * rotations_host);
extern "C" void ggml_cuda_rocmfp2_mix_register_host(
    const void * base, size_t nb02, int n_experts, int out, int in,
    const void * codebooks_bf16_host, const uint8_t * modes_host,
    const uint8_t * rotations_host);
bool ggml_cuda_rocmfp3_mix_registered(const void * vx);
bool ggml_cuda_rocmfp2_mix_registered(const void * vx);

static int g_fails = 0;
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", (msg)); ++g_fails; }  \
    } while (0)

namespace {

// Distinct, aligned, non-overlapping stand-ins for device bases. Spaced well beyond the
// registered span (nb02 * n_experts) so no two ranges can be confused for one another.
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

void register_as(bool is105, const void * base) {
    const int K = is105 ? 8 : 4;
    std::vector<uint16_t> books((size_t) E * 2 * (size_t) K, 0x3f80);  // bf16 ~1.0
    std::vector<uint8_t>  modes(E, 1), rots(E, 0);
    if (is105) {
        ggml_cuda_rocmfp3_mix_register_host(base, NB02, E, OUT, IN,
                                           books.data(), modes.data(), rots.data());
    } else {
        ggml_cuda_rocmfp2_mix_register_host(base, NB02, E, OUT, IN,
                                           books.data(), modes.data(), rots.data());
    }
}

}  // namespace

int main() {
    // Tensors live in a no_alloc context: only type and data are read by the teardown, and
    // free_deepseek4_weights owns the context afterwards.
    struct ggml_init_params ip = { /*mem_size=*/ 32u * 1024u * 1024u,
                                  /*mem_buffer=*/ nullptr, /*no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        std::fprintf(stderr, "SKIP: ggml_init failed\n");
        return 0;
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
    DeepSeek4Layer L{};
    int slot = 0;
    std::vector<const void *> dense105, dense106, expert105, expert106;

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
        if (!t) { std::fprintf(stderr, "SKIP: tensor alloc failed\n"); return 0; }
        t->data = (void *) fake_base(slot++);
        const bool is105 = (t->type == GGML_TYPE_Q3_1_ROCMFP3_MIX);
        register_as(is105, t->data);
        (is105 ? dense105 : dense106).push_back(t->data);
    }

    // Experts, the population the original teardown did cover.
    L.ffn_down_exps = mk(GGML_TYPE_Q3_1_ROCMFP3_MIX);
    L.ffn_gate_exps = mk(GGML_TYPE_Q2_1_ROCMFP2_MIX);
    L.ffn_up_exps   = mk(GGML_TYPE_Q2_1_ROCMFP2_MIX);
    ggml_tensor * const exp_t[3] = { L.ffn_down_exps, L.ffn_gate_exps, L.ffn_up_exps };
    for (ggml_tensor * t : exp_t) {
        if (!t) { std::fprintf(stderr, "SKIP: tensor alloc failed\n"); return 0; }
        t->data = (void *) fake_base(slot++);
        const bool is105 = (t->type == GGML_TYPE_Q3_1_ROCMFP3_MIX);
        register_as(is105, t->data);
        (is105 ? expert105 : expert106).push_back(t->data);
    }
    w.layers.push_back(L);

    // The fixture must actually contain dense entries, or everything below is vacuous. This
    // is the assertion the review asked for, and it is why the fixture is synthetic.
    CHECK(!dense105.empty() && !dense106.empty(),
          "fixture contains dense mix tensors of BOTH qtypes (else the test proves nothing)");

    size_t live = 0;
    for (const void * b : dense105)  live += ggml_cuda_rocmfp3_mix_registered(b) ? 1 : 0;
    for (const void * b : expert105) live += ggml_cuda_rocmfp3_mix_registered(b) ? 1 : 0;
    for (const void * b : dense106)  live += ggml_cuda_rocmfp2_mix_registered(b) ? 1 : 0;
    for (const void * b : expert106) live += ggml_cuda_rocmfp2_mix_registered(b) ? 1 : 0;
    const size_t total = dense105.size() + dense106.size() +
                         expert105.size() + expert106.size();
    CHECK(live == total, "every registration resolves before teardown");

    free_deepseek4_weights(w);
    CHECK(w.ctx == nullptr, "context released by teardown");
    CHECK(w.layers.empty(), "layers cleared by teardown");

    // THE REGRESSION: dense entries must be gone, not just the expert ones.
    size_t stale_dense = 0, stale_expert = 0;
    for (const void * b : dense105)  stale_dense  += ggml_cuda_rocmfp3_mix_registered(b) ? 1 : 0;
    for (const void * b : dense106)  stale_dense  += ggml_cuda_rocmfp2_mix_registered(b) ? 1 : 0;
    for (const void * b : expert105) stale_expert += ggml_cuda_rocmfp3_mix_registered(b) ? 1 : 0;
    for (const void * b : expert106) stale_expert += ggml_cuda_rocmfp2_mix_registered(b) ? 1 : 0;
    std::fprintf(stderr, "note: after teardown %zu/%zu dense and %zu/%zu expert entries "
                 "still resolve\n", stale_dense, dense105.size() + dense106.size(),
                 stale_expert, expert105.size() + expert106.size());
    CHECK(stale_dense == 0, "NO dense attention mix entry survives free_deepseek4_weights");
    CHECK(stale_expert == 0, "no expert mix entry survives free_deepseek4_weights");

    std::fprintf(stderr, g_fails ? "MIX REGISTRY TEARDOWN TEST FAILED (%d)\n"
                                 : "MIX REGISTRY TEARDOWN TEST OK\n", g_fails);
    return g_fails ? 1 : 0;
}
