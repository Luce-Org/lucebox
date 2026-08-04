// Regression test for transactional cleanup of a failed DeepSeek4 load.
//
// When a qtype-105 model's sidecar is missing/invalid the load must fail AND
// leave the DeepSeek4Weights empty (ctx/buf released, registry unwound), so that
// load_model()'s fallback — which reuses the SAME DeepSeek4Weights for
// init_hybrid_model() — does not overwrite (and permanently leak) the first
// allocation's context + GPU buffer handles.
//
// Model-gated: needs a real qtype-105 model. Set DS4_TEST_MODEL to a *.gguf that
// has a valid <model>.p4mix.bin sidecar next to it. Without it the test skips
// (returns 0) so it is CI-safe on hosts without the fixture model.
//
// It never touches the real model or sidecar: a temp symlink stands in for the
// model, and the presence of the temp sidecar symlink is toggled to exercise the
// failure and success paths against the same weights object.

#include "deepseek4_internal.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

// C++ linkage, matching the definitions in rocmfp{2,3}_mix.cu. Pure pointer-range lookups.
bool ggml_cuda_rocmfp3_mix_registered(const void * vx);
bool ggml_cuda_rocmfp2_mix_registered(const void * vx);

using dflash::common::DeepSeek4Weights;
using dflash::common::load_deepseek4_gguf;
using dflash::common::free_deepseek4_weights;

static int g_fails = 0;
#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", (msg)); ++g_fails; } \
    } while (0)

int main() {
    const char * model = std::getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        std::fprintf(stderr, "SKIP: set DS4_TEST_MODEL to a qtype-105 *.gguf "
                     "(with a valid .p4mix.bin) to run this test\n");
        return 0;
    }
    const std::string real_model = model;
    const std::string real_side  = real_model + ".p4mix.bin";

    // temp symlinks (unique per pid) so we never mutate the real model/sidecar
    const std::string tmp_model = "/tmp/ds4_txn_" + std::to_string(getpid()) + ".gguf";
    const std::string tmp_side  = tmp_model + ".p4mix.bin";
    ::unlink(tmp_model.c_str());
    ::unlink(tmp_side.c_str());
    if (symlink(real_model.c_str(), tmp_model.c_str()) != 0) {
        std::fprintf(stderr, "SKIP: could not create temp model symlink\n");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "SKIP: no CUDA/HIP device\n");
        ::unlink(tmp_model.c_str());
        return 0;
    }

    DeepSeek4Weights w;  // the SAME object is reused across both loads

    // 1. Missing sidecar => load must fail and release everything.
    ::unlink(tmp_side.c_str());  // ensure no sidecar
    bool ok1 = load_deepseek4_gguf(tmp_model, backend, w);
    CHECK(!ok1, "load fails when the qtype-105 sidecar is missing");
    CHECK(w.ctx == nullptr, "ctx released after failed load");
    CHECK(w.buf == nullptr, "GPU buffer released after failed load");
    CHECK(w.layers.empty(), "layers cleared after failed load");

    // 2. Retry with a valid sidecar into the SAME weights object. This must
    //    succeed — proving the failed load left no live handles to overwrite.
    if (symlink(real_side.c_str(), tmp_side.c_str()) == 0) {
        bool ok2 = load_deepseek4_gguf(tmp_model, backend, w);
        CHECK(ok2, "retry with a valid sidecar succeeds on the reused object");
        CHECK(w.ctx != nullptr && w.buf != nullptr, "reused load holds fresh handles");

        // 3. Registry teardown must cover EVERY mix tensor class, not just the experts.
        //    free_deepseek4_weights originally walked only ffn_down/gate/up, so the dense
        //    attention classes registered by the dmix sidecar were left behind: their device
        //    codebooks leaked and their ranges kept resolving against released buffers, so a
        //    later allocation at the same address answered with stale side data.
        //
        //    Bases are captured BEFORE the free and queried after. registered() is pure
        //    pointer-range arithmetic and never dereferences the key, so this is safe on a
        //    freed address -- and it is precisely the stale-range question being asked.
        std::vector<const void *> mix105, mix106;
        for (const auto & L : w.layers) {
            ggml_tensor * const dense[] = {
                L.attn_q_a, L.attn_q_b, L.attn_kv, L.attn_output_a, L.attn_output_b,
                L.ffn_down_exps, L.ffn_gate_exps, L.ffn_up_exps,
            };
            for (ggml_tensor * t : dense) {
                if (!t || !t->data) continue;
                if (t->type == GGML_TYPE_Q3_1_ROCMFP3_MIX) mix105.push_back(t->data);
                else if (t->type == GGML_TYPE_Q2_1_ROCMFP2_MIX) mix106.push_back(t->data);
            }
        }
        std::fprintf(stderr, "note: %zu qtype-105 and %zu qtype-106 mix tensors registered\n",
                     mix105.size(), mix106.size());

        free_deepseek4_weights(w);
        CHECK(w.ctx == nullptr && w.buf == nullptr, "clean release after success");

        size_t stale = 0;
        for (const void * b : mix105) if (ggml_cuda_rocmfp3_mix_registered(b)) ++stale;
        for (const void * b : mix106) if (ggml_cuda_rocmfp2_mix_registered(b)) ++stale;
        CHECK(stale == 0, "no mix registry entry survives free (dense classes included)");

        // 4. And it must be reloadable afterwards: a surviving range would either be hit by
        //    the new allocation or trip the registry's own duplicate handling.
        bool ok3 = load_deepseek4_gguf(tmp_model, backend, w);
        CHECK(ok3, "reload succeeds after a clean free");
        free_deepseek4_weights(w);
        CHECK(w.ctx == nullptr && w.buf == nullptr, "clean release after reload");
    } else {
        std::fprintf(stderr, "note: no real sidecar at %s; skipped success-retry leg\n",
                     real_side.c_str());
    }

    ggml_backend_free(backend);
    ::unlink(tmp_side.c_str());
    ::unlink(tmp_model.c_str());

    std::fprintf(stderr, g_fails ? "TRANSACTIONAL LOAD TEST FAILED (%d)\n"
                                 : "TRANSACTIONAL LOAD TEST OK\n", g_fails);
    return g_fails ? 1 : 0;
}
