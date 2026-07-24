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
        free_deepseek4_weights(w);
        CHECK(w.ctx == nullptr && w.buf == nullptr, "clean release after success");
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
