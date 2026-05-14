// Phase 2 RED test: Gemma4 MTP loader (load_gemma4_mtp_assistant)
//
// Should NOT compile today — MtpDrafterWeights and load_gemma4_mtp_assistant
// do not yet exist in internal.h. Once Phase 2 GREEN lands, the test compiles
// and 7 assertions verify the loader contract per
// .sisyphus/notes/mtp-spike-2026-05-09.md (sections "Contract — Phase 2").
//
// Run:
//   cd dflash && cmake --build build --target test_mtp_loader && \
//     MTP_GGUF=$ROOT/models/gemma4-mtp-31B/gemma-4-31B-it-assistant.Q4_K_M.gguf \
//     ./build/test_mtp_loader

#include "../src/internal.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace dflash27b;

static int fail(const char *msg) {
    std::fprintf(stderr, "[red] FAIL: %s\n", msg);
    return 1;
}

int main() {
    const char *p = std::getenv("MTP_GGUF");
    if (!p) {
        std::fprintf(stderr, "[skip] MTP_GGUF env not set; expected:\n");
        std::fprintf(stderr, "       /home/peppi/Dev/lucebox-hub/models/gemma4-mtp-31B/gemma-4-31B-it-assistant.Q4_K_M.gguf\n");
        return 77; // autotools skip
    }

    // Backend init (reuse the pattern from test_gemma4_dflash.cpp)
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        return fail("ggml_backend_cuda_init failed");
    }

    // The function under test (Phase 2 GREEN must define this)
    MtpDrafterWeights mtp;
    bool ok = load_gemma4_mtp_assistant(std::string(p), backend, mtp);
    if (!ok) {
        ggml_backend_free(backend);
        return fail("load_gemma4_mtp_assistant returned false");
    }

    // Assertion 1: n_embd_backbone matches target hidden (Dense 31B = 5376)
    if (mtp.n_embd_backbone != 5376) {
        std::fprintf(stderr, "  n_embd_backbone=%d expected 5376\n", mtp.n_embd_backbone);
        ggml_backend_free(backend);
        return fail("n_embd_backbone mismatch");
    }

    // Assertion 2: requires_target_arch == "gemma4" (vLLM #41789 guard)
    if (mtp.requires_target_arch != "gemma4") {
        std::fprintf(stderr, "  requires_target_arch=\"%s\" expected \"gemma4\"\n",
                     mtp.requires_target_arch.c_str());
        ggml_backend_free(backend);
        return fail("requires_target_arch mismatch");
    }

    // Assertion 3: 4 MTP transformer blocks (per MTP.md spec)
    if (mtp.layers.size() != 4) {
        std::fprintf(stderr, "  layers.size()=%zu expected 4\n", mtp.layers.size());
        ggml_backend_free(backend);
        return fail("MTP block count mismatch");
    }

    // Assertion 4: attention_k_eq_v=true (Gemma4 quirk; V always read from cache)
    if (!mtp.attention_k_eq_v) {
        ggml_backend_free(backend);
        return fail("attention_k_eq_v should be true for Gemma4");
    }

    // Assertion 5: pre_projection tensor shape [2*n_embd_backbone, n_embd_mtp]
    // pre_projection concatenates [tok_embd(n_embd_backbone) + h_prev(n_embd_backbone)]
    // and projects to MTP's own hidden size n_embd.
    // ne[0] = 2*n_embd_backbone = 10752, ne[1] = mtp.n_embd (the MTP model's hidden size)
    if (!mtp.pre_projection ||
        mtp.pre_projection->ne[0] != 2 * (int64_t)mtp.n_embd_backbone) {
        std::fprintf(stderr, "  pre_projection->ne[0]=%lld expected %d\n",
                     (long long)(mtp.pre_projection ? mtp.pre_projection->ne[0] : -1),
                     2 * mtp.n_embd_backbone);
        ggml_backend_free(backend);
        return fail("pre_projection shape mismatch (ne[0] != 2*n_embd_backbone)");
    }

    // Assertion 6: post_projection tensor shape [n_embd_mtp, n_embd_backbone]
    // Projects MTP hidden back to target backbone dimension.
    // ne[0] = mtp.n_embd, ne[1] = n_embd_backbone = 5376
    if (!mtp.post_projection ||
        mtp.post_projection->ne[1] != (int64_t)mtp.n_embd_backbone) {
        std::fprintf(stderr, "  post_projection->ne[1]=%lld expected %d\n",
                     (long long)(mtp.post_projection ? mtp.post_projection->ne[1] : -1),
                     mtp.n_embd_backbone);
        ggml_backend_free(backend);
        return fail("post_projection shape mismatch (ne[1] != n_embd_backbone)");
    }

    // Assertion 7: per-MTP-layer donor KV resolution (NOT global pair).
    // For Dense 31B (60 target layers, SWA pattern from gemma4_target_graph):
    //   even-indexed target layers = full attention,  last = 58
    //   odd-indexed  target layers = SWA attention,   last = 59
    // Each MTP layer's donor_target_layer must be exactly 58 (full) or 59 (SWA)
    // depending on that layer's own attention type.  A bounds-only check would
    // accept any value in [0, 60), which misses wrong-type assignments.
    for (size_t il = 0; il < mtp.layers.size(); ++il) {
        const int32_t got  = mtp.layers[il].donor_target_layer;
        const int32_t want = mtp.layers[il].is_swa ? 59 : 58;  // last SWA / last full-attn
        if (got != want) {
            std::fprintf(stderr,
                         "  layer %zu is_swa=%d donor_target_layer=%d expected %d\n",
                         il, (int)mtp.layers[il].is_swa, got, want);
            ggml_backend_free(backend);
            return fail("donor_target_layer does not point to last matching-type target layer");
        }
    }

    ggml_backend_free(backend);
    std::fprintf(stderr, "[red->green] all 7 assertions PASS\n");
    return 0;
}
