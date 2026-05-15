// layer_split_daemon.cpp — Layer-split request handler for qwen35 daemon mode.

#include "layer_split_daemon.h"

#include "internal.h"
#include "io_utils.h"
#include "qwen35_layer_split_dflash_target.h"
#include "common/dflash_spec_decode.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace dflash27b {

bool run_target_layer_split_request(
        std::vector<TargetLayerSplitShard> & shards,
        DraftWeights * draft_weights,
        ggml_backend_t draft_backend,
        int draft_gpu,
        DraftFeatureMirror * feature_ring,
        const std::vector<int32_t> & prompt,
        int n_gen,
        int max_ctx,
        bool run_dflash,
        const char * out_path,
        int kq_stride_pad,
        int fa_window,
        int draft_ctx_max,
        int stream_fd) {
    if (shards.empty() || prompt.empty()) return false;
    if ((int)prompt.size() + n_gen + 1 > max_ctx) {
        std::fprintf(stderr, "target-split prompt (%zu) + gen (%d) exceeds max_ctx (%d)\n",
                     prompt.size(), n_gen, max_ctx);
        return false;
    }

    int ubatch = (prompt.size() > 2048) ? 384 : 16;
    if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
        ubatch = std::max(1, std::atoi(s));
    }
    int last_tok = -1;
    if (!run_target_layer_split_forward(shards, shards.front().weights,
                                        prompt, 0, ubatch, last_tok,
                                        kq_stride_pad, fa_window,
                                        feature_ring)) {
        std::fprintf(stderr, "target-split prefill failed\n");
        return false;
    }

    if (run_dflash && draft_weights && feature_ring && feature_ring->target_feat) {
        Qwen35LayerSplitDFlashTarget target(shards, feature_ring,
                                            kq_stride_pad, fa_window,
                                            /*remote_draft=*/nullptr);
        const bool ok = run_dflash_spec_decode(
            target, *draft_weights, draft_backend, *feature_ring,
            prompt, n_gen, last_tok, out_path, draft_ctx_max, stream_fd,
            /*remote_draft=*/nullptr);
        stream_emit_fd(stream_fd, -1);
        return ok;
    }

    std::vector<int32_t> out_all = prompt;
    int generated = 0;
    for (; generated < n_gen; generated++) {
        std::vector<int32_t> one(1, last_tok);
        int next_tok = -1;
        if (!run_target_layer_split_forward(shards, shards.front().weights,
                                            one, (int)out_all.size(), 1, next_tok,
                                            kq_stride_pad, fa_window,
                                            feature_ring)) {
            std::fprintf(stderr, "target-split decode failed at %d\n", generated);
            stream_emit_fd(stream_fd, -1);
            return false;
        }
        out_all.push_back(last_tok);
        stream_emit_fd(stream_fd, last_tok);
        if (is_eos_tok(last_tok, shards.front().weights)) {
            generated++;
            break;
        }
        last_tok = next_tok;
    }
    if (out_path) write_int32_file(out_path, out_all);
    stream_emit_fd(stream_fd, -1);
    return true;
}

} // namespace dflash27b
