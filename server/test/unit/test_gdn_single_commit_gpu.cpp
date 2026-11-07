// GPU checks for GDN single commit (qwen35/gdn_single_commit.h):
// KEEP_BASE, SWAP_SNAPSHOT and
// SINGLE_REPLAY. Exits 77 (ctest SKIP) without a GPU.
//
// 1. Kernel. The GDN op's final state equals its last F32 checkpoint byte for
//    byte: grouped-cols and scalar kernels, raw and plain gates, 1..16
//    tokens. KEEP_BASE commits that checkpoint instead of the final-state
//    write, so this is the property KEEP_BASE rests on.
// 2. Target. Qwen35DFlashTarget over a synthetic one-GDN-layer model runs the
//    do_spec_decode call sequences: rollback_to for commit_n 1..16 (all
//    accepted included), emit-phase repairs (restore twice, replay a prefix),
//    the slow path, AR seed steps, a failed verify and a request reset. After
//    each call it checks the parent's semantics: committed state ==
//    intermediate[commit_n - 1], restore == the pre-step state, reset == 0.
//    ctest runs this binary once per mode (--parts: off, k, ks, kr, ksr, the
//    parts of gdn_single_commit.h); every run writes or checks one digest of
//    all argmax outputs and committed states, and the runs with parts on must
//    match the off run exactly.
//    SINGLE_REPLAY verifies write no checkpoints, so in that mode the
//    committed state is held to intermediate[commit_n - 1] through the digest
//    (the unset run checks it directly), and each such verify must have run
//    the grouped-cols GDN kernel. On a device where the GDN op does not take
//    that kernel SINGLE_REPLAY stays inert; the run then checks the
//    F32-checkpoint fallback directly instead of failing.
//
// Usage: test_gdn_single_commit_gpu --parts off|k|ks|kr|ksr
//            [--digest-out FILE | --digest-expect FILE]

#include "qwen35_tiny_gdn_target.h"

#include "gdn_single_commit.h"
#include "qwen35_dflash_target.h"
#include "step_graph.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"  // ggml_backend_cuda_init; maps to HIP under GGML_USE_HIP
#include "ggml-cuda-gdn-chain.h"  // ggml_backend_cuda_gdn_chain_uses_grouped_cols

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using luce::common::Qwen35DFlashTarget;
using luce::common::StepGraph;
using luce_test::TinyGdnTarget;

namespace {

int failures = 0;

#define CHECK_MSG(cond, ...) do {                                      \
    if (!(cond)) {                                                     \
        std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);      \
        std::fprintf(stderr, __VA_ARGS__);                             \
        std::fprintf(stderr, "\n");                                    \
        failures++;                                                    \
    }                                                                  \
} while (0)

using Bytes = std::vector<uint8_t>;

Bytes read_bytes(const ggml_tensor * t, size_t offset, size_t size) {
    Bytes b(size);
    ggml_backend_tensor_get(t, b.data(), offset, size);
    return b;
}

Bytes read_all(const ggml_tensor * t) {
    return read_bytes(t, 0, ggml_nbytes(t));
}

void upload_random(ggml_tensor * t, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v((size_t)ggml_nelements(t));
    for (float & x : v) x = d(rng);
    ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
}

// ── 1. Kernel: final state == last F32 checkpoint ────────────────────────

void kernel_case(ggml_backend_t backend, int S, int H, int T, bool raw_gates,
                 const char * route) {
    ggml_init_params ip{};
    ip.mem_size = 32 * ggml_tensor_overhead() + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { CHECK_MSG(false, "ggml_init"); return; }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, 1);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, 1);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, 1);
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, 1);
    ggml_tensor * persist = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, T);
    ggml_tensor * gate_ba = raw_gates
        ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * H) : nullptr;
    // The chain capture verify's op: non-in-place, persistent checkpoints in
    // src[7] (qwen35_target_graph.cpp build_delta_net_block).
    ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, beta, state);
    result->src[7] = persist;
    if (gate_ba) ggml_gated_delta_net_set_raw_gates(result, gate_ba);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { CHECK_MSG(false, "alloc"); ggml_free(ctx); return; }

    std::mt19937 rng(1234u + (unsigned)(S * 131 + T * 7 + (raw_gates ? 1 : 0)));
    const float qk = 1.0f / std::sqrt((float)S);
    upload_random(q, rng, -qk, qk);
    upload_random(k, rng, -qk, qk);
    upload_random(v, rng, -1.0f, 1.0f);
    upload_random(state, rng, -0.5f, 0.5f);
    if (raw_gates) {
        upload_random(g, rng, -2.0f, 2.0f);      // alpha before softplus
        upload_random(beta, rng, -2.0f, 2.0f);   // beta before sigmoid
        std::vector<float> ba((size_t)2 * H);
        for (int h = 0; h < H; ++h) {
            ba[(size_t)h] = 0.1f * (float)h;            // dt_bias
            ba[(size_t)H + h] = -1.0f - 0.05f * (float)h;  // A
        }
        ggml_backend_tensor_set(gate_ba, ba.data(), 0, ggml_nbytes(gate_ba));
    } else {
        upload_random(g, rng, -1.0f, -0.05f);    // log decay
        upload_random(beta, rng, 0.1f, 0.9f);
    }

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    CHECK_MSG(st == GGML_STATUS_SUCCESS, "%s S=%d T=%d compute", route, S, T);
    if (st == GGML_STATUS_SUCCESS) {
        const size_t f32 = sizeof(float);
        const size_t state_bytes = (size_t)S * S * H * f32;
        const size_t attn_bytes = (size_t)S * H * T * f32;
        const Bytes final_state = read_bytes(result, attn_bytes, state_bytes);
        const Bytes last = read_bytes(persist, (size_t)(T - 1) * persist->nb[3], state_bytes);
        CHECK_MSG(final_state == last,
                  "%s S=%d T=%d raw=%d: final state != intermediate[T-1]",
                  route, S, T, (int)raw_gates);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

void run_kernel_cases(ggml_backend_t backend) {
    struct Route { const char * name; int route; };  // ggml_backend_cuda_set_gdn_route_override
    const Route routes[] = {
        {"grouped-cols", 1},
        {"scalar", 0},
    };
    for (const Route & r : routes) {
        ggml_backend_cuda_set_gdn_route_override(r.route);
        for (int T = 1; T <= 16; ++T) {
            for (bool raw : {false, true}) {
                kernel_case(backend, 128, 4, T, raw, r.name);
            }
        }
    }
    ggml_backend_cuda_set_gdn_route_override(-1);
    for (int T : {1, 2, 7, 16}) kernel_case(backend, 64, 4, T, true, "scalar-64");
}

// ── 2. Target: the spec-decode call sequences ───────────────────────────

struct Digest {
    uint64_t h = 1469598103934665603ull;
    void add(const void * p, size_t n) {
        const auto * b = static_cast<const uint8_t *>(p);
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    }
    void add(const Bytes & b) { add(b.data(), b.size()); }
    void add(const std::vector<int32_t> & v) { add(v.data(), v.size() * sizeof(int32_t)); }
};

struct Run {
    TinyGdnTarget & t;
    Qwen35DFlashTarget & target;
    Digest digest;
    int base = 0;
    int salt = 0;

    std::vector<int32_t> tokens(int n) {
        std::vector<int32_t> out((size_t)n);
        for (int i = 0; i < n; ++i) {
            out[(size_t)i] = (int32_t)((salt * 37 + i * 13 + 11) % TinyGdnTarget::kVocab);
        }
        salt++;
        return out;
    }
    const ggml_tensor * live() const { return t.cache.ssm_state[0]; }
    Bytes state() const { return read_all(t.cache.ssm_state[0]); }
    Bytes conv() const { return read_all(t.cache.conv_state[0]); }
    Bytes checkpoint(int idx) const {
        const ggml_tensor * inter = t.cache.ssm_intermediate[0];
        return read_bytes(inter, (size_t)idx * inter->nb[3], ggml_nbytes(live()));
    }

    bool verify(const std::vector<int32_t> & toks, bool capture) {
        int last = -1;
        std::vector<int32_t> argmax;
        if (!target.verify_batch(toks, base, last, &argmax, capture)) return false;
        digest.add(argmax);
        return true;
    }

    void commit(const char * where) {
        digest.add(state());
        digest.add(conv());
        (void)where;
    }

    // A repair after the step's emit loop: restore twice (EOS, then budget
    // close), replay `n` tokens of the step.
    void repair(const std::vector<int32_t> & toks, int n,
                const Bytes & pre_state, const Bytes & pre_conv) {
        CHECK_MSG(target.restore_kv(), "restore_kv");
        CHECK_MSG(state() == pre_state && conv() == pre_conv, "restore != pre-step");
        CHECK_MSG(target.restore_kv(), "restore_kv again");
        CHECK_MSG(state() == pre_state && conv() == pre_conv, "second restore != pre-step");
        t.cache.cur_pos = base;
        if (n > 0) {
            std::vector<int32_t> prefix(toks.begin(), toks.begin() + n);
            CHECK_MSG(verify(prefix, /*capture=*/false), "replay");
            base += n;
            t.cache.cur_pos = base;
        }
        commit("repair");
    }
};

void run_target(ggml_backend_t backend, Digest & digest_out) {
    TinyGdnTarget t(backend, /*state_dim=*/128, /*raw_gates=*/true);
    if (!t.ok) { CHECK_MSG(false, "tiny target"); return; }
    if (t.cache.ssm_intermediate.empty() || !t.cache.ssm_intermediate[0] ||
        t.cache.ssm_intermediate[0]->type != GGML_TYPE_F32) {
        CHECK_MSG(false, "needs F32 SSM checkpoints (LUCE_SINGLE_CHAIN_CHECKPOINT_F32)");
        return;
    }
    StepGraph sg;
    {
        Qwen35DFlashTarget target(t.w, t.cache, t.backend, sg,
                                  luce::common::KQ_MASK_PAD, /*fa_window=*/0);
        target.set_fast_rollback(true);
        Run run{t, target};

        // Prompt: a replay-style forward leaves a non-trivial state.
        CHECK_MSG(run.verify(run.tokens(8), /*capture=*/false), "prompt");
        run.base = 8;
        run.commit("prompt");

        // SINGLE_REPLAY engages on every 16-token capture verify here, provided
        // the GDN op takes the grouped-cols kernel on this device (the query
        // gdn_single_replay_ids asks, with the same device and head size).
        // Elsewhere (pre-Ampere, or Ampere NVIDIA, where the scalar kernel
        // runs) the part stays inert and the verify
        // keeps the F32 checkpoints, so the committed state is checked against
        // them directly, as in the unset run: a verify that wrongly recorded a
        // replay log would leave them stale.
        const bool replay_switches = luce::common::gdn_commit::parts().keep_base &&
                                     luce::common::gdn_commit::parts().single_replay;
        const int64_t head_v_dim = t.w.ssm_d_inner / t.w.ssm_dt_rank;
        const bool replay = replay_switches &&
            ggml_backend_cuda_gdn_chain_uses_grouped_cols(
                ggml_backend_cuda_get_device_id(t.backend), head_v_dim);
        if (replay_switches && !replay) {
            std::fprintf(stderr,
                         "test_gdn_single_commit_gpu: the GDN op does not take the "
                         "grouped-cols kernel on this device; SINGLE_REPLAY stays "
                         "inert, checking the F32-checkpoint fallback\n");
        }

        // Fast rollback for every accept count, the all-accepted case included.
        for (int c = 1; c <= 16; ++c) {
            const Bytes pre_state = run.state();
            const Bytes pre_conv = run.conv();
            const std::vector<int32_t> toks = run.tokens(16);
            CHECK_MSG(target.snapshot_kv(), "snapshot_kv");
            const size_t grouped_before = ggml_backend_cuda_get_gdn_grouped_cols_launch_count();
            CHECK_MSG(run.verify(toks, /*capture=*/true), "verify");
            if (replay) {
                CHECK_MSG(ggml_backend_cuda_get_gdn_grouped_cols_launch_count() ==
                              grouped_before + 1,
                          "single-replay verify did not run the grouped-cols kernel");
            }
            const Bytes expect = replay ? Bytes{} : run.checkpoint(c - 1);
            const size_t commits_before = ggml_backend_cuda_get_gdn_chain_commit_count();
            CHECK_MSG(target.rollback_to(run.base, c), "rollback_to commit_n=%d", c);
            // The target must take SINGLE_REPLAY exactly where the query says:
            // one chain commit per rollback with it, none without.
            CHECK_MSG(ggml_backend_cuda_get_gdn_chain_commit_count() ==
                          commits_before + (replay ? 1u : 0u),
                      "commit_n=%d: %s", c,
                      replay ? "rollback_to did not run the chain commit"
                             : "rollback_to ran the chain commit with SINGLE_REPLAY off");
            CHECK_MSG(replay || run.state() == expect,
                      "commit_n=%d: committed state != intermediate[commit_n-1]", c);
            CHECK_MSG(t.cache.cur_pos == run.base + c, "cur_pos");
            run.commit("rollback_to");
            if (c % 4 == 0) {
                run.repair(toks, c - 1, pre_state, pre_conv);
            } else {
                run.base += c;
                t.cache.cur_pos = run.base;
            }
        }

        // Slow path: restore + replay, then a repair on top.
        {
            const Bytes pre_state = run.state();
            const Bytes pre_conv = run.conv();
            const std::vector<int32_t> toks = run.tokens(16);
            CHECK_MSG(target.snapshot_kv(), "snapshot_kv");
            CHECK_MSG(run.verify(toks, /*capture=*/true), "verify");
            const int base0 = run.base;
            run.repair(toks, 5, pre_state, pre_conv);
            run.base = base0;
            run.repair(toks, 3, pre_state, pre_conv);
        }

        // AR-burst seed steps: one-token capture verify, no rollback_to; the
        // state is as verify left it, which is its single checkpoint.
        for (int i = 0; i < 4; ++i) {
            const Bytes pre_state = run.state();
            const Bytes pre_conv = run.conv();
            const bool snap = i % 2 == 0;
            if (snap) CHECK_MSG(target.snapshot_kv(), "snapshot_kv");
            const std::vector<int32_t> toks = run.tokens(1);
            CHECK_MSG(run.verify(toks, /*capture=*/true), "AR verify");
            CHECK_MSG(run.state() == run.checkpoint(0), "AR seed state");
            run.base += 1;
            t.cache.cur_pos = run.base;
            run.commit("AR seed");
            if (i == 2) {
                // Budget close on the AR step: back to the pre-step state.
                CHECK_MSG(target.restore_kv(), "AR restore");
                CHECK_MSG(run.state() == pre_state && run.conv() == pre_conv,
                          "AR restore != pre-step");
                run.base -= 1;
                t.cache.cur_pos = run.base;
            }
        }

        // A failed step: the verify ran, the loop restores and gives up.
        {
            const Bytes pre_state = run.state();
            const Bytes pre_conv = run.conv();
            CHECK_MSG(target.snapshot_kv(), "snapshot_kv");
            CHECK_MSG(run.verify(run.tokens(16), /*capture=*/true), "verify");
            CHECK_MSG(target.restore_kv(), "restore_kv");
            CHECK_MSG(run.state() == pre_state && run.conv() == pre_conv,
                      "restore after a failed step != pre-step");
            t.cache.cur_pos = run.base;
        }

        CHECK_MSG(target.finish_speculative_state(), "finish_speculative_state");
        run.commit("finish");
        digest_out = run.digest;

        // Next request: the recurrent state starts from zero in base_buf.
        luce::common::reset_recurrent_state(t.cache);
        CHECK_MSG(!t.cache.ssm_state_swapped, "reset left the pairs exchanged");
        CHECK_MSG(t.cache.ssm_state[0]->buffer == t.cache.base_buf,
                  "reset left the live state outside base_buf");
        const Bytes zeros(ggml_nbytes(t.cache.ssm_state[0]), 0);
        CHECK_MSG(run.state() == zeros, "reset did not zero the live state");
    }
    luce::common::step_graph_destroy(sg);
}

}  // namespace

int main(int argc, char ** argv) {
    const char * digest_out = nullptr;
    const char * digest_expect = nullptr;
    const char * parts_arg = nullptr;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--digest-out") == 0) digest_out = argv[i + 1];
        else if (std::strcmp(argv[i], "--digest-expect") == 0) digest_expect = argv[i + 1];
        else if (std::strcmp(argv[i], "--parts") == 0) parts_arg = argv[i + 1];
    }
    // off, or any of k (KEEP_BASE), s (SWAP_SNAPSHOT), r (SINGLE_REPLAY).
    luce::common::gdn_commit::Parts parts{false, false, false};
    if (!parts_arg) {
        std::fprintf(stderr, "test_gdn_single_commit_gpu: --parts off|k|ks|kr|ksr required\n");
        return 2;
    }
    if (std::strcmp(parts_arg, "off") != 0) {
        parts.keep_base = std::strchr(parts_arg, 'k') != nullptr;
        parts.swap_snapshot = std::strchr(parts_arg, 's') != nullptr;
        parts.single_replay = std::strchr(parts_arg, 'r') != nullptr;
    }
    luce::common::gdn_commit::set_parts_for_test(parts);
    if (ggml_backend_cuda_get_device_count() < 1) {
        std::fprintf(stderr, "test_gdn_single_commit_gpu: no GPU\n");
        return 77;
    }
    ggml_backend_t kernel_backend = ggml_backend_cuda_init(0);
    if (!kernel_backend) {
        std::fprintf(stderr, "test_gdn_single_commit_gpu: device 0 present but its backend failed to init\n");
        return 1;
    }
    run_kernel_cases(kernel_backend);
    ggml_backend_free(kernel_backend);

    Digest digest;
    run_target(ggml_backend_cuda_init(0), digest);   // TinyGdnTarget owns it
    std::fprintf(stderr, "test_gdn_single_commit_gpu: digest %016" PRIx64 "\n", digest.h);

    if (digest_out) {
        FILE * f = std::fopen(digest_out, "w");
        CHECK_MSG(f != nullptr, "open %s", digest_out);
        if (f) { std::fprintf(f, "%016" PRIx64 "\n", digest.h); std::fclose(f); }
    }
    if (digest_expect) {
        FILE * f = std::fopen(digest_expect, "r");
        CHECK_MSG(f != nullptr, "open %s (run --parts off first)", digest_expect);
        if (f) {
            char line[64] = {};
            const bool read = std::fgets(line, sizeof(line), f) != nullptr;
            std::fclose(f);
            char mine[64];
            std::snprintf(mine, sizeof(mine), "%016" PRIx64 "\n", digest.h);
            CHECK_MSG(read && std::strcmp(line, mine) == 0,
                      "digest %s differs from the off run's %s", mine, line);
        }
    }
    if (failures) std::fprintf(stderr, "test_gdn_single_commit_gpu: %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
