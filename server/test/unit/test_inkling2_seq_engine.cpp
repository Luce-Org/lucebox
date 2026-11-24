// Unit + contract tests for SlotKv v2, Inkling2SlotStateAdapter, Inkling2SlotKv,
// and Inkling2SeqEngine (Phase 3 concurrent serving on Inkling 2 Mini).
//
// Tests:
//   1. SlotKv v2: PoolSlotKv backwards-compatibility + Inkling2SlotKv dual-tier
//      build_table_patch() (Tier 0 + Tier 1.5 disjoint tables with -1 holes) & caps().
//   2. Inkling2SlotStateAdapter GPU round-trip bit-exactness (1D 48-slot layout:
//      ShortConv + SWA-1024 + DSpark last_taps_dev, plus preamble_end()).
//   3. Inkling2SlotStateAdapter GPU round-trip bit-exactness (2DSD Tree + Shared-KV
//      W=64 tail rings + 5-layer sliding drafter ring).
//   4. Full official check_seq_engine_contract(engine) on a 2-slot Inkling2SeqEngine
//      (0 violations) + offload_kv / restore_kv / evict_kv / recompute resume cycle.

#include "inkling2/inkling2_kvflash.h"
#include "inkling2/inkling2_seq_engine.h"
#include "inkling2/inkling2_slot_state.h"
#include "common/concurrency/slot_kv.h"
#include "seq_engine_contract.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace luce::common;

namespace {

static int g_pass = 0;
static int g_fail = 0;

static void check(bool ok, const char * name, const char * detail = "") {
    if (ok) {
        std::printf("[PASS] %s %s\n", name, detail);
        ++g_pass;
    } else {
        std::printf("[FAIL] %s %s\n", name, detail);
        ++g_fail;
    }
}

static void fill_tensor_pattern(ggml_tensor * t, uint8_t seed) {
    if (!t) return;
    const size_t n = ggml_nbytes(t);
    std::vector<uint8_t> buf(n);
    for (size_t i = 0; i < n; ++i) {
        buf[i] = (uint8_t) ((i * 31u + seed * 17u + 13u) & 0xFFu);
    }
    ggml_backend_tensor_set(t, buf.data(), 0, n);
}

static bool tensor_matches_pattern(ggml_tensor * t, uint8_t seed) {
    if (!t) return false;
    const size_t n = ggml_nbytes(t);
    std::vector<uint8_t> got(n);
    ggml_backend_tensor_get(t, got.data(), 0, n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t exp = (uint8_t) ((i * 31u + seed * 17u + 13u) & 0xFFu);
        if (got[i] != exp) return false;
    }
    return true;
}

static Inkling2Weights make_synthetic_weights(bool with_2dsd) {
    Inkling2Weights w;
    w.n_embd = 64;
    w.n_head = 4;
    w.n_head_kv = 2;
    w.head_dim = 32;
    w.swa_window = 64;
    w.n_layer_phys = 4;
    w.n_layer_eff = 6; // 0 (SWA), core [1,3) x 2 passes -> 1,2,1,2, then 3 (Global)
    w.is_loop = true;
    w.core_start = 1;
    w.core_end = 3;
    w.n_passes = 2;
    w.n_vocab_unpadded = 256;
    w.eos_id = 2;
    w.eos_chat_id = 3;
    w.layers.resize((size_t) w.n_layer_phys);
    w.layers[0].is_swa = true;
    w.layers[1].is_swa = true;
    w.layers[2].is_swa = false;
    w.layers[3].is_swa = false;
    if (with_2dsd) {
        w.draft_mode = Inkling2DraftMode::TwoDSD;
        w.twodsd.loaded = true;
        w.twodsd.n_layer = 2;
        w.twodsd.n_head_kv = 2;
        w.twodsd.head_dim = 32;
    }
    return w;
}

// Allocate a lightweight synthetic 1D Inkling2Cache on `be` (conv_state + SWA + global + last_taps_dev)
static void alloc_synthetic_1d_cache(ggml_backend_t be, const Inkling2Weights & w, Inkling2Cache & c) {
    c.max_ctx = 512;
    c.swa_cap = 128;
    c.kv_type = GGML_TYPE_F16;
    const size_t nt = (size_t) w.n_layer_eff * 3 + 4;
    struct ggml_init_params gp = { nt * ggml_tensor_overhead() + 4096, nullptr, true };
    c.ctx = ggml_init(gp);
    c.k_cache.resize((size_t) w.n_layer_eff, nullptr);
    c.v_cache.resize((size_t) w.n_layer_eff, nullptr);
    c.conv_state.resize((size_t) w.n_layer_eff, nullptr);
    for (int il = 0; il < w.n_layer_eff; ++il) {
        const int ip = w.phys_layer_for_slot(il);
        const int rows = w.layers[ip].is_swa ? c.swa_cap : 64;
        c.k_cache[il]    = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, w.head_dim, rows, w.n_head_kv);
        c.v_cache[il]    = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, w.head_dim, rows, w.n_head_kv);
        c.conv_state[il] = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, w.n_embd, 3);
    }
    c.last_taps_dev = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, w.n_embd, 4);
    c.buf = ggml_backend_alloc_ctx_tensors(c.ctx, be);
}

static void free_synthetic_cache(Inkling2Cache & c) {
    if (c.kvflash) {
        c.kvflash->detach();
        c.kvflash.reset();
    }
    if (c.buf) { ggml_backend_buffer_free(c.buf); c.buf = nullptr; }
    if (c.ctx) { ggml_free(c.ctx); c.ctx = nullptr; }
}

static void test_slot_kv_v2_and_dual_tier_patch(ggml_backend_t be) {
    // 1a. PoolSlotKv backwards-compatibility + build_table_patch()
    PagedKvPool base_pool(/*blocks=*/16, /*max_seqs=*/2, /*block_size=*/64);
    PoolSlotKv pool_kv(base_pool);
    PagedKvSequenceHandle h{};
    SlotAcquireOpts opts;
    opts.total_tokens = 256;
    const bool acq_ok = (pool_kv.acquire_reserved(101, 128, opts, h) == PagedKvStatus::Ok);
    const auto app = pool_kv.append(h, 128, false);
    const bool app_ok = (app.status == PagedKvStatus::Ok);
    SlotTablePatch p;
    const bool patch_ok = pool_kv.build_table_patch(h, 128, 16, p, /*remote=*/false) &&
                          p.first_block == 0 && p.blocks.size() == 2 && p.blocks[0] >= 0 && p.blocks[1] >= 0;
    const bool remote_rejected = !pool_kv.build_table_patch(h, 128, 16, p, /*remote=*/true);
    pool_kv.release(h);

    // 1b. Inkling2SlotKv with 3-tier Inkling2KvFlash attached on slot 0
    constexpr int L = 2, D = 128, Hq = 8, Hk = 4, CT = 64;
    constexpr int T0_TOK = 320;  // 5 blocks (1 sink + 2 hot tail + 2 evictable/recall)
    constexpr int T15_TOK = 192; // 3 blocks
    constexpr int MAX_CTX = 640; // 10 chunks
    struct ggml_init_params gp = { 32 * ggml_tensor_overhead() + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    std::vector<ggml_tensor *> t0_k(L), t0_v(L), t15_k(L), t15_v(L);
    for (int l = 0; l < L; ++l) {
        t0_k[l]  = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, D, T0_TOK, Hk);
        t0_v[l]  = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, D, T0_TOK, Hk);
        t15_k[l] = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, D, T15_TOK, Hk);
        t15_v[l] = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, D, T15_TOK, Hk);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    Inkling2Cache c0, c1;
    c0.kvflash = std::make_unique<Inkling2KvFlash>();
    Inkling2KvFlashConfig cfg;
    cfg.enabled = true;
    cfg.chunk_tokens = CT;
    cfg.tier0_pool_tokens = T0_TOK;
    cfg.tier15_pool_tokens = T15_TOK;
    cfg.sink_chunks = 1;
    cfg.hot_tail_chunks = 2;
    cfg.recall_blocks = 1;
    cfg.mode4_enabled = false;
    const bool attached = c0.kvflash->attach(cfg, MAX_CTX, D, Hq, Hk, be, be, t0_k, t0_v, t15_k, t15_v);
    for (int c = 0; c < 10; ++c) c0.kvflash->prepare_span(c * CT, CT);

    Inkling2SlotKv skv(MAX_CTX, {&c0, &c1}, CT);
    const SlotKvCaps caps = skv.caps();
    PagedKvSequenceHandle h0{};
    const bool acq0 = (skv.acquire_reserved(201, MAX_CTX, h0) == PagedKvStatus::Ok);
    SlotTablePatch plocal, premote;
    const bool p0_ok = skv.build_table_patch(h0, MAX_CTX, 16, plocal, /*remote=*/false);
    const bool p1_ok = skv.build_table_patch(h0, MAX_CTX, 16, premote, /*remote=*/true);

    int n_t0 = 0, n_t15 = 0, n_hole = 0;
    bool disjoint = true;
    if (p0_ok && p1_ok && plocal.blocks.size() == 10 && premote.blocks.size() == 10) {
        for (size_t i = 0; i < 10; ++i) {
            const int b0 = plocal.blocks[i];
            const int b1 = premote.blocks[i];
            if (b0 >= 0 && b1 >= 0) disjoint = false;
            else if (b0 >= 0) ++n_t0;
            else if (b1 >= 0) ++n_t15;
            else ++n_hole;
        }
    } else {
        disjoint = false;
    }
    skv.release(h0);

    char detail[160];
    std::snprintf(detail, sizeof(detail),
                  "(pool=%d caps[mut=%d dual=%d park=%d] t0=%d t15=%d hole=%d disjoint=%d)",
                  (acq_ok && app_ok && patch_ok && remote_rejected) ? 1 : 0,
                  caps.mutates_existing_blocks ? 1 : 0, caps.dual_tier_tables ? 1 : 0,
                  caps.parkable_offload ? 1 : 0, n_t0, n_t15, n_hole, disjoint ? 1 : 0);
    check(acq_ok && app_ok && patch_ok && remote_rejected &&
          attached && acq0 && p0_ok && p1_ok && disjoint &&
          caps.mutates_existing_blocks && caps.dual_tier_tables && caps.parkable_offload &&
          n_t0 == 5 && n_t15 == 3 && n_hole == 2,
          "slot_kv_v2_and_dual_tier_patch", detail);

    c0.kvflash->detach();
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static void test_slot_state_adapter_1d(ggml_backend_t be) {
    Inkling2Weights w = make_synthetic_weights(/*with_2dsd=*/false);
    Inkling2Cache c0, c1;
    alloc_synthetic_1d_cache(be, w, c0);
    alloc_synthetic_1d_cache(be, w, c1);

    // Populate c0 with deterministic byte patterns across conv_state, SWA K/V, and last_taps_dev
    for (int il = 0; il < w.n_layer_eff; ++il) {
        fill_tensor_pattern(c0.conv_state[il], (uint8_t) (10 + il));
        if (w.layers[w.phys_layer_for_slot(il)].is_swa) {
            fill_tensor_pattern(c0.k_cache[il], (uint8_t) (40 + il));
            fill_tensor_pattern(c0.v_cache[il], (uint8_t) (70 + il));
        }
    }
    fill_tensor_pattern(c0.last_taps_dev, 99);
    c0.cur_pos = 317;
    c0.dspark_ctx_pos = 315;

    Inkling2SlotStateAdapter adapter(w, {&c0, &c1}, /*include_draft_ring=*/false);
    adapter.set_preamble_markers(/*user=*/100, /*system=*/99);
    const int p_end = adapter.preamble_end({99, 10, 11, 12, 100, 20, 21});

    SlotStateSlab slab;
    const bool saved = adapter.save_slot_state(0, slab);
    const bool size_ok = (slab.size() == adapter.slab_bytes() && slab.size() > 64);

    // Zero c0, then restore into c1 and verify bit-exact equality!
    adapter.reset_slot_state(0);
    const bool restored = adapter.restore_slot_state(1, slab);
    bool exact = (c1.cur_pos == 317 && c1.dspark_ctx_pos == 315);
    for (int il = 0; il < w.n_layer_eff; ++il) {
        if (!tensor_matches_pattern(c1.conv_state[il], (uint8_t) (10 + il))) exact = false;
        if (w.layers[w.phys_layer_for_slot(il)].is_swa) {
            if (!tensor_matches_pattern(c1.k_cache[il], (uint8_t) (40 + il))) exact = false;
            if (!tensor_matches_pattern(c1.v_cache[il], (uint8_t) (70 + il))) exact = false;
        }
    }
    if (!tensor_matches_pattern(c1.last_taps_dev, 99)) exact = false;

    // Tampered magic must fail safely
    SlotStateSlab bad = slab;
    bad[0] ^= 0xFF;
    const bool rejected_bad = !adapter.restore_slot_state(1, bad);

    char detail[128];
    std::snprintf(detail, sizeof(detail), "(slab=%zu B exact=%d preamble_end=%d bad_rej=%d)",
                  slab.size(), exact ? 1 : 0, p_end, rejected_bad ? 1 : 0);
    check(saved && size_ok && restored && exact && p_end == 4 && rejected_bad,
          "slot_state_adapter_1d_roundtrip", detail);

    free_synthetic_cache(c0);
    free_synthetic_cache(c1);
}

static void test_slot_state_adapter_2dsd_shared_kv(ggml_backend_t be) {
    Inkling2Weights w = make_synthetic_weights(/*with_2dsd=*/true);
    auto make_tree_cache = [&](Inkling2Cache & c) {
        c.max_ctx = 512;
        c.swa_cap = 128;
        c.kv_type = GGML_TYPE_F16;
        auto & tc = c.tree;
        tc.active = true;
        tc.prefix_len = 0;
        tc.global_pool_cap = 128;
        tc.shared_kv_window = 64;
        tc.shared_tail_cap = 96;
        tc.draft_cap = 128;
        tc.planes = {1, 2, 2, 3}; // ip=2 is looped global (2 planes -> 1 tail ring), ip=3 is coda global (3 planes -> 2 tail rings)

        struct ggml_init_params gp = { 64 * ggml_tensor_overhead() + 4096, nullptr, true };
        c.ctx = ggml_init(gp);
        tc.prefix_k.resize((size_t) w.n_layer_phys, nullptr);
        tc.prefix_v.resize((size_t) w.n_layer_phys, nullptr);
        tc.prefix_conv.resize((size_t) w.n_layer_phys, nullptr);
        for (int ip = 0; ip < w.n_layer_phys; ++ip) {
            const int pl = tc.planes[ip];
            if (w.layers[ip].is_swa) {
                tc.prefix_k[ip] = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, w.head_dim, pl * c.swa_cap, w.n_head_kv);
                tc.prefix_v[ip] = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, w.head_dim, pl * c.swa_cap, w.n_head_kv);
            } else {
                const int rows = tc.global_pool_cap + (pl - 1) * tc.shared_tail_cap;
                tc.prefix_k[ip] = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, w.head_dim, rows, w.n_head_kv);
                tc.prefix_v[ip] = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, w.head_dim, rows, w.n_head_kv);
            }
            tc.prefix_conv[ip] = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, w.n_embd, 3, pl);
        }
        tc.draft_k.resize((size_t) w.twodsd.n_layer, nullptr);
        tc.draft_v.resize((size_t) w.twodsd.n_layer, nullptr);
        for (int dl = 0; dl < w.twodsd.n_layer; ++dl) {
            tc.draft_k[dl] = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, w.twodsd.head_dim, tc.draft_cap, w.twodsd.n_head_kv);
            tc.draft_v[dl] = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, w.twodsd.head_dim, tc.draft_cap, w.twodsd.n_head_kv);
        }
        c.buf = ggml_backend_alloc_ctx_tensors(c.ctx, be);
    };

    Inkling2Cache c0, c1;
    make_tree_cache(c0);
    make_tree_cache(c1);

    // Fill c0 patterns; also place distinct canary bytes in c1's [0..global_pool_cap) global pool
    // to prove restore_slot_state ONLY overwrites the Shared-KV tail ring [global_pool_cap..rows)!
    for (int ip = 0; ip < w.n_layer_phys; ++ip) {
        fill_tensor_pattern(c0.tree.prefix_conv[ip], (uint8_t) (20 + ip));
        fill_tensor_pattern(c0.tree.prefix_k[ip], (uint8_t) (50 + ip));
        fill_tensor_pattern(c0.tree.prefix_v[ip], (uint8_t) (80 + ip));
        if (!w.layers[ip].is_swa) {
            ggml_backend_tensor_memset(c1.tree.prefix_k[ip], 0xCC, 0, ggml_nbytes(c1.tree.prefix_k[ip]));
            ggml_backend_tensor_memset(c1.tree.prefix_v[ip], 0xDD, 0, ggml_nbytes(c1.tree.prefix_v[ip]));
        }
    }
    for (int dl = 0; dl < w.twodsd.n_layer; ++dl) {
        fill_tensor_pattern(c0.tree.draft_k[dl], (uint8_t) (110 + dl));
        fill_tensor_pattern(c0.tree.draft_v[dl], (uint8_t) (130 + dl));
    }
    c0.tree.prefix_len = 412;

    Inkling2SlotStateAdapter adapter(w, {&c0, &c1}, /*include_draft_ring=*/true);
    SlotStateSlab slab;
    const bool saved = adapter.save_slot_state(0, slab);
    const bool restored = adapter.restore_slot_state(1, slab);

    bool exact = (c1.tree.prefix_len == 412);
    for (int ip = 0; ip < w.n_layer_phys; ++ip) {
        if (!tensor_matches_pattern(c1.tree.prefix_conv[ip], (uint8_t) (20 + ip))) exact = false;
        if (w.layers[ip].is_swa) {
            if (!tensor_matches_pattern(c1.tree.prefix_k[ip], (uint8_t) (50 + ip))) exact = false;
            if (!tensor_matches_pattern(c1.tree.prefix_v[ip], (uint8_t) (80 + ip))) exact = false;
        } else {
            // Verify global pool [0..pool_cap) still has 0xCC/0xDD canary, while tail [pool_cap..end) matches c0!
            const auto & tc = c1.tree;
            const size_t pool_b = (size_t) tc.global_pool_cap * tc.prefix_k[ip]->nb[1];
            const size_t tail_b = (size_t) (tc.planes[ip] - 1) * tc.shared_tail_cap * tc.prefix_k[ip]->nb[1];
            std::vector<uint8_t> pbuf(pool_b), t0(tail_b), t1(tail_b);
            for (int h = 0; h < w.n_head_kv; ++h) {
                ggml_backend_tensor_get(c1.tree.prefix_k[ip], pbuf.data(), (size_t) h * tc.prefix_k[ip]->nb[2], pool_b);
                for (uint8_t b : pbuf) if (b != 0xCC) exact = false;
                ggml_backend_tensor_get(c0.tree.prefix_k[ip], t0.data(), (size_t) h * tc.prefix_k[ip]->nb[2] + pool_b, tail_b);
                ggml_backend_tensor_get(c1.tree.prefix_k[ip], t1.data(), (size_t) h * tc.prefix_k[ip]->nb[2] + pool_b, tail_b);
                if (t0 != t1) exact = false;
                ggml_backend_tensor_get(c0.tree.prefix_v[ip], t0.data(), (size_t) h * tc.prefix_v[ip]->nb[2] + pool_b, tail_b);
                ggml_backend_tensor_get(c1.tree.prefix_v[ip], t1.data(), (size_t) h * tc.prefix_v[ip]->nb[2] + pool_b, tail_b);
                if (t0 != t1) exact = false;
            }
        }
    }
    for (int dl = 0; dl < w.twodsd.n_layer; ++dl) {
        if (!tensor_matches_pattern(c1.tree.draft_k[dl], (uint8_t) (110 + dl))) exact = false;
        if (!tensor_matches_pattern(c1.tree.draft_v[dl], (uint8_t) (130 + dl))) exact = false;
    }

    char detail[128];
    std::snprintf(detail, sizeof(detail), "(slab=%zu B exact=%d)", slab.size(), exact ? 1 : 0);
    check(saved && restored && exact, "slot_state_adapter_2dsd_shared_kv", detail);

    free_synthetic_cache(c0);
    free_synthetic_cache(c1);
}

static void test_seq_engine_contract_and_offload(ggml_backend_t be) {
    Inkling2Weights w = make_synthetic_weights(/*with_2dsd=*/false);
    Inkling2Cache c0, c1;
    alloc_synthetic_1d_cache(be, w, c0);
    alloc_synthetic_1d_cache(be, w, c1);

    // Deterministic synthetic forward hook (avoids needing a 17 GB GGUF in unit tests)
    auto fake_fwd = [&w](int slot, const int32_t * tokens, int n_tokens, int pos0,
                         bool need_logits, int /*capture_q_rows*/) -> Inkling2SeqStepResult {
        Inkling2SeqStepResult r;
        r.ok = true;
        const int32_t last = (n_tokens > 0) ? tokens[n_tokens - 1] : 1;
        const int32_t next = 10 + ((last + pos0 + slot * 7 + 1) % 200);
        r.greedy_token = next;
        if (need_logits) {
            r.logits.assign((size_t) w.n_vocab_unpadded, -5.0f);
            r.logits[(size_t) next] = 8.0f;
            r.logits[(size_t) ((next + 1) % w.n_vocab_unpadded)] = 7.5f;
        }
        return r;
    };

    Inkling2SeqEngine engine(be, w, {&c0, &c1}, /*max_ctx=*/512, /*prefill_chunk=*/64,
                             /*verify_width=*/1, fake_fwd);

    // 1. Official Lucebox SeqEngine contract checker (must return 0 violations!)
    const std::vector<std::string> violations = check_seq_engine_contract(engine);
    if (!violations.empty()) {
        for (const auto & v : violations) {
            std::fprintf(stderr, "  [CONTRACT VIOLATION] %s\n", v.c_str());
        }
    }
    char cdetail[64];
    std::snprintf(cdetail, sizeof(cdetail), "(violations=%zu)", violations.size());
    check(violations.empty(), "seq_engine_official_contract", cdetail);

    // 2. Offload -> restore -> evict -> recompute resume lifecycle
    SamplerCfg greedy{};
    auto a0 = engine.admit(9001, {11, 12, 13}, greedy);
    bool ok = (a0.status == SeqEngine::AdmitResult::Status::admitted);
    SeqEngine::StepPlan p1;
    p1.prefills.push_back({a0.slot, 3});
    auto r1 = engine.step(p1);
    ok = ok && r1.ok() && r1.prefills.size() == 1 &&
         r1.prefills[0].status == SeqEngine::PrefillOutput::Status::completed;
    const int32_t tok1 = ok ? r1.prefills[0].token : 0;

    // Plant canary pattern in slot's conv_state before offload_kv, clobber it, then restore_kv!
    fill_tensor_pattern(c0.conv_state[0], 77);
    std::string err;
    ok = ok && engine.offload_kv(a0.slot, 0, err);
    const auto st_parked = engine.kv_offload_state(a0.slot);
    ok = ok && st_parked.parked && !st_parked.recompute && st_parked.bytes == engine.state_adapter().slab_bytes();
    ggml_backend_tensor_memset(c0.conv_state[0], 0, 0, ggml_nbytes(c0.conv_state[0]));
    ok = ok && engine.kv_restore_feasible(a0.slot);
    ok = ok && engine.restore_kv(a0.slot, err);
    ok = ok && tensor_matches_pattern(c0.conv_state[0], 77);

    // Decode 1 step after restore
    SeqEngine::StepPlan p2;
    p2.decode.push_back({a0.slot, tok1});
    auto r2 = engine.step(p2);
    ok = ok && r2.ok() && r2.decode.size() == 1 && !r2.decode[0].failed;

    // Evict for recompute -> restore_kv -> re-prefill -> decode!
    ok = ok && engine.evict_kv(a0.slot, r2.decode[0].token, err);
    const auto st_rec = engine.kv_offload_state(a0.slot);
    ok = ok && st_rec.recompute && st_rec.parked;
    ok = ok && engine.restore_kv(a0.slot, err);
    SeqEngine::StepPlan p3;
    p3.prefills.push_back({a0.slot, 64});
    auto r3 = engine.step(p3);
    ok = ok && r3.ok() && r3.prefills.size() == 1 &&
         r3.prefills[0].status == SeqEngine::PrefillOutput::Status::completed;

    engine.retire(a0.slot);
    check(ok, "seq_engine_offload_restore_evict_cycle", err.c_str());

    free_synthetic_cache(c0);
    free_synthetic_cache(c1);
}

} // namespace

int main() {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) {
        std::fprintf(stderr, "No GPU backend available; skipping test_inkling2_seq_engine\n");
        return 77;
    }

    test_slot_kv_v2_and_dual_tier_patch(be);
    test_slot_state_adapter_1d(be);
    test_slot_state_adapter_2dsd_shared_kv(be);
    test_seq_engine_contract_and_offload(be);

    ggml_backend_free(be);
    std::printf("\n=== test_inkling2_seq_engine: %d/%d PASSED ===\n", g_pass, g_pass + g_fail);
    return (g_fail == 0) ? 0 : 1;
}
