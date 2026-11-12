#include "common/draft_attn_mask.h"
#include "host_check.h"

#include <cstdio>
#include <vector>

using namespace luce::common;

static int g_checks = 0;

static constexpr int CAP = 32;
static constexpr int Q = 8;
static constexpr int KV_TOTAL = 64;   // cap + q + padding

// Ring holding positions [committed - n, committed) at slot p % CAP.
static std::vector<int32_t> ring(int committed) {
    std::vector<int32_t> sp(CAP, -1);
    for (int p = std::max(0, committed - CAP); p < committed; p++) sp[p % CAP] = p;
    return sp;
}

static bool vis(const std::vector<uint16_t> & m, int q, int col) {
    return m[(size_t)q * KV_TOTAL + col] == kDraftMaskF16Zero;
}
static int ctx_slot(int pos) { return pos % CAP; }

int main() {
    // --- non-causal SWA: per-query window over ctx, noise fully visible ---
    {
        const int committed = 100, W = 16;
        auto sp = ring(committed);
        auto m = build_draft_swa_mask(CAP, KV_TOTAL, Q, committed, sp.data(), W, false);
        for (int q = 0; q < Q; q++) {
            const int qpos = committed + q;
            for (int p = committed - CAP; p < committed; p++)
                CHECK(vis(m, q, ctx_slot(p)) == (qpos - p < W));
            for (int j = 0; j < Q; j++) CHECK(vis(m, q, CAP + j));     // incl. j > q
            for (int c = CAP + Q; c < KV_TOTAL; c++) CHECK(!vis(m, q, c));
        }
        // window slides per row: row 0 sees 85..99 (15 keys), row 7 sees 92..99 (8 keys)
        CHECK(vis(m, 0, ctx_slot(85)) && !vis(m, 0, ctx_slot(84)));
        CHECK(vis(m, 7, ctx_slot(92)) && !vis(m, 7, ctx_slot(91)));
    }
    // window larger than history: everything in the ring is visible
    {
        const int committed = 10;
        auto sp = ring(committed);
        auto m = build_draft_swa_mask(CAP, KV_TOTAL, Q, committed, sp.data(), 2048, false);
        for (int q = 0; q < Q; q++) {
            for (int p = 0; p < committed; p++) CHECK(vis(m, q, ctx_slot(p)));
            for (int s = committed; s < CAP; s++) CHECK(!vis(m, q, s));   // empty slots
        }
    }
    // window == 0 (disabled) non-causal: full ring visible, still no stale slots
    {
        const int committed = 100;
        auto sp = ring(committed);
        auto m = build_draft_swa_mask(CAP, KV_TOTAL, Q, committed, sp.data(), 0, false);
        for (int q = 0; q < Q; q++)
            for (int p = committed - CAP; p < committed; p++) CHECK(vis(m, q, ctx_slot(p)));
    }
    // stale slot (position >= committed after a rollback) is never visible
    {
        const int committed = 50;
        auto sp = ring(committed);
        sp[7] = 70;   // shadow from a rolled-back draft
        auto m = build_draft_swa_mask(CAP, KV_TOTAL, Q, committed, sp.data(), 2048, false);
        for (int q = 0; q < Q; q++) CHECK(!vis(m, q, 7));
    }

    // --- causal SWA (DFlash 1 behaviour, must stay bit-identical) ---
    {
        const int committed = 100, W = 16;
        auto sp = ring(committed);
        auto m = build_draft_swa_mask(CAP, KV_TOTAL, Q, committed, sp.data(), W, true);
        for (int q = 0; q < Q; q++) {
            // one window shared by all rows, anchored at committed: [84, 100)
            for (int p = committed - CAP; p < committed; p++)
                CHECK(vis(m, q, ctx_slot(p)) == (p >= committed - W));
            for (int j = 0; j < Q; j++) CHECK(vis(m, q, CAP + j) == (j <= q));
        }
    }
    {
        const int committed = 20;   // shorter than the window
        auto sp = ring(committed);
        auto m = build_draft_swa_mask(CAP, KV_TOTAL, Q, committed, sp.data(), 2048, true);
        for (int q = 0; q < Q; q++)
            for (int p = 0; p < committed; p++) CHECK(vis(m, q, ctx_slot(p)));
    }
    // causal and non-causal rows agree on ctx when the window covers the ring
    {
        const int committed = 100;
        auto sp = ring(committed);
        auto a = build_draft_swa_mask(CAP, KV_TOTAL, Q, committed, sp.data(), 4096, true);
        auto b = build_draft_swa_mask(CAP, KV_TOTAL, Q, committed, sp.data(), 4096, false);
        for (int q = 0; q < Q; q++)
            for (int s = 0; s < CAP; s++) CHECK(vis(a, q, s) == vis(b, q, s));
    }

    std::printf("test_draft_attn_mask: %d checks passed\n", g_checks);
    return 0;
}
