// TDD: anchor transitive multi-pass.
//
// T1 — single-pass query-match preserved (regression pin, PASS today)
// T2 — single-pass misses chain hops (characterises limitation, PASS today)
// T3 — transitive rescues all hops (RED until Phase 2)
//
// Pure CPU — no GPU, no model load.

#include "../src/qwen3/anchor_scan.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define REQUIRE(cond) \
    do { if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s line %d: %s\n", __FILE__, __LINE__, #cond); \
        std::exit(1); \
    } } while (0)

static constexpr int32_t FILLER = 1;
static constexpr int32_t M1 = 1001, M2 = 1002, M3 = 1003;
static constexpr int CHUNK = 64;

// Place a marker 4-gram [FILLER, FILLER, MARKER, FILLER] at position pos.
static void place_marker_4gram(std::vector<int32_t>& ids, int pos, int32_t marker) {
    ids[(size_t)pos]     = FILLER;
    ids[(size_t)pos + 1] = FILLER;
    ids[(size_t)pos + 2] = marker;
    ids[(size_t)pos + 3] = FILLER;
}

// T1 — single-pass finds a query-matching marker in the body.
static void t1_single_pass_match() {
    const int N = 2048;
    std::vector<int32_t> ids((size_t)N, FILLER);

    // Body marker at pos 100 (chunk 1).
    place_marker_4gram(ids, 100, M3);
    // Same 4-gram in the query suffix at pos 2044 (inside query window).
    place_marker_4gram(ids, 2044, M3);

    const int q0 = 1948;  // N - 100
    std::vector<int32_t> query_pool(ids.begin() + q0, ids.end());

    const int n_chunks = (N + CHUNK - 1) / CHUNK;
    std::vector<uint8_t> forced((size_t)n_chunks, 0);

    dflash::qwen3::AnchorScanCfg cfg{CHUNK, /*anchor_radius=*/0,
                                     /*max_anchor_hits=*/8, /*ngram=*/4};
    dflash::qwen3::scan_and_force(ids, q0, query_pool, cfg, forced);

    // Chunk containing pos 100 must be forced.
    const int target_chunk = 100 / CHUNK;  // chunk 1
    REQUIRE(forced[(size_t)target_chunk] == 1);

    std::printf("T1 PASS: chunk %d forced by single-pass M3 match\n", target_chunk);
}

// T2 — single-pass only forces the direct match; chain hops stay unforced.
static void t2_single_pass_misses_hops() {
    const int N = 2048;
    std::vector<int32_t> ids((size_t)N, FILLER);

    // hop1 at pos 200 (chunk 3): contains M1.
    place_marker_4gram(ids, 200, M1);

    // hop2 at pos 600 (chunk 9): contains M2 + M1 (bridge to hop1).
    place_marker_4gram(ids, 600, M2);
    place_marker_4gram(ids, 604, M1);

    // hop3 at pos 1200 (chunk 18): contains M3 + M2 (bridge to hop2).
    place_marker_4gram(ids, 1200, M3);
    place_marker_4gram(ids, 1204, M2);

    // Query suffix at pos 2044: contains M3.
    place_marker_4gram(ids, 2044, M3);

    const int q0 = 1948;
    std::vector<int32_t> query_pool(ids.begin() + q0, ids.end());

    const int n_chunks = (N + CHUNK - 1) / CHUNK;
    std::vector<uint8_t> forced((size_t)n_chunks, 0);

    dflash::qwen3::AnchorScanCfg cfg{CHUNK, /*anchor_radius=*/0,
                                     /*max_anchor_hits=*/8, /*ngram=*/4};
    dflash::qwen3::scan_and_force(ids, q0, query_pool, cfg, forced);

    const int chunk_hop3 = 1200 / CHUNK;  // 18
    const int chunk_hop2 = 600  / CHUNK;  // 9
    const int chunk_hop1 = 200  / CHUNK;  // 3

    // Single-pass: only the direct M3 match at pos 1200 is forced.
    REQUIRE(forced[(size_t)chunk_hop3] == 1);
    REQUIRE(forced[(size_t)chunk_hop2] == 0);
    REQUIRE(forced[(size_t)chunk_hop1] == 0);

    std::printf("T2 PASS: chunk(%d) forced, chunk(%d) and chunk(%d) NOT forced (single-pass)\n",
                chunk_hop3, chunk_hop2, chunk_hop1);
}

// T3 — transitive rescues all hops (FAILS until Phase 2 implements the function).
static void t3_transitive_rescues_all() {
    const int N = 2048;
    std::vector<int32_t> ids((size_t)N, FILLER);

    place_marker_4gram(ids, 200, M1);

    place_marker_4gram(ids, 600, M2);
    place_marker_4gram(ids, 604, M1);

    place_marker_4gram(ids, 1200, M3);
    place_marker_4gram(ids, 1204, M2);

    place_marker_4gram(ids, 2044, M3);

    const int q0 = 1948;
    std::vector<int32_t> initial_query_pool(ids.begin() + q0, ids.end());

    const int n_chunks = (N + CHUNK - 1) / CHUNK;
    std::vector<uint8_t> forced((size_t)n_chunks, 0);

    dflash::qwen3::AnchorScanCfg cfg{CHUNK, /*anchor_radius=*/0,
                                     /*max_anchor_hits=*/8, /*ngram=*/4};
    dflash::qwen3::scan_and_force_transitive(ids, q0, initial_query_pool,
                                              cfg, /*max_iters=*/3, forced);

    const int chunk_hop3 = 1200 / CHUNK;
    const int chunk_hop2 = 600  / CHUNK;
    const int chunk_hop1 = 200  / CHUNK;

    REQUIRE(forced[(size_t)chunk_hop3] == 1);
    REQUIRE(forced[(size_t)chunk_hop2] == 1);
    REQUIRE(forced[(size_t)chunk_hop1] == 1);

    std::printf("T3 PASS: all hops forced transitively\n");
}

// T4 — variable-name reuse across templates (FAILS until v2 adds rare-token match).
//
// Token layout:
//   FILLER=1, V1=2001(X42), V2=2002(Y42), V3=2003(Z42)
//   Template-context tokens: A=3001,B=3002,C=3003,D=3004,E=3005,F=3006
//   Query-match tokens: X1=4001,X2=4002,X3=4003
//
// hop3 (chunk 18, pos 1200): [X1,X2,V3,X3,E,V2,F,FILL] — 4-gram [X1,X2,V3,X3] matches query
// hop2 (chunk  9, pos  600): [C,V2,FILL,V1,D,FILL,FILL] — V2 in DIFFERENT context than hop3
// hop1 (chunk  3, pos  200): [A,V1,FILL,B]              — V1 in DIFFERENT context than hop2
// query (pos 2044):          [X1,X2,V3,X3]              — matches hop3 4-gram exactly
//
// Pass 1 (4-gram): forces hop3.
// Pass 1 rare-token: V2 (freq=2) found in hop3 → also at pos 601 (hop2 chunk 9) → forces hop2.
// Pass 2 rare-token: V1 (freq=2) found in hop2 → also at pos 201 (hop1 chunk 3) → forces hop1.
// Today's impl (4-gram only) fails because V2 4-grams in hop3 ≠ V2 4-grams in hop2.
static void t4_rare_token_bridges_different_context() {
    static constexpr int32_t V1 = 2001, V2 = 2002, V3 = 2003;
    static constexpr int32_t A = 3001, B = 3002, C = 3003, D = 3004, E = 3005, F = 3006;
    static constexpr int32_t X1 = 4001, X2 = 4002, X3 = 4003;

    const int N = 2048;
    std::vector<int32_t> ids((size_t)N, FILLER);

    // hop1 (chunk 3, pos 200): [A, V1, FILL, B]
    ids[200] = A; ids[201] = V1; ids[202] = FILLER; ids[203] = B;

    // hop2 (chunk 9, pos 600): [C, V2, FILL, V1, D, FILL, FILL]
    ids[600] = C; ids[601] = V2; ids[602] = FILLER; ids[603] = V1;
    ids[604] = D; ids[605] = FILLER; ids[606] = FILLER;

    // hop3 (chunk 18, pos 1200): [X1, X2, V3, X3, E, V2, F, FILL]
    // V2 here is in 4-gram context [E,V2,F,FILL] — differs from hop2's [C,V2,FILL,V1]
    ids[1200] = X1; ids[1201] = X2; ids[1202] = V3; ids[1203] = X3;
    ids[1204] = E;  ids[1205] = V2; ids[1206] = F;  ids[1207] = FILLER;

    // query suffix (pos 2044): [X1, X2, V3, X3] — exact 4-gram match to hop3
    ids[2044] = X1; ids[2045] = X2; ids[2046] = V3; ids[2047] = X3;

    const int q0 = 1948;
    std::vector<int32_t> initial_query_pool(ids.begin() + q0, ids.end());

    const int n_chunks = (N + CHUNK - 1) / CHUNK;
    std::vector<uint8_t> forced((size_t)n_chunks, 0);

    dflash::qwen3::AnchorScanCfg cfg{CHUNK, /*anchor_radius=*/0,
                                     /*max_anchor_hits=*/8, /*ngram=*/4,
                                     /*rare_token_max_freq=*/8};
    dflash::qwen3::scan_and_force_transitive(ids, q0, initial_query_pool,
                                              cfg, /*max_iters=*/3, forced);

    const int chunk_hop3 = 1200 / CHUNK;  // 18
    const int chunk_hop2 =  600 / CHUNK;  //  9
    const int chunk_hop1 =  200 / CHUNK;  //  3

    REQUIRE(forced[(size_t)chunk_hop3] == 1);
    REQUIRE(forced[(size_t)chunk_hop2] == 1);
    REQUIRE(forced[(size_t)chunk_hop1] == 1);

    std::printf("T4 PASS: all hops forced via rare-token bridge (V2 freq=2, V1 freq=2)\n");
}

// T5: gate closes when pass-1 already finds >= cascade_min_anchor_count chunks.
//
// Layout (N=4096, chunk=64 → 64 chunks):
//   A common 4-gram [CMN,CMN,CMN,CMN] appears 50 times at scattered body positions.
//   One forced chunk (chunk 5, pos 320) also contains a unique rare token RT (freq=1).
//   RT appears once more at a separate body position in chunk 60 (pos 3840).
//   Query suffix contains the common 4-gram → pass-1 forces all 50 matching chunks.
//
// With cascade_min_anchor_count=5: gained=50 >= 5 → gate closes → cascade skipped.
// chunk 60 (pos 3840, which has RT but is only reachable via cascade) stays UNFORCED.
//
// With cascade_min_anchor_count=0: gate open → cascade runs → chunk 60 gets forced.
// This contrast proves the gate is operative.
static void t5_gate_closes_when_pass1_finds_many() {
    static constexpr int32_t CMN = 5001;  // common token (4-gram made of it)
    static constexpr int32_t RT  = 5002;  // rare token (freq=2)

    const int N = 4096;
    const int n_chunks = (N + CHUNK - 1) / CHUNK;  // 64
    std::vector<int32_t> ids((size_t)N, FILLER);

    // Place common 4-gram at 50 scattered body positions (chunks 0..49).
    // Spaced 64 tokens apart to land in different chunks.
    for (int i = 0; i < 50; ++i) {
        int pos = i * 64 + 4;  // pos 4, 68, 132, ... (well within body)
        ids[(size_t)pos]     = CMN;
        ids[(size_t)pos + 1] = CMN;
        ids[(size_t)pos + 2] = CMN;
        ids[(size_t)pos + 3] = CMN;
    }

    // RT appears in chunk 5 (pos 320) and chunk 60 (pos 3840).
    ids[320] = RT;
    ids[3840] = RT;

    // Query suffix: just the common 4-gram so pass-1 fires on all 50 body positions.
    const int q0 = N - 32;
    ids[(size_t)q0]     = CMN;
    ids[(size_t)q0 + 1] = CMN;
    ids[(size_t)q0 + 2] = CMN;
    ids[(size_t)q0 + 3] = CMN;
    std::vector<int32_t> query_pool(ids.begin() + q0, ids.end());

    // --- Test A: gate CLOSED (cascade_min_anchor_count=5) ---
    {
        std::vector<uint8_t> forced_a((size_t)n_chunks, 0);
        dflash::qwen3::AnchorScanCfg cfg{CHUNK, /*anchor_radius=*/0,
                                         /*max_anchor_hits=*/64, /*ngram=*/4,
                                         /*rare_token_max_freq=*/2,
                                         /*cascade_min_anchor_count=*/5,
                                         /*max_forced_count=*/INT_MAX};
        dflash::qwen3::scan_and_force_transitive(ids, q0, query_pool,
                                                  cfg, /*max_iters=*/3, forced_a);

        // Pass-1 forces chunks 0..49 (50 chunks); gate closes → cascade skipped.
        // chunk 60 (pos 3840 has RT but only reachable via cascade) must be UNFORCED.
        const int chunk_rt_extra = 3840 / CHUNK;  // 60
        REQUIRE(forced_a[(size_t)chunk_rt_extra] == 0);
        // chunk 5 (contains RT at pos 320) is forced by pass-1 (common 4-gram at pos 324).
        REQUIRE(forced_a[5] == 1);

        std::printf("T5a PASS: gate closed (gained=50 >= min=5), chunk %d unforced\n",
                    chunk_rt_extra);
    }

    // --- Test B: gate OPEN (cascade_min_anchor_count=0) → cascade forces chunk 60 ---
    {
        std::vector<uint8_t> forced_b((size_t)n_chunks, 0);
        dflash::qwen3::AnchorScanCfg cfg{CHUNK, /*anchor_radius=*/0,
                                         /*max_anchor_hits=*/64, /*ngram=*/4,
                                         /*rare_token_max_freq=*/2,
                                         /*cascade_min_anchor_count=*/0,
                                         /*max_forced_count=*/INT_MAX};
        dflash::qwen3::scan_and_force_transitive(ids, q0, query_pool,
                                                  cfg, /*max_iters=*/3, forced_b);

        // Cascade runs; chunk 5 is forced by pass-1 and contains RT;
        // RT at pos 3840 → chunk 60 forced via rare-token cascade.
        const int chunk_rt_extra = 3840 / CHUNK;
        REQUIRE(forced_b[(size_t)chunk_rt_extra] == 1);

        std::printf("T5b PASS: gate open (min=0), cascade forced chunk %d via RT\n",
                    chunk_rt_extra);
    }
}

// T6: hard cap (max_forced_count) prevents runaway cascade.
//
// Layout (N=2048, chunk=64 → 32 chunks):
//   Query contains 4-gram [TGR,TGR,TGR,TGR] which matches body chunk 0.
//   Chunk 0 contains chain token C0 (freq=2): also appears in chunk 1.
//   Chunk 1 contains chain token C1 (freq=2): also appears in chunk 2.
//   ... 20 such chain links.
//   Pass-1 forces chunk 0 (1 chunk gained < cascade_min_anchor_count=0 → gate open).
//   Cascade rare-token worklist propagates: chunk 0→1→2→...→20 (20 more).
//   max_forced_count=5 → cascade stops when total > 5. Result: forced <= 5.
static void t6_hard_cap_prevents_runaway() {
    static constexpr int32_t TGR = 7000;  // trigger token for 4-gram pass-1 match

    const int N = 2048;
    const int n_chunks = (N + CHUNK - 1) / CHUNK;  // 32
    std::vector<int32_t> ids((size_t)N, FILLER);

    // body chunk 0 (pos 0): place 4-gram [TGR,TGR,TGR,TGR] so pass-1 forces it.
    ids[0] = TGR; ids[1] = TGR; ids[2] = TGR; ids[3] = TGR;

    // Rare-token chain: C_i appears in chunk i (at offset 8) and chunk i+1 (at offset 9).
    // Offsets 8 and 9 within each chunk don't collide between consecutive tokens.
    // Cascade worklist: chunk i forced → C_i found at offset 8 → chunk i+1 forced.
    for (int i = 0; i < 20; ++i) {
        int32_t tok = 7100 + i;
        ids[(size_t)(i * 64 + 8)]           = tok;  // in chunk i, offset 8
        ids[(size_t)((i + 1) * 64 + 9)]     = tok;  // in chunk i+1, offset 9
    }

    // Query suffix: contains [TGR,TGR,TGR,TGR] → pass-1 matches body chunk 0.
    const int q0 = N - 64;
    ids[(size_t)q0]     = TGR;
    ids[(size_t)q0 + 1] = TGR;
    ids[(size_t)q0 + 2] = TGR;
    ids[(size_t)q0 + 3] = TGR;
    std::vector<int32_t> query_pool(ids.begin() + q0, ids.end());

    // Without cap: cascade forces chunks 0..20 (21 chunks total).
    // With cap=5: stops at 5.
    std::vector<uint8_t> forced((size_t)n_chunks, 0);
    dflash::qwen3::AnchorScanCfg cfg{CHUNK, /*anchor_radius=*/0,
                                     /*max_anchor_hits=*/8, /*ngram=*/4,
                                     /*rare_token_max_freq=*/2,
                                     /*cascade_min_anchor_count=*/0,
                                     /*max_forced_count=*/5};
    dflash::qwen3::scan_and_force_transitive(ids, q0, query_pool,
                                              cfg, /*max_iters=*/25, forced);

    int total_forced = 0;
    for (int c = 0; c < n_chunks; ++c) total_forced += (int)forced[(size_t)c];

    REQUIRE(total_forced <= 5);
    REQUIRE(forced[0] == 1);  // chunk 0 always forced by pass-1

    std::printf("T6 PASS: hard cap engaged, forced=%d (cap=5, chain length=20)\n",
                total_forced);
}

int main() {
    t1_single_pass_match();
    t2_single_pass_misses_hops();
    t3_transitive_rescues_all();
    t4_rare_token_bridges_different_context();
    t5_gate_closes_when_pass1_finds_many();
    t6_hard_cap_prevents_runaway();
    std::printf("\nAll anchor_transitive tests passed.\n");
    return 0;
}
