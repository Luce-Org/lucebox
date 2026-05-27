// Unit tests for the tail-capture chunk-boundary guard in qwen3_graph.cpp.
// Reproduces Bug #42: ggml_view_3d overrun when S % chunk_size ∈ {1..7}
// and n_lookahead == 8.
//
// Pure integer arithmetic — no ggml, no GPU, no server deps.
//
// Root cause (codex's diagnosis, confirmed by momus's data audit):
//   tail_lo = S - n_lookahead
//   When chunk 0 contains S = chunk_size + r tokens (r ∈ {1..7}), a second
//   chunk was dispatched but we still evaluate the first chunk's guard with
//   cs=0, cl=chunk_size. tail_lo = chunk_size + r - n_lookahead = 4088 + r.
//
//   OLD guard:  tail_lo >= cs && tail_lo < cs + cl
//     r=1..7: (4088+r) >= 0 && (4088+r) < 4096  → TRUE  ← BUG: tail overruns
//
//   NEW guard:  tail_lo >= cs && tail_lo + n_lookahead <= cs + cl
//     r=1..7: (4088+r) + 8 <= 4096 → 4096+r <= 4096 → FALSE ← correct: skip
//
// TDD RED/GREEN:
//   RED  (before patch): TAIL_GUARD_USE_NEW_FORMULA undefined → old guard inline → test FAILS.
//   GREEN (after patch): TAIL_GUARD_USE_NEW_FORMULA defined via compiler flag → test PASSES.
//   The patch to qwen3_graph.cpp changes the same 2 lines as this toggle.

#include <cstdio>
#include <cstdlib>

#define REQUIRE(cond) \
    do { if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s line %d: %s\n", __FILE__, __LINE__, #cond); \
        std::exit(1); \
    } } while (0)

// The guard being tested — toggled by compile-time flag to reproduce RED/GREEN.
#ifdef TAIL_GUARD_USE_NEW_FORMULA
static bool tail_fits(int tail_lo, int cs, int cl, int n_lookahead) {
    return tail_lo >= cs && tail_lo + n_lookahead <= cs + cl;  // NEW (fix)
}
#else
static bool tail_fits(int tail_lo, int cs, int cl, int n_lookahead) {
    (void)n_lookahead;
    return tail_lo >= cs && tail_lo < cs + cl;  // OLD (Bug #42)
}
#endif

// T1: First chunk (cs=0, cl=4096), S = chunk_size + r for r ∈ {1..7}.
// Tail straddles the chunk boundary: tail_lo ∈ [4089..4095], needs 8 tokens
// → runs 1..7 tokens past the end → view must be SKIPPED.
// CORRECT answer: false. Old guard returns true → BUG → RED test FAILS.
static void t1_straddling_tail_must_be_skipped() {
    const int chunk_size = 4096, n_lookahead = 8;
    const int cs = 0, cl = chunk_size;  // first chunk

    for (int r = 1; r <= 7; r++) {
        const int S       = chunk_size + r;
        const int tail_lo = S - n_lookahead;  // = 4088 + r ∈ [4089..4095]

        const bool result = tail_fits(tail_lo, cs, cl, n_lookahead);
        std::printf("T1 r=%d S=%d tail_lo=%d tail_hi=%d chunk=[%d,%d): fits=%d (expect 0)\n",
                    r, S, tail_lo, tail_lo + n_lookahead, cs, cs + cl, (int)result);
        REQUIRE(!result && "tail overruns chunk boundary — guard must return false");
    }
}

// T2: r=0 (S == chunk_size exactly). tail_lo=4088, tail_hi=4096=chunk end. Fits exactly.
// Both old and new guards agree: true.
static void t2_tail_fits_exactly_at_chunk_end() {
    const int chunk_size = 4096, n_lookahead = 8;
    const int cs = 0, cl = chunk_size;
    const int S       = chunk_size;
    const int tail_lo = S - n_lookahead;  // 4088

    const bool result = tail_fits(tail_lo, cs, cl, n_lookahead);
    std::printf("T2 r=0 S=%d tail_lo=%d: fits=%d (expect 1)\n", S, tail_lo, (int)result);
    REQUIRE(result && "tail fits exactly at chunk end — must return true");
}

// T3: r=8 (S = chunk_size + 8). tail_lo=4096 — at cs+cl boundary, outside chunk.
// Both guards agree: false.
static void t3_tail_starts_outside_chunk() {
    const int chunk_size = 4096, n_lookahead = 8;
    const int cs = 0, cl = chunk_size;
    const int S       = chunk_size + 8;
    const int tail_lo = S - n_lookahead;  // 4096

    const bool result = tail_fits(tail_lo, cs, cl, n_lookahead);
    std::printf("T3 r=8 S=%d tail_lo=%d: fits=%d (expect 0)\n", S, tail_lo, (int)result);
    REQUIRE(!result && "tail starts at next chunk — must return false");
}

// T4: Second chunk (cs=4096, cl=4096), S=8192, tail fully inside.
// tail_lo=8184, tail_hi=8192 == cs+cl. Both guards agree: true.
static void t4_second_chunk_tail_fits_exactly() {
    const int chunk_size = 4096, n_lookahead = 8;
    const int cs = chunk_size, cl = chunk_size;  // second chunk
    const int S       = 2 * chunk_size;
    const int tail_lo = S - n_lookahead;  // 8184

    const bool result = tail_fits(tail_lo, cs, cl, n_lookahead);
    std::printf("T4 second chunk S=%d tail_lo=%d cs=%d: fits=%d (expect 1)\n",
                S, tail_lo, cs, (int)result);
    REQUIRE(result && "tail fits exactly in second chunk — must return true");
}

// T5: Second chunk, r=3. tail straddles end of second chunk.
// S = 2*4096 + 3 = 8195. tail_lo = 8187, tail_hi = 8195. cs+cl = 8192.
// New guard: 8195 <= 8192 → false. Old guard: 8187 < 8192 → true (BUG).
static void t5_second_chunk_straddling_tail_skipped() {
    const int chunk_size = 4096, n_lookahead = 8;
    const int cs = chunk_size, cl = chunk_size;  // second chunk [4096,8192)
    const int r = 3;
    const int S       = 2 * chunk_size + r;
    const int tail_lo = S - n_lookahead;  // 8187

    const bool result = tail_fits(tail_lo, cs, cl, n_lookahead);
    std::printf("T5 second chunk r=%d S=%d tail_lo=%d: fits=%d (expect 0)\n",
                r, S, tail_lo, (int)result);
    REQUIRE(!result && "tail straddles end of second chunk — must return false");
}

int main() {
    t1_straddling_tail_must_be_skipped();
    t2_tail_fits_exactly_at_chunk_end();
    t3_tail_starts_outside_chunk();
    t4_second_chunk_tail_fits_exactly();
    t5_second_chunk_straddling_tail_skipped();
    std::printf("All tail_capture guard tests passed.\n");
    return 0;
}
