// Exactness check of the strict scorer's session resume (pflash_scoring_resume.h) on the real drafter.
//
//   pflash_resume_exact <drafter.gguf> <S1> <S2> [<S3>...]
//
// with PFLASH_SCORING_HEAD_GGUF / PFLASH_SEGMENT_PROBE_GGUF / PFLASH_REQUEST_PROBE_GGUF set as for serving.
// One random prompt of max(S) tokens; for each S (given order, each a prefix of the next prompt) the block-15 token
// mass and probe segments are scored fresh (sessions off) and then chained (sessions on: each call resumes from the
// previous one's session). Exits 0 only when every chained result equals its fresh one bit for bit. Read the
// "resumed at" line of the [qwen35-scorer] log to see where each chained call resumed.

#include "pflash/pflash_drafter.h"
#include "pflash/qwen35_drafter.h"
#include "luce.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace luce::common;

namespace {

struct Result {
    std::vector<int32_t> kept;
    std::vector<float> mass;
    std::vector<PFlashTokenSpan> segments;
    bool density = false;
};

bool score(DrafterContext & ctx, const std::vector<int32_t> & ids, Result & out) {
    auto * st = static_cast<Qwen35DrafterState *>(ctx.state);
    luce::pflash::PFlashSelectionConfig experiment;
    st->session_ctx_limit = luce::pflash::pflash_session_ctx_limit(st->ctx_train, ctx.engine_max_ctx);
    qwen35_restore_sessions(ctx, *st, ids);
    out.kept = qwen35_strict_score_and_compress(*st, ids, 0.1875f, 8, (int) ids.size(), experiment, {}, &out.mass,
                                                &out.segments, &out.density);
    return !out.kept.empty() && out.mass.size() == ids.size();
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: pflash_resume_exact <drafter.gguf> <S1> [<S2>...]\n");
        return 2;
    }
    std::vector<int> lengths;
    for (int i = 2; i < argc; ++i) lengths.push_back(std::atoi(argv[i]));
    const int max_s = *std::max_element(lengths.begin(), lengths.end());
    DrafterContext ctx;
    if (!load_drafter(argv[1], -1, ctx)) {
        std::fprintf(stderr, "load_drafter failed: %s\n", luce_last_error());
        return 2;
    }
    std::vector<int32_t> all((size_t) max_s);
    uint64_t x = 0x9E3779B97F4A7C15ull;
    for (auto & t : all) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        t = 1000 + (int32_t) ((x >> 33) % 100000);
    }
    std::vector<Result> fresh(lengths.size()), chained(lengths.size());
    setenv("PFLASH_DRAFTER_SESSIONS", "0", 1);
    for (size_t i = 0; i < lengths.size(); ++i) {
        const std::vector<int32_t> ids(all.begin(), all.begin() + lengths[i]);
        if (!score(ctx, ids, fresh[i])) { std::fprintf(stderr, "fresh S=%d failed: %s\n", lengths[i], luce_last_error()); return 2; }
    }
    setenv("PFLASH_DRAFTER_SESSIONS", "2", 1);
    int bad = 0;
    for (size_t i = 0; i < lengths.size(); ++i) {
        const std::vector<int32_t> ids(all.begin(), all.begin() + lengths[i]);
        if (!score(ctx, ids, chained[i])) { std::fprintf(stderr, "chained S=%d failed: %s\n", lengths[i], luce_last_error()); return 2; }
        const bool mass_eq = fresh[i].mass.size() == chained[i].mass.size() &&
            std::memcmp(fresh[i].mass.data(), chained[i].mass.data(), fresh[i].mass.size() * sizeof(float)) == 0;
        bool seg_eq = fresh[i].segments.size() == chained[i].segments.size() && fresh[i].density == chained[i].density;
        for (size_t k = 0; seg_eq && k < fresh[i].segments.size(); ++k) {
            seg_eq = fresh[i].segments[k].begin == chained[i].segments[k].begin &&
                     fresh[i].segments[k].end == chained[i].segments[k].end;
        }
        const bool kept_eq = fresh[i].kept == chained[i].kept;
        size_t first_diff = fresh[i].mass.size();
        for (size_t t = 0; t < fresh[i].mass.size() && t < chained[i].mass.size(); ++t) {
            if (std::memcmp(&fresh[i].mass[t], &chained[i].mass[t], sizeof(float)) != 0) { first_diff = t; break; }
        }
        std::printf("S=%d mass %s segments %s kept %s (first mass diff at %zu)\n", lengths[i],
                    mass_eq ? "IDENTICAL" : "DIFFERENT", seg_eq ? "IDENTICAL" : "DIFFERENT",
                    kept_eq ? "IDENTICAL" : "DIFFERENT", first_diff);
        if (!mass_eq || !seg_eq || !kept_eq) ++bad;
    }
    std::printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
