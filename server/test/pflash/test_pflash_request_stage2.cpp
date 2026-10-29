// Request stage 2 (STAGE2-CONTRACT.md v3): cand_v5 rule, query assembly, the fail-closed loader, the CPU path, and
// parity with the trainer's float64 fixtures.
//
// Fixture tests (env-gated; they need generated files):
//   PFLASH_STAGE2_FIXTURE_DIR=OUT  (python -m pflash_scorer.request_probe.stage2_fixtures write OUT)
//     - replay.jsonl: assembly on the same input numbers, exact;
//     - numeric cases on the CPU path, contract tolerances;
//   + PFLASH_STAGE2_FIXTURE_GPU=1 (hold the GPU lock): the same numeric cases on the device path.
// PFLASH_STAGE2_FIXTURE_REPORT=FILE appends the max |diff| per quantity as JSON lines.

#include "CppUnitTestFramework.hpp"

#include "common/gguf_inspect.h"
#include "pflash/pflash_request_probe.h"
#include "pflash/pflash_request_stage2.h"
#include "server/http_server.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

// Checks inside helper functions (REQUIRE needs the fixture object): a failure throws, which fails the test.
#define S2_NEED(cond) \
    do { \
        if (!(cond)) throw std::runtime_error(std::string("check failed: ") + #cond + " (line " + \
                                              std::to_string(__LINE__) + ")"); \
    } while (0)

using namespace luce::pflash;
using luce::common::PFlashTokenSpan;

namespace {

struct PFlashStage2Fixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;
};

using Role = PFlashRequestRole;
using Messages = std::vector<PFlashRequestMessage>;

PFlashRequestMessage msg(Role role, int b, int e) { return {role, {b, e}}; }

std::vector<float> filled(int n, float v) { return std::vector<float>((size_t) n, v); }

std::vector<int> range(int b, int e) {
    std::vector<int> out;
    for (int t = b; t < e; ++t) out.push_back(t);
    return out;
}

const float kNaN = std::numeric_limits<float>::quiet_NaN();

} // namespace

// ---------------------------------------------------------------- cand_v5 and the assembly (no fixtures)

TEST_CASE(PFlashStage2Fixture, by_score_breaks_exact_ties_by_the_later_position) {
    std::vector<float> s = {1.0f, 3.0f, 3.0f, 2.0f, kNaN, 3.0f};
    REQUIRE(pflash_request_by_score({0, 1, 2, 3, 5, 2}, s) == std::vector<int>({5, 2, 1, 3, 0}));
    REQUIRE(std::string(kRequestCandidateRuleV5) == "cand_v5_latest_user_seed_reserve256_postlatest_then_score_fill");
    REQUIRE(kStage2PromotedCap == 512);
}

TEST_CASE(PFlashStage2Fixture, v5_rule_seeds_the_latest_user_then_fills_every_other_message_by_score) {
    // system [0,4), user [5,15), assistant [16,30), tool [31,40), user (latest) [41,50)
    const Messages m = {msg(Role::System, 0, 4), msg(Role::User, 5, 15), msg(Role::Assistant, 16, 30),
                        msg(Role::Tool, 31, 40), msg(Role::User, 41, 50)};
    auto s1 = filled(52, -5.0f);
    for (int t = 1; t < 4; ++t) s1[(size_t) t] = 9.0f;            // system: never a candidate
    for (int t = 42; t < 45; ++t) s1[(size_t) t] = 1.0f;          // latest: seed
    for (int t = 6; t < 9; ++t) s1[(size_t) t] = 4.0f;            // earlier user
    for (int t = 20; t < 22; ++t) s1[(size_t) t] = 2.0f;          // assistant
    s1[33] = 3.0f;                                                // a 1-token run: dropped
    const auto set = pflash_request_query_set_v5(s1, m, 0.0f, 6);
    REQUIRE(set.fallback.empty());
    REQUIRE(set.seed == std::vector<int>({42, 43, 44}));
    // 3 slots left: the earlier user's 4.0 tokens beat the assistant's 2.0 ones; ties -> later position first.
    REQUIRE(set.tokens == std::vector<int>({6, 7, 8, 42, 43, 44}));
    REQUIRE(set.latest_message == 4);
    REQUIRE(set.capped);
    const auto big = pflash_request_query_set_v5(s1, m, 0.0f);
    REQUIRE(big.tokens == std::vector<int>({6, 7, 8, 20, 21, 42, 43, 44}));
    // The seed itself is capped by score (later position on ties).
    const auto tiny = pflash_request_query_set_v5(s1, m, 0.0f, 2);
    REQUIRE(tiny.seed == std::vector<int>({43, 44}));
    REQUIRE(tiny.seed_capped);
    // No runs anywhere -> today's windows.
    REQUIRE(pflash_request_query_set_v5(filled(52, -1.0f), m, 0.0f).fallback == "no_request_runs");
}

TEST_CASE(PFlashStage2Fixture, v5_rule_reserves_slots_for_post_latest_cues) {
    // user (latest) [0,40), assistant [41,45), tool [46,80): 30 seed run tokens, 20 post-latest run tokens.
    REQUIRE(kRequestPostReserve == 256);
    const Messages m = {msg(Role::User, 0, 40), msg(Role::Assistant, 41, 45), msg(Role::Tool, 46, 80)};
    auto s1 = filled(80, -1.0f);
    for (int t = 5; t < 35; ++t) s1[(size_t) t] = 5.0f + 0.01f * (float) t;   // seed, all above the cues
    for (int t = 50; t < 70; ++t) s1[(size_t) t] = 1.0f + 0.01f * (float) t;  // post-latest cues
    int R = 0;
    std::vector<int> post;
    bool capped = false;
    // cap 16, reserve 6: R = 6, the seed keeps its 10 best (latest positions), the 6 best cues are reserved.
    const auto seed = pflash_request_capped_seed(s1, 0.0f, m, 16, 6, R, post, &capped);
    REQUIRE(R == 6);
    REQUIRE(post == range(50, 70));
    REQUIRE(seed == range(25, 35));
    REQUIRE(capped);
    const auto set = pflash_request_query_set_v5(s1, m, 0.0f, 16, 6);
    REQUIRE(set.reserve == 6);
    std::vector<int> want = range(25, 35);
    for (int t = 64; t < 70; ++t) want.push_back(t);
    REQUIRE(set.tokens == want);
    // Reserve 0 is the cand_v4 order: the seed takes the whole cap.
    const auto v4 = pflash_request_query_set_v5(s1, m, 0.0f, 16, 0);
    REQUIRE(v4.reserve == 0);
    REQUIRE(v4.tokens == range(19, 35));
    // Fewer cues than the reserve: R = |P|, the seed gives back only that many slots.
    const auto few = pflash_request_query_set_v5(s1, m, 0.0f, 32, 256);
    REQUIRE(few.reserve == 20);
    REQUIRE(few.seed == range(23, 35));
    // Stage 2: seed capped at cap - R, promoted at min(promoted_cap, cap - R - |seed|), post-latest gets the rest.
    const Messages m2 = {msg(Role::User, 0, 4), msg(Role::Assistant, 4, 20), msg(Role::User, 20, 40),
                         msg(Role::Tool, 40, 60)};
    auto t1 = filled(60, -1.0f);
    for (int t = 22; t < 38; ++t) t1[(size_t) t] = 3.0f;      // 16 seed run tokens
    for (int t = 42; t < 52; ++t) t1[(size_t) t] = 2.0f;      // 10 post-latest run tokens
    auto s2 = filled(60, kNaN);
    for (int t = 0; t < 20; ++t) s2[(size_t) t] = 1.0f;       // every earlier token promotable
    const auto a = pflash_stage2_assemble(m2, t1, s2, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10, 60, 20, 8, 4);
    REQUIRE(a.reserve == 4);
    REQUIRE(a.seed.size() == 16);                              // cap - R = 16
    REQUIRE(a.promoted.empty());                               // min(8, 20 - 4 - 16) = 0
    REQUIRE(a.post.size() == 4);                               // 20 - 16
    const auto b = pflash_stage2_assemble(m2, t1, s2, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10, 60, 30, 8, 4);
    REQUIRE(b.seed.size() == 16);
    REQUIRE(b.promoted.size() == 8);                           // min(8, 30 - 4 - 16) = 8
    REQUIRE(b.post.size() == 6);                               // 30 - 24 >= R
}

TEST_CASE(PFlashStage2Fixture, assembly_seed_then_promoted_then_post_latest) {
    // user [0,10), assistant [10,30), user (latest) [30,40), tool [40,60)
    const Messages m = {msg(Role::User, 0, 10), msg(Role::Assistant, 10, 30), msg(Role::User, 30, 40),
                        msg(Role::Tool, 40, 60)};
    auto s1 = filled(60, -1.0f);
    for (int t = 32; t < 35; ++t) s1[(size_t) t] = 2.0f;     // seed
    for (int t = 2; t < 5; ++t) s1[(size_t) t] = 5.0f;       // an earlier stage-1 run: never admitted by stage 1
    for (int t = 45; t < 48; ++t) s1[(size_t) t] = 1.0f;     // post-latest
    auto s2 = filled(60, kNaN);
    for (int t = 0; t < 30; ++t) s2[(size_t) t] = -3.0f;
    for (int t = 12; t < 16; ++t) s2[(size_t) t] = 0.5f;     // promoted run
    s2[20] = 0.9f;                                           // 1-token run: dropped
    const auto a = pflash_stage2_assemble(m, s1, s2, /*l_lb=*/1.0f, /*l_vague=*/-1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                          100, 400);
    REQUIRE(a.active);
    REQUIRE(a.lookback);
    REQUIRE(a.seed == std::vector<int>({32, 33, 34}));
    REQUIRE(a.promoted == std::vector<int>({12, 13, 14, 15}));
    REQUIRE(a.post == std::vector<int>({45, 46, 47}));
    REQUIRE(a.query == std::vector<int>({12, 13, 14, 15, 32, 33, 34, 45, 46, 47}));
    REQUIRE(a.promoted_runs == 1);
    REQUIRE_FALSE(a.widened);
    REQUIRE(a.budget == 100);
    // Look-back gate below tau (and NaN): nothing earlier enters.
    const auto off = pflash_stage2_assemble(m, s1, s2, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 100, 400);
    REQUIRE_FALSE(off.lookback);
    REQUIRE(off.promoted.empty());
    REQUIRE(off.query == std::vector<int>({32, 33, 34, 45, 46, 47}));
    const auto nan = pflash_stage2_assemble(m, s1, s2, kNaN, kNaN, 0.0f, 0.0f, 0.0f, 0.0f, 100, 400);
    REQUIRE_FALSE(nan.lookback);
    REQUIRE_FALSE(nan.widened);
    // Caps: the seed first, then promoted (cap 2 here), then post-latest into what is left of cap 7.
    REQUIRE(a.reserve == 3);
    // Caps without the reserve (cand_v4 order): the seed first, then promoted (cap 2 here), then post-latest into
    // what is left of cap 7.
    const auto capped = pflash_stage2_assemble(m, s1, s2, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 100, 400, 7, 2, 0);
    REQUIRE(capped.promoted == std::vector<int>({14, 15}));    // equal scores: the later positions
    REQUIRE(capped.promoted_capped);
    REQUIRE(capped.post == std::vector<int>({46, 47}));
    REQUIRE(capped.query.size() == 7);
    // With the reserve: R = 3 holds the post-latest slots, promoted gets min(2, 7 - 3 - 3) = 1.
    const auto reserved = pflash_stage2_assemble(m, s1, s2, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 100, 400, 7, 2);
    REQUIRE(reserved.reserve == 3);
    REQUIRE(reserved.promoted == std::vector<int>({15}));
    REQUIRE(reserved.post == std::vector<int>({45, 46, 47}));
    REQUIRE(reserved.query.size() == 7);
}

TEST_CASE(PFlashStage2Fixture, assembly_widening_raw_fits_and_inactive_cases) {
    bool raw = false;
    REQUIRE(pflash_stage2_widened_budget(100, 400, raw) == 150);
    REQUIRE_FALSE(raw);
    REQUIRE(pflash_stage2_widened_budget(101, 400, raw) == 151);   // floor(101 / 2) = 50
    REQUIRE(pflash_stage2_widened_budget(300, 400, raw) == 400);
    REQUIRE(raw);
    const Messages m = {msg(Role::User, 0, 10), msg(Role::User, 10, 20)};
    const auto s1 = filled(20, -1.0f);
    // Empty query: no_request_runs, widening still applied.
    const auto a = pflash_stage2_assemble(m, s1, filled(20, kNaN), 1.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.5f, 10, 20);
    REQUIRE(a.query.empty());
    REQUIRE(a.reason == "no_request_runs");
    REQUIRE(a.widened);
    REQUIRE(a.budget == 15);
    // No user message: stage 2 inactive, the cand_v5 rule alone, no widening.
    const Messages agent = {msg(Role::System, 0, 3), msg(Role::Tool, 4, 10), msg(Role::Assistant, 11, 20)};
    auto t1 = filled(20, -1.0f);
    t1[5] = t1[6] = 1.0f;
    const auto b = pflash_stage2_assemble(agent, t1, filled(20, kNaN), 9.0f, 9.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10, 20);
    REQUIRE_FALSE(b.active);
    REQUIRE(b.reason == "no_user_message");
    REQUIRE(b.query == std::vector<int>({5, 6}));
    REQUIRE_FALSE(b.widened);
}

TEST_CASE(PFlashStage2Fixture, layout_builds_the_earlier_set_and_fails_on_overlaps) {
    // system, user [2,5), tool [6,8), tool [8,9), assistant [10,10) (empty), user latest [12,15), tool [16,18)
    const Messages m = {msg(Role::System, 0, 2), msg(Role::User, 2, 5), msg(Role::Tool, 6, 8),
                        msg(Role::Tool, 8, 9), msg(Role::Assistant, 10, 10), msg(Role::User, 12, 15),
                        msg(Role::Tool, 16, 18)};
    PFlashStage2Layout layout;
    std::string error;
    REQUIRE(pflash_stage2_layout(m, 20, layout, error));
    REQUIRE(layout.latest == 5);
    REQUIRE(layout.earlier == std::vector<int>({2, 3, 4, 6, 7, 8}));
    // ranks among non-system messages: user 0, tool 1, tool 2, assistant 3, user 4 -> d = 4, 3, 2
    REQUIRE(layout.back == std::vector<int>({4, 4, 4, 3, 3, 2}));
    REQUIRE(layout.role == std::vector<uint8_t>({0, 0, 0, 2, 2, 2}));
    REQUIRE(layout.offset == std::vector<int>({0, 1, 2, 0, 1, 0}));
    REQUIRE(layout.length == std::vector<int>({3, 3, 3, 2, 2, 1}));
    Messages overlap = m;
    overlap[3].span = {7, 9};
    REQUIRE_FALSE(pflash_stage2_layout(overlap, 20, layout, error));
    Messages outside = m;
    outside[6].span = {16, 21};
    REQUIRE_FALSE(pflash_stage2_layout(outside, 20, layout, error));
}

TEST_CASE(PFlashStage2Fixture, content_spans_come_from_a_verbatim_placeholder_render) {
    using luce::common::http_detail::pflash_char_span_tokens;
    using luce::common::http_detail::pflash_content_char_spans;
    const std::string prompt = "<s>sys\nS<e>\n<u>\n hi <e>\n<a>\nok<e>\n<g>";
    const std::string render = "<s>sys\nS<e>\n<u>\n@1@<e>\n<a>\n@2@<e>\n<g>";
    std::vector<std::pair<size_t, size_t>> spans;
    std::string error;
    REQUIRE(pflash_content_char_spans(render, {"", "@1@", "@2@"}, {"S", " hi ", "ok"}, prompt, spans, error));
    REQUIRE(spans[1].first == prompt.find(" hi "));
    REQUIRE(spans[1].second == spans[1].first + 4);
    REQUIRE(spans[2].first == prompt.find("ok"));
    REQUIRE(spans[0] == std::make_pair((size_t) 0, (size_t) 0));
    // A template that trims content, a missing or repeated placeholder: fail closed.
    REQUIRE_FALSE(pflash_content_char_spans(render, {"", "@1@", "@2@"}, {"S", "hi", "ok"}, prompt, spans, error));
    REQUIRE_FALSE(pflash_content_char_spans(render, {"", "@9@", "@2@"}, {"S", " hi ", "ok"}, prompt, spans, error));
    REQUIRE_FALSE(pflash_content_char_spans(render + "@2@", {"", "@1@", "@2@"}, {"S", " hi ", "ok"}, prompt, spans,
                                            error));
    // A template that trims every message (Qwen 3.8): the contents go back trimmed, spans on the rendered text.
    const std::string trimmed = "<s>sys\nS<e>\n<u>\nhi<e>\n<t>\nout<e>\n<g>";
    const std::string trimmed_render = "<s>sys\nS<e>\n<u>\n@1@<e>\n<t>\n@2@<e>\n<g>";
    REQUIRE(pflash_content_char_spans(trimmed_render, {"", "@1@", "@2@"}, {"S", " hi ", "out\n\n"}, trimmed, spans,
                                      error));
    REQUIRE(spans[1].first == trimmed.find("hi") && spans[1].second == spans[1].first + 2);
    REQUIRE(spans[2].first == trimmed.find("out") && spans[2].second == spans[2].first + 3);
    REQUIRE(luce::common::http_detail::pflash_template_trim(" \t\nx y\r\n") == "x y");
    // A template that renders messages out of order (DeepSeek sorts tool results by call): spans where they are.
    const std::string swapped = "<s>sys\nS<e>\n<t>\nb<e>\n<t>\na<e>\n<g>";
    const std::string swapped_render = "<s>sys\nS<e>\n<t>\n@2@<e>\n<t>\n@1@<e>\n<g>";
    REQUIRE(pflash_content_char_spans(swapped_render, {"", "@1@", "@2@"}, {"S", "a", "b"}, swapped, spans, error));
    REQUIRE(spans[1].first == swapped.find("a") && spans[2].first == swapped.find("b"));
    REQUIRE(spans[2].first < spans[1].first);
    // Tokens overlapping [5, 9): token starts 0, 4, 6, 9, 12 -> tokens 1 ([4,6)), 2 ([6,9)).
    const std::vector<size_t> starts = {0, 4, 6, 9, 12};
    REQUIRE(pflash_char_span_tokens(starts, 15, 5, 9) == PFlashTokenSpan({1, 3}));
    REQUIRE(pflash_char_span_tokens(starts, 15, 9, 10) == PFlashTokenSpan({3, 4}));
    REQUIRE(pflash_char_span_tokens(starts, 15, 13, 15) == PFlashTokenSpan({4, 5}));
    REQUIRE(pflash_char_span_tokens(starts, 15, 7, 7) == PFlashTokenSpan({0, 0}));
}

// ---------------------------------------------------------------- the loader (fail closed)

namespace {

struct Stage2File {
    std::map<std::string, std::string> strings;
    std::map<std::string, float> floats;
    std::map<std::string, uint32_t> u32s;
    std::map<std::string, std::vector<int64_t>> shapes;   // name -> ggml ne
    std::vector<std::string> extra_tensors;
};

Stage2File valid_stage2_file() {
    Stage2File f;
    f.strings = {
        {"general.architecture", kStage2Arch},
        {"requeststage2.schema", kStage2Schema},
        {"requeststage2.feature_version", kStage2FeatureVersion},
        {"requeststage2.assembly_version", kStage2AssemblyVersion},
        {"requeststage2.fallback_version", kStage2FallbackVersion},
        {"requeststage2.base_model", kStage2BaseModel},
        {"requeststage2.feature_tap", kStage2FeatureTap},
        {"requeststage2.runtime_gguf_sha256", "drafter-sha"},
        {"requeststage2.head_gguf_sha256", "head-sha"},
        {"requeststage2.stage1_checkpoint_sha256", "stage1-sha"},
        {"requeststage2.stage1_candidate_rule", kRequestCandidateRuleV5},
        {"requeststage2.tau_source", kStage2TauSourceDev},
    };
    f.floats = {
        {"requeststage2.norm_eps", 1e-5f}, {"requeststage2.rms_eps", 1e-6f}, {"requeststage2.s1_clamp", 16.0f},
        {"requeststage2.tau2", 0.25f}, {"requeststage2.tau_lb", -0.5f}, {"requeststage2.tau_fb", 1.5f},
        {"requeststage2.budget_factor", 1.5f},
    };
    f.u32s = {
        {"requeststage2.query_cap", 1024}, {"requeststage2.promoted_cap", 512}, {"requeststage2.min_run", 2},
        {"requeststage2.post_reserve", 256},
        {"requeststage2.input_dim", 143}, {"requeststage2.hidden", 64}, {"requeststage2.embed", 32},
        {"requeststage2.heads", 8},
    };
    f.shapes = {
        {"requeststage2.cand.weight", {1024, 32}}, {"requeststage2.cand.bias", {32}},
        {"requeststage2.lat_all.weight", {1024, 32}}, {"requeststage2.lat_all.bias", {32}},
        {"requeststage2.lat_s1.weight", {1024, 32}}, {"requeststage2.lat_s1.bias", {32}},
        {"requeststage2.norm.weight", {143}}, {"requeststage2.norm.bias", {143}},
        {"requeststage2.fc1.weight", {143, 64}}, {"requeststage2.fc1.bias", {64}},
        {"requeststage2.fc2.weight", {64}}, {"requeststage2.fc2.bias", {1}},
        {"requeststage2.msg_fc1.weight", {2048, 32}}, {"requeststage2.msg_fc1.bias", {32}},
        {"requeststage2.msg_fc2.weight", {32, 2}}, {"requeststage2.msg_fc2.bias", {2}},
    };
    return f;
}

std::string write_stage2_file(const Stage2File & f, const std::string & path) {
    gguf_context * g = gguf_init_empty();
    for (const auto & [k, v] : f.strings) gguf_set_val_str(g, k.c_str(), v.c_str());
    for (const auto & [k, v] : f.floats) gguf_set_val_f32(g, k.c_str(), v);
    for (const auto & [k, v] : f.u32s) gguf_set_val_u32(g, k.c_str(), v);
    size_t bytes = 0;
    std::vector<std::pair<std::string, std::vector<int64_t>>> all(f.shapes.begin(), f.shapes.end());
    for (const auto & name : f.extra_tensors) all.push_back({name, {4}});
    for (const auto & [name, ne] : all) {
        int64_t n = 1;
        for (const int64_t d : ne) n *= d;
        bytes += (size_t) n * sizeof(float) + ggml_tensor_overhead() + 64;
    }
    ggml_init_params ip{};
    ip.mem_size = bytes + 1024;
    ip.no_alloc = false;
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0f, 0.05f);
    for (const auto & [name, ne] : all) {
        ggml_tensor * t = ne.size() == 1 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[0])
                                         : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne[0], ne[1]);
        ggml_set_name(t, name.c_str());
        float * d = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = normal(rng);
        gguf_add_tensor(g, t);
    }
    gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

PFlashStage2Pins valid_pins() {
    PFlashStage2Pins p;
    p.drafter_sha256 = "drafter-sha";
    p.head_sha256 = "head-sha";
    p.stage1_checkpoint_sha256 = "stage1-sha";
    p.stage1_candidate_rule = kRequestCandidateRuleV5;
    p.drafter_rms_eps = 1e-6f;
    return p;
}

std::string temp_path(const std::string & name) {
    return (std::filesystem::temp_directory_path() / ("pflash-stage2-test-" + std::to_string(::getpid()) + "-" +
                                                      name)).string();
}

} // namespace

TEST_CASE(PFlashStage2Fixture, loader_accepts_the_contract_and_refuses_every_deviation) {
    const std::string path = temp_path("valid.gguf");
    write_stage2_file(valid_stage2_file(), path);
    PFlashStage2Weights w;
    std::string error;
    REQUIRE(pflash_stage2_load(path, valid_pins(), w, error));
    REQUIRE(w.cand_w.size() == (size_t) 32 * 1024);
    REQUIRE(w.fc1_w.size() == (size_t) 64 * 143);
    REQUIRE(w.tau_source == kStage2TauSourceDev);
    REQUIRE(w.tau_fb == 1.5f);
    // Each pin.
    const auto pinned = [&](auto mutate) {
        PFlashStage2Pins p = valid_pins();
        mutate(p);
        PFlashStage2Weights out;
        std::string e;
        return pflash_stage2_load(path, p, out, e);
    };
    REQUIRE_FALSE(pinned([](PFlashStage2Pins & p) { p.drafter_sha256 = "other"; }));
    REQUIRE_FALSE(pinned([](PFlashStage2Pins & p) { p.head_sha256 = "other"; }));
    REQUIRE_FALSE(pinned([](PFlashStage2Pins & p) { p.stage1_checkpoint_sha256 = ""; }));
    REQUIRE_FALSE(pinned([](PFlashStage2Pins & p) { p.stage1_candidate_rule = kRequestCandidateRule; }));
    REQUIRE_FALSE(pinned([](PFlashStage2Pins & p) { p.drafter_rms_eps = std::nextafter(1e-6f, 1.0f); }));
    // Each metadata key: missing, then wrong.
    int refused = 0;
    int cases = 0;
    const auto refuses = [&](const Stage2File & f, const std::string & tag) {
        const std::string p = temp_path(tag + ".gguf");
        write_stage2_file(f, p);
        PFlashStage2Weights out;
        std::string e;
        const bool ok = pflash_stage2_load(p, valid_pins(), out, e);
        std::filesystem::remove(p);
        ++cases;
        if (!ok) ++refused;
        return !ok;
    };
    const Stage2File base = valid_stage2_file();
    for (const auto & [k, v] : base.strings) {
        Stage2File f = base;
        f.strings.erase(k);
        REQUIRE(refuses(f, "nostr"));
        f = base;
        f.strings[k] = v + "x";
        if (k == "requeststage2.tau_source") f.strings[k] = "dev";
        REQUIRE(refuses(f, "badstr"));
    }
    for (const auto & [k, v] : base.floats) {
        Stage2File f = base;
        f.floats.erase(k);
        REQUIRE(refuses(f, "nof32"));
        if (k == "requeststage2.tau2" || k == "requeststage2.tau_lb" || k == "requeststage2.tau_fb") {
            f = base;
            f.floats[k] = kNaN;
            REQUIRE(refuses(f, "nanf32"));
            continue;
        }
        f = base;
        f.floats[k] = k == "requeststage2.norm_eps" ? 0.0f : v * 2.0f;
        REQUIRE(refuses(f, "badf32"));
    }
    for (const auto & [k, v] : base.u32s) {
        Stage2File f = base;
        f.u32s.erase(k);
        REQUIRE(refuses(f, "nou32"));
        f = base;
        f.u32s[k] = v + 1;
        REQUIRE(refuses(f, "badu32"));
    }
    for (const auto & [name, ne] : base.shapes) {
        Stage2File f = base;
        f.shapes.erase(name);
        REQUIRE(refuses(f, "notensor"));
        f = base;
        f.shapes[name][0] += 1;
        REQUIRE(refuses(f, "badshape"));
    }
    {
        Stage2File f = base;
        f.extra_tensors.push_back("requeststage2.extra");
        REQUIRE(refuses(f, "extra"));
    }
    REQUIRE(refused == cases);
    REQUIRE_FALSE(pflash_stage2_load(temp_path("missing.gguf"), valid_pins(), w, error));
    std::filesystem::remove(path);
}

// ---------------------------------------------------------------- trainer fixtures (env-gated)

namespace {

std::string fixture_dir() {
    const char * dir = std::getenv("PFLASH_STAGE2_FIXTURE_DIR");
    if (!dir || !*dir) throw CppUnitTestFramework::TestSkippedException("PFLASH_STAGE2_FIXTURE_DIR is not set");
    return dir;
}

nlohmann::json read_json(const std::string & path) {
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    return nlohmann::json::parse(text.str());
}

std::vector<float> read_f32(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(0, std::ios::end);
    const size_t bytes = (size_t) in.tellg();
    in.seekg(0);
    std::vector<float> out(bytes / sizeof(float));
    in.read((char *) out.data(), (std::streamsize) (out.size() * sizeof(float)));
    return out;
}

Messages case_messages(const nlohmann::json & c) {
    Messages m;
    for (const auto & j : c["messages"]) {
        PFlashRequestMessage x;
        S2_NEED(pflash_request_role_from_name(j["role"].get<std::string>(), x.role));
        x.span = {j["span"][0].get<int>(), j["span"][1].get<int>()};
        m.push_back(x);
    }
    return m;
}

std::vector<float> floats_or_nan(const nlohmann::json & a) {
    std::vector<float> out;
    for (const auto & v : a) out.push_back(v.is_null() ? kNaN : v.get<float>());
    return out;
}

std::vector<int> ints(const nlohmann::json & a) {
    std::vector<int> out;
    for (const auto & v : a) out.push_back(v.get<int>());
    return out;
}

// All tensors of a GGUF by name (host F32 copies) plus its context for metadata.
struct GgufFile {
    gguf_context * g = nullptr;
    ggml_context * data = nullptr;
    ~GgufFile() {
        if (g) gguf_free(g);
        if (data) ggml_free(data);
    }
    std::vector<float> tensor(const char * name) const {
        ggml_tensor * t = ggml_get_tensor(data, name);
        S2_NEED(t != nullptr);
        S2_NEED(t->type == GGML_TYPE_F32);
        return std::vector<float>((const float *) t->data, (const float *) t->data + ggml_nelements(t));
    }
    std::string str(const char * key) const {
        const int64_t id = gguf_find_key(g, key);
        S2_NEED(id >= 0);
        return gguf_get_val_str(g, id);
    }
    float f32(const char * key) const {
        const int64_t id = gguf_find_key(g, key);
        S2_NEED(id >= 0);
        return gguf_get_val_f32(g, id);
    }
};

void open_gguf(const std::string & path, GgufFile & f) {
    gguf_init_params params{ /*no_alloc=*/ false, /*ctx=*/ &f.data };
    f.g = gguf_init_from_file(path.c_str(), params);
    S2_NEED(f.g != nullptr);
}

struct Stage1Trunk {
    std::vector<float> norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b;
    float eps = 1e-5f;
    int width = 256;
};

// LayerNorm(1024) -> fc1 -> GELU(erf) -> fc2, double accumulation.
float stage1_cpu(const Stage1Trunk & t, const float * x) {
    const int H = kStage2Hidden;
    double mu = 0.0;
    for (int k = 0; k < H; ++k) mu += x[k];
    mu /= H;
    double var = 0.0;
    for (int k = 0; k < H; ++k) var += ((double) x[k] - mu) * ((double) x[k] - mu);
    var /= H;
    const double inv = 1.0 / std::sqrt(var + (double) t.eps);
    std::vector<double> v((size_t) H);
    for (int k = 0; k < H; ++k) v[(size_t) k] = ((double) x[k] - mu) * inv * t.norm_w[(size_t) k] + t.norm_b[(size_t) k];
    double out = t.fc2_b[0];
    for (int o = 0; o < t.width; ++o) {
        double acc = t.fc1_b[(size_t) o];
        const float * row = t.fc1_w.data() + (size_t) o * H;
        for (int k = 0; k < H; ++k) acc += (double) row[k] * v[(size_t) k];
        out += (double) t.fc2_w[(size_t) o] * 0.5 * acc * (1.0 + std::erf(acc / std::sqrt(2.0)));
    }
    return (float) out;
}

struct Fixture {
    std::string dir;
    nlohmann::json fx;
    std::vector<float> rows, features, norms;
    PFlashStage2HeadQK head;
    PFlashStage2Weights w;
    Stage1Trunk trunk;
};

void load_fixture(Fixture & f) {
    f.dir = fixture_dir();
    f.fx = read_json(f.dir + "/FIXTURE.json");
    S2_NEED(f.fx["schema"] == "request_stage2_parity_fixture_v1");
    S2_NEED(f.fx["input_dim"].get<int>() == kStage2InputDim);
    S2_NEED(f.fx["contract_version"].get<int>() == 3);
    S2_NEED(f.fx["feature_version"] == kStage2FeatureVersion);
    S2_NEED(f.fx["assembly_version"] == kStage2AssemblyVersion);
    S2_NEED(f.fx["stage1_candidate_rule"] == kRequestCandidateRuleV5);
    S2_NEED(f.fx["post_reserve"].get<int>() == kRequestPostReserve);
    f.rows = read_f32(f.dir + "/rows.f32");
    f.features = read_f32(f.dir + "/features.f32");
    f.norms = read_f32(f.dir + "/norms.f32");
    S2_NEED(f.norms.size() == (size_t) (1024 + 256 + 256));
    GgufFile head, s1;
    open_gguf(f.dir + "/head.gguf", head);
    open_gguf(f.dir + "/stage1.gguf", s1);
    f.head.wq = head.tensor("scoringhead.attn_q.weight");
    f.head.wk = head.tensor("scoringhead.attn_k.weight");
    f.head.w_in.assign(f.norms.begin(), f.norms.begin() + 1024);
    f.head.w_qn.assign(f.norms.begin() + 1024, f.norms.begin() + 1280);
    f.head.w_kn.assign(f.norms.begin() + 1280, f.norms.end());
    f.head.rms_eps = (float) f.fx["rms_eps"].get<double>();
    S2_NEED(s1.str("requestprobe.candidate_rule") == kRequestCandidateRuleV5);
    f.trunk.norm_w = s1.tensor("requestprobe.norm.weight");
    f.trunk.norm_b = s1.tensor("requestprobe.norm.bias");
    f.trunk.fc1_w = s1.tensor("requestprobe.fc1.weight");
    f.trunk.fc1_b = s1.tensor("requestprobe.fc1.bias");
    f.trunk.fc2_w = s1.tensor("requestprobe.fc2.weight");
    f.trunk.fc2_b = s1.tensor("requestprobe.fc2.bias");
    f.trunk.eps = s1.f32("requestprobe.norm_eps");
    f.trunk.width = (int) f.trunk.fc2_w.size();
    // The loader with the fixture's own pins: its fake drafter sha, the head file's sha, the stage-1 checkpoint.
    PFlashStage2Pins pins;
    pins.drafter_sha256 = head.str("scoringhead.runtime_gguf_sha256");
    pins.head_sha256 = luce::common::read_gguf_metadata(f.dir + "/head.gguf", true).sha256;
    pins.stage1_checkpoint_sha256 = s1.str("requestprobe.checkpoint_sha256");
    pins.stage1_candidate_rule = s1.str("requestprobe.candidate_rule");
    pins.drafter_rms_eps = f.head.rms_eps;
    std::string error;
    const bool loaded = pflash_stage2_load(f.dir + "/stage2.gguf", pins, f.w, error);
    if (!loaded) std::fprintf(stderr, "stage-2 fixture load: %s\n", error.c_str());
    S2_NEED(loaded);
}

// Device-path tolerances. HIP (gfx1201) runs the stage-2 F32 GEMMs in F32 and meets the contract's CPU-reference
// tolerances. The CUDA backend runs every F32 GEMM in TF32 (ggml-cuda sets cuBLAS TF32 tensor-op math, and its mmf
// kernels use TF32 mma), as all CUDA serving to date: it is held to the contract's served GPU bound (section 7,
// |d| <= 2e-2 on s1, s2, l) with identical query sets, and the maxima of every quantity are printed (coordinator
// decision, 7 Oct). The tile-consistency check compares one seed tile with 2-query tiles. On HIP it stays strict
// (|d| <= 1e-6 + 1e-4 |a|). On CUDA it uses the same served GPU rule as every other CUDA check, absolute |d| <= 2e-2
// (with the query sets identical): a relative bound is ill-conditioned near zero (TF32 measured max |d| 9.6e-5 while
// the relative error of tiny masses reached 8.7e-3), and the 1e-3 relative guess predated any CUDA measurement
// (coordinator decision, 7 Oct). The absolute and relative maxima are printed either way.
#if defined(LUCE_BACKEND_CUDA)
constexpr bool kDeviceTf32 = true;
#else
constexpr bool kDeviceTf32 = false;
#endif
constexpr double kServedGpuBound = 2e-2;
constexpr double kTileRel = 1e-4;   // HIP only
bool tile_violates(double d, double a) {
    return kDeviceTf32 ? d > kServedGpuBound : d > 1e-6 + kTileRel * std::fabs(a);
}

struct Worst {
    double s1 = 0, A = 0, features = 0, s2 = 0, message = 0, m_all = 0;
    int cases = 0, sets_identical = 0;
    // Device path only: one seed tile vs 2-query tiles (the same masses up to summation order and GEMM shapes).
    double tile_max_abs = 0;
    double tile_max_rel = 0;       // over masses above 1e-6
    long tile_violations = 0;      // tile_violates(|a - a2|, a)
};

void report(const char * path_kind, const Worst & worst, bool tiles_checked) {
    const char * rule = !tiles_checked ? "no tile check on this path"
        : kDeviceTf32 ? "tile rule |d| <= 2e-2 (CUDA served GPU bound)" : "tile rule |d| <= 1e-6 + 1e-4 |a|";
    std::fprintf(stderr,
        "[stage2-parity] %s cases=%d sets_identical=%d max|d| s1=%.3g A=%.3g x=%.3g s2=%.3g l=%.3g m_all=%.3g "
        "tile_max_abs=%.3g tile_max_rel=%.3g tile_violations=%ld (%s)\n",
        path_kind, worst.cases, worst.sets_identical, worst.s1, worst.A, worst.features, worst.s2, worst.message,
        worst.m_all, worst.tile_max_abs, worst.tile_max_rel, worst.tile_violations, rule);
    if (const char * out = std::getenv("PFLASH_STAGE2_FIXTURE_REPORT")) {
        std::ofstream f(out, std::ios::app);
        f << nlohmann::json{{"path", path_kind}, {"cases", worst.cases}, {"sets_identical", worst.sets_identical},
                            {"s1", worst.s1}, {"A", worst.A}, {"features", worst.features}, {"s2", worst.s2},
                            {"message", worst.message}, {"m_all_head", worst.m_all},
                            {"tile_max_abs", worst.tile_max_abs}, {"tile_max_rel", worst.tile_max_rel},
                            {"tile_violations", worst.tile_violations}, {"tile_rule", rule}}.dump()
          << "\n";
    }
}

// Compares one case's numbers and decisions with the fixture; updates ``worst``.
void check_case(const Fixture & f, const nlohmann::json & c, const std::vector<float> & s1,
                const PFlashStage2Layout & layout, const PFlashStage2Latest & lat, const std::vector<float> & x,
                const std::vector<float> & s2, Worst & worst) {
    const auto & e = c["expected"];
    const auto tol = f.fx["tolerances"];
    const int n = c["n_tokens"].get<int>();
    const auto want_s1 = floats_or_nan(e["s1"]);
    S2_NEED((int) want_s1.size() == n);
    for (int t = 0; t < n; ++t) {
        if (std::isnan(want_s1[(size_t) t])) continue;   // template token: no expected value
        worst.s1 = std::max(worst.s1, (double) std::fabs(s1[(size_t) t] - want_s1[(size_t) t]));
    }
    S2_NEED(layout.latest == e["latest"].get<int>());
    S2_NEED(layout.earlier == ints(e["earlier"]));
    const size_t N = layout.earlier.size();
    S2_NEED((size_t) c["features_rows"].get<int>() == N);
    const float * want_x = f.features.data() + (size_t) c["feature_offset"].get<int>() * kStage2InputDim;
    double worst_A = 0.0, worst_x = 0.0;
    for (size_t i = 0; i < N; ++i) {
        for (int k = 0; k < kStage2InputDim; ++k) {
            const double d = std::fabs((double) x[i * kStage2InputDim + (size_t) k] -
                                       (double) want_x[i * kStage2InputDim + (size_t) k]);
            if (k < kStage2ColBack) worst_A = std::max(worst_A, d);
            else worst_x = std::max(worst_x, d);
        }
        for (int h = 0; h < 8; ++h) {
            const double want_a = e["A"][i][(size_t) h].get<double>();
            worst_A = std::max(worst_A, std::fabs((double) x[i * kStage2InputDim + (size_t) h] - want_a));
        }
    }
    worst.A = std::max(worst.A, worst_A);
    worst.features = std::max(worst.features, worst_x);
    const auto want_s2 = floats_or_nan(e["s2"]);
    S2_NEED(want_s2.size() == N);
    for (size_t i = 0; i < N; ++i) worst.s2 = std::max(worst.s2, (double) std::fabs(s2[i] - want_s2[i]));
    if (!e["lookback_logit"].is_null()) {
        worst.message = std::max(worst.message, std::fabs((double) lat.lookback_logit - e["lookback_logit"].get<double>()));
        worst.message = std::max(worst.message, std::fabs((double) lat.vague_logit - e["vague_logit"].get<double>()));
    } else {
        S2_NEED(layout.latest < 0);
    }
    for (size_t k = 0; k < 8; ++k) {
        worst.m_all = std::max(worst.m_all, std::fabs(lat.m_all[k] - e["m_all_head"][k].get<double>()));
    }
    // Decisions: the runtime's assembly on its own numbers must give the fixture's sets exactly.
    const Messages m = case_messages(c);
    const auto a = pflash_stage2_assemble(m, s1, pflash_stage2_full_s2(layout, s2, n), lat.lookback_logit,
                                          lat.vague_logit, c["tau1"].get<float>(), c["tau2"].get<float>(),
                                          c["tau_lb"].get<float>(), c["tau_fb"].get<float>(), c["budget"].get<int>(),
                                          c["n_prompt"].get<int>(), c["cap"].get<int>(), c["promoted_cap"].get<int>(),
                                          c["reserve"].get<int>());
    const bool same = a.seed == ints(e["seed"]) && a.promoted == ints(e["promoted"]) && a.post == ints(e["post"]) &&
        a.reserve == e["reserve_R"].get<int>() &&
        a.query == ints(e["query"]) && a.widened == e["widened"].get<bool>() &&
        a.budget == e["budget_out"].get<int>() && a.raw_fits == e["raw_fits"].get<bool>() &&
        a.lookback == e["lookback"].get<bool>() && a.reason == e["reason"].get<std::string>();
    if (!same) std::fprintf(stderr, "[stage2-parity] case %s: sets differ\n", c["id"].get<std::string>().c_str());
    worst.sets_identical += same ? 1 : 0;
    ++worst.cases;
    }

// Contract section 7, served GPU parity: |d| <= 2e-2 on s1, s2, l, and (stricter than its >= 99 %) every query set
// identical.
bool within_served_gpu_bound(const Fixture & f, const Worst & worst) {
    return worst.cases == (int) f.fx["cases"].size() && worst.sets_identical == worst.cases &&
        worst.s1 <= kServedGpuBound && worst.s2 <= kServedGpuBound && worst.message <= kServedGpuBound;
}

bool within_tolerances(const Fixture & f, const Worst & worst) {
    const auto tol = f.fx["tolerances"];
    return worst.cases == (int) f.fx["cases"].size() && worst.sets_identical == worst.cases &&
        worst.s1 <= tol["s1"].get<double>() && worst.A <= tol["A"].get<double>() &&
        worst.features <= tol["features"].get<double>() && worst.s2 <= tol["s2"].get<double>() &&
        worst.message <= tol["message"].get<double>() && worst.m_all <= tol["features"].get<double>();
}

} // namespace

TEST_CASE(PFlashStage2Fixture, fixture_replay_cases_match_exactly) {
    const std::string dir = fixture_dir();
    std::ifstream in(dir + "/replay.jsonl");
    REQUIRE(in.good());
    std::string line;
    int cases = 0, identical = 0, stage2_cases = 0, with_reserve = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto c = nlohmann::json::parse(line);
        REQUIRE(c["rule"] == kRequestCandidateRuleV5);
        const Messages m = case_messages(c);
        const auto s1 = floats_or_nan(c["s1"]);
        const auto & e = c["expected"];
        with_reserve += e.value("reserve", 0) > 0 ? 1 : 0;
        bool same = false;
        if (c["stage2"].get<bool>()) {
            REQUIRE(c["assembly_version"] == kStage2AssemblyVersion);
            ++stage2_cases;
            const float l_lb = c["lookback_logit"].is_null() ? kNaN : c["lookback_logit"].get<float>();
            const auto a = pflash_stage2_assemble(
                m, s1, floats_or_nan(c["s2"]), l_lb, c["vague_logit"].get<float>(), c["tau1"].get<float>(),
                c["tau2"].get<float>(), c["tau_lb"].get<float>(), c["tau_fb"].get<float>(), c["budget"].get<int>(),
                c["n_prompt"].get<int>(), c["cap"].get<int>(), c["promoted_cap"].get<int>(), c["reserve"].get<int>());
            same = a.seed == ints(e["seed"]) && a.promoted == ints(e["promoted"]) && a.post == ints(e["post"]) &&
                a.reserve == e["reserve"].get<int>() &&
                a.query == ints(e["query"]) && a.widened == e["widened"].get<bool>() &&
                a.budget == e["budget"].get<int>() && a.raw_fits == e["raw_fits"].get<bool>() &&
                a.lookback == e["lookback"].get<bool>();
        } else {
            const auto set = pflash_request_query_set_v5(s1, m, c["tau1"].get<float>(), c["cap"].get<int>(),
                                                         c["reserve"].get<int>());
            same = set.tokens == ints(e["query"]) && set.seed == ints(e["seed"]) &&
                set.reserve == e["reserve"].get<int>();
        }
        if (!same) std::fprintf(stderr, "[stage2-replay] %s differs\n", c["id"].get<std::string>().c_str());
        identical += same ? 1 : 0;
        ++cases;
    }
    std::fprintf(stderr, "[stage2-replay] %d/%d identical (%d stage-2, %d cand_v5 only, %d with R > 0)\n", identical,
                 cases, stage2_cases, cases - stage2_cases, with_reserve);
    if (const char * out = std::getenv("PFLASH_STAGE2_FIXTURE_REPORT")) {
        std::ofstream f(out, std::ios::app);
        f << nlohmann::json{{"path", "replay"}, {"cases", cases}, {"identical", identical}}.dump() << "\n";
    }
    REQUIRE(cases > 0);
    REQUIRE(identical == cases);
}

TEST_CASE(PFlashStage2Fixture, fixture_numeric_cases_meet_the_contract_tolerances_on_the_cpu) {
    Fixture f;
    load_fixture(f);
    Worst worst;
    for (const auto & c : f.fx["cases"]) {
        const int n = c["n_tokens"].get<int>();
        const float * rows = f.rows.data() + (size_t) c["row_offset"].get<int>() * kStage2Hidden;
        std::vector<float> s1((size_t) n);
        for (int t = 0; t < n; ++t) s1[(size_t) t] = stage1_cpu(f.trunk, rows + (size_t) t * kStage2Hidden);
        PFlashStage2Pass1 p;
        std::string error;
        REQUIRE(pflash_stage2_cpu_reference(rows, n, case_messages(c), s1, c["tau1"].get<float>(), f.head, f.w, p,
                                            error));
        REQUIRE(p.seed == ints(c["expected"]["seed"]));
        REQUIRE(p.reserve == c["expected"]["reserve_R"].get<int>());
        check_case(f, c, s1, p.layout, p.latest, p.x, p.s2, worst);
    }
    report("cpu", worst, /*tiles_checked=*/false);
    REQUIRE(within_tolerances(f, worst));
}

// The device path: rows on the GPU, x̂ / K / c(t) / the m_all column sums / the stage-1 trunk built with the same
// ops the drafter's key graph uses, then the look-back attention and the token MLP of pflash_request_stage2.
TEST_CASE(PFlashStage2Fixture, fixture_numeric_cases_meet_the_contract_tolerances_on_the_gpu) {
    const char * gpu = std::getenv("PFLASH_STAGE2_FIXTURE_GPU");
    if (!gpu || std::string(gpu) != "1") {
        throw CppUnitTestFramework::TestSkippedException("PFLASH_STAGE2_FIXTURE_GPU is not 1");
    }
    Fixture f;
    load_fixture(f);
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    REQUIRE(backend != nullptr);
    struct BackendFree { ggml_backend_t b; ~BackendFree() { ggml_backend_free(b); } } backend_free{backend};
    PFlashStage2Device dev;
    std::string error;
    REQUIRE(pflash_stage2_device_upload(backend, f.w, dev, error));
    // Fixed weights on the device.
    ggml_init_params wp{};
    wp.mem_size = 16 * ggml_tensor_overhead();
    wp.no_alloc = true;
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * wq = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 1024, 2048);
    ggml_tensor * wk = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 1024, 512);
    ggml_tensor * w_in = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 1024);
    ggml_tensor * w_qn = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 256);
    ggml_tensor * w_kn = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 256);
    ggml_tensor * r_nw = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 1024);
    ggml_tensor * r_nb = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 1024);
    ggml_tensor * r_f1w = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 1024, f.trunk.width);
    ggml_tensor * r_f1b = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, f.trunk.width);
    ggml_tensor * r_f2w = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, f.trunk.width);
    ggml_tensor * r_f2b = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 1);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    REQUIRE(wbuf != nullptr);
    const auto put = [](ggml_tensor * t, const std::vector<float> & v) {
        S2_NEED((int64_t) v.size() == ggml_nelements(t));
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
    };
    put(wq, f.head.wq); put(wk, f.head.wk); put(w_in, f.head.w_in); put(w_qn, f.head.w_qn); put(w_kn, f.head.w_kn);
    put(r_nw, f.trunk.norm_w); put(r_nb, f.trunk.norm_b); put(r_f1w, f.trunk.fc1_w); put(r_f1b, f.trunk.fc1_b);
    put(r_f2w, f.trunk.fc2_w); put(r_f2b, f.trunk.fc2_b);
    const float eps = f.head.rms_eps;
    Worst worst;
    for (const auto & c : f.fx["cases"]) {
        const int n = c["n_tokens"].get<int>();
        const float * rows = f.rows.data() + (size_t) c["row_offset"].get<int>() * kStage2Hidden;
        const Messages m = case_messages(c);
        PFlashStage2Layout layout;
        REQUIRE(pflash_stage2_layout(m, n, layout, error));
        const PFlashTokenSpan latest = layout.latest_span;
        // Per-token pass (the key graph): K into a keys tensor, c(t), s1, the latest message's x̂ sum.
        ggml_init_params ip{};
        ip.mem_size = 8 * ggml_tensor_overhead();
        ip.no_alloc = true;
        ggml_context * pctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, 1024, n);
        ggml_tensor * keys = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, 256, 2, n);
        ggml_tensor * cand = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, kStage2Embed, n);
        ggml_tensor * s1d = ggml_new_tensor_1d(pctx, GGML_TYPE_F32, n);
        ggml_tensor * msum = ggml_new_tensor_1d(pctx, GGML_TYPE_F32, 1024);
        ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors(pctx, backend);
        REQUIRE(pbuf != nullptr);
        ggml_backend_tensor_set(x, rows, 0, (size_t) n * 1024 * sizeof(float));
        ggml_backend_tensor_memset(msum, 0, 0, ggml_nbytes(msum));
        ggml_init_params gp{};
        gp.mem_size = 64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(256, false) + 4096;
        gp.no_alloc = true;
        ggml_context * gctx = ggml_init(gp);
        ggml_cgraph * gf = ggml_new_graph_custom(gctx, 256, false);
        ggml_tensor * x_norm = ggml_mul(gctx, ggml_rms_norm(gctx, x, eps), w_in);   // as the key graph
        ggml_tensor * K = ggml_reshape_3d(gctx, ggml_mul_mat(gctx, wk, x_norm), 256, 2, n);
        K = ggml_mul(gctx, ggml_rms_norm(gctx, K, eps), w_kn);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, K, keys));
        ggml_build_forward_expand(gf, ggml_cpy(gctx, pflash_stage2_build_cand(gctx, dev, x_norm), cand));
        ggml_tensor * r = ggml_norm(gctx, x, f.trunk.eps);                          // the request trunk
        r = ggml_add(gctx, ggml_mul(gctx, r, r_nw), r_nb);
        r = ggml_gelu_erf(gctx, ggml_add(gctx, ggml_mul_mat(gctx, r_f1w, r), r_f1b));
        ggml_tensor * req = ggml_add(gctx, ggml_mul_mat(gctx, r_f2w, r), r_f2b);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, ggml_reshape_1d(gctx, req, n), s1d));
        if (latest.end > latest.begin) {
            ggml_build_forward_expand(gf, ggml_cpy(gctx,
                pflash_stage2_build_column_sum(gctx, x_norm, latest.begin, latest.end), msum));
        }
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        REQUIRE(ggml_gallocr_alloc_graph(alloc, gf));
        REQUIRE(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
        ggml_gallocr_free(alloc);
        ggml_free(gctx);
        std::vector<float> s1((size_t) n), cand_all((size_t) n * kStage2Embed), msum_h(1024);
        ggml_backend_tensor_get(s1d, s1.data(), 0, s1.size() * sizeof(float));
        ggml_backend_tensor_get(cand, cand_all.data(), 0, cand_all.size() * sizeof(float));
        ggml_backend_tensor_get(msum, msum_h.data(), 0, msum_h.size() * sizeof(float));
        // Latest-request embeddings and the message head (host), as the drafter's pass 1.
        int reserve = 0;
        std::vector<int> post_latest;
        const auto seed = pflash_request_capped_seed(s1, c["tau1"].get<float>(), m, c["cap"].get<int>(),
                                                     c["reserve"].get<int>(), reserve, post_latest);
        S2_NEED(reserve == c["expected"]["reserve_R"].get<int>());
        PFlashStage2Latest lat;
        lat.m_all.assign(1024, 0.0);
        if (latest.end > latest.begin) {
            for (int k = 0; k < 1024; ++k) lat.m_all[(size_t) k] = (double) msum_h[(size_t) k] / (latest.end - latest.begin);
        }
        std::vector<float> seed_rows(seed.size() * 1024);
        for (size_t q = 0; q < seed.size(); ++q) {
            std::copy_n(rows + (size_t) seed[q] * 1024, 1024, seed_rows.data() + q * 1024);
        }
        if (seed.empty()) {
            lat.m_s1 = lat.m_all;
        } else {
            lat.m_s1.assign(1024, 0.0);
            std::vector<double> xh(1024);
            for (size_t q = 0; q < seed.size(); ++q) {
                pflash_stage2_xhat(seed_rows.data() + q * 1024, f.head.w_in.data(), 1024, eps, xh.data());
                for (int k = 0; k < 1024; ++k) lat.m_s1[(size_t) k] += xh[(size_t) k];
            }
            for (auto & v : lat.m_s1) v /= (double) seed.size();
        }
        pflash_stage2_message_head(f.w, lat);
        const size_t N = layout.earlier.size();
        std::vector<float> a, xf, s2;
        PFlashStage2AttentionInputs in;
        in.backend = backend;
        in.keys = keys;
        in.wq = wq;
        in.w_in = w_in;
        in.w_qn = w_qn;
        in.rms_eps = eps;
        int tiles = 0;
        REQUIRE(pflash_stage2_attention_gpu(in, layout.earlier, seed_rows.data(), (int) seed.size(), a, tiles,
                                            error));
        if (N > 0 && !seed.empty()) {
            // A smaller tile budget forces several seed tiles: the same masses up to F32 summation order.
            in.tile_bytes = (size_t) N * 8 * sizeof(float) * 2;
            std::vector<float> a2;
            int tiles2 = 0;
            REQUIRE(pflash_stage2_attention_gpu(in, layout.earlier, seed_rows.data(), (int) seed.size(), a2, tiles2,
                                                error));
            REQUIRE(tiles2 == (int) (seed.size() + 1) / 2);
            // Reported, then required after the report (the device's GEMM may round per shape, e.g. TF32 on CUDA).
            for (size_t k = 0; k < a.size(); ++k) {
                const double d = std::fabs((double) a[k] - (double) a2[k]);
                worst.tile_max_abs = std::max(worst.tile_max_abs, d);
                if (std::fabs((double) a[k]) > 1e-6) {
                    worst.tile_max_rel = std::max(worst.tile_max_rel, d / std::fabs((double) a[k]));
                }
                if (tile_violates(d, (double) a[k])) ++worst.tile_violations;
            }
        }
        std::vector<float> cand_e(N * kStage2Embed);
        for (size_t i = 0; i < N; ++i) {
            std::copy_n(cand_all.data() + (size_t) layout.earlier[i] * kStage2Embed, kStage2Embed,
                        cand_e.data() + i * kStage2Embed);
        }
        pflash_stage2_features(layout, a, s1, cand_e, lat, xf);
        REQUIRE(pflash_stage2_mlp_gpu(backend, dev, f.w.norm_eps, xf, (int) N, s2, error));
        check_case(f, c, s1, layout, lat, xf, s2, worst);
        ggml_backend_buffer_free(pbuf);
        ggml_free(pctx);
    }
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    pflash_stage2_device_free(dev);
    report(kDeviceTf32 ? "gpu-cuda-tf32" : "gpu", worst, /*tiles_checked=*/true);
    REQUIRE(worst.tile_violations == 0);
    if (kDeviceTf32) {
        REQUIRE(within_served_gpu_bound(f, worst));
    } else {
        REQUIRE(within_tolerances(f, worst));
    }
}

// Pass-1 cost on the device at served lengths (PFLASH_STAGE2_BENCH=1; hold the GPU lock): the look-back attention
// (key gather, seed Q, seed tiles), the host feature assembly and the token MLP, for |E| = 32K / 64K / 128K and a
// 64- or 1024-token seed (1024 = the cap, the contract's 0.55 TFLOP case). Random keys and rows: the cost does not
// depend on the values. PFLASH_STAGE2_FIXTURE_REPORT receives one JSON line per point.
TEST_CASE(PFlashStage2Fixture, bench_pass1_on_the_gpu) {
    const char * bench = std::getenv("PFLASH_STAGE2_BENCH");
    if (!bench || std::string(bench) != "1") {
        throw CppUnitTestFramework::TestSkippedException("PFLASH_STAGE2_BENCH is not 1");
    }
    const std::string path = temp_path("bench.gguf");
    write_stage2_file(valid_stage2_file(), path);
    PFlashStage2Weights w;
    std::string error;
    REQUIRE(pflash_stage2_load(path, valid_pins(), w, error));
    std::filesystem::remove(path);
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    REQUIRE(backend != nullptr);
    PFlashStage2Device dev;
    REQUIRE(pflash_stage2_device_upload(backend, w, dev, error));
    const int max_n = 131072;
    const int latest_len = 2048;
    ggml_init_params ip{};
    ip.mem_size = 8 * ggml_tensor_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * keys = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 2, max_n + latest_len);
    ggml_tensor * wq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 2048);
    ggml_tensor * w_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    ggml_tensor * w_qn = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buf != nullptr);
    std::mt19937 rng(11);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    {
        std::vector<float> host((size_t) ggml_nelements(keys));
        for (auto & v : host) v = normal(rng);
        ggml_backend_tensor_set(keys, host.data(), 0, host.size() * sizeof(float));
        host.assign((size_t) ggml_nelements(wq), 0.0f);
        for (auto & v : host) v = 0.03f * normal(rng);
        ggml_backend_tensor_set(wq, host.data(), 0, host.size() * sizeof(float));
        host.assign(1024, 1.0f);
        ggml_backend_tensor_set(w_in, host.data(), 0, 1024 * sizeof(float));
        ggml_backend_tensor_set(w_qn, host.data(), 0, 256 * sizeof(float));
    }
    std::vector<float> seed_rows((size_t) 1024 * 1024);
    for (auto & v : seed_rows) v = normal(rng);
    for (const int N : {32768, 65536, 131072}) {
        // Earlier set: [0, N) in 32 messages of N/32 tokens, every 9th token template (not in E).
        Messages m;
        const int per = N / 32;
        int pos = 0;
        for (int k = 0; k < 32; ++k) {
            m.push_back(msg(k % 2 ? Role::Assistant : Role::Tool, pos, pos + per - per / 9));
            pos += per;
        }
        m.push_back(msg(Role::User, pos, pos + latest_len));
        PFlashStage2Layout layout;
        REQUIRE(pflash_stage2_layout(m, pos + latest_len, layout, error));
        const size_t E = layout.earlier.size();
        std::vector<float> s1((size_t) (pos + latest_len), 0.5f), cand(E * kStage2Embed, 0.1f);
        PFlashStage2Latest lat;
        lat.m_all.assign(1024, 0.01);
        lat.m_s1 = lat.m_all;
        pflash_stage2_message_head(w, lat);
        std::vector<size_t> tile_mb = {64};
        if (const char * sweep = std::getenv("PFLASH_STAGE2_BENCH_TILE_SWEEP")) {
            if (std::string(sweep) == "1") tile_mb = {32, 64, 128, 256};
        }
        for (const size_t mb : tile_mb)
        for (const int nseed : {64, 1024}) {
            PFlashStage2AttentionInputs in;
            in.tile_bytes = mb << 20;
            in.backend = backend;
            in.keys = keys;
            in.wq = wq;
            in.w_in = w_in;
            in.w_qn = w_qn;
            in.rms_eps = 1e-6f;
            std::vector<float> a, x, s2;
            int tiles = 0;
            double best_att = 1e30, best_feat = 1e30, best_mlp = 1e30;
            for (int rep = 0; rep < 4; ++rep) {   // rep 0 warms up
                const auto t0 = std::chrono::steady_clock::now();
                REQUIRE(pflash_stage2_attention_gpu(in, layout.earlier, seed_rows.data(), nseed, a, tiles, error));
                const auto t1 = std::chrono::steady_clock::now();
                pflash_stage2_features(layout, a, s1, cand, lat, x);
                const auto t2 = std::chrono::steady_clock::now();
                REQUIRE(pflash_stage2_mlp_gpu(backend, dev, w.norm_eps, x, (int) E, s2, error));
                const auto t3 = std::chrono::steady_clock::now();
                if (rep == 0) continue;
                best_att = std::min(best_att, std::chrono::duration<double, std::milli>(t1 - t0).count());
                best_feat = std::min(best_feat, std::chrono::duration<double, std::milli>(t2 - t1).count());
                best_mlp = std::min(best_mlp, std::chrono::duration<double, std::milli>(t3 - t2).count());
            }
            double mass = 0.0;
            for (size_t i = 0; i < E; ++i) mass += a[i * 8];
            REQUIRE(std::fabs(mass - 1.0) < 1e-3);          // head 0 sums to one over E
            std::fprintf(stderr,
                "[stage2-bench] E=%zu seed=%d tile_mb=%zu tiles=%d attention=%.1fms features=%.1fms mlp=%.1fms "
                "total=%.1fms\n",
                E, nseed, mb, tiles, best_att, best_feat, best_mlp, best_att + best_feat + best_mlp);
            if (const char * out = std::getenv("PFLASH_STAGE2_FIXTURE_REPORT")) {
                std::ofstream f(out, std::ios::app);
                f << nlohmann::json{{"path", "bench"}, {"earlier", E}, {"seed", nseed}, {"tile_mb", mb}, {"tiles", tiles},
                                    {"attention_ms", best_att}, {"features_ms", best_feat}, {"mlp_ms", best_mlp}}
                         .dump() << "\n";
            }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    pflash_stage2_device_free(dev);
    ggml_backend_free(backend);
}
