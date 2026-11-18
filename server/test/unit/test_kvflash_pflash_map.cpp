// Unit test for the mode-5 segment-to-chunk mapping (no GPU).

#include "CppUnitTestFramework.hpp"

#include "kvflash_pflash_map.h"

#include <map>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashPflashMapFixture {};
}

TEST_CASE(KvflashPflashMapFixture, a_chunk_takes_the_max_density_of_the_segments_overlapping_it) {
    // chunks of 64: [0,64) [64,128) [128,192) [192,256)
    const auto s = kvflash_pflash_chunk_scores({{10, 70}, {70, 100}, {200, 210}}, {2.0, 5.0, 3.0}, 256, 64);
    CHECK((s == std::vector<float>{2.0f, 5.0f, 0.0f, 3.0f}));
}

TEST_CASE(KvflashPflashMapFixture, a_segment_spanning_chunks_scores_each_of_them_and_gaps_stay_zero) {
    // [100, 300) touches chunks 1..4 (299 / 64 = 4); chunks 0 and 5 are gaps
    const auto s = kvflash_pflash_chunk_scores({{100, 300}}, {4.0}, 384, 64);
    CHECK((s == std::vector<float>{0.0f, 4.0f, 4.0f, 4.0f, 4.0f, 0.0f}));
}

TEST_CASE(KvflashPflashMapFixture, the_end_is_exclusive_and_the_partial_last_chunk_counts) {
    const auto s = kvflash_pflash_chunk_scores({{0, 64}, {128, 129}}, {1.0, 2.0}, 130, 64);
    CHECK((s == std::vector<float>{1.0f, 0.0f, 2.0f}));
}

TEST_CASE(KvflashPflashMapFixture, spans_past_the_prompt_are_clipped_and_empty_or_mismatched_input_scores_zero) {
    CHECK((kvflash_pflash_chunk_scores({{100, 900}}, {1.0}, 192, 64) == std::vector<float>{0.0f, 1.0f, 1.0f}));
    CHECK((kvflash_pflash_chunk_scores({{70, 70}, {-5, 3}}, {9.0, 1.0}, 128, 64) == std::vector<float>{1.0f, 0.0f}));
    CHECK((kvflash_pflash_chunk_scores({{0, 64}}, {}, 128, 64) == std::vector<float>{0.0f, 0.0f}));
    CHECK(kvflash_pflash_chunk_scores({{0, 64}}, {1.0}, 0, 64).empty());
    CHECK(kvflash_pflash_chunk_scores({{0, 64}}, {1.0}, 64, 0).empty());
}

// ---- the drafter gets the target's ids unchanged ----------------
TEST_CASE(KvflashPflashMapFixture, drafter_ids_are_the_prompt_ids_unchanged) {
    std::string err;
    const std::vector<int32_t> ids = {5, 99999, 100000, 248068, 248076};   // <think> = 248068
    CHECK(kvflash_pflash2_ids_fit(ids, 248077, err));
    CHECK(err.empty());
}
TEST_CASE(KvflashPflashMapFixture, an_id_outside_the_drafter_vocab_is_refused_by_name) {
    std::string err;
    CHECK(!kvflash_pflash2_ids_fit({1, 248077}, 248077, err));
    CHECK(err.find("248077") != std::string::npos);
}

// ---- the message map of a rendered prompt ----------------------
namespace {
constexpr int32_t kS = 1, kU = 2, kA = 3, kSys = 4, kNL = 5, kEnd = 6, kTR = 7;
const KvFlashChatIds kChat{kS, kU, kNL, kEnd, kTR, kA, kSys};
}
TEST_CASE(KvflashPflashMapFixture, the_message_map_has_role_and_content_span_per_message) {
    const std::vector<int32_t> ids = {kS, kSys, kNL, 10, 11, kEnd, kNL,           // 0-6   system [3,5)
                                      kS, kU, kNL, 20, 21, kEnd, kNL,             // 7-13  user [10,12)
                                      kS, kA, kNL, 30, kEnd, kNL,                 // 14-19 assistant [17,18)
                                      kS, kU, kNL, kTR, kNL, 40, 41, kEnd, kNL,   // 20-28 tool result [25,27)
                                      kS, kA, kNL};                               // 29-31 generation prompt: no message
    const auto m = kvflash_message_map(ids, kChat);
    CHECK(m.all_error.empty());
    CHECK((m.all_roles == std::vector<std::string>{"system", "user", "assistant", "tool"}));
    REQUIRE(m.all_spans.size() == 4);
    CHECK((m.all_spans[0].begin == 3 && m.all_spans[0].end == 5));
    CHECK((m.all_spans[1].begin == 10 && m.all_spans[1].end == 12));
    CHECK((m.all_spans[2].begin == 17 && m.all_spans[2].end == 18));
    CHECK((m.all_spans[3].begin == 25 && m.all_spans[3].end == 27));
    REQUIRE(m.human.size() == 1);                       // the tool result is not human
    CHECK((m.human[0].begin == 10 && m.human_index == std::vector<int>{1}));
    CHECK(kvflash_message_query_end(m) == 27);
}
TEST_CASE(KvflashPflashMapFixture, the_message_map_gives_the_last_two_human_turns_latest_first_and_empty_content_is_0_0) {
    const std::vector<int32_t> ids = {kS, kU, kNL, 1, kEnd, kNL, kS, kU, kNL, kEnd, kNL, kS, kU, kNL, 2, 3, kEnd, kNL, kS, kA, kNL};
    const auto m = kvflash_message_map(ids, kChat);
    REQUIRE(m.all_spans.size() == 3);
    CHECK((m.all_spans[1].begin == 0 && m.all_spans[1].end == 0));
    CHECK((m.all_spans[2].begin == 14 && m.all_spans[2].end == 16));
    REQUIRE(m.human.size() == 2);
    CHECK((m.human[0].begin == 14 && m.human[1].end == 0));   // an empty user message still counts, as in the daemon
    CHECK((m.human_index == std::vector<int>{2, 1}));
}
TEST_CASE(KvflashPflashMapFixture, an_unknown_role_token_is_a_named_error_and_no_spans) {
    const auto m = kvflash_message_map({kS, 99, kNL, 1, kEnd, kNL}, kChat);
    CHECK(!m.all_error.empty());
    CHECK(m.all_spans.empty());
    CHECK(!kvflash_message_map({}, kChat).all_error.empty());   // nothing to map
}

// ---- the mode-5 startup gate ----------------------------
namespace {
std::map<std::string, std::string> release_env() {
    return {{"PFLASH_SELECT_MODE", "budget_only"}, {"PFLASH_SELECT_SCORE", "density"}, {"PFLASH_SELECT_SEGMENTS", "probe"},
            {"PFLASH_SELECT_QUERY_HEAD", "1"}, {"PFLASH_SELECT_QUERY_PARSER", "latest_user"}, {"PFLASH_SELECT_QUERY_PROBE", "1"},
            {"PFLASH_SCORING_HEAD_GGUF", "h.gguf"}, {"PFLASH_SEGMENT_PROBE_GGUF", "s.gguf"}, {"PFLASH_REQUEST_PROBE_GGUF", "r.gguf"},
            {"PFLASH_QUERY_CANDIDATES", "1"}, {"PFLASH_QUERY_PIN", "512"}, {"PFLASH_DRAFTER_SESSIONS", "2"}};
}
std::string gate(const std::map<std::string, std::string> & e, int slots) {
    return kvflash_pflash2_config_error([&](const char * k) { auto i = e.find(k); return i == e.end() ? std::string() : i->second; }, slots);
}
}
TEST_CASE(KvflashPflashMapFixture, the_release_profile_passes_the_mode_5_gate) {
    CHECK(gate(release_env(), 1).empty());
    CHECK(gate(release_env(), 2).empty());
}
TEST_CASE(KvflashPflashMapFixture, an_unset_select_mode_is_refused_naming_it) {
    auto e = release_env(); e.erase("PFLASH_SELECT_MODE");
    CHECK(gate(e, 1).find("PFLASH_SELECT_MODE") != std::string::npos);
    e = release_env(); e["PFLASH_SELECT_MODE"] = "top_p";
    CHECK(gate(e, 1).find("PFLASH_SELECT_MODE") != std::string::npos);
}
TEST_CASE(KvflashPflashMapFixture, a_missing_request_probe_or_query_step_is_refused_naming_it) {
    for (const char * k : {"PFLASH_REQUEST_PROBE_GGUF", "PFLASH_QUERY_CANDIDATES", "PFLASH_QUERY_PIN", "PFLASH_SCORING_HEAD_GGUF"}) {
        auto e = release_env(); e.erase(k);
        CHECK(gate(e, 1).find(k) != std::string::npos);
    }
}
TEST_CASE(KvflashPflashMapFixture, fewer_drafter_sessions_than_slots_is_refused) {
    auto e = release_env();
    CHECK(gate(e, 3).find("PFLASH_DRAFTER_SESSIONS") != std::string::npos);
    e.erase("PFLASH_DRAFTER_SESSIONS");                     // default 2
    CHECK(gate(e, 2).empty());
    CHECK(!gate(e, 3).empty());
}
TEST_CASE(KvflashPflashMapFixture, a_malformed_or_overflowing_drafter_session_count_is_refused) {
    for (const char * v : {"2junk", "abc", "2 ", " 2", "2.5", "0x2", "99999999999999999999", "-"}) {
        auto e = release_env(); e["PFLASH_DRAFTER_SESSIONS"] = v;
        const std::string err = gate(e, 1);
        CHECK(err.find("PFLASH_DRAFTER_SESSIONS") != std::string::npos);
        CHECK(err.find(v) != std::string::npos);
    }
    auto e = release_env(); e["PFLASH_DRAFTER_SESSIONS"] = "3";
    CHECK(gate(e, 3).empty());
}
TEST_CASE(KvflashPflashMapFixture, an_empty_final_message_has_no_query_end) {
    const std::vector<int32_t> ids = {kS, kSys, kNL, 10, 11, kEnd, kNL,   // system [3,5)
                                      kS, kU, kNL, kEnd, kNL,             // user, empty content
                                      kS, kA, kNL};
    const auto m = kvflash_message_map(ids, kChat);
    REQUIRE(m.all_spans.size() == 2);
    CHECK(kvflash_message_query_end(m) == 0);   // not the system message's end (5)
}

// A tool output may hold the literal chat markers (the server tokenizes them as the special tokens).
TEST_CASE(KvflashPflashMapFixture, literal_chat_markers_inside_a_message_do_not_split_it) {
    //                      0  1   2   3   4  5   6   7   8    9   10   11   12 13  14  15   16   17   18 19  20
    const std::vector<int32_t> ids = {kS, kU, kNL, 20, kS, kU, kNL, 21, kEnd, 22, kEnd, kNL, kS, kA, kNL, 30, kEnd, kNL, kS, kA, kNL};
    const auto m = kvflash_message_map(ids, kChat);
    CHECK(m.all_error.empty());
    REQUIRE(m.all_spans.size() == 2);
    CHECK((m.all_spans[0].begin == 3 && m.all_spans[0].end == 10));   // not cut at the literal <|im_end|> at 8
    CHECK((m.all_spans[1].begin == 15 && m.all_spans[1].end == 16));
    // a literal <|im_end|> followed by a newline and text still belongs to the message
    const std::vector<int32_t> ids2 = {kS, kU, kNL, 20, kEnd, kNL, 21, kEnd, kNL, kS, kA, kNL};
    const auto m2 = kvflash_message_map(ids2, kChat);
    REQUIRE(m2.all_spans.size() == 1);
    CHECK((m2.all_spans[0].begin == 3 && m2.all_spans[0].end == 7));
}
