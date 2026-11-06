#include "CppUnitTestFramework.hpp"

#include "server/pflash_unified.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

using namespace pflash_unified;

namespace {

struct PFlashUnifiedFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;
};

// "line 1" .. "line n", '\n'-joined, with a trailing newline.
std::string numbered_text(int n, const char * prefix = "line ") {
    std::string out;
    for (int i = 1; i <= n; ++i) out += prefix + std::to_string(i) + "\n";
    return out;
}

// Every line start and separator start: both tokenisations agree everywhere.
std::vector<size_t> all_boundaries(const Lines & lines) {
    std::vector<size_t> out;
    for (size_t i = 0; i < lines.count(); ++i) {
        out.push_back(lines.starts[i]);
        out.push_back(lines.ends[i]);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::string line_at(const std::string & text, const Lines & lines, size_t i) {
    return text.substr(lines.starts[i], lines.ends[i] - lines.starts[i]);
}

}  // namespace

TEST_CASE(PFlashUnifiedFixture, test_unified_region_tools_parse) {
    ToolList t;
    std::string err;
    REQUIRE(parse_tool_list("", t, err));
    REQUIRE(!t.all);
    REQUIRE(t.names.empty());
    REQUIRE(!t.listed("read_file"));

    REQUIRE(parse_tool_list(" search_files , read_file,read_file ", t, err));
    REQUIRE(t.names == std::vector<std::string>({"read_file", "search_files"}));
    REQUIRE(t.identity == "read_file,search_files");
    REQUIRE(t.listed("read_file"));
    REQUIRE(!t.listed("terminal"));

    REQUIRE(parse_tool_list("*", t, err));
    REQUIRE(t.all);
    REQUIRE(t.identity == "*");
    REQUIRE(t.listed("anything"));

    REQUIRE(!parse_tool_list("*,read_file", t, err));
    REQUIRE(!parse_tool_list("a,,b", t, err));
    REQUIRE(!parse_tool_list("read file", t, err));
    REQUIRE(!parse_tool_list("a[0]", t, err));
}

TEST_CASE(PFlashUnifiedFixture, test_unified_labels_name_and_arguments) {
    const nlohmann::json obj = {{"path", "docs/a.md"}, {"offset", 10}};
    REQUIRE(call_brief("read_file", obj) == "read_file {\"offset\":10,\"path\":\"docs/a.md\"}");
    // JSON-encoded arguments (the OpenAI wire form) read the same.
    REQUIRE(call_brief("read_file", nlohmann::json(obj.dump())) == call_brief("read_file", obj));
    // Newlines and non-ASCII never reach the label raw; long arguments are cut.
    const std::string brief = call_brief("run\ncmd", {{"cmd", "echo a\nb \xC3\xA9"}});
    REQUIRE(brief.find('\n') == std::string::npos);
    REQUIRE(brief.find("\\n") != std::string::npos);
    REQUIRE(brief.find("\\u00e9") != std::string::npos);
    const std::string long_brief = call_brief("t", {{"q", std::string(500, 'x')}}, 40);
    REQUIRE(long_brief.size() == 2 + 40 + 3);
    REQUIRE(long_brief.substr(long_brief.size() - 3) == "\xE2\x80\xA6");
    REQUIRE(call_brief("list", nlohmann::json::object()) == "list");
    REQUIRE(call_brief("list", nullptr) == "list");

    REQUIRE(recall_label_text(RegionKind::ToolOutput, "read_file {\"path\":\"a\"}") ==
            "[From the earlier result of read_file {\"path\":\"a\"}]");
    REQUIRE(recall_label_text(RegionKind::UserContent, "") == "[From an earlier user message]");
}

TEST_CASE(PFlashUnifiedFixture, unified_split_lines_plain_and_escaped) {
    const std::string plain = "a\r\nbb\n\nc\n";
    const Lines p = split_lines(plain, 0, plain.size(), false);
    REQUIRE(p.count() == 4);
    REQUIRE(line_at(plain, p, 0) == "a\r");
    REQUIRE(line_at(plain, p, 1) == "bb");
    REQUIRE(line_at(plain, p, 2) == "");
    REQUIRE(line_at(plain, p, 3) == "c");
    REQUIRE(split_lines(plain, 0, plain.size() - 1, false).count() == 4);
    REQUIRE(split_lines(plain, 0, 0, false).count() == 0);

    // Inside a JSON string: "\n" separates, an escaped backslash followed by n does not, \u escapes are skipped.
    const std::string esc = "1|a\\n2|b\\\\n still 2\\n3|\\u005cn";
    const Lines e = split_lines(esc, 0, esc.size(), true);
    REQUIRE(e.count() == 3);
    REQUIRE(line_at(esc, e, 1) == "2|b\\\\n still 2");
    REQUIRE(line_at(esc, e, 2) == "3|\\u005cn");
    REQUIRE(std::string(e.sep()) == "\\n");
}

TEST_CASE(PFlashUnifiedFixture, test_unified_line_model_json_string) {
    // A JSON-wrapped file read (the shape Hermes and many MCP tools return): the lines live in a string value.
    std::string body;
    for (int i = 1; i <= 40; ++i) body += std::to_string(i) + "|text " + std::to_string(i) + "\n";
    const nlohmann::json result = {{"content", body}, {"total_lines", 40}, {"_hint", "a\nb"}};
    const std::string text = result.dump();
    const Lines lines = detect_lines(text, 0, text.size());
    REQUIRE(lines.escaped);
    REQUIRE(lines.count() == 40);
    REQUIRE(line_at(text, lines, 0) == "1|text 1");
    REQUIRE(line_at(text, lines, 39) == "40|text 40");

    // The same lines as plain text are found as plain lines; a pretty-printed JSON whose plain lines outnumber
    // its string's lines stays plain.
    const Lines plain = detect_lines(body, 0, body.size());
    REQUIRE(!plain.escaped);
    REQUIRE(plain.count() == 40);
    nlohmann::json many = nlohmann::json::array();
    for (int i = 0; i < 30; ++i) many.push_back({{"k", i}, {"v", "x\ny"}});
    const std::string pretty = many.dump(2);
    REQUIRE(!detect_lines(pretty, 0, pretty.size()).escaped);
    REQUIRE(json_string_values("not json {", 0, 10).empty());
    const std::string keys = "{\"a\\nb\\nc\\nd\\ne\\nf\": 1}";
    REQUIRE(json_string_values(keys, 0, keys.size()).empty());

    // Paging inside the string keeps the result valid JSON with every other field unchanged.
    std::vector<size_t> common = all_boundaries(lines);
    std::vector<size_t> proposed;
    for (int l = 5; l < 40; l += 5) proposed.push_back(lines.starts[(size_t) l]);
    Layout layout;
    std::string why;
    REQUIRE(layout_region(lines, common, proposed, layout, why));
    std::vector<char> admitted((size_t) layout.units(), 0);
    admitted[1] = 1;
    const std::string served = render_region_text(text, 0, text.size(), lines, layout, admitted,
                                                  RegionKind::ToolOutput);
    const nlohmann::json parsed = nlohmann::json::parse(served, nullptr, false);
    REQUIRE(!parsed.is_discarded());
    REQUIRE(parsed["total_lines"] == 40);
    REQUIRE(parsed["_hint"] == "a\nb");
    const std::string content = parsed["content"].get<std::string>();
    REQUIRE(content.find("1|text 1\n2|text 2\n[Server note: this tool result was shortened:") == 0);
    REQUIRE(content.find("[\xE2\x80\xA6 3 lines not shown \xE2\x80\xA6]\n6|text 6\n") != std::string::npos);
    REQUIRE(content.find("\n39|text 39\n40|text 40\n") != std::string::npos);
}

TEST_CASE(PFlashUnifiedFixture, test_unified_markers_count_lines) {
    // CRLF content without a trailing newline: served bytes keep CRLF; markers count the dropped lines.
    std::string text;
    for (int i = 1; i <= 12; ++i) text += "row " + std::to_string(i) + (i < 12 ? "\r\n" : "");
    const Lines lines = detect_lines(text, 0, text.size());
    REQUIRE(lines.count() == 12);
    std::vector<size_t> proposed;
    for (int l = 3; l < 12; l += 2) proposed.push_back(lines.starts[(size_t) l] + 1);  // mid-line: opens that line
    Layout layout;
    std::string why;
    REQUIRE(layout_region(lines, all_boundaries(lines), proposed, layout, why));
    REQUIRE(layout.cut_line == std::vector<int>({2, 3, 5, 7, 9, 10}));
    std::vector<char> admitted = {1, 0, 0, 1, 0};
    const std::string served = render_region_text(text, 0, text.size(), lines, layout, admitted,
                                                  RegionKind::UserContent);
    const std::string expect =
        "row 1\r\nrow 2\r\n"
        "[Server note: this message was shortened: 7 of 12 lines are shown; each omitted range is marked inline.]\n"
        "row 3\r\n"
        "[\xE2\x80\xA6 4 lines not shown \xE2\x80\xA6]\n"
        "row 8\r\nrow 9\r\n"
        "[\xE2\x80\xA6 1 line not shown \xE2\x80\xA6]\n"
        "row 11\r\nrow 12";
    REQUIRE(served == expect);
    REQUIRE(marker_text(2) == "[\xE2\x80\xA6 2 lines not shown \xE2\x80\xA6]");
}

TEST_CASE(PFlashUnifiedFixture, unified_layout_cuts_only_at_clean_line_boundaries) {
    const std::string text = numbered_text(20);
    const Lines lines = detect_lines(text, 0, text.size());
    // Separator starts only (a tokenizer that glues the newline to the next word): cuts sit before the separator.
    std::vector<size_t> seps;
    for (size_t i = 0; i < lines.count(); ++i) seps.push_back(lines.ends[i]);
    Layout layout;
    std::string why;
    REQUIRE(layout_region(lines, seps, {lines.starts[10]}, layout, why));
    REQUIRE(layout.cuts.front() == lines.ends[1]);
    REQUIRE(layout.cuts[1] == lines.ends[9]);
    REQUIRE(layout.cuts.back() == lines.ends[17]);
    std::vector<char> admitted = {0, 1};
    const std::string served = render_region_text(text, 0, text.size(), lines, layout, admitted,
                                                  RegionKind::UserContent);
    REQUIRE(served.find("line 2\n[Server note:") != std::string::npos);
    REQUIRE(served.find("inline.]\n[\xE2\x80\xA6 8 lines not shown \xE2\x80\xA6]\nline 11\n") != std::string::npos);

    // A line with no clean boundary opens no unit; the edges move inward to the first clean line.
    std::vector<size_t> sparse;
    for (int l : {3, 8, 16}) sparse.push_back(lines.starts[(size_t) l]);
    REQUIRE(layout_region(lines, sparse, {lines.starts[5], lines.starts[8]}, layout, why));
    REQUIRE(layout.cut_line == std::vector<int>({3, 8, 16}));

    // Too few lines, or no clean cut at all: never a region.
    const std::string four = numbered_text(4);
    const Lines four_lines = detect_lines(four, 0, four.size());
    REQUIRE(!layout_region(four_lines, all_boundaries(four_lines), {}, layout, why));
    REQUIRE(!layout_region(lines, {}, {}, layout, why));
    const std::string one_line(5000, 'x');
    REQUIRE(detect_lines(one_line, 0, one_line.size()).count() == 1);
}

TEST_CASE(PFlashUnifiedFixture, unified_admit_paging_counts_markers_and_respects_cap) {
    // Five units of 10 tokens, marker cost 3 per gap, preamble 4.
    const std::vector<int> cost = {10, 10, 10, 10, 10};
    auto marker = [](int, int) { return 3; };
    const std::vector<int> order = {2, 0, 4, 1, 3};
    // Start: one gap (3) + preamble (4) = 7. Unit 2 splits the gap: +10 +3 = 20. Unit 0: +10 +0 = 30. Unit 4:
    // +10 + 0 = 40 > 35, skipped; unit 1 joins [0..2]: +10 -3 = 37 > 35, skipped; unit 3: +10 -3 +3 = 40, skipped.
    Admission a = admit_paging(cost, {}, order, 4, 35, marker);
    REQUIRE(a.ok);
    REQUIRE(a.admitted == std::vector<char>({1, 0, 1, 0, 0}));
    REQUIRE(a.cost == 30);
    REQUIRE(a.taken == 2);
    // Locked units are kept and cost nothing against the cap.
    a = admit_paging(cost, {0, 0, 0, 0, 1}, order, 4, 7, marker);
    REQUIRE(a.ok);
    REQUIRE(a.admitted == std::vector<char>({0, 0, 0, 0, 1}));
    REQUIRE(a.cost == 7);
    // A cap below the preamble plus one marker cannot page.
    REQUIRE(!admit_paging(cost, {}, order, 4, 6, marker).ok);
}

TEST_CASE(PFlashUnifiedFixture, unified_admit_recall_skips_held_and_charges_labels_per_run) {
    const std::vector<int> cost = {10, 10, 10, 10, 10};
    auto label = [](int, int) { return 5; };
    // Held 1; order 2, 3, 0, 4: unit 2 = 15; unit 3 joins the run: +10 = 25; unit 0 new run: +15 = 40 > 30.
    Admission a = admit_recall(cost, {0, 1, 0, 0, 0}, {2, 3, 0, 4}, 30, label);
    REQUIRE(a.admitted == std::vector<char>({0, 0, 1, 1, 0}));
    REQUIRE(a.cost == 25);
    // Unit 4 fits after unit 0 is skipped.
    a = admit_recall(cost, {0, 1, 0, 0, 0}, {2, 3, 0, 4}, 35, label);
    REQUIRE(a.admitted == std::vector<char>({0, 0, 1, 1, 1}));
    REQUIRE(a.cost == 35);
    // Bridging two runs merges their labels.
    a = admit_recall(cost, {}, {0, 2, 1}, 1000, label);
    REQUIRE(a.cost == 35);
    // Worst case: every unheld unit comes back once, nothing more.
    a = admit_recall(cost, {1, 0, 1, 0, 0}, {0, 1, 2, 3, 4}, 1000, label);
    REQUIRE(a.admitted == std::vector<char>({0, 1, 0, 1, 1}));
}

TEST_CASE(PFlashUnifiedFixture, unified_recall_gate_appends_nothing_when_the_present_context_suffices) {
    RecallRegion r;
    r.unit_cost = {10, 10, 10, 10};
    r.held = {1, 0, 0, 1};
    r.score = {9.0, 2.0, 1.0, 9.0};
    r.mass = {0.40, 0.03, 0.01, 0.30};   // held units carry most of the mass; D = 0.04
    r.cap = 1000;
    r.label_cost = 5;
    GatedRecall g = admit_recall_gated({r}, 1.0, 0.05);
    REQUIRE(g.gated);
    REQUIRE(g.regions[0].taken == 0);
    REQUIRE(g.regions[0].admitted == std::vector<char>({0, 0, 0, 0}));
    REQUIRE(std::fabs(g.deficit_before - 0.04) < 1e-12);
    REQUIRE(std::fabs(g.deficit_after - 0.04) < 1e-12);
    // Held units never count towards D, whatever their mass.
    r.mass = {5.0, 0.03, 0.01, 5.0};
    REQUIRE(std::fabs(admit_recall_gated({r}, 1.0, 0.05).deficit_before - 0.04) < 1e-12);
}

TEST_CASE(PFlashUnifiedFixture, unified_recall_gate_stops_once_the_deficit_is_covered) {
    RecallRegion r;
    r.unit_cost = {10, 10, 10, 10, 10};
    r.held = {0, 0, 0, 0, 0};
    r.score = {1.0, 8.0, 2.0, 0.5, 0.4};
    r.mass = {0.02, 0.30, 0.05, 0.01, 0.01};   // D = 0.39 of S = 1
    r.cap = 1000;
    r.label_cost = 5;
    // theta .10: unit 1 leaves .09 < .10: one unit, nothing more.
    GatedRecall g = admit_recall_gated({r}, 1.0, 0.10);
    REQUIRE(!g.gated);
    REQUIRE(g.regions[0].admitted == std::vector<char>({0, 1, 0, 0, 0}));
    REQUIRE(g.regions[0].cost == 15);
    REQUIRE(std::fabs(g.deficit_after - 0.09) < 1e-12);
    // theta .05: unit 1 (.09 left), unit 2 (.04 left): stop.
    g = admit_recall_gated({r}, 1.0, 0.05);
    REQUIRE(g.regions[0].admitted == std::vector<char>({0, 1, 1, 0, 0}));
    REQUIRE(g.regions[0].cost == 25);
    // S scales the share: the same units against S = 4 are under .10 at the start.
    REQUIRE(admit_recall_gated({r}, 4.0, 0.10).gated);
}

TEST_CASE(PFlashUnifiedFixture, unified_recall_gate_at_zero_fills_like_admit_recall_and_never_past_the_cap) {
    RecallRegion r;
    r.unit_cost = {10, 10, 10, 10, 10};
    r.held = {0, 1, 0, 0, 0};
    r.score = {3.0, 9.0, 5.0, 4.0, 1.0};
    r.mass = {0.1, 0.5, 0.1, 0.1, 0.1};
    r.cap = 30;
    r.label_cost = 5;
    const auto label = [](int, int) { return 5; };
    const Admission fill = admit_recall(r.unit_cost, r.held, rank_units(r.score, r.held), r.cap, label);
    const GatedRecall g = admit_recall_gated({r}, 1.0, 0.0);
    REQUIRE(g.regions[0].admitted == fill.admitted);
    REQUIRE(g.regions[0].cost == fill.cost);
    REQUIRE(g.regions[0].cost <= r.cap);
    // Diffuse mass under a strict theta: still bounded by the cap.
    REQUIRE(admit_recall_gated({r}, 1.0, 0.01).regions[0].cost <= r.cap);
}

TEST_CASE(PFlashUnifiedFixture, unified_recall_gate_orders_units_across_regions_under_their_own_caps) {
    RecallRegion a;
    a.unit_cost = {10, 10};
    a.held = {0, 0};
    a.score = {1.0, 6.0};
    a.mass = {0.05, 0.10};
    a.cap = 15;
    a.label_cost = 5;
    RecallRegion b;
    b.unit_cost = {10, 10};
    b.held = {0, 0};
    b.score = {7.0, 2.0};
    b.mass = {0.20, 0.05};
    b.cap = 1000;
    b.label_cost = 5;
    // D = .40. Order: b0 (7), a1 (6), b1 (2), a0 (1). theta .12: b0 -> .20, a1 -> .10 < .12: stop.
    GatedRecall g = admit_recall_gated({a, b}, 1.0, 0.12);
    REQUIRE(g.regions[0].admitted == std::vector<char>({0, 1}));
    REQUIRE(g.regions[1].admitted == std::vector<char>({1, 0}));
    // theta .01: a0 does not fit a's cap (15 + 10 + label change) and is skipped; b1 is taken.
    g = admit_recall_gated({a, b}, 1.0, 0.01);
    REQUIRE(g.regions[0].admitted == std::vector<char>({0, 1}));
    REQUIRE(g.regions[0].cost <= a.cap);
    REQUIRE(g.regions[1].admitted == std::vector<char>({1, 1}));
}

TEST_CASE(PFlashUnifiedFixture, unified_rank_units_orders_by_score_then_document) {
    REQUIRE(rank_units({0.5, 2.0, 0.5, NAN, 3.0}, {}) == std::vector<int>({4, 1, 0, 2, 3}));
    REQUIRE(rank_units({0.5, 2.0, 0.5}, {0, 1, 0}) == std::vector<int>({0, 2}));
}

TEST_CASE(PFlashUnifiedFixture, test_unified_reread_by_content) {
    std::string file;
    for (int i = 1; i <= 100; ++i) file += "def f" + std::to_string(i) + "():\n    return " + std::to_string(i) + "\n}\n\n";
    const Lines region_lines = detect_lines(file, 0, file.size());
    const auto region = line_hashes(file, region_lines);

    // A partial re-read of the same file, delivered JSON-wrapped: decoded lines compare equal.
    std::string part;
    for (int i = 40; i <= 60; ++i) part += "def f" + std::to_string(i) + "():\n    return " + std::to_string(i) + "\n}\n\n";
    const std::string wrapped = nlohmann::json({{"content", part}}).dump();
    const Lines part_lines = detect_lines(wrapped, 0, wrapped.size());
    REQUIRE(part_lines.escaped);
    RereadMatch m = match_reread(line_hashes(wrapped, part_lines), region);
    REQUIRE(m.is_reread());
    REQUIRE(m.matched_candidate == m.candidate_informative);
    REQUIRE(m.candidate_informative == 21 * 2);             // "}" (100 copies) and blank lines are skipped
    int covered = 0;
    for (char c : m.region_covered) covered += c;
    REQUIRE(covered == 21 * 2 + 20 * 2);                    // + the "}" and blank lines between matched blocks
    REQUIRE(!m.region_covered[(size_t) (59 * 4 + 2)]);      // the "}" after the last block: next line unmatched
    REQUIRE(m.region_covered[(size_t) (39 * 4)]);           // "def f40():"
    REQUIRE(!m.region_covered[(size_t) (38 * 4)]);          // "def f39():"
    REQUIRE(m.region_informative[(size_t) (39 * 4)]);
    REQUIRE(!m.region_informative[(size_t) (39 * 4 + 2)]);  // "}"
    REQUIRE(!m.region_informative[(size_t) (39 * 4 + 3)]);  // blank
    // A unit ending in the last block's "}" and blank line is covered: its informative lines are; one reaching
    // into f61 is not.
    REQUIRE(lines_covered(m.region_covered, m.region_informative, 59 * 4, 60 * 4));
    REQUIRE(!lines_covered(m.region_covered, m.region_informative, 59 * 4, 60 * 4 + 1));
    // Lines with nothing informative count only when covered themselves (between two covered blocks).
    REQUIRE(lines_covered(m.region_covered, m.region_informative, 39 * 4 + 2, 40 * 4));
    REQUIRE(!lines_covered(m.region_covered, m.region_informative, 59 * 4 + 2, 60 * 4));

    // A changed file matches only where it is unchanged; "}" lines (more than 32 copies) never match alone.
    std::string changed = part;
    for (char & c : changed) if (c == 'r') c = 'R';
    m = match_reread(line_hashes(changed, detect_lines(changed, 0, changed.size())), region);
    REQUIRE(m.matched_candidate == 0);
    REQUIRE(!m.is_reread());
    const std::string unrelated = numbered_text(30, "other ");
    m = match_reread(line_hashes(unrelated, detect_lines(unrelated, 0, unrelated.size())), region);
    REQUIRE(!m.is_reread());
}

TEST_CASE(PFlashUnifiedFixture, test_unified_reread_multiple_regions) {
    const std::string v1 = numbered_text(50, "alpha ");
    std::string v2 = v1;
    v2.replace(v2.find("alpha 20\n"), 9, "ALPHA 20\n");
    const auto h1 = line_hashes(v1, detect_lines(v1, 0, v1.size()));
    const auto h2 = line_hashes(v2, detect_lines(v2, 0, v2.size()));
    const std::string reread = numbered_text(25, "alpha ");
    const auto hr = line_hashes(reread, detect_lines(reread, 0, reread.size()));
    const RereadMatch m1 = match_reread(hr, h1);
    const RereadMatch m2 = match_reread(hr, h2);
    REQUIRE(m1.region_covered[19]);
    REQUIRE(!m2.region_covered[19]);
    REQUIRE(m2.region_covered[18]);
    REQUIRE(m2.region_covered[20]);
}

TEST_CASE(PFlashUnifiedFixture, unified_view_store_finds_longest_prefix_and_keeps_ancestors) {
    ViewStore store(3, (size_t) 1 << 30);
    auto make = [](std::vector<int32_t> raw, int gen, const std::string & cfg) {
        View v;
        v.raw_tokens = raw;
        v.raw_gen_begin = gen;
        v.drafter_ids = raw;
        v.drafter_gen_begin = gen;
        v.config = cfg;
        return v;
    };
    store.remember(make({1, 2, 3, 9}, 3, "c"));
    store.remember(make({1, 2, 3, 4, 5, 9}, 5, "c"));
    View out;
    REQUIRE(store.find({1, 2, 3, 4, 5, 6, 7}, {1, 2, 3, 4, 5, 6, 7}, "c", out));
    REQUIRE(out.raw_gen_begin == 5);
    REQUIRE(store.find({1, 2, 3, 7}, {1, 2, 3, 7}, "c", out));   // a fork continues the ancestor
    REQUIRE(out.raw_gen_begin == 3);
    REQUIRE(!store.find({1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}, "other", out));
    REQUIRE(!store.find({1, 2, 3, 4, 5, 6}, {1, 2, 9, 4, 5, 6}, "c", out));   // drafter ids must match too
    // A retry replaces its own entry; capacity evicts the least recently used, never the newest.
    store.remember(make({1, 2, 3, 4, 5, 9}, 5, "c"));
    REQUIRE(store.size() == 2);
    store.remember(make({7, 7}, 1, "c"));
    store.remember(make({8, 8}, 1, "c"));
    REQUIRE(store.size() == 3);
    REQUIRE(!store.find({1, 2, 3, 0}, {1, 2, 3, 0}, "c", out));
    REQUIRE(store.find({8, 8, 1}, {8, 8, 1}, "c", out));
}

TEST_CASE(PFlashUnifiedFixture, unified_inline_layout_cuts_inside_lines) {
    // A document on one long line between short lines: line units cannot page it (four lines), inline units can.
    std::string body;
    for (int i = 0; i < 40; ++i) body += "word " + std::to_string(i) + " is here. ";
    const std::string text = "<doc>\n" + body + "\n</doc>\nWhich?";
    const Lines lines = detect_lines(text, 0, text.size());
    REQUIRE(lines.count() == 4);
    std::vector<size_t> every;
    for (size_t i = 0; i <= text.size(); ++i) every.push_back(i);
    std::vector<size_t> proposed;
    for (size_t p = lines.starts[1] + 100; p < lines.ends[1]; p += 100) proposed.push_back(p);
    Layout layout;
    std::string why;
    REQUIRE(!layout_region(lines, every, proposed, layout, why));
    REQUIRE(layout_region_inline(text, lines, every, proposed, true, true, layout, why));
    REQUIRE(layout.inline_units);
    REQUIRE(layout.head_end == lines.starts[1]);           // "<doc>\n" kept whole
    REQUIRE(layout.tail_begin == lines.starts[3]);         // "Which?" kept whole
    REQUIRE(layout.cut_line[1] == 1 && layout.cut_line[2] == 1);   // cuts inside line 1
    REQUIRE(unit_lines(lines, layout, 1, 1) == std::make_pair(1, 2));
    std::vector<char> admitted((size_t) layout.units(), 1);
    admitted[1] = 0;
    const std::string served = render_region_text(text, 0, text.size(), lines, layout, admitted,
                                                  RegionKind::UserContent);
    REQUIRE(served.find("<doc>\n[Server note: this message was shortened; each omitted range is marked inline.]\n") ==
            0);
    REQUIRE(served.find(text.substr(proposed[0], 20) + "") == std::string::npos);
    REQUIRE(served.find("\n[\xE2\x80\xA6 text not shown \xE2\x80\xA6]\n") != std::string::npos);
    REQUIRE(served.size() > 6 && served.compare(served.size() - 6, 6, "Which?") == 0);
    REQUIRE(unit_marker_text(layout, 0, 0) == "[\xE2\x80\xA6 text not shown \xE2\x80\xA6]");

    // Without the edge lines kept, the head and tail are the first and last inline units.
    REQUIRE(layout_region_inline(text, lines, every, proposed, false, false, layout, why));
    REQUIRE(layout.head_end == lines.starts[1] && layout.tail_begin == lines.starts[3]);
    REQUIRE(layout_region_inline(text, lines, every, {lines.starts[3] + 2}, false, false, layout, why));
    REQUIRE(layout.tail_begin == lines.starts[3] + 2);
    // Too few cuts: no layout.
    REQUIRE(!layout_region_inline(text, lines, {}, proposed, true, true, layout, why));

    // No cut inside a UTF-8 character or a JSON escape: an offset there moves to the next clean one.
    const std::string utf = std::string(60, 'a') + "\xC3\xA9" + std::string(60, 'b') + "\nend";
    const Lines utf_lines = detect_lines(utf, 0, utf.size());
    std::vector<size_t> utf_every;
    for (size_t i = 0; i <= utf.size(); ++i) utf_every.push_back(i);
    REQUIRE(layout_region_inline(utf, utf_lines, utf_every, {10, 61, 90}, false, false, layout, why));
    REQUIRE(std::find(layout.cuts.begin(), layout.cuts.end(), (size_t) 61) == layout.cuts.end());
    REQUIRE(std::find(layout.cuts.begin(), layout.cuts.end(), (size_t) 62) != layout.cuts.end());
    nlohmann::json value = {{"content", std::string(50, 'x') + "\u00e9\n" + std::string(50, 'y') + "\"q\""}};
    const std::string json_text = value.dump(-1, ' ', true);
    const Lines escaped = split_lines(json_text, 12, json_text.size() - 2, true);
    REQUIRE(escaped.escaped);
    std::vector<size_t> json_every;
    for (size_t i = 0; i <= json_text.size(); ++i) json_every.push_back(i);
    const size_t u = json_text.find("\\u00e9");
    REQUIRE(u != std::string::npos);
    REQUIRE(layout_region_inline(json_text, escaped, json_every, {20, u + 3, u + 30}, false, false, layout, why));
    for (size_t c : layout.cuts) REQUIRE(!(c > u && c < u + 6));
}


// A small view with one region whose per-unit and per-line vectors agree.
static View disk_test_view(const std::vector<int32_t> & raw, int gen, const std::string & config, uint64_t lineage) {
    View v;
    v.lineage = lineage;
    v.version = 2;
    v.config = config;
    v.raw_tokens = raw;
    v.raw_gen_begin = gen;
    v.drafter_ids = raw;
    v.drafter_gen_begin = gen;
    v.view_tokens = {raw.front(), 99, raw.back()};
    v.view_gen_begin = 2;
    v.turns = 1;
    Region r;
    r.kind = RegionKind::ToolOutput;
    r.brief = "read_file(a.md)";
    r.tool_call_id = "call_1";
    r.content_sha = "ab";
    r.cb = 3;
    r.ce = 30;
    r.te = 2;
    r.de = 2;
    r.lines.escaped = true;
    r.lines.begin = 3;
    r.lines.end = 30;
    r.lines.starts = {3, 12, 21};
    r.lines.ends = {10, 19, 30};
    r.layout.head_end = 12;
    r.layout.tail_begin = 21;
    r.layout.cuts = {12, 21};
    r.layout.cut_line = {1, 2};
    r.hashes = {0, 18446744073709551615ULL, 7};
    r.unit_tb = {0};
    r.unit_te = {1};
    r.unit_db = {0};
    r.unit_de = {1};
    r.mandatory = {0};
    r.kept = {1};
    r.held = {1};
    r.reread = {0, 1, 0};
    r.cap = 5;
    r.droppable = 9;
    r.recalled_tokens = 2;
    v.regions.push_back(r);
    v.pieces.push_back({0, 1, "raw", -1, 0});
    v.pieces.push_back({1, 2, "paged_text", 0, 0});
    return v;
}

TEST_CASE(PFlashUnifiedFixture, unified_view_json_round_trip_fails_closed) {
    const View v = disk_test_view({1, 2, 3, 4, 9}, 4, "c", 42);
    View back;
    std::string error;
    REQUIRE(view_from_json(view_to_json(v), back, error));
    REQUIRE(view_to_json(back) == view_to_json(v));
    REQUIRE(back.regions[0].hashes[1] == 18446744073709551615ULL);
    REQUIRE(back.lineage == 42 && back.regions[0].kind == RegionKind::ToolOutput && back.regions[0].lines.escaped);
    const auto broken = [&] (const std::function<void(nlohmann::json &)> & edit) {
        auto j = view_to_json(v);
        edit(j);
        View out;
        std::string why;
        return !view_from_json(j, out, why) && !why.empty();
    };
    REQUIRE(broken([] (nlohmann::json & j) { j["format"] = 2; }));
    REQUIRE(broken([] (nlohmann::json & j) { j.erase("raw_tokens"); }));
    REQUIRE(broken([] (nlohmann::json & j) { j["view_gen_begin"] = 9; }));
    REQUIRE(broken([] (nlohmann::json & j) { j["regions"][0]["kept"] = "11"; }));       // one unit
    REQUIRE(broken([] (nlohmann::json & j) { j["regions"][0]["held"] = "x"; }));
    REQUIRE(broken([] (nlohmann::json & j) { j["regions"][0]["te"] = 50; }));          // past the raw prompt
    REQUIRE(broken([] (nlohmann::json & j) { j["regions"][0]["kind"] = "other"; }));
    REQUIRE(broken([] (nlohmann::json & j) { j["pieces"][0][1] = 50; }));             // past the view
    REQUIRE(broken([] (nlohmann::json & j) { j["pieces"][1][3] = 3; }));              // no such region
    REQUIRE(broken([] (nlohmann::json & j) { j["pieces"][1][3] = -2; }));
    // Vectors serving indexes per line or per unit, and the offsets it walks, must agree.
    REQUIRE(broken([] (nlohmann::json & j) { j["regions"][0]["reread"] = ""; }));
    REQUIRE(broken([] (nlohmann::json & j) { j["regions"][0]["hashes"] = nlohmann::json::array(); }));
    REQUIRE(broken([] (nlohmann::json & j) { j["regions"][0]["lines"]["starts"] = {3, 2, 21}; }));
    REQUIRE(broken([] (nlohmann::json & j) { j["regions"][0]["lines"]["ends"] = {10, 19, 40}; }));   // past the content
    REQUIRE(broken([] (nlohmann::json & j) { j["regions"][0]["layout"]["cuts"] = {21, 12}; }));
    REQUIRE(broken([] (nlohmann::json & j) { j["regions"][0]["layout"]["cut_line"] = {1, 3}; }));   // no line 3
    REQUIRE(broken([] (nlohmann::json & j) { j["regions"][0]["layout"]["head_end"] = 11; }));
    REQUIRE(broken([] (nlohmann::json & j) { j["regions"][0]["unit_te"] = {3}; }));     // past the content's tokens
    REQUIRE(broken([] (nlohmann::json & j) { j["regions"][0]["unit_db"] = {2}; }));
    REQUIRE(broken([] (nlohmann::json & j) { j["raw_gen_begin"] = 1; }));            // region after the generation
    REQUIRE(broken([] (nlohmann::json & j) { j["regions"][0]["layout"].erase("inline"); }));
    // Inline units may cut one line several times; line units may not.
    View in_line = v;
    in_line.regions[0].layout.cut_line = {1, 1};
    in_line.regions[0].layout.inline_units = true;
    REQUIRE(view_from_json(view_to_json(in_line), back, error));
    REQUIRE(back.regions[0].layout.inline_units);
    in_line.regions[0].layout.inline_units = false;
    REQUIRE(!view_from_json(view_to_json(in_line), back, error));
}

TEST_CASE(PFlashUnifiedFixture, unified_view_stores_name_the_lineages_a_prompt_continues) {
    namespace fs = std::filesystem;
    ViewStore store;
    store.remember(disk_test_view({1, 2, 3, 9}, 3, "c", 1));
    store.remember(disk_test_view({1, 2, 3, 4, 5, 9}, 5, "other", 2));
    store.remember(disk_test_view({7, 7, 9}, 2, "c", 3));
    auto lineages = store.lineages_continued_by({1, 2, 3, 4, 5, 6});
    std::sort(lineages.begin(), lineages.end());
    REQUIRE(lineages == std::vector<uint64_t>({1, 2}));   // any config: none of them may continue
    REQUIRE(store.lineages_continued_by({1, 2, 4}).empty());

    const fs::path dir = fs::temp_directory_path() / ("luce_test_view_lineages_" + std::to_string(getpid()));
    fs::remove_all(dir);
    ViewDiskStore disk;
    REQUIRE(disk.lineages_continued_by({1, 2, 3, 4}).empty());   // not initialised
    REQUIRE(disk.init(dir.string(), "id-a", 0, 8));
    REQUIRE(disk.save(disk_test_view({1, 2, 3, 9}, 3, "c", 1)));
    REQUIRE(disk.save(disk_test_view({7, 7, 9}, 2, "c", 3)));
    REQUIRE(disk.lineages_continued_by({1, 2, 3, 4}) == std::vector<uint64_t>({1}));
    REQUIRE(disk.lineages_continued_by({1, 2}).empty());
    fs::remove_all(dir);
}

TEST_CASE(PFlashUnifiedFixture, unified_view_disk_store_applies_the_ttl_at_lookup) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("luce_test_view_ttl_" + std::to_string(getpid()));
    fs::remove_all(dir);
    uint64_t now = 1'000'000'000;
    ViewDiskStore store;
    store.set_clock([&] { return now; });
    REQUIRE(store.init(dir.string(), "id-a", 3600, 8));
    REQUIRE(store.save(disk_test_view({1, 2, 3, 9}, 3, "c", 1)));
    View out;
    now += 3600 * 1000;
    REQUIRE(store.find({1, 2, 3, 7}, {1, 2, 3, 7}, "c", 0, out));   // at the TTL: still there
    now += 1;
    REQUIRE(!store.find({1, 2, 3, 7}, {1, 2, 3, 7}, "c", 0, out));  // past it: gone, file and all
    REQUIRE(store.size() == 0);
    REQUIRE(fs::is_empty(dir));
    fs::remove_all(dir);
}

TEST_CASE(PFlashUnifiedFixture, unified_view_disk_store_restart_identity_cap_and_ttl) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("luce_test_view_disk_" + std::to_string(getpid()));
    fs::remove_all(dir);
    const auto view_files = [&] {
        std::vector<fs::path> out;
        for (const auto & e : fs::directory_iterator(dir)) {
            if (e.path().extension() == ".view") out.push_back(e.path());
        }
        return out;
    };
    {
        ViewDiskStore store;
        REQUIRE(store.init(dir.string(), "id-a", 0, 3));
        REQUIRE(store.save(disk_test_view({1, 2, 3, 9}, 3, "c", 1)));
        REQUIRE(store.save(disk_test_view({1, 2, 3, 4, 5, 9}, 5, "c", 1)));
        REQUIRE(store.save(disk_test_view({1, 2, 3, 4, 5, 9}, 5, "c", 1)));   // the same prefixes: replaced
        REQUIRE(store.size() == 2 && view_files().size() == 2);
    }
    {
        // A new process rebuilds the index from the headers and reads a view only when it matches.
        ViewDiskStore store;
        REQUIRE(store.init(dir.string(), "id-a", 0, 3));
        REQUIRE(store.size() == 2);
        REQUIRE(store.max_lineage() == 1);
        View out;
        REQUIRE(store.find({1, 2, 3, 4, 5, 6, 7}, {1, 2, 3, 4, 5, 6, 7}, "c", 0, out));
        REQUIRE(view_to_json(out) == view_to_json(disk_test_view({1, 2, 3, 4, 5, 9}, 5, "c", 1)));
        REQUIRE(store.find({1, 2, 3, 7}, {1, 2, 3, 7}, "c", 0, out));   // a fork continues the ancestor
        REQUIRE(out.raw_gen_begin == 3);
        REQUIRE(!store.find({1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}, "c", 5, out));   // RAM holds as long a view
        REQUIRE(!store.find({1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}, "other", 0, out));
        REQUIRE(!store.find({1, 2, 3, 4, 5, 6}, {1, 2, 9, 4, 5, 6}, "c", 0, out));   // drafter ids must match too
        // Over the cap the least recently written goes.
        REQUIRE(store.save(disk_test_view({7, 7, 9}, 2, "c", 2)));
        REQUIRE(store.save(disk_test_view({8, 8, 9}, 2, "c", 3)));
        REQUIRE(store.size() == 3 && view_files().size() == 3);
        REQUIRE(!store.find({1, 2, 3, 7}, {1, 2, 3, 7}, "c", 0, out));
        REQUIRE(store.find({1, 2, 3, 4, 5, 6, 7}, {1, 2, 3, 4, 5, 6, 7}, "c", 0, out));
        // A conversation the chat view took over leaves the disk too.
        store.forget_lineage(1);
        REQUIRE(store.size() == 2 && view_files().size() == 2);
        REQUIRE(!store.find({1, 2, 3, 4, 5, 6, 7}, {1, 2, 3, 4, 5, 6, 7}, "c", 0, out));
    }
    {
        // Another identity (model, template, context) never reads these views.
        ViewDiskStore other;
        REQUIRE(other.init(dir.string(), "id-b", 0, 3));
        REQUIRE(other.size() == 0);
        View out;
        REQUIRE(!other.find({7, 7, 1}, {7, 7, 1}, "c", 0, out));
        REQUIRE(view_files().size() == 2);
    }
    {
        // A view whose body does not load is deleted when read; the request then has no view.
        for (const auto & file : view_files()) {
            std::ifstream in(file);
            std::string header;
            std::getline(in, header);
            in.close();
            if (header.find("\"lineage\":2") == std::string::npos) continue;
            std::ofstream(file, std::ios::trunc) << header << "\n{\"format\":1}\n";
        }
        ViewDiskStore store;
        REQUIRE(store.init(dir.string(), "id-a", 0, 3));
        View out;
        REQUIRE(!store.find({7, 7, 1}, {7, 7, 1}, "c", 0, out));
        REQUIRE(store.size() == 1 && view_files().size() == 1);
        REQUIRE(store.find({8, 8, 1}, {8, 8, 1}, "c", 0, out));
    }
    {
        // At startup: an unfinished write is removed, and so is a view older than the TTL.
        std::ofstream(dir / "x.view.tmp") << "partial";
        std::ofstream(dir / "old.view") << nlohmann::json({{"format", 1}, {"identity", "id-a"}, {"lineage", 9},
                                                             {"config", "c"}, {"raw_len", 1}, {"raw_hash", 1},
                                                             {"drafter_len", 1}, {"drafter_hash", 1},
                                                             {"written", 1000}}).dump() << "\n{}\n";
        ViewDiskStore store;
        REQUIRE(store.init(dir.string(), "id-a", 3600, 3));
        REQUIRE(store.size() == 1);
        REQUIRE(!fs::exists(dir / "x.view.tmp") && !fs::exists(dir / "old.view"));
    }
    fs::remove_all(dir);
}

TEST_CASE(PFlashUnifiedFixture, unified_view_store_reports_evicted_views) {
    ViewStore store(2, (size_t) 1 << 30);
    REQUIRE(store.remember(disk_test_view({1, 9}, 1, "c", 1)).empty());
    REQUIRE(store.remember(disk_test_view({2, 9}, 1, "c", 2)).empty());
    const auto evicted = store.remember(disk_test_view({3, 9}, 1, "c", 3));
    REQUIRE(evicted.size() == 1 && evicted[0].lineage == 1);
    const auto all = store.all();
    REQUIRE(all.size() == 2 && all[0].lineage == 3 && all[1].lineage == 2);   // most recently used first
}

TEST_CASE(PFlashUnifiedFixture, unified_text_helpers) {
    REQUIRE(json_escape("a\"b\\c\nd\te\x01 \xC3\xA9") == "a\\\"b\\\\c\\nd\\te\\u0001 \xC3\xA9");
    REQUIRE(sha1_hex("abc") == "a9993e364706816aba3e25717850c26c9cd0d89d");
    REQUIRE(preamble_text(RegionKind::ToolOutput, 3, 10) ==
            "[Server note: this tool result was shortened: 3 of 10 lines are shown; each omitted range is marked "
            "inline. Call the tool again if you need an omitted part.]");
}
