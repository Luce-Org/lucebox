#include "CppUnitTestFramework.hpp"

#include "common/gbnf.h"
#include "common/output_constraint.h"
#include "server/json_schema_grammar.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace luce::common;

namespace {

struct GbnfUtf8Fixture {};

std::string bytes(std::initializer_list<int> values) {
    std::string out;
    for (int v : values) out.push_back(static_cast<char>(v));
    return out;
}

std::vector<std::vector<std::string>> splits(const std::string & text) {
    std::vector<std::vector<std::string>> out{ { text } };
    for (size_t i = 1; i < text.size(); ++i) out.push_back({ text.substr(0, i), text.substr(i) });
    if (text.size() > 2) {
        std::vector<std::string> each;
        for (char c : text) each.emplace_back(1, c);
        out.push_back(each);
    }
    return out;
}

std::string hex(const std::vector<std::string> & pieces) {
    static const char digits[] = "0123456789ABCDEF";
    std::string out;
    for (const auto & piece : pieces) {
        if (!out.empty()) out += " |";
        for (unsigned char c : piece) {
            if (!out.empty()) out += ' ';
            out += digits[c >> 4];
            out += digits[c & 0xF];
        }
    }
    return out;
}

struct Decoded {
    std::vector<uint32_t> code_points;  // without terminators
    gbnf::PartialUtf8 partial;
};

Decoded decode_pieces(const std::vector<std::string> & pieces) {
    Decoded out;
    for (const auto & piece : pieces) {
        const auto decoded = gbnf::decode_utf8(piece, out.partial);
        out.code_points.insert(out.code_points.end(), decoded.first.begin(), decoded.first.end() - 1);
        out.partial = decoded.second;
    }
    return out;
}

std::unique_ptr<gbnf::Matcher> compile(const std::string & grammar) {
    std::string error;
    auto matcher = gbnf::Matcher::compile(grammar, "root", error);
    if (!matcher) throw std::runtime_error("test grammar does not compile: " + error);
    return matcher;
}

bool candidate_rejected(const std::string & grammar, const std::string & piece) {
    const auto decoded = gbnf::decode_utf8(piece, {});
    return !compile(grammar)->reject({ { 0, decoded.first.data(), decoded.second } }).empty();
}

std::unique_ptr<gbnf::Matcher> fed(const std::string & grammar, const std::vector<std::string> & pieces) {
    auto matcher = compile(grammar);
    for (const auto & piece : pieces) matcher->accept_bytes(piece);
    return matcher;
}

// Malformed UTF-8 must not match a character allowed by the grammar
const std::string kAcute = "root ::= \"\\u00C1\"\n";
const std::string kAny   = "root ::= .*\n";

struct Valid { std::string text; uint32_t code_point; };

const std::vector<Valid> kValid = {
    { bytes({ 0xC2, 0x80 }), 0x80 },
    { bytes({ 0xE0, 0xA0, 0x80 }), 0x800 },
    { bytes({ 0xED, 0x9F, 0xBF }), 0xD7FF },
    { bytes({ 0xEF, 0xBF, 0xBF }), 0xFFFF },
    { bytes({ 0xF0, 0x90, 0x80, 0x80 }), 0x10000 },
    { bytes({ 0xF4, 0x8F, 0xBF, 0xBF }), 0x10FFFF },
};

const std::vector<std::string> kMalformed = {
    bytes({ 0xC3, 0x41 }),              // ASCII where a continuation belongs
    bytes({ 0xC3, 0xC3 }),              // a lead where a continuation belongs
    bytes({ 0xC0, 0x80 }),              // overlong U+0000
    bytes({ 0xC1, 0x81 }),              // overlong 'A'
    bytes({ 0xE0, 0x9F, 0xBF }),        // overlong U+07FF
    bytes({ 0xE1, 0x80, 0x41 }),        // third byte not a continuation
    bytes({ 0xED, 0xA0, 0x80 }),        // surrogate U+D800
    bytes({ 0xF0, 0x8F, 0xBF, 0xBF }),  // overlong U+FFFF
    bytes({ 0xF1, 0x80, 0x80, 0x41 }),  // fourth byte not a continuation
    bytes({ 0xF4, 0x90, 0x80, 0x80 }),  // U+110000
    bytes({ 0xF5, 0x80, 0x80, 0x80 }),  // lead past U+10FFFF
    bytes({ 0xFF }),
};

}  // namespace

TEST_CASE(GbnfUtf8Fixture, gbnf_utf8_decodes_boundary_code_points_in_any_split) {
    for (const auto & valid : kValid) {
        const auto section = EnterSection("bytes " + hex({ valid.text }));
        CHECK(gbnf::decode_utf8(valid.text, {}).first == std::vector<uint32_t>({ valid.code_point, 0 }));
        for (const auto & pieces : splits(valid.text)) {
            const auto split = EnterSection("pieces " + hex(pieces));
            const auto decoded = decode_pieces(pieces);
            CHECK(decoded.code_points == std::vector<uint32_t>({ valid.code_point }));
            CHECK(decoded.partial.n_remain == 0);
        }
    }
}

TEST_CASE(GbnfUtf8Fixture, gbnf_utf8_rejects_malformed_sequences_in_any_split) {
    for (const auto & text : kMalformed) {
        const auto section = EnterSection("bytes " + hex({ text }));
        const auto whole = gbnf::decode_utf8("ok" + text, {});
        CHECK(whole.first == std::vector<uint32_t>({ 0 }));
        CHECK(whole.second.n_remain == -1);
        for (const auto & pieces : splits(text)) {
            const auto split = EnterSection("pieces " + hex(pieces));
            CHECK(decode_pieces(pieces).partial.n_remain == -1);
        }
    }
}

TEST_CASE(GbnfUtf8Fixture, gbnf_utf8_leaves_incomplete_sequences_open) {
    const auto open = gbnf::decode_utf8("ab" + bytes({ 0xF0, 0x9F }), {});
    CHECK(open.first == std::vector<uint32_t>({ 'a', 'b', 0 }));
    CHECK(open.second.n_remain == 2);
    const auto closed = gbnf::decode_utf8(bytes({ 0x98, 0x80 }) + "c", open.second);
    CHECK(closed.first == std::vector<uint32_t>({ 0x1F600, 'c', 0 }));
    CHECK(closed.second.n_remain == 0);
}

TEST_CASE(GbnfUtf8Fixture, gbnf_utf8_invalid_state_stays_invalid) {
    const auto broken = gbnf::decode_utf8(bytes({ 0x80 }), {}).second;
    CHECK(broken.n_remain == -1);
    const auto next = gbnf::decode_utf8("a", broken);
    CHECK(next.first == std::vector<uint32_t>({ 0 }));
    CHECK(next.second.n_remain == -1);
}

TEST_CASE(GbnfUtf8Fixture, gbnf_utf8_matcher_rejects_malformed_sequences) {
    for (const auto & pieces : splits(bytes({ 0xC3, 0x81 }))) {
        const auto section = EnterSection("pieces " + hex(pieces));
        auto matcher = fed(kAcute, pieces);
        CHECK(!matcher->dead() && matcher->can_end());
    }
    for (const auto & text : { bytes({ 0xC3, 0x41 }), bytes({ 0xE0, 0x83, 0x81 }) }) {
        for (const auto & pieces : splits(text)) {
            const auto section = EnterSection("pieces " + hex(pieces));
            CHECK(fed(kAcute, pieces)->dead());
        }
    }
    auto prefix = fed(kAcute, { bytes({ 0xC3 }) });
    CHECK(!prefix->dead());
    CHECK(!prefix->can_end());

    for (const auto & pieces : splits(bytes({ 0xED, 0xA0, 0x80 }))) {
        const auto section = EnterSection("pieces " + hex(pieces));
        CHECK(fed(kAny, pieces)->dead());
    }
}

TEST_CASE(GbnfUtf8Fixture, gbnf_utf8_batch_rejection_drops_only_malformed_candidates) {
    struct Case { std::string piece; bool rejected; };
    const auto check_batch = [&](gbnf::Matcher & matcher, const std::vector<Case> & cases) {
        std::vector<std::pair<std::vector<uint32_t>, gbnf::PartialUtf8>> decoded;
        for (const auto & c : cases) decoded.push_back(gbnf::decode_utf8(c.piece, matcher.partial()));
        std::vector<gbnf::Candidate> candidates;
        for (size_t i = 0; i < cases.size(); ++i) {
            candidates.push_back({ i, decoded[i].first.data(), decoded[i].second });
        }
        std::vector<bool> rejected(cases.size(), false);
        for (const auto & reject : matcher.reject(candidates)) rejected[reject.index] = true;
        for (size_t i = 0; i < cases.size(); ++i) {
            const auto section = EnterSection("piece " + hex({ cases[i].piece }));
            CHECK(rejected[i] == cases[i].rejected);
        }
    };

    check_batch(*compile(kAny), {
        { bytes({ 0xED, 0x9F, 0xBF }), false },
        { bytes({ 0xED, 0xA0, 0x80 }), true },
        { bytes({ 0xED, 0x9F }), false },
        { bytes({ 0xC3, 0x41 }), true },
        { "A", false },
    });
    check_batch(*fed(kAny, { bytes({ 0xED }) }), {
        { bytes({ 0x9F, 0xBF }), false },
        { bytes({ 0xA0, 0x80 }), true },
        { "A", true },
    });
}

TEST_CASE(GbnfUtf8Fixture, gbnf_utf8_constraint_allows_and_masks_only_valid_utf8) {
    enum Tok : int32_t { VALID, MISSING_CONT, OVERLONG, LEAD, CONT, ASCII, END, N_TOK };
    std::vector<std::string> pieces(N_TOK);
    pieces[VALID] = bytes({ 0xC3, 0x81 });
    pieces[MISSING_CONT] = bytes({ 0xC3, 0x41 });
    pieces[OVERLONG] = bytes({ 0xE0, 0x83, 0x81 });
    pieces[LEAD] = bytes({ 0xC3 });
    pieces[CONT] = bytes({ 0x81 });
    pieces[ASCII] = "A";
    pieces[END] = "<|im_end|>";
    OutputConstraint constraint(ConstraintVocab::build(pieces, { END }), compile(kAcute), {});

    const auto allowed = [&]() {
        std::vector<bool> out;
        for (int32_t t = 0; t < N_TOK; ++t) out.push_back(constraint.allows(t));
        return out;
    };
    const auto check_mask = [&](const std::vector<bool> & expected, int expected_kept) {
        std::vector<float> logits;
        for (int32_t t = 0; t < N_TOK; ++t) logits.push_back(1.0f + t);
        CHECK(constraint.mask(logits.data(), N_TOK) == expected_kept);
        for (int32_t t = 0; t < N_TOK; ++t) {
            const auto section = EnterSection("token " + std::to_string(t));
            CHECK(logits[t] == (expected[t] ? 1.0f + t : -INFINITY));
        }
    };

    const std::vector<bool> at_start = { true, false, false, true, false, false, false };
    CHECK(allowed() == at_start);
    check_mask(at_start, 2);

    constraint.accept(LEAD);
    const std::vector<bool> after_lead = { false, false, false, false, true, false, false };
    CHECK(allowed() == after_lead);
    check_mask(after_lead, 1);

    constraint.accept(CONT);
    CHECK(constraint.complete());
    const std::vector<bool> at_end = { false, false, false, false, false, false, true };
    CHECK(allowed() == at_end);
    check_mask(at_end, 1);
}

TEST_CASE(GbnfUtf8Fixture, gbnf_utf8_open_lead_is_admitted_up_to_the_last_scalar) {
    struct Case { std::string grammar; std::string lead; std::string last; };
    const std::vector<Case> cases = {
        // ED completes to U+D7FF at most
        { "root ::= [\\uD7FF-\\uE000]\n", bytes({ 0xED }), bytes({ 0xED, 0x9F, 0xBF }) },
        // F4 completes to U+10FFFF at most
        { "root ::= \"\\U0010FFFF\"\n", bytes({ 0xF4 }), bytes({ 0xF4, 0x8F, 0xBF, 0xBF }) },
        // Use a GBNF range beyond Unicode to isolate the F4 prefix bound.
        { "root ::= [^\\U00110000-\\U0013FFFF]+\n", bytes({ 0xF4 }), bytes({ 0xF4, 0x8F, 0xBF, 0xBF }) },
    };
    for (const auto & c : cases) {
        const auto section = EnterSection("grammar " + c.grammar.substr(0, c.grammar.find('\n')));
        CHECK(!fed(c.grammar, { c.lead })->dead());
        CHECK(!candidate_rejected(c.grammar, c.lead));
        auto matcher = fed(c.grammar, { c.last });
        CHECK(!matcher->dead() && matcher->can_end());
    }
}

TEST_CASE(GbnfUtf8Fixture, gbnf_utf8_schema_pattern_without_surrogates_admits_ed_text) {
    enum Tok : int32_t { QUOTE, ED_LEAD, HANGUL_REST, BEFORE_SURROGATES, SURROGATE, END, N_TOK };
    std::vector<std::string> pieces(N_TOK);
    pieces[QUOTE] = "\"";
    pieces[ED_LEAD] = bytes({ 0xED });
    pieces[HANGUL_REST] = bytes({ 0x95, 0x9C });  // after ED_LEAD: U+D55C
    pieces[BEFORE_SURROGATES] = bytes({ 0xED, 0x9F, 0xBF });
    pieces[SURROGATE] = bytes({ 0xED, 0xA0, 0x80 });
    pieces[END] = "<|im_end|>";
    const auto schema = nlohmann::ordered_json::parse(R"({"type": "string", "pattern": "^[^\\uD800-\\uDFFF]+$"})");
    OutputConstraint constraint(ConstraintVocab::build(pieces, { END }), compile(json_schema_to_grammar(schema)), {});

    constraint.accept(QUOTE);
    CHECK(constraint.allows(ED_LEAD));
    CHECK(constraint.allows(BEFORE_SURROGATES));
    CHECK(!constraint.allows(SURROGATE));
    constraint.accept(ED_LEAD);
    CHECK(constraint.allows(HANGUL_REST));
    constraint.accept(HANGUL_REST);
    constraint.accept(QUOTE);
    CHECK(constraint.complete());
}

TEST_CASE(GbnfUtf8Fixture, gbnf_repetition_counts_are_bounded) {
    // Within the bound, a count means what it says.
    CHECK(fed("root ::= \"a\"{2,3}\n", { "aa" })->can_end());
    CHECK(fed("root ::= \"a\"{2,3}\n", { "aaaa" })->dead());
    // A maximum above the bound is "no maximum": the grammar stays small.
    auto open = fed("root ::= \"a\"{1,2000000000}\n", { std::string(5000, 'a') });
    CHECK(!open->dead() && open->can_end());
    CHECK(open->rules().size() < 16);
    // A minimum above the bound is refused, not expanded.
    std::string error;
    CHECK(!gbnf::Matcher::compile("root ::= \"a\"{2000000000}\n", "root", error));
    CHECK(error.find("repetition count") != std::string::npos);
    // A schema's own limits go the same way.
    const auto schema = nlohmann::ordered_json::parse(R"({"type": "string", "maxLength": 2000000000})");
    CHECK(compile(json_schema_to_grammar(schema))->rules().size() < 64);
}
