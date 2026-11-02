// GBNF parser and pushdown matcher. See gbnf.h for provenance: the parsing
// and matching below follow llama.cpp src/llama-grammar.cpp (b6500) function
// by function, so upstream fixes port over directly.

#include "gbnf.h"

#include <algorithm>
#include <stdexcept>

namespace luce::common::gbnf {

namespace {

// NOTE: assumes valid UTF-8 (but checks for overrun).
std::pair<uint32_t, const char *> decode_utf8_char(const char * src) {
    static const int lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    const uint8_t first_byte = static_cast<uint8_t>(*src);
    const uint8_t highbits   = first_byte >> 4;
    const int     len        = lookup[highbits];
    const uint8_t mask       = (1 << (8 - len)) - 1;
    uint32_t      value      = first_byte & mask;
    const char *  end        = src + len;  // may overrun!
    const char *  pos        = src + 1;
    for (; pos < end && *pos; pos++) {
        value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
    }
    return std::make_pair(value, pos);
}

bool is_digit_char(char c) {
    return '0' <= c && c <= '9';
}

bool is_word_char(char c) {
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' || is_digit_char(c);
}

std::pair<uint32_t, const char *> parse_hex(const char * src, int size) {
    const char * pos   = src;
    const char * end   = src + size;
    uint32_t     value = 0;
    for (; pos < end && *pos; pos++) {
        value <<= 4;
        const char c = *pos;
        if ('a' <= c && c <= 'f') {
            value += c - 'a' + 10;
        } else if ('A' <= c && c <= 'F') {
            value += c - 'A' + 10;
        } else if ('0' <= c && c <= '9') {
            value += c - '0';
        } else {
            break;
        }
    }
    if (pos != end) {
        throw std::runtime_error("expecting " + std::to_string(size) + " hex chars at " + src);
    }
    return std::make_pair(value, pos);
}

const char * parse_space(const char * src, bool newline_ok) {
    const char * pos = src;
    while (*pos == ' ' || *pos == '\t' || *pos == '#' ||
           (newline_ok && (*pos == '\r' || *pos == '\n'))) {
        if (*pos == '#') {
            while (*pos && *pos != '\r' && *pos != '\n') pos++;
        } else {
            pos++;
        }
    }
    return pos;
}

const char * parse_name(const char * src) {
    const char * pos = src;
    while (is_word_char(*pos)) pos++;
    if (pos == src) {
        throw std::runtime_error(std::string("expecting name at ") + src);
    }
    return pos;
}

const char * parse_int(const char * src) {
    const char * pos = src;
    while (is_digit_char(*pos)) pos++;
    if (pos == src) {
        throw std::runtime_error(std::string("expecting integer at ") + src);
    }
    return pos;
}

std::pair<uint32_t, const char *> parse_char(const char * src) {
    if (*src == '\\') {
        switch (src[1]) {
            case 'x': return parse_hex(src + 2, 2);
            case 'u': return parse_hex(src + 2, 4);
            case 'U': return parse_hex(src + 2, 8);
            case 't': return std::make_pair('\t', src + 2);
            case 'r': return std::make_pair('\r', src + 2);
            case 'n': return std::make_pair('\n', src + 2);
            case '\\':
            case '"':
            case '[':
            case ']':
                return std::make_pair(src[1], src + 2);
            default:
                throw std::runtime_error(std::string("unknown escape at ") + src);
        }
    } else if (*src) {
        return decode_utf8_char(src);
    }
    throw std::runtime_error("unexpected end of input");
}

void print_grammar_char(FILE * file, uint32_t c) {
    if (0x20 <= c && c <= 0x7f) {
        std::fprintf(file, "%c", static_cast<char>(c));
    } else {
        std::fprintf(file, "<U+%04X>", c);
    }
}

bool is_char_element(Element elem) {
    switch (elem.type) {
        case ElementType::CHAR:
        case ElementType::CHAR_NOT:
        case ElementType::CHAR_ALT:
        case ElementType::CHAR_RNG_UPPER:
        case ElementType::CHAR_ANY:
            return true;
        default:
            return false;
    }
}

void print_rule(FILE * file, uint32_t rule_id, const Rule & rule,
                const std::map<uint32_t, std::string> & symbol_id_names) {
    if (rule.empty() || rule.back().type != ElementType::END) {
        throw std::runtime_error(
            "malformed rule, does not end with END: " + std::to_string(rule_id));
    }
    std::fprintf(file, "%s ::= ", symbol_id_names.at(rule_id).c_str());
    for (size_t i = 0, end = rule.size() - 1; i < end; i++) {
        const Element elem = rule[i];
        switch (elem.type) {
            case ElementType::END:
                throw std::runtime_error(
                    "unexpected end of rule: " + std::to_string(rule_id) + "," + std::to_string(i));
            case ElementType::ALT:
                std::fprintf(file, "| ");
                break;
            case ElementType::RULE_REF:
                std::fprintf(file, "%s ", symbol_id_names.at(elem.value).c_str());
                break;
            case ElementType::CHAR:
                std::fprintf(file, "[");
                print_grammar_char(file, elem.value);
                break;
            case ElementType::CHAR_NOT:
                std::fprintf(file, "[^");
                print_grammar_char(file, elem.value);
                break;
            case ElementType::CHAR_RNG_UPPER:
                if (i == 0 || !is_char_element(rule[i - 1])) {
                    throw std::runtime_error(
                        "CHAR_RNG_UPPER without preceding char: " +
                        std::to_string(rule_id) + "," + std::to_string(i));
                }
                std::fprintf(file, "-");
                print_grammar_char(file, elem.value);
                break;
            case ElementType::CHAR_ALT:
                if (i == 0 || !is_char_element(rule[i - 1])) {
                    throw std::runtime_error(
                        "CHAR_ALT without preceding char: " +
                        std::to_string(rule_id) + "," + std::to_string(i));
                }
                print_grammar_char(file, elem.value);
                break;
            case ElementType::CHAR_ANY:
                std::fprintf(file, ".");
                break;
        }
        if (is_char_element(elem)) {
            switch (rule[i + 1].type) {
                case ElementType::CHAR_ALT:
                case ElementType::CHAR_RNG_UPPER:
                case ElementType::CHAR_ANY:
                    break;
                default:
                    std::fprintf(file, "] ");
            }
        }
    }
    std::fprintf(file, "\n");
}

// True when pos ends one of the definitions of a rule.
bool is_end_of_sequence(const Element * pos) {
    return pos->type == ElementType::END || pos->type == ElementType::ALT;
}

// Whether chr satisfies the char range at pos (regular or inverse range),
// and the element after the range. pos must point at a char range element.
std::pair<bool, const Element *> match_char(const Element * pos, uint32_t chr) {
    bool found = false;
    const bool is_positive_char =
        pos->type == ElementType::CHAR || pos->type == ElementType::CHAR_ANY;
    if (!is_positive_char && pos->type != ElementType::CHAR_NOT) {
        throw std::logic_error("gbnf: match_char on a non-char element");
    }
    do {
        if (pos[1].type == ElementType::CHAR_RNG_UPPER) {
            found = found || (pos->value <= chr && chr <= pos[1].value);
            pos += 2;
        } else if (pos->type == ElementType::CHAR_ANY) {
            found = true;
            pos += 1;
        } else {
            found = found || pos->value == chr;
            pos += 1;
        }
    } while (pos->type == ElementType::CHAR_ALT);
    return std::make_pair(found == is_positive_char, pos);
}

// Whether some continuation of the partial UTF-8 sequence could satisfy the
// char range at pos (regular or inverse range).
bool match_partial_char(const Element * pos, PartialUtf8 partial_utf8) {
    const bool is_positive_char =
        pos->type == ElementType::CHAR || pos->type == ElementType::CHAR_ANY;
    const uint32_t partial_value = partial_utf8.value;
    const int      n_remain      = partial_utf8.n_remain;

    // invalid sequence or 7-bit char split across 2 bytes (overlong)
    if (n_remain < 0 || (n_remain == 1 && partial_value < 2)) {
        return false;
    }

    // range of possible code points this partial UTF-8 sequence could complete to
    uint32_t low  = partial_value << (n_remain * 6);
    uint32_t high = low | ((1 << (n_remain * 6)) - 1);
    if (low == 0) {
        if (n_remain == 2) {
            low = 1 << 11;
        } else if (n_remain == 3) {
            low = 1 << 16;
        }
    }

    // ED and F4 are the only lead bytes that need their max values
    // limited to exclude values UTF-8 doesn't allow
    if (n_remain == 2 && partial_value == 0xD) {
        high = 0xD7FF;
    } else if (n_remain == 3 && partial_value == 0x4) {
        high = 0x10FFFF;
    }

    do {
        if (pos[1].type == ElementType::CHAR_RNG_UPPER) {
            if (pos->value <= high && low <= pos[1].value) return is_positive_char;
            pos += 2;
        } else if (pos->type == ElementType::CHAR_ANY) {
            return true;
        } else {
            if (low <= pos->value && pos->value <= high) return is_positive_char;
            pos += 1;
        }
    } while (pos->type == ElementType::CHAR_ALT);

    return !is_positive_char;
}

// Expand a pushdown stack into the stacks that end at a character range
// (terminal element), or the empty stack when the sentence can end there.
void advance_stack(const Rules & rules, const Stack & stack, Stacks & new_stacks) {
    if (stack.empty()) {
        if (std::find(new_stacks.begin(), new_stacks.end(), stack) == new_stacks.end()) {
            new_stacks.emplace_back(stack);
        }
        return;
    }

    const Element * pos = stack.back();
    switch (pos->type) {
        case ElementType::RULE_REF: {
            const size_t    rule_id = static_cast<size_t>(pos->value);
            const Element * subpos  = rules[rule_id].data();
            do {
                // the stack without the top (pos)
                Stack new_stack(stack.begin(), stack.end() - 1);
                if (!is_end_of_sequence(pos + 1)) {
                    // the rule ref is followed by another element
                    new_stack.push_back(pos + 1);
                }
                if (!is_end_of_sequence(subpos)) {
                    // the alternate is nonempty
                    new_stack.push_back(subpos);
                }
                advance_stack(rules, new_stack, new_stacks);
                while (!is_end_of_sequence(subpos)) subpos++;
                if (subpos->type == ElementType::ALT) {
                    subpos++;  // another alternate of this rule
                } else {
                    break;
                }
            } while (true);
            break;
        }
        case ElementType::CHAR:
        case ElementType::CHAR_NOT:
        case ElementType::CHAR_ANY:
            if (std::find(new_stacks.begin(), new_stacks.end(), stack) == new_stacks.end()) {
                new_stacks.emplace_back(stack);
            }
            break;
        default:
            // END/ALT or the middle of a char range: a stack never rests there.
            throw std::logic_error("gbnf: stack rests on a non-terminal element");
    }
}

std::vector<Candidate> reject_for_stack(const Rules & rules, const Stack & stack,
                                        const std::vector<Candidate> & candidates);

std::vector<Candidate> reject_for_stacks(const Rules & rules, const Stacks & stacks,
                                         const std::vector<Candidate> & candidates) {
    if (candidates.empty()) return {};
    if (stacks.empty()) return candidates;
    auto rejects = reject_for_stack(rules, stacks.front(), candidates);
    for (size_t i = 1, size = stacks.size(); i < size; ++i) {
        rejects = reject_for_stack(rules, stacks[i], rejects);
    }
    return rejects;
}

std::vector<Candidate> reject_for_stack(const Rules & rules, const Stack & stack,
                                        const std::vector<Candidate> & candidates) {
    std::vector<Candidate> rejects;
    rejects.reserve(candidates.size());

    if (stack.empty()) {
        for (const auto & tok : candidates) {
            if (*tok.code_points != 0 || tok.partial.n_remain != 0) {
                rejects.push_back(tok);
            }
        }
        return rejects;
    }

    const Element * stack_pos = stack.back();

    std::vector<Candidate> next_candidates;
    next_candidates.reserve(candidates.size());
    for (const auto & tok : candidates) {
        if (*tok.code_points == 0) {
            // End of the token's full code points: reject iff it ended in a
            // partial sequence that cannot satisfy this position.
            if (tok.partial.n_remain != 0 && !match_partial_char(stack_pos, tok.partial)) {
                rejects.push_back(tok);
            }
        } else if (match_char(stack_pos, *tok.code_points).first) {
            next_candidates.push_back({ tok.index, tok.code_points + 1, tok.partial });
        } else {
            rejects.push_back(tok);
        }
    }

    const Element * stack_pos_after = match_char(stack_pos, 0).second;

    // the top of the stack moves to the next element, if any
    Stack stack_after(stack.begin(), stack.end() - 1);
    if (!is_end_of_sequence(stack_pos_after)) {
        stack_after.push_back(stack_pos_after);
    }
    Stacks next_stacks;
    advance_stack(rules, stack_after, next_stacks);

    const auto next_rejects = reject_for_stacks(rules, next_stacks, next_candidates);
    for (const auto & tok : next_rejects) {
        rejects.push_back({ tok.index, tok.code_points - 1, tok.partial });
    }
    return rejects;
}

bool detect_left_recursion(const Rules & rules, size_t rule_index,
                           std::vector<bool> & rules_visited,
                           std::vector<bool> & rules_in_progress,
                           std::vector<bool> & rules_may_be_empty) {
    if (rules_in_progress[rule_index]) return true;
    rules_in_progress[rule_index] = true;

    const Rule & rule = rules[rule_index];

    // Can the rule produce the empty string?
    bool at_rule_start = true;
    for (size_t i = 0; i < rule.size(); i++) {
        if (is_end_of_sequence(&rule[i])) {
            if (at_rule_start) {
                rules_may_be_empty[rule_index] = true;
                break;
            }
            at_rule_start = true;
        } else {
            at_rule_start = false;
        }
    }

    // Recurse into leftmost nonterminals (or next-leftmost while the previous
    // nonterminal may be empty).
    bool recurse_into_nonterminal = true;
    for (size_t i = 0; i < rule.size(); i++) {
        if (rule[i].type == ElementType::RULE_REF && recurse_into_nonterminal) {
            if (detect_left_recursion(rules, (size_t) rule[i].value, rules_visited,
                                      rules_in_progress, rules_may_be_empty)) {
                return true;
            }
            if (!rules_may_be_empty[(size_t) rule[i].value]) {
                recurse_into_nonterminal = false;
            }
        } else if (is_end_of_sequence(&rule[i])) {
            recurse_into_nonterminal = true;
        } else {
            recurse_into_nonterminal = false;
        }
    }

    rules_in_progress[rule_index] = false;
    rules_visited[rule_index] = true;
    return false;
}

bool is_continuation_byte(uint8_t byte, uint32_t value, int n_remain) {
    if ((byte >> 6) != 2) return false;
    // For the value to be < 0x10 with two bytes remaining the lead must encode 3-bytes.
    if (n_remain == 2 && value < 0x10) {
        if (value == 0x0) return byte >= 0xA0;  // for E0, a second byte below 0xA0 would make an overlong encoding
        if (value == 0xD) return byte <= 0x9F;  // for ED, a second byte above 0x9F is invalid UTF-8
    } else if (n_remain == 3) {
        if (value == 0x0) return byte >= 0x90;  // for F0, a second byte below 0x90 would make an overlong encoding
        if (value == 0x4) return byte <= 0x8F;  // for F4, a second byte above 0x8F is invalid UTF-8
    }
    return true;
}

}  // namespace

std::pair<std::vector<uint32_t>, PartialUtf8> decode_utf8(
        const std::string & bytes, PartialUtf8 partial_start) {
    static const int lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4 };
    const char * pos = bytes.c_str();
    std::vector<uint32_t> code_points;
    // common english strings have the same number of codepoints and bytes;
    // + 1 for the terminating 0
    code_points.reserve(bytes.size() + 1);
    uint32_t value    = partial_start.value;
    int      n_remain = partial_start.n_remain;

    const auto invalid = [&code_points]() {
        code_points.clear();
        code_points.push_back(0);
        return std::make_pair(std::move(code_points), PartialUtf8{ 0, -1 });
    };
    if (n_remain < 0) return invalid();

    // continue the previous decode, if any
    while (*pos != 0 && n_remain > 0) {
        const uint8_t next_byte = static_cast<uint8_t>(*pos);
        if (!is_continuation_byte(next_byte, value, n_remain)) {
            // invalid sequence, abort
            return invalid();
        }
        value = (value << 6) + (next_byte & 0x3F);
        ++pos;
        --n_remain;
    }

    if (partial_start.n_remain > 0 && n_remain == 0) {
        code_points.push_back(value);
    }

    // decode the following sequences, which may end in an incomplete one
    while (*pos != 0) {
        const uint8_t first_byte = static_cast<uint8_t>(*pos);
        const uint8_t highbits   = first_byte >> 4;
        n_remain = lookup[highbits] - 1;

        // ASCII and C2-F4 are the only valid leading bytes.
        if (n_remain < 0 || first_byte == 0xC0 || first_byte == 0xC1 || first_byte > 0xF4) {
            return invalid();
        }

        const uint8_t mask = (1 << (7 - n_remain)) - 1;
        value = first_byte & mask;

        ++pos;
        while (*pos != 0 && n_remain > 0) {
            const uint8_t next_byte = static_cast<uint8_t>(*pos);
            if (!is_continuation_byte(next_byte, value, n_remain)) return invalid();
            value = (value << 6) + (next_byte & 0x3F);
            ++pos;
            --n_remain;
        }
        if (n_remain == 0) {
            code_points.push_back(value);
        }
    }
    code_points.push_back(0);

    return std::make_pair(std::move(code_points), PartialUtf8{ value, n_remain });
}

uint32_t Parser::get_symbol_id(const char * src, size_t len) {
    const uint32_t next_id = static_cast<uint32_t>(symbol_ids.size());
    const auto result = symbol_ids.emplace(std::string(src, len), next_id);
    return result.first->second;
}

uint32_t Parser::generate_symbol_id(const std::string & base_name) {
    const uint32_t next_id = static_cast<uint32_t>(symbol_ids.size());
    symbol_ids[base_name + '_' + std::to_string(next_id)] = next_id;
    return next_id;
}

void Parser::add_rule(uint32_t rule_id, const Rule & rule) {
    if (rules.size() <= rule_id) rules.resize(rule_id + 1);
    rules[rule_id] = rule;
}

const char * Parser::parse_alternates(const char * src, const std::string & rule_name,
                                      uint32_t rule_id, bool is_nested) {
    Rule rule;
    const char * pos = parse_sequence(src, rule_name, rule, is_nested);
    while (*pos == '|') {
        rule.push_back({ ElementType::ALT, 0 });
        pos = parse_space(pos + 1, true);
        pos = parse_sequence(pos, rule_name, rule, is_nested);
    }
    rule.push_back({ ElementType::END, 0 });
    add_rule(rule_id, rule);
    return pos;
}

const char * Parser::parse_sequence(const char * src, const std::string & rule_name,
                                    Rule & rule, bool is_nested) {
    size_t last_sym_start = rule.size();
    const char * pos = src;

    // S{m,n} --> S S S (m times) S'(n-m), S'(x) ::= S S'(x-1) | ... S'(1) ::= S |
    // S{m,}  --> S S S (m times) S',      S' ::= S S' |
    // S* = S{0,}, S+ = S{1,}, S? = S{0,1}
    auto handle_repetitions = [&](int min_times, int max_times) {
        if (last_sym_start == rule.size()) {
            throw std::runtime_error(std::string("expecting preceding item to */+/?/{ at ") + pos);
        }
        Rule prev_rule(rule.begin() + last_sym_start, rule.end());
        if (min_times == 0) {
            rule.resize(last_sym_start);
        } else {
            // repeat the previous elements (min_times - 1) times
            for (int i = 1; i < min_times; i++) {
                rule.insert(rule.end(), prev_rule.begin(), prev_rule.end());
            }
        }

        uint32_t last_rec_rule_id = 0;
        const int n_opt = max_times < 0 ? 1 : max_times - min_times;

        Rule rec_rule(prev_rule);
        for (int i = 0; i < n_opt; i++) {
            rec_rule.resize(prev_rule.size());
            const uint32_t rec_rule_id = generate_symbol_id(rule_name);
            if (i > 0 || max_times < 0) {
                rec_rule.push_back({ ElementType::RULE_REF,
                                     max_times < 0 ? rec_rule_id : last_rec_rule_id });
            }
            rec_rule.push_back({ ElementType::ALT, 0 });
            rec_rule.push_back({ ElementType::END, 0 });
            add_rule(rec_rule_id, rec_rule);
            last_rec_rule_id = rec_rule_id;
        }
        if (n_opt > 0) {
            rule.push_back({ ElementType::RULE_REF, last_rec_rule_id });
        }
    };

    while (*pos) {
        if (*pos == '"') {  // literal string
            pos++;
            last_sym_start = rule.size();
            while (*pos != '"') {
                if (!*pos) throw std::runtime_error("unexpected end of input");
                const auto char_pair = parse_char(pos);
                pos = char_pair.second;
                rule.push_back({ ElementType::CHAR, char_pair.first });
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '[') {  // char range(s)
            pos++;
            ElementType start_type = ElementType::CHAR;
            if (*pos == '^') {
                pos++;
                start_type = ElementType::CHAR_NOT;
            }
            last_sym_start = rule.size();
            while (*pos != ']') {
                if (!*pos) throw std::runtime_error("unexpected end of input");
                const auto char_pair = parse_char(pos);
                pos = char_pair.second;
                const ElementType type = last_sym_start < rule.size()
                    ? ElementType::CHAR_ALT
                    : start_type;
                rule.push_back({ type, char_pair.first });
                if (pos[0] == '-' && pos[1] != ']') {
                    if (!pos[1]) throw std::runtime_error("unexpected end of input");
                    const auto endchar_pair = parse_char(pos + 1);
                    pos = endchar_pair.second;
                    rule.push_back({ ElementType::CHAR_RNG_UPPER, endchar_pair.first });
                }
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (is_word_char(*pos)) {  // rule reference
            const char * name_end = parse_name(pos);
            const uint32_t ref_rule_id = get_symbol_id(pos, name_end - pos);
            pos = parse_space(name_end, is_nested);
            last_sym_start = rule.size();
            rule.push_back({ ElementType::RULE_REF, ref_rule_id });
        } else if (*pos == '(') {  // grouping
            // nested alternates become a synthesized rule
            pos = parse_space(pos + 1, true);
            const uint32_t sub_rule_id = generate_symbol_id(rule_name);
            pos = parse_alternates(pos, rule_name, sub_rule_id, true);
            last_sym_start = rule.size();
            rule.push_back({ ElementType::RULE_REF, sub_rule_id });
            if (*pos != ')') throw std::runtime_error(std::string("expecting ')' at ") + pos);
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '.') {  // any char
            last_sym_start = rule.size();
            rule.push_back({ ElementType::CHAR_ANY, 0 });
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '*') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(0, -1);
        } else if (*pos == '+') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(1, -1);
        } else if (*pos == '?') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(0, 1);
        } else if (*pos == '{') {
            pos = parse_space(pos + 1, is_nested);
            if (!is_digit_char(*pos)) throw std::runtime_error(std::string("expecting an int at ") + pos);
            const char * int_end = parse_int(pos);
            const int min_times = std::stoul(std::string(pos, int_end - pos));
            pos = parse_space(int_end, is_nested);

            int max_times = -1;
            if (*pos == '}') {
                max_times = min_times;
                pos = parse_space(pos + 1, is_nested);
            } else if (*pos == ',') {
                pos = parse_space(pos + 1, is_nested);
                if (is_digit_char(*pos)) {
                    const char * max_end = parse_int(pos);
                    max_times = std::stoul(std::string(pos, max_end - pos));
                    pos = parse_space(max_end, is_nested);
                }
                if (*pos != '}') throw std::runtime_error(std::string("expecting '}' at ") + pos);
                pos = parse_space(pos + 1, is_nested);
            } else {
                throw std::runtime_error(std::string("expecting ',' at ") + pos);
            }
            handle_repetitions(min_times, max_times);
        } else {
            break;
        }
    }
    return pos;
}

const char * Parser::parse_rule(const char * src) {
    const char * name_end = parse_name(src);
    const char * pos      = parse_space(name_end, false);
    const size_t name_len = name_end - src;
    const uint32_t rule_id = get_symbol_id(src, name_len);
    const std::string name(src, name_len);

    if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) {
        throw std::runtime_error(std::string("expecting ::= at ") + pos);
    }
    pos = parse_space(pos + 3, true);
    pos = parse_alternates(pos, name, rule_id, false);

    if (*pos == '\r') {
        pos += pos[1] == '\n' ? 2 : 1;
    } else if (*pos == '\n') {
        pos++;
    } else if (*pos) {
        throw std::runtime_error(std::string("expecting newline or end at ") + pos);
    }
    return parse_space(pos, true);
}

bool Parser::parse(const char * src) {
    try {
        const char * pos = parse_space(src, true);
        while (*pos) pos = parse_rule(pos);
        // every referenced rule must be defined
        for (const auto & rule : rules) {
            if (rule.empty()) throw std::runtime_error("Undefined rule");
            for (const auto & elem : rule) {
                if (elem.type != ElementType::RULE_REF) continue;
                if (elem.value >= rules.size() || rules[elem.value].empty()) {
                    for (const auto & kv : symbol_ids) {
                        if (kv.second == elem.value) {
                            throw std::runtime_error("Undefined rule identifier '" + kv.first + "'");
                        }
                    }
                    throw std::runtime_error("Undefined rule");
                }
            }
        }
    } catch (const std::exception & err) {
        error = err.what();
        rules.clear();
        return false;
    }
    return true;
}

void Parser::print(FILE * file) const {
    try {
        std::map<uint32_t, std::string> symbol_id_names;
        for (const auto & kv : symbol_ids) symbol_id_names[kv.second] = kv.first;
        for (size_t i = 0, end = rules.size(); i < end; i++) {
            print_rule(file, uint32_t(i), rules[i], symbol_id_names);
        }
    } catch (const std::exception & err) {
        std::fprintf(stderr, "\ngbnf: error printing grammar: %s\n", err.what());
    }
}

std::unique_ptr<Matcher> Matcher::compile(const std::string & text,
                                          const std::string & root,
                                          std::string & error) {
    Parser parser;
    if (!parser.parse(text.c_str()) || parser.rules.empty()) {
        error = parser.error.empty() ? "empty grammar" : parser.error;
        return nullptr;
    }
    const auto root_it = parser.symbol_ids.find(root);
    if (root_it == parser.symbol_ids.end()) {
        error = "grammar does not contain a '" + root + "' rule";
        return nullptr;
    }

    const size_t n_rules = parser.rules.size();
    std::vector<bool> rules_visited(n_rules);
    std::vector<bool> rules_in_progress(n_rules);
    std::vector<bool> rules_may_be_empty(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        if (rules_visited[i]) continue;
        if (detect_left_recursion(parser.rules, i, rules_visited, rules_in_progress,
                                  rules_may_be_empty)) {
            error = "unsupported grammar, left recursion at rule " + std::to_string(i);
            return nullptr;
        }
    }

    std::unique_ptr<Matcher> matcher(new Matcher());
    auto rules = std::make_shared<Rules>(std::move(parser.rules));
    matcher->symbol_ids_ = std::move(parser.symbol_ids);

    // the initial stacks: one per alternate of the root rule
    const Element * pos = (*rules)[root_it->second].data();
    do {
        Stack stack;
        if (!is_end_of_sequence(pos)) stack.push_back(pos);
        advance_stack(*rules, stack, matcher->stacks_);
        while (!is_end_of_sequence(pos)) pos++;
        if (pos->type == ElementType::ALT) {
            pos++;
        } else {
            break;
        }
    } while (true);

    // The stacks point into *rules: shared and never mutated, so copies of
    // the matcher stay valid.
    matcher->rules_ = std::move(rules);
    return matcher;
}

bool Matcher::can_end() const {
    if (partial_.n_remain != 0) return false;
    for (const auto & stack : stacks_) {
        if (stack.empty()) return true;
    }
    return false;
}

int Matcher::rule_id(const std::string & name) const {
    const auto it = symbol_ids_.find(name);
    return it == symbol_ids_.end() ? -1 : (int) it->second;
}

void Matcher::accept_bytes(const std::string & bytes) {
    const auto decoded = decode_utf8(bytes, partial_);
    if (decoded.second.n_remain < 0) {
        stacks_.clear();
        return;
    }
    const auto & code_points = decoded.first;
    for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
        Stacks stacks_new;
        stacks_new.reserve(stacks_.size());
        for (const auto & stack : stacks_) {
            if (stack.empty()) continue;
            const auto match = match_char(stack.back(), *it);
            if (!match.first) continue;
            // the top of the stack moves to the next element, if any
            Stack new_stack(stack.begin(), stack.end() - 1);
            if (!is_end_of_sequence(match.second)) new_stack.push_back(match.second);
            advance_stack(*rules_, new_stack, stacks_new);
        }
        stacks_ = std::move(stacks_new);
        if (stacks_.empty()) return;
    }
    partial_ = decoded.second;
    if (partial_.n_remain > 0) {
        // The piece ends inside a character: some stack must accept a
        // continuation of it, or the text is already off the grammar.
        const bool reachable = std::any_of(stacks_.begin(), stacks_.end(),
            [&](const Stack & stack) {
                return !stack.empty() && match_partial_char(stack.back(), partial_);
            });
        if (!reachable) stacks_.clear();
    }
}

std::vector<Candidate> Matcher::reject(const std::vector<Candidate> & candidates) const {
    return reject_for_stacks(*rules_, stacks_, candidates);
}

}  // namespace luce::common::gbnf
