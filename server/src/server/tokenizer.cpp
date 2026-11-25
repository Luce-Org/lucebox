// BPE tokenizer implementation.
//
// GGUF loading uses ggml's gguf_init_from_file API (already vendored).
// Pre-tokenization is a hand-coded state machine matching the Qwen3/3.5
// regex pattern without pulling in a regex library.

#include "tokenizer.h"

#include "gguf.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <utility>

namespace luce::common {

// ─── Unicode helpers ────────────────────────────────────────────────────

static int utf8_len(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;  // invalid — advance one byte
}

static uint32_t utf8_decode(const char * s, size_t remaining, int * len) {
    if (remaining == 0) { *len = 0; return 0xFFFD; }
    uint8_t c = (uint8_t)s[0];
    if (c < 0x80) { *len = 1; return c; }
    if ((c & 0xE0) == 0xC0 && remaining >= 2) {
        *len = 2;
        return ((uint32_t)(c & 0x1F) << 6) |
               ((uint32_t)((uint8_t)s[1]) & 0x3F);
    }
    if ((c & 0xF0) == 0xE0 && remaining >= 3) {
        *len = 3;
        return ((uint32_t)(c & 0x0F) << 12) |
               (((uint32_t)((uint8_t)s[1]) & 0x3F) << 6) |
               ((uint32_t)((uint8_t)s[2]) & 0x3F);
    }
    if ((c & 0xF8) == 0xF0 && remaining >= 4) {
        *len = 4;
        return ((uint32_t)(c & 0x07) << 18) |
               (((uint32_t)((uint8_t)s[1]) & 0x3F) << 12) |
               (((uint32_t)((uint8_t)s[2]) & 0x3F) << 6) |
               ((uint32_t)((uint8_t)s[3]) & 0x3F);
    }
    *len = 1;
    return 0xFFFD;
}

// Unicode character property tests (simplified — covers the ranges needed
// for Qwen3/3.5 tokenizer pre-split).

static bool is_letter(uint32_t cp) {
    // ASCII letters
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    // Common Latin-1 Supplement letters
    if (cp >= 0xC0 && cp <= 0xFF && cp != 0xD7 && cp != 0xF7) return true;
    // Latin Extended-A/B
    if (cp >= 0x100 && cp <= 0x24F) return true;
    // Greek and Coptic
    if (cp >= 0x370 && cp <= 0x3FF) return true;
    // Cyrillic
    if (cp >= 0x400 && cp <= 0x4FF) return true;
    // Arabic
    if (cp >= 0x600 && cp <= 0x6FF) return true;
    // Devanagari
    if (cp >= 0x900 && cp <= 0x97F) return true;
    // CJK Unified Ideographs
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    // CJK Extension A
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;
    // Hangul Syllables
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;
    // Hiragana
    if (cp >= 0x3040 && cp <= 0x309F) return true;
    // Katakana
    if (cp >= 0x30A0 && cp <= 0x30FF) return true;
    // CJK Compatibility Ideographs
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;
    // Fullwidth Latin
    if (cp >= 0xFF21 && cp <= 0xFF3A) return true;
    if (cp >= 0xFF41 && cp <= 0xFF5A) return true;
    // Thai
    if (cp >= 0x0E01 && cp <= 0x0E3A) return true;
    // Hebrew
    if (cp >= 0x05D0 && cp <= 0x05EA) return true;
    // General: Other Letter (Lo) ranges for common scripts
    if (cp >= 0x1100 && cp <= 0x11FF) return true;  // Hangul Jamo
    if (cp >= 0x2E80 && cp <= 0x2EFF) return true;  // CJK Radicals
    return false;
}

static bool is_digit(uint32_t cp) {
    if (cp >= '0' && cp <= '9') return true;              // ASCII
    if (cp >= 0xFF10 && cp <= 0xFF19) return true;        // Fullwidth digits
    if (cp >= 0x0660 && cp <= 0x0669) return true;        // Arabic-Indic digits
    if (cp >= 0x06F0 && cp <= 0x06F9) return true;        // Extended Arabic-Indic
    if (cp >= 0x0966 && cp <= 0x096F) return true;        // Devanagari digits
    if (cp >= 0x09E6 && cp <= 0x09EF) return true;        // Bengali digits
    if (cp >= 0x0E50 && cp <= 0x0E59) return true;        // Thai digits
    return false;
}

static bool is_mark(uint32_t cp) {
    // Unicode Mark category (Mn, Mc, Me) — combining marks.
    // Common ranges for diacritics used in many languages.
    if (cp >= 0x0300 && cp <= 0x036F) return true;   // Combining Diacritical Marks
    if (cp >= 0x0591 && cp <= 0x05BD) return true;   // Hebrew accents
    if (cp >= 0x0610 && cp <= 0x061A) return true;   // Arabic
    if (cp >= 0x064B && cp <= 0x065F) return true;   // Arabic
    if (cp >= 0x0900 && cp <= 0x0903) return true;   // Devanagari
    if (cp >= 0x093A && cp <= 0x094F) return true;   // Devanagari
    if (cp == 0x0E31) return true;   // Thai
    if (cp >= 0x0E34 && cp <= 0x0E3A) return true;   // Thai
    if (cp >= 0xFE20 && cp <= 0xFE2F) return true;   // Combining Half Marks
    if (cp >= 0x20D0 && cp <= 0x20FF) return true;   // Combining for Symbols
    return false;
}

static bool is_whitespace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == '\f' || cp == '\v' ||
           cp == 0x00A0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

static bool is_newline(uint32_t cp) {
    return cp == '\n' || cp == '\r';
}

// ─── Pre-tokenizer ─────────────────────────────────────────────────────
// Matches the Qwen/DeepSeek pattern:
//   (?:'[sStTmMdD]|...) |
//   [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ |
//   \p{N}{1,3} |
//   ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]* |
//   \s*[\r\n]+ |
//   \s+(?!\S) |
//   \s+

std::vector<std::string> Tokenizer::pre_tokenize(const std::string & text) const {
    std::vector<std::string> pieces;
    const char * s = text.c_str();
    const size_t len = text.size();
    size_t pos = 0;

    auto peek_cp = [&](size_t p, int * cplen) -> uint32_t {
        if (p >= len) { *cplen = 0; return 0; }
        return utf8_decode(s + p, len - p, cplen);
    };

    while (pos < len) {
        size_t start = pos;
        int cplen = 0;
        uint32_t cp = peek_cp(pos, &cplen);

        // Pattern 1: English contractions 's 't 're 've 'm 'll 'd
        if (cp == '\'') {
            size_t save = pos;
            pos++;
            bool matched = false;
            if (pos < len) {
                char c = s[pos] | 0x20;  // lowercase
                if (c == 's' || c == 't' || c == 'm' || c == 'd') {
                    pos++;
                    matched = true;
                } else if (c == 'r' && pos + 1 < len && (s[pos+1] | 0x20) == 'e') {
                    pos += 2;
                    matched = true;
                } else if (c == 'v' && pos + 1 < len && (s[pos+1] | 0x20) == 'e') {
                    pos += 2;
                    matched = true;
                } else if (c == 'l' && pos + 1 < len && (s[pos+1] | 0x20) == 'l') {
                    pos += 2;
                    matched = true;
                }
            }
            if (matched) {
                pieces.push_back(text.substr(start, pos - start));
                continue;
            }
            pos = save;  // reset, try other patterns
        }

        // Pattern 2: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        {
            size_t p = pos;
            int cl = 0;
            uint32_t c = peek_cp(p, &cl);
            // Optional leading non-letter, non-digit, non-newline char
            if (cl > 0 && !is_newline(c) && !is_letter(c) && !is_digit(c)) {
                p += cl;
                c = peek_cp(p, &cl);
            }
            // One or more letter/mark chars
            if (cl > 0 && (is_letter(c) || is_mark(c))) {
                while (cl > 0 && (is_letter(c) || is_mark(c))) {
                    p += cl;
                    c = peek_cp(p, &cl);
                }
                pieces.push_back(text.substr(pos, p - pos));
                pos = p;
                continue;
            }
        }

        // Pattern 3: \p{N}{1,3}  (up to 3 digits)
        if (is_digit(cp)) {
            int digit_count = 0;
            while (digit_count < 3 && is_digit(cp)) {
                pos += cplen;
                digit_count++;
                cp = peek_cp(pos, &cplen);
            }
            pieces.push_back(text.substr(start, pos - start));
            continue;
        }

        // Pattern 4: ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        {
            size_t p = pos;
            int cl = 0;
            uint32_t c = peek_cp(p, &cl);
            // Optional leading space
            if (c == ' ') {
                p += cl;
                c = peek_cp(p, &cl);
            }
            // One or more non-whitespace, non-letter, non-mark, non-digit
            size_t punc_start = p;
            while (cl > 0 && !is_whitespace(c) && !is_letter(c) &&
                   !is_mark(c) && !is_digit(c)) {
                p += cl;
                c = peek_cp(p, &cl);
            }
            if (p > punc_start) {
                // Trailing newlines
                while (cl > 0 && is_newline(c)) {
                    p += cl;
                    c = peek_cp(p, &cl);
                }
                pieces.push_back(text.substr(pos, p - pos));
                pos = p;
                continue;
            }
        }

        // Pattern 5: \s*[\r\n]+ — the greedy \s* backtracks to the last
        // newline of the whitespace run, so "\n \n" is one piece.
        if (is_whitespace(cp)) {
            size_t p = pos, nl_end = pos;
            int cl = 0;
            uint32_t c = peek_cp(p, &cl);
            while (cl > 0 && is_whitespace(c)) {
                p += cl;
                if (is_newline(c)) nl_end = p;
                c = peek_cp(p, &cl);
            }
            if (nl_end > pos) {
                pieces.push_back(text.substr(pos, nl_end - pos));
                pos = nl_end;
                continue;
            }
            // Pattern 6: \s+(?!\S) — whitespace NOT followed by non-whitespace
            // Pattern 7: \s+ — general whitespace (fallback)
            // Pattern 6 uses backtracking: greedily match all whitespace,
            // then check if followed by \S. If so, backtrack by 1 char so
            // the last space can be consumed by Pattern 2's optional leader.
            p = pos;
            c = peek_cp(p, &cl);
            size_t prev_p = pos;  // position before last whitespace char
            while (cl > 0 && is_whitespace(c)) {
                prev_p = p;
                p += cl;
                c = peek_cp(p, &cl);
            }
            {
                bool followed_by_non_ws = (p < len && !is_whitespace(c));
                if (!followed_by_non_ws) {
                    // Pattern 6: at end or before more whitespace — match all
                    pieces.push_back(text.substr(pos, p - pos));
                    pos = p;
                } else if (prev_p > pos) {
                    // Pattern 6 backtrack: leave last space for Pattern 2
                    pieces.push_back(text.substr(pos, prev_p - pos));
                    pos = prev_p;
                } else {
                    // Single space before non-ws: Pattern 6 fails, Pattern 7
                    pieces.push_back(text.substr(pos, p - pos));
                    pos = p;
                }
            }
            continue;
        }

        // Fallback: single character (shouldn't normally hit this)
        pos += cplen > 0 ? cplen : 1;
        pieces.push_back(text.substr(start, pos - start));
    }

    return pieces;
}

// The CJK class of DeepSeek's second split, [一-龥぀-ゟ゠-ヿ]: CJK Unified
// Ideographs U+4E00..U+9FA5, Hiragana and Katakana.
static bool is_deepseek_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FA5) || (cp >= 0x3040 && cp <= 0x30FF);
}

// \p{N}, the Unicode Number category (Nd, Nl, No), for DeepSeek's first split:
// its 144 ranges in Unicode 16.0 (the code points whose Python
// unicodedata.category() starts with "N"), the same 1,911 code points Hugging
// Face tokenizers match with \p{N}. is_digit() covers only some decimal digits.
static constexpr uint32_t kUnicodeNumberRanges[][2] = {
    {0x0030, 0x0039}, {0x00B2, 0x00B3}, {0x00B9, 0x00B9}, {0x00BC, 0x00BE}, {0x0660, 0x0669},
    {0x06F0, 0x06F9}, {0x07C0, 0x07C9}, {0x0966, 0x096F}, {0x09E6, 0x09EF}, {0x09F4, 0x09F9},
    {0x0A66, 0x0A6F}, {0x0AE6, 0x0AEF}, {0x0B66, 0x0B6F}, {0x0B72, 0x0B77}, {0x0BE6, 0x0BF2},
    {0x0C66, 0x0C6F}, {0x0C78, 0x0C7E}, {0x0CE6, 0x0CEF}, {0x0D58, 0x0D5E}, {0x0D66, 0x0D78},
    {0x0DE6, 0x0DEF}, {0x0E50, 0x0E59}, {0x0ED0, 0x0ED9}, {0x0F20, 0x0F33}, {0x1040, 0x1049},
    {0x1090, 0x1099}, {0x1369, 0x137C}, {0x16EE, 0x16F0}, {0x17E0, 0x17E9}, {0x17F0, 0x17F9},
    {0x1810, 0x1819}, {0x1946, 0x194F}, {0x19D0, 0x19DA}, {0x1A80, 0x1A89}, {0x1A90, 0x1A99},
    {0x1B50, 0x1B59}, {0x1BB0, 0x1BB9}, {0x1C40, 0x1C49}, {0x1C50, 0x1C59}, {0x2070, 0x2070},
    {0x2074, 0x2079}, {0x2080, 0x2089}, {0x2150, 0x2182}, {0x2185, 0x2189}, {0x2460, 0x249B},
    {0x24EA, 0x24FF}, {0x2776, 0x2793}, {0x2CFD, 0x2CFD}, {0x3007, 0x3007}, {0x3021, 0x3029},
    {0x3038, 0x303A}, {0x3192, 0x3195}, {0x3220, 0x3229}, {0x3248, 0x324F}, {0x3251, 0x325F},
    {0x3280, 0x3289}, {0x32B1, 0x32BF}, {0xA620, 0xA629}, {0xA6E6, 0xA6EF}, {0xA830, 0xA835},
    {0xA8D0, 0xA8D9}, {0xA900, 0xA909}, {0xA9D0, 0xA9D9}, {0xA9F0, 0xA9F9}, {0xAA50, 0xAA59},
    {0xABF0, 0xABF9}, {0xFF10, 0xFF19}, {0x10107, 0x10133}, {0x10140, 0x10178}, {0x1018A, 0x1018B},
    {0x102E1, 0x102FB}, {0x10320, 0x10323}, {0x10341, 0x10341}, {0x1034A, 0x1034A},
    {0x103D1, 0x103D5}, {0x104A0, 0x104A9}, {0x10858, 0x1085F}, {0x10879, 0x1087F},
    {0x108A7, 0x108AF}, {0x108FB, 0x108FF}, {0x10916, 0x1091B}, {0x109BC, 0x109BD},
    {0x109C0, 0x109CF}, {0x109D2, 0x109FF}, {0x10A40, 0x10A48}, {0x10A7D, 0x10A7E},
    {0x10A9D, 0x10A9F}, {0x10AEB, 0x10AEF}, {0x10B58, 0x10B5F}, {0x10B78, 0x10B7F},
    {0x10BA9, 0x10BAF}, {0x10CFA, 0x10CFF}, {0x10D30, 0x10D39}, {0x10D40, 0x10D49},
    {0x10E60, 0x10E7E}, {0x10F1D, 0x10F26}, {0x10F51, 0x10F54}, {0x10FC5, 0x10FCB},
    {0x11052, 0x1106F}, {0x110F0, 0x110F9}, {0x11136, 0x1113F}, {0x111D0, 0x111D9},
    {0x111E1, 0x111F4}, {0x112F0, 0x112F9}, {0x11450, 0x11459}, {0x114D0, 0x114D9},
    {0x11650, 0x11659}, {0x116C0, 0x116C9}, {0x116D0, 0x116E3}, {0x11730, 0x1173B},
    {0x118E0, 0x118F2}, {0x11950, 0x11959}, {0x11BF0, 0x11BF9}, {0x11C50, 0x11C6C},
    {0x11D50, 0x11D59}, {0x11DA0, 0x11DA9}, {0x11F50, 0x11F59}, {0x11FC0, 0x11FD4},
    {0x12400, 0x1246E}, {0x16130, 0x16139}, {0x16A60, 0x16A69}, {0x16AC0, 0x16AC9},
    {0x16B50, 0x16B59}, {0x16B5B, 0x16B61}, {0x16D70, 0x16D79}, {0x16E80, 0x16E96},
    {0x1CCF0, 0x1CCF9}, {0x1D2C0, 0x1D2D3}, {0x1D2E0, 0x1D2F3}, {0x1D360, 0x1D378},
    {0x1D7CE, 0x1D7FF}, {0x1E140, 0x1E149}, {0x1E2F0, 0x1E2F9}, {0x1E4F0, 0x1E4F9},
    {0x1E5F1, 0x1E5FA}, {0x1E8C7, 0x1E8CF}, {0x1E950, 0x1E959}, {0x1EC71, 0x1ECAB},
    {0x1ECAD, 0x1ECAF}, {0x1ECB1, 0x1ECB4}, {0x1ED01, 0x1ED2D}, {0x1ED2F, 0x1ED3D},
    {0x1F100, 0x1F10C}, {0x1FBF0, 0x1FBF9},
};

static bool is_unicode_number(uint32_t cp) {
    if (cp < 0x80) return cp >= '0' && cp <= '9';
    const auto * end = std::end(kUnicodeNumberRanges);
    const auto * it = std::upper_bound(std::begin(kUnicodeNumberRanges), end, cp,
        [](uint32_t c, const uint32_t (&range)[2]) { return c < range[0]; });
    return it != std::begin(kUnicodeNumberRanges) && cp <= (*(it - 1))[1];
}

// DeepSeek V3/V4's tokenizer.json runs Split(\p{N}{1,3}), then
// Split([一-龥぀-ゟ゠-ヿ]+), both Isolated, before the main pattern, which then
// sees each piece on its own. The main pattern's \s+(?!\S) therefore keeps a
// whitespace run before a number or CJK text whole: "a  1" is a | "  " | 1,
// where one pass of the main pattern gives a | " " | " " | 1. The main pattern
// matches nothing in a run of numbers, so the run stays one piece. The two
// classes are disjoint, so one scan isolating both equals the two splits.
std::vector<std::string> Tokenizer::pre_split(const std::string & text) const {
    if (pre_type_ != PreTokenizer::DEEPSEEK3) return pre_tokenize(text);

    std::vector<std::string> pieces;
    const auto main_split = [&](size_t from, size_t to) {
        if (to <= from) return;
        for (auto & piece : pre_tokenize(text.substr(from, to - from))) {
            pieces.push_back(std::move(piece));
        }
    };
    const char * s = text.c_str();
    const size_t len = text.size();
    size_t seg = 0, pos = 0;
    while (pos < len) {
        int cplen = 0;
        const uint32_t cp = utf8_decode(s + pos, len - pos, &cplen);
        if (cplen <= 0) cplen = 1;
        const bool number = is_unicode_number(cp);
        if (!number && !is_deepseek_cjk(cp)) {
            pos += cplen;
            continue;
        }
        size_t end = pos;
        int run = 0;
        while (end < len) {
            int cl = 0;
            const uint32_t c = utf8_decode(s + end, len - end, &cl);
            if (cl <= 0) cl = 1;
            if (number ? (run == 3 || !is_unicode_number(c)) : !is_deepseek_cjk(c)) break;
            end += cl;
            ++run;
        }
        main_split(seg, pos);
        if (number) {
            pieces.push_back(text.substr(pos, end - pos));
        } else {
            main_split(pos, end);
        }
        seg = pos = end;
    }
    main_split(seg, len);
    return pieces;
}

// ─── BPE encoding ──────────────────────────────────────────────────────

// GPT-2 byte-level BPE maps printable byte ranges directly to codepoints,
// and all other 68 non-printable bytes into U+0100..U+0143 (256..256+68).
static constexpr uint32_t GPT2_BPE_OFFSET = 256;
static constexpr uint32_t GPT2_BPE_NUM_RESERVED = 68;

static inline bool is_gpt2_printable_byte(uint32_t b) {
    return (b >= 33 && b <= 126) ||
           (b >= 161 && b <= 172) ||
           (b >= 174 && b <= 255);
}

// Forward GPT-2 byte encoding: raw byte → Unicode codepoint (UTF-8 string).
// This is the inverse of gpt2_unicode_to_byte (defined later, near decode).
// Bytes in {33-126, 161-172, 174-255} map to themselves as a codepoint;
// all others (0-32, 127-160, 173) map to U+0100..U+0143.
static std::string byte_to_gpt2_unicode(uint8_t b) {
    // Build forward table once (thread-safe via C++11 static init).
    static const auto fwd = []() {
        std::array<uint32_t, 256> t{};
        int n = 0;
        for (int i = 0; i < 256; i++) {
            if (is_gpt2_printable_byte(i)) {
                t[i] = (uint32_t)i;
            } else {
                t[i] = GPT2_BPE_OFFSET + n;
                n++;
            }
        }
        return t;
    }();
    uint32_t cp = fwd[b];
    // Encode codepoint as UTF-8.
    char buf[4];
    int len;
    if (cp < 0x80) {
        buf[0] = (char)cp; len = 1;
    } else if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        len = 2;
    } else {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        len = 3;
    }
    return std::string(buf, len);
}

// Convert a raw UTF-8 text piece to GPT-2 byte-encoded form for BPE lookup.
static std::string encode_gpt2_bpe(const std::string & text) {
    std::string out;
    out.reserve(text.size() * 2);  // GPT-2 encoding may expand
    for (uint8_t b : text) {
        out += byte_to_gpt2_unicode(b);
    }
    return out;
}

// Encode a single pre-tokenized piece using BPE merges.
std::vector<int32_t> Tokenizer::bpe_encode_piece(const std::string & piece) const {
    if (piece.empty()) return {};

    std::vector<std::string> symbols;

    if (is_sentencepiece_) {
        // SentencePiece: replace leading space with ▁, tokens are raw UTF-8.
        std::string sp_piece;
        sp_piece.reserve(piece.size() + 2);
        size_t start = 0;
        if (!piece.empty() && piece[0] == ' ') {
            sp_piece += "\xe2\x96\x81";  // ▁ (U+2581)
            start = 1;
        }
        sp_piece += piece.substr(start);
        // Replace any remaining spaces with ▁
        std::string encoded;
        encoded.reserve(sp_piece.size());
        for (char c : sp_piece) {
            if (c == ' ') {
                encoded += "\xe2\x96\x81";
            } else {
                encoded += c;
            }
        }

        // Try whole piece as single token.
        auto it = token_to_id_.find(encoded);
        if (it != token_to_id_.end()) {
            return { it->second };
        }

        // Split into individual UTF-8 characters as initial BPE symbols.
        const char * p = encoded.c_str();
        const char * end = p + encoded.size();
        while (p < end) {
            int cplen;
            utf8_decode(p, (size_t)(end - p), &cplen);
            if (cplen <= 0) cplen = 1;
            std::string sym(p, cplen);
            auto sit = token_to_id_.find(sym);
            if (sit != token_to_id_.end()) {
                symbols.push_back(sym);
            } else {
                // Byte-fallback: <0xNN>
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>", (unsigned)(uint8_t)*p);
                symbols.push_back(buf);
            }
            p += cplen;
        }
    } else {
        // GPT-2 BPE: convert raw text to GPT-2 byte encoding for vocab lookup.
        std::string encoded = encode_gpt2_bpe(piece);

        // Try to find the encoded piece as a single token first.
        auto it = token_to_id_.find(encoded);
        if (it != token_to_id_.end()) {
            return { it->second };
        }

        // Split into individual GPT-2-encoded bytes as initial BPE symbols.
        for (size_t i = 0; i < piece.size(); i++) {
            std::string sym = byte_to_gpt2_unicode((uint8_t)piece[i]);
            auto sit = token_to_id_.find(sym);
            if (sit != token_to_id_.end()) {
                symbols.push_back(sym);
            } else {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>",
                              (unsigned)(uint8_t)piece[i]);
                symbols.push_back(buf);
            }
        }
    }

    if (symbols.size() <= 1) {
        if (symbols.empty()) return {};
        auto sit = token_to_id_.find(symbols[0]);
        if (sit != token_to_id_.end()) return { sit->second };
        return {};  // unknown token
    }

    // Iteratively merge the highest-priority pair until no more merges apply.
    while (symbols.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_pos = SIZE_MAX;

        for (size_t i = 0; i + 1 < symbols.size(); i++) {
            std::string pair = symbols[i] + " " + symbols[i + 1];
            auto mit = merge_rank_.find(pair);
            if (mit != merge_rank_.end() && mit->second < best_rank) {
                best_rank = mit->second;
                best_pos = i;
            }
        }

        if (best_pos == SIZE_MAX) break;  // no more merges

        // Merge the best pair.
        symbols[best_pos] = symbols[best_pos] + symbols[best_pos + 1];
        symbols.erase(symbols.begin() + best_pos + 1);
    }

    // Convert merged symbols to token IDs.
    std::vector<int32_t> ids;
    ids.reserve(symbols.size());
    for (const auto & sym : symbols) {
        auto sit = token_to_id_.find(sym);
        if (sit != token_to_id_.end()) {
            ids.push_back(sit->second);
        } else {
            // Unknown symbol — emit byte-fallback tokens if available.
            // Symbols are in GPT-2 byte encoding; decode each Unicode codepoint
            // back to the original byte before emitting <0xNN>.
            static const auto gpt2_rev = []() {
                // Build reverse table: codepoint → original byte.
                std::array<uint8_t, GPT2_BPE_OFFSET + GPT2_BPE_NUM_RESERVED> t{};  // covers U+0000..U+0143
                int n = 0;
                for (int b = 0; b < 256; b++) {
                    if (is_gpt2_printable_byte(b)) {
                        t[b] = (uint8_t)b;
                    } else {
                        t[GPT2_BPE_OFFSET + n] = (uint8_t)b;
                        n++;
                    }
                }
                return t;
            }();
            const char * p = sym.c_str();
            const char * end = p + sym.size();
            while (p < end) {
                int cplen;
                uint32_t cp = utf8_decode(p, (size_t)(end - p), &cplen);
                uint8_t orig_byte;
                if (is_gpt2_printable_byte(cp)) {
                    orig_byte = (uint8_t)cp;
                } else if (cp >= GPT2_BPE_OFFSET && cp < GPT2_BPE_OFFSET + GPT2_BPE_NUM_RESERVED) {
                    orig_byte = gpt2_rev[cp];
                } else {
                    orig_byte = '?';
                }
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>", (unsigned)orig_byte);
                auto bit = token_to_id_.find(buf);
                if (bit != token_to_id_.end()) {
                    ids.push_back(bit->second);
                }
                p += cplen;
            }
        }
    }

    return ids;
}

// ─── Public API ─────────────────────────────────────────────────────────

bool Tokenizer::load_from_gguf(const char * model_path) {
    // A second load replaces the vocabulary: nothing of the first one stays.
    id_to_token_.clear();
    token_to_id_.clear();
    merge_rank_.clear();
    added_tokens_.clear();
    user_added_tokens_.clear();
    control_ids_.clear();
    bos_id_ = eos_id_ = eos_chat_id_ = -1;
    pre_type_ = PreTokenizer::QWEN35;
    is_sentencepiece_ = false;

    struct gguf_init_params params = { /*.no_alloc=*/ true, /*.ctx=*/ nullptr };
    struct gguf_context * gctx = gguf_init_from_file(model_path, params);
    if (!gctx) {
        std::fprintf(stderr, "[tokenizer] failed to open GGUF: %s\n", model_path);
        return false;
    }

    // Load token strings.
    int tokens_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    if (tokens_key < 0) {
        std::fprintf(stderr, "[tokenizer] missing tokenizer.ggml.tokens in %s\n",
                     model_path);
        gguf_free(gctx);
        return false;
    }

    const int n_vocab = gguf_get_arr_n(gctx, tokens_key);
    id_to_token_.resize(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        const char * tok = gguf_get_arr_str(gctx, tokens_key, i);
        id_to_token_[i] = tok ? tok : "";
        token_to_id_[id_to_token_[i]] = i;
    }

    // Load merge table.
    int merges_key = gguf_find_key(gctx, "tokenizer.ggml.merges");
    if (merges_key >= 0) {
        const int n_merges = gguf_get_arr_n(gctx, merges_key);
        for (int i = 0; i < n_merges; i++) {
            const char * merge = gguf_get_arr_str(gctx, merges_key, i);
            if (merge) {
                merge_rank_[merge] = i;
            }
        }
    }

    // Load token types and build added-tokens list.
    // GGUF token_type: 1=normal, 3=control, 4=user-defined, 5=unused
    // Types 3 and 4 are "special" tokens matched as whole strings before BPE.
    int type_key = gguf_find_key(gctx, "tokenizer.ggml.token_type");
    if (type_key >= 0) {
        const int n_types = gguf_get_arr_n(gctx, type_key);
        for (int i = 0; i < n_types && i < n_vocab; i++) {
            uint32_t ttype = ((const uint32_t *)gguf_get_arr_data(gctx, type_key))[i];
            if (ttype == 3 || ttype == 4) {
                const std::string & tok = id_to_token_[i];
                if (!tok.empty()) {
                    added_tokens_.push_back({tok, (int32_t)i});
                    if (ttype == 3) control_ids_.push_back((int32_t)i);
                }
            }
        }
        // Some DeepSeek V4.1 conversions type the chat, thinking and tool
        // markers as normal tokens (V4 files type them control/user-defined).
        // BPE would then spell <｜User｜> as five text pieces and the model
        // would never see its turn markers. On a DeepSeek vocabulary, match
        // every token in the full-width control form, and the markers the
        // DeepSeek template emits, as a whole string like any added token.
        static const std::string kDsBos = "<｜begin▁of▁sentence｜>";
        if (!id_to_token_.empty() && id_to_token_[0] == kDsBos) {
            static const char * kDsMarkers[] = {
                "<think>", "</think>", "<|EOT|>", "｜DSML｜", "<dsml:", "</dsml:",
            };
            const std::string open = "<｜", close = "｜>";
            size_t repaired = 0;
            for (int i = 0; i < n_types && i < n_vocab; i++) {
                const uint32_t ttype = ((const uint32_t *)gguf_get_arr_data(gctx, type_key))[i];
                if (ttype == 3 || ttype == 4) continue;
                const std::string & tok = id_to_token_[i];
                bool marker = tok.size() > open.size() + close.size() &&
                    tok.compare(0, open.size(), open) == 0 &&
                    tok.compare(tok.size() - close.size(), close.size(), close) == 0;
                for (const char * m : kDsMarkers) marker = marker || tok == m;
                if (marker) { added_tokens_.push_back({tok, (int32_t)i}); ++repaired; }
            }
            if (repaired > 0) {
                std::fprintf(stderr, "[tokenizer] %zu DeepSeek control tokens were typed "
                             "normal in the GGUF; matching them as whole tokens\n", repaired);
            }
        }
        // Sort longest-first for greedy matching.
        std::sort(added_tokens_.begin(), added_tokens_.end(),
                  [](const auto & a, const auto & b) {
                      return a.first.size() > b.first.size();
                  });
        std::sort(control_ids_.begin(), control_ids_.end());
        for (const auto & entry : added_tokens_) {
            if (!is_control_token(entry.second)) user_added_tokens_.push_back(entry);
        }
        std::fprintf(stderr, "[tokenizer] added_tokens: %zu special tokens\n",
                     added_tokens_.size());
    }

    // Detect tokenizer model type (sentencepiece vs bpe).
    int model_key = gguf_find_key(gctx, "tokenizer.ggml.model");
    if (model_key >= 0) {
        const char * model = gguf_get_val_str(gctx, model_key);
        // SentencePiece models store tokens as raw UTF-8 with ▁ for space.
        // GPT-2/BPE models use byte-level Unicode encoding.
        if (model && (std::strcmp(model, "llama") == 0 ||
                      std::strncmp(model, "gemma", 5) == 0)) {
            is_sentencepiece_ = true;
        }
    }

    // Detect pre-tokenizer type.
    int pre_key = gguf_find_key(gctx, "tokenizer.ggml.pre");
    if (pre_key >= 0) {
        const char * pre = gguf_get_val_str(gctx, pre_key);
        if (pre && std::strcmp(pre, "qwen35") == 0) {
            pre_type_ = PreTokenizer::QWEN35;
        } else if (pre && (std::strcmp(pre, "deepseek-v3") == 0 ||
                           std::strcmp(pre, "joyai-llm") == 0)) {
            // llama.cpp's DEEPSEEK3_LLM / JOYAI_LLM pre types.
            pre_type_ = PreTokenizer::DEEPSEEK3;
        } else {
            pre_type_ = PreTokenizer::QWEN2;
        }
    }

    // Load special token IDs.
    auto get_i32 = [&](const char * key) -> int32_t {
        int k = gguf_find_key(gctx, key);
        if (k < 0) return -1;
        return (int32_t)gguf_get_val_u32(gctx, k);
    };

    bos_id_ = get_i32("tokenizer.ggml.bos_token_id");
    eos_id_ = get_i32("tokenizer.ggml.eos_token_id");
    eos_chat_id_ = get_i32("tokenizer.ggml.eot_token_id");
    const int arch_key = gguf_find_key(gctx, "general.architecture");
    const char * arch = arch_key >= 0 ? gguf_get_val_str(gctx, arch_key) : nullptr;
    if (eos_chat_id_ < 0 && arch && std::strcmp(arch, "laguna") == 0) {
        // Laguna GGUFs ship only the EOS token; the backend ends a turn at
        // </assistant> (laguna_target_loader.cpp), and so must the server.
        // Checked before the markers below, which the backend never uses
        // for Laguna even when the vocabulary has them.
        auto eot = token_to_id_.find("</assistant>");
        if (eot != token_to_id_.end()) eos_chat_id_ = eot->second;
    }
    if (eos_chat_id_ < 0 && arch && (std::strcmp(arch, "inkling2_loop") == 0 || std::strcmp(arch, "inkling2") == 0)) {
        auto eot = token_to_id_.find("<|content_model_end_sampling|>");
        if (eot != token_to_id_.end()) eos_chat_id_ = eot->second;
    }
    if (eos_chat_id_ < 0) {
        // Qwen3 uses <|im_end|> as EOT.
        auto eot = token_to_id_.find("<|im_end|>");
        if (eot != token_to_id_.end()) eos_chat_id_ = eot->second;
    }
    if (eos_chat_id_ < 0) {
        // Gemma4 uses <turn|> as end-of-turn.
        auto eot = token_to_id_.find("<turn|>");
        if (eot != token_to_id_.end()) eos_chat_id_ = eot->second;
    }

    gguf_free(gctx);

    std::fprintf(stderr, "[tokenizer] loaded vocab=%d merges=%zu bos=%d eos=%d eot=%d pre=%s sp=%s\n",
                 n_vocab, merge_rank_.size(), bos_id_, eos_id_, eos_chat_id_,
                 pre_type_ == PreTokenizer::QWEN35      ? "qwen35"
                 : pre_type_ == PreTokenizer::DEEPSEEK3 ? "deepseek3"
                                                        : "qwen2",
                 is_sentencepiece_ ? "yes" : "no");
    return true;
}

std::vector<int32_t> Tokenizer::encode(const std::string & text, bool match_control) const {
    return encode_matching(text, match_control ? added_tokens_ : user_added_tokens_, {}, {});
}

std::vector<int32_t> Tokenizer::encode(const std::string & text, bool match_control,
                                       const std::vector<int32_t> & literal,
                                       const std::vector<std::pair<size_t, size_t>> & spans,
                                       const std::vector<int32_t> & as_text) const {
    const auto & all = match_control ? added_tokens_ : user_added_tokens_;
    if (literal.empty()) return encode_matching(text, all, spans, as_text);
    std::vector<std::pair<std::string, int32_t>> added;
    for (const auto & entry : all) {
        if (std::find(literal.begin(), literal.end(), entry.second) == literal.end()) {
            added.push_back(entry);
        }
    }
    return encode_matching(text, added, spans, as_text);
}

std::vector<int32_t> Tokenizer::encode_matching(
        const std::string & text, const std::vector<std::pair<std::string, int32_t>> & added,
        const std::vector<std::pair<size_t, size_t>> & spans,
        const std::vector<int32_t> & as_text) const {
    std::vector<int32_t> ids;
    // Pre-tokenize + BPE a run of plain text.
    const auto plain = [&](const std::string & segment) {
        for (const auto & piece : pre_split(segment)) {
            auto piece_ids = bpe_encode_piece(piece);
            ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
        }
    };
    // If no added tokens, fast path: pre-tokenize → BPE entire text.
    if (added.empty()) {
        plain(text);
        return ids;
    }

    // Whether an added token found at `at` stays text: one of `as_text`
    // inside a span. Matches come in order, so the spans are walked once.
    size_t span = 0;
    const auto stays_text = [&](size_t at, int32_t id) {
        while (span < spans.size() && spans[span].second <= at) ++span;
        return span < spans.size() && spans[span].first <= at &&
               std::find(as_text.begin(), as_text.end(), id) != as_text.end();
    };

    // Split text into segments: alternating normal text and special tokens.
    // Special tokens are matched greedily (longest first).
    size_t pos = 0;
    while (pos < text.size()) {
        // Try to match any added token at current position.
        bool matched = false;
        for (const auto & [tok_str, tok_id] : added) {
            if (pos + tok_str.size() <= text.size() &&
                text.compare(pos, tok_str.size(), tok_str) == 0) {
                if (stays_text(pos, tok_id)) plain(tok_str);
                else ids.push_back(tok_id);
                pos += tok_str.size();
                matched = true;
                break;
            }
        }
        if (matched) continue;

        // Find the next special token (or end of string).
        size_t next_special = text.size();
        for (const auto & [tok_str, tok_id] : added) {
            size_t found = text.find(tok_str, pos);
            if (found != std::string::npos && found < next_special) {
                next_special = found;
            }
        }

        // Pre-tokenize + BPE the normal segment.
        plain(text.substr(pos, next_special - pos));
        pos = next_special;
    }
    return ids;
}

// GPT-2 byte-level BPE uses a Unicode mapping where each byte 0-255 is
// represented by a specific Unicode codepoint.  Bytes that already have a
// printable representation (33-126, 161-172, 174-255) map to themselves;
// all other bytes (0-32, 127-160, 173) are offset into U+0100..U+0143.
// Token strings in the GGUF vocabulary are stored in this encoding, so we
// must reverse-map each codepoint back to its original byte.

static uint8_t gpt2_unicode_to_byte(uint32_t cp) {
    // Direct-mapped ranges: the codepoint IS the byte.
    if (is_gpt2_printable_byte(cp)) {
        return (uint8_t)cp;
    }
    // Offset-mapped range: U+0100..U+0143 → non-printable bytes.
    // Build the reverse table once (thread-safe via C++11 static init).
    static const auto table = []() {
        std::array<uint8_t, GPT2_BPE_NUM_RESERVED> t{};
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (is_gpt2_printable_byte(b)) continue;
            t[n] = (uint8_t)b;
            n++;
        }
        return t;
    }();
    if (cp >= GPT2_BPE_OFFSET && cp < GPT2_BPE_OFFSET + GPT2_BPE_NUM_RESERVED) {
        return table[cp - GPT2_BPE_OFFSET];
    }
    // Shouldn't happen for valid BPE tokens — return replacement.
    return '?';
}

static std::string decode_gpt2_bpe(const std::string & tok) {
    std::string out;
    out.reserve(tok.size());
    const char * p = tok.c_str();
    const char * end = p + tok.size();
    while (p < end) {
        int cplen;
        uint32_t cp = utf8_decode(p, (size_t)(end - p), &cplen);
        if (is_gpt2_printable_byte(cp)) {
            out.push_back((char)cp);
        } else if (cp >= GPT2_BPE_OFFSET && cp < GPT2_BPE_OFFSET + GPT2_BPE_NUM_RESERVED) {
            out.push_back((char)gpt2_unicode_to_byte(cp));
        } else {
            // Raw Unicode codepoint (e.g. U+FF5C '｜', U+2581 '▁') — emit raw bytes
            out.append(p, cplen);
        }
        p += cplen;
    }
    return out;
}

std::string Tokenizer::token_text(int32_t id) const {
    if (id < 0 || id >= (int32_t)id_to_token_.size()) return "";
    const std::string & tok = id_to_token_[id];

    // Handle byte-fallback tokens like <0xNN>.
    if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' &&
        tok[2] == 'x' && tok[5] == '>') {
        unsigned val = 0;
        if (std::sscanf(tok.c_str(), "<0x%02X>", &val) == 1) {
            return std::string(1, (char)(uint8_t)val);
        }
    }

    // Special tokens (e.g. <|im_start|>, <turn|>) — return as-is.
    if (!tok.empty() && tok[0] == '<' && tok.back() == '>') {
        return tok;
    }

    if (is_sentencepiece_) {
        // SentencePiece: tokens are raw UTF-8 with ▁ (U+2581) for space.
        std::string out;
        out.reserve(tok.size());
        const char * p = tok.c_str();
        const char * end = p + tok.size();
        while (p < end) {
            // ▁ is 3 bytes: 0xE2 0x96 0x81
            if (end - p >= 3 &&
                (uint8_t)p[0] == 0xE2 &&
                (uint8_t)p[1] == 0x96 &&
                (uint8_t)p[2] == 0x81) {
                out.push_back(' ');
                p += 3;
            } else {
                out.push_back(*p);
                p++;
            }
        }
        return out;
    }

    // Decode GPT-2 byte-level BPE encoding → raw bytes.
    return decode_gpt2_bpe(tok);
}

std::string Tokenizer::decode(const std::vector<int32_t> & ids) const {
    std::string result;
    for (int32_t id : ids) {
        result += token_text(id);
    }
    return result;
}

const std::string & Tokenizer::raw_token(int32_t id) const {
    static const std::string empty;
    if (id < 0 || id >= (int32_t)id_to_token_.size()) return empty;
    return id_to_token_[id];
}

bool Tokenizer::is_added_token(int32_t id) const {
    for (const auto & entry : added_tokens_) {
        if (entry.second == id) return true;
    }
    return false;
}

bool Tokenizer::is_control_token(int32_t id) const {
    return std::binary_search(control_ids_.begin(), control_ids_.end(), id);
}

int32_t Tokenizer::token_to_id(const std::string & token) const {
    auto it = token_to_id_.find(token);
    return it != token_to_id_.end() ? it->second : -1;
}

}  // namespace luce::common
