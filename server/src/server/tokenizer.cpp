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
#include <limits>
#include <utility>

namespace dflash::common {

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
// Matches the Qwen3.5 pattern:
//   (?:'[sStTmMdD]|...) |
//   [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ |
//   \p{N} |
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

        // Pattern 3: \p{N}  (single digit)
        if (is_digit(cp)) {
            pos += cplen;
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

        // Pattern 5: \s*[\r\n]+
        if (is_whitespace(cp)) {
            size_t p = pos;
            int cl = 0;
            uint32_t c = peek_cp(p, &cl);
            // Consume leading whitespace
            while (cl > 0 && is_whitespace(c) && !is_newline(c)) {
                p += cl;
                c = peek_cp(p, &cl);
            }
            if (cl > 0 && is_newline(c)) {
                while (cl > 0 && is_newline(c)) {
                    p += cl;
                    c = peek_cp(p, &cl);
                }
                pieces.push_back(text.substr(pos, p - pos));
                pos = p;
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

// ─── BPE encoding ──────────────────────────────────────────────────────

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
            if ((i >= 33  && i <= 126) ||
                (i >= 161 && i <= 172) ||
                (i >= 174 && i <= 255)) {
                t[i] = (uint32_t)i;
            } else {
                t[i] = 256 + n;
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
                std::array<uint8_t, 324> t{};  // covers U+0000..U+0143
                int n = 0;
                for (int b = 0; b < 256; b++) {
                    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
                        t[b] = (uint8_t)b;
                    } else {
                        t[256 + n] = (uint8_t)b;
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
                if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
                    orig_byte = (uint8_t)cp;
                } else if (cp >= 256 && cp < 256 + 68) {
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
                }
            }
        }
        // Sort longest-first for greedy matching.
        std::sort(added_tokens_.begin(), added_tokens_.end(),
                  [](const auto & a, const auto & b) {
                      return a.first.size() > b.first.size();
                  });
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
                 pre_type_ == PreTokenizer::QWEN35 ? "qwen35" : "qwen2",
                 is_sentencepiece_ ? "yes" : "no");
    return true;
}

std::vector<int32_t> Tokenizer::encode(const std::string & text) const {
    // If no added tokens, fast path: pre-tokenize → BPE entire text.
    if (added_tokens_.empty()) {
        std::vector<std::string> pieces = pre_tokenize(text);
        std::vector<int32_t> ids;
        for (const auto & piece : pieces) {
            auto piece_ids = bpe_encode_piece(piece);
            ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
        }
        return ids;
    }

    // Split text into segments: alternating normal text and special tokens.
    // Special tokens are matched greedily (longest first).
    std::vector<int32_t> ids;
    size_t pos = 0;
    while (pos < text.size()) {
        // Try to match any added token at current position.
        bool matched = false;
        for (const auto & [tok_str, tok_id] : added_tokens_) {
            if (pos + tok_str.size() <= text.size() &&
                text.compare(pos, tok_str.size(), tok_str) == 0) {
                ids.push_back(tok_id);
                pos += tok_str.size();
                matched = true;
                break;
            }
        }
        if (matched) continue;

        // Find the next special token (or end of string).
        size_t next_special = text.size();
        for (const auto & [tok_str, tok_id] : added_tokens_) {
            size_t found = text.find(tok_str, pos);
            if (found != std::string::npos && found < next_special) {
                next_special = found;
            }
        }

        // Pre-tokenize + BPE the normal segment.
        std::string segment = text.substr(pos, next_special - pos);
        std::vector<std::string> pieces = pre_tokenize(segment);
        for (const auto & piece : pieces) {
            auto piece_ids = bpe_encode_piece(piece);
            ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
        }
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
    if ((cp >= 33  && cp <= 126) ||
        (cp >= 161 && cp <= 172) ||
        (cp >= 174 && cp <= 255)) {
        return (uint8_t)cp;
    }
    // Offset-mapped range: U+0100..U+0143 → non-printable bytes.
    // Build the reverse table once (thread-safe via C++11 static init).
    static const auto table = []() {
        std::array<uint8_t, 68> t{};
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if ((b >= 33  && b <= 126) ||
                (b >= 161 && b <= 172) ||
                (b >= 174 && b <= 255)) continue;
            t[n] = (uint8_t)b;
            n++;
        }
        return t;
    }();
    if (cp >= 256 && cp < 256 + 68) {
        return table[cp - 256];
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
        out.push_back((char)gpt2_unicode_to_byte(cp));
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

int32_t Tokenizer::token_to_id(const std::string & token) const {
    auto it = token_to_id_.find(token);
    return it != token_to_id_.end() ? it->second : -1;
}

}  // namespace dflash::common
