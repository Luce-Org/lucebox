// UTF-8 utility functions for safe streaming and JSON serialization.
//
// Extracted from sse_emitter.cpp so that unit tests can validate these
// independently without constructing a full SseEmitter.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace dflash::common {

// Snap a byte offset back to a UTF-8 code-point boundary.
// Returns the largest position <= `pos` that doesn't split a multi-byte sequence.
// (Mirrors ds4_server.c's utf8_stream_safe_len.)
inline size_t utf8_safe_len(const std::string & s, size_t pos) {
    if (pos >= s.size()) return s.size();
    while (pos > 0 && (s[pos] & 0xC0) == 0x80) pos--;
    return pos;
}

inline size_t utf8_sequence_length(uint8_t c) {
    if (c < 0x80) return 1;
    if (c >= 0xC2 && c <= 0xDF) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if (c >= 0xF0 && c <= 0xF4) return 4;
    return 0;
}

inline bool utf8_valid_trailing_prefix(const std::string & s,
                                       size_t start, size_t seq_len) {
    if (seq_len < 2 || start >= s.size()) return false;
    for (size_t i = start + 1; i < s.size(); ++i) {
        if (((uint8_t)s[i] & 0xC0) != 0x80) return false;
    }
    if (start + 1 < s.size()) {
        const uint8_t lead = (uint8_t)s[start];
        const uint8_t next = (uint8_t)s[start + 1];
        // Once the first continuation byte is present, reject prefixes that
        // can only complete as an overlong, surrogate, or out-of-range value.
        if (lead == 0xE0 && next < 0xA0) return false;
        if (lead == 0xED && next > 0x9F) return false;
        if (lead == 0xF0 && next < 0x90) return false;
        if (lead == 0xF4 && next > 0x8F) return false;
    }
    return s.size() - start < seq_len;
}

// Return the number of bytes that are safe to sanitize now. Only a valid
// incomplete trailing prefix is held back, and that prefix is at most three
// bytes. Invalid bytes are included in the returned prefix so they are
// replaced promptly instead of being retained indefinitely.
inline size_t utf8_stream_safe_len(const std::string & s) {
    size_t i = 0;
    while (i < s.size()) {
        const uint8_t c = (uint8_t)s[i];
        const size_t seq_len = utf8_sequence_length(c);
        if (seq_len == 0) {
            ++i;
            continue;
        }
        if (seq_len == 1) {
            ++i;
            continue;
        }
        if (i + seq_len > s.size()) {
            if (utf8_valid_trailing_prefix(s, i, seq_len)) return i;
            ++i;
            continue;
        }

        bool valid = true;
        for (size_t j = 1; j < seq_len; ++j) {
            if (((uint8_t)s[i + j] & 0xC0) != 0x80) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            ++i;
            continue;
        }

        uint32_t cp = 0;
        if (seq_len == 2) {
            cp = ((uint32_t)(c & 0x1F) << 6) |
                 ((uint32_t)((uint8_t)s[i + 1]) & 0x3F);
            if (cp < 0x80) valid = false;
        } else if (seq_len == 3) {
            cp = ((uint32_t)(c & 0x0F) << 12) |
                 ((uint32_t)((uint8_t)s[i + 1] & 0x3F) << 6) |
                 ((uint32_t)((uint8_t)s[i + 2]) & 0x3F);
            if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) valid = false;
        } else {
            cp = ((uint32_t)(c & 0x07) << 18) |
                 ((uint32_t)((uint8_t)s[i + 1] & 0x3F) << 12) |
                 ((uint32_t)((uint8_t)s[i + 2] & 0x3F) << 6) |
                 ((uint32_t)((uint8_t)s[i + 3]) & 0x3F);
            if (cp < 0x10000 || cp > 0x10FFFF) valid = false;
        }
        i += valid ? seq_len : 1;
    }
    return s.size();
}

// Sanitize a string for JSON: replace invalid/incomplete UTF-8 with U+FFFD.
inline std::string utf8_sanitize(const std::string & s) {
    // ASCII is by far the common path for model output. Avoid allocating or
    // walking the validation state machine when no UTF-8 work is required.
    bool has_high_bit = false;
    for (unsigned char c : s) {
        if (c >= 0x80) {
            has_high_bit = true;
            break;
        }
    }
    if (!has_high_bit) return s;

    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        const uint8_t c = (uint8_t)s[i];
        const size_t seq_len = utf8_sequence_length(c);
        if (seq_len == 0 || i + seq_len > s.size()) {
            if (seq_len > 1 && utf8_valid_trailing_prefix(s, i, seq_len)) {
                // A valid prefix that reaches end-of-input is one truncated
                // sequence, not one independent invalid byte per continuation.
                out += "\xEF\xBF\xBD";
                break;
            }
            out += "\xEF\xBF\xBD";
            ++i;
            continue;
        }

        bool valid = true;
        for (size_t j = 1; j < seq_len; ++j) {
            if (((uint8_t)s[i + j] & 0xC0) != 0x80) {
                valid = false;
                break;
            }
        }
        if (valid) {
            uint32_t cp = 0;
            if (seq_len == 1) {
                cp = c;
            } else if (seq_len == 2) {
                cp = ((uint32_t)(c & 0x1F) << 6) |
                     ((uint32_t)((uint8_t)s[i+1]) & 0x3F);
                if (cp < 0x80) valid = false;
            } else if (seq_len == 3) {
                cp = ((uint32_t)(c & 0x0F) << 12) |
                     ((uint32_t)((uint8_t)s[i+1] & 0x3F) << 6) |
                     ((uint32_t)((uint8_t)s[i+2]) & 0x3F);
                if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) valid = false;
            } else {
                cp = ((uint32_t)(c & 0x07) << 18) |
                     ((uint32_t)((uint8_t)s[i+1] & 0x3F) << 12) |
                     ((uint32_t)((uint8_t)s[i+2] & 0x3F) << 6) |
                     ((uint32_t)((uint8_t)s[i+3]) & 0x3F);
                if (cp < 0x10000 || cp > 0x10FFFF) valid = false;
            }
        }
        if (valid) {
            out.append(s, i, seq_len);
            i += seq_len;
        } else {
            out += "\xEF\xBF\xBD";
            ++i;  // only skip lead byte; next byte may be a valid start
        }
    }
    return out;
}

}  // namespace dflash::common
