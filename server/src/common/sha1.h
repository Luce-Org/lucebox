// Minimal portable SHA-1 — shared by disk_prefix_cache, prefix_cache, and
// any other TU that needs a lightweight hash with no OpenSSL dependency.
// Include once per translation unit; the function is defined inline.
//
// Usage: sha1_hash(data, len, out20)  — writes 20 bytes to out20.

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace dflash::common {

inline void sha1_hash(const void * data, size_t len, uint8_t out[20]) {
    auto rotl = [](uint32_t x, int n) -> uint32_t {
        return (x << n) | (x >> (32 - n));
    };

    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE,
             h3 = 0x10325476, h4 = 0xC3D2E1F0;

    size_t new_len = len + 1;
    while (new_len % 64 != 56) new_len++;
    std::vector<uint8_t> msg(new_len + 8, 0);
    std::memcpy(msg.data(), data, len);
    msg[len] = 0x80;
    uint64_t bit_len = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) {
        msg[new_len + i] = (uint8_t)(bit_len >> (56 - 8 * i));
    }

    for (size_t offset = 0; offset < msg.size(); offset += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t)msg[offset + 4*i]   << 24) |
                   ((uint32_t)msg[offset + 4*i+1] << 16) |
                   ((uint32_t)msg[offset + 4*i+2] <<  8) |
                   ((uint32_t)msg[offset + 4*i+3]);
        }
        for (int i = 16; i < 80; i++) {
            w[i] = rotl(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
        }

        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if      (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d;           k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else              { f = b ^ c ^ d;           k = 0xCA62C1D6; }
            uint32_t temp = rotl(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rotl(b, 30); b = a; a = temp;
        }
        h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
    }

    auto store32 = [](uint8_t * p, uint32_t v) {
        p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16);
        p[2] = (uint8_t)(v >>  8); p[3] = (uint8_t)v;
    };
    store32(out,      h0);
    store32(out +  4, h1);
    store32(out +  8, h2);
    store32(out + 12, h3);
    store32(out + 16, h4);
}

}  // namespace dflash::common
