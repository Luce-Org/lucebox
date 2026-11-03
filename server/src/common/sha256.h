// SHA-256 (RFC 6234), self-contained so the server needs no crypto library.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace luce::common {

struct Sha256Ctx {
    uint32_t state[8];
    uint64_t bit_len;
    uint8_t  buf[64];
    size_t   buf_len;
};

inline uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

inline void sha256_init(Sha256Ctx & c) {
    c.state[0] = 0x6a09e667u;
    c.state[1] = 0xbb67ae85u;
    c.state[2] = 0x3c6ef372u;
    c.state[3] = 0xa54ff53au;
    c.state[4] = 0x510e527fu;
    c.state[5] = 0x9b05688cu;
    c.state[6] = 0x1f83d9abu;
    c.state[7] = 0x5be0cd19u;
    c.bit_len = 0;
    c.buf_len = 0;
}

inline void sha256_compress(Sha256Ctx & c, const uint8_t * block) {
    static const uint32_t K[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t(block[i*4+0]) << 24) | (uint32_t(block[i*4+1]) << 16) |
               (uint32_t(block[i*4+2]) << 8 ) |  uint32_t(block[i*4+3]);
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = c.state[0], b = c.state[1], cc = c.state[2], d = c.state[3];
    uint32_t e = c.state[4], f = c.state[5], g = c.state[6], h = c.state[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t mj = (a & b) ^ (a & cc) ^ (b & cc);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1;
        d = cc; cc = b; b = a; a = t1 + t2;
    }
    c.state[0] += a; c.state[1] += b; c.state[2] += cc; c.state[3] += d;
    c.state[4] += e; c.state[5] += f; c.state[6] += g;  c.state[7] += h;
}

inline void sha256_update(Sha256Ctx & c, const uint8_t * data, size_t len) {
    if (len == 0) return;  // data may be null (an empty string_view)
    c.bit_len += uint64_t(len) * 8;
    if (c.buf_len) {
        size_t take = std::min(size_t(64) - c.buf_len, len);
        std::memcpy(c.buf + c.buf_len, data, take);
        c.buf_len += take;
        data += take;
        len  -= take;
        if (c.buf_len == 64) {
            sha256_compress(c, c.buf);
            c.buf_len = 0;
        }
    }
    while (len >= 64) {
        sha256_compress(c, data);
        data += 64;
        len  -= 64;
    }
    if (len) {
        std::memcpy(c.buf, data, len);
        c.buf_len = len;
    }
}

inline std::string sha256_final(Sha256Ctx & c) {
    uint64_t bits = c.bit_len;
    c.buf[c.buf_len++] = 0x80;
    if (c.buf_len > 56) {
        std::memset(c.buf + c.buf_len, 0, 64 - c.buf_len);
        sha256_compress(c, c.buf);
        c.buf_len = 0;
    }
    std::memset(c.buf + c.buf_len, 0, 56 - c.buf_len);
    for (int i = 7; i >= 0; --i) {
        c.buf[56 + i] = uint8_t(bits & 0xff);
        bits >>= 8;
    }
    sha256_compress(c, c.buf);

    static const char * hex = "0123456789abcdef";
    std::string out;
    out.resize(64);
    for (int i = 0; i < 8; ++i) {
        uint32_t v = c.state[i];
        for (int j = 0; j < 4; ++j) {
            uint8_t byte = uint8_t((v >> (24 - j * 8)) & 0xff);
            out[i*8 + j*2 + 0] = hex[byte >> 4];
            out[i*8 + j*2 + 1] = hex[byte & 0x0f];
        }
    }
    return out;
}

// Incremental SHA-256: hex() reads the digest of what was fed so far, and
// update() can go on after it.
class Sha256 {
public:
    Sha256() { sha256_init(ctx_); }
    void update(std::string_view data) {
        sha256_update(ctx_, reinterpret_cast<const uint8_t *>(data.data()), data.size());
    }
    std::string hex() const {
        Sha256Ctx done = ctx_;
        return sha256_final(done);
    }

private:
    Sha256Ctx ctx_;
};

}  // namespace luce::common
