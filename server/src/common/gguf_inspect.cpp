#include "gguf_inspect.h"
#include "gguf.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace dflash::common {

GgufModelInfo inspect_gguf_model_info(const char * path) {
    GgufModelInfo info;

    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx = nullptr;
    gguf_context * gctx = gguf_init_from_file(path, gip);
    if (!gctx) return info;

    // Read architecture
    int64_t arch_id = gguf_find_key(gctx, "general.architecture");
    if (arch_id >= 0) {
        const char * v = gguf_get_val_str(gctx, arch_id);
        if (v) info.arch = v;
    }

    // Read layer count: <arch>.block_count
    if (!info.arch.empty()) {
        std::string key = info.arch + ".block_count";
        int64_t kid = gguf_find_key(gctx, key.c_str());
        if (kid >= 0) {
            info.n_layer = (int)gguf_get_val_u32(gctx, kid);
        }
    }

    gguf_free(gctx);
    return info;
}

// ─── SHA-256 (RFC 6234) ─────────────────────────────────────────────────
//
// Self-contained mini-implementation so we don't pull in OpenSSL just for
// one hash. Performance is "fine" — hashing a 17 GB GGUF takes ~30s on a
// fast NVMe, which is comparable to the per-file numbers `sha256sum` gets.
// We sidecar the result so this only happens on the first server start
// after a model is downloaded.

namespace {

struct Sha256Ctx {
    uint32_t state[8];
    uint64_t bit_len;
    uint8_t  buf[64];
    size_t   buf_len;
};

inline uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

void sha256_init(Sha256Ctx & c) {
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

void sha256_compress(Sha256Ctx & c, const uint8_t * block) {
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

void sha256_update(Sha256Ctx & c, const uint8_t * data, size_t len) {
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

std::string sha256_final(Sha256Ctx & c) {
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

std::string sha256_of_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    Sha256Ctx c;
    sha256_init(c);
    // 4 MiB read buffer: empirically best throughput on NVMe without
    // gulping the page cache. std::vector heap-allocates so we don't
    // blow the C++ thread stack.
    constexpr size_t BUF = 4 * 1024 * 1024;
    std::vector<uint8_t> buf(BUF);
    while (f) {
        f.read(reinterpret_cast<char*>(buf.data()), BUF);
        std::streamsize got = f.gcount();
        if (got > 0) sha256_update(c, buf.data(), size_t(got));
    }
    return sha256_final(c);
}

// Map LLAMA_FTYPE_* int → operator-friendly tag (Q4_K_M, IQ4_XS, BF16, …).
// Kept inline so we don't pull in llama.h here — those enum values are part
// of the GGUF on-disk format and won't change without a format bump.
const char * llama_ftype_name(int32_t v) {
    switch (v) {
    case 0:  return "F32";
    case 1:  return "F16";
    case 2:  return "Q4_0";
    case 3:  return "Q4_1";
    case 7:  return "Q8_0";
    case 8:  return "Q5_0";
    case 9:  return "Q5_1";
    case 10: return "Q2_K";
    case 11: return "Q3_K_S";
    case 12: return "Q3_K_M";
    case 13: return "Q3_K_L";
    case 14: return "Q4_K_S";
    case 15: return "Q4_K_M";
    case 16: return "Q5_K_S";
    case 17: return "Q5_K_M";
    case 18: return "Q6_K";
    case 19: return "IQ2_XXS";
    case 20: return "IQ2_XS";
    case 21: return "Q2_K_S";
    case 22: return "IQ3_XS";
    case 23: return "IQ3_XXS";
    case 24: return "IQ1_S";
    case 25: return "IQ4_NL";
    case 26: return "IQ3_S";
    case 27: return "IQ3_M";
    case 28: return "IQ2_S";
    case 29: return "IQ2_M";
    case 30: return "IQ4_XS";
    case 31: return "IQ1_M";
    case 32: return "BF16";
    case 36: return "TQ1_0";
    case 37: return "TQ2_0";
    case 38: return "MXFP4_MOE";
    case 39: return "NVFP4";
    case 40: return "Q1_0";
    case 1024: return "GUESSED";
    default: return "";
    }
}

bool read_sidecar_sha(const std::string & path, std::string & out) {
    std::ifstream f(path + ".sha256");
    if (!f) return false;
    std::string s;
    f >> s;  // tolerate `<hex>  filename\n` (sha256sum format) — we only want the first token
    if (s.size() != 64) return false;
    for (char c : s) {
        bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        if (!hex) return false;
    }
    out = std::move(s);
    return true;
}

void write_sidecar_sha(const std::string & path, const std::string & sha) {
    // Best-effort. If the directory isn't writable (read-only mount, model
    // dir owned by another user), we just skip — the in-memory hash is
    // already what /props will report this run.
    std::ofstream f(path + ".sha256");
    if (!f) return;
    f << sha << "\n";
}

}  // namespace

GgufMetadata read_gguf_metadata(const std::string & path,
                                bool compute_sha256) {
    GgufMetadata m;
    m.path = path;

    struct stat st{};
    if (::stat(path.c_str(), &st) == 0) {
        m.size_bytes = int64_t(st.st_size);
    }

    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx = nullptr;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) {
        // No GGUF header → bail. Still report path/size if we got them.
        return m;
    }
    m.ok = true;

    auto get_str = [&](const char * key, std::string & out) {
        int64_t id = gguf_find_key(gctx, key);
        if (id < 0) return;
        const char * v = gguf_get_val_str(gctx, id);
        if (v) out = v;
    };
    auto get_u32 = [&](const char * key, int32_t & out) {
        int64_t id = gguf_find_key(gctx, key);
        if (id < 0) return;
        out = int32_t(gguf_get_val_u32(gctx, id));
    };

    get_str("general.architecture",         m.general_architecture);
    get_str("general.name",                 m.general_name);
    get_u32("general.file_type",            m.file_type);
    get_u32("general.quantization_version", m.quantization_version);
    if (m.file_type >= 0) {
        const char * name = llama_ftype_name(m.file_type);
        if (name) m.file_type_name = name;
    }

    if (!m.general_architecture.empty()) {
        const std::string a = m.general_architecture;
        get_u32((a + ".block_count").c_str(),      m.block_count);
        get_u32((a + ".embedding_length").c_str(), m.embedding_length);
        get_u32((a + ".context_length").c_str(),   m.context_length);
        // vocab_size: prefer the explicit <arch>.vocab_size key. Fall back
        // to the tokenizer token array length (the canonical source on
        // models that don't write the redundant key).
        get_u32((a + ".vocab_size").c_str(),       m.vocab_size);
    }
    if (m.vocab_size < 0) {
        int64_t toks_id = gguf_find_key(gctx, "tokenizer.ggml.tokens");
        if (toks_id >= 0) {
            m.vocab_size = int32_t(gguf_get_arr_n(gctx, toks_id));
        }
    }

    gguf_free(gctx);

    if (compute_sha256) {
        std::string cached;
        if (read_sidecar_sha(path, cached)) {
            m.sha256 = std::move(cached);
        } else {
            std::string hash = sha256_of_file(path);
            if (!hash.empty()) {
                m.sha256 = hash;
                write_sidecar_sha(path, hash);
            }
        }
    }

    return m;
}

}  // namespace dflash::common
