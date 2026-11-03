#include "gguf_inspect.h"
#include "gguf.h"
#include "sha256.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace luce::common {

bool derive_effective_target_layer_count(const std::string & arch,
                                         uint32_t block_count,
                                         uint32_t nextn_predict_layers,
                                         uint32_t & target_layer_count,
                                         std::string & error) {
    target_layer_count = block_count;
    error.clear();

    if (block_count == 0) {
        error = "block_count must be greater than zero";
        return false;
    }

    // Embedded NextN blocks are defined by the Qwen3.5/3.6 and BailingMoE3
    // GGUF layouts. Do not reinterpret similarly named metadata elsewhere.
    if (arch != "qwen35" && arch != "qwen35moe" &&
        arch != "bailingmoe3") {
        return true;
    }
    if (nextn_predict_layers == 0) {
        return true;
    }
    if (nextn_predict_layers >= block_count) {
        error = "nextn_predict_layers must be smaller than block_count";
        return false;
    }

    target_layer_count = block_count - nextn_predict_layers;
    return true;
}

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

    int64_t name_id = gguf_find_key(gctx, "general.name");
    if (name_id >= 0) {
        const char * v = gguf_get_val_str(gctx, name_id);
        if (v) info.name = v;
    }

    int64_t tmpl_id = gguf_find_key(gctx, "tokenizer.chat_template");
    if (tmpl_id >= 0) {
        const char * v = gguf_get_val_str(gctx, tmpl_id);
        if (v) info.chat_template = v;
    }

    // Read target layer count. Qwen3.5/3.6 GGUFs can include trailing
    // embedded MTP blocks in block_count.
    if (!info.arch.empty()) {
        const std::string block_key = info.arch + ".block_count";
        const int64_t block_id = gguf_find_key(gctx, block_key.c_str());
        if (block_id >= 0) {
            const uint32_t block_count = gguf_get_val_u32(gctx, block_id);

            const std::string nextn_key = info.arch + ".nextn_predict_layers";
            const int64_t nextn_id = gguf_find_key(gctx, nextn_key.c_str());
            const uint32_t nextn = nextn_id >= 0
                ? gguf_get_val_u32(gctx, nextn_id)
                : 0;

            uint32_t target_layers = 0;
            std::string error;
            if (derive_effective_target_layer_count(
                    info.arch, block_count, nextn, target_layers, error)) {
                info.n_layer = (int)target_layers;
            }
        }
    }

    gguf_free(gctx);
    return info;
}

// ─── SHA-256 of a model file ────────────────────────────────────────────
//
// Hashing a 17 GB GGUF takes ~30s on a fast NVMe, comparable to the
// per-file numbers `sha256sum` gets. We sidecar the result so this only
// happens on the first server start after a model is downloaded.

namespace {

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
    // If the loop exited on anything other than clean EOF (disk error, etc.),
    // bail rather than return a finalized hash over a partial read — caching
    // that as the model's SHA-256 would silently misidentify the file.
    if (f.bad() || (f.fail() && !f.eof())) return {};
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

// Sidecar layout (extends standard sha256sum format with a validation hint):
//   line 1: "<64-hex>  <basename>\n"   (sha256sum-compatible)
//   line 2: "# size=<bytes>\n"         (our extension; required to trust line 1)
//
// The size guard is what protects us from a stale sidecar after the GGUF was
// replaced/edited in place without the sidecar being updated. We deliberately
// don't trust legacy/external sidecars that lack the size hint — silently
// reporting the wrong model identity at /props is worse than re-hashing once.
bool read_sidecar_sha(const std::string & path, int64_t expected_size, std::string & out) {
    if (expected_size < 0) return false;  // can't validate without a known size
    std::ifstream f(path + ".sha256");
    if (!f) return false;
    std::string hex;
    f >> hex;  // tolerate `<hex>  filename\n` (sha256sum format) — we only want the first token
    if (hex.size() != 64) return false;
    for (char c : hex) {
        bool is_hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        if (!is_hex) return false;
    }
    // Scan the rest of the file for a `# size=<n>` directive. Refuse to
    // trust the cached hash if it's missing or doesn't match — that
    // indicates either a legacy sidecar (pre-validation-guard) or that the
    // underlying GGUF has been replaced since the hash was written.
    std::string line;
    std::getline(f, line);  // consume rest of line 1
    bool size_matches = false;
    while (std::getline(f, line)) {
        // Strip leading whitespace, then look for "# size=" prefix.
        size_t i = 0;
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
        const std::string prefix = "# size=";
        if (line.compare(i, prefix.size(), prefix) != 0) continue;
        const char * num = line.c_str() + i + prefix.size();
        char * end = nullptr;
        long long n = std::strtoll(num, &end, 10);
        if (end == num) continue;
        if (n == (long long)expected_size) size_matches = true;
        break;
    }
    if (!size_matches) return false;
    out = std::move(hex);
    return true;
}

void write_sidecar_sha(const std::string & path, const std::string & sha, int64_t size_bytes) {
    // Best-effort. If the directory isn't writable (read-only mount, model
    // dir owned by another user), we just skip — the in-memory hash is
    // already what /props will report this run.
    std::ofstream f(path + ".sha256");
    if (!f) return;
    // Emit sha256sum-compatible line + our size guard. The basename keeps
    // `sha256sum -c` happy if a human ever runs it against the sidecar.
    std::string base = path;
    auto slash = base.find_last_of('/');
    if (slash != std::string::npos) base = base.substr(slash + 1);
    f << sha << "  " << base << "\n";
    if (size_bytes >= 0) f << "# size=" << size_bytes << "\n";
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
        if (read_sidecar_sha(path, m.size_bytes, cached)) {
            m.sha256 = std::move(cached);
        } else {
            std::string hash = sha256_of_file(path);
            if (!hash.empty()) {
                m.sha256 = hash;
                write_sidecar_sha(path, hash, m.size_bytes);
            }
        }
    }

    return m;
}

}  // namespace luce::common
