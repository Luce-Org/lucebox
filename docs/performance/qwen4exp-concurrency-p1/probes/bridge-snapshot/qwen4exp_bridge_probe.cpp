#include "qwen4exp_bridge_cache.h"
#include "qwen4exp_tail_replay.h"
#include "qwen4exp/qwen4exp_graph.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace luce::common;
using namespace luce::common::qwen4exp_bridge;

namespace {

struct Runtime {
    ggml_backend_t backend = nullptr;
    Qwen4ExpWeights weights;
    Qwen4ExpCache cache;

    ~Runtime() {
        free_qwen4exp_cache(cache);
        free_qwen4exp_weights(weights);
        if (backend) ggml_backend_free(backend);
    }

    bool init(const std::string & model, int max_ctx) {
        backend = ggml_backend_cuda_init(0);
        if (!backend) {
            std::fprintf(stderr, "no GPU backend available\n");
            return false;
        }
        if (!load_qwen4exp_gguf(model, backend, weights)) {
            std::fprintf(stderr, "load_qwen4exp_gguf failed: %s\n",
                         luce_last_error());
            return false;
        }
        if (!create_qwen4exp_cache(backend, weights, max_ctx,
                                   GGML_TYPE_F16, cache)) {
            std::fprintf(stderr, "create_qwen4exp_cache failed\n");
            return false;
        }
        if (cache.input_ring.enabled || cache.input_ring.buf) {
            std::fprintf(stderr, "UMA input ring must be disabled for bridge probe\n");
            return false;
        }
        std::printf("model layers=%d vocab=%d cache_ctx=%d full=%zu linear=%zu ple=%zu\n",
                    weights.n_layer, weights.n_vocab, cache.max_ctx,
                    cache.full_layer_ids.size(), cache.linear_layer_ids.size(),
                    cache.ple_layer_ids.size());
        return true;
    }
};

bool force_probe_environment() {
#if defined(_WIN32)
    return _putenv_s("LUCE_HIP_NO_UMA_RING", "1") == 0 &&
           _putenv_s("QWEN4EXP_DECODE_STABLEGRAPH", "0") == 0;
#else
    return setenv("LUCE_HIP_NO_UMA_RING", "1", 1) == 0 &&
           setenv("QWEN4EXP_DECODE_STABLEGRAPH", "0", 1) == 0;
#endif
}

bool parse_int(const char * text, int & value, bool allow_zero) {
    if (!text || !*text || *text == '-') return false;
    char * end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (!end || *end || parsed > std::numeric_limits<int>::max() ||
        parsed < (allow_zero ? 0 : 1)) return false;
    value = static_cast<int>(parsed);
    return true;
}

bool read_tokens(const std::string & path, std::vector<int32_t> & tokens) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "cannot open token file: %s\n", path.c_str());
        return false;
    }
    const std::streamoff size = in.tellg();
    if (size < 0 || size % 4 != 0 ||
        static_cast<uint64_t>(size) / 4 > std::numeric_limits<size_t>::max()) {
        std::fprintf(stderr, "TOKENS.bin length must be divisible by 4\n");
        return false;
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    in.seekg(0);
    if (!bytes.empty() && !in.read(reinterpret_cast<char *>(bytes.data()), size)) {
        std::fprintf(stderr, "failed to read token file\n");
        return false;
    }
    tokens.resize(bytes.size() / 4);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const size_t at = 4 * i;
        const uint32_t value = static_cast<uint32_t>(bytes[at]) |
            (static_cast<uint32_t>(bytes[at + 1]) << 8) |
            (static_cast<uint32_t>(bytes[at + 2]) << 16) |
            (static_cast<uint32_t>(bytes[at + 3]) << 24);
        tokens[i] = static_cast<int32_t>(value);
    }
    return true;
}

bool valid_tokens(const std::vector<int32_t> & tokens, int vocab) {
    for (int32_t token : tokens)
        if (token < 0 || token >= vocab) return false;
    return true;
}

bool read_file(const std::string & path, std::vector<uint8_t> & bytes) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    const std::streamoff size = in.tellg();
    if (size < 0 || static_cast<uint64_t>(size) > std::numeric_limits<size_t>::max())
        return false;
    bytes.resize(static_cast<size_t>(size));
    in.seekg(0);
    return bytes.empty() || static_cast<bool>(in.read(
        reinterpret_cast<char *>(bytes.data()), size));
}

bool write_file(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    return out && (bytes.empty() || static_cast<bool>(out.write(
        reinterpret_cast<const char *>(bytes.data()), bytes.size())));
}

bool publish_file(const std::string & path, const std::vector<uint8_t> & bytes) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::exists(path, ec) || ec) return false;
    const std::string temporary = path + ".tmp." + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    if (!write_file(temporary, bytes)) {
        fs::remove(temporary, ec);
        return false;
    }
    // A hard-link publication fails rather than replacing a concurrently
    // created result. The temporary lives beside the output, on one filesystem.
    fs::create_hard_link(temporary, path, ec);
    std::error_code remove_error;
    fs::remove(temporary, remove_error);
    return !ec;
}

uint64_t fnv1a64(const std::vector<uint8_t> & bytes) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint8_t byte : bytes) {
        hash ^= byte;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

bool read_exact(std::istream & in, void * data, size_t bytes) {
    return bytes <= static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) &&
           (bytes == 0 || static_cast<bool>(in.read(static_cast<char *>(data),
                                                     static_cast<std::streamsize>(bytes))));
}

bool skip_exact(std::istream & in, uint64_t bytes) {
    std::array<char, 64 * 1024> scratch{};
    while (bytes != 0) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(bytes,
                                                                    scratch.size()));
        if (!read_exact(in, scratch.data(), chunk)) return false;
        bytes -= chunk;
    }
    return true;
}

bool read_u32(std::istream & in, uint32_t & value) {
    uint8_t bytes[4];
    if (!read_exact(in, bytes, sizeof(bytes))) return false;
    value = static_cast<uint32_t>(bytes[0]) |
            (static_cast<uint32_t>(bytes[1]) << 8) |
            (static_cast<uint32_t>(bytes[2]) << 16) |
            (static_cast<uint32_t>(bytes[3]) << 24);
    return true;
}

bool read_u64(std::istream & in, uint64_t & value) {
    uint8_t bytes[8];
    if (!read_exact(in, bytes, sizeof(bytes))) return false;
    value = 0;
    for (unsigned i = 0; i < 8; ++i)
        value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    return true;
}

bool read_string(std::istream & in, uint32_t limit, std::string & value) {
    uint32_t size = 0;
    if (!read_u32(in, size) || size == 0 || size > limit) return false;
    value.resize(size);
    return read_exact(in, value.data(), size);
}

struct SourceSpec {
    std::string type;
    std::array<uint32_t, 4> ne;
    bool keep = false;
};

bool same_checkpoint(const Checkpoint & a, const Checkpoint & b);

bool source_checkpoint(const std::string & path,
                       const std::vector<int32_t> & tokens,
                       Checkpoint & target, std::string & error) {
    constexpr int kSourceFullMap[] = {0, 1, 3, 4, 5, 7, 8, 10, 11, 12, 14, 15};
    constexpr int kSourceLinearMap[] = {
        0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 23,
        24, 26, 27, 28, 30, 31, 32, 34, 35, 36, 38, 39, 40, 42, 43, 44, 46, 47};
    const std::set<int> kept_full(std::begin(kSourceFullMap), std::end(kSourceFullMap));
    const std::set<int> kept_linear(std::begin(kSourceLinearMap),
                                    std::end(kSourceLinearMap));

    std::ifstream in(path, std::ios::binary);
    char magic[8];
    uint32_t cut = 0, prompt_len = 0, count = 0;
    if (!in || !read_exact(in, magic, sizeof(magic)) ||
        std::memcmp(magic, "LBSNAP01", sizeof(magic)) != 0 ||
        !read_u32(in, cut) || !read_u32(in, prompt_len) || !read_u32(in, count) ||
        cut == 0 || cut > prompt_len || count != 129) {
        error = "invalid pinned source checkpoint header";
        return false;
    }
    if (tokens.size() != prompt_len) {
        error = "TOKENS.bin count does not match source prompt_len";
        return false;
    }

    std::map<std::string, SourceSpec> expected;
    for (int ordinal = 0; ordinal < 16; ++ordinal) {
        const SourceSpec spec{"f16", {256, cut, 4, 1}, kept_full.count(ordinal) != 0};
        expected.emplace("snap_cache_k_" + std::to_string(ordinal), spec);
        expected.emplace("snap_cache_v_" + std::to_string(ordinal), spec);
    }
    for (int ordinal = 0; ordinal < 48; ++ordinal) {
        const bool keep = kept_linear.count(ordinal) != 0;
        expected.emplace("snap_ssm_state_" + std::to_string(ordinal),
                         SourceSpec{"f32", {128, 128, 48, 1}, keep});
        expected.emplace("snap_conv_state_" + std::to_string(ordinal),
                         SourceSpec{"f32", {3, 10240, 1, 1}, keep});
    }
    expected.emplace("snap_target_feat",
                     SourceSpec{"bf16", {25600, std::min<uint32_t>(cut, 4096), 1, 1}, false});

    std::map<std::string, Tensor> kept;
    std::set<std::string> seen;
    for (uint32_t i = 0; i < count; ++i) {
        std::string name, type;
        uint32_t ndim = 0;
        std::array<uint32_t, 4> ne{};
        uint64_t bytes = 0;
        if (!read_string(in, 256, name) || !read_string(in, 8, type) ||
            !read_u32(in, ndim) || ndim != 4) {
            error = "truncated source tensor header";
            return false;
        }
        for (uint32_t & extent : ne)
            if (!read_u32(in, extent)) {
                error = "truncated source tensor shape";
                return false;
            }
        if (!read_u64(in, bytes)) {
            error = "truncated source tensor byte count";
            return false;
        }
        const auto found = expected.find(name);
        if (found == expected.end() || !seen.insert(name).second ||
            type != found->second.type || ne != found->second.ne) {
            error = "unrecognized, duplicate, or malformed source tensor: " + name;
            return false;
        }
        uint64_t elements = 1;
        for (uint32_t extent : ne) elements *= extent;
        const uint64_t expected_bytes = elements * (type == "f32" ? 4 : 2);
        if (bytes != expected_bytes || bytes > std::numeric_limits<size_t>::max()) {
            error = "source tensor payload size mismatch: " + name;
            return false;
        }
        if (found->second.keep) {
            Tensor tensor{name, type, ne, std::vector<uint8_t>(static_cast<size_t>(bytes))};
            if (!read_exact(in, tensor.data.data(), tensor.data.size())) {
                error = "truncated source tensor payload: " + name;
                return false;
            }
            kept.emplace(name, std::move(tensor));
        } else {
            if (!skip_exact(in, bytes)) {
                error = "truncated source tensor payload: " + name;
                return false;
            }
        }
    }
    if (seen.size() != expected.size() || in.peek() != std::char_traits<char>::eof()) {
        error = "missing source tensor or trailing checkpoint bytes";
        return false;
    }

    target.cut = cut;
    target.prompt_len = prompt_len;
    target.indexer_blocks = -1;
    target.ple_layer_ids = {1};
    for (int layer = 0; layer < 48; ++layer) {
        if (layer >= 3 && (layer - 3) % 4 == 0) target.full_layer_ids.push_back(layer);
        else target.linear_layer_ids.push_back(layer);
    }
    auto take = [&](const std::string & source, const std::string & name) {
        Tensor tensor = std::move(kept.at(source));
        tensor.name = name;
        target.tensors.push_back(std::move(tensor));
    };
    for (size_t ordinal = 0; ordinal < target.full_layer_ids.size(); ++ordinal) {
        const int source_ordinal = kSourceFullMap[ordinal];
        const int target_layer = target.full_layer_ids[ordinal];
        for (const char * kind : {"k", "v"}) {
            const std::string source = std::string("snap_cache_") + kind + "_" +
                                       std::to_string(source_ordinal);
            const Tensor & input = kept.at(source);
            const size_t head_bytes = static_cast<size_t>(256) * cut * 2;
            Tensor output;
            output.name = std::string("qwen4exp.attn_") + kind + "." +
                          std::to_string(target_layer);
            output.type = "f16";
            output.ne = {256, cut, 2, 1};
            output.data.reserve(2 * head_bytes);
            output.data.insert(output.data.end(), input.data.begin(),
                               input.data.begin() + head_bytes);
            output.data.insert(output.data.end(), input.data.begin() + 2 * head_bytes,
                               input.data.begin() + 3 * head_bytes);
            target.tensors.push_back(std::move(output));
        }
    }
    for (size_t ordinal = 0; ordinal < target.linear_layer_ids.size(); ++ordinal) {
        const int source_ordinal = kSourceLinearMap[ordinal];
        const int target_layer = target.linear_layer_ids[ordinal];
        take("snap_ssm_state_" + std::to_string(source_ordinal),
             "qwen4exp.ssm_state." + std::to_string(target_layer));
        take("snap_conv_state_" + std::to_string(source_ordinal),
             "qwen4exp.conv_state." + std::to_string(target_layer));
    }
    Tensor ple;
    ple.name = "qwen4exp.ple_conv_state.1";
    ple.type = "f32";
    ple.ne = {9, 10240, 1, 1};
    ple.data.assign(static_cast<size_t>(9) * 10240 * 4, 0);
    target.tensors.push_back(std::move(ple));
    Tensor previous;
    previous.name = "qwen4exp.ple_prev";
    previous.type = "i32";
    const uint32_t history = std::min<uint32_t>(cut, 2);
    previous.ne = {history, 1, 1, 1};
    for (uint32_t i = cut - history; i < cut; ++i) {
        const uint32_t token = static_cast<uint32_t>(tokens[i]);
        for (unsigned shift = 0; shift < 32; shift += 8)
            previous.data.push_back(static_cast<uint8_t>(token >> shift));
    }
    target.tensors.push_back(std::move(previous));
    return true;
}

bool translate_smoke_mode(const std::string & source_path,
                          const std::string & token_path,
                          const std::string & output_path) {
    std::error_code exists_error;
    if (std::filesystem::exists(output_path, exists_error) || exists_error) {
        std::fprintf(stderr, "refusing to overwrite output: %s\n", output_path.c_str());
        return false;
    }
    std::vector<int32_t> tokens;
    Checkpoint checkpoint, decoded;
    std::string error;
    if (!read_tokens(token_path, tokens) || !valid_tokens(tokens, 248320) ||
        !source_checkpoint(source_path, tokens, checkpoint, error)) {
        std::fprintf(stderr, "source translation input invalid: %s\n", error.c_str());
        return false;
    }
    const std::vector<uint8_t> bytes = encode(checkpoint, &error);
    if (bytes.empty() || !decode(bytes, decoded, &error) ||
        !same_checkpoint(checkpoint, decoded)) {
        std::fprintf(stderr, "translated checkpoint self-check failed: %s\n", error.c_str());
        return false;
    }
    if (!publish_file(output_path, bytes)) {
        std::fprintf(stderr, "refusing to overwrite or publish output: %s\n",
                     output_path.c_str());
        return false;
    }
    std::printf("TRANSLATE_SMOKE structural_only=yes semantic_parity_claim=no "
                "cut=%u prompt_len=%u tensors=%zu indexer_blocks=%d bytes=%zu path=%s\n",
                checkpoint.cut, checkpoint.prompt_len, checkpoint.tensors.size() + 1,
                checkpoint.indexer_blocks, bytes.size(), output_path.c_str());
    return true;
}

bool compose_kv_mode(const std::string & native_path,
                     const std::string & donor_path,
                     const std::string & output_path) {
    std::vector<uint8_t> native_bytes, donor_bytes;
    Checkpoint native, donor, composed, decoded;
    std::string error;
    if (!read_file(native_path, native_bytes) ||
        !decode(native_bytes, native, &error)) {
        std::fprintf(stderr, "invalid native checkpoint: %s\n", error.c_str());
        return false;
    }
    if (!read_file(donor_path, donor_bytes) ||
        !decode(donor_bytes, donor, &error)) {
        std::fprintf(stderr, "invalid donor checkpoint: %s\n", error.c_str());
        return false;
    }
    size_t replacements = 0;
    if (!compose_kv(native, donor, composed, &replacements, &error)) {
        std::fprintf(stderr, "K/V composition rejected: %s\n", error.c_str());
        return false;
    }
    const std::vector<uint8_t> output_bytes = encode(composed, &error);
    if (output_bytes.empty() || !decode(output_bytes, decoded, &error) ||
        !same_checkpoint(composed, decoded)) {
        std::fprintf(stderr, "composed checkpoint self-check failed: %s\n",
                     error.c_str());
        return false;
    }
    if (!publish_file(output_path, output_bytes)) {
        std::fprintf(stderr, "refusing to overwrite or publish output: %s\n",
                     output_path.c_str());
        return false;
    }
    std::printf("COMPOSE_KV host_only=yes rope_transform=no replacements=%zu "
                "native_fnv1a64=%016llx donor_fnv1a64=%016llx "
                "output_fnv1a64=%016llx bytes=%zu path=%s\n",
                replacements,
                static_cast<unsigned long long>(fnv1a64(native_bytes)),
                static_cast<unsigned long long>(fnv1a64(donor_bytes)),
                static_cast<unsigned long long>(fnv1a64(output_bytes)),
                output_bytes.size(), output_path.c_str());
    return true;
}

bool forward_chunks(Runtime & rt, const int32_t * tokens, int count, int pos,
                    std::vector<float> & logits) {
    constexpr int kChunk = 2048;
    int done = 0;
    while (done < count) {
        const int n = std::min(kChunk, count - done);
        const Qwen4ExpForwardResult result = qwen4exp_forward(
            rt.backend, rt.weights, rt.cache, tokens + done, n, pos + done, logits);
        if (!result.ok || logits.size() != static_cast<size_t>(rt.weights.n_vocab)) {
            std::fprintf(stderr, "qwen4exp_forward failed at pos=%d count=%d\n",
                         pos + done, n);
            return false;
        }
        done += n;
    }
    return true;
}

bool prefill(Runtime & rt, const std::vector<int32_t> & tokens, size_t count,
             std::vector<float> & logits) {
    if (count == 0 || count > static_cast<size_t>(std::numeric_limits<int>::max()))
        return false;
    return forward_chunks(rt, tokens.data(), static_cast<int>(count), 0, logits);
}

bool all_finite(const std::vector<float> & values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

int argmax(const std::vector<float> & values) {
    return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

struct GreedyRun {
    std::vector<int32_t> tokens;
    std::vector<float> seed_logits;
};

bool continue_greedy(Runtime & rt, const int32_t * suffix,
                     int cut, int count, const std::vector<float> & seed,
                     int generated, GreedyRun & out) {
    std::vector<float> logits = seed;
    if (count > 0 && !forward_chunks(rt, suffix, count, cut, logits))
        return false;
    if (!all_finite(logits)) return false;
    out.seed_logits = logits;
    out.tokens.clear();
    out.tokens.reserve(static_cast<size_t>(generated));
    int pos = cut + count;
    for (int i = 0; i < generated; ++i) {
        const int32_t next = argmax(logits);
        out.tokens.push_back(next);
        if (next == rt.weights.eos_id || next == rt.weights.eos_chat_id ||
            i + 1 == generated) break;
        if (!forward_chunks(rt, &next, 1, pos++, logits) || !all_finite(logits))
            return false;
    }
    return true;
}

bool same_checkpoint(const Checkpoint & a, const Checkpoint & b) {
    if (a.cut != b.cut || a.prompt_len != b.prompt_len ||
        a.indexer_blocks != b.indexer_blocks ||
        a.full_layer_ids != b.full_layer_ids ||
        a.linear_layer_ids != b.linear_layer_ids ||
        a.indexer_layer_ids != b.indexer_layer_ids ||
        a.ple_layer_ids != b.ple_layer_ids || a.tensors.size() != b.tensors.size())
        return false;
    for (size_t i = 0; i < a.tensors.size(); ++i) {
        const Tensor & x = a.tensors[i];
        const Tensor & y = b.tensors[i];
        if (x.name != y.name || x.type != y.type || x.ne != y.ne || x.data != y.data)
            return false;
    }
    return true;
}

enum AblationBits : unsigned {
    kAblateNone = 0,
    kAblateRec  = 1U << 0,
    kAblatePle  = 1U << 1,
    kAblateKv   = 1U << 2,
    kAblateSsm  = 1U << 3,
    kAblateConv = 1U << 4,
};

struct AblationCase {
    const char * name;
    unsigned bits;
};

constexpr AblationCase kAblations[] = {
    {"none", kAblateNone},
    {"rec", kAblateRec},
    {"ple", kAblatePle},
    {"rec+ple", kAblateRec | kAblatePle},
    {"kv-sensitivity", kAblateKv},
};

constexpr AblationCase kRecurrentSplitAblations[] = {
    {"none", kAblateNone},
    {"conv-only", kAblateConv},
    {"ssm-only", kAblateSsm},
    {"kv-sensitivity", kAblateKv},
};

bool prefix(const std::string & value, const char * expected) {
    return value.compare(0, std::strlen(expected), expected) == 0;
}

bool canonicalize_history(Checkpoint & checkpoint,
                          const std::vector<int32_t> & tokens,
                          std::string & error) {
    if (checkpoint.cut > tokens.size()) {
        error = "checkpoint cut exceeds canonical token stream";
        return false;
    }
    const auto found = std::find_if(checkpoint.tensors.begin(), checkpoint.tensors.end(),
        [](const Tensor & tensor) { return tensor.name == "qwen4exp.ple_prev"; });
    const uint32_t count = std::min<uint32_t>(checkpoint.cut, 2);
    if (found == checkpoint.tensors.end() || found->type != "i32" ||
        found->ne != std::array<uint32_t, 4>{count, 1, 1, 1}) {
        error = "checkpoint PLE history is not the pinned two-token shape";
        return false;
    }
    found->data.clear();
    for (uint32_t i = checkpoint.cut - count; i < checkpoint.cut; ++i) {
        const uint32_t token = static_cast<uint32_t>(tokens[i]);
        for (unsigned shift = 0; shift < 32; shift += 8)
            found->data.push_back(static_cast<uint8_t>(token >> shift));
    }
    return true;
}

bool ablate_checkpoint(const Checkpoint & native,
                       const std::vector<int32_t> & tokens,
                       const AblationCase & ablation,
                       Checkpoint & output, std::string & error) {
    output = native;
    if (!canonicalize_history(output, tokens, error)) return false;
    size_t selected = 0;
    for (Tensor & tensor : output.tensors) {
        const bool ssm = prefix(tensor.name, "qwen4exp.ssm_state.");
        const bool conv = prefix(tensor.name, "qwen4exp.conv_state.");
        const bool recurrent = ssm || conv;
        const bool ple = prefix(tensor.name, "qwen4exp.ple_conv_state.");
        const bool kv = prefix(tensor.name, "qwen4exp.attn_k.") ||
                        prefix(tensor.name, "qwen4exp.attn_v.");
        const bool zero = ((ablation.bits & kAblateRec) && recurrent) ||
                          ((ablation.bits & kAblateSsm) && ssm) ||
                          ((ablation.bits & kAblateConv) && conv) ||
                          ((ablation.bits & kAblatePle) && ple) ||
                          ((ablation.bits & kAblateKv) && kv);
        if (zero) {
            std::fill(tensor.data.begin(), tensor.data.end(), 0);
            ++selected;
        }
    }
    if ((ablation.bits != 0 && selected == 0) || !validate(output, &error)) {
        if (error.empty()) error = "ablation selected no checkpoint tensors";
        return false;
    }

    // Fail closed on scope: only the named payload families may differ. Metadata,
    // QSA indexer, and canonical PLE host history remain native.
    if (native.cut != output.cut || native.prompt_len != output.prompt_len ||
        native.indexer_blocks != output.indexer_blocks ||
        native.full_layer_ids != output.full_layer_ids ||
        native.linear_layer_ids != output.linear_layer_ids ||
        native.indexer_layer_ids != output.indexer_layer_ids ||
        native.ple_layer_ids != output.ple_layer_ids ||
        native.tensors.size() != output.tensors.size()) {
        error = "ablation changed checkpoint metadata";
        return false;
    }
    for (size_t i = 0; i < native.tensors.size(); ++i) {
        const Tensor & before = native.tensors[i];
        const Tensor & after = output.tensors[i];
        const bool selected_family =
            ((ablation.bits & kAblateRec) &&
             (prefix(before.name, "qwen4exp.ssm_state.") ||
              prefix(before.name, "qwen4exp.conv_state."))) ||
            ((ablation.bits & kAblateSsm) &&
             prefix(before.name, "qwen4exp.ssm_state.")) ||
            ((ablation.bits & kAblateConv) &&
             prefix(before.name, "qwen4exp.conv_state.")) ||
            ((ablation.bits & kAblatePle) &&
             prefix(before.name, "qwen4exp.ple_conv_state.")) ||
            ((ablation.bits & kAblateKv) &&
             (prefix(before.name, "qwen4exp.attn_k.") ||
              prefix(before.name, "qwen4exp.attn_v.")));
        if (before.name != after.name || before.type != after.type ||
            before.ne != after.ne ||
            (selected_family ? !std::all_of(after.data.begin(), after.data.end(),
                                             [](uint8_t value) { return value == 0; })
                             : before.data != after.data)) {
            error = "ablation changed a tensor outside its declared scope";
            return false;
        }
    }
    error.clear();
    return true;
}

size_t element_size(const std::string & type) {
    return type == "f16" ? 2 : 4;
}

uint32_t raw_element(const Tensor & tensor, size_t index) {
    uint32_t raw = 0;
    std::memcpy(&raw, tensor.data.data() + index * element_size(tensor.type),
                element_size(tensor.type));
    return raw;
}

double float_element(const Tensor & tensor, uint32_t raw) {
    if (tensor.type == "f16")
        return ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(raw));
    float value = 0.0f;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

int compare_mode(const std::string & a_path, const std::string & b_path) {
    std::vector<uint8_t> a_bytes, b_bytes;
    Checkpoint a, b;
    std::string error;
    if (!read_file(a_path, a_bytes)) {
        std::fprintf(stderr, "cannot read checkpoint: %s\n", a_path.c_str());
        return 2;
    }
    if (!decode(a_bytes, a, &error)) {
        std::fprintf(stderr, "invalid checkpoint %s: %s\n", a_path.c_str(),
                     error.c_str());
        return 2;
    }
    if (!read_file(b_path, b_bytes)) {
        std::fprintf(stderr, "cannot read checkpoint: %s\n", b_path.c_str());
        return 2;
    }
    if (!decode(b_bytes, b, &error)) {
        std::fprintf(stderr, "invalid checkpoint %s: %s\n", b_path.c_str(),
                     error.c_str());
        return 2;
    }

    bool equal = true;
    if (a.cut != b.cut || a.prompt_len != b.prompt_len ||
        a.indexer_blocks != b.indexer_blocks ||
        a.full_layer_ids != b.full_layer_ids ||
        a.linear_layer_ids != b.linear_layer_ids ||
        a.indexer_layer_ids != b.indexer_layer_ids ||
        a.ple_layer_ids != b.ple_layer_ids) {
        std::printf("METADATA differs cut=%u/%u prompt_len=%u/%u "
                    "indexer_blocks=%d/%d\n", a.cut, b.cut, a.prompt_len,
                    b.prompt_len, a.indexer_blocks, b.indexer_blocks);
        equal = false;
    }

    std::map<std::string, const Tensor *> a_tensors, b_tensors;
    for (const Tensor & tensor : a.tensors) a_tensors.emplace(tensor.name, &tensor);
    for (const Tensor & tensor : b.tensors) b_tensors.emplace(tensor.name, &tensor);

    size_t differing_tensors = 0;
    size_t mismatched_elements = 0;
    size_t mismatched_bytes = 0;
    for (const auto & [name, x] : a_tensors) {
        const auto found = b_tensors.find(name);
        if (found == b_tensors.end()) {
            std::printf("TENSOR_SCHEMA name=%s missing_from=B\n", name.c_str());
            ++differing_tensors;
            equal = false;
            continue;
        }
        const Tensor & y = *found->second;
        if (x->type != y.type || x->ne != y.ne) {
            std::printf("TENSOR_SCHEMA name=%s type=%s/%s "
                        "shape=%u,%u,%u,%u/%u,%u,%u,%u\n", name.c_str(),
                        x->type.c_str(), y.type.c_str(), x->ne[0], x->ne[1],
                        x->ne[2], x->ne[3], y.ne[0], y.ne[1], y.ne[2], y.ne[3]);
            ++differing_tensors;
            equal = false;
            continue;
        }
        if (x->data == y.data) continue;

        const size_t width = element_size(x->type);
        const size_t count = x->data.size() / width;
        size_t tensor_mismatches = 0;
        size_t tensor_byte_mismatches = 0;
        size_t first = count;
        uint32_t first_a = 0, first_b = 0;
        double max_abs = 0.0;
        for (size_t i = 0; i < count; ++i) {
            const uint32_t raw_a = raw_element(*x, i);
            const uint32_t raw_b = raw_element(y, i);
            if (raw_a == raw_b) continue;
            if (first == count) {
                first = i;
                first_a = raw_a;
                first_b = raw_b;
            }
            ++tensor_mismatches;
            if (x->type == "f16" || x->type == "f32") {
                const double diff = std::abs(float_element(*x, raw_a) -
                                             float_element(y, raw_b));
                if (std::isnan(diff)) max_abs = diff;
                else if (!std::isnan(max_abs)) max_abs = std::max(max_abs, diff);
            }
        }
        for (size_t i = 0; i < x->data.size(); ++i)
            tensor_byte_mismatches += x->data[i] != y.data[i];

        std::printf("TENSOR name=%s elements=%zu mismatched_elements=%zu "
                    "byte_mismatches=%zu first_index=%zu first_a=0x%0*x "
                    "first_b=0x%0*x", name.c_str(), count, tensor_mismatches,
                    tensor_byte_mismatches, first, static_cast<int>(width * 2),
                    static_cast<unsigned>(first_a),
                    static_cast<int>(width * 2), static_cast<unsigned>(first_b));
        if (x->type == "f16" || x->type == "f32")
            std::printf(" max_abs=%.9g", max_abs);
        std::printf("\n");
        ++differing_tensors;
        mismatched_elements += tensor_mismatches;
        mismatched_bytes += tensor_byte_mismatches;
        equal = false;
    }
    for (const auto & [name, tensor] : b_tensors) {
        (void) tensor;
        if (a_tensors.count(name) == 0) {
            std::printf("TENSOR_SCHEMA name=%s missing_from=A\n", name.c_str());
            ++differing_tensors;
            equal = false;
        }
    }
    std::printf("COMPARE equal=%s tensors=%zu/%zu differing_tensors=%zu "
                "mismatched_elements=%zu byte_mismatches=%zu\n",
                equal ? "yes" : "no", a.tensors.size(), b.tensors.size(),
                differing_tensors, mismatched_elements, mismatched_bytes);
    return equal ? 0 : 1;
}

void logit_diff(const std::vector<float> & a, const std::vector<float> & b,
                double & max_abs, size_t & exact_mismatches) {
    max_abs = 0.0;
    exact_mismatches = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = std::abs(static_cast<double>(a[i]) - b[i]);
        max_abs = std::max(max_abs, diff);
        exact_mismatches += a[i] != b[i];
    }
}

struct RepairTrace {
    Checkpoint at_1024;
    Checkpoint post_32;
    std::vector<float> full_logits;
    std::vector<std::vector<float>> step_logits;
    std::vector<int32_t> greedy_tokens;
};

struct RepairDiff {
    bool state_1024_equal = false;
    bool post_state_equal = false;
    bool tokens_equal = false;
    size_t full_logit_mismatches = 0;
    size_t step_logit_mismatches = 0;
    double full_logit_max_abs = 0.0;
    double step_logit_max_abs = 0.0;

    bool exact() const {
        return state_1024_equal && post_state_equal && tokens_equal &&
               full_logit_mismatches == 0 && step_logit_mismatches == 0;
    }
};

bool capture_exact(Runtime & rt, uint32_t cut, uint32_t prompt_len,
                   Checkpoint & checkpoint, std::string & error) {
    return capture(rt.backend, rt.weights, rt.cache, cut, prompt_len,
                   checkpoint, &error);
}

bool run_repair_trace(Runtime & rt, const Checkpoint & start,
                      const std::vector<int32_t> & canonical,
                      RepairTrace & out, std::string & error,
                      const std::vector<int32_t> * forced_tokens = nullptr) {
    constexpr uint32_t kEnd = 1024;
    constexpr int kGenerated = 32;
    if (start.cut >= kEnd || canonical.size() != 2048) {
        error = "repair trace requires a 2048-token stream and B below 1024";
        return false;
    }
    if (!restore(rt.backend, rt.weights, rt.cache, start, &error)) return false;
    Checkpoint immediate;
    if (!capture_exact(rt, start.cut, static_cast<uint32_t>(canonical.size()),
                       immediate, error) || !same_checkpoint(start, immediate)) {
        error = "immediate post-restore checkpoint differs from branch input";
        return false;
    }

    std::vector<float> logits;
    const int repair = static_cast<int>(kEnd - start.cut);
    if (!forward_chunks(rt, canonical.data() + start.cut, repair,
                        static_cast<int>(start.cut), logits) || !all_finite(logits)) {
        error = "repair forward failed or produced nonfinite logits";
        return false;
    }
    // The QSA graph writes indexer columns only for chunks of at least 128
    // tokens. R32/R1 therefore preserve the exact B checkpoint prefix (248),
    // unlike a fresh T=1024 prefill, which writes all 256 columns at once.
    if (rt.cache.indexer_blocks != start.indexer_blocks ||
        !capture_exact(rt, kEnd, static_cast<uint32_t>(canonical.size()),
                       out.at_1024, error)) {
        if (error.empty()) {
            error = "sub-128 repair changed QSA indexer prefix: start=" +
                std::to_string(start.indexer_blocks) + " actual=" +
                std::to_string(rt.cache.indexer_blocks) + " repair=" +
                std::to_string(repair);
        }
        return false;
    }
    out.full_logits = logits;
    out.step_logits.clear();
    out.greedy_tokens.clear();
    out.step_logits.reserve(kGenerated);
    out.greedy_tokens.reserve(kGenerated);
    if (forced_tokens != nullptr && forced_tokens->size() != kGenerated) {
        error = "forced continuation must contain exactly 32 tokens";
        return false;
    }
    int position = static_cast<int>(kEnd);
    for (int step = 0; step < kGenerated; ++step) {
        const int32_t next = argmax(logits);
        out.greedy_tokens.push_back(next);
        const int32_t forwarded = forced_tokens == nullptr ? next : (*forced_tokens)[step];
        if (!forward_chunks(rt, &forwarded, 1, position++, logits) ||
            !all_finite(logits)) {
            error = "generated continuation failed or produced nonfinite logits";
            return false;
        }
        out.step_logits.push_back(logits);
    }
    if (!capture_exact(rt, kEnd + kGenerated,
                       static_cast<uint32_t>(canonical.size()),
                       out.post_32, error)) return false;
    error.clear();
    return true;
}

RepairDiff compare_repair_trace(const RepairTrace & oracle,
                                const RepairTrace & candidate) {
    RepairDiff diff;
    diff.state_1024_equal = same_checkpoint(oracle.at_1024, candidate.at_1024);
    diff.post_state_equal = same_checkpoint(oracle.post_32, candidate.post_32);
    diff.tokens_equal = oracle.greedy_tokens == candidate.greedy_tokens;
    logit_diff(oracle.full_logits, candidate.full_logits,
               diff.full_logit_max_abs, diff.full_logit_mismatches);
    if (oracle.step_logits.size() != candidate.step_logits.size()) {
        diff.step_logit_mismatches = std::numeric_limits<size_t>::max();
        diff.step_logit_max_abs = std::numeric_limits<double>::infinity();
        return diff;
    }
    for (size_t step = 0; step < oracle.step_logits.size(); ++step) {
        double max_abs = 0.0;
        size_t mismatches = 0;
        logit_diff(oracle.step_logits[step], candidate.step_logits[step],
                   max_abs, mismatches);
        diff.step_logit_max_abs = std::max(diff.step_logit_max_abs, max_abs);
        if (diff.step_logit_mismatches <=
            std::numeric_limits<size_t>::max() - mismatches)
            diff.step_logit_mismatches += mismatches;
        else
            diff.step_logit_mismatches = std::numeric_limits<size_t>::max();
    }
    return diff;
}

bool repair_ablate_mode(const std::string & model,
                        const std::string & token_path) {
    constexpr uint32_t kPrimaryBase = 992;
    constexpr uint32_t kDiagnosticBase = 1023;
    constexpr uint32_t kPromptCut = 1024;
    constexpr int kRepeats = 2;
    std::vector<int32_t> tokens;
    if (!read_tokens(token_path, tokens) || tokens.size() != 2048) {
        std::fprintf(stderr, "repair ablation requires exactly 2048 canonical tokens\n");
        return false;
    }
    Runtime rt;
    if (!rt.init(model, 1056) || !valid_tokens(tokens, rt.weights.n_vocab)) return false;

    std::vector<float> logits;
    if (!prefill(rt, tokens, kPrimaryBase, logits) || !all_finite(logits)) return false;
    std::string error;
    Checkpoint base_992;
    if (!capture_exact(rt, kPrimaryBase, static_cast<uint32_t>(tokens.size()),
                       base_992, error) || base_992.indexer_blocks != 248) {
        std::fprintf(stderr, "B992 capture failed: %s\n", error.c_str());
        return false;
    }
    Checkpoint canonical_992;
    if (!ablate_checkpoint(base_992, tokens, kAblations[0], canonical_992, error) ||
        !same_checkpoint(base_992, canonical_992)) {
        std::fprintf(stderr, "B992 PLE history is not canonical: %s\n", error.c_str());
        return false;
    }

    // The R1 diagnostic must be derived from the same native C992 checkpoint,
    // not from an independent 1023-token prefill schedule.
    if (!restore(rt.backend, rt.weights, rt.cache, base_992, &error) ||
        !forward_chunks(rt, tokens.data() + kPrimaryBase,
                        static_cast<int>(kDiagnosticBase - kPrimaryBase),
                        static_cast<int>(kPrimaryBase), logits) || !all_finite(logits)) {
        std::fprintf(stderr, "native B1023 derivation failed: %s\n", error.c_str());
        return false;
    }
    Checkpoint base_1023;
    if (!capture_exact(rt, kDiagnosticBase, static_cast<uint32_t>(tokens.size()),
                       base_1023, error)) {
        std::fprintf(stderr, "B1023 capture failed: %s\n", error.c_str());
        return false;
    }
    Checkpoint canonical_1023;
    if (!ablate_checkpoint(base_1023, tokens, kAblations[0], canonical_1023, error) ||
        !same_checkpoint(base_1023, canonical_1023)) {
        std::fprintf(stderr, "B1023 PLE history is not canonical: %s\n", error.c_str());
        return false;
    }

    struct Boundary {
        const char * name;
        const Checkpoint * start;
        int repair;
        RepairTrace oracle;
        bool native_aa_exact = false;
    } boundaries[] = {
        {"B992_R32_C1024", &base_992, 32, {}, false},
        {"B1023_R1_C1024", &base_1023, 1, {}, false},
    };
    for (Boundary & boundary : boundaries) {
        if (!run_repair_trace(rt, *boundary.start, tokens, boundary.oracle, error)) {
            std::fprintf(stderr, "native oracle failed boundary=%s: %s\n",
                         boundary.name, error.c_str());
            return false;
        }
        RepairTrace repeated_oracle;
        if (!run_repair_trace(rt, *boundary.start, tokens, repeated_oracle, error)) {
            std::fprintf(stderr, "native A/A failed boundary=%s: %s\n",
                         boundary.name, error.c_str());
            return false;
        }
        boundary.native_aa_exact = compare_repair_trace(
            boundary.oracle, repeated_oracle).exact();
        std::printf("REPAIR_ABLATE_NATIVE_AA boundary=%s repeats=2 exact=%s\n",
                    boundary.name, boundary.native_aa_exact ? "yes" : "no");
    }

    bool no_op_exact = true;
    bool sensitivity_detected = true;
    bool mandatory_exact = true;
    for (Boundary & boundary : boundaries) {
        for (const AblationCase & ablation : kAblations) {
            Checkpoint branch;
            if (!ablate_checkpoint(*boundary.start, tokens, ablation, branch, error)) {
                std::fprintf(stderr, "branch construction failed boundary=%s mask=%s: %s\n",
                             boundary.name, ablation.name, error.c_str());
                return false;
            }
            RepairTrace first;
            for (int repeat = 0; repeat < kRepeats; ++repeat) {
                RepairTrace candidate;
                if (!run_repair_trace(rt, branch, tokens, candidate, error)) {
                    std::fprintf(stderr,
                                 "branch failed boundary=%s mask=%s repeat=%d: %s\n",
                                 boundary.name, ablation.name, repeat + 1, error.c_str());
                    return false;
                }
                const RepairDiff oracle_diff = compare_repair_trace(
                    boundary.oracle, candidate);
                const RepairDiff repeat_diff = repeat == 0 ? RepairDiff{
                    true, true, true, 0, 0, 0.0, 0.0} :
                    compare_repair_trace(first, candidate);
                const bool is_primary = boundary.repair == 32;
                const bool is_mandatory = is_primary &&
                    (ablation.bits == kAblateRec || ablation.bits == kAblatePle ||
                     ablation.bits == (kAblateRec | kAblatePle));
                std::printf(
                    "REPAIR_ABLATE boundary=%s base=%u repair=%d cut=%u mask=%s "
                    "repeat=%d mandatory=%s indexer_blocks=%d "
                    "input_restore_exact=yes "
                    "retained_bytes_equal=yes masked_bytes_zero=yes "
                    "canonical_ple_prev=yes "
                    "c1024_checkpoint_equal=%s c1024_seed_logits_equal=%s "
                    "c1024_seed_logit_mismatches=%zu c1024_seed_logit_max_abs=%.9g "
                    "following32_logits_equal=%s following32_logit_mismatches=%zu "
                    "following32_logit_max_abs=%.9g forced32_tokens_equal=%s "
                    "token32_forwarded=yes c1056_post_state_equal=%s "
                    "repeat_exact=%s exact=%s\n",
                    boundary.name, boundary.start->cut, boundary.repair, kPromptCut,
                    ablation.name, repeat + 1, is_mandatory ? "yes" : "no",
                    candidate.at_1024.indexer_blocks,
                    oracle_diff.state_1024_equal ? "yes" : "no",
                    oracle_diff.full_logit_mismatches == 0 ? "yes" : "no",
                    oracle_diff.full_logit_mismatches, oracle_diff.full_logit_max_abs,
                    oracle_diff.step_logit_mismatches == 0 ? "yes" : "no",
                    oracle_diff.step_logit_mismatches, oracle_diff.step_logit_max_abs,
                    oracle_diff.tokens_equal ? "yes" : "no",
                    oracle_diff.post_state_equal ? "yes" : "no",
                    repeat_diff.exact() ? "yes" : "no",
                    oracle_diff.exact() ? "yes" : "no");
                if (ablation.bits == kAblateNone)
                    no_op_exact = no_op_exact && oracle_diff.exact() && repeat_diff.exact();
                if (ablation.bits == kAblateKv && is_primary)
                    sensitivity_detected = sensitivity_detected &&
                                           oracle_diff.full_logit_mismatches != 0 &&
                                           repeat_diff.exact();
                if (is_mandatory)
                    mandatory_exact = mandatory_exact && oracle_diff.exact() &&
                                      repeat_diff.exact();
                if (repeat == 0) first = std::move(candidate);
            }
        }
    }
    bool native_aa_exact = true;
    for (const Boundary & boundary : boundaries)
        native_aa_exact = native_aa_exact && boundary.native_aa_exact;
    const bool controls_pass = native_aa_exact && no_op_exact && sensitivity_detected;
    const char * outcome = mandatory_exact ? "exact_pass" : "causal_negative";
    std::printf("REPAIR_ABLATE_SUMMARY records=1 boundaries=2 masks=5 repeats=2 "
                "native_aa_exact=%s token_derived_ple_prev_exact=yes "
                "no_op_exact=%s kv_sensitivity_c1024_logits_detected=%s "
                "mandatory_exact_gate_fraction=%.6f controls_pass=%s "
                "infrastructure_valid=%s outcome=%s "
                "indexer_blocks=248 indexer_policy=preserved_sub128 "
                "translation_claim=no native_target_oracle=yes\n",
                native_aa_exact ? "yes" : "no", no_op_exact ? "yes" : "no",
                sensitivity_detected ? "yes" : "no", mandatory_exact ? 1.0 : 0.0,
                controls_pass ? "yes" : "no", controls_pass ? "yes" : "no",
                controls_pass ? outcome : "invalid");
    return controls_pass;
}

bool recurrent_split_repair_mode(const std::string & model,
                                 const std::string & token_path) {
    constexpr uint32_t kBase = 992;
    constexpr uint32_t kEnd = 1024;
    constexpr int kRepeats = 2;
    std::vector<int32_t> tokens;
    if (!read_tokens(token_path, tokens) || tokens.size() != 2048) {
        std::fprintf(stderr, "recurrent split repair requires exactly 2048 canonical tokens\n");
        return false;
    }
    Runtime rt;
    if (!rt.init(model, 1056) || !valid_tokens(tokens, rt.weights.n_vocab)) return false;

    std::vector<float> logits;
    if (!prefill(rt, tokens, kBase, logits) || !all_finite(logits)) return false;
    std::string error;
    Checkpoint base;
    if (!capture_exact(rt, kBase, static_cast<uint32_t>(tokens.size()), base, error) ||
        base.indexer_blocks != 248) {
        std::fprintf(stderr, "B992 capture failed: %s\n", error.c_str());
        return false;
    }
    Checkpoint canonical;
    if (!ablate_checkpoint(base, tokens, kRecurrentSplitAblations[0], canonical,
                           error) || !same_checkpoint(base, canonical)) {
        std::fprintf(stderr, "B992 PLE history is not canonical: %s\n", error.c_str());
        return false;
    }

    size_t ssm_count = 0, conv_count = 0, k_count = 0, v_count = 0;
    for (const Tensor & tensor : base.tensors) {
        ssm_count += prefix(tensor.name, "qwen4exp.ssm_state.");
        conv_count += prefix(tensor.name, "qwen4exp.conv_state.");
        k_count += prefix(tensor.name, "qwen4exp.attn_k.");
        v_count += prefix(tensor.name, "qwen4exp.attn_v.");
    }
    if (ssm_count != 36 || conv_count != 36 || k_count != 12 || v_count != 12) {
        std::fprintf(stderr,
                     "unexpected B992 family counts ssm=%zu conv=%zu k=%zu v=%zu\n",
                     ssm_count, conv_count, k_count, v_count);
        return false;
    }

    RepairTrace oracle;
    if (!run_repair_trace(rt, base, tokens, oracle, error)) {
        std::fprintf(stderr, "native oracle failed: %s\n", error.c_str());
        return false;
    }
    RepairTrace repeated_oracle;
    if (!run_repair_trace(rt, base, tokens, repeated_oracle, error)) {
        std::fprintf(stderr, "native A/A failed: %s\n", error.c_str());
        return false;
    }
    const bool native_aa_exact = compare_repair_trace(oracle, repeated_oracle).exact();
    std::printf("RECURRENT_SPLIT_NATIVE_AA boundary=B992_R32_C1024 "
                "repeats=2 indexer_blocks=248 exact=%s\n",
                native_aa_exact ? "yes" : "no");

    bool no_op_exact = true;
    bool sensitivity_detected = true;
    bool conv_exact = true;
    bool ssm_exact = true;
    for (const AblationCase & ablation : kRecurrentSplitAblations) {
        Checkpoint branch;
        if (!ablate_checkpoint(base, tokens, ablation, branch, error)) {
            std::fprintf(stderr, "branch construction failed mask=%s: %s\n",
                         ablation.name, error.c_str());
            return false;
        }
        const size_t selected_count = ablation.bits == kAblateConv ? 36 :
                                      ablation.bits == kAblateSsm ? 36 :
                                      ablation.bits == kAblateKv ? 24 : 0;
        RepairTrace first;
        for (int repeat = 0; repeat < kRepeats; ++repeat) {
            RepairTrace candidate;
            if (!run_repair_trace(rt, branch, tokens, candidate, error,
                                  &oracle.greedy_tokens)) {
                std::fprintf(stderr, "branch failed mask=%s repeat=%d: %s\n",
                             ablation.name, repeat + 1, error.c_str());
                return false;
            }
            const RepairDiff oracle_diff = compare_repair_trace(oracle, candidate);
            const RepairDiff repeat_diff = repeat == 0 ? RepairDiff{
                true, true, true, 0, 0, 0.0, 0.0} :
                compare_repair_trace(first, candidate);
            const bool is_mandatory = ablation.bits == kAblateConv;
            std::printf(
                "RECURRENT_SPLIT boundary=B992_R32_C1024 base=%u repair=32 cut=%u "
                "mask=%s repeat=%d mandatory=%s selected_tensor_count=%zu "
                "indexer_blocks=%d input_restore_exact=yes retained_bytes_equal=yes "
                "masked_bytes_zero=yes canonical_ple_prev=yes "
                "c1024_checkpoint_equal=%s c1024_seed_logits_equal=%s "
                "c1024_seed_logit_mismatches=%zu c1024_seed_logit_max_abs=%.9g "
                "following32_logits_equal=%s following32_logit_mismatches=%zu "
                "following32_logit_max_abs=%.9g forced32_tokens_equal=%s "
                "token32_forwarded=yes c1056_post_state_equal=%s "
                "repeat_exact=%s exact=%s\n",
                kBase, kEnd, ablation.name, repeat + 1,
                is_mandatory ? "yes" : "no", selected_count,
                candidate.at_1024.indexer_blocks,
                oracle_diff.state_1024_equal ? "yes" : "no",
                oracle_diff.full_logit_mismatches == 0 ? "yes" : "no",
                oracle_diff.full_logit_mismatches, oracle_diff.full_logit_max_abs,
                oracle_diff.step_logit_mismatches == 0 ? "yes" : "no",
                oracle_diff.step_logit_mismatches, oracle_diff.step_logit_max_abs,
                oracle_diff.tokens_equal ? "yes" : "no",
                oracle_diff.post_state_equal ? "yes" : "no",
                repeat_diff.exact() ? "yes" : "no",
                oracle_diff.exact() ? "yes" : "no");
            const bool exact_repeat = oracle_diff.exact() && repeat_diff.exact();
            if (ablation.bits == kAblateNone) no_op_exact &= exact_repeat;
            if (ablation.bits == kAblateKv)
                sensitivity_detected &= oracle_diff.full_logit_mismatches != 0 &&
                                        repeat_diff.exact();
            if (ablation.bits == kAblateConv) conv_exact &= exact_repeat;
            if (ablation.bits == kAblateSsm) ssm_exact &= exact_repeat;
            if (repeat == 0) first = std::move(candidate);
        }
    }
    const bool controls_pass = native_aa_exact && no_op_exact && sensitivity_detected;
    const char * outcome = conv_exact ? "exact_pass" : "causal_negative";
    std::printf(
        "RECURRENT_SPLIT_SUMMARY records=1 boundaries=1 masks=4 repeats=2 "
        "native_aa_exact=%s token_derived_ple_prev_exact=yes no_op_exact=%s "
        "kv_sensitivity_c1024_logits_detected=%s "
        "conv_only_r32_exact_prompt_fraction=%.6f "
        "ssm_only_r32_exact_prompt_fraction=%.6f controls_pass=%s "
        "infrastructure_valid=%s outcome=%s indexer_blocks=248 "
        "indexer_policy=preserved_sub128 translation_claim=no "
        "native_target_oracle=yes\n",
        native_aa_exact ? "yes" : "no", no_op_exact ? "yes" : "no",
        sensitivity_detected ? "yes" : "no", conv_exact ? 1.0 : 0.0,
        ssm_exact ? 1.0 : 0.0, controls_pass ? "yes" : "no",
        controls_pass ? "yes" : "no", controls_pass ? outcome : "invalid");
    return controls_pass;
}

double logit_sse(const RepairTrace & oracle, const RepairTrace & candidate) {
    auto add = [](const std::vector<float> & a, const std::vector<float> & b,
                  long double & sum) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            const long double delta = static_cast<long double>(a[i]) - b[i];
            sum += delta * delta;
        }
        return true;
    };
    long double sum = 0.0;
    if (!add(oracle.full_logits, candidate.full_logits, sum) ||
        oracle.step_logits.size() != candidate.step_logits.size())
        return std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < oracle.step_logits.size(); ++i)
        if (!add(oracle.step_logits[i], candidate.step_logits[i], sum))
            return std::numeric_limits<double>::infinity();
    return static_cast<double>(sum);
}

size_t forced_top1_matches(const RepairTrace & oracle,
                           const RepairTrace & candidate) {
    if (oracle.step_logits.size() != 32 || candidate.step_logits.size() != 32)
        return 0;
    size_t matches = argmax(oracle.full_logits) == argmax(candidate.full_logits);
    for (size_t i = 0; i < 32; ++i)
        matches += argmax(oracle.step_logits[i]) == argmax(candidate.step_logits[i]);
    return matches;
}

bool retained_fa_equal(const Checkpoint & native, const Checkpoint & replayed) {
    if (native.indexer_blocks != replayed.indexer_blocks ||
        native.full_layer_ids != replayed.full_layer_ids ||
        native.indexer_layer_ids != replayed.indexer_layer_ids) return false;
    std::map<std::string, const Tensor *> right;
    for (const Tensor & tensor : replayed.tensors) right.emplace(tensor.name, &tensor);
    for (const Tensor & tensor : native.tensors) {
        if (!prefix(tensor.name, "qwen4exp.attn_k.") &&
            !prefix(tensor.name, "qwen4exp.attn_v.") &&
            !prefix(tensor.name, "qwen4exp.indexer_k.")) continue;
        const auto found = right.find(tensor.name);
        if (found == right.end() || tensor.type != found->second->type ||
            tensor.ne != found->second->ne || tensor.data != found->second->data)
            return false;
    }
    return true;
}

bool tail_replay_mode(const std::string & model, const std::string & token_path) {
    constexpr uint32_t kBase = 992;
    constexpr std::array<const char *, 12> kBoundaryFiles = {
        "/tmp/our_L00.emb.bin", "/tmp/our_L03.res.bin", "/tmp/our_L07.res.bin",
        "/tmp/our_L11.res.bin", "/tmp/our_L15.res.bin", "/tmp/our_L19.res.bin",
        "/tmp/our_L23.res.bin", "/tmp/our_L27.res.bin", "/tmp/our_L31.res.bin",
        "/tmp/our_L35.res.bin", "/tmp/our_L39.res.bin", "/tmp/our_L43.res.bin"};
    constexpr char kBoundaries[] =
        "L00.emb,L03.res,L07.res,L11.res,L15.res,L19.res,L23.res,"
        "L27.res,L31.res,L35.res,L39.res,L43.res";
    std::vector<int32_t> tokens;
    if (!read_tokens(token_path, tokens) || tokens.size() != 2048) {
        std::fprintf(stderr, "tail replay requires exactly 2048 canonical tokens\n");
        return false;
    }
#if defined(_WIN32)
    if (_putenv_s("QWEN4EXP_DUMP", "1") != 0 ||
        _putenv_s("QWEN4EXP_DUMP_BIN", kBoundaries) != 0) return false;
#else
    if (setenv("QWEN4EXP_DUMP", "1", 1) != 0 ||
        setenv("QWEN4EXP_DUMP_BIN", kBoundaries, 1) != 0) return false;
#endif
    Runtime rt;
    if (!rt.init(model, 1056) || !valid_tokens(tokens, rt.weights.n_vocab)) return false;

    for (const char * path : kBoundaryFiles) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        if (remove_error) {
            std::fprintf(stderr, "cannot remove stale boundary %s: %s\n",
                         path, remove_error.message().c_str());
            return false;
        }
    }

    std::vector<float> logits;
    if (!prefill(rt, tokens, kBase, logits) || !all_finite(logits)) return false;
    const uintmax_t plane_bytes = static_cast<uintmax_t>(rt.weights.n_embd) *
                                  rt.weights.n_hc * kBase * sizeof(float);
    for (size_t i = 0; i < kBoundaryFiles.size(); ++i) {
        std::error_code size_error;
        const uintmax_t actual = std::filesystem::file_size(kBoundaryFiles[i], size_error);
        const uintmax_t expected = plane_bytes * (i == 0 ? 1 : 2);
        if (size_error || actual != expected) {
            std::fprintf(stderr,
                         "native boundary capture missing/malformed path=%s actual=%ju expected=%ju error=%s\n",
                         kBoundaryFiles[i], actual, expected,
                         size_error ? size_error.message().c_str() : "none");
            return false;
        }
    }
    std::string error;
    Checkpoint native;
    if (!capture_exact(rt, kBase, static_cast<uint32_t>(tokens.size()), native, error) ||
        native.indexer_blocks != 248) {
        std::fprintf(stderr, "native B992 capture failed: %s\n", error.c_str());
        return false;
    }
    Checkpoint zero;
    if (!ablate_checkpoint(native, tokens, kAblations[3], zero, error)) {
        std::fprintf(stderr, "zero-state control construction failed: %s\n", error.c_str());
        return false;
    }
    // QWEN4EXP_DUMP_BIN uses fixed paths and every later multi-token forward
    // overwrites them. Consume both reconstructions while they still contain
    // the freshly size-checked native B992 boundaries.
    auto reconstruct = [&](Checkpoint & out) {
        if (!restore(rt.backend, rt.weights, rt.cache, zero, &error) ||
            !qwen4exp_tail_replay_992_100(rt.backend, rt.weights, rt.cache,
                                          tokens.data(), error) ||
            !capture_exact(rt, kBase, static_cast<uint32_t>(tokens.size()), out, error))
            return false;
        return retained_fa_equal(native, out);
    };
    Checkpoint replayed, replayed_repeat;
    if (!reconstruct(replayed) || !reconstruct(replayed_repeat)) {
        std::fprintf(stderr, "group-independent reconstruction failed: %s\n", error.c_str());
        return false;
    }
    const bool reconstruction_repeat_exact = same_checkpoint(replayed, replayed_repeat);

    Checkpoint native_noop;
    if (!ablate_checkpoint(native, tokens, kAblations[0], native_noop, error) ||
        !same_checkpoint(native, native_noop)) {
        std::fprintf(stderr, "native no-op restore construction failed: %s\n",
                     error.c_str());
        return false;
    }
    RepairTrace oracle, oracle_repeat, noop_restore, zero_control;
    if (!run_repair_trace(rt, native, tokens, oracle, error) ||
        !run_repair_trace(rt, native, tokens, oracle_repeat, error) ||
        !run_repair_trace(rt, native_noop, tokens, noop_restore, error) ||
        !run_repair_trace(rt, zero, tokens, zero_control, error,
                          &oracle.greedy_tokens)) {
        std::fprintf(stderr, "tail replay control failed: %s\n", error.c_str());
        return false;
    }
    const bool native_aa_exact = compare_repair_trace(oracle, oracle_repeat).exact();
    const bool native_noop_exact = compare_repair_trace(oracle, noop_restore).exact();

    RepairTrace forced, forced_repeat, autonomous;
    if (!run_repair_trace(rt, replayed, tokens, forced, error, &oracle.greedy_tokens) ||
        !run_repair_trace(rt, replayed_repeat, tokens, forced_repeat, error,
                          &oracle.greedy_tokens) ||
        !run_repair_trace(rt, replayed, tokens, autonomous, error)) {
        std::fprintf(stderr, "tail replay evaluation failed: %s\n", error.c_str());
        return false;
    }
    const size_t top1_matches = forced_top1_matches(oracle, forced);
    const double replay_sse = logit_sse(oracle, forced);
    const double zero_sse = logit_sse(oracle, zero_control);
    const bool forced_repeat_exact = compare_repair_trace(forced, forced_repeat).exact();
    const bool autonomous_exact = oracle.greedy_tokens == autonomous.greedy_tokens;
    const bool controls_pass = native_aa_exact && native_noop_exact &&
                               reconstruction_repeat_exact && forced_repeat_exact &&
                               retained_fa_equal(native, replayed);
    std::printf(
        "TAIL_REPLAY boundary=B992 repair=100 groups=12 forced_vectors=33 "
        "forced_top1_matches=%zu forced_top1_fraction=%.9f "
        "replay_logit_sse=%.17g zero_logit_sse=%.17g replay_better_than_zero=%s "
        "autonomous_tokens=32 autonomous_exact=%s native_aa_exact=%s "
        "native_noop_exact=%s "
        "reconstruction_repeat_exact=%s forced_repeat_exact=%s "
        "fa_kv_indexer_retained_exact=yes indexer_blocks=%d controls_pass=%s "
        "infrastructure_valid=%s translation_claim=no native_target_ceiling=yes\n",
        top1_matches, top1_matches / 33.0, replay_sse, zero_sse,
        replay_sse < zero_sse ? "yes" : "no", autonomous_exact ? "yes" : "no",
        native_aa_exact ? "yes" : "no",
        native_noop_exact ? "yes" : "no",
        reconstruction_repeat_exact ? "yes" : "no",
        forced_repeat_exact ? "yes" : "no", replayed.indexer_blocks,
        controls_pass ? "yes" : "no", controls_pass ? "yes" : "no");
    return controls_pass;
}

bool dump_mode(const std::string & model, const std::string & token_path,
               int cut, const std::string & output_path) {
    std::vector<int32_t> tokens;
    if (!read_tokens(token_path, tokens) || tokens.size() < static_cast<size_t>(cut))
        return false;
    const uint64_t max_ctx64 = static_cast<uint64_t>(cut) + 64;
    if (max_ctx64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "dump cache capacity exceeds INT_MAX\n");
        return false;
    }
    const int max_ctx = static_cast<int>(max_ctx64);
    Runtime rt;
    if (!rt.init(model, max_ctx) || !valid_tokens(tokens, rt.weights.n_vocab))
        return false;
    std::vector<float> logits;
    if (!prefill(rt, tokens, static_cast<size_t>(cut), logits)) return false;
    Checkpoint checkpoint;
    std::string error;
    if (!capture(rt.backend, rt.weights, rt.cache, cut,
                 static_cast<uint32_t>(tokens.size()), checkpoint, &error)) {
        std::fprintf(stderr, "capture failed: %s\n", error.c_str());
        return false;
    }
    const std::vector<uint8_t> bytes = encode(checkpoint, &error);
    if (bytes.empty() || !write_file(output_path, bytes)) {
        std::fprintf(stderr, "checkpoint write failed: %s\n", error.c_str());
        return false;
    }
    std::printf("DUMP cut=%d prompt_len=%zu indexer_blocks=%d bytes=%zu path=%s\n",
                cut, tokens.size(), checkpoint.indexer_blocks, bytes.size(),
                output_path.c_str());
    return true;
}

bool inject_mode(const std::string & model, const std::string & checkpoint_path,
                 const std::string & suffix_path, int generated) {
    std::vector<uint8_t> bytes;
    std::vector<int32_t> suffix;
    Checkpoint checkpoint;
    std::string error;
    if (!read_tokens(suffix_path, suffix) || suffix.size() != 128) {
        std::fprintf(stderr, "checkpoint/suffix input invalid: %s\n", error.c_str());
        return false;
    }
    const auto read_start = std::chrono::steady_clock::now();
    if (!read_file(checkpoint_path, bytes) || !decode(bytes, checkpoint, &error)) {
        std::fprintf(stderr, "checkpoint/suffix input invalid: %s\n", error.c_str());
        return false;
    }
    const auto read_end = std::chrono::steady_clock::now();
    if (checkpoint.cut != 896 || checkpoint.prompt_len != 1024) {
        std::fprintf(stderr, "inject requires a C896 checkpoint for a 1024-token prompt\n");
        return false;
    }
    const uint64_t max_ctx64 = static_cast<uint64_t>(checkpoint.cut) + suffix.size() +
                               static_cast<uint64_t>(generated);
    if (max_ctx64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) return false;
    Runtime rt;
    const auto load_start = std::chrono::steady_clock::now();
    if (!rt.init(model, static_cast<int>(max_ctx64))) return false;
    const auto load_end = std::chrono::steady_clock::now();
    if (!valid_tokens(suffix, rt.weights.n_vocab)) {
        std::fprintf(stderr, "checkpoint/suffix input invalid: out-of-vocabulary token\n");
        return false;
    }

    ggml_backend_synchronize(rt.backend);
    const auto adopt_start = std::chrono::steady_clock::now();
    if (!restore(rt.backend, rt.weights, rt.cache, checkpoint, &error)) {
        std::fprintf(stderr, "checkpoint adoption failed: %s\n", error.c_str());
        return false;
    }
    ggml_backend_synchronize(rt.backend);
    const auto adopt_end = std::chrono::steady_clock::now();
    if (rt.cache.indexer_blocks != 224) {
        std::fprintf(stderr, "C896 restore QSA mismatch: expected=224 actual=%d\n",
                     rt.cache.indexer_blocks);
        return false;
    }

    // The suffix is prompt input, not generated output. C896 plus exactly
    // prompt[896:1024] produces the first native Flash output logit.
    std::vector<float> logits;
    const auto suffix_start = std::chrono::steady_clock::now();
    if (!forward_chunks(rt, suffix.data(), static_cast<int>(suffix.size()),
                        static_cast<int>(checkpoint.cut), logits))
        return false;
    ggml_backend_synchronize(rt.backend);
    const auto first_logit_end = std::chrono::steady_clock::now();
    if (!all_finite(logits)) return false;
    if (rt.cache.indexer_blocks != 256) {
        std::fprintf(stderr, "R128 suffix QSA mismatch: expected=256 actual=%d\n",
                     rt.cache.indexer_blocks);
        return false;
    }
    GreedyRun run;
    const auto generation_start = std::chrono::steady_clock::now();
    if (!continue_greedy(rt, nullptr, 1024, 0, logits, generated, run)) return false;
    ggml_backend_synchronize(rt.backend);
    const auto generation_end = std::chrono::steady_clock::now();
    const bool eos = !run.tokens.empty() &&
        (run.tokens.back() == rt.weights.eos_id ||
         run.tokens.back() == rt.weights.eos_chat_id);

    const double read_ms = std::chrono::duration<double, std::milli>(
        read_end - read_start).count();
    const double load_ms = std::chrono::duration<double, std::milli>(
        load_end - load_start).count();
    const double adopt_ms = std::chrono::duration<double, std::milli>(
        adopt_end - adopt_start).count();
    const double suffix_ms = std::chrono::duration<double, std::milli>(
        first_logit_end - suffix_start).count();
    const double generation_ms = std::chrono::duration<double, std::milli>(
        generation_end - generation_start).count();
    // Direct spans retain small host gaps (allocation and finite validation)
    // that summing the named phase timers would otherwise hide.
    const double resident_to_logit_ms = std::chrono::duration<double, std::milli>(
        first_logit_end - adopt_start).count();
    const double resident_to_complete_ms = std::chrono::duration<double, std::milli>(
        generation_end - adopt_start).count();
    std::printf("TIMING kind=checkpoint_inject cut=%u suffix_tokens=%zu "
                "checkpoint_read_decode_ms=%.3f model_load_ms=%.3f "
                "adopt_ms=%.3f suffix_to_first_logit_ms=%.3f "
                "continuation_after_suffix_ms=%.3f "
                "paid_read_adopt_suffix_to_first_logit_ms=%.3f "
                "paid_read_adopt_suffix_generation_complete_ms=%.3f "
                "model_load_excluded=yes\n",
                checkpoint.cut, suffix.size(), read_ms, load_ms, adopt_ms,
                suffix_ms, generation_ms, read_ms + resident_to_logit_ms,
                read_ms + resident_to_complete_ms);
    std::printf("INJECT cut=%u suffix=%zu qsa_before=224 qsa_after=256 "
                "generated_count=%zu termination=%s token_ids=",
                checkpoint.cut, suffix.size(), run.tokens.size(),
                eos ? "eos" : "limit");
    for (size_t i = 0; i < run.tokens.size(); ++i)
        std::printf("%s%d", i == 0 ? "" : ",", run.tokens[i]);
    std::printf("\n");
    return true;
}

bool parity_mode(const std::string & model, const std::string & token_path,
                 int cut, int repair, int generated) {
    std::vector<int32_t> tokens;
    if (!read_tokens(token_path, tokens)) return false;
    const uint64_t input_count = static_cast<uint64_t>(cut) + repair;
    const uint64_t max_ctx64 = input_count + generated;
    if (tokens.size() != input_count || max_ctx64 > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "TOKENS.bin must contain exactly C+R IDs and fit cache capacity\n");
        return false;
    }
    Runtime rt;
    if (!rt.init(model, static_cast<int>(max_ctx64)) ||
        !valid_tokens(tokens, rt.weights.n_vocab)) return false;

    std::vector<float> cut_logits;
    if (!prefill(rt, tokens, static_cast<size_t>(cut), cut_logits) ||
        !all_finite(cut_logits)) return false;
    Checkpoint start;
    std::string error;
    if (!capture(rt.backend, rt.weights, rt.cache, cut,
                 static_cast<uint32_t>(tokens.size()), start, &error)) {
        std::fprintf(stderr, "cut capture failed: %s\n", error.c_str());
        return false;
    }

    GreedyRun uninterrupted;
    if (!continue_greedy(rt, tokens.data() + cut, cut, repair,
                         cut_logits, generated, uninterrupted)) return false;

    // Native A/A control: reconstruct the prefix without serialization, then
    // run the identical suffix and greedy continuation before testing restore.
    reset_qwen4exp_state(rt.backend, rt.cache);
    rt.cache.ple_prev.clear();
    clear_qwen4exp_decode_workspace(rt.cache.decode_workspace);
    std::vector<float> repeated_cut_logits;
    if (!prefill(rt, tokens, static_cast<size_t>(cut), repeated_cut_logits) ||
        !all_finite(repeated_cut_logits)) return false;
    GreedyRun native_repeat;
    if (!continue_greedy(rt, tokens.data() + cut, cut, repair,
                         repeated_cut_logits, generated, native_repeat)) return false;

    if (!restore(rt.backend, rt.weights, rt.cache, start, &error)) {
        std::fprintf(stderr, "restore preflight/copy failed: %s\n", error.c_str());
        return false;
    }
    Checkpoint restored;
    if (!capture(rt.backend, rt.weights, rt.cache, cut,
                 static_cast<uint32_t>(tokens.size()), restored, &error) ||
        !same_checkpoint(start, restored)) {
        std::fprintf(stderr, "restored semantic bytes differ from captured state: %s\n",
                     error.c_str());
        return false;
    }

    GreedyRun bridged;
    if (!continue_greedy(rt, tokens.data() + cut, cut, repair,
                         cut_logits, generated, bridged)) return false;
    double aa_max_abs = 0.0, restore_max_abs = 0.0;
    size_t aa_exact_mismatches = 0, restore_exact_mismatches = 0;
    logit_diff(uninterrupted.seed_logits, native_repeat.seed_logits,
               aa_max_abs, aa_exact_mismatches);
    // At R=0, bridged seed logits are supplied out of band; restored bytes and
    // the subsequent continuation are the restore proof in that case.
    logit_diff(uninterrupted.seed_logits, bridged.seed_logits,
               restore_max_abs, restore_exact_mismatches);
    const bool aa_equal = uninterrupted.tokens == native_repeat.tokens;
    const bool restore_equal = uninterrupted.tokens == bridged.tokens;
    std::printf("PARITY cut=%d repair=%d indexer_blocks=%d state_bytes_equal=yes "
                "native_aa_equal=%s native_aa_logits_equal=%s "
                "native_aa_logit_max_abs=%.9g "
                "native_aa_logit_mismatches=%zu restore_logit_max_abs=%.9g "
                "restore_logit_mismatches=%zu restore_logits_equal=%s "
                "greedy_tokens=%zu "
                "restore_equal=%s\n",
                cut, repair, start.indexer_blocks, aa_equal ? "yes" : "no",
                aa_exact_mismatches == 0 ? "yes" : "no", aa_max_abs,
                aa_exact_mismatches, restore_max_abs, restore_exact_mismatches,
                restore_exact_mismatches == 0 ? "yes" : "no",
                native_repeat.tokens.size(),
                restore_equal ? "yes" : "no");
    if (aa_exact_mismatches != 0 || restore_exact_mismatches != 0) {
        std::fprintf(stderr,
                     "exact logit parity failed: native_a/a=%zu restored=%zu\n",
                     aa_exact_mismatches, restore_exact_mismatches);
    }
    if (!aa_equal || !restore_equal) {
        std::fprintf(stderr, "uninterrupted:");
        for (int32_t token : uninterrupted.tokens) std::fprintf(stderr, " %d", token);
        std::fprintf(stderr, "\nnative A/A:");
        for (int32_t token : native_repeat.tokens) std::fprintf(stderr, " %d", token);
        std::fprintf(stderr, "\nrestored:");
        for (int32_t token : bridged.tokens) std::fprintf(stderr, " %d", token);
        std::fprintf(stderr, "\n");
    }
    return aa_exact_mismatches == 0 && restore_exact_mismatches == 0 &&
           aa_equal && restore_equal;
}

void usage(const char * exe) {
    std::fprintf(stderr,
        "usage:\n"
        "  %s --parity MODEL TOKENS.bin C R [--gen N]\n"
        "  %s --dump MODEL TOKENS.bin C OUT.lbsnap\n"
        "  %s --inject MODEL CHECKPOINT.lbsnap SUFFIX.bin [--gen N]\n"
        "  %s --compose-kv NATIVE.lbsnap DONOR.lbsnap OUT.lbsnap\n"
        "  %s --translate-smoke SOURCE.lbsnap TOKENS.bin OUT.lbsnap\n"
        "  %s --compare A.lbsnap B.lbsnap\n"
        "  %s --repair-ablate MODEL TOKENS.bin\n"
        "  %s --recurrent-split-repair MODEL TOKENS.bin\n"
        "  %s --tail-replay MODEL TOKENS.bin\n",
        exe, exe, exe, exe, exe, exe, exe, exe, exe);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) { usage(argv[0]); return 2; }
    const std::string mode = argv[1];
    if (mode == "--compare") {
        if (argc != 4) { usage(argv[0]); return 2; }
        return compare_mode(argv[2], argv[3]);
    }
    if (mode == "--translate-smoke") {
        if (argc != 5) { usage(argv[0]); return 2; }
        return translate_smoke_mode(argv[2], argv[3], argv[4]) ? 0 : 1;
    }
    if (mode == "--compose-kv") {
        if (argc != 5) { usage(argv[0]); return 2; }
        return compose_kv_mode(argv[2], argv[3], argv[4]) ? 0 : 1;
    }
    if (!force_probe_environment()) {
        std::fprintf(stderr, "failed to set probe environment\n");
        return 2;
    }
    bool ok = false;
    if (mode == "--parity" && (argc == 6 || argc == 8)) {
        int cut = 0, repair = 0, generated = 64;
        if (!parse_int(argv[4], cut, false) || !parse_int(argv[5], repair, true) ||
            (argc == 8 && (std::string(argv[6]) != "--gen" ||
                           !parse_int(argv[7], generated, false)))) {
            usage(argv[0]); return 2;
        }
        ok = parity_mode(argv[2], argv[3], cut, repair, generated);
    } else if (mode == "--dump" && argc == 6) {
        int cut = 0;
        if (!parse_int(argv[4], cut, false)) { usage(argv[0]); return 2; }
        ok = dump_mode(argv[2], argv[3], cut, argv[5]);
    } else if (mode == "--inject" && (argc == 5 || argc == 7)) {
        int generated = 1;
        if (argc == 7 && (std::string(argv[5]) != "--gen" ||
                          !parse_int(argv[6], generated, false) ||
                          generated > 4096)) {
            usage(argv[0]); return 2;
        }
        ok = inject_mode(argv[2], argv[3], argv[4], generated);
    } else if (mode == "--repair-ablate" && argc == 4) {
        ok = repair_ablate_mode(argv[2], argv[3]);
    } else if (mode == "--recurrent-split-repair" && argc == 4) {
        ok = recurrent_split_repair_mode(argv[2], argv[3]);
    } else if (mode == "--tail-replay" && argc == 4) {
        ok = tail_replay_mode(argv[2], argv[3]);
    } else {
        usage(argv[0]); return 2;
    }
    return ok ? 0 : 1;
}
