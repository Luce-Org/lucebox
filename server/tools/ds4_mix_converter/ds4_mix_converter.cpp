#include "ggml.h"
#include "gguf.h"
#include "rocmfpx.h"
#include "expert_batches.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <climits>
#include <cstdlib>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

constexpr uint32_t kQtypeP4Mix = 105;
constexpr uint32_t kQtypeGuMix = 106;
constexpr uint32_t kCodebooks = 2;
constexpr uint32_t kBlock = 32;
constexpr uint32_t kP4Levels = 8;
constexpr uint32_t kGuLevels = 4;
constexpr uint32_t kAlignment = 32;
constexpr size_t kCopyChunk = 8u * 1024u * 1024u;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

uint64_t checked_mul(uint64_t a, uint64_t b, const std::string & what) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        fail("overflow computing " + what);
    }
    return a*b;
}

uint64_t element_count(const std::vector<int64_t> & shape, const std::string & name) {
    if (shape.empty() || shape.size() > GGML_MAX_DIMS) {
        fail("unsupported rank for " + name);
    }
    uint64_t n = 1;
    for (int64_t d : shape) {
        if (d <= 0) fail("non-positive dimension for " + name);
        n = checked_mul(n, static_cast<uint64_t>(d), "element count for " + name);
    }
    return n;
}

size_t dtype_size(const std::string & dtype) {
    if (dtype == "BF16" || dtype == "F16") return 2;
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
    if (dtype == "I64" || dtype == "U64") return 8;
    if (dtype == "I8" || dtype == "U8" || dtype == "F8_E4M3" || dtype == "F8_E8M0") return 1;
    fail("unsupported safetensors dtype " + dtype);
}

uint16_t float_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

float bf16_to_float(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float fp4_e2m1(uint8_t code) {
    static constexpr float table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f,
    };
    return table[code & 15u];
}

float fp8_e4m3fn(uint8_t value) {
    const int sign = (value & 0x80u) ? -1 : 1;
    const int exponent = (value >> 3) & 15;
    const int mantissa = value & 7;
    if (exponent == 15 && mantissa == 7) {
        fail("F8_E4M3 contains NaN encoding");
    }
    const float magnitude = exponent == 0
        ? std::ldexp(static_cast<float>(mantissa), -9)
        : std::ldexp(static_cast<float>(8 + mantissa), exponent - 10);
    return sign * magnitude;
}

float fp8_e8m0(uint8_t value) {
    if (value == 0xffu) fail("F8_E8M0 contains NaN encoding");
    return std::ldexp(1.0f, static_cast<int>(value) - 127);
}

struct FileDescriptor {
    int fd = -1;
    explicit FileDescriptor(const fs::path & path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) fail("open " + path.string() + ": " + std::strerror(errno));
    }
    FileDescriptor(const FileDescriptor &) = delete;
    FileDescriptor & operator=(const FileDescriptor &) = delete;
    ~FileDescriptor() { if (fd >= 0) ::close(fd); }
};

void pread_exact(int fd, void * dst, size_t n, uint64_t offset, const std::string & what) {
    auto * p = static_cast<uint8_t *>(dst);
    size_t done = 0;
    while (done < n) {
        const ssize_t got = ::pread(fd, p + done, n - done, static_cast<off_t>(offset + done));
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) fail("short read for " + what);
        done += static_cast<size_t>(got);
    }
}

struct StEntry {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    fs::path path;
    uint64_t offset = 0;
    uint64_t size = 0;
};

class SafeTensorSet {
public:
    explicit SafeTensorSet(fs::path root) : root_(std::move(root)) {
        if (!fs::is_directory(root_)) fail("input is not a directory: " + root_.string());
        config_ = read_json(root_ / "config.json");
        tokenizer_ = read_json(root_ / "tokenizer.json");
        tokenizer_config_ = read_json(root_ / "tokenizer_config.json");
        const json index = read_json(root_ / "model.safetensors.index.json");
        if (!index.contains("weight_map") || !index["weight_map"].is_object()) {
            fail("model.safetensors.index.json has no object weight_map");
        }
        std::map<std::string, std::string> weight_map;
        std::set<std::string> shards;
        for (const auto & item : index["weight_map"].items()) {
            if (!item.value().is_string()) fail("weight_map value is not a string for " + item.key());
            const std::string shard = item.value().get<std::string>();
            if (fs::path(shard).is_absolute() || shard.find("..") != std::string::npos) {
                fail("unsafe shard path " + shard);
            }
            weight_map.emplace(item.key(), shard);
            shards.insert(shard);
        }
        for (const std::string & shard : shards) load_shard(shard);
        if (entries_.size() != weight_map.size()) {
            fail("safetensors headers contain " + std::to_string(entries_.size()) +
                 " tensors but index names " + std::to_string(weight_map.size()));
        }
        for (const auto & item : weight_map) {
            const auto it = entries_.find(item.first);
            if (it == entries_.end()) fail("index tensor missing from shard header: " + item.first);
            if (it->second.path.filename() != item.second) {
                fail("index shard mismatch for " + item.first);
            }
        }
    }

    const StEntry & at(const std::string & name) const {
        const auto it = entries_.find(name);
        if (it == entries_.end()) fail("missing required tensor " + name);
        return it->second;
    }
    bool contains(const std::string & name) const { return entries_.count(name) != 0; }
    const std::unordered_map<std::string, StEntry> & entries() const { return entries_; }
    const json & config() const { return config_; }
    const json & tokenizer() const { return tokenizer_; }
    const json & tokenizer_config() const { return tokenizer_config_; }

private:
    static json read_json(const fs::path & path) {
        std::ifstream in(path);
        if (!in) fail("cannot open " + path.string());
        json result;
        try { in >> result; }
        catch (const std::exception & e) { fail("invalid JSON " + path.string() + ": " + e.what()); }
        return result;
    }

    void load_shard(const std::string & filename) {
        const fs::path path = root_ / filename;
        FileDescriptor fd(path);
        struct stat st{};
        if (::fstat(fd.fd, &st) != 0 || st.st_size < 8) fail("invalid shard file " + path.string());
        uint64_t header_len = 0;
        pread_exact(fd.fd, &header_len, sizeof(header_len), 0, path.string() + " header length");
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
        fail("big-endian hosts are not supported");
#endif
        if (header_len == 0 || header_len > static_cast<uint64_t>(st.st_size) - 8 || header_len > (1ull << 31)) {
            fail("invalid safetensors header length in " + path.string());
        }
        std::string header(static_cast<size_t>(header_len), 0);
        pread_exact(fd.fd, header.data(), header.size(), 8, path.string() + " header");
        json parsed;
        try { parsed = json::parse(header); }
        catch (const std::exception & e) { fail("invalid safetensors header in " + path.string() + ": " + e.what()); }
        for (const auto & item : parsed.items()) {
            if (item.key() == "__metadata__") continue;
            const json & info = item.value();
            if (!info.contains("dtype") || !info.contains("shape") || !info.contains("data_offsets")) {
                fail("incomplete safetensors entry " + item.key());
            }
            StEntry entry;
            entry.name = item.key();
            entry.dtype = info["dtype"].get<std::string>();
            entry.shape = info["shape"].get<std::vector<int64_t>>();
            const auto range = info["data_offsets"].get<std::vector<uint64_t>>();
            if (range.size() != 2 || range[1] < range[0]) fail("bad data_offsets for " + entry.name);
            entry.path = path;
            entry.offset = 8 + header_len + range[0];
            entry.size = range[1] - range[0];
            const uint64_t expected = checked_mul(element_count(entry.shape, entry.name), dtype_size(entry.dtype), "byte size for " + entry.name);
            if (entry.size != expected || entry.offset > static_cast<uint64_t>(st.st_size) ||
                entry.size > static_cast<uint64_t>(st.st_size) - entry.offset) {
                fail("size or bounds mismatch for " + entry.name);
            }
            if (!entries_.emplace(entry.name, std::move(entry)).second) {
                fail("duplicate tensor " + item.key());
            }
        }
    }

    fs::path root_;
    json config_;
    json tokenizer_;
    json tokenizer_config_;
    std::unordered_map<std::string, StEntry> entries_;
};

struct ImatrixEntry {
    int32_t calls = 0;
    std::vector<float> values;
};
using Imatrix = std::unordered_map<std::string, ImatrixEntry>;

int32_t read_i32(std::ifstream & in, const std::string & what) {
    int32_t value = 0;
    if (!in.read(reinterpret_cast<char *>(&value), sizeof(value))) fail("truncated imatrix " + what);
    return value;
}

Imatrix load_imatrix(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open imatrix " + path.string());
    const int32_t count = read_i32(in, "entry count");
    if (count <= 0 || count > 1000000) fail("invalid imatrix entry count");
    Imatrix result;
    for (int32_t i = 0; i < count; ++i) {
        const int32_t name_len = read_i32(in, "name length");
        if (name_len <= 0 || name_len > 4096) fail("invalid imatrix name length");
        std::string name(static_cast<size_t>(name_len), 0);
        if (!in.read(name.data(), name.size())) fail("truncated imatrix name");
        ImatrixEntry entry;
        entry.calls = read_i32(in, "call count");
        const int32_t nvalues = read_i32(in, "value count");
        if (entry.calls <= 0 || nvalues <= 0 || nvalues > (1 << 24)) fail("invalid imatrix entry dimensions for " + name);
        entry.values.resize(static_cast<size_t>(nvalues));
        if (!in.read(reinterpret_cast<char *>(entry.values.data()), entry.values.size()*sizeof(float))) {
            fail("truncated imatrix values for " + name);
        }
        for (float value : entry.values) {
            if (!std::isfinite(value) || value < 0.0f) fail("invalid imatrix value for " + name);
        }
        if (!result.emplace(name, std::move(entry)).second) fail("duplicate imatrix entry " + name);
    }
    char extra = 0;
    if (in.read(&extra, 1)) fail("trailing bytes in imatrix " + path.string());
    std::cerr << "[calibration] loaded " << result.size() << " imatrix entries from " << path << "\n";
    return result;
}

const std::vector<float> * require_imatrix(
        const std::optional<Imatrix> & imatrix, const std::string & name, size_t in) {
    if (!imatrix) return nullptr;
    const auto it = imatrix->find(name);
    if (it == imatrix->end()) fail("imatrix is missing required entry " + name);
    if (it->second.values.size() != in) {
        fail("imatrix entry " + name + " has " + std::to_string(it->second.values.size()) +
             " values, expected " + std::to_string(in));
    }
    return &it->second.values;
}

enum class Surface : uint32_t { Gate = 0, Up = 1, Down = 2 };
enum class BookSource { GateUpJoint, DownOnly };

struct ExpertRecipe {
    const char * source_leaf;
    const char * target_leaf;
    Surface surface;
    ggml_type qtype;
    BookSource books;
    uint32_t levels;
};

constexpr std::array<ExpertRecipe, 3> kExpertRecipes{{
    {"w1", "ffn_gate_exps.weight", Surface::Gate, GGML_TYPE_Q2_1_ROCMFP2_MIX, BookSource::GateUpJoint, kGuLevels},
    {"w3", "ffn_up_exps.weight",   Surface::Up,   GGML_TYPE_Q2_1_ROCMFP2_MIX, BookSource::GateUpJoint, kGuLevels},
    {"w2", "ffn_down_exps.weight", Surface::Down, GGML_TYPE_Q3_1_ROCMFP3_MIX, BookSource::DownOnly,   kP4Levels},
}};

std::string source_expert_name(int layer, int expert, const ExpertRecipe & recipe, const char * suffix) {
    return "layers." + std::to_string(layer) + ".ffn.experts." + std::to_string(expert) +
           "." + recipe.source_leaf + "." + suffix;
}
std::string target_expert_name(int layer, const ExpertRecipe & recipe) {
    return "blk." + std::to_string(layer) + "." + recipe.target_leaf;
}

struct TensorShape {
    uint32_t in = 0;
    uint32_t out = 0;
};

TensorShape validate_expert_source(
        const SafeTensorSet & source, int layer, int expert, const ExpertRecipe & recipe) {
    const StEntry & weight = source.at(source_expert_name(layer, expert, recipe, "weight"));
    const StEntry & scale = source.at(source_expert_name(layer, expert, recipe, "scale"));
    if (weight.dtype != "I8" || weight.shape.size() != 2) {
        fail("expert weight must be rank-2 packed I8: " + weight.name);
    }
    const int64_t out = weight.shape[0];
    const int64_t in = weight.shape[1]*2;
    if (out <= 0 || in <= 0 || in % kBlock != 0 || out > UINT32_MAX || in > UINT32_MAX) {
        fail("unsupported expert dimensions for " + weight.name);
    }
    if (recipe.qtype == GGML_TYPE_Q2_1_ROCMFP2_MIX && in % 128 != 0) {
        fail("qtype-106 input dimension must be a multiple of 128: " + weight.name);
    }
    if (scale.dtype != "F8_E8M0" || scale.shape != std::vector<int64_t>{out, in/32}) {
        fail("expert scale shape/dtype mismatch for " + scale.name);
    }
    return {static_cast<uint32_t>(in), static_cast<uint32_t>(out)};
}

class HistogramFitter {
public:
    static constexpr int kBins = 4097;
    HistogramFitter() { for (auto & h : hist_) h.assign(kBins, 0.0); }

    void add_half(const float values[16], const float * importance) {
        float max_abs = 0.0f;
        double sum_sq = 0.0;
        for (int i = 0; i < 16; ++i) {
            if (!std::isfinite(values[i])) fail("non-finite expert value during calibration");
            max_abs = std::max(max_abs, std::fabs(values[i]));
            sum_sq += static_cast<double>(values[i])*values[i];
        }
        if (!(max_abs > 0.0f)) {
            hist_[0][kBins/2] += 16.0;
            return;
        }
        const double rms_ratio = std::sqrt(sum_sq/16.0)/max_abs;
        const int population = rms_ratio < 0.52 ? 0 : 1;
        for (int i = 0; i < 16; ++i) {
            const float normalized = std::max(-1.0f, std::min(1.0f, values[i]/max_abs));
            const int bin = std::max(0, std::min(kBins - 1,
                static_cast<int>(std::lround((normalized + 1.0f)*0.5f*(kBins - 1)))));
            double w = importance ? importance[i] : 1.0;
            if (w > 0.0 && std::isfinite(w)) hist_[population][bin] += w;
        }
    }

    static std::vector<uint16_t> round_centers(const std::vector<float> & centers, bool & repaired) {
        std::vector<uint16_t> bits;
        for (float center : centers) {
            if (!std::isfinite(center) || center < -1.0f || center > 1.0f)
                fail("invalid adaptive codebook center");
            bits.push_back(float_to_bf16(center));
        }
        for (size_t i = 1; i < bits.size(); ++i) {
            if (bf16_to_float(bits[i]) <= bf16_to_float(bits[i-1])) {
                bits[i] = bf16_step(bits[i-1], true);
                repaired = true;
            }
        }
        if (!bits.empty() && bf16_to_float(bits.back()) > 1.0f) {
            bits.back() = float_to_bf16(1.0f);
            for (size_t i = bits.size()-1; i > 0; --i) {
                if (bf16_to_float(bits[i-1]) >= bf16_to_float(bits[i]))
                    bits[i-1] = bf16_step(bits[i], false);
            }
        }
        float previous = -std::numeric_limits<float>::infinity();
        for (uint16_t b : bits) {
            const float value = bf16_to_float(b);
            if (!std::isfinite(value) || value < -1.0f || value > 1.0f || value <= previous)
                fail("cannot separate adaptive codebook levels in BF16");
            previous = value;
        }
        return bits;
    }

    std::vector<uint16_t> fit(int levels, const std::string & label = "unlabelled",
                              std::vector<std::string> * repairs = nullptr) const {
        if (levels != 4 && levels != 8) fail("unsupported fitted codebook size");
        std::vector<uint16_t> result;
        result.reserve(static_cast<size_t>(2*levels));
        std::vector<double> fallback(kBins, 0.0);
        for (int i = 0; i < kBins; ++i) fallback[i] = hist_[0][i] + hist_[1][i];
        for (int population = 0; population < 2; ++population) {
            const std::vector<double> & h = total(hist_[population]) > 0.0 ? hist_[population] : fallback;
            const std::vector<float> centers = lloyd(h, levels);
            bool repaired = false;
            const auto bits = round_centers(centers, repaired);
            result.insert(result.end(), bits.begin(), bits.end());
            if (repaired) {
                const std::string stamp = label + " population=" + std::to_string(population) +
                    " levels=" + std::to_string(levels) + " repair=bf16-epsilon-v1";
                std::cerr << "[calibration] WARNING: " << stamp << "\n";
                if (repairs) repairs->push_back(stamp);
            }
        }
        return result;
    }

private:
    static uint16_t bf16_step(uint16_t bits, bool up) {
        if ((bits & 0x7fffu) == 0) return up ? 0x0080u : 0x8080u;
        return static_cast<uint16_t>(bits + ((up != ((bits & 0x8000u) != 0)) ? 1 : -1));
    }
    static double total(const std::vector<double> & h) {
        double out = 0.0;
        for (double value : h) out += value;
        return out;
    }
    static float coordinate(int bin) {
        return -1.0f + 2.0f*static_cast<float>(bin)/static_cast<float>(kBins - 1);
    }
    static std::vector<float> lloyd(const std::vector<double> & h, int k) {
        const double mass = total(h);
        if (!(mass > 0.0)) fail("cannot fit an empty codebook histogram");
        std::vector<float> c(static_cast<size_t>(k));
        for (int j = 0; j < k; ++j) {
            const double target = mass*(j + 0.5)/k;
            double running = 0.0;
            int bin = 0;
            for (; bin < kBins - 1; ++bin) {
                running += h[bin];
                if (running >= target) break;
            }
            c[j] = coordinate(bin);
        }
        for (int iteration = 0; iteration < 24; ++iteration) {
            std::vector<double> sums(static_cast<size_t>(k), 0.0);
            std::vector<double> weights(static_cast<size_t>(k), 0.0);
            int cluster = 0;
            for (int bin = 0; bin < kBins; ++bin) {
                const float x = coordinate(bin);
                while (cluster + 1 < k && x > 0.5f*(c[cluster] + c[cluster + 1])) ++cluster;
                sums[cluster] += h[bin]*x;
                weights[cluster] += h[bin];
            }
            for (int j = 0; j < k; ++j) if (weights[j] > 0.0) c[j] = static_cast<float>(sums[j]/weights[j]);
            std::sort(c.begin(), c.end());
        }
        return c;
    }

    std::array<std::vector<double>, 2> hist_;
};

struct OpenTensorPair {
    FileDescriptor weight_fd;
    FileDescriptor scale_fd;
    const StEntry & weight;
    const StEntry & scale;
    OpenTensorPair(const StEntry & w, const StEntry & s)
        : weight_fd(w.path), scale_fd(s.path), weight(w), scale(s) {}
};

void decode_expert_row(
        const OpenTensorPair & input, uint32_t row, uint32_t in,
        std::vector<uint8_t> & packed, std::vector<uint8_t> & scales,
        std::vector<float> & values) {
    packed.resize(in/2);
    scales.resize(in/32);
    values.resize(in);
    pread_exact(input.weight_fd.fd, packed.data(), packed.size(),
                input.weight.offset + static_cast<uint64_t>(row)*packed.size(), input.weight.name);
    pread_exact(input.scale_fd.fd, scales.data(), scales.size(),
                input.scale.offset + static_cast<uint64_t>(row)*scales.size(), input.scale.name);
    for (uint32_t col = 0; col < in; ++col) {
        const uint8_t byte = packed[col/2];
        const uint8_t nibble = (col & 1) ? (byte >> 4) : (byte & 15u);
        values[col] = fp4_e2m1(nibble)*fp8_e8m0(scales[col/32]);
        if (!std::isfinite(values[col])) fail("non-finite decoded expert value in " + input.weight.name);
    }
}

void add_expert_to_fitter(
        const SafeTensorSet & source, int layer, int expert,
        const ExpertRecipe & recipe, const std::vector<float> * importance,
        HistogramFitter & fitter) {
    const TensorShape shape = validate_expert_source(source, layer, expert, recipe);
    const StEntry & w = source.at(source_expert_name(layer, expert, recipe, "weight"));
    const StEntry & s = source.at(source_expert_name(layer, expert, recipe, "scale"));
    OpenTensorPair input(w, s);
    std::vector<uint8_t> packed, scales;
    std::vector<float> values;
    for (uint32_t row = 0; row < shape.out; ++row) {
        decode_expert_row(input, row, shape.in, packed, scales, values);
        for (uint32_t col = 0; col < shape.in; col += 16) {
            fitter.add_half(values.data() + col,
                            importance ? importance->data() + col : nullptr);
        }
    }
}

struct CodebookRegistry {
    uint32_t levels = 0;
    std::vector<std::vector<uint16_t>> experts;
};

struct LayerCalibration {
    int layer = 0;
    TensorShape gate_up_shape;
    TensorShape down_shape;
    CodebookRegistry gate_up;
    CodebookRegistry down;
    std::vector<std::string> repairs;
};

struct Options {
    std::string recipe = "mix";
    std::string imatrix_provenance;
    bool plan_only = false;
    unsigned encode_threads = 1;
    fs::path input;
    fs::path output;
    std::optional<fs::path> imatrix;
    bool absmax_only = false;
    bool experts_only = false;
    bool force = false;
    bool validate_input_only = false;
    int layer_start = 0;
    int layer_count = -1;
    int expert_limit = -1;
};

void usage(const char * argv0) {
    std::cerr << "Usage: " << argv0 << " --input DIR --output FILE (--imatrix FILE | --absmax-only)\n"
              << "       [--layer-start N] [--layer-count N] [--expert-limit N] [--experts-only] [--validate-input-only] [--force]\n";
    std::cerr << "IQ85: --recipe iq85 --imatrix FILE --imatrix-provenance "
              << "uniform-unvalidated|activation-derived|transferred-text-calibration "
              << "[--plan-only] [--encode-threads 1..16]; fresh output only\n";
}

int parse_nonnegative(const char * value, const std::string & option, bool allow_zero = true) {
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno || !end || *end || parsed < (allow_zero ? 0 : 1) || parsed > INT32_MAX) {
        fail("invalid value for " + option + ": " + value);
    }
    return static_cast<int>(parsed);
}

Options parse_options(int argc, char ** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> const char * {
            if (++i >= argc) fail("missing value after " + arg);
            return argv[i];
        };
        if (arg == "--input") out.input = value();
        else if (arg == "--recipe") out.recipe = value();
        else if (arg == "--imatrix-provenance") out.imatrix_provenance = value();
        else if (arg == "--plan-only") out.plan_only = true;
        else if (arg == "--encode-threads") out.encode_threads = parse_nonnegative(value(), arg, false);
        else if (arg == "--output") out.output = value();
        else if (arg == "--imatrix") out.imatrix = fs::path(value());
        else if (arg == "--absmax-only") out.absmax_only = true;
        else if (arg == "--experts-only") out.experts_only = true;
        else if (arg == "--force") out.force = true;
        else if (arg == "--validate-input-only") out.validate_input_only = true;
        else if (arg == "--layer-start") out.layer_start = parse_nonnegative(value(), arg);
        else if (arg == "--layer-count") out.layer_count = parse_nonnegative(value(), arg, false);
        else if (arg == "--expert-limit") out.expert_limit = parse_nonnegative(value(), arg, false);
        else if (arg == "--help" || arg == "-h") { usage(argv[0]); std::exit(0); }
        else fail("unknown option " + arg);
    }
    if (out.input.empty() || out.output.empty()) fail("--input and --output are required");
    if (out.recipe != "mix" && out.recipe != "iq85") fail("--recipe must be mix or iq85");
    if (out.encode_threads > 16) fail("--encode-threads must be in 1..16");
    if (out.recipe == "iq85") {
        if (out.force) fail("iq85 never overwrites artifacts; use a fresh output path");
        if (out.absmax_only || !out.imatrix) fail("iq85 requires --imatrix; --absmax-only is unsupported");
        if (out.imatrix_provenance != "uniform-unvalidated" && out.imatrix_provenance != "activation-derived" &&
            out.imatrix_provenance != "transferred-text-calibration")
            fail("iq85 requires --imatrix-provenance uniform-unvalidated|activation-derived|transferred-text-calibration");
    } else if (out.plan_only || out.encode_threads != 1 || !out.imatrix_provenance.empty()) {
        fail("--plan-only, --encode-threads and --imatrix-provenance require --recipe iq85");
    }
    if (out.absmax_only == out.imatrix.has_value()) {
        fail("choose exactly one of --imatrix FILE or --absmax-only");
    }
    return out;
}

uint32_t config_u32(const json & c, const char * key, uint32_t fallback = 0) {
    if (!c.contains(key)) {
        if (fallback != 0) return fallback;
        fail(std::string("config.json is missing ") + key);
    }
    const int64_t value = c[key].get<int64_t>();
    if (value <= 0 || value > UINT32_MAX) fail(std::string("invalid config value ") + key);
    return static_cast<uint32_t>(value);
}

float config_f32(const json & c, const char * key, float fallback) {
    if (!c.contains(key)) return fallback;
    const float value = c[key].get<float>();
    if (!std::isfinite(value)) fail(std::string("non-finite config value ") + key);
    return value;
}

template<typename T>
void append_le(std::vector<uint8_t> & out, T value) {
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    fail("big-endian hosts are not supported");
#endif
    const auto * p = reinterpret_cast<const uint8_t *>(&value);
    out.insert(out.end(), p, p + sizeof(value));
}

std::vector<uint8_t> make_p4_blob(const std::vector<LayerCalibration> & layers, uint32_t experts) {
    std::vector<uint8_t> out;
    const char magic[8] = {'P','4','M','I','X','v','1','\0'};
    out.insert(out.end(), magic, magic + 8);
    append_le<uint32_t>(out, static_cast<uint32_t>(layers.size()));
    append_le<uint32_t>(out, 0);
    for (const LayerCalibration & layer : layers) {
        if (layer.down.levels != kP4Levels || layer.down.experts.size() != experts) {
            fail("incomplete qtype-105 codebook registry at layer " + std::to_string(layer.layer));
        }
        append_le<uint32_t>(out, static_cast<uint32_t>(layer.layer));
        append_le<uint32_t>(out, experts);
        append_le<uint32_t>(out, layer.down_shape.out);
        append_le<uint32_t>(out, layer.down_shape.in);
        append_le<uint32_t>(out, kCodebooks);
        append_le<uint32_t>(out, kP4Levels);
        out.insert(out.end(), experts, 1u);
        out.insert(out.end(), experts, 0u);
        for (const auto & book : layer.down.experts) {
            if (book.size() != kCodebooks*kP4Levels) fail("bad qtype-105 codebook length");
            for (uint16_t value : book) append_le<uint16_t>(out, value);
        }
    }
    return out;
}

std::vector<uint8_t> make_gumix_blob(const std::vector<LayerCalibration> & layers, uint32_t experts) {
    std::vector<uint8_t> out;
    const char magic[8] = {'G','U','M','I','X','s','1','\0'};
    out.insert(out.end(), magic, magic + 8);
    append_le<uint32_t>(out, static_cast<uint32_t>(layers.size()*2));
    append_le<uint32_t>(out, 0);
    for (const LayerCalibration & layer : layers) {
        if (layer.gate_up.levels != kGuLevels || layer.gate_up.experts.size() != experts) {
            fail("incomplete qtype-106 codebook registry at layer " + std::to_string(layer.layer));
        }
        for (Surface surface : {Surface::Gate, Surface::Up}) {
            append_le<uint32_t>(out, static_cast<uint32_t>(layer.layer));
            append_le<uint32_t>(out, static_cast<uint32_t>(surface));
            append_le<uint32_t>(out, experts);
            append_le<uint32_t>(out, layer.gate_up_shape.out);
            append_le<uint32_t>(out, layer.gate_up_shape.in);
            append_le<uint32_t>(out, kCodebooks);
            append_le<uint32_t>(out, kGuLevels);
            out.insert(out.end(), experts, 1u);
            for (const auto & book : layer.gate_up.experts) {
                if (book.size() != kCodebooks*kGuLevels) fail("bad qtype-106 codebook length");
                for (uint16_t value : book) append_le<uint16_t>(out, value);
            }
        }
    }
    return out;
}

void write_atomic_bytes(const fs::path & path, const std::vector<uint8_t> & bytes, bool force) {
    if (!force && fs::exists(path)) fail("output exists: " + path.string());
    const fs::path temporary = path.string() + ".partial";
    if (fs::exists(temporary)) fs::remove(temporary);
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) fail("cannot create " + temporary.string());
    out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    out.close();
    if (!out) fail("failed writing " + temporary.string());
    if (force && fs::exists(path)) fs::remove(path);
    fs::rename(temporary, path);
}

enum class Producer { Raw, DenseFp8, Int64ToInt32, Expert };

struct TensorSpec {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
    std::vector<int64_t> ne;
    Producer producer = Producer::Raw;
    const StEntry * source = nullptr;
    const StEntry * scale = nullptr;
    int layer = -1;
    const ExpertRecipe * recipe = nullptr;
};

std::vector<int64_t> reverse_shape(const StEntry & entry) {
    std::vector<int64_t> ne(entry.shape.rbegin(), entry.shape.rend());
    while (ne.size() < GGML_MAX_DIMS) ne.push_back(1);
    return ne;
}

ggml_type direct_ggml_type(const std::string & dtype) {
    if (dtype == "BF16") return GGML_TYPE_BF16;
    if (dtype == "F16") return GGML_TYPE_F16;
    if (dtype == "F32") return GGML_TYPE_F32;
    if (dtype == "I32") return GGML_TYPE_I32;
    fail("unsupported direct tensor dtype " + dtype);
}

const std::unordered_map<std::string, std::string> kLayerNameMap = {
    {"attn.attn_sink", "attn_sinks.weight"},
    {"attn.kv_norm.weight", "attn_kv_a_norm.weight"},
    {"attn.q_norm.weight", "attn_q_a_norm.weight"},
    {"attn.wkv.weight", "attn_kv.weight"},
    {"attn.wo_a.weight", "attn_output_a.weight"},
    {"attn.wo_b.weight", "attn_output_b.weight"},
    {"attn.wq_a.weight", "attn_q_a.weight"},
    {"attn.wq_b.weight", "attn_q_b.weight"},
    {"attn_norm.weight", "attn_norm.weight"},
    {"attn.compressor.ape", "attn_compressor_ape.weight"},
    {"attn.compressor.norm.weight", "attn_compressor_norm.weight"},
    {"attn.compressor.wgate.weight", "attn_compressor_gate.weight"},
    {"attn.compressor.wkv.weight", "attn_compressor_kv.weight"},
    {"attn.indexer.compressor.ape", "indexer_compressor_ape.weight"},
    {"attn.indexer.compressor.norm.weight", "indexer_compressor_norm.weight"},
    {"attn.indexer.compressor.wgate.weight", "indexer_compressor_gate.weight"},
    {"attn.indexer.compressor.wkv.weight", "indexer_compressor_kv.weight"},
    {"attn.indexer.weights_proj.weight", "indexer.proj.weight"},
    {"attn.indexer.wq_b.weight", "indexer.attn_q_b.weight"},
    {"ffn.gate.bias", "exp_probs_b.bias"},
    {"ffn.gate.tid2eid", "ffn_gate_tid2eid.weight"},
    {"ffn.gate.weight", "ffn_gate_inp.weight"},
    {"ffn.shared_experts.w1.weight", "ffn_gate_shexp.weight"},
    {"ffn.shared_experts.w2.weight", "ffn_down_shexp.weight"},
    {"ffn.shared_experts.w3.weight", "ffn_up_shexp.weight"},
    {"ffn_norm.weight", "ffn_norm.weight"},
    {"hc_attn_base", "hc_attn_base.weight"},
    {"hc_attn_fn", "hc_attn_fn.weight"},
    {"hc_attn_scale", "hc_attn_scale.weight"},
    {"hc_ffn_base", "hc_ffn_base.weight"},
    {"hc_ffn_fn", "hc_ffn_fn.weight"},
    {"hc_ffn_scale", "hc_ffn_scale.weight"},
};

const std::unordered_map<std::string, std::string> kGlobalNameMap = {
    {"embed.weight", "token_embd.weight"},
    {"head.weight", "output.weight"},
    {"norm.weight", "output_norm.weight"},
    {"hc_head_base", "output_hc_base.weight"},
    {"hc_head_fn", "output_hc_fn.weight"},
    {"hc_head_scale", "output_hc_scale.weight"},
};

bool starts_with(const std::string & value, const std::string & prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}
bool ends_with(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::optional<std::pair<int, std::string>> parse_layer_name(const std::string & name) {
    if (!starts_with(name, "layers.")) return std::nullopt;
    const size_t dot = name.find('.', 7);
    if (dot == std::string::npos) fail("malformed layer tensor name " + name);
    const std::string number = name.substr(7, dot - 7);
    if (number.empty() || number.find_first_not_of("0123456789") != std::string::npos) {
        fail("malformed layer index in " + name);
    }
    return std::make_pair(std::stoi(number), name.substr(dot + 1));
}

TensorSpec mapped_source_spec(const std::string & target, const StEntry & source,
                              const SafeTensorSet & all) {
    TensorSpec spec;
    spec.name = target;
    spec.source = &source;
    if (source.dtype == "I64") {
        if (!ends_with(source.name, ".tid2eid")) fail("unsupported I64 tensor " + source.name);
        spec.type = GGML_TYPE_I32;
        spec.producer = Producer::Int64ToInt32;
    } else if (source.dtype == "F8_E4M3") {
        if (!ends_with(source.name, ".weight")) fail("FP8 tensor is not a weight: " + source.name);
        const std::string scale_name = source.name.substr(0, source.name.size() - 6) + "scale";
        const StEntry & scale = all.at(scale_name);
        if (scale.dtype != "F8_E8M0" || source.shape.size() != 2 || scale.shape.size() != 2) {
            fail("unsupported FP8 tensor/scale pair " + source.name);
        }
        const std::vector<int64_t> expected = {
            (source.shape[0] + 127)/128, (source.shape[1] + 127)/128,
        };
        if (scale.shape != expected) fail("FP8 scale shape mismatch for " + source.name);
        spec.type = GGML_TYPE_BF16;
        spec.producer = Producer::DenseFp8;
        spec.scale = &scale;
    } else {
        spec.type = direct_ggml_type(source.dtype);
        spec.producer = Producer::Raw;
    }
    spec.ne = reverse_shape(source);
    if (spec.name.size() >= GGML_MAX_NAME) fail("GGUF tensor name too long: " + spec.name);
    return spec;
}

std::vector<TensorSpec> make_plan(
        const SafeTensorSet & source, const std::vector<LayerCalibration> & layers,
        uint32_t experts, bool experts_only) {
    std::vector<TensorSpec> plan;
    std::set<std::string> names;
    auto add = [&](TensorSpec spec) {
        if (!names.insert(spec.name).second) fail("duplicate output tensor name " + spec.name);
        plan.push_back(std::move(spec));
    };

    if (!experts_only) {
        for (const auto & mapping : kGlobalNameMap) {
            add(mapped_source_spec(mapping.second, source.at(mapping.first), source));
        }
    }

    for (const LayerCalibration & layer : layers) {
        if (!experts_only) {
            const std::string prefix = "layers." + std::to_string(layer.layer) + ".";
            for (const auto & mapping : kLayerNameMap) {
                const std::string src = prefix + mapping.first;
                if (!source.contains(src)) continue;
                if (ends_with(mapping.first, ".scale")) continue;
                add(mapped_source_spec("blk." + std::to_string(layer.layer) + "." + mapping.second,
                                       source.at(src), source));
            }
        }
        for (const ExpertRecipe & recipe : kExpertRecipes) {
            const TensorShape shape = recipe.books == BookSource::GateUpJoint
                ? layer.gate_up_shape : layer.down_shape;
            TensorSpec spec;
            spec.name = target_expert_name(layer.layer, recipe);
            spec.type = recipe.qtype;
            spec.ne = {shape.in, shape.out, static_cast<int64_t>(experts), 1};
            spec.producer = Producer::Expert;
            spec.layer = layer.layer;
            spec.recipe = &recipe;
            add(std::move(spec));
        }
    }

    if (!experts_only) {
        bool dropped_mtp = false;
        static const std::regex expert_leaf(
            R"(^ffn\.experts\.[0-9]+\.(w1|w2|w3)\.(weight|scale)$)");
        for (const auto & item : source.entries()) {
            const std::string & name = item.first;
            if (starts_with(name, "vision.") || starts_with(name, "aligner.") || starts_with(name, "image_")) {
                TensorSpec spec = mapped_source_spec(name, item.second, source);
                if (spec.producer != Producer::Raw) fail("vision pass-through is not byte-lossless for " + name);
                add(std::move(spec));
                continue;
            }
            if (starts_with(name, "mtp.")) {
                dropped_mtp = true;
                continue;
            }
            if (kGlobalNameMap.count(name)) continue;
            const auto parsed = parse_layer_name(name);
            if (!parsed) fail("unsupported top-level tensor " + name);
            const int layer = parsed->first;
            const std::string & leaf = parsed->second;
            const bool selected = std::any_of(layers.begin(), layers.end(),
                [layer](const LayerCalibration & c) { return c.layer == layer; });
            if (!selected) continue;
            if (leaf == "ffn.gate.bias_vl") {
                TensorSpec spec = mapped_source_spec(name, item.second, source);
                if (spec.producer != Producer::Raw) fail("bias_vl pass-through is not byte-lossless for " + name);
                add(std::move(spec));
                continue;
            }
            if (starts_with(leaf, "ffn.experts.")) {
                if (!std::regex_match(leaf, expert_leaf)) fail("unsupported expert tensor " + name);
                continue;
            }
            if (kLayerNameMap.count(leaf) || (ends_with(leaf, ".scale") &&
                source.contains(name.substr(0, name.size() - 5) + "weight"))) continue;
            fail("unsupported selected-layer tensor " + name);
        }
        if (dropped_mtp) {
            std::cerr << "[plan] NOTE: mtp.* predictor tensors are intentionally omitted; "
                      << "the DeepSeek4 target loader does not consume them\n";
        }
    }

    std::sort(plan.begin(), plan.end(), [](const TensorSpec & a, const TensorSpec & b) {
        return a.name < b.name;
    });
    return plan;
}

void set_model_metadata(gguf_context * ctx, const SafeTensorSet & source,
                        uint32_t layers, uint32_t experts, bool absmax_only,
                        bool smoke_artifact, const std::vector<uint8_t> & p4_blob) {
    const json & c = source.config();
    gguf_set_val_str(ctx, "general.architecture", "deepseek4");
    gguf_set_val_str(ctx, "general.name", "DeepSeek-V4-Flash-Vision-Uncensored MIX");
    gguf_set_val_u32(ctx, "general.alignment", kAlignment);
    gguf_set_val_u32(ctx, "general.file_type", 119);
    gguf_set_val_str(ctx, "deepseek4.mix.calibration",
                     absmax_only ? "absmax-only (LOWER QUALITY; no imatrix)" : "importance-matrix weighted");
    gguf_set_val_bool(ctx, "deepseek4.mix.lower_quality_absmax_only", absmax_only);
    gguf_set_val_bool(ctx, "deepseek4.mix.experts_only_smoke_artifact", smoke_artifact);

    gguf_set_val_u32(ctx, "deepseek4.block_count", layers);
    gguf_set_val_u32(ctx, "deepseek4.embedding_length", config_u32(c, "hidden_size"));
    gguf_set_val_u32(ctx, "deepseek4.vocab_size", config_u32(c, "vocab_size"));
    gguf_set_val_u32(ctx, "deepseek4.attention.head_count", config_u32(c, "num_attention_heads"));
    gguf_set_val_u32(ctx, "deepseek4.attention.head_count_kv", config_u32(c, "num_key_value_heads"));
    gguf_set_val_u32(ctx, "deepseek4.attention.key_length", config_u32(c, "head_dim"));
    gguf_set_val_u32(ctx, "deepseek4.rope.dimension_count", config_u32(c, "qk_rope_head_dim"));
    gguf_set_val_u32(ctx, "deepseek4.attention.q_lora_rank", config_u32(c, "q_lora_rank"));
    gguf_set_val_u32(ctx, "deepseek4.attention.output_lora_rank", config_u32(c, "o_lora_rank"));
    gguf_set_val_u32(ctx, "deepseek4.attention.output_group_count", config_u32(c, "o_groups"));
    gguf_set_val_u32(ctx, "deepseek4.expert_count", experts);
    gguf_set_val_u32(ctx, "deepseek4.expert_used_count", std::min(experts, config_u32(c, "num_experts_per_tok")));
    gguf_set_val_u32(ctx, "deepseek4.expert_shared_count", config_u32(c, "n_shared_experts"));
    gguf_set_val_u32(ctx, "deepseek4.expert_feed_forward_length", config_u32(c, "moe_intermediate_size"));
    gguf_set_val_u32(ctx, "deepseek4.hash_layer_count", std::min(layers, config_u32(c, "num_hash_layers")));
    gguf_set_val_u32(ctx, "deepseek4.attention.sliding_window", config_u32(c, "sliding_window"));
    gguf_set_val_u32(ctx, "deepseek4.attention.indexer.head_count", config_u32(c, "index_n_heads"));
    gguf_set_val_u32(ctx, "deepseek4.attention.indexer.key_length", config_u32(c, "index_head_dim"));
    gguf_set_val_u32(ctx, "deepseek4.attention.indexer.top_k", config_u32(c, "index_topk"));
    gguf_set_val_u32(ctx, "deepseek4.hyper_connection.count", config_u32(c, "hc_mult"));
    gguf_set_val_u32(ctx, "deepseek4.hyper_connection.sinkhorn_iterations", config_u32(c, "hc_sinkhorn_iters"));

    gguf_set_val_f32(ctx, "deepseek4.rope.freq_base", config_f32(c, "rope_theta", 10000.0f));
    gguf_set_val_f32(ctx, "deepseek4.rope.scaling.factor", 16.0f);
    gguf_set_val_f32(ctx, "deepseek4.rope.scaling.yarn_beta_fast", 32.0f);
    gguf_set_val_f32(ctx, "deepseek4.rope.scaling.yarn_beta_slow", 1.0f);
    gguf_set_val_f32(ctx, "deepseek4.attention.compress_rope_freq_base", config_f32(c, "compress_rope_theta", 160000.0f));
    gguf_set_val_u64(ctx, "deepseek4.rope.scaling.original_context_length", 65536);
    gguf_set_val_f32(ctx, "deepseek4.attention.layer_norm_rms_epsilon", 1e-6f);
    gguf_set_val_f32(ctx, "deepseek4.hyper_connection.epsilon", config_f32(c, "hc_eps", 1e-6f));
    gguf_set_val_f32(ctx, "deepseek4.expert_weights_scale", config_f32(c, "routed_scaling_factor", 1.5f));
    gguf_set_val_f32(ctx, "deepseek4.swiglu_clamp_exp", config_f32(c, "swiglu_limit", 10.0f));

    std::vector<uint32_t> ratios(layers, 0);
    if (c.contains("compress_ratios") && c["compress_ratios"].is_array()) {
        if (c["compress_ratios"].size() < layers) fail("config compress_ratios is shorter than selected layers");
        for (uint32_t i = 0; i < layers; ++i) ratios[i] = c["compress_ratios"][i].get<uint32_t>();
    }
    gguf_set_arr_data(ctx, "deepseek4.attention.compress_ratios", GGUF_TYPE_UINT32,
                      ratios.data(), ratios.size());

    gguf_set_val_str(ctx, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(ctx, "tokenizer.ggml.pre", "deepseek-v3");
    gguf_set_val_u32(ctx, "tokenizer.ggml.bos_token_id", c.value("bos_token_id", 0u));
    gguf_set_val_u32(ctx, "tokenizer.ggml.eos_token_id", c.value("eos_token_id", 1u));
    gguf_set_val_u32(ctx, "tokenizer.ggml.padding_token_id", c.value("eos_token_id", 1u));
    gguf_set_val_bool(ctx, "tokenizer.ggml.add_bos_token", false);
    gguf_set_val_bool(ctx, "tokenizer.ggml.add_eos_token", false);

    const uint32_t vocab_size = config_u32(c, "vocab_size");
    std::vector<std::string> tokens(vocab_size);
    std::vector<int32_t> types(vocab_size, 1);
    std::vector<float> scores(vocab_size, 0.0f);
    std::vector<bool> assigned(vocab_size, false);
    const json & tok = source.tokenizer();
    if (!tok.contains("model") || !tok["model"].contains("vocab") ||
        !tok["model"].contains("merges")) fail("tokenizer.json is missing BPE model fields");
    for (const auto & item : tok["model"]["vocab"].items()) {
        const int64_t id = item.value().get<int64_t>();
        if (id < 0 || id >= vocab_size) fail("tokenizer vocab id out of range");
        tokens[id] = item.key();
        assigned[id] = true;
    }
    if (tok.contains("added_tokens")) {
        for (const json & added : tok["added_tokens"]) {
            const int64_t id = added.at("id").get<int64_t>();
            if (id < 0 || id >= vocab_size) fail("added token id out of range");
            tokens[id] = added.at("content").get<std::string>();
            types[id] = added.value("special", true) ? 3 : 4;
            assigned[id] = true;
        }
    }
    for (uint32_t i = 0; i < vocab_size; ++i) {
        if (!assigned[i]) fail("tokenizer has no token for id " + std::to_string(i));
    }
    std::vector<const char *> token_ptrs;
    token_ptrs.reserve(tokens.size());
    for (const std::string & token : tokens) token_ptrs.push_back(token.c_str());
    gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(ctx, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(ctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());

    std::vector<std::string> merges = tok["model"]["merges"].get<std::vector<std::string>>();
    std::vector<const char *> merge_ptrs;
    merge_ptrs.reserve(merges.size());
    for (const std::string & merge : merges) merge_ptrs.push_back(merge.c_str());
    gguf_set_arr_str(ctx, "tokenizer.ggml.merges", merge_ptrs.data(), merge_ptrs.size());

    gguf_set_arr_data(ctx, "deepseek4.p4mix.sidecar", GGUF_TYPE_UINT8,
                      p4_blob.data(), p4_blob.size());
}

std::unique_ptr<ggml_tensor> make_tensor_descriptor(const TensorSpec & spec) {
    auto tensor = std::make_unique<ggml_tensor>();
    std::memset(tensor.get(), 0, sizeof(*tensor));
    tensor->type = spec.type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) tensor->ne[i] = i < static_cast<int>(spec.ne.size()) ? spec.ne[i] : 1;
    const int64_t block = ggml_blck_size(spec.type);
    if (block <= 0 || tensor->ne[0] % block != 0) {
        fail("tensor " + spec.name + " ne[0] violates qtype block size");
    }
    tensor->nb[0] = ggml_type_size(spec.type);
    tensor->nb[1] = tensor->nb[0]*static_cast<size_t>(tensor->ne[0]/block);
    for (int i = 2; i < GGML_MAX_DIMS; ++i) tensor->nb[i] = tensor->nb[i - 1]*static_cast<size_t>(tensor->ne[i - 1]);
    std::snprintf(tensor->name, sizeof(tensor->name), "%s", spec.name.c_str());
    return tensor;
}

void fwrite_exact(FILE * out, const void * data, size_t n, const std::string & what) {
    if (n && std::fwrite(data, 1, n, out) != n) fail("failed writing " + what);
}

void copy_raw(FILE * out, const StEntry & source) {
    FileDescriptor fd(source.path);
    std::vector<uint8_t> buffer(std::min<uint64_t>(source.size, kCopyChunk));
    uint64_t done = 0;
    while (done < source.size) {
        const size_t n = static_cast<size_t>(std::min<uint64_t>(buffer.size(), source.size - done));
        pread_exact(fd.fd, buffer.data(), n, source.offset + done, source.name);
        fwrite_exact(out, buffer.data(), n, source.name);
        done += n;
    }
}

void write_int64_to_int32(FILE * out, const StEntry & source) {
    FileDescriptor fd(source.path);
    const uint64_t count = element_count(source.shape, source.name);
    std::vector<int64_t> input(std::min<uint64_t>(count, kCopyChunk/sizeof(int64_t)));
    std::vector<int32_t> output(input.size());
    uint64_t done = 0;
    while (done < count) {
        const size_t n = static_cast<size_t>(std::min<uint64_t>(input.size(), count - done));
        pread_exact(fd.fd, input.data(), n*sizeof(int64_t), source.offset + done*sizeof(int64_t), source.name);
        for (size_t i = 0; i < n; ++i) {
            if (input[i] < INT32_MIN || input[i] > INT32_MAX) fail("I64 routing id out of I32 range in " + source.name);
            output[i] = static_cast<int32_t>(input[i]);
        }
        fwrite_exact(out, output.data(), n*sizeof(int32_t), source.name);
        done += n;
    }
}

void write_dense_fp8(FILE * out, const StEntry & weight, const StEntry & scale) {
    const uint32_t rows = static_cast<uint32_t>(weight.shape[0]);
    const uint32_t cols = static_cast<uint32_t>(weight.shape[1]);
    FileDescriptor wf(weight.path), sf(scale.path);
    std::vector<uint8_t> scales(scale.size);
    pread_exact(sf.fd, scales.data(), scales.size(), scale.offset, scale.name);
    std::vector<uint8_t> input(cols);
    std::vector<uint16_t> output(cols);
    const uint32_t scale_cols = static_cast<uint32_t>(scale.shape[1]);
    for (uint32_t row = 0; row < rows; ++row) {
        pread_exact(wf.fd, input.data(), input.size(), weight.offset + static_cast<uint64_t>(row)*cols, weight.name);
        for (uint32_t col = 0; col < cols; ++col) {
            const uint8_t scale_byte = scales[(row/128)*scale_cols + col/128];
            const float decoded = fp8_e4m3fn(input[col])*fp8_e8m0(scale_byte);
            const uint16_t b = float_to_bf16(decoded);
            if (bf16_to_float(b) != decoded) {
                fail("FP8->BF16 is not exact for " + weight.name + " at row " +
                     std::to_string(row) + " col " + std::to_string(col));
            }
            output[col] = b;
        }
        fwrite_exact(out, output.data(), output.size()*sizeof(uint16_t), weight.name);
    }
}

const LayerCalibration & find_calibration(const std::vector<LayerCalibration> & all, int layer) {
    const auto it = std::find_if(all.begin(), all.end(), [layer](const LayerCalibration & c) { return c.layer == layer; });
    if (it == all.end()) fail("internal error: missing calibration for layer " + std::to_string(layer));
    return *it;
}

void write_expert_tensor(FILE * out, const SafeTensorSet & source,
                         const LayerCalibration & calibration, uint32_t experts,
                         const ExpertRecipe & recipe, const std::optional<Imatrix> & imatrix) {
    const TensorShape expected = recipe.books == BookSource::GateUpJoint
        ? calibration.gate_up_shape : calibration.down_shape;
    const CodebookRegistry & registry = recipe.books == BookSource::GateUpJoint
        ? calibration.gate_up : calibration.down;
    const std::string target = target_expert_name(calibration.layer, recipe);
    const std::vector<float> * importance = require_imatrix(imatrix, target, expected.in);
    std::vector<uint8_t> packed, scales;
    std::vector<float> values;
    std::vector<block_rocmfp2> q2(expected.in/kBlock);
    std::vector<block_rocmfp3> q3(expected.in/kBlock);
    for (uint32_t expert = 0; expert < experts; ++expert) {
        const TensorShape shape = validate_expert_source(source, calibration.layer, expert, recipe);
        if (shape.in != expected.in || shape.out != expected.out) fail("expert shape drift in " + target);
        const StEntry & w = source.at(source_expert_name(calibration.layer, expert, recipe, "weight"));
        const StEntry & s = source.at(source_expert_name(calibration.layer, expert, recipe, "scale"));
        OpenTensorPair input(w, s);
        const auto & books = registry.experts.at(expert);
        for (uint32_t row = 0; row < shape.out; ++row) {
            decode_expert_row(input, row, shape.in, packed, scales, values);
            if (recipe.qtype == GGML_TYPE_Q2_1_ROCMFP2_MIX) {
                if (!rocmfpx_quantize_row_fp2_mix_ref(values.data(), q2.data(), shape.in,
                                                       books.data(), importance ? importance->data() : nullptr)) {
                    fail("qtype-106 reference encoder rejected " + w.name);
                }
                fwrite_exact(out, q2.data(), q2.size()*sizeof(q2[0]), target);
            } else if (recipe.qtype == GGML_TYPE_Q3_1_ROCMFP3_MIX) {
                if (!rocmfpx_quantize_row_fp3_mix_ref(values.data(), q3.data(), shape.in,
                                                       books.data(), importance ? importance->data() : nullptr)) {
                    fail("qtype-105 reference encoder rejected " + w.name);
                }
                fwrite_exact(out, q3.data(), q3.size()*sizeof(q3[0]), target);
            } else {
                fail("recipe table contains unsupported qtype");
            }
        }
        std::cerr << "[encode] " << target << " expert " << (expert + 1) << "/" << experts << "\n";
    }
}

std::vector<LayerCalibration> validate_input_layout(
        const SafeTensorSet & source, uint32_t layers, uint32_t experts) {
    std::vector<LayerCalibration> result;
    result.reserve(layers);
    for (uint32_t layer = 0; layer < layers; ++layer) {
        LayerCalibration current;
        current.layer = static_cast<int>(layer);
        for (uint32_t expert = 0; expert < experts; ++expert) {
            for (const ExpertRecipe & recipe : kExpertRecipes) {
                const TensorShape shape = validate_expert_source(source, layer, expert, recipe);
                TensorShape & expected = recipe.books == BookSource::GateUpJoint
                    ? current.gate_up_shape : current.down_shape;
                if (expected.in == 0) expected = shape;
                if (shape.in != expected.in || shape.out != expected.out) {
                    fail("expert shape drift during input validation at layer " + std::to_string(layer));
                }
            }
        }
        result.push_back(std::move(current));
    }
    return result;
}

std::vector<LayerCalibration> calibrate(
        const SafeTensorSet & source, uint32_t layers, uint32_t experts,
        const std::optional<Imatrix> & imatrix) {
    std::vector<LayerCalibration> result;
    result.reserve(layers);
    for (uint32_t layer = 0; layer < layers; ++layer) {
        LayerCalibration current;
        current.layer = static_cast<int>(layer);
        current.gate_up.levels = kGuLevels;
        current.down.levels = kP4Levels;
        for (uint32_t expert = 0; expert < experts; ++expert) {
            HistogramFitter gate_up_fitter;
            TensorShape gate_shape{};
            for (const ExpertRecipe & recipe : kExpertRecipes) {
                if (recipe.books != BookSource::GateUpJoint) continue;
                const TensorShape shape = validate_expert_source(source, layer, expert, recipe);
                if (gate_shape.in == 0) gate_shape = shape;
                if (shape.in != gate_shape.in || shape.out != gate_shape.out) {
                    fail("qtype-106 gate/up shape mismatch at layer " + std::to_string(layer));
                }
                const std::string target = target_expert_name(layer, recipe);
                const auto * importance = require_imatrix(imatrix, target, shape.in);
                add_expert_to_fitter(source, layer, expert, recipe, importance, gate_up_fitter);
            }
            HistogramFitter down_fitter;
            const ExpertRecipe & down_recipe = kExpertRecipes[2];
            const TensorShape down_shape = validate_expert_source(source, layer, expert, down_recipe);
            const auto * down_importance = require_imatrix(
                imatrix, target_expert_name(layer, down_recipe), down_shape.in);
            add_expert_to_fitter(source, layer, expert, down_recipe, down_importance, down_fitter);

            if (current.gate_up_shape.in == 0) current.gate_up_shape = gate_shape;
            if (current.down_shape.in == 0) current.down_shape = down_shape;
            if (current.gate_up_shape.in != gate_shape.in || current.gate_up_shape.out != gate_shape.out ||
                current.down_shape.in != down_shape.in || current.down_shape.out != down_shape.out) {
                fail("expert shapes vary within layer " + std::to_string(layer));
            }
            const std::string label = "layer=" + std::to_string(layer) + " expert=" + std::to_string(expert);
            current.gate_up.experts.push_back(gate_up_fitter.fit(kGuLevels, label + " gate_up", &current.repairs));
            current.down.experts.push_back(down_fitter.fit(kP4Levels, label + " down", &current.repairs));
            std::cerr << "[calibration] layer " << layer << " expert " << (expert + 1)
                      << "/" << experts << " fitted joint gate/up and down codebooks\n";
        }
        result.push_back(std::move(current));
    }
    return result;
}

void compare_raw_passthrough(int output_fd, uint64_t output_offset, const StEntry & input) {
    FileDescriptor source_fd(input.path);
    std::vector<uint8_t> a(std::min<uint64_t>(input.size, kCopyChunk));
    std::vector<uint8_t> b(a.size());
    uint64_t done = 0;
    while (done < input.size) {
        const size_t n = static_cast<size_t>(std::min<uint64_t>(a.size(), input.size - done));
        pread_exact(source_fd.fd, a.data(), n, input.offset + done, input.name + " source verification");
        pread_exact(output_fd, b.data(), n, output_offset + done, input.name + " output verification");
        if (std::memcmp(a.data(), b.data(), n) != 0) fail("lossless pass-through verification failed for " + input.name);
        done += n;
    }
}

std::vector<uint8_t> read_file(const fs::path & path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) fail("cannot open " + path.string());
    const auto end = in.tellg();
    if (end < 0) fail("cannot size " + path.string());
    std::vector<uint8_t> out(static_cast<size_t>(end));
    in.seekg(0);
    if (!out.empty() && !in.read(reinterpret_cast<char *>(out.data()), out.size())) fail("cannot read " + path.string());
    return out;
}

void verify_artifact(const fs::path & output, const fs::path & gumix_path,
                     const std::vector<TensorSpec> & plan,
                     const std::vector<uint8_t> & expected_p4,
                     const std::vector<uint8_t> & expected_gumix) {
    ggml_context * meta = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &meta;
    gguf_context * ctx = gguf_init_from_file(output.c_str(), params);
    if (!ctx) fail("GGUF verification could not parse " + output.string());
    const uint64_t file_size = fs::file_size(output);
    const uint64_t data_offset = gguf_get_data_offset(ctx);
    const int64_t p4_key = gguf_find_key(ctx, "deepseek4.p4mix.sidecar");
    if (p4_key < 0 || gguf_get_kv_type(ctx, p4_key) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(ctx, p4_key) != GGUF_TYPE_UINT8 ||
        gguf_get_arr_n(ctx, p4_key) != expected_p4.size() ||
        std::memcmp(gguf_get_arr_data(ctx, p4_key), expected_p4.data(), expected_p4.size()) != 0) {
        fail("embedded deepseek4.p4mix.sidecar verification failed");
    }
    if (read_file(gumix_path) != expected_gumix) fail("qtype-106 gumix sidecar verification failed");

    FileDescriptor output_fd(output);
    for (const TensorSpec & spec : plan) {
        const int64_t id = gguf_find_tensor(ctx, spec.name.c_str());
        if (id < 0) fail("output is missing tensor " + spec.name);
        if (gguf_get_tensor_type(ctx, id) != spec.type) fail("output qtype mismatch for " + spec.name);
        const uint64_t offset = data_offset + gguf_get_tensor_offset(ctx, id);
        const uint64_t size = gguf_get_tensor_size(ctx, id);
        if (offset > file_size || size > file_size - offset) fail("output tensor is out of file bounds: " + spec.name);
        if (spec.producer == Producer::Raw) compare_raw_passthrough(output_fd.fd, offset, *spec.source);
    }
    if (gguf_get_n_tensors(ctx) != static_cast<int64_t>(plan.size())) fail("unexpected tensor count in output");
    gguf_free(ctx);
    if (meta) ggml_free(meta);
    std::cerr << "[verify] parsed GGUF, checked qtypes/bounds, exact sidecars, and raw pass-through bytes\n";
}

void write_gguf(const Options & options, const SafeTensorSet & source,
                const std::vector<LayerCalibration> & calibration,
                uint32_t layers, uint32_t experts, const std::optional<Imatrix> & imatrix) {
    const fs::path gumix_path = options.output.string() + ".gumix.bin";
    if (!options.force && (fs::exists(options.output) || fs::exists(gumix_path))) {
        fail("output or qtype-106 sidecar exists: " + options.output.string());
    }
    if (!options.output.parent_path().empty()) fs::create_directories(options.output.parent_path());
    const fs::path temporary = options.output.string() + ".partial";
    if (fs::exists(temporary)) fs::remove(temporary);

    const std::vector<uint8_t> p4 = make_p4_blob(calibration, experts);
    const std::vector<uint8_t> gumix = make_gumix_blob(calibration, experts);
    const std::vector<TensorSpec> plan = make_plan(source, calibration, experts, options.experts_only);

    gguf_context * ctx = gguf_init_empty();
    if (!ctx) fail("gguf_init_empty failed");
    set_model_metadata(ctx, source, layers, experts, options.absmax_only,
                       options.experts_only, p4);
    std::vector<const char *> repair_stamps;
    for (const auto & layer : calibration)
        for (const auto & stamp : layer.repairs) repair_stamps.push_back(stamp.c_str());
    gguf_set_val_str(ctx, "deepseek4.mix.codebook_repair", "bf16-epsilon-v1");
    gguf_set_arr_str(ctx, "deepseek4.mix.codebook_repairs", repair_stamps.data(), repair_stamps.size());
    std::vector<std::unique_ptr<ggml_tensor>> descriptors;
    descriptors.reserve(plan.size());
    for (const TensorSpec & spec : plan) {
        descriptors.push_back(make_tensor_descriptor(spec));
        gguf_add_tensor(ctx, descriptors.back().get());
    }
    if (!gguf_write_to_file(ctx, temporary.c_str(), true)) fail("failed writing GGUF metadata");
    FILE * out = std::fopen(temporary.c_str(), "ab");
    if (!out) fail("cannot append tensor data to " + temporary.string());
    const off_t initial_position = ::ftello(out);
    if (initial_position < 0) fail("cannot determine GGUF data offset");
    const uint64_t data_offset = static_cast<uint64_t>(initial_position);
    for (const TensorSpec & spec : plan) {
        const int64_t id = gguf_find_tensor(ctx, spec.name.c_str());
        if (id < 0) fail("internal GGUF tensor lookup failed for " + spec.name);
        const uint64_t expected = data_offset + gguf_get_tensor_offset(ctx, id);
        const off_t position = ::ftello(out);
        if (position < 0 || static_cast<uint64_t>(position) > expected) {
            fail("GGUF stream offset overran " + spec.name + " (position=" +
                 std::to_string(position) + " expected=" + std::to_string(expected) + ")");
        }
        std::array<uint8_t, kAlignment> zeros{};
        uint64_t gap = expected - static_cast<uint64_t>(position);
        while (gap) {
            const size_t n = static_cast<size_t>(std::min<uint64_t>(gap, zeros.size()));
            fwrite_exact(out, zeros.data(), n, "GGUF alignment");
            gap -= n;
        }
        const off_t before = ::ftello(out);
        if (spec.producer == Producer::Raw) {
            copy_raw(out, *spec.source);
        } else if (spec.producer == Producer::DenseFp8) {
            write_dense_fp8(out, *spec.source, *spec.scale);
        } else if (spec.producer == Producer::Int64ToInt32) {
            write_int64_to_int32(out, *spec.source);
        } else {
            write_expert_tensor(out, source, find_calibration(calibration, spec.layer),
                                experts, *spec.recipe, imatrix);
        }
        const off_t after = ::ftello(out);
        if (before < 0 || after < before || static_cast<uint64_t>(after - before) != gguf_get_tensor_size(ctx, id)) {
            fail("producer wrote wrong byte count for " + spec.name);
        }
    }
    const bool flush_failed = std::fflush(out) != 0;
    const bool sync_failed = !flush_failed && ::fsync(::fileno(out)) != 0;
    const bool close_failed = std::fclose(out) != 0;
    if (flush_failed || sync_failed || close_failed) {
        fail("failed finalizing " + temporary.string());
    }
    gguf_free(ctx);

    write_atomic_bytes(gumix_path, gumix, options.force);
    if (options.force && fs::exists(options.output)) fs::remove(options.output);
    fs::rename(temporary, options.output);
    verify_artifact(options.output, gumix_path, plan, p4, gumix);
}

#include "iq85_converter.inc"

} // namespace

int main(int argc, char ** argv) {
    try {
        static_assert(static_cast<int>(GGML_TYPE_Q3_1_ROCMFP3_MIX) == static_cast<int>(kQtypeP4Mix));
        static_assert(static_cast<int>(GGML_TYPE_Q2_1_ROCMFP2_MIX) == static_cast<int>(kQtypeGuMix));
        const Options options = parse_options(argc, argv);
        if (options.layer_start != 0) fail("loader-compatible partial artifacts must start at layer 0");
        SafeTensorSet source(options.input);
        const uint32_t source_layers = config_u32(source.config(), "num_hidden_layers");
        const uint32_t source_experts = config_u32(source.config(), "n_routed_experts");
        const uint32_t layers = options.layer_count < 0 ? source_layers : static_cast<uint32_t>(options.layer_count);
        const uint32_t experts = options.expert_limit < 0 ? source_experts : static_cast<uint32_t>(options.expert_limit);
        if (layers == 0 || layers > source_layers || experts == 0 || experts > source_experts) {
            fail("selected layer/expert range exceeds config.json");
        }
        if (!options.experts_only && (layers != source_layers || experts != source_experts)) {
            fail("layer/expert limits are permitted only with --experts-only smoke artifacts");
        }
        if (options.recipe == "iq85") {
            run_iq85(options, source, layers, experts);
            return 0;
        }
        if (options.validate_input_only) {
            const auto layout = validate_input_layout(source, layers, experts);
            const auto plan = make_plan(source, layout, experts, options.experts_only);
            std::cerr << "[validate-input] PASS: " << plan.size()
                      << " loader and pass-through tensors; all expert sources/dtypes/shapes validated\n";
            return 0;
        }
        std::optional<Imatrix> imatrix;
        if (options.imatrix) {
            imatrix = load_imatrix(*options.imatrix);
        } else {
            std::cerr << "[calibration] WARNING: ABSMAX-ONLY LOWER QUALITY mode explicitly selected; no imatrix weighting\n";
        }
        std::cerr << "[plan] CPU-only conversion layers=0.." << (layers - 1)
                  << " experts=0.." << (experts - 1)
                  << (options.experts_only ? " experts-only smoke artifact" : " complete text+vision artifact") << "\n";
        const auto calibration = calibrate(source, layers, experts, imatrix);
        write_gguf(options, source, calibration, layers, experts, imatrix);
        std::cerr << "[done] " << options.output << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "ds4_mix_converter: ERROR: " << e.what() << "\n";
        return 1;
    }
}
