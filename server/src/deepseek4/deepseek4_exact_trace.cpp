#include "deepseek4_exact_trace.h"

#include "common/model_backend.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <utility>

namespace dflash::common {
namespace {

constexpr const char * kSchema = "lucebox.ds4.exact-diff/v1";
constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

void hash_update(uint64_t & hash, const void * data, size_t bytes) {
    const auto * input = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= input[i];
        hash *= kFnvPrime;
    }
}

std::string hash_hex(uint64_t hash) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << hash;
    return stream.str();
}

struct TensorDigest {
    std::string hash;
    size_t bytes = 0;
};

TensorDigest tensor_digest(const ggml_tensor * tensor, size_t bytes) {
    if (!tensor || bytes == 0) return {};
    bytes = std::min(bytes, ggml_nbytes(tensor));
    std::vector<uint8_t> data(bytes);
    ggml_backend_tensor_get(tensor, data.data(), 0, bytes);
    return {DeepSeek4ExactTraceWriter::hash_bytes(data.data(), data.size()), data.size()};
}

TensorDigest tensor_digest(const ggml_tensor * tensor) {
    return tensor_digest(tensor, tensor ? ggml_nbytes(tensor) : 0);
}

size_t tensor_prefix_bytes(const ggml_tensor * tensor, int rows) {
    if (!tensor || rows <= 0) return 0;
    return std::min(
        ggml_nbytes(tensor),
        ggml_row_size(tensor->type, tensor->ne[0]) * static_cast<size_t>(rows));
}

void write_nullable_string(std::ofstream & output, const std::string & value) {
    if (value.empty()) {
        output << "null";
    } else {
        output << '"' << value << '"';
    }
}

bool write_float_array(
        std::ofstream & output,
        const float * values,
        size_t count,
        size_t begin = 0,
        size_t end = std::numeric_limits<size_t>::max()) {
    begin = std::min(begin, count);
    end = std::min(end, count);
    bool non_finite = false;
    output << '[';
    for (size_t i = begin; i < end; ++i) {
        if (i != begin) output << ',';
        if (std::isfinite(values[i])) {
            output << std::setprecision(std::numeric_limits<float>::max_digits10)
                   << values[i];
        } else {
            output << "null";
            non_finite = true;
        }
    }
    output << ']';
    return non_finite;
}

void write_int_array(
        std::ofstream & output,
        const int32_t * values,
        size_t begin,
        size_t end) {
    output << '[';
    for (size_t i = begin; i < end; ++i) {
        if (i != begin) output << ',';
        output << values[i];
    }
    output << ']';
}

bool near_boundary(int begin, int end, int boundary) {
    return begin <= boundary + 4 && end >= boundary - 4;
}

void append_unique(std::vector<std::string> & values, std::string value) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(std::move(value));
    }
}

std::string boundary_relation(
        const std::string & prefix,
        int begin,
        int end,
        int boundary) {
    if (begin < boundary && end < boundary) return prefix + "_before";
    if (begin < boundary && end >= boundary) return prefix + "_on";
    return prefix + "_after";
}

}  // namespace

std::unique_ptr<DeepSeek4ExactTraceWriter>
DeepSeek4ExactTraceWriter::create_from_env(const DeepSeek4Weights & weights) {
    const char * path = std::getenv("DFLASH_DS4_EXACT_TRACE_PATH");
    if (!path || !*path) return nullptr;

    const char * raw_width = std::getenv("DFLASH_DS4_EXACT_TRACE_Q");
    char * end = nullptr;
    const long parsed = raw_width ? std::strtol(raw_width, &end, 10) : 0;
    if (!raw_width || !*raw_width || !end || *end || parsed < 1 || parsed > 4) {
        std::fprintf(stderr,
                     "[deepseek4-exact-trace] DFLASH_DS4_EXACT_TRACE_Q must be 1..4\n");
        return nullptr;
    }

    std::ofstream output(path, std::ios::out | std::ios::app);
    if (!output) {
        std::fprintf(stderr,
                     "[deepseek4-exact-trace] cannot append trace path %s\n", path);
        return nullptr;
    }
    std::fprintf(stderr,
                 "[deepseek4-exact-trace] enabled path=%s q=%ld\n", path, parsed);
    return std::unique_ptr<DeepSeek4ExactTraceWriter>(
        new DeepSeek4ExactTraceWriter(weights, std::move(output), static_cast<int>(parsed)));
}

DeepSeek4ExactTraceWriter::DeepSeek4ExactTraceWriter(
        const DeepSeek4Weights & weights,
        std::ofstream output,
        int width)
    : weights_(weights), output_(std::move(output)), width_(width) {
    std::set<int> unique;
    for (uint32_t ratio : weights_.compress_ratios) {
        if (ratio > 0) unique.insert(static_cast<int>(ratio));
    }
    compressor_boundaries_.assign(unique.begin(), unique.end());
    output_ << "{\"schema\":\"" << kSchema
            << "\",\"type\":\"model_config\",\"n_swa\":" << weights_.n_swa
            << ",\"compressor_boundaries\":[";
    for (size_t i = 0; i < compressor_boundaries_.size(); ++i) {
        if (i) output_ << ',';
        output_ << compressor_boundaries_[i];
    }
    output_ << "]}\n";
    output_.flush();
}

bool DeepSeek4ExactTraceWriter::begin_request(
        const GenerateRequest & request,
        bool restored,
        int cache_position) {
    ++request_index_;
    prompt_tokens_ = static_cast<int>(request.prompt.size());
    snapshot_position_ = request.snap_pos;
    output_ << "{\"schema\":\"" << kSchema
            << "\",\"type\":\"request_start\",\"request\":" << request_index_
            << ",\"width\":" << width_
            << ",\"restored\":" << (restored ? "true" : "false")
            << ",\"cache_position\":" << cache_position
            << ",\"prompt_tokens\":" << request.prompt.size()
            << ",\"prompt_token_hash\":\"" << hash_token_ids(request.prompt) << "\""
            << ",\"prompt_token_ids\":";
    write_int_array(output_, request.prompt.data(), 0, request.prompt.size());
    output_
            << ",\"n_gen\":" << request.n_gen
            << ",\"snap_slot\":" << request.snap_slot
            << ",\"snap_pos\":" << request.snap_pos
            << ",\"temperature\":" << request.sampler.temp
            << ",\"top_p\":" << request.sampler.top_p
            << ",\"top_k\":" << request.sampler.top_k
            << ",\"seed\":" << request.sampler.seed << "}\n";
    output_.flush();
    return static_cast<bool>(output_);
}

void DeepSeek4ExactTraceWriter::end_request(
        bool ok,
        int cache_position,
        const std::vector<int32_t> & tokens) {
    output_ << "{\"schema\":\"" << kSchema
            << "\",\"type\":\"tokens\",\"request\":" << request_index_
            << ",\"token_ids\":";
    write_int_array(output_, tokens.data(), 0, tokens.size());
    output_ << "}\n{\"schema\":\"" << kSchema
            << "\",\"type\":\"request_end\",\"request\":" << request_index_
            << ",\"ok\":" << (ok ? "true" : "false")
            << ",\"cache_position\":" << cache_position << "}\n";
    output_.flush();
}

bool DeepSeek4ExactTraceWriter::wants_step(int position_begin, int n_tokens) const {
    if (request_index_ < 0 || n_tokens <= 0) return false;
    const int position_end = position_begin + n_tokens;
    if (position_begin < 16 || position_end >= std::max(0, prompt_tokens_ - 8)) return true;
    if (snapshot_position_ >= 0 &&
        near_boundary(position_begin, position_end, snapshot_position_)) {
        return true;
    }
    if (weights_.n_swa > 0 && near_boundary(position_begin, position_end, weights_.n_swa)) {
        return true;
    }
    for (int ratio : compressor_boundaries_) {
        const int first = ratio;
        const int last = prompt_tokens_ / ratio * ratio;
        if (near_boundary(position_begin, position_end, first) ||
            (last > first && near_boundary(position_begin, position_end, last))) {
            return true;
        }
    }
    const int capture_begin = std::max(0, prompt_tokens_ - weights_.n_swa);
    return near_boundary(position_begin, position_end, capture_begin);
}

void DeepSeek4ExactTraceWriter::record_reset(int cache_position) {
    output_ << "{\"schema\":\"" << kSchema
            << "\",\"type\":\"reset\",\"request\":" << request_index_
            << ",\"cache_position\":" << cache_position << "}\n";
}

void DeepSeek4ExactTraceWriter::write_relations(int position_begin, int position_end) {
    std::vector<std::string> relations;
    const int step_width = position_end - position_begin;
    if (step_width == width_ && position_end < prompt_tokens_) {
        append_unique(relations, "ordinary");
    }
    if (position_end == prompt_tokens_ && step_width < width_) {
        append_unique(relations, "tail_width_" + std::to_string(step_width));
    }
    for (int ratio : compressor_boundaries_) {
        const int first = ratio;
        const int last = prompt_tokens_ / ratio * ratio;
        if (near_boundary(position_begin, position_end, first)) {
            append_unique(relations,
                          boundary_relation("compressor", position_begin, position_end, first));
            append_unique(
                relations,
                boundary_relation(
                    "compressor_" + std::to_string(ratio),
                    position_begin, position_end, first));
        }
        if (last > first && near_boundary(position_begin, position_end, last)) {
            append_unique(relations,
                          boundary_relation("compressor", position_begin, position_end, last));
            append_unique(
                relations,
                boundary_relation(
                    "compressor_" + std::to_string(ratio),
                    position_begin, position_end, last));
        }
    }
    if (weights_.n_swa > 0 && near_boundary(position_begin, position_end, weights_.n_swa)) {
        append_unique(relations,
                      boundary_relation("swa", position_begin, position_end, weights_.n_swa));
        append_unique(
            relations,
            boundary_relation(
                "swa_" + std::to_string(weights_.n_swa),
                position_begin, position_end, weights_.n_swa));
    }
    if (snapshot_position_ >= 0 &&
        near_boundary(position_begin, position_end, snapshot_position_)) {
        append_unique(relations, "snapshot_boundary");
    }
    const int capture_begin = std::max(0, prompt_tokens_ - weights_.n_swa);
    if (near_boundary(position_begin, position_end, capture_begin) ||
        position_end >= std::max(0, prompt_tokens_ - 8)) {
        append_unique(relations, "dspark_capture_window");
    }
    output_ << '[';
    for (size_t i = 0; i < relations.size(); ++i) {
        if (i) output_ << ',';
        output_ << '"' << relations[i] << '"';
    }
    output_ << ']';
}

void DeepSeek4ExactTraceWriter::record_step(
        int position_begin,
        int n_tokens,
        int cache_position,
        bool logits_present) {
    if (!wants_step(position_begin, n_tokens)) return;
    const int position_end = position_begin + n_tokens;
    output_ << "{\"schema\":\"" << kSchema
            << "\",\"type\":\"step\",\"request\":" << request_index_
            << ",\"position_begin\":" << position_begin
            << ",\"position_end\":" << position_end
            << ",\"position\":" << position_end
            << ",\"width\":" << n_tokens
            << ",\"cache_position\":" << cache_position
            << ",\"logits_present\":" << (logits_present ? "true" : "false")
            << ",\"relations\":";
    write_relations(position_begin, position_end);
    output_ << "}\n";
}

void DeepSeek4ExactTraceWriter::record_layer(
        ggml_backend_t,
        const DeepSeek4Cache & cache,
        const std::vector<float> & hc_state,
        int layer,
        int position_begin,
        int n_tokens,
        bool hash_routed,
        const std::vector<int32_t> & routing_ids,
        const std::vector<float> & routing_weights) {
    if (!wants_step(position_begin, n_tokens) || layer < 0 ||
        layer >= static_cast<int>(cache.layers.size()) || n_tokens <= 0) {
        return;
    }
    const DeepSeek4LayerCache & layer_cache = cache.layers[static_cast<size_t>(layer)];
    const int position_end = position_begin + n_tokens;
    const int ratio = static_cast<int>(weights_.compress_ratios[static_cast<size_t>(layer)]);
    const int active_raw_rows = std::min(position_end, weights_.n_swa);
    const int active_comp_rows = ratio > 0
        ? std::max(layer_cache.n_comp, position_end / ratio) : 0;
    const int active_index_rows = ratio == 4
        ? std::max(layer_cache.n_index_comp, position_end / ratio) : 0;
    const TensorDigest raw = tensor_digest(
        layer_cache.raw_kv, tensor_prefix_bytes(layer_cache.raw_kv, active_raw_rows));
    const TensorDigest compressed = tensor_digest(
        layer_cache.comp_kv, tensor_prefix_bytes(layer_cache.comp_kv, active_comp_rows));
    const TensorDigest index_compressed = tensor_digest(
        layer_cache.index_comp_kv,
        tensor_prefix_bytes(layer_cache.index_comp_kv, active_index_rows));
    const TensorDigest attn_state_kv = tensor_digest(layer_cache.attn_compressor.state_kv);
    const TensorDigest attn_state_score = tensor_digest(layer_cache.attn_compressor.state_score);
    const TensorDigest index_state_kv = tensor_digest(layer_cache.indexer_compressor.state_kv);
    const TensorDigest index_state_score = tensor_digest(layer_cache.indexer_compressor.state_score);

    const size_t hc_stride = static_cast<size_t>(weights_.n_hc) * weights_.n_embd;
    const size_t hc_begin = hc_stride * static_cast<size_t>(n_tokens - 1);
    const bool hc_available = hc_begin + hc_stride <= hc_state.size();
    const std::string hc_hash = hc_available
        ? hash_bytes(hc_state.data() + hc_begin, hc_stride * sizeof(float)) : std::string{};
    bool non_finite = false;
    if (hc_available) {
        for (size_t i = hc_begin; i < hc_begin + hc_stride; ++i) {
            non_finite = non_finite || !std::isfinite(hc_state[i]);
        }
    }

    const size_t route_width = routing_ids.size() >= static_cast<size_t>(n_tokens)
        ? routing_ids.size() / static_cast<size_t>(n_tokens) : 0;
    const size_t route_begin = route_width * static_cast<size_t>(n_tokens - 1);
    const bool route_available = route_width > 0 &&
        route_begin + route_width <= routing_ids.size() &&
        route_begin + route_width <= routing_weights.size();
    if (route_available) {
        for (size_t i = route_begin; i < route_begin + route_width; ++i) {
            non_finite = non_finite || !std::isfinite(routing_weights[i]);
        }
    }

    output_ << "{\"schema\":\"" << kSchema
            << "\",\"type\":\"layer\",\"request\":" << request_index_
            << ",\"layer\":" << layer
            << ",\"position_begin\":" << position_begin
            << ",\"position_end\":" << position_end
            << ",\"token_position\":" << (position_end - 1)
            << ",\"routing\":{\"mode\":\""
            << (hash_routed ? "hash" : "learned") << "\",\"ids\":";
    if (route_available) {
        write_int_array(output_, routing_ids.data(), route_begin, route_begin + route_width);
    } else {
        output_ << "[]";
    }
    output_ << ",\"weights\":";
    if (route_available) {
        write_float_array(
            output_, routing_weights.data(), routing_weights.size(),
            route_begin, route_begin + route_width);
    } else {
        output_ << "[]";
    }
    output_ << "},\"hc_token_hashes\":[";
    write_nullable_string(output_, hc_hash);
    output_ << "],\"raw_kv_hash\":";
    write_nullable_string(output_, raw.hash);
    output_ << ",\"raw_kv_bytes\":" << raw.bytes
            << ",\"compressed_kv_hash\":";
    write_nullable_string(output_, compressed.hash);
    output_ << ",\"compressed_kv_bytes\":" << compressed.bytes
            << ",\"compressor_state\":{\"kv\":";
    write_nullable_string(output_, attn_state_kv.hash);
    output_ << ",\"score\":";
    write_nullable_string(output_, attn_state_score.hash);
    output_ << ",\"n_comp\":" << active_comp_rows
            << "},\"indexer_state\":{\"compressed_kv\":";
    write_nullable_string(output_, index_compressed.hash);
    output_ << ",\"compressed_kv_bytes\":" << index_compressed.bytes
            << ",\"kv\":";
    write_nullable_string(output_, index_state_kv.hash);
    output_ << ",\"score\":";
    write_nullable_string(output_, index_state_score.hash);
    output_ << ",\"n_comp\":" << active_index_rows
            << "},\"non_finite\":" << (non_finite ? "true" : "false") << "}\n";
}

void DeepSeek4ExactTraceWriter::record_capture(
        int position_begin,
        int n_tokens,
        int capture_begin,
        const std::vector<int> & layer_ids,
        const std::vector<float> & rows) {
    const size_t row_width = static_cast<size_t>(weights_.n_embd) * layer_ids.size();
    if (rows.empty() || n_tokens <= 0 || capture_begin < 0 ||
        capture_begin >= n_tokens || row_width == 0 || rows.size() % row_width != 0) {
        return;
    }
    const size_t captured_rows = rows.size() / row_width;
    if (capture_begin + static_cast<int>(captured_rows) > n_tokens) return;
    for (size_t row_index = 0; row_index < captured_rows; ++row_index) {
        const float * row = rows.data() + row_index * row_width;
        bool non_finite = false;
        for (size_t i = 0; i < row_width; ++i) {
            non_finite = non_finite || !std::isfinite(row[i]);
        }
        const int position = position_begin + capture_begin + static_cast<int>(row_index);
        output_ << "{\"schema\":\"" << kSchema
                << "\",\"type\":\"capture\",\"request\":" << request_index_
                << ",\"position_begin\":" << position
                << ",\"position_end\":" << (position + 1)
                << ",\"layer_ids\":[";
        for (size_t i = 0; i < layer_ids.size(); ++i) {
            if (i) output_ << ',';
            output_ << layer_ids[i];
        }
        output_ << "],\"row_hash\":\"" << hash_bytes(row, row_width * sizeof(float))
                << "\",\"total_values\":" << row_width << ",\"rows\":";
        non_finite = write_float_array(output_, row, row_width) || non_finite;
        output_ << ",\"non_finite\":" << (non_finite ? "true" : "false") << "}\n";
    }
}

void DeepSeek4ExactTraceWriter::record_logits(
        int position,
        const std::vector<float> & logits) {
    bool non_finite = false;
    output_ << "{\"schema\":\"" << kSchema
            << "\",\"type\":\"logits\",\"request\":" << request_index_
            << ",\"position\":" << position
            << ",\"hash\":\"" << hash_bytes(logits.data(), logits.size() * sizeof(float))
            << "\",\"values\":";
    non_finite = write_float_array(output_, logits.data(), logits.size());
    output_ << ",\"non_finite\":" << (non_finite ? "true" : "false") << "}\n";
}

std::string DeepSeek4ExactTraceWriter::cache_state_hash(
        const DeepSeek4Cache & cache) const {
    uint64_t hash = kFnvOffset;
    hash_update(hash, &cache.cur_pos, sizeof(cache.cur_pos));
    for (size_t layer_index = 0; layer_index < cache.layers.size(); ++layer_index) {
        const DeepSeek4LayerCache & layer = cache.layers[layer_index];
        hash_update(hash, &layer.n_comp, sizeof(layer.n_comp));
        hash_update(hash, &layer.n_index_comp, sizeof(layer.n_index_comp));
        const int raw_rows = std::min(cache.cur_pos, weights_.n_swa);
        const std::pair<const ggml_tensor *, size_t> tensors[] = {
            {layer.raw_kv, tensor_prefix_bytes(layer.raw_kv, raw_rows)},
            {layer.comp_kv, tensor_prefix_bytes(layer.comp_kv, layer.n_comp)},
            {layer.index_comp_kv,
             tensor_prefix_bytes(layer.index_comp_kv, layer.n_index_comp)},
            {layer.attn_compressor.state_kv,
             layer.attn_compressor.state_kv
                 ? ggml_nbytes(layer.attn_compressor.state_kv) : 0},
            {layer.attn_compressor.state_score,
             layer.attn_compressor.state_score
                 ? ggml_nbytes(layer.attn_compressor.state_score) : 0},
            {layer.indexer_compressor.state_kv,
             layer.indexer_compressor.state_kv
                 ? ggml_nbytes(layer.indexer_compressor.state_kv) : 0},
            {layer.indexer_compressor.state_score,
             layer.indexer_compressor.state_score
                 ? ggml_nbytes(layer.indexer_compressor.state_score) : 0},
        };
        for (const auto & [tensor, bytes] : tensors) {
            if (!tensor) continue;
            std::vector<uint8_t> data(bytes);
            if (data.empty()) continue;
            ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
            hash_update(hash, data.data(), data.size());
        }
    }
    if (cache.hc_state) {
        std::vector<uint8_t> data(ggml_nbytes(cache.hc_state));
        ggml_backend_tensor_get(cache.hc_state, data.data(), 0, data.size());
        hash_update(hash, data.data(), data.size());
    }
    return hash_hex(hash);
}

void DeepSeek4ExactTraceWriter::record_snapshot(
        const char * kind,
        int slot,
        const DeepSeek4Cache & cache) {
    output_ << "{\"schema\":\"" << kSchema
            << "\",\"type\":\"" << kind << "\",\"request\":" << request_index_
            << ",\"slot\":" << slot
            << ",\"cache_position\":" << cache.cur_pos
            << ",\"state_hash\":\"" << cache_state_hash(cache) << "\"}\n";
}

std::string DeepSeek4ExactTraceWriter::hash_bytes(const void * data, size_t bytes) {
    uint64_t hash = kFnvOffset;
    if (data && bytes) hash_update(hash, data, bytes);
    return hash_hex(hash);
}

std::string DeepSeek4ExactTraceWriter::hash_token_ids(
        const std::vector<int32_t> & tokens) {
    uint64_t hash = kFnvOffset;
    for (int32_t token : tokens) {
        const uint32_t value = static_cast<uint32_t>(token);
        const uint8_t bytes[] = {
            static_cast<uint8_t>(value),
            static_cast<uint8_t>(value >> 8),
            static_cast<uint8_t>(value >> 16),
            static_cast<uint8_t>(value >> 24),
        };
        hash_update(hash, bytes, sizeof(bytes));
    }
    return hash_hex(hash);
}

}  // namespace dflash::common
