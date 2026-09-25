#include "qwen4exp_bridge_cache.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>

namespace dflash::common::qwen4exp_bridge {
namespace {

using LiveTensors = std::map<std::string, ggml_tensor *>;

bool kv_name(const std::string & name);
bool indexer_name(const std::string & name);

template <size_t N>
bool starts_with(const std::string & value, const char (&prefix)[N]) {
    return value.compare(0, N - 1, prefix) == 0;
}

bool reject(std::string * error, const char * message) {
    if (error) *error = message;
    return false;
}

std::string state_name(const char * kind, int32_t layer) {
    return std::string("qwen4exp.") + kind + "." + std::to_string(layer);
}

bool equal_ids(const std::vector<int32_t> & a, const std::vector<int> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return false;
    return true;
}

bool live_layout(const Qwen4ExpWeights & w, const Qwen4ExpCache & c,
                 int32_t indexer_blocks, LiveTensors & tensors,
                 std::vector<int> & full_ids, std::vector<int> & linear_ids,
                 std::vector<int> & indexer_ids, std::vector<int> & ple_ids,
                 std::string * error) {
    if (w.n_layer <= 0 || static_cast<size_t>(w.n_layer) != w.layers.size() ||
        c.max_ctx <= 0 || w.n_vocab <= 0 || w.ple_ngram_size < 1 ||
        c.input_ring.enabled || c.input_ring.buf)
        return reject(error, "incomplete Qwen4Exp target metadata");
    for (int il = 0; il < w.n_layer; ++il) {
        if (w.layers[il].is_full_attention) full_ids.push_back(il);
        else linear_ids.push_back(il);
        if (w.ple_conv_kernel > 1 && w.layers[il].is_ple) ple_ids.push_back(il);
    }
    if (!equal_ids(c.full_layer_ids, full_ids) ||
        !equal_ids(c.linear_layer_ids, linear_ids) ||
        !equal_ids(c.ple_layer_ids, ple_ids) ||
        !equal_ids(w.ple_layer_ids, ple_ids))
        return reject(error, "cache layer inventories differ from loaded model order");
    if ((!ple_ids.empty() && !w.ple_reader.available()) ||
        c.attn_k.size() != full_ids.size() || c.attn_v.size() != full_ids.size() ||
        c.indexer_k.size() != full_ids.size() ||
        c.ssm_state.size() != linear_ids.size() ||
        c.conv_state.size() != linear_ids.size() ||
        c.ple_conv_state.size() != ple_ids.size())
        return reject(error, "cache tensor inventory is incomplete");

    for (size_t i = 0; i < full_ids.size(); ++i) {
        if (!c.attn_k[i] || !c.attn_v[i])
            return reject(error, "full-attention K/V tensor is missing");
        tensors.emplace(state_name("attn_k", full_ids[i]), c.attn_k[i]);
        tensors.emplace(state_name("attn_v", full_ids[i]), c.attn_v[i]);
        if (indexer_blocks > 0) {
            if (!c.indexer_k[i])
                return reject(error, "populated indexer state has no target tensor");
            indexer_ids.push_back(full_ids[i]);
            tensors.emplace(state_name("indexer_k", full_ids[i]), c.indexer_k[i]);
        }
    }
    for (size_t i = 0; i < linear_ids.size(); ++i) {
        if (!c.ssm_state[i] || !c.conv_state[i])
            return reject(error, "GDN state tensor is missing");
        tensors.emplace(state_name("ssm_state", linear_ids[i]), c.ssm_state[i]);
        tensors.emplace(state_name("conv_state", linear_ids[i]), c.conv_state[i]);
    }
    for (size_t i = 0; i < ple_ids.size(); ++i) {
        if (!c.ple_conv_state[i])
            return reject(error, "PLE convolution state tensor is missing");
        tensors.emplace(state_name("ple_conv_state", ple_ids[i]), c.ple_conv_state[i]);
    }
    return true;
}

bool kv_name(const std::string & name) {
    return starts_with(name, "qwen4exp.attn_k.") ||
           starts_with(name, "qwen4exp.attn_v.");
}

bool indexer_name(const std::string & name) {
    return starts_with(name, "qwen4exp.indexer_k.");
}

bool validate_live_tensors(const Qwen4ExpWeights & w, const Qwen4ExpCache & c,
                           const LiveTensors & tensors, uint32_t cut,
                           int32_t indexer_blocks, bool check_history,
                           std::string * error) {
    const int64_t conv_channels =
        2LL * w.ssm_n_group * w.ssm_d_state + w.ssm_d_inner;
    const int64_t ple_hist =
        static_cast<int64_t>(w.ple_conv_kernel - 1) * w.ple_ngram_size;
    const int64_t hc_dim = static_cast<int64_t>(w.n_embd) * w.n_hc;
    for (const auto & entry : tensors) {
        const ggml_tensor * t = entry.second;
        if (!t || !ggml_is_contiguous(t) || !ggml_type_name(t->type))
            return reject(error, "target state tensor is null or non-contiguous");
        if (kv_name(entry.first)) {
            const bool is_key = starts_with(entry.first, "qwen4exp.attn_k.");
            const int64_t dim = is_key ? w.n_embd_head_k : w.n_embd_head_v;
            if (t->type != c.kv_type || t->ne[0] != dim || t->ne[1] < cut ||
                t->ne[2] != w.n_head_kv || t->ne[3] != 1 ||
                t->nb[2] < static_cast<size_t>(t->ne[0]) * cut *
                    (ggml_type_size(t->type) / ggml_blck_size(t->type)))
                return reject(error, "target K/V cache layout does not match checkpoint");
        } else if (indexer_name(entry.first)) {
            if (t->type != GGML_TYPE_F32 || t->ne[0] != w.indexer_head_size ||
                t->ne[1] < indexer_blocks || t->ne[2] != 1 || t->ne[3] != 1)
                return reject(error, "target indexer layout does not match checkpoint");
        } else if (starts_with(entry.first, "qwen4exp.ssm_state.")) {
            if (t->type != GGML_TYPE_F32 || t->ne[0] != w.ssm_d_state ||
                t->ne[1] != w.ssm_d_state || t->ne[2] != w.linear_value_heads ||
                t->ne[3] != 1)
                return reject(error, "target SSM layout differs from model config");
        } else if (starts_with(entry.first, "qwen4exp.conv_state.")) {
            if (t->type != GGML_TYPE_F32 || t->ne[0] != w.ssm_d_conv - 1 ||
                t->ne[1] != conv_channels || t->ne[2] != 1 || t->ne[3] != 1)
                return reject(error, "target convolution layout differs from model config");
        } else {
            if (t->type != GGML_TYPE_F32 || t->ne[0] != ple_hist ||
                t->ne[1] != hc_dim || t->ne[2] != 1 || t->ne[3] != 1)
                return reject(error, "target PLE layout differs from model config");
        }
    }
    if (indexer_blocks < -1)
        return reject(error, "invalid live indexer count");
    if (check_history) {
        if (c.ple_prev.size() >
                static_cast<size_t>(std::max(0, w.ple_ngram_size - 1)))
            return reject(error, "live PLE history is too long for model n-gram size");
        for (int32_t token : c.ple_prev)
            if (token < 0 || token >= w.n_vocab)
                return reject(error, "live PLE history contains an invalid token");
        const size_t expected_history = c.ple_layer_ids.empty() ? 0 :
            std::min<size_t>(cut, static_cast<size_t>(w.ple_ngram_size - 1));
        if (c.ple_prev.size() != expected_history)
            return reject(error, "live PLE history length does not match explicit cut");
    }
    return true;
}

bool state_tensor(const ggml_tensor * t, Tensor & out) {
    if (!t || !ggml_is_contiguous(t)) return false;
    out.type = ggml_type_name(t->type);
    for (size_t i = 0; i < 4; ++i) out.ne[i] = static_cast<uint32_t>(t->ne[i]);
    const size_t element_size = ggml_type_size(t->type) / ggml_blck_size(t->type);
    size_t elements = 1;
    for (uint32_t ne : out.ne) {
        if (ne == 0 || elements > std::numeric_limits<size_t>::max() / ne)
            return false;
        elements *= ne;
    }
    if (elements > std::numeric_limits<size_t>::max() / element_size) return false;
    out.data.resize(elements * element_size);
    return true;
}

bool compact_prefix(ggml_tensor * source, uint32_t cut, Tensor & out) {
    if (!source || !ggml_is_contiguous(source) || cut > source->ne[1] ||
        source->ne[2] <= 0 || source->ne[3] != 1) return false;
    out.type = ggml_type_name(source->type);
    out.ne = {static_cast<uint32_t>(source->ne[0]), cut,
              static_cast<uint32_t>(source->ne[2]), 1};
    const size_t element_size = ggml_type_size(source->type) /
                                ggml_blck_size(source->type);
    const size_t head_bytes = static_cast<size_t>(source->ne[0]) * cut * element_size;
    if (source->nb[2] < head_bytes) return false;
    out.data.resize(head_bytes * static_cast<size_t>(source->ne[2]));
    for (size_t h = 0; h < static_cast<size_t>(source->ne[2]); ++h)
        ggml_backend_tensor_get(source, out.data.data() + h * head_bytes,
                                h * source->nb[2], head_bytes);
    return true;
}

bool capture_prefix(ggml_tensor * source, uint32_t columns,
                    const std::string & name, Tensor & out) {
    if (!source || !ggml_is_contiguous(source) || columns > source->ne[1] ||
        source->ne[2] != 1 || source->ne[3] != 1) return false;
    out.name = name;
    out.type = ggml_type_name(source->type);
    out.ne = {static_cast<uint32_t>(source->ne[0]), columns, 1, 1};
    const size_t element_size = ggml_type_size(source->type) /
                                ggml_blck_size(source->type);
    const size_t bytes = static_cast<size_t>(out.ne[0]) * columns * element_size;
    out.data.resize(bytes);
    if (bytes != 0) ggml_backend_tensor_get(source, out.data.data(), 0, bytes);
    return true;
}

std::vector<int32_t> read_i32(const Tensor & tensor) {
    std::vector<int32_t> values(tensor.ne[0]);
    for (size_t i = 0; i < values.size(); ++i) {
        const size_t at = i * 4;
        uint32_t v = static_cast<uint32_t>(tensor.data[at]) |
                     (static_cast<uint32_t>(tensor.data[at + 1]) << 8) |
                     (static_cast<uint32_t>(tensor.data[at + 2]) << 16) |
                     (static_cast<uint32_t>(tensor.data[at + 3]) << 24);
        values[i] = static_cast<int32_t>(v);
    }
    return values;
}

bool write_tensor_bytes(ggml_tensor * target, const Tensor & source) {
    if (!target || source.data.size() > ggml_nbytes(target)) return false;
    if (kv_name(source.name)) {
        const size_t element_size = ggml_type_size(target->type) /
                                    ggml_blck_size(target->type);
        const size_t head_bytes = static_cast<size_t>(target->ne[0]) *
                                  source.ne[1] * element_size;
        if (target->nb[2] < head_bytes) return false;
        for (size_t h = 0; h < static_cast<size_t>(target->ne[2]); ++h)
            ggml_backend_tensor_set(target,
                source.data.data() + h * head_bytes,
                h * target->nb[2], head_bytes);
    } else {
        if (!ggml_is_contiguous(target)) return false;
        ggml_backend_tensor_set(target, source.data.data(), 0, source.data.size());
    }
    return true;
}

void append_i32(Tensor & tensor, const std::vector<int32_t> & values) {
    tensor.name = "qwen4exp.ple_prev";
    tensor.type = "i32";
    tensor.ne = {static_cast<uint32_t>(values.size()), 1, 1, 1};
    tensor.data.reserve(values.size() * 4);
    for (int32_t value : values) {
        const uint32_t v = static_cast<uint32_t>(value);
        tensor.data.push_back(static_cast<uint8_t>(v));
        tensor.data.push_back(static_cast<uint8_t>(v >> 8));
        tensor.data.push_back(static_cast<uint8_t>(v >> 16));
        tensor.data.push_back(static_cast<uint8_t>(v >> 24));
    }
}

} // namespace

bool validate_for_target(const Checkpoint & cp, const Qwen4ExpWeights & w,
                         const Qwen4ExpCache & c, std::string * error) {
    if (!validate(cp, error)) return false;
    if (cp.cut > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        cp.cut > static_cast<uint32_t>(c.max_ctx))
        return reject(error, "checkpoint cut exceeds target cache capacity");
    if (c.indexer_blocks < -1)
        return reject(error, "target indexer count is below the -1 sentinel");

    LiveTensors live;
    std::vector<int> full, linear, indexer, ple;
    if (!live_layout(w, c, cp.indexer_blocks, live, full, linear, indexer, ple,
                     error)) return false;
    if (!validate_live_tensors(w, c, live, cp.cut, cp.indexer_blocks, false, error))
        return false;
    if (!equal_ids(cp.full_layer_ids, full) ||
        !equal_ids(cp.linear_layer_ids, linear) ||
        !equal_ids(cp.ple_layer_ids, ple) ||
        !equal_ids(cp.indexer_layer_ids, indexer))
        return reject(error, "checkpoint ordered layer inventory differs from target");

    const int expected_history = ple.empty() ? 0 :
        std::min<int>(static_cast<int>(cp.cut), w.ple_ngram_size - 1);
    const auto history = std::find_if(cp.tensors.begin(), cp.tensors.end(),
        [](const Tensor & tensor) { return tensor.name == "qwen4exp.ple_prev"; });
    if (history == cp.tensors.end() ||
        history->ne[0] != static_cast<uint32_t>(expected_history))
        return reject(error, "PLE history length does not match cut and n-gram size");
    for (int32_t token : read_i32(*history))
        if (token < 0 || token >= w.n_vocab)
            return reject(error, "PLE history contains an out-of-vocabulary token");

    if (cp.indexer_blocks > 0) {
        if (indexer.empty()) return reject(error, "checkpoint has no target indexer tensors");
        for (int id : indexer) {
            const auto it = live.find(state_name("indexer_k", id));
            if (it == live.end() || cp.indexer_blocks > it->second->ne[1])
                return reject(error, "checkpoint indexer prefix exceeds target capacity");
        }
    }

    std::map<std::string, const Tensor *> saved;
    for (const Tensor & tensor : cp.tensors) saved.emplace(tensor.name, &tensor);
    if (saved.size() != cp.tensors.size() || saved.size() != live.size() + 1)
        return reject(error, "checkpoint tensor inventory does not cover target state");
    for (const auto & entry : live) {
        const auto it = saved.find(entry.first);
        if (it == saved.end()) return reject(error, "checkpoint omits a target state tensor");
        const ggml_tensor * target = entry.second;
        const Tensor & source = *it->second;
        if (source.type != ggml_type_name(target->type))
            return reject(error, "checkpoint tensor type differs from target");
        std::array<uint32_t, 4> expected{};
        for (size_t i = 0; i < 4; ++i)
            expected[i] = static_cast<uint32_t>(target->ne[i]);
        if (kv_name(entry.first)) {
            expected[1] = cp.cut;
            expected[3] = 1;
        } else if (indexer_name(entry.first)) {
            expected[1] = static_cast<uint32_t>(cp.indexer_blocks);
            expected[2] = expected[3] = 1;
        }
        if (source.ne != expected)
            return reject(error, "checkpoint tensor shape differs from target");
    }
    if (error) error->clear();
    return true;
}

bool capture(ggml_backend_t backend, const Qwen4ExpWeights & w,
             const Qwen4ExpCache & c, uint32_t cut, uint32_t prompt_len,
             Checkpoint & checkpoint, std::string * error) {
    if (!backend || cut == 0 || cut > prompt_len ||
        cut > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        prompt_len > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        cut > static_cast<uint32_t>(c.max_ctx) || c.indexer_blocks < -1)
        return reject(error, "invalid explicit capture cut or cache position");
    Checkpoint captured;
    captured.cut = cut;
    captured.prompt_len = prompt_len;
    captured.indexer_blocks = c.indexer_blocks;
    LiveTensors live;
    std::vector<int> full, linear, indexer, ple;
    if (!live_layout(w, c, captured.indexer_blocks, live, full, linear,
                     indexer, ple, error)) return false;
    if (!validate_live_tensors(w, c, live, cut, captured.indexer_blocks, true, error))
        return false;
    captured.full_layer_ids.assign(full.begin(), full.end());
    captured.linear_layer_ids.assign(linear.begin(), linear.end());
    captured.indexer_layer_ids.assign(indexer.begin(), indexer.end());
    captured.ple_layer_ids.assign(ple.begin(), ple.end());

    for (const auto & entry : live) {
        Tensor tensor;
        tensor.name = entry.first;
        if (kv_name(entry.first)) {
            if (!compact_prefix(entry.second, cut, tensor))
                return reject(error, "failed to capture strided K/V prefix");
            tensor.name = entry.first;
        } else if (indexer_name(entry.first)) {
            if (!capture_prefix(entry.second,
                                static_cast<uint32_t>(captured.indexer_blocks),
                                entry.first, tensor))
                return reject(error, "invalid live indexer tensor");
        } else {
            if (!state_tensor(entry.second, tensor))
                return reject(error, "unsupported non-contiguous target state tensor");
            tensor.name = entry.first;
            ggml_backend_tensor_get(entry.second, tensor.data.data(), 0,
                                    tensor.data.size());
        }
        captured.tensors.push_back(std::move(tensor));
    }
    Tensor history;
    append_i32(history, c.ple_prev);
    captured.tensors.push_back(std::move(history));
    if (!validate_for_target(captured, w, c, error)) return false;
    checkpoint = std::move(captured);
    return true;
}

bool restore(ggml_backend_t backend, const Qwen4ExpWeights & w,
             Qwen4ExpCache & c, const Checkpoint & cp, std::string * error) {
    if (!backend || !validate_for_target(cp, w, c, error)) return false;
    LiveTensors live;
    std::vector<int> full, linear, indexer, ple;
    if (!live_layout(w, c, cp.indexer_blocks, live, full, linear,
                     indexer, ple, error)) return false;

    // Recreate transient graph/workspace state; the checkpoint owns only the
    // semantic prefix and host metadata. Input-ring allocation is disabled by
    // the probe before cache creation.
    clear_qwen4exp_decode_workspace(c.decode_workspace);
    for (const Tensor & tensor : cp.tensors) {
        if (tensor.name == "qwen4exp.ple_prev") continue;
        const auto target = live.find(tensor.name);
        if (target == live.end() || !write_tensor_bytes(target->second, tensor))
            return reject(error, "failed to restore validated target tensor");
    }
    const auto history = std::find_if(cp.tensors.begin(), cp.tensors.end(),
        [](const Tensor & tensor) { return tensor.name == "qwen4exp.ple_prev"; });
    c.ple_prev = read_i32(*history);
    c.indexer_blocks = cp.indexer_blocks;
    c.cur_pos = static_cast<int>(cp.cut);
    if (error) error->clear();
    return true;
}

} // namespace dflash::common::qwen4exp_bridge
