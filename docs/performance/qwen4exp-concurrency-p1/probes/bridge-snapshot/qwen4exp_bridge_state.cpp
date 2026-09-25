#include "qwen4exp_bridge_state.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>

namespace dflash::common::qwen4exp_bridge {
namespace {

constexpr char kMagic[] = "LBSNAP01";
constexpr uint32_t kSchemaVersion = 1;
constexpr uint32_t kMaxTensors = 4096;
constexpr uint32_t kMaxNameBytes = 256;

bool reject(std::string * error, const char * message) {
    if (error) *error = message;
    return false;
}

void put_u32(std::vector<uint8_t> & out, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
        out.push_back(static_cast<uint8_t>(value >> shift));
}

void put_u64(std::vector<uint8_t> & out, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.push_back(static_cast<uint8_t>(value >> shift));
}

bool take_u32(const std::vector<uint8_t> & in, size_t & at, uint32_t & value) {
    if (in.size() - at < 4) return false;
    value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
        value |= static_cast<uint32_t>(in[at++]) << shift;
    return true;
}

bool take_u64(const std::vector<uint8_t> & in, size_t & at, uint64_t & value) {
    if (in.size() - at < 8) return false;
    value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
        value |= static_cast<uint64_t>(in[at++]) << shift;
    return true;
}

bool take_string(const std::vector<uint8_t> & in, size_t & at,
                 uint32_t limit, std::string & value) {
    uint32_t size = 0;
    if (!take_u32(in, at, size) || size == 0 || size > limit ||
        in.size() - at < size) return false;
    value.assign(reinterpret_cast<const char *>(in.data() + at), size);
    at += size;
    return true;
}

size_t type_size(const std::string & type) {
    if (type == "f16") return 2;
    if (type == "f32" || type == "i32") return 4;
    return 0;
}

bool tensor_bytes(const Tensor & tensor, size_t & bytes) {
    const size_t element_size = type_size(tensor.type);
    if (element_size == 0) return false;
    if (tensor.name == "qwen4exp.ple_prev" && tensor.type == "i32" &&
        tensor.ne[0] == 0 && tensor.ne[1] == 1 && tensor.ne[2] == 1 &&
        tensor.ne[3] == 1) {
        bytes = 0;
        return tensor.data.empty();
    }
    size_t count = 1;
    for (uint32_t extent : tensor.ne) {
        if (extent == 0 || count > std::numeric_limits<size_t>::max() / extent)
            return false;
        count *= extent;
    }
    if (count > std::numeric_limits<size_t>::max() / element_size) return false;
    bytes = count * element_size;
    return true;
}

bool unique_nonnegative(const std::vector<int32_t> & ids) {
    std::set<int32_t> unique;
    for (int32_t id : ids)
        if (id < 0 || !unique.insert(id).second) return false;
    return true;
}

std::string state_name(const char * kind, int32_t layer) {
    return std::string("qwen4exp.") + kind + "." + std::to_string(layer);
}

Tensor make_meta(const Checkpoint & cp) {
    std::vector<int32_t> values = {
        static_cast<int32_t>(kSchemaVersion), cp.indexer_blocks,
        static_cast<int32_t>(cp.full_layer_ids.size()),
        static_cast<int32_t>(cp.linear_layer_ids.size()),
        static_cast<int32_t>(cp.indexer_layer_ids.size()),
        static_cast<int32_t>(cp.ple_layer_ids.size())};
    values.insert(values.end(), cp.full_layer_ids.begin(), cp.full_layer_ids.end());
    values.insert(values.end(), cp.linear_layer_ids.begin(), cp.linear_layer_ids.end());
    values.insert(values.end(), cp.indexer_layer_ids.begin(), cp.indexer_layer_ids.end());
    values.insert(values.end(), cp.ple_layer_ids.begin(), cp.ple_layer_ids.end());

    Tensor meta;
    meta.name = "qwen4exp.__meta";
    meta.type = "i32";
    meta.ne = {static_cast<uint32_t>(values.size()), 1, 1, 1};
    meta.data.reserve(values.size() * sizeof(int32_t));
    for (int32_t value : values) put_u32(meta.data, static_cast<uint32_t>(value));
    return meta;
}

bool read_meta(const Tensor & meta, Checkpoint & cp) {
    if (meta.type != "i32" || meta.ne[1] != 1 || meta.ne[2] != 1 ||
        meta.ne[3] != 1 || meta.data.size() != meta.ne[0] * sizeof(int32_t) ||
        meta.ne[0] < 6) return false;
    std::vector<int32_t> v;
    v.reserve(meta.ne[0]);
    for (size_t at = 0; at < meta.data.size();) {
        uint32_t word = 0;
        for (unsigned shift = 0; shift < 32; shift += 8)
            word |= static_cast<uint32_t>(meta.data[at++]) << shift;
        v.push_back(static_cast<int32_t>(word));
    }
    if (v[0] != static_cast<int32_t>(kSchemaVersion) || v[1] < -1 ||
        v[2] < 0 || v[3] < 0 || v[4] < 0 || v[5] < 0) return false;
    const size_t expected = 6 + static_cast<size_t>(v[2]) + v[3] + v[4] + v[5];
    if (expected != v.size() || expected > 4096) return false;
    cp.indexer_blocks = v[1];
    size_t at = 6;
    auto take_ids = [&](int32_t count, std::vector<int32_t> & ids) {
        ids.assign(v.begin() + at, v.begin() + at + count);
        at += static_cast<size_t>(count);
    };
    take_ids(v[2], cp.full_layer_ids);
    take_ids(v[3], cp.linear_layer_ids);
    take_ids(v[4], cp.indexer_layer_ids);
    take_ids(v[5], cp.ple_layer_ids);
    return true;
}

bool finite_payload(const Tensor & tensor) {
    const size_t width = type_size(tensor.type);
    if (width != 2 && width != 4) return false;
    for (size_t at = 0; at < tensor.data.size(); at += width) {
        if (width == 2) {
            uint16_t bits = 0;
            std::memcpy(&bits, tensor.data.data() + at, sizeof(bits));
            if ((bits & 0x7c00U) == 0x7c00U) return false;
        } else {
            uint32_t bits = 0;
            std::memcpy(&bits, tensor.data.data() + at, sizeof(bits));
            if ((bits & 0x7f800000U) == 0x7f800000U) return false;
        }
    }
    return true;
}

bool kv_name(const std::string & name) {
    return name.compare(0, 16, "qwen4exp.attn_k.") == 0 ||
           name.compare(0, 16, "qwen4exp.attn_v.") == 0;
}

} // namespace

bool validate(const Checkpoint & cp, std::string * error) {
    if (cp.cut == 0 || cp.cut > cp.prompt_len)
        return reject(error, "cut must be a nonzero prefix of prompt_len");
    if (cp.indexer_blocks < -1 || cp.tensors.size() > kMaxTensors)
        return reject(error, "invalid indexer count or tensor count");
    if (!unique_nonnegative(cp.full_layer_ids) ||
        !unique_nonnegative(cp.linear_layer_ids) ||
        !unique_nonnegative(cp.indexer_layer_ids) ||
        !unique_nonnegative(cp.ple_layer_ids))
        return reject(error, "layer IDs must be unique and nonnegative");

    std::set<int32_t> all_layers(cp.full_layer_ids.begin(), cp.full_layer_ids.end());
    for (int32_t id : cp.linear_layer_ids)
        if (!all_layers.insert(id).second)
            return reject(error, "full and linear layers overlap");
    for (int32_t id : cp.indexer_layer_ids)
        if (std::find(cp.full_layer_ids.begin(), cp.full_layer_ids.end(), id) ==
            cp.full_layer_ids.end())
            return reject(error, "indexer layer is not a model layer");
    for (int32_t id : cp.ple_layer_ids)
        if (all_layers.count(id) == 0)
            return reject(error, "PLE layer is not a model layer");
    if ((cp.indexer_blocks <= 0) != cp.indexer_layer_ids.empty())
        return reject(error, "indexer layer inventory disagrees with populated block count");

    std::unordered_map<std::string, const Tensor *> tensors;
    for (const Tensor & tensor : cp.tensors) {
        size_t expected_bytes = 0;
        if (tensor.name.empty() || tensor.name.size() > kMaxNameBytes ||
            !tensor_bytes(tensor, expected_bytes) || tensor.data.size() != expected_bytes)
            return reject(error, "invalid tensor name, type, shape, or payload size");
        if (!tensors.emplace(tensor.name, &tensor).second)
            return reject(error, "duplicate tensor name");
    }

    auto require = [&](const std::string & name, const char * type) {
        auto it = tensors.find(name);
        if (it == tensors.end() || it->second->type != type) return false;
        return true;
    };

    std::set<std::string> expected;
    for (int32_t id : cp.full_layer_ids) {
        const std::string k_name = state_name("attn_k", id);
        const std::string v_name = state_name("attn_v", id);
        for (const char * kind : {"attn_k", "attn_v"}) {
            const std::string name = state_name(kind, id);
            auto it = tensors.find(name);
            if (it == tensors.end() ||
                (it->second->type != "f16" && it->second->type != "f32") ||
                it->second->ne[1] != cp.cut || it->second->ne[2] != 2 ||
                it->second->ne[3] != 1)
                return reject(error, "missing or malformed full-attention K/V prefix");
            expected.insert(name);
        }
        if (tensors.at(k_name)->type != tensors.at(v_name)->type)
            return reject(error, "K/V cache types differ within a layer");
    }
    for (int32_t id : cp.indexer_layer_ids) {
        const std::string name = state_name("indexer_k", id);
        if (!require(name, "f32") ||
            tensors.at(name)->ne[1] != static_cast<uint32_t>(cp.indexer_blocks) ||
            tensors.at(name)->ne[2] != 1 || tensors.at(name)->ne[3] != 1)
            return reject(error, "missing or malformed indexer K prefix");
        expected.insert(name);
    }
    for (int32_t id : cp.linear_layer_ids) {
        const std::string ssm = state_name("ssm_state", id);
        const std::string conv = state_name("conv_state", id);
        if (!require(ssm, "f32") || tensors.at(ssm)->ne[0] != tensors.at(ssm)->ne[1] ||
            tensors.at(ssm)->ne[3] != 1 || !require(conv, "f32") ||
            tensors.at(conv)->ne[2] != 1 || tensors.at(conv)->ne[3] != 1)
            return reject(error, "missing or malformed GDN state");
        expected.insert(ssm);
        expected.insert(conv);
    }
    for (int32_t id : cp.ple_layer_ids) {
        const std::string name = state_name("ple_conv_state", id);
        if (!require(name, "f32") || tensors.at(name)->ne[2] != 1 ||
            tensors.at(name)->ne[3] != 1)
            return reject(error, "missing or malformed PLE convolution state");
        expected.insert(name);
    }
    const auto prev = tensors.find("qwen4exp.ple_prev");
    if (prev == tensors.end() || prev->second->type != "i32" ||
        prev->second->ne[1] != 1 || prev->second->ne[2] != 1 ||
        prev->second->ne[3] != 1 || prev->second->ne[0] > cp.cut)
        return reject(error, "missing or malformed PLE token history");
    for (size_t i = 0; i < prev->second->ne[0]; ++i) {
        const size_t at = i * 4;
        const uint32_t value = static_cast<uint32_t>(prev->second->data[at]) |
            (static_cast<uint32_t>(prev->second->data[at + 1]) << 8) |
            (static_cast<uint32_t>(prev->second->data[at + 2]) << 16) |
            (static_cast<uint32_t>(prev->second->data[at + 3]) << 24);
        if (static_cast<int32_t>(value) < 0)
            return reject(error, "PLE token history contains a negative token ID");
    }
    expected.insert(prev->first);
    if (expected.size() != tensors.size())
        return reject(error, "checkpoint has unrecognized or non-semantic tensors");
    if (error) error->clear();
    return true;
}

std::vector<uint8_t> encode(const Checkpoint & cp, std::string * error) {
    if (!validate(cp, error)) return {};
    std::vector<uint8_t> out(kMagic, kMagic + 8);
    put_u32(out, cp.cut);
    put_u32(out, cp.prompt_len);
    put_u32(out, static_cast<uint32_t>(cp.tensors.size() + 1));
    auto write_tensor = [&](const Tensor & tensor) {
        put_u32(out, static_cast<uint32_t>(tensor.name.size()));
        out.insert(out.end(), tensor.name.begin(), tensor.name.end());
        put_u32(out, static_cast<uint32_t>(tensor.type.size()));
        out.insert(out.end(), tensor.type.begin(), tensor.type.end());
        put_u32(out, 4);
        for (uint32_t extent : tensor.ne) put_u32(out, extent);
        put_u64(out, tensor.data.size());
        out.insert(out.end(), tensor.data.begin(), tensor.data.end());
    };
    write_tensor(make_meta(cp));
    for (const Tensor & tensor : cp.tensors) write_tensor(tensor);
    if (error) error->clear();
    return out;
}

bool decode(const std::vector<uint8_t> & in, Checkpoint & checkpoint,
            std::string * error) {
    if (in.size() < 20 || !std::equal(kMagic, kMagic + 8, in.begin()))
        return reject(error, "not an LBSNAP01 checkpoint");
    size_t at = 8;
    Checkpoint decoded;
    uint32_t count = 0;
    if (!take_u32(in, at, decoded.cut) || !take_u32(in, at, decoded.prompt_len) ||
        !take_u32(in, at, count) || count == 0 || count > kMaxTensors + 1)
        return reject(error, "invalid checkpoint header");

    bool saw_meta = false;
    for (uint32_t i = 0; i < count; ++i) {
        Tensor tensor;
        if (!take_string(in, at, kMaxNameBytes, tensor.name) ||
            !take_string(in, at, 8, tensor.type))
            return reject(error, "truncated or oversized tensor header");
        uint32_t ndim = 0;
        if (!take_u32(in, at, ndim) || ndim != 4)
            return reject(error, "unsupported tensor rank");
        for (uint32_t & extent : tensor.ne)
            if (!take_u32(in, at, extent))
                return reject(error, "truncated tensor shape");
        uint64_t bytes = 0;
        if (!take_u64(in, at, bytes) || bytes > in.size() - at)
            return reject(error, "truncated tensor payload");
        tensor.data.assign(in.begin() + at, in.begin() + at + static_cast<size_t>(bytes));
        at += static_cast<size_t>(bytes);
        if (tensor.name == "qwen4exp.__meta") {
            if (saw_meta || !read_meta(tensor, decoded))
                return reject(error, "missing, duplicate, or invalid schema metadata");
            saw_meta = true;
        } else {
            decoded.tensors.push_back(std::move(tensor));
        }
    }
    if (at != in.size() || !saw_meta)
        return reject(error, "trailing data or missing schema metadata");
    if (!validate(decoded, error)) return false;
    checkpoint = std::move(decoded);
    return true;
}

bool compose_kv(const Checkpoint & native, const Checkpoint & donor,
                Checkpoint & output, size_t * replacements,
                std::string * error) {
    if (!validate(native, error) || !validate(donor, error)) return false;
    static const std::vector<int32_t> kTargetFullLayers = {
        3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47};
    if (native.cut != 896 || native.prompt_len != 1024 ||
        native.full_layer_ids != kTargetFullLayers)
        return reject(error, "native checkpoint is not the pinned C896/R128 layout");
    if (donor.cut != native.cut || donor.prompt_len != native.prompt_len ||
        donor.full_layer_ids != native.full_layer_ids)
        return reject(error, "donor K/V checkpoint metadata differs from native");

    std::map<std::string, const Tensor *> donor_kv;
    for (const Tensor & tensor : donor.tensors)
        if (kv_name(tensor.name)) donor_kv.emplace(tensor.name, &tensor);
    if (donor_kv.size() != 24)
        return reject(error, "donor must contain exactly 24 target K/V tensors");

    Checkpoint composed = native;
    size_t replaced = 0;
    for (Tensor & target : composed.tensors) {
        if (!kv_name(target.name)) continue;
        const auto found = donor_kv.find(target.name);
        if (found == donor_kv.end())
            return reject(error, "donor is missing a named target K/V tensor");
        const Tensor & source = *found->second;
        if (source.type != target.type || source.ne != target.ne ||
            source.data.size() != target.data.size())
            return reject(error, "donor K/V metadata or byte count differs from native");
        if (!finite_payload(source))
            return reject(error, "donor K/V contains a nonfinite value");
        target.data = source.data;
        donor_kv.erase(found);
        ++replaced;
    }
    if (replaced != 24 || !donor_kv.empty())
        return reject(error, "donor contains missing or extra target K/V tensors");
    if (!validate(composed, error)) return false;

    // Fail closed: the clone's metadata and every non-K/V tensor stay native.
    if (composed.cut != native.cut || composed.prompt_len != native.prompt_len ||
        composed.indexer_blocks != native.indexer_blocks ||
        composed.full_layer_ids != native.full_layer_ids ||
        composed.linear_layer_ids != native.linear_layer_ids ||
        composed.indexer_layer_ids != native.indexer_layer_ids ||
        composed.ple_layer_ids != native.ple_layer_ids ||
        composed.tensors.size() != native.tensors.size())
        return reject(error, "composition changed native checkpoint metadata");
    for (size_t i = 0; i < native.tensors.size(); ++i) {
        const Tensor & before = native.tensors[i];
        const Tensor & after = composed.tensors[i];
        if (before.name != after.name || before.type != after.type ||
            before.ne != after.ne ||
            (!kv_name(before.name) && before.data != after.data))
            return reject(error, "composition changed native non-K/V content");
    }
    output = std::move(composed);
    if (replacements) *replacements = replaced;
    if (error) error->clear();
    return true;
}

} // namespace dflash::common::qwen4exp_bridge
