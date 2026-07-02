#include "gguf_tensor_data.h"

#include "common/gguf_mmap.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>

namespace dflash::common {

namespace {

uint16_t get_u16_or(const gguf_context * g, const char * key, uint16_t fallback) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    return gguf_get_val_u16(g, id);
}

bool parse_positive_int(const std::string & text, int & out) {
    if (text.empty()) return false;
    errno = 0;
    char * end = nullptr;
    const long v = std::strtol(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' ||
        v <= 0 || v > std::numeric_limits<int>::max()) {
        return false;
    }
    out = (int)v;
    return true;
}

bool split_prefix_from_path(const std::string & path,
                            int split_no,
                            int split_count,
                            std::string & prefix,
                            std::string & err) {
    static const std::string suffix = ".gguf";
    if (path.size() <= suffix.size() ||
        path.compare(path.size() - suffix.size(), suffix.size(), suffix) != 0) {
        err = "split GGUF path does not end in .gguf: " + path;
        return false;
    }

    const size_t suffix_pos = path.size() - suffix.size();
    const size_t of_pos = path.rfind("-of-", suffix_pos);
    if (of_pos == std::string::npos) {
        err = "split GGUF path is missing -of- marker: " + path;
        return false;
    }
    const size_t dash_pos = path.rfind('-', of_pos == 0 ? 0 : of_pos - 1);
    if (dash_pos == std::string::npos || dash_pos + 1 >= of_pos) {
        err = "split GGUF path is missing split number: " + path;
        return false;
    }

    int file_no = 0;
    int file_count = 0;
    if (!parse_positive_int(path.substr(dash_pos + 1, of_pos - dash_pos - 1), file_no) ||
        !parse_positive_int(path.substr(of_pos + 4, suffix_pos - (of_pos + 4)), file_count)) {
        err = "split GGUF path has invalid split numbers: " + path;
        return false;
    }
    if (file_no != split_no + 1 || file_count != split_count) {
        std::ostringstream ss;
        ss << "split GGUF path metadata mismatch: path says "
           << file_no << "-of-" << file_count
           << " but metadata says " << (split_no + 1)
           << "-of-" << split_count;
        err = ss.str();
        return false;
    }

    prefix = path.substr(0, dash_pos);
    return true;
}

std::string split_path_from_prefix(const std::string & prefix,
                                   int split_no,
                                   int split_count) {
    char tail[64];
    std::snprintf(tail, sizeof(tail), "-%05d-of-%05d.gguf",
                  split_no + 1, split_count);
    return prefix + tail;
}

} // namespace

struct GgufTensorDataReader::Impl {
    struct Shard {
        std::string path;
        gguf_context * gctx = nullptr;
        ggml_context * tensor_ctx = nullptr;
        GgufMmap mmap;
    };

    std::vector<std::unique_ptr<Shard>> shards;
    ggml_context * merged_ctx = nullptr;

    ~Impl() {
        if (merged_ctx) {
            ggml_free(merged_ctx);
            merged_ctx = nullptr;
        }
        for (auto & shard : shards) {
            if (!shard) continue;
            if (shard->tensor_ctx) {
                ggml_free(shard->tensor_ctx);
                shard->tensor_ctx = nullptr;
            }
            if (shard->gctx) {
                gguf_free(shard->gctx);
                shard->gctx = nullptr;
            }
        }
    }

    bool open_one(const std::string & path,
                  bool build_tensor_context,
                  int expected_no,
                  int expected_count,
                  std::string & err) {
        auto shard = std::make_unique<Shard>();
        shard->path = path;

        gguf_init_params gip{};
        gip.no_alloc = true;
        if (build_tensor_context) {
            gip.ctx = &shard->tensor_ctx;
        }
        shard->gctx = gguf_init_from_file(path.c_str(), gip);
        if (!shard->gctx) {
            err = "gguf_init_from_file failed: " + path;
            return false;
        }

        if (expected_count > 1) {
            const uint16_t split_no =
                get_u16_or(shard->gctx, "split.no", UINT16_MAX);
            const uint16_t split_count =
                get_u16_or(shard->gctx, "split.count", 0);
            if (split_no != (uint16_t)expected_no ||
                split_count != (uint16_t)expected_count) {
                std::ostringstream ss;
                ss << "split metadata mismatch in " << path
                   << ": split.no=" << split_no
                   << " split.count=" << split_count
                   << " expected " << expected_no
                   << "/" << expected_count;
                err = ss.str();
                return false;
            }
        }

        shards.push_back(std::move(shard));
        return true;
    }

    bool build_merged_context(std::string & err) {
        size_t n_tensors = 0;
        for (const auto & shard : shards) {
            n_tensors += (size_t)gguf_get_n_tensors(shard->gctx);
        }
        if (n_tensors == 0) {
            err = "GGUF shard set contains no tensors";
            return false;
        }
        if (n_tensors > std::numeric_limits<size_t>::max() / ggml_tensor_overhead()) {
            err = "GGUF tensor metadata context size overflow";
            return false;
        }

        ggml_init_params ip{};
        ip.mem_size = n_tensors * ggml_tensor_overhead();
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        merged_ctx = ggml_init(ip);
        if (!merged_ctx) {
            err = "failed to allocate merged GGUF tensor context";
            return false;
        }

        for (const auto & shard : shards) {
            for (int64_t tid = 0; tid < gguf_get_n_tensors(shard->gctx); ++tid) {
                const char * name = gguf_get_tensor_name(shard->gctx, tid);
                ggml_tensor * src = shard->tensor_ctx
                    ? ggml_get_tensor(shard->tensor_ctx, name)
                    : nullptr;
                if (!src) {
                    err = std::string("missing shard tensor metadata: ") + name;
                    return false;
                }
                if (ggml_get_tensor(merged_ctx, name)) {
                    err = std::string("duplicate tensor across GGUF shards: ") + name;
                    return false;
                }
                ggml_tensor * dst =
                    ggml_new_tensor(merged_ctx, src->type, GGML_MAX_DIMS, src->ne);
                if (!dst) {
                    err = std::string("failed to create merged tensor metadata: ") + name;
                    return false;
                }
                ggml_set_name(dst, name);
            }
        }
        return true;
    }
};

GgufTensorDataReader::GgufTensorDataReader()
    : impl_(std::make_unique<Impl>()) {}

GgufTensorDataReader::~GgufTensorDataReader() = default;

bool GgufTensorDataReader::open(const std::string & path,
                                bool build_merged_tensor_context,
                                std::string & err) {
    impl_ = std::make_unique<Impl>();

    gguf_context * probe = gguf_init_from_file(path.c_str(), gguf_init_params{});
    if (!probe) {
        err = "gguf_init_from_file failed: " + path;
        return false;
    }

    const uint16_t split_count = get_u16_or(probe, "split.count", 0);
    const uint16_t split_no = get_u16_or(probe, "split.no", 0);
    gguf_free(probe);

    if (split_count <= 1) {
        if (!impl_->open_one(path, build_merged_tensor_context,
                             /*expected_no=*/0, /*expected_count=*/1, err)) {
            return false;
        }
    } else {
        std::string prefix;
        if (!split_prefix_from_path(path, split_no, split_count, prefix, err)) {
            return false;
        }
        for (int i = 0; i < (int)split_count; ++i) {
            const std::string shard_path =
                split_path_from_prefix(prefix, i, (int)split_count);
            if (!impl_->open_one(shard_path, build_merged_tensor_context,
                                 i, (int)split_count, err)) {
                return false;
            }
        }
    }

    if (build_merged_tensor_context &&
        !impl_->build_merged_context(err)) {
        return false;
    }
    return true;
}

bool GgufTensorDataReader::open_mmaps(std::string & err) {
    for (auto & shard : impl_->shards) {
        if (!shard->mmap.open(shard->path, err)) {
            return false;
        }
    }
    return true;
}

const gguf_context * GgufTensorDataReader::primary_context() const {
    return impl_->shards.empty() ? nullptr : impl_->shards.front()->gctx;
}

ggml_context * GgufTensorDataReader::merged_context() const {
    return impl_->merged_ctx;
}

ggml_context * GgufTensorDataReader::release_merged_context() {
    ggml_context * out = impl_->merged_ctx;
    impl_->merged_ctx = nullptr;
    return out;
}

int GgufTensorDataReader::shard_count() const {
    return (int)impl_->shards.size();
}

const gguf_context * GgufTensorDataReader::shard_context(int shard_index) const {
    if (shard_index < 0 || shard_index >= shard_count()) return nullptr;
    return impl_->shards[(size_t)shard_index]->gctx;
}

bool GgufTensorDataReader::find_tensor(const char * name, GgufTensorRef & out) const {
    out = {};
    if (!name || !*name) return false;
    for (size_t si = 0; si < impl_->shards.size(); ++si) {
        const auto * shard = impl_->shards[si].get();
        const int64_t tid = gguf_find_tensor(shard->gctx, name);
        if (tid < 0) continue;

        out.shard_index = (int)si;
        out.tensor_id = tid;
        out.tensor = impl_->merged_ctx ? ggml_get_tensor(impl_->merged_ctx, name)
                                       : nullptr;
        out.size = gguf_get_tensor_size(shard->gctx, tid);
        out.type = gguf_get_tensor_type(shard->gctx, tid);
        if (shard->mmap.is_open()) {
            const size_t off =
                gguf_get_data_offset(shard->gctx) +
                gguf_get_tensor_offset(shard->gctx, tid);
            if (off <= shard->mmap.size() &&
                out.size <= shard->mmap.size() - off) {
                out.data = static_cast<const uint8_t *>(shard->mmap.data()) + off;
            }
        }
        return true;
    }
    return false;
}

const std::string & GgufTensorDataReader::shard_path(int shard_index) const {
    return impl_->shards[(size_t)shard_index]->path;
}

const void * GgufTensorDataReader::shard_mmap_data(int shard_index) const {
    if (shard_index < 0 || shard_index >= shard_count()) return nullptr;
    return impl_->shards[(size_t)shard_index]->mmap.data();
}

size_t GgufTensorDataReader::shard_mmap_size(int shard_index) const {
    if (shard_index < 0 || shard_index >= shard_count()) return 0;
    return impl_->shards[(size_t)shard_index]->mmap.size();
}

bool GgufTensorDataReader::release_single_mmap(
        const void *& data,
        size_t & size,
        int & fd,
        std::string & err) {
    data = nullptr;
    size = 0;
    fd = -1;
    if (shard_count() != 1) {
        err = "release_single_mmap requires exactly one GGUF shard";
        return false;
    }
    GgufMmap::OwnedRegion region = impl_->shards.front()->mmap.release();
    data = region.data;
    size = region.size;
    fd = region.fd;
    return data != nullptr && size > 0;
}

std::vector<LayerExpertFileData> make_layer_expert_file_data(
        const GgufTensorDataReader & reader,
        int n_layer) {
    std::vector<LayerExpertFileData> layer_file_data((size_t)n_layer);
    for (int il = 0; il < n_layer; ++il) {
        char name[128];
        auto find_tensor_data = [&](const char * suffix) -> ExpertTensorFileData {
            std::snprintf(name, sizeof(name), "blk.%d.%s.weight", il, suffix);
            GgufTensorRef ref;
            if (!reader.find_tensor(name, ref) || !ref.data) return {};
            return { ref.data, ref.size };
        };
        layer_file_data[(size_t)il].gate_exps    = find_tensor_data("ffn_gate_exps");
        layer_file_data[(size_t)il].up_exps      = find_tensor_data("ffn_up_exps");
        layer_file_data[(size_t)il].down_exps    = find_tensor_data("ffn_down_exps");
        layer_file_data[(size_t)il].gate_up_exps = find_tensor_data("ffn_gate_up_exps");
    }
    return layer_file_data;
}

int64_t gguf_tensor_ref_row_width(const GgufTensorRef & ref) {
    if (!ref.tensor) return 0;
    return ref.tensor->ne[0];
}

int64_t gguf_tensor_ref_row_count(const GgufTensorRef & ref) {
    if (!ref.tensor) return 0;
    int64_t rows = 1;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        rows *= ref.tensor->ne[i];
    }
    return rows;
}

bool gguf_tensor_ref_rows_to_host_f32(
        const GgufTensorRef & ref,
        int64_t row_begin,
        int64_t row_count,
        std::vector<float> & out,
        std::string * err) {
    out.clear();
    if (err) err->clear();
    if (!ref.tensor || ref.tensor_id < 0) {
        if (err) *err = "GGUF tensor ref is not initialized";
        return false;
    }
    if (!ref.data) {
        if (err) *err = "GGUF tensor ref has no mapped data";
        return false;
    }

    const int64_t row_width = gguf_tensor_ref_row_width(ref);
    const int64_t total_rows = gguf_tensor_ref_row_count(ref);
    if (row_width <= 0 || total_rows <= 0) {
        if (err) *err = "GGUF tensor ref shape is invalid";
        return false;
    }
    if (row_begin < 0 || row_count < 0 || row_begin > total_rows ||
        row_count > total_rows - row_begin) {
        if (err) *err = "GGUF tensor ref row range is out of bounds";
        return false;
    }
    if (row_count == 0) return true;

    const ggml_type type = ref.type;
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    const bool native_f32 = type == GGML_TYPE_F32;
    const bool native_f16 = type == GGML_TYPE_F16;
    const bool native_bf16 = type == GGML_TYPE_BF16;
    if (!(native_f32 || native_f16 || native_bf16) &&
        (!traits || !traits->to_float)) {
        if (err) *err = "GGUF tensor ref type cannot be dequantized to f32";
        return false;
    }

    const int64_t block = ggml_blck_size(type);
    if (block <= 0 || row_width % block != 0) {
        if (err) *err = "GGUF tensor ref row width is incompatible with tensor block size";
        return false;
    }

    const size_t row_bytes = ggml_row_size(type, row_width);
    if (row_bytes == 0) {
        if (err) *err = "GGUF tensor ref row size is zero";
        return false;
    }
    if ((size_t) total_rows > std::numeric_limits<size_t>::max() / row_bytes) {
        if (err) *err = "GGUF tensor ref row byte size overflow";
        return false;
    }
    const size_t total_bytes = (size_t) total_rows * row_bytes;
    if (total_bytes > ref.size) {
        if (err) *err = "GGUF tensor ref byte extent is smaller than tensor rows";
        return false;
    }
    if ((size_t) row_count > std::numeric_limits<size_t>::max() / (size_t) row_width) {
        if (err) *err = "GGUF tensor ref output size overflow";
        return false;
    }

    out.resize((size_t) row_count * (size_t) row_width);
    for (int64_t row = 0; row < row_count; ++row) {
        const size_t src_off = ((size_t) row_begin + (size_t) row) * row_bytes;
        const uint8_t * src = ref.data + src_off;
        float * dst = out.data() + (size_t) row * (size_t) row_width;
        if (native_f32) {
            std::memcpy(dst, src, (size_t) row_width * sizeof(float));
        } else if (native_f16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, row_width);
        } else if (native_bf16) {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, dst, row_width);
        } else {
            traits->to_float(src, dst, row_width);
        }
    }
    return true;
}

bool gguf_tensor_ref_all_to_host_f32(
        const GgufTensorRef & ref,
        std::vector<float> & out,
        std::string * err) {
    return gguf_tensor_ref_rows_to_host_f32(
        ref, 0, gguf_tensor_ref_row_count(ref), out, err);
}

} // namespace dflash::common
