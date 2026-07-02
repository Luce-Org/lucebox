#pragma once

#include "ggml.h"
#include "gguf.h"
#include "moe_hybrid_storage.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dflash::common {

struct GgufTensorRef {
    int shard_index = -1;
    int64_t tensor_id = -1;
    ggml_tensor * tensor = nullptr;
    const uint8_t * data = nullptr;
    size_t size = 0;
    ggml_type type = GGML_TYPE_COUNT;
};

class GgufTensorDataReader {
public:
    GgufTensorDataReader();
    ~GgufTensorDataReader();

    GgufTensorDataReader(const GgufTensorDataReader &) = delete;
    GgufTensorDataReader & operator=(const GgufTensorDataReader &) = delete;

    bool open(const std::string & path,
              bool build_merged_tensor_context,
              std::string & err);
    bool open_mmaps(std::string & err);

    const gguf_context * primary_context() const;
    ggml_context * merged_context() const;
    ggml_context * release_merged_context();

    int shard_count() const;
    const gguf_context * shard_context(int shard_index) const;

    bool find_tensor(const char * name, GgufTensorRef & out) const;
    const std::string & shard_path(int shard_index) const;
    const void * shard_mmap_data(int shard_index) const;
    size_t shard_mmap_size(int shard_index) const;

    // Transfer the sole mmap to the caller. Valid only for single-shard readers.
    bool release_single_mmap(const void *& data, size_t & size, int & fd, std::string & err);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::vector<LayerExpertFileData> make_layer_expert_file_data(
    const GgufTensorDataReader & reader,
    int n_layer);

int64_t gguf_tensor_ref_row_width(const GgufTensorRef & ref);
int64_t gguf_tensor_ref_row_count(const GgufTensorRef & ref);

bool gguf_tensor_ref_rows_to_host_f32(
    const GgufTensorRef & ref,
    int64_t row_begin,
    int64_t row_count,
    std::vector<float> & out,
    std::string * err = nullptr);

bool gguf_tensor_ref_all_to_host_f32(
    const GgufTensorRef & ref,
    std::vector<float> & out,
    std::string * err = nullptr);

} // namespace dflash::common
