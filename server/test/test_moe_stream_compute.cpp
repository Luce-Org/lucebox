#include "CppUnitTestFramework.hpp"
#include "common/moe_hybrid_stream.h"

#include "ggml-cuda.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

using namespace dflash::common;

#define STREAM_REQUIRE(cond) do { \
    if (!(cond)) throw std::runtime_error(std::string(__FILE__) + ":" + \
        std::to_string(__LINE__) + ": " + #cond); \
} while (0)

namespace {

struct MoeStreamComputeFixture {};

constexpr int kExperts = 3;
// 256 deliberately does not satisfy CUDA/HIP's 512-element quantized matrix
// row padding. The MXFP4 case below therefore exercises the padded GPU-slot
// path that real Kimi-K3 exposed.
constexpr int kInput = 256;
constexpr int kFf = 64;
constexpr int kOutput = 128;
constexpr int kTokens = 2;
constexpr int kTopK = 2;

struct TempFile {
    int fd = -1;

    explicit TempFile(const std::vector<uint8_t> & bytes) {
        char path[] = "/tmp/luce-moe-stream-XXXXXX";
        fd = ::mkstemp(path);
        if (fd < 0) throw std::runtime_error("mkstemp failed");
        (void) ::unlink(path);
        size_t done = 0;
        while (done < bytes.size()) {
            const ssize_t wrote = ::pwrite(
                fd, bytes.data() + done, bytes.size() - done, (off_t) done);
            if (wrote <= 0) throw std::runtime_error("pwrite failed");
            done += (size_t) wrote;
        }
    }

    ~TempFile() {
        if (fd >= 0) ::close(fd);
    }
};

float gate_value(int expert, int row, int column) {
    return 0.08f * std::sin(
        0.17f * (float) (1 + expert * 11 + row * 5 + column));
}

float up_value(int expert, int row, int column) {
    return 0.07f * std::cos(
        0.13f * (float) (3 + expert * 7 + row * 3 + column));
}

float down_value(int expert, int row, int column) {
    return 0.06f * std::sin(
        0.11f * (float) (5 + expert * 13 + row * 2 + column));
}

void fill_weights(std::vector<float> & gate,
                  std::vector<float> & up,
                  std::vector<float> & down) {
    gate.resize((size_t) kExperts * kFf * kInput);
    up.resize(gate.size());
    down.resize((size_t) kExperts * kOutput * kFf);
    for (int expert = 0; expert < kExperts; ++expert) {
        for (int row = 0; row < kFf; ++row) {
            for (int column = 0; column < kInput; ++column) {
                const size_t i = ((size_t) expert * kFf + row) * kInput + column;
                gate[i] = gate_value(expert, row, column);
                up[i] = up_value(expert, row, column);
            }
        }
        for (int row = 0; row < kOutput; ++row) {
            for (int column = 0; column < kFf; ++column) {
                const size_t i = ((size_t) expert * kOutput + row) * kFf + column;
                down[i] = down_value(expert, row, column);
            }
        }
    }
}

void append_at(std::vector<uint8_t> & file, size_t offset,
               const float * values, size_t count) {
    STREAM_REQUIRE(offset <= file.size());
    STREAM_REQUIRE(count * sizeof(float) <= file.size() - offset);
    std::memcpy(file.data() + offset, values, count * sizeof(float));
}

struct ModelBytes {
    std::vector<uint8_t> file;
    LayerExpertRegions regions;
    size_t slot_bytes = 0;
};

ModelBytes make_model_bytes(bool expert_major,
                            const std::vector<float> & gate,
                            const std::vector<float> & up,
                            const std::vector<float> & down) {
    const size_t gate_bytes = (size_t) kInput * kFf * sizeof(float);
    const size_t up_bytes = gate_bytes;
    const size_t down_bytes = (size_t) kFf * kOutput * sizeof(float);
    ModelBytes model;
    model.regions.expert_bytes_gate = gate_bytes;
    model.regions.expert_bytes_up = up_bytes;
    model.regions.expert_bytes_down = down_bytes;

    if (!expert_major) {
        const size_t gate_stack = gate_bytes * kExperts;
        const size_t up_stack = up_bytes * kExperts;
        const size_t down_stack = down_bytes * kExperts;
        model.file.resize(gate_stack + up_stack + down_stack);
        model.regions.gate_exps = {0, gate_stack};
        model.regions.up_exps = {gate_stack, up_stack};
        model.regions.down_exps = {gate_stack + up_stack, down_stack};
        std::memcpy(model.file.data(), gate.data(), gate_stack);
        std::memcpy(model.file.data() + gate_stack, up.data(), up_stack);
        std::memcpy(model.file.data() + gate_stack + up_stack,
                    down.data(), down_stack);
        model.slot_bytes = gate_bytes + up_bytes + down_bytes;
        return model;
    }

    constexpr size_t kGap = 256;
    const size_t gate_offset = 0;
    const size_t up_offset = gate_bytes + kGap;
    const size_t down_offset = up_offset + up_bytes + kGap;
    const size_t stride = down_offset + down_bytes;
    model.file.assign(stride * kExperts, 0);
    model.regions.expert_major.enabled = true;
    model.regions.expert_major.experts = {0, model.file.size()};
    model.regions.expert_major.expert_stride = stride;
    model.regions.expert_major.gate_offset = gate_offset;
    model.regions.expert_major.up_offset = up_offset;
    model.regions.expert_major.down_offset = down_offset;
    for (int expert = 0; expert < kExperts; ++expert) {
        const size_t base = (size_t) expert * stride;
        append_at(model.file, base + gate_offset,
                  gate.data() + (size_t) expert * kFf * kInput,
                  (size_t) kFf * kInput);
        append_at(model.file, base + up_offset,
                  up.data() + (size_t) expert * kFf * kInput,
                  (size_t) kFf * kInput);
        append_at(model.file, base + down_offset,
                  down.data() + (size_t) expert * kOutput * kFf,
                  (size_t) kOutput * kFf);
    }
    model.slot_bytes = stride;
    return model;
}

std::vector<uint8_t> quantize_mxfp4(const std::vector<float> & values,
                                    int columns, int rows) {
    const size_t bytes = ggml_row_size(GGML_TYPE_MXFP4, columns) *
                         static_cast<size_t>(rows);
    std::vector<uint8_t> quantized(bytes);
    const size_t written = ggml_quantize_chunk(
        GGML_TYPE_MXFP4, values.data(), quantized.data(), 0,
        rows, columns, nullptr);
    STREAM_REQUIRE(written == bytes);
    return quantized;
}

std::vector<float> dequantize_mxfp4(const std::vector<uint8_t> & values,
                                    int columns, int rows) {
    const size_t row_bytes = ggml_row_size(GGML_TYPE_MXFP4, columns);
    STREAM_REQUIRE(values.size() == row_bytes * static_cast<size_t>(rows));
    std::vector<float> dequantized(
        static_cast<size_t>(columns) * static_cast<size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        dequantize_row_mxfp4(
            reinterpret_cast<const block_mxfp4 *>(
                values.data() + static_cast<size_t>(row) * row_bytes),
            dequantized.data() + static_cast<size_t>(row) * columns,
            columns);
    }
    return dequantized;
}

ModelBytes make_mxfp4_model_bytes(const std::vector<float> & gate,
                                  const std::vector<float> & up,
                                  const std::vector<float> & down,
                                  std::vector<float> & gate_dequantized,
                                  std::vector<float> & up_dequantized,
                                  std::vector<float> & down_dequantized) {
    const std::vector<uint8_t> gate_q = quantize_mxfp4(
        gate, kInput, kExperts * kFf);
    const std::vector<uint8_t> up_q = quantize_mxfp4(
        up, kInput, kExperts * kFf);
    const std::vector<uint8_t> down_q = quantize_mxfp4(
        down, kFf, kExperts * kOutput);
    gate_dequantized = dequantize_mxfp4(
        gate_q, kInput, kExperts * kFf);
    up_dequantized = dequantize_mxfp4(
        up_q, kInput, kExperts * kFf);
    down_dequantized = dequantize_mxfp4(
        down_q, kFf, kExperts * kOutput);

    ModelBytes model;
    model.regions.expert_bytes_gate =
        ggml_row_size(GGML_TYPE_MXFP4, kInput) * kFf;
    model.regions.expert_bytes_up = model.regions.expert_bytes_gate;
    model.regions.expert_bytes_down =
        ggml_row_size(GGML_TYPE_MXFP4, kFf) * kOutput;
    model.regions.gate_exps = {0, gate_q.size()};
    model.regions.up_exps = {gate_q.size(), up_q.size()};
    model.regions.down_exps = {
        gate_q.size() + up_q.size(), down_q.size()};
    model.file.reserve(gate_q.size() + up_q.size() + down_q.size());
    model.file.insert(model.file.end(), gate_q.begin(), gate_q.end());
    model.file.insert(model.file.end(), up_q.begin(), up_q.end());
    model.file.insert(model.file.end(), down_q.begin(), down_q.end());
    model.slot_bytes = model.regions.expert_bytes_gate +
                       model.regions.expert_bytes_up +
                       model.regions.expert_bytes_down;
    return model;
}

std::vector<float> cpu_reference(
        const std::vector<float> & gate,
        const std::vector<float> & up,
        const std::vector<float> & down,
        const std::vector<float> & input,
        const int32_t * ids,
        const float * weights) {
    constexpr float gate_scale = 0.8f;
    constexpr float up_scale = 1.1f;
    constexpr float down_scale = 0.9f;
    constexpr float beta = 4.0f;
    constexpr float linear_beta = 25.0f;
    std::vector<float> output((size_t) kTokens * kOutput, 0.0f);
    std::vector<float> activated(kFf);
    for (int token = 0; token < kTokens; ++token) {
        for (int rank = 0; rank < kTopK; ++rank) {
            const int expert = ids[token * kTopK + rank];
            for (int row = 0; row < kFf; ++row) {
                float g = 0.0f;
                float u = 0.0f;
                for (int column = 0; column < kInput; ++column) {
                    const size_t wi =
                        ((size_t) expert * kFf + row) * kInput + column;
                    const float x = input[(size_t) token * kInput + column];
                    g += gate[wi] * x;
                    u += up[wi] * x;
                }
                g *= gate_scale;
                u *= up_scale;
                const float nonlinear =
                    beta * std::tanh(g / beta) / (1.0f + std::exp(-g));
                const float linear = linear_beta * std::tanh(u / linear_beta);
                activated[(size_t) row] = nonlinear * linear;
            }
            for (int row = 0; row < kOutput; ++row) {
                float value = 0.0f;
                for (int column = 0; column < kFf; ++column) {
                    const size_t wi =
                        ((size_t) expert * kOutput + row) * kFf + column;
                    value += down[wi] * activated[(size_t) column];
                }
                output[(size_t) token * kOutput + row] +=
                    weights[token * kTopK + rank] * down_scale * value;
            }
        }
    }
    return output;
}

void run_layout_case(ggml_backend_t backend, bool expert_major) {
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> down;
    fill_weights(gate, up, down);
    ModelBytes model = make_model_bytes(expert_major, gate, up, down);
    TempFile file(model.file);

    MoeHybridStorage storage;
    storage.mmap_size = model.file.size();
    storage.mmap_fd = ::dup(file.fd);
    STREAM_REQUIRE(storage.mmap_fd >= 0);
    storage.layer_regions.push_back(model.regions);

    MoeStreamConfig config;
    config.device_slots = 2;
    config.device_cache_bytes = 0;
    config.graph_cache_entries = 4;
    config.nvme.backend = MoeNvmeBackend::ThreadPool;
    config.nvme.direct_io = MoeNvmeDirectMode::Disabled;
    config.nvme.host_slots = 6;
    config.nvme.io_threads = 2;

    MoeHybridStreamEngine engine;
    std::string error;
    STREAM_REQUIRE(engine.init(
        backend, model.slot_bytes, storage, config, &error));

    MoeStreamExpertSpec spec;
    spec.input_dim = kInput;
    spec.intermediate_dim = kFf;
    spec.output_dim = kOutput;
    spec.gate_type = GGML_TYPE_F32;
    spec.up_type = GGML_TYPE_F32;
    spec.down_type = GGML_TYPE_F32;
    spec.gated_activation = MoeGatedActivation::Situ;
    spec.gate_scale = 0.8f;
    spec.up_scale = 1.1f;
    spec.down_scale = 0.9f;

    std::vector<float> input((size_t) kTokens * kInput);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = 0.12f * std::sin(0.07f * (float) (i + 1));
    }
    const int32_t ids[kTokens * kTopK] = {2, 0, 1, 2};
    const float weights[kTokens * kTopK] = {0.65f, 0.35f, 0.55f, 0.45f};
    MoeStreamRouteBatch batch;
    batch.layer = 0;
    batch.n_expert = kExperts;
    batch.top_k = kTopK;
    batch.n_tokens = kTokens;
    batch.inputs = input.data();
    batch.selected_ids = ids;
    batch.selected_weights = weights;

    const std::vector<float> expected =
        cpu_reference(gate, up, down, input, ids, weights);
    std::vector<float> actual;
    STREAM_REQUIRE(eval_moe_streamed_experts(
        engine, spec, batch, actual, &error));
    STREAM_REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = 2.0e-5f + 2.0e-4f * std::fabs(expected[i]);
        STREAM_REQUIRE(std::fabs(actual[i] - expected[i]) <= tolerance);
    }

    const MoeStreamComputeStats first = engine.compute_stats();
    STREAM_REQUIRE(first.graph_builds == 2);
    STREAM_REQUIRE(first.graph_launches == 3);
    STREAM_REQUIRE(eval_moe_streamed_experts(
        engine, spec, batch, actual, &error));
    const MoeStreamComputeStats second = engine.compute_stats();
    STREAM_REQUIRE(second.graph_builds == first.graph_builds);
    STREAM_REQUIRE(second.graph_cache_hits > first.graph_cache_hits);
    STREAM_REQUIRE(second.graph_launches == 6);
    engine.destroy();
}

void run_mxfp4_padding_case(ggml_backend_t backend) {
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> down;
    fill_weights(gate, up, down);
    std::vector<float> gate_dequantized;
    std::vector<float> up_dequantized;
    std::vector<float> down_dequantized;
    ModelBytes model = make_mxfp4_model_bytes(
        gate, up, down, gate_dequantized, up_dequantized,
        down_dequantized);
    TempFile file(model.file);

    MoeHybridStorage storage;
    storage.mmap_size = model.file.size();
    storage.mmap_fd = ::dup(file.fd);
    STREAM_REQUIRE(storage.mmap_fd >= 0);
    storage.layer_regions.push_back(model.regions);

    MoeStreamConfig config;
    config.device_slots = 2;
    config.graph_cache_entries = 4;
    config.nvme.backend = MoeNvmeBackend::ThreadPool;
    config.nvme.direct_io = MoeNvmeDirectMode::Disabled;
    config.nvme.host_slots = 6;

    MoeHybridStreamEngine engine;
    std::string error;
    STREAM_REQUIRE(engine.init(
        backend, model.slot_bytes, storage, config, &error));

    MoeStreamExpertSpec spec;
    spec.input_dim = kInput;
    spec.intermediate_dim = kFf;
    spec.output_dim = kOutput;
    spec.gate_type = GGML_TYPE_MXFP4;
    spec.up_type = GGML_TYPE_MXFP4;
    spec.down_type = GGML_TYPE_MXFP4;
    spec.gated_activation = MoeGatedActivation::Situ;
    spec.gate_scale = 0.8f;
    spec.up_scale = 1.1f;
    spec.down_scale = 0.9f;

    std::vector<float> input(static_cast<size_t>(kTokens) * kInput);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = 0.12f * std::sin(0.07f * static_cast<float>(i + 1));
    }
    const int32_t ids[kTokens * kTopK] = {2, 0, 1, 2};
    const float weights[kTokens * kTopK] = {0.65f, 0.35f, 0.55f, 0.45f};
    MoeStreamRouteBatch batch;
    batch.layer = 0;
    batch.n_expert = kExperts;
    batch.top_k = kTopK;
    batch.n_tokens = kTokens;
    batch.inputs = input.data();
    batch.selected_ids = ids;
    batch.selected_weights = weights;

    const std::vector<float> expected = cpu_reference(
        gate_dequantized, up_dequantized, down_dequantized,
        input, ids, weights);
    std::vector<float> actual;
    STREAM_REQUIRE(eval_moe_streamed_experts(
        engine, spec, batch, actual, &error));
    STREAM_REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        const float tolerance =
            2.0e-4f + 2.0e-3f * std::fabs(expected[i]);
        STREAM_REQUIRE(std::fabs(actual[i] - expected[i]) <= tolerance);
    }
    const MoeStreamComputeStats stats = engine.compute_stats();
    STREAM_REQUIRE(stats.graph_builds == 2);
    STREAM_REQUIRE(stats.graph_launches == 3);
    engine.destroy();
}

} // namespace

TEST_CASE(MoeStreamComputeFixture, persistent_graph_matches_cpu_and_padded_mxfp4) {
    int device = 0;
    if (const char * value = std::getenv("DFLASH_TEST_GPU")) {
        device = std::max(0, std::atoi(value));
    }
    if (device >= ggml_backend_cuda_get_device_count()) {
        std::fprintf(stderr, "skip: requested CUDA/HIP device is unavailable\n");
        return;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(device);
    if (!backend) {
        std::fprintf(stderr, "skip: no CUDA/HIP backend available\n");
        return;
    }
    run_layout_case(backend, false);
    run_layout_case(backend, true);
    run_mxfp4_padding_case(backend);
    ggml_backend_free(backend);
}
