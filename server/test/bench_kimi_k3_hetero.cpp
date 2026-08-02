// Read-only Kimi-K3 routed-core qualification for heterogeneous Lucebox.
//
// It accepts either a real Kimi-K3 GGUF (preferred) or a raw source file large
// enough to emulate the released 2.8T geometry. With a GGUF, tensor types,
// dimensions, layer offsets, and per-expert strides come from model metadata;
// the bytes evaluated by the common stream engine are the actual checkpoint
// weights. The source file is read-only and is never modified.

#include "common/moe_hybrid_stream.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace dflash::common;

namespace {

constexpr int kKimiExperts = 896;
constexpr int kKimiTopK = 16;
constexpr int kKimiMoeLayers = 92;
constexpr int64_t kKimiLatent = 3584;
constexpr int64_t kKimiExpertFf = 3072;
constexpr float kSituBeta = 4.0f;
constexpr float kSituLinearBeta = 25.0f;

bool parse_nonnegative(const char * text, uint64_t & out) {
    if (!text || !text[0] || text[0] == '-') return false;
    char * end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') return false;
    out = (uint64_t) value;
    return true;
}

double gib(uint64_t bytes) {
    return (double) bytes / (1024.0 * 1024.0 * 1024.0);
}

std::vector<int32_t> route_for(int token, int layer, int n_expert,
                               int top_k, bool repeat) {
    std::vector<int32_t> population((size_t) n_expert);
    std::iota(population.begin(), population.end(), 0);
    const uint64_t token_key = repeat ? 0 : (uint64_t) token;
    std::mt19937_64 rng(
        0x4b494d49334c5543ULL ^ (token_key * 0x9e3779b97f4a7c15ULL) ^
        ((uint64_t) layer * 0xbf58476d1ce4e5b9ULL));
    for (int i = 0; i < top_k; ++i) {
        std::uniform_int_distribution<int> choose(i, n_expert - 1);
        const int selected = choose(rng);
        std::swap(population[(size_t) i], population[(size_t) selected]);
    }
    population.resize((size_t) top_k);
    return population;
}

struct KimiGgufStreamLayout {
    bool detected = false;
    int n_expert = kKimiExperts;
    int top_k = kKimiTopK;
    int latent = (int) kKimiLatent;
    int expert_ff = (int) kKimiExpertFf;
    float situ_beta = kSituBeta;
    float situ_linear_beta = kSituLinearBeta;
    ggml_type gate_type = GGML_TYPE_IQ1_S;
    ggml_type up_type = GGML_TYPE_IQ1_S;
    ggml_type down_type = GGML_TYPE_IQ1_S;
    size_t expert_bytes = 0;
    std::vector<LayerExpertRegions> regions;
};

uint32_t gguf_u32_or(const gguf_context * g, const char * key, uint32_t fallback) {
    const int64_t id = gguf_find_key(g, key);
    return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_UINT32
        ? gguf_get_val_u32(g, id) : fallback;
}

float gguf_f32_or(const gguf_context * g, const char * key, float fallback) {
    const int64_t id = gguf_find_key(g, key);
    return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_FLOAT32
        ? gguf_get_val_f32(g, id) : fallback;
}

bool inspect_kimi_gguf_layout(const char * path,
                              KimiGgufStreamLayout & out,
                              std::string & error) {
    ggml_context * meta = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &meta;
    gguf_context * g = gguf_init_from_file(path, params);
    if (!g || !meta) {
        if (g) gguf_free(g);
        if (meta) ggml_free(meta);
        return false; // Non-GGUF inputs retain the synthetic compatibility path.
    }
    const int64_t arch_id = gguf_find_key(g, "general.architecture");
    const char * arch = arch_id >= 0 ? gguf_get_val_str(g, arch_id) : "";
    if (std::strcmp(arch, "kimi-k3") != 0) {
        gguf_free(g);
        ggml_free(meta);
        return false;
    }

    out.detected = true;
    out.n_expert = (int) gguf_u32_or(g, "kimi-k3.expert_count", 0);
    out.top_k = (int) gguf_u32_or(g, "kimi-k3.expert_used_count", 0);
    out.latent = (int) gguf_u32_or(g, "kimi-k3.expert_latent_length", 0);
    out.expert_ff = (int) gguf_u32_or(g, "kimi-k3.expert_feed_forward_length", 0);
    out.situ_beta = gguf_f32_or(g, "kimi-k3.activation.situ_beta", kSituBeta);
    out.situ_linear_beta = gguf_f32_or(
        g, "kimi-k3.activation.situ_linear_beta", kSituLinearBeta);
    const int n_layer = (int) gguf_u32_or(g, "kimi-k3.block_count", 0);
    const int dense_lead = (int) gguf_u32_or(
        g, "kimi-k3.leading_dense_block_count", 0);
    if (out.n_expert <= 0 || out.top_k <= 0 || out.top_k > out.n_expert ||
        out.latent <= 0 || out.expert_ff <= 0 || n_layer <= dense_lead) {
        error = "invalid Kimi-K3 GGUF routing metadata";
        gguf_free(g);
        ggml_free(meta);
        return false;
    }

    const size_t data_start = gguf_get_data_offset(g);
    for (int il = dense_lead; il < n_layer; ++il) {
        char gate_name[128], up_name[128], down_name[128];
        std::snprintf(gate_name, sizeof(gate_name),
                      "blk.%d.ffn_gate_exps.weight", il);
        std::snprintf(up_name, sizeof(up_name),
                      "blk.%d.ffn_up_exps.weight", il);
        std::snprintf(down_name, sizeof(down_name),
                      "blk.%d.ffn_down_exps.weight", il);
        const int64_t gate_id = gguf_find_tensor(g, gate_name);
        const int64_t up_id = gguf_find_tensor(g, up_name);
        const int64_t down_id = gguf_find_tensor(g, down_name);
        ggml_tensor * gate = ggml_get_tensor(meta, gate_name);
        ggml_tensor * up = ggml_get_tensor(meta, up_name);
        ggml_tensor * down = ggml_get_tensor(meta, down_name);
        if (gate_id < 0 || up_id < 0 || down_id < 0 || !gate || !up || !down ||
            gate->ne[0] != out.latent || gate->ne[1] != out.expert_ff ||
            gate->ne[2] != out.n_expert ||
            up->ne[0] != out.latent || up->ne[1] != out.expert_ff ||
            up->ne[2] != out.n_expert ||
            down->ne[0] != out.expert_ff || down->ne[1] != out.latent ||
            down->ne[2] != out.n_expert) {
            error = "Kimi-K3 expert tensor shape mismatch at model layer " +
                    std::to_string(il);
            gguf_free(g);
            ggml_free(meta);
            return false;
        }

        const size_t gate_size = gguf_get_tensor_size(g, gate_id);
        const size_t up_size = gguf_get_tensor_size(g, up_id);
        const size_t down_size = gguf_get_tensor_size(g, down_id);
        if (gate_size % (size_t) out.n_expert != 0 ||
            up_size % (size_t) out.n_expert != 0 ||
            down_size % (size_t) out.n_expert != 0) {
            error = "Kimi-K3 expert tensor is not expert-major stridable at layer " +
                    std::to_string(il);
            gguf_free(g);
            ggml_free(meta);
            return false;
        }
        const ggml_type gt = gguf_get_tensor_type(g, gate_id);
        const ggml_type ut = gguf_get_tensor_type(g, up_id);
        const ggml_type dt = gguf_get_tensor_type(g, down_id);
        if (out.regions.empty()) {
            out.gate_type = gt;
            out.up_type = ut;
            out.down_type = dt;
        } else if (out.gate_type != gt || out.up_type != ut || out.down_type != dt) {
            error = "mixed expert tensor types need per-layer stream specs";
            gguf_free(g);
            ggml_free(meta);
            return false;
        }

        LayerExpertRegions regions;
        regions.fused_gate_up = false;
        regions.expert_bytes_gate = gate_size / (size_t) out.n_expert;
        regions.expert_bytes_up = up_size / (size_t) out.n_expert;
        regions.expert_bytes_down = down_size / (size_t) out.n_expert;
        regions.gate_exps = {
            data_start + gguf_get_tensor_offset(g, gate_id), gate_size, 0};
        regions.up_exps = {
            data_start + gguf_get_tensor_offset(g, up_id), up_size, 0};
        regions.down_exps = {
            data_start + gguf_get_tensor_offset(g, down_id), down_size, 0};
        const size_t bytes = regions.expert_bytes_gate +
                             regions.expert_bytes_up +
                             regions.expert_bytes_down;
        if (out.expert_bytes == 0) out.expert_bytes = bytes;
        if (out.expert_bytes != bytes) {
            error = "variable per-layer expert bytes need per-layer stream slots";
            gguf_free(g);
            ggml_free(meta);
            return false;
        }
        out.regions.push_back(regions);
    }
    gguf_free(g);
    ggml_free(meta);
    return !out.regions.empty();
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 8) {
        std::fprintf(stderr,
            "usage: %s MODEL_FILE [device=1] [tokens=1] [layers=all] "
            "[top_k=model] [compute=1] [repeat_routes=0]\n",
            argv[0]);
        return 2;
    }

    uint64_t device_arg = 1;
    uint64_t tokens_arg = 1;
    uint64_t layers_arg = 0;
    uint64_t top_k_arg = 0;
    uint64_t compute_arg = 1;
    uint64_t repeat_arg = 0;
    uint64_t * values[] = {
        &device_arg, &tokens_arg, &layers_arg, &top_k_arg,
        &compute_arg, &repeat_arg,
    };
    for (int i = 2; i < argc; ++i) {
        if (!parse_nonnegative(argv[i], *values[i - 2])) {
            std::fprintf(stderr, "invalid integer: %s\n", argv[i]);
            return 2;
        }
    }
    if (tokens_arg == 0 || compute_arg > 1 || repeat_arg > 1) {
        std::fprintf(stderr, "Kimi benchmark arguments are out of range\n");
        return 2;
    }
    if (device_arg >= (uint64_t) ggml_backend_cuda_get_device_count()) {
        std::fprintf(stderr, "GPU device is out of range\n");
        return 2;
    }

#if defined(_WIN32)
    const int fd = ::_open(argv[1], _O_RDONLY | _O_BINARY);
    struct _stat64 st{};
    if (fd < 0 || ::_fstat64(fd, &st) != 0 || st.st_size <= 0) {
#else
    const int fd = ::open(argv[1], O_RDONLY | O_CLOEXEC);
    struct stat st{};
    if (fd < 0 || ::fstat(fd, &st) != 0 || st.st_size <= 0) {
#endif
        std::fprintf(stderr, "cannot open input file: %s\n", std::strerror(errno));
        return 1;
    }
    const size_t file_bytes = (size_t) st.st_size;

    KimiGgufStreamLayout layout;
    std::string layout_error;
    const bool actual_kimi = inspect_kimi_gguf_layout(argv[1], layout, layout_error);
    if (!actual_kimi && layout.detected) {
        std::fprintf(stderr, "Kimi GGUF layout inspection failed: %s\n",
                     layout_error.c_str());
        return 2;
    }

    int n_expert = layout.n_expert;
    int latent = layout.latent;
    int expert_ff = layout.expert_ff;
    ggml_type gate_type = layout.gate_type;
    ggml_type up_type = layout.up_type;
    ggml_type down_type = layout.down_type;
    float situ_beta = layout.situ_beta;
    float situ_linear_beta = layout.situ_linear_beta;
    size_t expert_bytes = layout.expert_bytes;
    std::vector<LayerExpertRegions> regions;

    if (actual_kimi) {
        if (layers_arg == 0) layers_arg = layout.regions.size();
        if (top_k_arg == 0) top_k_arg = (uint64_t) layout.top_k;
        if (layers_arg > layout.regions.size()) {
            std::fprintf(stderr, "requested layers exceed Kimi GGUF MoE layers\n");
            return 2;
        }
        regions.assign(layout.regions.begin(),
                       layout.regions.begin() + (size_t) layers_arg);
    } else {
        if (layers_arg == 0) layers_arg = kKimiMoeLayers;
        if (top_k_arg == 0) top_k_arg = kKimiTopK;
        const size_t gate_bytes =
            ggml_row_size(gate_type, latent) * (size_t) expert_ff;
        const size_t up_bytes = gate_bytes;
        const size_t down_bytes =
            ggml_row_size(down_type, expert_ff) * (size_t) latent;
        expert_bytes = gate_bytes + up_bytes + down_bytes;
        const size_t gate_stack = gate_bytes * (size_t) n_expert;
        const size_t up_stack = up_bytes * (size_t) n_expert;
        const size_t down_stack = down_bytes * (size_t) n_expert;
        const size_t required_bytes = gate_stack + up_stack + down_stack;
        if (file_bytes < required_bytes) {
            std::fprintf(stderr,
                "input needs at least %.3f GiB for one Kimi expert stack\n",
                gib(required_bytes));
            return 2;
        }
        LayerExpertRegions one_layer;
        one_layer.fused_gate_up = false;
        one_layer.expert_bytes_gate = gate_bytes;
        one_layer.expert_bytes_up = up_bytes;
        one_layer.expert_bytes_down = down_bytes;
        one_layer.gate_exps = {0, gate_stack};
        one_layer.up_exps = {gate_stack, up_stack};
        one_layer.down_exps = {gate_stack + up_stack, down_stack};
        regions.assign((size_t) layers_arg, one_layer);
    }
    if (layers_arg == 0 || top_k_arg == 0 || top_k_arg > (uint64_t) n_expert) {
        std::fprintf(stderr, "Kimi benchmark routing arguments are out of range\n");
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init((int) device_arg);
    if (!backend) {
        std::fprintf(stderr, "failed to initialize GPU backend\n");
        return 1;
    }

    MoeHybridStorage storage;
    storage.mmap_size = file_bytes;
#if defined(_WIN32)
    storage.mmap_fd = -1;
#else
    storage.mmap_fd = ::dup(fd);
#endif
    storage.layer_regions = regions;

    MoeStreamConfig stream_config = MoeStreamConfig::from_env();
    MoeHybridStreamEngine engine;
    std::string error;
    if (!engine.init(backend, expert_bytes, storage, stream_config, &error)) {
        std::fprintf(stderr, "stream engine initialization failed: %s\n", error.c_str());
        ggml_backend_free(backend);
        return 1;
    }

    MoeStreamExpertSpec expert_spec;
    expert_spec.input_dim = latent;
    expert_spec.intermediate_dim = expert_ff;
    expert_spec.output_dim = latent;
    expert_spec.gate_type = gate_type;
    expert_spec.up_type = up_type;
    expert_spec.down_type = down_type;
    expert_spec.gated_activation = MoeGatedActivation::Situ;
    expert_spec.situ_beta = situ_beta;
    expert_spec.situ_linear_beta = situ_linear_beta;

    std::vector<float> model_input((size_t) latent);
    for (size_t i = 0; i < model_input.size(); ++i) {
        model_input[i] = 0.01f * std::sin((float) i * 0.013f);
    }
    std::vector<float> route_weights(
        (size_t) top_k_arg, 1.0f / (float) top_k_arg);
    std::vector<float> routed_output;

    const uint64_t accesses =
        tokens_arg * layers_arg * top_k_arg;
    const auto begin = std::chrono::steady_clock::now();
    for (int token = 0; token < (int) tokens_arg; ++token) {
        for (int layer = 0; layer < (int) layers_arg; ++layer) {
            const std::vector<int32_t> experts = route_for(
                token, layer, n_expert, (int) top_k_arg, repeat_arg != 0);
            if (compute_arg) {
                MoeStreamRouteBatch batch;
                batch.layer = layer;
                batch.n_expert = n_expert;
                batch.top_k = (int) top_k_arg;
                batch.n_tokens = 1;
                batch.inputs = model_input.data();
                batch.selected_ids = experts.data();
                batch.selected_weights = route_weights.data();
                if (!eval_moe_streamed_experts(
                        engine, expert_spec, batch, routed_output, &error)) {
                    std::fprintf(stderr,
                        "common streamed-expert evaluation failed: %s\n",
                        error.c_str());
                    return 1;
                }
                continue;
            }

            engine.request_experts(
                layer, experts.data(), (int) experts.size(), MoeNvmePriority::Demand);

            int staged_slot = -1;
            if (!engine.stage_expert_cached_async(
                    layer, experts[0], &staged_slot, &error)) {
                std::fprintf(stderr, "initial expert stage failed: %s\n", error.c_str());
                return 1;
            }
            for (size_t i = 0; i < experts.size(); ++i) {
                const int current_slot = staged_slot;
                if (!engine.activate_device_slot(current_slot, &error)) {
                    std::fprintf(stderr, "expert activation failed: %s\n", error.c_str());
                    return 1;
                }
                // Disk/H2D for expert N+1 overlaps bookkeeping for expert N.
                if (i + 1 < experts.size()) {
                    int next_slot = -1;
                    if (!engine.stage_expert_cached_async(
                            layer, experts[i + 1], &next_slot, &error)) {
                        ggml_backend_synchronize(backend);
                        std::fprintf(stderr, "pipelined expert stage failed: %s\n", error.c_str());
                        return 1;
                    }
                    staged_slot = next_slot;
                }
                engine.release_device_slot(current_slot);
            }
        }
    }
    ggml_backend_synchronize(backend);
    const auto end = std::chrono::steady_clock::now();
    const double seconds =
        std::chrono::duration<double>(end - begin).count();
    const MoeNvmeStats stats = engine.io_stats();
    const MoeStreamComputeStats compute_stats = engine.compute_stats();
    const double misses = expert_bytes > 0
        ? (double) stats.payload_bytes / (double) expert_bytes : 0.0;
    const double hit_rate = accesses > 0
        ? std::max(0.0, 1.0 - misses / (double) accesses) : 0.0;
    char description[256] = {};
    ggml_backend_cuda_get_device_description(
        (int) device_arg, description, sizeof(description));

    std::printf(
        "kimi_k3 device=%" PRIu64 " description=%s latent=%lld ff=%lld "
        "experts=%d top_k=%" PRIu64 " layers=%" PRIu64 " tokens=%" PRIu64
        " compute=%s repeat_routes=%s\n",
        device_arg, description, (long long) latent,
        (long long) expert_ff, n_expert, top_k_arg, layers_arg,
        tokens_arg, compute_arg ? "situ" : "off",
        repeat_arg ? "yes" : "no");
    std::printf("source=%s gate_type=%s up_type=%s down_type=%s\n",
                actual_kimi ? "actual-kimi-gguf" : "synthetic-layout",
                ggml_type_name(gate_type), ggml_type_name(up_type),
                ggml_type_name(down_type));
    std::printf(
        "expert_bytes=%zu expert_mib=%.6f accesses=%" PRIu64
        " elapsed_s=%.6f routed_core_tok_s=%.6f experts_s=%.2f\n",
        expert_bytes, expert_bytes / 1024.0 / 1024.0, accesses, seconds,
        seconds > 0 ? (double) tokens_arg / seconds : 0.0,
        seconds > 0 ? (double) accesses / seconds : 0.0);
    std::printf(
        "ssd_payload_gib=%.6f physical_gib=%.6f pipeline_gib_s=%.6f "
        "estimated_device_cache_hit=%.4f cache_gib=%.3f io_errors=%" PRIu64
        " graph_builds=%" PRIu64 " graph_hits=%" PRIu64
        " graph_launches=%" PRIu64 " fused_decode_launches=%" PRIu64
        " fused_decode_experts=%" PRIu64 "\n",
        gib(stats.payload_bytes), gib(stats.physical_bytes),
        seconds > 0 ? gib(stats.payload_bytes) / seconds : 0.0,
        hit_rate, gib(engine.device_cache_bytes()), stats.errors,
        compute_stats.graph_builds, compute_stats.graph_cache_hits,
        compute_stats.graph_launches,
        compute_stats.fused_decode_launches,
        compute_stats.fused_decode_experts);

    engine.destroy();
    ggml_backend_free(backend);
#if defined(_WIN32)
    ::_close(fd);
#else
    ::close(fd);
#endif
    return stats.errors == 0 ? 0 : 1;
}
