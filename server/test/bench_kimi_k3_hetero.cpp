// Read-only Kimi-K3 routed-core qualification for heterogeneous Lucebox.
//
// This is not a substitute for the Kimi model graph. It isolates the part
// whose placement is genuinely new: exact routed experts moving from NVMe to
// one GPU while that GPU evaluates the persistent IQ1_S + SiTU expert graph.
// The source file only supplies bytes and is never modified.

#include "common/moe_hybrid_stream.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

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

std::vector<int32_t> route_for(int token, int layer, int top_k, bool repeat) {
    std::vector<int32_t> population((size_t) kKimiExperts);
    std::iota(population.begin(), population.end(), 0);
    const uint64_t token_key = repeat ? 0 : (uint64_t) token;
    std::mt19937_64 rng(
        0x4b494d49334c5543ULL ^ (token_key * 0x9e3779b97f4a7c15ULL) ^
        ((uint64_t) layer * 0xbf58476d1ce4e5b9ULL));
    for (int i = 0; i < top_k; ++i) {
        std::uniform_int_distribution<int> choose(i, kKimiExperts - 1);
        const int selected = choose(rng);
        std::swap(population[(size_t) i], population[(size_t) selected]);
    }
    population.resize((size_t) top_k);
    return population;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 8) {
        std::fprintf(stderr,
            "usage: %s MODEL_FILE [device=1] [tokens=1] [layers=92] "
            "[top_k=16] [compute=1] [repeat_routes=0]\n",
            argv[0]);
        return 2;
    }

    uint64_t device_arg = 1;
    uint64_t tokens_arg = 1;
    uint64_t layers_arg = kKimiMoeLayers;
    uint64_t top_k_arg = kKimiTopK;
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
    if (tokens_arg == 0 || layers_arg == 0 || layers_arg > kKimiMoeLayers ||
        top_k_arg == 0 || top_k_arg > kKimiExperts || compute_arg > 1 ||
        repeat_arg > 1) {
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

    const size_t gate_bytes =
        ggml_row_size(GGML_TYPE_IQ1_S, kKimiLatent) * (size_t) kKimiExpertFf;
    const size_t up_bytes = gate_bytes;
    const size_t down_bytes =
        ggml_row_size(GGML_TYPE_IQ1_S, kKimiExpertFf) * (size_t) kKimiLatent;
    const size_t expert_bytes = gate_bytes + up_bytes + down_bytes;
    const size_t gate_stack = gate_bytes * (size_t) kKimiExperts;
    const size_t up_stack = up_bytes * (size_t) kKimiExperts;
    const size_t down_stack = down_bytes * (size_t) kKimiExperts;
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
    std::vector<LayerExpertRegions> regions((size_t) layers_arg, one_layer);

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
    expert_spec.input_dim = (int) kKimiLatent;
    expert_spec.intermediate_dim = (int) kKimiExpertFf;
    expert_spec.output_dim = (int) kKimiLatent;
    expert_spec.gate_type = GGML_TYPE_IQ1_S;
    expert_spec.up_type = GGML_TYPE_IQ1_S;
    expert_spec.down_type = GGML_TYPE_IQ1_S;
    expert_spec.gated_activation = MoeGatedActivation::Situ;
    expert_spec.situ_beta = kSituBeta;
    expert_spec.situ_linear_beta = kSituLinearBeta;

    std::vector<float> model_input((size_t) kKimiLatent);
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
                token, layer, (int) top_k_arg, repeat_arg != 0);
            if (compute_arg) {
                MoeStreamRouteBatch batch;
                batch.layer = layer;
                batch.n_expert = kKimiExperts;
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
        device_arg, description, (long long) kKimiLatent,
        (long long) kKimiExpertFf, kKimiExperts, top_k_arg, layers_arg,
        tokens_arg, compute_arg ? "situ-iq1s" : "off",
        repeat_arg ? "yes" : "no");
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
        " graph_launches=%" PRIu64 "\n",
        gib(stats.payload_bytes), gib(stats.physical_bytes),
        seconds > 0 ? gib(stats.payload_bytes) / seconds : 0.0,
        hit_rate, gib(engine.device_cache_bytes()), stats.errors,
        compute_stats.graph_builds, compute_stats.graph_cache_hits,
        compute_stats.graph_launches);

    engine.destroy();
    ggml_backend_free(backend);
#if defined(_WIN32)
    ::_close(fd);
#else
    ::close(fd);
#endif
    return stats.errors == 0 ? 0 : 1;
}
