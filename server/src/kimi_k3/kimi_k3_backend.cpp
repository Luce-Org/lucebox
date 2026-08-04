#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include "kimi_k3_backend.h"

#include "common/dynamic_backend.h"
#include "common/moe_expert_package.h"
#include "common/moe_hybrid_placement.h"
#include "common/moe_stream_cache_policy.h"
#include "common/sampler.h"
#include "dflash27b.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash::common {
namespace {

void close_file_descriptor(int fd) {
#if defined(_WIN32)
    ::_close(fd);
#else
    ::close(fd);
#endif
}

class ScopedFileDescriptors {
public:
    ~ScopedFileDescriptors() {
        for (int fd : values_) close_file_descriptor(fd);
    }

    void add(int fd) { values_.push_back(fd); }

    void close(int fd) {
        const auto found = std::find(values_.begin(), values_.end(), fd);
        if (found == values_.end()) return;
        close_file_descriptor(fd);
        values_.erase(found);
    }

private:
    std::vector<int> values_;
};

bool open_nvme_source(const std::string & path,
                      ScopedFileDescriptors & descriptors,
                      MoeNvmeSource & out,
                      std::string & error) {
#if defined(_WIN32)
    const int fd = ::_open(path.c_str(), _O_RDONLY | _O_BINARY);
#else
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
#endif
    if (fd < 0) {
        error = "cannot open " + path + ": " + std::strerror(errno);
        return false;
    }
    uint64_t bytes = 0;
#if defined(_WIN32)
    struct _stat64 stat_buffer {};
    if (::_fstat64(fd, &stat_buffer) == 0 && stat_buffer.st_size > 0) {
        bytes = static_cast<uint64_t>(stat_buffer.st_size);
    }
#else
    struct stat stat_buffer {};
    if (::fstat(fd, &stat_buffer) == 0 && stat_buffer.st_size > 0) {
        bytes = static_cast<uint64_t>(stat_buffer.st_size);
    }
#endif
    if (bytes == 0 || bytes > std::numeric_limits<size_t>::max()) {
        close_file_descriptor(fd);
        error = "cannot determine file size for " + path;
        return false;
    }
    descriptors.add(fd);
    out = {nullptr, static_cast<size_t>(bytes), fd};
    return true;
}

bool file_path_exists(const char * path) {
    if (!path || !*path) return false;
#if defined(_WIN32)
    struct _stat64 st {};
    return ::_stat64(path, &st) == 0 && st.st_size > 0;
#else
    struct stat st {};
    return ::stat(path, &st) == 0 && st.st_size > 0;
#endif
}

bool create_package_output(const std::string & path,
                           ScopedFileDescriptors & descriptors,
                           int & fd, std::string & error) {
#if defined(_WIN32)
    fd = ::_open(path.c_str(),
                 _O_CREAT | _O_TRUNC | _O_RDWR | _O_BINARY,
                 _S_IREAD | _S_IWRITE);
#else
    fd = ::open(path.c_str(),
                O_CREAT | O_TRUNC | O_RDWR | O_CLOEXEC, 0644);
#endif
    if (fd < 0) {
        error = "cannot create " + path + ": " + std::strerror(errno);
        return false;
    }
    descriptors.add(fd);
    return true;
}

std::string package_temporary_path(const std::string & path) {
#if defined(_WIN32)
    const int process_id = ::_getpid();
#else
    const int process_id = static_cast<int>(::getpid());
#endif
    return path + ".partial." + std::to_string(process_id);
}

bool publish_package(const std::string & temporary,
                     const std::string & destination,
                     std::string & error) {
#if defined(_WIN32)
    if (::MoveFileExA(temporary.c_str(), destination.c_str(),
                      MOVEFILE_REPLACE_EXISTING |
                      MOVEFILE_WRITE_THROUGH)) {
        return true;
    }
    error = "cannot publish expert package " + destination +
            ": Windows error " + std::to_string(::GetLastError());
#else
    if (::rename(temporary.c_str(), destination.c_str()) == 0) return true;
    error = "cannot publish expert package " + destination + ": " +
            std::strerror(errno);
#endif
    return false;
}

} // namespace

KimiK3Backend::KimiK3Backend(const KimiK3BackendConfig & cfg) : cfg_(cfg) {}

KimiK3Backend::~KimiK3Backend() {
    shutdown();
}

void KimiK3Backend::release_expert_backend() {
    if (expert_backend_) {
        ggml_backend_free(expert_backend_);
        expert_backend_ = nullptr;
    }
    expert_gpu_ = -1;
    expert_backend_kind_ = PlacementBackend::Auto;
}

bool KimiK3Backend::init_streaming() {
    routing_stats_.reset();
    routing_stats_out_path_.clear();
    if (!weights_.routed_experts_streamed ||
        weights_.streamed_layer_regions.empty() ||
        weights_.max_streamed_expert_bytes == 0) {
        std::fprintf(stderr,
                     "[kimi-k3] routed expert streaming metadata is incomplete\n");
        return false;
    }

    MoeExpertOwnerPlacement owner;
    std::string error;
    if (!resolve_moe_expert_owner_placement(
            cfg_.device.primary_gpu(), cfg_.expert_gpu,
            owner, &error)) {
        std::fprintf(stderr,
                     "[kimi-k3] invalid expert-owner placement: %s\n",
                     error.c_str());
        return false;
    }
    expert_gpu_ = owner.expert_gpu;
    const PlacementBackend primary_kind =
        cfg_.device.backend == PlacementBackend::Auto
            ? compiled_placement_backend() : cfg_.device.backend;
    PlacementBackend expert_kind = primary_kind;
    if (const char * raw = std::getenv("DFLASH_MOE_TP_BACKEND")) {
        if (*raw && (!parse_placement_backend(raw, expert_kind) ||
                     expert_kind == PlacementBackend::Auto)) {
            std::fprintf(stderr,
                         "[kimi-k3] invalid DFLASH_MOE_TP_BACKEND=%s; "
                         "expected cuda or hip\n", raw);
            return false;
        }
    }
    const bool heterogeneous =
        expert_kind != primary_kind || owner.heterogeneous();
    if (heterogeneous) {
        expert_backend_ = init_placement_backend(
            expert_kind, expert_gpu_, &error);
        if (!expert_backend_) {
            std::fprintf(stderr,
                         "[kimi-k3] expert backend init failed for %s:%d: %s\n",
                         placement_backend_name(expert_kind), expert_gpu_,
                         error.c_str());
            expert_gpu_ = -1;
            return false;
        }
        expert_backend_kind_ = expert_kind;
        std::fprintf(stderr,
                     "[kimi-k3] in-process routed owners primary=%s:%d "
                     "secondary=%s:%d transfer=backend-staged\n",
                     placement_backend_name(primary_kind),
                     cfg_.device.primary_gpu(),
                     placement_backend_name(expert_kind), expert_gpu_);
    }
    auto fail_streaming = [&]() {
        dual_stream_executor_.destroy();
        stream_engine_.destroy();
        secondary_stream_engine_.destroy();
        stream_owner_policy_ = MoeStreamDualOwnerPolicy{};
        stream_placement_ = MoeHybridPlacement{};
        release_expert_backend();
        return false;
    };

    size_t routed_pool_bytes = 0;
    for (const LayerExpertRegions & regions :
         weights_.streamed_layer_regions) {
        size_t bytes_per_expert = 0;
        bool component_overflow = false;
        for (size_t component : {
                 regions.expert_bytes_gate, regions.expert_bytes_up,
                 regions.expert_bytes_down, regions.expert_bytes_gate_up}) {
            if (component >
                std::numeric_limits<size_t>::max() - bytes_per_expert) {
                component_overflow = true;
                break;
            }
            bytes_per_expert += component;
        }
        if (component_overflow) {
            routed_pool_bytes = std::numeric_limits<size_t>::max();
            break;
        }
        if (bytes_per_expert >
            (std::numeric_limits<size_t>::max() - routed_pool_bytes) /
                static_cast<size_t>(weights_.n_expert)) {
            routed_pool_bytes = std::numeric_limits<size_t>::max();
            break;
        }
        routed_pool_bytes +=
            bytes_per_expert * static_cast<size_t>(weights_.n_expert);
    }
    auto stream_config_for = [&](ggml_backend_t owner_backend, int gpu,
                                 PlacementBackend backend_kind,
                                 const char * owner_name) {
        MoeStreamConfig stream_config = MoeStreamConfig::from_env();
        if (std::getenv("DFLASH_MOE_NVME_DEVICE_CACHE_MB")) {
            stream_config.device_cache_bytes =
                std::min(stream_config.device_cache_bytes, routed_pool_bytes);
            return stream_config;
        }
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (ggml_backend_dev_t device =
                ggml_backend_get_device(owner_backend)) {
            ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
        }
        // ggml's generic UMA query intentionally reports available system RAM
        // so model loaders can use managed memory. The streamed cache needs one
        // contiguous native allocation, so cap it using the runtime's native
        // pool even when the owner is provided by a dynamically loaded peer.
        size_t allocation_free = free_bytes;
        size_t allocation_total = total_bytes;
        backend_native_memory(
            owner_backend, &allocation_free, &allocation_total);
        const size_t gib = 1024ULL * 1024ULL * 1024ULL;
        const size_t system_reserve =
            std::max<size_t>(2 * gib, total_bytes / 20);
        const size_t allocation_reserve =
            std::max<size_t>(2 * gib, allocation_total / 20);
        const size_t system_budget = free_bytes > system_reserve
            ? free_bytes - system_reserve : 0;
        const size_t allocation_budget = allocation_free > allocation_reserve
            ? allocation_free - allocation_reserve : 0;
        stream_config.device_cache_bytes = std::min(
            {system_budget, allocation_budget, routed_pool_bytes});
        std::fprintf(stderr,
            "[kimi-k3] %s streamed cache: device=%s:%d free=%.2f GiB "
            "alloc-free=%.2f GiB reserve=%.2f/%.2f GiB pool=%.2f GiB "
            "cache=%.2f GiB\n",
            owner_name, placement_backend_name(backend_kind), gpu,
            static_cast<double>(free_bytes) / gib,
            static_cast<double>(allocation_free) / gib,
            static_cast<double>(system_reserve) / gib,
            static_cast<double>(allocation_reserve) / gib,
            static_cast<double>(routed_pool_bytes) / gib,
            static_cast<double>(stream_config.device_cache_bytes) / gib);
        return stream_config;
    };

    stream_owner_policy_ = MoeStreamDualOwnerPolicy::from_env();
    stream_placement_ = MoeHybridPlacement{};
    const char * placement_path = std::getenv("DFLASH_MOE_PLACEMENT");
    if (placement_path && *placement_path) {
        if (!MoeHybridPlacement::load_json(
                placement_path, stream_placement_, &error) ||
            !stream_placement_.matches(
                static_cast<int>(weights_.streamed_layer_regions.size()),
                weights_.n_expert, weights_.n_expert_used)) {
            std::fprintf(stderr,
                         "[kimi-k3] invalid streamed-expert placement %s: %s\n",
                         placement_path,
                         error.empty() ? "model shape mismatch" : error.c_str());
            return fail_streaming();
        }
        if (expert_backend_) {
            stream_owner_policy_.primary_placement = &stream_placement_;
        }
    }

    ScopedFileDescriptors descriptors;
    std::vector<MoeNvmeSource> model_sources;
    model_sources.reserve(weights_.shard_paths.size());
    for (const std::string & shard : weights_.shard_paths) {
        MoeNvmeSource source;
        if (!open_nvme_source(shard, descriptors, source, error)) {
            std::fprintf(stderr,
                         "[kimi-k3] %s\n", error.c_str());
            return fail_streaming();
        }
        model_sources.push_back(source);
    }

    std::vector<uint32_t> expert_counts(
        weights_.streamed_layer_regions.size(),
        static_cast<uint32_t>(weights_.n_expert));
    const uint64_t source_layout_hash = moe_expert_source_layout_hash(
        model_sources, weights_.streamed_layer_regions, expert_counts);
    std::vector<MoeNvmeSource> package_sources;
    MoeExpertPackageManifest package_manifest;
    const std::vector<MoeNvmeSource> * active_sources = &model_sources;
    const std::vector<LayerExpertRegions> * active_regions =
        &weights_.streamed_layer_regions;
    size_t max_streamed_expert_bytes = weights_.max_streamed_expert_bytes;
    const char * package_path = std::getenv("DFLASH_MOE_EXPERT_PACKAGE");
    const char * package_build =
        std::getenv("DFLASH_MOE_EXPERT_PACKAGE_BUILD");
    const bool build_requested =
        package_build && *package_build &&
        std::strcmp(package_build, "0") != 0;
    const bool force_build =
        build_requested && std::strcmp(package_build, "force") == 0;
    if (build_requested && (!package_path || !*package_path)) {
        std::fprintf(stderr,
            "[kimi-k3] DFLASH_MOE_EXPERT_PACKAGE_BUILD requires "
            "DFLASH_MOE_EXPERT_PACKAGE=<output>\n");
        return fail_streaming();
    }
    if (package_path && *package_path) {
        MoeNvmeSource package_source;
        auto load_matching_package = [&]() -> bool {
            MoeNvmeSource candidate;
            MoeExpertPackageManifest candidate_manifest;
            if (!open_nvme_source(
                    package_path, descriptors, candidate, error)) {
                return false;
            }
            if (!read_moe_expert_package(
                    candidate, candidate_manifest, &error)) {
                descriptors.close(candidate.fd);
                return false;
            }
            const bool shape_matches =
                candidate_manifest.layer_regions.size() ==
                    weights_.streamed_layer_regions.size() &&
                candidate_manifest.expert_counts == expert_counts;
            if (!shape_matches ||
                candidate_manifest.source_layout_hash != source_layout_hash) {
                descriptors.close(candidate.fd);
                error = "expert package does not match this model, "
                        "checkpoint, or shard layout";
                return false;
            }
            package_source = candidate;
            package_manifest = std::move(candidate_manifest);
            return true;
        };

        bool package_loaded = !force_build && load_matching_package();
        if (!package_loaded && !build_requested) {
            std::fprintf(stderr,
                         "[kimi-k3] invalid expert package %s: %s\n",
                         package_path, error.c_str());
            return fail_streaming();
        }
        if (!package_loaded) {
            if (!force_build && file_path_exists(package_path)) {
                std::fprintf(stderr,
                    "[kimi-k3] rebuilding invalid expert package %s: %s\n",
                    package_path, error.c_str());
            }
            error.clear();
            const std::string temporary_path =
                package_temporary_path(package_path);
            int output_fd = -1;
            if (!create_package_output(
                    temporary_path, descriptors, output_fd, error)) {
                std::fprintf(stderr,
                             "[kimi-k3] expert package build failed: %s\n",
                             error.c_str());
                return fail_streaming();
            }
            MoeExpertPackageOptions options;
            options.progress = [](size_t completed, size_t total, void *) {
                std::fprintf(stderr,
                    "[kimi-k3] expert package progress %zu/%zu layers\n",
                    completed, total);
                std::fflush(stderr);
            };
            std::fprintf(stderr,
                "[kimi-k3] compiling expert-major package=%s "
                "(exact weights, one aligned record/expert)\n",
                package_path);
            if (!write_moe_expert_package(
                    output_fd, model_sources,
                    weights_.streamed_layer_regions, expert_counts,
                    options, &package_manifest, &error)) {
                descriptors.close(output_fd);
                (void) std::remove(temporary_path.c_str());
                std::fprintf(stderr,
                             "[kimi-k3] expert package build failed: %s\n",
                             error.c_str());
                return fail_streaming();
            }
            descriptors.close(output_fd);
            if (!publish_package(
                    temporary_path, package_path, error)) {
                (void) std::remove(temporary_path.c_str());
                std::fprintf(stderr,
                             "[kimi-k3] expert package build failed: %s\n",
                             error.c_str());
                return fail_streaming();
            }
            package_loaded = load_matching_package();
            if (!package_loaded) {
                std::fprintf(stderr,
                    "[kimi-k3] published expert package is invalid %s: %s\n",
                    package_path, error.c_str());
                return fail_streaming();
            }
        }
        package_sources.push_back(package_source);
        active_sources = &package_sources;
        active_regions = &package_manifest.layer_regions;
        max_streamed_expert_bytes = std::max(
            max_streamed_expert_bytes, package_manifest.max_record_bytes);
        std::fprintf(stderr,
            "[kimi-k3] using expert-major package=%s layers=%zu "
            "record<=%.2f MiB (one read/expert)\n",
            package_path, package_manifest.layer_regions.size(),
            static_cast<double>(package_manifest.max_record_bytes) /
                (1024.0 * 1024.0));
    }

    const MoeStreamConfig primary_config = stream_config_for(
        backend_, cfg_.device.primary_gpu(), primary_kind, "primary");
    if (!stream_engine_.init(
            backend_, max_streamed_expert_bytes,
            primary_config, &error)) {
        std::fprintf(stderr,
                     "[kimi-k3] primary stream engine initialization failed: %s\n",
                     error.c_str());
        return fail_streaming();
    }
    if (expert_backend_) {
        const MoeStreamConfig secondary_config = stream_config_for(
            expert_backend_, expert_gpu_, expert_backend_kind_, "secondary");
        if (!secondary_stream_engine_.init(
                expert_backend_, max_streamed_expert_bytes,
                secondary_config, &error)) {
            std::fprintf(stderr,
                         "[kimi-k3] secondary stream engine initialization failed: %s\n",
                         error.c_str());
            return fail_streaming();
        }
    }

    const bool primary_bound = stream_engine_.bind_sources(
        *active_sources, *active_regions, &error);
    bool secondary_bound = true;
    std::string secondary_error;
    if (primary_bound && expert_backend_) {
        secondary_bound = secondary_stream_engine_.bind_sources(
            *active_sources, *active_regions, &secondary_error);
    }
    if (!primary_bound || !secondary_bound) {
        std::fprintf(stderr,
                     "[kimi-k3] stream source binding failed: %s\n",
                     primary_bound ? secondary_error.c_str() : error.c_str());
        return fail_streaming();
    }

    std::vector<MoeStreamExpertSpec> layer_specs;
    std::vector<uint64_t> layer_expert_bytes;
    layer_specs.reserve(weights_.streamed_layer_regions.size());
    layer_expert_bytes.reserve(weights_.streamed_layer_regions.size());
    for (size_t local_layer = 0;
         local_layer < weights_.streamed_layer_regions.size(); ++local_layer) {
        const size_t model_layer =
            (size_t) weights_.n_dense_lead + local_layer;
        if (model_layer >= weights_.layers.size()) {
            std::fprintf(stderr,
                         "[kimi-k3] streamed layer metadata is out of range\n");
            return fail_streaming();
        }
        const KimiK3Layer & layer = weights_.layers[model_layer];
        if (!layer.ffn_gate_exps || !layer.ffn_up_exps ||
            !layer.ffn_down_exps) {
            std::fprintf(stderr,
                         "[kimi-k3] streamed layer is missing expert types\n");
            return fail_streaming();
        }
        MoeStreamExpertSpec spec;
        spec.input_dim = weights_.n_expert_latent;
        spec.intermediate_dim = weights_.n_ff_exp;
        spec.output_dim = weights_.n_expert_latent;
        spec.gate_type = layer.ffn_gate_exps->type;
        spec.up_type = layer.ffn_up_exps->type;
        spec.down_type = layer.ffn_down_exps->type;
        spec.gated_activation = MoeGatedActivation::Situ;
        spec.situ_beta = weights_.situ_beta;
        spec.situ_linear_beta = weights_.situ_linear_beta;
        layer_specs.push_back(spec);
        const LayerExpertRegions & regions =
            weights_.streamed_layer_regions[local_layer];
        layer_expert_bytes.push_back(
            (uint64_t) regions.expert_bytes_gate +
            (uint64_t) regions.expert_bytes_up +
            (uint64_t) regions.expert_bytes_down +
            (uint64_t) regions.expert_bytes_gate_up);
    }

    const char * hotness_path = std::getenv("DFLASH_MOE_HOTNESS_CSV");
    if (!hotness_path || !*hotness_path) {
        hotness_path = std::getenv("DFLASH_DS4_HOTNESS_CSV");
    }
    MoeHybridRoutingStats routing_profile;
    const bool have_routing_profile = hotness_path && *hotness_path;
    if (have_routing_profile &&
        (!MoeHybridRoutingStats::load_csv(
             hotness_path, routing_profile, &error) ||
         !routing_profile.matches(
             static_cast<int>(weights_.streamed_layer_regions.size()),
             weights_.n_expert, weights_.n_expert_used))) {
        std::fprintf(stderr,
                     "[kimi-k3] invalid expert hotness profile %s: %s\n",
                     hotness_path,
                     error.empty() ? "model shape mismatch" : error.c_str());
        return fail_streaming();
    }

    auto warm_owner = [&](MoeHybridStreamEngine & engine,
                          MoeStreamCacheOwner owner_kind,
                          const char * owner_name) -> bool {
        const int slots = engine.device_slot_count();
        // Preserve a meaningful adaptive region when the profile drifts. Tiny
        // caches still retain the two slots needed by the miss pipeline.
        const int reserve_slots = std::max(2, slots / 4);
        const size_t max_entries = slots > reserve_slots
            ? (size_t) (slots - reserve_slots) : 0;
        if (max_entries == 0) return true;

        std::vector<MoeStreamCacheWarmEntry> plan;
        if (have_routing_profile) {
            MoeStreamCachePlanConfig plan_config;
            plan_config.max_entries = max_entries;
            plan_config.owner = owner_kind;
            const MoeStreamDualOwnerPolicy * policy =
                owner_kind == MoeStreamCacheOwner::All
                    ? nullptr : &stream_owner_policy_;
            if (!build_moe_stream_cache_plan(
                    routing_profile, layer_expert_bytes,
                    plan_config, policy, plan, &error)) {
                return false;
            }
        } else if (!stream_placement_.empty() &&
                   owner_kind != MoeStreamCacheOwner::Secondary) {
            for (int layer = 0; layer < stream_placement_.n_layer; ++layer) {
                const auto & ids =
                    stream_placement_.hot_expert_ids[(size_t) layer];
                for (size_t rank = 0; rank < ids.size(); ++rank) {
                    plan.push_back({
                        (int32_t) layer, ids[rank],
                        (uint64_t) std::max<size_t>(1, ids.size() - rank),
                        layer_expert_bytes[(size_t) layer]});
                }
            }
        }
        if (plan.empty()) return true;

        MoeStreamCacheWarmStats warm_stats;
        if (!engine.warm_and_pin_device_cache(
                layer_specs, plan, reserve_slots, &warm_stats, &error)) {
            return false;
        }
        std::fprintf(stderr,
            "[kimi-k3] %s profile-warm cache: requested=%zu pinned=%zu "
            "resident=%zu capacity-drops=%zu source=%s\n",
            owner_name, warm_stats.requested, warm_stats.admitted,
            warm_stats.already_resident, warm_stats.capacity_drops,
            have_routing_profile ? hotness_path : placement_path);
        return true;
    };

    if (!warm_owner(
            stream_engine_,
            expert_backend_ ? MoeStreamCacheOwner::Primary
                            : MoeStreamCacheOwner::All,
            "primary") ||
        (expert_backend_ &&
         !warm_owner(secondary_stream_engine_,
                     MoeStreamCacheOwner::Secondary,
                     "secondary"))) {
        std::fprintf(stderr,
                     "[kimi-k3] profile-guided cache warmup failed: %s\n",
                     error.c_str());
        return fail_streaming();
    }

    if (expert_backend_ && !dual_stream_executor_.init(
            stream_engine_, secondary_stream_engine_, &error)) {
        std::fprintf(stderr,
                     "[kimi-k3] dual-owner executor initialization failed: %s\n",
                     error.c_str());
        return fail_streaming();
    }
    if (const char * stats_path = std::getenv("DFLASH_MOE_ROUTE_STATS_OUT")) {
        if (*stats_path) {
            routing_stats_ = std::make_shared<MoeHybridRoutingStats>();
            if (!routing_stats_->init(
                    static_cast<int>(weights_.streamed_layer_regions.size()),
                    weights_.n_expert, weights_.n_expert_used)) {
                std::fprintf(stderr,
                    "[kimi-k3] failed to initialize routed-expert statistics\n");
                return fail_streaming();
            }
            routing_stats_out_path_ = stats_path;
            std::fprintf(stderr,
                "[kimi-k3] recording native route counts to %s\n",
                routing_stats_out_path_.c_str());
        }
    }
    if (expert_backend_) {
        std::fprintf(stderr,
            "[kimi-k3] routed experts dual-owner: shards=%zu layers=%zu "
            "primary=%s:%d/%s/%.2fGiB secondary=%s:%d/%s/%.2fGiB "
            "primary_share=%d/1000 placement=%s\n",
            weights_.shard_paths.size(),
            weights_.streamed_layer_regions.size(),
            placement_backend_name(primary_kind), cfg_.device.primary_gpu(),
            stream_engine_.io_backend_name(),
            static_cast<double>(stream_engine_.device_cache_bytes()) /
                (1024.0 * 1024.0 * 1024.0),
            placement_backend_name(expert_backend_kind_), expert_gpu_,
            secondary_stream_engine_.io_backend_name(),
            static_cast<double>(secondary_stream_engine_.device_cache_bytes()) /
                (1024.0 * 1024.0 * 1024.0),
            stream_owner_policy_.primary_share_per_mille,
            stream_owner_policy_.primary_placement ? "profile" : "hash");
    } else {
        std::fprintf(stderr,
            "[kimi-k3] routed experts file-backed: shards=%zu layers=%zu "
            "io=%s gpu=%d cache=%.2f GiB\n",
            weights_.shard_paths.size(),
            weights_.streamed_layer_regions.size(),
            stream_engine_.io_backend_name(),
            cfg_.device.primary_gpu(),
            static_cast<double>(stream_engine_.device_cache_bytes()) /
                (1024.0 * 1024.0 * 1024.0));
    }
    return true;
}

bool KimiK3Backend::init() {
    if (!cfg_.model_path) {
        std::fprintf(stderr, "[kimi-k3] model path is null\n");
        return false;
    }
    backend_ = ggml_backend_cuda_init(cfg_.device.primary_gpu());
    if (!backend_) {
        std::fprintf(stderr, "[kimi-k3] GPU backend init failed for device %d\n",
                     cfg_.device.primary_gpu());
        return false;
    }
    const bool stream_routed_experts =
        cfg_.moe_storage != MoeStoragePolicy::Resident;
    if (!load_kimi_k3_gguf(
            cfg_.model_path, backend_, weights_,
            stream_routed_experts)) {
        std::fprintf(stderr, "[kimi-k3] model load failed: %s\n",
                     dflash27b_last_error());
        return false;
    }
    const int max_ctx = std::max(1, cfg_.device.max_ctx);
    if (!create_kimi_k3_cache(backend_, weights_, max_ctx, cache_)) {
        std::fprintf(stderr, "[kimi-k3] cache allocation failed (max_ctx=%d)\n",
                     max_ctx);
        return false;
    }
    if (weights_.routed_experts_streamed && !init_streaming()) return false;
    std::fprintf(stderr,
        "[kimi-k3] native backend ready on device %d (max_ctx=%d, "
        "experts=%s, correctness-first sequential prefill)\n",
        cfg_.device.primary_gpu(), max_ctx,
        !weights_.routed_experts_streamed ? "resident" :
            (expert_backend_ ? "nvme-dual-owner" : "nvme-single-owner"));
    std::fflush(stderr);
    return true;
}

void KimiK3Backend::print_ready_banner() const {
    std::printf("[kimi-k3-daemon] ready (layers=%d hidden=%d experts=%d "
                "vocab=%d max_ctx=%d)\n",
                weights_.n_layer, weights_.n_embd, weights_.n_expert,
                weights_.n_vocab, cache_.max_ctx);
    std::fflush(stdout);
}

bool KimiK3Backend::park(ParkTarget target) {
    if (!park_target_includes_target_model(target)) return false;
    if (!parked_) {
        maybe_save_routing_stats();
        dual_stream_executor_.destroy();
        stream_engine_.destroy();
        secondary_stream_engine_.destroy();
        release_expert_backend();
        free_kimi_k3_weights(weights_);
        parked_ = true;
    }
    return true;
}

bool KimiK3Backend::unpark(ParkTarget target) {
    if (!park_target_includes_target_model(target)) return false;
    if (parked_) {
        if (!load_kimi_k3_gguf(
                cfg_.model_path, backend_, weights_,
                cfg_.moe_storage != MoeStoragePolicy::Resident) ||
            (weights_.routed_experts_streamed && !init_streaming())) {
            return false;
        }
        parked_ = false;
    }
    return true;
}

int32_t KimiK3Backend::choose_token(const std::vector<float> & logits,
                                    const SamplerCfg & sampler,
                                    const std::vector<int32_t> & history) {
    if (sampler.needs_logit_processing()) {
        return sample_logits(logits.data(), weights_.n_vocab,
                             sampler, history, rng_);
    }
    return static_cast<int32_t>(std::distance(logits.begin(),
        std::max_element(logits.begin(), logits.end())));
}

GenerateResult KimiK3Backend::generate_impl(const GenerateRequest & req,
                                            const DaemonIO & io) {
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    if (parked_) {
        result.fail(GenerateErrorCode::ModelParked);
        out_io.emit(-1);
        return result;
    }
    if (req.prompt.empty()) {
        result.fail(GenerateErrorCode::PrefillFailed, "empty prompt");
        out_io.emit(-1);
        return result;
    }
    if (req.prompt.size() + static_cast<size_t>(std::max(0, req.n_gen)) >
        static_cast<size_t>(cache_.max_ctx)) {
        result.fail(GenerateErrorCode::ContextOverflow,
                    "prompt plus generation exceeds Kimi-K3 cache");
        out_io.emit(-1);
        return result;
    }
    if (req.do_sample && req.sampler.seed != 0) rng_.seed(req.sampler.seed);

    reset_kimi_k3_cache(cache_);
    std::vector<float> logits;
    const auto prefill_begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < req.prompt.size(); ++i) {
        if (!kimi_k3_step(
                backend_, weights_, cache_, req.prompt[i],
                static_cast<int>(i), logits, &stream_engine_,
                dual_stream_executor_.is_ready()
                    ? &dual_stream_executor_ : nullptr,
                &stream_owner_policy_, routing_stats_.get())) {
            result.fail(GenerateErrorCode::PrefillFailed,
                        dflash27b_last_error());
            out_io.emit(-1);
            return result;
        }
    }
    const auto prefill_end = std::chrono::steady_clock::now();
    result.prefill_s = std::chrono::duration<double>(prefill_end - prefill_begin).count();

    const auto decode_begin = std::chrono::steady_clock::now();
    bool budget_close_started = false;
    size_t close_inject_pos = 0;
    for (int i = 0; i < req.n_gen; ++i) {
        int32_t next = choose_token(logits, req.sampler, result.tokens);

        // Preserve the shared Level-2 budget contract even before speculative
        // decode support lands for Kimi-K3.
        const auto & close_ids = req.budget_hook.close_token_ids;
        if (!close_ids.empty()) {
            if (budget_close_started && close_inject_pos < close_ids.size()) {
                next = close_ids[close_inject_pos++];
                result.budget_forced_close = true;
            } else if (!budget_close_started &&
                       req.n_gen - i <=
                           req.budget_hook.hard_limit_remaining) {
                budget_close_started = true;
                if (next == close_ids.front()) {
                    close_inject_pos = 1;
                } else {
                    next = close_ids.front();
                    close_inject_pos = 1;
                    result.budget_forced_close = true;
                }
            }
        }

        result.tokens.push_back(next);
        out_io.emit(next);
        if (out_io.cancelled || next == weights_.eos_token_id) break;
        if (i + 1 < req.n_gen) {
            if (!kimi_k3_step(
                    backend_, weights_, cache_, next,
                    cache_.cur_pos, logits, &stream_engine_,
                    dual_stream_executor_.is_ready()
                        ? &dual_stream_executor_ : nullptr,
                    &stream_owner_policy_, routing_stats_.get())) {
                result.fail(GenerateErrorCode::DecodeFailed,
                            dflash27b_last_error());
                out_io.emit(-1);
                return result;
            }
        }
    }
    const auto decode_end = std::chrono::steady_clock::now();
    result.decode_s = std::chrono::duration<double>(decode_end - decode_begin).count();
    maybe_save_routing_stats();
    out_io.emit(-1);
    result.succeed();
    return result;
}

bool KimiK3Backend::snapshot_save(int slot) {
    (void)slot;
    return false;
}

void KimiK3Backend::snapshot_free(int slot) {
    (void)slot;
}

bool KimiK3Backend::snapshot_used(int slot) const {
    (void)slot;
    return false;
}

int KimiK3Backend::snapshot_cur_pos(int slot) const {
    (void)slot;
    return 0;
}

GenerateResult KimiK3Backend::restore_and_generate_impl(
        int slot, const GenerateRequest & req, const DaemonIO & io) {
    (void)slot;
    (void)req;
    GenerateResult result;
    result.fail(GenerateErrorCode::InvalidSnapshotSlot,
                "Kimi-K3 prefix snapshots are not implemented yet");
    io.emit(-1);
    return result;
}

bool KimiK3Backend::handle_compress(const std::string & line,
                                    const DaemonIO & io) {
    (void)line;
    (void)io;
    return false;
}

void KimiK3Backend::shutdown() {
    maybe_save_routing_stats();
    dual_stream_executor_.destroy();
    stream_engine_.destroy();
    secondary_stream_engine_.destroy();
    release_expert_backend();
    free_kimi_k3_cache(cache_);
    free_kimi_k3_weights(weights_);
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    routing_stats_.reset();
    routing_stats_out_path_.clear();
    parked_ = false;
}

void KimiK3Backend::maybe_save_routing_stats() {
    if (!routing_stats_ || routing_stats_out_path_.empty()) return;
    std::string error;
    if (!routing_stats_->save_csv(routing_stats_out_path_, &error)) {
        std::fprintf(stderr,
            "[kimi-k3] failed to save route statistics %s: %s\n",
            routing_stats_out_path_.c_str(), error.c_str());
    }
}

} // namespace dflash::common
