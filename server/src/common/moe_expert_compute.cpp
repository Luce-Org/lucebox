#include "moe_expert_compute.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>

#if !defined(_WIN32)
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash::common {

namespace {

uint64_t hash_u64(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

const char * nonempty_env(const char * name) {
    const char * raw = std::getenv(name);
    return raw && *raw ? raw : nullptr;
}

int parse_nonnegative_env(const char * name, int fallback) {
    const char * raw = nonempty_env(name);
    if (!raw) return fallback;
    errno = 0;
    char * end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (errno == ERANGE || end == raw || *end != '\0' ||
        value < 0 || value > std::numeric_limits<int>::max()) {
        return fallback;
    }
    return (int)value;
}

std::string make_runtime_key(const MoeExpertComputeRuntimeConfig & cfg) {
    std::string key = cfg.target_path;
    key += "|n_layer=" + std::to_string(cfg.n_layer);
    key += "|n_expert=" + std::to_string(cfg.n_expert);
    key += "|n_used=" + std::to_string(cfg.n_expert_used);
    key += "|n_embd=" + std::to_string(cfg.n_embd);
    key += "|n_ff=" + std::to_string(cfg.n_ff_exp);
    if (const char * ipc_bin = nonempty_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_BIN")) {
        key += "|ipc_bin=";
        key += ipc_bin;
        key += "|ipc_gpu=" + std::to_string(
            parse_nonnegative_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_GPU", 0));
        key += "|ipc_work=";
        if (const char * work_dir = nonempty_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_WORK_DIR")) {
            key += work_dir;
        }
        key += "|ipc_required=" + std::to_string(
            parse_nonnegative_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_REQUIRED", 0));
    } else {
        key += "|cpu";
    }
    return key;
}

std::string make_multi_target_runtime_key(const MoeExpertComputeRuntimeConfig & cfg,
                                          const ExpertSplitComputeRuntime & split_runtime) {
    std::string key = make_runtime_key(cfg);
    key += "|multi-target";
    key += "|targets=" + std::to_string(split_runtime.targets.size());
    key += "|split_fp=" +
        std::to_string(expert_split_compute_runtime_fingerprint(split_runtime));
    for (const auto & target : split_runtime.targets) {
        key += "|";
        key += target.target.name;
        key += "=";
        key += std::to_string(target.placement.total_hot);
    }
    return key;
}

bool validate_executable_file(const char * path, std::string * err) {
#if defined(_WIN32)
    (void)path;
    (void)err;
    return true;
#else
    struct stat st {};
    if (::stat(path, &st) != 0) {
        if (err) *err = std::string("MoE expert compute IPC binary is not accessible: ") +
                        path + ": " + std::strerror(errno);
        return false;
    }
    if (!S_ISREG(st.st_mode)) {
        if (err) *err = std::string("MoE expert compute IPC binary is not a regular file: ") + path;
        return false;
    }
    if (::access(path, X_OK) != 0) {
        if (err) *err = std::string("MoE expert compute IPC binary is not executable: ") +
                        path + ": " + std::strerror(errno);
        return false;
    }
    return true;
#endif
}

class MultiTargetMoeExpertCompute final : public MoeExpertCompute {
public:
    MultiTargetMoeExpertCompute(MoeMultiTargetExpertRuntime * runtime)
        : runtime_(runtime) {}

    bool healthy() const override {
        if (!runtime_) return false;
        for (const auto & target : runtime_->targets) {
            if (!target.compute_active) continue;
            if (!target.runtime.compute_ptr() || !target.runtime.compute->healthy()) {
                return false;
            }
        }
        return true;
    }

    bool prefers_padded_batch() const override {
        if (!runtime_) return false;
        int active_targets = 0;
        bool padded_preferred = false;
        for (const auto & target : runtime_->targets) {
            if (!target.compute_active) continue;
            ++active_targets;
            const MoeExpertCompute * compute = target.runtime.compute_ptr();
            if (compute && compute->prefers_padded_batch()) {
                padded_preferred = true;
            }
        }
        // A single active target can benefit from padded batches to maximize
        // backend-side graph reuse. With multiple active targets, padded dummy
        // slots amplify cross-target scatter and can dominate expert_split
        // prefill cost, so prefer grouped exact-count batches instead.
        return active_targets == 1 && padded_preferred;
    }

    bool compute(const MoeExpertLayer & layer,
                 const float * input,
                 const int32_t * ids,
                 const float * weights,
                 int n_selected,
                 int n_embd,
                 int n_ff,
                 float * output) override {
        if (!output || n_selected < 0 || n_embd <= 0 || n_ff <= 0) return false;
        if (n_selected > 0 && !input) return false;
        if (n_selected == 0) return true;
        if (!runtime_ || layer.layer_idx < 0 ||
            (size_t) layer.layer_idx >= runtime_->layer_routes.size()) {
            return false;
        }

        const MoeMultiTargetLayerRuntime & layer_rt =
            runtime_->layer_routes[(size_t) layer.layer_idx];
        ensure_single_scratch(n_selected, n_embd);
        if (!scatter_selected(layer_rt, ids, weights, n_selected)) {
            return false;
        }

        bool wrote_output = false;
        for (size_t ti = 0; ti < runtime_->targets.size(); ++ti) {
            const int n_target = target_counts_[(size_t)ti];
            if (n_target <= 0) continue;
            auto & target = runtime_->targets[ti];
            const MoeExpertLayer * target_layer = target.runtime.layer_ptr((size_t)layer.layer_idx);
            if (!target_layer || !target.runtime.compute_ptr()) {
                return false;
            }
            float * tmp = target_output_scratch_[(size_t)ti].data();
            if (!target.runtime.compute->compute(
                    *target_layer,
                    input,
                    target_ids_[(size_t)ti].data(),
                    target_weights_[(size_t)ti].data(),
                    n_target,
                    n_embd,
                    n_ff,
                    tmp)) {
                return false;
            }
            if (!wrote_output) {
                std::memcpy(output, tmp, sizeof(float) * (size_t)n_embd);
                wrote_output = true;
            } else {
                for (int i = 0; i < n_embd; ++i) {
                    output[i] += tmp[i];
                }
            }
        }
        if (!wrote_output) {
            std::fill(output, output + n_embd, 0.0f);
        }
        return true;
    }

    bool compute_batch(const MoeExpertLayer & layer,
                       const float * input,
                       const int32_t * ids,
                       const float * weights,
                       int n_tokens,
                       int n_selected,
                       int n_embd,
                       int n_ff,
                       float * output) override {
        if (!output || n_tokens < 0 || n_selected < 0 || n_embd <= 0 || n_ff <= 0) {
            return false;
        }
        if (n_tokens > 0 && n_selected > 0 && (!input || !ids || !weights)) {
            return false;
        }
        if (n_tokens == 0 || n_selected == 0) {
            std::fill(output, output + (size_t)n_tokens * (size_t)n_embd, 0.0f);
            return true;
        }
        if (!runtime_ || layer.layer_idx < 0 ||
            (size_t) layer.layer_idx >= runtime_->layer_routes.size()) {
            return false;
        }

        const MoeMultiTargetLayerRuntime & layer_rt =
            runtime_->layer_routes[(size_t) layer.layer_idx];
        ensure_batch_scratch(n_tokens, n_selected);
        if (!scatter_batch(layer_rt, ids, weights, n_tokens, n_selected)) {
            return false;
        }

        bool wrote_output = false;
        const size_t target_count = runtime_->targets.size();
        for (size_t ti = 0; ti < target_count; ++ti) {
            const int max_target_selected = batch_target_max_selected_[(size_t)ti];
            if (max_target_selected <= 0) continue;

            auto & target = runtime_->targets[ti];
            const MoeExpertLayer * target_layer =
                target.runtime.layer_ptr((size_t)layer.layer_idx);
            MoeExpertCompute * target_compute = target.runtime.compute_ptr();
            if (!target_layer || !target_compute) return false;

            const auto & dense_ids = batch_target_ids_[ti];
            const auto & dense_weights = batch_target_weights_[ti];
            const auto & active_selected_counts =
                batch_active_selected_counts_[(size_t)ti];
            for (int n_target : active_selected_counts) {
                const auto & token_group =
                    batch_token_groups_[ti][(size_t)n_target];
                if (token_group.empty()) continue;
                if (!dispatch_target_batch(
                        *target_compute, *target_layer, input, dense_ids,
                        dense_weights, token_group, n_selected, n_target,
                        n_embd, n_ff, output, batch_output_written_)) {
                    return false;
                }
                wrote_output = true;
            }
        }
        if (!wrote_output) {
            std::fill(output, output + (size_t)n_tokens * (size_t)n_embd, 0.0f);
        }

        return true;
    }

    bool prepare_batch(const MoeExpertLayer & layer,
                       int n_tokens,
                       int n_selected,
                       int n_embd,
                       int n_ff) override {
        if (n_tokens < 0 || n_selected < 0 || n_embd <= 0 || n_ff <= 0) return false;
        if (!runtime_ || layer.layer_idx < 0 ||
            (size_t) layer.layer_idx >= runtime_->layer_routes.size()) {
            return false;
        }
        bool ok = true;
        for (auto & target : runtime_->targets) {
            if (!target.compute_active) continue;
            const MoeExpertLayer * target_layer = target.runtime.layer_ptr((size_t)layer.layer_idx);
            if (!target_layer) return false;
            const int max_selected =
                std::min<int>(n_selected, (int)target_layer->cold_global_by_local.size());
            if (max_selected <= 0) continue;
            if (!target.runtime.compute_ptr()) return false;
            ok = target.runtime.compute->prepare_batch(
                     *target_layer, n_tokens, max_selected, n_embd, n_ff) && ok;
        }
        return ok;
    }

    bool prepare_single(const MoeExpertLayer & layer,
                        int n_selected,
                        int n_embd,
                        int n_ff) override {
        if (n_selected < 0 || n_embd <= 0 || n_ff <= 0) return false;
        if (!runtime_ || layer.layer_idx < 0 ||
            (size_t) layer.layer_idx >= runtime_->layer_routes.size()) {
            return false;
        }
        bool ok = true;
        for (auto & target : runtime_->targets) {
            if (!target.compute_active) continue;
            const MoeExpertLayer * target_layer = target.runtime.layer_ptr((size_t)layer.layer_idx);
            if (!target_layer) return false;
            const int max_selected =
                std::min<int>(n_selected, (int)target_layer->cold_global_by_local.size());
            if (max_selected <= 0) continue;
            if (!target.runtime.compute_ptr()) return false;
            ok = target.runtime.compute->prepare_single(
                     *target_layer, max_selected, n_embd, n_ff) && ok;
        }
        return ok;
    }

private:
    void ensure_single_scratch(int n_selected, int n_embd) {
        const size_t target_count = runtime_ ? runtime_->targets.size() : 0;
        target_counts_.assign(target_count, 0);
        target_ids_.resize(target_count);
        target_weights_.resize(target_count);
        target_output_scratch_.resize(target_count);
        for (size_t ti = 0; ti < target_count; ++ti) {
            target_ids_[ti].clear();
            target_weights_[ti].clear();
            if (target_ids_[ti].capacity() < (size_t)n_selected) {
                target_ids_[ti].reserve((size_t)n_selected);
            }
            if (target_weights_[ti].capacity() < (size_t)n_selected) {
                target_weights_[ti].reserve((size_t)n_selected);
            }
            if (target_output_scratch_[ti].size() < (size_t)n_embd) {
                target_output_scratch_[ti].resize((size_t)n_embd);
            }
        }
    }

    bool scatter_selected(const MoeMultiTargetLayerRuntime & layer_rt,
                          const int32_t * ids,
                          const float * weights,
                          int n_selected) {
        if (!ids || !weights) return false;
        const size_t target_count = runtime_ ? runtime_->targets.size() : 0;
        for (size_t ti = 0; ti < target_count; ++ti) {
            target_counts_[ti] = 0;
            target_ids_[ti].clear();
            target_weights_[ti].clear();
        }

        for (int i = 0; i < n_selected; ++i) {
            const int32_t union_local = ids[i];
            if (union_local < 0 ||
                (size_t) union_local >= layer_rt.route_by_union_local.size()) {
                return false;
            }
            const MoeMultiTargetLayerRoute & route =
                layer_rt.route_by_union_local[(size_t) union_local];
            if (route.target_slot < 0 || (size_t) route.target_slot >= target_count ||
                route.target_local < 0) {
                return false;
            }
            target_ids_[(size_t) route.target_slot].push_back(route.target_local);
            target_weights_[(size_t) route.target_slot].push_back(weights[i]);
            target_counts_[(size_t) route.target_slot]++;
        }
        return true;
    }

    void ensure_batch_scratch(int n_tokens, int n_selected) {
        const size_t target_count = runtime_ ? runtime_->targets.size() : 0;
        const size_t dense_slots = (size_t)n_tokens * (size_t)n_selected;
        batch_target_counts_.resize(target_count);
        batch_target_ids_.resize(target_count);
        batch_target_weights_.resize(target_count);
        batch_token_groups_.resize(target_count);
        batch_active_selected_counts_.resize(target_count);
        batch_target_max_selected_.assign(target_count, 0);
        for (size_t ti = 0; ti < target_count; ++ti) {
            auto & token_groups = batch_token_groups_[ti];
            auto & active_selected_counts =
                batch_active_selected_counts_[ti];
            for (int n_target : active_selected_counts) {
                if (n_target > 0 && (size_t)n_target < token_groups.size()) {
                    token_groups[(size_t)n_target].clear();
                }
            }
            active_selected_counts.clear();
            batch_target_counts_[ti].assign((size_t)n_tokens, 0);
            batch_target_ids_[ti].resize(dense_slots);
            batch_target_weights_[ti].resize(dense_slots);
            if (token_groups.size() != (size_t)n_selected + 1) {
                token_groups.resize((size_t)n_selected + 1);
            }
        }
        batch_output_written_.assign((size_t)n_tokens, 0);
        if (scatter_touched_targets_.capacity() < (size_t)n_selected) {
            scatter_touched_targets_.reserve((size_t)n_selected);
        }
    }

    bool scatter_batch(const MoeMultiTargetLayerRuntime & layer_rt,
                       const int32_t * ids,
                       const float * weights,
                       int n_tokens,
                       int n_selected) {
        const size_t target_count = runtime_ ? runtime_->targets.size() : 0;
        for (int t = 0; t < n_tokens; ++t) {
            scatter_touched_targets_.clear();
            for (int i = 0; i < n_selected; ++i) {
                const size_t src = (size_t)t * (size_t)n_selected + (size_t)i;
                const int32_t union_local = ids[src];
                if (union_local < 0 ||
                    (size_t) union_local >= layer_rt.route_by_union_local.size()) {
                    return false;
                }
                const MoeMultiTargetLayerRoute & route =
                    layer_rt.route_by_union_local[(size_t) union_local];
                if (route.target_slot < 0 ||
                    (size_t) route.target_slot >= target_count ||
                    route.target_local < 0) {
                    return false;
                }
                auto & counts = batch_target_counts_[(size_t)route.target_slot];
                if (counts[(size_t)t] == 0) {
                    scatter_touched_targets_.push_back(route.target_slot);
                }
                const int offset = counts[(size_t)t]++;
                if (offset < 0 || offset >= n_selected) return false;
                const size_t dst =
                    (size_t)t * (size_t)n_selected + (size_t)offset;
                batch_target_ids_[(size_t)route.target_slot][dst] =
                    route.target_local;
                batch_target_weights_[(size_t)route.target_slot][dst] =
                    weights[src];
                int & max_selected =
                    batch_target_max_selected_[(size_t)route.target_slot];
                if (offset + 1 > max_selected) {
                    max_selected = offset + 1;
                }
            }
            for (int target_slot : scatter_touched_targets_) {
                auto & active_selected_counts =
                    batch_active_selected_counts_[(size_t)target_slot];
                const int count =
                    batch_target_counts_[(size_t)target_slot][(size_t)t];
                auto pos = std::lower_bound(
                    active_selected_counts.begin(),
                    active_selected_counts.end(),
                    count);
                if (pos == active_selected_counts.end() || *pos != count) {
                    active_selected_counts.insert(pos, count);
                }
                auto & token_group =
                    batch_token_groups_[(size_t)target_slot][(size_t)count];
                if (token_group.empty() &&
                    token_group.capacity() < (size_t)n_tokens) {
                    token_group.reserve((size_t)n_tokens);
                }
                token_group.push_back(t);
            }
        }
        return true;
    }

    bool dispatch_target_batch(MoeExpertCompute & compute,
                               const MoeExpertLayer & target_layer,
                               const float * input,
                               const std::vector<int32_t> & dense_ids,
                               const std::vector<float> & dense_weights,
                               const std::vector<int> & token_group,
                               int n_selected_stride,
                               int n_target,
                               int n_embd,
                               int n_ff,
                               float * output,
                               std::vector<uint8_t> & output_written) {
        const int tc = (int)token_group.size();
        if (tc <= 0) return true;

        group_ids_.resize((size_t)tc * (size_t)n_target);
        group_weights_.resize((size_t)tc * (size_t)n_target);
        for (int gi = 0; gi < tc; ++gi) {
            const int t = token_group[(size_t)gi];
            const size_t src_base =
                (size_t)t * (size_t)n_selected_stride;
            const size_t dst_base =
                (size_t)gi * (size_t)n_target;
            for (int i = 0; i < n_target; ++i) {
                group_ids_[dst_base + (size_t)i] =
                    dense_ids[src_base + (size_t)i];
                group_weights_[dst_base + (size_t)i] =
                    dense_weights[src_base + (size_t)i];
            }
        }

        bool contiguous_tokens = true;
        for (int gi = 1; gi < tc; ++gi) {
            if (token_group[(size_t)gi] != token_group[0] + gi) {
                contiguous_tokens = false;
                break;
            }
        }
        bool all_unwritten = true;
        for (int gi = 0; gi < tc; ++gi) {
            const int t = token_group[(size_t)gi];
            if (t < 0 || (size_t)t >= output_written.size()) return false;
            if (output_written[(size_t)t]) {
                all_unwritten = false;
            }
        }

        const float * compute_input = input;
        if (contiguous_tokens) {
            compute_input = input + (size_t)token_group[0] * (size_t)n_embd;
        } else {
            group_input_.resize((size_t)tc * (size_t)n_embd);
            for (int gi = 0; gi < tc; ++gi) {
                const int t = token_group[(size_t)gi];
                std::memcpy(group_input_.data() + (size_t)gi * (size_t)n_embd,
                            input + (size_t)t * (size_t)n_embd,
                            sizeof(float) * (size_t)n_embd);
            }
            compute_input = group_input_.data();
        }

        const bool direct_output = contiguous_tokens && all_unwritten;
        float * compute_output = direct_output
            ? output + (size_t)token_group[0] * (size_t)n_embd
            : nullptr;
        if (!direct_output) {
            group_output_.resize((size_t)tc * (size_t)n_embd);
            compute_output = group_output_.data();
        }
        if (!compute.compute_batch(target_layer, compute_input,
                                   group_ids_.data(), group_weights_.data(),
                                   tc, n_target, n_embd, n_ff,
                                   compute_output)) {
            return false;
        }

        if (direct_output) {
            for (int gi = 0; gi < tc; ++gi) {
                output_written[(size_t)token_group[(size_t)gi]] = 1;
            }
            return true;
        }

        for (int gi = 0; gi < tc; ++gi) {
            const int t = token_group[(size_t)gi];
            float * dst = output + (size_t)t * (size_t)n_embd;
            const float * src =
                group_output_.data() + (size_t)gi * (size_t)n_embd;
            if (output_written[(size_t)t]) {
                for (int i = 0; i < n_embd; ++i) {
                    dst[i] += src[i];
                }
            } else {
                std::memcpy(dst, src, sizeof(float) * (size_t)n_embd);
                output_written[(size_t)t] = 1;
            }
        }
        return true;
    }

    MoeMultiTargetExpertRuntime * runtime_ = nullptr;
    std::vector<std::vector<int32_t>> target_ids_;
    std::vector<std::vector<float>> target_weights_;
    std::vector<std::vector<float>> target_output_scratch_;
    std::vector<int> target_counts_;
    std::vector<std::vector<int32_t>> batch_target_ids_;
    std::vector<std::vector<float>> batch_target_weights_;
    std::vector<std::vector<int>> batch_target_counts_;
    std::vector<std::vector<std::vector<int>>> batch_token_groups_;
    std::vector<std::vector<int>> batch_active_selected_counts_;
    std::vector<int> batch_target_max_selected_;
    std::vector<uint8_t> batch_output_written_;
    std::vector<int> scatter_touched_targets_;
    std::vector<int32_t> group_ids_;
    std::vector<float> group_weights_;
    std::vector<float> group_input_;
    std::vector<float> group_output_;
};

class RemappedMoeExpertCompute final : public MoeExpertCompute {
public:
    RemappedMoeExpertCompute(std::unique_ptr<MoeExpertCompute> primary,
                             std::unique_ptr<MoeExpertCompute> fallback,
                             std::vector<MoeExpertLayer> fallback_layers,
                             std::vector<std::vector<int32_t>> base_local_by_target_local)
        : primary_(std::move(primary)),
          fallback_(std::move(fallback)),
          fallback_layers_(std::move(fallback_layers)),
          base_local_by_target_local_(std::move(base_local_by_target_local)) {}

    bool healthy() const override {
        return (primary_ && primary_->healthy()) ||
               (fallback_ && fallback_->healthy());
    }

    bool prefers_padded_batch() const override {
        return primary_ && primary_->prefers_padded_batch();
    }

    bool compute(const MoeExpertLayer & layer,
                 const float * input,
                 const int32_t * ids,
                 const float * weights,
                 int n_selected,
                 int n_embd,
                 int n_ff,
                 float * output) override {
        if (primary_ &&
            primary_->compute(layer, input, ids, weights,
                              n_selected, n_embd, n_ff, output)) {
            return true;
        }
        if (!fallback_) return false;
        if (!remap_ids(layer.layer_idx, ids, n_selected)) return false;
        const MoeExpertLayer * fallback_layer = fallback_layer_ptr(layer.layer_idx);
        if (!fallback_layer) return false;
        return fallback_->compute(*fallback_layer, input, remapped_ids_.data(), weights,
                                  n_selected, n_embd, n_ff, output);
    }

    bool compute_batch(const MoeExpertLayer & layer,
                       const float * input,
                       const int32_t * ids,
                       const float * weights,
                       int n_tokens,
                       int n_selected,
                       int n_embd,
                       int n_ff,
                       float * output) override {
        if (primary_ &&
            primary_->compute_batch(layer, input, ids, weights,
                                    n_tokens, n_selected, n_embd, n_ff, output)) {
            return true;
        }
        if (!fallback_) return false;
        if (!remap_ids_batch(layer.layer_idx, ids, n_tokens, n_selected)) return false;
        const MoeExpertLayer * fallback_layer = fallback_layer_ptr(layer.layer_idx);
        if (!fallback_layer) return false;
        return fallback_->compute_batch(*fallback_layer, input, remapped_ids_.data(), weights,
                                        n_tokens, n_selected, n_embd, n_ff, output);
    }

    bool prepare_batch(const MoeExpertLayer & layer,
                       int n_tokens,
                       int n_selected,
                       int n_embd,
                       int n_ff) override {
        if (primary_ && primary_->prepare_batch(layer, n_tokens, n_selected, n_embd, n_ff)) {
            return true;
        }
        const MoeExpertLayer * fallback_layer = fallback_layer_ptr(layer.layer_idx);
        return fallback_ && fallback_layer &&
               fallback_->prepare_batch(*fallback_layer, n_tokens, n_selected, n_embd, n_ff);
    }

    bool prepare_single(const MoeExpertLayer & layer,
                        int n_selected,
                        int n_embd,
                        int n_ff) override {
        if (primary_ && primary_->prepare_single(layer, n_selected, n_embd, n_ff)) {
            return true;
        }
        const MoeExpertLayer * fallback_layer = fallback_layer_ptr(layer.layer_idx);
        return fallback_ && fallback_layer &&
               fallback_->prepare_single(*fallback_layer, n_selected, n_embd, n_ff);
    }

private:
    const MoeExpertLayer * fallback_layer_ptr(int layer_idx) const {
        if (layer_idx < 0 || (size_t)layer_idx >= fallback_layers_.size()) {
            return nullptr;
        }
        return &fallback_layers_[(size_t)layer_idx];
    }

    bool remap_ids(int layer_idx,
                   const int32_t * ids,
                   int n_selected) {
        if (n_selected < 0 || (n_selected > 0 && !ids) ||
            layer_idx < 0 ||
            (size_t)layer_idx >= base_local_by_target_local_.size()) {
            return false;
        }
        const std::vector<int32_t> & lut = base_local_by_target_local_[(size_t)layer_idx];
        remapped_ids_.resize((size_t)n_selected);
        for (int i = 0; i < n_selected; ++i) {
            const int32_t local = ids[i];
            if (local < 0 || (size_t)local >= lut.size() || lut[(size_t)local] < 0) {
                return false;
            }
            remapped_ids_[(size_t)i] = lut[(size_t)local];
        }
        return true;
    }

    bool remap_ids_batch(int layer_idx,
                         const int32_t * ids,
                         int n_tokens,
                         int n_selected) {
        if (n_tokens < 0 || n_selected < 0 ||
            ((n_tokens > 0 && n_selected > 0) && !ids) ||
            layer_idx < 0 ||
            (size_t)layer_idx >= base_local_by_target_local_.size()) {
            return false;
        }
        const std::vector<int32_t> & lut = base_local_by_target_local_[(size_t)layer_idx];
        remapped_ids_.resize((size_t)n_tokens * (size_t)n_selected);
        for (int t = 0; t < n_tokens; ++t) {
            for (int i = 0; i < n_selected; ++i) {
                const size_t idx = (size_t)t * (size_t)n_selected + (size_t)i;
                const int32_t local = ids[idx];
                if (local < 0 || (size_t)local >= lut.size() || lut[(size_t)local] < 0) {
                    return false;
                }
                remapped_ids_[idx] = lut[(size_t)local];
            }
        }
        return true;
    }

    std::unique_ptr<MoeExpertCompute> primary_;
    std::unique_ptr<MoeExpertCompute> fallback_;
    std::vector<MoeExpertLayer> fallback_layers_;
    std::vector<std::vector<int32_t>> base_local_by_target_local_;
    std::vector<int32_t> remapped_ids_;
};

std::vector<MoeExpertLayer> build_union_target_layers(
        const MoeHybridStorage & hybrid,
        const ExpertSplitComputeRuntime & split_runtime,
        const std::vector<MoeMultiTargetExpertRuntimeTarget> & targets,
        std::vector<MoeMultiTargetLayerRuntime> & out_routes) {
    std::vector<MoeExpertLayer> layers(hybrid.layers.size());
    out_routes.assign(hybrid.layers.size(), {});
    for (size_t il = 0; il < hybrid.layers.size(); ++il) {
        MoeExpertLayer & layer = layers[(size_t)il];
        const MoeHybridLayerStorage & storage = hybrid.layers[il];
        layer.layer_idx = (int)il;
        layer.cold_global_by_local = storage.cold_expert_ids;
        out_routes[il].route_by_union_local.resize(layer.cold_global_by_local.size());
        for (size_t union_local = 0; union_local < layer.cold_global_by_local.size(); ++union_local) {
            MoeMultiTargetLayerRoute & route = out_routes[il].route_by_union_local[union_local];
            route.target_slot = -1;
            route.target_local = -1;
            route.union_local = (int)union_local;
        }
    }

    for (size_t il = 0; il < hybrid.layers.size(); ++il) {
        const MoeHybridLayerStorage & storage = hybrid.layers[il];
        MoeMultiTargetLayerRuntime & layer_rt = out_routes[il];
        for (size_t union_local = 0; union_local < storage.cold_expert_ids.size(); ++union_local) {
            const int32_t global = storage.cold_expert_ids[union_local];
            if ((int)il >= split_runtime.n_layer || global < 0 || global >= split_runtime.n_expert) {
                continue;
            }
            const int target_slot = split_runtime.target_index((int)il, global);
            const int target_local = split_runtime.local_index((int)il, global);
            if (target_slot < 0 || target_local < 0 ||
                (size_t)target_slot >= targets.size()) {
                continue;
            }
            MoeMultiTargetLayerRoute route;
            route.target_slot = target_slot;
            route.target_local = target_local;
            route.union_local = (int)union_local;
            layer_rt.route_by_union_local[union_local] = route;
        }
    }
    return layers;
}

bool parse_target_suffix_device(const std::string & name,
                                const std::string & prefix,
                                int & out) {
    if (name.size() <= prefix.size() + 1) return false;
    if (name.compare(0, prefix.size(), prefix) != 0 ||
        name[prefix.size()] != ':') {
        return false;
    }
    char * end = nullptr;
    long value = std::strtol(name.c_str() + prefix.size() + 1, &end, 10);
    if (end == name.c_str() + prefix.size() + 1 || *end != '\0' || value < 0 ||
        value > std::numeric_limits<int>::max()) {
        return false;
    }
    out = (int)value;
    return true;
}

bool build_target_runtime_from_split(
    MoeMultiTargetExpertRuntimeTarget & out,
    const MoeExpertComputeRuntimeConfig & cfg,
    const ExpertSplitComputeTargetRuntime & split_target,
    const MoeHybridStorage & hybrid,
    const std::vector<MoeLayerDesc> & layer_descs,
    std::string * err) {
    out.target_index = split_target.target_index;
    out.target = split_target.target;
    out.placement = split_target.placement;

    MoeExpertComputeRuntimeConfig target_cfg = cfg;
    const bool is_primary = split_target.target_index == 0;
    const bool is_cpu_target = split_target.target.backend == "cpu";
    out.compute_active =
        cfg.enabled && !is_primary && split_target.placement.total_hot > 0;
    target_cfg.enabled = out.compute_active;

    if (!target_cfg.enabled) {
        out.runtime.reset();
        if (is_primary && split_target.placement.total_hot > 0) {
            std::fprintf(stderr,
                         "%s multi-target target_index=%d name=%s backend=%s "
                         "placement_hot=%d compute=primary-local-skip\n",
                         cfg.log_prefix ? cfg.log_prefix : "[moe-expert-compute]",
                         split_target.target_index,
                         split_target.target.name.c_str(),
                         split_target.target.backend.c_str(),
                         split_target.placement.total_hot);
        }
        return true;
    }

    const char * ipc_bin = nonempty_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_BIN");
    const bool remote_required =
        parse_nonnegative_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_REQUIRED", 0) != 0;
    bool started_remote = false;
    std::unique_ptr<MoeExpertCompute> primary_compute;
    if (ipc_bin && !is_cpu_target) {
        if (!validate_executable_file(ipc_bin, err)) {
            return false;
        }

        int target_gpu = 0;
        if (const char * gpu_env = nonempty_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_GPU")) {
            target_gpu = parse_nonnegative_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_GPU", 0);
            (void)gpu_env;
        }
        if (!split_target.target.backend.empty()) {
            if (split_target.target.backend == "cuda") {
                (void) parse_target_suffix_device(split_target.target.name, "cuda", target_gpu);
            } else if (split_target.target.backend == "hip") {
                (void) parse_target_suffix_device(split_target.target.name, "hip", target_gpu);
            }
        }

        const char * work_dir = nonempty_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_WORK_DIR");
        std::fprintf(stderr,
                     "%s spawning multi-target expert IPC target_index=%d "
                     "name=%s backend=%s device=%d placement_hot=%d "
                     "required=%d reason=secondary-non-cpu\n",
                     cfg.log_prefix ? cfg.log_prefix : "[moe-expert-compute]",
                     split_target.target_index,
                     split_target.target.name.c_str(),
                     split_target.target.backend.c_str(),
                     target_gpu,
                     split_target.placement.total_hot,
                     remote_required ? 1 : 0);
        MoeExpertComputeIpcStartResult remote =
            make_moe_expert_compute_ipc_for_placement(
                ipc_bin,
                cfg.target_path,
                target_gpu,
                split_target.placement,
                cfg.n_embd,
                cfg.n_ff_exp,
                cfg.n_expert_used,
                work_dir ? work_dir : "",
                remote_required);
        if (remote_required && !remote.started_remote) {
            if (err) *err = "required multi-target expert IPC did not start";
            return false;
        }
        started_remote = remote.started_remote;
        if (remote.started_remote) {
            primary_compute = std::move(remote.compute);
        }
    } else if (is_cpu_target && split_target.placement.total_hot > 0) {
        std::fprintf(stderr,
                     "%s multi-target target_index=%d name=%s backend=cpu "
                     "placement_hot=%d compute=cpu-local\n",
                     cfg.log_prefix ? cfg.log_prefix : "[moe-expert-compute]",
                     split_target.target_index,
                     split_target.target.name.c_str(),
                     split_target.placement.total_hot);
    }

    std::unique_ptr<MoeExpertCompute> fallback_compute;
    std::vector<MoeExpertLayer> fallback_layers;
    std::vector<std::vector<int32_t>> base_local_by_target_local;
    const bool need_cpu_fallback = is_cpu_target || !started_remote || !remote_required;
    if (need_cpu_fallback) {
        fallback_compute = make_cpu_moe_expert_compute(cfg.n_ff_exp);
        fallback_layers = make_moe_expert_layers(hybrid, layer_descs);
        base_local_by_target_local.resize((size_t)cfg.n_layer);
        for (int il = 0; il < cfg.n_layer; ++il) {
            const MoeExpertLayer & base_layer = fallback_layers[(size_t)il];
            const std::vector<int32_t> & globals =
                split_target.placement.hot_expert_ids[(size_t)il];
            std::vector<int32_t> remap(globals.size(), -1);
            for (size_t local = 0; local < globals.size(); ++local) {
                const int32_t global = globals[local];
                if (global < 0 ||
                    (size_t)global >= hybrid.layers[(size_t)il].cold_local_by_global.size()) {
                    continue;
                }
                const int32_t base_local =
                    hybrid.layers[(size_t)il].cold_local_by_global[(size_t)global];
                if (base_local < 0 ||
                    (size_t)base_local >= base_layer.cold_global_by_local.size() ||
                    base_layer.cold_global_by_local[(size_t)base_local] != global) {
                    continue;
                }
                remap[local] = base_local;
            }
            base_local_by_target_local[(size_t)il] = std::move(remap);
        }
    }

    out.runtime.compute = std::make_unique<RemappedMoeExpertCompute>(
        std::move(primary_compute),
        std::move(fallback_compute),
        std::move(fallback_layers),
        std::move(base_local_by_target_local));
    out.runtime.layers.resize((size_t)cfg.n_layer);
    for (int il = 0; il < cfg.n_layer; ++il) {
        MoeExpertLayer & layer = out.runtime.layers[(size_t)il];
        layer.layer_idx = il;
        const auto & globals = split_target.placement.hot_expert_ids[(size_t)il];
        layer.cold_global_by_local.assign(globals.begin(), globals.end());
    }
    out.runtime.target_path = cfg.target_path;
    out.runtime.runtime_key =
        split_target.target.name + (started_remote ? "|ipc" : "|remapped");
    out.runtime.placement_fingerprint =
        moe_expert_placement_fingerprint(hybrid, cfg.n_layer, cfg.n_expert,
                                         cfg.n_expert_used) ^
        (uint64_t)(split_target.target_index + 1);
    out.runtime.remote_started = started_remote;
    return out.runtime.compute_ptr() != nullptr;
}

}  // namespace

int moe_expert_compute_prepare_batch_limit_from_env() {
    const char * raw = std::getenv("DFLASH_MOE_EXPERT_COMPUTE_PREPARE_BATCH");
    if (!raw || !*raw) return 0;
    char * end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (end == raw || value <= 0) return 0;
    if (value > 4096) return 4096;
    return (int)value;
}

int moe_expert_compute_prepare_selected_limit_from_env() {
    const char * raw = std::getenv("DFLASH_MOE_EXPERT_COMPUTE_PREPARE_SELECTED");
    if (!raw || !*raw) return 0;
    char * end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (end == raw || value <= 0) return 0;
    if (value > 4096) return 4096;
    return (int)value;
}

int moe_expert_compute_batch_limit_from_env() {
    const char * raw = std::getenv("DFLASH_MOE_EXPERT_COMPUTE_BATCH");
    if (!raw || !*raw) return 32;
    char * end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (end == raw || value <= 0) return 32;
    if (value > 4096) return 4096;
    return (int)value;
}

int moe_expert_compute_daemon_batch_limit_from_env() {
    const char * raw = std::getenv("DFLASH_MOE_EXPERT_COMPUTE_DAEMON_BATCH");
    if (!raw || !*raw) return 8;
    char * end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (end == raw || value <= 0) return 8;
    if (value > 4096) return 4096;
    return (int)value;
}

std::unique_ptr<MoeExpertCompute> make_multi_target_moe_expert_compute(
    MoeMultiTargetExpertRuntime * runtime) {
    return std::make_unique<MultiTargetMoeExpertCompute>(runtime);
}

uint64_t moe_expert_placement_fingerprint(const MoeHybridStorage & hybrid,
                                         int n_layer,
                                         int n_expert,
                                         int n_expert_used) {
    uint64_t h = 1469598103934665603ULL;
    h = hash_u64(h, (uint64_t)n_layer);
    h = hash_u64(h, (uint64_t)n_expert);
    h = hash_u64(h, (uint64_t)n_expert_used);
    h = hash_u64(h, (uint64_t)hybrid.placement.total_hot);
    for (size_t il = 0; il < hybrid.placement.hot_expert_ids.size(); ++il) {
        h = hash_u64(h, (uint64_t)il);
        for (int32_t expert : hybrid.placement.hot_expert_ids[il]) {
            h = hash_u64(h, (uint64_t)(uint32_t)expert);
        }
    }
    return h;
}

std::vector<MoeExpertLayer> make_moe_expert_layers(
    const MoeHybridStorage & hybrid,
    const std::vector<MoeLayerDesc> & layer_descs) {
    std::vector<MoeExpertLayer> layers(hybrid.layers.size());
    for (size_t il = 0; il < hybrid.layers.size(); ++il) {
        const auto & storage = hybrid.layers[il];
        const MoeLayerDesc * desc =
            il < layer_descs.size() ? &layer_descs[il] : nullptr;
        auto & cl = layers[il];
        cl.layer_idx = (int)il;
        cl.cold_global_by_local = storage.cold_expert_ids;
        cl.fused_gate_up = (storage.gate_up_cold != nullptr);
        if (cl.fused_gate_up) {
            cl.gate_up_data =
                storage.gate_up_cold ? storage.gate_up_cold->data : nullptr;
            cl.gate_up_stride =
                storage.gate_up_cold ? storage.gate_up_cold->nb[2] : 0;
            cl.gate_up_type =
                storage.gate_up_cold ? storage.gate_up_cold->type : GGML_TYPE_Q4_K;
            cl.gate_up_scale = desc ? desc->ffn_gate_up_exps_s : 1.0f;
        } else {
            cl.gate_data = storage.gate_cold ? storage.gate_cold->data : nullptr;
            cl.up_data = storage.up_cold ? storage.up_cold->data : nullptr;
            cl.gate_stride = storage.gate_cold ? storage.gate_cold->nb[2] : 0;
            cl.up_stride = storage.up_cold ? storage.up_cold->nb[2] : 0;
            cl.gate_type =
                storage.gate_cold ? storage.gate_cold->type : GGML_TYPE_Q4_K;
            cl.up_type =
                storage.up_cold ? storage.up_cold->type : GGML_TYPE_Q4_K;
            cl.gate_scale = desc ? desc->ffn_gate_exps_s : 1.0f;
            cl.up_scale = desc ? desc->ffn_up_exps_s : 1.0f;
        }
        cl.down_data = storage.down_cold ? storage.down_cold->data : nullptr;
        cl.down_stride = storage.down_cold ? storage.down_cold->nb[2] : 0;
        cl.down_type =
            storage.down_cold ? storage.down_cold->type : GGML_TYPE_Q4_K;
        cl.down_scale = desc ? desc->ffn_down_exps_s : 1.0f;
    }
    return layers;
}

void MoeExpertComputeRuntime::reset() {
    compute.reset();
    layers.clear();
    target_path.clear();
    runtime_key.clear();
    placement_fingerprint = 0;
    remote_started = false;
}

void MoeMultiTargetExpertRuntime::reset() {
    compute.reset();
    layers.clear();
    targets.clear();
    layer_routes.clear();
    runtime_key.clear();
    placement_fingerprint = 0;
    enabled = false;
}

bool ensure_moe_expert_compute_runtime(
    MoeExpertComputeRuntime & runtime,
    const MoeExpertComputeRuntimeConfig & cfg,
    const MoeHybridStorage & hybrid,
    const std::vector<MoeLayerDesc> & layer_descs,
    std::string * err) {
    if (!cfg.enabled) {
        runtime.reset();
        return true;
    }
    if (cfg.n_layer <= 0 || cfg.n_expert <= 0 || cfg.n_expert_used <= 0 ||
        cfg.n_embd <= 0 || cfg.n_ff_exp <= 0) {
        if (err) *err = "invalid MoE expert compute runtime config";
        runtime.reset();
        return false;
    }

    const uint64_t fingerprint =
        moe_expert_placement_fingerprint(hybrid, cfg.n_layer, cfg.n_expert,
                                         cfg.n_expert_used);
    const std::string runtime_key = make_runtime_key(cfg);
    const bool can_reuse =
        runtime.compute &&
        runtime.compute->healthy() &&
        runtime.runtime_key == runtime_key &&
        runtime.placement_fingerprint == fingerprint;
    if (!can_reuse) {
        runtime.compute.reset();
        runtime.target_path.clear();
        runtime.runtime_key.clear();
        runtime.placement_fingerprint = 0;
        runtime.remote_started = false;
    }

    if (!runtime.compute) {
        bool started_remote = false;
        if (const char * ipc_bin = nonempty_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_BIN")) {
            if (!validate_executable_file(ipc_bin, err)) {
                std::fprintf(stderr, "%s %s\n", cfg.log_prefix ? cfg.log_prefix : "[moe-expert-compute]",
                             err ? err->c_str() : "invalid remote IPC binary");
                runtime.reset();
                return false;
            }
            const char * work_dir = nonempty_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_WORK_DIR");
            const int remote_gpu =
                parse_nonnegative_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_GPU", 0);
            const bool required =
                parse_nonnegative_env("DFLASH_MOE_EXPERT_COMPUTE_IPC_REQUIRED", 0) != 0;
            MoeExpertComputeIpcStartResult remote = make_moe_expert_compute_ipc(
                ipc_bin, cfg.target_path, remote_gpu, hybrid.placement,
                cfg.n_embd, cfg.n_ff_exp, cfg.n_expert_used,
                work_dir ? work_dir : "", required);
            if (required && !remote.started_remote) {
                if (err) *err = "remote MoE expert compute IPC is required but did not start";
                std::fprintf(stderr, "%s %s\n", cfg.log_prefix ? cfg.log_prefix : "[moe-expert-compute]",
                             err ? err->c_str() : "remote IPC did not start");
                runtime.reset();
                return false;
            }
            started_remote = remote.started_remote;
            runtime.compute = std::move(remote.compute);
        }
        if (!runtime.compute) {
            runtime.compute = make_cpu_moe_expert_compute(cfg.n_ff_exp);
        }
        runtime.remote_started = started_remote;
    }

    runtime.layers = make_moe_expert_layers(hybrid, layer_descs);
    runtime.target_path = cfg.target_path;
    runtime.runtime_key = runtime_key;
    runtime.placement_fingerprint = fingerprint;

    if (runtime.remote_started && runtime.compute && runtime.compute->healthy()) {
        int prepared = 0;
        int failed = 0;
        int prepare_tokens = moe_expert_compute_prepare_batch_limit_from_env();
        if (prepare_tokens > 0) {
            const bool prepare_full_selected_only = []() {
                const char * raw = std::getenv("DFLASH_MOE_EXPERT_COMPUTE_PREPARE_FULL_SELECTED_ONLY");
                return raw && *raw && std::strcmp(raw, "0") != 0 &&
                    std::strcmp(raw, "false") != 0 && std::strcmp(raw, "off") != 0;
            }();
            const int prepare_selected_limit =
                moe_expert_compute_prepare_selected_limit_from_env();
            for (const MoeExpertLayer & layer : runtime.layers) {
                if (layer.layer_idx < 0 || layer.cold_global_by_local.empty()) continue;
                const int max_selected = std::min<int>(
                    cfg.n_expert_used, (int)layer.cold_global_by_local.size());
                const int first_selected = prepare_full_selected_only
                    ? max_selected
                    : 1;
                const int last_selected = prepare_full_selected_only
                    ? max_selected
                    : (prepare_selected_limit > 0
                        ? std::min(max_selected, prepare_selected_limit)
                        : max_selected);
                for (int n_selected = 1; n_selected <= max_selected; ++n_selected) {
                    if (n_selected < first_selected || n_selected > last_selected) {
                        continue;
                    }
                    const bool prepare_single =
                        runtime.compute->prepare_single(layer, n_selected,
                                                        cfg.n_embd, cfg.n_ff_exp);
                    const bool prepare_prefill =
                        prepare_tokens == 1 ? true : runtime.compute->prepare_batch(
                            layer, prepare_tokens, n_selected, cfg.n_embd,
                            cfg.n_ff_exp);
                    if (prepare_single) ++prepared; else ++failed;
                    if (prepare_tokens != 1) {
                        if (prepare_prefill) ++prepared; else ++failed;
                    }
                }
            }
        }
        if (prepared > 0 || failed > 0) {
            std::fprintf(stderr,
                         "%s prepared remote MoE graphs=%d failed=%d batch=%d\n",
                         cfg.log_prefix ? cfg.log_prefix : "[moe-expert-compute]",
                         prepared, failed, prepare_tokens);
        }
    }
    return true;
}

bool ensure_multi_target_moe_expert_compute_runtime(
    MoeMultiTargetExpertRuntime & runtime,
    const MoeExpertComputeRuntimeConfig & cfg,
    const ExpertSplitComputeRuntime & split_runtime,
    const MoeHybridStorage & hybrid,
    const std::vector<MoeLayerDesc> & layer_descs,
    std::string * err) {
    if (!cfg.enabled) {
        runtime.reset();
        return true;
    }
    if (!split_runtime.matches(cfg.n_layer, cfg.n_expert, cfg.n_expert_used) ||
        split_runtime.targets.empty()) {
        if (err) *err = "expert split compute runtime not initialized";
        runtime.reset();
        return false;
    }
    if (split_runtime.targets.size() <= 1) {
        runtime.reset();
        return true;
    }

    const std::string runtime_key =
        make_multi_target_runtime_key(cfg, split_runtime);
    const uint64_t fingerprint =
        moe_expert_placement_fingerprint(hybrid, cfg.n_layer, cfg.n_expert,
                                         cfg.n_expert_used);
    int explicit_non_cpu_targets = 0;
    for (const auto & target : split_runtime.targets) {
        if (target.target.backend != "cpu") {
            ++explicit_non_cpu_targets;
        }
    }
    if (explicit_non_cpu_targets <= 1) {
        runtime.reset();
        return true;
    }
    const bool can_reuse =
        runtime.enabled &&
        runtime.compute &&
        runtime.compute->healthy() &&
        runtime.runtime_key == runtime_key &&
        runtime.placement_fingerprint == fingerprint &&
        runtime.targets.size() == split_runtime.targets.size();
    if (!can_reuse) {
        runtime.reset();
    }

    if (!runtime.compute) {
        std::vector<MoeMultiTargetExpertRuntimeTarget> targets;
        targets.resize(split_runtime.targets.size());
        for (size_t i = 0; i < split_runtime.targets.size(); ++i) {
            if (!build_target_runtime_from_split(
                    targets[i], cfg, split_runtime.targets[i], hybrid, layer_descs, err)) {
                runtime.reset();
                return false;
            }
        }

        std::vector<MoeMultiTargetLayerRuntime> layer_routes;
        std::vector<MoeExpertLayer> union_layers =
            build_union_target_layers(hybrid, split_runtime, targets,
                                      layer_routes);
        runtime.targets = std::move(targets);
        runtime.layer_routes = std::move(layer_routes);
        runtime.layers = std::move(union_layers);
        runtime.runtime_key = runtime_key;
        runtime.placement_fingerprint = fingerprint;
        runtime.compute = make_multi_target_moe_expert_compute(&runtime);
        runtime.enabled = true;
        if (!runtime.compute->healthy()) {
            runtime.reset();
            if (err) *err = "multi-target expert compute runtime is unhealthy";
            return false;
        }
    }

    return true;
}

}  // namespace dflash::common
