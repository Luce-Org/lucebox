#include "deepseek4_image_admission.h"

#include "deepseek4_internal.h"
#include "common/moe_hybrid_placement.h"
#include "common/moe_hybrid_types.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace dflash::vision {
namespace {
constexpr uint64_t MAX = std::numeric_limits<uint64_t>::max();

bool fail(std::string & error, const std::string & message) {
    error = message;
    return false;
}

bool add(uint64_t & sum, uint64_t value) {
    if (value > MAX - sum) return false;
    sum += value;
    return true;
}

bool mul(uint64_t a, uint64_t b, uint64_t & result) {
    if (a && b > MAX / a) return false;
    result = a * b;
    return true;
}

bool aligned(uint64_t value, uint64_t alignment, uint64_t & result) {
    if (!alignment) return false;
    result = value;
    return add(result, (alignment - value % alignment) % alignment);
}

bool enabled(const char * name) {
    const char * raw = std::getenv(name);
    return raw && *raw && std::strcmp(raw, "0") != 0;
}

bool gpu_device(ggml_backend_t backend) {
    if (!backend || !ggml_backend_get_device(backend)) return false;
    const auto type = ggml_backend_dev_type(ggml_backend_get_device(backend));
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

bool selected_allocation(const ggml_tensor & source, uint64_t count,
                         ggml_backend_buffer_type_t buft,
                         uint64_t & payload, uint64_t & allocation,
                         std::string & error) {
    payload = allocation = 0;
    if (!count) return true;
    if (!mul(uint64_t(source.nb[2]), count, payload) || payload > SIZE_MAX) {
        return fail(error, "selected expert tensor payload overflow");
    }
    // Storage recreates each owner tensor contiguously with only ne[2] changed.
    // The metadata copy avoids allocating even a temporary GGML context.
    ggml_tensor selected = source;
    selected.ne[2] = int64_t(count);
    selected.ne[3] = 1;
    selected.nb[3] = size_t(payload);
    selected.data = nullptr;
    selected.buffer = nullptr;
    selected.view_src = nullptr;
    selected.view_offs = 0;
    const uint64_t requested = ggml_backend_buft_get_alloc_size(buft, &selected);
    if (requested < payload ||
        !aligned(requested, ggml_backend_buft_get_alignment(buft), allocation) ||
        allocation > ggml_backend_buft_get_max_size(buft)) {
        return fail(error, "selected expert tensor allocation size is invalid or exceeds backend limit");
    }
    return true;
}

bool owner_activation_estimate(const common::MoeHybridConfig & config, int tokens,
                               uint64_t alignment, uint64_t & bytes, std::string & error) {
    if (tokens <= 0 || tokens > 1024 || config.n_embd <= 0 ||
        config.n_ff_exp <= 0 || config.n_ff_shexp < 0 || config.n_expert_used <= 0) {
        return fail(error, "image owner estimate requires positive dimensions and chunk capacity at most 1024");
    }
    const uint64_t d = uint64_t(config.n_embd);
    const uint64_t f = uint64_t(config.n_ff_exp);
    const uint64_t shared_f = uint64_t(config.n_ff_shexp);
    uint64_t pairs = 0, routed_width = 0, shared_width = 0, part = 0, elements = 0;
    // Sum materialized tensors without lifetime reuse in the expert-major graph:
    // packed input/output, matmul/scaled output, route gather (5*D per route);
    // gate/up, optional scales/contiguous copies/clamps and GLU (<=9*F).
    // Input/shared branch/reduction contributes <=4*D+9*shared_F per token.
    // This intentionally leaves additional arena/pool/cached-graph headroom to
    // the named reservation rather than inventing a backend-wide upper bound.
    if (!mul(uint64_t(tokens), uint64_t(config.n_expert_used), pairs) ||
        !mul(5, d, routed_width) || !mul(9, f, part) || !add(routed_width, part) ||
        !mul(4, d, shared_width) || !mul(9, shared_f, part) || !add(shared_width, part) ||
        !mul(routed_width, pairs, elements) || !mul(shared_width, uint64_t(tokens), part) ||
        !add(elements, part) || !mul(elements, sizeof(float), bytes) ||
        !mul(12, pairs, part) || !add(bytes, part) ||
        !mul(16384, alignment, part) || !add(bytes, part)) {
        return fail(error, "owner activation estimate overflow");
    }
    return true;
}

bool host_available(uint64_t & bytes, std::string & error) {
#if defined(__linux__)
    std::ifstream input("/proc/meminfo");
    if (!input) return fail(error, "cannot read host MemAvailable");
    std::string line;
    bool found = false;
    while (std::getline(input, line)) {
        if (line.compare(0, 13, "MemAvailable:") != 0) continue;
        if (found) return fail(error, "duplicate host MemAvailable field");
        std::istringstream fields(line.substr(13));
        uint64_t kb = 0;
        std::string unit, extra;
        if (!(fields >> kb >> unit) || unit != "kB" || (fields >> extra) ||
            !mul(kb, 1024, bytes)) return fail(error, "invalid host MemAvailable");
        found = true;
    }
    return found || fail(error, "host MemAvailable is missing");
#else
    (void) bytes;
    return fail(error, "host admission currently requires Linux MemAvailable");
#endif
}

bool device_free(ggml_backend_t backend, uint64_t & available, std::string & error) {
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(backend), &free, &total);
    if (!total) return fail(error, "device memory query returned no capacity");
    // GGML's UMA query can legitimately report free RAM above dedicated total.
    available = free;
    return true;
}
} // namespace

bool estimate_deepseek4_image_storage(
    const common::DeepSeek4Weights & w, const common::MoeHybridPlacement & placement,
    const common::MoeHybridConfig & config, ggml_backend_t primary, ggml_backend_t cold,
    bool duplicate, ImageStorageEstimate & out, std::string & error) {
    out = {};
    error.clear();
    if (!gpu_device(primary) || !gpu_device(cold) ||
        ggml_backend_get_device(primary) == ggml_backend_get_device(cold)) {
        return fail(error, "image storage admission requires two distinct GPU owners");
    }
    if (config.cold_expert_backend != common::MoeHybridColdBackend::Gpu ||
        !config.materialize_hot_experts || !config.materialize_cold_experts ||
        w.n_expert <= 0 || w.n_layer <= 0 || size_t(w.n_layer) != w.layers.size() ||
        config.n_layer != w.n_layer || config.n_expert != w.n_expert ||
        config.n_embd != w.n_embd || config.n_ff_exp != w.n_ff_exp ||
        config.n_expert_used != w.n_expert_used || !placement.matches(config) ||
        !placement.valid(&error)) {
        return fail(error, "image storage admission requires valid materialized DS4 placement");
    }
    if (duplicate != enabled("DFLASH_MOE_DUPLICATE_HOT_ON_COLD")) {
        return fail(error, "image admission duplicate mode differs from storage environment");
    }
    const auto hot_buft = ggml_backend_get_default_buffer_type(primary);
    const auto cold_buft = ggml_backend_get_default_buffer_type(cold);
    if (!hot_buft || !cold_buft) return fail(error, "owner buffer type is missing");
    for (int layer_index = 0; layer_index < w.n_layer; ++layer_index) {
        const auto & layer = w.layers[size_t(layer_index)];
        const uint64_t hot_count = uint64_t(placement.hot_counts[size_t(layer_index)]);
        const uint64_t cold_count = duplicate ? uint64_t(w.n_expert) : uint64_t(w.n_expert) - hot_count;
        if (enabled("DFLASH_DS4_DECODE_ALL_COLD") && cold_count != uint64_t(w.n_expert)) {
            return fail(error, "decode-all-cold requires a full physical cold stack before allocation");
        }
        if ((layer.ffn_gate_shexp && layer.ffn_gate_shexp->ne[1] > config.n_ff_shexp) ||
            (layer.ffn_up_shexp && layer.ffn_up_shexp->ne[1] > config.n_ff_shexp)) {
            return fail(error, "shared expert metadata exceeds the configured activation width");
        }
        uint64_t hot_layer = 0, cold_layer = 0;
        uint64_t hot_buffer = 0, cold_buffer = 0;
        const auto count_buffer = [&](uint64_t allocation, uint64_t maximum,
                                      uint64_t & current, uint64_t & count) {
            if (current && allocation > maximum - current) {
                if (!add(count, 1)) return false;
                current = 0;
            }
            return add(current, allocation);
        };
        const std::array<const ggml_tensor *, 3> surfaces{
            layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps};
        for (const auto * tensor : surfaces) {
            if (!tensor) continue;
            if (tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] != w.n_expert ||
                tensor->ne[3] != 1 || tensor->nb[2] == 0 || !ggml_is_contiguous(tensor)) {
                return fail(error, "expert metadata must be positive contiguous [in,out,E,1]");
            }
            uint64_t full_payload = 0;
            if (!mul(uint64_t(tensor->nb[2]), uint64_t(w.n_expert), full_payload) ||
                full_payload > SIZE_MAX || full_payload != ggml_nbytes(tensor)) {
                return fail(error, "expert metadata payload overflow or stride mismatch");
            }
            uint64_t hot_payload = 0, hot_alloc = 0, cold_payload = 0, cold_alloc = 0;
            if (!selected_allocation(*tensor, hot_count, hot_buft, hot_payload, hot_alloc, error) ||
                !selected_allocation(*tensor, cold_count, cold_buft, cold_payload, cold_alloc, error)) return false;
            if (!add(out.hot_payload_bytes, hot_payload) || !add(out.cold_payload_bytes, cold_payload) ||
                !add(hot_layer, hot_alloc) || !add(cold_layer, cold_alloc)) {
                return fail(error, "expert owner storage sum overflow");
            }
            if (!count_buffer(hot_alloc, ggml_backend_buft_get_max_size(hot_buft),
                              hot_buffer, out.hot_buffer_count) ||
                !count_buffer(cold_alloc, ggml_backend_buft_get_max_size(cold_buft),
                              cold_buffer, out.cold_buffer_count)) {
                return fail(error, "owner split-buffer accounting overflow");
            }
            out.largest_copy_bytes = std::max({out.largest_copy_bytes, hot_payload, cold_payload});
            uint64_t per_expert_table = 0;
            if (tensor->type == GGML_TYPE_Q3_1_ROCMFP3_MIX) per_expert_table = 33;
            if (tensor->type == GGML_TYPE_Q2_1_ROCMFP2_MIX) per_expert_table = 17;
            if (per_expert_table) {
                uint64_t hot_table = 0, cold_table = 0, host_table = 0;
                if (!mul(per_expert_table, hot_count, hot_table) ||
                    !mul(per_expert_table, cold_count, cold_table) ||
                    !add(out.hot_mix_table_bytes, hot_table) || !add(out.cold_mix_table_bytes, cold_table) ||
                    !mul(per_expert_table + (per_expert_table == 33 ? 1 : 0),
                         uint64_t(w.n_expert), host_table) ||
                    !add(host_table, std::max(hot_table, cold_table)) ||
                    !add(host_table, uint64_t(w.n_expert))) {
                    return fail(error, "MIX table accounting overflow");
                }
                // File entry books/modes (+ P4 rotation bytes), compact selected
                // books/modes, and conservatively one byte per seen expert.
                out.host_mix_payload_peak_bytes = std::max(out.host_mix_payload_peak_bytes, host_table);
                if (!add(out.mix_device_allocation_count, 2 * uint64_t(hot_count > 0) +
                                                          2 * uint64_t(cold_count > 0))) {
                    return fail(error, "MIX allocation count overflow");
                }
            }
        }
        if (!add(out.hot_allocation_bytes, hot_layer) || !add(out.cold_allocation_bytes, cold_layer) ||
            !add(out.hot_buffer_count, uint64_t(hot_buffer > 0)) ||
            !add(out.cold_buffer_count, uint64_t(cold_buffer > 0))) {
            return fail(error, "owner allocation accounting overflow");
        }
    }
    if (!out.hot_payload_bytes && !out.cold_payload_bytes) return fail(error, "no expert storage metadata found");
    // The qualified Linux libstdc++ vector grows to at most old_size + max(old_size,
    // requested_delta). Retained old storage during growth adds at most another M.
    // Hot and cold staging vectors have disjoint scopes; layers are copied serially.
#if defined(__GLIBCXX__)
    if (!mul(out.largest_copy_bytes, 3, out.host_copy_peak_bytes)) {
        return fail(error, "host expert staging peak overflow");
    }
#else
    return fail(error, "host expert staging growth bound requires the qualified libstdc++ runtime");
#endif
    return true;
}

bool check_deepseek4_image_admission(
    const common::DeepSeek4Weights & w, const common::MoeHybridPlacement & placement,
    const common::MoeHybridConfig & config, ggml_backend_t primary, ggml_backend_t cold,
    const ImageAdmissionReserves & reserves, ImageAdmissionReport & out, std::string & error) {
    out = {};
    if (!estimate_deepseek4_image_storage(w, placement, config, primary, cold,
            reserves.duplicate_hot_on_cold, out.storage, error)) return false;
    out.storage_estimated = true;
    out.cold_runtime_reservation_bytes = reserves.cold_runtime_reservation_bytes;
    if (!owner_activation_estimate(config, reserves.max_chunk_tokens,
            ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(cold)),
            out.cold_activation_estimate_bytes, error)) return false;
    if (reserves.primary_domain == ImageMemoryDomain::Unknown ||
        reserves.cold_domain == ImageMemoryDomain::Unknown) {
        return fail(error, "actual owner host-memory sharing must be classified before admission");
    }
    if (!device_free(primary, out.primary_free_bytes, error) ||
        !device_free(cold, out.cold_free_bytes, error) ||
        !host_available(out.host_available_bytes, error)) return false;
    const ImageStorageEstimate storage = out.storage;
    const uint64_t activation_bytes = out.cold_activation_estimate_bytes;
    const ImageMemorySnapshot snapshot{out.primary_free_bytes, out.cold_free_bytes, out.host_available_bytes};
    return assess_deepseek4_image_admission(storage, activation_bytes, reserves, snapshot, out, error);
}

bool check_deepseek4_image_host_preparation(uint64_t required_bytes, std::string & error) {
    error.clear();
    if (!required_bytes) return fail(error, "image preparation requires a nonzero host reservation");
    uint64_t available = 0;
    if (!host_available(available, error)) return false;
    if (required_bytes > available) {
        return fail(error, "insufficient host memory for image preparation: required=" +
            std::to_string(required_bytes) + " available=" + std::to_string(available));
    }
    return true;
}

bool check_deepseek4_image_runtime_admission(
    const common::MoeHybridConfig & config, ggml_backend_t primary, ggml_backend_t cold,
    const ImageAdmissionReserves & reserves, ImageAdmissionReport & out, std::string & error) {
    out = {};
    error.clear();
    if (!gpu_device(primary) || !gpu_device(cold) ||
        ggml_backend_get_device(primary) == ggml_backend_get_device(cold)) {
        return fail(error, "image runtime admission requires two distinct GPU owners");
    }
    uint64_t activation_bytes = 0;
    const auto cold_buft = ggml_backend_get_default_buffer_type(cold);
    if (!cold_buft || !owner_activation_estimate(config, reserves.max_chunk_tokens,
            ggml_backend_buft_get_alignment(cold_buft), activation_bytes, error)) return false;
    ImageMemorySnapshot snapshot;
    if (!device_free(primary, snapshot.primary_free_bytes, error) ||
        !device_free(cold, snapshot.cold_free_bytes, error) ||
        !host_available(snapshot.host_available_bytes, error)) return false;
    ImageAdmissionReserves runtime_reserves = reserves;
    runtime_reserves.host_loader_overhead_bytes = 0;
    return assess_deepseek4_image_admission({}, activation_bytes, runtime_reserves, snapshot, out, error);
}

bool assess_deepseek4_image_admission(
    const ImageStorageEstimate & storage, uint64_t cold_activation_estimate_bytes,
    const ImageAdmissionReserves & reserves, const ImageMemorySnapshot & snapshot,
    ImageAdmissionReport & out, std::string & error) {
    const ImageStorageEstimate storage_copy = storage;
    out = {};
    error.clear();
    out.storage = storage_copy;
    out.storage_estimated = true;
    out.cold_activation_estimate_bytes = cold_activation_estimate_bytes;
    out.cold_runtime_reservation_bytes = reserves.cold_runtime_reservation_bytes;
    out.primary_free_bytes = snapshot.primary_free_bytes;
    out.cold_free_bytes = snapshot.cold_free_bytes;
    out.host_available_bytes = snapshot.host_available_bytes;
    const auto known_domain = [](ImageMemoryDomain domain) {
        return domain == ImageMemoryDomain::Dedicated || domain == ImageMemoryDomain::HostShared;
    };
    if (!known_domain(reserves.primary_domain) || !known_domain(reserves.cold_domain)) {
        return fail(error, "actual owner host-memory sharing must be classified before admission");
    }
    out.primary_required_bytes = out.storage.hot_allocation_bytes;
    out.cold_required_bytes = out.storage.cold_allocation_bytes;
    if (!add(out.primary_required_bytes, out.storage.hot_mix_table_bytes) ||
        !add(out.primary_required_bytes, reserves.primary_future_bytes) ||
        !add(out.cold_required_bytes, out.storage.cold_mix_table_bytes) ||
        !add(out.cold_required_bytes, reserves.cold_future_bytes) ||
        !add(out.cold_required_bytes, reserves.cold_runtime_reservation_bytes)) {
        return fail(error, "device admission charge overflow");
    }
    uint64_t loading = out.storage.host_copy_peak_bytes;
    if (!add(loading, out.storage.host_mix_payload_peak_bytes) ||
        !add(loading, reserves.host_loader_overhead_bytes)) {
        return fail(error, "host model-loading charge overflow");
    }
    // Loading and serving are separate phases. Conservatively keep runtime
    // host buffers alongside whichever phase has the larger temporary peak.
    out.host_required_bytes = std::max(loading, reserves.host_request_bytes);
    if (!add(out.host_required_bytes, reserves.host_runtime_bytes) ||
        (reserves.primary_domain == ImageMemoryDomain::HostShared &&
         !add(out.host_required_bytes, out.primary_required_bytes)) ||
        (reserves.cold_domain == ImageMemoryDomain::HostShared &&
         !add(out.host_required_bytes, out.cold_required_bytes))) {
        return fail(error, "combined host/UMA admission charge overflow");
    }
    out.known_charges_fit = out.primary_required_bytes <= out.primary_free_bytes &&
        out.cold_required_bytes <= out.cold_free_bytes && out.host_required_bytes <= out.host_available_bytes;
    if (!out.known_charges_fit) return fail(error, "insufficient primary, cold-owner, or combined host/UMA headroom");
    if (reserves.cold_runtime_reservation_bytes < out.cold_activation_estimate_bytes) {
        return fail(error, "cold-owner runtime reservation is below the metadata-derived activation estimate");
    }
    const bool loading_storage = storage_copy.hot_allocation_bytes || storage_copy.cold_allocation_bytes ||
        storage_copy.hot_mix_table_bytes || storage_copy.cold_mix_table_bytes ||
        storage_copy.host_copy_peak_bytes || storage_copy.host_mix_payload_peak_bytes;
    if (!reserves.host_request_bytes || (loading_storage && !reserves.host_loader_overhead_bytes)) {
        return fail(error, "request/decode and metadata/host-safety reservations are required");
    }
    return true;
}
} // namespace dflash::vision
