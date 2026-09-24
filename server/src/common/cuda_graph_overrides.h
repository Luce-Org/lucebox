#pragma once

#include "ggml-cuda.h"

namespace luce::common {

// Thread-local CUDA/HIP graph controls used while executing a graph whose
// topology or backend metadata requires a temporary policy override. The
// setters return their prior state, which makes this scope safe to nest.
class ScopedCudaGraphOverrides {
public:
    explicit ScopedCudaGraphOverrides(
            bool disable_graphs = false,
            int mmvq_max_ncols = 0,
            bool skip_property_check = false,
            int ds4_mix_mmv_max_tokens = 0,
            bool mmvq_batch_invariant = false)
        : disable_graphs_(disable_graphs),
          override_mmvq_(mmvq_max_ncols > 0),
          skip_property_check_(skip_property_check),
          override_ds4_mix_(ds4_mix_mmv_max_tokens > 0),
          batch_invariant_(mmvq_batch_invariant) {
        if (batch_invariant_) {
            previous_batch_invariant_ =
                ggml_backend_cuda_set_mmvq_batch_invariant(true);
        }
        if (disable_graphs_) {
            previous_graphs_disabled_ =
                ggml_backend_cuda_set_graphs_disabled_override(true);
        }
        if (override_mmvq_) {
            previous_mmvq_max_ncols_ =
                ggml_backend_cuda_set_mmvq_max_ncols_override(
                    mmvq_max_ncols);
        }
        if (skip_property_check_) {
            previous_skip_property_check_ =
                ggml_backend_cuda_set_skip_props_check(true);
        }
        if (override_ds4_mix_) {
            previous_ds4_mix_max_tokens_ =
                ggml_backend_cuda_set_ds4_mix_mmv_max_tokens_override(ds4_mix_mmv_max_tokens);
        }
    }

    ~ScopedCudaGraphOverrides() {
        if (batch_invariant_) {
            ggml_backend_cuda_set_mmvq_batch_invariant(previous_batch_invariant_);
        }
        if (override_ds4_mix_) {
            ggml_backend_cuda_set_ds4_mix_mmv_max_tokens_override(previous_ds4_mix_max_tokens_);
        }
        if (skip_property_check_) {
            ggml_backend_cuda_set_skip_props_check(
                previous_skip_property_check_);
        }
        if (override_mmvq_) {
            ggml_backend_cuda_set_mmvq_max_ncols_override(
                previous_mmvq_max_ncols_);
        }
        if (disable_graphs_) {
            ggml_backend_cuda_set_graphs_disabled_override(
                previous_graphs_disabled_);
        }
    }

    ScopedCudaGraphOverrides(const ScopedCudaGraphOverrides &) = delete;
    ScopedCudaGraphOverrides & operator=(const ScopedCudaGraphOverrides &) = delete;

private:
    bool disable_graphs_ = false;
    bool override_mmvq_ = false;
    bool skip_property_check_ = false;
    bool override_ds4_mix_ = false;
    bool batch_invariant_ = false;
    bool previous_graphs_disabled_ = false;
    bool previous_batch_invariant_ = false;
    bool previous_skip_property_check_ = false;
    int previous_mmvq_max_ncols_ = 0;
    int previous_ds4_mix_max_tokens_ = 0;
};

}  // namespace luce::common
