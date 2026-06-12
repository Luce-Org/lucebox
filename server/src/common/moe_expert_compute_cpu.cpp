// CpuMoeExpertCompute: fused selected expert FFN using ggml vec_dot primitives.
// Bypasses ggml graph dispatch overhead. Uses OpenMP to saturate memory bandwidth.
// Memory-bandwidth bound at ~45 GB/s DDR4. Target: 15.7ms → ~3ms/token.

#include "moe_expert_compute.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dflash::common {

class CpuMoeExpertCompute : public MoeExpertCompute {
    int n_ff_max_;
    int n_threads_;

    // Per-thread scratch buffers for parallel down matmul
    struct ThreadBuf {
        std::vector<float> scratch;   // [n_ff * 2] gate_up result + SwiGLU
        std::vector<uint8_t> mid_conv; // down input converted to vec_dot_type
    };
    std::vector<ThreadBuf> thread_bufs_;
    std::vector<uint8_t> inp_conv_;  // input converted (shared, read-only during matmul)

public:
    explicit CpuMoeExpertCompute(int n_ff_max, int n_threads = 0) : n_ff_max_(n_ff_max) {
#ifdef _OPENMP
        if (n_threads <= 0) {
            const char * env = std::getenv("DFLASH_MOE_EXPERT_COMPUTE_THREADS");
            n_threads = env ? std::atoi(env) : 0;
        }
        n_threads_ = n_threads > 0 ? n_threads : std::min(omp_get_max_threads(), 8);
#else
        n_threads_ = 1;
#endif
        fprintf(stderr, "[moe-expert-compute-cpu] using %d threads\n", n_threads_);
        thread_bufs_.resize(n_threads_);
        for (auto & tb : thread_bufs_) {
            tb.scratch.resize((size_t)n_ff_max * 2);
        }
    }

    bool compute(
        const MoeExpertLayer & layer,
        const float * input,
        const int32_t * ids,
        const float * weights,
        int n_cold,
        int n_embd,
        int n_ff,
        float * output) override {

        if (n_cold <= 0) return true;
        std::memset(output, 0, sizeof(float) * (size_t)n_embd);

        // Gate/up phase type traits
        const ggml_type gu_type = layer.fused_gate_up ? layer.gate_up_type : layer.gate_type;
        const auto * gu_cpu_traits = ggml_get_type_traits_cpu(gu_type);
        const auto gu_vec_dot = gu_cpu_traits->vec_dot;
        const auto gu_vec_dot_type = gu_cpu_traits->vec_dot_type;
        const auto gu_from_float = ggml_get_type_traits_cpu(gu_vec_dot_type)->from_float;

        // Down phase type traits (may differ from gate/up)
        const auto * dn_cpu_traits = ggml_get_type_traits_cpu(layer.down_type);
        const auto dn_vec_dot = dn_cpu_traits->vec_dot;
        const auto dn_vec_dot_type = dn_cpu_traits->vec_dot_type;
        const auto dn_from_float = ggml_get_type_traits_cpu(dn_vec_dot_type)->from_float;

        const size_t inp_row_size = ggml_row_size(gu_vec_dot_type, n_embd);
        const size_t mid_row_size = ggml_row_size(dn_vec_dot_type, n_ff);
        const size_t gu_weight_row = ggml_row_size(gu_type, n_embd);
        const size_t dn_weight_row = ggml_row_size(layer.down_type, n_ff);

        // For separate gate/up — up may have a different type than gate
        size_t up_weight_row = gu_weight_row;
        const ggml_type up_type_actual = layer.fused_gate_up ? gu_type : layer.up_type;
        (void)up_type_actual;
        ggml_vec_dot_t up_vec_dot = gu_vec_dot;
        ggml_type up_vdt = gu_vec_dot_type;
        if (!layer.fused_gate_up && layer.up_type != layer.gate_type) {
            const auto * up_cpu_traits = ggml_get_type_traits_cpu(layer.up_type);
            up_vec_dot = up_cpu_traits->vec_dot;
            up_vdt = up_cpu_traits->vec_dot_type;
            up_weight_row = ggml_row_size(layer.up_type, n_embd);
        }

        // Ensure input conversion buffer is large enough
        if (inp_conv_.size() < inp_row_size) inp_conv_.resize(inp_row_size);
        // Ensure per-thread mid_conv buffers
        for (auto & tb : thread_bufs_) {
            if (tb.mid_conv.size() < mid_row_size) tb.mid_conv.resize(mid_row_size);
        }

        // Convert input for up if different type
        std::vector<uint8_t> inp_conv_up;
        if (!layer.fused_gate_up && up_vdt != gu_vec_dot_type) {
            size_t up_inp_row_size = ggml_row_size(up_vdt, n_embd);
            inp_conv_up.resize(up_inp_row_size);
            auto up_from_float = ggml_get_type_traits_cpu(up_vdt)->from_float;
            up_from_float(input, inp_conv_up.data(), n_embd);
        }

        // Convert input to gate's vec_dot format once
        gu_from_float(input, inp_conv_.data(), n_embd);

        for (int e = 0; e < n_cold; ++e) {
            const int32_t eid = ids[e];
            const float w = weights[e];
            if (w == 0.0f) continue;

            // Use thread 0's scratch for gate_up (serial phase)
            float * scratch = thread_bufs_[0].scratch.data();

            // ── Phase 1: gate_up matmul → scratch[0..n_ff*2) ──
            // Parallel over rows (each row is independent, reading shared inp_conv_)
            if (layer.fused_gate_up) {
                const char * expert = (const char *)layer.gate_up_data + (size_t)eid * layer.gate_up_stride;
                const int n_rows = n_ff * 2;
#ifdef _OPENMP
                #pragma omp parallel for num_threads(n_threads_) schedule(static)
#endif
                for (int row = 0; row < n_rows; ++row) {
                    const void * row_ptr = expert + (size_t)row * gu_weight_row;
                    gu_vec_dot(n_embd, &scratch[row], 0, row_ptr, 0, inp_conv_.data(), 0, 1);
                }
                if (layer.gate_up_scale != 1.0f) {
                    for (int i = 0; i < n_rows; ++i) scratch[i] *= layer.gate_up_scale;
                }
            } else {
                const char * gate_expert = (const char *)layer.gate_data + (size_t)eid * layer.gate_stride;
                const char * up_expert = (const char *)layer.up_data + (size_t)eid * layer.up_stride;
                const uint8_t * up_inp = (!inp_conv_up.empty()) ? inp_conv_up.data() : inp_conv_.data();
#ifdef _OPENMP
                #pragma omp parallel for num_threads(n_threads_) schedule(static)
#endif
                for (int row = 0; row < n_ff; ++row) {
                    const void * gp = gate_expert + (size_t)row * gu_weight_row;
                    gu_vec_dot(n_embd, &scratch[row], 0, gp, 0, inp_conv_.data(), 0, 1);
                    const void * up = up_expert + (size_t)row * up_weight_row;
                    up_vec_dot(n_embd, &scratch[n_ff + row], 0, up, 0, up_inp, 0, 1);
                }
                if (layer.gate_scale != 1.0f) {
                    for (int i = 0; i < n_ff; ++i) scratch[i] *= layer.gate_scale;
                }
                if (layer.up_scale != 1.0f) {
                    for (int i = 0; i < n_ff; ++i) scratch[n_ff + i] *= layer.up_scale;
                }
            }

            // ── Phase 2: SwiGLU activation ──
            for (int i = 0; i < n_ff; ++i) {
                const float gate = scratch[i];
                const float up = scratch[n_ff + i];
                scratch[i] = (gate / (1.0f + expf(-gate))) * up;
            }

            // ── Phase 3: down matmul → output (weighted accumulate) ──
            // Convert SwiGLU result to down's vec_dot format (serial, small)
            dn_from_float(scratch, thread_bufs_[0].mid_conv.data(), n_ff);
            const uint8_t * mid_conv_data = thread_bufs_[0].mid_conv.data();

            const char * down_expert = (const char *)layer.down_data + (size_t)eid * layer.down_stride;
            const float scale = w * layer.down_scale;

            // Parallel down matmul — each thread accumulates its own output rows
#ifdef _OPENMP
            #pragma omp parallel for num_threads(n_threads_) schedule(static)
#endif
            for (int row = 0; row < n_embd; ++row) {
                float val;
                const void * row_ptr = down_expert + (size_t)row * dn_weight_row;
                dn_vec_dot(n_ff, &val, 0, row_ptr, 0, mid_conv_data, 0, 1);
                output[row] += scale * val;
            }
        }
        return true;
    }
};

std::unique_ptr<MoeExpertCompute> make_cpu_moe_expert_compute(int n_ff_max) {
    return std::make_unique<CpuMoeExpertCompute>(n_ff_max);
}

}  // namespace dflash::common
