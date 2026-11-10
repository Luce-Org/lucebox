#include "moe_split_step.h"

namespace luce::common {

void moe_split_queue_expert(ggml_cgraph * gf, ggml_tensor * expert_sum) {
    if (gf && expert_sum) ggml_build_forward_expand(gf, expert_sum);
}

ggml_tensor * moe_split_join(ggml_context * ctx, ggml_tensor * expert_sum, ggml_tensor * target_sum) {
    if (expert_sum->type != GGML_TYPE_F32) expert_sum = ggml_cast(ctx, expert_sum, GGML_TYPE_F32);
    return ggml_add(ctx, expert_sum, target_sum);
}

void moe_split_sched_init(ggml_backend_sched_t sched) {
    ggml_backend_sched_set_late_cross_input_split(sched, true);
}

}  // namespace luce::common
