#include "qwen35moe/qwen35moe_ffn.h"
#include <unordered_set>
#include "common/moe_hybrid_ffn_eval.h"
#include "common/moe_hybrid_placement.h"
#include "support/environment.h"
#include <nlohmann/json.hpp>
#include "support/test_assert.h"
#include <vector>

using json = nlohmann::json;
using namespace dflash::common;

TEST_CASE(ServerUnitFixture, test_moe_hybrid_expert_compute_batch_default) {
    dflash_unsetenv("DFLASH_MOE_EXPERT_COMPUTE_BATCH");
    dflash_unsetenv("DFLASH_MOE_EXPERT_COMPUTE_BATCH_MAX");
    TEST_ASSERT(moe_hybrid_expert_compute_batch_limit() == 32);
}

TEST_CASE(ServerUnitFixture, test_moe_hybrid_expert_compute_ipc_mode_batch_limit) {
    dflash_unsetenv("DFLASH_MOE_EXPERT_COMPUTE_IPC_MODE");
    dflash_unsetenv("DFLASH_MOE_EXPERT_COMPUTE_IPC_BATCH_CAPACITY");
    TEST_ASSERT(moe_hybrid_expert_compute_ipc_batch_limit(2048) == 1024);

    dflash_setenv("DFLASH_MOE_EXPERT_COMPUTE_IPC_MODE", "auto");
    dflash_setenv("DFLASH_MOE_EXPERT_COMPUTE_IPC_BATCH_CAPACITY", "512");
    TEST_ASSERT(moe_hybrid_expert_compute_ipc_batch_limit(2048) == 512);

    dflash_setenv("DFLASH_MOE_EXPERT_COMPUTE_IPC_MODE", "batched");
    TEST_ASSERT(moe_hybrid_expert_compute_ipc_batch_limit(2048) == 512);

    dflash_setenv("DFLASH_MOE_EXPERT_COMPUTE_IPC_MODE", "stream");
    TEST_ASSERT(moe_hybrid_expert_compute_ipc_batch_limit(2048) == 32);

    dflash_unsetenv("DFLASH_MOE_EXPERT_COMPUTE_IPC_MODE");
    dflash_unsetenv("DFLASH_MOE_EXPERT_COMPUTE_IPC_BATCH_CAPACITY");
}

TEST_CASE(ServerUnitFixture, test_moe_hybrid_prefill_hot_sub_batch_limit) {
    dflash_unsetenv("DFLASH_MOE_PREFILL_HOT_SUB_BATCH");
    TEST_ASSERT(moe_hybrid_prefill_hot_sub_batch_limit() == 4);

    dflash_setenv("DFLASH_MOE_PREFILL_HOT_SUB_BATCH", "0");
    TEST_ASSERT(moe_hybrid_prefill_hot_sub_batch_limit() == 4);

    dflash_setenv("DFLASH_MOE_PREFILL_HOT_SUB_BATCH", "3");
    TEST_ASSERT(moe_hybrid_prefill_hot_sub_batch_limit() == 3);

    dflash_setenv("DFLASH_MOE_PREFILL_HOT_SUB_BATCH", "8");
    TEST_ASSERT(moe_hybrid_prefill_hot_sub_batch_limit() == 4);

    dflash_unsetenv("DFLASH_MOE_PREFILL_HOT_SUB_BATCH");
}

TEST_CASE(ServerUnitFixture, test_moe_hybrid_uma_core_memory_is_saturating) {
    constexpr size_t gib = (size_t) 1024 * 1024 * 1024;
    TEST_ASSERT(moe_hybrid_core_bytes_from_memory(
        "test", 6 * gib, 8 * gib) == 2 * gib);
    TEST_ASSERT(moe_hybrid_core_bytes_from_memory(
        "test", 10 * gib, 8 * gib) == 0);
}

TEST_CASE(ServerUnitFixture, test_moe_hybrid_canonical_rocmfp2_q2_is_tokenwise) {
    // ROCmFP2's safe q>1 fallback builds one owner graph per token. Canonical
    // route-order joins must preserve [hidden, route, token] while appending
    // those token slices; concatenating the route dimension makes the final
    // owner reduction invalid.
    dflash_unsetenv("DFLASH_MOE_TP_GROUPED_MMVQ");
    dflash_unsetenv("DFLASH_DS4_TP_GROUPED_MMVQ");

    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr);

    constexpr int n_embd = 32;
    constexpr int n_ff = 32;
    constexpr int n_expert = 4;
    constexpr int n_used = 2;
    constexpr int n_tokens = 2;

    MoeHybridConfig cfg;
    cfg.n_embd = n_embd;
    cfg.n_ff_exp = n_ff;
    cfg.n_expert = n_expert;
    cfg.n_expert_used = n_used;

    MoeHybridLayerStorage storage;
    storage.gate_hot = ggml_new_tensor_3d(
        ctx, GGML_TYPE_Q2_0_ROCMFP2, n_embd, n_ff, 2);
    storage.up_hot = ggml_new_tensor_3d(
        ctx, GGML_TYPE_Q2_0_ROCMFP2, n_embd, n_ff, 2);
    storage.down_hot = ggml_new_tensor_3d(
        ctx, GGML_TYPE_Q3_0_ROCMFPX, n_ff, n_embd, 2);
    storage.gate_cold = ggml_new_tensor_3d(
        ctx, GGML_TYPE_Q2_0_ROCMFP2, n_embd, n_ff, 2);
    storage.up_cold = ggml_new_tensor_3d(
        ctx, GGML_TYPE_Q2_0_ROCMFP2, n_embd, n_ff, 2);
    storage.down_cold = ggml_new_tensor_3d(
        ctx, GGML_TYPE_Q3_0_ROCMFPX, n_ff, n_embd, 2);
    storage.hot_local_by_global = {0, 1, -1, -1};
    storage.cold_local_by_global = {-1, -1, 0, 1};

    MoeLayerDesc desc;
    ggml_tensor * input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * weights = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_used, n_tokens);
    MoeHybridGraphInputs out;
    const bool built = build_moe_hybrid_ffn_graph(
        ctx, nullptr, cfg, desc, storage, input, ids, weights, n_tokens,
        out, /*include_shared=*/false, /*allow_fused_combine=*/false,
        MoeHybridJoinMode::CanonicalRouteOrder);
    TEST_ASSERT(built);
    TEST_ASSERT(out.output != nullptr);
    TEST_ASSERT(out.output->ne[0] == n_embd);
    TEST_ASSERT(out.output->ne[1] == n_tokens);

    const auto count_mul_mat_id = [](const std::vector<ggml_tensor *> & nodes) {
        int count = 0;
        for (const ggml_tensor * node : nodes) {
            if (node && node->op == GGML_OP_MUL_MAT_ID) ++count;
        }
        return count;
    };
    TEST_ASSERT(count_mul_mat_id(out.hot_nodes) == 3 * n_tokens);
    TEST_ASSERT(count_mul_mat_id(out.cold_nodes) == 3 * n_tokens);

    ggml_free(ctx);
}

TEST_CASE(ServerUnitFixture, test_bailingmoe3_router_builds_group_mask) {
    ggml_init_params params{};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr);

    TargetWeights weights;
    weights.n_expert = 512;
    weights.n_expert_used = 8;
    weights.n_expert_groups = 8;
    weights.n_expert_groups_used = 4;
    weights.expert_gating_func = 2;
    weights.expert_weights_norm = true;
    weights.expert_weights_scale = 2.5f;

    TargetLayer layer;
    layer.ffn_gate_inp = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, 16, weights.n_expert);
    layer.ffn_exp_probs_b = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F32, weights.n_expert);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 2);

    const Qwen35MoeRouterOutputs router = build_qwen35moe_router(
        ctx, input, weights, layer);
    TEST_ASSERT(router.selected != nullptr);
    TEST_ASSERT(router.weights != nullptr);

    std::unordered_set<const ggml_tensor *> visited;
    const auto contains_op = [&](const auto & self,
                                 const ggml_tensor * tensor,
                                 ggml_op op) -> bool {
        if (!tensor || !visited.insert(tensor).second) return false;
        if (tensor->op == op) return true;
        for (const ggml_tensor * source : tensor->src) {
            if (self(self, source, op)) return true;
        }
        return false;
    };
    TEST_ASSERT(contains_op(contains_op, router.selected, GGML_OP_SET_ROWS));

    ggml_free(ctx);
}
