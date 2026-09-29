// Streamed expert slots are shared by layers of different quant types, each reading them through
// a view whose expert stride the kernels divide into whole blocks: the stride must divide exactly.
#include "moe_hybrid_expert_cache.h"
#include "ggml.h"
#include <cstdio>
#include <vector>
using luce::common::moe_expert_slot_stride;
int main() {
    bool ok = true;
    // DS4.1 gate/up experts, 2304 x 5120: IQ2_XXS, IQ2_XS and IQ3_XXS layers share the gate slots.
    const size_t rows = 2304, blocks = 5120 / 256;
    const std::vector<ggml_type> gate = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS};
    std::vector<size_t> sizes; size_t largest = 0;
    for (ggml_type t : gate) { sizes.push_back(ggml_type_size(t)); largest = std::max(largest, rows * blocks * ggml_type_size(t)); }
    const size_t stride = moe_expert_slot_stride(largest, sizes);
    if (stride < largest || stride - largest >= 2 * 2 * 3 * 7 * 7 * 11 * 37) ok = false;   // < one lcm unit of slack
    for (size_t ts : sizes) if (stride % ts) ok = false;
    if (largest % ggml_type_size(GGML_TYPE_IQ2_XXS) == 0) ok = false;   // the unrounded stride was not whole IQ2_XXS blocks
    // One type: the largest expert itself.
    const size_t q2 = 5120 * 9 * ggml_type_size(GGML_TYPE_Q2_K);
    if (moe_expert_slot_stride(q2, {ggml_type_size(GGML_TYPE_Q2_K)}) != q2) ok = false;
    // Down experts, 5120 x 2304, five types.
    std::vector<size_t> down; size_t dl = 0;
    for (ggml_type t : {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_Q2_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_Q3_K}) {
        down.push_back(ggml_type_size(t)); dl = std::max(dl, (size_t) 5120 * 9 * ggml_type_size(t));
    }
    const size_t ds = moe_expert_slot_stride(dl, down);
    for (size_t ts : down) if (ds % ts) ok = false;
    std::printf("gate stride %zu (largest %zu), down stride %zu (largest %zu): %s\n", stride, largest, ds, dl, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
