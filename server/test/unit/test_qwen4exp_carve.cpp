// Standalone host policy check: c++ -std=c++17 -Iserver/src server/test/unit/test_qwen4exp_carve.cpp -o /tmp/test_qwen4exp_carve
#include "qwen4exp/qwen4exp_carve.h"

#include <cassert>
#include <cstdio>
#include <vector>

using luce::common::kQwen4ExpCarveReserve;
using luce::common::kQwen4ExpHostKeep;
using luce::common::qwen4exp_carve_cut_valid;
using luce::common::qwen4exp_plan_carve;
using luce::common::Qwen4ExpCarvePlan;
using luce::common::Qwen4ExpExpertSpan;

static constexpr uint64_t GiB = 1ull << 30;

int main() {
    // Three 1 GiB stacks with a 256-byte gap after each (alignment), 3 GiB + 512 bytes in all.
    const std::vector<Qwen4ExpExpertSpan> spans = {
        {0, (size_t) GiB}, {(size_t) GiB + 256, (size_t) GiB}, {2 * (size_t) GiB + 512, (size_t) GiB}};
    const size_t size = 3 * (size_t) GiB + 512;

    // Every stack fits beside the reserve: no host part.
    Qwen4ExpCarvePlan p = qwen4exp_plan_carve(spans, size, kQwen4ExpCarveReserve + size, 64 * GiB);
    assert(p.ok() && p.cut == size && p.host_bytes == 0);

    // Room for two stacks and a byte short of the third: the third starts the host part, nothing straddles.
    p = qwen4exp_plan_carve(spans, size, kQwen4ExpCarveReserve + size - 1, 64 * GiB);
    assert(p.ok() && p.cut == spans[2].offset && p.host_bytes == GiB);

    // Room for one and a half stacks: the second starts the host part.
    p = qwen4exp_plan_carve(spans, size, kQwen4ExpCarveReserve + GiB + GiB / 2, 64 * GiB);
    assert(p.ok() && p.cut == spans[1].offset && p.host_bytes == size - spans[1].offset);

    // The carve holds no stack (less free than the reserve, or less than the first stack beside it).
    p = qwen4exp_plan_carve(spans, size, kQwen4ExpCarveReserve / 2, 64 * GiB);
    assert(!p.ok() && !p.carve_holds_one && p.cut == 0);
    p = qwen4exp_plan_carve(spans, size, kQwen4ExpCarveReserve + GiB - 1, 64 * GiB);
    assert(!p.ok() && !p.carve_holds_one);

    // The host part must leave kQwen4ExpHostKeep of the available host memory.
    p = qwen4exp_plan_carve(spans, size, kQwen4ExpCarveReserve + GiB, kQwen4ExpHostKeep + size - spans[1].offset);
    assert(p.ok() && p.cut == spans[1].offset);
    p = qwen4exp_plan_carve(spans, size, kQwen4ExpCarveReserve + GiB, kQwen4ExpHostKeep + size - spans[1].offset - 1);
    assert(!p.ok() && p.carve_holds_one && !p.host_fits);
    p = qwen4exp_plan_carve(spans, size, kQwen4ExpCarveReserve + GiB, kQwen4ExpHostKeep / 2);
    assert(!p.ok() && !p.host_fits);

    // Flash Next's shape: 48 layers x (gate, up, down), 71.73 GiB in all, on a carve with 60 GiB free.
    {
        std::vector<Qwen4ExpExpertSpan> stacks;
        const size_t stack = (size_t) (71.73 * GiB / 144);
        for (size_t i = 0; i < 144; ++i) stacks.push_back({i * stack, stack});
        const uint64_t carve_free = 60 * GiB;
        p = qwen4exp_plan_carve(stacks, 144 * stack, carve_free, 56 * GiB);
        assert(p.ok() && p.cut % stack == 0 && p.cut + kQwen4ExpCarveReserve <= carve_free);
        assert(p.cut + stack + kQwen4ExpCarveReserve > carve_free);   // the carve takes every stack it can
        assert(p.cut + p.host_bytes == 144 * stack);
    }

    // A kept split can place the stacks again only at a stack's start or the buffer's end.
    assert(qwen4exp_carve_cut_valid(spans, size, size));
    assert(qwen4exp_carve_cut_valid(spans, size, 0));
    assert(qwen4exp_carve_cut_valid(spans, size, spans[1].offset));
    assert(qwen4exp_carve_cut_valid(spans, size, spans[2].offset));
    assert(!qwen4exp_carve_cut_valid(spans, size, GiB / 2));     // inside the first stack
    assert(!qwen4exp_carve_cut_valid(spans, size, GiB + 128));   // in the gap after it, where no stack starts

    std::puts("test_qwen4exp_carve: ok");
    return 0;
}
