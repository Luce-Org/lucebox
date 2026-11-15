#pragma once

// Split mode on an integrated expert device (Strix Halo) whose carve, the GPU's own memory, cannot hold every routed
// expert stack: the stacks fill the carve in allocation order and the rest go to locked host memory, which the GPU
// reads from the same DRAM (ggml_backend_cuda_set_host_spill, the tier DeepSeek V4 also places experts in past the
// carve). On R9700 + Strix Halo with a 64 GiB carve, Flash Next's 71.7 GiB of routed experts need both.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace luce::common {

// What the carve keeps free after the expert stacks: the device's headroom (4 GiB on an APU, as DeepSeek V4 keeps)
// and room for what split mode allocates there afterwards: its part of the prompt chunk and decode graphs, the
// read-ahead router rows and the LM head tail.
constexpr uint64_t kQwen4ExpCarveReserve = 10ull << 30;
// Host memory the locked part leaves to the OS and the server's own host buffers, as DeepSeek V4 keeps.
constexpr uint64_t kQwen4ExpHostKeep = 5ull << 30;

// One routed expert stack in the expert buffer: [offset, offset + bytes).
struct Qwen4ExpExpertSpan {
    size_t offset = 0;
    size_t bytes = 0;
};

struct Qwen4ExpCarvePlan {
    size_t cut = 0;          // expert buffer offsets below this stay in the carve, the rest go to the host part
    size_t host_bytes = 0;   // the host part's size (0: the carve holds every stack)
    bool carve_holds_one = false;   // the carve holds at least the first stack
    bool host_fits = false;         // the host part leaves kQwen4ExpHostKeep of `host_available`
    bool ok() const { return carve_holds_one && host_fits; }
};

// The split of an expert buffer of `size` bytes, its stacks `spans` in increasing offset order: the first stack that
// would end past `device_free` less kQwen4ExpCarveReserve starts the host part, so no stack straddles the two.
inline Qwen4ExpCarvePlan qwen4exp_plan_carve(const std::vector<Qwen4ExpExpertSpan> & spans, size_t size,
                                             uint64_t device_free, uint64_t host_available) {
    Qwen4ExpCarvePlan plan;
    const uint64_t room = device_free > kQwen4ExpCarveReserve ? device_free - kQwen4ExpCarveReserve : 0;
    plan.cut = size;
    for (const Qwen4ExpExpertSpan & s : spans) {
        if ((uint64_t) s.offset + s.bytes > room) {
            plan.cut = s.offset;
            break;
        }
    }
    plan.host_bytes = size - plan.cut;
    plan.carve_holds_one = plan.cut > 0;
    plan.host_fits = host_available >= kQwen4ExpHostKeep && plan.host_bytes <= host_available - kQwen4ExpHostKeep;
    return plan;
}

// True when `cut` falls between two stacks (or at the buffer's end): a kept split can place the stacks again.
inline bool qwen4exp_carve_cut_valid(const std::vector<Qwen4ExpExpertSpan> & spans, size_t size, size_t cut) {
    if (cut == size) return true;
    for (const Qwen4ExpExpertSpan & s : spans) {
        if (s.offset == cut) return true;
        if (s.offset < cut && s.offset + s.bytes > cut) return false;
    }
    return false;
}

}  // namespace luce::common
