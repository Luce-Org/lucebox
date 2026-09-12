#pragma once

#include "seq_slot_manager.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <optional>
#include <limits>
#include <vector>

namespace dflash::common {

// Reserved configuration value; resolved before model workers start.
inline constexpr size_t kAutoKvOffloadBytes = (std::numeric_limits<size_t>::max)();

// Available physical RAM, bounded by Linux cgroup v2 limits when present.
// Unknown on unsupported platforms/controllers: automatic sizing disables.
// Root arguments also allow tests to exercise captured proc/cgroup trees.
std::optional<size_t> available_kv_offload_memory(
    const char * proc_root = "/proc", const char * cgroup_root = "/sys/fs/cgroup");

// At most a quarter of remaining host headroom across all models, including
// explicit caps. The model supplies the maximum useful checkpoint payload.
size_t auto_kv_offload_budget(size_t pool_bytes, size_t available_bytes,
                              size_t explicit_bytes, size_t auto_models);

// One physical-page plane, e.g. one Qwen KV head or one DS4 compressed
// tensor. The engine defines the layout; no model-specific state moves here.
struct PagedKvTensor {
    ggml_tensor * tensor = nullptr;
    size_t offset = 0;
    size_t block_bytes = 0;
};

// Same-slot suspension: only paged KV leaves the device. Recurrent/raw-ring/
// compressor/draft state stays in its reserved slot, and the slot manager
// retains positions, history and RNG. Copies complete before blocks change
// ownership. All calls run on the engine's sole worker thread.
class PagedKvOffload {
public:
    PagedKvOffload(SeqSlotManager & slots, PagedKvPool & pool,
                   ggml_backend_t backend, std::vector<PagedKvTensor> tensors);
    size_t capacity() const;
    SeqEngine::KvOffloadState state(int slot) const;
    bool suspend(int slot, size_t available_bytes, std::string & error);
    bool restore(int slot, std::vector<int32_t> & blocks, std::string & error);
    void discard(int slot);

private:
    bool payload_size(size_t blocks, size_t & bytes, std::string & error) const;
    SeqSlotManager & slots_;
    PagedKvPool & pool_;
    ggml_backend_t backend_;
    std::vector<PagedKvTensor> tensors_;
    std::vector<std::vector<uint8_t>> saved_;
};

} // namespace dflash::common
