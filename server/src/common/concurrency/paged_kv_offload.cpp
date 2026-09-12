#include "paged_kv_offload.h"

#include <limits>
#include <new>
#include <utility>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <filesystem>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace dflash::common {

size_t auto_kv_offload_budget(size_t pool_bytes, size_t available_bytes,
                              size_t explicit_bytes, size_t auto_models) {
    const size_t allowance = available_bytes / 4;
    return auto_models ? std::min(pool_bytes,
        (allowance - std::min(allowance, explicit_bytes)) / auto_models) : 0;
}

std::optional<size_t> available_kv_offload_memory(const char * proc_root, const char * cgroup_root) {
#if defined(__linux__)
    std::ifstream meminfo(std::filesystem::path(proc_root) / "meminfo");
    std::string line;
    std::optional<size_t> available;
    while (std::getline(meminfo, line)) {
        std::istringstream fields(line);
        std::string key, unit;
        size_t kib = 0;
        if (fields >> key >> kib >> unit && key == "MemAvailable:" && unit == "kB" &&
            kib <= (std::numeric_limits<size_t>::max)() / 1024) {
            available = kib * 1024;
            break;
        }
    }
    if (!available) return std::nullopt;
    std::ifstream groups(std::filesystem::path(proc_root) / "self/cgroup");
    if (!groups) return std::nullopt;
    while (std::getline(groups, line)) {
        // cgroup v1 memory accounting is not supported by this probe.
        if (line.find(":memory:") != std::string::npos ||
            line.find(",memory") != std::string::npos ||
            line.find(":memory,") != std::string::npos) return std::nullopt;
        if (line.rfind("0::/", 0) != 0) continue;
        const std::filesystem::path root(cgroup_root);
        const std::filesystem::path relative = line.substr(4);
        for (const auto & part : relative) if (part == "..") return std::nullopt;
        auto path = relative.empty() ? root : root / relative;
        // Check ancestors too: an unlimited child can have a bounded parent.
        for (;;) {
            std::ifstream limit_file(path / "memory.max");
            std::string limit;
            if (!(limit_file >> limit)) {
                // The host hierarchy root has no memory.max. A namespaced
                // container root may have one, which must still be applied.
                std::error_code ec;
                if (path == root && !std::filesystem::exists(root / "memory.max", ec) && !ec &&
                    std::filesystem::exists(root / "cgroup.controllers", ec) && !ec) break;
                return std::nullopt;
            }
            if (limit != "max") {
                size_t maximum = 0, used = 0;
                std::istringstream value(limit);
                std::ifstream current(path / "memory.current");
                if (!(value >> maximum) || !(current >> used)) return std::nullopt;
                *available = std::min(*available, maximum - std::min(maximum, used));
            }
            if (path == root) break;
            path = path.parent_path();
        }
    }
    return available;
#elif defined(_WIN32)
    (void)proc_root;
    (void)cgroup_root;
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) return static_cast<size_t>(status.ullAvailPhys);
    return std::nullopt;
#else
    (void)proc_root;
    (void)cgroup_root;
    return std::nullopt;
#endif
}

size_t PagedKvOffload::capacity() const {
    size_t bytes = 0;
    std::string error;
    const size_t per_request = std::min<size_t>(pool_.physical_block_count(),
        (static_cast<size_t>(slots_.max_context()) + pool_.block_size() - 1) / pool_.block_size());
    const size_t suspended_slots = static_cast<size_t>(std::max(0, slots_.slot_count() - 1));
    if (suspended_slots && per_request > (std::numeric_limits<size_t>::max)() / suspended_slots) return 0;
    return payload_size(per_request * suspended_slots, bytes, error) ? bytes : 0;
}

PagedKvOffload::PagedKvOffload(SeqSlotManager & slots, PagedKvPool & pool,
                               ggml_backend_t backend,
                               std::vector<PagedKvTensor> tensors)
    : slots_(slots), pool_(pool), backend_(backend), tensors_(std::move(tensors)),
      saved_(static_cast<size_t>(slots.slot_count())) {}

SeqEngine::KvOffloadState PagedKvOffload::state(int slot) const {
    if (!slots_.is_active(slot)) return {};
    const SeqSlot & s = slots_.slot(slot);
    return {s.evicted(), s.recomputing(), saved_[(size_t)slot].size()};
}

bool PagedKvOffload::payload_size(size_t blocks, size_t & bytes,
                                  std::string & error) const {
    bytes = 0;
    for (const auto & plane : tensors_) {
        if (!plane.tensor || !plane.block_bytes ||
            plane.offset > ggml_nbytes(plane.tensor) ||
            pool_.physical_block_count() >
                (ggml_nbytes(plane.tensor) - plane.offset) / plane.block_bytes ||
            blocks > ((std::numeric_limits<size_t>::max)() - bytes) / plane.block_bytes) {
            error = "invalid paged KV offload tensor layout";
            return false;
        }
        bytes += blocks * plane.block_bytes;
    }
    return true;
}

bool PagedKvOffload::suspend(int slot, size_t available_bytes,
                             std::string & error) {
    error.clear();
    if (!slots_.is_active(slot) || state(slot).suspended ||
        !slots_.slot(slot).staged_tokens.empty()) {
        error = "KV suspension requires a resident slot at a committed step boundary";
        return false;
    }
    try {
        PagedKvSequenceSnapshot sequence;
        if (pool_.sequence(slots_.slot(slot).handle, sequence) != PagedKvStatus::Ok ||
            sequence.kv_seq_len != static_cast<uint32_t>(slots_.slot(slot).cur_pos)) {
            error = "KV suspension position does not match the pool";
            return false;
        }
        size_t bytes = 0;
        if (!payload_size(sequence.block_table.size(), bytes, error)) return false;
        if (bytes > available_bytes) {
            error = "decode KV offload RAM budget exhausted";
            return false;
        }
        // Model loading and other processes may have consumed headroom since
        // startup. Probe only on pressure, never on the steady decode path.
        const auto available = available_kv_offload_memory();
        if (available && bytes > *available / 4) {
            error = "insufficient current host memory for decode KV offload";
            return false;
        }
        std::vector<uint8_t> saved(bytes);
        ggml_backend_synchronize(backend_);
        size_t offset = 0;
        for (const auto & plane : tensors_) {
            for (uint32_t block : sequence.block_table) {
                ggml_backend_tensor_get(plane.tensor, saved.data() + offset,
                    plane.offset + static_cast<size_t>(block) * plane.block_bytes,
                    plane.block_bytes);
                offset += plane.block_bytes;
            }
        }
        if (!slots_.detach_kv(slot)) {
            error = "could not release preserved KV blocks";
            return false;
        }
        saved_[(size_t)slot] = std::move(saved);
        return true;
    } catch (const std::bad_alloc &) {
        error = "could not allocate decode KV offload RAM";
        return false;
    }
}

bool PagedKvOffload::restore(int slot, std::vector<int32_t> & blocks,
                             std::string & error) {
    error.clear();
    blocks.clear();
    if (!slots_.is_active(slot) || !slots_.slot(slot).suspended()) {
        error = slots_.is_active(slot) && slots_.slot(slot).recomputing()
            ? "KV copy restore requires a checkpointed suspension"
            : "KV restore requires a suspended slot";
        return false;
    }
    const size_t count = (static_cast<size_t>(slots_.slot(slot).cur_pos) +
                          pool_.block_size() - 1) / pool_.block_size();
    size_t bytes = 0;
    if (!payload_size(count, bytes, error)) return false;
    if (bytes != saved_[(size_t)slot].size()) {
        error = "saved KV payload size does not match the sequence";
        return false;
    }
    try {
        blocks.resize(count);
        if (!slots_.attach_kv(slot)) return false;
        PagedKvSequenceSnapshot sequence;
        if (pool_.sequence(slots_.slot(slot).handle, sequence) != PagedKvStatus::Ok) {
            slots_.detach_kv(slot);
            error = "restored KV sequence is unavailable";
            return false;
        }
        ggml_backend_synchronize(backend_);
        size_t offset = 0;
        for (const auto & plane : tensors_) {
            for (uint32_t block : sequence.block_table) {
                ggml_backend_tensor_set(plane.tensor, saved_[(size_t)slot].data() + offset,
                    plane.offset + static_cast<size_t>(block) * plane.block_bytes,
                    plane.block_bytes);
                offset += plane.block_bytes;
            }
        }
        for (size_t i = 0; i < count; ++i) blocks[i] = (int32_t)sequence.block_table[i];
        ggml_backend_synchronize(backend_);
        discard(slot);
        return true;
    } catch (const std::bad_alloc &) {
        if (!state(slot).suspended) slots_.detach_kv(slot);
        error = "could not allocate KV restore metadata";
        return false;
    }
}

void PagedKvOffload::discard(int slot) {
    if (slot >= 0 && static_cast<size_t>(slot) < saved_.size()) {
        std::vector<uint8_t>().swap(saved_[(size_t)slot]);
    }
}

} // namespace dflash::common
