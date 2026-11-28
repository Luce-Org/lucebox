#pragma once

// Host copies of drafter scoring sessions (qwen35_drafter.cpp), so a session survives the drafter being released
// (--draft-residency auto: ReleaseAfterUse frees the drafter after every compression).
//
// When the drafter is released, every session this load created or extended is copied to host RAM next to its host
// bookkeeping: its recurrent-state tensors whole, and its token-indexed tensors (attention KV, block-15 keys) only
// for the tokens a resume can read (below the session's checkpoint). A session restored and left untouched keeps the
// host copy it came from. When the drafter is loaded again, only the stored session a request can resume from is
// restored, into freshly allocated tensors of the same layout, before the scoring call; resume runs unchanged.
// A copy is restored only under the identity it was stored with (drafter file, build, session parameters) and only
// if every tensor matches in type, shape, strides and size; anything else drops it and the call scores from token 0.
// Nothing is approximated.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace luce::pflash {

// Host bytes: pinned (a buffer of the device's host buffer type) when one is given and can be allocated, else plain
// memory.
class PFlashHostBlob {
public:
    PFlashHostBlob() = default;
    PFlashHostBlob(const PFlashHostBlob &) = delete;
    PFlashHostBlob & operator=(const PFlashHostBlob &) = delete;
    PFlashHostBlob(PFlashHostBlob && o) noexcept { *this = std::move(o); }
    PFlashHostBlob & operator=(PFlashHostBlob && o) noexcept;
    ~PFlashHostBlob() { reset(); }
    // At least ``size`` bytes; keeps the current storage when it is large enough. ``pinned`` may be null.
    bool reserve(size_t size, ggml_backend_buffer_type_t pinned);
    void reset();
    uint8_t * data() { return data_; }
    const uint8_t * data() const { return data_; }
    size_t capacity() const { return capacity_; }
    bool pinned() const { return buf_ != nullptr; }
private:
    ggml_backend_buffer_t buf_ = nullptr;
    std::vector<uint8_t> vec_;
    uint8_t * data_ = nullptr;
    size_t capacity_ = 0;
};

// One tensor's layout and the byte chunks of it that were copied (offsets into the tensor, in blob order).
struct PFlashTensorCopy {
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[GGML_MAX_DIMS] = {0, 0, 0, 0};
    size_t nb[GGML_MAX_DIMS] = {0, 0, 0, 0};
    std::vector<std::pair<size_t, size_t>> chunks;   // {offset in tensor, bytes}
    int token_dim = -1;          // -1: whole tensor
    int64_t n_tokens = 0;        // positions copied along token_dim
};

struct PFlashHostCopy {
    std::vector<PFlashTensorCopy> tensors;
    PFlashHostBlob blob;
    size_t bytes = 0;    // bytes used in blob
};

// Every tensor of ``ctx`` that owns device memory (allocated, not a view), in creation order.
std::vector<ggml_tensor *> pflash_owned_tensors(ggml_context * ctx);

// The byte chunks of ``t`` that hold its first ``n_tokens`` positions along ``token_dim`` (-1: the whole tensor).
// Empty with an error when the tensor is not contiguous or the axis is out of range.
bool pflash_token_chunks(const ggml_tensor * t, int token_dim, int64_t n_tokens,
                         std::vector<std::pair<size_t, size_t>> & chunks, std::string & error);

// Copies those chunks of every tensor to host (``token_dims[i]`` per tensor). Reuses ``out.blob`` when it is large
// enough. Fails (``out`` emptied, reason in ``error``) on a null, unallocated, view or non-contiguous tensor.
bool pflash_copy_to_host(const std::vector<ggml_tensor *> & tensors, const std::vector<int> & token_dims,
                         int64_t n_tokens, ggml_backend_buffer_type_t pinned, PFlashHostCopy & out,
                         std::string & error);

// Writes a host copy back into ``tensors``. Every pair is checked first (count, type, shape, strides, chunk bounds,
// allocated); on any mismatch nothing is written and the reason is in ``error``. A token-prefix copy may go into a
// tensor that is longer along its token axis (a session reallocated with more capacity); every other axis must match
// and the chunks are recomputed for the target layout. Bytes outside the copied chunks are left as they are (a resume
// never reads them before writing them).
bool pflash_copy_from_host(const PFlashHostCopy & host, const std::vector<ggml_tensor *> & tensors,
                           std::string & error);

// The stored session a request over ``ids`` resumes from: the largest pflash_scoring_resume(checkpoint, shared
// prefix, S, S) > 0, or -1. Host-side only; the scorer's own acquire step still decides.
struct PFlashStoredPrompt {
    const std::vector<int32_t> * ids = nullptr;
    int checkpoint = 0;
};
int pflash_pick_restore(const std::vector<PFlashStoredPrompt> & stored, const std::vector<int32_t> & ids);

// What the offload does with each session, given the store's entries and the device sessions. A device session keeps
// its host entry only when it was restored from it and never acquired since (``host_valid``); a session acquired by
// a scoring call is copied again, whatever host copy it came from. Entries whose session was restored but now holds no
// prompt are dropped. The result keeps at most ``limit`` sessions, the most recently used.
struct PFlashOffloadEntry {
    uint64_t id = 0;
    uint64_t last_used = 0;
    bool on_device = false;       // restored into a device session during this load
};
struct PFlashOffloadDevice {
    uint64_t host_id = 0;         // entry it was restored from (0: none)
    bool host_valid = false;      // restored and not acquired since
    bool has_prompt = false;      // ids non-empty and checkpoint > 0 (resumable)
    uint64_t last_used = 0;
};
struct PFlashOffloadPlan {
    std::vector<uint64_t> keep_entries;     // store entries kept as they are
    std::vector<size_t> copy_devices;       // device sessions copied out (index into devices)
    std::vector<uint64_t> drop_entries;     // store entries released
};
PFlashOffloadPlan pflash_plan_offload(const std::vector<PFlashOffloadEntry> & entries,
                                      const std::vector<PFlashOffloadDevice> & devices, int limit);

// "<path>|dev:ino|size|mtime_ns" for an existing file, "<path>|missing" otherwise, "-" for an empty path: a file
// identity cheap enough to take at every load (the drafter's sha256 is not always computed).
std::string pflash_file_identity(const std::string & path);

// Why a stored identity does not match the current one: "" when equal, else the first differing "key=value" field
// (identities are "key=value" lines).
std::string pflash_identity_mismatch(const std::string & stored, const std::string & current);

} // namespace luce::pflash
