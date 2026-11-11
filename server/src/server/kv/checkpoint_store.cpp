#include "server/kv/checkpoint_store.h"

#include <utility>

namespace luce::common {

namespace {

// The last backend slot: no prefix-cache entry is ever placed there
// (PrefixCache::MAX_CACHE_SLOTS).
constexpr int kStagingSlot = ModelBackend::kMaxSlots - 1;

}  // namespace

int BackendCheckpoints::granularity() const {
    return backend_.snapshot_granularity();
}

int BackendCheckpoints::restore_point_spacing() const {
    return backend_.restore_point_spacing();
}

bool BackendCheckpoints::prefill_cuts_at_restore_points() const {
    return backend_.prefill_cuts_at_restore_points();
}

size_t BackendCheckpoints::estimate_bytes(int tokens) const {
    return backend_.snapshot_bytes_estimate(tokens);
}

size_t BackendCheckpoints::estimate_state_bytes(int tokens) const {
    return backend_.snapshot_state_bytes_estimate(tokens);
}

bool BackendCheckpoints::drop_kv(Checkpoint checkpoint) {
    return checkpoint.valid() && backend_.snapshot_drop_kv(checkpoint.cache_slot());
}

bool BackendCheckpoints::state_only(Checkpoint checkpoint) const {
    return checkpoint.valid() && backend_.snapshot_state_only(checkpoint.cache_slot());
}

bool BackendCheckpoints::used(Checkpoint checkpoint) const {
    return checkpoint.valid() && backend_.snapshot_used(checkpoint.cache_slot());
}

int BackendCheckpoints::position(Checkpoint checkpoint) const {
    return checkpoint.valid() ? backend_.snapshot_cur_pos(checkpoint.cache_slot()) : 0;
}

void BackendCheckpoints::free(Checkpoint checkpoint) {
    if (checkpoint.valid()) backend_.snapshot_free(checkpoint.cache_slot());
}

Checkpoint BackendCheckpoints::staging() const {
    return Checkpoint::of_cache_slot(kStagingSlot);
}

ModelBackend::SnapshotRef BackendCheckpoints::ref(Checkpoint checkpoint) const {
    return checkpoint.valid() ? backend_.snapshot_ref(checkpoint.cache_slot())
                              : ModelBackend::SnapshotRef{};
}

OwnedSnapshot BackendCheckpoints::detach(Checkpoint checkpoint) {
    return checkpoint.valid() ? backend_.snapshot_detach(checkpoint.cache_slot())
                              : OwnedSnapshot{};
}

bool BackendCheckpoints::adopt(Checkpoint checkpoint, OwnedSnapshot & snapshot) {
    if (!checkpoint.valid() || snapshot.empty()) return false;
    if (!backend_.snapshot_adopt(checkpoint.cache_slot(), snapshot.ctx(),
                                 snapshot.buf(), snapshot.cur_pos(),
                                 snapshot.last_tok())) {
        return false;
    }
    snapshot.release();
    return true;
}

void BackendCheckpoints::bind_capture(GenerateRequest & request,
                                      Checkpoint checkpoint, int position) const {
    request.snap_slot = checkpoint.cache_slot();
    request.snap_pos = checkpoint.valid() ? position : -1;
}

GenerateResult BackendCheckpoints::restore_and_generate(
        Checkpoint checkpoint, const GenerateRequest & request, const DaemonIO & io) {
    return backend_.restore_and_generate(checkpoint.cache_slot(), request, io);
}

GenerateResult BackendCheckpoints::prefill(Checkpoint into,
                                           const std::vector<int32_t> & prompt,
                                           std::vector<int> restore_points,
                                           Checkpoint from, CancellationProbe cancel) {
    GenerateRequest request;
    request.prompt = prompt;
    request.n_gen = 0;
    request.restore_points = std::move(restore_points);
    bind_capture(request, into, (int)prompt.size());
    DaemonIO io;
    io.stream_fd = -1;
    io.should_cancel = std::move(cancel);
    return from.valid() ? restore_and_generate(from, request, io)
                        : backend_.generate(request, io);
}

int BackendCheckpoints::restorable_prefix(const std::vector<int32_t> & prompt,
                                          Checkpoint checkpoint, int tokens) const {
    return backend_.restorable_prefix(prompt, checkpoint.cache_slot(), tokens);
}

bool BackendCheckpoints::save_now(Checkpoint checkpoint) {
    return checkpoint.valid() && backend_.snapshot_save(checkpoint.cache_slot());
}

bool BackendCheckpoints::save_deferred(Checkpoint checkpoint) {
    return checkpoint.valid() && backend_.snapshot_save_deferred(checkpoint.cache_slot());
}

void BackendCheckpoints::flush_deferred() {
    backend_.snapshot_flush_deferred();
}

OwnedSnapshot BackendCheckpoints::take_displaced(Checkpoint checkpoint) {
    return checkpoint.valid() ? backend_.snapshot_take_displaced(checkpoint.cache_slot())
                              : OwnedSnapshot{};
}

bool EngineCheckpoints::supported() const {
    return engine_.supports_prefix_store();
}

bool EngineCheckpoints::in_backend_slots() const {
    return engine_.prefix_store_in_backend_snapshots();
}

size_t EngineCheckpoints::estimate_bytes(int tokens) const {
    return engine_.estimate_prefix_store_bytes(tokens);
}

size_t EngineCheckpoints::estimate_state_bytes(int tokens) const {
    return engine_.estimate_prefix_state_bytes(tokens);
}

bool EngineCheckpoints::drop_kv(Checkpoint checkpoint, int tokens) {
    return engine_.drop_prefix_kv(ref(checkpoint, tokens));
}

void EngineCheckpoints::discard(PrefixStoreRef ref) {
    engine_.discard_prefix_store(ref);
}

}  // namespace luce::common
