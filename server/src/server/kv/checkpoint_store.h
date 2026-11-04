// Where checkpoint payloads live, behind one interface.
//
// The server never names a backend snapshot slot: it holds Checkpoint
// handles and asks a store. BackendCheckpoints keeps them in the backend's
// snapshot slots (the single-sequence path, and concurrent engines that
// keep their checkpoints there); EngineCheckpoints adds what the concurrent
// path speaks to its SeqEngine: PrefixStoreRef ids and discards. The store
// owns the disk-staging checkpoint, so no caller knows which slot that is.

#pragma once

#include "common/concurrency/prefix_store.h"
#include "common/model_backend.h"
#include "server/kv/checkpoint.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace luce::common {

// Payload, geometry and the disk boundary: what both serving paths share.
class CheckpointStore {
public:
    virtual ~CheckpointStore() = default;

    // Where a checkpoint can land and where prefill is cut: see
    // ModelBackend::snapshot_granularity, restore_point_spacing and
    // prefill_cuts_at_restore_points.
    virtual int granularity() const = 0;
    virtual int restore_point_spacing() const = 0;
    virtual bool prefill_cuts_at_restore_points() const = 0;
    // System-memory bytes a checkpoint of the first `tokens` positions
    // holds; 0 when the store cannot tell.
    virtual size_t estimate_bytes(int tokens) const = 0;

    virtual bool used(Checkpoint checkpoint) const = 0;
    // The positions (KV rows) the checkpoint holds; 0 when unused.
    virtual int position(Checkpoint checkpoint) const = 0;
    virtual void free(Checkpoint checkpoint) = 0;
    // The checkpoint a disk read is adopted into before it is restored.
    // No prefix-index entry ever lives there.
    virtual Checkpoint staging() const = 0;

    // The disk boundary. `ref` reads the payload in place (layout, digest,
    // a copy for the writer); `detach` hands it to a new owner and leaves the
    // checkpoint empty; `adopt` takes a payload read from disk, and leaves it
    // with `snapshot` when it fails.
    virtual ModelBackend::SnapshotRef ref(Checkpoint checkpoint) const = 0;
    virtual OwnedSnapshot detach(Checkpoint checkpoint) = 0;
    virtual bool adopt(Checkpoint checkpoint, OwnedSnapshot & snapshot) = 0;
};

// Checkpoints in the backend's snapshot slots.
class BackendCheckpoints final : public CheckpointStore {
public:
    explicit BackendCheckpoints(ModelBackend & backend) : backend_(backend) {}

    int granularity() const override;
    int restore_point_spacing() const override;
    bool prefill_cuts_at_restore_points() const override;
    size_t estimate_bytes(int tokens) const override;
    bool used(Checkpoint checkpoint) const override;
    int position(Checkpoint checkpoint) const override;
    void free(Checkpoint checkpoint) override;
    Checkpoint staging() const override;
    ModelBackend::SnapshotRef ref(Checkpoint checkpoint) const override;
    OwnedSnapshot detach(Checkpoint checkpoint) override;
    bool adopt(Checkpoint checkpoint, OwnedSnapshot & snapshot) override;

    // Generation, for the single-sequence path. A request saves at most one
    // checkpoint during its prefill: `bind_capture` names it and its
    // position.
    void bind_capture(GenerateRequest & request, Checkpoint checkpoint, int position) const;
    GenerateResult restore_and_generate(Checkpoint checkpoint,
                                        const GenerateRequest & request,
                                        const DaemonIO & io);
    // How many prompt tokens a restore from `checkpoint` (holding the first
    // `tokens` of `prompt`) would skip; see ModelBackend::restorable_prefix.
    int restorable_prefix(const std::vector<int32_t> & prompt,
                          Checkpoint checkpoint, int tokens) const;

    // Save the live state between generations: now, or deferred while the
    // backend can keep the live state itself (ModelBackend::snapshot_save_deferred).
    bool save_now(Checkpoint checkpoint);
    bool save_deferred(Checkpoint checkpoint);
    void flush_deferred();

    // A concurrent capture's replaced payload (ModelBackend::snapshot_take_displaced).
    OwnedSnapshot take_displaced(Checkpoint checkpoint);

private:
    ModelBackend & backend_;
};

// What the concurrent path says to its SeqEngine about checkpoints. A
// Checkpoint's id is the PrefixStoreRef id the engine knows.
class EngineCheckpoints {
public:
    EngineCheckpoints(SeqEngine & engine, BackendCheckpoints & slots)
        : engine_(engine), slots_(slots) {}

    // The engine takes prefix plans at all.
    bool supported() const;
    // Its checkpoints live in the backend's snapshot slots, where the disk
    // cache can write and read them (slots()).
    bool in_backend_slots() const;
    size_t estimate_bytes(int tokens) const;

    static PrefixStoreRef ref(Checkpoint checkpoint, int tokens) {
        return {checkpoint.id, tokens};
    }
    static Checkpoint of(PrefixStoreRef ref) {
        return Checkpoint{(uint32_t)ref.id};
    }
    // Frees the payload only while it still holds `tokens` positions.
    void discard(PrefixStoreRef ref);
    void discard(Checkpoint checkpoint, int tokens) { discard(ref(checkpoint, tokens)); }

    Checkpoint staging() const { return slots_.staging(); }
    BackendCheckpoints & slots() { return slots_; }

private:
    SeqEngine & engine_;
    BackendCheckpoints & slots_;
};

}  // namespace luce::common
