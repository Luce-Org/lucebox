# One SSD session cache for every model

A design spec for extending the disk session cache (`server/src/server/kv/`,
described in `server/docs/PREFIX_CACHE.md`, "Disk Session Cache") so that
the models and serving paths that ran without it keep their sessions on the
SSD, and so that several `luce_server` processes on one machine share one
cache directory with one budget.

## 1. Before

- The file format is model-agnostic: a snapshot's tensors are stored verbatim
  with the key tokens, an auxiliary blob and a checksum, in
  `<dir>/<layout>/<hash>.dkv`. A layout fingerprint (tensor names, types and
  row shapes) and an identity salt (model file, `--max-ctx`, chat template,
  KV types) keep one model from loading another's files.
- It reaches a model through the backend's snapshot slots
  (`BackendCheckpoints`: `snapshot_ref`, `snapshot_detach`, `snapshot_adopt`).
- One process per directory: the first server took an exclusive lock on
  `<dir>/.lock`; a second server on the same directory ran without a disk
  cache.
- Off for: the concurrent `qwen4exp` engine (its checkpoints lived in its own
  store), the single-sequence `qwen4exp` backend (no `snapshot_ref`), and the
  concurrent DeepSeek4 engine.

## 2. Engines on the SSD

The concurrent `qwen4exp` engine keeps its checkpoints in the backend's
snapshot slots (`Qwen4ExpSeqEngine::use_backend_snapshots`), as the qwen35
engine does: checkpoint id N is snapshot slot N-1, and the disk-staging id is
the last slot. The disk reader and writer that serve qwen35 then serve it
unchanged.

- A capture keeps the checkpoint it replaces aside
  (`ModelBackend::snapshot_take_displaced`) until the scheduler claims it:
  written when the entry is worth keeping, freed otherwise
  (`snapshot_drop_displaced`, so a backend that packs its snapshots does not
  copy one for nothing). One the scheduler never claims goes at the next
  step. A failed capture leaves the checkpoint it would have replaced.
- `qwen4exp` snapshots are strips of a slot cache plus host fields, not plain
  tensors, so the backend packs them (`snapshot_exports`, `snapshot_copy`,
  `snapshot_detach`, `snapshot_adopt`): the snapshot buffer as one I8 tensor
  named after the cache's strip layout, and an I32 metadata tensor with the
  positions, tokens, draft-layer state, logits and each strip's place. Both
  have their length in dimension 1, which the layout fingerprint ignores, so
  the fingerprint is the same at every position and the file format does not
  change.
- A snapshot in system memory (a CPU buffer) is handed to the writer without
  a copy; the metadata gets a small buffer of its own (`OwnedSnapshot` can
  own a second buffer; it moved to `common/owned_snapshot.h`). A disk read is
  checked strip by strip against the live cache and kept in place.

## 3. One directory, several servers

- **Locks.** Each server holds a shared lock on `<dir>/.lock` for its
  lifetime. An older server, which takes it exclusively, keeps new servers
  off and is kept off by them. Each model identity is held by one server
  (`<dir>/.identity-<salt>.lock`), so no two servers write the same file,
  pins file or views.
- **Publish.** Before a write is renamed into place, the writer thread takes
  `<dir>/.index.lock`, merges the directory into its index (other servers'
  new files, the ones they deleted, last-used times and pins from their
  headers; ids and queued writes kept), applies the TTL and evicts the least
  recently used files of any model until the directory fits, then renames.
  Only the writer thread deletes other servers' files, right after the merge.
- **Queueing.** A queued write evicts only this process's files; the publish
  makes the rest of the room or drops the file when pins fill the budget.
- **Reads.** A read sizes the file it opened (`fstat`): another server's
  delete leaves an open file readable. A file deleted before the open is a
  quiet miss and leaves the index.
- **Temporaries.** A write holds a lock on its temporary file until the
  rename; start-up removes only temporaries nobody holds.
- **Budget and TTL.** They are the directory's: each server keeps the whole
  directory within its own `--kv-cache-budget` and `--kv-cache-ttl`, so with
  different values the smaller one wins once that server writes. The
  directory exceeds the budget by at most the files being written.
- Windows keeps one server per directory.

## 4. Not covered

The concurrent DeepSeek4 engine keeps its own checkpoint type; moving it into
the backend's snapshot slots is the same change and is left to a follow-up
tested on its machine.

## 5. Measurement

On R9700 + Ryzen AI MAX+: a 4-slot `qwen4exp` server and a Qwen 3.8 27B
server sharing one directory. For a long conversation on each: a cold
prefill, a RAM hit, a restore from the SSD after a RAM eviction and after a
restart (page cache dropped first), the greedy text compared, the bytes and
write times, and a small budget that makes one server evict the other's
files.
