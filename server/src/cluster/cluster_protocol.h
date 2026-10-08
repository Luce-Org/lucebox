// cluster_protocol.h - control-channel wire protocol between cluster ranks.
//
// The control channel is a plain TCP connection from every worker to the head
// (rank 0). It carries small, latency-tolerant messages: bootstrap (Hello /
// Welcome with the RCCL unique id), one RequestDescriptor per generation, the
// per-step decisions the head makes on behalf of all ranks (Decision for AR,
// Draft + Accept for DSpark), backend mutations that must happen on every
// rank (BackendOp), heartbeats, aborts and end-of-request telemetry.
//
// Bulk tensor traffic (the per-layer partial-sum reduction) never travels on
// this channel; that is RCCL over RoCE (cluster_comm.h).
//
// Encoding: every frame is a 12-byte little-endian header followed by the
// payload. Integers are little-endian; strings and vectors carry a u32 length
// prefix. The encoding is deliberately hand-rolled so the protocol test can
// run without ggml, HIP or RCCL and so a wire dump stays readable with xxd.
//
// Compatibility rule: bump kProtocolVersion on any field change; the head
// refuses a Hello whose protocol version differs.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace luce::cluster {

inline constexpr uint32_t kFrameMagic      = 0x4C43424Cu;  // "LBCL"
// 2: RequestMsg carries the inline snapshot position and separates the
//    restore slot from the inline save slot, so the prefix cache can be
//    replicated (a request may restore from one slot and check-point into
//    another; before this the workers never took the inline snapshot at all).
inline constexpr uint16_t kProtocolVersion = 2;
inline constexpr size_t   kFrameHeaderBytes = 12;
inline constexpr size_t   kMaxPayloadBytes  = 64ull * 1024ull * 1024ull;
inline constexpr size_t   kRcclUniqueIdBytes = 128;  // == NCCL_UNIQUE_ID_BYTES

enum class MsgType : uint16_t {
    Hello         = 1,   // worker -> head, once after connect
    Welcome       = 2,   // head -> worker, carries the RCCL unique id
    Heartbeat     = 3,   // both directions, every ~1 s while idle
    Abort         = 4,   // any -> all, request or process is being torn down
    Request       = 5,   // head -> workers, start of one generation
    Decision      = 6,   // head -> workers, AR: sampled token + stop flags
    Draft         = 7,   // head -> workers, DSpark: proposed tokens for verify
    Accept        = 8,   // head -> workers, DSpark: accepted count + bonus token
    HashProbe     = 9,   // worker -> head (and head local), determinism check
    RequestReport = 10,  // worker -> head, per-request telemetry
    BackendOp     = 11,  // head -> workers, snapshot/park/etc. replicated call
    RequestEnd    = 12,  // head -> workers, generation finished (normal path)
    Shutdown      = 13,  // head -> workers, orderly process exit
};

const char * msg_type_name(MsgType type);

struct FrameHeader {
    uint32_t magic       = kFrameMagic;
    uint16_t version     = kProtocolVersion;
    uint16_t type        = 0;
    uint32_t payload_len = 0;
};

struct Frame {
    MsgType              type = MsgType::Heartbeat;
    std::vector<uint8_t> payload;
};

// ─── Byte-level (de)serialization ───────────────────────────────────────

class ByteWriter {
public:
    void u8(uint8_t v);
    void u16(uint16_t v);
    void u32(uint32_t v);
    void u64(uint64_t v);
    void i32(int32_t v);
    void i64(int64_t v);
    void f32(float v);
    void str(const std::string & s);                 // u32 length + bytes
    void bytes(const void * data, size_t n);         // u32 length + bytes
    void vec_i32(const std::vector<int32_t> & v);    // u32 count + i32...
    void vec_i64(const std::vector<int64_t> & v);
    void vec_u64(const std::vector<uint64_t> & v);

    const std::vector<uint8_t> & data() const { return buf_; }
    std::vector<uint8_t> take() { return std::move(buf_); }
    size_t size() const { return buf_.size(); }

private:
    std::vector<uint8_t> buf_;
};

class ByteReader {
public:
    ByteReader(const uint8_t * data, size_t size) : data_(data), size_(size) {}
    explicit ByteReader(const std::vector<uint8_t> & v) : data_(v.data()), size_(v.size()) {}

    bool u8(uint8_t & v);
    bool u16(uint16_t & v);
    bool u32(uint32_t & v);
    bool u64(uint64_t & v);
    bool i32(int32_t & v);
    bool i64(int64_t & v);
    bool f32(float & v);
    bool str(std::string & s, size_t max_len = 1 << 20);
    bool bytes(std::vector<uint8_t> & out, size_t max_len = kMaxPayloadBytes);
    bool vec_i32(std::vector<int32_t> & v, size_t max_count = 1 << 22);
    bool vec_i64(std::vector<int64_t> & v, size_t max_count = 1 << 20);
    bool vec_u64(std::vector<uint64_t> & v, size_t max_count = 1 << 20);

    bool at_end() const { return pos_ == size_; }
    size_t remaining() const { return size_ - pos_; }
    bool failed() const { return failed_; }

private:
    const uint8_t * data_;
    size_t size_;
    size_t pos_ = 0;
    bool failed_ = false;
};

// Header encode/decode. decode_header validates magic and version and
// rejects payload_len > kMaxPayloadBytes.
void encode_header(const FrameHeader & h, uint8_t out[kFrameHeaderBytes]);
bool decode_header(const uint8_t in[kFrameHeaderBytes], FrameHeader & h, std::string * err);

// ─── Messages ───────────────────────────────────────────────────────────
// Every message has encode()/decode() and a static type(). decode() returns
// false on truncated or malformed payloads and never throws.

struct HelloMsg {
    int32_t  rank = -1;
    int32_t  size = 0;
    uint16_t protocol_version = kProtocolVersion;
    std::string build_sha;      // git describe of the binary
    std::string model_sha;      // sha256 of the target GGUF header or path hash
    uint64_t placement_hash = 0; // ClusterExpertPlacement::hash()
    std::string hostname;

    static MsgType type() { return MsgType::Hello; }
    void encode(ByteWriter & w) const;
    static bool decode(ByteReader & r, HelloMsg & out);
};

struct WelcomeMsg {
    int32_t  size = 0;
    std::array<uint8_t, kRcclUniqueIdBytes> rccl_unique_id{};
    uint64_t placement_hash = 0;
    uint32_t timeout_ms = 0;
    uint8_t  allreduce_dtype = 0;   // AllreduceDType as u8
    uint8_t  shared_expert   = 0;   // SharedExpertMode as u8
    int32_t  verify_hash_every = 0;
    std::vector<std::string> rank_hostnames;  // index = rank

    static MsgType type() { return MsgType::Welcome; }
    void encode(ByteWriter & w) const;
    static bool decode(ByteReader & r, WelcomeMsg & out);
};

struct HeartbeatMsg {
    int32_t  rank = -1;
    uint64_t mono_us = 0;  // sender's monotonic clock

    static MsgType type() { return MsgType::Heartbeat; }
    void encode(ByteWriter & w) const;
    static bool decode(ByteReader & r, HeartbeatMsg & out);
};

struct AbortMsg {
    int32_t  rank = -1;
    int32_t  code = 0;        // 0 = operator shutdown, >0 = failure class
    uint64_t request_id = 0;  // 0 = process-level
    std::string reason;

    static MsgType type() { return MsgType::Abort; }
    void encode(ByteWriter & w) const;
    static bool decode(ByteReader & r, AbortMsg & out);
};

// Decode mode requested for one generation.
enum class DecodeMode : uint8_t {
    Autoregressive = 0,
    Speculative    = 1,  // DSpark: head drafts, all ranks verify
};

// Everything a worker needs to run the identical generate() call. Field
// names mirror the head's GenerateRequest/SamplerCfg; the head decorator
// fills this before invoking the inner DeepSeek4Backend::generate.
struct RequestMsg {
    uint64_t request_id = 0;
    std::vector<int32_t> prompt_tokens;
    int32_t n_gen = 0;
    int32_t max_ctx = 0;
    // Sampler configuration (rank 0 is the only rank that samples; workers
    // still need these so the AR-vs-spec routing decision is identical).
    float    temperature = 0.0f;
    float    top_p = 1.0f;
    int32_t  top_k = 0;
    float    min_p = 0.0f;
    float    repeat_penalty = 1.0f;
    uint64_t seed = 0;
    DecodeMode decode_mode = DecodeMode::Autoregressive;
    bool     force_ar = false;
    // Prefix-cache state. The two halves are independent: a request may
    // resume from `restore_slot` (its KV holds `kv_offset` tokens) and still
    // be asked to write an inline snapshot into `snapshot_slot` once prefill
    // reaches `snapshot_pos`. -1 means "none" for both slots.
    int32_t  restore_slot = -1;
    int32_t  kv_offset = 0;
    int32_t  snapshot_slot = -1;
    int32_t  snapshot_pos = -1;
    std::vector<int32_t> stop_token_ids;     // budget-hook close tokens etc.

    static MsgType type() { return MsgType::Request; }
    void encode(ByteWriter & w) const;
    static bool decode(ByteReader & r, RequestMsg & out);
};

enum DecisionFlags : uint8_t {
    kDecisionEos      = 1u << 0,
    kDecisionStop     = 1u << 1,  // stop string / stop token matched on head
    kDecisionCancel   = 1u << 2,  // client disconnected / cancelled
    kDecisionBudget   = 1u << 3,  // budget hook forced a close token
    kDecisionFinal    = 1u << 4,  // last decision of this request
};

// AR decode: the token rank 0 sampled for this step, plus termination flags.
struct DecisionMsg {
    uint64_t request_id = 0;
    uint32_t step = 0;
    int32_t  token = -1;
    uint8_t  flags = 0;

    static MsgType type() { return MsgType::Decision; }
    void encode(ByteWriter & w) const;
    static bool decode(ByteReader & r, DecisionMsg & out);
};

// DSpark: tokens the head drafted for this verify step. tokens.size() == q
// (seed + q-1 drafted), pos is the target position of tokens[0].
struct DraftMsg {
    uint64_t request_id = 0;
    uint32_t step = 0;
    int32_t  pos = 0;
    std::vector<int32_t> tokens;

    static MsgType type() { return MsgType::Draft; }
    void encode(ByteWriter & w) const;
    static bool decode(ByteReader & r, DraftMsg & out);
};

// DSpark: verification outcome computed on rank 0. accept = number of
// drafted tokens accepted, bonus = the target's token after the accepted
// prefix (becomes next step's seed), flags as DecisionFlags.
struct AcceptMsg {
    uint64_t request_id = 0;
    uint32_t step = 0;
    int32_t  accept = 0;
    int32_t  bonus = -1;
    uint8_t  flags = 0;

    static MsgType type() { return MsgType::Accept; }
    void encode(ByteWriter & w) const;
    static bool decode(ByteReader & r, AcceptMsg & out);
};

// Determinism probe (--cluster-verify-hash n).
struct HashProbeMsg {
    uint64_t request_id = 0;
    uint32_t step = 0;
    int32_t  rank = -1;
    uint64_t hc_state_hash = 0;   // FNV-1a over the final-layer HC state bytes
    uint64_t token_hash = 0;      // FNV-1a over accepted token ids so far
    int32_t  first_divergent_layer = -1;  // filled only under LUCE_CLUSTER_TRACE

    static MsgType type() { return MsgType::HashProbe; }
    void encode(ByteWriter & w) const;
    static bool decode(ByteReader & r, HashProbeMsg & out);
};

// Per-request telemetry sent by every worker after RequestEnd; the head
// merges it into usage.timings.cluster.
struct RequestReportMsg {
    uint64_t request_id = 0;
    int32_t  rank = -1;
    uint32_t steps = 0;
    uint64_t compute_us = 0;
    uint64_t allreduce_calls = 0;
    uint64_t allreduce_bytes = 0;
    uint64_t allreduce_wait_us = 0;
    uint64_t ctrl_wait_us = 0;
    uint64_t peak_device_bytes = 0;

    static MsgType type() { return MsgType::RequestReport; }
    void encode(ByteWriter & w) const;
    static bool decode(ByteReader & r, RequestReportMsg & out);
};

// Backend mutations that must be mirrored on every rank so the replicated
// KV/snapshot state stays identical.
enum class BackendOpKind : uint8_t {
    SnapshotSave      = 1,
    SnapshotFree      = 2,
    SnapshotRestore   = 3,
    Park              = 4,
    Unpark            = 5,
    FreeDrafter       = 6,
    ResetRequestState = 7,
    HandleCompress    = 8,
};

struct BackendOpMsg {
    uint64_t request_id = 0;
    BackendOpKind kind = BackendOpKind::ResetRequestState;
    std::vector<int64_t> args;   // op-specific scalar arguments

    static MsgType type() { return MsgType::BackendOp; }
    void encode(ByteWriter & w) const;
    static bool decode(ByteReader & r, BackendOpMsg & out);
};

struct RequestEndMsg {
    uint64_t request_id = 0;
    uint32_t steps = 0;
    int32_t  status = 0;   // 0 = ok, otherwise GenerateErrorCode as int

    static MsgType type() { return MsgType::RequestEnd; }
    void encode(ByteWriter & w) const;
    static bool decode(ByteReader & r, RequestEndMsg & out);
};

struct ShutdownMsg {
    int32_t reason = 0;

    static MsgType type() { return MsgType::Shutdown; }
    void encode(ByteWriter & w) const;
    static bool decode(ByteReader & r, ShutdownMsg & out);
};

// ─── Frame helpers ──────────────────────────────────────────────────────

template <typename Msg>
Frame make_frame(const Msg & msg) {
    ByteWriter w;
    msg.encode(w);
    Frame f;
    f.type = Msg::type();
    f.payload = w.take();
    return f;
}

template <typename Msg>
bool parse_frame(const Frame & f, Msg & out) {
    if (f.type != Msg::type()) return false;
    ByteReader r(f.payload);
    if (!Msg::decode(r, out)) return false;
    return r.at_end();
}

// FNV-1a 64-bit, used for placement hashes and determinism probes.
uint64_t fnv1a64(const void * data, size_t n, uint64_t seed = 0xcbf29ce484222325ull);

}  // namespace luce::cluster
