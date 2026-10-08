// cluster_protocol.cpp - hand-rolled little-endian wire encoding for the
// control channel (cluster_protocol.h). No exceptions: every decode returns
// false on truncation, over-long lengths or malformed enums.

#include "cluster/cluster_protocol.h"

#include <cstring>

namespace luce::cluster {

// ─── ByteWriter ─────────────────────────────────────────────────────────

void ByteWriter::u8(uint8_t v) {
    buf_.push_back(v);
}

void ByteWriter::u16(uint16_t v) {
    buf_.push_back((uint8_t)(v & 0xFFu));
    buf_.push_back((uint8_t)((v >> 8) & 0xFFu));
}

void ByteWriter::u32(uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        buf_.push_back((uint8_t)((v >> (8 * i)) & 0xFFu));
    }
}

void ByteWriter::u64(uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        buf_.push_back((uint8_t)((v >> (8 * i)) & 0xFFu));
    }
}

void ByteWriter::i32(int32_t v) {
    u32((uint32_t)v);
}

void ByteWriter::i64(int64_t v) {
    u64((uint64_t)v);
}

void ByteWriter::f32(float v) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(v), "float must be 32-bit");
    std::memcpy(&bits, &v, sizeof(bits));
    u32(bits);
}

void ByteWriter::str(const std::string & s) {
    u32((uint32_t)s.size());
    buf_.insert(buf_.end(), s.begin(), s.end());
}

void ByteWriter::bytes(const void * data, size_t n) {
    u32((uint32_t)n);
    if (n > 0) {
        const uint8_t * p = (const uint8_t *)data;
        buf_.insert(buf_.end(), p, p + n);
    }
}

void ByteWriter::vec_i32(const std::vector<int32_t> & v) {
    u32((uint32_t)v.size());
    for (int32_t x : v) i32(x);
}

void ByteWriter::vec_i64(const std::vector<int64_t> & v) {
    u32((uint32_t)v.size());
    for (int64_t x : v) i64(x);
}

void ByteWriter::vec_u64(const std::vector<uint64_t> & v) {
    u32((uint32_t)v.size());
    for (uint64_t x : v) u64(x);
}

// ─── ByteReader ─────────────────────────────────────────────────────────

bool ByteReader::u8(uint8_t & v) {
    if (failed_ || size_ - pos_ < 1) { failed_ = true; return false; }
    v = data_[pos_++];
    return true;
}

bool ByteReader::u16(uint16_t & v) {
    if (failed_ || size_ - pos_ < 2) { failed_ = true; return false; }
    v = (uint16_t)data_[pos_] | (uint16_t)((uint16_t)data_[pos_ + 1] << 8);
    pos_ += 2;
    return true;
}

bool ByteReader::u32(uint32_t & v) {
    if (failed_ || size_ - pos_ < 4) { failed_ = true; return false; }
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        r |= (uint32_t)data_[pos_ + (size_t)i] << (8 * i);
    }
    pos_ += 4;
    v = r;
    return true;
}

bool ByteReader::u64(uint64_t & v) {
    if (failed_ || size_ - pos_ < 8) { failed_ = true; return false; }
    uint64_t r = 0;
    for (int i = 0; i < 8; ++i) {
        r |= (uint64_t)data_[pos_ + (size_t)i] << (8 * i);
    }
    pos_ += 8;
    v = r;
    return true;
}

bool ByteReader::i32(int32_t & v) {
    uint32_t u = 0;
    if (!u32(u)) return false;
    v = (int32_t)u;
    return true;
}

bool ByteReader::i64(int64_t & v) {
    uint64_t u = 0;
    if (!u64(u)) return false;
    v = (int64_t)u;
    return true;
}

bool ByteReader::f32(float & v) {
    uint32_t bits = 0;
    if (!u32(bits)) return false;
    std::memcpy(&v, &bits, sizeof(v));
    return true;
}

bool ByteReader::str(std::string & s, size_t max_len) {
    uint32_t n = 0;
    if (!u32(n)) return false;
    if (n > max_len || size_ - pos_ < n) { failed_ = true; return false; }
    s.assign((const char *)data_ + pos_, n);
    pos_ += n;
    return true;
}

bool ByteReader::bytes(std::vector<uint8_t> & out, size_t max_len) {
    uint32_t n = 0;
    if (!u32(n)) return false;
    if (n > max_len || size_ - pos_ < n) { failed_ = true; return false; }
    out.assign(data_ + pos_, data_ + pos_ + n);
    pos_ += n;
    return true;
}

bool ByteReader::vec_i32(std::vector<int32_t> & v, size_t max_count) {
    uint32_t n = 0;
    if (!u32(n)) return false;
    if (n > max_count || size_ - pos_ < (size_t)n * 4) { failed_ = true; return false; }
    v.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        if (!i32(v[i])) return false;
    }
    return true;
}

bool ByteReader::vec_i64(std::vector<int64_t> & v, size_t max_count) {
    uint32_t n = 0;
    if (!u32(n)) return false;
    if (n > max_count || size_ - pos_ < (size_t)n * 8) { failed_ = true; return false; }
    v.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        if (!i64(v[i])) return false;
    }
    return true;
}

bool ByteReader::vec_u64(std::vector<uint64_t> & v, size_t max_count) {
    uint32_t n = 0;
    if (!u32(n)) return false;
    if (n > max_count || size_ - pos_ < (size_t)n * 8) { failed_ = true; return false; }
    v.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        if (!u64(v[i])) return false;
    }
    return true;
}

// ─── Header ─────────────────────────────────────────────────────────────

void encode_header(const FrameHeader & h, uint8_t out[kFrameHeaderBytes]) {
    ByteWriter w;
    w.u32(h.magic);
    w.u16(h.version);
    w.u16(h.type);
    w.u32(h.payload_len);
    std::memcpy(out, w.data().data(), kFrameHeaderBytes);
}

bool decode_header(const uint8_t in[kFrameHeaderBytes], FrameHeader & h, std::string * err) {
    ByteReader r(in, kFrameHeaderBytes);
    FrameHeader tmp;
    if (!r.u32(tmp.magic) || !r.u16(tmp.version) || !r.u16(tmp.type) || !r.u32(tmp.payload_len)) {
        if (err) *err = "frame header truncated";
        return false;
    }
    if (tmp.magic != kFrameMagic) {
        if (err) *err = "bad frame magic";
        return false;
    }
    if (tmp.version != kProtocolVersion) {
        if (err) {
            *err = "protocol version mismatch (peer " + std::to_string(tmp.version) +
                   ", local " + std::to_string(kProtocolVersion) + ")";
        }
        return false;
    }
    if (tmp.payload_len > kMaxPayloadBytes) {
        if (err) *err = "frame payload too large (" + std::to_string(tmp.payload_len) + " bytes)";
        return false;
    }
    h = tmp;
    return true;
}

const char * msg_type_name(MsgType type) {
    switch (type) {
        case MsgType::Hello:         return "Hello";
        case MsgType::Welcome:       return "Welcome";
        case MsgType::Heartbeat:     return "Heartbeat";
        case MsgType::Abort:         return "Abort";
        case MsgType::Request:       return "Request";
        case MsgType::Decision:      return "Decision";
        case MsgType::Draft:         return "Draft";
        case MsgType::Accept:        return "Accept";
        case MsgType::HashProbe:     return "HashProbe";
        case MsgType::RequestReport: return "RequestReport";
        case MsgType::BackendOp:     return "BackendOp";
        case MsgType::RequestEnd:    return "RequestEnd";
        case MsgType::Shutdown:      return "Shutdown";
    }
    return "Unknown";
}

// ─── Messages ───────────────────────────────────────────────────────────

void HelloMsg::encode(ByteWriter & w) const {
    w.i32(rank);
    w.i32(size);
    w.u16(protocol_version);
    w.str(build_sha);
    w.str(model_sha);
    w.u64(placement_hash);
    w.str(hostname);
}

bool HelloMsg::decode(ByteReader & r, HelloMsg & out) {
    return r.i32(out.rank) && r.i32(out.size) && r.u16(out.protocol_version) &&
           r.str(out.build_sha, 4096) && r.str(out.model_sha, 4096) &&
           r.u64(out.placement_hash) && r.str(out.hostname, 1024);
}

void WelcomeMsg::encode(ByteWriter & w) const {
    w.i32(size);
    w.bytes(rccl_unique_id.data(), rccl_unique_id.size());
    w.u64(placement_hash);
    w.u32(timeout_ms);
    w.u8(allreduce_dtype);
    w.u8(shared_expert);
    w.i32(verify_hash_every);
    w.u32((uint32_t)rank_hostnames.size());
    for (const std::string & h : rank_hostnames) w.str(h);
}

bool WelcomeMsg::decode(ByteReader & r, WelcomeMsg & out) {
    if (!r.i32(out.size)) return false;
    std::vector<uint8_t> id;
    if (!r.bytes(id, kRcclUniqueIdBytes)) return false;
    if (id.size() != kRcclUniqueIdBytes) return false;
    std::memcpy(out.rccl_unique_id.data(), id.data(), kRcclUniqueIdBytes);
    if (!r.u64(out.placement_hash) || !r.u32(out.timeout_ms) || !r.u8(out.allreduce_dtype) ||
        !r.u8(out.shared_expert) || !r.i32(out.verify_hash_every)) {
        return false;
    }
    uint32_t n = 0;
    if (!r.u32(n)) return false;
    if (n > 256) return false;
    out.rank_hostnames.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        if (!r.str(out.rank_hostnames[i], 1024)) return false;
    }
    return true;
}

void HeartbeatMsg::encode(ByteWriter & w) const {
    w.i32(rank);
    w.u64(mono_us);
}

bool HeartbeatMsg::decode(ByteReader & r, HeartbeatMsg & out) {
    return r.i32(out.rank) && r.u64(out.mono_us);
}

void AbortMsg::encode(ByteWriter & w) const {
    w.i32(rank);
    w.i32(code);
    w.u64(request_id);
    w.str(reason);
}

bool AbortMsg::decode(ByteReader & r, AbortMsg & out) {
    return r.i32(out.rank) && r.i32(out.code) && r.u64(out.request_id) &&
           r.str(out.reason, 1 << 16);
}

void RequestMsg::encode(ByteWriter & w) const {
    w.u64(request_id);
    w.vec_i32(prompt_tokens);
    w.i32(n_gen);
    w.i32(max_ctx);
    w.f32(temperature);
    w.f32(top_p);
    w.i32(top_k);
    w.f32(min_p);
    w.f32(repeat_penalty);
    w.u64(seed);
    w.u8((uint8_t)decode_mode);
    w.u8(force_ar ? 1 : 0);
    w.i32(restore_slot);
    w.i32(kv_offset);
    w.i32(snapshot_slot);
    w.i32(snapshot_pos);
    w.vec_i32(stop_token_ids);
}

bool RequestMsg::decode(ByteReader & r, RequestMsg & out) {
    uint8_t mode = 0;
    uint8_t force = 0;
    if (!r.u64(out.request_id) || !r.vec_i32(out.prompt_tokens) || !r.i32(out.n_gen) ||
        !r.i32(out.max_ctx) || !r.f32(out.temperature) || !r.f32(out.top_p) ||
        !r.i32(out.top_k) || !r.f32(out.min_p) || !r.f32(out.repeat_penalty) ||
        !r.u64(out.seed) || !r.u8(mode) || !r.u8(force) || !r.i32(out.restore_slot) ||
        !r.i32(out.kv_offset) || !r.i32(out.snapshot_slot) || !r.i32(out.snapshot_pos) ||
        !r.vec_i32(out.stop_token_ids, 1 << 16)) {
        return false;
    }
    if (mode > (uint8_t)DecodeMode::Speculative) return false;
    if (force > 1) return false;
    out.decode_mode = (DecodeMode)mode;
    out.force_ar = force != 0;
    return true;
}

void DecisionMsg::encode(ByteWriter & w) const {
    w.u64(request_id);
    w.u32(step);
    w.i32(token);
    w.u8(flags);
}

bool DecisionMsg::decode(ByteReader & r, DecisionMsg & out) {
    return r.u64(out.request_id) && r.u32(out.step) && r.i32(out.token) && r.u8(out.flags);
}

void DraftMsg::encode(ByteWriter & w) const {
    w.u64(request_id);
    w.u32(step);
    w.i32(pos);
    w.vec_i32(tokens);
}

bool DraftMsg::decode(ByteReader & r, DraftMsg & out) {
    return r.u64(out.request_id) && r.u32(out.step) && r.i32(out.pos) &&
           r.vec_i32(out.tokens, 1 << 12);
}

void AcceptMsg::encode(ByteWriter & w) const {
    w.u64(request_id);
    w.u32(step);
    w.i32(accept);
    w.i32(bonus);
    w.u8(flags);
}

bool AcceptMsg::decode(ByteReader & r, AcceptMsg & out) {
    return r.u64(out.request_id) && r.u32(out.step) && r.i32(out.accept) &&
           r.i32(out.bonus) && r.u8(out.flags);
}

void HashProbeMsg::encode(ByteWriter & w) const {
    w.u64(request_id);
    w.u32(step);
    w.i32(rank);
    w.u64(hc_state_hash);
    w.u64(token_hash);
    w.i32(first_divergent_layer);
}

bool HashProbeMsg::decode(ByteReader & r, HashProbeMsg & out) {
    return r.u64(out.request_id) && r.u32(out.step) && r.i32(out.rank) &&
           r.u64(out.hc_state_hash) && r.u64(out.token_hash) &&
           r.i32(out.first_divergent_layer);
}

void RequestReportMsg::encode(ByteWriter & w) const {
    w.u64(request_id);
    w.i32(rank);
    w.u32(steps);
    w.u64(compute_us);
    w.u64(allreduce_calls);
    w.u64(allreduce_bytes);
    w.u64(allreduce_wait_us);
    w.u64(ctrl_wait_us);
    w.u64(peak_device_bytes);
}

bool RequestReportMsg::decode(ByteReader & r, RequestReportMsg & out) {
    return r.u64(out.request_id) && r.i32(out.rank) && r.u32(out.steps) &&
           r.u64(out.compute_us) && r.u64(out.allreduce_calls) &&
           r.u64(out.allreduce_bytes) && r.u64(out.allreduce_wait_us) &&
           r.u64(out.ctrl_wait_us) && r.u64(out.peak_device_bytes);
}

void BackendOpMsg::encode(ByteWriter & w) const {
    w.u64(request_id);
    w.u8((uint8_t)kind);
    w.vec_i64(args);
}

bool BackendOpMsg::decode(ByteReader & r, BackendOpMsg & out) {
    uint8_t kind = 0;
    if (!r.u64(out.request_id) || !r.u8(kind) || !r.vec_i64(out.args, 1 << 12)) {
        return false;
    }
    if (kind < (uint8_t)BackendOpKind::SnapshotSave || kind > (uint8_t)BackendOpKind::HandleCompress) {
        return false;
    }
    out.kind = (BackendOpKind)kind;
    return true;
}

void RequestEndMsg::encode(ByteWriter & w) const {
    w.u64(request_id);
    w.u32(steps);
    w.i32(status);
}

bool RequestEndMsg::decode(ByteReader & r, RequestEndMsg & out) {
    return r.u64(out.request_id) && r.u32(out.steps) && r.i32(out.status);
}

void ShutdownMsg::encode(ByteWriter & w) const {
    w.i32(reason);
}

bool ShutdownMsg::decode(ByteReader & r, ShutdownMsg & out) {
    return r.i32(out.reason);
}

// ─── Hash ───────────────────────────────────────────────────────────────

uint64_t fnv1a64(const void * data, size_t n, uint64_t seed) {
    const uint8_t * p = (const uint8_t *)data;
    uint64_t h = seed;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t)p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

}  // namespace luce::cluster
