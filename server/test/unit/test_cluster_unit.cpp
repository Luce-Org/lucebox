// GPU-free unit tests for the expert-parallel cluster runtime.
//
// Covers the control-channel wire protocol (cluster_protocol.h), the CLI
// config helpers (cluster_config.h), the N-rank expert placement
// (cluster_expert_placement.h) and the rank-0-decides hooks
// (cluster_decision_hooks.h) driven over a real loopback control channel.
// None of this needs ggml, HIP or RCCL: the target links only the cluster
// sources plus moe_hybrid_placement / moe_hybrid_routing_stats / sampler.
//
// Build: cmake --build . --target test_cluster_unit
// Run:   ./test_cluster_unit
//
// main() comes from test/test_unit_main.cpp (CMake adds it, like every
// target in _new_cppunit_test_targets).

#include "CppUnitTestFramework.hpp"

#include "cluster/cluster_config.h"
#include "cluster/cluster_decision_hooks.h"
#include "cluster/cluster_expert_placement.h"
#include "cluster/cluster_protocol.h"
#include "common/moe_hybrid_placement.h"
#include "common/moe_hybrid_routing_stats.h"

#if !defined(_WIN32)
#include "cluster/cluster_control.h"
#include <atomic>
#include <thread>
#include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

using namespace CppUnitTestFramework;
using namespace luce::cluster;

namespace {

// DeepSeek V4 Flash shape used throughout the placement tests.
constexpr int kLayers = 43;
constexpr int kExperts = 256;
constexpr int kTopK = 6;
constexpr int kSlots = 8;

luce::common::MoeHybridConfig ds4_cfg() {
    luce::common::MoeHybridConfig cfg;
    cfg.n_embd = 4096;
    cfg.n_expert = kExperts;
    cfg.n_expert_used = kTopK;
    cfg.n_ff_exp = 2048;
    cfg.n_layer = kLayers;
    cfg.first_moe_layer = 0;
    return cfg;
}

// Helpers that assert (REQUIRE/CHECK) must be fixture members: the framework's
// HandleAssert is a CommonFixture method, so free functions cannot call it.
struct ClusterUnitFixture : CommonFixture {
    using CommonFixture::CommonFixture;

    // Hotness that is adversarial to e % N: every fourth expert is hot, so
    // uniform piles all heat onto rank 0 for N = 2 and N = 4.
    luce::common::MoeHybridRoutingStats skewed_stats() {
        luce::common::MoeHybridRoutingStats s;
        REQUIRE(s.init(kLayers, kExperts, kTopK));
        for (int l = 0; l < kLayers; l++) {
            uint64_t total = 0;
            for (int e = 0; e < kExperts; e++) {
                const uint64_t c = (e % 4 == 0) ? 1000u + (uint64_t) (kExperts - e) : 1u + (uint64_t) (e % 7);
                s.counts[(size_t) l * kExperts + e] = c;
                total += c;
            }
            s.layer_totals[(size_t) l] = total;
        }
        return s;
    }

    // Round-trips `msg` through encode/decode and checks that every strict
    // prefix of the payload is rejected and that trailing bytes fail parse_frame.
    template <typename Msg>
    void check_roundtrip_and_truncation(const Msg & in, Msg & out) {
        Frame f = make_frame(in);
        REQUIRE(f.type == Msg::type());
        REQUIRE(parse_frame(f, out));

        for (size_t cut = 0; cut < f.payload.size(); cut++) {
            Frame t;
            t.type = f.type;
            t.payload.assign(f.payload.begin(), f.payload.begin() + (std::ptrdiff_t) cut);
            Msg scratch;
            CHECK_FALSE(parse_frame(t, scratch));
        }
        Frame g = f;
        g.payload.push_back(0xAA);
        Msg scratch;
        CHECK_FALSE(parse_frame(g, scratch));

        Frame wrong = f;
        wrong.type = (f.type == MsgType::Heartbeat) ? MsgType::Shutdown : MsgType::Heartbeat;
        CHECK_FALSE(parse_frame(wrong, scratch));
    }
};

// Per-layer per-rank load from stats over owned experts.
std::vector<uint64_t> rank_loads(const ClusterExpertPlacement & p,
                                 const luce::common::MoeHybridRoutingStats & s, int layer) {
    std::vector<uint64_t> load((size_t) p.n_ranks, 0);
    for (int e = 0; e < p.n_expert; e++) {
        const int32_t o = p.owner_of(layer, e);
        if (o >= 0) load[(size_t) o] += s.count(layer, e);
    }
    return load;
}

double max_over_mean(const std::vector<uint64_t> & load) {
    const uint64_t mx = *std::max_element(load.begin(), load.end());
    const double mean = (double) std::accumulate(load.begin(), load.end(), (uint64_t) 0) /
                        (double) load.size();
    return mean > 0 ? (double) mx / mean : 0.0;
}

}  // namespace

// ─── Protocol: bytes and header ─────────────────────────────────────────

TEST_CASE(ClusterUnitFixture, ByteWriterReaderRoundTrip) {
    ByteWriter w;
    w.u8(0xAB);
    w.u16(0xBEEF);
    w.u32(0xDEADBEEFu);
    w.u64(0x0123456789ABCDEFull);
    w.i32(-17);
    w.i64(-9000000000ll);
    w.f32(1.5f);
    w.str("hello");
    w.str("");
    const uint8_t raw[3] = {1, 2, 3};
    w.bytes(raw, 3);
    w.vec_i32({});
    w.vec_i32({-1, 0, 1});
    w.vec_i64({-5, 5});
    w.vec_u64({7});

    ByteReader r(w.data());
    uint8_t a; uint16_t b; uint32_t c; uint64_t d; int32_t e; int64_t f; float g;
    std::string s1, s2;
    std::vector<uint8_t> bytes;
    std::vector<int32_t> v0, v1;
    std::vector<int64_t> v2;
    std::vector<uint64_t> v3;
    REQUIRE(r.u8(a));  CHECK_EQUAL(a, 0xAB);
    REQUIRE(r.u16(b)); CHECK_EQUAL(b, 0xBEEF);
    REQUIRE(r.u32(c)); CHECK_EQUAL(c, 0xDEADBEEFu);
    REQUIRE(r.u64(d)); CHECK_EQUAL(d, 0x0123456789ABCDEFull);
    REQUIRE(r.i32(e)); CHECK_EQUAL(e, -17);
    REQUIRE(r.i64(f)); CHECK_EQUAL(f, -9000000000ll);
    REQUIRE(r.f32(g)); CHECK_EQUAL(g, 1.5f);
    REQUIRE(r.str(s1)); CHECK_EQUAL(s1, std::string("hello"));
    REQUIRE(r.str(s2)); CHECK(s2.empty());
    REQUIRE(r.bytes(bytes)); REQUIRE_EQUAL(bytes.size(), (size_t) 3); CHECK_EQUAL(bytes[2], 3);
    REQUIRE(r.vec_i32(v0)); CHECK(v0.empty());
    REQUIRE(r.vec_i32(v1)); REQUIRE_EQUAL(v1.size(), (size_t) 3); CHECK_EQUAL(v1[0], -1);
    REQUIRE(r.vec_i64(v2)); REQUIRE_EQUAL(v2.size(), (size_t) 2); CHECK_EQUAL(v2[0], -5);
    REQUIRE(r.vec_u64(v3)); REQUIRE_EQUAL(v3.size(), (size_t) 1); CHECK_EQUAL(v3[0], 7u);
    CHECK(r.at_end());
    CHECK_FALSE(r.failed());

    // Reading past the end fails and latches.
    uint32_t extra = 0;
    CHECK_FALSE(r.u32(extra));
    CHECK(r.failed());
}

TEST_CASE(ClusterUnitFixture, ByteReaderRejectsOversizedVectors) {
    ByteWriter w;
    w.u32(1000);  // declares 1000 elements, none follow
    ByteReader r(w.data());
    std::vector<int32_t> v;
    CHECK_FALSE(r.vec_i32(v));

    ByteWriter w2;
    w2.u32(10);
    w2.u8(1);
    ByteReader r2(w2.data());
    std::string s;
    CHECK_FALSE(r2.str(s, /*max_len=*/4));  // exceeds the caller's cap
}

TEST_CASE(ClusterUnitFixture, FrameHeaderEncodeDecode) {
    FrameHeader h;
    h.type = (uint16_t) MsgType::Decision;
    h.payload_len = 17;
    uint8_t buf[kFrameHeaderBytes];
    encode_header(h, buf);
    // Little-endian magic "LBCL" == 0x4C43424C.
    CHECK_EQUAL(buf[0], 0x4C); CHECK_EQUAL(buf[1], 0x42); CHECK_EQUAL(buf[2], 0x43); CHECK_EQUAL(buf[3], 0x4C);

    FrameHeader out;
    std::string err;
    REQUIRE(decode_header(buf, out, &err));
    CHECK_EQUAL(out.magic, kFrameMagic);
    CHECK_EQUAL(out.version, kProtocolVersion);
    CHECK_EQUAL(out.type, (uint16_t) MsgType::Decision);
    CHECK_EQUAL(out.payload_len, 17u);

    uint8_t bad_magic[kFrameHeaderBytes];
    std::memcpy(bad_magic, buf, sizeof(buf));
    bad_magic[0] ^= 0xFF;
    CHECK_FALSE(decode_header(bad_magic, out, &err));
    CHECK_FALSE(err.empty());

    FrameHeader v = h;
    v.version = kProtocolVersion + 1;
    encode_header(v, buf);
    CHECK_FALSE(decode_header(buf, out, &err));

    FrameHeader big = h;
    big.payload_len = (uint32_t) kMaxPayloadBytes + 1;
    encode_header(big, buf);
    CHECK_FALSE(decode_header(buf, out, &err));

    FrameHeader max_ok = h;
    max_ok.payload_len = (uint32_t) kMaxPayloadBytes;
    encode_header(max_ok, buf);
    CHECK(decode_header(buf, out, &err));
}

TEST_CASE(ClusterUnitFixture, Fnv1a64KnownVectors) {
    // Reference values of the 64-bit FNV-1a function.
    CHECK_EQUAL(fnv1a64(nullptr, 0), 0xcbf29ce484222325ull);
    CHECK_EQUAL(fnv1a64("a", 1), 0xaf63dc4c8601ec8cull);
    CHECK_EQUAL(fnv1a64("foobar", 6), 0x85944171f73967e8ull);
    // Chaining with the previous hash as seed equals hashing the concatenation.
    const uint64_t h1 = fnv1a64("foo", 3);
    CHECK_EQUAL(fnv1a64("bar", 3, h1), fnv1a64("foobar", 6));

    const std::vector<int32_t> toks = {1, 2, 3};
    CHECK_EQUAL(hash_tokens(toks), fnv1a64(toks.data(), toks.size() * sizeof(int32_t)));
    CHECK_EQUAL(hash_tokens({}), fnv1a64(nullptr, 0));
    const float fl[2] = {1.0f, -2.0f};
    CHECK_EQUAL(hash_floats(fl, 2), fnv1a64(fl, sizeof(fl)));
    CHECK_EQUAL(hash_floats(nullptr, 0), fnv1a64(nullptr, 0));
}

TEST_CASE(ClusterUnitFixture, MsgTypeNamesAreDistinct) {
    std::set<std::string> names;
    for (uint16_t t = 1; t <= 13; t++) {
        const char * n = msg_type_name((MsgType) t);
        REQUIRE_NOT_NULL(n);
        names.insert(n);
    }
    CHECK_EQUAL(names.size(), (size_t) 13);
}

// ─── Protocol: messages ─────────────────────────────────────────────────

TEST_CASE(ClusterUnitFixture, HelloWelcomeRoundTrip) {
    HelloMsg in;
    in.rank = 3;
    in.size = 4;
    in.build_sha = "v1.2.3-7-gabcdef";
    in.model_sha = std::string(64, 'f');
    in.placement_hash = 0x1122334455667788ull;
    in.hostname = "halo-3";
    HelloMsg out;
    check_roundtrip_and_truncation(in, out);
    CHECK_EQUAL(out.rank, 3);
    CHECK_EQUAL(out.size, 4);
    CHECK_EQUAL(out.protocol_version, kProtocolVersion);
    CHECK_EQUAL(out.build_sha, in.build_sha);
    CHECK_EQUAL(out.model_sha, in.model_sha);
    CHECK_EQUAL(out.placement_hash, in.placement_hash);
    CHECK_EQUAL(out.hostname, in.hostname);

    HelloMsg empty;  // defaults with empty strings still round-trip
    HelloMsg empty_out;
    check_roundtrip_and_truncation(empty, empty_out);
    CHECK_EQUAL(empty_out.rank, -1);
    CHECK(empty_out.build_sha.empty());

    WelcomeMsg w;
    w.size = 4;
    for (size_t i = 0; i < kRcclUniqueIdBytes; i++) w.rccl_unique_id[i] = (uint8_t) (i * 7);
    w.placement_hash = 42;
    w.timeout_ms = 30000;
    w.allreduce_dtype = 2;
    w.shared_expert = 1;
    w.verify_hash_every = 8;
    w.rank_hostnames = {"halo-0", "halo-1", "", "halo-3"};
    WelcomeMsg wo;
    check_roundtrip_and_truncation(w, wo);
    CHECK_EQUAL(wo.size, 4);
    CHECK(wo.rccl_unique_id == w.rccl_unique_id);
    CHECK_EQUAL(wo.placement_hash, 42u);
    CHECK_EQUAL(wo.timeout_ms, 30000u);
    CHECK_EQUAL(wo.allreduce_dtype, 2);
    CHECK_EQUAL(wo.shared_expert, 1);
    CHECK_EQUAL(wo.verify_hash_every, 8);
    REQUIRE_EQUAL(wo.rank_hostnames.size(), (size_t) 4);
    CHECK(wo.rank_hostnames[2].empty());
    CHECK_EQUAL(wo.rank_hostnames[3], std::string("halo-3"));

    WelcomeMsg w_empty;
    WelcomeMsg w_empty_out;
    check_roundtrip_and_truncation(w_empty, w_empty_out);
    CHECK(w_empty_out.rank_hostnames.empty());
}

TEST_CASE(ClusterUnitFixture, HeartbeatAbortShutdownRoundTrip) {
    HeartbeatMsg hb;
    hb.rank = 2;
    hb.mono_us = 123456789012ull;
    HeartbeatMsg hbo;
    check_roundtrip_and_truncation(hb, hbo);
    CHECK_EQUAL(hbo.rank, 2);
    CHECK_EQUAL(hbo.mono_us, hb.mono_us);

    AbortMsg ab;
    ab.rank = 1;
    ab.code = 3;
    ab.request_id = 99;
    ab.reason = "worker generate failed while head succeeded";
    AbortMsg abo;
    check_roundtrip_and_truncation(ab, abo);
    CHECK_EQUAL(abo.rank, 1);
    CHECK_EQUAL(abo.code, 3);
    CHECK_EQUAL(abo.request_id, 99u);
    CHECK_EQUAL(abo.reason, ab.reason);

    AbortMsg ab_empty;
    AbortMsg ab_empty_out;
    check_roundtrip_and_truncation(ab_empty, ab_empty_out);
    CHECK(ab_empty_out.reason.empty());

    ShutdownMsg sd;
    sd.reason = 7;
    ShutdownMsg sdo;
    check_roundtrip_and_truncation(sd, sdo);
    CHECK_EQUAL(sdo.reason, 7);
}

TEST_CASE(ClusterUnitFixture, RequestRoundTripEmptyAndLarge) {
    RequestMsg in;
    in.request_id = 7;
    in.n_gen = 512;
    in.max_ctx = 32768;
    in.temperature = 0.7f;
    in.top_p = 0.95f;
    in.top_k = 40;
    in.min_p = 0.05f;
    in.repeat_penalty = 1.1f;
    in.seed = 0xDEADBEEFCAFEBABEull;
    in.decode_mode = DecodeMode::Speculative;
    in.force_ar = true;
    in.restore_slot = 5;
    in.kv_offset = 1024;
    in.snapshot_slot = 3;
    in.snapshot_pos = 2048;
    RequestMsg out;

    SECTION("empty prompt and stop tokens") {
        check_roundtrip_and_truncation(in, out);
        CHECK(out.prompt_tokens.empty());
        CHECK(out.stop_token_ids.empty());
        CHECK_EQUAL(out.request_id, 7u);
        CHECK_EQUAL(out.n_gen, 512);
        CHECK_EQUAL(out.max_ctx, 32768);
        CHECK_EQUAL(out.temperature, 0.7f);
        CHECK_EQUAL(out.top_p, 0.95f);
        CHECK_EQUAL(out.top_k, 40);
        CHECK_EQUAL(out.min_p, 0.05f);
        CHECK_EQUAL(out.repeat_penalty, 1.1f);
        CHECK_EQUAL(out.seed, in.seed);
        CHECK(out.decode_mode == DecodeMode::Speculative);
        CHECK(out.force_ar);
        // Restore and inline save are independent halves; a request that
        // resumes from one slot and check-points into another must survive
        // the wire with both, or the workers' slots drift from the head's.
        CHECK_EQUAL(out.restore_slot, 5);
        CHECK_EQUAL(out.snapshot_slot, 3);
        CHECK_EQUAL(out.snapshot_pos, 2048);
        CHECK_EQUAL(out.kv_offset, 1024);
    }
    SECTION("128k prompt") {
        in.prompt_tokens.resize(131072);
        for (size_t i = 0; i < in.prompt_tokens.size(); i++) in.prompt_tokens[i] = (int32_t) (i * 31 % 129280);
        in.stop_token_ids = {1718, 37947, 32};
        Frame f = make_frame(in);
        REQUIRE(parse_frame(f, out));
        CHECK(out.prompt_tokens == in.prompt_tokens);
        CHECK(out.stop_token_ids == in.stop_token_ids);
        CHECK(f.payload.size() < kMaxPayloadBytes);
        // Truncating inside the token vector must fail cleanly.
        Frame t = f;
        t.payload.resize(t.payload.size() / 2);
        RequestMsg scratch;
        CHECK_FALSE(parse_frame(t, scratch));
    }
    SECTION("defaults") {
        RequestMsg d;
        RequestMsg d_out;
        check_roundtrip_and_truncation(d, d_out);
        CHECK(d_out.decode_mode == DecodeMode::Autoregressive);
        CHECK_FALSE(d_out.force_ar);
        CHECK_EQUAL(d_out.snapshot_slot, -1);
    }
}

TEST_CASE(ClusterUnitFixture, DecisionDraftAcceptRoundTrip) {
    DecisionMsg d;
    d.request_id = 5;
    d.step = 4095;
    d.token = 129279;
    d.flags = kDecisionEos | kDecisionFinal;
    DecisionMsg d_out;
    check_roundtrip_and_truncation(d, d_out);
    CHECK_EQUAL(d_out.request_id, 5u);
    CHECK_EQUAL(d_out.step, 4095u);
    CHECK_EQUAL(d_out.token, 129279);
    CHECK_EQUAL(d_out.flags, (uint8_t) (kDecisionEos | kDecisionFinal));

    DecisionMsg neg;
    neg.token = -1;
    neg.flags = kDecisionCancel;
    DecisionMsg neg_out;
    check_roundtrip_and_truncation(neg, neg_out);
    CHECK_EQUAL(neg_out.token, -1);

    DraftMsg dr;
    dr.request_id = 5;
    dr.step = 9;
    dr.pos = 2047;
    dr.tokens = {11, 22, 33, 44, 55};
    DraftMsg dr_out;
    check_roundtrip_and_truncation(dr, dr_out);
    CHECK_EQUAL(dr_out.pos, 2047);
    CHECK(dr_out.tokens == dr.tokens);

    DraftMsg dr_empty;
    DraftMsg dr_empty_out;
    check_roundtrip_and_truncation(dr_empty, dr_empty_out);
    CHECK(dr_empty_out.tokens.empty());

    AcceptMsg ac;
    ac.request_id = 5;
    ac.step = 9;
    ac.accept = 3;
    ac.bonus = 777;
    ac.flags = kDecisionStop;
    AcceptMsg ac_out;
    check_roundtrip_and_truncation(ac, ac_out);
    CHECK_EQUAL(ac_out.accept, 3);
    CHECK_EQUAL(ac_out.bonus, 777);
    CHECK_EQUAL(ac_out.flags, (uint8_t) kDecisionStop);
}

TEST_CASE(ClusterUnitFixture, HashProbeReportBackendOpRequestEndRoundTrip) {
    HashProbeMsg hp;
    hp.request_id = 1;
    hp.step = 8;
    hp.rank = 2;
    hp.hc_state_hash = 0xAAAAAAAAAAAAAAAAull;
    hp.token_hash = 0x5555555555555555ull;
    hp.first_divergent_layer = 17;
    HashProbeMsg hp_out;
    check_roundtrip_and_truncation(hp, hp_out);
    CHECK_EQUAL(hp_out.rank, 2);
    CHECK_EQUAL(hp_out.hc_state_hash, hp.hc_state_hash);
    CHECK_EQUAL(hp_out.token_hash, hp.token_hash);
    CHECK_EQUAL(hp_out.first_divergent_layer, 17);

    RequestReportMsg rr;
    rr.request_id = 1;
    rr.rank = 1;
    rr.steps = 512;
    rr.compute_us = 20'000'000;
    rr.allreduce_calls = 43 * 512;
    rr.allreduce_bytes = 16384ull * 43 * 512;
    rr.allreduce_wait_us = 1'500'000;
    rr.ctrl_wait_us = 400'000;
    rr.peak_device_bytes = 90ull << 30;
    RequestReportMsg rr_out;
    check_roundtrip_and_truncation(rr, rr_out);
    CHECK_EQUAL(rr_out.steps, 512u);
    CHECK_EQUAL(rr_out.allreduce_bytes, rr.allreduce_bytes);
    CHECK_EQUAL(rr_out.peak_device_bytes, rr.peak_device_bytes);

    BackendOpMsg op;
    op.request_id = 0;
    op.kind = BackendOpKind::Park;
    op.args = {2};
    BackendOpMsg op_out;
    check_roundtrip_and_truncation(op, op_out);
    CHECK(op_out.kind == BackendOpKind::Park);
    REQUIRE_EQUAL(op_out.args.size(), (size_t) 1);
    CHECK_EQUAL(op_out.args[0], (int64_t) 2);

    BackendOpMsg op_noargs;
    op_noargs.kind = BackendOpKind::FreeDrafter;
    BackendOpMsg op_noargs_out;
    check_roundtrip_and_truncation(op_noargs, op_noargs_out);
    CHECK(op_noargs_out.args.empty());

    RequestEndMsg re;
    re.request_id = 1;
    re.steps = 12;
    re.status = 6;
    RequestEndMsg re_out;
    check_roundtrip_and_truncation(re, re_out);
    CHECK_EQUAL(re_out.steps, 12u);
    CHECK_EQUAL(re_out.status, 6);
}

TEST_CASE(ClusterUnitFixture, DecisionFlagHelpers) {
    CHECK_EQUAL(make_decision_flags(false, false, false, false, false), (uint8_t) 0);
    CHECK_EQUAL(make_decision_flags(true, false, false, false, false), (uint8_t) kDecisionEos);
    CHECK_EQUAL(make_decision_flags(false, true, true, true, true),
                (uint8_t) (kDecisionStop | kDecisionCancel | kDecisionBudget | kDecisionFinal));
    CHECK_FALSE(decision_terminates(0));
    CHECK_FALSE(decision_terminates(kDecisionBudget));
    CHECK(decision_terminates(kDecisionEos));
    CHECK(decision_terminates(kDecisionCancel));
    CHECK(decision_terminates(kDecisionFinal));
}

// ─── Config ─────────────────────────────────────────────────────────────

TEST_CASE(ClusterUnitFixture, ClusterConfigValidate) {
    ClusterConfig c;
    CHECK_FALSE(c.enabled());
    CHECK(c.validate().empty());  // "not a cluster run" is valid

    c.size = 2;
    c.rank = 0;
    c.head_host = "10.0.0.1";
    CHECK(c.validate().empty());
    CHECK(c.is_head());
    CHECK_FALSE(c.is_worker());

    c.rank = 1;
    CHECK(c.validate().empty());
    CHECK(c.is_worker());

    ClusterConfig bad = c;
    bad.rank = 2;  // == size
    CHECK_FALSE(bad.validate().empty());
    bad = c; bad.rank = -1;
    CHECK_FALSE(bad.validate().empty());
    bad = c; bad.size = 1;
    CHECK_FALSE(bad.validate().empty());
    bad = c; bad.size = kClusterMaxSize + 1;
    CHECK_FALSE(bad.validate().empty());
    bad = c; bad.head_host.clear();
    CHECK_FALSE(bad.validate().empty());
    bad = c; bad.head_port = 0;
    CHECK_FALSE(bad.validate().empty());
    bad = c; bad.replicate_hot = -1;
    CHECK_FALSE(bad.validate().empty());
    bad = c; bad.timeout_ms = 0;
    CHECK_FALSE(bad.validate().empty());
    bad = c; bad.gid_index = -2;
    CHECK_FALSE(bad.validate().empty());
    bad = c; bad.placement_source = PlacementSource::File; bad.placement_file.clear();
    CHECK_FALSE(bad.validate().empty());
    bad.placement_file = "placement.json";
    CHECK(bad.validate().empty());

    ClusterConfig eight = c;
    eight.size = kClusterMaxSize;
    eight.rank = kClusterMaxSize - 1;
    CHECK(eight.validate().empty());
}

TEST_CASE(ClusterUnitFixture, ClusterConfigParsers) {
    std::string host;
    int port = 0;
    std::string err;
    REQUIRE(parse_host_port("10.0.0.1:9400", host, port, &err));
    CHECK_EQUAL(host, std::string("10.0.0.1"));
    CHECK_EQUAL(port, 9400);
    REQUIRE(parse_host_port("halo-0:1", host, port, &err));
    CHECK_EQUAL(host, std::string("halo-0"));
    CHECK_EQUAL(port, 1);
    CHECK_FALSE(parse_host_port("", host, port, &err));
    // A bare host takes the default control port.
    REQUIRE(parse_host_port("10.0.0.1", host, port, &err));
    CHECK_EQUAL(port, kClusterDefaultPort);
    CHECK_FALSE(parse_host_port(":9400", host, port, &err));
    CHECK_FALSE(parse_host_port("10.0.0.1:", host, port, &err));
    CHECK_FALSE(parse_host_port("10.0.0.1:abc", host, port, &err));
    CHECK_FALSE(parse_host_port("10.0.0.1:70000", host, port, &err));
    CHECK_FALSE(parse_host_port("10.0.0.1:0", host, port, &err));

    SharedExpertMode se;
    REQUIRE(parse_shared_expert_mode("replicate", se, &err)); CHECK(se == SharedExpertMode::Replicate);
    REQUIRE(parse_shared_expert_mode("shard", se, &err));     CHECK(se == SharedExpertMode::Shard);
    REQUIRE(parse_shared_expert_mode("rank0", se, &err));     CHECK(se == SharedExpertMode::Rank0);
    CHECK_FALSE(parse_shared_expert_mode("bogus", se, &err));
    CHECK_FALSE(parse_shared_expert_mode("", se, &err));
    CHECK_EQUAL(std::string(shared_expert_mode_name(SharedExpertMode::Shard)), std::string("shard"));

    AllreduceDType dt;
    REQUIRE(parse_allreduce_dtype("f32", dt, &err));  CHECK(dt == AllreduceDType::F32);
    REQUIRE(parse_allreduce_dtype("bf16", dt, &err)); CHECK(dt == AllreduceDType::BF16);
    REQUIRE(parse_allreduce_dtype("auto", dt, &err)); CHECK(dt == AllreduceDType::Auto);
    CHECK_FALSE(parse_allreduce_dtype("fp16", dt, &err));
    CHECK_EQUAL(std::string(allreduce_dtype_name(AllreduceDType::BF16)), std::string("bf16"));

    PlacementSource ps;
    std::string file;
    REQUIRE(parse_placement_source("uniform", ps, file, &err));
    CHECK(ps == PlacementSource::Uniform);
    CHECK(file.empty());
    REQUIRE(parse_placement_source("balanced", ps, file, &err));
    CHECK(ps == PlacementSource::Balanced);
    REQUIRE(parse_placement_source("/tmp/placement.json", ps, file, &err));
    CHECK(ps == PlacementSource::File);
    CHECK_EQUAL(file, std::string("/tmp/placement.json"));
    CHECK_FALSE(parse_placement_source("random", ps, file, &err));
    CHECK_FALSE(parse_placement_source("placement.yaml", ps, file, &err));
    CHECK_FALSE(parse_placement_source("", ps, file, &err));
    CHECK_EQUAL(std::string(placement_source_name(PlacementSource::Balanced)), std::string("balanced"));
}

// ─── Expert placement ───────────────────────────────────────────────────

TEST_CASE(ClusterUnitFixture, UniformPlacementExactlyOneOwnerPerSlot) {
    for (int n : {2, 3, 4}) {
        ClusterExpertPlacement p;
        std::string err;
        REQUIRE(ClusterExpertPlacement::build_uniform(n, ds4_cfg(), 0, nullptr, p, &err));
        REQUIRE(p.valid(&err));
        CHECK(p.matches(ds4_cfg()));
        CHECK_EQUAL(p.n_ranks, n);
        CHECK_EQUAL(p.n_layer, kLayers);
        CHECK_EQUAL(p.n_expert, kExperts);
        CHECK_EQUAL(p.owner.size(), (size_t) kLayers * kExperts);

        // owner_of follows e % N and nothing is replicated.
        for (int l = 0; l < kLayers; l++) {
            for (int e = 0; e < kExperts; e++) {
                CHECK_EQUAL(p.owner_of(l, e), (int32_t) (e % n));
                CHECK_FALSE(p.is_replicated(l, e));
            }
        }

        // Exactly one rank evaluates every (layer, expert, slot).
        std::vector<std::vector<uint8_t>> validity((size_t) n);
        for (int r = 0; r < n; r++) {
            p.slot_validity(r, kSlots, validity[(size_t) r]);
            REQUIRE_EQUAL(validity[(size_t) r].size(), (size_t) kLayers * kSlots * kExperts);
        }
        for (int l = 0; l < kLayers; l++) {
            for (int s = 0; s < kSlots; s++) {
                for (int e = 0; e < kExperts; e++) {
                    int owners = 0;
                    for (int r = 0; r < n; r++) {
                        owners += validity[(size_t) r][((size_t) l * kSlots + s) * kExperts + e] ? 1 : 0;
                    }
                    REQUIRE_EQUAL(owners, 1);
                    const int32_t so = p.slot_owner(l, e, s);
                    REQUIRE(so >= 0 && so < n);
                    REQUIRE(validity[(size_t) so][((size_t) l * kSlots + s) * kExperts + e] != 0);
                }
            }
        }

        // Shard sizes within +-1 of n_expert / n per layer, and in total.
        for (int l = 0; l < kLayers; l++) {
            int sum = 0;
            for (int r = 0; r < n; r++) {
                const int c = p.owned_count(l, r);
                CHECK(std::abs(c - kExperts / n) <= 1);
                sum += c;
            }
            CHECK_EQUAL(sum, kExperts);
        }
        int total = 0;
        for (int r = 0; r < n; r++) total += p.total_owned(r);
        CHECK_EQUAL(total, kLayers * kExperts);
    }
}

TEST_CASE(ClusterUnitFixture, ReplicateHotMarksHottestAndKeepsOneOwnerPerSlot) {
    const auto stats = skewed_stats();
    const int n = 4;
    const int k = 8;
    ClusterExpertPlacement p;
    std::string err;
    REQUIRE(ClusterExpertPlacement::build_uniform(n, ds4_cfg(), k, &stats, p, &err));
    REQUIRE(p.valid(&err));
    CHECK_EQUAL(p.replicate_hot, k);

    for (int l = 0; l < kLayers; l++) {
        const std::vector<int> hottest = stats.hot_experts(l, k);
        REQUIRE_EQUAL(hottest.size(), (size_t) k);
        int replicated = 0;
        for (int e = 0; e < kExperts; e++) {
            if (p.is_replicated(l, e)) {
                replicated++;
                CHECK(std::find(hottest.begin(), hottest.end(), e) != hottest.end());
            }
        }
        CHECK_EQUAL(replicated, k);

        // Replicated experts rotate over ranks by slot; owned ones do not.
        for (int e : hottest) {
            std::set<int32_t> seen;
            for (int s = 0; s < kSlots; s++) {
                const int32_t so = p.slot_owner(l, e, s);
                CHECK_EQUAL(so, (int32_t) ((e + s) % n));
                seen.insert(so);
            }
            CHECK_EQUAL(seen.size(), (size_t) n);
        }
    }

    // One evaluator per (layer, expert, slot) still holds with replication.
    std::vector<std::vector<uint8_t>> validity((size_t) n);
    for (int r = 0; r < n; r++) p.slot_validity(r, kSlots, validity[(size_t) r]);
    for (int l = 0; l < kLayers; l++) {
        for (int s = 0; s < kSlots; s++) {
            for (int e = 0; e < kExperts; e++) {
                int owners = 0;
                for (int r = 0; r < n; r++) {
                    owners += validity[(size_t) r][((size_t) l * kSlots + s) * kExperts + e] ? 1 : 0;
                }
                REQUIRE_EQUAL(owners, 1);
            }
        }
    }

    // Resident sets: owned + all replicated, sorted ascending.
    for (int r = 0; r < n; r++) {
        const std::vector<int32_t> res = p.resident_experts(0, r);
        CHECK(std::is_sorted(res.begin(), res.end()));
        CHECK_EQUAL((int) res.size(), p.owned_count(0, r) + k);
    }
}

TEST_CASE(ClusterUnitFixture, BalancedPlacementBeatsUniformOnSkewedStats) {
    const auto stats = skewed_stats();
    for (int n : {2, 4}) {
        ClusterExpertPlacement uni, bal;
        std::string err;
        REQUIRE(ClusterExpertPlacement::build_uniform(n, ds4_cfg(), 0, nullptr, uni, &err));
        REQUIRE(ClusterExpertPlacement::build_balanced(n, stats, 0, bal, &err));
        REQUIRE(bal.valid(&err));
        CHECK(bal.matches(ds4_cfg()));

        double worst_uni = 0.0, worst_bal = 0.0;
        for (int l = 0; l < kLayers; l++) {
            const double u = max_over_mean(rank_loads(uni, stats, l));
            const double b = max_over_mean(rank_loads(bal, stats, l));
            worst_uni = std::max(worst_uni, u);
            worst_bal = std::max(worst_bal, b);
            CHECK(b <= u);
            // Memory balance constraint: shard sizes stay within +-1.
            for (int r = 0; r < n; r++) {
                CHECK(std::abs(bal.owned_count(l, r) - kExperts / n) <= 1);
            }
        }
        // Uniform sends every hot expert (e % 4 == 0) to rank 0.
        CHECK(worst_uni > 1.5);
        CHECK(worst_bal < worst_uni);
        CHECK(worst_bal < 1.1);
        CHECK(bal.hash() != uni.hash());
    }
}

TEST_CASE(ClusterUnitFixture, PlacementJsonRoundTripAndHash) {
    const auto stats = skewed_stats();
    ClusterExpertPlacement p;
    std::string err;
    REQUIRE(ClusterExpertPlacement::build_uniform(4, ds4_cfg(), 2, &stats, p, &err));
    const uint64_t h0 = p.hash();
    CHECK(h0 != 0u);

    const long long stamp = (long long) std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("cluster_placement_test_" + std::to_string(stamp) + ".json");
    REQUIRE(p.save_json(path.string(), &err));
    ClusterExpertPlacement q;
    REQUIRE(ClusterExpertPlacement::load_json(path.string(), q, &err));
    std::filesystem::remove(path);
    CHECK_EQUAL(q.n_ranks, p.n_ranks);
    CHECK_EQUAL(q.n_layer, p.n_layer);
    CHECK_EQUAL(q.n_expert, p.n_expert);
    CHECK_EQUAL(q.n_expert_used, p.n_expert_used);
    CHECK_EQUAL(q.replicate_hot, p.replicate_hot);
    CHECK(q.owner == p.owner);
    CHECK_EQUAL(q.hash(), h0);

    // Any ownership change changes the hash; the same content hashes equally.
    ClusterExpertPlacement r = p;
    CHECK_EQUAL(r.hash(), h0);
    const size_t idx = (size_t) 5 * kExperts + 17;
    r.owner[idx] = (r.owner[idx] + 1) % 4;
    CHECK(r.hash() != h0);
    r.owner[idx] = p.owner[idx];
    CHECK_EQUAL(r.hash(), h0);

    // Loading garbage fails cleanly.
    ClusterExpertPlacement bad;
    CHECK_FALSE(ClusterExpertPlacement::load_json("/nonexistent/dir/placement.json", bad, &err));
    CHECK_FALSE(err.empty());
    CHECK_FALSE(p.describe(&stats).empty());
}

TEST_CASE(ClusterUnitFixture, ToRankPlacementIsValidAndSorted) {
    const auto stats = skewed_stats();
    for (int n : {2, 3, 4}) {
        ClusterExpertPlacement p;
        std::string err;
        REQUIRE(ClusterExpertPlacement::build_uniform(n, ds4_cfg(), 4, &stats, p, &err));
        for (int r = 0; r < n; r++) {
            luce::common::MoeHybridPlacement mp;
            REQUIRE(p.to_rank_placement(r, mp, &err));
            REQUIRE(mp.valid(&err));
            CHECK(mp.matches(ds4_cfg()));
            CHECK_EQUAL(mp.n_layer, kLayers);
            CHECK_EQUAL(mp.n_expert, kExperts);
            REQUIRE_EQUAL(mp.hot_expert_ids.size(), (size_t) kLayers);
            REQUIRE_EQUAL(mp.hot_counts.size(), (size_t) kLayers);
            int total = 0;
            for (int l = 0; l < kLayers; l++) {
                const auto & ids = mp.hot_expert_ids[(size_t) l];
                CHECK(std::is_sorted(ids.begin(), ids.end()));
                CHECK(std::adjacent_find(ids.begin(), ids.end()) == ids.end());
                CHECK_EQUAL((int) ids.size(), mp.hot_counts[(size_t) l]);
                CHECK(ids == p.resident_experts(l, r));
                for (int32_t e : ids) CHECK(mp.is_hot(l, e));
                total += (int) ids.size();
            }
            CHECK_EQUAL(mp.total_hot, total);
        }
        luce::common::MoeHybridPlacement out_of_range;
        CHECK_FALSE(p.to_rank_placement(n, out_of_range, &err));
        CHECK_FALSE(err.empty());
    }
}

TEST_CASE(ClusterUnitFixture, PlacementBuildersRejectBadInput) {
    ClusterExpertPlacement p;
    std::string err;
    CHECK_FALSE(ClusterExpertPlacement::build_uniform(0, ds4_cfg(), 0, nullptr, p, &err));
    CHECK_FALSE(ClusterExpertPlacement::build_uniform(2, ds4_cfg(), -1, nullptr, p, &err));
    // n_ranks == 1 is a legal degenerate placement (loopback comm, size-1 runs).
    ClusterExpertPlacement single;
    CHECK(ClusterExpertPlacement::build_uniform(1, ds4_cfg(), 0, nullptr, single, &err));
    CHECK_EQUAL(single.owned_count(0, 0), kExperts);
    luce::common::MoeHybridConfig empty;
    CHECK_FALSE(ClusterExpertPlacement::build_uniform(2, empty, 0, nullptr, p, &err));
    // replicate_hot without stats cannot pick "hottest": either refused or zero replicated.
    ClusterExpertPlacement q;
    if (ClusterExpertPlacement::build_uniform(2, ds4_cfg(), 3, nullptr, q, &err)) {
        int replicated = 0;
        for (int32_t o : q.owner) replicated += (o == kReplicated) ? 1 : 0;
        CHECK_EQUAL(replicated, 0);
    }
    // replicate_hot >= n_expert leaves nothing to shard.
    const auto stats = skewed_stats();
    CHECK_FALSE(ClusterExpertPlacement::build_balanced(2, stats, kExperts, p, &err));
    luce::common::MoeHybridRoutingStats none;
    CHECK_FALSE(ClusterExpertPlacement::build_balanced(2, none, 0, p, &err));
}

// ─── Decision hooks ─────────────────────────────────────────────────────

TEST_CASE(ClusterUnitFixture, LocalHooksAreTransparent) {
    LocalHooks h;
    CHECK(h.is_head());
    CHECK_FALSE(h.is_cluster());
    CHECK_EQUAL(h.verify_hash_every(), 0);
    int32_t token = 123;
    uint8_t flags = kDecisionEos;
    luce::common::SamplerCfg cfg;
    std::string err;
    const float logits[3] = {0.f, 1.f, 0.f};
    CHECK(h.decide_next_token(1, 0, logits, 3, cfg, false, token, flags, &err));
    CHECK_EQUAL(token, 123);
    CHECK_EQUAL(flags, (uint8_t) kDecisionEos);
    std::vector<int32_t> draft = {1, 2};
    CHECK(h.decide_draft(1, 0, 0, draft, &err));
    CHECK_EQUAL(draft.size(), (size_t) 2);
    int32_t accept = 2, bonus = 9;
    CHECK(h.decide_accept(1, 0, accept, bonus, flags, &err));
    CHECK_EQUAL(accept, 2);
    CHECK_EQUAL(bonus, 9);
    CHECK(h.hash_probe(1, 0, 1, 2, &err));
    CHECK(h.backend_op(BackendOpMsg{}, &err));
    h.set_current_request(77);
    CHECK_EQUAL(h.current_request(), 77u);
    CHECK_EQUAL(h.counters().decisions, 0u);
}

#if !defined(_WIN32)

namespace {

// Head and worker on 127.0.0.1. The head picks a port from a small range
// (no ephemeral-port query on ClusterHeadControl); the worker retries.
struct LoopbackPair {
    ClusterHeadControl head;
    ClusterWorkerControl worker;
    HelloMsg head_id;
    HelloMsg worker_id;
    WelcomeMsg welcome;
    std::vector<HelloMsg> hellos;
    int port = 0;
    std::string head_err, worker_err;
    bool ok = false;

    LoopbackPair(const std::string & build = "test-build", uint64_t placement = 0x1234) {
        head_id.rank = 0; head_id.size = 2; head_id.build_sha = build;
        head_id.model_sha = "model"; head_id.placement_hash = placement; head_id.hostname = "head";
        worker_id = head_id; worker_id.rank = 1; worker_id.hostname = "worker";

        ControlTimeouts t;
        t.connect_ms = 1000;
        t.connect_retries = 20;
        t.recv_ms = 5000;
        head.set_timeouts(t);
        worker.set_timeouts(t);

        const int base = 39400 + (int) (getpid() % 500);
        for (int attempt = 0; attempt < 20 && port == 0; attempt++) {
            if (head.listen("127.0.0.1", base + attempt, &head_err)) port = base + attempt;
        }
    }

    // Runs the handshake with the worker on a second thread.
    bool handshake(const HelloMsg & wid) {
        if (port == 0) return false;
        WelcomeMsg send;
        send.size = 2;
        send.timeout_ms = 5000;
        send.verify_hash_every = 4;
        send.rank_hostnames = {"head", "worker"};
        std::atomic<bool> w_ok{false};
        std::thread wt([&] {
            w_ok = worker.connect_and_handshake("127.0.0.1", port, wid, welcome, &worker_err);
        });
        const bool h_ok = head.accept_workers(1, head_id, 10000, hellos, &head_err);
        bool sent = false;
        if (h_ok) sent = head.broadcast(make_frame(send), &head_err);
        wt.join();
        ok = h_ok && sent && w_ok;
        return ok;
    }
};

}  // namespace

TEST_CASE(ClusterUnitFixture, ControlHandshakeAcceptsMatchingIdentity) {
    LoopbackPair lp;
    if (lp.port == 0) SKIP("no loopback port available");
    REQUIRE(lp.handshake(lp.worker_id));
    REQUIRE_EQUAL(lp.hellos.size(), (size_t) 1);
    CHECK_EQUAL(lp.hellos[0].rank, 1);
    CHECK_EQUAL(lp.hellos[0].hostname, std::string("worker"));
    CHECK_EQUAL(lp.welcome.size, 2);
    CHECK_EQUAL(lp.welcome.verify_hash_every, 4);
    CHECK_EQUAL(lp.head.n_workers(), 1);
    CHECK(lp.head.all_alive());
    CHECK(lp.worker.alive());
    lp.worker.close();
    lp.head.close();
}

TEST_CASE(ClusterUnitFixture, ControlHandshakeRefusesMismatch) {
    SECTION("placement hash") {
        LoopbackPair lp;
        if (lp.port == 0) SKIP("no loopback port available");
        HelloMsg wid = lp.worker_id;
        wid.placement_hash ^= 1;
        CHECK_FALSE(lp.handshake(wid));
        CHECK_FALSE(lp.head_err.empty());
    }
    SECTION("build sha") {
        LoopbackPair lp;
        if (lp.port == 0) SKIP("no loopback port available");
        HelloMsg wid = lp.worker_id;
        wid.build_sha = "other";
        CHECK_FALSE(lp.handshake(wid));
    }
    SECTION("rank out of range") {
        LoopbackPair lp;
        if (lp.port == 0) SKIP("no loopback port available");
        HelloMsg wid = lp.worker_id;
        wid.rank = 5;
        CHECK_FALSE(lp.handshake(wid));
    }
}

TEST_CASE(ClusterUnitFixture, HeadAndWorkerHooksOverLoopback) {
    LoopbackPair lp;
    if (lp.port == 0) SKIP("no loopback port available");
    REQUIRE(lp.handshake(lp.worker_id));

    HeadHooksConfig hc;
    hc.timeout_ms = 5000;
    hc.verify_hash_every = 4;
    HeadHooks head(lp.head, hc);
    WorkerHooksConfig wc;
    wc.rank = 1;
    wc.timeout_ms = 5000;
    wc.verify_hash_every = 4;
    WorkerHooks worker(lp.worker, wc);
    CHECK(head.is_head());  CHECK(head.is_cluster());
    CHECK_FALSE(worker.is_head()); CHECK(worker.is_cluster());
    CHECK_EQUAL(worker.verify_hash_every(), 4);

    const uint64_t req = 42;
    head.set_current_request(req);
    worker.set_current_request(req);
    luce::common::SamplerCfg scfg;
    const float logits[4] = {0.1f, 0.9f, 0.2f, 0.3f};

    SECTION("decision, draft, accept, hash probe and backend op") {
        std::string w_err;
        int32_t w_tok = -1; uint8_t w_flags = 0;
        std::vector<int32_t> w_draft;
        int32_t w_accept = -1, w_bonus = -1; uint8_t w_aflags = 0;
        bool w_dec = false, w_dr = false, w_ac = false, w_hp = false, w_hp2 = false;
        std::thread wt([&] {
            w_dec = worker.decide_next_token(req, 0, nullptr, 0, scfg, false, w_tok, w_flags, &w_err);
            w_dr  = worker.decide_draft(req, 1, 100, w_draft, &w_err);
            w_ac  = worker.decide_accept(req, 1, w_accept, w_bonus, w_aflags, &w_err);
            w_hp  = worker.hash_probe(req, 4, 0xAAAA, 0xBBBB, &w_err);   // matches
            w_hp2 = worker.hash_probe(req, 8, 0xAAAA, 0xCCCC, &w_err);   // token hash differs
        });

        std::string h_err;
        int32_t tok = 7; uint8_t flags = make_decision_flags(true, false, false, false, true);
        REQUIRE(head.decide_next_token(req, 0, logits, 4, scfg, true, tok, flags, &h_err));
        CHECK_EQUAL(tok, 7);  // caller's token is passed through untouched
        std::vector<int32_t> draft = {7, 8, 9, 10};
        REQUIRE(head.decide_draft(req, 1, 100, draft, &h_err));
        int32_t accept = 3, bonus = 11; uint8_t aflags = 0;
        REQUIRE(head.decide_accept(req, 1, accept, bonus, aflags, &h_err));
        REQUIRE(head.hash_probe(req, 4, 0xAAAA, 0xBBBB, &h_err));
        CHECK_EQUAL(head.counters().hash_mismatches, 0u);
        CHECK_EQUAL(head.first_mismatch_rank(), -1);
        REQUIRE(head.hash_probe(req, 8, 0xAAAA, 0xBBBB, &h_err));  // non-strict: logged, not fatal
        CHECK_EQUAL(head.counters().hash_mismatches, 1u);
        CHECK_EQUAL(head.first_mismatch_rank(), 1);
        CHECK_EQUAL(head.first_mismatch_step(), 8u);
        wt.join();

        CHECK(w_dec); CHECK_EQUAL(w_tok, 7); CHECK_EQUAL(w_flags, flags);
        CHECK(w_dr);  CHECK(w_draft == draft);
        CHECK(w_ac);  CHECK_EQUAL(w_accept, 3); CHECK_EQUAL(w_bonus, 11);
        CHECK(w_hp); CHECK(w_hp2);
        CHECK_EQUAL(head.counters().decisions, 1u);
        CHECK_EQUAL(head.counters().drafts, 1u);
        CHECK_EQUAL(head.counters().accepts, 1u);
        CHECK_EQUAL(head.counters().hash_probes, 2u);
        CHECK_EQUAL(worker.counters().decisions, 1u);
        CHECK_EQUAL(worker.counters().hash_probes, 2u);

        // Backend op reaches the worker as a plain frame (the worker main
        // loop executes it; WorkerHooks::backend_op is a no-op).
        BackendOpMsg op;
        op.kind = BackendOpKind::SnapshotFree;
        op.args = {5};
        REQUIRE(head.backend_op(op, &h_err));
        Frame f;
        REQUIRE(lp.worker.recv(f, 5000, &w_err));
        BackendOpMsg got;
        REQUIRE(parse_frame(f, got));
        CHECK(got.kind == BackendOpKind::SnapshotFree);
        CHECK(worker.backend_op(got, &w_err));
        CHECK_EQUAL(head.counters().backend_ops, 1u);

        // Head samples itself when the caller passes token < 0 (greedy).
        int32_t sampled = -1; uint8_t sflags = 0;
        std::thread wt2([&] {
            int32_t t = -1; uint8_t fl = 0; std::string e;
            worker.decide_next_token(req, 1, nullptr, 0, scfg, false, t, fl, &e);
        });
        REQUIRE(head.decide_next_token(req, 1, logits, 4, scfg, false, sampled, sflags, &h_err));
        CHECK_EQUAL(sampled, 1);
        wt2.join();
    }

    SECTION("worker rejects wrong request or step") {
        std::string w_err;
        int32_t w_tok = -1; uint8_t w_flags = 0;
        bool w_ok = true;
        std::thread wt([&] {
            w_ok = worker.decide_next_token(req, 3, nullptr, 0, scfg, false, w_tok, w_flags, &w_err);
        });
        std::string h_err;
        int32_t tok = 1; uint8_t flags = 0;
        REQUIRE(head.decide_next_token(req, 2, logits, 4, scfg, false, tok, flags, &h_err));  // step 2 != 3
        wt.join();
        CHECK_FALSE(w_ok);
        CHECK_FALSE(w_err.empty());
    }

    SECTION("unexpected RequestEnd is parked, Abort is fatal") {
        std::string w_err;
        int32_t w_tok = -1; uint8_t w_flags = 0;
        bool w_ok = true;
        std::thread wt([&] {
            w_ok = worker.decide_next_token(req, 0, nullptr, 0, scfg, false, w_tok, w_flags, &w_err);
        });
        RequestEndMsg end;
        end.request_id = req;
        end.status = 6;
        std::string h_err;
        REQUIRE(lp.head.broadcast(make_frame(end), &h_err));
        wt.join();
        CHECK_FALSE(w_ok);
        Frame parked;
        REQUIRE(worker.take_pending_frame(parked));
        CHECK(parked.type == MsgType::RequestEnd);
        CHECK_FALSE(worker.take_pending_frame(parked));

        bool w_ok2 = true;
        std::thread wt2([&] {
            std::vector<int32_t> d;
            w_ok2 = worker.decide_draft(req, 0, 0, d, &w_err);
        });
        AbortMsg abort;
        abort.rank = 0;
        abort.code = 1;
        abort.reason = "test abort";
        REQUIRE(lp.head.broadcast(make_frame(abort), &h_err));
        wt2.join();
        CHECK_FALSE(w_ok2);
        CHECK(worker.aborted());
        CHECK_EQUAL(worker.abort_reason(), std::string("test abort"));
        // Once aborted every further call fails immediately without blocking.
        int32_t a = 0, b = 0; uint8_t fl = 0;
        CHECK_FALSE(worker.decide_accept(req, 0, a, b, fl, &w_err));
    }

    lp.worker.close();
    lp.head.close();
}

// An empty draft block is the head's end marker for a request only it
// can see ending (client cancel). It must survive the wire as an empty
// vector, and it must clear whatever the worker had drafted locally.
TEST_CASE(ClusterUnitFixture, EmptyDraftBlockIsTheEndMarker) {
    LoopbackPair lp;
    if (lp.port == 0) SKIP("no loopback port available");
    REQUIRE(lp.handshake(lp.worker_id));

    HeadHooksConfig hc; hc.timeout_ms = 5000;
    HeadHooks head(lp.head, hc);
    WorkerHooksConfig wc; wc.rank = 1; wc.timeout_ms = 5000;
    WorkerHooks worker(lp.worker, wc);

    const uint64_t req = 9;
    head.set_current_request(req);
    worker.set_current_request(req);

    std::string w_err;
    // The worker drafted its own block; the head's end marker must replace it.
    std::vector<int32_t> w_draft = {5, 6, 7};
    bool w_ok = false;
    std::thread wt([&] { w_ok = worker.decide_draft(req, 3, 128, w_draft, &w_err); });

    std::string h_err;
    std::vector<int32_t> end_marker;
    REQUIRE(head.decide_draft(req, 3, 128, end_marker, &h_err));
    wt.join();

    CHECK(w_ok);
    CHECK(w_draft.empty());
    CHECK_EQUAL(head.counters().drafts, 1u);
    CHECK_EQUAL(worker.counters().drafts, 1u);
}

// The end marker also has to round-trip through the frame codec on its own:
// a zero-length token vector is a valid Draft, not a malformed one.
TEST_CASE(ClusterUnitFixture, EmptyDraftMsgRoundTrip) {
    DraftMsg msg;
    msg.request_id = 77;
    msg.step = 12;
    msg.pos = 4096;
    const Frame f = make_frame(msg);
    DraftMsg got;
    REQUIRE(parse_frame(f, got));
    CHECK_EQUAL(got.request_id, 77u);
    CHECK_EQUAL(got.step, 12u);
    CHECK_EQUAL(got.pos, 4096);
    CHECK(got.tokens.empty());
}

TEST_CASE(ClusterUnitFixture, HeadHooksFailWhenWorkerGone) {
    LoopbackPair lp;
    if (lp.port == 0) SKIP("no loopback port available");
    REQUIRE(lp.handshake(lp.worker_id));
    lp.worker.close();

    HeadHooksConfig hc;
    hc.timeout_ms = 500;
    HeadHooks head(lp.head, hc);
    std::string err;
    // A gather with no live worker must return false within the deadline.
    CHECK_FALSE(head.hash_probe(1, 0, 1, 2, &err));
    CHECK_FALSE(err.empty());
    lp.head.close();
}

#endif  // !_WIN32
