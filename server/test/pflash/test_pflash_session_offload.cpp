// Drafter-session host copies (pflash_session_offload.h): the byte-exact round trip (whole tensors and token
// prefixes, plain and buffer-backed host memory), the checks that refuse a copy into a different layout (nothing
// written), the restore pick, the offload plan (incl. the stale-copy rule) and the identity rule. CPU backend,
// model-free.

#include "CppUnitTestFramework.hpp"

#include "pflash/pflash_session_offload.h"
#include "pflash/pflash_scoring_resume.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace luce::pflash;

namespace {

struct PFlashSessionOffloadFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;
};

// A session-like layout: quantized and F16 attention K/V [64, tokens, 2] (token axis 1), F32 recurrent state, I32,
// keys [32, 2, tokens] (token axis 2), and a view (not owned).
struct Layout {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * view = nullptr;

    explicit Layout(int tokens, ggml_type kv_type = GGML_TYPE_Q8_0) {
        backend = ggml_backend_cpu_init();
        ggml_init_params p{};
        p.mem_size = 16 * ggml_tensor_overhead();
        p.no_alloc = true;
        ctx = ggml_init(p);
        ggml_set_name(ggml_new_tensor_3d(ctx, kv_type, 64, tokens, 2), "attn_k_0");
        ggml_set_name(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 64, tokens, 2), "attn_v_0");
        ggml_set_name(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 4), "ssm_state_0");
        ggml_set_name(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 37), "ids");
        ggml_tensor * keys = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 2, tokens);
        ggml_set_name(keys, "keys");
        view = ggml_view_1d(ctx, keys, 8, 0);
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }
    ~Layout() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
};

const std::vector<int> kWhole = {-1, -1, -1, -1, -1};
const std::vector<int> kTokens = {1, 1, -1, -1, 2};

std::vector<uint8_t> bytes_of(ggml_tensor * t) {
    std::vector<uint8_t> out(ggml_nbytes(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size());
    return out;
}

// Random raw bytes everywhere (any bit pattern: NaN payloads, -0, denormals, arbitrary quant blocks), plus explicit
// special floats at the start of every F32 tensor.
void fill_random(const std::vector<ggml_tensor *> & tensors, uint32_t seed) {
    std::mt19937 rng(seed);
    for (ggml_tensor * t : tensors) {
        std::vector<uint8_t> b(ggml_nbytes(t));
        for (auto & x : b) x = (uint8_t) rng();
        if (t->type == GGML_TYPE_F32 && b.size() >= 5 * sizeof(float)) {
            const uint32_t specials[5] = {0x7fc12345u, 0xff800001u, 0x80000000u, 0x00000001u, 0x7f800000u};
            std::memcpy(b.data(), specials, sizeof(specials));
        }
        ggml_backend_tensor_set(t, b.data(), 0, b.size());
    }
}

std::string temp_path(const char * tag) {
    return std::string("/tmp/pflash_session_offload_") + tag + "_" + std::to_string((long) getpid());
}

std::vector<int32_t> seq(int n, int salt = 0) {
    std::vector<int32_t> v((size_t) n);
    for (int i = 0; i < n; ++i) v[(size_t) i] = (i * 7 + salt) % 50000;
    return v;
}

} // namespace

TEST_CASE(PFlashSessionOffloadFixture, owned_tensors_skip_views_and_keep_creation_order) {
    Layout a(256);
    REQUIRE(a.buf != nullptr);
    const auto owned = pflash_owned_tensors(a.ctx);
    REQUIRE(owned.size() == 5u);
    CHECK(std::string(ggml_get_name(owned[0])) == "attn_k_0");
    CHECK(std::string(ggml_get_name(owned[4])) == "keys");
    for (ggml_tensor * t : owned) CHECK(t != a.view);
    CHECK(pflash_owned_tensors(nullptr).empty());
}

TEST_CASE(PFlashSessionOffloadFixture, whole_round_trip_is_byte_exact_in_plain_and_buffer_memory) {
    for (int buffer_backed = 0; buffer_backed < 2; ++buffer_backed) {
        Layout a(512);
        Layout b(512);
        REQUIRE(a.buf != nullptr);
        REQUIRE(b.buf != nullptr);
        const auto src = pflash_owned_tensors(a.ctx);
        const auto dst = pflash_owned_tensors(b.ctx);
        fill_random(src, 7);
        fill_random(dst, 99);
        PFlashHostCopy host;
        std::string error;
        REQUIRE(pflash_copy_to_host(src, kWhole, 0, buffer_backed ? ggml_backend_cpu_buffer_type() : nullptr, host,
                                    error));
        CHECK(host.blob.pinned() == (buffer_backed == 1));
        size_t expected = 0;
        for (ggml_tensor * t : src) expected += ggml_nbytes(t);
        CHECK(host.bytes == expected);
        std::vector<std::vector<uint8_t>> want;
        for (ggml_tensor * t : src) want.push_back(bytes_of(t));
        fill_random(src, 1234);   // the source is gone before the restore (the drafter is freed)
        REQUIRE(pflash_copy_from_host(host, dst, error));
        for (size_t i = 0; i < dst.size(); ++i) CHECK(bytes_of(dst[i]) == want[i]);
    }
}

TEST_CASE(PFlashSessionOffloadFixture, token_prefix_round_trip_copies_only_the_prefix_and_is_byte_exact) {
    const int T = 1024, n = 300;
    Layout a(T);
    Layout b(T);
    const auto src = pflash_owned_tensors(a.ctx);
    const auto dst = pflash_owned_tensors(b.ctx);
    fill_random(src, 11);
    fill_random(dst, 12);
    std::vector<std::vector<uint8_t>> src_bytes, dst_before;
    for (ggml_tensor * t : src) src_bytes.push_back(bytes_of(t));
    for (ggml_tensor * t : dst) dst_before.push_back(bytes_of(t));
    PFlashHostCopy host;
    std::string error;
    REQUIRE(pflash_copy_to_host(src, kTokens, n, nullptr, host, error));
    size_t expected = 0;
    expected += 2 * (size_t) n * src[0]->nb[1];      // attn_k: two heads, n rows each
    expected += 2 * (size_t) n * src[1]->nb[1];      // attn_v
    expected += ggml_nbytes(src[2]) + ggml_nbytes(src[3]);
    expected += (size_t) n * src[4]->nb[2];          // keys: one run
    CHECK(host.bytes == expected);
    REQUIRE(pflash_copy_from_host(host, dst, error));
    // Inside the prefix every byte equals the source; outside it the destination keeps its own bytes.
    for (size_t i = 0; i < dst.size(); ++i) {
        const auto got = bytes_of(dst[i]);
        std::vector<uint8_t> in_prefix(got.size(), 0);
        std::vector<std::pair<size_t, size_t>> chunks;
        REQUIRE(pflash_token_chunks(src[i], kTokens[i], n, chunks, error));
        for (const auto & c : chunks) std::fill(in_prefix.begin() + (long) c.first,
                                                 in_prefix.begin() + (long) (c.first + c.second), 1);
        for (size_t k = 0; k < got.size(); ++k) {
            if (in_prefix[k]) {
                if (got[k] != src_bytes[i][k]) { CHECK(false); break; }
            } else if (got[k] != dst_before[i][k]) {
                CHECK(false);
                break;
            }
        }
    }
    // A prefix longer than the axis is clipped to it; zero tokens copies nothing of the token tensors.
    REQUIRE(pflash_copy_to_host(src, kTokens, T + 500, nullptr, host, error));
    size_t whole = 0;
    for (ggml_tensor * t : src) whole += ggml_nbytes(t);
    CHECK(host.bytes == whole);
    REQUIRE(pflash_copy_to_host(src, kTokens, 0, nullptr, host, error));
    CHECK(host.bytes == ggml_nbytes(src[2]) + ggml_nbytes(src[3]));
}

// A session restored for a request longer than its stored capacity is allocated with more positions; the stored prefix
// lands at the same token positions of the longer tensors, byte for byte. A whole-tensor copy never goes into a longer
// tensor.
TEST_CASE(PFlashSessionOffloadFixture, token_prefix_restores_into_a_longer_session) {
    const int n = 300;
    Layout a(512);
    Layout b(1280);
    const auto src = pflash_owned_tensors(a.ctx);
    const auto dst = pflash_owned_tensors(b.ctx);
    fill_random(src, 21);
    fill_random(dst, 22);
    PFlashHostCopy host;
    std::string error;
    REQUIRE(pflash_copy_to_host(src, kTokens, n, nullptr, host, error));
    REQUIRE(pflash_copy_from_host(host, dst, error));
    // attn_k / attn_v [64, T, 2]: rows < n of each head equal; keys [32, 2, T]: positions < n equal.
    for (int i : {0, 1}) {
        const auto s_b = bytes_of(src[(size_t) i]);
        const auto d_b = bytes_of(dst[(size_t) i]);
        for (int h = 0; h < 2; ++h) {
            const size_t row = src[(size_t) i]->nb[1];
            CHECK(std::memcmp(s_b.data() + h * src[(size_t) i]->nb[2], d_b.data() + h * dst[(size_t) i]->nb[2],
                              row * n) == 0);
        }
    }
    CHECK(bytes_of(src[2]) == bytes_of(dst[2]));
    CHECK(bytes_of(src[3]) == bytes_of(dst[3]));
    const auto ks = bytes_of(src[4]);
    const auto kd = bytes_of(dst[4]);
    CHECK(std::memcmp(ks.data(), kd.data(), src[4]->nb[2] * n) == 0);
    // Whole copies keep requiring the exact shape.
    PFlashHostCopy whole;
    REQUIRE(pflash_copy_to_host(src, kWhole, 0, nullptr, whole, error));
    std::string why;
    CHECK(!pflash_copy_from_host(whole, dst, why));
    CHECK(why.find("shape differs") != std::string::npos);
    // A shorter target is refused.
    Layout c(256);
    CHECK(!pflash_copy_from_host(host, pflash_owned_tensors(c.ctx), why));
}

TEST_CASE(PFlashSessionOffloadFixture, mismatched_layouts_are_refused_and_nothing_is_written) {
    Layout a(512);
    REQUIRE(a.buf != nullptr);
    fill_random(pflash_owned_tensors(a.ctx), 5);
    PFlashHostCopy host;
    std::string error;
    REQUIRE(pflash_copy_to_host(pflash_owned_tensors(a.ctx), kTokens, 200, nullptr, host, error));

    const auto refused = [&](Layout & other, std::vector<ggml_tensor *> dst, const char * expect) {
        fill_random(pflash_owned_tensors(other.ctx), 77);
        std::vector<std::vector<uint8_t>> before;
        for (ggml_tensor * t : pflash_owned_tensors(other.ctx)) before.push_back(bytes_of(t));
        std::string why;
        CHECK(!pflash_copy_from_host(host, dst, why));
        CHECK(why.find(expect) != std::string::npos);
        size_t i = 0;
        for (ggml_tensor * t : pflash_owned_tensors(other.ctx)) CHECK(bytes_of(t) == before[i++]);
    };
    Layout shorter(256);          // fewer positions than the stored layout
    refused(shorter, pflash_owned_tensors(shorter.ctx), "shape differs");
    Layout other_type(512, GGML_TYPE_F16);
    refused(other_type, pflash_owned_tensors(other_type.ctx), "type differs");
    Layout fewer(512);
    auto dst = pflash_owned_tensors(fewer.ctx);
    dst.pop_back();
    refused(fewer, dst, "tensor count");
    Layout with_view(512);
    auto dst_view = pflash_owned_tensors(with_view.ctx);
    dst_view.back() = with_view.view;
    refused(with_view, dst_view, "not an allocated owned tensor");

    // A tampered copy (chunk out of bounds, byte count off) is refused too.
    Layout same(512);
    PFlashHostCopy bad;
    REQUIRE(pflash_copy_to_host(pflash_owned_tensors(a.ctx), kTokens, 200, nullptr, bad, error));
    bad.tensors[2].chunks.back().first = ggml_nbytes(pflash_owned_tensors(a.ctx)[2]);   // a whole-tensor copy
    std::string why;
    CHECK(!pflash_copy_from_host(bad, pflash_owned_tensors(same.ctx), why));
    CHECK(why.find("out of bounds") != std::string::npos);
    REQUIRE(pflash_copy_to_host(pflash_owned_tensors(a.ctx), kTokens, 200, nullptr, bad, error));
    bad.bytes -= 1;
    CHECK(!pflash_copy_from_host(bad, pflash_owned_tensors(same.ctx), why));
    CHECK(why.find("size") != std::string::npos);
}

TEST_CASE(PFlashSessionOffloadFixture, unallocated_view_or_mismatched_axes_are_not_offloaded) {
    ggml_init_params p{};
    p.mem_size = 4 * ggml_tensor_overhead();
    p.no_alloc = true;
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    PFlashHostCopy host;
    std::string error;
    CHECK(!pflash_copy_to_host({t}, {-1}, 0, nullptr, host, error));
    CHECK(host.tensors.empty());
    CHECK(!pflash_copy_to_host({nullptr}, {-1}, 0, nullptr, host, error));
    ggml_free(ctx);
    Layout a(256);
    CHECK(!pflash_copy_to_host({a.view}, {-1}, 0, nullptr, host, error));
    CHECK(!pflash_copy_to_host(pflash_owned_tensors(a.ctx), {-1}, 0, nullptr, host, error));   // axes count
}

TEST_CASE(PFlashSessionOffloadFixture, host_blob_reuses_storage_that_is_large_enough) {
    PFlashHostBlob blob;
    REQUIRE(blob.reserve(4096, ggml_backend_cpu_buffer_type()));
    const uint8_t * first = blob.data();
    CHECK(blob.pinned());
    REQUIRE(blob.reserve(1000, ggml_backend_cpu_buffer_type()));
    CHECK(blob.data() == first);
    REQUIRE(blob.reserve(8192, nullptr));
    CHECK(!blob.pinned());
    CHECK(blob.capacity() >= 8192u);
    PFlashHostBlob moved = std::move(blob);
    CHECK(moved.capacity() >= 8192u);
    CHECK(blob.data() == nullptr);
}

TEST_CASE(PFlashSessionOffloadFixture, restore_pick_is_the_largest_exact_resume_point) {
    const int G = kScoringResumeGrid;
    const auto a = seq(3 * G + 100);        // checkpoint 3G
    auto b = seq(2 * G + 100);              // checkpoint 2G
    auto other = seq(3 * G + 100, 13);      // another conversation
    std::vector<PFlashStoredPrompt> stored = {{&b, 2 * G}, {&other, 3 * G}, {&a, 3 * G}};
    auto req = a;
    req.push_back(1);
    req.push_back(2);
    CHECK(pflash_pick_restore(stored, req) == 2);          // a resumes at 3G, b at 2G
    auto edited = a;
    edited[(size_t) (G + 5)] ^= 1;                          // diverges before every checkpoint
    CHECK(pflash_pick_restore(stored, edited) == -1);      // shared prefix G+5 < every checkpoint
    auto short_req = seq(3 * G + 10);                       // S - 64 < 3G: only b's 2G is exact
    CHECK(pflash_pick_restore(stored, short_req) == 0);
    CHECK(pflash_pick_restore(stored, seq(100, 99)) == -1);
    std::vector<int32_t> empty;
    std::vector<PFlashStoredPrompt> none = {{&empty, 0}, {&a, 0}};
    CHECK(pflash_pick_restore(none, req) == -1);           // checkpoint 0 never resumes
}

TEST_CASE(PFlashSessionOffloadFixture, offload_plan_never_keeps_a_stale_host_copy) {
    // Session restored from entry 5 and then acquired by a scoring call (host_valid false): it is copied again and
    // entry 5 is released, so a later restore can only see the new state.
    auto plan = pflash_plan_offload({{5, 10, true}}, {{5, false, true, 20}}, 2);
    CHECK(plan.copy_devices == std::vector<size_t>{0});
    CHECK(plan.keep_entries.empty());
    CHECK(plan.drop_entries == std::vector<uint64_t>{5});
    // Restored and untouched: entry 5 is kept, nothing is copied.
    plan = pflash_plan_offload({{5, 10, true}}, {{5, true, true, 10}}, 2);
    CHECK(plan.copy_devices.empty());
    CHECK(plan.keep_entries == std::vector<uint64_t>{5});
    CHECK(plan.drop_entries.empty());
    // Restored, then its call failed (no prompt left): the old entry is released, nothing copied.
    plan = pflash_plan_offload({{5, 10, true}}, {{5, false, false, 20}}, 2);
    CHECK(plan.copy_devices.empty());
    CHECK(plan.drop_entries == std::vector<uint64_t>{5});
    // host_valid without its entry (e.g. dropped meanwhile) is copied, never assumed stored.
    plan = pflash_plan_offload({}, {{9, true, true, 20}}, 2);
    CHECK(plan.copy_devices == std::vector<size_t>{0});
}

TEST_CASE(PFlashSessionOffloadFixture, offload_plan_keeps_the_most_recent_sessions_up_to_the_limit) {
    // Two stored conversations on the host, a new one scored on the device: the least recently used one goes.
    auto plan = pflash_plan_offload({{1, 5, false}, {2, 8, false}}, {{0, false, true, 9}}, 2);
    CHECK(plan.copy_devices == std::vector<size_t>{0});
    CHECK(plan.keep_entries == std::vector<uint64_t>{2});
    CHECK(plan.drop_entries == std::vector<uint64_t>{1});
    // Unrestored entries stay untouched while within the limit (entry 2 restored and unchanged; also when the
    // entry's on_device flag was not set, it is counted once).
    plan = pflash_plan_offload({{1, 5, false}, {2, 8, false}}, {{2, true, true, 8}}, 2);
    CHECK(plan.copy_devices.empty());
    CHECK(plan.drop_entries.empty());
    plan = pflash_plan_offload({{1, 5, false}, {2, 8, true}}, {{2, true, true, 8}}, 2);
    CHECK(plan.copy_devices.empty());
    CHECK(plan.drop_entries.empty());
    CHECK(plan.keep_entries.size() == 2u);
    // A session too short to resume (no checkpoint) is not stored; limit 1 keeps the newest only.
    plan = pflash_plan_offload({{1, 5, false}}, {{0, false, false, 9}, {0, false, true, 7}}, 1);
    CHECK(plan.copy_devices == std::vector<size_t>{1});
    CHECK(plan.drop_entries == std::vector<uint64_t>{1});
    plan = pflash_plan_offload({{1, 5, false}}, {{0, false, true, 9}}, 0);
    CHECK(plan.copy_devices.empty());
    CHECK(plan.drop_entries == std::vector<uint64_t>{1});
}

TEST_CASE(PFlashSessionOffloadFixture, identity_names_the_first_differing_field) {
    const std::string a = "format=1\ndrafter=/m/a.gguf|1:2|10|5\nsessions=2\n";
    CHECK(pflash_identity_mismatch(a, a).empty());
    CHECK(pflash_identity_mismatch(a, "format=1\ndrafter=/m/b.gguf|1:3|10|5\nsessions=2\n") == "drafter differs");
    CHECK(pflash_identity_mismatch(a, "format=1\ndrafter=/m/a.gguf|1:2|10|5\nsessions=3\n") == "sessions differs");
    CHECK(pflash_identity_mismatch(a, a + "head=x\n") == "head added");
    CHECK(pflash_identity_mismatch(a + "head=x\n", a) == "head removed");
}

TEST_CASE(PFlashSessionOffloadFixture, file_identity_changes_with_the_file) {
    CHECK(pflash_file_identity("") == "-");
    const std::string path = temp_path("ident");
    std::remove(path.c_str());
    CHECK(pflash_file_identity(path) == path + "|missing");
    { std::ofstream(path) << "drafter-v1"; }
    const std::string first = pflash_file_identity(path);
    CHECK(first.rfind(path + "|", 0) == 0);
    CHECK(pflash_file_identity(path) == first);
    { std::ofstream(path) << "drafter-v2 longer"; }
    CHECK(pflash_file_identity(path) != first);
    std::remove(path.c_str());
}
