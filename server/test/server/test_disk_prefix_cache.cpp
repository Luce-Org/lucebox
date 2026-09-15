#include "ggml-cpu.h"
#include "server/disk_prefix_cache.h"
#include "server/prefix_cache.h"
#include "support/mock_backend.h"
#include "support/environment.h"
#include <nlohmann/json.hpp>
#include "support/test_assert.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include <unistd.h>

using json = nlohmann::json;
using namespace dflash::common;
namespace fs = std::filesystem;

// Disk Prefix Cache Tests
// ═══════════════════════════════════════════════════════════════════════

// Minimal mock backend for testing (no GPU needed).
struct MockBatchCompressBackend : MockBackend {
    int compress_calls = 0;

    CompressResult compress(const CompressRequest & request) override {
        ++compress_calls;
        CompressResult result;
        result.ok = !request.input_ids.empty();
        if (result.ok) result.compressed_ids = {request.input_ids.front()};
        return result;
    }
};

TEST_CASE(ServerUnitFixture, test_compress_batch_default_preserves_order) {
    MockBatchCompressBackend backend;
    std::vector<ModelBackend::CompressRequest> requests(3);
    requests[0].input_ids = {11, 12};
    requests[1].input_ids = {21, 22};
    requests[2].input_ids = {31, 32};

    const auto results = backend.compress_batch(requests);
    TEST_ASSERT(results.size() == requests.size());
    TEST_ASSERT(backend.compress_calls == 3);
    TEST_ASSERT(results[0].compressed_ids == std::vector<int32_t>({11}));
    TEST_ASSERT(results[1].compressed_ids == std::vector<int32_t>({21}));
    TEST_ASSERT(results[2].compressed_ids == std::vector<int32_t>({31}));
}

struct MockMemoryOnlySnapshotBackend : MockBackend {
    bool snapshot_used(int slot) const override { return slot == 0; }
};

// ─── MockBackendWithLayout ──────────────────────────────────────────────
// Extends MockBackend with a real ggml_context so DiskPrefixCache can
// iterate tensors in compute_layout_id and write a real .dkv file.
// KV: one layer, K=[16,32,4,1] F32 + V=[32,16,4,1] F32.
struct MockBackendWithLayout : MockBackend {
    static constexpr int     kNLayer  = 1;
    static constexpr int64_t kHeadDim = 16;
    static constexpr int64_t kNHead   = 4;
    static constexpr int     kMaxPos  = 32;

    ggml_context         * kv_ctx_ = nullptr;
    ggml_backend_t         cpu_be_ = nullptr;
    ggml_backend_buffer_t  kv_buf_ = nullptr;
    ggml_tensor          * k_[kNLayer] = {};
    ggml_tensor          * v_[kNLayer] = {};

    MockBackendWithLayout() {
        cpu_be_ = ggml_backend_cpu_init();
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * (kNLayer * 2 + 4) + 4096;
        ip.no_alloc = true;
        kv_ctx_ = ggml_init(ip);
        int64_t ne_k[4] = {kHeadDim, kMaxPos, kNHead, 1};
        int64_t ne_v[4] = {kMaxPos, kHeadDim, kNHead, 1};
        char name[64];
        for (int il = 0; il < kNLayer; ++il) {
            k_[il] = ggml_new_tensor(kv_ctx_, GGML_TYPE_F32, 4, ne_k);
            std::snprintf(name, sizeof(name), "snap_k_%d", il);
            ggml_set_name(k_[il], name);
            v_[il] = ggml_new_tensor(kv_ctx_, GGML_TYPE_F32, 4, ne_v);
            std::snprintf(name, sizeof(name), "snap_v_%d", il);
            ggml_set_name(v_[il], name);
        }
        kv_buf_ = ggml_backend_alloc_ctx_tensors(kv_ctx_, cpu_be_);
        for (int il = 0; il < kNLayer; ++il) {
            std::vector<float> ones(ggml_nelements(k_[il]), 1.0f);
            std::vector<float> twos(ggml_nelements(v_[il]), 2.0f);
            ggml_backend_tensor_set(k_[il], ones.data(), 0, ggml_nbytes(k_[il]));
            ggml_backend_tensor_set(v_[il], twos.data(), 0, ggml_nbytes(v_[il]));
        }
    }
    ~MockBackendWithLayout() {
        if (kv_buf_) ggml_backend_buffer_free(kv_buf_);
        if (kv_ctx_) ggml_free(kv_ctx_);
        if (cpu_be_) ggml_backend_free(cpu_be_);
    }

    SnapshotRef snapshot_ref(int /*slot*/) const override {
        SnapshotRef ref;
        ref.ctx      = kv_ctx_;
        ref.buf      = kv_buf_;
        ref.cur_pos  = kMaxPos;
        ref.last_tok = 42;
        return ref;
    }

    bool snapshot_save(int) override { return true; }
    bool snapshot_used(int) const override { return true; }
    int  snapshot_cur_pos(int) const override { return kMaxPos; }
};

// Helper: recursively remove a directory.
static void rm_rf(const std::string & path) {
    std::error_code ec;
    fs::remove_all(path, ec);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_config_defaults) {
    DiskCacheConfig cfg;
    TEST_ASSERT(cfg.cache_dir.empty());
    TEST_ASSERT(cfg.budget_bytes == (size_t)4 * 1024 * 1024 * 1024);
    TEST_ASSERT(cfg.min_tokens == 512);
    TEST_ASSERT(cfg.continued_interval == 10240);
    TEST_ASSERT(cfg.cold_max_tokens == 10240);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_policy_parse) {
    DiskPrefixCachePolicy policy;
    TEST_ASSERT(parse_disk_prefix_cache_policy("off", policy));
    TEST_ASSERT(policy.mode == DiskPrefixCacheMode::Off);
    TEST_ASSERT(parse_disk_prefix_cache_policy("full", policy));
    TEST_ASSERT(policy.mode == DiskPrefixCacheMode::Full);
    TEST_ASSERT(parse_disk_prefix_cache_policy("auto", policy));
    TEST_ASSERT(policy.mode == DiskPrefixCacheMode::Auto);
    TEST_ASSERT(policy.auto_window == 30);
    TEST_ASSERT(parse_disk_prefix_cache_policy("auto:30", policy));
    TEST_ASSERT(policy.mode == DiskPrefixCacheMode::Auto);
    TEST_ASSERT(policy.auto_window == 30);
    TEST_ASSERT(parse_disk_prefix_cache_policy("1000", policy));
    TEST_ASSERT(policy.mode == DiskPrefixCacheMode::Fixed);
    TEST_ASSERT(policy.fixed_tokens == 1000);
    TEST_ASSERT(!parse_disk_prefix_cache_policy("core", policy));
    TEST_ASSERT(!parse_disk_prefix_cache_policy("task", policy));
    TEST_ASSERT(!parse_disk_prefix_cache_policy("auto:0", policy));
}

// BUG-A: apply_request_scope_override must preserve server-level compress flag.
// A request-level scope override (e.g. "auto") must NOT clear compress=true
// that was set by the server configuration.
TEST_CASE(ServerUnitFixture, test_scope_override_preserves_compress) {
    // Server policy: compress=true, mode=Full.
    DiskPrefixCachePolicy server;
    server.mode = DiskPrefixCacheMode::Full;
    server.compress = true;

    // Request sends scope="auto" — should change mode but keep compress.
    TEST_ASSERT(apply_request_scope_override(server, "auto"));
    TEST_ASSERT(server.mode == DiskPrefixCacheMode::Auto);
    TEST_ASSERT_MSG(server.compress,
        "BUG-A: scope override dropped server-level compress=true");

    // Same with a fixed-token scope.
    DiskPrefixCachePolicy server2;
    server2.mode = DiskPrefixCacheMode::Full;
    server2.compress = true;
    TEST_ASSERT(apply_request_scope_override(server2, "1000"));
    TEST_ASSERT(server2.mode == DiskPrefixCacheMode::Fixed);
    TEST_ASSERT(server2.fixed_tokens == 1000);
    TEST_ASSERT_MSG(server2.compress,
        "BUG-A: fixed-token scope override dropped server-level compress=true");

    // scope="off" must also preserve compress flag.
    DiskPrefixCachePolicy server3;
    server3.compress = true;
    TEST_ASSERT(apply_request_scope_override(server3, "off"));
    TEST_ASSERT(server3.mode == DiskPrefixCacheMode::Off);
    TEST_ASSERT_MSG(server3.compress,
        "BUG-A: off scope override dropped server-level compress=true");

    // Invalid scope string must return false and leave policy unchanged.
    DiskPrefixCachePolicy server4;
    server4.compress = true;
    server4.mode = DiskPrefixCacheMode::Full;
    TEST_ASSERT(!apply_request_scope_override(server4, "core"));
    TEST_ASSERT(server4.compress);
    TEST_ASSERT(server4.mode == DiskPrefixCacheMode::Full);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_fixed_boundary) {
    DiskPrefixCachePolicy policy;
    TEST_ASSERT(parse_disk_prefix_cache_policy("1000", policy));
    TEST_ASSERT(disk_prefix_cache_fixed_boundary(policy, 2000) == 1000);
    TEST_ASSERT(disk_prefix_cache_fixed_boundary(policy, 500) == 0);
    TEST_ASSERT(disk_prefix_cache_fixed_boundary(policy, 2000, 1001) == 0);
    TEST_ASSERT(disk_prefix_cache_fixed_boundary(policy, 2000, 1000) == 1000);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_auto_boundary_lcp) {
    std::vector<int32_t> current{1, 2, 3, 4, 5, 9};
    std::vector<std::vector<int32_t>> recent{
        {1, 2, 3, 4, 8},
        {1, 2, 3, 4, 7},
        {7, 8},
    };
    std::vector<int> safe_boundaries{2, 4};
    TEST_ASSERT(disk_prefix_cache_auto_boundary(
        current, recent, 2, safe_boundaries, 2) == 4);
    TEST_ASSERT(disk_prefix_cache_auto_boundary(
        current, recent, 2, {}, 2) == 4);
    TEST_ASSERT(disk_prefix_cache_auto_boundary(
        current, recent, 3, {}, 2) == 4);
    TEST_ASSERT(disk_prefix_cache_auto_boundary(
        current, recent, 2, safe_boundaries, 5) == 0);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_auto_window_limits_history) {
    std::vector<int32_t> current{1, 2, 3, 4, 5};
    std::vector<std::vector<int32_t>> recent{
        {9},
        {1, 2, 3, 4, 0},
        {1, 2, 3, 4, 9},
    };
    TEST_ASSERT(disk_prefix_cache_auto_boundary(
        current, recent, 1, {}, 2) == 0);
    TEST_ASSERT(disk_prefix_cache_auto_boundary(
        current, recent, 2, {}, 2) == 4);
    TEST_ASSERT(disk_prefix_cache_auto_boundary(
        current, recent, 3, {}, 2) == 4);
    TEST_ASSERT(disk_prefix_cache_auto_boundary(
        current, recent, 0, {}, 2) == 0);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_disabled_when_no_dir) {
    MockBackend backend;
    DiskCacheConfig cfg;
    cfg.cache_dir = "";
    DiskPrefixCache cache(cfg, backend);
    TEST_ASSERT(cache.disabled());
    // Operations should be no-ops.
    std::vector<int32_t> ids = {1, 2, 3, 4, 5};
    TEST_ASSERT(!cache.lookup(ids, 0));
    TEST_ASSERT(!cache.save(0, ids));
}

TEST_CASE(ServerUnitFixture, test_disk_cache_disables_memory_only_backend) {
    MockMemoryOnlySnapshotBackend backend;
    DiskCacheConfig cfg;
    cfg.cache_dir = test_tmp_path("dflash_test_disk_cache_memory_only").string();
    DiskPrefixCache cache(cfg, backend);
    TEST_ASSERT(!cache.disabled());

    // A live in-memory snapshot with no SnapshotRef means this backend cannot
    // serialize or adopt disk entries. Detect it once and skip later disk work.
    cache.learn_layout(0);
    TEST_ASSERT(cache.disabled());
}

TEST_CASE(ServerUnitFixture, test_disk_cache_init_creates_directory) {
    MockBackend backend;
    std::string dir = test_tmp_path("dflash_test_disk_cache_init").string();
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    DiskPrefixCache cache(cfg, backend);
    TEST_ASSERT(!cache.disabled());
    TEST_ASSERT(cache.init());

    // Directory should exist.
    std::error_code ec;
    TEST_ASSERT(fs::is_directory(dir, ec) && !ec);

    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_header_size) {
    // The header should be exactly 80 bytes.
    TEST_ASSERT(DISK_CACHE_HEADER_SIZE == 80);
    // Bumped to 2 when the K-rotation default changed: a cache written by an
    // older binary stores K in the rotated basis, and the layout id does not
    // cover that, so the version is what rejects it.
    TEST_ASSERT(DISK_CACHE_VERSION == 2);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_continued_boundary) {
    // Test maybe_store_continued logic: saves at interval boundaries.
    MockBackend backend;
    std::string dir = test_tmp_path("dflash_test_continued").string();
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    cfg.min_tokens = 100;
    cfg.continued_interval = 1000;
    DiskPrefixCache cache(cfg, backend);
    cache.init();

    // Without layout known, save should fail gracefully.
    std::vector<int32_t> tokens(1500, 42);
    TEST_ASSERT(!cache.maybe_store_continued(0, tokens, 1000));

    // Reset continued tracking.
    cache.reset_continued();

    // Below interval, no save (even if tokens available).
    TEST_ASSERT(!cache.maybe_store_continued(0, tokens, 500));

    // At exactly 1000 tokens — would save if layout were known.
    // But backend mock can't provide snapshots, so it fails gracefully.
    TEST_ASSERT(!cache.maybe_store_continued(0, tokens, 1000));

    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_cold_prefix_short_prompt) {
    // Cold prefix should not trigger for short prompts. The layout is
    // learned first so the length gate — not the !layout_known_ guard —
    // is what returns 0.
    MockBackendWithLayout backend;
    std::string dir = test_tmp_path("dflash_test_cold_short").string();
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    cfg.cold_max_tokens = 10240;
    cfg.min_tokens = 512;
    DiskPrefixCache cache(cfg, backend);
    cache.init();
    cache.learn_layout(0);

    // Prompt shorter than cold_max_tokens.
    std::vector<int32_t> prompt(5000, 1);
    std::vector<int> boundaries = {1000, 2000, 3000, 4000};
    TEST_ASSERT(cache.cold_prefix_boundary(prompt, boundaries) == 0);

    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_cold_prefix_no_boundaries) {
    // Cold prefix should not trigger if no boundaries provided — the empty
    // boundary gate fires even with the layout learned and a long prompt.
    MockBackendWithLayout backend;
    std::string dir = test_tmp_path("dflash_test_cold_nobound").string();
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    cfg.cold_max_tokens = 5000;
    cfg.min_tokens = 512;
    DiskPrefixCache cache(cfg, backend);
    cache.init();
    cache.learn_layout(0);

    std::vector<int32_t> prompt(10000, 1);
    std::vector<int> empty_boundaries;
    TEST_ASSERT(cache.cold_prefix_boundary(prompt, empty_boundaries) == 0);

    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_cold_prefix_finds_boundary) {
    // With the layout learned, cold_prefix_boundary returns the last turn
    // boundary inside [min_tokens, cold_max_tokens].
    MockBackendWithLayout backend;
    std::string dir = test_tmp_path("dflash_test_cold_finds").string();
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    cfg.cold_max_tokens = 5000;
    cfg.min_tokens = 512;
    DiskPrefixCache cache(cfg, backend);
    cache.init();

    std::vector<int32_t> prompt(10000, 1);
    const std::vector<int> boundaries = {1000, 2000, 3000, 4000, 6000, 8000};

    // Before learn_layout the layout guard short-circuits to 0.
    TEST_ASSERT(cache.cold_prefix_boundary(prompt, boundaries) == 0);

    cache.learn_layout(0);
    // 4000 is the last boundary <= cold_max_tokens; 6000/8000 exceed it.
    TEST_ASSERT(cache.cold_prefix_boundary(prompt, boundaries) == 4000);
    // A boundary exactly at the cap is eligible.
    TEST_ASSERT(cache.cold_prefix_boundary(prompt, {5000}) == 5000);
    // Boundaries below min_tokens cannot host a checkpoint.
    TEST_ASSERT(cache.cold_prefix_boundary(prompt, {100, 400}) == 0);

    // Once a disk entry covers that prefix, no cold checkpoint is needed.
    const std::vector<int32_t> head(prompt.begin(), prompt.begin() + 4000);
    TEST_ASSERT(cache.save(0, head));
    TEST_ASSERT(cache.cold_prefix_boundary(prompt, {4000}) == 0);

    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_lookup_miss_no_layout) {
    // Lookup with no layout known should return false.
    MockBackend backend;
    std::string dir = test_tmp_path("dflash_test_lookup_miss").string();
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    DiskPrefixCache cache(cfg, backend);
    cache.init();

    std::vector<int32_t> ids = {1, 2, 3, 4, 5, 6, 7, 8};
    TEST_ASSERT(!cache.lookup(ids, 0));

    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_save_below_min_tokens) {
    // Save with fewer tokens than min_tokens should be rejected.
    MockBackend backend;
    std::string dir = test_tmp_path("dflash_test_save_below").string();
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    cfg.min_tokens = 100;
    DiskPrefixCache cache(cfg, backend);
    cache.init();

    std::vector<int32_t> ids(50, 1);  // only 50 tokens
    TEST_ASSERT(!cache.save(0, ids));

    rm_rf(dir);
}

// ─── Disk-cache identity salt tests (manifest hardening) ────────────────
//
// (a) Different salts → different layout_id; same salt → same layout_id.
// (b) All-zero salt (default) ≡ no salt at all (back-compat).

// Helper: read layout_id from the first .dkv file found under base/.
static std::array<uint8_t, 16> read_layout_id_from_cache_dir(const std::string & base) {
    std::array<uint8_t, 16> id{};
    std::error_code ec;
    for (const fs::directory_entry & entry : fs::directory_iterator(base, ec)) {
        if (ec) break;
        if (!entry.is_directory(ec)) {
            ec.clear();
            continue;
        }
        for (const fs::directory_entry & file_entry :
             fs::directory_iterator(entry.path(), ec)) {
            if (ec) break;
            const std::string name = file_entry.path().filename().string();
            if (name.size() < 4 || name.compare(name.size() - 4, 4, ".dkv") != 0) {
                continue;
            }
            FILE * f = std::fopen(file_entry.path().string().c_str(), "rb");
            if (!f) continue;
            std::fseek(f, 8, SEEK_SET);  // skip magic(4) + version(4)
            std::fread(id.data(), 1, 16, f);
            std::fclose(f);
            return id;
        }
    }
    return id;
}

TEST_CASE(ServerUnitFixture, test_disk_identity_salt_changes_layout_id) {
    MockBackendWithLayout backend;
    std::vector<int32_t> prompt;
    for (int i = 0; i < MockBackendWithLayout::kMaxPos; ++i) prompt.push_back(i + 1);

    // Salt A: non-zero.
    std::array<uint8_t, 16> salt_a{};
    salt_a[0] = 0x01; salt_a[15] = 0xAB;

    std::string dir_a = test_tmp_path("dflash_test_salt_a").string();
    rm_rf(dir_a);
    {
        DiskCacheConfig cfg; cfg.cache_dir = dir_a; cfg.min_tokens = 1;
        DiskPrefixCache cache(cfg, backend);
        cache.set_identity_salt(salt_a);
        cache.init();
        cache.learn_layout(0);
        TEST_ASSERT(cache.save(0, prompt));
    }

    // Salt B: different from A.
    std::array<uint8_t, 16> salt_b{};
    salt_b[0] = 0x02; salt_b[15] = 0xCD;

    std::string dir_b = test_tmp_path("dflash_test_salt_b").string();
    rm_rf(dir_b);
    {
        DiskCacheConfig cfg; cfg.cache_dir = dir_b; cfg.min_tokens = 1;
        DiskPrefixCache cache(cfg, backend);
        cache.set_identity_salt(salt_b);
        cache.init();
        cache.learn_layout(0);
        TEST_ASSERT(cache.save(0, prompt));
    }

    auto id_a = read_layout_id_from_cache_dir(dir_a);
    auto id_b = read_layout_id_from_cache_dir(dir_b);

    // Different salts → different layout_id.
    TEST_ASSERT(id_a != id_b);

    // Same salt A applied again → identical layout_id.
    std::string dir_a2 = test_tmp_path("dflash_test_salt_a2").string();
    rm_rf(dir_a2);
    {
        DiskCacheConfig cfg; cfg.cache_dir = dir_a2; cfg.min_tokens = 1;
        DiskPrefixCache cache(cfg, backend);
        cache.set_identity_salt(salt_a);
        cache.init();
        cache.learn_layout(0);
        TEST_ASSERT(cache.save(0, prompt));
    }
    auto id_a2 = read_layout_id_from_cache_dir(dir_a2);
    TEST_ASSERT(id_a == id_a2);

    rm_rf(dir_a);
    rm_rf(dir_b);
    rm_rf(dir_a2);
}

TEST_CASE(ServerUnitFixture, test_disk_identity_salt_zero_is_backcompat) {
    // Explicit all-zero salt must produce the same layout_id as no salt call
    // (default-constructed identity_salt_ is already all-zero).
    MockBackendWithLayout backend;
    std::vector<int32_t> prompt;
    for (int i = 0; i < MockBackendWithLayout::kMaxPos; ++i) prompt.push_back(i + 1);

    std::string dir1 = test_tmp_path("dflash_test_salt_zero1").string();
    rm_rf(dir1);
    {
        DiskCacheConfig cfg; cfg.cache_dir = dir1; cfg.min_tokens = 1;
        DiskPrefixCache cache(cfg, backend);
        // No set_identity_salt call — stays all-zero.
        cache.init();
        cache.learn_layout(0);
        TEST_ASSERT(cache.save(0, prompt));
    }

    std::string dir2 = test_tmp_path("dflash_test_salt_zero2").string();
    rm_rf(dir2);
    {
        DiskCacheConfig cfg; cfg.cache_dir = dir2; cfg.min_tokens = 1;
        DiskPrefixCache cache(cfg, backend);
        std::array<uint8_t, 16> zero_salt{};
        cache.set_identity_salt(zero_salt);
        cache.init();
        cache.learn_layout(0);
        TEST_ASSERT(cache.save(0, prompt));
    }

    auto id1 = read_layout_id_from_cache_dir(dir1);
    auto id2 = read_layout_id_from_cache_dir(dir2);
    TEST_ASSERT(id1 == id2);

    rm_rf(dir1);
    rm_rf(dir2);
}

// Extends MockBackendWithLayout with adoptable snapshots: save() stores the
// backend's buffers and lookup() hands them back through snapshot_adopt.
struct MockBackendWithAdopt : MockBackendWithLayout {
    std::vector<std::pair<ggml_context *, ggml_backend_buffer_t>> adopted_;
    int adopted_cur_pos_ = 0;
    int cur_pos_ = kMaxPos;
    ~MockBackendWithAdopt() {
        for (auto & p : adopted_) {
            if (p.second) ggml_backend_buffer_free(p.second);
            if (p.first) ggml_free(p.first);
        }
    }
    SnapshotRef snapshot_ref(int slot) const override {
        SnapshotRef ref = MockBackendWithLayout::snapshot_ref(slot);
        ref.cur_pos = cur_pos_;
        return ref;
    }
    int snapshot_cur_pos(int) const override { return cur_pos_; }
    bool snapshot_adopt(int, ggml_context * ctx, ggml_backend_buffer_t buf,
                        int cur_pos, int32_t) override {
        adopted_.push_back({ctx, buf});
        adopted_cur_pos_ = cur_pos;
        return true;
    }
};

TEST_CASE(ServerUnitFixture, test_disk_cache_header_round_trip) {
    // Drive the production serializer end to end: save() writes the header
    // through DiskPrefixCache::write_header, the on-disk fields are checked
    // against the known fixture, and lookup() loads the same file back
    // through read_header/read_file.
    MockBackendWithAdopt backend;
    std::string dir = test_tmp_path("dflash_test_header_rt").string();
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    cfg.min_tokens = 1;
    DiskPrefixCache cache(cfg, backend);
    TEST_ASSERT(cache.init());
    cache.learn_layout(0);

    // The key must cover the whole snapshot (kMaxPos tokens minimum).
    std::vector<int32_t> prompt;
    for (int i = 0; i < MockBackendWithLayout::kMaxPos + 8; ++i) {
        prompt.push_back(i + 1);
    }
    TEST_ASSERT(cache.save(0, prompt));

    // Locate the file production wrote: <dir>/<layout_id_hex>/<hash>.dkv.
    std::string path;
    for (const auto & entry : fs::recursive_directory_iterator(dir)) {
        if (entry.path().extension() == ".dkv") {
            path = entry.path().string();
            break;
        }
    }
    TEST_ASSERT(!path.empty());
    const std::string layout_hex =
        fs::path(path).parent_path().filename().string();

    FILE * f = std::fopen(path.c_str(), "rb");
    TEST_ASSERT(f != nullptr);
    char magic[4]; std::fread(magic, 4, 1, f);
    TEST_ASSERT(std::memcmp(magic, "DKVC", 4) == 0);
    uint32_t rv;
    std::fread(&rv, 4, 1, f); TEST_ASSERT(rv == DISK_CACHE_VERSION);
    uint8_t lid[16]; std::fread(lid, 16, 1, f);
    {
        // The layout directory name is the hex of the header's layout_id.
        static const char * digits = "0123456789abcdef";
        std::string got;
        for (uint8_t b : lid) { got += digits[b >> 4]; got += digits[b & 0xF]; }
        TEST_ASSERT(got == layout_hex);
    }
    std::fread(&rv, 4, 1, f);
    TEST_ASSERT(rv == (uint32_t)MockBackendWithLayout::kMaxPos);  // cur_pos
    std::fread(&rv, 4, 1, f);
    TEST_ASSERT(rv == 2u * MockBackendWithLayout::kNLayer);       // n_tensors
    std::fread(&rv, 4, 1, f);
    TEST_ASSERT(rv == (uint32_t)prompt.size());                   // token_count
    uint8_t th[16]; std::fread(th, 16, 1, f);
    {
        const PrefixHash ph = hash_prefix(prompt.data(), (int)prompt.size());
        TEST_ASSERT(std::memcmp(th, ph.data(), 16) == 0);
    }
    uint64_t ru64; std::fread(&ru64, 8, 1, f);
    const uint64_t payload =
        (uint64_t)MockBackendWithLayout::kNLayer *
        (ggml_nbytes(backend.k_[0]) + ggml_nbytes(backend.v_[0]));
    TEST_ASSERT(ru64 == payload);                                 // payload_bytes
    uint64_t created; std::fread(&created, 8, 1, f);
    TEST_ASSERT(created > 0);                                     // created_at
    std::fread(&ru64, 8, 1, f); TEST_ASSERT(ru64 == created);     // last_used
    int32_t ri32; std::fread(&ri32, 4, 1, f); TEST_ASSERT(ri32 == 42);  // last_tok
    std::fclose(f);

    // The same file must load back through read_header/read_file.
    TEST_ASSERT(cache.lookup(prompt, 1));
    TEST_ASSERT(backend.adopted_cur_pos_ == MockBackendWithLayout::kMaxPos);

    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_full_lookup_lengths) {
    // Whole prompt first, then every boundary deepest first, skipping cuts
    // below the persistence minimum and the prompt end itself.
    TEST_ASSERT(disk_prefix_cache_full_lookup_lengths(
                    6000, {300, 2000, 4000, 6000}, 512) ==
                std::vector<int>({6000, 4000, 2000}));
    TEST_ASSERT(disk_prefix_cache_full_lookup_lengths(6000, {}, 512) ==
                std::vector<int>({6000}));
    TEST_ASSERT(disk_prefix_cache_full_lookup_lengths(0, {100}, 512).empty());
    // Below the persistence minimum nothing was ever written: no probes.
    TEST_ASSERT(disk_prefix_cache_full_lookup_lengths(300, {100}, 512).empty());
}

TEST_CASE(ServerUnitFixture, test_disk_cache_rejects_snapshot_past_key) {
    // A snapshot of kMaxPos positions may only be filed under a key that
    // covers at least kMaxPos tokens; shorter keys are refused on save and,
    // for files that already exist, on read.
    MockBackendWithAdopt backend;
    std::string dir = test_tmp_path("dflash_test_past_key").string();
    rm_rf(dir);
    DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1;
    DiskPrefixCache cache(cfg, backend);
    TEST_ASSERT(cache.init());
    cache.learn_layout(0);

    std::vector<int32_t> short_key;
    for (int i = 0; i < MockBackendWithLayout::kMaxPos - 4; ++i) short_key.push_back(i + 1);
    std::vector<int32_t> full_key;
    for (int i = 0; i < MockBackendWithLayout::kMaxPos; ++i) full_key.push_back(i + 1);

    TEST_ASSERT(!cache.save(0, short_key));
    TEST_ASSERT(cache.total_bytes() == 0);
    TEST_ASSERT(cache.save(0, full_key));
    TEST_ASSERT(cache.total_bytes() > 0);
    TEST_ASSERT(!cache.lookup(short_key, 1));
    TEST_ASSERT(cache.lookup(full_key, 1));
    TEST_ASSERT(backend.adopted_cur_pos_ == MockBackendWithLayout::kMaxPos);

    // Forge the pre-fix shape on disk: key shorter than the snapshot. The
    // scan keys entries by the header's token hash, so only the header
    // fields need rewriting (token_count at byte 32, token_hash at byte 36
    // of the 80-byte field-by-field header).
    {
        std::string forged;
        for (auto & entry : fs::recursive_directory_iterator(dir)) {
            if (entry.path().extension() == ".dkv") {
                forged = (entry.path().parent_path() /
                          (std::string(32, 'f') + ".dkv")).string();
                fs::copy_file(entry.path(), forged, fs::copy_options::overwrite_existing);
                break;
            }
        }
        TEST_ASSERT(!forged.empty());
        FILE * f = std::fopen(forged.c_str(), "r+b");
        TEST_ASSERT(f != nullptr);
        if (f) {
            const uint32_t short_count = (uint32_t)short_key.size();
            PrefixHash ph = hash_prefix(short_key.data(), (int)short_key.size());
            std::fseek(f, 32, SEEK_SET);
            TEST_ASSERT(std::fwrite(&short_count, 4, 1, f) == 1);
            TEST_ASSERT(std::fwrite(ph.data(), 16, 1, f) == 1);
            std::fclose(f);
        }
        DiskPrefixCache reopened(cfg, backend);
        TEST_ASSERT(reopened.init());
        reopened.learn_layout(0);
        TEST_ASSERT(!reopened.lookup(short_key, 2));   // rejected + removed
        TEST_ASSERT(!fs::exists(forged));
        TEST_ASSERT(reopened.lookup(full_key, 2));     // consistent file survives
    }
    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_continued_keys_full_prefix) {
    // Continued checkpoints are paced by the interval but keyed by the
    // tokens the snapshot really covers, so only a prompt containing all of
    // them can hit.
    MockBackendWithAdopt backend;
    std::string dir = test_tmp_path("dflash_test_continued_key").string();
    rm_rf(dir);
    DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1;
    cfg.continued_interval = 10;   // 32 positions -> crosses at 30
    DiskPrefixCache cache(cfg, backend);
    TEST_ASSERT(cache.init());
    cache.learn_layout(0);

    std::vector<int32_t> tokens;
    for (int i = 0; i < 60; ++i) tokens.push_back(100 + i);
    const int cur_pos = MockBackendWithLayout::kMaxPos;  // 32 -> bucket 30
    backend.cur_pos_ = cur_pos;
    TEST_ASSERT(cache.maybe_store_continued(0, tokens, cur_pos));
    std::vector<int32_t> aligned(tokens.begin(), tokens.begin() + 30);
    std::vector<int32_t> covered(tokens.begin(), tokens.begin() + cur_pos);
    TEST_ASSERT(!cache.lookup(aligned, 1));
    TEST_ASSERT(cache.lookup(covered, 1));
    TEST_ASSERT(backend.adopted_cur_pos_ == cur_pos);
    const size_t bytes_after_first = cache.total_bytes();
    TEST_ASSERT(bytes_after_first > 0);

    // Same interval bucket: no second checkpoint.
    TEST_ASSERT(!cache.maybe_store_continued(0, tokens, cur_pos));
    backend.cur_pos_ = 38;  // still bucket 30
    TEST_ASSERT(!cache.maybe_store_continued(0, tokens, 38));
    TEST_ASSERT(cache.total_bytes() == bytes_after_first);

    // Crossing into bucket 40 fires again, keyed by the 42 covered tokens.
    backend.cur_pos_ = 42;
    TEST_ASSERT(cache.maybe_store_continued(0, tokens, 42));
    TEST_ASSERT(cache.total_bytes() > bytes_after_first);
    std::vector<int32_t> bucket(tokens.begin(), tokens.begin() + 40);
    std::vector<int32_t> covered2(tokens.begin(), tokens.begin() + 42);
    TEST_ASSERT(!cache.lookup(bucket, 2));
    TEST_ASSERT(cache.lookup(covered2, 2));
    TEST_ASSERT(backend.adopted_cur_pos_ == 42);
    // The earlier checkpoint is still there, and bucket 40 does not refire.
    TEST_ASSERT(cache.lookup(covered, 3));
    backend.cur_pos_ = 47;
    TEST_ASSERT(!cache.maybe_store_continued(0, tokens, 47));
    rm_rf(dir);
}
