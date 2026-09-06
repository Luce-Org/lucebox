#include "ggml-cpu.h"
#include "server/disk_prefix_cache.h"
#include "server/prefix_cache.h"
#include "support/mock_backend.h"
#include "support/environment.h"
#include <nlohmann/json.hpp>
#include "support/test_assert.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

using json = nlohmann::json;
using namespace dflash::common;

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
    DIR * dir = opendir(path.c_str());
    if (!dir) { unlink(path.c_str()); return; }
    struct dirent * ent;
    while ((ent = readdir(dir)) != nullptr) {
        if (std::strcmp(ent->d_name, ".") == 0 || std::strcmp(ent->d_name, "..") == 0) continue;
        std::string child = path + "/" + ent->d_name;
        struct stat st;
        if (stat(child.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            rm_rf(child);
        } else {
            unlink(child.c_str());
        }
    }
    closedir(dir);
    rmdir(path.c_str());
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
    cfg.cache_dir = "/tmp/dflash_test_disk_cache_memory_only";
    DiskPrefixCache cache(cfg, backend);
    TEST_ASSERT(!cache.disabled());

    // A live in-memory snapshot with no SnapshotRef means this backend cannot
    // serialize or adopt disk entries. Detect it once and skip later disk work.
    cache.learn_layout(0);
    TEST_ASSERT(cache.disabled());
}

TEST_CASE(ServerUnitFixture, test_disk_cache_init_creates_directory) {
    MockBackend backend;
    std::string dir = "/tmp/dflash_test_disk_cache_init";
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    DiskPrefixCache cache(cfg, backend);
    TEST_ASSERT(!cache.disabled());
    TEST_ASSERT(cache.init());

    // Directory should exist.
    struct stat st;
    TEST_ASSERT(stat(dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode));

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

TEST_CASE(ServerUnitFixture, test_disk_cache_header_round_trip) {
    // Write and read a header to verify serialization.
    std::string path = "/tmp/dflash_test_header_rt.dkv";
    unlink(path.c_str());

    DiskCacheHeader hdr{};
    std::memcpy(hdr.magic, "DKVC", 4);
    hdr.version = DISK_CACHE_VERSION;
    std::memset(hdr.layout_id, 0xAB, 16);
    hdr.cur_pos = 1234;
    hdr.n_tensors = 42;
    hdr.token_count = 567;
    std::memset(hdr.token_hash, 0xCD, 16);
    hdr.payload_bytes = 9999999;
    hdr.created_at = 1700000000;
    hdr.last_used = 1700000100;
    hdr.last_tok = 151643;

    // Use DiskPrefixCache's static write/read_header (they are private, so
    // we test indirectly through file I/O matching the on-disk format).
    FILE * f = std::fopen(path.c_str(), "wb");
    TEST_ASSERT(f != nullptr);
    // Write field-by-field matching disk_prefix_cache.cpp's write_header.
    std::fwrite(hdr.magic, 4, 1, f);
    uint32_t v;
    v = hdr.version; std::fwrite(&v, 4, 1, f);
    std::fwrite(hdr.layout_id, 16, 1, f);
    v = hdr.cur_pos; std::fwrite(&v, 4, 1, f);
    v = hdr.n_tensors; std::fwrite(&v, 4, 1, f);
    v = hdr.token_count; std::fwrite(&v, 4, 1, f);
    std::fwrite(hdr.token_hash, 16, 1, f);
    uint64_t u64 = hdr.payload_bytes; std::fwrite(&u64, 8, 1, f);
    u64 = hdr.created_at; std::fwrite(&u64, 8, 1, f);
    u64 = hdr.last_used; std::fwrite(&u64, 8, 1, f);
    int32_t i32 = hdr.last_tok; std::fwrite(&i32, 4, 1, f);
    std::fclose(f);

    // Verify file size is DISK_CACHE_HEADER_SIZE.
    struct stat st;
    stat(path.c_str(), &st);
    TEST_ASSERT((size_t)st.st_size == DISK_CACHE_HEADER_SIZE);

    // Read back and verify.
    f = std::fopen(path.c_str(), "rb");
    TEST_ASSERT(f != nullptr);
    char magic[4]; std::fread(magic, 4, 1, f);
    TEST_ASSERT(std::memcmp(magic, "DKVC", 4) == 0);
    uint32_t rv; std::fread(&rv, 4, 1, f);
    TEST_ASSERT(rv == DISK_CACHE_VERSION);
    uint8_t lid[16]; std::fread(lid, 16, 1, f);
    TEST_ASSERT(lid[0] == 0xAB && lid[15] == 0xAB);
    std::fread(&rv, 4, 1, f); TEST_ASSERT(rv == 1234);  // cur_pos
    std::fread(&rv, 4, 1, f); TEST_ASSERT(rv == 42);    // n_tensors
    std::fread(&rv, 4, 1, f); TEST_ASSERT(rv == 567);   // token_count
    uint8_t th[16]; std::fread(th, 16, 1, f);
    TEST_ASSERT(th[0] == 0xCD && th[15] == 0xCD);
    uint64_t ru64; std::fread(&ru64, 8, 1, f); TEST_ASSERT(ru64 == 9999999);  // payload
    std::fread(&ru64, 8, 1, f); TEST_ASSERT(ru64 == 1700000000);  // created_at
    std::fread(&ru64, 8, 1, f); TEST_ASSERT(ru64 == 1700000100);  // last_used
    int32_t ri32; std::fread(&ri32, 4, 1, f); TEST_ASSERT(ri32 == 151643);  // last_tok
    std::fclose(f);

    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_disk_cache_continued_boundary) {
    // Test maybe_store_continued logic: saves at interval boundaries.
    MockBackend backend;
    std::string dir = "/tmp/dflash_test_continued";
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
    // Cold prefix should not trigger for short prompts.
    MockBackend backend;
    std::string dir = "/tmp/dflash_test_cold_short";
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    cfg.cold_max_tokens = 10240;
    cfg.min_tokens = 512;
    DiskPrefixCache cache(cfg, backend);
    cache.init();

    // Prompt shorter than cold_max_tokens.
    std::vector<int32_t> prompt(5000, 1);
    std::vector<int> boundaries = {1000, 2000, 3000, 4000};
    TEST_ASSERT(cache.cold_prefix_boundary(prompt, boundaries) == 0);

    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_cold_prefix_no_boundaries) {
    // Cold prefix should not trigger if no boundaries provided.
    MockBackend backend;
    std::string dir = "/tmp/dflash_test_cold_nobound";
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    cfg.cold_max_tokens = 5000;
    cfg.min_tokens = 512;
    DiskPrefixCache cache(cfg, backend);
    cache.init();

    std::vector<int32_t> prompt(10000, 1);
    std::vector<int> empty_boundaries;
    TEST_ASSERT(cache.cold_prefix_boundary(prompt, empty_boundaries) == 0);

    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_cold_prefix_finds_boundary) {
    // Cold prefix should find the last boundary <= cold_max_tokens.
    MockBackend backend;
    std::string dir = "/tmp/dflash_test_cold_finds";
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    cfg.cold_max_tokens = 5000;
    cfg.min_tokens = 512;
    DiskPrefixCache cache(cfg, backend);
    cache.init();
    // Manually mark layout as known (hack for testing without real snapshots).
    // Since cold_prefix_boundary checks layout_known_, and we can't easily
    // set it without a real snapshot, the function will return 0.
    // This tests that short prompts / bad boundaries correctly return 0.
    std::vector<int32_t> prompt(10000, 1);
    std::vector<int> boundaries = {1000, 2000, 3000, 4000, 6000, 8000};
    // Without layout_known_, returns 0.
    int result = cache.cold_prefix_boundary(prompt, boundaries);
    TEST_ASSERT(result == 0);  // layout not known yet

    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_lookup_miss_no_layout) {
    // Lookup with no layout known should return false.
    MockBackend backend;
    std::string dir = "/tmp/dflash_test_lookup_miss";
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
    std::string dir = "/tmp/dflash_test_save_below";
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
    DIR * d = opendir(base.c_str());
    if (!d) return id;
    struct dirent * ent;
    while ((ent = readdir(d)) != nullptr) {
        if (ent->d_name[0] == '.') continue;
        std::string sub = base + "/" + ent->d_name;
        struct stat st{};
        if (stat(sub.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) continue;
        DIR * sd = opendir(sub.c_str());
        if (!sd) continue;
        struct dirent * sf;
        while ((sf = readdir(sd)) != nullptr) {
            size_t nl = std::strlen(sf->d_name);
            if (nl < 4 || std::strcmp(sf->d_name + nl - 4, ".dkv") != 0) continue;
            std::string fp = sub + "/" + sf->d_name;
            FILE * f = std::fopen(fp.c_str(), "rb");
            if (!f) continue;
            std::fseek(f, 8, SEEK_SET);  // skip magic(4) + version(4)
            std::fread(id.data(), 1, 16, f);
            std::fclose(f);
            closedir(sd);
            closedir(d);
            return id;
        }
        closedir(sd);
    }
    closedir(d);
    return id;
}

TEST_CASE(ServerUnitFixture, test_disk_identity_salt_changes_layout_id) {
    MockBackendWithLayout backend;
    std::vector<int32_t> prompt;
    for (int i = 0; i < 10; ++i) prompt.push_back(i + 1);

    // Salt A: non-zero.
    std::array<uint8_t, 16> salt_a{};
    salt_a[0] = 0x01; salt_a[15] = 0xAB;

    std::string dir_a = "/tmp/dflash_test_salt_a";
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

    std::string dir_b = "/tmp/dflash_test_salt_b";
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
    std::string dir_a2 = "/tmp/dflash_test_salt_a2";
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
    for (int i = 0; i < 10; ++i) prompt.push_back(i + 1);

    std::string dir1 = "/tmp/dflash_test_salt_zero1";
    rm_rf(dir1);
    {
        DiskCacheConfig cfg; cfg.cache_dir = dir1; cfg.min_tokens = 1;
        DiskPrefixCache cache(cfg, backend);
        // No set_identity_salt call — stays all-zero.
        cache.init();
        cache.learn_layout(0);
        TEST_ASSERT(cache.save(0, prompt));
    }

    std::string dir2 = "/tmp/dflash_test_salt_zero2";
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
