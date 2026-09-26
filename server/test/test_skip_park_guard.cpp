// Unit tests for the skip-park policy — GPU-free.
//
// Covers: the VMM crash guard (skip_park_allowed) and its VMM-pool scope, the
// auto|on|off mode parse, the startup estimator (skip_park_required_bytes,
// inspect_drafter_footprint on a synthetic GGUF header), the precedence
// contract of resolve_skip_park — explicit off/on beat the estimate, the
// guard beats an explicit on, auto follows the footprint vs. free VRAM and
// alone may keep the drafter loaded — and the runtime fail-safe
// (SkipParkFallback, run_skip_park_window): only out-of-memory windows retry
// parked, and the park latch backs off instead of lasting forever.

#include "CppUnitTestFramework.hpp"
#include "common/gguf_inspect.h"
#include "placement/skip_park_guard.h"

#include "gguf.h"
#include "kv_quant.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
struct SkipParkGuardFixture {};
}

using namespace luce::common;

static constexpr size_t GiB = 1024ull * 1024 * 1024;

TEST_CASE(SkipParkGuardFixture, T1_not_requested_stays_off) {
    CHECK(!skip_park_allowed(false, 24 * GiB, 32768));
}

TEST_CASE(SkipParkGuardFixture, T2_big_card_any_ctx) {
    CHECK(skip_park_allowed(true, 32 * GiB, 131072));
}

TEST_CASE(SkipParkGuardFixture, T3_small_card_small_ctx_allowed) {
    CHECK(skip_park_allowed(true, 24 * GiB, 65536));
}

TEST_CASE(SkipParkGuardFixture, T4_small_card_big_ctx_downgraded) {
    CHECK(!skip_park_allowed(true, 24 * GiB, 131072));
}

TEST_CASE(SkipParkGuardFixture, T5_boundary_ctx_one_over) {
    CHECK(!skip_park_allowed(true, 24 * GiB, 65537));
}

TEST_CASE(SkipParkGuardFixture, T6_boundary_vram_just_under_32g) {
    CHECK(!skip_park_allowed(true, 32 * GiB - 1, 131072));
}

// ── Mode parse ────────────────────────────────────────────────────────────

TEST_CASE(SkipParkGuardFixture, T7_mode_parse) {
    SkipParkMode m = SkipParkMode::Auto;
    CHECK(parse_skip_park_mode("auto", m) && m == SkipParkMode::Auto);
    CHECK(parse_skip_park_mode("on", m) && m == SkipParkMode::On);
    CHECK(parse_skip_park_mode("off", m) && m == SkipParkMode::Off);
    CHECK(!parse_skip_park_mode("yes", m));
    CHECK(!parse_skip_park_mode("", m));
    CHECK(std::string(skip_park_mode_name(SkipParkMode::Auto)) == "auto");
    CHECK(std::string(skip_park_mode_name(SkipParkMode::On)) == "on");
    CHECK(std::string(skip_park_mode_name(SkipParkMode::Off)) == "off");
}

// ── Startup resolution ────────────────────────────────────────────────────

// Qwen3.5-0.8B-ish footprint: ~1.6 GiB weights, ~15 KB/token hybrid runtime.
static SkipParkDrafterInfo qwen35_like_info() {
    SkipParkDrafterInfo i;
    i.recognized = true;
    i.weights_bytes = int64_t(1668 * 1024 * 1024);
    i.runtime_bytes_per_token = 15000;
    i.fixed_bytes = 384ll * 1024 * 1024;
    i.context_length = 262144;
    return i;
}

TEST_CASE(SkipParkGuardFixture, T8_explicit_off_always_wins) {
    const auto d = resolve_skip_park(SkipParkMode::Off, /*drafter=*/true,
                                     qwen35_like_info(),
                                     /*free=*/64 * GiB, /*total=*/64 * GiB,
                                     /*max_ctx=*/131072);
    CHECK(!d.enabled);
}

TEST_CASE(SkipParkGuardFixture, T9_explicit_on_bypasses_estimate) {
    // Free VRAM far too small for the estimate — explicit on wins anyway.
    const auto d = resolve_skip_park(SkipParkMode::On, /*drafter=*/true,
                                     qwen35_like_info(),
                                     /*free=*/1, /*total=*/32 * GiB,
                                     /*max_ctx=*/131072);
    CHECK(d.enabled);
}

TEST_CASE(SkipParkGuardFixture, T10_explicit_on_blocked_by_guard) {
    // 24 GiB card with ctx > 64K: the crash guard beats even force-on.
    const auto d = resolve_skip_park(SkipParkMode::On, /*drafter=*/true,
                                     qwen35_like_info(),
                                     /*free=*/16 * GiB, /*total=*/24 * GiB,
                                     /*max_ctx=*/131072);
    CHECK(!d.enabled);
    // Same card at 64K ctx is under the guard — force-on honored.
    const auto ok = resolve_skip_park(SkipParkMode::On, /*drafter=*/true,
                                      qwen35_like_info(),
                                      /*free=*/16 * GiB, /*total=*/24 * GiB,
                                      /*max_ctx=*/65536);
    CHECK(ok.enabled);
}

TEST_CASE(SkipParkGuardFixture, T11_auto_fits_with_margin) {
    // required = (1.63 GiB + 0.375 GiB + 15000·131072) × 1.25 ≈ 4.9 GiB.
    const auto d = resolve_skip_park(SkipParkMode::Auto, /*drafter=*/true,
                                     qwen35_like_info(),
                                     /*free=*/8 * GiB, /*total=*/32 * GiB,
                                     /*max_ctx=*/131072);
    CHECK(d.enabled);
    CHECK(d.window_tokens == 131072);
    CHECK(d.required_bytes > 0);
}

TEST_CASE(SkipParkGuardFixture, T12_auto_exceeds_free) {
    const auto d = resolve_skip_park(SkipParkMode::Auto, /*drafter=*/true,
                                     qwen35_like_info(),
                                     /*free=*/1 * GiB, /*total=*/32 * GiB,
                                     /*max_ctx=*/131072);
    CHECK(!d.enabled);
}

TEST_CASE(SkipParkGuardFixture, T13_auto_unknown_footprint_off) {
    const auto d = resolve_skip_park(SkipParkMode::Auto, /*drafter=*/true,
                                     SkipParkDrafterInfo{},
                                     /*free=*/32 * GiB, /*total=*/32 * GiB,
                                     /*max_ctx=*/131072);
    CHECK(!d.enabled);
}

TEST_CASE(SkipParkGuardFixture, T14_no_drafter_off_in_all_modes) {
    CHECK(!resolve_skip_park(SkipParkMode::Auto, false, qwen35_like_info(),
                           32 * GiB, 32 * GiB, 131072).enabled);
    CHECK(!resolve_skip_park(SkipParkMode::On, false, qwen35_like_info(),
                           32 * GiB, 32 * GiB, 131072).enabled);
}

TEST_CASE(SkipParkGuardFixture, T15_window_capped_by_drafter_ctx) {
    // max_ctx above the drafter's native window must not inflate the estimate.
    const auto d = resolve_skip_park(SkipParkMode::Auto, /*drafter=*/true,
                                     qwen35_like_info(),
                                     /*free=*/64 * GiB, /*total=*/80 * GiB,
                                     /*max_ctx=*/1048576);
    CHECK(d.window_tokens == 262144);
}

TEST_CASE(SkipParkGuardFixture, T16_estimate_margin_and_guard_on_auto) {
    const auto info = qwen35_like_info();
    const int64_t raw = info.weights_bytes + info.fixed_bytes +
                        info.runtime_bytes_per_token * 65536;
    CHECK(skip_park_required_bytes(info, 65536) ==
          raw + raw * kSkipParkSafetyMarginPercent / 100);
    // Guard still applies under auto: <32 GiB with ctx>64K resolves off even
    // when the footprint would fit.
    const auto d = resolve_skip_park(SkipParkMode::Auto, /*drafter=*/true,
                                     info, /*free=*/20 * GiB,
                                     /*total=*/24 * GiB, /*max_ctx=*/131072);
    CHECK(!d.enabled);
}

// ── VMM guard scope ───────────────────────────────────────────────────────

TEST_CASE(SkipParkGuardFixture, T17_guard_only_with_vmm_pool) {
    // The R9700 reports 34.2 GB = 31.86 GiB: under the 32 GiB line. Without
    // the VMM pool (HIP default) the guard does not apply.
    const size_t r9700 = 34208743424ull;
    CHECK(!skip_park_allowed(true, r9700, 131072, /*vmm_pool=*/true));
    CHECK(skip_park_allowed(true, r9700, 131072, /*vmm_pool=*/false));
    CHECK(!skip_park_allowed(false, r9700, 131072, /*vmm_pool=*/false));
    // A 24 GiB CUDA card with the VMM pool keeps the guard.
    CHECK(!skip_park_allowed(true, 24 * GiB, 131072, /*vmm_pool=*/true));

    const auto on_hip = resolve_skip_park(SkipParkMode::On, true,
                                          qwen35_like_info(), 16 * GiB,
                                          int64_t(r9700), 131072,
                                          /*vmm_pool=*/false);
    CHECK(on_hip.enabled);
    const auto on_cuda = resolve_skip_park(SkipParkMode::On, true,
                                           qwen35_like_info(), 16 * GiB,
                                           int64_t(r9700), 131072,
                                           /*vmm_pool=*/true);
    CHECK(!on_cuda.enabled);
}

// ── Keep-loaded decision ──────────────────────────────────────────────────

TEST_CASE(SkipParkGuardFixture, T18_keep_loaded_only_from_auto_estimate) {
    const auto info = qwen35_like_info();
    // Ample: window and keep-loaded footprints both fit.
    const auto ample = resolve_skip_park(SkipParkMode::Auto, true, info,
                                         /*free=*/16 * GiB, 32 * GiB, 32768);
    CHECK(ample.enabled);
    CHECK(ample.keep_drafter_loaded);
    CHECK(ample.keep_loaded_bytes ==
          skip_park_with_margin(skip_park_raw_bytes(info, 32768) +
                                kTargetComputeReserveBytes));
    CHECK(ample.keep_loaded_bytes > ample.required_bytes);

    // Free VRAM between the two: skip-park on, drafter still released.
    const int64_t between = (ample.required_bytes + ample.keep_loaded_bytes) / 2;
    const auto tight = resolve_skip_park(SkipParkMode::Auto, true, info,
                                         between, 32 * GiB, 32768);
    CHECK(tight.enabled);
    CHECK(!tight.keep_drafter_loaded);

    // Explicit on runs no estimate, so it never keeps the drafter loaded.
    const auto forced = resolve_skip_park(SkipParkMode::On, true, info,
                                          /*free=*/64 * GiB, 80 * GiB, 32768);
    CHECK(forced.enabled);
    CHECK(!forced.keep_drafter_loaded);
    CHECK(!resolve_skip_park(SkipParkMode::Off, true, info, 64 * GiB,
                             80 * GiB, 32768).keep_drafter_loaded);
}

// ── Estimator on a synthetic drafter header ───────────────────────────────

namespace {

std::string write_drafter_header(const char * arch, bool full) {
    gguf_context * ctx = gguf_init_empty();
    gguf_set_val_str(ctx, "general.architecture", arch);
    const std::string a = arch;
    if (full) {
        gguf_set_val_u32(ctx, (a + ".block_count").c_str(), 24);
        gguf_set_val_u32(ctx, (a + ".embedding_length").c_str(), 1024);
        gguf_set_val_u32(ctx, (a + ".attention.head_count").c_str(), 8);
        gguf_set_val_u32(ctx, (a + ".attention.head_count_kv").c_str(), 2);
        gguf_set_val_u32(ctx, (a + ".attention.key_length").c_str(), 256);
        gguf_set_val_u32(ctx, (a + ".full_attention_interval").c_str(), 4);
        gguf_set_val_u32(ctx, (a + ".context_length").c_str(), 262144);
    }
    const auto path = std::filesystem::temp_directory_path() /
        ("luce-skippark-" + std::to_string(getpid()) + "-" + a +
         (full ? "-full" : "-bare") + ".gguf");
    gguf_write_to_file(ctx, path.c_str(), false);
    gguf_free(ctx);
    return path.string();
}

}  // namespace

TEST_CASE(SkipParkGuardFixture, T19_estimator_qwen35_header) {
    const std::string full = write_drafter_header("qwen35", true);
    const std::string bare = write_drafter_header("qwen35", false);
    const std::string other = write_drafter_header("qwen3", true);

    SkipParkDrafterInfo q8, q128, no_sessions, defaults, unknown;
    CHECK(inspect_drafter_footprint(full, 8, 2, q8));
    CHECK(inspect_drafter_footprint(full, 128, 2, q128));
    CHECK(inspect_drafter_footprint(full, 128, 0, no_sessions));
    CHECK(inspect_drafter_footprint(bare, 128, 2, defaults));
    CHECK(!inspect_drafter_footprint(other, 128, 2, unknown));
    CHECK(!unknown.recognized);

    CHECK(q8.recognized && q8.context_length == 262144);
    // The query window's logits cost (n_head + 1)·4 bytes per query token.
    CHECK(q128.runtime_bytes_per_token - q8.runtime_bytes_per_token ==
          int64_t(128 - 8) * (8 + 1) * 4);
    // Kept sessions (×2, sized ×1.5) dominate a scratch session.
    CHECK(no_sessions.runtime_bytes_per_token < q128.runtime_bytes_per_token);
    CHECK(no_sessions.fixed_bytes < q128.fixed_bytes);
    // A header without dims falls back to the Qwen3.5-0.8B shape.
    CHECK(defaults.runtime_bytes_per_token == q128.runtime_bytes_per_token);

    // Strict scorer per token: 2 sessions × 1.5 × (KV of blocks 0..14 +
    // f32 keys) + activations + logits/mask + probe.
    ggml_type k = GGML_TYPE_Q4_0, v = GGML_TYPE_Q4_0;
    luce::resolve_kv_types(k, v);
    const int64_t session = int64_t(luce::kv_reservation_bytes_per_token(
        15, 4, 2, k, 256, v, 256)) + 256 * 2 * 4;
    const int64_t strict = 2 * session * 3 / 2 + 2 * 1024 * 4 +
                           128 * 9 * 4 + 8;
    CHECK(q128.runtime_bytes_per_token >= strict);

    for (const auto & p : {full, bare, other}) std::filesystem::remove(p);
}

// ── Runtime fail-safe ─────────────────────────────────────────────────────

namespace {

// Fake window: each attempt pops the next outcome; records park flags.
struct FakeWindows {
    std::vector<SkipParkWindowOutcome> script;
    std::vector<bool> parks;
    int drops = 0;
    SkipParkWindowOutcome run(bool park) {
        parks.push_back(park);
        if (script.empty()) return SkipParkWindowOutcome::Ok;
        const auto next = script.front();
        script.erase(script.begin());
        return next;
    }
};

SkipParkWindowOutcome window(bool skip, SkipParkFallback & fb,
                             FakeWindows & fake) {
    return run_skip_park_window(
        skip, fb, [&](bool park) { return fake.run(park); },
        [](SkipParkWindowOutcome o) { return o; },
        [&]() { ++fake.drops; }, "[test]");
}

}  // namespace

TEST_CASE(SkipParkGuardFixture, T20_non_oom_failure_does_not_retry) {
    // e.g. non-finite scoring-head scores: parking cannot fix it.
    SkipParkFallback fb;
    FakeWindows fake{{SkipParkWindowOutcome::Failed}};
    CHECK(window(true, fb, fake) == SkipParkWindowOutcome::Failed);
    CHECK(fake.parks == std::vector<bool>{false});
    CHECK(fake.drops == 0);
    CHECK(!fb.parking_forced());
    CHECK(!fb.memory_tight());
}

TEST_CASE(SkipParkGuardFixture, T21_oom_retries_parked_once) {
    SkipParkFallback fb;
    FakeWindows fake{{SkipParkWindowOutcome::OutOfMemory,
                      SkipParkWindowOutcome::Ok}};
    CHECK(window(true, fb, fake) == SkipParkWindowOutcome::Ok);
    CHECK((fake.parks == std::vector<bool>{false, true}));
    CHECK(fake.drops == 1);
    CHECK(fb.memory_tight());
    CHECK(fb.forced_windows() == SkipParkFallback::kInitialBackoffWindows);

    // A parked retry that fails too returns its failure; no latch change
    // beyond the earlier one, and no second retry.
    SkipParkFallback fb2;
    FakeWindows fake2{{SkipParkWindowOutcome::OutOfMemory,
                       SkipParkWindowOutcome::OutOfMemory}};
    CHECK(window(true, fb2, fake2) == SkipParkWindowOutcome::OutOfMemory);
    CHECK(fake2.parks.size() == 2);
    CHECK(!fb2.parking_forced());
}

TEST_CASE(SkipParkGuardFixture, T22_latch_backs_off_then_reprobes) {
    SkipParkFallback fb;
    FakeWindows fake{{SkipParkWindowOutcome::OutOfMemory,
                      SkipParkWindowOutcome::Ok}};
    window(true, fb, fake);
    fake.parks.clear();
    // The next kInitialBackoffWindows windows park outright.
    for (int i = 0; i < SkipParkFallback::kInitialBackoffWindows; ++i) {
        CHECK(window(true, fb, fake) == SkipParkWindowOutcome::Ok);
    }
    CHECK(fake.parks ==
          std::vector<bool>(SkipParkFallback::kInitialBackoffWindows, true));
    CHECK(fb.memory_tight());   // no clean no-park window yet
    // Then skip-park is probed again; OOM again doubles the backoff.
    fake.parks.clear();
    fake.script = {SkipParkWindowOutcome::OutOfMemory,
                   SkipParkWindowOutcome::Ok};
    window(true, fb, fake);
    CHECK((fake.parks == std::vector<bool>{false, true}));
    CHECK(fb.forced_windows() == 2 * SkipParkFallback::kInitialBackoffWindows);
    // Drain, then a clean no-park window resets everything.
    for (int i = 0; i < 2 * SkipParkFallback::kInitialBackoffWindows; ++i) {
        window(true, fb, fake);
    }
    fake.parks.clear();
    CHECK(window(true, fb, fake) == SkipParkWindowOutcome::Ok);
    CHECK(fake.parks == std::vector<bool>{false});
    CHECK(!fb.memory_tight());
    CHECK(fb.backoff() == 0);
}

TEST_CASE(SkipParkGuardFixture, T23_backoff_caps_and_park_requests_untouched) {
    SkipParkFallback fb;
    for (int i = 0; i < 10; ++i) fb.on_oom_recovered();
    CHECK(fb.backoff() == SkipParkFallback::kMaxBackoffWindows);
    // A window that asks to park never consults or advances the latch.
    FakeWindows fake;
    const int before = fb.forced_windows();
    CHECK(window(false, fb, fake) == SkipParkWindowOutcome::Ok);
    CHECK(fake.parks == std::vector<bool>{true});
    CHECK(fb.forced_windows() == before);
}
