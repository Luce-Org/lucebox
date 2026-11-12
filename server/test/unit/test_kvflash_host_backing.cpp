// LUCE_KVFLASH_HOST_BACKING, the host-backing policy of the KVFlash slot pagers (default pinned).
// Pure parsing and the failure state machine, plus a CPU pager with an injected pinned-allocation failure
// for the ledger (host_bytes) and the apply-to-every-slot contract. No GPU.

#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"
#include "kvflash_pager_test_host.h"

#include "kvflash_host_backing.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashHostBackingFixture {};

struct Harness {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;
    Harness() {
        backend = ggml_backend_cpu_init();
        ggml_init_params ip{};
        ip.mem_size = 16 * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        for (int l = 0; l < 2; l++) {
            k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, 512, 2));
            v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, 512, 2));
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }
    ~Harness() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
};
KvFlashConfig cfg8() {
    KvFlashConfig c;
    c.pool_tokens = 512; c.chunk_tokens = 64; c.sink_chunks = 1; c.tail_window_chunks = 2;
    return c;
}
}  // namespace

TEST_CASE(KvflashHostBackingFixture, parse_accepts_the_three_values_and_defaults_to_pinned) {
    bool ok = false;
    CHECK(kvflash_host_backing_parse(nullptr, &ok) == KvFlashHostBacking::Pinned); CHECK(ok);
    CHECK(kvflash_host_backing_parse("", &ok) == KvFlashHostBacking::Pinned);      CHECK(ok);
    CHECK(kvflash_host_backing_parse("auto", &ok) == KvFlashHostBacking::Auto);    CHECK(ok);
    CHECK(kvflash_host_backing_parse("pinned", &ok) == KvFlashHostBacking::Pinned);   CHECK(ok);
    CHECK(kvflash_host_backing_parse("pageable", &ok) == KvFlashHostBacking::Pageable); CHECK(ok);
    CHECK(kvflash_host_backing_parse("PINNED", &ok) == KvFlashHostBacking::Pinned);   CHECK(ok);   // case-insensitive
    CHECK(kvflash_host_backing_parse("mmap", &ok) == KvFlashHostBacking::Pinned);  CHECK(!ok);     // unknown: pinned, flagged
}

TEST_CASE(KvflashHostBackingFixture, env_selects_the_mode) {
    {
        luce_test::ScopedEnvVar e("LUCE_KVFLASH_HOST_BACKING", "pageable");
        CHECK(kvflash_host_backing_from_env() == KvFlashHostBacking::Pageable);
    }
    {
        luce_test::ScopedEnvVar e("LUCE_KVFLASH_HOST_BACKING", "auto");
        CHECK(kvflash_host_backing_from_env() == KvFlashHostBacking::Auto);
    }
    {
        luce_test::ScopedEnvVar e("LUCE_KVFLASH_HOST_BACKING", nullptr);
        CHECK(kvflash_host_backing_from_env() == KvFlashHostBacking::Pinned);   // unset: today's behaviour
    }
}

TEST_CASE(KvflashHostBackingFixture, state_machine_auto_falls_back_once_and_logs_it) {
    KvFlashBackingState s(KvFlashHostBacking::Auto);
    CHECK(s.pinned());
    const KvFlashBackingDecision d = s.on_pinned_failure("out of memory");
    CHECK(d.retry_pageable);
    CHECK(!s.pinned());
    CHECK(s.fallbacks() == 1);
    CHECK(d.log.find("host backing: pageable (pinned refused") != std::string::npos);
    CHECK(d.log.find("out of memory") != std::string::npos);
}

TEST_CASE(KvflashHostBackingFixture, state_machine_pinned_never_falls_back_and_logs_the_failure) {
    KvFlashBackingState s(KvFlashHostBacking::Pinned);
    const KvFlashBackingDecision d = s.on_pinned_failure("out of memory");
    CHECK(!d.retry_pageable);
    CHECK(s.pinned());                                   // the policy stays
    CHECK(s.failures() == 1);
    CHECK(d.log.find("pinned allocation failed") != std::string::npos);
    CHECK(d.log.find("no fallback") != std::string::npos);
}

TEST_CASE(KvflashHostBackingFixture, state_machine_pageable_starts_unpinned_and_pageable_failure_is_counted) {
    KvFlashBackingState s(KvFlashHostBacking::Pageable);
    CHECK(!s.pinned());
    const KvFlashBackingDecision d = s.on_pageable_failure();
    CHECK(!d.retry_pageable);
    CHECK(s.failures() == 1);
    CHECK(d.log.find("pageable allocation failed") != std::string::npos);
}

#ifdef KVFLASH_HAS_ASYNC_DMA
TEST_CASE(KvflashHostBackingFixture, auto_pager_survives_a_refused_pinned_allocation_and_the_ledger_counts_bytes_once) {
    Harness h;
    KvFlashPager p;
    p.set_host_backing(KvFlashHostBacking::Auto);
    int refused = 0;
    p.pinned_alloc_hook = [&](void **, size_t) { refused++; return false; };
    REQUIRE(p.attach(cfg8(), h.k, h.v));
    REQUIRE(p.alloc_span(0, 3 * 64));
    REQUIRE(p.page_out(1));                              // the pinned allocation is refused, the pageable one serves
    CHECK(refused == 1);
    CHECK(p.stats().host_bytes == (int64_t)p.chunk_bytes());
    REQUIRE(p.page_out(2));                              // the pager stays pageable: no second refusal
    CHECK(refused == 1);
    CHECK(p.stats().host_bytes == 2 * (int64_t)p.chunk_bytes());
    CHECK(p.host_backing_state().fallbacks() == 1);
    CHECK(!p.host_backing_state().pinned());
    REQUIRE(p.page_in(1));
    p.synchronize_paging();
}

TEST_CASE(KvflashHostBackingFixture, pinned_pager_fails_the_page_out_without_charging_the_ledger) {
    Harness h;
    KvFlashPager p;
    p.set_host_backing(KvFlashHostBacking::Pinned);
    p.pinned_alloc_hook = [](void **, size_t) { return false; };
    REQUIRE(p.attach(cfg8(), h.k, h.v));
    REQUIRE(p.alloc_span(0, 3 * 64));
    CHECK(!p.page_out(1));
    CHECK(p.stats().host_bytes == 0);
    CHECK(p.host_backing_state().failures() == 1);
}
#endif

TEST_CASE(KvflashHostBackingFixture, apply_reaches_every_slot_pager) {
    Harness h;
    std::vector<std::unique_ptr<KvFlashPager>> ps;
    for (int i = 0; i < 3; i++) ps.push_back(std::make_unique<KvFlashPager>());
    std::vector<KvFlashPager *> ptrs;
    for (auto & p : ps) ptrs.push_back(p.get());
    kvflash_apply_host_backing(ptrs, KvFlashHostBacking::Pageable);
    for (auto & p : ps) CHECK(!p->host_backing_state().pinned());
    kvflash_apply_host_backing(ptrs, KvFlashHostBacking::Pinned);
    for (auto & p : ps) CHECK(p->host_backing_state().pinned());
}
