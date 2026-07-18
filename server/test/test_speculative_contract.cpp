#include "common/speculative_contract.h"

#include <cstdio>

using namespace dflash::common;

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

int main() {
    SpeculativeContract parsed = SpeculativeContract::Approximate;
    CHECK(parse_speculative_contract("exact", parsed));
    CHECK(parsed == SpeculativeContract::Exact);
    CHECK(parse_speculative_contract("approximate", parsed));
    CHECK(parsed == SpeculativeContract::Approximate);
    CHECK(!parse_speculative_contract("fast", parsed));

    const auto exact_chain = resolve_speculative_route(
        SpeculativeContract::Exact, true, false, false, false);
    CHECK(!exact_chain.use_speculative);
    CHECK(exact_chain.effective_mode == EffectiveDecodeMode::Autoregressive);
    CHECK(exact_chain.fallback_reason ==
          SpeculativeFallbackReason::ExactContractRequiresAR);

    const auto exact_tree = resolve_speculative_route(
        SpeculativeContract::Exact, true, true, false, false);
    CHECK(!exact_tree.use_speculative);
    CHECK(exact_tree.effective_mode == EffectiveDecodeMode::Autoregressive);

    const auto approximate_chain = resolve_speculative_route(
        SpeculativeContract::Approximate, true, false, false, false);
    CHECK(approximate_chain.use_speculative);
    CHECK(approximate_chain.effective_mode ==
          EffectiveDecodeMode::ApproximateChain);

    const auto approximate_tree = resolve_speculative_route(
        SpeculativeContract::Approximate, true, true, false, false);
    CHECK(approximate_tree.use_speculative);
    CHECK(approximate_tree.effective_mode ==
          EffectiveDecodeMode::ApproximateDDTree);

    const auto sampled = resolve_speculative_route(
        SpeculativeContract::Approximate, true, true, true, false);
    CHECK(!sampled.use_speculative);
    CHECK(sampled.fallback_reason ==
          SpeculativeFallbackReason::SamplingUnsupported);

    const auto forced = resolve_speculative_route(
        SpeculativeContract::Approximate, true, false, false, true);
    CHECK(!forced.use_speculative);
    CHECK(forced.fallback_reason ==
          SpeculativeFallbackReason::ForcedAutoregressive);

    const auto no_draft = resolve_speculative_route(
        SpeculativeContract::Approximate, false, true, false, false);
    CHECK(!no_draft.use_speculative);
    CHECK(no_draft.fallback_reason == SpeculativeFallbackReason::None);

    if (failures) return 1;
    std::fprintf(stderr, "speculative contract routing: PASS\n");
    return 0;
}
