#include "deepseek4_image_budget.h"
#include "deepseek4_image_spans.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using dflash::vision::ImageSpanView;
using dflash::vision::TokenSpan;
using dflash::vision::atomic_image_chunk;
using dflash::vision::image_block_at;
using dflash::vision::remaining_expert_budget;
using dflash::vision::valid_image_spans;
constexpr uint64_t MAX = std::numeric_limits<uint64_t>::max();
size_t checks = 0;

void require(bool condition, const std::string & description) {
    ++checks;
    if (!condition) throw std::runtime_error(description);
}

ImageSpanView view(const std::vector<TokenSpan> & spans) {
    return {spans.data(), spans.size()};
}

bool bisects(const std::vector<TokenSpan> & spans, uint64_t point) {
    for (const auto & span : spans) {
        if (span.block_begin < point && point < span.block_end) return true;
    }
    return false;
}

// Enumerate every legal endpoint, independently of the helper's interval walk.
// A small preferred chunk may expand only when no positive legal endpoint fits.
int oracle(const std::vector<TokenSpan> & spans, uint64_t position,
           int preferred, uint64_t remaining, int capacity) {
    if (preferred <= 0 || capacity <= 0 || uint64_t(preferred) > remaining ||
        remaining > MAX - position || bisects(spans, position)) return 0;
    int below = 0;
    int above = 0;
    for (int length = 1; length <= capacity && uint64_t(length) <= remaining; ++length) {
        if (bisects(spans, position + uint64_t(length))) continue;
        if (length <= preferred) below = length;
        else if (!above) above = length;
    }
    return below ? below : above;
}

void check_oracle(const std::vector<TokenSpan> & spans, uint64_t position,
                  int preferred, uint64_t remaining, int capacity) {
    const int expected = oracle(spans, position, preferred, remaining, capacity);
    const int actual = atomic_image_chunk(view(spans), position, preferred, remaining, capacity);
    require(actual == expected, "endpoint oracle mismatch at position=" + std::to_string(position) +
            " preferred=" + std::to_string(preferred) + " capacity=" + std::to_string(capacity));
    if (actual) {
        require(actual > 0 && actual <= capacity && uint64_t(actual) <= remaining,
                "chunk must make bounded positive progress");
        require(!bisects(spans, position + uint64_t(actual)), "chunk endpoint splits an image block");
    }
}

void validation_and_lookup() {
    require(valid_image_spans({}, 0), "empty prompt/view is valid");
    require(!image_block_at({}, 0), "empty lookup is null");
    require(!valid_image_spans({nullptr, 1}, 100), "nonempty null view rejected");
    std::array<TokenSpan, 5> too_many{};
    require(!valid_image_spans({too_many.data(), too_many.size()}, 100), "more than four images rejected");
    const std::vector<TokenSpan> spans{{10, 13, 18, 20}, {20, 20, 25, 25}, {30, 31, 33, 35}};
    require(valid_image_spans(view(spans), 35), "adjacent and separated blocks valid");
    require(!image_block_at(view(spans), 9), "text before block excluded");
    require(image_block_at(view(spans), 10) == &spans[0], "leading padding belongs to image block");
    require(image_block_at(view(spans), 12) == &spans[0], "leading padding is not ordinary text");
    require(image_block_at(view(spans), 13) == &spans[0], "visible begin included");
    require(image_block_at(view(spans), 18) == &spans[0], "trailing padding belongs to block");
    require(image_block_at(view(spans), 20) == &spans[1], "adjacent boundary belongs to next block");
    require(!image_block_at(view(spans), 25), "half-open end excluded");
    require(!image_block_at(view(spans), 29), "gap excluded");
    require(!image_block_at(view(spans), 35), "final end excluded");
    for (const auto & invalid : std::vector<std::vector<TokenSpan>>{
             {{9, 8, 10, 11}}, {{9, 9, 9, 11}}, {{9, 9, 12, 11}},
             {{9, 9, 10, 36}}, {{0, 0, 1, 385}}, {{5, 5, 7, 9}, {8, 8, 10, 12}},
             {{10, 10, 11, 12}, {1, 1, 2, 3}}, {{MAX, 0, 1, 2}}}) {
        require(!valid_image_spans(view(invalid), 35), "malformed/overlapping/out-of-range span rejected");
    }
    const std::vector<TokenSpan> largest{{0, 3, 380, 384}};
    require(valid_image_spans(view(largest), 384), "384-token block admitted");
    const std::vector<TokenSpan> overlong{{0, 3, 380, 385}};
    require(!valid_image_spans(view(overlong), 385), "385-token block rejected even inside prompt");
    const std::vector<TokenSpan> upper{{MAX - 384, MAX - 381, MAX - 1, MAX}};
    require(valid_image_spans(view(upper), MAX), "valid upper-limit span does not overflow");
    require(atomic_image_chunk(view(upper), MAX - 384, 1, 384, 1024) == 384,
            "upper-limit atomic extension reaches exact end");
    require(atomic_image_chunk({}, MAX - 3, 1, 4, 1024) == 0, "position plus remaining overflow rejected");
    require(atomic_image_chunk({}, 0, 0, 10, 1024) == 0, "zero proposed chunk rejected");
    require(atomic_image_chunk({}, 0, -1, 10, 1024) == 0, "negative proposed chunk rejected");
    require(atomic_image_chunk({}, 0, 1, 10, 0) == 0, "zero capacity rejected");
    require(atomic_image_chunk({}, 0, 1, 10, -1) == 0, "negative capacity rejected");
    require(atomic_image_chunk({}, 0, 11, 10, 1024) == 0, "caller must clamp preferred chunk to remaining");
}

void production_boundary_matrix() {
    const std::vector<TokenSpan> spans{
        {100, 103, 479, 484}, {484, 484, 500, 500},
        {32640, 32644, 33020, 33024}, {33024, 33024, 33031, 33032}};
    constexpr uint64_t prompt = 35000;
    require(valid_image_spans(view(spans), prompt), "four-image fixture valid");
    for (int preferred : {1, 128, 512, 1024, 4096}) {
        for (uint64_t position : {uint64_t(0), uint64_t(99), uint64_t(100), uint64_t(101),
                uint64_t(483), uint64_t(484), uint64_t(499), uint64_t(500), uint64_t(32639),
                uint64_t(32640), uint64_t(32768), uint64_t(33023), uint64_t(33024), uint64_t(33032)}) {
            check_oracle(spans, position, preferred, prompt - position, 1024);
        }
        uint64_t position = 0;
        size_t steps = 0;
        while (position < prompt) {
            const int proposed = int(std::min<uint64_t>(uint64_t(preferred), prompt - position));
            const int count = atomic_image_chunk(view(spans), position, proposed, prompt - position, 1024);
            require(count > 0, "complete traversal must not stall when every block fits capacity");
            require(!bisects(spans, position + uint64_t(count)), "complete traversal preserves atomic blocks");
            require(count <= 1024, "preferred 4096 cannot exceed independent hard capacity 1024");
            position += uint64_t(count);
            require(++steps <= prompt, "traversal terminates");
        }
        require(position == prompt, "complete traversal consumes exact prompt");
    }
    require(atomic_image_chunk(view(spans), 32640, 128, prompt - 32640, 1024) == 384,
            "image crossing 32768 extends past context boundary atomically");
    require(atomic_image_chunk(view(spans), 100, 1, prompt - 100, 383) == 0,
            "capacity smaller than complete block rejected");
    require(atomic_image_chunk(view(spans), 100, 1, 383, 1024) == 0,
            "remaining prompt smaller than complete block rejected");
    require(atomic_image_chunk(view(spans), 99, 128, prompt - 99, 128) == 1,
            "ordinary prefix remains consumable before oversized image");
    for (int proposed : {100, 484, 500, 1024}) check_oracle(spans, 0, proposed, prompt, 1024);
}

void exhaustive_endpoints() {
    constexpr uint64_t prompt = 14;
    // All ordered pairs of blocks, including adjacency, varying image lengths,
    // all positions, and capacities that can both fit and reject a whole image.
    for (uint64_t a = 0; a < prompt; ++a) {
        for (uint64_t b = a + 1; b <= std::min(prompt, a + 4); ++b) {
            for (uint64_t c = b; c < prompt; ++c) {
                for (uint64_t d = c + 1; d <= std::min(prompt, c + 4); ++d) {
                    const std::vector<TokenSpan> spans{{a, a, b, b}, {c, c, d, d}};
                    require(valid_image_spans(view(spans), prompt), "generated spans valid");
                    for (uint64_t position = 0; position <= prompt; ++position) {
                        for (int capacity : {1, 2, 3, 4, 7, 16}) {
                            for (int preferred : {1, 2, 4, 8, 16}) {
                                const uint64_t remaining = prompt - position;
                                check_oracle(spans, position, preferred, remaining, capacity);
                            }
                        }
                    }
                }
            }
        }
    }
}

void memory_budget() {
    constexpr uint64_t MiB = 1024 * 1024;
    constexpr uint64_t GiB = 1024 * MiB;
    constexpr uint64_t vision = dflash::vision::SCRATCH_RESERVATION;
    constexpr uint64_t workspace = 76 * MiB;
    require(vision == 2 * GiB, "retain qualified two-GiB vision reservation");
    const uint64_t initial = remaining_expert_budget(24 * GiB, 4 * GiB, GiB, 256 * MiB, 512 * MiB, vision);
    require(initial == 16640 * MiB, "combined fixed charges leave expected expert budget");
    // A free-memory snapshot turns every completed allocation into core usage.
    // Preloading the projector must be charged exactly once in either order.
    constexpr uint64_t projector = 700 * MiB;
    const uint64_t before_projector = remaining_expert_budget(
        24 * GiB, 4 * GiB, GiB, 256 * MiB, 512 * MiB, vision + projector);
    const uint64_t after_projector = remaining_expert_budget(
        24 * GiB, 4 * GiB + projector, GiB, 256 * MiB, 512 * MiB, vision);
    require(before_projector == after_projector, "preloaded projector charged once regardless of load order");
    const uint64_t after_workspace = remaining_expert_budget(
        24 * GiB, 4 * GiB + workspace, GiB, 256 * MiB, 512 * MiB, vision, workspace);
    require(initial == after_workspace, "resident workspace offsets only its included reservation");
    require(remaining_expert_budget(24 * GiB, 4 * GiB + workspace, GiB, 256 * MiB,
                512 * MiB, vision) == initial - workspace,
            "unproven workspace residency cannot receive a credit");
    require(remaining_expert_budget(MAX, 0, 0, 0, 0, vision, vision + 1) == 0,
            "workspace credit cannot exceed reservation");
    require(remaining_expert_budget(100, 20, 20, 20, 20, 20) == 0, "exact exhaustion returns zero");
    require(remaining_expert_budget(99, 20, 20, 20, 20, 20) == 0, "one-byte shortage fails closed");
    require(remaining_expert_budget(MAX, MAX, 1, 0, 0, 0) == 0, "overflowing sum cannot wrap into budget");
    require(remaining_expert_budget(MAX, 1, MAX, 0, 0, 0) == 0, "late exhaustion cannot wrap");
    require(remaining_expert_budget(MAX, 0, 0, 0, 0, 0) == MAX, "uncharged maximum preserved");
    require(remaining_expert_budget(MAX, MAX - 7, 2, 1, 1, 2) == 1, "near-maximum arithmetic remains exact");
    // Independent conservation law for uniform expert rounds: tighter primary
    // headroom must transfer the same quantized storage amount to the cold owner.
    constexpr uint64_t round = 64 * MiB;
    constexpr uint64_t total_experts = 32 * GiB;
    uint64_t last_hot = initial / round * round;
    uint64_t last_cold = total_experts - last_hot;
    for (uint64_t extra = 0; extra <= 4 * GiB; extra += 17 * MiB) {
        const uint64_t budget = remaining_expert_budget(24 * GiB, 4 * GiB, GiB,
            256 * MiB, 512 * MiB, vision + extra);
        const uint64_t hot = budget / round * round;
        const uint64_t cold = total_experts - hot;
        require(hot <= last_hot && cold >= last_cold, "reduced hot budget cannot reduce cold demand");
        require(hot + cold == total_experts && budget - hot < round, "placement conserves expert bytes");
        require(last_hot - hot == cold - last_cold, "lost hot bytes equal additional cold bytes");
        last_hot = hot;
        last_cold = cold;
    }
}
} // namespace

int main() {
    try {
        validation_and_lookup();
        production_boundary_matrix();
        exhaustive_endpoints();
        memory_budget();
        std::cout << "PASS: DS4 image integration invariants checks=" << checks << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
