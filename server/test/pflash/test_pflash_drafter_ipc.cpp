#include "CppUnitTestFramework.hpp"

#include "common/pflash_drafter_ipc.h"

#include <cmath>
#include <string>

using namespace luce::common;

namespace {

struct PFlashDrafterIpcFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;
};

} // namespace

TEST_CASE(PFlashDrafterIpcFixture, compress2_round_trips_paper_ratio_budget) {
    const float keep = 16384.0f / 120000.0f;
    std::string line;
    std::string error;

    REQUIRE(format_pflash_drafter_ipc_compress_command(
        keep, 119900, 128, "/tmp/pflash ids.bin", line, error));
    REQUIRE(error.empty());

    PFlashDrafterIpcCompressCommand parsed;
    REQUIRE(parse_pflash_drafter_ipc_compress_command(line, parsed, error));
    REQUIRE(error.empty());
    REQUIRE(!parsed.legacy_quantized_ratio);
    REQUIRE(parsed.keep_ratio == keep);
    REQUIRE((int) std::floor(120000.0 * (double) parsed.keep_ratio) == 16384);
    REQUIRE(parsed.score_query_end == 119900);
    REQUIRE(parsed.score_query_tokens == 128);
    REQUIRE(parsed.required_instruction_spans.empty());
    REQUIRE(parsed.path == "/tmp/pflash ids.bin");

    // The narrowest scorer query window round-trips too.
    REQUIRE(format_pflash_drafter_ipc_compress_command(
        keep, 119900, 1, "/tmp/pflash ids.bin", line, error));
    REQUIRE(parse_pflash_drafter_ipc_compress_command(line, parsed, error));
    REQUIRE(parsed.score_query_tokens == 1);
}

TEST_CASE(PFlashDrafterIpcFixture, compress3_round_trips_ordered_instruction_spans) {
    const float keep = 16384.0f / 120000.0f;
    const std::vector<PFlashTokenSpan> instructions{{0, 384}, {4096, 4352}};
    std::string line;
    std::string error;

    REQUIRE(format_pflash_drafter_ipc_compress_command(
        keep, 119900, 128, instructions,
        "/tmp/pflash ids.bin", line, error));
    REQUIRE(error.empty());
    REQUIRE(line.rfind("compress3 ", 0) == 0);

    PFlashDrafterIpcCompressCommand parsed;
    REQUIRE(parse_pflash_drafter_ipc_compress_command(line, parsed, error));
    REQUIRE(error.empty());
    REQUIRE(parsed.keep_ratio == keep);
    REQUIRE(parsed.score_query_end == 119900);
    REQUIRE(parsed.score_query_tokens == 128);
    REQUIRE(parsed.required_instruction_spans == instructions);
    REQUIRE(parsed.path == "/tmp/pflash ids.bin");
}

TEST_CASE(PFlashDrafterIpcFixture, legacy_x1000_parser_is_supported_but_quantized) {
    PFlashDrafterIpcCompressCommand parsed;
    std::string error;

    REQUIRE(parse_pflash_drafter_ipc_compress_command(
        "compress 137 119900 128 /tmp/pflash_ids.bin", parsed, error));
    REQUIRE(error.empty());
    REQUIRE(parsed.legacy_quantized_ratio);
    REQUIRE(parsed.keep_ratio == 0.137f);
    REQUIRE((int) std::floor(120000.0 * (double) parsed.keep_ratio) > 16384);
    REQUIRE(parsed.score_query_end == 119900);
    REQUIRE(parsed.score_query_tokens == 128);
    REQUIRE(parsed.path == "/tmp/pflash_ids.bin");
}

TEST_CASE(PFlashDrafterIpcFixture, parser_rejects_malformed_values) {
    const char * bad_lines[] = {
        "compress2 nan 10 8 /tmp/ids.bin",
        "compress2 inf 10 8 /tmp/ids.bin",
        "compress2 -0.1 10 8 /tmp/ids.bin",
        "compress2 1.1 10 8 /tmp/ids.bin",
        "compress2 0.5 10 0 /tmp/ids.bin",
        "compress2 0.5 10 8",
        "compress 1001 10 8 /tmp/ids.bin",
        "compress -1 10 8 /tmp/ids.bin",
        "compress 500 10 0 /tmp/ids.bin",
        "compress3 0.5 10 8 -1 /tmp/ids.bin",
        "compress3 0.5 10 8 1 0 /tmp/ids.bin",
        "compress3 0.5 10 8 1 4 4 /tmp/ids.bin",
        "compress3 0.5 10 8 2 0 4 3 6 /tmp/ids.bin",
        "compress3 0.5 10 8 65 /tmp/ids.bin",
        "unknown 0.5 10 8 /tmp/ids.bin",
    };

    for (const char * line : bad_lines) {
        PFlashDrafterIpcCompressCommand parsed;
        std::string error;
        REQUIRE(!parse_pflash_drafter_ipc_compress_command(
            line, parsed, error));
        REQUIRE(!error.empty());
    }
}

TEST_CASE(PFlashDrafterIpcFixture, formatter_fails_closed_on_invalid_values) {
    struct BadFormatInput {
        float keep_ratio;
        int score_query_tokens;
        const char * path;
    };
    const BadFormatInput bad_inputs[] = {
        {-0.1f, 8, "/tmp/ids.bin"},
        {1.1f, 8, "/tmp/ids.bin"},
        {0.5f, 0, "/tmp/ids.bin"},
        {0.5f, -1, "/tmp/ids.bin"},
        {0.5f, 8, ""},
    };

    for (const auto & input : bad_inputs) {
        std::string line;
        std::string error;
        REQUIRE(!format_pflash_drafter_ipc_compress_command(
            input.keep_ratio, 10, input.score_query_tokens,
            input.path, line, error));
        REQUIRE(line.empty());
        REQUIRE(!error.empty());
    }
}
