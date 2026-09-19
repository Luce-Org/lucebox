#include "CppUnitTestFramework.hpp"

#include "common/pflash_drafter_ipc.h"
#include "common/model_backend.h"
#include "qwen3/pflash_selection.h"

#include <cmath>
#include <string>

using namespace dflash::common;

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

TEST_CASE(PFlashDrafterIpcFixture, compress3_matches_the_local_selector_contract) {
    ModelBackend::CompressRequest local;
    local.input_ids.resize(32);
    local.keep_ratio = 0.5f;
    local.score_query_end = 32;
    local.score_query_tokens = 4;
    local.required_instruction_spans = {{0, 4}, {12, 14}};

    std::string line;
    std::string error;
    REQUIRE(format_pflash_drafter_ipc_compress_command(
        local.keep_ratio, local.score_query_end, local.score_query_tokens,
        local.required_instruction_spans, "/tmp/ids.bin", line, error));
    PFlashDrafterIpcCompressCommand remote;
    REQUIRE(parse_pflash_drafter_ipc_compress_command(line, remote, error));
    REQUIRE(remote.keep_ratio == local.keep_ratio);
    REQUIRE(remote.score_query_end == local.score_query_end);
    REQUIRE(remote.score_query_tokens == local.score_query_tokens);
    REQUIRE(remote.required_instruction_spans == local.required_instruction_spans);

    const auto select = [&] (const std::vector<PFlashTokenSpan> & spans) {
        std::vector<dflash::qwen3::PFlashSelectionCandidate> candidates;
        constexpr double scores[]{0.0, 9.0, 1.0, 0.0, 2.0, 3.0, 4.0, 0.0};
        for (int chunk = 0; chunk < 8; ++chunk) {
            const int begin = chunk * 4;
            const int end = begin + 4;
            candidates.push_back({
                (size_t) chunk, begin, end, scores[chunk],
                dflash::qwen3::pflash_chunk_is_structurally_required(
                    begin, end, 28, 32, 32, spans),
            });
        }
        return dflash::qwen3::select_pflash_candidates(
            candidates, {16, 0.95},
            dflash::qwen3::PFlashSelectionMode::BudgetOnly);
    };
    const auto local_result = select(local.required_instruction_spans);
    const auto remote_result = select(remote.required_instruction_spans);
    REQUIRE(local_result.ok);
    REQUIRE(remote_result.ok);
    REQUIRE(local_result.ordinals == remote_result.ordinals);
    REQUIRE(local_result.retained_tokens == remote_result.retained_tokens);
    REQUIRE(local_result.stop == remote_result.stop);

    const std::vector<PFlashTokenSpan> out_of_range{{0, 33}};
    std::string local_error;
    std::string remote_error;
    REQUIRE(!dflash::qwen3::validate_pflash_instruction_spans(
        out_of_range, (int) local.input_ids.size(), local_error));
    REQUIRE(format_pflash_drafter_ipc_compress_command(
        local.keep_ratio, local.score_query_end, local.score_query_tokens,
        out_of_range, "/tmp/ids.bin", line, error));
    REQUIRE(parse_pflash_drafter_ipc_compress_command(line, remote, error));
    REQUIRE(!dflash::qwen3::validate_pflash_instruction_spans(
        remote.required_instruction_spans, (int) local.input_ids.size(),
        remote_error));
    REQUIRE(local_error == remote_error);
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
