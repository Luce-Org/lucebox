// Helpers shared by the Qwen3-0.6B and Qwen3.5-0.8B drafter paths.
//
// Moved verbatim out of qwen3_drafter.cpp so qwen35_drafter.cpp can use the
// same selector, trace writer and environment readers without a second copy.

#pragma once

#include "pflash_selection.h"
#include "common/pflash_types.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dflash::common {

int env_int(const char * name, int fallback);
float env_float(const char * name, float def);
void force_chunk_neighborhood(std::vector<uint8_t> & forced, int n_chunks,
                              int chunk, int radius);

struct PFlashTraceFields {
    const std::vector<int32_t> * input_ids = nullptr;
    int query_begin = -1;
    int query_end = -1;
    dflash::qwen3::PFlashSelectionMode selector_mode =
        dflash::qwen3::PFlashSelectionMode::Legacy;
    dflash::qwen3::PFlashQueryParser query_parser =
        dflash::qwen3::PFlashQueryParser::SemanticUser;
    int token_budget = 0;
    dflash::qwen3::PFlashSelectionStop stop =
        dflash::qwen3::PFlashSelectionStop::InvalidInput;
    int retained_tokens = 0;
    double retained_mass = 0.0;
    const std::vector<double> * exact_chunk_scores = nullptr;
    const std::vector<PFlashTokenSpan> * required_instruction_spans = nullptr;
    // Variable-length candidates (segment probe): spans in candidate order.
    const std::vector<PFlashTokenSpan> * segments = nullptr;
    const char * segmentation = "fixed";
    const char * candidate_score = "sum";
    // Two-scorer selection: the other scorer's candidate scores, same order.
    const char * scorer = "head";
    double split_fraction = 0.0;
    const std::vector<double> * other_chunk_scores = nullptr;
};

void write_compression_trace(
        int input_tokens,
        float keep_ratio,
        int chunk_size,
        int n_lookahead,
        int pool_kernel,
        int n_keep,
        const std::vector<std::pair<float, int>> & chunk_means,
        const std::vector<uint8_t> & selected,
        const std::vector<uint8_t> & forced,
        const std::vector<int32_t> & compressed_ids,
        const PFlashTraceFields * trace_fields = nullptr);

std::vector<int32_t> select_pflash_chunks(
        const std::vector<int32_t> & ids,
        const std::vector<float> & token_scores,
        float keep_ratio,
        int n_lookahead,
        int score_query_end,
        int pool_kernel,
        const dflash::qwen3::PFlashSelectionConfig & config,
        const std::vector<PFlashTokenSpan> & required_instruction_spans,
        bool direct_mass,
        bool write_trace,
        const std::vector<PFlashTokenSpan> * segments = nullptr,
        bool density = false,
        const std::vector<float> * other_token_scores = nullptr,
        double split_fraction = 0.0);

} // namespace dflash::common
