// pflash_drafter_ipc.cpp - PFlash drafter IPC client + daemon body.

#include "pflash_drafter_ipc.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace dflash::common {

namespace {

bool parse_int_token(const std::string & raw, int & out) {
    if (raw.empty()) return false;
    errno = 0;
    char * end = nullptr;
    const long value = std::strtol(raw.c_str(), &end, 10);
    if (errno == ERANGE || end == raw.c_str() || *end != '\0' ||
        value < INT_MIN || value > INT_MAX) {
        return false;
    }
    out = (int) value;
    return true;
}

bool parse_float_token(const std::string & raw, float & out) {
    if (raw.empty()) return false;
    errno = 0;
    char * end = nullptr;
    const float value = std::strtof(raw.c_str(), &end);
    if (errno == ERANGE || end == raw.c_str() || *end != '\0' ||
        !std::isfinite(value)) {
        return false;
    }
    out = value;
    return true;
}

bool validate_request_fields(
        float keep_ratio,
        int score_query_tokens,
        const std::vector<PFlashTokenSpan> & instruction_spans,
        const std::string & path,
        std::string & error) {
    if (!std::isfinite(keep_ratio) || keep_ratio < 0.0f || keep_ratio > 1.0f) {
        error = "PFlash IPC keep_ratio must be finite and in [0, 1]";
        return false;
    }
    if (score_query_tokens < 1) {
        error = "PFlash IPC score_query_tokens must be positive";
        return false;
    }
    if (instruction_spans.size() > kPFlashMaxInstructionSpans) {
        error = "PFlash IPC has too many instruction spans";
        return false;
    }
    int previous_end = 0;
    for (const auto & span : instruction_spans) {
        if (span.begin < 0 || span.end <= span.begin) {
            error = "PFlash IPC instruction span is invalid";
            return false;
        }
        if (span.begin < previous_end) {
            error = "PFlash IPC instruction spans must be ordered and non-overlapping";
            return false;
        }
        previous_end = span.end;
    }
    if (path.empty()) {
        error = "PFlash IPC token path must not be empty";
        return false;
    }
    return true;
}

} // namespace

bool format_pflash_drafter_ipc_compress_command(
        float keep_ratio,
        int score_query_end,
        int score_query_tokens,
        const std::string & path,
        std::string & out,
        std::string & error) {
    out.clear();
    error.clear();
    if (!validate_request_fields(
            keep_ratio, score_query_tokens, {}, path, error)) {
        return false;
    }

    char keep_text[64];
    std::snprintf(keep_text, sizeof(keep_text), "%.9g", keep_ratio);

    std::ostringstream line;
    line << "compress2 " << keep_text << ' ' << score_query_end << ' '
         << score_query_tokens << ' ' << path;
    out = line.str();
    return true;
}

bool format_pflash_drafter_ipc_compress_command(
        float keep_ratio,
        int score_query_end,
        int score_query_tokens,
        const std::vector<PFlashTokenSpan> & required_instruction_spans,
        const std::string & path,
        std::string & out,
        std::string & error) {
    out.clear();
    error.clear();
    if (required_instruction_spans.empty()) {
        return format_pflash_drafter_ipc_compress_command(
            keep_ratio, score_query_end, score_query_tokens,
            path, out, error);
    }
    if (!validate_request_fields(
            keep_ratio, score_query_tokens,
            required_instruction_spans, path, error)) {
        return false;
    }

    char keep_text[64];
    std::snprintf(keep_text, sizeof(keep_text), "%.9g", keep_ratio);
    std::ostringstream line;
    line << "compress3 " << keep_text << ' ' << score_query_end << ' '
         << score_query_tokens << ' ' << required_instruction_spans.size();
    for (const auto & span : required_instruction_spans) {
        line << ' ' << span.begin << ' ' << span.end;
    }
    line << ' ' << path;
    out = line.str();
    return true;
}

bool parse_pflash_drafter_ipc_compress_command(
        const std::string & line,
        PFlashDrafterIpcCompressCommand & out,
        std::string & error) {
    out = {};
    error.clear();

    std::istringstream iss(line);
    std::string command;
    if (!(iss >> command)) {
        error = "PFlash IPC command is empty";
        return false;
    }

    std::string keep_raw;
    std::string query_end_raw;
    std::string query_tokens_raw;
    if (!(iss >> keep_raw >> query_end_raw >> query_tokens_raw)) {
        error = "PFlash IPC compress command is missing fields";
        return false;
    }
    if (!parse_int_token(query_end_raw, out.score_query_end) ||
        !parse_int_token(query_tokens_raw, out.score_query_tokens)) {
        error = "PFlash IPC query fields must be integers";
        return false;
    }
    if (command == "compress3") {
        if (!parse_float_token(keep_raw, out.keep_ratio)) {
            error = "PFlash IPC keep_ratio must be a float";
            return false;
        }
        std::string count_raw;
        int span_count = -1;
        if (!(iss >> count_raw) || !parse_int_token(count_raw, span_count) ||
            span_count < 0 ||
            (size_t) span_count > kPFlashMaxInstructionSpans) {
            error = "PFlash IPC instruction span count is invalid";
            return false;
        }
        out.required_instruction_spans.reserve((size_t) span_count);
        for (int index = 0; index < span_count; ++index) {
            std::string begin_raw;
            std::string end_raw;
            PFlashTokenSpan span;
            if (!(iss >> begin_raw >> end_raw) ||
                !parse_int_token(begin_raw, span.begin) ||
                !parse_int_token(end_raw, span.end)) {
                error = "PFlash IPC instruction span fields must be integers";
                return false;
            }
            out.required_instruction_spans.push_back(span);
        }
        out.path = read_line_tail(iss);
    } else if (command == "compress2") {
        if (!parse_float_token(keep_raw, out.keep_ratio)) {
            error = "PFlash IPC keep_ratio must be a float";
            return false;
        }
        out.path = read_line_tail(iss);
    } else if (command == "compress") {
        int keep_x1000 = 0;
        if (!parse_int_token(keep_raw, keep_x1000) ||
            keep_x1000 < 0 || keep_x1000 > 1000) {
            error = "PFlash IPC legacy keep_x1000 must be in [0, 1000]";
            return false;
        }
        out.legacy_quantized_ratio = true;
        out.keep_ratio = (float) keep_x1000 / 1000.0f;
        out.path = read_line_tail(iss);
    } else {
        error = "unknown PFlash IPC command";
        return false;
    }

    return validate_request_fields(
        out.keep_ratio, out.score_query_tokens,
        out.required_instruction_spans, out.path, error);
}

bool PFlashDrafterIpcClient::start(
        const std::string & bin,
        const std::string & drafter_path,
        int drafter_gpu,
        const std::string & work_dir) {
#if defined(_WIN32)
    (void)bin; (void)drafter_path; (void)drafter_gpu; (void)work_dir;
    std::fprintf(stderr, "PFlash drafter IPC is only implemented on POSIX hosts\n");
    return false;
#else
    close();
    if (bin.empty() || drafter_path.empty()) return false;
    BackendIpcLaunchConfig launch;
    launch.bin = bin;
    launch.mode = BackendIpcMode::PFlashCompress;
    launch.payload_path = drafter_path;
    launch.work_dir = work_dir;
    launch.args.push_back("--draft-gpu=" + std::to_string(std::max(0, drafter_gpu)));
    if (!process_.start(launch)) {
        std::fprintf(stderr, "pflash-ipc backend process start failed\n");
        return false;
    }
    active_ = true;
    std::fprintf(stderr, "[pflash-ipc] ready bin=%s gpu=%d work_dir=%s\n",
                 bin.c_str(), drafter_gpu, process_.work_dir().c_str());
    return true;
#endif
}

bool PFlashDrafterIpcClient::compress(
        const std::vector<int32_t> & input_ids,
        float keep_ratio,
        std::vector<int32_t> & compressed_ids,
        int score_query_end,
        int score_query_tokens,
        const std::vector<PFlashTokenSpan> & required_instruction_spans) {
#if defined(_WIN32)
    (void)input_ids; (void)keep_ratio; (void)compressed_ids;
    (void)score_query_end; (void)score_query_tokens;
    (void)required_instruction_spans;
    return false;
#else
    compressed_ids.clear();
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    if (!active_ || !cmd || stream_fd < 0 || input_ids.empty()) return false;

    const std::string path = process_.next_path("pflash_tokens");
    if (!write_int32_file(path, input_ids)) {
        std::fprintf(stderr, "pflash-ipc write tokens failed: %s\n", path.c_str());
        return false;
    }
    std::string line;
    std::string error;
    if (!format_pflash_drafter_ipc_compress_command(
            keep_ratio, score_query_end, score_query_tokens,
            required_instruction_spans, path, line, error)) {
        std::fprintf(stderr, "pflash-ipc bad compress request: %s\n", error.c_str());
        std::remove(path.c_str());
        return false;
    }
    std::fprintf(cmd, "%s\n", line.c_str());
    std::fflush(cmd);

    int32_t status = -1;
    bool ok = read_exact_fd(stream_fd, &status, sizeof(status)) && status == 0;
    if (ok) {
        int32_t n_out = -1;
        ok = read_exact_fd(stream_fd, &n_out, sizeof(n_out)) && n_out > 0;
        if (ok) {
            compressed_ids.assign((size_t)n_out, 0);
            ok = read_exact_fd(stream_fd, compressed_ids.data(),
                               compressed_ids.size() * sizeof(int32_t));
        }
    }
    std::remove(path.c_str());
    if (!ok) {
        std::fprintf(stderr, "pflash-ipc compress failed status=%d\n", status);
        compressed_ids.clear();
        close();
    }
    return ok;
#endif
}

void PFlashDrafterIpcClient::close() {
    process_.close();
    active_ = false;
}

} // namespace dflash::common
