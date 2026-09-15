#include "response_error.h"

#include "common/model_backend.h"

#include <utility>

namespace dflash::common {

namespace {

std::string message_or(std::string message, const char * fallback) {
    return message.empty() ? fallback : std::move(message);
}

const char * fallback_message(GenerateErrorCode code) {
    switch (code) {
    case GenerateErrorCode::Incomplete:
        return "generation did not complete";
    case GenerateErrorCode::AdapterUnavailable:
        return "requested adapter is unavailable";
    case GenerateErrorCode::ResourceExhausted:
        return "request exceeds available generation capacity";
    case GenerateErrorCode::ContextOverflow:
        return "request exceeds the model context";
    case GenerateErrorCode::SamplingUnsupported:
        return "requested sampling settings are unsupported";
    case GenerateErrorCode::PrefillFailed:
        return "generation prefill failed";
    case GenerateErrorCode::DecodeSeedMissing:
        return "generation decode seed is missing";
    case GenerateErrorCode::DecodeFailed:
        return "generation decode failed";
    case GenerateErrorCode::InvalidSnapshotSlot:
        return "generation snapshot is unavailable";
    case GenerateErrorCode::ModelParked:
        return "model is unavailable";
    case GenerateErrorCode::BackendSpecific:
        return "generation failed";
    }
    return "generation failed";
}

const char * openai_error_type(ResponseErrorKind kind) {
    return kind == ResponseErrorKind::InvalidRequest
        ? "invalid_request_error" : "server_error";
}

const char * anthropic_error_type(ResponseErrorKind kind) {
    switch (kind) {
    case ResponseErrorKind::InvalidRequest: return "invalid_request_error";
    case ResponseErrorKind::Unavailable:    return "overloaded_error";
    case ResponseErrorKind::Internal:       return "api_error";
    }
    return "api_error";
}

}  // namespace

ResponseError ResponseError::invalid_request(
        std::string code, std::string message) {
    return {ResponseErrorKind::InvalidRequest,
            std::move(code),
            message_or(std::move(message), "invalid request")};
}

ResponseError ResponseError::unavailable(
        std::string code, std::string message) {
    return {ResponseErrorKind::Unavailable,
            std::move(code),
            message_or(std::move(message), "service unavailable")};
}

ResponseError ResponseError::internal(
        std::string code, std::string message) {
    return {ResponseErrorKind::Internal,
            std::move(code),
            message_or(std::move(message), "generation failed")};
}

ResponseError to_response_error(const GenerateError & error) {
    const std::string code(generate_error_code(error.code));
    const std::string message = error.detail.empty()
        ? fallback_message(error.code) : error.detail;

    switch (error.code) {
    case GenerateErrorCode::ContextOverflow:
    case GenerateErrorCode::SamplingUnsupported:
        return ResponseError::invalid_request(code, message);
    case GenerateErrorCode::AdapterUnavailable:
    case GenerateErrorCode::ResourceExhausted:
    case GenerateErrorCode::ModelParked:
        return ResponseError::unavailable(code, message);
    case GenerateErrorCode::Incomplete:
    case GenerateErrorCode::PrefillFailed:
    case GenerateErrorCode::DecodeSeedMissing:
    case GenerateErrorCode::DecodeFailed:
    case GenerateErrorCode::InvalidSnapshotSlot:
    case GenerateErrorCode::BackendSpecific:
        return ResponseError::internal(code, message);
    }
    return ResponseError::internal("unknown_error", "generation failed");
}

int response_error_http_status(const ResponseError & error) {
    switch (error.kind) {
    case ResponseErrorKind::InvalidRequest: return 400;
    case ResponseErrorKind::Unavailable:    return 503;
    case ResponseErrorKind::Internal:       return 500;
    }
    return 500;
}

nlohmann::json build_error_response(
        ApiFormat format, const ResponseError & error,
        const std::string & request_id) {
    if (format == ApiFormat::ANTHROPIC) {
        nlohmann::json body = {
            {"type", "error"},
            {"error", {
                {"type", anthropic_error_type(error.kind)},
                {"message", error.message},
            }},
        };
        if (!request_id.empty()) body["request_id"] = request_id;
        return body;
    }

    return {{"error", {
        {"message", error.message},
        {"type", openai_error_type(error.kind)},
        {"code", error.code},
    }}};
}

}  // namespace dflash::common
