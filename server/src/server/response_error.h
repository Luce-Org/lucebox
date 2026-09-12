// Server-side generation failure classification and API response encoding.
#pragma once

#include "api_types.h"

#include <nlohmann/json.hpp>

#include <string>

namespace dflash::common {

struct GenerateError;

enum class ResponseErrorKind {
    InvalidRequest,
    Unavailable,
    Internal,
};

// API-neutral description of a failed serving request. Backends remain the
// source of truth for generation failures through GenerateError; this value
// records only the consequence at the server boundary.
struct ResponseError {
    ResponseErrorKind kind = ResponseErrorKind::Internal;
    std::string code;
    std::string message;

    static ResponseError invalid_request(
        std::string code, std::string message);
    static ResponseError unavailable(
        std::string code, std::string message);
    static ResponseError internal(
        std::string code, std::string message);
};

// Total backend-to-server mapping. Every backend error receives a stable code,
// a non-empty client message, and a server-owned availability classification.
ResponseError to_response_error(const GenerateError & error);

int response_error_http_status(const ResponseError & error);

// Build the protocol body for a non-streaming request. HTTP framing and socket
// ownership stay with HttpServer.
nlohmann::json build_error_response(
    ApiFormat format, const ResponseError & error,
    const std::string & request_id = {});

}  // namespace dflash::common
