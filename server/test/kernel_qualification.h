#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace lucebox::kernel_qualification {

enum class Status {
    pass,
    fail,
    unsupported,
};

struct Metric {
    std::string name;
    std::string error_measure = "max_abs";
    double tolerance = 0.0;
    double observed_error = 0.0;
    size_t worst_index = 0;
    bool finite = true;
    bool passed = false;
    std::string reason;
};

struct Route {
    std::string expected;
    std::string observed;
    bool matched = false;
};

struct CaseResult {
    std::string name;
    Status status = Status::fail;
    std::string reason;
    Route route;
    std::vector<Metric> metrics;
};

inline const char * status_name(Status status) {
    switch (status) {
        case Status::pass: return "pass";
        case Status::fail: return "fail";
        case Status::unsupported: return "unsupported";
    }
    return "fail";
}

inline Metric compare_f32_impl(
        std::string name,
        const std::vector<float> & candidate,
        const std::vector<float> & reference,
        double tolerance,
        bool normalized_mse) {
    Metric result;
    result.name = std::move(name);
    result.error_measure = normalized_mse ? "normalized_mse" : "max_abs";
    result.tolerance = tolerance;
    if (!std::isfinite(tolerance)) {
        result.reason = "non-finite tolerance";
        return result;
    }
    if (tolerance < 0.0) {
        result.reason = "negative tolerance";
        return result;
    }
    if (candidate.size() != reference.size()) {
        result.observed_error = std::numeric_limits<double>::infinity();
        result.finite = false;
        result.reason = "size mismatch";
        return result;
    }
    if (candidate.empty()) {
        result.reason = "no samples";
        return result;
    }

    double squared_error = 0.0;
    double reference_power = 0.0;
    double worst_error = 0.0;
    for (size_t i = 0; i < candidate.size(); ++i) {
        if (!std::isfinite(candidate[i]) || !std::isfinite(reference[i])) {
            result.observed_error = std::numeric_limits<double>::infinity();
            result.worst_index = i;
            result.finite = false;
            result.reason = "non-finite value";
            return result;
        }
        const double error = static_cast<double>(candidate[i]) - reference[i];
        const double absolute_error = std::fabs(error);
        if (absolute_error > worst_error) {
            worst_error = absolute_error;
            result.worst_index = i;
        }
        if (normalized_mse) {
            squared_error += error*error;
            reference_power +=
                static_cast<double>(reference[i])*reference[i];
        } else {
            result.observed_error = worst_error;
        }
    }
    if (normalized_mse) {
        result.observed_error =
            squared_error/std::max(reference_power, 1.0e-30);
    }
    result.passed = result.observed_error <= tolerance;
    if (!result.passed) result.reason = "tolerance exceeded";
    return result;
}

inline Metric compare_f32(
        std::string name,
        const std::vector<float> & candidate,
        const std::vector<float> & reference,
        double tolerance) {
    return compare_f32_impl(
        std::move(name), candidate, reference, tolerance, false);
}

inline Metric compare_f32_nmse(
        std::string name,
        const std::vector<float> & candidate,
        const std::vector<float> & reference,
        double tolerance) {
    return compare_f32_impl(
        std::move(name), candidate, reference, tolerance, true);
}

inline CaseResult evaluate(
        std::string name,
        std::vector<Metric> metrics,
        Route route) {
    CaseResult result;
    result.name = std::move(name);
    result.metrics = std::move(metrics);
    result.route = std::move(route);
    if (result.metrics.empty()) {
        result.reason = "no metrics";
        return result;
    }
    result.status = result.route.matched &&
            std::all_of(result.metrics.begin(), result.metrics.end(),
                        [](const Metric & metric) {
                            return metric.finite && metric.passed;
                        })
        ? Status::pass
        : Status::fail;
    if (!result.route.matched) {
        result.reason = "candidate route was not observed";
    } else {
        const auto failed = std::find_if(
            result.metrics.begin(), result.metrics.end(),
            [](const Metric & metric) {
                return !metric.finite || !metric.passed;
            });
        if (failed != result.metrics.end()) {
            result.reason = failed->reason.empty() && !failed->finite
                ? "non-finite metric"
                : failed->reason;
        }
    }
    return result;
}

inline CaseResult unsupported(std::string name, std::string reason) {
    CaseResult result;
    result.name = std::move(name);
    result.status = Status::unsupported;
    result.reason = std::move(reason);
    return result;
}

inline Status aggregate_status(const std::vector<CaseResult> & cases) {
    if (cases.empty()) return Status::fail;
    if (std::any_of(cases.begin(), cases.end(), [](const CaseResult & result) {
            return result.status == Status::fail;
        })) {
        return Status::fail;
    }
    if (std::any_of(cases.begin(), cases.end(), [](const CaseResult & result) {
            return result.status == Status::pass;
        })) {
        return Status::pass;
    }
    return Status::unsupported;
}

inline int exit_code(const std::vector<CaseResult> & cases) {
    switch (aggregate_status(cases)) {
        case Status::pass: return 0;
        case Status::fail: return 1;
        case Status::unsupported: return 77;
    }
    return 1;
}

inline std::string json_string(const std::string & value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << static_cast<int>(ch)
                        << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    out << '"';
    return out.str();
}

inline void write_json(
        std::ostream & out,
        const std::string & backend,
        const std::vector<CaseResult> & cases) {
    out << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "{\"schema\":\"lucebox.kernel_qualification.v1\""
        << ",\"backend\":" << json_string(backend)
        << ",\"status\":" << json_string(status_name(aggregate_status(cases)))
        << ",\"cases\":[";
    for (size_t case_index = 0; case_index < cases.size(); ++case_index) {
        const CaseResult & result = cases[case_index];
        if (case_index != 0) out << ',';
        out << "{\"name\":" << json_string(result.name)
            << ",\"status\":" << json_string(status_name(result.status))
            << ",\"reason\":" << json_string(result.reason)
            << ",\"route\":{\"expected\":"
            << json_string(result.route.expected)
            << ",\"observed\":" << json_string(result.route.observed)
            << ",\"matched\":" << (result.route.matched ? "true" : "false")
            << "},\"metrics\":[";
        for (size_t metric_index = 0;
             metric_index < result.metrics.size(); ++metric_index) {
            const Metric & metric = result.metrics[metric_index];
            if (metric_index != 0) out << ',';
            out << "{\"name\":" << json_string(metric.name)
                << ",\"error_measure\":"
                << json_string(metric.error_measure)
                << ",\"tolerance\":";
            if (std::isfinite(metric.tolerance)) {
                out << metric.tolerance;
            } else {
                out << "null";
            }
            out << ",\"observed_error\":";
            if (std::isfinite(metric.observed_error)) {
                out << metric.observed_error;
            } else {
                out << "null";
            }
            out << ",\"worst_index\":" << metric.worst_index
                << ",\"finite\":" << (metric.finite ? "true" : "false")
                << ",\"passed\":" << (metric.passed ? "true" : "false")
                << ",\"reason\":" << json_string(metric.reason) << '}';
        }
        out << "]}";
    }
    out << "]}\n";
}

}  // namespace lucebox::kernel_qualification
