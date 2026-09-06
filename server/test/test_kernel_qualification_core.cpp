#include "kernel_qualification.h"
#include "CppUnitTestFramework.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace kq = lucebox::kernel_qualification;
using CppUnitTestFramework::CommonFixture;

namespace {

struct KernelQualificationFixture : CommonFixture {
    using CommonFixture::CommonFixture;
};

}  // namespace

TEST_CASE(KernelQualificationFixture, comparisons) {
    const kq::Metric within = kq::compare_f32(
        "output", {1.0f, 2.0f}, {1.0f, 2.00001f}, 2.0e-5);
    REQUIRE_TRUE(within.passed && within.finite);

    const kq::Metric outside = kq::compare_f32(
        "state", {1.0f, 2.1f}, {1.0f, 2.0f}, 1.0e-3);
    REQUIRE_TRUE(!outside.passed && outside.reason == "tolerance exceeded");

    const kq::Metric nonfinite = kq::compare_f32(
        "nonfinite", {std::numeric_limits<float>::quiet_NaN()}, {0.0f}, 1.0);
    REQUIRE_TRUE(!nonfinite.passed && !nonfinite.finite);

    const kq::Metric infinite_tolerance = kq::compare_f32(
        "infinite_tolerance", {1.0f}, {1.0f},
        std::numeric_limits<double>::infinity());
    const kq::Metric nan_tolerance = kq::compare_f32_nmse(
        "nan_tolerance", {1.0f}, {1.0f},
        std::numeric_limits<double>::quiet_NaN());
    REQUIRE_TRUE(
        !infinite_tolerance.passed && infinite_tolerance.finite &&
        infinite_tolerance.reason == "non-finite tolerance");
    REQUIRE_TRUE(
        !nan_tolerance.passed && nan_tolerance.finite &&
        nan_tolerance.reason == "non-finite tolerance");

    const kq::Metric negative_tolerance = kq::compare_f32(
        "negative_tolerance", {1.0f}, {1.0f}, -1.0);
    const kq::Metric negative_nmse_tolerance = kq::compare_f32_nmse(
        "negative_nmse_tolerance", {1.0f}, {1.0f}, -1.0);
    REQUIRE_TRUE(
        !negative_tolerance.passed && negative_tolerance.finite &&
        negative_tolerance.reason == "negative tolerance");
    REQUIRE_TRUE(
        !negative_nmse_tolerance.passed &&
        negative_nmse_tolerance.finite &&
        negative_nmse_tolerance.reason == "negative tolerance");

    const kq::Metric empty = kq::compare_f32("empty", {}, {}, 0.0);
    const kq::Metric empty_nmse =
        kq::compare_f32_nmse("empty_nmse", {}, {}, 0.0);
    REQUIRE_TRUE(!empty.passed && empty.reason == "no samples");
    REQUIRE_TRUE(
        !empty_nmse.passed && empty_nmse.reason == "no samples");

    const kq::Metric nmse = kq::compare_f32_nmse(
        "output", {1.0f, 2.001f}, {1.0f, 2.0f}, 1.0e-6);
    REQUIRE_TRUE(nmse.passed && nmse.error_measure == "normalized_mse");
}

TEST_CASE(KernelQualificationFixture, statuses) {
    const kq::Metric within = kq::compare_f32(
        "output", {1.0f, 2.0f}, {1.0f, 2.00001f}, 2.0e-5);
    const kq::CaseResult passed = kq::evaluate(
        "gdn", {within}, {"gdn.grouped_cols", "gdn.grouped_cols", true});
    const kq::CaseResult wrong_route = kq::evaluate(
        "gdn", {within}, {"gdn.grouped_cols", "gdn.scalar", false});
    const kq::CaseResult empty = kq::evaluate(
        "gdn", {}, {"gdn.grouped_cols", "gdn.grouped_cols", true});
    const kq::CaseResult skipped = kq::unsupported(
        "gdn", "GPU backend unavailable");
    kq::Metric contradictory = within;
    contradictory.finite = false;
    contradictory.passed = true;
    const kq::CaseResult nonfinite = kq::evaluate(
        "gdn", {contradictory},
        {"gdn.grouped_cols", "gdn.grouped_cols", true});
    REQUIRE_TRUE(passed.status == kq::Status::pass);
    REQUIRE_TRUE(wrong_route.status == kq::Status::fail);
    REQUIRE_TRUE(skipped.status == kq::Status::unsupported);
    REQUIRE_TRUE(empty.status == kq::Status::fail && empty.reason == "no metrics");
    REQUIRE_TRUE(
        nonfinite.status == kq::Status::fail &&
        nonfinite.reason == "non-finite metric");
    REQUIRE_TRUE(kq::exit_code({skipped}) == 77);
    REQUIRE_TRUE(kq::aggregate_status({}) == kq::Status::fail);
    REQUIRE_TRUE(kq::exit_code({}) == 1);
}

TEST_CASE(KernelQualificationFixture, report_preserves_double_precision) {
    kq::Metric precise;
    precise.name = "precise";
    precise.tolerance = 1.2345678901234567e-12;
    precise.observed_error = 9.8765432109876543e-13;
    precise.finite = true;
    precise.passed = true;
    const kq::CaseResult passed = kq::evaluate(
        "report", {precise}, {"route", "route", true});
    std::ostringstream report;
    kq::write_json(report, "self-test", {passed});
    const std::string json = report.str();
    REQUIRE_TRUE(
        json.find("\"schema\":\"lucebox.kernel_qualification.v1\"") !=
            std::string::npos &&
        json.find("\"status\":\"pass\"") != std::string::npos);
    REQUIRE_TRUE(
        json.find("1.2345678901234567e-12") != std::string::npos);
    REQUIRE_TRUE(
        json.find("9.8765432109876542e-13") != std::string::npos);
}

TEST_CASE(KernelQualificationFixture, report_rejects_nonfinite_numbers) {
    kq::Metric invalid;
    invalid.name = "invalid";
    invalid.tolerance = std::numeric_limits<double>::infinity();
    invalid.observed_error = std::numeric_limits<double>::quiet_NaN();
    invalid.finite = false;
    const kq::CaseResult failed = kq::evaluate(
        "report", {invalid}, {"route", "route", true});
    std::ostringstream report;
    kq::write_json(report, "self-test", {failed});
    const std::string json = report.str();
    REQUIRE_TRUE(json.find("\"tolerance\":null") != std::string::npos);
    REQUIRE_TRUE(
        json.find("\"observed_error\":null") != std::string::npos);
}
