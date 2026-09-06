#include "common/model_backend.h"
#include "support/mock_backend.h"
#include <nlohmann/json.hpp>
#include "support/test_assert.h"

using json = nlohmann::json;
using namespace dflash::common;

// ModelBackend common empty-spec retry tests
// ═══════════════════════════════════════════════════════════════════════

struct EmptySpecRetryBackend : MockBackend {
    int generate_calls = 0;
    int restore_calls = 0;
    bool generate_saw_force_ar = false;
    bool restore_saw_force_ar = false;
    bool generate_first_empty_visible = false;
    bool restore_first_empty_visible = false;

    GenerateResult generate_impl(const GenerateRequest & req,
                            const DaemonIO &) override {
        generate_calls++;
        GenerateResult result;
        result.succeed();
        if (req.force_ar_decode) {
            generate_saw_force_ar = true;
            result.tokens = {42};
        } else {
            result.spec_decode_ran = true;
            if (generate_first_empty_visible) {
                result.tokens = {2};
                result.empty_visible_output = true;
            }
        }
        return result;
    }

    GenerateResult restore_and_generate_impl(int, const GenerateRequest & req,
                                        const DaemonIO &) override {
        restore_calls++;
        GenerateResult result;
        result.succeed();
        result.restored_prefix_tokens = req.force_ar_decode ? 2 : 3;
        if (req.force_ar_decode) {
            restore_saw_force_ar = true;
            result.tokens = {84};
        } else {
            result.spec_decode_ran = true;
            if (restore_first_empty_visible) {
                result.tokens = {2};
                result.empty_visible_output = true;
            }
        }
        return result;
    }
};

TEST_CASE(ServerUnitFixture, test_model_backend_retries_empty_spec_generate_once_with_ar) {
    EmptySpecRetryBackend backend;
    GenerateRequest req;
    req.prompt = {1, 2, 3};
    req.n_gen = 4;
    DaemonIO io;

    GenerateResult result = backend.generate(req, io);

    TEST_ASSERT(result.ok());
    TEST_ASSERT(result.tokens.size() == 1);
    TEST_ASSERT(result.tokens[0] == 42);
    TEST_ASSERT(result.spec_decode_ran);
    TEST_ASSERT(backend.generate_calls == 2);
    TEST_ASSERT(backend.generate_saw_force_ar);
}

TEST_CASE(ServerUnitFixture, test_model_backend_retries_empty_spec_restore_once_with_ar) {
    EmptySpecRetryBackend backend;
    GenerateRequest req;
    req.prompt = {1, 2, 3};
    req.n_gen = 4;
    DaemonIO io;

    GenerateResult result =
        backend.restore_and_generate(7, req, io);

    TEST_ASSERT(result.ok());
    TEST_ASSERT(result.tokens.size() == 1);
    TEST_ASSERT(result.tokens[0] == 84);
    TEST_ASSERT(result.spec_decode_ran);
    TEST_ASSERT(result.restored_prefix_tokens == 3);
    TEST_ASSERT(backend.restore_calls == 2);
    TEST_ASSERT(backend.restore_saw_force_ar);
}

TEST_CASE(ServerUnitFixture, test_model_backend_retries_empty_visible_spec_generate_once_with_ar) {
    EmptySpecRetryBackend backend;
    backend.generate_first_empty_visible = true;
    GenerateRequest req;
    req.prompt = {1, 2, 3};
    req.n_gen = 4;
    DaemonIO io;

    GenerateResult result = backend.generate(req, io);

    TEST_ASSERT(result.ok());
    TEST_ASSERT(result.tokens.size() == 1);
    TEST_ASSERT(result.tokens[0] == 42);
    TEST_ASSERT(!result.empty_visible_output);
    TEST_ASSERT(result.spec_decode_ran);
    TEST_ASSERT(backend.generate_calls == 2);
    TEST_ASSERT(backend.generate_saw_force_ar);
}

TEST_CASE(ServerUnitFixture, test_model_backend_retries_empty_visible_spec_restore_once_with_ar) {
    EmptySpecRetryBackend backend;
    backend.restore_first_empty_visible = true;
    GenerateRequest req;
    req.prompt = {1, 2, 3};
    req.n_gen = 4;
    DaemonIO io;

    GenerateResult result = backend.restore_and_generate(7, req, io);

    TEST_ASSERT(result.ok());
    TEST_ASSERT(result.tokens.size() == 1);
    TEST_ASSERT(result.tokens[0] == 84);
    TEST_ASSERT(!result.empty_visible_output);
    TEST_ASSERT(result.spec_decode_ran);
    TEST_ASSERT(backend.restore_calls == 2);
    TEST_ASSERT(backend.restore_saw_force_ar);
}

// GenerateResult speculative telemetry plumbing tests (Day 1 of bandit MVP)
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE(ServerUnitFixture, test_generate_result_accept_rate_defaults_to_zero) {
    GenerateResult r;
    TEST_ASSERT(r.accept_rate == 0.0f);
}



TEST_CASE(ServerUnitFixture, test_generate_result_accept_rate_zero_when_no_spec_decode) {
    // When spec decode doesn't run (no draft model), accept_rate stays 0.
    GenerateResult r;
    r.succeed();
    // Telemetry not set → accept_rate is zero and speculative decode is false.
    TEST_ASSERT(r.accept_rate == 0.0f);
    TEST_ASSERT(!r.spec_decode_ran);
}

TEST_CASE(ServerUnitFixture, test_generate_result_error_state_is_consistent) {
    GenerateResult result;
    TEST_ASSERT(!result.ok());
    TEST_ASSERT(result.error->code == GenerateErrorCode::Incomplete);
    TEST_ASSERT(result.error_code() == "incomplete");

    result.fail(GenerateErrorCode::BackendSpecific, "prefill graph allocation failed");
    TEST_ASSERT(!result.ok());
    TEST_ASSERT(result.error->code == GenerateErrorCode::BackendSpecific);
    TEST_ASSERT(result.error_code() == "backend_specific");
    TEST_ASSERT(result.error_detail() == "prefill graph allocation failed");

    result.succeed();
    TEST_ASSERT(result.ok());
    TEST_ASSERT(!result.error.has_value());
    TEST_ASSERT(result.error_code().empty());
    TEST_ASSERT(result.error_detail().empty());
}
