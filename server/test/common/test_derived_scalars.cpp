// Unit tests for dflash::common::verify_derived_scalars — no GPU, no model files.

#include "CppUnitTestFramework.hpp"
#include "common/derived_scalars.h"

#include <string>

using namespace dflash::common;

namespace {
struct DerivedScalarsFixture {};
}

TEST_CASE(DerivedScalarsFixture, match_returns_true) {
    std::string err;
    bool ok = verify_derived_scalars(
        /*wq_ne1*/ 4096, /*wk_ne1*/ 512, /*wq_ne0*/ 5120,
        /*exp_q_dim*/ 4096, /*exp_kv_dim*/ 512, /*exp_n_embd*/ 5120,
        "blk.0", err);
    CHECK(ok);
    CHECK(err.empty());
}

TEST_CASE(DerivedScalarsFixture, mismatch_q_dim_returns_false) {
    std::string err;
    bool ok = verify_derived_scalars(
        /*wq_ne1*/ 4097, /*wk_ne1*/ 512, /*wq_ne0*/ 5120,
        /*exp_q_dim*/ 4096, /*exp_kv_dim*/ 512, /*exp_n_embd*/ 5120,
        "blk.0", err);
    CHECK(!ok);
    CHECK(!err.empty());
}

TEST_CASE(DerivedScalarsFixture, mismatch_kv_dim_returns_false) {
    std::string err;
    bool ok = verify_derived_scalars(
        /*wq_ne1*/ 4096, /*wk_ne1*/ 513, /*wq_ne0*/ 5120,
        /*exp_q_dim*/ 4096, /*exp_kv_dim*/ 512, /*exp_n_embd*/ 5120,
        "blk.0", err);
    CHECK(!ok);
    CHECK(!err.empty());
}

TEST_CASE(DerivedScalarsFixture, mismatch_n_embd_returns_false) {
    std::string err;
    bool ok = verify_derived_scalars(
        /*wq_ne1*/ 4096, /*wk_ne1*/ 512, /*wq_ne0*/ 5184,
        /*exp_q_dim*/ 4096, /*exp_kv_dim*/ 512, /*exp_n_embd*/ 5120,
        "blk.0", err);
    CHECK(!ok);
    CHECK(!err.empty());
}

TEST_CASE(DerivedScalarsFixture, draft_dims_match) {
    std::string err;
    bool ok = verify_derived_scalars(
        4096, 1024, 5120,
        (int64_t)32 * 128, (int64_t)8 * 128, 5120,
        "blk.0", err);
    CHECK(ok);
    CHECK(err.empty());
}

TEST_CASE(DerivedScalarsFixture, qwen35_target_dims_match) {
    std::string err;
    bool ok = verify_derived_scalars(
        /*wq_ne1*/ 12288, /*wk_ne1*/ 1024, /*wq_ne0*/ 5120,
        /*exp_q_dim*/ (int64_t)24 * 256 * 2,
        /*exp_kv_dim*/ (int64_t)4 * 256,
        /*exp_n_embd*/ 5120,
        "blk.3", err);
    CHECK(ok);
    CHECK(err.empty());
}

TEST_CASE(DerivedScalarsFixture, err_contains_layer_tag) {
    std::string err;
    verify_derived_scalars(4097, 1024, 5120, 4096, 1024, 5120, "blk.15", err);
    CHECK(err.find("blk.15") != std::string::npos);
}
