// Lucepilot switches (common/pilot_switch.h and its ggml-cuda counterpart)
// and the [pilot] boot line (common/pilot_boot_line.h). GPU-free.

#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"

#include "common/pilot_boot_line.h"
#include "common/pilot_switch.h"
#include "../../deps/llama.cpp/ggml/src/ggml-cuda/luce-pilot.cuh"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace {

struct PilotSwitch {};
struct PilotBootLine {};

std::vector<std::string> & captured() {
    static std::vector<std::string> lines;
    return lines;
}

void capture(const char * line) {
    captured().emplace_back(line);
}

// Captures the pilot lines of one test with the logging gate forced; restores
// the environment-decided stderr default afterwards.
struct CapturedLines {
    explicit CapturedLines(int active) {
        captured().clear();
        luce::pilot::detail::reset_for_test(&capture, active);
    }
    ~CapturedLines() {
        luce::pilot::detail::reset_for_test(nullptr, -1);
    }
    const std::vector<std::string> & lines() const { return captured(); }
};

nlohmann::json parse_boot_json(const std::string & line) {
    return nlohmann::json::parse(line);
}

}  // namespace

// ── switch semantics ────────────────────────────────────────────────────

TEST_CASE(PilotSwitch, parse_unset_off_on) {
    using luce::pilot::SwitchState;
    CHECK_TRUE(luce::pilot::parse_switch(nullptr) == SwitchState::Unset);
    CHECK_TRUE(luce::pilot::parse_switch("0") == SwitchState::Off);
    CHECK_TRUE(luce::pilot::parse_switch("1") == SwitchState::On);
    CHECK_TRUE(luce::pilot::parse_switch("64") == SwitchState::On);
    CHECK_TRUE(luce::pilot::parse_switch("off") == SwitchState::On);
    CHECK_TRUE(luce::pilot::parse_switch("00") == SwitchState::On);
    CHECK_TRUE(luce::pilot::parse_switch("") == SwitchState::On);
}

TEST_CASE(PilotSwitch, switch_reads_environment) {
    CapturedLines lines(0);
    {
        luce_test::ScopedEnvVar env("LUCE_PILOT_NOOP_TEST_A", nullptr);
        const luce::pilot::Switch s("LUCE_PILOT_NOOP_TEST_A");
        CHECK_FALSE(s.is_set());
        CHECK_FALSE(s.on());
        CHECK_FALSE(s.off());
        CHECK_TRUE(s.value().empty());
    }
    {
        luce_test::ScopedEnvVar env("LUCE_PILOT_NOOP_TEST_A", "0");
        const luce::pilot::Switch s("LUCE_PILOT_NOOP_TEST_A");
        CHECK_TRUE(s.is_set());
        CHECK_TRUE(s.off());
        CHECK_FALSE(s.on());
    }
    {
        luce_test::ScopedEnvVar env("LUCE_PILOT_NOOP_TEST_A", "1");
        const luce::pilot::Switch s("LUCE_PILOT_NOOP_TEST_A");
        CHECK_TRUE(s.on());
        CHECK_EQUAL(s.value(), std::string("1"));
    }
    {
        luce_test::ScopedEnvVar env("LUCE_PILOT_NOOP_TEST_A", "rows4");
        const luce::pilot::Switch s("LUCE_PILOT_NOOP_TEST_A");
        CHECK_TRUE(s.on());
        CHECK_EQUAL(s.value(), std::string("rows4"));
    }
}

TEST_CASE(PilotSwitch, tuning_knobs_default_to_parent) {
    CapturedLines lines(0);
    {
        luce_test::ScopedEnvVar env("LUCE_PILOT_MMQ_TEST_X", nullptr);
        const luce::pilot::Switch s("LUCE_PILOT_MMQ_TEST_X");
        CHECK_EQUAL(s.int_or(128), 128LL);
        CHECK_EQUAL(s.string_or("auto"), std::string("auto"));
    }
    {
        luce_test::ScopedEnvVar env("LUCE_PILOT_MMQ_TEST_X", "64");
        const luce::pilot::Switch s("LUCE_PILOT_MMQ_TEST_X");
        CHECK_EQUAL(s.int_or(128), 64LL);
        CHECK_EQUAL(s.string_or("auto"), std::string("64"));
    }
    {
        luce_test::ScopedEnvVar env("LUCE_PILOT_MMQ_TEST_X", "0");
        const luce::pilot::Switch s("LUCE_PILOT_MMQ_TEST_X");
        CHECK_EQUAL(s.int_or(128), 0LL);
    }
    {
        luce_test::ScopedEnvVar env("LUCE_PILOT_MMQ_TEST_X", "-3");
        const luce::pilot::Switch s("LUCE_PILOT_MMQ_TEST_X");
        CHECK_EQUAL(s.int_or(128), -3LL);
    }
    for (const char * bad : {"", "64k", "x", "1.5", " 7"}) {
        luce_test::ScopedEnvVar env("LUCE_PILOT_MMQ_TEST_X", bad);
        const luce::pilot::Switch s("LUCE_PILOT_MMQ_TEST_X");
        CHECK_EQUAL(s.int_or(128), 128LL);
    }
}

// ── one-shot read / engaged lines ───────────────────────────────────────

TEST_CASE(PilotSwitch, read_line_once_per_name) {
    CapturedLines lines(1);
    luce_test::ScopedEnvVar set("LUCE_PILOT_GDN_TEST_ON", "1");
    luce_test::ScopedEnvVar unset("LUCE_PILOT_GDN_TEST_UNSET", nullptr);
    {
        const luce::pilot::Switch a("LUCE_PILOT_GDN_TEST_ON");
        const luce::pilot::Switch b("LUCE_PILOT_GDN_TEST_ON");  // same name: no second line
        const luce::pilot::Switch c("LUCE_PILOT_GDN_TEST_UNSET");
        (void) a; (void) b; (void) c;
    }
    REQUIRE_EQUAL(lines.lines().size(), size_t(2));
    CHECK_EQUAL(lines.lines()[0], std::string("[pilot] read LUCE_PILOT_GDN_TEST_ON=1"));
    CHECK_EQUAL(lines.lines()[1], std::string("[pilot] read LUCE_PILOT_GDN_TEST_UNSET=unset"));
}

TEST_CASE(PilotSwitch, read_line_keeps_one_line) {
    CapturedLines lines(1);
    luce_test::ScopedEnvVar env("LUCE_PILOT_GLUE_TEST_NL", "a\nb");
    const luce::pilot::Switch s("LUCE_PILOT_GLUE_TEST_NL");
    CHECK_EQUAL(s.value(), std::string("a\nb"));
    REQUIRE_EQUAL(lines.lines().size(), size_t(1));
    CHECK_EQUAL(lines.lines()[0], std::string("[pilot] read LUCE_PILOT_GLUE_TEST_NL=a?b"));
}

TEST_CASE(PilotSwitch, engaged_line_once_per_name) {
    CapturedLines lines(1);
    luce_test::ScopedEnvVar env("LUCE_PILOT_TREE_TEST_E", "1");
    const luce::pilot::Switch s("LUCE_PILOT_TREE_TEST_E");
    for (int i = 0; i < 3; ++i) s.engaged("depth=6");
    luce::pilot::engaged("LUCE_PILOT_TREE_TEST_E", "other");  // already printed
    luce::pilot::engaged("LUCE_PILOT_TREE_TEST_F");
    REQUIRE_EQUAL(lines.lines().size(), size_t(3));
    CHECK_EQUAL(lines.lines()[0], std::string("[pilot] read LUCE_PILOT_TREE_TEST_E=1"));
    CHECK_EQUAL(lines.lines()[1], std::string("[pilot] engaged LUCE_PILOT_TREE_TEST_E depth=6"));
    CHECK_EQUAL(lines.lines()[2], std::string("[pilot] engaged LUCE_PILOT_TREE_TEST_F"));
}

TEST_CASE(PilotSwitch, no_lines_without_pilot_environment) {
    CapturedLines lines(0);
    luce_test::ScopedEnvVar env("LUCE_PILOT_ATTN_TEST_Q", "1");
    const luce::pilot::Switch s("LUCE_PILOT_ATTN_TEST_Q");
    s.engaged("x");
    CHECK_TRUE(s.on());
    CHECK_TRUE(lines.lines().empty());
}

TEST_CASE(PilotSwitch, gate_follows_environment_block) {
    char a[] = "PATH=/bin";
    char b[] = "LUCE_X=1";
    char c[] = "LUCE_PILOT_CANDIDATE=c7";
    char * without[] = {a, b, nullptr};
    char * with[] = {a, c, nullptr};
    CHECK_FALSE(luce::pilot::has_pilot_variable(without));
    CHECK_TRUE(luce::pilot::has_pilot_variable(with));
    CHECK_FALSE(luce::pilot::has_pilot_variable(nullptr));
}

TEST_CASE(PilotSwitch, macro_reads_once_per_call_site) {
    CapturedLines lines(1);
    luce_test::ScopedEnvVar env("LUCE_PILOT_FAULT_TEST_M", "1");
    bool seen = false;
    for (int i = 0; i < 3; ++i) {
        seen = LUCE_PILOT_SWITCH("LUCE_PILOT_FAULT_TEST_M").on();
    }
    CHECK_TRUE(seen);
    REQUIRE_EQUAL(lines.lines().size(), size_t(1));
    CHECK_EQUAL(lines.lines()[0], std::string("[pilot] read LUCE_PILOT_FAULT_TEST_M=1"));
}

TEST_CASE(PilotSwitch, noop_probe_is_read_once) {
    CapturedLines lines(1);
    luce_test::ScopedEnvVar env(luce::pilot::kNoopProbe, "1");
    luce::pilot::read_noop_probe();
    luce::pilot::read_noop_probe();
    REQUIRE_EQUAL(lines.lines().size(), size_t(1));
    CHECK_EQUAL(lines.lines()[0], std::string("[pilot] read LUCE_PILOT_NOOP_PROBE=1"));
}

TEST_CASE(PilotSwitch, line_formats) {
    CHECK_EQUAL(luce::pilot::read_line("LUCE_PILOT_X", nullptr),
                std::string("[pilot] read LUCE_PILOT_X=unset"));
    CHECK_EQUAL(luce::pilot::read_line("LUCE_PILOT_X", ""),
                std::string("[pilot] read LUCE_PILOT_X="));
    CHECK_EQUAL(luce::pilot::engaged_line("LUCE_PILOT_X", nullptr),
                std::string("[pilot] engaged LUCE_PILOT_X"));
    CHECK_EQUAL(luce::pilot::engaged_line("LUCE_PILOT_X", "a b"),
                std::string("[pilot] engaged LUCE_PILOT_X a b"));
}

// The ggml-cuda counterpart compiles as host code and parses alike. Its
// lines go to stderr only when the process holds a LUCE_PILOT_* variable.
TEST_CASE(PilotSwitch, ggml_counterpart_semantics) {
    CHECK_TRUE(ggml_luce_pilot::parse(nullptr) == ggml_luce_pilot::state::unset);
    CHECK_TRUE(ggml_luce_pilot::parse("0") == ggml_luce_pilot::state::off);
    CHECK_TRUE(ggml_luce_pilot::parse("1") == ggml_luce_pilot::state::on);
    CHECK_TRUE(ggml_luce_pilot::parse("x") == ggml_luce_pilot::state::on);
    {
        luce_test::ScopedEnvVar env("LUCE_PILOT_MMQ_TEST_G", "32");
        CHECK_TRUE(ggml_luce_pilot::read("LUCE_PILOT_MMQ_TEST_G") == ggml_luce_pilot::state::on);
        CHECK_EQUAL(ggml_luce_pilot::read_int("LUCE_PILOT_MMQ_TEST_G", 8), 32LL);
        CHECK_TRUE(GGML_LUCE_PILOT_ON("LUCE_PILOT_MMQ_TEST_G"));
    }
    {
        luce_test::ScopedEnvVar env("LUCE_PILOT_MMQ_TEST_H", nullptr);
        CHECK_TRUE(ggml_luce_pilot::read("LUCE_PILOT_MMQ_TEST_H") == ggml_luce_pilot::state::unset);
        CHECK_EQUAL(ggml_luce_pilot::read_int("LUCE_PILOT_MMQ_TEST_H", 8), 8LL);
    }
    for (const char * bad : {"", "x", " 7", "7x", "-"}) {
        luce_test::ScopedEnvVar env("LUCE_PILOT_MMQ_TEST_I", bad);
        CHECK_EQUAL(ggml_luce_pilot::read_int("LUCE_PILOT_MMQ_TEST_I", 8), 8LL);
    }
    {
        luce_test::ScopedEnvVar env("LUCE_PILOT_MMQ_TEST_I", "-4");
        CHECK_EQUAL(ggml_luce_pilot::read_int("LUCE_PILOT_MMQ_TEST_I", 8), -4LL);
    }
    ggml_luce_pilot::engaged("LUCE_PILOT_MMQ_TEST_G", "x=32");
}

// ── boot line ───────────────────────────────────────────────────────────

TEST_CASE(PilotBootLine, empty_environment) {
    const std::string line = luce::pilot::boot_line(nullptr, nullptr);
    CHECK_EQUAL(line.rfind("[pilot] {", 0), size_t(0));
    CHECK_EQUAL(line.find('\n'), std::string::npos);
    const nlohmann::json j = parse_boot_json(luce::pilot::boot_line_json(nullptr, nullptr));
    CHECK_EQUAL(j["schema"].get<int>(), 1);
    CHECK_TRUE(j["candidate"].is_null());
    CHECK_EQUAL(j["build_sha"].get<std::string>(), std::string("unknown"));
    CHECK_TRUE(j["pilot"].is_object() && j["pilot"].empty());
    CHECK_TRUE(j["env"].is_object() && j["env"].empty());
}

TEST_CASE(PilotBootLine, filters_prefixes_and_splits_pilot) {
    char e0[] = "PATH=/usr/bin";
    char e1[] = "HOME=/home/x";
    char e2[] = "LUCE_TP_PROFILE=1";
    char e3[] = "GGML_CUDA_DISABLE_GRAPHS=1";
    char e4[] = "HIP_VISIBLE_DEVICES=0";
    char e5[] = "ROCR_VISIBLE_DEVICES=0";
    char e6[] = "DFLASH_DEBUG=2";
    char e7[] = "LUCE_PILOT_CANDIDATE=gdn-rowsplit-3";
    char e8[] = "LUCE_PILOT_GDN_ROWSPLIT=1";
    char e9[] = "LUCEX_NOT=1";
    char e10[] = "XLUCE_NOT=1";
    char e11[] = "LUCE_TP_PROFILE=second";  // first definition wins, as getenv
    char e12[] = "NOEQUALS";
    char e13[] = "HIP=no-underscore";
    char * envp[] = {e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, nullptr};
    const nlohmann::json j = parse_boot_json(luce::pilot::boot_line_json(envp, "abc123"));
    CHECK_EQUAL(j["schema"].get<int>(), 1);
    CHECK_EQUAL(j["candidate"].get<std::string>(), std::string("gdn-rowsplit-3"));
    CHECK_EQUAL(j["build_sha"].get<std::string>(), std::string("abc123"));
    const nlohmann::json expected_pilot = {
        {"LUCE_PILOT_CANDIDATE", "gdn-rowsplit-3"},
        {"LUCE_PILOT_GDN_ROWSPLIT", "1"},
    };
    CHECK_TRUE(j["pilot"] == expected_pilot);
    const nlohmann::json expected_env = {
        {"LUCE_TP_PROFILE", "1"},
        {"GGML_CUDA_DISABLE_GRAPHS", "1"},
        {"HIP_VISIBLE_DEVICES", "0"},
        {"ROCR_VISIBLE_DEVICES", "0"},
        {"DFLASH_DEBUG", "2"},
    };
    CHECK_TRUE(j["env"] == expected_env);
}

TEST_CASE(PilotBootLine, top_level_key_order) {
    const std::string json = luce::pilot::boot_line_json(nullptr, "s");
    CHECK_EQUAL(json, std::string(
        "{\"schema\":1,\"candidate\":null,\"build_sha\":\"s\",\"pilot\":{},\"env\":{}}"));
}

TEST_CASE(PilotBootLine, escapes_values) {
    char e0[] = "LUCE_ARGS=--x \"quoted\" back\\slash\ttab\nnewline";
    char e1[] = "LUCE_PILOT_CANDIDATE=c\"1";
    char e2[] = "LUCE_UTF8=caf\xc3\xa9";
    char e3[] = "LUCE_BAD_UTF8=a\xff" "b";
    char e4[] = "GGML_CTRL=\x01";
    char * envp[] = {e0, e1, e2, e3, e4, nullptr};
    const std::string line = luce::pilot::boot_line(envp, "sha\"x");
    CHECK_EQUAL(line.find('\n'), std::string::npos);
    CHECK_EQUAL(line.find('\t'), std::string::npos);
    const nlohmann::json j = parse_boot_json(line.substr(std::string("[pilot] ").size()));
    CHECK_EQUAL(j["env"]["LUCE_ARGS"].get<std::string>(),
                std::string("--x \"quoted\" back\\slash\ttab\nnewline"));
    CHECK_EQUAL(j["candidate"].get<std::string>(), std::string("c\"1"));
    CHECK_EQUAL(j["build_sha"].get<std::string>(), std::string("sha\"x"));
    CHECK_EQUAL(j["env"]["LUCE_UTF8"].get<std::string>(), std::string("caf\xc3\xa9"));
    CHECK_EQUAL(j["env"]["LUCE_BAD_UTF8"].get<std::string>(), std::string("a\xef\xbf\xbd" "b"));
    CHECK_EQUAL(j["env"]["GGML_CTRL"].get<std::string>(), std::string("\x01"));
}

TEST_CASE(PilotBootLine, process_environment_round_trip) {
    luce_test::ScopedEnvVar candidate("LUCE_PILOT_CANDIDATE", "unit-test");
    luce_test::ScopedEnvVar knob("LUCE_PILOT_NOOP_BOOT_TEST", "v");
    luce_test::ScopedEnvVar env("GGML_BOOT_LINE_TEST", "1");
    const nlohmann::json j = parse_boot_json(
        luce::pilot::boot_line_json(luce::pilot::process_environment(), "deadbeef"));
    CHECK_EQUAL(j["candidate"].get<std::string>(), std::string("unit-test"));
    CHECK_EQUAL(j["pilot"]["LUCE_PILOT_NOOP_BOOT_TEST"].get<std::string>(), std::string("v"));
    CHECK_EQUAL(j["env"]["GGML_BOOT_LINE_TEST"].get<std::string>(), std::string("1"));
    CHECK_FALSE(j["env"].contains("LUCE_PILOT_CANDIDATE"));
}
