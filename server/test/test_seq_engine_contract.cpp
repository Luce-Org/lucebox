// Host-only mutation test for the model-neutral SeqEngine contract.

#include "seq_engine_contract.h"
#include "host_check.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace dflash::common;

static int g_checks = 0;

struct Faults {
    bool hard_error_when_full = false;
    bool reuse_live_slot = false;
    bool drop_decode_output = false;
    bool serialize_prefills = false;
    bool accept_omitted_decode = false;
    bool accept_invalid_prefill = false;
    bool lose_other_pending = false;
    bool overconsume_prefill = false;
    bool drop_second_completion = false;
    bool retire_leaks = false;
};

struct FakeCapabilities {
    StepPlanLimits idle{2, 2, 4};
    StepPlanLimits mixed{2, 1, 2};
};

class FakeSeqEngine final : public SeqEngine {
public:
    explicit FakeSeqEngine(int count, Faults faults = {},
                           FakeCapabilities capabilities = {})
        : slots_((size_t)count), faults_(faults),
          capabilities_(capabilities) {}

    int slot_count() const override { return (int)slots_.size(); }
    int max_context() const override { return 128; }
    StepPlanLimits step_plan_limits(int decode_rows) const override {
        return decode_rows > 0 ? capabilities_.mixed : capabilities_.idle;
    }
    bool token_is_eos(int32_t token) const override { return token == 2; }

    AdmitResult admit(uint64_t,
                      const std::vector<int32_t> & prompt,
                      const SamplerCfg &) override {
        AdmitResult result;
        if (prompt.empty() || prompt.size() > (size_t)max_context()) {
            result.error = "invalid prompt";
            return result;
        }

        int chosen = -1;
        if (faults_.reuse_live_slot) {
            chosen = 0;
        } else {
            for (size_t i = 0; i < slots_.size(); ++i) {
                if (!slots_[i].active) {
                    chosen = (int)i;
                    break;
                }
            }
        }
        if (chosen < 0) {
            result.status = faults_.hard_error_when_full
                ? AdmitResult::Status::failed
                : AdmitResult::Status::busy;
            result.error = "all slots are live";
            return result;
        }

        Slot & slot = slots_[(size_t)chosen];
        slot.active = true;
        slot.prefilling = true;
        slot.remaining = (int)prompt.size();
        slot.fed.clear();
        result.status = AdmitResult::Status::admitted;
        result.slot = chosen;
        return result;
    }

    StepResult step(const StepPlan & plan) override {
        StepResult result;
        std::string validation_error;
        if (!valid_decode(plan, validation_error) &&
            !faults_.accept_omitted_decode) {
            result.error = validation_error;
            return result;
        }
        if (!valid_prefills(plan, validation_error) &&
            !faults_.accept_invalid_prefill) {
            result.error = validation_error;
            return result;
        }

        size_t decode_count = plan.decode.size();
        if (faults_.drop_decode_output && plan.prefills.empty() &&
            decode_count == 2) {
            --decode_count;
        }
        for (size_t i = 0; i < decode_count; ++i) {
            const StepInput & input = plan.decode[i];
            if (input.slot < 0 || input.slot >= slot_count()) continue;
            Slot & slot = slots_[(size_t)input.slot];
            slot.fed.push_back(input.token);
            result.decode.push_back({
                input.slot,
                100 + input.slot + (int32_t)slot.fed.size(),
                false, {},
            });
        }

        std::vector<int> completed_this_step;
        for (size_t i = 0; i < plan.prefills.size(); ++i) {
            if (faults_.serialize_prefills && i > 0) continue;
            const PrefillSlice & slice = plan.prefills[i];
            if (slice.slot < 0 || slice.slot >= slot_count()) continue;
            Slot & slot = slots_[(size_t)slice.slot];
            if (!slot.active || !slot.prefilling || slot.remaining <= 0) {
                continue;
            }
            int consumed = std::min(slice.max_tokens, slot.remaining);
            if (faults_.overconsume_prefill) consumed = slice.max_tokens + 1;
            slot.remaining -= consumed;
            if (slot.remaining <= 0) {
                slot.prefilling = false;
                completed_this_step.push_back(slice.slot);
                const bool omit = faults_.drop_second_completion && i > 0;
                if (!omit) {
                    result.prefills.push_back({
                        slice.slot, PrefillOutput::Status::completed,
                        100 + slice.slot, {},
                    });
                }
            } else {
                result.prefills.push_back({
                    slice.slot, PrefillOutput::Status::advanced, -1, {},
                });
            }
        }

        if (faults_.lose_other_pending && !completed_this_step.empty()) {
            for (Slot & slot : slots_) {
                if (slot.active && slot.prefilling) slot.prefilling = false;
            }
        }

        return result;
    }

    void retire(int slot) override {
        if (slot < 0 || slot >= slot_count() || faults_.retire_leaks) return;
        slots_[(size_t)slot] = Slot{};
    }

private:
    struct Slot {
        bool active = false;
        bool prefilling = false;
        int remaining = 0;
        std::vector<int32_t> fed;
    };

    int decoding_count() const {
        int count = 0;
        for (const Slot & slot : slots_) {
            if (slot.active && !slot.prefilling) ++count;
        }
        return count;
    }

    bool valid_decode(const StepPlan & plan, std::string & error) const {
        std::vector<bool> seen(slots_.size(), false);
        if ((int)plan.decode.size() != decoding_count()) {
            error = "plan omits a decoding slot";
            return false;
        }
        for (const StepInput & input : plan.decode) {
            if (input.slot < 0 || input.slot >= slot_count() ||
                seen[(size_t)input.slot] || input.token < 0 ||
                !slots_[(size_t)input.slot].active ||
                slots_[(size_t)input.slot].prefilling) {
                error = "invalid decode input";
                return false;
            }
            seen[(size_t)input.slot] = true;
        }
        for (int slot = 0; slot < slot_count(); ++slot) {
            if (slots_[(size_t)slot].active &&
                !slots_[(size_t)slot].prefilling &&
                !seen[(size_t)slot]) {
                error = "plan omits a decoding slot";
                return false;
            }
        }
        return true;
    }

    bool valid_prefills(const StepPlan & plan, std::string & error) const {
        const StepPlanLimits limits =
            step_plan_limits((int)plan.decode.size());
        const int token_limit = limits.max_prefill_tokens_per_sequence;
        if ((int)plan.prefills.size() > limits.max_prefill_sequences) {
            error = "too many prefills";
            return false;
        }
        std::vector<bool> seen(slots_.size(), false);
        int total_tokens = 0;
        for (const PrefillSlice & slice : plan.prefills) {
            if (slice.slot < 0 || slice.slot >= slot_count() ||
                seen[(size_t)slice.slot] || slice.max_tokens <= 0 ||
                slice.max_tokens > token_limit ||
                !slots_[(size_t)slice.slot].active ||
                !slots_[(size_t)slice.slot].prefilling) {
                error = "invalid prefill slice";
                return false;
            }
            seen[(size_t)slice.slot] = true;
            total_tokens += slice.max_tokens;
            if (total_tokens > limits.max_prefill_tokens_total) {
                error = "too many total prefill tokens";
                return false;
            }
        }
        return true;
    }

    std::vector<Slot> slots_;
    Faults faults_;
    FakeCapabilities capabilities_;
};

static void print_violations(const char * label,
                             const std::vector<std::string> & violations) {
    for (const std::string & violation : violations) {
        std::fprintf(stderr, "  [%s] %s\n", label, violation.c_str());
    }
}

static bool mentions(const std::vector<std::string> & violations,
                     const char * needle) {
    return std::any_of(
        violations.begin(), violations.end(),
        [&](const std::string & violation) {
            return violation.find(needle) != std::string::npos;
        });
}

int main() {
    for (const int slots : {2, 4}) {
        FakeSeqEngine engine(slots);
        const auto violations = check_seq_engine_contract(engine);
        if (!violations.empty()) print_violations("conforming", violations);
        CHECK(violations.empty());
    }

    {
        FakeCapabilities capabilities;
        capabilities.idle = {1, 2, 2};
        capabilities.mixed = {1, 1, 1};
        FakeSeqEngine engine(2, {}, capabilities);
        const auto violations = check_seq_engine_contract(engine);
        if (!violations.empty()) print_violations("conforming-k1", violations);
        CHECK(violations.empty());
    }

    {
        FakeCapabilities capabilities;
        capabilities.idle = {2, 2, 2};
        capabilities.mixed = {1, 1, 1};
        FakeSeqEngine engine(2, {}, capabilities);
        const auto violations = check_seq_engine_contract(engine);
        if (!violations.empty()) {
            print_violations("conforming-width-dependent", violations);
        }
        CHECK(violations.empty());
    }

    struct Case {
        const char * label;
        bool Faults::*fault;
        const char * expected;
    };
    const Case cases[] = {
        {"hard-error-full", &Faults::hard_error_when_full, "full engine"},
        {"reuse-live", &Faults::reuse_live_slot, "reused a live slot"},
        {"drop-decode", &Faults::drop_decode_output, "omitted an output"},
        {"serialize-prefill", &Faults::serialize_prefills,
         "omitted an output"},
        {"accept-omitted", &Faults::accept_omitted_decode,
         "omits a decoding slot"},
        {"accept-invalid-prefill", &Faults::accept_invalid_prefill,
         "prefill work for a decoding slot"},
        {"lose-pending", &Faults::lose_other_pending,
         "valid planned work must succeed"},
        {"overconsume", &Faults::overconsume_prefill,
         "completion before its final token"},
        {"scalar-completion", &Faults::drop_second_completion,
         "omitted an output"},
        {"retire-leak", &Faults::retire_leaks,
         "succeed while a slot is free"},
    };

    for (const Case & test : cases) {
        Faults faults;
        faults.*test.fault = true;
        FakeSeqEngine engine(2, faults);
        const auto violations = check_seq_engine_contract(engine);
        if (!mentions(violations, test.expected)) {
            std::fprintf(stderr,
                         "FAIL %s: expected violation containing '%s'\n",
                         test.label, test.expected);
            print_violations(test.label, violations);
        }
        CHECK(mentions(violations, test.expected));
    }

    std::printf("test_seq_engine_contract: %d checks passed\n", g_checks);
    return 0;
}
