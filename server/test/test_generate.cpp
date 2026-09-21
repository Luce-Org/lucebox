// End-to-end generation test for our qwen35 target forward.
//
// Reads a binary int32 token file (produced by scripts/tokenize_prompt.py),
// runs single-token decode over every token (no batched prefill), generates
// N new tokens via greedy argmax, and writes the resulting int32 token stream
// to an output file for Python-side detokenization.
//
// Also reports decode tok/s (generation only, prompt steps excluded).
//
// Usage:
//   test_generate <qwen35.gguf> <prompt_ids.bin> <n_gen> <out_ids.bin>
//   test_generate --seq-engine-contract <qwen35.gguf> [slots]
//   test_generate --seq-engine-mixed-spec-contract <qwen35.gguf>
//                 <draft.gguf|safetensors> [slots]

#include "dflash27b.h"
#include "internal.h"
#include "qwen35/qwen35_backend.h"
#include "seq_engine_contract.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#ifdef _WIN32
#define setenv(name, value, overwrite) _putenv_s(name, value)
#define unsetenv(name) _putenv_s(name, "")
#endif

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace dflash::common;

struct GenerateStepGraph {
    ggml_context *    ctx = nullptr;
    ggml_cgraph *     gf  = nullptr;
    ggml_gallocr_t    alloc = nullptr;
    ggml_tensor *     inp_embed = nullptr;
    ggml_tensor *     positions = nullptr;
    ggml_tensor *     logits    = nullptr;
};

class ContractQwen35Backend final : public Qwen35Backend {
public:
    using Qwen35Backend::Qwen35Backend;

    const StepGraph & step_graph() const { return target_step_graph(); }
    const TargetCache & cache() const { return target_cache(); }
};

// Build a fresh single-token forward graph. We rebuild per step so that
// `kv_start` updates drive the correct KV cache slot. The graph is cheap to
// rebuild — all the weights + KV cache stay persistent.
static bool build_step_graph(
    GenerateStepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start
) {
    if (sg.alloc) { ggml_gallocr_free(sg.alloc); sg.alloc = nullptr; }
    if (sg.ctx)   { ggml_free(sg.ctx); sg.ctx = nullptr; }

    ggml_init_params ip{};
    ip.mem_size   = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int n_tokens = 1;
    const int hidden = DFLASH27B_TARGET_HIDDEN;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_input(sg.inp_embed);
    ggml_set_input(sg.positions);

    sg.gf = ggml_new_graph_custom(sg.ctx, 8192, false);

    QwenGraphInputs gi{};
    gi.inp_embed      = sg.inp_embed;
    gi.positions      = sg.positions;
    gi.attn_mask      = nullptr;        // n_tokens==1, no mask needed
    gi.n_tokens       = n_tokens;
    gi.kv_start       = kv_start;
    gi.capture_layers = false;

    QwenGraphOutputs go = build_qwen35_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    ggml_set_output(go.logits);
    ggml_build_forward_expand(sg.gf, go.logits);
    sg.logits = go.logits;

    sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

static std::vector<int32_t> read_generate_tokens(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<int32_t> out(sz / sizeof(int32_t));
    f.read((char *)out.data(), sz);
    return out;
}

static bool write_generate_tokens(const std::string & path,
                                  const std::vector<int32_t> & v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write((const char *)v.data(), v.size() * sizeof(int32_t));
    return (bool)f;
}

static bool capture_recurrent_state(
        const TargetCache & cache, int slot, std::vector<uint8_t> & bytes) {
    if (slot < 0 || slot >= cache.n_seq_slots) return false;
    bytes.clear();
    const auto append_slot = [&](ggml_tensor * tensor) {
        if (!tensor || ggml_nbytes(tensor) % cache.n_seq_slots != 0) {
            return false;
        }
        const size_t slab = ggml_nbytes(tensor) / cache.n_seq_slots;
        const size_t old_size = bytes.size();
        bytes.resize(old_size + slab);
        ggml_backend_tensor_get(
            tensor, bytes.data() + old_size,
            static_cast<size_t>(slot) * slab, slab);
        return true;
    };
    for (ggml_tensor * tensor : cache.conv_state) {
        if (!append_slot(tensor)) return false;
    }
    for (ggml_tensor * tensor : cache.ssm_state) {
        if (!append_slot(tensor)) return false;
    }
    return !bytes.empty();
}

// Seed every case at C4 before retiring peers. This exposes a request's
// MMVQ/MMQ or float matvec/matrix transition without changing its input state.
static std::vector<std::string> check_decode_batch_invariance(
        ContractQwen35Backend & backend, SeqEngine & engine, bool with_draft) {
    std::vector<std::string> violations;
    const char * wmma = std::getenv("DFLASH27B_PAGED_WMMA");
    if (wmma && std::atoi(wmma) != 0) {
        std::printf("decode batch invariance skipped: opt-in WMMA uses a separate numerical policy\n");
        return violations;
    }
    const auto require = [&](bool ok, const char * message) {
        if (!ok) violations.emplace_back(message);
        return ok;
    };
    struct CaseResult {
        std::vector<int32_t> pending, emitted;
        std::vector<uint8_t> before, after, logits;
    };
    uint64_t next_request_id = 10000;
    const auto run_case = [&](int cohort, bool speculate, CaseResult & output) {
        std::vector<int> slots;
        std::vector<int32_t> pending(4, -1);
        const auto finish = [&](bool ok) {
            for (int slot : slots) if (slot >= 0) engine.retire(slot);
            return ok;
        };
        for (int i = 0; i < 4; ++i) {
            std::vector<int32_t> prompt(i == 3 ? 33 : 2);
            for (size_t j = 0; j < prompt.size(); ++j) {
                prompt[j] = 31 + 10 * i + static_cast<int32_t>(j % 4);
            }
            const auto admitted = engine.admit(next_request_id++, prompt, SamplerCfg{});
            if (!require(admitted.status == SeqEngine::AdmitResult::Status::admitted,
                         "decode invariance fixture could not admit C4")) {
                return finish(false);
            }
            slots.push_back(admitted.slot);
        }
        int step_index = 0;
        const auto execute = [&](const SeqEngine::StepPlan & plan,
                                  SeqEngine::StepResult & result) {
            ++step_index;
            result = engine.step(plan);
            const std::string protocol_error =
                validate_step_result(plan, result, engine.slot_count());
            if (!result.ok() || !protocol_error.empty()) {
                violations.emplace_back(
                    "decode invariance C" + std::to_string(cohort) +
                    " speculate=" + std::to_string(speculate) +
                    " step=" + std::to_string(step_index) +
                    " engine='" + result.error + "' protocol='" + protocol_error + "'");
                return false;
            }
            for (const auto & row : result.prefills) {
                if (!require(row.status == SeqEngine::PrefillOutput::Status::completed &&
                                 row.token >= 0,
                             "decode invariance fixture prefill failed")) return false;
                const auto it = std::find(slots.begin(), slots.end(), row.slot);
                if (it != slots.end()) pending[it - slots.begin()] = row.token;
            }
            for (const auto & row : result.decode) {
                if (!require(!row.failed && row.token >= 0,
                             "decode invariance fixture decode failed")) return false;
                const auto it = std::find(slots.begin(), slots.end(), row.slot);
                if (it != slots.end()) pending[it - slots.begin()] = row.token;
            }
            return true;
        };
        SeqEngine::StepResult result;
        // Include every already-decoding peer while prefilling the next pair.
        for (int begin : {0, 2}) {
            SeqEngine::StepPlan plan;
            for (int i = 0; i < begin; ++i) {
                plan.decode.push_back({slots[i], pending[i], false});
            }
            for (int i = begin; i < begin + 2; ++i) {
                plan.prefills.push_back({slots[i], i == 3 ? 33 : 2});
            }
            if (!execute(plan, result)) return finish(false);
        }
        SeqEngine::StepPlan seed;
        for (int i = 0; i < 4; ++i) seed.decode.push_back({slots[i], pending[i], false});
        if (!execute(seed, result) ||
            !require(!backend.step_graph().parent_ids &&
                         std::all_of(result.decode.begin(), result.decode.end(),
                             [](const auto & row) { return row.committed_tokens.empty(); }),
                     "decode invariance common seed attempted speculation")) {
            return finish(false);
        }
        output.pending = pending;
        for (int i = cohort - 1; i < 3; ++i) {
            engine.retire(slots[i]);
            slots[i] = -1;
        }
        if (!require(capture_recurrent_state(backend.cache(), slots[3], output.before),
                     "decode invariance pre-step state capture failed")) return finish(false);
        SeqEngine::StepPlan plan;
        for (int i = 0; i < 4; ++i) {
            if (slots[i] >= 0) plan.decode.push_back({slots[i], pending[i], speculate});
        }
        if (!execute(plan, result)) return finish(false);
        const StepGraph & graph = backend.step_graph();
        const bool expected_shape = speculate
            ? graph.parent_ids && graph.parent_ids->ne[0] == 8 &&
                graph.parent_ids->ne[1] == cohort
            : !graph.parent_ids;
        const int root_row = (cohort - 1) * (speculate ? 8 : 1);
        if (!require(expected_shape && graph.logits &&
                         graph.logits->type == GGML_TYPE_F32 &&
                         root_row < graph.logits->ne[1],
                     "decode invariance fixture used an unexpected target graph")) {
            return finish(false);
        }
        output.logits.resize(graph.logits->ne[0] * sizeof(float));
        ggml_backend_tensor_get(graph.logits, output.logits.data(),
                                root_row * graph.logits->nb[1], output.logits.size());
        for (const auto & row : result.decode) {
            if (!speculate && !require(row.committed_tokens.empty(),
                    "ordinary decode invariance fixture returned a speculative burst")) {
                return finish(false);
            }
            if (row.slot == slots[3]) {
                output.emitted = row.committed_tokens;
                output.emitted.push_back(row.token);
            }
        }
        if (!require(!output.emitted.empty() &&
                         capture_recurrent_state(backend.cache(), slots[3], output.after),
                     "decode invariance post-step state capture failed")) return finish(false);
        return finish(true);
    };
    for (bool speculate : {false, true}) {
        if (speculate && !with_draft) continue;
        CaseResult reference;
        if (!run_case(4, speculate, reference)) continue;
        for (int cohort : {1, 2, 3}) {
            if (speculate && cohort != 3) continue;
            CaseResult other;
            if (!run_case(cohort, speculate, other)) continue;
            if (!require(reference.pending == other.pending && reference.before == other.before,
                         "decode invariance fixture did not reproduce its C4 seed")) continue;
            require(reference.logits == other.logits,
                    "changing the decode cohort changed the request's root logits");
            const size_t common = std::min(reference.emitted.size(), other.emitted.size());
            require(std::equal(reference.emitted.begin(), reference.emitted.begin() + common,
                               other.emitted.begin()),
                    "changing the decode cohort changed the request's emitted tokens");
            // Different accepted counts end at different positions. Root logits
            // remain comparable; durable state is comparable at equal positions.
            if (reference.emitted.size() == other.emitted.size()) {
                require(reference.after == other.after,
                        "changing the decode cohort changed durable recurrent state");
            }
        }
    }
    return violations;
}

static std::vector<std::string> check_mixed_spec_prompt_tail(
        ContractQwen35Backend & backend, SeqEngine & engine) {
    std::vector<std::string> violations;
    const auto require = [&](bool ok, const char * message) {
        if (!ok) violations.emplace_back(message);
        return ok;
    };

    std::vector<int32_t> completing_prompt(1024);
    std::vector<int32_t> remaining_prompt(1025);
    for (size_t i = 0; i < completing_prompt.size(); ++i) {
        completing_prompt[i] = 11 + static_cast<int32_t>(i % 4);
    }
    for (size_t i = 0; i < remaining_prompt.size(); ++i) {
        remaining_prompt[i] = 21 + static_cast<int32_t>(i % 5);
    }

    struct CaseResult {
        int32_t prompt_token = -1;
        int32_t decoder_token = -1;
        int32_t peer_token = -1;
        std::vector<uint8_t> recurrent_state;
    };
    const char * paged_wmma_env = std::getenv("DFLASH27B_PAGED_WMMA");
    const bool paged_wmma = paged_wmma_env && std::atoi(paged_wmma_env) != 0;
    uint64_t next_request_id = 100;
    const auto run_case = [&](int tail_tokens, bool allow_speculation,
                              bool ar_peer, CaseResult & output) {
        const SamplerCfg greedy{};
        const PrefixCaptureTicket completing_capture{
            next_request_id++, {1, 1024}};
        const PrefixCaptureTicket remaining_capture{
            next_request_id++, {2, 1024}};
        std::vector<int> live_slots;
        const auto retire_all = [&]() {
            for (int slot : live_slots) engine.retire(slot);
            live_slots.clear();
            engine.discard_prefix_store(completing_capture.checkpoint);
            engine.discard_prefix_store(remaining_capture.checkpoint);
        };
        const auto admit = [&](const std::vector<int32_t> & prompt,
                               PrefixCaptureTicket capture = {}) {
            PrefixStorePlan prefix;
            prefix.capture = capture;
            const SeqEngine::AdmitResult admitted = capture.valid()
                ? engine.admit_with_prefix(
                      next_request_id++, prompt, greedy, prefix)
                : engine.admit(next_request_id++, prompt, greedy);
            if (admitted.status !=
                SeqEngine::AdmitResult::Status::admitted) {
                return -1;
            }
            live_slots.push_back(admitted.slot);
            if (!require(admitted.prefix_store.capture == capture,
                         "mixed-spec prefill did not arm its prefix capture")) {
                return -1;
            }
            return admitted.slot;
        };
        const auto decode_token = [](
                const SeqEngine::StepResult & result, int slot,
                int32_t & token) {
            const auto it = std::find_if(
                result.decode.begin(), result.decode.end(),
                [slot](const SeqEngine::DecodeOutput & row) {
                    return row.slot == slot;
                });
            if (it == result.decode.end() || it->failed || it->token < 0) {
                return false;
            }
            token = it->token;
            return true;
        };

        const int established_a = admit({31, 32});
        const int established_b = admit({41, 42});
        if (!require(established_a >= 0 && established_b >= 0,
                     "mixed-spec boundary fixture could not admit decoders")) {
            retire_all();
            return false;
        }
        SeqEngine::StepPlan establish_plan;
        establish_plan.prefills.push_back({established_a, 2});
        establish_plan.prefills.push_back({established_b, 2});
        const SeqEngine::StepResult established = engine.step(establish_plan);
        if (!require(established.ok() && established.prefills.size() == 2,
                     "mixed-spec boundary fixture could not establish decoders")) {
            retire_all();
            return false;
        }
        int32_t token_a = -1;
        int32_t token_b = -1;
        for (const SeqEngine::PrefillOutput & row : established.prefills) {
            if (row.slot == established_a) token_a = row.token;
            if (row.slot == established_b) token_b = row.token;
        }
        if (!require(token_a >= 0 && token_b >= 0,
                     "mixed-spec boundary decoder prefill did not complete")) {
            retire_all();
            return false;
        }

        const int completing = admit(completing_prompt, completing_capture);
        const int remaining = admit(remaining_prompt, remaining_capture);
        if (!require(completing >= 0 && remaining >= 0,
                     "mixed-spec boundary fixture could not admit prefills")) {
            retire_all();
            return false;
        }

        // Preserve the original 512+512 mixed case. For shorter tails,
        // seed both runs with identical ordinary steps so the final graph
        // alone determines any durable-state difference.
        const std::vector<int> seed_chunks = tail_tokens == 512
            ? std::vector<int>{512} : std::vector<int>{512, 512 - tail_tokens};
        for (int chunk : seed_chunks) {
            const bool seed_speculation = allow_speculation && tail_tokens == 512;
            SeqEngine::StepPlan first_plan;
            first_plan.decode.push_back(
                {established_a, token_a, seed_speculation});
            first_plan.decode.push_back(
                {established_b, token_b, seed_speculation && !ar_peer});
            first_plan.prefills.push_back({completing, chunk});
            first_plan.prefills.push_back({remaining, chunk});
            const SeqEngine::StepResult first = engine.step(first_plan);
            if (!require(first.ok() && first.prefills.size() == 2,
                         "mixed-spec boundary first prefill chunks failed") ||
                !require(validate_step_result(
                             first_plan, first, engine.slot_count()).empty(),
                         "mixed-spec seed step returned invalid row results") ||
                !require(decode_token(first, established_a, token_a) &&
                             decode_token(first, established_b, token_b),
                         "mixed-spec boundary first decode rows failed")) {
                retire_all();
                return false;
            }
            for (const auto & row : first.prefills) {
                require(!row.prefix_store.attempted(),
                        "mixed-spec prefix capture occurred before its boundary");
            }
        }

        SeqEngine::StepPlan boundary_plan;
        boundary_plan.decode.push_back(
            {established_a, token_a, allow_speculation});
        boundary_plan.decode.push_back(
            {established_b, token_b, allow_speculation && !ar_peer});
        boundary_plan.prefills.push_back({completing, tail_tokens});
        boundary_plan.prefills.push_back({remaining, tail_tokens});
        const SeqEngine::StepResult boundary = engine.step(boundary_plan);
        if (!require(boundary.ok(),
                     "mixed-spec 1024-token boundary step failed") ||
            !require(validate_step_result(
                         boundary_plan, boundary, engine.slot_count()).empty(),
                     "mixed-spec boundary returned invalid row results") ||
            !require(decode_token(boundary, established_a, token_a) &&
                         decode_token(boundary, established_b, token_b),
                     "mixed-spec boundary decode rows failed")) {
            retire_all();
            return false;
        }
        const auto completing_row = std::find_if(
            boundary.prefills.begin(), boundary.prefills.end(),
            [completing](const SeqEngine::PrefillOutput & row) {
                return row.slot == completing;
            });
        const auto remaining_row = std::find_if(
            boundary.prefills.begin(), boundary.prefills.end(),
            [remaining](const SeqEngine::PrefillOutput & row) {
                return row.slot == remaining;
            });
        if (!require(
                completing_row != boundary.prefills.end() &&
                    completing_row->status ==
                        SeqEngine::PrefillOutput::Status::completed,
                "mixed-spec boundary completing prefill did not complete") ||
            !require(
                remaining_row != boundary.prefills.end() &&
                    remaining_row->status ==
                        SeqEngine::PrefillOutput::Status::advanced,
                "mixed-spec boundary peer did not remain in prefill")) {
            retire_all();
            return false;
        }
        const auto saved_capture = [](const PrefixStoreEvent & event,
                                      PrefixCaptureTicket expected) {
            return event.status == PrefixStoreEvent::Status::saved &&
                event.ticket == expected && event.bytes > 0 &&
                event.error.empty();
        };
        require(saved_capture(completing_row->prefix_store, completing_capture),
                "mixed-spec completing prefill did not save its prefix capture");
        require(saved_capture(remaining_row->prefix_store, remaining_capture),
                "mixed-spec advancing prefill did not save its prefix capture");
        output.prompt_token = completing_row->token;
        output.decoder_token = token_a;
        output.peer_token = token_b;
        if (!require(capture_recurrent_state(
                         backend.cache(), completing, output.recurrent_state),
                     "mixed-spec boundary recurrent-state capture failed")) {
            retire_all();
            return false;
        }

        // The canonical projection and attention policies also cover tiny
        // tails. Opt-in WMMA retains its separate ordinary graph.
        const bool expect_tree = allow_speculation && !paged_wmma;
        const StepGraph & graph = backend.step_graph();
        if (!expect_tree) {
            require(!graph.parent_ids,
                    "mixed prefill changed the protected ordinary target graph");
            for (const auto & row : boundary.decode) {
                require(row.committed_tokens.empty(),
                        "ordinary mixed-prefill fallback returned speculative tokens");
            }
        } else {
            const bool tree_graph = require(
                graph.parent_ids && graph.logits_row_indices &&
                    graph.argmax_tokens,
                "mixed prefill did not retain the fixed-chain target graph");
            if (tree_graph) {
                const int tree_width =
                    static_cast<int>(graph.parent_ids->ne[0]);
                const int tree_lanes =
                    static_cast<int>(graph.parent_ids->ne[1]);
                const int logits_count =
                    static_cast<int>(graph.logits_row_indices->ne[0]);
                const int ar_rows = ar_peer ? 1 : 0;
                require(tree_width == 8 && tree_lanes == 2 - ar_rows,
                        "mixed prefill used the wrong fixed-chain W8 bucket");
                const int expected_logits = tree_width * tree_lanes + 1 + ar_rows;
                require(logits_count == expected_logits,
                        "mixed prefill produced the wrong compact logits shape");
                if (logits_count == expected_logits) {
                    std::vector<int32_t> logits_rows(
                        static_cast<size_t>(logits_count), -1);
                    ggml_backend_tensor_get(
                        graph.logits_row_indices, logits_rows.data(), 0,
                        sizeof(int32_t) * logits_rows.size());
                    require(logits_rows[0] == tail_tokens - 1,
                            "mixed speculative prompt tail did not gather its final row");
                    for (int row = 1; row < logits_count; ++row) {
                        require(
                            logits_rows[static_cast<size_t>(row)] ==
                                2 * tail_tokens + row - 1,
                            "mixed speculative tree logits were not gathered "
                            "after the direct prefix");
                    }
                }
            }
        }
        // The peer's capture boundary precedes prompt completion. Its next
        // ordinary step must complete without emitting the capture a second time.
        SeqEngine::StepPlan finish_plan;
        finish_plan.decode = {
            {established_a, token_a, false},
            {established_b, token_b, false},
            {completing, output.prompt_token, false}};
        finish_plan.prefills.push_back({remaining, 1});
        const SeqEngine::StepResult finished = engine.step(finish_plan);
        require(finished.ok() &&
                    std::all_of(finished.decode.begin(), finished.decode.end(),
                                [](const auto & row) { return !row.failed; }) &&
                    validate_step_result(
                        finish_plan, finished, engine.slot_count()).empty() &&
                    finished.prefills.size() == 1 &&
                    finished.prefills[0].status ==
                        SeqEngine::PrefillOutput::Status::completed &&
                    !finished.prefills[0].prefix_store.attempted(),
                "mixed-spec prefix capture disrupted the remaining prefill");
        retire_all();
        return true;
    };

    for (int tail_tokens : {512, 9, 3, 1}) {
        CaseResult ordinary;
        if (!run_case(tail_tokens, false, false, ordinary)) continue;
        // Tail nine also checks compact AR rows beside a speculative lane
        // with identical seed history.
        const int variants = tail_tokens == 9 ? 2 : 1;
        for (int variant = 0; variant < variants; ++variant) {
            CaseResult speculative;
            if (!run_case(tail_tokens, true, variant == 1, speculative)) continue;
            require(speculative.prompt_token == ordinary.prompt_token,
                    "mixed speculation changed the 1024-token prompt-tail result");
            require(speculative.recurrent_state == ordinary.recurrent_state,
                    "mixed speculation changed durable convolution/Gated DeltaNet state");
            const bool ordinary_route = paged_wmma;
            if (ordinary_route) {
                require(speculative.decoder_token == ordinary.decoder_token,
                        "mixed-prefill fallback changed the decoder's next token");
            }
            if (variant == 1 || ordinary_route) {
                require(speculative.peer_token == ordinary.peer_token,
                        "mixed speculation changed the ordinary peer's next token");
            }
        }
    }
    // The first pending token is at position 249, so a width-eight proposal
    // crosses the ordinary 256-token attention launch window. The tree
    // launch must cover the proposal beyond that boundary.
    std::vector<int> boundary_slots;
    const auto admit_boundary = [&](int count, int32_t token) {
        const auto admitted = engine.admit(
            next_request_id++, std::vector<int32_t>(count, token), SamplerCfg{});
        if (!require(admitted.status == SeqEngine::AdmitResult::Status::admitted,
                     "launch-boundary fixture could not admit a request")) {
            return -1;
        }
        boundary_slots.push_back(admitted.slot);
        return admitted.slot;
    };
    const int boundary_a = admit_boundary(249, 31);
    const int boundary_b = admit_boundary(249, 41);
    if (boundary_a >= 0 && boundary_b >= 0) {
        SeqEngine::StepPlan seed;
        seed.prefills = {{boundary_a, 249}, {boundary_b, 249}};
        const auto seeded = engine.step(seed);
        if (require(seeded.ok() && seeded.prefills.size() == 2,
                    "launch-boundary fixture could not seed decoders")) {
            const int prefill_a = admit_boundary(9, 11);
            const int prefill_b = admit_boundary(9, 21);
            if (prefill_a >= 0 && prefill_b >= 0) {
                SeqEngine::StepPlan crossing;
                for (const auto & row : seeded.prefills) {
                    crossing.decode.push_back({row.slot, row.token, true});
                }
                crossing.prefills = {{prefill_a, 9}, {prefill_b, 9}};
                const auto result = engine.step(crossing);
                require(result.ok() &&
                            validate_step_result(crossing, result,
                                                 engine.slot_count()).empty(),
                        "launch-boundary step returned invalid results");
                require(bool(backend.step_graph().parent_ids) == !paged_wmma,
                        "launch-boundary step selected the wrong target graph");
                for (const auto & row : result.decode) {
                    require(!row.failed && row.token >= 0,
                            "launch-boundary decode failed");
                    if (paged_wmma) {
                        require(row.committed_tokens.empty(),
                                "WMMA fallback attempted speculation");
                    }
                }
            }
        }
    }
    for (int slot : boundary_slots) engine.retire(slot);
    return violations;
}

static int run_seq_engine_contract(
        const char * gguf_path, const char * draft_path, int slots) {
    const int min_slots = draft_path ? 4 : 2;
    if (slots < min_slots || slots > 64) {
        std::fprintf(stderr, "slots must be in [%d, 64], got %d\n",
                     min_slots, slots);
        return 2;
    }

    // Keep this mode independent of a caller's KVFlash environment: paged
    // attention and KVFlash are intentionally mutually exclusive. Prevent
    // arbitrary raw token IDs from ending the checker before its three
    // batched steps have exercised state carry.
    unsetenv("DFLASH_KVFLASH");
    setenv("DFLASH_MIN_TOKENS", "8", 1);

    Qwen35Config cfg;
    cfg.target_path = gguf_path;
    cfg.draft_path = draft_path;
    cfg.draft_block_size = draft_path ? 8 : 0;
    cfg.device.gpu = 0;
    cfg.device.max_ctx = draft_path ? 1152 : 256;
    cfg.draft_gpu = 0;
    cfg.paged_attention = true;
    cfg.max_concurrency = slots;
    cfg.kv_pool_tokens = draft_path
        ? static_cast<int64_t>(cfg.device.max_ctx) * slots
        : 0;

    ContractQwen35Backend backend(cfg);
    if (!backend.init()) {
        std::fprintf(stderr, "seq-engine backend init failed: %s\n",
                     dflash27b_last_error());
        return 1;
    }

    SeqEngine * engine = backend.seq_engine();
    if (!engine) {
        std::fprintf(stderr,
                     "seq-engine backend did not expose a concurrent engine\n");
        return 1;
    }

    std::vector<std::string> violations = check_seq_engine_contract(*engine);
    if (violations.empty() && slots >= 4) {
        const auto batch_violations =
            check_decode_batch_invariance(backend, *engine, draft_path != nullptr);
        violations.insert(violations.end(), batch_violations.begin(), batch_violations.end());
    }
    if (violations.empty() && draft_path) {
        std::vector<std::string> mixed_violations =
            check_mixed_spec_prompt_tail(backend, *engine);
        violations.insert(
            violations.end(), mixed_violations.begin(),
            mixed_violations.end());
    }
    for (const std::string & violation : violations) {
        std::fprintf(stderr, "seq-engine contract: %s\n", violation.c_str());
    }
    if (!violations.empty()) return 1;

    std::printf("seq-engine %scontract passed against Qwen35Backend (%d slots)\n",
                draft_path ? "mixed-spec " : "", slots);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc >= 2 &&
        std::strcmp(argv[1], "--seq-engine-contract") == 0) {
        if (argc < 3 || argc > 4) {
            std::fprintf(stderr,
                "usage: %s --seq-engine-contract <qwen35.gguf> [slots]\n",
                argv[0]);
            return 2;
        }
        const int slots = argc == 4 ? std::atoi(argv[3]) : 4;
        return run_seq_engine_contract(argv[2], nullptr, slots);
    }
    if (argc >= 2 &&
        std::strcmp(argv[1], "--seq-engine-mixed-spec-contract") == 0) {
        if (argc < 4 || argc > 5) {
            std::fprintf(stderr,
                "usage: %s --seq-engine-mixed-spec-contract "
                "<qwen35.gguf> <draft.gguf|safetensors> [slots]\n",
                argv[0]);
            return 2;
        }
        const int slots = argc == 5 ? std::atoi(argv[4]) : 4;
        return run_seq_engine_contract(argv[2], argv[3], slots);
    }

    if (argc < 5) {
        std::fprintf(stderr,
            "usage: %s <qwen35.gguf> <prompt_ids.bin> <n_gen> <out_ids.bin>\n"
            "       %s --seq-engine-contract <qwen35.gguf> [slots]\n"
            "       %s --seq-engine-mixed-spec-contract <qwen35.gguf> "
            "<draft.gguf|safetensors> [slots]\n",
            argv[0], argv[0], argv[0]);
        return 2;
    }
    const char * gguf_path   = argv[1];
    const char * prompt_path = argv[2];
    const int    n_gen       = std::atoi(argv[3]);
    const char * out_path    = argv[4];
    int stream_fd = -1;
    for (int i = 5; i < argc; i++) {
        if (std::strncmp(argv[i], "--stream-fd=", 12) == 0) {
            stream_fd = std::atoi(argv[i] + 12);
        }
        // KV cache type flags (mirror llama-cli -ctk / -ctv).
        // Set the env var before resolve_kv_types() reads it inside create_target_cache.
        else if (std::strcmp(argv[i], "--cache-type-k") == 0 || std::strcmp(argv[i], "-ctk") == 0) {
            if (i + 1 < argc) setenv("DFLASH27B_KV_K", argv[++i], 1);
        }
        else if (std::strncmp(argv[i], "--cache-type-k=", 15) == 0) {
            setenv("DFLASH27B_KV_K", argv[i] + 15, 1);
        }
        else if (std::strncmp(argv[i], "-ctk=", 5) == 0) {
            setenv("DFLASH27B_KV_K", argv[i] + 5, 1);
        }
        else if (std::strcmp(argv[i], "--cache-type-v") == 0 || std::strcmp(argv[i], "-ctv") == 0) {
            if (i + 1 < argc) setenv("DFLASH27B_KV_V", argv[++i], 1);
        }
        else if (std::strncmp(argv[i], "--cache-type-v=", 15) == 0) {
            setenv("DFLASH27B_KV_V", argv[i] + 15, 1);
        }
        else if (std::strncmp(argv[i], "-ctv=", 5) == 0) {
            setenv("DFLASH27B_KV_V", argv[i] + 5, 1);
        }
    }
    auto stream_emit = [&](int32_t tok) {
        if (stream_fd < 0) return;
        int32_t v = tok;
#if defined(_WIN32)
        DWORD written;
        WriteFile((HANDLE)(intptr_t)stream_fd, &v, sizeof(v), &written, nullptr);
#else
        ssize_t n = ::write(stream_fd, &v, sizeof(v));
        (void)n;
#endif
    };

    // ── Load model and cache
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    TargetWeights w;
    if (!load_target_gguf(gguf_path, backend, w)) {
        std::fprintf(stderr, "load: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[target] %s\n", dflash27b_last_error());

    const int max_ctx = 4096;
    TargetCache cache;
    if (!create_target_cache(w, max_ctx, /*max_verify_tokens=*/0, backend, cache)) {
        std::fprintf(stderr, "cache: %s\n", dflash27b_last_error());
        return 1;
    }

    auto prompt = read_generate_tokens(prompt_path);
    if (prompt.empty()) { std::fprintf(stderr, "empty prompt bin\n"); return 1; }
    std::printf("[prompt] %zu tokens: ", prompt.size());
    for (auto t : prompt) std::printf("%d ", t);
    std::printf("\n");

    if ((int)prompt.size() + n_gen > max_ctx) {
        std::fprintf(stderr, "prompt+gen exceeds max_ctx\n");
        return 1;
    }

    std::vector<int32_t> all_tokens = prompt;
    all_tokens.reserve(prompt.size() + n_gen);

    const int hidden = DFLASH27B_TARGET_HIDDEN;
    std::vector<float> embed_buf(hidden);

    GenerateStepGraph sg;

    // ── Helper: run one step given current token + absolute position
    auto run_step = [&](int32_t tok, int pos) -> int32_t {
        if (!build_step_graph(sg, w, cache, backend, pos)) {
            std::fprintf(stderr, "build_step_graph failed at pos=%d\n", pos);
            std::exit(1);
        }

        // CPU embed
        int32_t ids[1] = { tok };
        if (!w.embedder.embed(ids, 1, embed_buf.data())) {
            std::fprintf(stderr, "embed failed tok=%d\n", tok);
            std::exit(1);
        }
        ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0,
                                sizeof(float) * embed_buf.size());

        // M-RoPE positions: 4 copies of pos
        int32_t p4[4] = { pos, pos, pos, pos };
        ggml_backend_tensor_set(sg.positions, p4, 0, sizeof(int32_t) * 4);

        auto st = ggml_backend_graph_compute(backend, sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "compute failed at pos=%d (%d)\n", pos, (int)st);
            std::exit(1);
        }

        // argmax on logits
        const int vocab = DFLASH27B_TARGET_VOCAB;
        std::vector<float> logits(vocab);
        ggml_backend_tensor_get(sg.logits, logits.data(), 0, sizeof(float) * vocab);
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < vocab; i++) {
            if (logits[i] > bv) { bv = logits[i]; best = i; }
        }
        return best;
    };

    // ── Prefill: feed prompt tokens one at a time (decode-only mode).
    //    We throw away the logits for all prompt tokens except the last one.
    int next = -1;
    for (int i = 0; i < (int)prompt.size(); i++) {
        next = run_step(prompt[i], i);
    }
    std::printf("[prefill] last-token argmax=%d\n", next);

    // ── Generation loop
    auto t_start = std::chrono::steady_clock::now();
    int gen_start_pos = (int)prompt.size();
    for (int g = 0; g < n_gen; g++) {
        int32_t tok = next;
        all_tokens.push_back(tok);
        stream_emit(tok);
        next = run_step(tok, gen_start_pos + g);
    }
    auto t_end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t_end - t_start).count();
    double tps  = n_gen / std::max(1e-9, secs);

    // Also push the final next token so downstream sees it
    all_tokens.push_back(next);

    std::printf("[gen] %d new tokens in %.3f s  ->  %.2f tok/s\n", n_gen, secs, tps);
    std::printf("[gen] tokens: ");
    for (int i = 0; i < n_gen; i++) std::printf("%d ", all_tokens[prompt.size() + i]);
    std::printf("\n");

    write_generate_tokens(out_path, all_tokens);
    std::printf("[out] wrote %zu tokens to %s\n", all_tokens.size(), out_path);

    if (sg.alloc) ggml_gallocr_free(sg.alloc);
    if (sg.ctx)   ggml_free(sg.ctx);
    free_target_cache(cache);
    free_target_weights(w);
    ggml_backend_free(backend);
    return 0;
}
