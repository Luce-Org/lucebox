#include "stub_model_backend.h"

#include "server/tokenizer.h"

#include <chrono>
#include <cstdio>
#include <thread>

namespace dflash::common::test {

GenerateResult StubModelBackend::generate_impl(const GenerateRequest & req,
                                               const DaemonIO & io) {
    GenerateResult r;

    const std::string rendered = tokenizer_.decode(req.prompt);
    const Scenario * sc = store_.match(rendered);
    if (!sc) {
        r.ok = false;
        r.error = "stub: no scenario matches prompt (size="
                  + std::to_string(rendered.size()) + ")";
        std::fprintf(stderr,
            "[stub] no scenario match. last 120b of prompt: %s\n",
            rendered.substr(rendered.size() > 120 ? rendered.size() - 120 : 0).c_str());
        return r;
    }

    if (!sc->response.ok) {
        r.ok = false;
        r.error = sc->response.error.empty()
            ? std::string("stub: scenario '") + sc->name + "' declared failure"
            : sc->response.error;
        return r;
    }

    std::fprintf(stderr, "[stub] match=%s emitting %zu scripted tokens\n",
        sc->name.c_str(), sc->response.tokens.size());

    r.ok = true;
    bool aborted = false;

    auto emit = [&](int32_t id) -> bool {
        if (req.on_token && !req.on_token(id)) return false;
        if (io.on_token  && !io.on_token(id))  return false;
        r.tokens.push_back(id);
        return true;
    };

    for (const auto & t : sc->response.tokens) {
        if (t.special) {
            int32_t id = tokenizer_.token_to_id(t.text);
            if (id < 0) {
                r.ok = false;
                r.error = "stub: special token not in vocab: " + t.text;
                return r;
            }
            if (!emit(id)) { aborted = true; break; }
        } else {
            auto ids = tokenizer_.encode(t.text);
            for (int32_t id : ids) {
                if (!emit(id)) { aborted = true; break; }
            }
            if (aborted) break;
        }
        if (sc->response.decode_us > 0) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(sc->response.decode_us));
        }
    }

    return r;
}

}  // namespace dflash::common::test
