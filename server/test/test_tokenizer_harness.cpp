// Tokenizer test harness — reads commands from stdin, writes results to stdout.
// Used by test_tokenizer.py to compare C++ tokenizer output vs HuggingFace.
//
// Protocol (line-oriented, JSON):
//   → {"cmd":"encode","text":"Hello, world!"}
//   ← {"ids":[9707,11,1879,0]}
//   → {"cmd":"decode","ids":[9707,11,1879,0]}
//   ← {"text":"Hello, world!"}
//   → {"cmd":"token_text","id":9707}
//   ← {"text":"Hello"}
//   → {"cmd":"info"}
//   ← {"vocab_size":151936,"eos_id":151645,"bos_id":-1}
//   → {"cmd":"quit"}

#include "server/tokenizer.h"
#include <nlohmann/json.hpp>

#include <cstdio>
#include <iostream>
#include <string>

using json = nlohmann::json;
using namespace dflash::common;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: test_tokenizer_harness <model.gguf>\n");
        return 1;
    }

    Tokenizer tok;
    if (!tok.load_from_gguf(argv[1])) {
        std::fprintf(stderr, "Failed to load tokenizer from %s\n", argv[1]);
        return 1;
    }

    // Signal ready.
    std::fprintf(stderr, "[harness] ready\n");

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        try {
            json req = json::parse(line);
            std::string cmd = req.value("cmd", "");

            if (cmd == "encode") {
                std::string text = req["text"].get<std::string>();
                auto ids = tok.encode(text);
                json resp = {{"ids", ids}};
                std::cout << resp.dump() << "\n" << std::flush;
            } else if (cmd == "decode") {
                auto ids = req["ids"].get<std::vector<int32_t>>();
                std::string text = tok.decode(ids);
                json resp = {{"text", text}};
                std::cout << resp.dump() << "\n" << std::flush;
            } else if (cmd == "token_text") {
                int32_t id = req["id"].get<int32_t>();
                std::string text = tok.token_text(id);
                json resp = {{"text", text}};
                std::cout << resp.dump() << "\n" << std::flush;
            } else if (cmd == "raw_token") {
                int32_t id = req["id"].get<int32_t>();
                const std::string & raw = tok.raw_token(id);
                json resp = {{"text", raw}};
                std::cout << resp.dump() << "\n" << std::flush;
            } else if (cmd == "info") {
                json resp = {
                    {"vocab_size", tok.vocab_size()},
                    {"eos_id", tok.eos_id()},
                    {"bos_id", tok.bos_id()},
                };
                std::cout << resp.dump() << "\n" << std::flush;
            } else if (cmd == "quit") {
                break;
            } else {
                json resp = {{"error", "unknown cmd: " + cmd}};
                std::cout << resp.dump() << "\n" << std::flush;
            }
        } catch (const std::exception & e) {
            json resp = {{"error", std::string("parse error: ") + e.what()}};
            std::cout << resp.dump() << "\n" << std::flush;
        }
    }
    return 0;
}
