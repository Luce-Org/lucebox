#include "scenario_store.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

namespace dflash::common::test {

using json = nlohmann::json;

namespace {

bool parse_token(const json & j, ScenarioToken & out, const std::string & file) {
    if (j.is_string()) {
        out.text    = j.get<std::string>();
        out.special = false;
        return true;
    }
    if (j.is_object()) {
        if (!j.contains("text") || !j["text"].is_string()) {
            std::fprintf(stderr,
                "[scenario] %s: token object missing 'text' string\n",
                file.c_str());
            return false;
        }
        out.text    = j["text"].get<std::string>();
        out.special = j.value("special", false);
        return true;
    }
    std::fprintf(stderr,
        "[scenario] %s: token must be a string or {text,special} object\n",
        file.c_str());
    return false;
}

bool parse_scenario(const json & j, Scenario & out, const std::string & file) {
    if (!j.is_object()) {
        std::fprintf(stderr, "[scenario] %s: top level must be object\n", file.c_str());
        return false;
    }
    out.name        = j.value("name", file);
    out.description = j.value("description", "");

    if (!j.contains("match") || !j["match"].is_object()) {
        std::fprintf(stderr, "[scenario] %s: missing 'match' object\n", file.c_str());
        return false;
    }
    const auto & m = j["match"];
    if (!m.contains("prompt_suffix") || !m["prompt_suffix"].is_string()) {
        std::fprintf(stderr,
            "[scenario] %s: match.prompt_suffix must be a string\n", file.c_str());
        return false;
    }
    out.match_prompt_suffix = m["prompt_suffix"].get<std::string>();
    if (out.match_prompt_suffix.empty()) {
        std::fprintf(stderr,
            "[scenario] %s: match.prompt_suffix is empty — would match every prompt\n",
            file.c_str());
        return false;
    }

    if (!j.contains("response") || !j["response"].is_object()) {
        std::fprintf(stderr, "[scenario] %s: missing 'response' object\n", file.c_str());
        return false;
    }
    const auto & r = j["response"];
    out.response.ok            = r.value("ok", true);
    out.response.error         = r.value("error", "");
    out.response.finish_reason = r.value("finish_reason", "stop");
    out.response.decode_us     = r.value("decode_us", 0);

    if (out.response.ok) {
        if (!r.contains("tokens") || !r["tokens"].is_array()) {
            std::fprintf(stderr,
                "[scenario] %s: response.tokens must be an array when ok=true\n",
                file.c_str());
            return false;
        }
        out.response.tokens.reserve(r["tokens"].size());
        for (const auto & jt : r["tokens"]) {
            ScenarioToken tok;
            if (!parse_token(jt, tok, file)) return false;
            out.response.tokens.push_back(std::move(tok));
        }
    }
    return true;
}

bool ends_with(const std::string & s, const std::string & suffix) {
    if (suffix.size() > s.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

}  // namespace

bool ScenarioStore::load_file(const std::string & path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "[scenario] cannot open: %s\n", path.c_str());
        return false;
    }
    std::stringstream buf;
    buf << f.rdbuf();
    json j;
    try {
        j = json::parse(buf.str());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[scenario] %s: JSON parse error: %s\n",
            path.c_str(), e.what());
        return false;
    }
    Scenario sc;
    if (!parse_scenario(j, sc, path)) return false;
    scenarios_.push_back(std::move(sc));
    std::fprintf(stderr, "[scenario] loaded %s (suffix=%zub, tokens=%zu)\n",
        scenarios_.back().name.c_str(),
        scenarios_.back().match_prompt_suffix.size(),
        scenarios_.back().response.tokens.size());
    return true;
}

bool ScenarioStore::load_directory(const std::string & dir) {
    DIR * d = opendir(dir.c_str());
    if (!d) {
        std::fprintf(stderr, "[scenario] cannot open dir: %s\n", dir.c_str());
        return false;
    }
    std::vector<std::string> files;
    while (struct dirent * de = readdir(d)) {
        std::string name = de->d_name;
        if (name.size() < 6) continue;
        if (name.compare(name.size() - 5, 5, ".json") != 0) continue;
        files.push_back(dir + "/" + name);
    }
    closedir(d);
    std::sort(files.begin(), files.end());  // deterministic load order
    bool all_ok = true;
    for (const auto & p : files) {
        if (!load_file(p)) all_ok = false;
    }
    return all_ok;
}

const Scenario * ScenarioStore::match(const std::string & rendered_prompt) const {
    const Scenario * best = nullptr;
    for (const auto & sc : scenarios_) {
        if (!ends_with(rendered_prompt, sc.match_prompt_suffix)) continue;
        if (!best ||
            sc.match_prompt_suffix.size() > best->match_prompt_suffix.size()) {
            best = &sc;
        } else if (sc.match_prompt_suffix.size() == best->match_prompt_suffix.size()) {
            std::fprintf(stderr,
                "[scenario] tie on suffix length %zu between '%s' and '%s' — "
                "load-order earlier wins ('%s'); add a longer suffix to disambiguate\n",
                sc.match_prompt_suffix.size(),
                best->name.c_str(), sc.name.c_str(), best->name.c_str());
        }
    }
    return best;
}

}  // namespace dflash::common::test
