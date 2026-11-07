// Whether KVFlash sessions are kept on disk, and where: --kv-sessions on|off, the disk cache directory, and the old
// LUCE_KVFLASH_SLOT_BUNDLE_DIR variable, which stays one release as an override. Pure: the server turns the answer into
// a log line and an environment variable, the engine reads the environment.
#pragma once

#include <string>

namespace luce::common {

enum class KvFlashSessionsFlag { automatic, on, off };   // automatic: on whenever KVFlash and a disk dir are on

// "on" or "off"; anything else leaves `out` as it was.
inline bool kvflash_sessions_flag_parse(const std::string & v, KvFlashSessionsFlag & out) {
    if (v == "on") { out = KvFlashSessionsFlag::on; return true; }
    if (v == "off") { out = KvFlashSessionsFlag::off; return true; }
    return false;
}

struct KvFlashSessionsPlan {
    bool on = false;
    std::string root;   // the directory whose sessions/ holds them
    std::string note;   // one line for the log, "" when there is nothing to say
};

inline KvFlashSessionsPlan kvflash_sessions_plan(bool kvflash_on, KvFlashSessionsFlag flag, const std::string & cache_dir, const std::string & override_dir) {
    KvFlashSessionsPlan p;
    if (!kvflash_on) return p;   // no slots to park
    if (flag == KvFlashSessionsFlag::off) {
        p.note = "sessions are not kept on disk (--kv-sessions off)";
        return p;
    }
    if (!override_dir.empty()) {
        p.on = true;
        p.root = override_dir;
        p.note = "LUCE_KVFLASH_SLOT_BUNDLE_DIR is deprecated and goes next release: sessions go to " + override_dir +
                 "/sessions/, outside the one cache budget; use --kv-cache-dir (and --kv-sessions on|off)";
        return p;
    }
    if (cache_dir.empty()) {
        if (flag == KvFlashSessionsFlag::on) p.note = "--kv-sessions on needs a disk cache: pass --kv-cache-dir <path>";
        return p;
    }
    p.on = true;
    p.root = cache_dir;
    return p;
}

}  // namespace luce::common
