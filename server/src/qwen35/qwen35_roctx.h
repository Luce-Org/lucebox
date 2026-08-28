#pragma once

namespace dflash::common {

struct Qwen35RoctxMetadata {
    int live = -1;
    int bucket = -1;
    int prefill_tokens = -1;
    int prefill_segments = -1;
    int total_rows = -1;
    int max_kv_len = -1;
};

struct Qwen35RoctxCallbacks {
    int (*push)(const char * message) = nullptr;
    int (*pop)() = nullptr;
};

bool qwen35_roctx_env_enabled(const char * value);

class Qwen35RoctxRange {
public:
    Qwen35RoctxRange(const char * scope, const Qwen35RoctxMetadata & metadata);
    Qwen35RoctxRange(const char * scope, const Qwen35RoctxMetadata & metadata,
                     bool enabled, Qwen35RoctxCallbacks callbacks);
    ~Qwen35RoctxRange();

    Qwen35RoctxRange(const Qwen35RoctxRange &) = delete;
    Qwen35RoctxRange & operator=(const Qwen35RoctxRange &) = delete;

private:
    int (*pop_)() = nullptr;
    bool pushed_ = false;
};

} // namespace dflash::common
