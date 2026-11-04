#pragma once

namespace luce::common {

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

// The process's ROCTX callbacks: set on HIP builds when LUCE_QWEN35_ROCTX is
// on and the library loaded, empty otherwise. Resolved once.
Qwen35RoctxCallbacks qwen35_roctx_callbacks();

// Whether LUCE_QWEN35_ROCTX markers are live, cached: with the variable unset
// a single-path marker costs this one check.
inline bool qwen35_roctx_enabled() {
    static const bool enabled = qwen35_roctx_callbacks().push != nullptr;
    return enabled;
}

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

// Single-request (non-paged) path markers. Names of its ranges:
inline constexpr const char * kQwen35RoctxPrefill = "qwen35.prefill";
inline constexpr const char * kQwen35RoctxSpecStep = "qwen35.spec.step";
inline constexpr const char * kQwen35RoctxDraft = "qwen35.spec.draft";
inline constexpr const char * kQwen35RoctxProject = "qwen35.spec.project";
inline constexpr const char * kQwen35RoctxSnapshot = "qwen35.spec.snapshot";
inline constexpr const char * kQwen35RoctxVerify = "qwen35.spec.verify";
inline constexpr const char * kQwen35RoctxRollback = "qwen35.spec.rollback";
inline constexpr const char * kQwen35RoctxReplay = "qwen35.spec.replay";
inline constexpr const char * kQwen35RoctxFeatureCopy = "qwen35.spec.feature_copy";

// A range for the lifetime of a scope, without metadata.
class Qwen35RoctxScope {
public:
    explicit Qwen35RoctxScope(const char * scope) {
        if (qwen35_roctx_enabled()) begin(scope, qwen35_roctx_callbacks());
    }
    Qwen35RoctxScope(const char * scope, bool enabled, Qwen35RoctxCallbacks callbacks) {
        if (enabled) begin(scope, callbacks);
    }
    ~Qwen35RoctxScope() {
        if (pop_) pop_();
    }

    Qwen35RoctxScope(const Qwen35RoctxScope &) = delete;
    Qwen35RoctxScope & operator=(const Qwen35RoctxScope &) = delete;

private:
    void begin(const char * scope, Qwen35RoctxCallbacks callbacks) {
        if (callbacks.push && callbacks.pop && callbacks.push(scope) >= 0) pop_ = callbacks.pop;
    }

    int (*pop_)() = nullptr;
};

// Phase ranges opened and closed at separate statements: the LUCE_TP_PROFILE
// start/add pairs of the single-request speculative loop, which nest LIFO.
// Whatever is still open when it goes out of scope (an early return between
// a pair) is closed then, so the ROCTX stack stays balanced.
class Qwen35RoctxPhases {
public:
    Qwen35RoctxPhases() : enabled_(qwen35_roctx_enabled()) {
        if (enabled_) callbacks_ = qwen35_roctx_callbacks();
    }
    Qwen35RoctxPhases(bool enabled, Qwen35RoctxCallbacks callbacks)
        : enabled_(enabled && callbacks.push && callbacks.pop), callbacks_(callbacks) {}
    ~Qwen35RoctxPhases() {
        while (open_ > 0) close();
    }

    Qwen35RoctxPhases(const Qwen35RoctxPhases &) = delete;
    Qwen35RoctxPhases & operator=(const Qwen35RoctxPhases &) = delete;

    void open(const char * phase) {
        if (!enabled_) return;
        if (callbacks_.push(phase) >= 0) ++open_;
    }
    void close() {
        if (!enabled_ || open_ <= 0) return;
        callbacks_.pop();
        --open_;
    }
    int open_count() const { return open_; }

private:
    bool enabled_ = false;
    Qwen35RoctxCallbacks callbacks_{};
    int open_ = 0;
};

} // namespace luce::common
