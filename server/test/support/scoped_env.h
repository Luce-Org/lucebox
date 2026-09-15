#pragma once

#include <cstdlib>
#include <string>

namespace luce_test {

// Temporarily set or unset an environment variable, then restore the exact
// process state on scope exit. This keeps formerly standalone tests isolated
// now that they share one test process.
class ScopedEnvVar {
public:
    ScopedEnvVar(const char * name, const char * value) : name_(name) {
        if (const char * previous = std::getenv(name)) {
            had_previous_value_ = true;
            previous_value_ = previous;
        }
        assign(value);
    }

    ~ScopedEnvVar() {
        assign(had_previous_value_ ? previous_value_.c_str() : nullptr);
    }

    ScopedEnvVar(const ScopedEnvVar &) = delete;
    ScopedEnvVar & operator=(const ScopedEnvVar &) = delete;

private:
    void assign(const char * value) const {
#if defined(_WIN32)
        _putenv_s(name_.c_str(), value ? value : "");
#else
        if (value) {
            setenv(name_.c_str(), value, 1);
        } else {
            unsetenv(name_.c_str());
        }
#endif
    }

    std::string name_;
    std::string previous_value_;
    bool had_previous_value_ = false;
};

} // namespace luce_test
