// host_stream_flag.h - a host-written word a GPU stream can wait on.
//
// Lets the host queue GPU work that depends on data a host thread is still
// producing: the stream waits (on the command processor, no kernel) until the
// word reaches a value, and the producing thread raises it when the data is
// in place. HIP only; elsewhere init() fails and callers keep their blocking
// path.
#pragma once

#include <cstddef>
#include <cstdint>

namespace luce::common {

class HostStreamFlag {
public:
    HostStreamFlag() = default;
    HostStreamFlag(const HostStreamFlag &) = delete;
    HostStreamFlag & operator=(const HostStreamFlag &) = delete;
    ~HostStreamFlag();

    bool init();
    bool ready() const { return host_ != nullptr; }
    // Queue on `stream` a wait until the word equals `value`.
    bool enqueue_wait(void * stream, uint32_t value) const;
    // Raise the word to `value` (release: earlier host writes are visible
    // to the GPU work released by it).
    void signal(uint32_t value) const;
    // Queue an asynchronous host-to-device copy on `stream` (the source must
    // be pinned and stay valid until the stream ran the copy).
    static bool copy_to_device(void * stream, void * dst, const void * src, size_t bytes);

private:
    uint32_t * host_ = nullptr;
    uint32_t * dev_ = nullptr;
};

}  // namespace luce::common
