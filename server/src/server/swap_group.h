// Swap groups: several luce_server processes that share GPUs take turns on them.
//
// Each member writes <dir>/<port>.json (what it holds on which GPU) and takes
// <dir>/swap.lock (flock) for every load or reinstate. A member that needs a
// GPU asks the peers that hold memory on it to release it, over their
// POST /v1/residency, then loads. See docs/specs/model-swap.md.

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace luce::server {

struct SwapMember {
    int pid  = 0;
    int port = 0;
    std::string model;
    std::vector<int> gpus;        // every GPU it uses
    std::vector<int> released;    // GPUs it holds nothing on
    std::vector<int> keepable;    // GPUs it can keep while releasing the others
    std::vector<size_t> bytes;    // per entry of `gpus`

    // The GPUs it holds memory on now.
    std::vector<int> held() const;
};

class SwapGroup {
public:
    // Creates `dir` if needed. Null with `error` set when it cannot.
    static std::unique_ptr<SwapGroup> open(const std::string & dir, int port, std::string & error);
    ~SwapGroup();

    SwapGroup(const SwapGroup &) = delete;
    SwapGroup & operator=(const SwapGroup &) = delete;

    const std::string & dir() const { return dir_; }
    int port() const { return port_; }

    // Writes this member's file (atomically: a temporary file, then rename).
    bool publish(const SwapMember & self);
    // The other members whose process is alive. Files of dead processes are removed.
    std::vector<SwapMember> peers() const;

    // Takes the group lock. While it waits, `on_wait` runs every 20 ms (a
    // member answers its peers' release requests meanwhile); it returns false
    // to stop waiting. False when the wait stopped or timed out.
    bool lock(const std::function<bool()> & on_wait, int timeout_s, std::string & error);
    void unlock();
    bool locked() const { return lock_fd_ >= 0; }

    // Asks `peer` to release `gpus` and waits for its answer. False with
    // `error` set when the peer refused or could not be reached.
    static bool ask_release(const SwapMember & peer, const std::vector<int> & gpus, int timeout_s,
                            std::string & error);

private:
    SwapGroup(std::string dir, int port) : dir_(std::move(dir)), port_(port) {}
    std::string member_path(int port) const;

    std::string dir_;
    int port_ = 0;
    int lock_fd_ = -1;
};

// How long a member waits for the group lock, and for a peer to release its
// GPUs (the peer finishes its running requests first).
constexpr int kSwapLockTimeoutS    = 1800;
constexpr int kSwapReleaseTimeoutS = 900;

// POST `body` (JSON) to http://127.0.0.1:<port><path>. Returns the HTTP status,
// or 0 when the connection failed or timed out; `response` gets the body.
int local_http_post(int port, const std::string & path, const std::string & body,
                    std::string & response, int timeout_s);

// The GPUs in `a` that are also in `b`.
std::vector<int> gpu_overlap(const std::vector<int> & a, const std::vector<int> & b);

}  // namespace luce::server
