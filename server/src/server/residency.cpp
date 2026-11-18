// Residency: releasing and reinstating the model's GPUs so models that share
// them take turns without restarting (docs/specs/model-swap.md).
//
// A client thread queues an operation and waits. The serving thread runs it
// once no request runs: admission stops while one is queued, so a release
// waits for the requests in flight, and requests that arrive meanwhile wait
// behind it. A request for a released model brings it back first; in a swap
// group that means asking the peers on its GPUs to release them.

#include "http_server.h"
#include "response_error.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace luce::common {

namespace {

json residency_to_json(const ModelBackend::Residency & r) {
    json gpus = json::array();
    for (size_t i = 0; i < r.gpus.size(); ++i) {
        const int g = r.gpus[i];
        gpus.push_back({
            {"gpu", g},
            {"bytes", i < r.bytes.size() ? r.bytes[i] : 0},
            {"released", std::find(r.released.begin(), r.released.end(), g) != r.released.end()},
            {"keepable", std::find(r.keepable.begin(), r.keepable.end(), g) != r.keepable.end()},
        });
    }
    const char * state = r.released.empty() ? "resident"
                       : r.released.size() >= r.gpus.size() ? "released" : "partial";
    json j = {{"supported", r.supported}, {"state", state}, {"gpus", gpus}};
    if (!r.supported) j["reason"] = r.reason;
    return j;
}

}  // namespace

void HttpServer::refresh_residency(const char * last_event, double seconds, const std::string & error) {
    const ModelBackend::Residency r = backend_.residency();
    residency_released_.store(!r.released.empty(), std::memory_order_release);
    json info = residency_to_json(r);
    info["swap_group"] = swap_group_ ? json(swap_group_->dir()) : json(nullptr);
    {
        std::lock_guard<std::mutex> lk(residency_info_mu_);
        const bool had = residency_info_.is_object();
        json last = had ? residency_info_.value("last", json::object()) : json::object();
        json counts = had ? residency_info_.value("counts", json::object()) : json::object();
        if (last_event) {
            last[last_event] = {{"seconds", seconds}, {"ok", error.empty()}};
            if (!error.empty()) last[last_event]["error"] = error;
            counts[last_event] = counts.value(last_event, 0) + 1;
        }
        info["last"] = std::move(last);
        info["counts"] = std::move(counts);
        residency_info_ = std::move(info);
    }
    rebudget_prefix_cache();
    if (swap_group_) {
        luce::server::SwapMember self;
#if defined(_WIN32)
        self.pid = _getpid();
#else
        self.pid = (int) getpid();
#endif
        self.port = config_.port;
        self.model = config_.model_name;
        self.gpus = r.gpus;
        self.released = r.released;
        self.keepable = r.keepable;
        self.bytes = r.bytes;
        if (!swap_group_->publish(self)) {
            std::fprintf(stderr, "[swap] could not write this member's file in %s\n", swap_group_->dir().c_str());
        }
    }
}

bool HttpServer::residency_release(const std::vector<int> & gpus, std::string & error) {
    if (!gpus.empty()) {
        // A release only ever reduces what the model holds: none of these GPUs
        // holds any of it, so there is nothing to do.
        luce::server::SwapMember held;
        const ModelBackend::Residency r = backend_.residency();
        held.gpus = r.gpus;
        held.released = r.released;
        if (luce::server::gpu_overlap(held.held(), gpus).empty()) return true;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = backend_.release_gpus(gpus, error);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    refresh_residency("release", s, ok ? std::string() : error);
    publish_memory_report();
    std::fprintf(stderr, "[swap] release %s in %.2f s%s%s\n", ok ? "done" : "failed", s,
                 ok ? "" : ": ", ok ? "" : error.c_str());
    return ok;
}

bool HttpServer::ensure_resident(std::string & error) {
    if (!residency_released_.load(std::memory_order_acquire)) return true;
    const auto t0 = std::chrono::steady_clock::now();
    bool locked_here = false;
    if (swap_group_ && !swap_group_->locked()) {
        // While another member swaps, answer its release requests: this
        // server may hold a GPU it needs (the part it kept).
        auto on_wait = [this]() {
            service_residency_ops(/*releases_only=*/true);
            return !stopping_.load();
        };
        if (!swap_group_->lock(on_wait, luce::server::kSwapLockTimeoutS, error)) {
            refresh_residency("reinstate", 0.0, error);
            return false;
        }
        locked_here = true;
    }
    bool ok = true;
    if (swap_group_) {
        // The GPUs this model must load onto, and the peers holding them.
        const ModelBackend::Residency mine = backend_.residency();
        for (const luce::server::SwapMember & peer : swap_group_->peers()) {
            const std::vector<int> overlap = luce::server::gpu_overlap(peer.held(), mine.released);
            if (overlap.empty()) continue;
            std::fprintf(stderr, "[swap] asking %s (port %d) to release GPU", peer.model.c_str(), peer.port);
            for (int g : overlap) std::fprintf(stderr, " %d", g);
            std::fprintf(stderr, "\n");
            if (!luce::server::SwapGroup::ask_release(peer, overlap, luce::server::kSwapReleaseTimeoutS, error)) {
                ok = false;
                break;
            }
        }
    }
    if (ok) ok = backend_.reinstate(error);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    refresh_residency("reinstate", s, ok ? std::string() : error);
    if (locked_here) swap_group_->unlock();
    publish_memory_report();
    std::fprintf(stderr, "[swap] reinstate %s in %.2f s%s%s\n", ok ? "done" : "failed", s,
                 ok ? "" : ": ", ok ? "" : error.c_str());
    return ok;
}

void HttpServer::service_residency_ops(bool releases_only) {
    while (true) {
        ResidencyOp * op = nullptr;
        {
            std::lock_guard<std::mutex> lk(queue_mu_);
            ResidencyOp ** link = &residency_head_;
            while (*link && releases_only && !(*link)->release) link = &(*link)->next;
            op = *link;
            if (!op) return;
            unlink_residency_op_locked(link);
            op->running = true;   // its client now waits for it, even at shutdown
        }
        const auto t0 = std::chrono::steady_clock::now();
        std::string error;
        const bool ok = op->release ? residency_release(op->gpus, error) : ensure_resident(error);
        {
            std::lock_guard<std::mutex> lk(queue_mu_);
            op->ok = ok;
            op->error = std::move(error);
            op->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            op->done = true;
        }
        residency_cv_.notify_all();
    }
}

bool HttpServer::run_residency_op(ResidencyOp & op) {
    std::unique_lock<std::mutex> lk(queue_mu_);
    if (stopping_.load()) {
        op.error = "server stopping";
        return false;
    }
    op.next = nullptr;
    if (residency_tail_) residency_tail_->next = &op;
    else residency_head_ = &op;
    residency_tail_ = &op;
    queue_cv_.notify_all();
    // At shutdown an operation still queued gives up; one the serving thread
    // has started runs to the end, since that thread writes its result here.
    residency_cv_.wait(lk, [&] { return op.done || (stopping_.load() && !op.running); });
    if (!op.done) {
        // Still queued at shutdown: take it out so nothing points at it.
        for (ResidencyOp ** p = &residency_head_; *p; p = &(*p)->next) {
            if (*p == &op) {
                unlink_residency_op_locked(p);
                break;
            }
        }
        op.error = "server stopping";
        return false;
    }
    return op.ok;
}

void HttpServer::unlink_residency_op_locked(ResidencyOp ** link) {
    ResidencyOp * op = *link;
    *link = op->next;
    if (residency_tail_ == op) {
        residency_tail_ = nullptr;
        for (ResidencyOp * q = residency_head_; q; q = q->next) residency_tail_ = q;
    }
}

void HttpServer::cancel_queued_residency_ops() {
    // stopping_ is set: taking the lock first means no client is between its
    // check of stopping_ and its wait, so none misses this wake-up.
    { std::lock_guard<std::mutex> lk(queue_mu_); }
    residency_cv_.notify_all();
}

void HttpServer::handle_residency_request(SocketHandle fd, const std::string & method, const std::string & body) {
    if (!models_.empty()) {
        send_error(fd, 409, "residency is per server; this one serves several models");
        return;
    }
    if (method == "GET") {
        json info;
        {
            std::lock_guard<std::mutex> lk(residency_info_mu_);
            info = residency_info_;
        }
        send_response(fd, 200, "application/json", info.dump() + "\n");
        return;
    }
    json req = json::parse(body.empty() ? std::string("{}") : body, nullptr, /*allow_exceptions=*/false);
    if (req.is_discarded() || !req.is_object()) {
        send_error(fd, 400, "expected a JSON object: {\"state\": \"released\" | \"resident\", \"gpus\": [..]}");
        return;
    }
    if (req.contains("state") && !req["state"].is_string()) {
        send_error(fd, 400, "\"state\" must be \"released\" or \"resident\"");
        return;
    }
    const std::string state = req.value("state", std::string());
    ResidencyOp op;
    if (state == "released") {
        op.release = true;
        if (req.contains("gpus")) {
            if (!req["gpus"].is_array()) {
                send_error(fd, 400, "\"gpus\" must be an array of GPU ids");
                return;
            }
            for (const json & g : req["gpus"]) {
                if (!g.is_number_integer() || g.get<long long>() < 0 || g.get<long long>() > INT_MAX) {
                    send_error(fd, 400, "\"gpus\" must be an array of GPU ids");
                    return;
                }
                op.gpus.push_back(g.get<int>());
            }
        }
    } else if (state != "resident") {
        send_error(fd, 400, "\"state\" must be \"released\" or \"resident\"");
        return;
    }
    const bool ok = run_residency_op(op);
    json out;
    {
        std::lock_guard<std::mutex> lk(residency_info_mu_);
        out = residency_info_;
    }
    out["seconds"] = op.seconds;
    if (!ok) out["error"] = op.error;
    send_response(fd, ok ? 200 : 503, "application/json", out.dump() + "\n");
}

void HttpServer::rebudget_prefix_cache() {
    if (!swap_group_ || !config_.prefix_cache_budget_auto || prefix_cache_.disabled()) return;
    // Sized by someone else since the last call (start-up, or the concurrent scheduler once it can size a
    // checkpoint): that is the size without the group.
    if (prefix_cache_.max_resident_bytes() != prefix_budget_applied_) {
        prefix_budget_cap_ = prefix_cache_.max_resident_bytes();
    }
    if (prefix_budget_cap_ == 0) return;   // unlimited: nothing to follow
    // A quarter of what is available, as at start-up, counting what this cache holds as available.
    const size_t memory = budgetable_memory_bytes() + prefix_cache_.resident_bytes();
    size_t bytes = std::min(prefix_budget_cap_, memory / 4);
    const size_t current = prefix_cache_.max_resident_bytes();
    const size_t step = std::max<size_t>((size_t) 256 << 20, current / 20);
    // Small moves of the available memory leave the budget where it is, unless the cache is over the new one.
    if ((bytes > current ? bytes - current : current - bytes) < step && prefix_cache_.resident_bytes() <= bytes) {
        bytes = current;
    }
    prefix_budget_applied_ = bytes;
    if (bytes == current) return;
    std::fprintf(stderr, "[swap] RAM prefix budget %zu -> %zu MiB (%zu MiB held)\n",
                 prefix_cache_.max_resident_bytes() >> 20, bytes >> 20, prefix_cache_.resident_bytes() >> 20);
    prefix_cache_.set_max_resident_bytes(bytes);
    // Over it: evict the least recently used, to the SSD session cache when it is on.
    if (prefix_cache_.resident_bytes() > bytes) kv_cache_.enforce_budget(-1);
}

void HttpServer::fail_unstarted_job(ServerJob * job, const std::string & message) {
    std::fprintf(stderr, "[server] request failed: %s\n", message.c_str());
    const ResponseError error = ResponseError::unavailable("unavailable", message);
    const json body = build_error_response(job->req.format, error, job->req.response_id);
    send_response(job->fd, response_error_http_status(error), "application/json", body.dump() + "\n");
    std::lock_guard<std::mutex> lk(job->mu);
    job->done = true;
    job->cv.notify_one();
}

void HttpServer::attach_swap_group(std::unique_ptr<luce::server::SwapGroup> group) {
    swap_group_ = std::move(group);
    if (config_.swap_start_released) {
        // Keep only what never conflicts (a split model's expert stacks). The
        // serving thread runs it once its loop has started, which sizes the
        // prefix checkpoints from the live caches first.
        const ModelBackend::Residency r = backend_.residency();
        if (!r.supported) {
            std::fprintf(stderr, "[swap] --swap-start released ignored: %s\n", r.reason.c_str());
        } else {
            startup_release_ = std::make_unique<ResidencyOp>();
            startup_release_->release = true;
            for (int g : r.gpus) {
                if (std::find(r.keepable.begin(), r.keepable.end(), g) == r.keepable.end()) {
                    startup_release_->gpus.push_back(g);
                }
            }
            std::lock_guard<std::mutex> lk(queue_mu_);
            startup_release_->next = residency_head_;
            residency_head_ = startup_release_.get();
            if (!residency_tail_) residency_tail_ = residency_head_;
        }
    }
    refresh_residency();
    if (swap_group_) swap_group_->unlock();
}

}  // namespace luce::common
