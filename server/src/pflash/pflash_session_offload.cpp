#include "pflash_session_offload.h"

#include "pflash_scoring_resume.h"

#include <sys/stat.h>

#include <algorithm>
#include <cstring>
#include <sstream>

namespace luce::pflash {

PFlashHostBlob & PFlashHostBlob::operator=(PFlashHostBlob && o) noexcept {
    if (this != &o) {
        reset();
        buf_ = o.buf_;
        vec_ = std::move(o.vec_);
        data_ = buf_ ? o.data_ : vec_.data();
        capacity_ = o.capacity_;
        o.buf_ = nullptr;
        o.vec_.clear();
        o.data_ = nullptr;
        o.capacity_ = 0;
    }
    return *this;
}

void PFlashHostBlob::reset() {
    if (buf_) ggml_backend_buffer_free(buf_);
    buf_ = nullptr;
    std::vector<uint8_t>().swap(vec_);
    data_ = nullptr;
    capacity_ = 0;
}

bool PFlashHostBlob::reserve(size_t size, ggml_backend_buffer_type_t pinned) {
    if (size <= capacity_ && data_) return true;
    reset();
    if (size == 0) return true;
    if (pinned) {
        buf_ = ggml_backend_buft_alloc_buffer(pinned, size);
        if (buf_ && ggml_backend_buffer_is_host(buf_)) {
            data_ = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buf_));
            capacity_ = size;
            return true;
        }
        if (buf_) ggml_backend_buffer_free(buf_);
        buf_ = nullptr;
    }
    try {
        vec_.resize(size);
    } catch (...) {
        vec_.clear();
        return false;
    }
    data_ = vec_.data();
    capacity_ = size;
    return true;
}

std::vector<ggml_tensor *> pflash_owned_tensors(ggml_context * ctx) {
    std::vector<ggml_tensor *> out;
    if (!ctx) return out;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (t->view_src == nullptr && t->buffer != nullptr && t->data != nullptr) out.push_back(t);
    }
    return out;
}

bool pflash_token_chunks(const ggml_tensor * t, int token_dim, int64_t n_tokens,
                         std::vector<std::pair<size_t, size_t>> & chunks, std::string & error) {
    chunks.clear();
    if (!ggml_is_contiguous(t)) {
        error = "tensor is not contiguous";
        return false;
    }
    if (token_dim < 0) {
        chunks.push_back({0, ggml_nbytes(t)});
        return true;
    }
    if (token_dim >= GGML_MAX_DIMS) {
        error = "token axis out of range";
        return false;
    }
    const int64_t n = std::max<int64_t>(0, std::min<int64_t>(n_tokens, t->ne[token_dim]));
    const size_t run = t->nb[token_dim] * (size_t) n;
    if (run == 0) return true;
    // Every index of the axes above the token axis holds one contiguous run of the first n positions.
    int64_t hi[GGML_MAX_DIMS] = {1, 1, 1, 1};
    for (int d = token_dim + 1; d < GGML_MAX_DIMS; ++d) hi[d] = t->ne[d];
    for (int64_t i3 = 0; i3 < hi[3]; ++i3) {
        for (int64_t i2 = 0; i2 < hi[2]; ++i2) {
            for (int64_t i1 = 0; i1 < hi[1]; ++i1) {
                size_t off = 0;
                if (token_dim < 1) off += (size_t) i1 * t->nb[1];
                if (token_dim < 2) off += (size_t) i2 * t->nb[2];
                off += (size_t) i3 * t->nb[3];
                chunks.push_back({off, run});
            }
        }
    }
    return true;
}

bool pflash_copy_to_host(const std::vector<ggml_tensor *> & tensors, const std::vector<int> & token_dims,
                         int64_t n_tokens, ggml_backend_buffer_type_t pinned, PFlashHostCopy & out,
                         std::string & error) {
    out.tensors.clear();
    out.bytes = 0;
    if (token_dims.size() != tensors.size()) {
        error = "token axes do not match the tensors";
        return false;
    }
    std::vector<PFlashTensorCopy> copies(tensors.size());
    size_t total = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const ggml_tensor * t = tensors[i];
        if (!t || !t->buffer || !t->data || t->view_src) {
            error = "tensor " + std::to_string(i) + " is not an allocated owned tensor";
            return false;
        }
        PFlashTensorCopy & c = copies[i];
        c.type = t->type;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            c.ne[d] = t->ne[d];
            c.nb[d] = t->nb[d];
        }
        c.token_dim = token_dims[i];
        c.n_tokens = token_dims[i] < 0 ? 0 : std::min<int64_t>(std::max<int64_t>(n_tokens, 0), t->ne[token_dims[i]]);
        std::string why;
        if (!pflash_token_chunks(t, token_dims[i], n_tokens, c.chunks, why)) {
            error = "tensor " + std::to_string(i) + ": " + why;
            return false;
        }
        for (const auto & ch : c.chunks) total += ch.second;
    }
    if (!out.blob.reserve(total, pinned)) {
        error = "host allocation of " + std::to_string(total) + " bytes failed";
        return false;
    }
    size_t pos = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        for (const auto & ch : copies[i].chunks) {
            ggml_backend_tensor_get(tensors[i], out.blob.data() + pos, ch.first, ch.second);
            pos += ch.second;
        }
    }
    out.tensors = std::move(copies);
    out.bytes = total;
    return true;
}

bool pflash_copy_from_host(const PFlashHostCopy & host, const std::vector<ggml_tensor *> & tensors,
                           std::string & error) {
    if (host.tensors.size() != tensors.size()) {
        error = "tensor count " + std::to_string(tensors.size()) + " != stored " + std::to_string(host.tensors.size());
        return false;
    }
    std::vector<std::vector<std::pair<size_t, size_t>>> targets(tensors.size());
    size_t total = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const ggml_tensor * t = tensors[i];
        const PFlashTensorCopy & h = host.tensors[i];
        const std::string where = "tensor " + std::to_string(i);
        if (!t || !t->buffer || !t->data || t->view_src) {
            error = where + " is not an allocated owned tensor";
            return false;
        }
        if (h.type != t->type) {
            error = where + " type differs";
            return false;
        }
        const bool grow = h.token_dim >= 0 && h.token_dim < GGML_MAX_DIMS;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            const bool same = grow && d == h.token_dim ? t->ne[d] >= h.ne[d] : t->ne[d] == h.ne[d];
            const bool same_nb = grow && d > h.token_dim ? true : t->nb[d] == h.nb[d];
            if (!same || !same_nb) {
                error = where + " shape differs";
                return false;
            }
        }
        if (grow) {
            std::string why;
            if (!pflash_token_chunks(t, h.token_dim, h.n_tokens, targets[i], why)) {
                error = where + ": " + why;
                return false;
            }
            if (targets[i].size() != h.chunks.size()) {
                error = where + " chunk layout differs";
                return false;
            }
            for (size_t k = 0; k < h.chunks.size(); ++k) {
                if (targets[i][k].second != h.chunks[k].second) {
                    error = where + " chunk size differs";
                    return false;
                }
            }
        } else {
            targets[i] = h.chunks;
        }
        for (const auto & ch : targets[i]) {
            if (ch.first > ggml_nbytes(t) || ch.second > ggml_nbytes(t) - ch.first) {
                error = where + " chunk out of bounds";
                return false;
            }
            total += ch.second;
        }
    }
    if (total != host.bytes || total > host.blob.capacity() || (total > 0 && !host.blob.data())) {
        error = "host copy size " + std::to_string(host.bytes) + " does not match its chunks (" +
                std::to_string(total) + ")";
        return false;
    }
    size_t pos = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        for (const auto & ch : targets[i]) {
            ggml_backend_tensor_set(tensors[i], host.blob.data() + pos, ch.first, ch.second);
            pos += ch.second;
        }
    }
    return true;
}

int pflash_pick_restore(const std::vector<PFlashStoredPrompt> & stored, const std::vector<int32_t> & ids) {
    const int S = (int) ids.size();
    int best = -1, best_r = 0;
    for (size_t i = 0; i < stored.size(); ++i) {
        if (!stored[i].ids || stored[i].ids->empty()) continue;
        const std::vector<int32_t> & a = *stored[i].ids;
        const size_t n = std::min(a.size(), ids.size());
        const int shared = (int) (std::mismatch(a.begin(), a.begin() + (long) n, ids.begin()).first - a.begin());
        const int r = pflash_scoring_resume(stored[i].checkpoint, shared, S, S);
        if (r > best_r) {
            best_r = r;
            best = (int) i;
        }
    }
    return best;
}

PFlashOffloadPlan pflash_plan_offload(const std::vector<PFlashOffloadEntry> & entries,
                                      const std::vector<PFlashOffloadDevice> & devices, int limit) {
    PFlashOffloadPlan plan;
    struct Item { bool device; size_t index; uint64_t id; uint64_t last_used; };
    std::vector<Item> items;
    std::vector<uint64_t> kept_by_device;
    for (size_t i = 0; i < devices.size(); ++i) {
        const PFlashOffloadDevice & d = devices[i];
        if (!d.has_prompt) continue;
        const bool entry_exists = d.host_id != 0 &&
            std::any_of(entries.begin(), entries.end(), [&](const PFlashOffloadEntry & e) { return e.id == d.host_id; });
        if (d.host_valid && entry_exists) {
            items.push_back({false, 0, d.host_id, d.last_used});
            kept_by_device.push_back(d.host_id);
        } else {
            items.push_back({true, i, 0, d.last_used});
        }
    }
    for (const PFlashOffloadEntry & e : entries) {
        if (e.on_device) continue;   // its session decides above (kept when unchanged, else replaced or gone)
        if (std::find(kept_by_device.begin(), kept_by_device.end(), e.id) != kept_by_device.end()) continue;
        items.push_back({false, 0, e.id, e.last_used});
    }
    std::stable_sort(items.begin(), items.end(), [](const Item & a, const Item & b) {
        return a.last_used > b.last_used;
    });
    if (limit < 0) limit = 0;
    if ((int) items.size() > limit) items.resize((size_t) limit);
    for (const Item & it : items) {
        if (it.device) plan.copy_devices.push_back(it.index);
        else plan.keep_entries.push_back(it.id);
    }
    for (const PFlashOffloadEntry & e : entries) {
        if (std::find(plan.keep_entries.begin(), plan.keep_entries.end(), e.id) == plan.keep_entries.end()) {
            plan.drop_entries.push_back(e.id);
        }
    }
    return plan;
}

std::string pflash_file_identity(const std::string & path) {
    if (path.empty()) return "-";
    struct stat s {};
    if (::stat(path.c_str(), &s) != 0) return path + "|missing";
#ifdef _WIN32
    // MSVC's struct stat has no st_mtim; whole seconds are enough, the identity is only compared on one machine.
    const long long mtime_ns = (long long) s.st_mtime * 1000000000LL;
#else
    const long long mtime_ns = (long long) s.st_mtim.tv_sec * 1000000000LL + (long long) s.st_mtim.tv_nsec;
#endif
    std::ostringstream out;
    out << path << '|' << (unsigned long long) s.st_dev << ':' << (unsigned long long) s.st_ino << '|'
        << (long long) s.st_size << '|' << mtime_ns;
    return out.str();
}

std::string pflash_identity_mismatch(const std::string & stored, const std::string & current) {
    if (stored == current) return "";
    std::istringstream a(stored), b(current);
    std::string la, lb;
    while (true) {
        const bool ga = (bool) std::getline(a, la);
        const bool gb = (bool) std::getline(b, lb);
        if (!ga && !gb) return "identity differs";
        if (!ga) return lb.substr(0, lb.find('=')) + " added";
        if (!gb) return la.substr(0, la.find('=')) + " removed";
        if (la != lb) return la.substr(0, la.find('=')) + " differs";
    }
}

} // namespace luce::pflash
