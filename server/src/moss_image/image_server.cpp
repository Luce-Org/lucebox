#include "image_server.h"

#include "png.h"

#include <nlohmann/json.hpp>

#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <csignal>
#include <cstring>
#include <ctime>
#include <limits>
#include <mutex>
#include <random>
#include <system_error>
#include <thread>

namespace luce::moss_image {

namespace {

using json = nlohmann::json;

// A prompt is at most kMaxPrompt bytes, about the text encoder's 4,096 tokens: it is tokenized while the GPU waits,
// in a time that grows with the square of its longest word.
constexpr size_t kMaxPrompt   = 16 * 1024;
constexpr size_t kMaxHeader   = 64 * 1024;
constexpr size_t kMaxBody     = 1u << 20;  // JSON with two prompts at most
constexpr int    kIdleMs      = 30000;     // a client that sends or reads nothing for this long is dropped
constexpr int    kMaxImages   = 8;         // sd-server's batch_count cap
const char *     kExtraOpen   = "<sd_cpp_extra_args>";
const char *     kExtraClose  = "</sd_cpp_extra_args>";

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
};

// A request the client got wrong: answered with this message and status.
struct BadRequest {
    std::string message;
    int         status = 400;
};

bool wait_readable(const int fd) {
    pollfd pfd{ fd, POLLIN, 0 };
    int    ready = 0;
    do {
        ready = poll(&pfd, 1, kIdleMs);
    } while (ready < 0 && errno == EINTR);
    return ready > 0 && (pfd.revents & (POLLIN | POLLHUP)) != 0;
}

bool send_all(const int fd, const std::string & data) {
    size_t sent = 0;
    while (sent < data.size()) {
        const ssize_t n = send(fd, data.data() + sent, data.size() - sent, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        sent += size_t(n);
    }
    return true;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return s;
}

// False when the client sent no request in time or went away; throws BadRequest for one the server does not read.
bool read_request(const int fd, HttpRequest & req) {
    std::string buf;
    char        chunk[16384];
    size_t      header_end = std::string::npos;
    while (header_end == std::string::npos) {
        if (!wait_readable(fd)) {
            return false;
        }
        const ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) {
            return false;
        }
        buf.append(chunk, size_t(n));
        header_end = buf.find("\r\n\r\n");
        if (std::min(header_end, buf.size()) > kMaxHeader) {
            throw BadRequest{ "the request headers take more than " + std::to_string(kMaxHeader) + " bytes", 431 };
        }
    }
    const std::string head = lower(buf.substr(0, header_end));
    const size_t      sp1  = buf.find(' ');
    const size_t      sp2  = sp1 == std::string::npos ? sp1 : buf.find(' ', sp1 + 1);
    if (sp2 == std::string::npos || sp2 > header_end) {
        return false;
    }
    req.method = buf.substr(0, sp1);
    req.path   = buf.substr(sp1 + 1, sp2 - sp1 - 1);
    req.path   = req.path.substr(0, req.path.find('?'));

    if (head.find("\r\ntransfer-encoding:") != std::string::npos) {
        throw BadRequest{ "send the body with a Content-Length, not chunked", 411 };
    }
    size_t       length = 0;
    const size_t cl     = head.find("\r\ncontent-length:");
    if (cl != std::string::npos) {
        length = size_t(std::strtoull(head.c_str() + cl + 17, nullptr, 10));
    }
    if (length > kMaxBody) {
        throw BadRequest{ "the request body takes more than " + std::to_string(kMaxBody) + " bytes", 413 };
    }
    if (head.find("\r\nexpect: 100-continue") != std::string::npos &&
        !send_all(fd, "HTTP/1.1 100 Continue\r\n\r\n")) {
        return false;
    }
    req.body = buf.substr(header_end + 4);
    while (req.body.size() < length) {
        if (!wait_readable(fd)) {
            return false;
        }
        const ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) {
            return false;
        }
        req.body.append(chunk, size_t(n));
    }
    req.body.resize(length);
    return true;
}

// Closes once the client has read the answer and closed its side, or after 2 s. A request refused before its body was
// read still has that body on the way, and closing with unread data would reset the connection under the answer.
void close_after_answer(const int fd) {
    shutdown(fd, SHUT_WR);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    char       sink[16384];
    for (;;) {
        const auto left =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        pollfd pfd{ fd, POLLIN, 0 };
        if (left <= 0 || poll(&pfd, 1, int(left)) <= 0 || recv(fd, sink, sizeof(sink), 0) <= 0) {
            break;
        }
    }
    close(fd);
}

void respond(const int fd, const int status, const std::string & body,
             const char * content_type = "application/json") {
    const char * reason = status == 200   ? "OK"
                          : status == 204 ? "No Content"
                          : status == 400 ? "Bad Request"
                          : status == 404 ? "Not Found"
                          : status == 405 ? "Method Not Allowed"
                          : status == 411 ? "Length Required"
                          : status == 413 ? "Content Too Large"
                          : status == 431 ? "Request Header Fields Too Large"
                                          : "Internal Server Error";
    std::string out = "HTTP/1.1 " + std::to_string(status) + " " + reason + "\r\n";
    out += "Access-Control-Allow-Origin: *\r\n"
           "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
           "Access-Control-Allow-Headers: *\r\n";
    out += std::string("Content-Type: ") + content_type + "\r\n";
    out += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    out += "Connection: close\r\n\r\n";
    out += body;
    send_all(fd, out);
}

std::string base64(const std::string & in) {
    static const char * table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string         out((in.size() + 2) / 3 * 4, '=');
    size_t              o = 0;
    size_t              i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const uint32_t v = uint32_t(uint8_t(in[i])) << 16 | uint32_t(uint8_t(in[i + 1])) << 8 | uint8_t(in[i + 2]);
        out[o++]         = table[v >> 18];
        out[o++]         = table[(v >> 12) & 63];
        out[o++]         = table[(v >> 6) & 63];
        out[o++]         = table[v & 63];
    }
    if (i < in.size()) {
        const bool     two = i + 1 < in.size();
        const uint32_t v   = uint32_t(uint8_t(in[i])) << 16 | (two ? uint32_t(uint8_t(in[i + 1])) << 8 : 0);
        out[o++]           = table[v >> 18];
        out[o++]           = table[(v >> 12) & 63];
        if (two) {
            out[o] = table[(v >> 6) & 63];
        }
    }
    return out;
}

SampleMethod parse_sample_method(std::string name) {
    name = lower(name);
    if (name == "euler" || name == "k_euler") {
        return SampleMethod::euler;
    }
    if (name == "er_sde") {
        return SampleMethod::er_sde;
    }
    throw BadRequest{ "sample method " + name + " is not supported (euler, er_sde)" };
}

Scheduler parse_scheduler(std::string name) {
    name = lower(name);
    if (name == "discrete") {
        return Scheduler::discrete;
    }
    if (name == "simple") {
        return Scheduler::simple;
    }
    throw BadRequest{ "scheduler " + name + " is not supported (discrete, simple)" };
}

const char * name_of(const SampleMethod m) {
    return m == SampleMethod::euler ? "euler" : "er_sde";
}

const char * name_of(const Scheduler s) {
    return s == Scheduler::discrete ? "discrete" : "simple";
}

// sd-server reads only values of the expected JSON type and passes over the rest; a number the field cannot hold
// is the client's error.
template <typename T> void read_number(const json & j, const char * key, T & out) {
    if (!j.contains(key)) {
        return;
    }
    const json & v = j[key];
    bool         fits;
    if constexpr (std::is_integral_v<T>) {
        if (!v.is_number_integer()) {
            return;
        }
        fits = v.is_number_unsigned() ? v.get<uint64_t>() <= uint64_t(std::numeric_limits<T>::max())
                                      : v.get<int64_t>() >= int64_t(std::numeric_limits<T>::min()) &&
                                            v.get<int64_t>() <= int64_t(std::numeric_limits<T>::max());
    } else {
        if (!v.is_number()) {
            return;
        }
        fits = std::fabs(v.get<double>()) <= double(std::numeric_limits<T>::max());
    }
    if (!fits) {
        throw BadRequest{ std::string(key) + " is out of range" };
    }
    out = v.get<T>();
}

void read_string(const json & j, const char * key, std::string & out) {
    if (j.contains(key) && j[key].is_string()) {
        out = j[key].get<std::string>();
    }
}

// The JSON between the first <sd_cpp_extra_args> pair; every pair is cut from the prompt.
std::string take_extra_args(std::string & prompt) {
    std::string extra;
    bool        first = true;
    for (size_t open = prompt.find(kExtraOpen); open != std::string::npos; open = prompt.find(kExtraOpen, open)) {
        const size_t body  = open + std::strlen(kExtraOpen);
        const size_t close = prompt.find(kExtraClose, body);
        if (close == std::string::npos) {
            break;
        }
        if (first) {
            extra = prompt.substr(body, close - body);
            first = false;
        }
        prompt.erase(open, close + std::strlen(kExtraClose) - open);
    }
    return extra;
}

void apply_extra_args(const std::string & text, ImageRequest & r, int & n) {
    const json j = json::parse(text, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        throw BadRequest{ "invalid sd_cpp_extra_args" };
    }
    read_string(j, "prompt", r.prompt);
    read_string(j, "negative_prompt", r.negative_prompt);
    read_number(j, "width", r.width);
    read_number(j, "height", r.height);
    read_number(j, "batch_count", n);
    read_number(j, "seed", r.seed);
    if (j.contains("sample_params") && j["sample_params"].is_object()) {
        const json & s = j["sample_params"];
        read_number(s, "sample_steps", r.steps);
        read_number(s, "eta", r.eta);
        read_number(s, "flow_shift", r.flow_shift);
        if (s.contains("sample_method") && s["sample_method"].is_string()) {
            r.sample_method = parse_sample_method(s["sample_method"].get<std::string>());
        }
        if (s.contains("scheduler") && s["scheduler"].is_string()) {
            r.scheduler = parse_scheduler(s["scheduler"].get<std::string>());
        }
        if (s.contains("guidance") && s["guidance"].is_object()) {
            read_number(s["guidance"], "txt_cfg", r.cfg_scale);
        }
    }
}

// sd-server's clamps, plus everything the pipeline checks, so that it never refuses a request the server took.
void validate(ImageRequest & r, int & n) {
    if (r.prompt.empty()) {
        throw BadRequest{ "prompt required" };
    }
    if (r.prompt.size() > kMaxPrompt || r.negative_prompt.size() > kMaxPrompt) {
        throw BadRequest{ "the prompt and the negative prompt take at most " + std::to_string(kMaxPrompt) +
                          " bytes each" };
    }
    if (r.width % 16 != 0 || r.height % 16 != 0 || std::min(r.width, r.height) < kMinImageSide ||
        int64_t(r.width) * r.height > kMaxImagePixels) {
        throw BadRequest{ "width and height must be multiples of 16, at least " + std::to_string(kMinImageSide) +
                          ", and at most " + std::to_string(kMaxImagePixels) + " pixels together" };
    }
    if (!(r.cfg_scale >= 0.0f) || !(r.eta >= 0.0f)) {
        throw BadRequest{ "cfg_scale and eta must not be negative" };
    }
    if (!(r.flow_shift > 0.0f)) {
        throw BadRequest{ "flow_shift must be positive" };
    }
    n       = std::clamp(n, 1, kMaxImages);
    r.steps = std::clamp(r.steps, 1, kMaxImageSteps);
    if (r.seed < 0) {
        static std::mutex mutex;
        static std::mt19937_64 rng{ std::random_device{}() };
        std::lock_guard<std::mutex> lock(mutex);
        r.seed = int64_t(rng() & 0x7fffffff);
    } else if (r.seed > std::numeric_limits<int64_t>::max() - (n - 1)) {
        throw BadRequest{ "the seed is too large: image i of a request takes seed + i" };
    }
}

class Server {
public:
    Server(const ImageServerConfig & config, Generator & generator) : config_(config), generator_(generator) {}

    void handle(const int fd) {
        try {
            HttpRequest req;
            if (read_request(fd, req)) {
                route(fd, req);
            }
        } catch (const BadRequest & e) {
            respond(fd, e.status, json{ { "error", e.message } }.dump());
        } catch (const std::exception & e) {
            respond(fd, 500, json{ { "error", "server_error" }, { "message", e.what() } }.dump());
        }
    }

private:
    void route(const int fd, const HttpRequest & req) {
        if (req.method == "OPTIONS") {
            respond(fd, 204, {});
        } else if (req.path == "/v1/models") {
            if (req.method != "GET") {
                respond(fd, 405, json{ { "error", "use GET" } }.dump());
                return;
            }
            json model = { { "id", config_.model_id }, { "object", "model" }, { "owned_by", "local" } };
            respond(fd, 200, json{ { "object", "list" }, { "data", json::array({ model }) } }.dump());
        } else if (req.path == "/v1/images/generations" || req.path == "/sdapi/v1/txt2img") {
            if (req.method != "POST") {
                respond(fd, 405, json{ { "error", "use POST" } }.dump());
                return;
            }
            const json body = json::parse(req.body, nullptr, false);
            if (body.is_discarded() || !body.is_object()) {
                throw BadRequest{ "the body must be a JSON object" };
            }
            if (req.path == "/sdapi/v1/txt2img") {
                txt2img(fd, body);
            } else {
                generations(fd, body);
            }
        } else if (req.path == "/memory" || req.path == "/memory/release" || req.path == "/memory/load") {
            memory(fd, req);
        } else {
            respond(fd, 404, json{ { "error", "no route " + req.path } }.dump());
        }
    }

    // The GPU memory of a server in standby: where the weights are, and giving it back or taking the weights again.
    void memory(const int fd, const HttpRequest & req) {
        const bool release = req.path == "/memory/release";
        const bool load    = req.path == "/memory/load";
        if (req.method != (release || load ? "POST" : "GET")) {
            respond(fd, 405, json{ { "error", release || load ? "use POST" : "use GET" } }.dump());
            return;
        }
        bool keep_weights = false;
        if (release && !req.body.empty()) {
            const json body = json::parse(req.body, nullptr, false);
            if (body.is_discarded() || !body.is_object()) {
                throw BadRequest{ "the body must be a JSON object" };
            }
            if (body.contains("keep_weights") && body["keep_weights"].is_boolean()) {
                keep_weights = body["keep_weights"].get<bool>();
            }
        }
        // A release stops the loads asked for before it, the one copying and any still waiting for the GPU.
        const uint64_t releases = release ? ++releases_ : releases_.load();

        std::string error;
        bool        ok = true;
        MemoryState state;
        {
            std::lock_guard<std::mutex> lock(gpu_);
            if (release) {
                ok = generator_.release_memory(keep_weights, error);
            } else if (load) {
                ok = generator_.load_weights([this, releases] { return releases_.load() != releases; }, error);
            }
            state = generator_.memory();
        }
        json out = { { "standby", state.standby },          { "weights", state.weights_on_gpu ? "gpu" : "host" },
                     { "gpu_mib", state.gpu_bytes >> 20 },  { "host_mib", state.host_bytes >> 20 },
                     { "gpu_free_mib", state.gpu_free_bytes >> 20 } };
        if (!ok) {
            out["error"] = error.empty() ? "a release stopped it" : error;
        }
        respond(fd, ok ? 200 : 409, out.dump());
    }

    void generations(const int fd, const json & body) {
        ImageRequest r = config_.defaults;
        int          n = 1;
        read_string(body, "prompt", r.prompt);
        read_number(body, "n", n);
        std::string size;
        read_string(body, "size", size);
        const size_t x = size.find('x');
        if (x != std::string::npos) {
            try {
                r.width  = std::stoi(size.substr(0, x));
                r.height = std::stoi(size.substr(x + 1));
            } catch (const std::exception &) {
                // sd-server keeps its default size when "size" does not parse
            }
        }
        std::string format = "png";
        read_string(body, "output_format", format);
        if (format != "png") {
            throw BadRequest{ "output_format must be png" };
        }
        const std::string extra = take_extra_args(r.prompt);
        if (!extra.empty()) {
            apply_extra_args(extra, r, n);
        }
        validate(r, n);

        json out = { { "created", int64_t(std::time(nullptr)) }, { "data", json::array() }, { "output_format", "png" } };
        for (const std::string & png : generate(r, n)) {
            out["data"].push_back({ { "b64_json", base64(png) } });
        }
        respond(fd, 200, out.dump());
    }

    void txt2img(const int fd, const json & body) {
        ImageRequest r = config_.defaults;
        int          n = 1;
        read_string(body, "prompt", r.prompt);
        read_string(body, "negative_prompt", r.negative_prompt);
        read_number(body, "width", r.width);
        read_number(body, "height", r.height);
        read_number(body, "steps", r.steps);
        read_number(body, "cfg_scale", r.cfg_scale);
        read_number(body, "seed", r.seed);
        read_number(body, "batch_size", n);
        std::string sampler, scheduler;
        read_string(body, "sampler_name", sampler);
        read_string(body, "scheduler", scheduler);
        if (!sampler.empty()) {
            r.sample_method = parse_sample_method(sampler);
        }
        if (!scheduler.empty()) {
            r.scheduler = parse_scheduler(scheduler);
        }
        const std::string extra = take_extra_args(r.prompt);
        if (!extra.empty()) {
            apply_extra_args(extra, r, n);
        }
        validate(r, n);

        // The request as resolved, not the body: copying a client's JSON recurses as deep as it nests.
        const json parameters = { { "prompt", r.prompt }, { "negative_prompt", r.negative_prompt },
                                  { "width", r.width }, { "height", r.height }, { "steps", r.steps },
                                  { "cfg_scale", r.cfg_scale }, { "seed", r.seed }, { "batch_size", n },
                                  { "sampler_name", name_of(r.sample_method) }, { "scheduler", name_of(r.scheduler) } };
        json out  = { { "images", json::array() }, { "parameters", parameters } };
        json info = { { "prompt", r.prompt }, { "seed", r.seed }, { "cfg_scale", r.cfg_scale },
                      { "width", r.width }, { "height", r.height }, { "steps", r.steps },
                      { "sampler_name", name_of(r.sample_method) },
                      { "extra_generation_params", { { "Schedule type", name_of(r.scheduler) } } },
                      { "all_seeds", json::array() } };
        if (!r.negative_prompt.empty()) {
            info["negative_prompt"] = r.negative_prompt;
        }
        int i = 0;
        for (const std::string & png : generate(r, n)) {
            out["images"].push_back(base64(png));
            info["all_seeds"].push_back(r.seed + i++);
        }
        out["info"] = info.dump();
        respond(fd, 200, out.dump());
    }

    // The request's n images, one at a time on the GPU, image i with seed + i.
    std::vector<std::string> generate(const ImageRequest & r, const int n) {
        std::vector<std::string>    pngs;
        std::lock_guard<std::mutex> lock(gpu_);
        for (int i = 0; i < n; ++i) {
            ImageRequest one = r;
            one.seed         = r.seed + i;
            RgbImage    image;
            std::string error;
            const auto  start = std::chrono::steady_clock::now();
            if (!generator_.generate(one, image, error)) {
                throw std::runtime_error(error);
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            pngs.push_back(encode_png(image.rgb.data(), image.width, image.height));
            if (pngs.back().empty()) {
                throw std::runtime_error("PNG encoding failed");
            }
            std::fprintf(stderr, "[moss-image] generate_image completed in %.2fs (%dx%d, %d steps, %s, %s, cfg %.2f, "
                                 "seed %lld)\n",
                         seconds, one.width, one.height, one.steps, name_of(one.sample_method),
                         name_of(one.scheduler), one.cfg_scale, (long long) one.seed);
        }
        return pngs;
    }

    const ImageServerConfig & config_;
    Generator &               generator_;
    std::mutex                gpu_;
    std::atomic<uint64_t>     releases_{ 0 };  // POST /memory/release requests so far
};

int listen_on(const std::string & host, const int port) {
    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags    = AI_PASSIVE;
    addrinfo * found  = nullptr;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &found) != 0) {
        return -1;
    }
    int fd = -1;
    for (addrinfo * a = found; a != nullptr && fd < 0; a = a->ai_next) {
        fd = socket(a->ai_family, a->ai_socktype | SOCK_CLOEXEC, a->ai_protocol);
        const int one = 1;
        if (fd >= 0 && (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0 ||
                        bind(fd, a->ai_addr, a->ai_addrlen) != 0 || listen(fd, 64) != 0)) {
            close(fd);
            fd = -1;
        }
    }
    freeaddrinfo(found);
    return fd;
}

} // namespace

int run_image_server(const ImageServerConfig & config, Generator & generator) {
    std::signal(SIGPIPE, SIG_IGN);
    const int listener = listen_on(config.host, config.port);
    if (listener < 0) {
        std::fprintf(stderr, "[moss-image] cannot listen on %s:%d\n", config.host.c_str(), config.port);
        return 1;
    }
    std::fprintf(stderr, "[moss-image] listening on http://%s:%d\n", config.host.c_str(), config.port);
    Server server(config, generator);
    for (;;) {
        const int fd = accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
        if (fd < 0) {
            // Linux reports a new connection's network errors here, and descriptors or memory free up as
            // connections close: keep accepting, with a pause so that an error that lasts does not spin.
            if (errno != EINTR && errno != ECONNABORTED) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            continue;
        }
        const timeval send_timeout{ kIdleMs / 1000, 0 };
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &send_timeout, sizeof(send_timeout));
        try {
            std::thread([&server, fd]() {
                server.handle(fd);
                close_after_answer(fd);
            }).detach();
        } catch (const std::system_error &) {
            close(fd);  // no thread to spare: the client can try again
        }
    }
}

} // namespace luce::moss_image
