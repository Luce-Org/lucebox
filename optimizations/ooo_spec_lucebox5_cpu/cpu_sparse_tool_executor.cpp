// Deterministic read-only sparse-compute adapter for
// dflash.tool-speculation.v1 qualification.
//
// This is a benchmark tool, not an application-specific tool. It provides a
// reproducible CPU-bound workload whose exact result can be compared between
// sequential and speculative execution. The engine pins this child before it
// releases the JSON request, and the adapter verifies the observed mask.

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

using json = nlohmann::json;

namespace {

constexpr const char * kProtocol = "dflash.tool-speculation.v1";
constexpr const char * kToolName = "benchmark_cpu_sparse";
constexpr int kRows = 4096;
constexpr int kNonzerosPerRow = 16;
constexpr int kThreads = 2;
constexpr uint64_t kSeed = 731;

uint64_t splitmix64(uint64_t & state) {
    uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::vector<int> observed_affinity() {
#if defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (::sched_getaffinity(0, sizeof(mask), &mask) != 0) {
        throw std::runtime_error("sched_getaffinity failed");
    }
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &mask)) cpus.push_back(cpu);
    }
    return cpus;
#else
    return {};
#endif
}

uint64_t sparse_worker(int worker,
                       int rows,
                       int nonzeros_per_row,
                       int iterations,
                       uint64_t seed) {
    const size_t entries =
        static_cast<size_t>(rows) * static_cast<size_t>(nonzeros_per_row);
    std::vector<uint32_t> columns(entries);
    std::vector<uint32_t> values(entries);
    std::vector<uint32_t> input(static_cast<size_t>(rows));
    std::vector<uint32_t> output(static_cast<size_t>(rows));

    uint64_t state = seed ^
        (0xd6e8feb86659fd93ULL * static_cast<uint64_t>(worker + 1));
    for (size_t index = 0; index < entries; ++index) {
        columns[index] = static_cast<uint32_t>(splitmix64(state) % rows);
        values[index] = static_cast<uint32_t>(splitmix64(state) | 1ULL);
    }
    for (uint32_t & value : input) {
        value = static_cast<uint32_t>(splitmix64(state));
    }

    uint64_t rolling = 0xcbf29ce484222325ULL ^
        static_cast<uint64_t>(worker);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int row = 0; row < rows; ++row) {
            uint64_t accumulator =
                static_cast<uint64_t>(iteration + 1) * 0x9e3779b1U +
                static_cast<uint64_t>(row);
            const size_t start =
                static_cast<size_t>(row) * nonzeros_per_row;
            for (int offset = 0; offset < nonzeros_per_row; ++offset) {
                const size_t index = start + static_cast<size_t>(offset);
                accumulator += static_cast<uint64_t>(values[index]) *
                    input[columns[index]];
            }
            const uint32_t folded = static_cast<uint32_t>(
                accumulator ^ (accumulator >> 32U));
            output[static_cast<size_t>(row)] =
                folded + static_cast<uint32_t>(row * 2654435761U);
        }
        input.swap(output);
        rolling ^= static_cast<uint64_t>(input[
            static_cast<size_t>(iteration) % input.size()]);
        rolling *= 0x100000001b3ULL;
    }
    for (size_t index = 0; index < input.size(); index += 17) {
        rolling ^= static_cast<uint64_t>(input[index]) + index;
        rolling *= 0x100000001b3ULL;
    }
    return rolling;
}

int integer_argument(const json & arguments,
                     const char * name,
                     int minimum,
                     int maximum) {
    if (!arguments.contains(name) || !arguments[name].is_number_integer()) {
        throw std::runtime_error(std::string(name) + " must be an integer");
    }
    const int value = arguments[name].get<int>();
    if (value < minimum || value > maximum) {
        throw std::runtime_error(
            std::string(name) + " is outside the allowed range");
    }
    return value;
}

json execute(const json & request) {
    if (!request.is_object() || request.value("protocol", "") != kProtocol) {
        throw std::runtime_error("unsupported protocol");
    }
    if (!request.contains("call") || !request["call"].is_object() ||
        request["call"].value("name", "") != kToolName) {
        throw std::runtime_error("only benchmark_cpu_sparse is allowed");
    }
    const json & arguments = request["call"].at("arguments");
    if (!arguments.is_object()) {
        throw std::runtime_error("arguments must be an object");
    }
    const int rows = arguments.contains("rows")
        ? integer_argument(arguments, "rows", 64, 1 << 20) : kRows;
    const int nonzeros = arguments.contains("nonzeros_per_row")
        ? integer_argument(arguments, "nonzeros_per_row", 1, 256)
        : kNonzerosPerRow;
    const int iterations = integer_argument(
        arguments, "iterations", 1, 1'000'000);
    const int threads = arguments.contains("threads")
        ? integer_argument(arguments, "threads", 1, 64) : kThreads;
    if (static_cast<uint64_t>(rows) * static_cast<uint64_t>(nonzeros) >
        16ULL * 1024ULL * 1024ULL) {
        throw std::runtime_error("sparse matrix exceeds the 16M-entry limit");
    }
    uint64_t seed = kSeed;
    if (arguments.contains("seed")) {
        if (!arguments["seed"].is_number_integer()) {
            throw std::runtime_error("seed must be an unsigned integer");
        }
        if (arguments["seed"].is_number_unsigned()) {
            seed = arguments["seed"].get<uint64_t>();
        } else {
            const int64_t signed_seed = arguments["seed"].get<int64_t>();
            if (signed_seed < 0) {
                throw std::runtime_error("seed must be an unsigned integer");
            }
            seed = static_cast<uint64_t>(signed_seed);
        }
    }

    std::vector<int> expected_affinity;
    if (request.contains("cpu_affinity")) {
        expected_affinity = request["cpu_affinity"].get<std::vector<int>>();
        std::sort(expected_affinity.begin(), expected_affinity.end());
        expected_affinity.erase(
            std::unique(expected_affinity.begin(), expected_affinity.end()),
            expected_affinity.end());
    }
    const std::vector<int> affinity = observed_affinity();
    if (!expected_affinity.empty() && affinity != expected_affinity) {
        throw std::runtime_error("observed CPU affinity does not match request");
    }
    if (!affinity.empty() && threads > static_cast<int>(affinity.size())) {
        throw std::runtime_error("threads exceed the pinned logical CPU count");
    }

    const auto started = std::chrono::steady_clock::now();
    std::vector<uint64_t> partial(static_cast<size_t>(threads));
    std::vector<int> pin_errors(static_cast<size_t>(threads), 0);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));
    for (int worker = 0; worker < threads; ++worker) {
        workers.emplace_back([&, worker]() {
#if defined(__linux__)
            if (!affinity.empty()) {
                cpu_set_t worker_mask;
                CPU_ZERO(&worker_mask);
                CPU_SET(affinity[static_cast<size_t>(worker)], &worker_mask);
                pin_errors[static_cast<size_t>(worker)] =
                    ::pthread_setaffinity_np(
                        ::pthread_self(), sizeof(worker_mask), &worker_mask);
                if (pin_errors[static_cast<size_t>(worker)] != 0) return;
            }
#endif
            partial[static_cast<size_t>(worker)] = sparse_worker(
                worker, rows, nonzeros, iterations, seed);
        });
    }
    for (std::thread & worker : workers) worker.join();
    if (std::any_of(pin_errors.begin(), pin_errors.end(),
                    [](int error) { return error != 0; })) {
        throw std::runtime_error("worker CPU pinning failed");
    }
    const double compute_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();

    uint64_t checksum = 0x6a09e667f3bcc909ULL;
    for (const uint64_t value : partial) {
        checksum ^= value + 0x9e3779b97f4a7c15ULL +
            (checksum << 6U) + (checksum >> 2U);
    }
    return {
        {"ok", true},
        {"result", {
            {"checksum", std::to_string(checksum)},
            {"compute_ms", compute_ms},
            {"rows", rows},
            {"nonzeros_per_row", nonzeros},
            {"iterations", iterations},
            {"threads", threads},
            {"seed", seed},
            {"cpu_affinity", affinity},
            {"worker_cpus", std::vector<int>(
                affinity.begin(), affinity.begin() +
                    std::min(affinity.size(), static_cast<size_t>(threads)))},
        }},
    };
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2 || std::string(argv[1]) != "--dflash-tool-spec-v1") {
        std::cerr << "expected --dflash-tool-spec-v1\n";
        return 2;
    }
    try {
        std::string line;
        if (!std::getline(std::cin, line) || line.empty()) {
            throw std::runtime_error("missing request");
        }
        std::cout << execute(json::parse(line)).dump() << '\n';
        std::cout.flush();
        return 0;
    } catch (const std::exception & exception) {
        std::cerr << exception.what() << '\n';
        return 2;
    }
}
