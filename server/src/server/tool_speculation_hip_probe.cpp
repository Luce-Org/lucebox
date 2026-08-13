#include "tool_speculation_hip_probe.h"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace dflash::common {
namespace {

std::string hip_error(const char * operation, hipError_t status) {
    return std::string(operation) + ": " + hipGetErrorString(status);
}

std::string hipblas_error(const char * operation, hipblasStatus_t status) {
    return std::string(operation) + " failed with status " +
        std::to_string(static_cast<int>(status));
}

// HIP's current device is thread-local process state. The HTTP worker that
// launches a tool may immediately continue into model inference, so a trusted
// executor must leave that state exactly as it found it.
class ScopedHipDevice final {
public:
    explicit ScopedHipDevice(int device) {
        status_ = hipGetDevice(&previous_device_);
        if (status_ != hipSuccess) return;
        if (previous_device_ == device) return;
        status_ = hipSetDevice(device);
        switched_ = status_ == hipSuccess;
    }

    ~ScopedHipDevice() {
        if (switched_) (void) hipSetDevice(previous_device_);
    }

    bool ok() const { return status_ == hipSuccess; }
    hipError_t status() const { return status_; }

private:
    int previous_device_ = 0;
    hipError_t status_ = hipSuccess;
    bool switched_ = false;
};

class HipSgemmState;

class HipSgemmExecution final : public ToolSpeculationExecution {
public:
    explicit HipSgemmExecution(std::shared_ptr<HipSgemmState> state)
        : state_(std::move(state)) {}
    ~HipSgemmExecution() override;

    bool send_control(const std::string & operation) override;
    bool collect_result(int timeout_ms,
                        size_t max_result_bytes,
                        json & result,
                        double & wait_ms,
                        std::string & error) override;
    void terminate(bool allow_control_grace) override;

private:
    std::shared_ptr<HipSgemmState> state_;
    bool committed_ = false;
    bool finished_ = false;
};

class HipSgemmState final {
public:
    HipSgemmState(int device, int matrix_size, int total_cus)
        : device_(device), matrix_size_(matrix_size), total_cus_(total_cus) {}

    ~HipSgemmState() {
        std::lock_guard<std::mutex> lock(mutex_);
        ScopedHipDevice device(device_);
        if (!device.ok()) return;
        if (stream_) (void) hipStreamSynchronize(stream_);
        if (finished_) (void) hipEventDestroy(finished_);
        if (started_) (void) hipEventDestroy(started_);
        if (handle_) (void) hipblasDestroy(handle_);
        if (c_) (void) hipFree(c_);
        if (b_) (void) hipFree(b_);
        if (a_) (void) hipFree(a_);
        if (stream_) (void) hipStreamDestroy(stream_);
    }

    bool start(const json & request, std::string & error) {
        ScopedHipDevice device(device_);
        if (!device.ok()) {
            error = hip_error("hipSetDevice(tool)", device.status());
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_) {
            error = "HIP probe already has active work";
            return false;
        }
        try {
            const json & call = request.at("call");
            if (call.at("name").get<std::string>() != "benchmark_hip_sgemm") {
                error = "HIP probe only supports benchmark_hip_sgemm";
                return false;
            }
            const json & arguments = call.at("arguments");
            if (!arguments.is_object() ||
                !arguments.contains("iterations") ||
                !arguments["iterations"].is_number_integer()) {
                error = "benchmark_hip_sgemm.iterations must be an integer";
                return false;
            }
            iterations_ = arguments["iterations"].get<int>();
            if (iterations_ <= 0 || iterations_ > 1'000'000) {
                error = "benchmark_hip_sgemm.iterations must be 1..1000000";
                return false;
            }
            const int resource_percentage =
                request.at("resource_percentage").get<int>();
            if (resource_percentage <= 0 || resource_percentage > 100) {
                error = "resource_percentage must be 1..100";
                return false;
            }
            const int cu_count = std::clamp(
                (total_cus_ * resource_percentage + 99) / 100,
                1, total_cus_);
            if (!ensure_resources(cu_count, error)) return false;

            const float alpha = 1.0F;
            const float beta = 0.0F;
            hipError_t status = hipEventRecord(started_, stream_);
            if (status != hipSuccess) {
                error = hip_error("hipEventRecord(started)", status);
                return false;
            }
            for (int iteration = 0; iteration < iterations_; ++iteration) {
                const hipblasStatus_t blas_status = hipblasSgemm(
                    handle_, HIPBLAS_OP_N, HIPBLAS_OP_N,
                    matrix_size_, matrix_size_, matrix_size_,
                    &alpha, a_, matrix_size_, b_, matrix_size_,
                    &beta, c_, matrix_size_);
                if (blas_status != HIPBLAS_STATUS_SUCCESS) {
                    error = hipblas_error("hipblasSgemm", blas_status);
                    (void) hipStreamSynchronize(stream_);
                    return false;
                }
            }
            status = hipEventRecord(finished_, stream_);
            if (status != hipSuccess) {
                error = hip_error("hipEventRecord(finished)", status);
                (void) hipStreamSynchronize(stream_);
                return false;
            }
            active_ = true;
            error.clear();
            return true;
        } catch (const std::exception & exception) {
            error = std::string("invalid HIP probe request: ") + exception.what();
            return false;
        }
    }

    bool collect(int timeout_ms,
                 size_t max_result_bytes,
                 json & result,
                 double & wait_ms,
                 std::string & error) {
        ScopedHipDevice device(device_);
        if (!device.ok()) {
            error = hip_error("hipSetDevice(tool)", device.status());
            release_active();
            return false;
        }
        const auto wait_started = std::chrono::steady_clock::now();
        const auto deadline = wait_started +
            std::chrono::milliseconds(std::max(1, timeout_ms));
        while (true) {
            const hipError_t status = hipEventQuery(finished_);
            if (status == hipSuccess) break;
            if (status != hipErrorNotReady) {
                error = hip_error("hipEventQuery", status);
                release_active();
                return false;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                error = "executor_timeout";
                synchronize_and_release();
                wait_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - wait_started).count();
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        float gpu_ms = 0.0F;
        hipError_t status = hipEventElapsedTime(&gpu_ms, started_, finished_);
        if (status != hipSuccess) {
            error = hip_error("hipEventElapsedTime", status);
            release_active();
            return false;
        }
        float sample = 0.0F;
        status = hipMemcpyAsync(
            &sample, c_, sizeof(sample), hipMemcpyDeviceToHost, stream_);
        if (status == hipSuccess) status = hipStreamSynchronize(stream_);
        if (status != hipSuccess || !std::isfinite(sample)) {
            error = status == hipSuccess
                ? "HIP probe produced a non-finite sample"
                : hip_error("HIP probe result copy", status);
            release_active();
            return false;
        }
        result = {
            {"sample", sample},
            {"gpu_ms", gpu_ms},
            {"iterations", iterations_},
            {"matrix_size", matrix_size_},
            {"cu_count", current_cu_count_},
        };
        if (result.dump().size() > max_result_bytes) {
            error = "executor_result_too_large";
            release_active();
            return false;
        }
        wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wait_started).count();
        release_active();
        error.clear();
        return true;
    }

    void synchronize_and_release() {
        ScopedHipDevice device(device_);
        std::lock_guard<std::mutex> lock(mutex_);
        if (device.ok() && active_ && stream_) {
            (void) hipStreamSynchronize(stream_);
        }
        active_ = false;
    }

private:
    bool ensure_resources(int cu_count, std::string & error) {
        hipError_t status = hipSuccess;
        if (!stream_ || !handle_ || current_cu_count_ != cu_count) {
            if (stream_) {
                (void) hipStreamSynchronize(stream_);
                if (handle_) {
                    (void) hipblasDestroy(handle_);
                    handle_ = nullptr;
                }
                (void) hipStreamDestroy(stream_);
                stream_ = nullptr;
            }
            const size_t mask_words =
                static_cast<size_t>((total_cus_ + 31) / 32);
            std::vector<uint32_t> mask(mask_words, 0);
            for (int cu = 0; cu < cu_count; ++cu) {
                mask[static_cast<size_t>(cu / 32)] |=
                    uint32_t{1} << (cu % 32);
            }
            status = hipExtStreamCreateWithCUMask(
                &stream_, static_cast<uint32_t>(mask.size()), mask.data());
            if (status != hipSuccess) {
                error = hip_error("hipExtStreamCreateWithCUMask", status);
                return false;
            }
            const hipblasStatus_t create_status = hipblasCreate(&handle_);
            if (create_status != HIPBLAS_STATUS_SUCCESS) {
                error = hipblas_error("hipblasCreate", create_status);
                return false;
            }
            const hipblasStatus_t stream_status =
                hipblasSetStream(handle_, stream_);
            if (stream_status != HIPBLAS_STATUS_SUCCESS) {
                error = hipblas_error("hipblasSetStream", stream_status);
                return false;
            }
            current_cu_count_ = cu_count;
        }
        if (!a_ || !b_ || !c_) {
            if (c_) (void) hipFree(c_);
            if (b_) (void) hipFree(b_);
            if (a_) (void) hipFree(a_);
            a_ = nullptr;
            b_ = nullptr;
            c_ = nullptr;
            const size_t elements =
                static_cast<size_t>(matrix_size_) * matrix_size_;
            const size_t bytes = elements * sizeof(float);
            if ((status = hipMalloc(&a_, bytes)) != hipSuccess ||
                (status = hipMalloc(&b_, bytes)) != hipSuccess ||
                (status = hipMalloc(&c_, bytes)) != hipSuccess) {
                error = hip_error("hipMalloc", status);
                if (c_) (void) hipFree(c_);
                if (b_) (void) hipFree(b_);
                if (a_) (void) hipFree(a_);
                a_ = nullptr;
                b_ = nullptr;
                c_ = nullptr;
                return false;
            }
            if ((status = hipMemsetAsync(a_, 0x01, bytes, stream_)) != hipSuccess ||
                (status = hipMemsetAsync(b_, 0x02, bytes, stream_)) != hipSuccess ||
                (status = hipMemsetAsync(c_, 0, bytes, stream_)) != hipSuccess) {
                error = hip_error("hipMemsetAsync", status);
                return false;
            }
            const float alpha = 1.0F;
            const float beta = 0.0F;
            const hipblasStatus_t warm_status = hipblasSgemm(
                handle_, HIPBLAS_OP_N, HIPBLAS_OP_N,
                matrix_size_, matrix_size_, matrix_size_,
                &alpha, a_, matrix_size_, b_, matrix_size_,
                &beta, c_, matrix_size_);
            if (warm_status != HIPBLAS_STATUS_SUCCESS) {
                error = hipblas_error("hipblasSgemm(warmup)", warm_status);
                return false;
            }
            if ((status = hipStreamSynchronize(stream_)) != hipSuccess) {
                error = hip_error("hipStreamSynchronize(warmup)", status);
                return false;
            }
        }
        if (!started_ &&
            (status = hipEventCreate(&started_)) != hipSuccess) {
            error = hip_error("hipEventCreate(started)", status);
            return false;
        }
        if (!finished_ &&
            (status = hipEventCreate(&finished_)) != hipSuccess) {
            error = hip_error("hipEventCreate(finished)", status);
            return false;
        }
        return true;
    }

    void release_active() {
        std::lock_guard<std::mutex> lock(mutex_);
        active_ = false;
    }

    std::mutex mutex_;
    int device_ = 0;
    int matrix_size_ = 0;
    int total_cus_ = 0;
    int current_cu_count_ = 0;
    int iterations_ = 0;
    bool active_ = false;
    hipStream_t stream_ = nullptr;
    hipblasHandle_t handle_ = nullptr;
    hipEvent_t started_ = nullptr;
    hipEvent_t finished_ = nullptr;
    float * a_ = nullptr;
    float * b_ = nullptr;
    float * c_ = nullptr;
};

HipSgemmExecution::~HipSgemmExecution() {
    if (!finished_) terminate(false);
}

bool HipSgemmExecution::send_control(const std::string & operation) {
    if (operation == "commit") {
        committed_ = true;
        return true;
    }
    if (operation == "cancel") {
        committed_ = false;
        return true;
    }
    return false;
}

bool HipSgemmExecution::collect_result(
        int timeout_ms,
        size_t max_result_bytes,
        json & result,
        double & wait_ms,
        std::string & error) {
    if (finished_) {
        error = "executor already collected";
        return false;
    }
    if (!committed_) {
        error = "executor result requested before commit";
        terminate(false);
        return false;
    }
    finished_ = true;
    return state_->collect(
        timeout_ms, max_result_bytes, result, wait_ms, error);
}

void HipSgemmExecution::terminate(bool allow_control_grace) {
    (void) allow_control_grace;
    if (finished_) return;
    finished_ = true;
    state_->synchronize_and_release();
}

class HipSgemmExecutor final : public ToolSpeculationExecutor {
public:
    explicit HipSgemmExecutor(std::shared_ptr<HipSgemmState> state)
        : state_(std::move(state)) {}

    std::unique_ptr<ToolSpeculationExecution> start(
            const json & request, std::string & error) override {
        if (!state_->start(request, error)) return nullptr;
        return std::make_unique<HipSgemmExecution>(state_);
    }

    const char * mode_name() const override {
        return "in_process_hip_cu_mask";
    }

private:
    std::shared_ptr<HipSgemmState> state_;
};

}  // namespace

std::shared_ptr<ToolSpeculationExecutor>
create_hip_sgemm_tool_speculation_executor(
        int device,
        int matrix_size,
        int & total_compute_units,
        std::string & error) {
    total_compute_units = 0;
    if (device < 0 || matrix_size <= 0 || matrix_size > 8192) {
        error = "HIP probe needs DEVICE >= 0 and MATRIX_SIZE in 1..8192";
        return nullptr;
    }
    hipDeviceProp_t properties{};
    const hipError_t status = hipGetDeviceProperties(&properties, device);
    if (status != hipSuccess) {
        error = hip_error("hipGetDeviceProperties", status);
        return nullptr;
    }
    if (properties.multiProcessorCount <= 0) {
        error = "HIP device reports no compute units";
        return nullptr;
    }
    total_compute_units = properties.multiProcessorCount;
    error.clear();
    auto state = std::make_shared<HipSgemmState>(
        device, matrix_size, properties.multiProcessorCount);
    return std::make_shared<HipSgemmExecutor>(std::move(state));
}

}  // namespace dflash::common
