// Self-contained HIP kernel correctness smoke for the AMD GPUs on lucebox3.
// Compiled and run once for gfx1201 and once for gfx1151 by gpu-tests-amd to
// prove the ROCm/HIP compute path actually executes on AMD hardware (GitHub
// hosted runners have no GPU). No model weights, deterministic, fast.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error at line %d: %s\n", __LINE__, hipGetErrorString(e)); \
    return 2; } } while (0)

int main() {
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, 0));
    printf("HIP device 0: %s (%s, %d CUs)\n", p.name, p.gcnArchName, p.multiProcessorCount);

    const int n = 1 << 20;
    const size_t sz = n * sizeof(float);
    float *ha = (float*)malloc(sz), *hb = (float*)malloc(sz), *hc = (float*)malloc(sz);
    for (int i = 0; i < n; i++) { ha[i] = 1.0f * i; hb[i] = 2.0f * i; }

    float *da, *db, *dc;
    CK(hipMalloc(&da, sz)); CK(hipMalloc(&db, sz)); CK(hipMalloc(&dc, sz));
    CK(hipMemcpy(da, ha, sz, hipMemcpyHostToDevice));
    CK(hipMemcpy(db, hb, sz, hipMemcpyHostToDevice));
    vadd<<<(n + 255) / 256, 256>>>(da, db, dc, n);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(hc, dc, sz, hipMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < n; i++) if (hc[i] != 3.0f * i) bad++;
    printf(bad ? "FAIL: %d mismatches\n" : "PASS: vadd correct on %d elements\n", bad ? bad : n);
    return bad ? 1 : 0;
}
