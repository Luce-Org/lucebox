#pragma once

typedef short qd_half16 __attribute__((ext_vector_type(16)));
typedef float qd_float8 __attribute__((ext_vector_type(8)));

static __global__ __launch_bounds__(256) void qsa_decode_wmma_partial(
        const char * q, const char * k, const char * v, const char * mask, const char * ids,
        size_t q1, size_t q2, size_t k1, size_t k2, size_t v1, size_t v2, size_t m1, size_t i1,
        int nk, int ns, int nh, int splits, float scale, float * partial) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, r = lane & 15, hi = lane >> 4;
    const int kvh = blockIdx.x, query = blockIdx.y, split = blockIdx.z;
    const int * ir = (const int *) (ids + query*i1);
    const half * mr = mask ? (const half *) (mask + query*m1) : nullptr;
    __shared__ union {
        uint16_t query_stage[8][16][32];
        float4 scores[8][64];
    } shared;
    __shared__ __align__(16) uint16_t probabilities[16][16];
    __shared__ float alpha_s[16];
    qd_half16 qf[2];
    const _Float16 hs = (_Float16) scale;
    for (int j = lane; j < 16*32; j += 32) {
        const int head = j/32, d = j%32;
        const float value = head < 12 ? *(const float *) (q + query*q1 + (kvh*12+head)*q2 + (w*32+d)*4) : 0.0f;
        shared.query_stage[w][head][d] = __builtin_bit_cast(uint16_t, (_Float16)value*hs);
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        const uint16_t * src = &shared.query_stage[w][r][t*16];
        qf[t] = __builtin_bit_cast(qd_half16, (uint4[2]){*(const uint4 *)src, *(const uint4 *)(src+8)});
    }
    __syncthreads();
    qd_float8 output[2] = {};
    float maximum = -INFINITY, normalizer = 0.0f;
    const int end = min(ns, split*64+64);
    for (int start = split*64; start < end; start += 16) {
        int keys[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            int key = start+j < end ? ir[start+j] : -1;
            if (key < 0 || key >= nk || (mr && __half2float(mr[key]) == -INFINITY)) { key = -1; }
            keys[j] = key;
        }
        qd_float8 scores = {};
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            qd_half16 kf = {};
            const int key = keys[r];
            if (key >= 0 && key < nk) {
                const uint16_t * src = (const uint16_t *) (k + key*k1 + kvh*k2) + 32*w+16*t;
                kf = __builtin_bit_cast(qd_half16, (uint4[2]){*(const uint4 *)src, *(const uint4 *)(src+8)});
            }
            scores = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kf, qf[t], scores);
        }
        shared.scores[w][lane*2] = make_float4(scores[0],scores[1],scores[2],scores[3]);
        shared.scores[w][lane*2+1] = make_float4(scores[4],scores[5],scores[6],scores[7]);
        __syncthreads();
        if (w == 0) {
            float score[8] = {};
#pragma unroll
            for (int sw = 0; sw < 8; ++sw) {
                const float4 lo = shared.scores[sw][lane*2], high = shared.scores[sw][lane*2+1];
                score[0]+=lo.x;score[1]+=lo.y;score[2]+=lo.z;score[3]+=lo.w;
                score[4]+=high.x;score[5]+=high.y;score[6]+=high.z;score[7]+=high.w;
            }
            float local = -INFINITY;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int key = keys[2*e+hi];
                if (key < 0 || key >= nk) { score[e] = -INFINITY; }
                else if (mr) { score[e] += __half2float(mr[key]); }
                local = fmaxf(local,score[e]);
            }
            local = fmaxf(local,__shfl_xor(local,16));
            const float next = fmaxf(maximum,local);
            const float alpha = next == -INFINITY ? 1.0f : exp2f((maximum-next)*1.4426950408889634f);
            float sum = 0.0f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float p = next == -INFINITY ? 0.0f : exp2f((score[e]-next)*1.4426950408889634f);
                sum += p;
                probabilities[r][2*e+hi] = __builtin_bit_cast(uint16_t,(_Float16)p);
            }
            sum += __shfl_xor(sum,16);
            normalizer = normalizer*alpha+sum;
            maximum = next;
            if (hi == 0) { alpha_s[r] = alpha; }
        }
        __syncthreads();
        const float alpha = alpha_s[r];
        const qd_half16 pf = __builtin_bit_cast(qd_half16, (uint4[2]){
            *(const uint4 *)&probabilities[r][0], *(const uint4 *)&probabilities[r][8]});
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            uint16_t values[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int key = keys[j];
                values[j] = key >= 0 && key < nk ? *((const uint16_t *)(v+key*v1+kvh*v2)+32*w+16*t+r) : 0;
            }
            const qd_half16 vf = __builtin_bit_cast(qd_half16,values);
#pragma unroll
            for (int e = 0; e < 8; ++e) { output[t][e] *= alpha; }
            output[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vf,pf,output[t]);
        }
        __syncthreads();
    }
    if (r < 12) {
        float * dst = partial + ((query*nh+kvh*12+r)*splits+split)*258;
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) { dst[32*w+16*t+2*e+hi] = output[t][e]; }
        if (w == 0 && hi == 0) { dst[256] = maximum; dst[257] = normalizer; }
    }
}
