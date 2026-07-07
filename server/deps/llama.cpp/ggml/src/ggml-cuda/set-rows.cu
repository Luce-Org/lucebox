#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "tq3-quant.cuh"

template <typename idx_t>
static __global__ void k_set_rows_tq3_0(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_tq3_0 * __restrict__ dst,
        const int64_t ne_total,
        const int64_t num_groups_per_row,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const uint3   ne00g_fd,
        const uint3   ne01_fd,
        const uint3   ne02_fd,
        const uint3   ne11_fd,
        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i >= ne_total) return;

    uint32_t tmp = (uint32_t) i;
    uint2 div_mod;

    div_mod = fast_div_modulo(tmp, ne00g_fd);
    const int64_t ig = div_mod.y;
    tmp = div_mod.x;

    div_mod = fast_div_modulo(tmp, ne01_fd);
    const int64_t i01 = div_mod.y;
    tmp = div_mod.x;

    div_mod = fast_div_modulo(tmp, ne02_fd);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_tq3_0 * dst_row_ptr = (block_tq3_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);

    const float * src_grp = src0_row + ig * QK_TQ3_0_GROUP;
    block_tq3_0 * dst_grp = dst_row_ptr + ig * (QK_TQ3_0_GROUP / QK_TQ3_0);

    quantize_f32_tq3_0_group(src_grp, dst_grp);

    GGML_UNUSED(ne10); GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13);
}

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * __restrict__ src0,
                                  const idx_t * __restrict__ src1,
                                  dst_t * __restrict__ dst,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows<<<grid_size, block_size, 0, stream>>>(src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
                                                         s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
                                                         ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TQ3_0) {
        GGML_ASSERT(ne00 % QK_TQ3_0_GROUP == 0);
        const int64_t num_groups_per_row = ne00 / QK_TQ3_0_GROUP;
        const int64_t ne_total = num_groups_per_row * ne01 * ne02 * ne03;
        const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;

        const int64_t s01 = nb01 / sizeof(float);
        const int64_t s02 = nb02 / sizeof(float);
        const int64_t s03 = nb03 / sizeof(float);
        const int64_t s10 = nb10 / sizeof(idx_t);
        const int64_t s11 = nb11 / sizeof(idx_t);
        const int64_t s12 = nb12 / sizeof(idx_t);
        const int64_t s1  = nb1;
        const int64_t s2  = nb2;
        const int64_t s3  = nb3;

        if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
            const uint3 ne00g_fd = init_fastdiv_values((uint32_t) num_groups_per_row);
            const uint3 ne01_fd  = init_fastdiv_values((uint32_t) ne01);
            const uint3 ne02_fd  = init_fastdiv_values((uint32_t) ne02);
            const uint3 ne11_fd  = init_fastdiv_values((uint32_t) ne11);
            const uint3 ne12_fd  = init_fastdiv_values((uint32_t) ne12);

            k_set_rows_tq3_0<idx_t><<<num_blocks, CUDA_SET_ROWS_BLOCK_SIZE, 0, stream>>>(
                src0_d, src1_d, (block_tq3_0*)dst->data,
                ne_total, num_groups_per_row,
                ne10, ne11, ne12, ne13,
                s01, s02, s03, s10, s11, s12, s1, s2, s3,
                ne00g_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        }
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}



// ---- Fused dual set_rows -------------------------------------------------
// Two independent quantized SET_ROWS (the per-layer K and V cache appends)
// executed by one kernel launch. Each element runs the exact same math as
// k_set_rows_quant, so outputs are bit-identical to two separate launches.

template <typename idx_t>
struct set_rows_quant_params {
    const float * src0;
    const idx_t * src1;
    void        * dst;
    int64_t ne_total;
    int64_t s01, s02, s03, s10, s11, s12, s1, s2, s3;
    uint3   ne00, ne01, ne02, ne11_fd, ne12_fd;
};

template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant_dual(
        const set_rows_quant_params<idx_t> pa,
        const set_rows_quant_params<idx_t> pb) {
    int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    const bool second = i >= pa.ne_total;
    if (second) {
        i -= pa.ne_total;
        if (i >= pb.ne_total) {
            return;
        }
    }
    const set_rows_quant_params<idx_t> p = second ? pb : pa;

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, p.ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, p.ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, p.ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, p.ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, p.ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(p.src1 + i10*p.s10 + i11*p.s11 + i12*p.s12);

    const float * src0_row = p.src0 + i01*p.s01 + i02*p.s02 + i03*p.s03;
    block_type * dst_row_ptr = (block_type *) p.dst + (dst_row*p.s1 + i02*p.s2 + i03*p.s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);
}

template <typename idx_t>
static set_rows_quant_params<idx_t> make_set_rows_quant_params(const ggml_tensor * dst_t, const int qk) {
    const ggml_tensor * src0 = dst_t->src[0];
    const ggml_tensor * src1 = dst_t->src[1];
    const ggml_tensor * dst  = dst_t;

    GGML_TENSOR_BINARY_OP_LOCALS

    set_rows_quant_params<idx_t> p{};
    p.src0 = (const float *) src0->data;
    p.src1 = (const idx_t *) src1->data;
    p.dst  = dst->data;
    GGML_ASSERT(ne00 % qk == 0);
    p.ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    p.s01 = nb01 / sizeof(float);
    p.s02 = nb02 / sizeof(float);
    p.s03 = nb03 / sizeof(float);
    p.s10 = nb10 / sizeof(idx_t);
    p.s11 = nb11 / sizeof(idx_t);
    p.s12 = nb12 / sizeof(idx_t);
    p.s1  = nb1;
    p.s2  = nb2;
    p.s3  = nb3;
    p.ne00    = init_fastdiv_values((uint32_t) ne00);
    p.ne01    = init_fastdiv_values((uint32_t) ne01);
    p.ne02    = init_fastdiv_values((uint32_t) ne02);
    p.ne11_fd = init_fastdiv_values((uint32_t) ne11);
    p.ne12_fd = init_fastdiv_values((uint32_t) ne12);
    return p;
}

template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static void set_rows_dual_launch(ggml_backend_cuda_context & ctx, ggml_tensor * a, ggml_tensor * b) {
    const set_rows_quant_params<idx_t> pa = make_set_rows_quant_params<idx_t>(a, qk);
    const set_rows_quant_params<idx_t> pb = make_set_rows_quant_params<idx_t>(b, qk);
    const int64_t total = pa.ne_total + pb.ne_total;
    if (total <= 0) {
        return;
    }
    const int num_blocks = (total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    k_set_rows_quant_dual<idx_t, block_type, qk, quantize_func>
        <<<num_blocks, CUDA_SET_ROWS_BLOCK_SIZE, 0, ctx.stream()>>>(pa, pb);
}

template <typename idx_t>
static void set_rows_dual_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * a, ggml_tensor * b) {
    switch (a->type) {
        case GGML_TYPE_Q4_0:
            set_rows_dual_launch<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(ctx, a, b); break;
        case GGML_TYPE_Q4_1:
            set_rows_dual_launch<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(ctx, a, b); break;
        case GGML_TYPE_Q5_0:
            set_rows_dual_launch<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(ctx, a, b); break;
        case GGML_TYPE_Q5_1:
            set_rows_dual_launch<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(ctx, a, b); break;
        case GGML_TYPE_Q8_0:
            set_rows_dual_launch<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(ctx, a, b); break;
        case GGML_TYPE_IQ4_NL:
            set_rows_dual_launch<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(ctx, a, b); break;
        default:
            GGML_ABORT("unsupported dual set_rows type %s", ggml_type_name(a->type));
    }
}

bool ggml_cuda_set_rows_dual_supported(const ggml_tensor * a, const ggml_tensor * b) {
    if (a->op != GGML_OP_SET_ROWS || b->op != GGML_OP_SET_ROWS) {
        return false;
    }
    if (a->type != b->type || a->data == b->data) {
        return false;
    }
    switch (a->type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0: case GGML_TYPE_IQ4_NL:
            break;
        default:
            return false;
    }
    if (a->src[0]->type != GGML_TYPE_F32 || b->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    if (a->src[1]->type != b->src[1]->type ||
        (a->src[1]->type != GGML_TYPE_I32 && a->src[1]->type != GGML_TYPE_I64)) {
        return false;
    }
    const int qk = ggml_blck_size(a->type);
    if (a->src[0]->ne[0] % qk != 0 || b->src[0]->ne[0] % qk != 0) {
        return false;
    }
    if (ggml_nelements(a->src[0]) <= 0 || ggml_nelements(b->src[0]) <= 0) {
        return false;
    }
    return true;
}

void ggml_cuda_op_set_rows_dual(ggml_backend_cuda_context & ctx, ggml_tensor * a, ggml_tensor * b) {
    if (a->src[1]->type == GGML_TYPE_I64) {
        set_rows_dual_dispatch<int64_t>(ctx, a, b);
    } else {
        set_rows_dual_dispatch<int32_t>(ctx, a, b);
    }
}

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
    }
}
