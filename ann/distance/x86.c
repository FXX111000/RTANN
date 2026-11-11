// SPDX-License-Identifier: GPL-2.0

#include <linux/types.h>

#include "../../include/vecfs_ann_distance.h"

#ifdef CONFIG_X86_64

static __always_inline float vecfs_ann_horizontal_sum_sse(
    const float values[4])
{
    return values[0] +
           values[1] +
           values[2] +
           values[3];
}

float vecfs_ann_l2_sse2(
    const float *left,
    const float *right,
    u32 dimension)
{
    float lanes[4] __aligned(16);
    float result;
    u32 vector_count = dimension >> 2;
    u32 remainder = dimension & 3;
    const float *a = left;
    const float *b = right;
    u32 i;

    asm volatile(
        "xorps %%xmm0, %%xmm0\n\t"
        "test %[count], %[count]\n\t"
        "jz 2f\n\t"
        "1:\n\t"
        "movups (%[left]), %%xmm1\n\t"
        "movups (%[right]), %%xmm2\n\t"
        "subps %%xmm2, %%xmm1\n\t"
        "mulps %%xmm1, %%xmm1\n\t"
        "addps %%xmm1, %%xmm0\n\t"
        "add $16, %[left]\n\t"
        "add $16, %[right]\n\t"
        "dec %[count]\n\t"
        "jnz 1b\n\t"
        "2:\n\t"
        "movups %%xmm0, %[lanes]\n\t"
        : [left] "+r" (a),
          [right] "+r" (b),
          [count] "+r" (vector_count),
          [lanes] "=m" (lanes)
        :
        : "memory", "cc",
          "xmm0", "xmm1", "xmm2");

    result = vecfs_ann_horizontal_sum_sse(lanes);

    for (i = 0; i < remainder; i++) {
        float delta = a[i] - b[i];

        result += delta * delta;
    }

    return result;
}

float vecfs_ann_ip_sse2(
    const float *left,
    const float *right,
    u32 dimension)
{
    float lanes[4] __aligned(16);
    float result;
    u32 vector_count = dimension >> 2;
    u32 remainder = dimension & 3;
    const float *a = left;
    const float *b = right;
    u32 i;

    asm volatile(
        "xorps %%xmm0, %%xmm0\n\t"
        "test %[count], %[count]\n\t"
        "jz 2f\n\t"
        "1:\n\t"
        "movups (%[left]), %%xmm1\n\t"
        "movups (%[right]), %%xmm2\n\t"
        "mulps %%xmm2, %%xmm1\n\t"
        "addps %%xmm1, %%xmm0\n\t"
        "add $16, %[left]\n\t"
        "add $16, %[right]\n\t"
        "dec %[count]\n\t"
        "jnz 1b\n\t"
        "2:\n\t"
        "movups %%xmm0, %[lanes]\n\t"
        : [left] "+r" (a),
          [right] "+r" (b),
          [count] "+r" (vector_count),
          [lanes] "=m" (lanes)
        :
        : "memory", "cc",
          "xmm0", "xmm1", "xmm2");

    result = vecfs_ann_horizontal_sum_sse(lanes);

    for (i = 0; i < remainder; i++)
        result += a[i] * b[i];

    return 1.0f - result;
}

static __always_inline float vecfs_ann_horizontal_sum_avx(
    const float values[8])
{
    return values[0] +
           values[1] +
           values[2] +
           values[3] +
           values[4] +
           values[5] +
           values[6] +
           values[7];
}

float vecfs_ann_l2_avx2(
    const float *left,
    const float *right,
    u32 dimension)
{
    float lanes[8] __aligned(32);
    float result;
    u32 vector_count = dimension >> 3;
    u32 remainder = dimension & 7;
    const float *a = left;
    const float *b = right;
    u32 i;

    /*
     * 主循环展开 2×（16 floats/迭代，两条 ymm 交错），对齐 hnswlib
     * L2SqrSIMD16ExtAVX，提高 ILP 覆盖内存延迟。算术顺序不变，结果逐位一致。
     */
    asm volatile(
        "vxorps %%ymm0, %%ymm0, %%ymm0\n\t"
        "test %[count], %[count]\n\t"
        "jz 3f\n\t"
        "1:\n\t"
        "cmp $2, %[count]\n\t"
        "jb 2f\n\t"
        "vmovups (%[left]), %%ymm1\n\t"
        "vmovups (%[right]), %%ymm2\n\t"
        "vsubps %%ymm2, %%ymm1, %%ymm1\n\t"
        "vmulps %%ymm1, %%ymm1, %%ymm1\n\t"
        "vaddps %%ymm1, %%ymm0, %%ymm0\n\t"
        "vmovups 32(%[left]), %%ymm1\n\t"
        "vmovups 32(%[right]), %%ymm2\n\t"
        "vsubps %%ymm2, %%ymm1, %%ymm1\n\t"
        "vmulps %%ymm1, %%ymm1, %%ymm1\n\t"
        "vaddps %%ymm1, %%ymm0, %%ymm0\n\t"
        "add $64, %[left]\n\t"
        "add $64, %[right]\n\t"
        "sub $2, %[count]\n\t"
        "jnz 1b\n\t"
        "2:\n\t"
        "test %[count], %[count]\n\t"
        "jz 3f\n\t"
        "vmovups (%[left]), %%ymm1\n\t"
        "vmovups (%[right]), %%ymm2\n\t"
        "vsubps %%ymm2, %%ymm1, %%ymm1\n\t"
        "vmulps %%ymm1, %%ymm1, %%ymm1\n\t"
        "vaddps %%ymm1, %%ymm0, %%ymm0\n\t"
        "add $32, %[left]\n\t"
        "add $32, %[right]\n\t"
        "3:\n\t"
        "vmovups %%ymm0, %[lanes]\n\t"
        "vzeroupper\n\t"
        : [left] "+r" (a),
          [right] "+r" (b),
          [count] "+r" (vector_count),
          [lanes] "=m" (lanes)
        :
        : "memory", "cc",
          "ymm0", "ymm1", "ymm2");

    result = vecfs_ann_horizontal_sum_avx(lanes);

    for (i = 0; i < remainder; i++) {
        float delta = a[i] - b[i];

        result += delta * delta;
    }

    return result;
}

float vecfs_ann_ip_avx2(
    const float *left,
    const float *right,
    u32 dimension)
{
    float lanes[8] __aligned(32);
    float result;
    u32 vector_count = dimension >> 3;
    u32 remainder = dimension & 7;
    const float *a = left;
    const float *b = right;
    u32 i;

    /*
     * 主循环展开 2×（16 floats/迭代，两条 ymm 交错），对齐 hnswlib
     * InnerProductSIMD16ExtAVX，提高 ILP 覆盖内存延迟。算术顺序不变，
     * 结果逐位一致。
     */
    asm volatile(
        "vxorps %%ymm0, %%ymm0, %%ymm0\n\t"
        "test %[count], %[count]\n\t"
        "jz 3f\n\t"
        "1:\n\t"
        "cmp $2, %[count]\n\t"
        "jb 2f\n\t"
        "vmovups (%[left]), %%ymm1\n\t"
        "vmovups (%[right]), %%ymm2\n\t"
        "vmulps %%ymm2, %%ymm1, %%ymm1\n\t"
        "vaddps %%ymm1, %%ymm0, %%ymm0\n\t"
        "vmovups 32(%[left]), %%ymm1\n\t"
        "vmovups 32(%[right]), %%ymm2\n\t"
        "vmulps %%ymm2, %%ymm1, %%ymm1\n\t"
        "vaddps %%ymm1, %%ymm0, %%ymm0\n\t"
        "add $64, %[left]\n\t"
        "add $64, %[right]\n\t"
        "sub $2, %[count]\n\t"
        "jnz 1b\n\t"
        "2:\n\t"
        "test %[count], %[count]\n\t"
        "jz 3f\n\t"
        "vmovups (%[left]), %%ymm1\n\t"
        "vmovups (%[right]), %%ymm2\n\t"
        "vmulps %%ymm2, %%ymm1, %%ymm1\n\t"
        "vaddps %%ymm1, %%ymm0, %%ymm0\n\t"
        "add $32, %[left]\n\t"
        "add $32, %[right]\n\t"
        "3:\n\t"
        "vmovups %%ymm0, %[lanes]\n\t"
        "vzeroupper\n\t"
        : [left] "+r" (a),
          [right] "+r" (b),
          [count] "+r" (vector_count),
          [lanes] "=m" (lanes)
        :
        : "memory", "cc",
          "ymm0", "ymm1", "ymm2");

    result = vecfs_ann_horizontal_sum_avx(lanes);

    for (i = 0; i < remainder; i++)
        result += a[i] * b[i];

    return 1.0f - result;
}

#endif
