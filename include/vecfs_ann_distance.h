#ifndef _VECFS_ANN_DISTANCE_H
#define _VECFS_ANN_DISTANCE_H

#include <linux/types.h>

struct super_block;

enum vecfs_ann_simd_mode {
    VECFS_ANN_SIMD_SCALAR = 0,
    VECFS_ANN_SIMD_SSE2 = 1,
    VECFS_ANN_SIMD_AVX2 = 2,
    VECFS_ANN_SIMD_AVX512 = 3,
};

typedef float (*vecfs_ann_distance_raw_fn)(
    const float *left,
    const float *right,
    u32 dimension);

struct vecfs_ann_distance {
    u32 dimension;
    u16 distance_type;
    u16 simd_mode;

    vecfs_ann_distance_raw_fn raw_fn;
};

struct vecfs_ann_distance_session {
    const struct vecfs_ann_distance *distance;
    bool active;
};

int vecfs_ann_distance_init(struct super_block *sb);
void vecfs_ann_distance_destroy(struct super_block *sb);

void vecfs_ann_distance_begin(
    struct vecfs_ann_distance_session *session,
    const struct vecfs_ann_distance *distance);

void vecfs_ann_distance_end(
    struct vecfs_ann_distance_session *session);

static __always_inline float
vecfs_ann_distance_calculate(
    const struct vecfs_ann_distance_session *session,
    const float *left,
    const float *right)
{
    return session->distance->raw_fn(
        left,
        right,
        session->distance->dimension);
}

const char *vecfs_ann_simd_name(u16 simd_mode);

float vecfs_ann_l2_scalar(
    const float *left,
    const float *right,
    u32 dimension);

float vecfs_ann_ip_scalar(
    const float *left,
    const float *right,
    u32 dimension);

#ifdef CONFIG_X86_64
float vecfs_ann_l2_sse2(
    const float *left,
    const float *right,
    u32 dimension);

float vecfs_ann_ip_sse2(
    const float *left,
    const float *right,
    u32 dimension);

float vecfs_ann_l2_avx2(
    const float *left,
    const float *right,
    u32 dimension);

float vecfs_ann_ip_avx2(
    const float *left,
    const float *right,
    u32 dimension);
#endif

#endif
