// SPDX-License-Identifier: GPL-2.0

#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/module.h>

#ifdef CONFIG_X86_64
#include <asm/cpufeature.h>
#include <asm/fpu/api.h>
#endif

#include "../../include/vecfs.h"
#include "../../include/vecfs_ann_distance.h"
#include "../../include/vecfs_disk.h"

static bool vecfs_ann_enable_avx2 = true;

module_param_named(
    ann_avx2,
    vecfs_ann_enable_avx2,
    bool,
    0644);

static bool vecfs_ann_has_avx2(void)
{
#ifdef CONFIG_X86_64
    return vecfs_ann_enable_avx2 &&
           boot_cpu_has(X86_FEATURE_AVX) &&
           boot_cpu_has(X86_FEATURE_AVX2);
#else
    return false;
#endif
}

int vecfs_ann_distance_init(struct super_block *sb)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_distance *distance;

    if (!sbi)
        return -EINVAL;

    if (sbi->s_ann.config.element_type !=
        VECFS_ANN_ELEM_FP32)
        return -EOPNOTSUPP;

    distance = &sbi->s_ann_distance;

    memset(distance, 0, sizeof(*distance));

    distance->dimension =
        sbi->s_ann.config.dimension;
    distance->distance_type =
        sbi->s_ann.config.distance_type;

    switch (distance->distance_type) {
    case VECFS_ANN_DIST_L2:
#ifdef CONFIG_X86_64
        if (vecfs_ann_has_avx2()) {
            distance->raw_fn = vecfs_ann_l2_avx2;
            distance->simd_mode =
                VECFS_ANN_SIMD_AVX2;
        } else {
            distance->raw_fn = vecfs_ann_l2_sse2;
            distance->simd_mode =
                VECFS_ANN_SIMD_SSE2;
        }
#else
        distance->raw_fn = vecfs_ann_l2_scalar;
        distance->simd_mode =
            VECFS_ANN_SIMD_SCALAR;
#endif
        break;

    case VECFS_ANN_DIST_IP:
#ifdef CONFIG_X86_64
        if (vecfs_ann_has_avx2()) {
            distance->raw_fn = vecfs_ann_ip_avx2;
            distance->simd_mode =
                VECFS_ANN_SIMD_AVX2;
        } else {
            distance->raw_fn = vecfs_ann_ip_sse2;
            distance->simd_mode =
                VECFS_ANN_SIMD_SSE2;
        }
#else
        distance->raw_fn = vecfs_ann_ip_scalar;
        distance->simd_mode =
            VECFS_ANN_SIMD_SCALAR;
#endif
        break;

    default:
        return -EOPNOTSUPP;
    }

    return 0;
}

void vecfs_ann_distance_destroy(struct super_block *sb)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);

    if (sbi)
        memset(
            &sbi->s_ann_distance,
            0,
            sizeof(sbi->s_ann_distance));
}

void vecfs_ann_distance_begin(
    struct vecfs_ann_distance_session *session,
    const struct vecfs_ann_distance *distance)
{
    memset(session, 0, sizeof(*session));

    if (!distance || !distance->raw_fn)
        return;

#ifdef CONFIG_X86_64
    kernel_fpu_begin();
#endif

    session->distance = distance;
    session->active = true;
}

void vecfs_ann_distance_end(
    struct vecfs_ann_distance_session *session)
{
    if (!session || !session->active)
        return;

    session->active = false;

#ifdef CONFIG_X86_64
    kernel_fpu_end();
#endif

    session->distance = NULL;
}

const char *vecfs_ann_simd_name(u16 mode)
{
    switch (mode) {
    case VECFS_ANN_SIMD_SCALAR:
        return "scalar";
    case VECFS_ANN_SIMD_SSE2:
        return "sse2";
    case VECFS_ANN_SIMD_AVX2:
        return "avx2";
    case VECFS_ANN_SIMD_AVX512:
        return "avx512";
    default:
        return "unknown";
    }
}
