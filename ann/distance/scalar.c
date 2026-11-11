// SPDX-License-Identifier: GPL-2.0

#include <linux/types.h>

#include "../../include/vecfs_ann_distance.h"

float vecfs_ann_l2_scalar(
    const float *left,
    const float *right,
    u32 dimension)
{
    float distance = 0.0f;
    u32 i;

    for (i = 0; i < dimension; i++) {
        float delta = left[i] - right[i];

        distance += delta * delta;
    }

    return distance;
}

float vecfs_ann_ip_scalar(
    const float *left,
    const float *right,
    u32 dimension)
{
    float product = 0.0f;
    u32 i;

    for (i = 0; i < dimension; i++)
        product += left[i] * right[i];

    /*
     * 与 hnswlib 的 InnerProductDistance 对齐。
     */
    return 1.0f - product;
}
