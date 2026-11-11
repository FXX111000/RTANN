// SPDX-License-Identifier: GPL-2.0

#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/sort.h>

#include "../include/vecfs_ann_build.h"
#include "../include/vecfs_ann_cache.h"
#include "../include/vecfs_ann_distance.h"

static int vecfs_ann_build_item_compare(
    const void *left,
    const void *right)
{
    const struct vecfs_ann_heap_item *a = left;
    const struct vecfs_ann_heap_item *b = right;

    if (a->distance < b->distance)
        return -1;

    if (a->distance > b->distance)
        return 1;

    if (a->node_id < b->node_id)
        return -1;

    if (a->node_id > b->node_id)
        return 1;

    return 0;
}

int vecfs_ann_select_neighbors_heuristic(
    const struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_distance_session *distance,
    struct vecfs_ann_build_context *context,
    const struct vecfs_ann_heap_item *candidates,
    u32 candidate_count,
    u32 maximum,
    u32 *selected_count)
{
    u32 count = 0;
    u32 i;

    if (!snapshot || !distance ||
        !context || !selected_count)
        return -EINVAL;

    if (candidate_count >
        context->temporary_capacity)
        return -ENOSPC;

    if (maximum >
        context->temporary_capacity)
        return -ENOSPC;

    memcpy(
        context->ordered,
        candidates,
        (size_t)candidate_count *
            sizeof(*candidates));

    sort(
        context->ordered,
        candidate_count,
        sizeof(*context->ordered),
        vecfs_ann_build_item_compare,
        NULL);

    /*
     * hnswlib getNeighborsByHeuristic2：
     *
     * candidate 只有在它与所有已选择节点之间的距离都不小于
     * candidate 到 query/new node 的距离时才被接受。
     */
    for (i = 0;
         i < candidate_count && count < maximum;
         i++) {
        const struct vecfs_ann_heap_item *candidate =
            &context->ordered[i];
        const float *candidate_vector;
        bool good = true;
        u32 j;

        candidate_vector =
            vecfs_ann_snapshot_vector(
                snapshot,
                candidate->node_id);

        for (j = 0; j < count; j++) {
            const struct vecfs_ann_heap_item *selected =
                &context->selected[j];
            float inter_distance;

            inter_distance =
                vecfs_ann_distance_calculate(
                    distance,
                    candidate_vector,
                    vecfs_ann_snapshot_vector(
                        snapshot,
                        selected->node_id));

            if (inter_distance <
                candidate->distance) {
                good = false;
                break;
            }
        }

        if (good)
            context->selected[count++] =
                *candidate;
    }

    *selected_count = count;
    return 0;
}
