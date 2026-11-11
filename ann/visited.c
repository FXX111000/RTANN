// SPDX-License-Identifier: GPL-2.0

#include <linux/errno.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

#include "../include/vecfs_ann_visited.h"

int vecfs_ann_visited_init(
    struct vecfs_ann_visited *visited,
    u32 capacity,
    gfp_t gfp)
{
    if (!visited || !capacity)
        return -EINVAL;

    memset(visited, 0, sizeof(*visited));

    visited->generation = kvcalloc(
        capacity,
        sizeof(*visited->generation),
        gfp);
    if (!visited->generation)
        return -ENOMEM;

    visited->capacity = capacity;
    visited->cur_gen = 1;

    return 0;
}

void vecfs_ann_visited_destroy(
    struct vecfs_ann_visited *visited)
{
    if (!visited)
        return;

    kvfree(visited->generation);
    memset(visited, 0, sizeof(*visited));
}

void vecfs_ann_visited_reset(
    struct vecfs_ann_visited *visited)
{
    if (!visited || !visited->generation)
        return;

    visited->cur_gen++;

    if (unlikely(!visited->cur_gen)) {
        memset(
            visited->generation,
            0,
            (size_t)visited->capacity *
                sizeof(*visited->generation));

        visited->cur_gen = 1;
    }
}
