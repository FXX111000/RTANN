#ifndef _VECFS_ANN_VISITED_H
#define _VECFS_ANN_VISITED_H

#include <linux/types.h>

struct vecfs_ann_visited {
    u16 *generation;
    u32 capacity;
    u16 cur_gen;
};

int vecfs_ann_visited_init(
    struct vecfs_ann_visited *visited,
    u32 capacity,
    gfp_t gfp);

void vecfs_ann_visited_destroy(
    struct vecfs_ann_visited *visited);

void vecfs_ann_visited_reset(
    struct vecfs_ann_visited *visited);

static __always_inline bool
vecfs_ann_visited_test_and_set(
    struct vecfs_ann_visited *visited,
    u32 node_id)
{
    u16 *entry = &visited->generation[node_id];
    u16 cur_gen = visited->cur_gen;

    if (likely(READ_ONCE(*entry) == cur_gen))
        return true;

    WRITE_ONCE(*entry, cur_gen);
    return false;
}

#endif
