// SPDX-License-Identifier: GPL-2.0

#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/sort.h>

#include "../include/vecfs_ann_heap.h"

/*
 * 热路径的 top/push/pop/replace_top 与 sift 辅助已移到
 * include/vecfs_ann_heap.h 内联（与 hnswlib 的 std::priority_queue 对齐）。
 * 本文件只保留冷路径：init / destroy / copy_sorted / reserve。
 */

int vecfs_ann_heap_init(
    struct vecfs_ann_heap *heap,
    u32 capacity,
    enum vecfs_ann_heap_type type,
    gfp_t gfp)
{
    if (!heap || !capacity)
        return -EINVAL;

    if (type != VECFS_ANN_HEAP_MIN &&
        type != VECFS_ANN_HEAP_MAX)
        return -EINVAL;

    memset(heap, 0, sizeof(*heap));

    heap->items = kvmalloc_array(
        capacity,
        sizeof(*heap->items),
        gfp);
    if (!heap->items)
        return -ENOMEM;

    heap->capacity = capacity;
    heap->type = type;

    return 0;
}

void vecfs_ann_heap_destroy(
    struct vecfs_ann_heap *heap)
{
    if (!heap)
        return;

    kvfree(heap->items);
    memset(heap, 0, sizeof(*heap));
}

static int vecfs_ann_sorted_item_compare(
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

int vecfs_ann_heap_copy_sorted(
    const struct vecfs_ann_heap *heap,
    struct vecfs_ann_heap_item *output,
    u32 capacity,
    u32 *count)
{
    if (!heap || !output || !count)
        return -EINVAL;

    if (capacity < heap->size)
        return -ENOSPC;

    memcpy(
        output,
        heap->items,
        heap->size * sizeof(*output));

    sort(
        output,
        heap->size,
        sizeof(*output),
        vecfs_ann_sorted_item_compare,
        NULL);

    *count = heap->size;
    return 0;
}

int vecfs_ann_heap_reserve(
    struct vecfs_ann_heap *heap,
    u32 capacity,
    gfp_t gfp)
{
    struct vecfs_ann_heap_item *items;

    if (!heap || !heap->items)
        return -EINVAL;

    if (capacity <= heap->capacity)
        return 0;

    if (capacity > U32_MAX /
                   sizeof(*heap->items))
        return -EOVERFLOW;

    items = kvmalloc_array(
        capacity,
        sizeof(*items),
        gfp);
    if (!items)
        return -ENOMEM;

    memcpy(
        items,
        heap->items,
        (size_t)heap->size *
            sizeof(*items));

    kvfree(heap->items);

    heap->items = items;
    heap->capacity = capacity;

    return 0;
}
