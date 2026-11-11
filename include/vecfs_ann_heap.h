#ifndef _VECFS_ANN_HEAP_H
#define _VECFS_ANN_HEAP_H

#include <linux/types.h>

enum vecfs_ann_heap_type {
    VECFS_ANN_HEAP_MIN = 0,
    VECFS_ANN_HEAP_MAX = 1,
};

struct vecfs_ann_heap_item {
    float distance;
    u32 node_id;
};

struct vecfs_ann_heap {
    struct vecfs_ann_heap_item *items;
    u32 size;
    u32 capacity;
    u8 type;
};

/* ------------------------------------------------------------------ */
/* 冷路径：初始化 / 销毁 / 排序拷贝 / 扩容（在 heap.c，含 kvmalloc/sort） */
/* ------------------------------------------------------------------ */

int vecfs_ann_heap_init(
    struct vecfs_ann_heap *heap,
    u32 capacity,
    enum vecfs_ann_heap_type type,
    gfp_t gfp);

void vecfs_ann_heap_destroy(
    struct vecfs_ann_heap *heap);

int vecfs_ann_heap_copy_sorted(
    const struct vecfs_ann_heap *heap,
    struct vecfs_ann_heap_item *output,
    u32 capacity,
    u32 *count);

int vecfs_ann_heap_reserve(
    struct vecfs_ann_heap *heap,
    u32 capacity,
    gfp_t gfp);

/* ------------------------------------------------------------------ */
/* 热路径：top/push/pop/replace_top 与 sift 辅助全部内联。              */
/* 与 hnswlib 的 std::priority_queue（编译器内联）对齐，避免检索热循环 */
/* 每次迭代 5~6 次跨 TU 函数调用。                                      */
/* ------------------------------------------------------------------ */

static __always_inline bool
vecfs_ann_heap_before(
    const struct vecfs_ann_heap *heap,
    const struct vecfs_ann_heap_item *left,
    const struct vecfs_ann_heap_item *right)
{
    if (left->distance == right->distance) {
        if (heap->type == VECFS_ANN_HEAP_MIN)
            return left->node_id < right->node_id;

        return left->node_id > right->node_id;
    }

    if (heap->type == VECFS_ANN_HEAP_MIN)
        return left->distance < right->distance;

    return left->distance > right->distance;
}

static __always_inline void
vecfs_ann_heap_swap(
    struct vecfs_ann_heap_item *left,
    struct vecfs_ann_heap_item *right)
{
    struct vecfs_ann_heap_item temporary = *left;

    *left = *right;
    *right = temporary;
}

static __always_inline void
vecfs_ann_heap_sift_up(
    struct vecfs_ann_heap *heap,
    u32 index)
{
    while (index) {
        u32 parent = (index - 1) >> 1;

        if (!vecfs_ann_heap_before(
                heap,
                &heap->items[index],
                &heap->items[parent]))
            break;

        vecfs_ann_heap_swap(
            &heap->items[index],
            &heap->items[parent]);

        index = parent;
    }
}

static __always_inline void
vecfs_ann_heap_sift_down(
    struct vecfs_ann_heap *heap,
    u32 index)
{
    for (;;) {
        u32 left = index * 2 + 1;
        u32 right = left + 1;
        u32 selected = index;

        if (left < heap->size &&
            vecfs_ann_heap_before(
                heap,
                &heap->items[left],
                &heap->items[selected]))
            selected = left;

        if (right < heap->size &&
            vecfs_ann_heap_before(
                heap,
                &heap->items[right],
                &heap->items[selected]))
            selected = right;

        if (selected == index)
            break;

        vecfs_ann_heap_swap(
            &heap->items[index],
            &heap->items[selected]);

        index = selected;
    }
}

static __always_inline void
vecfs_ann_heap_reset(
    struct vecfs_ann_heap *heap)
{
    if (heap)
        heap->size = 0;
}

static __always_inline bool
vecfs_ann_heap_empty(
    const struct vecfs_ann_heap *heap)
{
    return !heap || !heap->size;
}

static __always_inline bool
vecfs_ann_heap_full(
    const struct vecfs_ann_heap *heap)
{
    return heap &&
           heap->size == heap->capacity;
}

static __always_inline u32
vecfs_ann_heap_size(
    const struct vecfs_ann_heap *heap)
{
    return heap ? heap->size : 0;
}

static __always_inline const struct vecfs_ann_heap_item *
vecfs_ann_heap_top(
    const struct vecfs_ann_heap *heap)
{
    if (!heap || !heap->size)
        return NULL;

    return &heap->items[0];
}

static __always_inline int
vecfs_ann_heap_push(
    struct vecfs_ann_heap *heap,
    u32 node_id,
    float distance)
{
    u32 index;

    if (!heap || !heap->items)
        return -EINVAL;

    if (heap->size >= heap->capacity)
        return -ENOSPC;

    index = heap->size++;

    heap->items[index].node_id = node_id;
    heap->items[index].distance = distance;

    vecfs_ann_heap_sift_up(heap, index);
    return 0;
}

static __always_inline int
vecfs_ann_heap_pop(
    struct vecfs_ann_heap *heap,
    struct vecfs_ann_heap_item *item)
{
    if (!heap || !heap->items || !heap->size)
        return -ENOENT;

    if (item)
        *item = heap->items[0];

    heap->size--;

    if (heap->size) {
        heap->items[0] =
            heap->items[heap->size];

        vecfs_ann_heap_sift_down(heap, 0);
    }

    return 0;
}

static __always_inline int
vecfs_ann_heap_replace_top(
    struct vecfs_ann_heap *heap,
    u32 node_id,
    float distance)
{
    if (!heap || !heap->items || !heap->size)
        return -ENOENT;

    heap->items[0].node_id = node_id;
    heap->items[0].distance = distance;

    vecfs_ann_heap_sift_down(heap, 0);
    return 0;
}

#endif
