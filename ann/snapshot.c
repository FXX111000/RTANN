// SPDX-License-Identifier: GPL-2.0

#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/overflow.h>
#include <linux/rcupdate.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

#include "../include/vecfs_ann_build.h"
#include "../include/vecfs_ann_cache.h"

#define VECFS_ANN_SNAPSHOT_MIN_CAPACITY 16U

static size_t vecfs_ann_snapshot_upper_stride(
    const struct vecfs_ann_snapshot *snapshot)
{
    return ALIGN(
        sizeof(u16) +
            (size_t)snapshot->m * sizeof(u32),
        sizeof(u32));
}

static u32 vecfs_ann_snapshot_growth_capacity(
    u32 cur_cap,
    u32 required)
{
    u32 capacity;

    capacity = max_t(
        u32,
        cur_cap,
        VECFS_ANN_SNAPSHOT_MIN_CAPACITY);

    while (capacity < required) {
        if (capacity > U32_MAX / 2)
            return required;

        capacity *= 2;
    }

    return capacity;
}

/*
 * chunk_idx 号块实际容纳的节点数：除末块可能收紧外均为 nodes_per_chunk。
 * 末块节点数 = capacity - chunk_idx*nodes_per_chunk。
 */
static u32 vecfs_ann_level0_chunk_nodes(
    const struct vecfs_ann_snapshot *snapshot,
    u32 chunk_idx)
{
    u32 nodes_per_chunk =
        1u << snapshot->level0_chunk_shift;
    u32 start = chunk_idx * nodes_per_chunk;

    if (start >= snapshot->capacity)
        return 0;

    return min_t(u32, nodes_per_chunk,
                 snapshot->capacity - start);
}

/*
 * 仅初始化 level0 分块几何并分配 chunk 结构数组（所有 chunk 的 memory
 * 置 NULL）。供 clone_locked 使用：它只共享/复制部分块，不整表分配。
 */
static int vecfs_ann_level0_init_geometry(
    struct vecfs_ann_snapshot *snapshot,
    u32 capacity)
{
    struct vecfs_ann_level0_chunk *chunks;
    u32 nodes_per_chunk;
    u32 shift;
    u32 mask;
    u32 chunk_count;

    if (!snapshot || !capacity)
        return -EINVAL;

    shift = vecfs_ann_level0_chunk_shift(
        snapshot->element_stride);
    nodes_per_chunk = 1u << shift;
    mask = nodes_per_chunk - 1;

    if (!nodes_per_chunk)
        return -EINVAL;

    chunk_count = (u32)DIV_ROUND_UP(
        (u64)capacity,
        nodes_per_chunk);

    chunks = kvcalloc(
        chunk_count,
        sizeof(*chunks),
        GFP_KERNEL);
    if (!chunks)
        return -ENOMEM;

    snapshot->level0_chunks = chunks;
    snapshot->level0_chunk_shift = shift;
    snapshot->level0_chunk_mask = mask;
    snapshot->level0_chunk_count = chunk_count;

    return 0;
}

int vecfs_ann_level0_alloc(
    struct vecfs_ann_snapshot *snapshot,
    u32 capacity)
{
    struct vecfs_ann_level0_chunk *chunks;
    u32 nodes_per_chunk;
    u32 remaining;
    u32 chunk_count;
    u32 i;
    int ret;

    ret = vecfs_ann_level0_init_geometry(
        snapshot, capacity);
    if (ret)
        return ret;

    chunks = snapshot->level0_chunks;
    nodes_per_chunk = 1u << snapshot->level0_chunk_shift;
    chunk_count = snapshot->level0_chunk_count;

    /*
     * 每块单独分配（last 块按剩余节点数收紧），单块字节数 <= 目标值（4MB），
     * 不再有单次超大 vmalloc。分配失败只回滚已分配的块，不影响调用方。
     */
    remaining = capacity;
    for (i = 0; i < chunk_count; i++) {
        u32 n = min_t(u32, remaining, nodes_per_chunk);

        /* 块头 = [refcount_t][元素数据...]，共享块时共享头部的 refcount。 */
        chunks[i].memory = kvzalloc(
            sizeof(refcount_t) +
                (size_t)n * snapshot->element_stride,
            GFP_KERNEL);
        if (!chunks[i].memory)
            goto fail;
        refcount_set(
            vecfs_ann_level0_chunk_refs(&chunks[i]),
            1);

        remaining -= n;
    }

    return 0;

fail:
    while (i > 0)
        kvfree(chunks[--i].memory);

    kvfree(chunks);
    snapshot->level0_chunks = NULL;
    snapshot->level0_chunk_shift = 0;
    snapshot->level0_chunk_mask = 0;
    snapshot->level0_chunk_count = 0;
    return -ENOMEM;
}

void vecfs_ann_level0_free(
    struct vecfs_ann_snapshot *snapshot)
{
    u32 i;

    if (!snapshot || !snapshot->level0_chunks)
        return;

    for (i = 0;
         i < snapshot->level0_chunk_count;
         i++) {
        if (snapshot->level0_chunks[i].memory &&
            refcount_dec_and_test(
                vecfs_ann_level0_chunk_refs(
                    &snapshot->level0_chunks[i])))
            kvfree(
                snapshot->level0_chunks[i].memory);
    }

    kvfree(snapshot->level0_chunks);

    snapshot->level0_chunks = NULL;
    snapshot->level0_chunk_shift = 0;
    snapshot->level0_chunk_mask = 0;
    snapshot->level0_chunk_count = 0;
}

void *vecfs_ann_level0_element_mutable(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    struct vecfs_ann_level0_chunk *chunk;
    u32 chunk_idx;
    u32 chunk_nodes;
    void *old_memory;
    void *new_memory;
    size_t bytes;

    if (!snapshot || !snapshot->level0_chunks ||
        !vecfs_ann_snapshot_id_valid(snapshot, node_id))
        return NULL;

    chunk_idx = node_id >> snapshot->level0_chunk_shift;
    chunk = &snapshot->level0_chunks[chunk_idx];

    if (!chunk->memory)
        return NULL;

    if (refcount_read(
            vecfs_ann_level0_chunk_refs(chunk)) > 1) {
        /* 被多个快照共享：COW 整块，递减旧块引用后换新块。 */
        chunk_nodes =
            vecfs_ann_level0_chunk_nodes(snapshot, chunk_idx);
        bytes = (size_t)chunk_nodes * snapshot->element_stride;

        new_memory = kvmalloc(
            sizeof(refcount_t) + bytes, GFP_KERNEL);
        if (!new_memory)
            return NULL;

        memcpy(new_memory, chunk->memory,
               sizeof(refcount_t) + bytes);

        old_memory = chunk->memory;
        chunk->memory = new_memory;
        refcount_set(
            vecfs_ann_level0_chunk_refs(chunk), 1);

        if (refcount_dec_and_test(
                (refcount_t *)old_memory))
            kvfree(old_memory);
    }

    return vecfs_ann_level0_chunk_data(chunk) +
           (size_t)(node_id & snapshot->level0_chunk_mask) *
               snapshot->element_stride;
}

void *vecfs_ann_upper_level_mutable(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id,
    u16 level)
{
    struct vecfs_ann_upper_block *block;
    void *old_memory;
    void *new_memory;
    size_t stride;
    size_t bytes;

    if (!snapshot || !snapshot->upper ||
        !vecfs_ann_snapshot_id_valid(snapshot, node_id) ||
        !level)
        return NULL;

    block = &snapshot->upper[node_id];

    if (!block->memory)
        return NULL;

    stride = ALIGN(
        sizeof(u16) + (size_t)snapshot->m * sizeof(u32),
        sizeof(u32));

    if (refcount_read(
            vecfs_ann_upper_block_refs(block)) > 1) {
        /* 被多个快照共享：COW 整个上层块，递减旧块引用后换新块。 */
        if (check_mul_overflow(
                (size_t)vecfs_ann_node_at(snapshot, node_id)->level,
                stride,
                &bytes))
            return NULL;

        new_memory = kvmalloc(
            sizeof(refcount_t) + bytes, GFP_KERNEL);
        if (!new_memory)
            return NULL;

        memcpy(new_memory, block->memory,
               sizeof(refcount_t) + bytes);

        old_memory = block->memory;
        block->memory = new_memory;
        refcount_set(
            vecfs_ann_upper_block_refs(block), 1);

        if (refcount_dec_and_test(
                (refcount_t *)old_memory))
            kvfree(old_memory);
    }

    return (u8 *)vecfs_ann_upper_block_data(block) +
           (size_t)(level - 1) * stride;
}

/* 仅分配扁平页表（所有页指针置 NULL），供 clone 共享页使用。 */
static int vecfs_ann_node_pages_init(
    struct vecfs_ann_snapshot *snapshot,
    u32 capacity)
{
    struct vecfs_ann_node_page **pages;
    u32 page_count;

    page_count = (u32)DIV_ROUND_UP(
        (u64)capacity, VECFS_ANN_NODE_PAGE_SIZE);

    pages = kvcalloc(
        page_count, sizeof(*pages), GFP_KERNEL);
    if (!pages)
        return -ENOMEM;

    snapshot->node_pages = pages;
    snapshot->node_page_count = page_count;
    return 0;
}

int vecfs_ann_node_pages_alloc(
    struct vecfs_ann_snapshot *snapshot,
    u32 capacity)
{
    struct vecfs_ann_node_page **pages;
    u32 page_count;
    u32 i;
    int ret;

    ret = vecfs_ann_node_pages_init(snapshot, capacity);
    if (ret)
        return ret;

    pages = snapshot->node_pages;
    page_count = snapshot->node_page_count;

    /*
     * 每页整块分配（末页未用槽保持零态），页尾的 refcount 用于 clone
     * 结构共享。分配失败只回滚已分配的页。
     */
    for (i = 0; i < page_count; i++) {
        pages[i] = kvzalloc(sizeof(*pages[i]), GFP_KERNEL);
        if (!pages[i])
            goto fail;
        refcount_set(&pages[i]->refcount, 1);
    }

    return 0;

fail:
    while (i > 0)
        kvfree(pages[--i]);

    kvfree(pages);
    snapshot->node_pages = NULL;
    snapshot->node_page_count = 0;
    return -ENOMEM;
}

void vecfs_ann_node_pages_free(
    struct vecfs_ann_snapshot *snapshot)
{
    struct vecfs_ann_node_page **pages = snapshot->node_pages;
    u32 i;

    if (!pages)
        return;

    for (i = 0; i < snapshot->node_page_count; i++) {
        if (pages[i] &&
            refcount_dec_and_test(&pages[i]->refcount))
            kvfree(pages[i]);
    }

    kvfree(pages);
    snapshot->node_pages = NULL;
    snapshot->node_page_count = 0;
}

struct vecfs_ann_cache_node *
vecfs_ann_node_mutable(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    struct vecfs_ann_node_page **pages;
    struct vecfs_ann_node_page *page;
    struct vecfs_ann_node_page *new_page;
    u32 page_idx;

    if (!snapshot || !snapshot->node_pages ||
        !vecfs_ann_snapshot_id_valid(snapshot, node_id))
        return NULL;

    pages = snapshot->node_pages;
    page_idx = node_id >> VECFS_ANN_NODE_PAGE_SHIFT;
    page = pages[page_idx];

    if (!page) {
        /* 新容量区域（clone 只建页表未分配页）：首次写入按需分配零态页。 */
        page = kvzalloc(sizeof(*page), GFP_KERNEL);
        if (!page)
            return NULL;
        refcount_set(&page->refcount, 1);
        pages[page_idx] = page;
    }

    if (refcount_read(&page->refcount) > 1) {
        /* 被多个快照共享：COW 整页，递减旧页引用后换新页。 */
        new_page = kvmalloc(sizeof(*new_page), GFP_KERNEL);
        if (!new_page)
            return NULL;

        memcpy(new_page, page, sizeof(*new_page));
        refcount_set(&new_page->refcount, 1);

        pages[page_idx] = new_page;

        if (refcount_dec_and_test(&page->refcount))
            kvfree(page);
    }

    return &pages[page_idx]->nodes[
        node_id & VECFS_ANN_NODE_PAGE_MASK];
}

static int vecfs_ann_snapshot_allocate_clone_arrays(
    struct vecfs_ann_snapshot *destination,
    u32 capacity,
    bool shared_level0)
{
    int ret;

    if (!destination || !capacity)
        return -EINVAL;

    /*
     * shared_level0=true（clone）：level0 只建几何，块内存由 clone_locked
     * 之后逐块共享/复制，避免 O(N) 整表分配。
     * shared_level0=false（empty-like）：整表分配零态块。
     */
    if (shared_level0)
        ret = vecfs_ann_level0_init_geometry(
            destination,
            capacity);
    else
        ret = vecfs_ann_level0_alloc(
            destination,
            capacity);
    if (ret)
        return ret;

    destination->upper = kvcalloc(
        capacity,
        sizeof(*destination->upper),
        GFP_KERNEL);
    if (!destination->upper)
        goto fail;

    if (shared_level0)
        ret = vecfs_ann_node_pages_init(
            destination,
            capacity);
    else
        ret = vecfs_ann_node_pages_alloc(
            destination,
            capacity);
    if (ret)
        goto fail;

    destination->stale_bitmap =
        bitmap_zalloc(
            capacity,
            GFP_KERNEL);
    if (!destination->stale_bitmap)
        goto fail;

    destination->valid_bitmap =
        bitmap_zalloc(
            capacity,
            GFP_KERNEL);
    if (!destination->valid_bitmap)
        goto fail;

    /*
     * 新容量区域的 valid 位默认全 1（有效）。
     *
     * 重挂载一致性模型（此处是「默认全有效」这一决定的依据）：
     *
     * 派生正确性只依赖 CONTENT_STALE 一个持久化标志——它经 rewrite.c 的
     * effective_flags 折叠进磁盘 object 的 flags 落盘，重挂载时由
     * load_node 读回并重建 stale_bitmap。因此「写后派生未完成」这一
     * 过期/无效状态跨重挂载保持，正确性不受影响。
     *
     * valid_bitmap（CONTENT_INVALID / 查询端 Valid(x,f)）是纯内存结构：
     * 它只在 SHIELD 条件发布的「派生中→提交」临界区内瞬时置 0，表示
     * 「尚未提交的新向量」；重挂载发生在派生静默期（无并发 worker），
     * 不存在「提交到一半的语义根」，故重建时重置全 1 是安全的，不会
     * 泄露无效结果。shield_version 与块级复用表同样是 in-memory 状态，
     * 丢失只影响去抖/复用效率，不影响正确性。
     */
    bitmap_fill(
        destination->valid_bitmap,
        capacity);

    destination->capacity = capacity;
    return 0;

fail:
    bitmap_free(
        destination->stale_bitmap);

    bitmap_free(
        destination->valid_bitmap);

    vecfs_ann_node_pages_free(destination);
    kvfree(destination->upper);
    vecfs_ann_level0_free(destination);

    destination->stale_bitmap = NULL;
    destination->valid_bitmap = NULL;
    destination->upper = NULL;
    destination->capacity = 0;

    return -ENOMEM;
}


static int vecfs_ann_snapshot_clone_upper_node(
    struct vecfs_ann_snapshot *destination,
    const struct vecfs_ann_snapshot *source,
    u32 node_id,
    bool copy_full)
{
    const struct vecfs_ann_cache_node *node;
    void *memory;
    size_t stride;
    size_t bytes;

    node = vecfs_ann_node_at(source, node_id);

    if (!node->level ||
        !source->upper[node_id].memory)
        return 0;

    if (!copy_full) {
        /*
         * 上层块结构共享：两快照指向同一块头、共享头部的 refcount。
         * 写侧经 vecfs_ann_upper_level_mutable() 触发 COW。
         */
        destination->upper[node_id].memory =
            source->upper[node_id].memory;

        refcount_inc(
            vecfs_ann_upper_block_refs(
                &source->upper[node_id]));

        return 0;
    }

    /* 全拷贝消融：整块重新分配并复制，不与 source 共享任何上层数据。 */
    stride = ALIGN(
        sizeof(u16) + (size_t)destination->m * sizeof(u32),
        sizeof(u32));
    if (check_mul_overflow(
            (size_t)node->level, stride, &bytes))
        return -EOVERFLOW;

    memory = kvmalloc(
        sizeof(refcount_t) + bytes, GFP_KERNEL);
    if (!memory)
        return -ENOMEM;

    memcpy(memory, source->upper[node_id].memory,
           sizeof(refcount_t) + bytes);

    destination->upper[node_id].memory = memory;
    refcount_set(
        vecfs_ann_upper_block_refs(
            &destination->upper[node_id]), 1);

    return 0;
}

static struct vecfs_ann_snapshot *
__vecfs_ann_snapshot_clone_locked(
    const struct vecfs_ann_snapshot *source,
    u32 minimum_capacity,
    bool copy_full)
{
    struct vecfs_ann_snapshot *destination;
    u32 capacity;
    u32 id;
    int ret;

    if (!source)
        return ERR_PTR(-EINVAL);

    if (minimum_capacity >=
        VECFS_ANN_INVALID_NODE_ID)
        return ERR_PTR(-ENOSPC);

    capacity =
        vecfs_ann_snapshot_growth_capacity(
            source->capacity,
            minimum_capacity);

    destination = kzalloc(
        sizeof(*destination),
        GFP_KERNEL);
    if (!destination)
        return ERR_PTR(-ENOMEM);

    /*
     * 复制普通布局字段，随后清除所有拥有所有权的指针。
     * atomic_t 不依赖结构赋值结果，后面显式设置。
     */
    *destination = *source;

    destination->capacity = 0;
    destination->level0_chunks = NULL;
    destination->level0_chunk_shift = 0;
    destination->level0_chunk_mask = 0;
    destination->level0_chunk_count = 0;
    destination->upper = NULL;
    destination->node_pages = NULL;
    destination->node_page_count = 0;
    destination->stale_bitmap = NULL;
    destination->valid_bitmap = NULL;

    /*
     * *destination = *source 会把 source 的 kref 一并拷贝过来；这里显式
     * 重置为 1，使派生快照拥有独立的引用计数（writer 只发布派生快照，
     * 其引用计数归发布路径统一管理）。
     */
    kref_init(&destination->refcount);

    atomic_set(
        &destination->stale_count,
        atomic_read(
            &source->stale_count));

    atomic_set(
        &destination->invalid_count,
        atomic_read(
            &source->invalid_count));

    ret =
        vecfs_ann_snapshot_allocate_clone_arrays(
            destination,
            capacity,
            true);
    if (ret)
        goto fail;

    /*
     * level0 分块结构共享 + COW 骨架：source 与 destination 的
     * element_stride 相同（clone 复用布局），分块几何一致。
     *
     * 满块（src_nodes == dst_nodes）直接共享（refcount++），不再 O(N)
     * 整块 memcpy；仅「源末块收紧」（src_nodes < dst_nodes，源容量非
     * nodes_per_chunk 整数倍）与新容量区域需要新分配（复制存活部分、
     * 余下零态）。写侧经 vecfs_ann_level0_element_mutable() 触发 COW。
     */
    {
        u32 chunk;

        for (chunk = 0;
             chunk < destination->level0_chunk_count;
             chunk++) {
            u32 src_nodes =
                vecfs_ann_level0_chunk_nodes(source, chunk);
            u32 dst_nodes =
                vecfs_ann_level0_chunk_nodes(destination, chunk);

            if (!copy_full &&
                chunk < source->level0_chunk_count &&
                src_nodes == dst_nodes) {
                destination->level0_chunks[chunk].memory =
                    source->level0_chunks[chunk].memory;

                refcount_inc(
                    vecfs_ann_level0_chunk_refs(
                        &source->level0_chunks[chunk]));
            } else {
                destination->level0_chunks[chunk].memory =
                    kvzalloc(
                        sizeof(refcount_t) +
                            (size_t)dst_nodes *
                                destination->element_stride,
                        GFP_KERNEL);
                if (!destination->level0_chunks[chunk].memory)
                    goto fail;

                refcount_set(
                    vecfs_ann_level0_chunk_refs(
                        &destination->level0_chunks[chunk]),
                    1);

                if (src_nodes)
                    memcpy(
                        vecfs_ann_level0_chunk_data(
                            &destination->level0_chunks[chunk]),
                        vecfs_ann_level0_chunk_data(
                            &source->level0_chunks[chunk]),
                        (size_t)src_nodes *
                            destination->element_stride);
            }
        }
    }

    /*
     * nodes[] 分页结构共享：页整块共享（refcount++），不再 O(N) memcpy。
     * 所有页（含末页未用零槽）均等长，直接共享即可；超出 source 的新容量
     * 页单独分配零态页（新节点写入时首次用 node_mutable 也会按需分配，此处
     * 预分配以保持与原 kvcalloc 一致的全零态语义）。
     */
    {
        u32 page;

        for (page = 0;
             page < destination->node_page_count;
             page++) {
            if (page < source->node_page_count && !copy_full) {
                destination->node_pages[page] =
                    source->node_pages[page];
                refcount_inc(
                    &source->node_pages[page]->refcount);
            } else if (page < source->node_page_count) {
                /* 全拷贝消融：整页重新分配并复制，不与 source 共享。 */
                destination->node_pages[page] =
                    kvmalloc(sizeof(struct vecfs_ann_node_page),
                             GFP_KERNEL);
                if (!destination->node_pages[page])
                    goto fail;
                memcpy(destination->node_pages[page],
                       source->node_pages[page],
                       sizeof(struct vecfs_ann_node_page));
                refcount_set(
                    &destination->node_pages[page]->refcount, 1);
            } else {
                destination->node_pages[page] =
                    kvzalloc(sizeof(struct vecfs_ann_node_page),
                             GFP_KERNEL);
                if (!destination->node_pages[page])
                    goto fail;
                refcount_set(
                    &destination->node_pages[page]->refcount, 1);
            }
        }
    }

    bitmap_copy(
        destination->stale_bitmap,
        source->stale_bitmap,
        source->capacity);

    bitmap_copy(
        destination->valid_bitmap,
        source->valid_bitmap,
        source->capacity);

    for (id = VECFS_ANN_FIRST_NODE_ID;
         id < source->capacity;
         id++) {
        ret =
            vecfs_ann_snapshot_clone_upper_node(
                destination,
                source,
                id,
                copy_full);
        if (ret)
            goto fail;
    }

    return destination;

fail:
    vecfs_ann_snapshot_free(
        destination);

    return ERR_PTR(ret);
}

struct vecfs_ann_snapshot *
vecfs_ann_snapshot_clone_locked(
    const struct vecfs_ann_snapshot *source,
    u32 minimum_capacity)
{
    return __vecfs_ann_snapshot_clone_locked(
        source, minimum_capacity, false);
}

struct vecfs_ann_snapshot *
vecfs_ann_snapshot_clone_locked_copy(
    const struct vecfs_ann_snapshot *source,
    u32 minimum_capacity,
    bool copy_full)
{
    return __vecfs_ann_snapshot_clone_locked(
        source, minimum_capacity, copy_full);
}

int vecfs_ann_snapshot_update_binding(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id,
    u64 content_offset,
    u64 content_length,
    u16 flags)
{
    struct vecfs_ann_cache_node *node;
    bool was_stale;
    bool is_stale;

    if (!snapshot)
        return -EINVAL;

    if (!vecfs_ann_snapshot_id_valid(
            snapshot,
            node_id))
        return -ENOENT;

    if ((flags &
         VECFS_ANN_OBJ_F_HAS_CONTENT_RANGE) &&
        content_offset >
            U64_MAX - content_length)
        return -EOVERFLOW;

    node = vecfs_ann_node_mutable(snapshot, node_id);
    if (!node)
        return -ENOMEM;

    was_stale =
        vecfs_ann_snapshot_content_stale(
            snapshot,
            node_id);

    is_stale =
        flags &
        VECFS_ANN_OBJ_F_CONTENT_STALE;

    node->content_offset =
        content_offset;
    node->content_length =
        content_length;

    node->flags =
        flags &
        ~VECFS_ANN_OBJ_F_CONTENT_STALE;

    if (is_stale && !was_stale) {
        set_bit(
            node_id,
            snapshot->stale_bitmap);
        atomic_inc(
            &snapshot->stale_count);
    } else if (!is_stale && was_stale) {
        clear_bit(
            node_id,
            snapshot->stale_bitmap);

        if (atomic_dec_return(
                &snapshot->stale_count) < 0) {
            WARN_ON_ONCE(1);
            atomic_set(
                &snapshot->stale_count,
                0);
        }
    }

    return 0;
}


int vecfs_ann_snapshot_init_node(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id,
    u16 level,
    const void *vector,
    const struct vecfs_ann_insert_request *request)
{
    struct vecfs_ann_cache_node *node;
    void *element;
    size_t upper_bytes;
    size_t upper_stride;

    if (!snapshot || !request || !vector)
        return -EINVAL;

    if (!vecfs_ann_snapshot_id_valid(
            snapshot, node_id))
        return -EINVAL;

    if (level >= snapshot->max_levels)
        return -EINVAL;

    node = vecfs_ann_node_mutable(snapshot, node_id);
    if (!node)
        return -ENOMEM;

    if (node->state == VECFS_ANN_OBJ_LIVE)
        return -EEXIST;

    element = vecfs_ann_level0_element_mutable(
        snapshot, node_id);

    if (!element)
        return -ENOMEM;

    memset(element, 0, snapshot->element_stride);

    memcpy(
        (u8 *)element + snapshot->vector_offset,
        vector,
        snapshot->vector_bytes);

    node->external_id = request->external_id;

    node->owner_ino = request->owner_ino;
    node->owner_generation =
        request->owner_generation;
    node->vector_ordinal =
        request->vector_ordinal;

    node->content_offset =
        request->content_offset;
    node->content_length =
        request->content_length;

    node->owner_type =
    request->owner_type;

    node->flags =
        request->flags &
        ~VECFS_ANN_OBJ_F_CONTENT_STALE;

    if (request->flags &
        VECFS_ANN_OBJ_F_CONTENT_STALE)
        __set_bit(
            node_id,
            snapshot->stale_bitmap);
    else
        __clear_bit(
            node_id,
            snapshot->stale_bitmap);

    /*
     * SHIELD 插入即无效：新节点在进入查询快照前即置 valid_bitmap = 0，
     * 与 invalid_count 同步。仅 SHIELD 原子发布路径设置 start_invalid，
     * 其余插入路径（ioctl/import/rebuild）保持默认有效。
     */
    if (request->start_invalid) {
        __clear_bit(
            node_id,
            snapshot->valid_bitmap);

        atomic_inc(
            &snapshot->invalid_count);
    }

    node->state = VECFS_ANN_OBJ_LIVE;
    node->level = level;

    if (level) {
        upper_stride =
            vecfs_ann_snapshot_upper_stride(snapshot);

        if (check_mul_overflow(
                (size_t)level,
                upper_stride,
                &upper_bytes))
            return -EOVERFLOW;

        /* 块头 = [refcount_t][层数据...]。 */
        snapshot->upper[node_id].memory =
            kvzalloc(
                sizeof(refcount_t) + upper_bytes,
                GFP_KERNEL);

        if (!snapshot->upper[node_id].memory)
            return -ENOMEM;

        refcount_set(
            vecfs_ann_upper_block_refs(
                &snapshot->upper[node_id]),
            1);
    }

    snapshot->live_count++;
    return 0;
}

int vecfs_ann_cache_publish_locked(
    struct vecfs_ann_cache *cache,
    struct vecfs_ann_snapshot *snapshot)
{
    struct vecfs_ann_snapshot *old;

    if (!cache || !snapshot)
        return -EINVAL;

    lockdep_assert_held(&cache->writer_lock);

    old = rcu_dereference_protected(
        cache->snapshot,
        lockdep_is_held(&cache->writer_lock));

    rcu_assign_pointer(cache->snapshot, snapshot);

    /*
     * 旧快照经 kref_put → call_rcu 延迟回收，读侧无需同步宽限期。
     * 不再在此处 synchronize_rcu()：它每文件阻塞一次完整宽限期，且
     * 独占 writer_lock 堵住其它派生，是派生吞吐的主要瓶颈之一。
     */
    vecfs_ann_snapshot_put(old);

    /*
     * 延迟 stale 落盘脏集合随容量增长扩容：否则 mount 时按初始（空）容量
     * 分配的位图覆盖不了后续导入/派生产生的 node_id，async-stale 恒退化为
     * 同步落盘。失败不致命（保留旧位图，新增 node_id 回退同步）。
     */
    vecfs_ann_cache_stale_dirty_grow(cache, snapshot->capacity);

    return 0;
}

struct vecfs_ann_snapshot *
vecfs_ann_snapshot_create_empty_like(
    const struct vecfs_ann_snapshot *source,
    u32 capacity)
{
    struct vecfs_ann_snapshot *snapshot;
    int ret;

    if (!source ||
        capacity <=
            VECFS_ANN_FIRST_NODE_ID ||
        capacity >=
            VECFS_ANN_INVALID_NODE_ID)
        return ERR_PTR(-EINVAL);

    snapshot = kzalloc(
        sizeof(*snapshot),
        GFP_KERNEL);
    if (!snapshot)
        return ERR_PTR(-ENOMEM);

    kref_init(&snapshot->refcount);

    snapshot->capacity = 0;
    snapshot->live_count = 0;

    snapshot->entry_node =
        VECFS_ANN_INVALID_NODE_ID;

    snapshot->max_level = 0;

    snapshot->m = source->m;
    snapshot->m0 = source->m0;

    snapshot->max_levels =
        source->max_levels;

    snapshot->dimension =
        source->dimension;

    snapshot->vector_bytes =
        source->vector_bytes;

    snapshot->level0_links_offset =
        source->level0_links_offset;

    snapshot->vector_offset =
        source->vector_offset;

    snapshot->element_stride =
        source->element_stride;

    snapshot->has_deletions = false;

    atomic_set(
        &snapshot->stale_count,
        0);

    atomic_set(
        &snapshot->invalid_count,
        0);

    ret =
        vecfs_ann_snapshot_allocate_clone_arrays(
            snapshot,
            capacity,
            false);
    if (ret) {
        vecfs_ann_snapshot_free(
            snapshot);

        return ERR_PTR(ret);
    }

    return snapshot;
}

