// SPDX-License-Identifier: GPL-2.0

#include <linux/bitmap.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/overflow.h>
#include <linux/rcupdate.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_cache.h"
#include "../include/vecfs_ann_store.h"

#define VECFS_ANN_CACHE_ALIGNMENT 64U

static size_t vecfs_ann_upper_stride(
    const struct vecfs_ann_snapshot *snapshot)
{
    return ALIGN(
        sizeof(u16) +
            (size_t)snapshot->m * sizeof(u32),
        sizeof(u32));
}

static void vecfs_ann_snapshot_free_upper(
    struct vecfs_ann_snapshot *snapshot)
{
    u32 id;

    if (!snapshot || !snapshot->upper)
        return;

    for (id = VECFS_ANN_FIRST_NODE_ID;
         id < snapshot->capacity;
         id++) {
        if (snapshot->upper[id].memory &&
            refcount_dec_and_test(
                vecfs_ann_upper_block_refs(
                    &snapshot->upper[id])))
            kvfree(snapshot->upper[id].memory);
    }
}

void vecfs_ann_snapshot_free(
    struct vecfs_ann_snapshot *snapshot)
{
    if (!snapshot)
        return;

    vecfs_ann_snapshot_free_upper(
        snapshot);

    bitmap_free(
        snapshot->stale_bitmap);

    bitmap_free(
        snapshot->valid_bitmap);

    vecfs_ann_node_pages_free(snapshot);
    kvfree(snapshot->upper);
    vecfs_ann_level0_free(snapshot);

    snapshot->stale_bitmap = NULL;
    snapshot->valid_bitmap = NULL;

    kfree(snapshot);
}

static struct workqueue_struct *vecfs_ann_free_wq;

static void vecfs_ann_snapshot_free_work(
    struct work_struct *work)
{
    struct vecfs_ann_snapshot *snapshot =
        container_of(
            work,
            struct vecfs_ann_snapshot,
            free_work);

    vecfs_ann_snapshot_free(snapshot);
}

static void vecfs_ann_snapshot_free_rcu(
    struct rcu_head *rhp)
{
    struct vecfs_ann_snapshot *snapshot =
        container_of(
            rhp,
            struct vecfs_ann_snapshot,
            rcu);

    /*
     * 读侧临界区已全部退出（宽限期结束），但 free() 内部 kvfree 可能落到
     * vfree（可睡眠），不能在 RCU 软中断上下文跑，转到 free_wq（进程上下文）。
     */
    INIT_WORK(&snapshot->free_work, vecfs_ann_snapshot_free_work);
    queue_work(vecfs_ann_free_wq, &snapshot->free_work);
}

static void vecfs_ann_snapshot_release(
    struct kref *kref)
{
    struct vecfs_ann_snapshot *snapshot =
        container_of(
            kref,
            struct vecfs_ann_snapshot,
            refcount);

    /*
     * 延迟到 RCU 宽限期后再释放：读侧在 rcu_read_lock 临界区中经
     * rcu_dereference 取到快照后可能仍在用；call_rcu 保证所有已在读侧
     * 临界区内的读者退出前不释放。可睡眠路径（RTANN worker）经 kref_get
     * 单独持有引用，不会受此影响。
     */
    call_rcu(&snapshot->rcu, vecfs_ann_snapshot_free_rcu);
}

void vecfs_ann_snapshot_put(
    struct vecfs_ann_snapshot *snapshot)
{
    if (snapshot)
        kref_put(
            &snapshot->refcount,
            vecfs_ann_snapshot_release);
}

int vecfs_ann_cache_init(void)
{
    vecfs_ann_free_wq =
        alloc_workqueue("vecfs_ann_free", WQ_MEM_RECLAIM, 0);
    if (!vecfs_ann_free_wq)
        return -ENOMEM;
    return 0;
}

void vecfs_ann_cache_exit(void)
{
    /*
     * 顺序：先 rcu_barrier 等所有 call_rcu 回调（它们把 free 排进 free_wq），
     * 再 flush_workqueue 等这些 free 真正跑完，最后销毁。由 vecfs_exit 在
     * rcu_barrier() 之后调用，见 core/vecfs_main.c。
     */
    if (vecfs_ann_free_wq) {
        flush_workqueue(vecfs_ann_free_wq);
        destroy_workqueue(vecfs_ann_free_wq);
        vecfs_ann_free_wq = NULL;
    }
}


static int vecfs_ann_snapshot_init_layout(
    struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_config *config)
{
    size_t offset;

    snapshot->m = config->m;
    snapshot->m0 = config->m0;
    snapshot->max_levels = config->max_levels;

    snapshot->dimension = config->dimension;
    snapshot->vector_bytes = config->vector_bytes;

    offset = sizeof(u32);

    snapshot->level0_links_offset = offset;

    offset += (size_t)snapshot->m0 * sizeof(u32);
    offset = ALIGN(offset, VECFS_ANN_CACHE_ALIGNMENT);

    snapshot->vector_offset = offset;

    offset += snapshot->vector_bytes;
    offset = ALIGN(offset, VECFS_ANN_CACHE_ALIGNMENT);

    if (offset > U32_MAX)
        return -EOVERFLOW;

    snapshot->element_stride = offset;
    return 0;
}

static int vecfs_ann_snapshot_allocate(
    struct vecfs_ann_snapshot *snapshot,
    u32 capacity)
{
    int ret;

    if (!snapshot || !capacity)
        return -EINVAL;

    ret = vecfs_ann_level0_alloc(
        snapshot,
        capacity);
    if (ret)
        return ret;

    snapshot->upper = kvcalloc(
        capacity,
        sizeof(*snapshot->upper),
        GFP_KERNEL);
    if (!snapshot->upper)
        goto fail;

    ret = vecfs_ann_node_pages_alloc(
        snapshot,
        capacity);
    if (ret)
        goto fail;

    snapshot->stale_bitmap =
        bitmap_zalloc(
            capacity,
            GFP_KERNEL);
    if (!snapshot->stale_bitmap)
        goto fail;

    snapshot->valid_bitmap =
        bitmap_zalloc(
            capacity,
            GFP_KERNEL);
    if (!snapshot->valid_bitmap)
        goto fail;

    /*
     * valid 位默认全 1（全部有效）：只有 SHIELD 发布/写路径会清位，
     * 非 SHIELD 环境与旧快照因此不受影响。
     */
    bitmap_fill(
        snapshot->valid_bitmap,
        capacity);

    snapshot->capacity = capacity;
    return 0;

fail:
    bitmap_free(
        snapshot->stale_bitmap);

    bitmap_free(
        snapshot->valid_bitmap);

    vecfs_ann_node_pages_free(snapshot);
    kvfree(snapshot->upper);
    vecfs_ann_level0_free(snapshot);

    snapshot->stale_bitmap = NULL;
    snapshot->valid_bitmap = NULL;
    snapshot->upper = NULL;
    snapshot->capacity = 0;

    return -ENOMEM;
}



static int vecfs_ann_snapshot_load_upper(
    struct vecfs_ann_snapshot *snapshot,
    struct vecfs_ann_store *store,
    u32 node_id,
    u16 node_level,
    const void *disk_object)
{
    const __le16 *disk_counts;
    const __le32 *disk_links;
    size_t upper_stride;
    size_t upper_bytes;
    u8 *memory;
    u16 level;

    if (!node_level)
        return 0;

    upper_stride = vecfs_ann_upper_stride(snapshot);

    if (check_mul_overflow(
            (size_t)node_level,
            upper_stride,
            &upper_bytes))
        return -EOVERFLOW;

    /* 块头 = [refcount_t][层数据...]。 */
    memory = kvzalloc(
        sizeof(refcount_t) + upper_bytes, GFP_KERNEL);
    if (!memory)
        return -ENOMEM;

    snapshot->upper[node_id].memory = memory;

    refcount_set(
        vecfs_ann_upper_block_refs(
            &snapshot->upper[node_id]),
        1);

    disk_counts = (const __le16 *)(
        (const u8 *)disk_object +
        store->layout.upper_counts_offset);

    disk_links = (const __le32 *)(
        (const u8 *)disk_object +
        store->layout.upper_links_offset);

    for (level = 1; level <= node_level; level++) {
        u8 *level_memory =
            (u8 *)vecfs_ann_upper_block_data(
                &snapshot->upper[node_id]) +
            (size_t)(level - 1) * upper_stride;
        u16 *count_ptr = (u16 *)level_memory;
        u32 *links =
            (u32 *)(level_memory + sizeof(u32));
        u16 count;
        u16 i;

        count = le16_to_cpu(disk_counts[level - 1]);
        if (count > snapshot->m)
            return -EUCLEAN;

        *count_ptr = count;

        for (i = 0; i < count; i++) {
            u32 neighbor = le32_to_cpu(
                disk_links[
                    (size_t)(level - 1) *
                        snapshot->m + i]);

            if (!vecfs_ann_snapshot_id_valid(
                    snapshot, neighbor))
                return -EUCLEAN;

            links[i] = neighbor;
        }
    }

    return 0;
}

static int vecfs_ann_snapshot_load_node(
    struct vecfs_ann_snapshot *snapshot,
    struct vecfs_ann_store *store,
    u32 node_id,
    void *disk_object)
{
    struct vecfs_ann_object_disk *disk = disk_object;
    struct vecfs_ann_cache_node *node;
    const __le32 *disk_links;
    void *element;
    u32 *links;
    u16 count;
    u16 level;
    u32 state;
    u16 i;
    int ret;

    ret = vecfs_ann_object_read(
        store->sb,
        node_id,
        disk_object,
        store->layout.stride);
    if (ret)
        return ret;

    state = le32_to_cpu(disk->state);

    if (state != VECFS_ANN_OBJ_LIVE &&
        state != VECFS_ANN_OBJ_DELETED)
        return 0;

    level = le16_to_cpu(disk->level);
    if (level >= snapshot->max_levels)
        return -EUCLEAN;

    count = le16_to_cpu(disk->level0_count);
    if (count > snapshot->m0)
        return -EUCLEAN;

    node = vecfs_ann_node_mutable(snapshot, node_id);
    if (!node)
        return -ENOMEM;

    node->state = state;
    node->level = level;
    node->external_id =
        le64_to_cpu(disk->external_id);

    node->owner_ino =
        le64_to_cpu(disk->owner_ino);
    node->owner_generation =
        le32_to_cpu(disk->owner_generation);
    node->vector_ordinal =
        le32_to_cpu(disk->vector_ordinal);

    node->content_offset =
        le64_to_cpu(disk->content_offset);
    node->content_length =
        le64_to_cpu(disk->content_length);

    node->owner_type =
        le16_to_cpu(disk->owner_type);
    node->flags =
        le16_to_cpu(disk->flags) &
        ~VECFS_ANN_OBJ_F_CONTENT_STALE;

    if (le16_to_cpu(disk->flags) &
        VECFS_ANN_OBJ_F_CONTENT_STALE) {
        __set_bit(
            node_id,
            snapshot->stale_bitmap);

        atomic_inc(
            &snapshot->stale_count);
    }



    element = vecfs_ann_level0_element(
        snapshot, node_id);

    *(u16 *)element = count;

    links = (u32 *)(
        (u8 *)element +
        snapshot->level0_links_offset);

    disk_links = (const __le32 *)(
        (const u8 *)disk_object +
        store->layout.level0_offset);

    for (i = 0; i < count; i++) {
        u32 neighbor =
            le32_to_cpu(disk_links[i]);

        if (!vecfs_ann_snapshot_id_valid(
                snapshot, neighbor))
            return -EUCLEAN;

        links[i] = neighbor;
    }

    memcpy(
        (u8 *)element + snapshot->vector_offset,
        (const u8 *)disk_object +
            store->layout.vector_offset,
        snapshot->vector_bytes);

    ret = vecfs_ann_snapshot_load_upper(
        snapshot,
        store,
        node_id,
        level,
        disk_object);
    if (ret)
        return ret;

    if (state == VECFS_ANN_OBJ_LIVE)
        snapshot->live_count++;
    else
        snapshot->has_deletions = true;

    return 0;
}

struct vecfs_ann_snapshot *
vecfs_ann_snapshot_build(struct super_block *sb)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_store *store;
    struct vecfs_ann_snapshot *snapshot;
    void *disk_object;
    u64 next_node_id;
    u32 id;
    int ret;

    if (!sbi || !sbi->s_ann_store)
        return ERR_PTR(-ENODEV);

    store = sbi->s_ann_store;

    snapshot = kzalloc(sizeof(*snapshot), GFP_KERNEL);
    if (!snapshot)
        return ERR_PTR(-ENOMEM);
    kref_init(
        &snapshot->refcount);
    atomic_set(
        &snapshot->stale_count,
        0);
    atomic_set(
        &snapshot->invalid_count,
        0);

    ret = vecfs_ann_snapshot_init_layout(
        snapshot, &sbi->s_ann.config);
    if (ret)
        goto fail;

    mutex_lock(&sbi->s_ann.lock);

    next_node_id = sbi->s_ann.next_node_id;
    snapshot->entry_node =
        sbi->s_ann.entry_node_id;
    snapshot->max_level =
        sbi->s_ann.config.current_max_level;

    mutex_unlock(&sbi->s_ann.lock);

    if (next_node_id > VECFS_ANN_INVALID_NODE_ID) {
        ret = -EFBIG;
        goto fail;
    }

    ret = vecfs_ann_snapshot_allocate(
        snapshot,
        max_t(u32, (u32)next_node_id, 2));
    if (ret)
        goto fail;

    disk_object = kvmalloc(
        store->layout.stride, GFP_KERNEL);
    if (!disk_object) {
        ret = -ENOMEM;
        goto fail;
    }

    for (id = VECFS_ANN_FIRST_NODE_ID;
         id < snapshot->capacity;
         id++) {
        ret = vecfs_ann_snapshot_load_node(
            snapshot,
            store,
            id,
            disk_object);

        if (ret == -ENOENT ||
            ret == -ENODATA ||
            ret == -EBADMSG) {
            ret = 0;
            continue;
        }

        if (ret)
            goto fail_object;

        if (!(id & 16383))
            schedule();
        else if (!(id & 63))
            cond_resched();
    }

    kvfree(disk_object);

    if (!snapshot->live_count) {
        snapshot->entry_node =
            VECFS_ANN_INVALID_NODE_ID;
        snapshot->max_level = 0;
        return snapshot;
    }

    if (!vecfs_ann_snapshot_id_valid(
            snapshot, snapshot->entry_node) ||
        vecfs_ann_node_at(snapshot, snapshot->entry_node)->state !=
            VECFS_ANN_OBJ_LIVE) {
        ret = -EUCLEAN;
        goto fail;
    }

    return snapshot;

fail_object:
    kvfree(disk_object);
fail:
    vecfs_ann_snapshot_free(snapshot);
    return ERR_PTR(ret);
}

int vecfs_ann_cache_publish(
    struct super_block *sb,
    struct vecfs_ann_snapshot *snapshot)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_cache *cache;
    int ret;

    if (!sbi || !snapshot)
        return -EINVAL;

    cache = sbi->s_ann_cache;
    if (!cache)
        return -ENODEV;

    mutex_lock(&cache->writer_lock);

    ret = vecfs_ann_cache_publish_locked(
        cache, snapshot);

    mutex_unlock(&cache->writer_lock);
    return ret;
}

int vecfs_ann_cache_mount(struct super_block *sb)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *snapshot;

    if (!sbi)
        return -EINVAL;

    cache = kzalloc(sizeof(*cache), GFP_KERNEL);
    if (!cache)
        return -ENOMEM;

    cache->sb = sb;
    cache->async_stale = true;  /* 默认异步 stale 落盘（消融 OFF 由 set_strategy 关） */
    mutex_init(&cache->writer_lock);

    snapshot = vecfs_ann_snapshot_build(sb);
    if (IS_ERR(snapshot)) {
        int ret = PTR_ERR(snapshot);

        mutex_destroy(&cache->writer_lock);
        kfree(cache);
        return ret;
    }

    RCU_INIT_POINTER(cache->snapshot, snapshot);
    sbi->s_ann_cache = cache;

    /*
     * Phase D：延迟 stale 落盘的脏集合位图，与 snapshot 容量对齐。
     * 分配失败不致命——mark 路径退化为同步落盘（见 stale.c）。
     */
    cache->stale_dirty =
        bitmap_zalloc(snapshot->capacity, GFP_KERNEL);
    cache->stale_dirty_bits =
        cache->stale_dirty ? snapshot->capacity : 0;

    pr_info(
        "vecfs: ANN snapshot loaded capacity=%u live=%u "
        "entry=%u level=%u deleted=%u stride=%u\n",
        snapshot->capacity,
        snapshot->live_count,
        snapshot->entry_node,
        snapshot->max_level,
        snapshot->has_deletions,
        snapshot->element_stride);

    return 0;
}

/*
 * 延迟 stale 落盘脏集合随快照容量扩容。mount 时按初始容量分配，但 store 随
 * 导入/派生增长，若不扩容则 node_id >= stale_dirty_bits 恒成立 → 异步 stale
 * 退化为同步落盘（async-stale 消融失效）。仅在 writer_lock 下调用。
 *
 * 失败不致命：保留旧位图，新增 node_id 回退同步落盘（mark_snapshot_stale 的
 * 兜底分支），语义正确性不变。
 */
int vecfs_ann_cache_stale_dirty_grow(
    struct vecfs_ann_cache *cache,
    u32 new_bits)
{
    unsigned long *new_bitmap;
    u32 old_bits;

    if (!cache)
        return -EINVAL;

    lockdep_assert_held(&cache->writer_lock);

    old_bits = cache->stale_dirty_bits;
    if (new_bits <= old_bits)
        return 0;

    new_bitmap = bitmap_zalloc(new_bits, GFP_NOFS);
    if (!new_bitmap)
        return -ENOMEM;

    if (old_bits && cache->stale_dirty)
        bitmap_copy(new_bitmap, cache->stale_dirty, old_bits);

    bitmap_free(cache->stale_dirty);
    cache->stale_dirty = new_bitmap;
    cache->stale_dirty_bits = new_bits;

    return 0;
}

int vecfs_ann_cache_reload(struct super_block *sb)
{
    struct vecfs_ann_snapshot *snapshot;

    snapshot = vecfs_ann_snapshot_build(sb);
    if (IS_ERR(snapshot))
        return PTR_ERR(snapshot);

    return vecfs_ann_cache_publish(sb, snapshot);
}

void vecfs_ann_cache_unmount(struct super_block *sb)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *snapshot;

    if (!sbi)
        return;

    cache = sbi->s_ann_cache;
    if (!cache)
        return;

    mutex_lock(&cache->writer_lock);

    snapshot = rcu_dereference_protected(
        cache->snapshot,
        lockdep_is_held(&cache->writer_lock));

    RCU_INIT_POINTER(cache->snapshot, NULL);
    synchronize_rcu();

    vecfs_ann_snapshot_put(snapshot);

    mutex_unlock(&cache->writer_lock);

    bitmap_free(cache->stale_dirty);
    cache->stale_dirty = NULL;
    cache->stale_dirty_bits = 0;

    mutex_destroy(&cache->writer_lock);

    sbi->s_ann_cache = NULL;
    kfree(cache);
}
