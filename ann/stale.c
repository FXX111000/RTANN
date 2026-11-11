// SPDX-License-Identifier: GPL-2.0

#include <linux/bitmap.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/overflow.h>
#include <linux/rcupdate.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_cache.h"
#include "../include/vecfs_ann_indirect.h"
#include "../include/vecfs_ann_stale.h"
#include "../include/vecfs_ann_store.h"
#include "../include/vecfs_shield.h"

struct vecfs_ann_stale_mapping {
    u32 ordinal;
    u32 node_id;
};

struct vecfs_ann_stale_snapshot {
    struct vecfs_ann_stale_mapping *entries;
    size_t count;
    size_t capacity;
};

enum vecfs_ann_stale_mode {
    VECFS_ANN_STALE_RANGE,
    VECFS_ANN_STALE_AFTER,
};

struct vecfs_ann_stale_request {
    enum vecfs_ann_stale_mode mode;

    u64 offset;
    u64 length;
    u64 end;
};

static int vecfs_ann_stale_collect(
    struct inode *inode,
    u32 ordinal,
    u32 node_id,
    void *private_data)
{
    struct vecfs_ann_stale_snapshot *snapshot =
        private_data;

    struct vecfs_ann_stale_mapping *entries;
    size_t capacity;

    if (snapshot->count ==
        snapshot->capacity) {
        capacity =
            snapshot->capacity ?
                snapshot->capacity * 2 :
                16;

        if (capacity >
            SIZE_MAX / sizeof(*entries))
            return -EOVERFLOW;

        entries = krealloc_array(
            snapshot->entries,
            capacity,
            sizeof(*entries),
            GFP_NOFS);
        if (!entries)
            return -ENOMEM;

        snapshot->entries = entries;
        snapshot->capacity = capacity;
    }

    snapshot->entries[
        snapshot->count].ordinal =
        ordinal;

    snapshot->entries[
        snapshot->count].node_id =
        node_id;

    snapshot->count++;

    return 0;
}

static bool vecfs_ann_ranges_overlap(
    u64 first_offset,
    u64 first_length,
    u64 second_offset,
    u64 second_length)
{
    u64 first_end;
    u64 second_end;

    if (!first_length ||
        !second_length)
        return false;

    if (check_add_overflow(
            first_offset,
            first_length,
            &first_end))
        first_end = U64_MAX;

    if (check_add_overflow(
            second_offset,
            second_length,
            &second_end))
        second_end = U64_MAX;

    return first_offset < second_end &&
           second_offset < first_end;
}

static bool vecfs_ann_node_affected(
    const struct vecfs_ann_cache_node *node,
    const struct vecfs_ann_stale_request *request)
{
    u16 flags;

    if (!node || !request)
        return false;

    if (node->owner_type !=
        VECFS_ANN_OWNER_FILE)
        return false;

    if (node->state !=
            VECFS_ANN_OBJ_LIVE &&
        node->state !=
            VECFS_ANN_OBJ_DELETED)
        return false;

    flags = READ_ONCE(node->flags);

    if (!(flags &
          VECFS_ANN_OBJ_F_HAS_CONTENT_RANGE))
        return false;

    if (!node->content_length)
        return false;

    switch (request->mode) {
    case VECFS_ANN_STALE_RANGE:
        return vecfs_ann_ranges_overlap(
            node->content_offset,
            node->content_length,
            request->offset,
            request->length);

    case VECFS_ANN_STALE_AFTER:
        /*
         * 文件缩小到 request->offset 时，只要绑定范围有任意部分
         * 位于新 EOF 之后，就需要重新生成向量。
         */
        if (node->content_offset >=
            request->offset)
            return true;

        return node->content_length >
               request->offset -
                   node->content_offset;

    default:
        return false;
    }
}

static int vecfs_ann_mark_snapshot_stale(
    struct inode *inode,
    const struct vecfs_ann_stale_snapshot *mappings,
    const struct vecfs_ann_stale_request *request)
{
    struct vecfs_sb_info *sbi =
        VECFS_SB(inode->i_sb);

    struct vecfs_ann_cache *cache =
        sbi->s_ann_cache;

    struct vecfs_ann_snapshot *snapshot;
    size_t i;
    int ret = 0;

    if (!cache)
        return -ENODEV;

    /*
     * stale bitmap 本身允许原子更新，但 writer_lock 仍然需要持有：
     *
     * 1. 防止 snapshot clone 时丢失并发设置的 stale bit；
     * 2. 防止 rebuild/import 在更新磁盘 object 期间切换 arena；
     * 3. 保证 node metadata 与 inode 映射来自同一个 snapshot epoch。
     *
     * ANN 查询不获取 writer_lock。
     */
    mutex_lock(&cache->writer_lock);

    snapshot = rcu_dereference_protected(
        cache->snapshot,
        lockdep_is_held(
            &cache->writer_lock));

    if (!snapshot) {
        ret = -ENODEV;
        goto out_unlock;
    }

    for (i = 0;
         i < mappings->count;
         i++) {
        const struct vecfs_ann_stale_mapping *mapping =
            &mappings->entries[i];

        const struct vecfs_ann_cache_node *node;
        u32 node_id = mapping->node_id;

        if (!vecfs_ann_snapshot_id_valid(
                snapshot,
                node_id))
            continue;

        node = vecfs_ann_node_at(snapshot, node_id);

        if (node->owner_type !=
                VECFS_ANN_OWNER_FILE ||
            node->owner_ino !=
                inode->i_ino ||
            node->owner_generation !=
                inode->i_generation ||
            node->vector_ordinal !=
                mapping->ordinal)
            continue;

        if (!vecfs_ann_node_affected(
                node,
                request))
            continue;

        if (vecfs_ann_snapshot_content_stale(
                snapshot,
                node_id))
            continue;

        /*
         * 先向查询结果发布 stale 状态（内存位），磁盘 CONTENT_STALE 标志
         * 延迟落盘（Phase D）：内存位是查询正确性的权威，磁盘标志只用于
         * 崩溃恢复。节点入脏集合，卸载/派生提交时统一 drain。
         */
        if (test_and_set_bit(
                node_id,
                snapshot->stale_bitmap))
            continue;

        atomic_inc(
            &snapshot->stale_count);

        if (!cache->async_stale ||
            !cache->stale_dirty ||
            node_id >= cache->stale_dirty_bits) {
            /*
             * 脏集合不可用（未分配/容量不足）：同步落盘兜底，保证磁盘
             * 标志不丢（写入失败则拒绝本次写，与旧语义一致）。
             */
            ret = vecfs_ann_object_update_flags(
                inode->i_sb,
                node_id,
                0,
                VECFS_ANN_OBJ_F_CONTENT_STALE);
            if (ret)
                goto out_unlock;
            atomic64_inc(&cache->stale_sync_writes);
            continue;
        }

        /*
         * 首条延迟落盘前置 superblock dirty（durable），确保崩溃后能检测到
         * 「有未落盘的 stale」。置 dirty 失败则本节点同步落盘兜底、脏集合
         * 保持空（此时 superblock 仍 clean，与磁盘一致）。
         */
        if (!sbi->s_stale_dirty_marked) {
            ret = vecfs_super_set_clean(
                inode->i_sb, false);
            if (ret) {
                ret = vecfs_ann_object_update_flags(
                    inode->i_sb,
                    node_id,
                    0,
                    VECFS_ANN_OBJ_F_CONTENT_STALE);
                if (ret)
                    goto out_unlock;
                atomic64_inc(&cache->stale_sync_writes);
                continue;
            }
            sbi->s_stale_dirty_marked = true;
        }

        __set_bit(node_id, cache->stale_dirty);
        atomic64_inc(&cache->stale_deferred);

    }

out_unlock:
    mutex_unlock(&cache->writer_lock);
    return ret;
}

static int vecfs_ann_mark_content_stale_internal(
    struct inode *inode,
    const struct vecfs_ann_stale_request *request)
{
    struct vecfs_ann_stale_snapshot mappings = {};
    int ret;

    if (!inode || !request)
        return -EINVAL;

    if (!S_ISREG(inode->i_mode))
        return 0;

    /*
     * 隐藏 ANN arena inode 不属于用户文件，不执行内容绑定检查。
     */
    if (inode->i_ino ==
            VECFS_ANN_ARENA_INO ||
        inode->i_ino ==
            VECFS_ANN_REBUILD_ARENA_INO)
        return 0;

    ret = vecfs_ann_for_each_node(
        inode,
        vecfs_ann_stale_collect,
        &mappings);
    if (ret)
        goto out;

    if (!mappings.count) {
        ret = 0;
        goto out;
    }

    ret = vecfs_ann_mark_snapshot_stale(
        inode,
        &mappings,
        request);

out:
    kvfree(mappings.entries);
    return ret;
}

int vecfs_ann_mark_content_stale(
    struct inode *inode,
    u64 offset,
    u64 length)
{
    struct vecfs_ann_stale_request request;

    if (!inode)
        return -EINVAL;

    if (!length)
        return 0;

    memset(&request, 0, sizeof(request));

    request.mode =
        VECFS_ANN_STALE_RANGE;

    request.offset = offset;
    request.length = length;

    if (check_add_overflow(
            offset,
            length,
            &request.end))
        request.end = U64_MAX;

    return vecfs_ann_mark_content_stale_internal(
        inode,
        &request);
}

int vecfs_ann_mark_content_stale_after(
    struct inode *inode,
    u64 new_size)
{
    struct vecfs_ann_stale_request request;

    if (!inode)
        return -EINVAL;

    memset(&request, 0, sizeof(request));

    request.mode =
        VECFS_ANN_STALE_AFTER;

    request.offset = new_size;

    return vecfs_ann_mark_content_stale_internal(
        inode,
        &request);
}

int vecfs_ann_clear_file_vector_stale(
    struct inode *inode,
    u32 ordinal,
    u64 expected_version)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *snapshot;
    const struct vecfs_ann_cache_node *node;
    u32 node_id;
    int ret;

    if (!inode)
        return -EINVAL;

    if (!S_ISREG(inode->i_mode))
        return -EINVAL;

    ret = vecfs_ann_get_node(
        inode,
        ordinal,
        &node_id);
    if (ret)
        return ret;

    sbi = VECFS_SB(inode->i_sb);
    cache = sbi->s_ann_cache;

    if (!cache)
        return -ENODEV;

    mutex_lock(
        &cache->writer_lock);

    snapshot =
        rcu_dereference_protected(
            cache->snapshot,
            lockdep_is_held(
                &cache->writer_lock));

    if (!snapshot ||
        !vecfs_ann_snapshot_id_valid(
            snapshot,
            node_id)) {
        ret = -ENOENT;
        goto out_unlock;
    }

    node = vecfs_ann_node_at(snapshot, node_id);

    if (node->owner_type !=
            VECFS_ANN_OWNER_FILE ||
        node->owner_ino !=
            inode->i_ino ||
        node->owner_generation !=
            inode->i_generation ||
        node->vector_ordinal !=
            ordinal) {
        ret = -ESTALE;
        goto out_unlock;
    }

    /*
     * 版本复核（复用路径的关键防护）：写路径先在 fs->lock 内推进 shield_version、
     * 再在 writer_lock 内 mark_content_stale，二者按程序顺序；此处持 writer_lock
     * 与 mark_content_stale 串行。若版本已推进，说明有并发写已（或即将）重新
     * 保守标记该向量 stale，此时清除会把它重新激活成可查询的过期向量，必须拒绝。
     */
    if (vecfs_shield_version(inode) != expected_version) {
        ret = -ESTALE;
        goto out_unlock;
    }

    if (!vecfs_ann_snapshot_content_stale(
            snapshot,
            node_id)) {
        ret = 0;
        goto out_unlock;
    }

    ret = vecfs_ann_object_update_flags(
        inode->i_sb,
        node_id,
        VECFS_ANN_OBJ_F_CONTENT_STALE,
        0);
    if (ret)
        goto out_unlock;

    if (test_and_clear_bit(
            node_id,
            snapshot->stale_bitmap)) {
        if (atomic_dec_return(
                &snapshot->stale_count) < 0) {
            WARN_ON_ONCE(1);

            atomic_set(
                &snapshot->stale_count,
                0);
        }
    }

    /*
     * Phase D：磁盘标志已同步清除，该节点不再需要 drain（否则卸载时会被
     * 误判为「仍 stale」而重新置 CONTENT_STALE）。
     */
    if (cache->stale_dirty &&
        node_id < cache->stale_dirty_bits)
        clear_bit(node_id, cache->stale_dirty);

out_unlock:
    mutex_unlock(
        &cache->writer_lock);

    return ret;
}

/*
 * SHIELD 查询端 Valid(x,f) 的 valid 位维护。
 *
 * valid 位只存在于内存（不落盘），重挂载后随 in-memory 派生状态重建。
 * mark_node_invalid 清位（置无效），mark_node_valid 置位（恢复有效），
 * 二者都在 writer_lock 下原子更新 invalid_count，与 stale 位机制同构。
 */
static int vecfs_ann_set_node_validity(
    struct inode *inode,
    u32 ordinal,
    bool valid)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *snapshot;
    const struct vecfs_ann_cache_node *node;
    u32 node_id;
    int ret;

    if (!inode)
        return -EINVAL;

    if (!S_ISREG(inode->i_mode))
        return -EINVAL;

    ret = vecfs_ann_get_node(
        inode,
        ordinal,
        &node_id);
    if (ret)
        return ret;

    sbi = VECFS_SB(inode->i_sb);
    cache = sbi->s_ann_cache;

    if (!cache)
        return -ENODEV;

    mutex_lock(
        &cache->writer_lock);

    snapshot =
        rcu_dereference_protected(
            cache->snapshot,
            lockdep_is_held(
                &cache->writer_lock));

    if (!snapshot ||
        !vecfs_ann_snapshot_id_valid(
            snapshot,
            node_id)) {
        ret = -ENOENT;
        goto out_unlock;
    }

    node = vecfs_ann_node_at(snapshot, node_id);

    if (node->owner_type !=
            VECFS_ANN_OWNER_FILE ||
        node->owner_ino !=
            inode->i_ino ||
        node->owner_generation !=
            inode->i_generation ||
        node->vector_ordinal !=
            ordinal) {
        ret = -ESTALE;
        goto out_unlock;
    }

    if (valid) {
        /* 置位 valid_bitmap（恢复有效）；若原为 0 则递减 invalid_count。 */
        if (!test_and_set_bit(
                node_id,
                snapshot->valid_bitmap)) {
            if (atomic_dec_return(
                    &snapshot->invalid_count) < 0) {
                WARN_ON_ONCE(1);
                atomic_set(
                    &snapshot->invalid_count,
                    0);
            }
        }
    } else {
        /* 清位 valid_bitmap（置无效）；若原为 1 则递增 invalid_count。 */
        if (test_and_clear_bit(
                node_id,
                snapshot->valid_bitmap))
            atomic_inc(
                &snapshot->invalid_count);
    }

    ret = 0;

out_unlock:
    mutex_unlock(
        &cache->writer_lock);

    return ret;
}

int vecfs_ann_mark_node_valid(
    struct inode *inode,
    u32 ordinal)
{
    return vecfs_ann_set_node_validity(
        inode,
        ordinal,
        true);
}

int vecfs_ann_mark_node_invalid(
    struct inode *inode,
    u32 ordinal)
{
    return vecfs_ann_set_node_validity(
        inode,
        ordinal,
        false);
}

/*
 * Phase D：把 stale_dirty 脏集合中「仍 stale」的节点 CONTENT_STALE 落盘。
 *
 * 对每个脏节点，若当前快照已非 stale（被 clear / 被删除 / 重建丢弃），跳过
 * 并清脏位；否则 object_update_flags 置 CONTENT_STALE。仅在 writer_lock 下
 * 调用，保证与 mark 的 __set_bit 互斥。
 */
int vecfs_ann_stale_drain(
    struct super_block *sb)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *snapshot;
    u32 node_id;
    int ret = 0;
    int count = 0;

    if (!sb)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi || !sbi->s_ann_cache)
        return -ENODEV;

    cache = sbi->s_ann_cache;

    mutex_lock(&cache->writer_lock);

    snapshot = rcu_dereference_protected(
        cache->snapshot,
        lockdep_is_held(&cache->writer_lock));
    if (!snapshot) {
        ret = -ENODEV;
        goto out_unlock;
    }

    if (!cache->stale_dirty)
        goto out_unlock;

    for_each_set_bit(node_id, cache->stale_dirty,
                     cache->stale_dirty_bits) {
        if (node_id >= snapshot->capacity ||
            !vecfs_ann_snapshot_content_stale(
                snapshot, node_id)) {
            clear_bit(node_id, cache->stale_dirty);
            continue;
        }

        ret = vecfs_ann_object_update_flags(
            sb, node_id, 0,
            VECFS_ANN_OBJ_F_CONTENT_STALE);
        if (ret)
            goto out_unlock;

        clear_bit(node_id, cache->stale_dirty);
        count++;
    }

    ret = count;

out_unlock:
    mutex_unlock(&cache->writer_lock);
    return ret;
}

/*
 * Phase D 崩溃恢复：把所有 file-bound LIVE 向量保守失效。
 *
 * 非干净卸载时，磁盘可能缺少「已置内存 stale 位、但 CONTENT_STALE 尚未
 * 落盘」的标志。无法廉价区分受影响节点，故保守地把所有 file-bound LIVE
 * 向量置 stale（同步落盘），并交由 SHIELD 侧（set_mode → schedule rederive）
 * 重派生恢复。失效一次后磁盘与内存一致，代价计入 R5。
 */
int vecfs_ann_recover_file_vectors(
    struct super_block *sb)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *snapshot;
    const struct vecfs_ann_cache_node *node;
    u32 node_id;
    int ret;
    int count = 0;

    if (!sb)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi || !sbi->s_ann_cache)
        return -ENODEV;

    cache = sbi->s_ann_cache;

    mutex_lock(&cache->writer_lock);

    snapshot = rcu_dereference_protected(
        cache->snapshot,
        lockdep_is_held(&cache->writer_lock));
    if (!snapshot) {
        ret = -ENODEV;
        goto out_unlock;
    }

    for (node_id = VECFS_ANN_FIRST_NODE_ID;
         node_id < snapshot->capacity;
         node_id++) {
        node = vecfs_ann_node_at(snapshot, node_id);

        if (node->state != VECFS_ANN_OBJ_LIVE)
            continue;

        if (node->owner_type != VECFS_ANN_OWNER_FILE)
            continue;

        /*
         * 同步落盘 CONTENT_STALE（恢复是一次性操作，非写热路径），
         * 保证崩溃后重挂也不丢失。
         */
        ret = vecfs_ann_object_update_flags(
            sb, node_id, 0,
            VECFS_ANN_OBJ_F_CONTENT_STALE);
        if (ret)
            goto out_unlock;

        if (!test_and_set_bit(
                node_id,
                snapshot->stale_bitmap))
            atomic_inc(&snapshot->stale_count);

        count++;
    }

    ret = count;

out_unlock:
    mutex_unlock(&cache->writer_lock);
    return ret;
}

