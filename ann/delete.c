// SPDX-License-Identifier: GPL-2.0

#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/rcupdate.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_indirect.h"
#include "../include/vecfs_ann_cache.h"
#include "../include/vecfs_ann_label.h"
#include "../include/vecfs_ann_meta.h"
#include "../include/vecfs_ann_mutation.h"
#include "../include/vecfs_ann_store.h"

static int vecfs_ann_check_access(
    const struct vecfs_ann_cache_node *node,
    enum vecfs_ann_access_class access)
{
    switch (access) {
    case VECFS_ANN_ACCESS_STANDALONE:
        if (node->owner_type ==
                VECFS_ANN_OWNER_FILE)
            return -EPERM;
        return 0;

    case VECFS_ANN_ACCESS_FILE:
        if (node->owner_type !=
                VECFS_ANN_OWNER_FILE)
            return -EPERM;
        return 0;

    case VECFS_ANN_ACCESS_INTERNAL:
        return 0;

    default:
        return -EINVAL;
    }
}

bool vecfs_ann_snapshot_has_deletions(
    const struct vecfs_ann_snapshot *snapshot)
{
    u32 id;

    for (id = VECFS_ANN_FIRST_NODE_ID;
         id < snapshot->capacity;
         id++) {
        if (vecfs_ann_node_at(snapshot, id)->state ==
            VECFS_ANN_OBJ_DELETED)
            return true;
    }

    return false;
}

static int vecfs_ann_set_deleted_locked(
    struct super_block *sb,
    u32 node_id,
    bool deleted,
    enum vecfs_ann_access_class access)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_cache *cache =
        sbi->s_ann_cache;
    struct vecfs_ann_snapshot *old;
    struct vecfs_ann_snapshot *next;
    const struct vecfs_ann_cache_node *node;
    int ret;

    old = rcu_dereference_protected(
        cache->snapshot,
        lockdep_is_held(&cache->writer_lock));

    if (!old ||
        !vecfs_ann_snapshot_id_valid(
            old, node_id))
        return -ENOENT;

    node = vecfs_ann_node_at(old, node_id);

    ret = vecfs_ann_check_access(node, access);
    if (ret)
        return ret;

    if (deleted) {
        if (node->state ==
            VECFS_ANN_OBJ_DELETED)
            return -EALREADY;

        if (node->state !=
            VECFS_ANN_OBJ_LIVE)
            return -ENOENT;
    } else {
        if (node->state ==
            VECFS_ANN_OBJ_LIVE)
            return -EALREADY;

        if (node->state !=
            VECFS_ANN_OBJ_DELETED)
            return -ENOENT;
    }

    next = vecfs_ann_snapshot_clone_locked(
        old,
        old->capacity);
    if (IS_ERR(next))
        return PTR_ERR(next);

    if (deleted) {
        ret = vecfs_ann_object_mark_deleted(
            sb, node_id);
        if (ret)
            goto fail;

        vecfs_ann_node_mutable(next, node_id)->state =
            VECFS_ANN_OBJ_DELETED;

        if (next->live_count)
            next->live_count--;

        next->has_deletions = true;
    } else {
        ret = vecfs_ann_object_unmark_deleted(
            sb, node_id);
        if (ret)
            goto fail;

        vecfs_ann_node_mutable(next, node_id)->state =
            VECFS_ANN_OBJ_LIVE;

        next->live_count++;

        next->has_deletions =
            vecfs_ann_snapshot_has_deletions(next);
    }

    /*
     * markDelete 与 hnswlib 一致：不改变图连接和 entry point。
     */
    ret = vecfs_ann_meta_commit(sb, true);
    if (ret)
        goto fail_rollback_disk;

    ret = vecfs_ann_cache_publish_locked(
        cache, next);
    if (ret)
        goto fail_rollback_disk;

    return 0;

fail_rollback_disk:
    /*
     * 尽力恢复磁盘状态。失败时 mount scan 会以磁盘对象状态为准。
     */
    if (deleted)
        vecfs_ann_object_unmark_deleted(
            sb, node_id);
    else
        vecfs_ann_object_mark_deleted(
            sb, node_id);

fail:
    vecfs_ann_snapshot_free(next);
    return ret;
}

int vecfs_ann_mark_delete_by_label(
    struct super_block *sb,
    u64 external_id,
    enum vecfs_ann_access_class access)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_cache *cache;
    u32 node_id;
    int ret;

    if (!sbi || !sbi->s_ann_cache)
        return -ENODEV;

    cache = sbi->s_ann_cache;

    mutex_lock(&cache->writer_lock);

    ret = vecfs_ann_label_lookup_locked(
        sb,
        external_id,
        &node_id);
    if (!ret)
        ret = vecfs_ann_set_deleted_locked(
            sb,
            node_id,
            true,
            access);

    mutex_unlock(&cache->writer_lock);
    return ret;
}

int vecfs_ann_unmark_delete_by_label(
    struct super_block *sb,
    u64 external_id,
    enum vecfs_ann_access_class access)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_cache *cache;
    u32 node_id;
    int ret;

    if (!sbi || !sbi->s_ann_cache)
        return -ENODEV;

    cache = sbi->s_ann_cache;

    mutex_lock(&cache->writer_lock);

    ret = vecfs_ann_label_lookup_locked(
        sb,
        external_id,
        &node_id);
    if (!ret)
        ret = vecfs_ann_set_deleted_locked(
            sb,
            node_id,
            false,
            access);

    mutex_unlock(&cache->writer_lock);
    return ret;
}

int vecfs_ann_file_mark_delete(
    struct inode *inode,
    u32 ordinal)
{
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *snapshot;
    u32 node_id;
    u64 external_id;
    int ret;

    if (!inode || !S_ISREG(inode->i_mode))
        return -EINVAL;

    ret = vecfs_ann_get_node(
        inode,
        ordinal,
        &node_id);
    if (ret)
        return ret;

    cache = VECFS_SB(inode->i_sb)->s_ann_cache;
    if (!cache)
        return -ENODEV;

    mutex_lock(&cache->writer_lock);

    snapshot = rcu_dereference_protected(
        cache->snapshot,
        lockdep_is_held(&cache->writer_lock));

    if (!snapshot ||
        !vecfs_ann_snapshot_id_valid(
            snapshot, node_id)) {
        ret = -ENOENT;
        goto out;
    }

    if (vecfs_ann_node_at(snapshot, node_id)->owner_type !=
            VECFS_ANN_OWNER_FILE ||
        vecfs_ann_node_at(snapshot, node_id)->owner_ino !=
            inode->i_ino ||
        vecfs_ann_node_at(snapshot, node_id)->owner_generation !=
            inode->i_generation ||
        vecfs_ann_node_at(snapshot, node_id)->vector_ordinal !=
            ordinal) {
        ret = -ESTALE;
        goto out;
    }

    external_id =
        vecfs_ann_node_at(snapshot, node_id)->external_id;

    ret = vecfs_ann_set_deleted_locked(
        inode->i_sb,
        node_id,
        true,
        VECFS_ANN_ACCESS_FILE);

    /*
     * 正常 mark-delete 保留 ordinal 映射，便于 update/unmark。
     * 文件最终 unlink 时由 remove_all_nodes 清除映射。
     */
    if (!ret)
        pr_debug(
            "vecfs: file vector deleted ino=%lu "
            "ordinal=%u label=%llu\n",
            inode->i_ino,
            ordinal,
            external_id);

out:
    mutex_unlock(&cache->writer_lock);
    return ret;
}
