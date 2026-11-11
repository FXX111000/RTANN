// SPDX-License-Identifier: GPL-2.0

#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/rcupdate.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_cache.h"
#include "../include/vecfs_ann_indirect.h"
#include "../include/vecfs_ann_info.h"
#include "../include/vecfs_ann_label.h"

static void vecfs_ann_cache_node_to_info(
    const struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_cache_node *node,
    u32 node_id,
    struct vecfs_ann_object_info *info)
{
    memset(info, 0, sizeof(*info));

    info->node_id = node_id;
    info->state = node->state;

    info->external_id =
        node->external_id;

    info->owner_ino =
        node->owner_ino;

    info->owner_generation =
        node->owner_generation;

    info->vector_ordinal =
        node->vector_ordinal;

    info->content_offset =
        node->content_offset;

    info->content_length =
        node->content_length;

    info->level =
        node->level;

    info->owner_type =
        node->owner_type;

    info->flags =
        vecfs_ann_snapshot_effective_flags(
            snapshot,
            node_id);
}


static int vecfs_ann_copy_snapshot_record_locked(
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id,
    bool include_deleted,
    bool allow_file_owner,
    struct vecfs_ann_vector_record *record)
{
    const struct vecfs_ann_cache_node *node;
    void *vector;

    if (!snapshot || !record)
        return -EINVAL;

    if (!vecfs_ann_snapshot_id_valid(
            snapshot,
            node_id))
        return -ENOENT;

    node = vecfs_ann_node_at(snapshot, node_id);

    if (node->state ==
        VECFS_ANN_OBJ_DELETED) {
        if (!include_deleted)
            return -ENOENT;
    } else if (node->state !=
               VECFS_ANN_OBJ_LIVE) {
        return -ENOENT;
    }

    if (!allow_file_owner &&
        node->owner_type ==
            VECFS_ANN_OWNER_FILE)
        return -EPERM;

    vector = kvmemdup(
        vecfs_ann_snapshot_vector(
            snapshot,
            node_id),
        snapshot->vector_bytes,
        GFP_NOFS);
    if (!vector)
        return -ENOMEM;

    memset(
        record,
        0,
        sizeof(*record));

    record->vector = vector;
    record->vector_bytes =
        snapshot->vector_bytes;

    vecfs_ann_cache_node_to_info(
        snapshot,
        node,
        node_id,
        &record->info);

    return 0;
}


int vecfs_ann_get_by_label(
    struct super_block *sb,
    u64 external_id,
    bool include_deleted,
    bool allow_file_owner,
    struct vecfs_ann_vector_record *record)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *snapshot;
    u32 node_id;
    int ret;

    if (!sb || !record)
        return -EINVAL;

    memset(record, 0, sizeof(*record));

    sbi = VECFS_SB(sb);
    if (!sbi || !sbi->s_ann_cache)
        return -ENODEV;

    cache = sbi->s_ann_cache;

    mutex_lock(&cache->writer_lock);

    ret = vecfs_ann_label_lookup_locked(
        sb,
        external_id,
        &node_id);
    if (ret)
        goto out_unlock;

    snapshot = rcu_dereference_protected(
        cache->snapshot,
        lockdep_is_held(
            &cache->writer_lock));

    if (!snapshot) {
        ret = -ENODEV;
        goto out_unlock;
    }

    ret = vecfs_ann_copy_snapshot_record_locked(
        snapshot,
        node_id,
        include_deleted,
        allow_file_owner,
        record);

out_unlock:
    mutex_unlock(&cache->writer_lock);
    return ret;
}

int vecfs_ann_get_file_vector(
    struct inode *inode,
    u32 ordinal,
    bool include_deleted,
    struct vecfs_ann_vector_record *record)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *snapshot;
    const struct vecfs_ann_cache_node *node;
    u32 node_id;
    int ret;

    if (!inode || !record)
        return -EINVAL;

    if (!S_ISREG(inode->i_mode))
        return -EINVAL;

    memset(record, 0, sizeof(*record));

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

    mutex_lock(&cache->writer_lock);

    snapshot = rcu_dereference_protected(
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
        node->vector_ordinal != ordinal) {
        ret = -ESTALE;
        goto out_unlock;
    }

    ret = vecfs_ann_copy_snapshot_record_locked(
        snapshot,
        node_id,
        include_deleted,
        true,
        record);

out_unlock:
    mutex_unlock(&cache->writer_lock);
    return ret;
}

void vecfs_ann_vector_record_destroy(
    struct vecfs_ann_vector_record *record)
{
    if (!record)
        return;

    kvfree(record->vector);
    memset(record, 0, sizeof(*record));
}
