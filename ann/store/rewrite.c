// SPDX-License-Identifier: GPL-2.0

#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

#include "../../include/vecfs.h"
#include "../../include/vecfs_ann_cache.h"
#include "../../include/vecfs_ann_store.h"
#include "../../include/vecfs_Indirect.h"
#include "internal.h"

static void vecfs_ann_rewrite_init_empty_links(
    struct vecfs_ann_store *store,
    void *object)
{
    __le32 *level0;
    __le16 *counts;
    __le32 *upper;
    size_t upper_slots;
    u32 i;

    level0 = (__le32 *)(
        (u8 *)object +
        store->layout.level0_offset);

    for (i = 0;
         i < store->layout.m0;
         i++)
        level0[i] = cpu_to_le32(
            VECFS_ANN_INVALID_NODE_ID);

    counts = (__le16 *)(
        (u8 *)object +
        store->layout.upper_counts_offset);

    for (i = 0;
         i < store->layout.max_levels - 1;
         i++)
        counts[i] = 0;

    upper = (__le32 *)(
        (u8 *)object +
        store->layout.upper_links_offset);

    upper_slots =
        (size_t)(store->layout.max_levels - 1) *
        store->layout.m;

    for (i = 0;
         i < upper_slots;
         i++)
        upper[i] = cpu_to_le32(
            VECFS_ANN_INVALID_NODE_ID);
}

static void vecfs_ann_rewrite_fill_header(
    struct vecfs_ann_store *store,
    void *object,
    u32 node_id,
    u32 state,
    u64 epoch)
{
    struct vecfs_ann_object_disk *disk =
        object;

    memset(
        object,
        0,
        store->layout.stride);

    disk->magic =
        cpu_to_le32(
            VECFS_ANN_OBJECT_MAGIC);

    disk->version =
        cpu_to_le16(
            VECFS_ANN_OBJECT_VERSION);

    disk->header_size =
        cpu_to_le16(sizeof(*disk));

    disk->node_id =
        cpu_to_le32(node_id);

    disk->state =
        cpu_to_le32(state);

    disk->external_id =
        cpu_to_le64(
            VECFS_ANN_EXTERNAL_ID_AUTO);

    disk->owner_type =
        cpu_to_le16(
            VECFS_ANN_OWNER_NONE);

    disk->object_epoch =
        cpu_to_le64(epoch);

    vecfs_ann_rewrite_init_empty_links(
        store,
        object);
}

static int vecfs_ann_rewrite_fill_live(
    struct vecfs_ann_store *store,
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id,
    u64 epoch,
    void *object)
{
    struct vecfs_ann_object_disk *disk;
    const struct vecfs_ann_cache_node *node;
    __le32 *level0;
    __le16 *upper_counts;
    __le32 *upper_links;
    const u32 *source_links;
    u16 effective_flags;
    u16 count;
    u16 level;
    u16 i;

    node = vecfs_ann_node_at(snapshot, node_id);

    vecfs_ann_rewrite_fill_header(
        store,
        object,
        node_id,
        VECFS_ANN_OBJ_LIVE,
        epoch);

    disk = object;

    disk->external_id =
        cpu_to_le64(node->external_id);

    disk->owner_ino =
        cpu_to_le64(node->owner_ino);

    disk->owner_generation =
        cpu_to_le32(
            node->owner_generation);

    disk->vector_ordinal =
        cpu_to_le32(
            node->vector_ordinal);

    disk->content_offset =
        cpu_to_le64(
            node->content_offset);

    disk->content_length =
        cpu_to_le64(
            node->content_length);

    disk->level =
        cpu_to_le16(node->level);

    disk->owner_type =
        cpu_to_le16(node->owner_type);

    effective_flags =
        vecfs_ann_snapshot_effective_flags(
            snapshot,
            node_id);

    disk->flags =
        cpu_to_le16(effective_flags);

    count = vecfs_ann_level0_count(
        snapshot,
        node_id);

    if (count > store->layout.m0)
        return -EUCLEAN;

    disk->level0_count =
        cpu_to_le16(count);

    level0 = (__le32 *)(
        (u8 *)object +
        store->layout.level0_offset);

    source_links =
        vecfs_ann_level0_links(
            snapshot,
            node_id);

    for (i = 0; i < count; i++)
        level0[i] =
            cpu_to_le32(
                source_links[i]);

    memcpy(
        (u8 *)object +
            store->layout.vector_offset,
        vecfs_ann_snapshot_vector(
            snapshot,
            node_id),
        store->layout.vector_bytes);

    upper_counts = (__le16 *)(
        (u8 *)object +
        store->layout.upper_counts_offset);

    upper_links = (__le32 *)(
        (u8 *)object +
        store->layout.upper_links_offset);

    for (level = 1;
         level <= node->level;
         level++) {
        count = vecfs_ann_upper_count(
            snapshot,
            node_id,
            level);

        if (count > store->layout.m)
            return -EUCLEAN;

        upper_counts[level - 1] =
            cpu_to_le16(count);

        source_links =
            vecfs_ann_upper_links(
                snapshot,
                node_id,
                level);

        if (!source_links && count)
            return -EUCLEAN;

        for (i = 0; i < count; i++) {
            upper_links[
                (size_t)(level - 1) *
                    store->layout.m + i] =
                cpu_to_le32(
                    source_links[i]);
        }
    }

    return 0;
}


static int vecfs_ann_rewrite_truncate_arena(
    struct inode *inode)
{
    int ret;

    inode_lock(inode);

    ret = vecfs_truncate_blocks(
        inode,
        0);

    if (!ret) {
        truncate_setsize(
            inode,
            0);

        mark_inode_dirty(inode);
    }

    inode_unlock(inode);
    return ret;
}

int vecfs_ann_store_rewrite_snapshot(
    struct super_block *sb,
    u32 arena_inode_number,
    const struct vecfs_ann_snapshot *snapshot,
    u64 epoch,
    struct vecfs_ann_store **result_store)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_store *store;
    struct inode *inode;
    void *object;
    u32 node_id;
    int ret;

    if (!sb || !snapshot || !result_store)
        return -EINVAL;

    *result_store = NULL;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return -EINVAL;

    if (arena_inode_number !=
            VECFS_ANN_ARENA_INO &&
        arena_inode_number !=
            VECFS_ANN_REBUILD_ARENA_INO)
        return -EINVAL;

    ret = vecfs_ann_get_arena_inode(
        sb,
        arena_inode_number,
        &inode);
    if (ret)
        return ret;

    if (sbi->s_ann_store &&
        sbi->s_ann_store->arena_inode ==
            inode) {
        iput(inode);
        return -EBUSY;
    }

    ret = vecfs_ann_rewrite_truncate_arena(
        inode);
    if (ret) {
        iput(inode);
        return ret;
    }

    store = kzalloc(
        sizeof(*store),
        GFP_KERNEL);
    if (!store) {
        iput(inode);
        return -ENOMEM;
    }

    store->sb = sb;
    store->arena_inode = inode;

    mutex_init(
        &store->allocation_lock);

    init_rwsem(
        &store->io_lock);

    ret = vecfs_ann_layout_init(
        &store->layout,
        &sbi->s_ann.config);
    if (ret)
        goto fail_store;

    object = kvmalloc(
        store->layout.stride,
        GFP_NOFS);
    if (!object) {
        ret = -ENOMEM;
        goto fail_store;
    }

    down_write(&store->io_lock);

    for (node_id = 0;
         node_id < snapshot->capacity;
         node_id++) {
        if (vecfs_ann_rebuild_should_cancel(sb)) {
            ret = -ECANCELED;
            goto out_unlock;
        }
        if (node_id ==
            VECFS_ANN_EMPTY_NODE_ID) {
            vecfs_ann_rewrite_fill_header(
                store,
                object,
                node_id,
                VECFS_ANN_OBJ_FREE,
                epoch);
        } else if (
            vecfs_ann_node_at(snapshot, node_id)->state ==
                VECFS_ANN_OBJ_LIVE) {
            ret = vecfs_ann_rewrite_fill_live(
                store,
                snapshot,
                node_id,
                epoch,
                object);
            if (ret)
                goto out_unlock;
        } else {
            /*
             * deleted/corrupt/unused object 完全清除。
             */
            vecfs_ann_rewrite_fill_header(
                store,
                object,
                node_id,
                VECFS_ANN_OBJ_FREE,
                epoch);
        }

        ret = vecfs_ann_object_store_locked(
            store,
            node_id,
            object,
            false);
        if (ret)
            goto out_unlock;

        if (!(node_id & 1023))
            cond_resched();
    }

out_unlock:
    up_write(&store->io_lock);
    kvfree(object);

    if (ret)
        goto fail_store;

    ret = write_inode_now(
        inode,
        1);
    if (ret)
        goto fail_store;

    ret = filemap_write_and_wait(
        inode->i_mapping);
    if (ret)
        goto fail_store;

    *result_store = store;
    return 0;

fail_store:
    vecfs_ann_store_release_instance(
        store,
        false);

    return ret;
}
