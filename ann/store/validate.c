// SPDX-License-Identifier: GPL-2.0

#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/rhashtable.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

#include "../../include/vecfs.h"
#include "../../include/vecfs_ann_indirect.h"
#include "../../include/vecfs_ann_meta.h"
#include "../../include/vecfs_ann_store.h"
#include "../../include/vecfs_ann_validate.h"

struct vecfs_ann_external_entry {
    struct rhash_head node;
    u64 external_id;
    u32 node_id;
};

static const struct rhashtable_params
vecfs_ann_external_ht_params = {
    .head_offset =
        offsetof(struct vecfs_ann_external_entry, node),
    .key_offset =
        offsetof(struct vecfs_ann_external_entry, external_id),
    .key_len = sizeof(u64),
    .automatic_shrinking = true,
};

struct vecfs_ann_mapping {
    u32 ordinal;
    u32 node_id;
};

struct vecfs_ann_mapping_snapshot {
    struct vecfs_ann_mapping *entries;
    size_t count;
    size_t capacity;
};

struct vecfs_ann_validate_context {
    struct super_block *sb;
    struct vecfs_ann_validate_report *report;
    struct rhashtable external_ids;

    u32 flags;
    bool inconsistent;

    u32 entry_node;
    u16 max_level;
};

static struct inode *vecfs_ann_iget(
    struct super_block *sb,
    unsigned long ino)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct inode *inode;
    int ret;

    if (!ino || ino > sbi->s_inodes_count)
        return ERR_PTR(-ENOENT);

    inode = iget_locked(sb, ino);
    if (!inode)
        return ERR_PTR(-ENOMEM);

    if (VECFS_INODE_IS_NEW(inode)) {
        ret = vecfs_read_inode(inode);
        if (ret) {
            iget_failed(inode);
            return ERR_PTR(ret);
        }

        unlock_new_inode(inode);
    }

    if (!inode->i_mode) {
        iput(inode);
        return ERR_PTR(-ENOENT);
    }

    return inode;
}

static int vecfs_ann_record_external_id(
    struct vecfs_ann_validate_context *ctx,
    u64 external_id,
    u32 node_id)
{
    struct vecfs_ann_external_entry *existing;
    struct vecfs_ann_external_entry *entry;
    int ret;

    if (external_id == VECFS_ANN_EXTERNAL_ID_AUTO)
        return -EINVAL;

    existing = rhashtable_lookup_fast(
        &ctx->external_ids,
        &external_id,
        vecfs_ann_external_ht_params);

    if (existing) {
        ctx->report->duplicate_external_ids++;
        ctx->inconsistent = true;
        return -EEXIST;
    }

    entry = kmalloc(sizeof(*entry), GFP_NOFS);
    if (!entry)
        return -ENOMEM;

    entry->external_id = external_id;
    entry->node_id = node_id;

    ret = rhashtable_insert_fast(
        &ctx->external_ids,
        &entry->node,
        vecfs_ann_external_ht_params);
    if (ret) {
        kfree(entry);
        return ret;
    }

    return 0;
}

static void vecfs_ann_destroy_external_table(
    struct rhashtable *table)
{
    struct rhashtable_iter iterator;
    struct vecfs_ann_external_entry *entry;

    rhashtable_walk_enter(table, &iterator);
    rhashtable_walk_start(&iterator);

    for (;;) {
        entry = rhashtable_walk_next(&iterator);

        if (IS_ERR(entry)) {
            if (PTR_ERR(entry) == -EAGAIN)
                continue;
            break;
        }

        if (!entry)
            break;

        rhashtable_remove_fast(
            table,
            &entry->node,
            vecfs_ann_external_ht_params);
        kfree(entry);
    }

    rhashtable_walk_stop(&iterator);
    rhashtable_walk_exit(&iterator);
    rhashtable_destroy(table);
}

static int vecfs_ann_mark_invalid_object(
    struct vecfs_ann_validate_context *ctx,
    u32 node_id)
{
    int ret;

    ctx->inconsistent = true;

    if (!(ctx->flags & VECFS_ANN_VALIDATE_F_REPAIR))
        return 0;

    if (sb_rdonly(ctx->sb))
        return -EROFS;

    ret = vecfs_ann_object_mark_deleted(
        ctx->sb, node_id);

    /*
     * 对象可能已在 arena scan 阶段变为 DELETED/CORRUPT。
     */
    if (ret == -EINVAL ||
        ret == -ENOENT ||
        ret == -ESTALE)
        ret = 0;

    if (!ret)
        ctx->report->objects_marked_deleted++;

    return ret;
}

static int vecfs_ann_validate_file_object(
    struct vecfs_ann_validate_context *ctx,
    const struct vecfs_ann_object_info *info)
{
    struct inode *inode;
    u32 mapped_node;
    int ret;

    inode = vecfs_ann_iget(
        ctx->sb, info->owner_ino);

    if (IS_ERR(inode)) {
        ret = PTR_ERR(inode);

        if (ret == -ENOENT) {
            ctx->report->missing_inodes++;
            return vecfs_ann_mark_invalid_object(
                ctx, info->node_id);
        }

        return ret;
    }

    if (!S_ISREG(inode->i_mode)) {
        ctx->report->owner_mismatches++;
        iput(inode);

        return vecfs_ann_mark_invalid_object(
            ctx, info->node_id);
    }

    if (inode->i_generation !=
        info->owner_generation) {
        ctx->report->generation_mismatches++;
        iput(inode);

        return vecfs_ann_mark_invalid_object(
            ctx, info->node_id);
    }

    ret = vecfs_ann_get_node(
        inode,
        info->vector_ordinal,
        &mapped_node);

    if (ret == -ENOENT) {
        ctx->report->missing_inode_mappings++;
        iput(inode);

        return vecfs_ann_mark_invalid_object(
            ctx, info->node_id);
    }

    if (ret) {
        iput(inode);
        return ret;
    }

    if (mapped_node != info->node_id) {
        ctx->report->owner_mismatches++;
        iput(inode);

        return vecfs_ann_mark_invalid_object(
            ctx, info->node_id);
    }

    iput(inode);
    return 0;
}

static int vecfs_ann_validate_object_links(
    struct vecfs_ann_validate_context *context,
    const struct vecfs_ann_object_info *info);

static int vecfs_ann_validate_objects(
    struct vecfs_ann_validate_context *ctx)
{
    struct vecfs_sb_info *sbi = VECFS_SB(ctx->sb);
    u64 next_node_id;
    u64 id;
    int ret;

    mutex_lock(&sbi->s_ann.lock);
    next_node_id = sbi->s_ann.next_node_id;
    mutex_unlock(&sbi->s_ann.lock);

    for (id = VECFS_ANN_FIRST_NODE_ID;
         id < next_node_id;
         id++) {
        struct vecfs_ann_object_info info;

        ret = vecfs_ann_object_read_info(
            ctx->sb,
            (u32)id,
            &info);

        ctx->report->objects_checked++;

        if (ret) {
            /*
             * arena scan 已负责隔离 CRC 损坏对象。这里记录即可。
             */
            ctx->report->corrupt_objects++;
            ctx->inconsistent = true;
            continue;
        }

        switch (info.state) {
        case VECFS_ANN_OBJ_FREE:
            /*
            * FREE 是 rebuild 产生的合法 node ID 空洞。
            */
            continue;

        case VECFS_ANN_OBJ_LIVE:
            ctx->report->live_objects++;
            break;

        case VECFS_ANN_OBJ_DELETED:
        case VECFS_ANN_OBJ_CORRUPT:
            ctx->report->deleted_objects++;
            continue;

        case VECFS_ANN_OBJ_PREPARED:
            ret = vecfs_ann_mark_invalid_object(
                ctx,
                info.node_id);
            if (ret)
                return ret;

            continue;

        default:
            ctx->report->corrupt_objects++;
            ctx->inconsistent = true;
            continue;
        }


        ret = vecfs_ann_record_external_id(
            ctx,
            info.external_id,
            info.node_id);

        if (ret == -EEXIST || ret == -EINVAL) {
            ret = vecfs_ann_mark_invalid_object(
                ctx, info.node_id);
            if (ret)
                return ret;
            continue;
        }

        if (ret)
            return ret;

        switch (info.owner_type) {
        case VECFS_ANN_OWNER_FILE:
            ctx->report->file_objects++;

            ret = vecfs_ann_validate_file_object(
                ctx, &info);
            if (ret)
                return ret;
            break;

        case VECFS_ANN_OWNER_STANDALONE:
            ctx->report->standalone_objects++;

            if (info.owner_ino ||
                info.owner_generation) {
                ctx->report->owner_mismatches++;

                ret = vecfs_ann_mark_invalid_object(
                    ctx, info.node_id);
                if (ret)
                    return ret;
            }
            break;

        case VECFS_ANN_OWNER_IMPORTED:
            ctx->report->imported_objects++;

            if (info.owner_ino ||
                info.owner_generation) {
                ctx->report->owner_mismatches++;

                ret = vecfs_ann_mark_invalid_object(
                    ctx, info.node_id);
                if (ret)
                    return ret;
            }
            break;

        default:
            ctx->report->owner_mismatches++;

            ret = vecfs_ann_mark_invalid_object(
                ctx, info.node_id);
            if (ret)
                return ret;
            continue;
        }
        ret = vecfs_ann_validate_object_links(
            ctx,
            &info);
        if (ret)
            return ret;

        /*
         * 只使用通过所有校验的 LIVE 节点恢复 entry point。
         */
        if (info.state == VECFS_ANN_OBJ_LIVE &&
            (ctx->entry_node ==
                 VECFS_ANN_INVALID_NODE_ID ||
             info.level > ctx->max_level)) {
            ctx->entry_node = info.node_id;
            ctx->max_level = info.level;
        }

        if (!(id & 16383))
            schedule();
        else if (!(id & 63))
            cond_resched();
    }

    return 0;
}

static int vecfs_ann_snapshot_callback(
    struct inode *inode,
    u32 ordinal,
    u32 node_id,
    void *private_data)
{
    struct vecfs_ann_mapping_snapshot *snapshot =
        private_data;
    struct vecfs_ann_mapping *entries;

    if (snapshot->count == snapshot->capacity) {
        size_t capacity =
            snapshot->capacity ?
                snapshot->capacity * 2 : 16;

        if (capacity > SIZE_MAX /
                       sizeof(*entries))
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

    snapshot->entries[snapshot->count].ordinal =
        ordinal;
    snapshot->entries[snapshot->count].node_id =
        node_id;
    snapshot->count++;

    return 0;
}

static int vecfs_ann_validate_inode_mapping(
    struct vecfs_ann_validate_context *ctx,
    struct inode *inode,
    const struct vecfs_ann_mapping *mapping)
{
    struct vecfs_ann_object_info info;
    bool valid = true;
    int ret;

    ctx->report->inode_mappings_checked++;

    ret = vecfs_ann_object_read_info(
        inode->i_sb,
        mapping->node_id,
        &info);

    if (ret) {
        ctx->report->stale_inode_mappings++;
        valid = false;
    } else if (info.state !=
               VECFS_ANN_OBJ_LIVE) {
        ctx->report->stale_inode_mappings++;
        valid = false;
    } else if (info.owner_type !=
               VECFS_ANN_OWNER_FILE) {
        ctx->report->owner_mismatches++;
        valid = false;
    } else if (info.owner_ino !=
               inode->i_ino) {
        ctx->report->owner_mismatches++;
        valid = false;
    } else if (info.owner_generation !=
               inode->i_generation) {
        ctx->report->generation_mismatches++;
        valid = false;
    } else if (info.vector_ordinal !=
               mapping->ordinal) {
        ctx->report->ordinal_mismatches++;
        valid = false;
    }

    if (valid)
        return 0;

    ctx->inconsistent = true;

    if (!(ctx->flags & VECFS_ANN_VALIDATE_F_REPAIR))
        return 0;

    if (sb_rdonly(ctx->sb))
        return -EROFS;

    ret = vecfs_ann_clear_node(
        inode,
        mapping->ordinal);
    if (!ret)
        ctx->report->mappings_cleared++;

    return ret;
}

static int vecfs_ann_validate_inode(
    struct vecfs_ann_validate_context *ctx,
    struct inode *inode)
{
    struct vecfs_ann_mapping_snapshot snapshot = {};
    size_t i;
    int ret;

    if (!S_ISREG(inode->i_mode))
        return 0;

    ret = vecfs_ann_for_each_node(
        inode,
        vecfs_ann_snapshot_callback,
        &snapshot);
    if (ret)
        goto out;

    /*
     * for_each 已释放 inode meta_lock，之后读取对象不会形成
     * inode meta_lock -> store io_lock 的长期锁嵌套。
     */
    for (i = 0; i < snapshot.count; i++) {
        ret = vecfs_ann_validate_inode_mapping(
            ctx,
            inode,
            &snapshot.entries[i]);
        if (ret)
            break;
    }

out:
    kfree(snapshot.entries);
    return ret;
}

static int vecfs_ann_validate_inode_mappings(
    struct vecfs_ann_validate_context *ctx)
{
    struct vecfs_sb_info *sbi = VECFS_SB(ctx->sb);
    unsigned long bit;
    int ret = 0;

    /*
     * bitmap bit N 对应 inode N + 1。
     */
    for_each_set_bit(
        bit,
        sbi->s_imap,
        sbi->s_inodes_count) {
        unsigned long ino = bit + 1;
        struct inode *inode;
        if (ino <= VECFS_FIRST_USER_INO)
            continue;
        if (ino == VECFS_ANN_ARENA_INO ||
            ino == VECFS_ANN_REBUILD_ARENA_INO)
            continue;

        inode = vecfs_ann_iget(ctx->sb, ino);
        if (IS_ERR(inode)) {
            if (PTR_ERR(inode) == -ENOENT)
                continue;
            return PTR_ERR(inode);
        }

        ret = vecfs_ann_validate_inode(
            ctx, inode);

        iput(inode);

        if (ret)
            return ret;

        if (!(bit & 16383))
            schedule();
        else if (!(bit & 63))
            cond_resched();
    }

    return 0;
}

static int vecfs_ann_repair_metadata(
    struct vecfs_ann_validate_context *ctx)
{
    struct vecfs_sb_info *sbi = VECFS_SB(ctx->sb);
    bool changed = false;
    int ret = 0;

    mutex_lock(&sbi->s_ann.lock);

    if (sbi->s_ann.entry_node_id !=
        ctx->entry_node) {
        sbi->s_ann.entry_node_id =
            ctx->entry_node;
        changed = true;
    }

    if (sbi->s_ann.config.current_max_level !=
        ctx->max_level) {
        sbi->s_ann.config.current_max_level =
            ctx->max_level;
        changed = true;
    }

    if (!ctx->report->live_objects &&
        sbi->s_ann.state !=
            VECFS_ANN_META_EMPTY) {
        sbi->s_ann.state =
            VECFS_ANN_META_EMPTY;
        changed = true;
    } else if (ctx->report->live_objects &&
               sbi->s_ann.state ==
                   VECFS_ANN_META_EMPTY) {
        sbi->s_ann.state =
            VECFS_ANN_META_ACTIVE;
        changed = true;
    }

    if (changed) {
        sbi->s_ann.dirty = true;
        ctx->report->metadata_repairs++;
    }

    mutex_unlock(&sbi->s_ann.lock);

    if (changed &&
        (ctx->flags & VECFS_ANN_VALIDATE_F_REPAIR) &&
        !sb_rdonly(ctx->sb))
        ret = vecfs_ann_meta_commit(ctx->sb, true);

    return ret;
}

int vecfs_ann_validate(
    struct super_block *sb,
    u32 flags,
    struct vecfs_ann_validate_report *report)
{
    struct vecfs_ann_validate_context context;
    int ret;

    if (!sb || !report)
        return -EINVAL;

    if (!VECFS_SB(sb)->s_ann_store)
        return -ENODEV;

    memset(report, 0, sizeof(*report));
    memset(&context, 0, sizeof(context));

    context.sb = sb;
    context.flags = flags;
    context.report = report;
    context.entry_node =
        VECFS_ANN_INVALID_NODE_ID;

    ret = rhashtable_init(
        &context.external_ids,
        &vecfs_ann_external_ht_params);
    if (ret)
        return ret;

    /*
     * 第一遍：object -> inode mapping。
     */
    ret = vecfs_ann_validate_objects(&context);
    if (ret)
        goto out;

    /*
     * 第二遍：inode mapping -> object。
     */
    ret = vecfs_ann_validate_inode_mappings(
        &context);
    if (ret)
        goto out;

    ret = vecfs_ann_repair_metadata(&context);
    if (ret)
        goto out;

    report->recovered_entry_node =
        context.entry_node;
    report->recovered_max_level =
        context.max_level;

    if (context.inconsistent &&
        (flags & VECFS_ANN_VALIDATE_F_STRICT))
        ret = -EUCLEAN;

out:
    vecfs_ann_destroy_external_table(
        &context.external_ids);

    pr_info(
        "vecfs: ANN validate objects=%llu live=%llu "
        "file=%llu mappings=%llu stale=%llu "
        "missing=%llu duplicates=%llu repaired=%llu/%llu\n",
        report->objects_checked,
        report->live_objects,
        report->file_objects,
        report->inode_mappings_checked,
        report->stale_inode_mappings,
        report->missing_inode_mappings,
        report->duplicate_external_ids,
        report->objects_marked_deleted,
        report->mappings_cleared);

    return ret;
}

int vecfs_ann_validate_mount(
    struct super_block *sb)
{
    struct vecfs_ann_validate_report *report;
    u32 flags = 0;
    int ret;

    report = kzalloc(sizeof(*report), GFP_KERNEL);
    if (!report)
        return -ENOMEM;

    /*
     * 只读挂载执行检查但不能修复。
     */
    if (!sb_rdonly(sb))
        flags |= VECFS_ANN_VALIDATE_F_REPAIR;

    ret = vecfs_ann_validate(
        sb, flags, report);

    kfree(report);
    return ret;
}

static int vecfs_ann_validate_object_links(
    struct vecfs_ann_validate_context *context,
    const struct vecfs_ann_object_info *info)
{
    struct vecfs_sb_info *sbi =
        VECFS_SB(context->sb);

    struct vecfs_ann_store *store =
        sbi->s_ann_store;

    u32 *neighbors;
    u16 count;
    u16 level;
    u16 capacity;
    u16 i;
    int ret;

    if (!(context->flags &
          VECFS_ANN_VALIDATE_F_CHECK_LINKS))
        return 0;

    capacity = max_t(
        u16,
        store->layout.m0,
        store->layout.m);

    neighbors = kvmalloc_array(
        capacity,
        sizeof(*neighbors),
        GFP_NOFS);

    if (!neighbors)
        return -ENOMEM;

    ret = vecfs_ann_object_read_level0(
        context->sb,
        info->node_id,
        neighbors,
        capacity,
        &count);
    if (ret)
        goto out;

    for (i = 0; i < count; i++) {
        struct vecfs_ann_object_info neighbor_info;

        if (neighbors[i] ==
                info->node_id ||
            neighbors[i] <
                VECFS_ANN_FIRST_NODE_ID) {
            context->report->
                invalid_neighbors++;

            context->inconsistent = true;
            continue;
        }

        ret = vecfs_ann_object_read_info(
            context->sb,
            neighbors[i],
            &neighbor_info);

        if (ret ||
            (neighbor_info.state !=
                 VECFS_ANN_OBJ_LIVE &&
             neighbor_info.state !=
                 VECFS_ANN_OBJ_DELETED)) {
            context->report->
                invalid_neighbors++;

            context->inconsistent = true;
            ret = 0;
        }
    }

    for (level = 1;
         level <= info->level;
         level++) {
        ret = vecfs_ann_object_read_upper(
            context->sb,
            info->node_id,
            level,
            neighbors,
            capacity,
            &count);
        if (ret)
            goto out;

        for (i = 0; i < count; i++) {
            struct vecfs_ann_object_info neighbor_info;

            if (neighbors[i] ==
                    info->node_id ||
                neighbors[i] <
                    VECFS_ANN_FIRST_NODE_ID) {
                context->report->
                    invalid_neighbors++;

                context->inconsistent = true;
                continue;
            }

            ret = vecfs_ann_object_read_info(
                context->sb,
                neighbors[i],
                &neighbor_info);

            if (ret ||
                neighbor_info.level < level ||
                (neighbor_info.state !=
                     VECFS_ANN_OBJ_LIVE &&
                 neighbor_info.state !=
                     VECFS_ANN_OBJ_DELETED)) {
                context->report->
                    invalid_neighbors++;

                context->inconsistent = true;
                ret = 0;
            }
        }
    }

out:
    kvfree(neighbors);
    return ret;
}
