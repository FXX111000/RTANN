// SPDX-License-Identifier: GPL-2.0

#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/rhashtable.h>
#include <linux/slab.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_cache.h"
#include "../include/vecfs_ann_label.h"

static const struct rhashtable_params
vecfs_ann_label_ht_params = {
    .head_offset = offsetof(
        struct vecfs_ann_label_entry,
        hash_node),

    .key_offset = offsetof(
        struct vecfs_ann_label_entry,
        external_id),

    .key_len = sizeof(u64),
    .automatic_shrinking = true,
};

static int vecfs_ann_label_build_from_snapshot(
    struct super_block *sb,
    const struct vecfs_ann_snapshot *snapshot)
{
    u32 node_id;
    int ret;

    for (node_id = VECFS_ANN_FIRST_NODE_ID;
         node_id < snapshot->capacity;
         node_id++) {
        const struct vecfs_ann_cache_node *node =
            vecfs_ann_node_at(snapshot, node_id);

        /*
         * DELETED label 仍然保留。
         *
         * 这与 hnswlib 一致：
         * markDelete 不会从 label_lookup_ 中删除 label，
         * 因而可以 unmarkDelete，且不能直接插入重复 label。
         */
        if (node->state != VECFS_ANN_OBJ_LIVE &&
            node->state != VECFS_ANN_OBJ_DELETED)
            continue;

        ret = vecfs_ann_label_insert_locked(
            sb,
            node->external_id,
            node_id);
        if (ret)
            return ret;
    }

    return 0;
}

int vecfs_ann_label_index_init(
    struct super_block *sb)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *snapshot;
    int ret;

    if (!sb)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi || !sbi->s_ann_cache)
        return -ENODEV;

    cache = sbi->s_ann_cache;

    ret = rhashtable_init(
        &sbi->s_ann_labels.table,
        &vecfs_ann_label_ht_params);
    if (ret)
        return ret;

    sbi->s_ann_labels.initialized = true;

    mutex_lock(&cache->writer_lock);

    snapshot = rcu_dereference_protected(
        cache->snapshot,
        lockdep_is_held(&cache->writer_lock));

    ret = vecfs_ann_label_build_from_snapshot(
        sb, snapshot);

    mutex_unlock(&cache->writer_lock);

    if (ret) {
        vecfs_ann_label_index_destroy(sb);
        return ret;
    }

    return 0;
}

static void vecfs_ann_label_free_entry(
    void *pointer,
    void *argument)
{
    kfree(pointer);
}

void vecfs_ann_label_index_destroy(
    struct super_block *sb)
{
    struct vecfs_sb_info *sbi;

    if (!sb)
        return;

    sbi = VECFS_SB(sb);
    if (!sbi || !sbi->s_ann_labels.initialized)
        return;

    rhashtable_free_and_destroy(
        &sbi->s_ann_labels.table,
        vecfs_ann_label_free_entry,
        NULL);

    sbi->s_ann_labels.initialized = false;
}

int vecfs_ann_label_lookup_locked(
    struct super_block *sb,
    u64 external_id,
    u32 *node_id)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_label_entry *entry;

    if (!sb || !node_id)
        return -EINVAL;

    if (external_id ==
        VECFS_ANN_EXTERNAL_ID_AUTO)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi || !sbi->s_ann_labels.initialized)
        return -ENODEV;

    entry = rhashtable_lookup_fast(
        &sbi->s_ann_labels.table,
        &external_id,
        vecfs_ann_label_ht_params);

    if (!entry)
        return -ENOENT;

    *node_id = entry->node_id;
    return 0;
}

int vecfs_ann_label_insert_locked(
    struct super_block *sb,
    u64 external_id,
    u32 node_id)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_label_entry *entry;
    struct vecfs_ann_label_entry *existing;
    int ret;

    if (!sb ||
        external_id ==
            VECFS_ANN_EXTERNAL_ID_AUTO ||
        node_id < VECFS_ANN_FIRST_NODE_ID)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi || !sbi->s_ann_labels.initialized)
        return -ENODEV;

    existing = rhashtable_lookup_fast(
        &sbi->s_ann_labels.table,
        &external_id,
        vecfs_ann_label_ht_params);

    if (existing)
        return -EEXIST;

    entry = kmalloc(sizeof(*entry), GFP_NOFS);
    if (!entry)
        return -ENOMEM;

    entry->external_id = external_id;
    entry->node_id = node_id;

    ret = rhashtable_insert_fast(
        &sbi->s_ann_labels.table,
        &entry->hash_node,
        vecfs_ann_label_ht_params);

    if (ret)
        kfree(entry);

    return ret;
}

int vecfs_ann_label_remove_locked(
    struct super_block *sb,
    u64 external_id,
    u32 expected_node_id)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_label_entry *entry;
    int ret;

    if (!sb)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi || !sbi->s_ann_labels.initialized)
        return -ENODEV;

    entry = rhashtable_lookup_fast(
        &sbi->s_ann_labels.table,
        &external_id,
        vecfs_ann_label_ht_params);

    if (!entry)
        return -ENOENT;

    if (expected_node_id &&
        entry->node_id != expected_node_id)
        return -ESTALE;

    ret = rhashtable_remove_fast(
        &sbi->s_ann_labels.table,
        &entry->hash_node,
        vecfs_ann_label_ht_params);

    if (!ret)
        kfree(entry);

    return ret;
}
int vecfs_ann_label_rebuild_locked(
    struct super_block *sb,
    const struct vecfs_ann_snapshot *snapshot)
{
    struct vecfs_sb_info *sbi;
    u32 node_id;
    int ret;

    if (!sb || !snapshot)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi || !sbi->s_ann_cache)
        return -ENODEV;

    lockdep_assert_held(
        &sbi->s_ann_cache->writer_lock);

    if (sbi->s_ann_labels.initialized) {
        rhashtable_free_and_destroy(
            &sbi->s_ann_labels.table,
            vecfs_ann_label_free_entry,
            NULL);

        sbi->s_ann_labels.initialized = false;
    }

    ret = rhashtable_init(
        &sbi->s_ann_labels.table,
        &vecfs_ann_label_ht_params);
    if (ret)
        return ret;

    sbi->s_ann_labels.initialized = true;

    for (node_id = VECFS_ANN_FIRST_NODE_ID;
         node_id < snapshot->capacity;
         node_id++) {
        const struct vecfs_ann_cache_node *node =
            vecfs_ann_node_at(snapshot, node_id);

        /*
         * tombstone 保留 label，以支持 undelete，并阻止直接重用
         * 尚未被 rebuild 清除的 external ID。
         */
        if (node->state !=
                VECFS_ANN_OBJ_LIVE &&
            node->state !=
                VECFS_ANN_OBJ_DELETED)
            continue;

        ret = vecfs_ann_label_insert_locked(
            sb,
            node->external_id,
            node_id);
        if (ret) {
            rhashtable_free_and_destroy(
                &sbi->s_ann_labels.table,
                vecfs_ann_label_free_entry,
                NULL);

            sbi->s_ann_labels.initialized =
                false;

            return ret;
        }
    }

    return 0;
}
