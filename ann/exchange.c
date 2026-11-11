// SPDX-License-Identifier: GPL-2.0

#include <linux/bitmap.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/random.h>
#include <linux/rcupdate.h>
#include <linux/rhashtable.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_build.h"
#include "../include/vecfs_ann_cache.h"
#include "../include/vecfs_ann_exchange.h"
#include "../include/vecfs_ann_label.h"
#include "../include/vecfs_ann_meta.h"
#include "../include/vecfs_ann_store.h"
#include "../include/vecfs_Indirect.h"

struct vecfs_ann_import_label_key {
    u64 external_id;
};

static const struct rhashtable_params
vecfs_ann_import_label_params = {
    .head_offset = offsetof(
        struct vecfs_ann_import_label,
        hash_node),

    .key_offset = offsetof(
        struct vecfs_ann_import_label,
        external_id),

    .key_len = sizeof(u64),
    .automatic_shrinking = true,
};

static void vecfs_ann_import_label_free(
    void *pointer,
    void *argument)
{
    kfree(pointer);
}

static void vecfs_ann_import_transaction_reset(
    struct vecfs_ann_import_transaction *transaction)
{
    if (!transaction)
        return;

    if (transaction->labels_initialized) {
        rhashtable_free_and_destroy(
            &transaction->labels,
            vecfs_ann_import_label_free,
            NULL);

        transaction->labels_initialized = false;
    }

    bitmap_free(
        transaction->present_nodes);

    transaction->present_nodes = NULL;

    vecfs_ann_snapshot_free(
        transaction->snapshot);

    transaction->snapshot = NULL;

    transaction->transaction_id = 0;
    transaction->expected_nodes = 0;
    transaction->imported_nodes = 0;
    transaction->capacity = 0;
    transaction->active = false;
}

int vecfs_ann_exchange_init(
    struct super_block *sb)
{
    struct vecfs_sb_info *sbi;

    if (!sb)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return -EINVAL;

    memset(
        &sbi->s_ann_exchange,
        0,
        sizeof(sbi->s_ann_exchange));

    mutex_init(
        &sbi->s_ann_exchange.lock);

    return 0;
}

void vecfs_ann_exchange_destroy(
    struct super_block *sb)
{
    struct vecfs_sb_info *sbi;

    if (!sb)
        return;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return;

    mutex_lock(
        &sbi->s_ann_exchange.lock);

    vecfs_ann_import_transaction_reset(
        &sbi->s_ann_exchange.import);

    mutex_unlock(
        &sbi->s_ann_exchange.lock);

    mutex_destroy(
        &sbi->s_ann_exchange.lock);
}

static int vecfs_ann_import_validate_begin(
    struct vecfs_sb_info *sbi,
    const struct vecfs_ann_import_begin *request)
{
    if (!request)
        return -EINVAL;

    if (request->element_type !=
        sbi->s_ann.config.element_type)
        return -EINVAL;

    if (request->distance_type !=
        sbi->s_ann.config.distance_type)
        return -EINVAL;

    if (request->dimension !=
        sbi->s_ann.config.dimension)
        return -EINVAL;

    if (request->m !=
        sbi->s_ann.config.m)
        return -EINVAL;

    if (request->m0 !=
        sbi->s_ann.config.m0)
        return -EINVAL;

    if (request->max_levels !=
        sbi->s_ann.config.max_levels)
        return -EINVAL;

    if (request->node_capacity <=
            VECFS_ANN_FIRST_NODE_ID ||
        request->node_capacity >=
            VECFS_ANN_INVALID_NODE_ID)
        return -EINVAL;

    if (request->expected_nodes >
        (u64)request->node_capacity -
            VECFS_ANN_FIRST_NODE_ID)
        return -EINVAL;

    if (request->flags &
        ~(VECFS_IMPORT_F_PRESERVE_NODE_IDS |
          VECFS_IMPORT_F_ABORT_EXISTING))
        return -EINVAL;

    return 0;
}

static void vecfs_ann_import_remove_label(
    struct vecfs_ann_import_transaction *transaction,
    u64 external_id)
{
    struct vecfs_ann_import_label *entry;

    entry = rhashtable_lookup_fast(
        &transaction->labels,
        &external_id,
        vecfs_ann_import_label_params);

    if (!entry)
        return;

    if (!rhashtable_remove_fast(
            &transaction->labels,
            &entry->hash_node,
            vecfs_ann_import_label_params))
        kfree(entry);
}


static bool vecfs_ann_snapshot_empty(
    const struct vecfs_ann_snapshot *snapshot)
{
    return snapshot &&
           !snapshot->live_count;
}

int vecfs_ann_import_begin(
    struct super_block *sb,
    const struct vecfs_ann_import_begin *request,
    u64 *transaction_id)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *cur_snapshot;
    struct vecfs_ann_import_transaction *transaction;
    struct vecfs_ann_snapshot *snapshot;
    int ret;

    if (!sb || !request || !transaction_id)
        return -EINVAL;

    if (sb_rdonly(sb))
        return -EROFS;

    sbi = VECFS_SB(sb);
    if (!sbi ||
        !sbi->s_ann_cache ||
        !sbi->s_ann_store)
        return -ENODEV;

    ret = vecfs_ann_import_validate_begin(
        sbi,
        request);
    if (ret)
        return ret;

    cache = sbi->s_ann_cache;

    mutex_lock(
        &sbi->s_ann_exchange.lock);

    transaction =
        &sbi->s_ann_exchange.import;

    if (transaction->active) {
        if (!(request->flags &
              VECFS_IMPORT_F_ABORT_EXISTING)) {
            ret = -EBUSY;
            goto out_exchange;
        }

        vecfs_ann_import_transaction_reset(
            transaction);
    }

    mutex_lock(
        &cache->writer_lock);

    cur_snapshot = rcu_dereference_protected(
        cache->snapshot,
        lockdep_is_held(
            &cache->writer_lock));

    if (!vecfs_ann_snapshot_empty(cur_snapshot)) {
        ret = -ENOTEMPTY;
        goto out_writer;
    }

    snapshot =
        vecfs_ann_snapshot_create_empty_like(
            cur_snapshot,
            request->node_capacity);

    if (IS_ERR(snapshot)) {
        ret = PTR_ERR(snapshot);
        goto out_writer;
    }

    transaction->present_nodes =
        bitmap_zalloc(
            request->node_capacity,
            GFP_NOFS);

    if (!transaction->present_nodes) {
        vecfs_ann_snapshot_free(snapshot);
        ret = -ENOMEM;
        goto out_writer;
    }

    ret = rhashtable_init(
        &transaction->labels,
        &vecfs_ann_import_label_params);
    if (ret) {
        bitmap_free(
            transaction->present_nodes);

        transaction->present_nodes = NULL;

        vecfs_ann_snapshot_free(snapshot);
        goto out_writer;
    }

    transaction->labels_initialized = true;
    transaction->snapshot = snapshot;
    transaction->capacity =
        request->node_capacity;
    transaction->expected_nodes =
        request->expected_nodes;
    transaction->imported_nodes = 0;

    get_random_bytes(
        &transaction->transaction_id,
        sizeof(transaction->transaction_id));

    if (!transaction->transaction_id)
        transaction->transaction_id = 1;

    transaction->active = true;

    *transaction_id =
        transaction->transaction_id;

    ret = 0;

out_writer:
    mutex_unlock(
        &cache->writer_lock);

out_exchange:
    mutex_unlock(
        &sbi->s_ann_exchange.lock);

    return ret;
}

static int vecfs_ann_import_insert_label(
    struct vecfs_ann_import_transaction *transaction,
    u64 external_id,
    u32 node_id)
{
    struct vecfs_ann_import_label *existing;
    struct vecfs_ann_import_label *entry;
    int ret;

    if (external_id ==
        VECFS_ANN_EXTERNAL_ID_AUTO)
        return -EINVAL;

    existing = rhashtable_lookup_fast(
        &transaction->labels,
        &external_id,
        vecfs_ann_import_label_params);

    if (existing)
        return -EEXIST;

    entry = kmalloc(
        sizeof(*entry),
        GFP_NOFS);
    if (!entry)
        return -ENOMEM;

    entry->external_id =
        external_id;

    entry->node_id =
        node_id;

    ret = rhashtable_insert_fast(
        &transaction->labels,
        &entry->hash_node,
        vecfs_ann_import_label_params);

    if (ret)
        kfree(entry);

    return ret;
}

static int vecfs_ann_import_validate_node(
    const struct vecfs_ann_import_transaction *transaction,
    const struct vecfs_ann_import_node *node)
{
    u16 level;
    u16 i;

    if (!transaction || !node)
        return -EINVAL;

    if (node->node_id <
            VECFS_ANN_FIRST_NODE_ID ||
        node->node_id >=
            transaction->capacity)
        return -EINVAL;

    if (test_bit(
            node->node_id,
            transaction->present_nodes))
        return -EEXIST;

    if (node->external_id ==
        VECFS_ANN_EXTERNAL_ID_AUTO)
        return -EINVAL;

    if (node->level >=
        transaction->snapshot->max_levels)
        return -EINVAL;

    if (!node->vector ||
        node->vector_bytes !=
            transaction->snapshot->vector_bytes)
        return -EINVAL;

    if (node->level0_count >
        transaction->snapshot->m0)
        return -EINVAL;

    if (node->level0_count &&
        !node->level0_links)
        return -EINVAL;

    if (node->level &&
        (!node->upper_counts ||
         !node->upper_links))
        return -EINVAL;

    for (i = 0;
         i < node->level0_count;
         i++) {
        u32 neighbor =
            node->level0_links[i];

        if (neighbor <
                VECFS_ANN_FIRST_NODE_ID ||
            neighbor >=
                transaction->capacity ||
            neighbor == node->node_id)
            return -EINVAL;
    }

    for (level = 1;
         level <= node->level;
         level++) {
        u16 count =
            node->upper_counts[level - 1];

        if (count >
            transaction->snapshot->m)
            return -EINVAL;

        for (i = 0; i < count; i++) {
            u32 neighbor =
                node->upper_links[
                    (size_t)(level - 1) *
                        transaction->snapshot->m +
                    i];

            if (neighbor <
                    VECFS_ANN_FIRST_NODE_ID ||
                neighbor >=
                    transaction->capacity ||
                neighbor == node->node_id)
                return -EINVAL;
        }
    }

    return 0;
}

static int vecfs_ann_import_copy_node(
    struct vecfs_ann_import_transaction *transaction,
    const struct vecfs_ann_import_node *node)
{
    struct vecfs_ann_snapshot *snapshot =
        transaction->snapshot;

    struct vecfs_ann_insert_request request;
    u32 *level0_links;
    u16 level;
    u16 i;
    int ret;

    memset(
        &request,
        0,
        sizeof(request));

    request.vector =
        node->vector;

    request.vector_bytes =
        node->vector_bytes;

    request.external_id =
        node->external_id;

    request.owner_type =
        VECFS_ANN_OWNER_IMPORTED;

    request.vector_ordinal =
        VECFS_ANN_ORDINAL_AUTO;

    ret = vecfs_ann_snapshot_init_node(
        snapshot,
        node->node_id,
        node->level,
        node->vector,
        &request);
    if (ret)
        return ret;

    level0_links =
        vecfs_ann_level0_links_mutable(
            snapshot,
            node->node_id);

    for (i = 0;
         i < node->level0_count;
         i++)
        level0_links[i] =
            node->level0_links[i];

    vecfs_ann_level0_set_count(
        snapshot,
        node->node_id,
        node->level0_count);

    for (level = 1;
         level <= node->level;
         level++) {
        u16 count =
            node->upper_counts[level - 1];

        u32 *links =
            vecfs_ann_upper_links_mutable(
                snapshot,
                node->node_id,
                level);

        for (i = 0; i < count; i++) {
            links[i] =
                node->upper_links[
                    (size_t)(level - 1) *
                        snapshot->m +
                    i];
        }

        vecfs_ann_upper_set_count(
            snapshot,
            node->node_id,
            level,
            count);
    }

    return 0;
}

int vecfs_ann_import_node(
    struct super_block *sb,
    u64 transaction_id,
    const struct vecfs_ann_import_node *node)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_import_transaction *transaction;
    bool label_inserted = false;
    int ret;

    if (!sb || !node)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return -EINVAL;

    mutex_lock(
        &sbi->s_ann_exchange.lock);

    transaction =
        &sbi->s_ann_exchange.import;

    if (!transaction->active ||
        transaction->transaction_id !=
            transaction_id) {
        ret = -ENOENT;
        goto out;
    }

    if (transaction->imported_nodes >=
        transaction->expected_nodes) {
        ret = -E2BIG;
        goto out;
    }

    ret = vecfs_ann_import_validate_node(
        transaction,
        node);
    if (ret)
        goto out;

    ret = vecfs_ann_import_insert_label(
        transaction,
        node->external_id,
        node->node_id);
    if (ret)
        goto out;

    label_inserted = true;

    ret = vecfs_ann_import_copy_node(
        transaction,
        node);
    if (ret)
        goto rollback_label;

    __set_bit(
        node->node_id,
        transaction->present_nodes);

    transaction->imported_nodes++;

    ret = 0;
    goto out;

rollback_label:
    if (label_inserted)
        vecfs_ann_import_remove_label(
            transaction,
            node->external_id);

out:
    mutex_unlock(
        &sbi->s_ann_exchange.lock);

    return ret;
}


static int vecfs_ann_import_validate_graph(
    const struct vecfs_ann_import_transaction *transaction,
    u32 entry_node,
    u16 current_max_level)
{
    const struct vecfs_ann_snapshot *snapshot =
        transaction->snapshot;

    unsigned long node_id;

    if (transaction->imported_nodes !=
        transaction->expected_nodes)
        return -ENODATA;

    if (!transaction->imported_nodes) {
        if (entry_node !=
            VECFS_ANN_INVALID_NODE_ID)
            return -EINVAL;

        return 0;
    }

    if (entry_node <
            VECFS_ANN_FIRST_NODE_ID ||
        entry_node >=
            transaction->capacity ||
        !test_bit(
            entry_node,
            transaction->present_nodes))
        return -EINVAL;

    if (current_max_level >=
        snapshot->max_levels)
        return -EINVAL;

    if (vecfs_ann_node_at(snapshot, entry_node)->level !=
        current_max_level)
        return -EINVAL;

    for_each_set_bit(
        node_id,
        transaction->present_nodes,
        transaction->capacity) {
        const u32 *links;
        u16 count;
        u16 level;
        u16 i;

        count = vecfs_ann_level0_count(
            snapshot,
            node_id);

        links = vecfs_ann_level0_links(
            snapshot,
            node_id);

        for (i = 0; i < count; i++) {
            if (!test_bit(
                    links[i],
                    transaction->present_nodes))
                return -EINVAL;
        }

        for (level = 1;
             level <=
                 vecfs_ann_node_at(snapshot, node_id)->level;
             level++) {
            count = vecfs_ann_upper_count(
                snapshot,
                node_id,
                level);

            links = vecfs_ann_upper_links(
                snapshot,
                node_id,
                level);

            for (i = 0; i < count; i++) {
                u32 neighbor =
                    links[i];

                if (!test_bit(
                        neighbor,
                        transaction->present_nodes))
                    return -EINVAL;

                if (vecfs_ann_node_at(snapshot, neighbor)->level <
                    level)
                    return -EINVAL;
            }
        }
    }

    return 0;
}

static u32 vecfs_ann_import_target_arena(
    u32 active_arena)
{
    if (active_arena ==
        VECFS_ANN_ARENA_INO)
        return VECFS_ANN_REBUILD_ARENA_INO;

    return VECFS_ANN_ARENA_INO;
}

static int vecfs_ann_import_truncate_store(
    struct vecfs_ann_store *store)
{
    struct inode *inode;
    int ret;

    if (!store || !store->arena_inode)
        return 0;

    inode = store->arena_inode;

    inode_lock(inode);

    ret = vecfs_truncate_blocks(
        inode,
        0);

    if (!ret) {
        truncate_setsize(
            inode,
            0);

        mark_inode_dirty(inode);

        ret = write_inode_now(
            inode,
            1);
    }

    inode_unlock(inode);

    return ret;
}

int vecfs_ann_import_commit(
    struct super_block *sb,
    u64 transaction_id,
    u32 entry_node,
    u16 current_max_level,
    u64 *imported_nodes,
    u64 *new_epoch)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_import_transaction *transaction;

    struct vecfs_ann_snapshot *cur_snapshot;
    struct vecfs_ann_snapshot *snapshot;
    struct vecfs_ann_snapshot *old_snapshot;

    struct vecfs_ann_store *new_store = NULL;
    struct vecfs_ann_store *old_store;

    u32 target_arena;
    u64 epoch;
    int cleanup_ret;
    int ret;

    if (!sb ||
        !imported_nodes ||
        !new_epoch)
        return -EINVAL;

    if (sb_rdonly(sb))
        return -EROFS;

    sbi = VECFS_SB(sb);

    if (!sbi ||
        !sbi->s_ann_cache ||
        !sbi->s_ann_store)
        return -ENODEV;

    cache = sbi->s_ann_cache;

    mutex_lock(
        &sbi->s_ann_exchange.lock);

    transaction =
        &sbi->s_ann_exchange.import;

    if (!transaction->active ||
        transaction->transaction_id !=
            transaction_id) {
        ret = -ENOENT;
        goto out_exchange;
    }

    ret = vecfs_ann_import_validate_graph(
        transaction,
        entry_node,
        current_max_level);
    if (ret)
        goto out_exchange;

    mutex_lock(
        &cache->writer_lock);

    cur_snapshot = rcu_dereference_protected(
        cache->snapshot,
        lockdep_is_held(
            &cache->writer_lock));

    if (!vecfs_ann_snapshot_empty(cur_snapshot)) {
        ret = -ENOTEMPTY;
        goto out_writer;
    }

    snapshot =
        transaction->snapshot;

    snapshot->entry_node =
        entry_node;

    snapshot->max_level =
        current_max_level;

    snapshot->has_deletions = false;

    mutex_lock(&sbi->s_ann.lock);

    epoch = sbi->s_ann.epoch + 1;

    target_arena =
        vecfs_ann_import_target_arena(
            sbi->s_ann.active_arena_ino);

    mutex_unlock(&sbi->s_ann.lock);

    ret = vecfs_ann_store_rewrite_snapshot(
        sb,
        target_arena,
        snapshot,
        epoch,
        &new_store);
    if (ret)
        goto out_writer;

    mutex_lock(&sbi->s_ann.lock);

    sbi->s_ann.epoch =
        epoch;

    sbi->s_ann.state =
        snapshot->live_count ?
            VECFS_ANN_META_ACTIVE :
            VECFS_ANN_META_EMPTY;

    sbi->s_ann.active_arena_ino =
        target_arena;

    sbi->s_ann.rebuild_arena_ino = 0;
    sbi->s_ann.rebuild_state =
        VECFS_ANN_REBUILD_DISK_NONE;

    sbi->s_ann.live_nodes =
        snapshot->live_count;

    sbi->s_ann.deleted_nodes = 0;

    sbi->s_ann.next_node_id =
        snapshot->capacity;

    sbi->s_ann.entry_node_id =
        snapshot->entry_node;

    sbi->s_ann.config.current_max_level =
        snapshot->max_level;

    sbi->s_ann.dirty = true;

    mutex_unlock(&sbi->s_ann.lock);

    ret = vecfs_ann_meta_commit(
        sb,
        true);
    if (ret)
        goto out_new_store;

    /*
     * metadata 已持久化切换到新 arena。
     */
    old_store =
        sbi->s_ann_store;

    sbi->s_ann_store =
        new_store;

    new_store = NULL;

    ret = vecfs_ann_label_rebuild_locked(
        sb,
        snapshot);
    if (ret) {
        /*
         * 磁盘切换已经完成，label table 可在下一次 mount 时重建。
         * 仍然发布 snapshot，避免内存与磁盘 active arena 不一致。
         */
        pr_err(
            "vecfs: imported graph committed but "
            "label index rebuild failed: %d\n",
            ret);
    }

    old_snapshot = cur_snapshot;

    rcu_assign_pointer(
        cache->snapshot,
        snapshot);

    transaction->snapshot = NULL;

    vecfs_ann_snapshot_put(old_snapshot);

    cleanup_ret =
        vecfs_ann_import_truncate_store(
            old_store);

    if (cleanup_ret)
        pr_warn(
            "vecfs: failed to truncate old empty "
            "ANN arena: %d\n",
            cleanup_ret);

    vecfs_ann_store_release_instance(
        old_store,
        false);

    *imported_nodes =
        transaction->imported_nodes;

    *new_epoch =
        epoch;

    vecfs_ann_import_transaction_reset(
        transaction);

    ret = 0;
    goto out_writer;

out_new_store:
    if (new_store) {
        cleanup_ret =
            vecfs_ann_import_truncate_store(
                new_store);

        if (cleanup_ret)
            pr_warn(
                "vecfs: failed to clean abandoned "
                "import arena: %d\n",
                cleanup_ret);

        vecfs_ann_store_release_instance(
            new_store,
            false);
    }

out_writer:
    mutex_unlock(
        &cache->writer_lock);

out_exchange:
    mutex_unlock(
        &sbi->s_ann_exchange.lock);

    return ret;
}

int vecfs_ann_import_abort(
    struct super_block *sb,
    u64 transaction_id)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_import_transaction *transaction;
    int ret = 0;

    if (!sb)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return -EINVAL;

    mutex_lock(
        &sbi->s_ann_exchange.lock);

    transaction =
        &sbi->s_ann_exchange.import;

    if (!transaction->active ||
        transaction->transaction_id !=
            transaction_id) {
        ret = -ENOENT;
        goto out;
    }

    vecfs_ann_import_transaction_reset(
        transaction);

out:
    mutex_unlock(
        &sbi->s_ann_exchange.lock);

    return ret;
}

int vecfs_ann_export_info(
    struct super_block *sb,
    u64 *node_count,
    u32 *capacity,
    u32 *entry_node,
    u16 *max_level,
    u64 *epoch)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *snapshot;

    if (!sb ||
        !node_count ||
        !capacity ||
        !entry_node ||
        !max_level ||
        !epoch)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi || !sbi->s_ann_cache)
        return -ENODEV;

    cache = sbi->s_ann_cache;

    rcu_read_lock();

    snapshot = rcu_dereference(
        cache->snapshot);

    if (!snapshot) {
        rcu_read_unlock();
        return -ENODEV;
    }

    *node_count =
        snapshot->live_count;

    *capacity =
        snapshot->capacity;

    *entry_node =
        snapshot->entry_node;

    *max_level =
        snapshot->max_level;

    *epoch = READ_ONCE(
        sbi->s_ann.epoch);

    rcu_read_unlock();

    return 0;
}

int vecfs_ann_export_node(
    struct super_block *sb,
    u32 node_id,
    struct vecfs_ann_export_node *node)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *snapshot;
    const struct vecfs_ann_cache_node *source;
    void *vector = NULL;
    u32 *level0 = NULL;
    u16 *upper_counts = NULL;
    u32 *upper_links = NULL;
    size_t upper_slots;
    u16 level;
    u16 count;
    int ret = 0;

    if (!sb || !node)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi || !sbi->s_ann_cache)
        return -ENODEV;

    cache = sbi->s_ann_cache;

    memset(node, 0, sizeof(*node));

    mutex_lock(
        &cache->writer_lock);

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

    source =
        vecfs_ann_node_at(snapshot, node_id);

    if (source->state !=
        VECFS_ANN_OBJ_LIVE) {
        ret = -ENOENT;
        goto out_unlock;
    }

    vector = kvmemdup(
        vecfs_ann_snapshot_vector(
            snapshot,
            node_id),
        snapshot->vector_bytes,
        GFP_NOFS);

    if (!vector) {
        ret = -ENOMEM;
        goto out_unlock;
    }

    count = vecfs_ann_level0_count(
        snapshot,
        node_id);

    if (count) {
        level0 = kvmemdup(
            vecfs_ann_level0_links(
                snapshot,
                node_id),
            (size_t)count * sizeof(u32),
            GFP_NOFS);

        if (!level0) {
            ret = -ENOMEM;
            goto out_free;
        }
    }

    if (source->level) {
        upper_counts = kvmalloc_array(
            source->level,
            sizeof(*upper_counts),
            GFP_NOFS);

        upper_slots =
            (size_t)source->level *
            snapshot->m;

        upper_links = kvcalloc(
            upper_slots,
            sizeof(*upper_links),
            GFP_NOFS);

        if (!upper_counts ||
            !upper_links) {
            ret = -ENOMEM;
            goto out_free;
        }

        for (level = 1;
             level <= source->level;
             level++) {
            const u32 *links;
            u16 i;

            count = vecfs_ann_upper_count(
                snapshot,
                node_id,
                level);

            upper_counts[level - 1] =
                count;

            links = vecfs_ann_upper_links(
                snapshot,
                node_id,
                level);

            for (i = 0; i < count; i++) {
                upper_links[
                    (size_t)(level - 1) *
                        snapshot->m +
                    i] = links[i];
            }
        }
    }

    node->node_id = node_id;
    node->external_id =
        source->external_id;
    node->level =
        source->level;

    node->level0_count =
        vecfs_ann_level0_count(
            snapshot,
            node_id);

    node->vector =
        vector;
    node->vector_bytes =
        snapshot->vector_bytes;

    node->level0_links =
        level0;

    node->upper_counts =
        upper_counts;

    node->upper_links =
        upper_links;

    mutex_unlock(
        &cache->writer_lock);

    return 0;

out_free:
    kvfree(upper_links);
    kvfree(upper_counts);
    kvfree(level0);
    kvfree(vector);

out_unlock:
    mutex_unlock(
        &cache->writer_lock);

    return ret;
}
