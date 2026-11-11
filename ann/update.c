// SPDX-License-Identifier: GPL-2.0

#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/rcupdate.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_indirect.h"
#include "../include/vecfs_ann_build.h"
#include "../include/vecfs_ann_cache.h"
#include "../include/vecfs_ann_label.h"
#include "../include/vecfs_ann_meta.h"
#include "../include/vecfs_ann_mutation.h"
#include "../include/vecfs_ann_store.h"

struct vecfs_ann_update_backup {
    u32 *node_ids;
    void **objects;
    u32 count;
    u32 capacity;
};

static int vecfs_ann_update_backup_init(
    struct vecfs_ann_update_backup *backup,
    u32 capacity,
    size_t object_bytes)
{
    if (!backup || !capacity)
        return -EINVAL;

    memset(
        backup,
        0,
        sizeof(*backup));

    backup->node_ids = kvmalloc_array(
        capacity,
        sizeof(*backup->node_ids),
        GFP_NOFS);

    backup->objects = kvcalloc(
        capacity,
        sizeof(*backup->objects),
        GFP_NOFS);

    if (!backup->node_ids ||
        !backup->objects) {
        kvfree(backup->objects);
        kvfree(backup->node_ids);
        memset(backup, 0, sizeof(*backup));
        return -ENOMEM;
    }

    backup->capacity = capacity;
    return 0;
}

static void vecfs_ann_update_backup_destroy(
    struct vecfs_ann_update_backup *backup)
{
    u32 i;

    if (!backup)
        return;

    for (i = 0;
         i < backup->count;
         i++)
        kvfree(backup->objects[i]);

    kvfree(backup->objects);
    kvfree(backup->node_ids);

    memset(
        backup,
        0,
        sizeof(*backup));
}

static int vecfs_ann_update_backup_add(
    struct super_block *sb,
    struct vecfs_ann_update_backup *backup,
    u32 node_id,
    size_t object_bytes)
{
    void *object;
    u32 i;
    int ret;

    for (i = 0;
         i < backup->count;
         i++) {
        if (backup->node_ids[i] ==
            node_id)
            return 0;
    }

    if (backup->count >=
        backup->capacity)
        return -ENOSPC;

    object = kvmalloc(
        object_bytes,
        GFP_NOFS);
    if (!object)
        return -ENOMEM;

    ret = vecfs_ann_object_read_raw(
        sb,
        node_id,
        object,
        object_bytes);
    if (ret) {
        kvfree(object);
        return ret;
    }

    backup->node_ids[
        backup->count] = node_id;

    backup->objects[
        backup->count] = object;

    backup->count++;

    return 0;
}

static int vecfs_ann_update_backup_restore(
    struct super_block *sb,
    const struct vecfs_ann_update_backup *backup,
    size_t object_bytes)
{
    u32 i;
    int first_error = 0;

    if (!backup)
        return -EINVAL;

    for (i = 0;
         i < backup->count;
         i++) {
        int ret;

        ret = vecfs_ann_object_restore_raw(
            sb,
            backup->node_ids[i],
            backup->objects[i],
            object_bytes);

        if (ret && !first_error)
            first_error = ret;
    }

    return first_error;
}


static int vecfs_ann_update_check_access(
    const struct vecfs_ann_cache_node *node,
    enum vecfs_ann_access_class access)
{
    if (access == VECFS_ANN_ACCESS_INTERNAL)
        return 0;

    if (access == VECFS_ANN_ACCESS_FILE)
        return node->owner_type ==
                   VECFS_ANN_OWNER_FILE ?
               0 : -EPERM;

    if (access == VECFS_ANN_ACCESS_STANDALONE)
        return node->owner_type ==
                   VECFS_ANN_OWNER_FILE ?
               -EPERM : 0;

    return -EINVAL;
}

static int vecfs_ann_update_persist_dirty(
    struct super_block *sb,
    const struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_build_context *build)
{
    unsigned long node_id;

    for_each_set_bit(
        node_id,
        build->dirty_nodes,
        build->dirty_capacity) {
        const u32 *links;
        u16 count;
        u16 level;
        int ret;

        if (node_id <
            VECFS_ANN_FIRST_NODE_ID)
            continue;

        count = vecfs_ann_level0_count(
            snapshot, node_id);

        links = vecfs_ann_level0_links(
            snapshot, node_id);

        ret = vecfs_ann_object_write_level0(
            sb,
            node_id,
            links,
            count);
        if (ret)
            return ret;

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

            ret = vecfs_ann_object_write_upper(
                sb,
                node_id,
                level,
                links,
                count);
            if (ret)
                return ret;
        }
    }

    return 0;
}

static int vecfs_ann_update_locked(
    struct super_block *sb,
    u32 node_id,
    const struct vecfs_ann_update_request *request,
    enum vecfs_ann_access_class access)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_store *store;

    struct vecfs_ann_snapshot *old;
    struct vecfs_ann_snapshot *next = NULL;

    struct vecfs_ann_build_context build;
    struct vecfs_ann_distance_session distance;
    struct vecfs_ann_update_backup backup;

    const struct vecfs_ann_cache_node *old_node;

    unsigned long dirty_node;

    u64 old_live_nodes;
    u64 old_deleted_nodes;
    u64 old_random_seed;

    u32 old_entry_node;
    u16 old_max_level;

    u64 random_state;

    u64 new_content_offset;
    u64 new_content_length;

    u16 old_effective_flags;
    u16 new_binding_flags;

    bool was_deleted;
    bool update_binding;
    bool write_binding;
    bool disk_may_be_modified = false;
    bool backup_initialized = false;
    bool build_initialized = false;

    int rollback_ret;
    int metadata_ret;
    int ret;

    if (!sb || !request)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return -EINVAL;

    cache = sbi->s_ann_cache;
    store = sbi->s_ann_store;

    if (!cache || !store)
        return -ENODEV;

    /*
     * 调用者必须持有 writer_lock。
     *
     * standalone update 和 file update 的外层入口都必须先取得该锁，
     * 从而保证：
     *
     * 1. old snapshot 在整个事务期间不会改变；
     * 2. label/node ID 映射不会改变；
     * 3. rebuild/import 不会切换 arena；
     * 4. 只有一个写事务修改持久对象。
     */
    lockdep_assert_held(
        &cache->writer_lock);

    memset(&build, 0, sizeof(build));
    memset(&distance, 0, sizeof(distance));
    memset(&backup, 0, sizeof(backup));

    old = rcu_dereference_protected(
        cache->snapshot,
        lockdep_is_held(
            &cache->writer_lock));

    if (!old ||
        !vecfs_ann_snapshot_id_valid(
            old,
            node_id))
        return -ENOENT;

    old_node = vecfs_ann_node_at(old, node_id);

    ret = vecfs_ann_update_check_access(
        old_node,
        access);
    if (ret)
        return ret;

    if (old_node->state !=
            VECFS_ANN_OBJ_LIVE &&
        old_node->state !=
            VECFS_ANN_OBJ_DELETED)
        return -ENOENT;

    if (!request->vector ||
        request->vector_bytes !=
            old->vector_bytes)
        return -EINVAL;

    if (request->neighbor_probability_ppm >
        1000000)
        return -EINVAL;

    if (request->update_binding &&
        old_node->owner_type !=
            VECFS_ANN_OWNER_FILE)
        return -EPERM;

    if (request->update_binding) {
        if (request->binding_flags &
            ~(VECFS_ANN_OBJ_F_HAS_CONTENT_RANGE |
              VECFS_ANN_OBJ_F_CONTENT_STALE |
              VECFS_ANN_OBJ_F_EXTERNAL_ID_AUTO))
            return -EINVAL;

        if ((request->binding_flags &
             VECFS_ANN_OBJ_F_HAS_CONTENT_RANGE) &&
            request->content_offset >
                U64_MAX -
                    request->content_length)
            return -EOVERFLOW;
    }

    was_deleted =
        old_node->state ==
        VECFS_ANN_OBJ_DELETED;

    old_effective_flags =
        vecfs_ann_snapshot_effective_flags(
            old,
            node_id);

    update_binding =
        request->update_binding;

    /*
     * FILE 向量更新的 binding 规则：
     *
     * - update_binding=true：
     *     使用请求给出的 offset/length/flags；
     *
     * - update_binding=false：
     *     保留旧 offset/length/range 标志；
     *     因为向量已经重新生成，默认清除 CONTENT_STALE。
     *
     * standalone/imported 节点不写 binding。
     */
    if (old_node->owner_type ==
        VECFS_ANN_OWNER_FILE) {
        write_binding = true;

        if (update_binding) {
            new_content_offset =
                request->content_offset;

            new_content_length =
                request->content_length;

            new_binding_flags =
                request->binding_flags;
        } else {
            new_content_offset =
                old_node->content_offset;

            new_content_length =
                old_node->content_length;

            new_binding_flags =
                old_effective_flags &
                ~VECFS_ANN_OBJ_F_CONTENT_STALE;
        }
    } else {
        write_binding = false;

        new_content_offset =
            old_node->content_offset;

        new_content_length =
            old_node->content_length;

        new_binding_flags =
            old_effective_flags;
    }

    /*
     * 在任何磁盘对象修改前保存 ANN 根元数据。
     */
    mutex_lock(&sbi->s_ann.lock);

    old_live_nodes =
        sbi->s_ann.live_nodes;

    old_deleted_nodes =
        sbi->s_ann.deleted_nodes;

    old_random_seed =
        sbi->s_ann.config.random_seed;

    old_entry_node =
        sbi->s_ann.entry_node_id;

    old_max_level =
        sbi->s_ann.config.current_max_level;

    random_state =
        old_random_seed;

    mutex_unlock(&sbi->s_ann.lock);

    /*
     * 构造私有的下一版本 snapshot。
     *
     * 查询继续通过 RCU 使用 old；next 在事务成功前不可见。
     */
    next = vecfs_ann_snapshot_clone_locked(
        old,
        old->capacity);
    if (IS_ERR(next))
        return PTR_ERR(next);

    /*
     * 在内存 snapshot 中更新向量。
     */
    memcpy(
        (void *)vecfs_ann_snapshot_vector(
            next,
            node_id),
        request->vector,
        next->vector_bytes);

    /*
     * 在 next 中更新 FILE binding。
     *
     * stale_count 和 stale_bitmap 也在该函数中同步更新。
     */
    if (write_binding) {
        ret = vecfs_ann_snapshot_update_binding(
            next,
            node_id,
            new_content_offset,
            new_content_length,
            new_binding_flags);
        if (ret)
            goto fail_snapshot;
    }

    /*
     * 更新已删除节点时，与 hnswlib 对齐，恢复其 LIVE 状态。
     *
     * 这里只先修改 next。磁盘状态会在完成目标对象备份后修改。
     */
    if (was_deleted) {
        vecfs_ann_node_mutable(next, node_id)->state =
            VECFS_ANN_OBJ_LIVE;

        next->live_count++;

        /*
         * DELETED -> LIVE 后需要重新扫描是否还有剩余
         * DELETED 节点；不可使用 has_stale（那是 FILE
         * content stale，与节点的 DELETED 状态无关）。
         */
        next->has_deletions =
            vecfs_ann_snapshot_has_deletions(
                next);
    }

    /*
     * 创建构图上下文。这里会完成全部内存分配。
     *
     * 进入 FPU 区域之后不允许分配内存、执行 I/O 或睡眠。
     */
    ret = vecfs_ann_build_context_init(
        sb,
        &build,
        next->capacity);
    if (ret)
        goto fail_snapshot;

    build_initialized = true;

    /*
     * 备份容量按 node capacity 分配，因此可以容纳所有可能被
     * updatePoint 修改的节点。
     */
    ret = vecfs_ann_update_backup_init(
        &backup,
        next->capacity,
        store->layout.stride);
    if (ret)
        goto fail_build;

    backup_initialized = true;

    /*
     * 目标节点可能在构图前因 DELETED -> LIVE 被修改，也一定会写入
     * 新向量，因此必须最先备份。
     */
    ret = vecfs_ann_update_backup_add(
        sb,
        &backup,
        node_id,
        store->layout.stride);
    if (ret)
        goto fail_backup_without_disk_change;

    /*
     * 更新已删除节点的持久状态。
     *
     * 此时目标对象已经完成备份，因此后续任意失败都能恢复原始
     * DELETED object。
     */
    if (was_deleted) {
        ret = vecfs_ann_object_unmark_deleted(
            sb,
            node_id);
        if (ret)
            goto fail_backup_without_disk_change;

        disk_may_be_modified = true;
    }

    /*
     * 整个 updatePoint 只有一次 FPU context 切换。
     */
    vecfs_ann_distance_begin(
        &distance,
        &sbi->s_ann_distance);

    if (!distance.active) {
        ret = -ENODEV;
        goto fail_rollback;
    }

    ret = vecfs_ann_build_update_node(
        next,
        &distance,
        &build,
        node_id,
        sbi->s_ann.config.ef_construction,
        request->neighbor_probability_ppm,
        &random_state);

    vecfs_ann_distance_end(
        &distance);

    if (ret)
        goto fail_rollback;

    /*
     * 构图完成后，dirty_nodes 精确描述即将被写盘的所有邻接表。
     * 在第一次邻接表写入前逐个备份其完整 object。
     *
     * vecfs_ann_update_backup_add() 会自动忽略已备份的目标节点。
     */
    for_each_set_bit(
        dirty_node,
        build.dirty_nodes,
        build.dirty_capacity) {
        if (dirty_node <
                VECFS_ANN_FIRST_NODE_ID ||
            dirty_node >=
                next->capacity)
            continue;

        ret = vecfs_ann_update_backup_add(
            sb,
            &backup,
            dirty_node,
            store->layout.stride);
        if (ret)
            goto fail_rollback;
    }

    /*
     * 以下所有持久化操作均由 backup 覆盖：
     *
     * 1. 目标向量；
     * 2. FILE binding；
     * 3. 目标节点和所有受影响邻居的 links；
     * 4. 目标节点 LIVE/DELETED 状态。
     */
    ret = vecfs_ann_object_write_vector(
        sb,
        node_id,
        request->vector,
        request->vector_bytes);
    if (ret)
        goto fail_rollback;

    disk_may_be_modified = true;

    if (write_binding) {
        ret = vecfs_ann_object_update_binding(
            sb,
            node_id,
            new_content_offset,
            new_content_length,
            new_binding_flags);
        if (ret)
            goto fail_rollback;
    }

    ret = vecfs_ann_update_persist_dirty(
        sb,
        next,
        &build);
    if (ret)
        goto fail_rollback;

    /*
     * 更新 ANN 根元数据。
     */
    mutex_lock(&sbi->s_ann.lock);

    sbi->s_ann.config.random_seed =
        random_state;

    sbi->s_ann.entry_node_id =
        next->entry_node;

    sbi->s_ann.config.current_max_level =
        next->max_level;

    /*
     * DELETED -> LIVE 的计数已由 object_unmark_deleted() 修改；
     * 其它更新不改变 live/deleted 数量。
     */
    sbi->s_ann.dirty = true;

    mutex_unlock(&sbi->s_ann.lock);

    ret = vecfs_ann_meta_commit(
        sb,
        true);
    if (ret)
        goto fail_rollback;

    /*
     * 持久对象和 metadata 全部提交后，才发布新 snapshot。
     *
     * 该函数内部执行 rcu_assign_pointer() 并 kref_put 旧快照
     * （call_rcu 延迟回收）。
     */
    ret = vecfs_ann_cache_publish_locked(
        cache,
        next);
    if (ret)
        goto fail_rollback;

    next = NULL;

    vecfs_ann_update_backup_destroy(
        &backup);
    backup_initialized = false;

    if (build.search)
        vecfs_ann_search_context_put(
            sb,
            build.search);

    build.search = NULL;

    vecfs_ann_build_context_destroy(
        &build);
    build_initialized = false;

    return 0;

fail_rollback:
    /*
     * 若当前仍处在 FPU session，必须先退出，再执行 I/O 回滚。
     */
    if (distance.active)
        vecfs_ann_distance_end(
            &distance);

    /*
     * 只要可能修改过任意对象，就恢复全部备份。
     *
     * 即使失败发生在第一次对象写入前，执行恢复也是安全的；这里用
     * disk_may_be_modified 减少不必要 I/O。
     */
    if (backup_initialized &&
        disk_may_be_modified) {
        rollback_ret =
            vecfs_ann_update_backup_restore(
                sb,
                &backup,
                store->layout.stride);

        if (rollback_ret)
            pr_err(
                "vecfs: failed to restore ANN objects "
                "after update failure: %d\n",
                rollback_ret);
    }

    /*
     * 恢复内存中的 ANN metadata，并通过一次新的 A/B commit 使
     * 恢复状态拥有更高 sequence。
     */
    mutex_lock(&sbi->s_ann.lock);

    sbi->s_ann.live_nodes =
        old_live_nodes;

    sbi->s_ann.deleted_nodes =
        old_deleted_nodes;

    sbi->s_ann.config.random_seed =
        old_random_seed;

    sbi->s_ann.entry_node_id =
        old_entry_node;

    sbi->s_ann.config.current_max_level =
        old_max_level;

    sbi->s_ann.dirty = true;

    mutex_unlock(&sbi->s_ann.lock);

    if (!sb_rdonly(sb)) {
        metadata_ret =
            vecfs_ann_meta_commit(
                sb,
                true);

        if (metadata_ret)
            pr_err(
                "vecfs: failed to commit restored ANN "
                "metadata after update failure: %d\n",
                metadata_ret);
    }

    goto fail_cleanup;

fail_backup_without_disk_change:
    /*
     * 目标对象尚未成功修改，不需要执行对象恢复，但 metadata 也没有
     * 发生有效状态变化。
     */
    if (distance.active)
        vecfs_ann_distance_end(
            &distance);

    goto fail_cleanup;

fail_build:
    if (distance.active)
        vecfs_ann_distance_end(
            &distance);

fail_cleanup:
    if (backup_initialized)
        vecfs_ann_update_backup_destroy(
            &backup);

    if (build_initialized) {
        if (build.search)
            vecfs_ann_search_context_put(
                sb,
                build.search);

        build.search = NULL;

        vecfs_ann_build_context_destroy(
            &build);
    }

fail_snapshot:
    vecfs_ann_snapshot_free(
        next);

    return ret;
}



int vecfs_ann_update_by_label(
    struct super_block *sb,
    u64 external_id,
    const struct vecfs_ann_update_request *request,
    enum vecfs_ann_access_class access)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_cache *cache;
    u32 node_id;
    int ret;

    if (!sbi || !request ||
        !sbi->s_ann_cache)
        return -EINVAL;

    cache = sbi->s_ann_cache;

    mutex_lock(&cache->writer_lock);

    ret = vecfs_ann_label_lookup_locked(
        sb,
        external_id,
        &node_id);

    if (!ret)
        ret = vecfs_ann_update_locked(
            sb,
            node_id,
            request,
            access);

    mutex_unlock(&cache->writer_lock);
    return ret;
}

int vecfs_ann_file_update(
    struct inode *inode,
    u32 ordinal,
    const struct vecfs_ann_update_request *request)
{
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *snapshot;
    u32 node_id;
    int ret;

    if (!inode ||
        !request ||
        !S_ISREG(inode->i_mode))
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

    ret = vecfs_ann_update_locked(
        inode->i_sb,
        node_id,
        request,
        VECFS_ANN_ACCESS_FILE);

out:
    mutex_unlock(&cache->writer_lock);
    return ret;
}
