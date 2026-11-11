// SPDX-License-Identifier: GPL-2.0

#include <linux/bitmap.h>
#include <linux/cpumask.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/ktime.h>
#include <linux/rcupdate.h>
#include <linux/slab.h>
#include <linux/spinlock.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_build.h"
#include "../include/vecfs_ann_cache.h"
#include "../include/vecfs_ann_indirect.h"
#include "../include/vecfs_ann_label.h"
#include "../include/vecfs_ann_meta.h"
#include "../include/vecfs_ann_rebuild.h"
#include "../include/vecfs_ann_store.h"
#include "../include/vecfs_Indirect.h"

struct vecfs_ann_rebuild_mapping {
    struct inode *inode;
    u32 ordinal;
};

static u32 vecfs_ann_rebuild_target_arena(
    u32 current_arena)
{
    if (current_arena ==
        VECFS_ANN_ARENA_INO)
        return VECFS_ANN_REBUILD_ARENA_INO;

    return VECFS_ANN_ARENA_INO;
}

static int vecfs_ann_rebuild_init_live_node(
    struct vecfs_ann_snapshot *destination,
    const struct vecfs_ann_snapshot *source,
    u32 node_id)
{
    const struct vecfs_ann_cache_node *source_node;
    struct vecfs_ann_insert_request request;

    source_node =
        vecfs_ann_node_at(source, node_id);

    memset(
        &request,
        0,
        sizeof(request));

    request.vector =
        vecfs_ann_snapshot_vector(
            source,
            node_id);

    request.vector_bytes =
        source->vector_bytes;

    request.external_id =
        source_node->external_id;

    request.owner_ino =
        source_node->owner_ino;

    request.owner_generation =
        source_node->owner_generation;

    request.vector_ordinal =
        source_node->vector_ordinal;

    request.content_offset =
        source_node->content_offset;

    request.content_length =
        source_node->content_length;

    request.owner_type =
        source_node->owner_type;

    request.flags =
        source_node->flags;

    return vecfs_ann_snapshot_init_node(
        destination,
        node_id,
        source_node->level,
        request.vector,
        &request);
}

static int vecfs_ann_rebuild_graph(
    struct super_block *sb,
    const struct vecfs_ann_snapshot *source,
    struct vecfs_ann_snapshot *destination)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_build_context build;
    struct vecfs_ann_distance_session distance;
    u32 node_id;
    int ret;

    memset(&build, 0, sizeof(build));
    memset(&distance, 0, sizeof(distance));

    ret = vecfs_ann_build_context_init(
        sb,
        &build,
        destination->capacity);
    if (ret)
        return ret;

    vecfs_ann_distance_begin(
        &distance,
        &sbi->s_ann_distance);

    if (!distance.active) {
        ret = -ENODEV;
        goto out_build;
    }

    /*
     * 按稳定 node ID 顺序重新插入所有 LIVE 节点。
     *
     * 每个节点保留原 level，因此不会改变节点层级分布。
     */
    for (node_id =
             VECFS_ANN_FIRST_NODE_ID;
         node_id < source->capacity;
         node_id++) {
        if (vecfs_ann_rebuild_should_cancel(sb)) {
            ret = -ECANCELED;
            break;
        }
        if (vecfs_ann_node_at(source, node_id)->state !=
            VECFS_ANN_OBJ_LIVE)
            continue;

        ret = vecfs_ann_rebuild_init_live_node(
            destination,
            source,
            node_id);
        if (ret)
            break;

        if (destination->entry_node ==
            VECFS_ANN_INVALID_NODE_ID) {
            destination->entry_node =
                node_id;

            destination->max_level =
                vecfs_ann_node_at(source, node_id)->level;

            continue;
        }

        ret = vecfs_ann_build_insert_node(
            destination,
            &distance,
            &build,
            node_id,
            vecfs_ann_node_at(source, node_id)->level,
            sbi->s_ann.config.ef_construction,
            NULL, NULL);
        if (ret)
            break;
    }

    vecfs_ann_distance_end(
        &distance);

out_build:
    if (distance.active)
        vecfs_ann_distance_end(
            &distance);

    if (build.search)
        vecfs_ann_search_context_put(
            sb,
            build.search);

    vecfs_ann_build_context_destroy(
        &build);

    return ret;
}

/* ---------------------------------------------------------------------
 * workqueue 并行全量重构（并发插入选项，hnswlib 式）。
 *
 * 与顺序 rebuild_graph 的唯一差异：build_insert_node 循环拆给多个
 * workqueue worker。每 worker 独立 build_context + distance_session
 * （各自 kernel_fpu_begin/end），节点邻接表用条纹链锁互斥，entry/max_level
 * 用 entry_lock 互斥。最终 publish 仍是单次 rcu_assign_pointer 原子切换，
 * 查询不感知并行。
 * ------------------------------------------------------------------- */

#define VECFS_ANN_REBUILD_MAX_WORKERS 8U

struct vecfs_ann_rebuild_parallel;

struct vecfs_ann_rebuild_worker {
    struct work_struct work;
    struct vecfs_ann_rebuild_parallel *ctx;
    u32 begin;   /* 本 worker 负责的 node_id 区间 [begin, end) */
    u32 end;
    int error;
    struct vecfs_ann_build_context build;
    struct vecfs_ann_distance_session distance;
};

struct vecfs_ann_rebuild_parallel {
    struct super_block *sb;
    const struct vecfs_ann_snapshot *source;
    struct vecfs_ann_snapshot *destination;
    struct vecfs_ann_link_locks locks;
    spinlock_t entry_lock;
    u32 first_live;   /* entry 种子：最低 node_id 的 LIVE 节点，不参与构图 */
    struct workqueue_struct *wq;
    struct vecfs_ann_rebuild_worker *workers;
};

static void vecfs_ann_rebuild_parallel_workfn(
    struct work_struct *work)
{
    struct vecfs_ann_rebuild_worker *w =
        container_of(work, struct vecfs_ann_rebuild_worker, work);
    struct vecfs_ann_rebuild_parallel *ctx = w->ctx;
    struct super_block *sb = ctx->sb;
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_snapshot *destination = ctx->destination;
    const struct vecfs_ann_snapshot *source = ctx->source;
    u32 node_id;
    int ret = 0;

    memset(&w->build, 0, sizeof(w->build));
    memset(&w->distance, 0, sizeof(w->distance));

    ret = vecfs_ann_build_context_init(
        sb,
        &w->build,
        destination->capacity);
    if (ret) {
        w->error = ret;
        return;
    }

    vecfs_ann_distance_begin(
        &w->distance,
        &sbi->s_ann_distance);

    if (!w->distance.active) {
        w->error = -ENODEV;
    } else {
        for (node_id = w->begin;
             node_id < w->end;
             node_id++) {
            if (vecfs_ann_rebuild_should_cancel(sb)) {
                ret = -ECANCELED;
                break;
            }
            if (vecfs_ann_node_at(source, node_id)->state !=
                VECFS_ANN_OBJ_LIVE)
                continue;
            if (node_id == ctx->first_live)
                continue;

            ret = vecfs_ann_build_insert_node(
                destination,
                &w->distance,
                &w->build,
                node_id,
                vecfs_ann_node_at(source, node_id)->level,
                sbi->s_ann.config.ef_construction,
                &ctx->locks,
                &ctx->entry_lock);
            if (ret)
                break;
        }

        vecfs_ann_distance_end(
            &w->distance);

        w->error = ret;
    }

    if (w->build.search)
        vecfs_ann_search_context_put(
            sb,
            w->build.search);

    vecfs_ann_build_context_destroy(
        &w->build);
}

static int vecfs_ann_rebuild_graph_parallel(
    struct super_block *sb,
    const struct vecfs_ann_snapshot *source,
    struct vecfs_ann_snapshot *destination)
{
    struct vecfs_ann_rebuild_parallel *ctx;
    u32 first_live = VECFS_ANN_INVALID_NODE_ID;
    u32 nworkers;
    u32 span;
    u32 node_id;
    u32 i;
    int ret;

    if (!source->live_count)
        return 0;

    /* Phase 0：顺序预初始化所有 LIVE 节点（仅 memcpy + 拷贝向量，快）。 */
    for (node_id = VECFS_ANN_FIRST_NODE_ID;
         node_id < source->capacity;
         node_id++) {
        if (vecfs_ann_node_at(source, node_id)->state !=
            VECFS_ANN_OBJ_LIVE)
            continue;

        ret = vecfs_ann_rebuild_init_live_node(
            destination,
            source,
            node_id);
        if (ret)
            return ret;

        if (first_live ==
            VECFS_ANN_INVALID_NODE_ID)
            first_live = node_id;

        if (!(node_id & 16383))
            schedule();
        else if (!(node_id & 63))
            cond_resched();
    }

    destination->entry_node = first_live;
    destination->max_level =
        vecfs_ann_node_at(source, first_live)->level;

    /* 单个 LIVE 节点：无链接可建。 */
    if (source->live_count == 1)
        return 0;

    nworkers = num_online_cpus();
    if (nworkers > VECFS_ANN_REBUILD_MAX_WORKERS)
        nworkers = VECFS_ANN_REBUILD_MAX_WORKERS;
    if (nworkers < 1)
        nworkers = 1;

    ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
    if (!ctx)
        return -ENOMEM;

    ctx->sb = sb;
    ctx->source = source;
    ctx->destination = destination;
    ctx->first_live = first_live;

    ctx->locks.locks = kcalloc(
        VECFS_ANN_BUILD_LOCK_STRIPES,
        sizeof(*ctx->locks.locks),
        GFP_KERNEL);
    if (!ctx->locks.locks) {
        ret = -ENOMEM;
        goto out_ctx;
    }
    ctx->locks.mask =
        VECFS_ANN_BUILD_LOCK_STRIPES - 1;
    for (i = 0; i < VECFS_ANN_BUILD_LOCK_STRIPES; i++)
        spin_lock_init(&ctx->locks.locks[i]);

    spin_lock_init(&ctx->entry_lock);

    ctx->wq = alloc_workqueue(
        "vecfs_ann_rebuild",
        WQ_UNBOUND,
        0);
    if (!ctx->wq) {
        ret = -ENOMEM;
        goto out_locks;
    }

    ctx->workers = kcalloc(
        nworkers,
        sizeof(*ctx->workers),
        GFP_KERNEL);
    if (!ctx->workers) {
        ret = -ENOMEM;
        goto out_wq;
    }

    /* [FIRST, capacity) 均分为 nworkers 段。 */
    span = source->capacity - VECFS_ANN_FIRST_NODE_ID;

    for (i = 0; i < nworkers; i++) {
        struct vecfs_ann_rebuild_worker *w =
            &ctx->workers[i];

        w->ctx = ctx;
        w->begin =
            VECFS_ANN_FIRST_NODE_ID +
            (u32)((u64)span * i / nworkers);
        w->end =
            VECFS_ANN_FIRST_NODE_ID +
            (u32)((u64)span * (i + 1) / nworkers);
        w->error = 0;

        INIT_WORK(&w->work,
                  vecfs_ann_rebuild_parallel_workfn);
    }

    for (i = 0; i < nworkers; i++)
        queue_work(ctx->wq, &ctx->workers[i].work);

    flush_workqueue(ctx->wq);

    ret = 0;
    for (i = 0; i < nworkers; i++) {
        if (ctx->workers[i].error && !ret)
            ret = ctx->workers[i].error;
    }

    destroy_workqueue(ctx->wq);
    kfree(ctx->workers);
    kfree(ctx->locks.locks);
    kfree(ctx);
    return ret;

out_wq:
    destroy_workqueue(ctx->wq);
out_locks:
    kfree(ctx->locks.locks);
out_ctx:
    kfree(ctx);
    return ret;
}

static void vecfs_ann_rebuild_clear_deleted_mapping(
    struct super_block *sb,
    const struct vecfs_ann_cache_node *node)
{
    struct inode *inode;
    u32 mapped_node;
    int ret;

    if (node->owner_type !=
            VECFS_ANN_OWNER_FILE ||
        !node->owner_ino)
        return;

    inode = ilookup(
        sb,
        node->owner_ino);

    if (!inode)
        return;

    if (inode->i_generation !=
        node->owner_generation)
        goto out_inode;

    ret = vecfs_ann_get_node(
        inode,
        node->vector_ordinal,
        &mapped_node);

    if (!ret &&
        mapped_node !=
            VECFS_ANN_EMPTY_NODE_ID)
        vecfs_ann_clear_node(
            inode,
            node->vector_ordinal);

out_inode:
    iput(inode);
}

static void vecfs_ann_rebuild_clear_deleted_mappings(
    struct super_block *sb,
    const struct vecfs_ann_snapshot *source)
{
    u32 node_id;

    for (node_id =
             VECFS_ANN_FIRST_NODE_ID;
         node_id < source->capacity;
         node_id++) {
        if (vecfs_ann_node_at(source, node_id)->state !=
            VECFS_ANN_OBJ_DELETED)
            continue;

        vecfs_ann_rebuild_clear_deleted_mapping(
            sb,
            vecfs_ann_node_at(source, node_id));
    }
}

static void vecfs_ann_rebuild_remove_deleted_labels(
    struct super_block *sb,
    const struct vecfs_ann_snapshot *source)
{
    u32 node_id;

    for (node_id =
             VECFS_ANN_FIRST_NODE_ID;
         node_id < source->capacity;
         node_id++) {
        int ret;

        if (vecfs_ann_node_at(source, node_id)->state !=
            VECFS_ANN_OBJ_DELETED)
            continue;

        ret = vecfs_ann_label_remove_locked(
            sb,
            vecfs_ann_node_at(source, node_id)->external_id,
            node_id);

        if (ret && ret != -ENOENT)
            pr_warn(
                "vecfs: rebuild failed to remove "
                "deleted label %llu: %d\n",
                vecfs_ann_node_at(source, node_id)->
                    external_id,
                ret);
    }
}

static int vecfs_ann_rebuild_release_old_arena(
    struct vecfs_ann_store *old_store)
{
    struct inode *inode;
    int ret;

    if (!old_store ||
        !old_store->arena_inode)
        return 0;

    inode = old_store->arena_inode;

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

int vecfs_ann_rebuild(
    struct super_block *sb,
    u32 flags,
    struct vecfs_ann_rebuild_report *report)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *source;
    struct vecfs_ann_snapshot *rebuilt = NULL;
    struct vecfs_ann_snapshot *old_snapshot;
    struct vecfs_ann_store *new_store = NULL;
    struct vecfs_ann_store *old_store;
    u32 old_arena;
    u32 new_arena;
    u64 old_epoch;
    u64 new_epoch;
    u64 start_ns;
    u32 node_id;
    int cleanup_ret;
    int ret;

    if (!sb || !report)
        return -EINVAL;

    if (flags &
        ~(VECFS_REBUILD_F_SYNC |
          VECFS_REBUILD_F_PRESERVE_LEVELS |
          VECFS_REBUILD_F_PARALLEL))
        return -EINVAL;

    if (sb_rdonly(sb))
        return -EROFS;

    sbi = VECFS_SB(sb);
    if (!sbi ||
        !sbi->s_ann_cache ||
        !sbi->s_ann_store)
        return -ENODEV;

    memset(report, 0, sizeof(*report));

    cache = sbi->s_ann_cache;
    start_ns = ktime_get_ns();

    mutex_lock(&cache->writer_lock);

    source =
        rcu_dereference_protected(
            cache->snapshot,
            lockdep_is_held(
                &cache->writer_lock));

    if (!source) {
        ret = -ENODEV;
        goto out_unlock;
    }

    mutex_lock(&sbi->s_ann.lock);

    old_arena =
        sbi->s_ann.active_arena_ino;

    old_epoch =
        sbi->s_ann.epoch;

    mutex_unlock(&sbi->s_ann.lock);

    new_arena =
        vecfs_ann_rebuild_target_arena(
            old_arena);

    new_epoch = old_epoch + 1;

    report->source_live_nodes =
        source->live_count;

    for (node_id =
             VECFS_ANN_FIRST_NODE_ID;
         node_id < source->capacity;
         node_id++) {
        if (vecfs_ann_node_at(source, node_id)->state ==
            VECFS_ANN_OBJ_DELETED)
            report->source_deleted_nodes++;
    }

    rebuilt =
        vecfs_ann_snapshot_create_empty_like(
            source,
            source->capacity);

    if (IS_ERR(rebuilt)) {
        ret = PTR_ERR(rebuilt);
        rebuilt = NULL;
        goto out_unlock;
    }

    if (flags & VECFS_REBUILD_F_PARALLEL)
        ret = vecfs_ann_rebuild_graph_parallel(
            sb,
            source,
            rebuilt);
    else
        ret = vecfs_ann_rebuild_graph(
            sb,
            source,
            rebuilt);
    if (ret)
        goto out_free_snapshot;

    /*
     * 先完整写入非活动 arena。
     */
    ret = vecfs_ann_store_rewrite_snapshot(
        sb,
        new_arena,
        rebuilt,
        new_epoch,
        &new_store);
    if (ret)
        goto out_free_snapshot;

    /*
     * 清除 deleted FILE 节点的 inode 映射。
     *
     * LIVE 节点保持原 node ID，因此不需要改写它们的映射。
     */
    vecfs_ann_rebuild_clear_deleted_mappings(
        sb,
        source);

    mutex_lock(&sbi->s_ann.lock);

    sbi->s_ann.state =
        rebuilt->live_count ?
            VECFS_ANN_META_ACTIVE :
            VECFS_ANN_META_EMPTY;

    sbi->s_ann.epoch =
        new_epoch;

    sbi->s_ann.active_arena_ino =
        new_arena;

    sbi->s_ann.rebuild_arena_ino = 0;
    sbi->s_ann.rebuild_state =
        VECFS_ANN_REBUILD_DISK_NONE;

    sbi->s_ann.live_nodes =
        rebuilt->live_count;

    sbi->s_ann.deleted_nodes = 0;

    sbi->s_ann.entry_node_id =
        rebuilt->entry_node;

    sbi->s_ann.config.current_max_level =
        rebuilt->max_level;

    /*
     * node ID 保持稳定且不复用。
     */
    sbi->s_ann.next_node_id =
        rebuilt->capacity;

    sbi->s_ann.dirty = true;

    mutex_unlock(&sbi->s_ann.lock);

    /*
     * A/B metadata commit 是新旧 arena 的持久化切换点。
     */
    ret = vecfs_ann_meta_commit(
        sb,
        true);
    if (ret)
        goto out_release_new_store;

    /*
     * metadata 已切换，之后 mount 将选择新 arena。
     */
    old_store =
        sbi->s_ann_store;

    sbi->s_ann_store =
        new_store;

    new_store = NULL;

    /*
     * LIVE label/node ID 保持不变，只需移除 tombstone label。
     */
    vecfs_ann_rebuild_remove_deleted_labels(
        sb,
        source);

    old_snapshot = source;

    rcu_assign_pointer(
        cache->snapshot,
        rebuilt);

    rebuilt = NULL;

    vecfs_ann_snapshot_put(old_snapshot);

    report->rebuilt_nodes =
        report->source_live_nodes;

    report->removed_nodes =
        report->source_deleted_nodes;

    report->old_epoch =
        old_epoch;

    report->new_epoch =
        new_epoch;

    report->old_arena_inode =
        old_arena;

    report->new_arena_inode =
        new_arena;

    /*
     * 新 snapshot 已发布，旧 store 不再是持久化后端。
     */
    cleanup_ret =
        vecfs_ann_rebuild_release_old_arena(
            old_store);

    if (cleanup_ret)
        pr_warn(
            "vecfs: failed to truncate old ANN "
            "arena %u: %d\n",
            old_arena,
            cleanup_ret);

    vecfs_ann_store_release_instance(
        old_store,
        false);

    report->duration_ns =
        ktime_get_ns() - start_ns;

    ret = 0;
    goto out_unlock;

out_release_new_store:
    /*
     * metadata 未切换，新 arena 仍然不可见，可以安全销毁。
     */
    if (new_store) {
        cleanup_ret =
            vecfs_ann_rebuild_release_old_arena(
                new_store);

        if (cleanup_ret)
            pr_warn(
                "vecfs: failed to clean abandoned "
                "rebuild arena: %d\n",
                cleanup_ret);

        vecfs_ann_store_release_instance(
            new_store,
            false);

        new_store = NULL;
    }

out_free_snapshot:
    vecfs_ann_snapshot_free(
        rebuilt);

out_unlock:
    mutex_unlock(&cache->writer_lock);

    report->duration_ns =
        ktime_get_ns() - start_ns;

    return ret;
}
