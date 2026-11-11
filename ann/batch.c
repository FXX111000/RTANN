// SPDX-License-Identifier: GPL-2.0

#include <linux/atomic.h>
#include <linux/bitmap.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/ktime.h>
#include <linux/rcupdate.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_batch.h"
#include "../include/vecfs_ann_build.h"
#include "../include/vecfs_ann_cache.h"
#include "../include/vecfs_ann_indirect.h"
#include "../include/vecfs_ann_label.h"
#include "../include/vecfs_ann_meta.h"
#include "../include/vecfs_ann_store.h"

struct vecfs_ann_batch_state {
    u32 *node_ids;
    u16 *levels;

    bool *object_prepared;
    bool *object_published;
    bool *label_inserted;
    bool *mapping_inserted;

    u32 count;
};

static u32 vecfs_ann_batch_prng_next(u64 *state)
{
    u64 value = *state;

    if (!value)
        value = 0x9e3779b97f4a7c15ULL;

    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;

    *state = value;

    return (u32)(
        value * 2685821657736338717ULL);
}

static u16 vecfs_ann_batch_random_level(
    u64 *state,
    u16 m,
    u16 max_levels)
{
    u16 level = 0;

    while (level + 1 < max_levels &&
           vecfs_ann_batch_prng_next(state) % m == 0)
        level++;

    return level;
}

static int vecfs_ann_batch_state_init(
    struct vecfs_ann_batch_state *state,
    u32 count)
{
    if (!state || !count)
        return -EINVAL;

    memset(state, 0, sizeof(*state));

    state->node_ids = kvmalloc_array(
        count,
        sizeof(*state->node_ids),
        GFP_NOFS);

    state->levels = kvmalloc_array(
        count,
        sizeof(*state->levels),
        GFP_NOFS);

    state->object_prepared = kvcalloc(
        count,
        sizeof(*state->object_prepared),
        GFP_NOFS);

    state->object_published = kvcalloc(
        count,
        sizeof(*state->object_published),
        GFP_NOFS);

    state->label_inserted = kvcalloc(
        count,
        sizeof(*state->label_inserted),
        GFP_NOFS);

    state->mapping_inserted = kvcalloc(
        count,
        sizeof(*state->mapping_inserted),
        GFP_NOFS);

    if (!state->node_ids ||
        !state->levels ||
        !state->object_prepared ||
        !state->object_published ||
        !state->label_inserted ||
        !state->mapping_inserted) {
        kvfree(state->mapping_inserted);
        kvfree(state->label_inserted);
        kvfree(state->object_published);
        kvfree(state->object_prepared);
        kvfree(state->levels);
        kvfree(state->node_ids);

        memset(state, 0, sizeof(*state));
        return -ENOMEM;
    }

    state->count = count;
    return 0;
}

static void vecfs_ann_batch_state_destroy(
    struct vecfs_ann_batch_state *state)
{
    if (!state)
        return;

    kvfree(state->mapping_inserted);
    kvfree(state->label_inserted);
    kvfree(state->object_published);
    kvfree(state->object_prepared);
    kvfree(state->levels);
    kvfree(state->node_ids);

    memset(state, 0, sizeof(*state));
}

static int vecfs_ann_batch_validate_item(
    struct inode *inode,
    const struct vecfs_ann_batch_item *item)
{
    const struct vecfs_ann_insert_request *request;
    struct vecfs_sb_info *sbi;

    if (!inode || !item)
        return -EINVAL;

    sbi = VECFS_SB(inode->i_sb);
    request = &item->request;

    if (!request->vector ||
        request->vector_bytes !=
            sbi->s_ann.config.vector_bytes)
        return -EINVAL;

    if (request->owner_type !=
        VECFS_ANN_OWNER_FILE)
        return -EPERM;

    if (request->owner_ino != inode->i_ino)
        return -EXDEV;

    if (request->owner_generation !=
        inode->i_generation)
        return -ESTALE;

    if (request->vector_ordinal ==
        VECFS_ANN_ORDINAL_AUTO)
        return -EINVAL;

    if (request->flags &
        ~(VECFS_ANN_OBJ_F_HAS_CONTENT_RANGE |
          VECFS_ANN_OBJ_F_CONTENT_STALE |
          VECFS_ANN_OBJ_F_EXTERNAL_ID_AUTO))
        return -EINVAL;

    if ((request->flags &
         VECFS_ANN_OBJ_F_HAS_CONTENT_RANGE) &&
        request->content_offset >
            U64_MAX - request->content_length)
        return -EOVERFLOW;

    return 0;
}

static int vecfs_ann_batch_validate_unique(
    struct inode *inode,
    struct vecfs_ann_batch_item *items,
    u32 count)
{
    u32 i;
    u32 j;
    int ret;

    for (i = 0; i < count; i++) {
        u32 existing;

        ret = vecfs_ann_batch_validate_item(
            inode,
            &items[i]);
        if (ret)
            return ret;

        ret = vecfs_ann_get_node(
            inode,
            items[i].request.vector_ordinal,
            &existing);

        if (!ret)
            return -EEXIST;

        if (ret != -ENOENT)
            return ret;

        for (j = 0; j < i; j++) {
            if (items[j].request.vector_ordinal ==
                items[i].request.vector_ordinal)
                return -EEXIST;

            if (items[i].request.external_id !=
                    VECFS_ANN_EXTERNAL_ID_AUTO &&
                items[j].request.external_id ==
                    items[i].request.external_id)
                return -EEXIST;
        }
    }

    return 0;
}

static int vecfs_ann_batch_validate_labels_locked(
    struct super_block *sb,
    struct vecfs_ann_batch_item *items,
    u32 count)
{
    u32 i;

    for (i = 0; i < count; i++) {
        u64 external_id =
            items[i].request.external_id;
        u32 existing;
        int ret;

        if (external_id ==
            VECFS_ANN_EXTERNAL_ID_AUTO)
            continue;

        ret = vecfs_ann_label_lookup_locked(
            sb,
            external_id,
            &existing);

        if (!ret)
            return -EEXIST;

        if (ret != -ENOENT)
            return ret;
    }

    return 0;
}

static int vecfs_ann_batch_prepare_objects(
    struct super_block *sb,
    struct vecfs_ann_batch_item *items,
    struct vecfs_ann_batch_state *state)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    u64 random_state;
    u16 m;
    u16 max_levels;
    u32 i;
    int ret;

    mutex_lock(&sbi->s_ann.lock);

    random_state = sbi->s_ann.config.random_seed;
    m = sbi->s_ann.config.m;
    max_levels =
        sbi->s_ann.config.max_levels;

    mutex_unlock(&sbi->s_ann.lock);

    for (i = 0; i < state->count; i++) {
        struct vecfs_ann_object_create create;
        struct vecfs_ann_object_info info;

        memset(&create, 0, sizeof(create));

        state->levels[i] =
            vecfs_ann_batch_random_level(
                &random_state,
                m,
                max_levels);

        create.external_id =
            items[i].request.external_id;

        create.owner_ino =
            items[i].request.owner_ino;
        create.owner_generation =
            items[i].request.owner_generation;
        create.vector_ordinal =
            items[i].request.vector_ordinal;

        create.content_offset =
            items[i].request.content_offset;
        create.content_length =
            items[i].request.content_length;

        create.level =
            state->levels[i];
        create.owner_type =
            VECFS_ANN_OWNER_FILE;
        create.flags =
            items[i].request.flags;

        create.vector =
            items[i].request.vector;
        create.vector_bytes =
            items[i].request.vector_bytes;

        ret = vecfs_ann_object_prepare(
            sb,
            &create,
            &state->node_ids[i]);
        if (ret)
            return ret;

        state->object_prepared[i] = true;

        ret = vecfs_ann_object_read_info(
            sb,
            state->node_ids[i],
            &info);
        if (ret)
            return ret;

        items[i].request.external_id =
            info.external_id;

        items[i].result.node_id =
            state->node_ids[i];
        items[i].result.level =
            state->levels[i];
        items[i].result.external_id =
            info.external_id;
    }

    mutex_lock(&sbi->s_ann.lock);

    sbi->s_ann.config.random_seed = random_state;
    sbi->s_ann.dirty = true;

    mutex_unlock(&sbi->s_ann.lock);

    return 0;
}

static int vecfs_ann_batch_init_snapshot_nodes(
    struct vecfs_ann_snapshot *snapshot,
    struct vecfs_ann_batch_item *items,
    const struct vecfs_ann_batch_state *state)
{
    u32 i;
    int ret;

    for (i = 0; i < state->count; i++) {
        ret = vecfs_ann_snapshot_init_node(
            snapshot,
            state->node_ids[i],
            state->levels[i],
            items[i].request.vector,
            &items[i].request);
        if (ret)
            return ret;
    }

    return 0;
}

static int vecfs_ann_batch_build_graph(
    struct super_block *sb,
    struct vecfs_ann_snapshot *snapshot,
    struct vecfs_ann_build_context *build,
    const struct vecfs_ann_batch_state *state)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_distance_session distance;
    u32 i;
    int ret = 0;

    memset(&distance, 0, sizeof(distance));

    /*
     * 整个批量事务只切换一次 FPU。
     */
    vecfs_ann_distance_begin(
        &distance,
        &sbi->s_ann_distance);

    if (!distance.active)
        return -ENODEV;

    for (i = 0; i < state->count; i++) {
        if (snapshot->entry_node ==
            VECFS_ANN_INVALID_NODE_ID) {
            snapshot->entry_node =
                state->node_ids[i];
            snapshot->max_level =
                state->levels[i];

            __set_bit(
                state->node_ids[i],
                build->dirty_nodes);
            continue;
        }

        ret = vecfs_ann_build_insert_node(
            snapshot,
            &distance,
            build,
            state->node_ids[i],
            state->levels[i],
            sbi->s_ann.config.ef_construction,
            NULL, NULL);
        if (ret)
            break;
    }

    vecfs_ann_distance_end(&distance);
    return ret;
}

static int vecfs_ann_batch_persist_node_links(
    struct super_block *sb,
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    const u32 *links;
    u16 count;
    u16 level;
    int ret;

    count = vecfs_ann_level0_count(
        snapshot,
        node_id);

    links = vecfs_ann_level0_links(
        snapshot,
        node_id);

    ret = vecfs_ann_object_write_level0(
        sb,
        node_id,
        links,
        count);
    if (ret)
        return ret;

    for (level = 1;
         level <= vecfs_ann_node_at(snapshot, node_id)->level;
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

    return 0;
}

static int vecfs_ann_batch_persist_dirty(
    struct super_block *sb,
    const struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_build_context *build)
{
    unsigned long node_id;
    int ret;

    for_each_set_bit(
        node_id,
        build->dirty_nodes,
        build->dirty_capacity) {
        if (node_id <
            VECFS_ANN_FIRST_NODE_ID)
            continue;

        ret = vecfs_ann_batch_persist_node_links(
            sb,
            snapshot,
            node_id);
        if (ret)
            return ret;
    }

    return 0;
}

static int vecfs_ann_batch_restore_old_links(
    struct super_block *sb,
    const struct vecfs_ann_snapshot *old,
    const struct vecfs_ann_build_context *build)
{
    unsigned long node_id;
    int first_error = 0;

    for_each_set_bit(
        node_id,
        build->dirty_nodes,
        build->dirty_capacity) {
        int ret;

        if (node_id <
                VECFS_ANN_FIRST_NODE_ID ||
            node_id >= old->capacity)
            continue;

        if (vecfs_ann_node_at(old, node_id)->state !=
                VECFS_ANN_OBJ_LIVE &&
            vecfs_ann_node_at(old, node_id)->state !=
                VECFS_ANN_OBJ_DELETED)
            continue;

        ret = vecfs_ann_batch_persist_node_links(
            sb,
            old,
            node_id);

        if (ret && !first_error)
            first_error = ret;
    }

    return first_error;
}

static int vecfs_ann_batch_bind_mappings(
    struct inode *inode,
    struct vecfs_ann_batch_item *items,
    struct vecfs_ann_batch_state *state)
{
    u32 i;
    int ret;

    for (i = 0; i < state->count; i++) {
        ret = vecfs_ann_bind_file_node(
            inode,
            items[i].request.vector_ordinal,
            state->node_ids[i],
            false);
        if (ret)
            return ret;

        state->mapping_inserted[i] = true;
    }

    return 0;
}

static int vecfs_ann_batch_insert_labels(
    struct super_block *sb,
    struct vecfs_ann_batch_item *items,
    struct vecfs_ann_batch_state *state)
{
    u32 i;
    int ret;

    for (i = 0; i < state->count; i++) {
        ret = vecfs_ann_label_insert_locked(
            sb,
            items[i].result.external_id,
            state->node_ids[i]);
        if (ret)
            return ret;

        state->label_inserted[i] = true;
    }

    return 0;
}

static int vecfs_ann_batch_publish_objects(
    struct super_block *sb,
    struct vecfs_ann_batch_state *state)
{
    u32 i;
    int ret;

    for (i = 0; i < state->count; i++) {
        ret = vecfs_ann_object_publish(
            sb,
            state->node_ids[i],
            false);
        if (ret)
            return ret;

        state->object_published[i] = true;
    }

    return 0;
}

static void vecfs_ann_batch_rollback(
    struct inode *inode,
    struct vecfs_ann_batch_item *items,
    struct vecfs_ann_batch_state *state)
{
    struct super_block *sb = inode->i_sb;
    u32 i = state->count;

    while (i--) {
        if (state->mapping_inserted[i]) {
            vecfs_ann_clear_node(
                inode,
                items[i].request.vector_ordinal);

            state->mapping_inserted[i] = false;
        }

        if (state->label_inserted[i]) {
            vecfs_ann_label_remove_locked(
                sb,
                items[i].result.external_id,
                state->node_ids[i]);

            state->label_inserted[i] = false;
        }

        if (state->object_prepared[i]) {
            int ret;

            ret = vecfs_ann_object_mark_deleted(
                sb,
                state->node_ids[i]);

            /* mark_deleted 返回 0 表示成功（对象已删）；-EALREADY/-EINVAL
             * 是幂等/参数失败，不算错。仅负值且非上述二者才告警。 */
            if (ret < 0 &&
                ret != -EALREADY &&
                ret != -EINVAL)
                pr_warn(
                    "vecfs: batch rollback node %u "
                    "failed: %d\n",
                    state->node_ids[i],
                    ret);
        }
    }
}

int vecfs_ann_batch_insert_files_many(
    struct super_block *sb,
    struct vecfs_ann_batch_file *files,
    u32 n_files,
    u32 total_items,
    vecfs_ann_batch_publish_guard_fn publish_guard,
    bool copy_full)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *old;
    struct vecfs_ann_snapshot *next = NULL;
    struct vecfs_ann_build_context build;
    struct vecfs_ann_batch_state *states = NULL;
    bool *state_init = NULL;
    u32 final_capacity = 0;
    u32 i;
    bool build_initialized = false;
    int restore_ret;
    int ret;

    if (!sb || !files || !n_files)
        return -EINVAL;

    if (total_items > VECFS_ANN_BATCH_MAX_ITEMS)
        return -E2BIG;

    sbi = VECFS_SB(sb);
    cache = sbi->s_ann_cache;

    if (!cache)
        return -ENODEV;

    states = kcalloc(n_files, sizeof(*states), GFP_NOFS);
    if (!states)
        return -ENOMEM;

    state_init = kcalloc(n_files, sizeof(*state_init), GFP_NOFS);
    if (!state_init) {
        kfree(states);
        return -ENOMEM;
    }

    memset(&build, 0, sizeof(build));

    /* 锁外：逐文件基础校验 + 幂等/唯一性校验 + state 初始化。 */
    for (i = 0; i < n_files; i++) {
        struct vecfs_ann_batch_file *f = &files[i];

        if (!f->inode || !f->items || !f->count) {
            ret = -EINVAL;
            goto out_release_states;
        }

        if (!S_ISREG(f->inode->i_mode)) {
            ret = -EINVAL;
            goto out_release_states;
        }

        if (f->inode->i_sb != sb) {
            ret = -EXDEV;
            goto out_release_states;
        }

        if (f->count > VECFS_ANN_BATCH_MAX_ITEMS) {
            ret = -E2BIG;
            goto out_release_states;
        }

        ret = vecfs_ann_batch_validate_unique(
            f->inode,
            f->items,
            f->count);
        if (ret)
            goto out_release_states;

        ret = vecfs_ann_batch_state_init(
            &states[i],
            f->count);
        if (ret)
            goto out_release_states;

        state_init[i] = true;
    }

    mutex_lock(&cache->writer_lock);

    old = rcu_dereference_protected(
        cache->snapshot,
        lockdep_is_held(&cache->writer_lock));

    if (!old) {
        ret = -ENODEV;
        goto out_unlock;
    }

    /*
     * 关键：所有文件的 prepare_objects（推进全局 next_node_id 的 node id
     * 分配）必须先于 clone 统一完成，final_capacity 才能覆盖跨文件的最大
     * node id。逐文件 validate_labels_locked + prepare_objects。
     */
    for (i = 0; i < n_files; i++) {
        struct vecfs_ann_batch_file *f = &files[i];

        ret = vecfs_ann_batch_validate_labels_locked(
            sb,
            f->items,
            f->count);
        if (ret)
            goto out_rollback;

        ret = vecfs_ann_batch_prepare_objects(
            sb,
            f->items,
            &states[i]);
        if (ret)
            goto out_rollback;
    }

    for (i = 0; i < n_files; i++) {
        u32 last = states[i].node_ids[states[i].count - 1] + 1;

        if (last > final_capacity)
            final_capacity = last;
    }

    {
        u64 clone_t0 = ktime_get_ns();

        next = vecfs_ann_snapshot_clone_locked_copy(
            old,
            final_capacity,
            copy_full);

        /* 消融观测：clone 成本（COW ON=refcount / OFF=全拷贝）与图构建解耦。 */
        atomic64_add(ktime_get_ns() - clone_t0, &cache->clone_ns);
        atomic64_inc(&cache->clone_count);
    }
    if (IS_ERR(next)) {
        ret = PTR_ERR(next);
        next = NULL;
        goto out_rollback;
    }

    for (i = 0; i < n_files; i++) {
        ret = vecfs_ann_batch_init_snapshot_nodes(
            next,
            files[i].items,
            &states[i]);
        if (ret)
            goto out_rollback;
    }

    ret = vecfs_ann_build_context_init(
        sb,
        &build,
        next->capacity);
    if (ret)
        goto out_rollback;

    build_initialized = true;

    /* 共享一个 build context，dirty_nodes 跨文件累积。 */
    {
        u64 build_t0 = ktime_get_ns();

        for (i = 0; i < n_files; i++) {
            ret = vecfs_ann_batch_build_graph(
                sb,
                next,
                &build,
                &states[i]);
            if (ret)
                goto out_restore;
        }

        /* 消融观测：HNSW 图构建耗时（逐节点插入 + ef_construction 贪心搜索）。 */
        atomic64_add(ktime_get_ns() - build_t0, &cache->build_ns);
        atomic64_inc(&cache->build_count);
    }

    {
        u64 persist_t0 = ktime_get_ns();

        ret = vecfs_ann_batch_persist_dirty(
            sb,
            next,
            &build);

        /* 消融观测：派生批量持久化耗时（node links 写缓冲，异步 mark_buffer_dirty）。 */
        atomic64_add(ktime_get_ns() - persist_t0, &cache->persist_ns);
        atomic64_inc(&cache->persist_count);
    }
    if (ret)
        goto out_restore;

    {
        u64 publish_t0 = ktime_get_ns();

        for (i = 0; i < n_files; i++) {
            struct vecfs_ann_batch_file *f = &files[i];

            ret = vecfs_ann_batch_bind_mappings(
                f->inode,
                f->items,
                &states[i]);
            if (ret)
                goto out_restore;

            ret = vecfs_ann_batch_insert_labels(
                sb,
                f->items,
                &states[i]);
            if (ret)
                goto out_restore;

            ret = vecfs_ann_batch_publish_objects(
                sb,
                &states[i]);
            if (ret)
                goto out_restore;
        }

        /* 消融观测：发布耗时（bind + labels + publish_objects）。 */
        atomic64_add(ktime_get_ns() - publish_t0, &cache->publish_ns);
        atomic64_inc(&cache->publish_count);
    }

    /* 逐文件版本复核（每文件自己的 guard_ctx）。任一失败整批回滚。 */
    if (publish_guard) {
        for (i = 0; i < n_files; i++) {
            if (!publish_guard(
                    files[i].inode,
                    files[i].guard_ctx)) {
                ret = -ESTALE;
                goto out_restore;
            }
        }
    }

    mutex_lock(&sbi->s_ann.lock);

    sbi->s_ann.entry_node_id =
        next->entry_node;

    sbi->s_ann.config.current_max_level =
        next->max_level;

    sbi->s_ann.state =
        VECFS_ANN_META_ACTIVE;

    sbi->s_ann.dirty = true;

    mutex_unlock(&sbi->s_ann.lock);

    {
        u64 persist_t0 = ktime_get_ns();

        ret = vecfs_ann_meta_commit(
            sb,
            false);

        /* 消融观测：metadata 提交耗时（sync=false，异步 mark_buffer_dirty）。 */
        atomic64_add(ktime_get_ns() - persist_t0, &cache->persist_ns);
        atomic64_inc(&cache->persist_count);
    }
    if (ret)
        goto out_restore;

    ret = vecfs_ann_cache_publish_locked(
        cache,
        next);
    if (ret)
        goto out_restore;

    next = NULL;
    ret = 0;
    goto out_release_build;

out_restore:
    restore_ret =
        vecfs_ann_batch_restore_old_links(
            sb,
            old,
            &build);

    if (restore_ret)
        pr_err(
            "vecfs: failed to restore old ANN links: %d\n",
            restore_ret);

out_rollback:
    for (i = 0; i < n_files; i++) {
        if (state_init[i])
            vecfs_ann_batch_rollback(
                files[i].inode,
                files[i].items,
                &states[i]);
    }

    /*
     * 回滚修改了对象计数，尽力提交修复后的 metadata。
     */
    if (!sb_rdonly(sb))
        vecfs_ann_meta_commit(
            sb,
            true);

out_release_build:
    if (build_initialized) {
        if (build.search)
            vecfs_ann_search_context_put(
                sb,
                build.search);

        vecfs_ann_build_context_destroy(
            &build);
    }

    vecfs_ann_snapshot_free(next);

out_unlock:
    mutex_unlock(&cache->writer_lock);

out_release_states:
    for (i = 0; i < n_files; i++) {
        if (state_init[i])
            vecfs_ann_batch_state_destroy(
                &states[i]);
    }

    kfree(state_init);
    kfree(states);

    return ret;
}

int vecfs_ann_batch_insert_file_guarded(
    struct inode *inode,
    struct vecfs_ann_batch_item *items,
    u32 count,
    vecfs_ann_batch_publish_guard_fn publish_guard,
    void *ctx)
{
    struct vecfs_ann_batch_file file;

    if (!inode || !items || !count)
        return -EINVAL;

    if (!S_ISREG(inode->i_mode))
        return -EINVAL;

    if (count > VECFS_ANN_BATCH_MAX_ITEMS)
        return -E2BIG;

    file.inode = inode;
    file.items = items;
    file.count = count;
    file.guard_ctx = ctx;

    return vecfs_ann_batch_insert_files_many(
        inode->i_sb,
        &file,
        1,
        count,
        publish_guard,
        false);
}

int vecfs_ann_batch_insert_file(
    struct inode *inode,
    struct vecfs_ann_batch_item *items,
    u32 count)
{
    return vecfs_ann_batch_insert_file_guarded(
        inode,
        items,
        count,
        NULL,
        NULL);
}
