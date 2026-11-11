// SPDX-License-Identifier: GPL-2.0

#include <linux/bitmap.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/rcupdate.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_indirect.h"

#include "../include/vecfs_ann_build.h"
#include "../include/vecfs_ann_cache.h"
#include "../include/vecfs_ann_distance.h"
#include "../include/vecfs_ann_label.h"
#include "../include/vecfs_ann_meta.h"
#include "../include/vecfs_ann_search.h"
#include "../include/vecfs_ann_store.h"

/*
 * xorshift64* PRNG。
 *
 * writer_lock 已保证所有 HNSW 图构建写操作串行化；
 * s_ann.lock 只用于保护持久化元数据字段。
 */
static u32 vecfs_ann_prng_next(u64 *state)
{
    u64 value;

    value = *state;

    if (!value)
        value = 0x9e3779b97f4a7c15ULL;

    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;

    *state = value;

    return (u32)(
        value * 2685821657736338717ULL);
}

/*
 * 对默认 level multiplier 1/log(M)，有：
 *
 *     P(level >= L) = M^(-L)
 *
 * 因此每层以 1/M 的概率继续上升，与 hnswlib 默认随机层数
 * 分布一致。
 */
static u16 vecfs_ann_random_level(
    struct vecfs_sb_info *sbi)
{
    u64 state;
    u16 maximum_levels;
    u16 m;
    u16 level = 0;

    mutex_lock(&sbi->s_ann.lock);

    state = sbi->s_ann.config.random_seed;
    m = sbi->s_ann.config.m;
    maximum_levels =
        sbi->s_ann.config.max_levels;

    while (level + 1 < maximum_levels &&
           vecfs_ann_prng_next(&state) % m == 0)
        level++;

    sbi->s_ann.config.random_seed = state;
    sbi->s_ann.dirty = true;

    mutex_unlock(&sbi->s_ann.lock);

    return level;
}

/*
 * 初始化插入/更新使用的构图上下文。
 *
 * 所有可能在构图期间使用的内存均在进入 FPU 区域前完成分配。
 */
int vecfs_ann_build_context_init(
    struct super_block *sb,
    struct vecfs_ann_build_context *context,
    u32 node_capacity)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    u32 temporary_capacity;
    int ret;

    if (!sbi || !context || !node_capacity)
        return -EINVAL;

    memset(context, 0, sizeof(*context));

    ret = vecfs_ann_search_context_get(
        sb,
        node_capacity,
        sbi->s_ann.config.ef_construction,
        &context->search);
    if (ret)
        return ret;

    temporary_capacity = max_t(
        u32,
        sbi->s_ann.config.ef_construction,
        sbi->s_ann.config.m0 + 1);

    context->ordered = kvmalloc_array(
        temporary_capacity,
        sizeof(*context->ordered),
        GFP_NOFS);

    context->selected = kvmalloc_array(
        temporary_capacity,
        sizeof(*context->selected),
        GFP_NOFS);

    context->neighbor_ids = kvmalloc_array(
        temporary_capacity,
        sizeof(*context->neighbor_ids),
        GFP_NOFS);

    context->dirty_nodes = bitmap_zalloc(
        node_capacity,
        GFP_NOFS);

    context->candidate_bitmap = bitmap_zalloc(
        node_capacity,
        GFP_NOFS);

    context->neighbor_bitmap = bitmap_zalloc(
        node_capacity,
        GFP_NOFS);

    context->candidate_nodes = kvmalloc_array(
        node_capacity,
        sizeof(*context->candidate_nodes),
        GFP_NOFS);

    context->selected_neighbors = kvmalloc_array(
        node_capacity,
        sizeof(*context->selected_neighbors),
        GFP_NOFS);

    if (!context->ordered ||
        !context->selected ||
        !context->neighbor_ids ||
        !context->dirty_nodes ||
        !context->candidate_bitmap ||
        !context->neighbor_bitmap ||
        !context->candidate_nodes ||
        !context->selected_neighbors) {
        ret = -ENOMEM;
        goto fail;
    }

    context->temporary_capacity =
        temporary_capacity;

    context->dirty_capacity =
        node_capacity;

    context->set_capacity =
        node_capacity;

    return 0;

fail:
    vecfs_ann_search_context_put(
        sb,
        context->search);

    context->search = NULL;

    vecfs_ann_build_context_destroy(
        context);

    return ret;
}


void vecfs_ann_build_context_destroy(
    struct vecfs_ann_build_context *context)
{
    if (!context)
        return;

    bitmap_free(
        context->candidate_bitmap);

    bitmap_free(
        context->neighbor_bitmap);

    bitmap_free(
        context->dirty_nodes);

    kvfree(
        context->candidate_nodes);

    kvfree(
        context->selected_neighbors);

    kvfree(
        context->neighbor_ids);

    kvfree(
        context->selected);

    kvfree(
        context->ordered);

    /*
     * search context 由调用方归还，因为这里只保存 context 指针，
     * 没有对应的 super_block。
     */
    context->search = NULL;
    context->candidate_bitmap = NULL;
    context->neighbor_bitmap = NULL;
    context->dirty_nodes = NULL;
    context->candidate_nodes = NULL;
    context->selected_neighbors = NULL;
    context->neighbor_ids = NULL;
    context->selected = NULL;
    context->ordered = NULL;

    context->temporary_capacity = 0;
    context->dirty_capacity = 0;
    context->set_capacity = 0;
}


/*
 * 将一个内存 snapshot 节点的邻接表写回持久化对象。
 *
 * 此函数在 FPU 区域外执行。
 */
static int vecfs_ann_persist_node_links(
    struct super_block *sb,
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    const struct vecfs_ann_cache_node *node;
    const u32 *links;
    u16 count;
    u16 level;
    int ret;

    if (!vecfs_ann_snapshot_id_valid(
            snapshot,
            node_id))
        return -EINVAL;

    node = vecfs_ann_node_at(snapshot, node_id);

    if (node->state != VECFS_ANN_OBJ_LIVE &&
        node->state != VECFS_ANN_OBJ_DELETED)
        return -ENOENT;

    count = vecfs_ann_level0_count(
        snapshot,
        node_id);

    if (count > snapshot->m0)
        return -EUCLEAN;

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
         level <= node->level;
         level++) {
        count = vecfs_ann_upper_count(
            snapshot,
            node_id,
            level);

        if (count > snapshot->m)
            return -EUCLEAN;

        links = vecfs_ann_upper_links(
            snapshot,
            node_id,
            level);

        if (!links && count)
            return -EUCLEAN;

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

/*
 * 只持久化本次构图实际修改过的节点。
 */
static int vecfs_ann_persist_dirty_links(
    struct super_block *sb,
    const struct vecfs_ann_snapshot *snapshot,
    const unsigned long *dirty_nodes,
    u32 dirty_capacity)
{
    unsigned long node_id;
    int ret;

    if (!dirty_nodes)
        return -EINVAL;

    for_each_set_bit(
        node_id,
        dirty_nodes,
        dirty_capacity) {
        if (node_id < VECFS_ANN_FIRST_NODE_ID)
            continue;

        ret = vecfs_ann_persist_node_links(
            sb,
            snapshot,
            node_id);
        if (ret)
            return ret;
    }

    return 0;
}

/*
 * 创建 PREPARED 状态的持久化对象。
 *
 * object_prepare() 负责：
 *
 * - 分配内部 node ID；
 * - 自动 external ID（如果请求为 AUTO）；
 * - 写入向量和对象 owner 元数据；
 * - 初始化空邻接表。
 */
static int vecfs_ann_prepare_store_object(
    struct super_block *sb,
    const struct vecfs_ann_insert_request *request,
    u16 level,
    u32 *node_id)
{
    struct vecfs_ann_object_create create;

    memset(&create, 0, sizeof(create));

    create.external_id =
        request->external_id;

    create.owner_ino =
        request->owner_ino;

    create.owner_generation =
        request->owner_generation;

    create.vector_ordinal =
        request->vector_ordinal;

    create.content_offset =
        request->content_offset;

    create.content_length =
        request->content_length;

    create.level = level;

    create.owner_type =
        request->owner_type;

    create.flags =
        request->flags;

    create.vector =
        request->vector;

    create.vector_bytes =
        request->vector_bytes;

    return vecfs_ann_object_prepare(
        sb,
        &create,
        node_id);
}

/*
 * 在 writer_lock 内检查显式 external ID。
 *
 * 不再扫描 snapshot->nodes[]，直接使用 rhashtable。
 */
static int vecfs_ann_check_explicit_label_locked(
    struct super_block *sb,
    u64 external_id)
{
    u32 existing_node;
    int ret;

    if (external_id ==
        VECFS_ANN_EXTERNAL_ID_AUTO)
        return 0;

    ret = vecfs_ann_label_lookup_locked(
        sb,
        external_id,
        &existing_node);

    if (!ret)
        return -EEXIST;

    if (ret == -ENOENT)
        return 0;

    return ret;
}

/*
 * 检查文件绑定对象所引用的 inode。
 *
 * 调用时 inode 通常仍由文件描述符持有，因此 ilookup() 应能够找到。
 */
static struct inode *vecfs_ann_get_owner_inode(
    struct super_block *sb,
    const struct vecfs_ann_insert_request *request)
{
    struct inode *inode;

    if (request->owner_type !=
        VECFS_ANN_OWNER_FILE)
        return NULL;

    if (!request->owner_ino ||
        request->owner_ino > ULONG_MAX)
        return ERR_PTR(-EINVAL);

    inode = ilookup(
        sb,
        (unsigned long)request->owner_ino);
    if (!inode)
        return ERR_PTR(-ENOENT);

    if (!S_ISREG(inode->i_mode)) {
        iput(inode);
        return ERR_PTR(-EINVAL);
    }

    if (inode->i_generation !=
        request->owner_generation) {
        iput(inode);
        return ERR_PTR(-ESTALE);
    }

    return inode;
}

/*
 * 尽力回滚文件 ordinal 映射。
 */
static void vecfs_ann_rollback_file_binding(
    struct inode *inode,
    u32 ordinal,
    u32 expected_node_id)
{
    u32 mapped_node;

    if (!inode)
        return;

    if (vecfs_ann_get_node(
            inode,
            ordinal,
            &mapped_node))
        return;

    /*
     * 只清除本次插入建立的映射，不能误删其它并发事务的映射。
     * writer_lock 已串行化 ANN writer，这里主要是防御磁盘损坏。
     */
    if (mapped_node == expected_node_id)
        vecfs_ann_clear_node(
            inode,
            ordinal);
}

/*
 * 插入失败后的对象回滚。
 *
 * HNSW 的邻接表允许暂时保留指向 DELETED node 的边：
 *
 * - DELETED node 仍可作为导航节点；
 * - 不会进入结果；
 * - rebuild 时彻底清理。
 *
 * 因此失败后将 PREPARED/LIVE object 标记为 DELETED 即可，不必同步
 * 遍历并恢复所有已修改的 backlink。
 */
static void vecfs_ann_abort_object(
    struct super_block *sb,
    u32 node_id,
    bool object_published)
{
    int ret;

    if (node_id < VECFS_ANN_FIRST_NODE_ID)
        return;

    ret = vecfs_ann_object_mark_deleted(
        sb,
        node_id);

    if (ret == -EALREADY ||
        ret == -EINVAL ||
        ret == -ENOENT)
        return;

    if (ret)
        pr_warn(
            "vecfs: failed to abort ANN node %u: %d "
            "(published=%u)\n",
            node_id,
            ret,
            object_published);
}

/*
 * 通用单节点插入。
 *
 * 查询侧从不获取 writer_lock；它始终使用 immutable RCU snapshot。
 */
int vecfs_ann_insert(
    struct super_block *sb,
    const struct vecfs_ann_insert_request *request,
    struct vecfs_ann_insert_result *result)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *old;
    struct vecfs_ann_snapshot *next = NULL;

    struct vecfs_ann_build_context build;
    struct vecfs_ann_distance_session distance;

    struct vecfs_ann_object_info object_info;
    struct vecfs_ann_insert_request normalized;

    struct inode *owner_inode = NULL;

    u32 node_id = VECFS_ANN_INVALID_NODE_ID;
    u16 level;

    bool object_prepared = false;
    bool object_published = false;
    bool label_registered = false;
    bool file_bound = false;

    int ret;

    if (!sb || !request || !result) {
        pr_err("vecfs: ann_insert: null params sb=%p req=%p res=%p\n",
               sb, request, result);
        return -EINVAL;
    }

    sbi = VECFS_SB(sb);
    if (!sbi) {
        pr_err("vecfs: ann_insert: null sbi\n");
        return -EINVAL;
    }

    cache = sbi->s_ann_cache;
    if (!cache) {
        pr_err("vecfs: ann_insert: null cache\n");
        return -ENODEV;
    }

    if (sb_rdonly(sb)) {
        pr_err("vecfs: ann_insert: readonly\n");
        return -EROFS;
    }

    if (!request->vector) {
        pr_err("vecfs: ann_insert: null vector\n");
        return -EINVAL;
    }

    if (request->vector_bytes !=
        sbi->s_ann.config.vector_bytes) {
        pr_err("vecfs: ann_insert: bytes=%zu expected=%u\n",
               request->vector_bytes,
               sbi->s_ann.config.vector_bytes);
        return -EINVAL;
    }

    if (request->owner_type !=
            VECFS_ANN_OWNER_STANDALONE &&
        request->owner_type !=
            VECFS_ANN_OWNER_FILE &&
        request->owner_type !=
            VECFS_ANN_OWNER_IMPORTED) {
        pr_err("vecfs: ann_insert: bad owner_type=%u\n",
               request->owner_type);
        return -EINVAL;
    }

    switch (request->owner_type) {
    case VECFS_ANN_OWNER_FILE:
        if (!request->owner_ino)
            return -EINVAL;
        break;

    case VECFS_ANN_OWNER_STANDALONE:
    case VECFS_ANN_OWNER_IMPORTED:
        /*
         * 独立/导入节点不能伪造文件绑定。
         */
        if (request->owner_ino ||
            request->owner_generation)
            return -EINVAL;
        break;

    default:
        return -EINVAL;
    }

    if ((request->flags &
         VECFS_ANN_OBJ_F_HAS_CONTENT_RANGE) &&
        request->content_offset >
            U64_MAX - request->content_length)
        return -EOVERFLOW;

    memset(result, 0, sizeof(*result));
    memset(&build, 0, sizeof(build));
    memset(&distance, 0, sizeof(distance));
    memset(&object_info, 0, sizeof(object_info));

    mutex_lock(&cache->writer_lock);

    old = rcu_dereference_protected(
        cache->snapshot,
        lockdep_is_held(&cache->writer_lock));

    if (!old) {
        ret = -ENODEV;
        goto out_unlock;
    }

    /*
     * 显式 label 在创建对象前检查。
     *
     * DELETED node 的 label 仍保留在哈希表中，因此不能直接插入
     * 同名 label；应使用 unmark/update，与 hnswlib 一致。
     */
    ret = vecfs_ann_check_explicit_label_locked(
        sb,
        request->external_id);
    if (ret)
        goto out_unlock;

    /*
     * 文件 owner 在创建对象前完成基本校验。
     */
    owner_inode = vecfs_ann_get_owner_inode(
        sb,
        request);
    if (IS_ERR(owner_inode)) {
        ret = PTR_ERR(owner_inode);
        owner_inode = NULL;
        goto out_unlock;
    }

    if (owner_inode) {
        u32 existing_node;

        ret = vecfs_ann_get_node(
            owner_inode,
            request->vector_ordinal,
            &existing_node);

        if (!ret) {
            ret = -EEXIST;
            goto out_owner;
        }

        if (ret != -ENOENT)
            goto out_owner;
    }

    level = vecfs_ann_random_level(sbi);

    /*
     * 先创建 PREPARED object，以获得稳定的 node ID 和最终
     * external ID。
     */
    ret = vecfs_ann_prepare_store_object(
        sb,
        request,
        level,
        &node_id);
    if (ret)
        goto out_owner;

    object_prepared = true;

    ret = vecfs_ann_object_read_info(
        sb,
        node_id,
        &object_info);
    if (ret)
        goto out_abort;

    /*
     * AUTO external ID 由 object store 分配。即使是 AUTO，也需要
     * 经过 label hash 注册，以防恢复后 next_external_id 异常导致
     * 冲突。
     */
    ret = vecfs_ann_label_insert_locked(
        sb,
        object_info.external_id,
        node_id);
    if (ret)
        goto out_abort;

    label_registered = true;

    /*
     * writer 修改 clone，查询继续无锁读取 old。
     */
    next = vecfs_ann_snapshot_clone_locked(
        old,
        node_id + 1);
    if (IS_ERR(next)) {
        ret = PTR_ERR(next);
        next = NULL;
        goto out_abort;
    }

    normalized = *request;
    normalized.external_id =
        object_info.external_id;

    ret = vecfs_ann_snapshot_init_node(
        next,
        node_id,
        level,
        request->vector,
        &normalized);
    if (ret)
        goto out_abort;

    /*
     * 在进入 FPU 之前完成所有 heap、visited、bitmap 和临时数组分配。
     */
    ret = vecfs_ann_build_context_init(
        sb,
        &build,
        next->capacity);
    if (ret)
        goto out_abort;

    vecfs_ann_distance_begin(
        &distance,
        &sbi->s_ann_distance);

    if (!distance.active) {
        ret = -ENODEV;
        goto out_build;
    }

    /*
     * 整个 HNSW 构图过程只进行一次 FPU 上下文切换。
     */
    if (next->live_count == 1 ||
        next->entry_node ==
            VECFS_ANN_INVALID_NODE_ID) {
        next->entry_node = node_id;
        next->max_level = level;

        __set_bit(
            node_id,
            build.dirty_nodes);

        ret = 0;
    } else {
        ret = vecfs_ann_build_insert_node(
            next,
            &distance,
            &build,
            node_id,
            level,
            sbi->s_ann.config.ef_construction,
            NULL, NULL);
    }

    vecfs_ann_distance_end(&distance);

    if (ret)
        goto out_build;

    /*
     * 离开 FPU 区域后再执行持久化 I/O。
     */
    ret = vecfs_ann_persist_dirty_links(
        sb,
        next,
        build.dirty_nodes,
        build.dirty_capacity);
    if (ret)
        goto out_build;

    /*
     * FILE object 的提交顺序：
     *
     * PREPARED object
     * -> 图邻接持久化
     * -> inode ordinal 映射
     * -> object LIVE
     * -> ANN metadata commit
     * -> RCU snapshot publish
     */
    if (owner_inode) {
        ret = vecfs_ann_bind_file_node(
            owner_inode,
            request->vector_ordinal,
            node_id,
            false);
        if (ret)
            goto out_build;

        file_bound = true;
    }

    ret = vecfs_ann_object_publish(
        sb,
        node_id,
        true);
    if (ret)
        goto out_unbind;

    object_published = true;

    /*
     * 更新 ANN 根元数据。
     */
    mutex_lock(&sbi->s_ann.lock);

    sbi->s_ann.entry_node_id =
        next->entry_node;

    sbi->s_ann.config.current_max_level =
        next->max_level;

    sbi->s_ann.state =
        VECFS_ANN_META_ACTIVE;

    sbi->s_ann.dirty = true;

    mutex_unlock(&sbi->s_ann.lock);

    ret = vecfs_ann_meta_commit(
        sb,
        true);
    if (ret)
        goto out_unbind;

    /*
     * 发布不可变新图。
     *
     * vecfs_ann_cache_publish_locked()：
     *
     * 1. rcu_assign_pointer(new);
     * 2. kref_put(old)（call_rcu 延迟回收）;
     *
     * 查询不会获取 writer_lock。
     */
    ret = vecfs_ann_cache_publish_locked(
        cache,
        next);
    if (ret)
        goto out_unbind;

    next = NULL;

    result->node_id = node_id;
    result->level = level;
    result->external_id =
        object_info.external_id;

    ret = 0;
    goto out_build;

out_unbind:
    if (file_bound) {
        vecfs_ann_rollback_file_binding(
            owner_inode,
            request->vector_ordinal,
            node_id);
        file_bound = false;
    }

out_build:
    if (distance.active)
        vecfs_ann_distance_end(
            &distance);

    if (build.search) {
        vecfs_ann_search_context_put(
            sb,
            build.search);
        build.search = NULL;
    }

    vecfs_ann_build_context_destroy(
        &build);

    if (!ret)
        goto out_owner;

out_abort:
    vecfs_ann_snapshot_free(next);
    next = NULL;

    if (file_bound) {
        vecfs_ann_rollback_file_binding(
            owner_inode,
            request->vector_ordinal,
            node_id);
        file_bound = false;
    }

    if (object_prepared) {
        vecfs_ann_abort_object(
            sb,
            node_id,
            object_published);
    }

    /*
     * 本次 insert 没有成功发布，因此移除运行期 label 注册。
     *
     * 失败对象已成为 tombstone，后续 rebuild 会彻底回收。
     */
    if (label_registered) {
        int label_ret;

        label_ret =
            vecfs_ann_label_remove_locked(
                sb,
                object_info.external_id,
                node_id);

        if (label_ret &&
            label_ret != -ENOENT)
            pr_warn(
                "vecfs: rollback label %llu "
                "for node %u failed: %d\n",
                object_info.external_id,
                node_id,
                label_ret);

        label_registered = false;
    }

    /*
     * object 状态变化会修改 live/deleted 计数，尽力提交修复后的
     * metadata；保留原始插入错误作为返回值。
     */
    if (object_prepared &&
        !sb_rdonly(sb)) {
        int meta_ret;

        meta_ret = vecfs_ann_meta_commit(
            sb,
            true);

        if (meta_ret)
            pr_warn(
                "vecfs: rollback ANN metadata "
                "commit failed: %d\n",
                meta_ret);
    }

out_owner:
    if (owner_inode) {
        iput(owner_inode);
        owner_inode = NULL;
    }

out_unlock:
    mutex_unlock(&cache->writer_lock);
    return ret;
}

/*
 * 文件绑定向量插入包装（公共主体）。
 *
 * 该接口只能创建 FILE owner 节点。start_invalid 控制新节点在发布进
 * 查询快照时是否已标记为 invalid（valid_bitmap = 0），供 SHIELD 原子
 * 条件发布使用。
 */
static int vecfs_ann_insert_file_vector_common(
    struct inode *inode,
    u32 ordinal,
    const void *vector,
    size_t vector_bytes,
    u64 external_id,
    u64 content_offset,
    u64 content_length,
    u32 flags,
    bool start_invalid,
    struct vecfs_ann_insert_result *result)
{
    struct vecfs_ann_insert_request request;
    u32 existing_node;
    int ret;

    if (!inode ||
        !S_ISREG(inode->i_mode) ||
        !vector ||
        !result)
        return -EINVAL;

    if (inode->i_ino ==
            VECFS_ANN_ARENA_INO ||
        inode->i_ino ==
            VECFS_ANN_REBUILD_ARENA_INO)
        return -EPERM;

    /*
     * 快速检查。真正的防重检查仍在 writer_lock 内执行。
     */
    ret = vecfs_ann_get_node(
        inode,
        ordinal,
        &existing_node);

    if (!ret)
        return -EEXIST;

    if (ret != -ENOENT)
        return ret;

    memset(&request, 0, sizeof(request));

    request.vector = vector;
    request.vector_bytes = vector_bytes;

    request.external_id = external_id;

    request.owner_ino = inode->i_ino;
    request.owner_generation =
        inode->i_generation;

    request.vector_ordinal = ordinal;

    request.content_offset = content_offset;
    request.content_length = content_length;

    request.owner_type =
        VECFS_ANN_OWNER_FILE;

    request.flags = flags;

    request.start_invalid = start_invalid;

    return vecfs_ann_insert(
        inode->i_sb,
        &request,
        result);
}

int vecfs_ann_insert_file_vector(
    struct inode *inode,
    u32 ordinal,
    const void *vector,
    size_t vector_bytes,
    u64 external_id,
    u64 content_offset,
    u64 content_length,
    u32 flags,
    struct vecfs_ann_insert_result *result)
{
    return vecfs_ann_insert_file_vector_common(
        inode,
        ordinal,
        vector,
        vector_bytes,
        external_id,
        content_offset,
        content_length,
        flags,
        false,
        result);
}

int vecfs_ann_insert_file_vector_invalid(
    struct inode *inode,
    u32 ordinal,
    const void *vector,
    size_t vector_bytes,
    u64 external_id,
    u64 content_offset,
    u64 content_length,
    u32 flags,
    struct vecfs_ann_insert_result *result)
{
    return vecfs_ann_insert_file_vector_common(
        inode,
        ordinal,
        vector,
        vector_bytes,
        external_id,
        content_offset,
        content_length,
        flags,
        true,
        result);
}
