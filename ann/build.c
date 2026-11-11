// SPDX-License-Identifier: GPL-2.0

#include <linux/bitmap.h>
#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/prefetch.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

#include "../include/vecfs_ann_build.h"
#include "../include/vecfs_ann_cache.h"
#include "../include/vecfs_ann_distance.h"

#define VECFS_ANN_BUILD_FLOAT_MAX 3.402823466e+38F

static __always_inline const u32 *
vecfs_ann_build_links(
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id,
    u16 level,
    u16 *count)
{
    if (!level) {
        *count = vecfs_ann_level0_count(
            snapshot, node_id);

        return vecfs_ann_level0_links(
            snapshot, node_id);
    }

    *count = vecfs_ann_upper_count(
        snapshot, node_id, level);

    return vecfs_ann_upper_links(
        snapshot, node_id, level);
}

static __always_inline u32 *
vecfs_ann_build_links_mutable(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id,
    u16 level)
{
    if (!level)
        return vecfs_ann_level0_links_mutable(
            snapshot, node_id);

    return vecfs_ann_upper_links_mutable(
        snapshot, node_id, level);
}

static __always_inline void
vecfs_ann_build_set_count(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id,
    u16 level,
    u16 count)
{
    if (!level)
        vecfs_ann_level0_set_count(
            snapshot, node_id, count);
    else
        vecfs_ann_upper_set_count(
            snapshot, node_id, level, count);
}

static __always_inline u16
vecfs_ann_build_max_neighbors(
    const struct vecfs_ann_snapshot *snapshot,
    u16 level)
{
    return level ? snapshot->m : snapshot->m0;
}

/*
 * 每节点链锁：locks 为 NULL 表示调用方已外部串行化（writer_lock），无需加锁。
 * 锁粒度是「某个节点自己的邻接表」（level0 + upper），node_id 散列到条纹。
 */
static __always_inline void
vecfs_ann_build_lock_node(
    struct vecfs_ann_link_locks *locks,
    u32 node_id)
{
    if (locks)
        spin_lock(&locks->locks[node_id & locks->mask]);
}

static __always_inline void
vecfs_ann_build_unlock_node(
    struct vecfs_ann_link_locks *locks,
    u32 node_id)
{
    if (locks)
        spin_unlock(&locks->locks[node_id & locks->mask]);
}

static int vecfs_ann_build_search_layer(
    const struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_distance_session *distance,
    struct vecfs_ann_build_context *build,
    const float *query,
    u32 entry_node,
    u32 new_node,
    u16 level,
    u32 ef,
    struct vecfs_ann_heap_item **result_items,
    u32 *result_count)
{
    struct vecfs_ann_search_context *context =
        build->search;
    struct vecfs_ann_visited *visited =
        &context->visited;
    float lower_bound;
    float entry_distance;
    int ret;

    vecfs_ann_visited_reset(visited);
    vecfs_ann_heap_reset(&context->candidates);
    vecfs_ann_heap_reset(&context->results);

    entry_distance =
        vecfs_ann_distance_calculate(
            distance,
            query,
            vecfs_ann_snapshot_vector(
                snapshot, entry_node));

    lower_bound = entry_distance;

    visited->generation[entry_node] =
        visited->cur_gen;

    ret = vecfs_ann_heap_push(
        &context->candidates,
        entry_node,
        entry_distance);
    if (ret)
        return ret;

    ret = vecfs_ann_heap_push(
        &context->results,
        entry_node,
        entry_distance);
    if (ret)
        return ret;

    while (context->candidates.size) {
        struct vecfs_ann_heap_item cur;
        const struct vecfs_ann_heap_item *nearest;
        const u32 *links;
        u16 count;
        u16 i;

        nearest = vecfs_ann_heap_top(
            &context->candidates);

        if (nearest->distance > lower_bound &&
            context->results.size >= ef)
            break;

        vecfs_ann_heap_pop(
            &context->candidates,
            &cur);

        links = vecfs_ann_build_links(
            snapshot,
            cur.node_id,
            level,
            &count);

        if (!links)
            continue;

        if (count)
            prefetch(
                vecfs_ann_snapshot_vector(
                    snapshot, links[0]));

        for (i = 0; i < count; i++) {
            u32 candidate = links[i];
            if (candidate == new_node)
                continue;

            float candidate_distance;

            if (i + 1 < count)
                prefetch(
                    vecfs_ann_snapshot_vector(
                        snapshot,
                        links[i + 1]));

            if (visited->generation[candidate] ==
                visited->cur_gen)
                continue;

            visited->generation[candidate] =
                visited->cur_gen;

            candidate_distance =
                vecfs_ann_distance_calculate(
                    distance,
                    query,
                    vecfs_ann_snapshot_vector(
                        snapshot, candidate));

            if (context->results.size < ef ||
                candidate_distance < lower_bound) {
                ret = vecfs_ann_heap_push(
                    &context->candidates,
                    candidate,
                    candidate_distance);
                if (ret)
                    return ret;

                if (context->results.size < ef) {
                    ret = vecfs_ann_heap_push(
                        &context->results,
                        candidate,
                        candidate_distance);
                } else {
                    const struct vecfs_ann_heap_item *worst =
                        vecfs_ann_heap_top(
                            &context->results);

                    if (candidate_distance <
                        worst->distance) {
                        ret =
                            vecfs_ann_heap_replace_top(
                                &context->results,
                                candidate,
                                candidate_distance);
                    } else {
                        ret = 0;
                    }
                }

                if (ret)
                    return ret;

                lower_bound =
                    vecfs_ann_heap_top(
                        &context->results)->distance;
            }
        }
    }

    ret = vecfs_ann_heap_copy_sorted(
        &context->results,
        context->sorted_results,
        context->sorted_capacity,
        result_count);
    if (ret)
        return ret;

    *result_items = context->sorted_results;
    return 0;
}

static int vecfs_ann_build_trim_neighbor_list(
    struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_distance_session *distance,
    struct vecfs_ann_build_context *build,
    u32 owner_node,
    u16 level,
    const u32 *candidate_ids,
    u32 candidate_count,
    u16 maximum)
{
    struct vecfs_ann_heap_item *items =
        build->ordered;
    const float *owner_vector;
    u32 selected_count;
    u32 *links;
    u32 i;
    int ret;

    if (candidate_count >
        build->temporary_capacity)
        return -ENOSPC;

    owner_vector =
        vecfs_ann_snapshot_vector(
            snapshot, owner_node);

    for (i = 0; i < candidate_count; i++) {
        items[i].node_id = candidate_ids[i];

        items[i].distance =
            vecfs_ann_distance_calculate(
                distance,
                owner_vector,
                vecfs_ann_snapshot_vector(
                    snapshot,
                    candidate_ids[i]));
    }

    ret = vecfs_ann_select_neighbors_heuristic(
        snapshot,
        distance,
        build,
        items,
        candidate_count,
        maximum,
        &selected_count);
    if (ret)
        return ret;

    links = vecfs_ann_build_links_mutable(
        snapshot, owner_node, level);
    if (!links)
        return -EUCLEAN;

    for (i = 0; i < selected_count; i++)
        links[i] =
            build->selected[i].node_id;

    vecfs_ann_build_set_count(
        snapshot,
        owner_node,
        level,
        selected_count);

    __set_bit(owner_node, build->dirty_nodes);
    return 0;
}

static int vecfs_ann_build_add_backlink(
    struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_distance_session *distance,
    struct vecfs_ann_build_context *build,
    u32 neighbor,
    u32 new_node,
    u16 level,
    struct vecfs_ann_link_locks *locks)
{
    u16 maximum;
    u32 *links;
    u16 count;
    u16 i;
    int ret;

    if (!vecfs_ann_snapshot_id_valid(
            snapshot, neighbor) ||
        !vecfs_ann_snapshot_id_valid(
            snapshot, new_node))
        return -EUCLEAN;

    if (neighbor == new_node) {
        pr_err(
            "vecfs: ANN self backlink "
            "node=%u level=%u\n",
            new_node,
            level);
        return -EUCLEAN;
    }

    if (level > vecfs_ann_node_at(snapshot, neighbor)->level)
        return 0;

    maximum = vecfs_ann_build_max_neighbors(
        snapshot, level);

    links = vecfs_ann_build_links_mutable(
        snapshot, neighbor, level);
    if (!links)
        return -EUCLEAN;

    /*
     * 邻接表 read-modify-write 全程持 neighbor 的链锁。并行构图下，
     * 另一个 worker 可能同时给同一 neighbor 加 backlink；单条/批量路径
     * （locks == NULL）由 writer_lock 串行化，等价于无锁。
     */
    vecfs_ann_build_lock_node(locks, neighbor);

    count = level ?
        vecfs_ann_upper_count(
            snapshot, neighbor, level) :
        vecfs_ann_level0_count(
            snapshot, neighbor);

    for (i = 0; i < count; i++) {
        if (links[i] == new_node) {
            vecfs_ann_build_unlock_node(
                locks, neighbor);
            return 0;
        }
    }

    if (count < maximum) {
        links[count] = new_node;

        vecfs_ann_build_set_count(
            snapshot,
            neighbor,
            level,
            count + 1);

        __set_bit(neighbor, build->dirty_nodes);

        vecfs_ann_build_unlock_node(
            locks, neighbor);
        return 0;
    }

    /*
     * 与 hnswlib mutuallyConnectNewElement 对齐：
     *
     * 当前邻居已满时，把 new node 与原邻居共同交给 heuristic。
     * trim_neighbor_list 假定调用方已持 owner(=neighbor) 的链锁，不再加锁。
     */
    if ((u32)count + 1 >
        build->temporary_capacity) {
        vecfs_ann_build_unlock_node(
            locks, neighbor);
        return -ENOSPC;
    }

    for (i = 0; i < count; i++)
        build->neighbor_ids[i] = links[i];

    build->neighbor_ids[count] = new_node;

    ret = vecfs_ann_build_trim_neighbor_list(
        snapshot,
        distance,
        build,
        neighbor,
        level,
        build->neighbor_ids,
        count + 1,
        maximum);

    vecfs_ann_build_unlock_node(
        locks, neighbor);

    return ret;
}

static int vecfs_ann_build_connect_new_node(
    struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_distance_session *distance,
    struct vecfs_ann_build_context *build,
    u32 new_node,
    u16 level,
    const struct vecfs_ann_heap_item *candidates,
    u32 candidate_count,
    u32 *next_entry,
    struct vecfs_ann_link_locks *locks)
{
    u16 maximum =
        vecfs_ann_build_max_neighbors(
            snapshot, level);
    u32 selected_count;
    u32 *new_links;
    u32 i;
    int ret;

    ret = vecfs_ann_select_neighbors_heuristic(
        snapshot,
        distance,
        build,
        candidates,
        candidate_count,
        maximum,
        &selected_count);
    if (ret)
        return ret;

    /*
    * add_backlink() 在邻接表已满时会再次调用 heuristic，
    * 从而覆盖 build->selected[]。先保存外层选择结果。
    */
    if (selected_count > build->set_capacity)
        return -EOVERFLOW;

    for (i = 0; i < selected_count; i++)
        build->selected_neighbors[i] =
            build->selected[i].node_id;

    new_links = vecfs_ann_build_links_mutable(
        snapshot,
        new_node,
        level);
    if (!new_links && selected_count)
        return -EUCLEAN;

    /*
     * 并行构图下，new_node 自己的邻接表也可能被别的 worker 当作 backlink
     * 目标写入（见 add_backlink），因此同样要持 new_node 的链锁写自己的 links。
     */
    vecfs_ann_build_lock_node(locks, new_node);

    for (i = 0; i < selected_count; i++)
        new_links[i] =
            build->selected_neighbors[i];

    vecfs_ann_build_set_count(
        snapshot,
        new_node,
        level,
        selected_count);

    vecfs_ann_build_unlock_node(locks, new_node);

    __set_bit(new_node, build->dirty_nodes);

    for (i = 0; i < selected_count; i++) {
        u32 neighbor =
            build->selected_neighbors[i];

        /*
        * 防止将嵌套临时数组覆盖转化为 self-loop。
        * 正常算法不会触发该条件。
        */
        if (WARN_ON_ONCE(neighbor == new_node))
            return -EUCLEAN;

        ret = vecfs_ann_build_add_backlink(
            snapshot,
            distance,
            build,
            neighbor,
            new_node,
            level,
            locks);
        if (ret)
            return ret;
    }

    if (selected_count)
        *next_entry =
            build->selected_neighbors[0];

    return 0;
}

int vecfs_ann_build_insert_node(
    struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_distance_session *distance,
    struct vecfs_ann_build_context *build,
    u32 new_node,
    u16 new_level,
    u32 ef_construction,
    struct vecfs_ann_link_locks *locks,
    spinlock_t *entry_lock)
{
    const float *query;
    u32 cur;
    u16 build_max_level;
    float current_distance;
    int level;
    int ret;

    if (!snapshot || !distance || !build)
    return -EINVAL;

    if (!vecfs_ann_snapshot_id_valid(
            snapshot, new_node))
        return -ENOENT;

    /*
    * 只有图中第一个节点允许直接成为 entry。
    * 已有图中不能从目标节点自身开始重新插入。
    */
    if (snapshot->live_count > 1 &&
        READ_ONCE(snapshot->entry_node) == new_node) {
        pr_err(
            "vecfs: cannot reinsert ANN entry node "
            "%u using itself as entry\n",
            new_node);
        return -EUCLEAN;
    }

    if (snapshot->live_count == 1 ||
        READ_ONCE(snapshot->entry_node) ==
            VECFS_ANN_INVALID_NODE_ID) {
        if (entry_lock)
            spin_lock(entry_lock);

        snapshot->entry_node = new_node;
        snapshot->max_level = new_level;

        if (entry_lock)
            spin_unlock(entry_lock);

        return 0;
    }

    query = vecfs_ann_snapshot_vector(
        snapshot, new_node);

    /*
     * entry / max_level 在并行构图下会被其他 worker 通过 entry_lock 更新；
     * 这里各取一次快照（READ_ONCE）用于本节点的贪心下降。entry 只会指向
     * 已完整连接的更高层节点、max_level 单调不减，读到旧值仅使本节点下降
     * 起点略保守，不影响最终图正确性。
     */
    cur = READ_ONCE(snapshot->entry_node);
    build_max_level = READ_ONCE(snapshot->max_level);

    current_distance =
        vecfs_ann_distance_calculate(
            distance,
            query,
            vecfs_ann_snapshot_vector(
                snapshot, cur));

    /*
     * 新节点层数以下才需要建立连接。
     * 高于新节点层数的层只执行 greedy descent。
     */
    for (level = build_max_level;
         level > new_level;
         level--) {
        bool changed = true;

        while (changed) {
            const u32 *links;
            u16 count;
            u16 i;

            changed = false;

            links = vecfs_ann_upper_links(
                snapshot, cur, level);
            count = vecfs_ann_upper_count(
                snapshot, cur, level);

            if (!links)
                break;

            for (i = 0; i < count; i++) {
                u32 candidate = links[i];
                float candidate_distance;

                candidate_distance =
                    vecfs_ann_distance_calculate(
                        distance,
                        query,
                        vecfs_ann_snapshot_vector(
                            snapshot, candidate));

                if (candidate_distance <
                    current_distance) {
                    cur = candidate;
                    current_distance =
                        candidate_distance;
                    changed = true;
                }
            }
        }
    }

    for (level = min_t(
             int,
             new_level,
             build_max_level);
         level >= 0;
         level--) {
        struct vecfs_ann_heap_item *candidates;
        u32 candidate_count;
        u32 next_entry = cur;

        ret = vecfs_ann_build_search_layer(
            snapshot,
            distance,
            build,
            query,
            cur,
            new_node,
            level,
            ef_construction,
            &candidates,
            &candidate_count);
        if (ret)
            return ret;

        ret = vecfs_ann_build_connect_new_node(
            snapshot,
            distance,
            build,
            new_node,
            level,
            candidates,
            candidate_count,
            &next_entry,
            locks);
        if (ret)
            return ret;

        cur = next_entry;
    }

    /*
     * 更高层节点替换 entry：并行构图下由 entry_lock 互斥，且须在锁内复核
     * （两个 worker 各自插入更高层节点时只有真正更高者成为 entry）。
     */
    if (entry_lock) {
        spin_lock(entry_lock);

        if (new_level > snapshot->max_level) {
            snapshot->entry_node = new_node;
            snapshot->max_level = new_level;
        }

        spin_unlock(entry_lock);
    } else if (new_level > snapshot->max_level) {
        snapshot->entry_node = new_node;
        snapshot->max_level = new_level;
    }

    return 0;
}
