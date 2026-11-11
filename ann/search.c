// SPDX-License-Identifier: GPL-2.0

#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/prefetch.h>
#include <linux/rcupdate.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_cache.h"
#include "../include/vecfs_ann_distance.h"
#include "../include/vecfs_ann_heap.h"
#include "../include/vecfs_ann_search.h"
#include "../include/vecfs_ann_runtime.h"

#define VECFS_ANN_FLOAT_MAX 3.402823466e+38F

enum vecfs_ann_result_filter_mode {
    VECFS_ANN_FILTER_ALL = 0,
    VECFS_ANN_FILTER_FILE,
    VECFS_ANN_FILTER_STANDALONE,
};

struct vecfs_ann_result_filter {
    enum vecfs_ann_result_filter_mode mode;

    bool include_stale;
};

static __always_inline void
vecfs_ann_prefetch_neighbor(
    const struct vecfs_ann_snapshot *snapshot,
    struct vecfs_ann_visited *visited,
    u32 node_id);

static __always_inline void
vecfs_ann_prefetch_level0(
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id);

static __always_inline int
vecfs_ann_candidate_push(
    struct vecfs_ann_search_context *context,
    u32 node_id,
    float distance);

static __always_inline void
vecfs_ann_update_lower_bound(
    const struct vecfs_ann_search_context *context,
    float *lower_bound);

static __always_inline bool
vecfs_ann_result_allowed(
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id,
    const struct vecfs_ann_result_filter *filter)
{
    const struct vecfs_ann_cache_node *node =
        vecfs_ann_node_at(snapshot, node_id);

    /*
     * DELETED 节点参与导航，但永远不能进入结果。
     */
    if (node->state !=
        VECFS_ANN_OBJ_LIVE)
        return false;

    switch (filter->mode) {
    case VECFS_ANN_FILTER_FILE:
        if (node->owner_type !=
            VECFS_ANN_OWNER_FILE)
            return false;
        break;

    case VECFS_ANN_FILTER_STANDALONE:
        /*
         * IMPORTED 节点通过 standalone API 管理，因此归入非 FILE
         * 结果集合。
         */
        if (node->owner_type ==
            VECFS_ANN_OWNER_FILE)
            return false;
        break;

    case VECFS_ANN_FILTER_ALL:
    default:
        break;
    }

    if (!filter->include_stale &&
        vecfs_ann_snapshot_content_stale(
            snapshot,
            node_id))
        return false;

    /*
     * SHIELD 查询端 Valid(x,f)：过滤掉不被任何当前语义根引用的节点。
     * 与 stale 位并列，成本相同（一次位测试）。
     */
    if (!filter->include_stale &&
        vecfs_ann_snapshot_content_invalid(
            snapshot,
            node_id))
        return false;

    return true;
}

/*
 * 对应 hnswlib searchBaseLayerST<false>() + BaseFilterFunctor。
 *
 * 不满足 filter 的节点：
 *
 * - 仍然进入 candidates；
 * - 仍然展开邻居；
 * - 不进入 results。
 *
 * 因此过滤不会破坏图导航连通性。
 */
static int vecfs_ann_search_base_filtered(
    const struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_distance_session *distance,
    struct vecfs_ann_search_context *context,
    const float *query,
    u32 entry_node,
    float entry_distance,
    u32 ef,
    const struct vecfs_ann_result_filter *filter)
{
    struct vecfs_ann_visited *visited =
        &context->visited;

    float lower_bound =
        VECFS_ANN_FLOAT_MAX;

    int ret;

    vecfs_ann_heap_reset(
        &context->candidates);

    vecfs_ann_heap_reset(
        &context->results);

    vecfs_ann_visited_reset(
        visited);

    visited->generation[entry_node] =
        visited->cur_gen;


    ret = vecfs_ann_heap_push(
        &context->candidates,
        entry_node,
        entry_distance);
    if (unlikely(ret))
        return ret;

    if (vecfs_ann_result_allowed(
            snapshot,
            entry_node,
            filter)) {
        ret = vecfs_ann_heap_push(
            &context->results,
            entry_node,
            entry_distance);
        if (unlikely(ret))
            return ret;

        lower_bound =
            entry_distance;
    }

    while (likely(
        context->candidates.size)) {
        struct vecfs_ann_heap_item cur;

        const struct vecfs_ann_heap_item *nearest;

        const u32 *links;
        u16 count;
        u16 i;

        nearest = vecfs_ann_heap_top(
            &context->candidates);

        /*
         * 与 hnswlib 非 bare-bone 路径一致：只有 result 数量已经达到
         * ef 时，candidate 超过 lowerBound 才能终止。
         */
        if (nearest->distance >
                lower_bound &&
            context->results.size ==
                ef)
            break;

        ret = vecfs_ann_heap_pop(
            &context->candidates,
            &cur);
        if (unlikely(ret))
            return ret;

        links = vecfs_ann_level0_links(
            snapshot,
            cur.node_id);

        count = vecfs_ann_level0_count(
            snapshot,
            cur.node_id);

        if (likely(count)) {
            u32 first = links[0];

            prefetch(
                &visited->generation[first]);

            prefetch(
                vecfs_ann_snapshot_vector(
                    snapshot,
                    first));

            if (count > 1)
                prefetch(&links[1]);
        }

        for (i = 0; i < count; i++) {
            u32 candidate = links[i];
            float candidate_distance;
            bool consider;

            if (i + 1 < count) {
                u32 next = links[i + 1];

                vecfs_ann_prefetch_neighbor(
                    snapshot,
                    visited,
                    next);
            }

            if (likely(
                    visited->generation[candidate] ==
                    visited->cur_gen))
                continue;

            visited->generation[candidate] =
                visited->cur_gen;

            candidate_distance =
                vecfs_ann_distance_calculate(
                    distance,
                    query,
                    vecfs_ann_snapshot_vector(
                        snapshot,
                        candidate));

            consider =
                context->results.size < ef ||
                lower_bound >
                    candidate_distance;

            if (!consider)
                continue;

            /*
             * 即使 candidate 不满足 filter，也必须加入 candidate set
             * 继续导航。
             */
            ret = vecfs_ann_candidate_push(
                context,
                candidate,
                candidate_distance);
            if (unlikely(ret))
                return ret;

            vecfs_ann_prefetch_level0(
                snapshot,
                vecfs_ann_heap_top(
                    &context->candidates)->
                        node_id);

            if (!vecfs_ann_result_allowed(
                    snapshot,
                    candidate,
                    filter))
                continue;

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

            if (unlikely(ret))
                return ret;

            vecfs_ann_update_lower_bound(
                context,
                &lower_bound);
        }
    }

    return 0;
}


static __always_inline void
vecfs_ann_prefetch_neighbor(
    const struct vecfs_ann_snapshot *snapshot,
    struct vecfs_ann_visited *visited,
    u32 node_id)
{
    prefetch(&visited->generation[node_id]);
    prefetch(vecfs_ann_snapshot_vector(snapshot, node_id));
}

static __always_inline void
vecfs_ann_prefetch_level0(
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    prefetch(vecfs_ann_level0_element(snapshot, node_id));
    prefetch(
        (const u8 *)vecfs_ann_level0_element(
            snapshot, node_id) + 64);
}

static int vecfs_ann_greedy_upper(
    const struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_distance_session *distance,
    const float *query,
    u32 *current_node,
    float *current_distance)
{
    u32 cur = *current_node;
    float current_dist = *current_distance;
    int level;

    for (level = snapshot->max_level;
         level > 0;
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

            if (unlikely(!links))
                break;

            if (count)
                prefetch(
                    vecfs_ann_snapshot_vector(
                        snapshot, links[0]));

            for (i = 0; i < count; i++) {
                u32 candidate = links[i];
                float candidate_distance;

                if (i + 1 < count)
                    prefetch(
                        vecfs_ann_snapshot_vector(
                            snapshot, links[i + 1]));

                candidate_distance =
                    vecfs_ann_distance_calculate(
                        distance,
                        query,
                        vecfs_ann_snapshot_vector(
                            snapshot, candidate));

                /*
                 * 与 hnswlib 一致，只接受严格更近，不使用 node ID
                 * 作为 upper greedy 的距离相同时 tie-break。
                 */
                if (candidate_distance < current_dist) {
                    current_dist = candidate_distance;
                    cur = candidate;
                    changed = true;
                }
            }
        }
    }

    *current_node = cur;
    *current_distance = current_dist;

    return 0;
}

static __always_inline int
vecfs_ann_candidate_push(
    struct vecfs_ann_search_context *context,
    u32 node_id,
    float distance)
{
    /*
     * candidates 容量等于 node capacity，每个 node 最多进入一次。
     */
    return vecfs_ann_heap_push(
        &context->candidates,
        node_id,
        distance);
}

static __always_inline void
vecfs_ann_update_lower_bound(
    const struct vecfs_ann_search_context *context,
    float *lower_bound)
{
    const struct vecfs_ann_heap_item *top =
        vecfs_ann_heap_top(&context->results);

    *lower_bound = top ?
        top->distance :
        VECFS_ANN_FLOAT_MAX;
}

/*
 * 等价于 hnswlib searchBaseLayerST<true>()。
 *
 * 适用条件：
 * - snapshot 中没有 deleted node；
 * - 没有 filter；
 * - 没有 stop condition。
 *
 * hot loop 中没有 state/filter 分支。
 */
static int vecfs_ann_search_base_bare(
    const struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_distance_session *distance,
    struct vecfs_ann_search_context *context,
    const float *query,
    u32 entry_node,
    float entry_distance,
    u32 ef)
{
    struct vecfs_ann_visited *visited =
        &context->visited;
    float lower_bound = entry_distance;
    int ret;

    vecfs_ann_heap_reset(&context->candidates);
    vecfs_ann_heap_reset(&context->results);
    vecfs_ann_visited_reset(visited);

    visited->generation[entry_node] =
        visited->cur_gen;


    ret = vecfs_ann_heap_push(
        &context->results,
        entry_node,
        entry_distance);
    if (unlikely(ret))
        return ret;

    ret = vecfs_ann_candidate_push(
        context,
        entry_node,
        entry_distance);
    if (unlikely(ret))
        return ret;

    while (likely(context->candidates.size)) {
        struct vecfs_ann_heap_item cur;
        const struct vecfs_ann_heap_item *nearest;
        const u32 *links;
        u16 count;
        u16 i;

        nearest = vecfs_ann_heap_top(
            &context->candidates);

        /*
         * hnswlib bare_bone_search 的终止条件：
         *
         *     candidate_dist > lowerBound
         *
         * 不额外检查 top_candidates.size() == ef。
         */
        if (nearest->distance > lower_bound)
            break;

        vecfs_ann_heap_pop(
            &context->candidates, &cur);

        links = vecfs_ann_level0_links(
            snapshot, cur.node_id);
        count = vecfs_ann_level0_count(
            snapshot, cur.node_id);

        if (likely(count)) {
            u32 first = links[0];

            prefetch(&visited->generation[first]);
            prefetch(
                &visited->generation[
                    min_t(u32,
                          first + 32,
                          snapshot->capacity - 1)]);
            prefetch(
                vecfs_ann_snapshot_vector(
                    snapshot, first));

            if (count > 1)
                prefetch(&links[1]);
        }

        for (i = 0; i < count; i++) {
            u32 candidate = links[i];
            float candidate_distance;

            if (i + 1 < count) {
                u32 next = links[i + 1];

                vecfs_ann_prefetch_neighbor(
                    snapshot,
                    visited,
                    next);
            }

            if (likely(
                    visited->generation[candidate] ==
                    visited->cur_gen))
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
                lower_bound > candidate_distance) {
                ret = vecfs_ann_candidate_push(
                    context,
                    candidate,
                    candidate_distance);
                if (unlikely(ret))
                    return ret;

                /*
                 * 与 hnswlib 一致：prefetch candidate_set.top()
                 * 的 level-0 link list。
                 */
                vecfs_ann_prefetch_level0(
                    snapshot,
                    vecfs_ann_heap_top(
                        &context->candidates)->node_id);

                ret = vecfs_ann_heap_push(
                    &context->results,
                    candidate,
                    candidate_distance);
                if (unlikely(ret == -ENOSPC)) {
                    /*
                     * results 固定为 ef。等价于先 push 再在 size>ef
                     * 时 pop：只有 candidate 比当前最远更近时替换。
                     */
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

                if (unlikely(ret))
                    return ret;

                vecfs_ann_update_lower_bound(
                    context, &lower_bound);
            }
        }
    }

    return 0;
}

/*
 * 等价于 hnswlib searchBaseLayerST<false>() 的 deleted-only 路径。
 *
 * deleted node：
 * - 可以进入 candidate set 并参与导航；
 * - 不能进入 top results。
 */
static int vecfs_ann_search_base_deleted(
    const struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_distance_session *distance,
    struct vecfs_ann_search_context *context,
    const float *query,
    u32 entry_node,
    float entry_distance,
    u32 ef)
{
    struct vecfs_ann_visited *visited =
        &context->visited;
    float lower_bound;
    int ret;

    vecfs_ann_heap_reset(&context->candidates);
    vecfs_ann_heap_reset(&context->results);
    vecfs_ann_visited_reset(visited);

    visited->generation[entry_node] =
        visited->cur_gen;


    if (vecfs_ann_node_at(snapshot, entry_node)->state ==
        VECFS_ANN_OBJ_LIVE) {
        lower_bound = entry_distance;

        ret = vecfs_ann_heap_push(
            &context->results,
            entry_node,
            entry_distance);
        if (unlikely(ret))
            return ret;
    } else {
        lower_bound = VECFS_ANN_FLOAT_MAX;
    }

    ret = vecfs_ann_candidate_push(
        context,
        entry_node,
        entry_distance);
    if (unlikely(ret))
        return ret;

    while (likely(context->candidates.size)) {
        struct vecfs_ann_heap_item cur;
        const struct vecfs_ann_heap_item *nearest;
        const u32 *links;
        u16 count;
        u16 i;

        nearest = vecfs_ann_heap_top(
            &context->candidates);

        if (nearest->distance > lower_bound &&
            context->results.size == ef)
            break;

        vecfs_ann_heap_pop(
            &context->candidates, &cur);

        links = vecfs_ann_level0_links(
            snapshot, cur.node_id);
        count = vecfs_ann_level0_count(
            snapshot, cur.node_id);

        if (likely(count))
            vecfs_ann_prefetch_neighbor(
                snapshot,
                visited,
                links[0]);

        for (i = 0; i < count; i++) {
            u32 candidate = links[i];
            float candidate_distance;

            if (i + 1 < count)
                vecfs_ann_prefetch_neighbor(
                    snapshot,
                    visited,
                    links[i + 1]);

            if (likely(
                    visited->generation[candidate] ==
                    visited->cur_gen))
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
                lower_bound > candidate_distance) {
                ret = vecfs_ann_candidate_push(
                    context,
                    candidate,
                    candidate_distance);
                if (unlikely(ret))
                    return ret;

                vecfs_ann_prefetch_level0(
                    snapshot,
                    vecfs_ann_heap_top(
                        &context->candidates)->node_id);

                if (vecfs_ann_node_at(snapshot, candidate)->state ==
                    VECFS_ANN_OBJ_LIVE) {
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

                    if (unlikely(ret))
                        return ret;

                    vecfs_ann_update_lower_bound(
                        context, &lower_bound);
                }
            }
        }
    }

    return 0;
}

static int vecfs_ann_copy_results(
    const struct vecfs_ann_snapshot *snapshot,
    struct vecfs_ann_search_context *context,
    struct vecfs_ann_search_result *results,
    u32 k,
    u32 *result_count)
{
    u32 sorted_count;
    u32 count;
    u32 i;
    int ret;

    ret = vecfs_ann_heap_copy_sorted(
        &context->results,
        context->sorted_results,
        context->sorted_capacity,
        &sorted_count);
    if (ret)
        return ret;

    count = min(
        k,
        sorted_count);

    for (i = 0; i < count; i++) {
        const struct vecfs_ann_heap_item *item =
            &context->sorted_results[i];

        const struct vecfs_ann_cache_node *node =
            vecfs_ann_node_at(snapshot, item->node_id);

        memset(
            &results[i],
            0,
            sizeof(results[i]));

        results[i].node_id =
            item->node_id;

        results[i].external_id =
            node->external_id;

        results[i].owner_ino =
            node->owner_ino;

        results[i].owner_generation =
            node->owner_generation;

        results[i].vector_ordinal =
            node->vector_ordinal;

        results[i].content_offset =
            node->content_offset;

        results[i].content_length =
            node->content_length;

        results[i].owner_type =
            node->owner_type;

        results[i].flags =
            vecfs_ann_snapshot_effective_flags(
                snapshot,
                item->node_id);

        results[i].distance =
            item->distance;
    }

    *result_count = count;
    return 0;
}


int vecfs_ann_search(
    struct super_block *sb,
    const struct vecfs_ann_search_request *request,
    struct vecfs_ann_search_result *results,
    u32 result_capacity,
    u32 *result_count)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *snapshot;
    struct vecfs_ann_search_context *context = NULL;
    struct vecfs_ann_distance_session distance;
    struct vecfs_ann_result_filter filter;

    const float *query;

    u32 snapshot_capacity;
    u32 entry_node;
    u32 ef;

    float entry_distance;

    bool filtered;
    int ret;

    if (!sb ||
        !request ||
        !results ||
        !result_count)
        return -EINVAL;

    *result_count = 0;

    if (request->flags &
        ~VECFS_ANN_SEARCH_VALID_FLAGS)
        return -EINVAL;

    if ((request->flags &
         VECFS_ANN_SEARCH_F_FILE_ONLY) &&
        (request->flags &
         VECFS_ANN_SEARCH_F_STANDALONE_ONLY))
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi ||
        !sbi->s_ann_cache)
        return -ENODEV;

    cache = sbi->s_ann_cache;

    if (!request->query_vector)
        return -EINVAL;

    if (!request->k ||
        request->k >
            VECFS_ANN_SEARCH_MAX_K)
        return -EINVAL;

    if (result_capacity <
        request->k)
        return -ENOSPC;

    /*
     * 本次查询唯一一次读取 runtime ef_search。
     */
    ef = vecfs_ann_runtime_capture_ef(
        sb,
        request->ef_search,
        request->k);

    if (!ef ||
        ef > VECFS_ANN_SEARCH_MAX_EF)
        return -EINVAL;

    memset(&filter, 0, sizeof(filter));

    if (request->flags &
        VECFS_ANN_SEARCH_F_FILE_ONLY) {
        filter.mode =
            VECFS_ANN_FILTER_FILE;
    } else if (
        request->flags &
        VECFS_ANN_SEARCH_F_STANDALONE_ONLY) {
        filter.mode =
            VECFS_ANN_FILTER_STANDALONE;
    } else {
        filter.mode =
            VECFS_ANN_FILTER_ALL;
    }

    filter.include_stale =
        request->flags &
        VECFS_ANN_SEARCH_F_INCLUDE_STALE;

retry_context:
    /*
     * 这里只读取容量，不能在 RCU 区间内获取可能睡眠的 context。
     */
    rcu_read_lock();

    snapshot = rcu_dereference(
        cache->snapshot);

    if (unlikely(!snapshot)) {
        rcu_read_unlock();
        return -ENODEV;
    }

    if (request->query_bytes !=
        snapshot->vector_bytes) {
        rcu_read_unlock();
        return -EINVAL;
    }

    snapshot_capacity =
        snapshot->capacity;

    rcu_read_unlock();

    ret = vecfs_ann_search_context_get(
        sb,
        snapshot_capacity,
        ef,
        &context);
    if (ret)
        return ret;

    rcu_read_lock();

    snapshot = rcu_dereference(
        cache->snapshot);

    if (unlikely(!snapshot)) {
        ret = -ENODEV;
        goto out_rcu;
    }

    if (unlikely(
            context->node_capacity <
            snapshot->capacity)) {
        u32 required =
            snapshot->capacity;

        rcu_read_unlock();

        ret =
            vecfs_ann_search_context_prepare(
                context,
                required,
                ef);

        vecfs_ann_search_context_put(
            sb,
            context);

        context = NULL;

        if (ret)
            return ret;

        goto retry_context;
    }

    if (!snapshot->live_count ||
        snapshot->entry_node ==
            VECFS_ANN_INVALID_NODE_ID) {
        ret = 0;
        goto out_rcu;
    }

    query = request->query_vector;
    entry_node = snapshot->entry_node;

    /*
     * 只在查询开始和结束切换一次 FPU。
     */
    memset(&distance, 0, sizeof(distance));

    vecfs_ann_distance_begin(
        &distance,
        &sbi->s_ann_distance);

    if (unlikely(!distance.active)) {
        ret = -ENODEV;
        goto out_rcu;
    }

    entry_distance =
        vecfs_ann_distance_calculate(
            &distance,
            query,
            vecfs_ann_snapshot_vector(
                snapshot,
                entry_node));

    ret = vecfs_ann_greedy_upper(
        snapshot,
        &distance,
        query,
        &entry_node,
        &entry_distance);
    if (unlikely(ret))
        goto out_fpu;

    /*
     * 只有以下条件全部成立才进入 hnswlib bare-bone 等价路径：
     *
     * - 图中无 deleted；
     * - 无 owner filter；
     * - stale 被包含，或者图中不存在 stale。
     */
    filtered =
        filter.mode !=
            VECFS_ANN_FILTER_ALL ||
        (!filter.include_stale &&
         (vecfs_ann_snapshot_has_stale(
              snapshot) ||
          vecfs_ann_snapshot_has_invalid(
              snapshot)));

    if (likely(
            !snapshot->has_deletions &&
            !filtered)) {
        ret = vecfs_ann_search_base_bare(
            snapshot,
            &distance,
            context,
            query,
            entry_node,
            entry_distance,
            ef);
    } else if (
        snapshot->has_deletions &&
        !filtered) {
        ret = vecfs_ann_search_base_deleted(
            snapshot,
            &distance,
            context,
            query,
            entry_node,
            entry_distance,
            ef);
    } else {
        ret = vecfs_ann_search_base_filtered(
            snapshot,
            &distance,
            context,
            query,
            entry_node,
            entry_distance,
            ef,
            &filter);
    }

out_fpu:
    vecfs_ann_distance_end(
        &distance);

    if (!ret) {
        ret = vecfs_ann_copy_results(
            snapshot,
            context,
            results,
            request->k,
            result_count);
    }

out_rcu:
    rcu_read_unlock();

    vecfs_ann_search_context_put(
        sb,
        context);

    return ret;
}


