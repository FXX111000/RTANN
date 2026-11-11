// SPDX-License-Identifier: GPL-2.0

#include <linux/bitmap.h>
#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/prefetch.h>

#include "../include/vecfs_ann_build.h"
#include "../include/vecfs_ann_cache.h"
#include "../include/vecfs_ann_distance.h"

static u32 vecfs_ann_update_random(u64 *state)
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

static bool vecfs_ann_update_selected(
    u64 *state,
    u32 probability_ppm)
{
    if (probability_ppm >= 1000000)
        return true;

    if (!probability_ppm)
        return false;

    return vecfs_ann_update_random(state) %
               1000000 <
           probability_ppm;
}

static const u32 *vecfs_ann_update_links(
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

static u32 *vecfs_ann_update_links_mutable(
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

static void vecfs_ann_update_set_count(
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

static u16 vecfs_ann_update_max_connections(
    const struct vecfs_ann_snapshot *snapshot,
    u16 level)
{
    return level ? snapshot->m : snapshot->m0;
}

static int vecfs_ann_update_rebuild_neighbor(
    struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_distance_session *distance,
    struct vecfs_ann_build_context *build,
    u32 neighbor,
    u16 level,
    u32 candidate_count,
    u32 ef_construction)
{
    struct vecfs_ann_heap *heap =
        &build->search->results;
    struct vecfs_ann_heap_item *items;
    const float *neighbor_vector;
    u32 elements_to_keep;
    u32 sorted_count;
    u32 selected_count;
    u32 index;
    u32 *links;
    int ret;

    vecfs_ann_heap_reset(heap);

    neighbor_vector =
        vecfs_ann_snapshot_vector(
            snapshot, neighbor);

    elements_to_keep = min(
        ef_construction,
        candidate_count ?
            candidate_count - 1 : 0);

    if (!elements_to_keep)
        return 0;

    for (index = 0;
         index < candidate_count;
         index++) {
        u32 candidate =
            build->candidate_nodes[index];
        float candidate_distance;

        if (candidate == neighbor)
            continue;

        candidate_distance =
            vecfs_ann_distance_calculate(
                distance,
                neighbor_vector,
                vecfs_ann_snapshot_vector(
                    snapshot, candidate));

        if (heap->size < elements_to_keep) {
            ret = vecfs_ann_heap_push(
                heap,
                candidate,
                candidate_distance);
            if (ret)
                return ret;
        } else {
            const struct vecfs_ann_heap_item *worst =
                vecfs_ann_heap_top(heap);

            if (candidate_distance <
                worst->distance) {
                ret =
                    vecfs_ann_heap_replace_top(
                        heap,
                        candidate,
                        candidate_distance);
                if (ret)
                    return ret;
            }
        }
    }

    ret = vecfs_ann_heap_copy_sorted(
        heap,
        build->search->sorted_results,
        build->search->sorted_capacity,
        &sorted_count);
    if (ret)
        return ret;

    items = build->search->sorted_results;

    ret = vecfs_ann_select_neighbors_heuristic(
        snapshot,
        distance,
        build,
        items,
        sorted_count,
        vecfs_ann_update_max_connections(
            snapshot, level),
        &selected_count);
    if (ret)
        return ret;

    links = vecfs_ann_update_links_mutable(
        snapshot,
        neighbor,
        level);
    if (!links)
        return -EUCLEAN;

    for (index = 0;
         index < selected_count;
         index++)
        links[index] =
            build->selected[index].node_id;

    vecfs_ann_update_set_count(
        snapshot,
        neighbor,
        level,
        selected_count);

    __set_bit(
        neighbor,
        build->dirty_nodes);

    return 0;
}

static int vecfs_ann_update_local_graph(
    struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_distance_session *distance,
    struct vecfs_ann_build_context *build,
    u32 node_id,
    u16 level,
    u32 ef_construction,
    u32 probability_ppm,
    u64 *random_state)
{
    const u32 *one_hop;
    u16 one_hop_count;
    unsigned long candidate;
    unsigned long neighbor;
    u32 candidate_count = 0;
    u16 i;
    int ret;

    bitmap_zero(
        build->candidate_bitmap,
        build->set_capacity);

    bitmap_zero(
        build->neighbor_bitmap,
        build->set_capacity);

    __set_bit(
        node_id,
        build->candidate_bitmap);

    one_hop = vecfs_ann_update_links(
        snapshot,
        node_id,
        level,
        &one_hop_count);

    if (!one_hop || !one_hop_count)
        return 0;

    for (i = 0; i < one_hop_count; i++) {
        u32 first = one_hop[i];

        __set_bit(
            first,
            build->candidate_bitmap);

        if (!vecfs_ann_update_selected(
                random_state,
                probability_ppm))
            continue;

        __set_bit(
            first,
            build->neighbor_bitmap);

        {
            const u32 *two_hop;
            u16 two_hop_count;
            u16 j;

            two_hop = vecfs_ann_update_links(
                snapshot,
                first,
                level,
                &two_hop_count);

            if (!two_hop)
                continue;

            for (j = 0;
                 j < two_hop_count;
                 j++)
                __set_bit(
                    two_hop[j],
                    build->candidate_bitmap);
        }
    }

    for_each_set_bit(
        candidate,
        build->candidate_bitmap,
        build->set_capacity) {
        build->candidate_nodes[
            candidate_count++] = candidate;
    }

    for_each_set_bit(
        neighbor,
        build->neighbor_bitmap,
        build->set_capacity) {
        ret = vecfs_ann_update_rebuild_neighbor(
            snapshot,
            distance,
            build,
            neighbor,
            level,
            candidate_count,
            ef_construction);
        if (ret)
            return ret;
    }

    return 0;
}

/*
 * 搜索并重新建立被更新节点自身的连接。
 *
 * 复用插入算法的搜索和 mutually-connect 逻辑时必须使用
 * update 模式，避免把旧连接简单追加到新连接后面。
 *
 * 当前实现先清空 node 自身各层连接，再重新执行 build insert。
 * 邻居侧 backlink 会由 mutually-connect 重建；旧邻居中残留的
 * backlink 与 hnswlib updatePoint 一样允许存在，之后由局部图修复
 * 和 rebuild 清理。
 */
static void vecfs_ann_update_clear_node_links(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    u16 level;

    vecfs_ann_level0_set_count(
        snapshot,
        node_id,
        0);

    for (level = 1;
         level <= vecfs_ann_node_at(snapshot, node_id)->level;
         level++)
        vecfs_ann_upper_set_count(
            snapshot,
            node_id,
            level,
            0);
}

int vecfs_ann_build_update_node(
    struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_distance_session *distance,
    struct vecfs_ann_build_context *build,
    u32 node_id,
    u32 ef_construction,
    u32 neighbor_probability_ppm,
    u64 *random_state)
{
    u16 node_level;
    u16 level;
    int ret;

    if (!snapshot ||
        !distance ||
        !build ||
        !random_state)
        return -EINVAL;

    if (!vecfs_ann_snapshot_id_valid(
            snapshot, node_id))
        return -ENOENT;

    if (vecfs_ann_node_at(snapshot, node_id)->state !=
            VECFS_ANN_OBJ_LIVE &&
        vecfs_ann_node_at(snapshot, node_id)->state !=
            VECFS_ANN_OBJ_DELETED)
        return -ENOENT;

    if (neighbor_probability_ppm > 1000000)
        return -EINVAL;

    node_level = vecfs_ann_node_at(snapshot, node_id)->level;

    if (snapshot->entry_node == node_id &&
        snapshot->live_count == 1)
        return 0;

    /*
     * hnswlib updatePoint 第一阶段：
     * one-hop + selected two-hop 局部重建。
     */
for (level = 0;
     level <= node_level;
     level++) {
    ret = vecfs_ann_update_local_graph(
        snapshot,
        distance,
        build,
        node_id,
        level,
        ef_construction,
        neighbor_probability_ppm,
        random_state);
    if (ret)
        return ret;
    }

    /*
    * hnswlib repairConnectionsForUpdate() 对 entry point 的处理：
    *
    * entry point 不能以自身为搜索入口重新插入，否则 level search
    * 会把 entry 本身放入候选集，最终产生 self-loop。
    *
    * 局部 one-hop/two-hop 邻居修复已经完成。目标 entry 节点保留
    * 原有连接，只更新向量及受影响邻居。
    */
    if (snapshot->entry_node == node_id)
        return 0;


    /*
    * 非 entry 节点执行 repairConnectionsForUpdate：
    * 清空自身连接后，从全局 entry point 重新搜索并连接。
    */
    vecfs_ann_update_clear_node_links(
        snapshot,
        node_id);

    __set_bit(
        node_id,
        build->dirty_nodes);

    return vecfs_ann_build_insert_node(
        snapshot,
        distance,
        build,
        node_id,
        node_level,
        ef_construction,
        NULL, NULL);

}
