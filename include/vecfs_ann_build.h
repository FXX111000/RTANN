#ifndef _VECFS_ANN_BUILD_H
#define _VECFS_ANN_BUILD_H

#include <linux/spinlock.h>
#include <linux/types.h>

#include "vecfs_ann_cache.h"
#include "vecfs_ann_distance.h"
#include "vecfs_ann_heap.h"
#include "vecfs_ann_search.h"

struct inode;
struct super_block;

#define VECFS_ANN_INSERT_F_HAS_CONTENT_RANGE  (1U << 0)

/*
 * 并行构图（workqueue 多 worker 的 hnswlib 式并发插入）用的每节点链锁。
 *
 * 每个节点（level0 + upper）的邻接表用一个锁保护，node_id 散列到条纹数组
 * 中的一个锁；两 worker 同时对同一邻居 add_backlink / trim 时据此互斥。
 * 为 NULL 表示调用方已在外部串行化（例如持有 cache->writer_lock 的单条插入、
 * 批量插入、顺序重构），无需逐节点加锁。STRIPES 必须是 2 的幂，配合 mask
 * 用位与定位，避免取模。
 */
#define VECFS_ANN_BUILD_LOCK_STRIPES 4096U

struct vecfs_ann_link_locks {
    spinlock_t *locks;
    u32 mask;   /* STRIPES - 1 */
};

struct vecfs_ann_insert_request {
    const void *vector;
    size_t vector_bytes;

    u64 external_id;

    u64 owner_ino;
    u32 owner_generation;
    u32 vector_ordinal;

    u64 content_offset;
    u64 content_length;

    u16 owner_type;
    u16 flags;

    /*
     * SHIELD 原子条件发布：为 true 时，新节点在发布进查询快照的
     * 那一刻即被标记为 valid_bitmap = 0（无效），直到派生 worker
     * 在版本复核通过后原子提交才翻转为有效。这样并发写入在新节点
     * 尚未提交前不会把过期向量暴露给查询。仅内存语义，不落盘。
     */
    bool start_invalid;
};

struct vecfs_ann_insert_result {
    u32 node_id;
    u16 level;
    u16 reserved;

    u64 external_id;
};

struct vecfs_ann_build_context {
    struct vecfs_ann_search_context *search;

    struct vecfs_ann_heap_item *ordered;
    struct vecfs_ann_heap_item *selected;

    u32 *neighbor_ids;
    u32 temporary_capacity;

    unsigned long *dirty_nodes;
    u32 dirty_capacity;

    /*
     * updatePoint 的一跳/二跳集合。
     */
    unsigned long *candidate_bitmap;
    unsigned long *neighbor_bitmap;

    u32 *candidate_nodes;
    u32 *selected_neighbors;

    u32 set_capacity;
};


int vecfs_ann_build_context_init(
    struct super_block *sb,
    struct vecfs_ann_build_context *context,
    u32 node_capacity);

void vecfs_ann_build_context_destroy(
    struct vecfs_ann_build_context *context);

int vecfs_ann_insert(
    struct super_block *sb,
    const struct vecfs_ann_insert_request *request,
    struct vecfs_ann_insert_result *result);

int vecfs_ann_insert_file_vector(
    struct inode *inode,
    u32 ordinal,
    const void *vector,
    size_t vector_bytes,
    u64 external_id,
    u64 content_offset,
    u64 content_length,
    u32 flags,
    struct vecfs_ann_insert_result *result);

/*
 * 与 vecfs_ann_insert_file_vector 相同，但以「插入即无效」语义发布：
 * 新节点在发布进查询快照时 valid_bitmap 已为 0，需后续 mark_node_valid
 * 提交后才对查询可见。SHIELD 原子条件发布专用。
 */
int vecfs_ann_insert_file_vector_invalid(
    struct inode *inode,
    u32 ordinal,
    const void *vector,
    size_t vector_bytes,
    u64 external_id,
    u64 content_offset,
    u64 content_length,
    u32 flags,
    struct vecfs_ann_insert_result *result);

int vecfs_ann_select_neighbors_heuristic(
    const struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_distance_session *distance,
    struct vecfs_ann_build_context *context,
    const struct vecfs_ann_heap_item *candidates,
    u32 candidate_count,
    u32 maximum,
    u32 *selected_count);

int vecfs_ann_build_insert_node(
    struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_distance_session *distance,
    struct vecfs_ann_build_context *build,
    u32 new_node,
    u16 new_level,
    u32 ef_construction,
    struct vecfs_ann_link_locks *locks,
    spinlock_t *entry_lock);

int vecfs_ann_build_update_node(
    struct vecfs_ann_snapshot *snapshot,
    const struct vecfs_ann_distance_session *distance,
    struct vecfs_ann_build_context *build,
    u32 node_id,
    u32 ef_construction,
    u32 neighbor_probability_ppm,
    u64 *random_state);

#endif
