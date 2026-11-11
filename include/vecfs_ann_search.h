#ifndef _VECFS_ANN_SEARCH_H
#define _VECFS_ANN_SEARCH_H

#include <linux/list.h>
#include <linux/mutex.h>
#include <linux/types.h>
#include <linux/wait.h>

#include "vecfs_ann_heap.h"
#include "vecfs_ann_visited.h"

#define VECFS_ANN_SEARCH_MAX_K   4096U
#define VECFS_ANN_SEARCH_MAX_EF  65535U

#define VECFS_ANN_SEARCH_F_FILE_ONLY \
    (1U << 0)

#define VECFS_ANN_SEARCH_F_STANDALONE_ONLY \
    (1U << 1)

#define VECFS_ANN_SEARCH_F_INCLUDE_STALE \
    (1U << 2)

#define VECFS_ANN_SEARCH_VALID_FLAGS \
    (VECFS_ANN_SEARCH_F_FILE_ONLY | \
     VECFS_ANN_SEARCH_F_STANDALONE_ONLY | \
     VECFS_ANN_SEARCH_F_INCLUDE_STALE)

struct super_block;

struct vecfs_ann_search_context {
    struct list_head pool_node;

    struct vecfs_ann_visited visited;

    /*
     * candidates 最坏情况下每个 node 进入一次，因此容量按 snapshot
     * capacity 分配，查询期间绝不扩容。
     */
    struct vecfs_ann_heap candidates;

    /*
     * results 容量按 ef 分配。
     */
    struct vecfs_ann_heap results;

    struct vecfs_ann_heap_item *sorted_results;
    u32 sorted_capacity;

    u32 node_capacity;
    u32 result_capacity;
};

struct vecfs_ann_search_pool {
    struct mutex lock;
    wait_queue_head_t wait;
    struct list_head free_contexts;

    u32 total_contexts;
    u32 free_count;
    u32 maximum_contexts;

    bool shutting_down;
};

struct vecfs_ann_search_request {
    const void *query_vector;
    size_t query_bytes;

    u32 k;
    u32 ef_search;
    u32 flags;
};

struct vecfs_ann_search_result {
    u32 node_id;
    u32 reserved32;

    u64 external_id;

    u64 owner_ino;
    u32 owner_generation;
    u32 vector_ordinal;

    u64 content_offset;
    u64 content_length;

    u16 owner_type;
    u16 flags;
    u32 reserved_result;

    float distance;
};

int vecfs_ann_search_context_init(
    struct vecfs_ann_search_context *context,
    u32 node_capacity,
    u32 result_capacity);

void vecfs_ann_search_context_destroy(
    struct vecfs_ann_search_context *context);

int vecfs_ann_search_context_prepare(
    struct vecfs_ann_search_context *context,
    u32 node_capacity,
    u32 result_capacity);

void vecfs_ann_search_context_reset(
    struct vecfs_ann_search_context *context);

int vecfs_ann_search_pool_init(
    struct super_block *sb,
    u32 maximum_contexts);

void vecfs_ann_search_pool_destroy(
    struct super_block *sb);

int vecfs_ann_search_context_get(
    struct super_block *sb,
    u32 node_capacity,
    u32 result_capacity,
    struct vecfs_ann_search_context **context);

void vecfs_ann_search_context_put(
    struct super_block *sb,
    struct vecfs_ann_search_context *context);

int vecfs_ann_search(
    struct super_block *sb,
    const struct vecfs_ann_search_request *request,
    struct vecfs_ann_search_result *results,
    u32 result_capacity,
    u32 *result_count);

#endif
