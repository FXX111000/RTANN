#ifndef _VECFS_ANN_EXCHANGE_H
#define _VECFS_ANN_EXCHANGE_H

#include <linux/mutex.h>
#include <linux/rhashtable.h>
#include <linux/types.h>

struct super_block;
struct vecfs_ann_snapshot;

#define VECFS_IMPORT_F_PRESERVE_NODE_IDS  (1U << 0)
#define VECFS_IMPORT_F_ABORT_EXISTING     (1U << 1)

struct vecfs_ann_import_label {
    struct rhash_head hash_node;

    u64 external_id;
    u32 node_id;
};

struct vecfs_ann_import_transaction {
    u64 transaction_id;

    struct vecfs_ann_snapshot *snapshot;

    unsigned long *present_nodes;

    struct rhashtable labels;
    bool labels_initialized;

    u64 expected_nodes;
    u64 imported_nodes;

    u32 capacity;

    bool active;
};

struct vecfs_ann_exchange {
    struct mutex lock;

    struct vecfs_ann_import_transaction import;
};

struct vecfs_ann_import_begin {
    u16 element_type;
    u16 distance_type;

    u32 dimension;

    u16 m;
    u16 m0;
    u16 max_levels;

    u64 expected_nodes;
    u32 node_capacity;
    u32 flags;
};

struct vecfs_ann_import_node {
    u32 node_id;

    u64 external_id;

    u16 level;

    const void *vector;
    u32 vector_bytes;

    const u32 *level0_links;
    u16 level0_count;

    const u16 *upper_counts;
    const u32 *upper_links;
};

struct vecfs_ann_export_node {
    u32 node_id;
    u64 external_id;

    u16 level;
    u16 level0_count;

    const void *vector;
    u32 vector_bytes;

    const u32 *level0_links;

    const u16 *upper_counts;
    const u32 *upper_links;
};

int vecfs_ann_exchange_init(
    struct super_block *sb);

void vecfs_ann_exchange_destroy(
    struct super_block *sb);

int vecfs_ann_import_begin(
    struct super_block *sb,
    const struct vecfs_ann_import_begin *request,
    u64 *transaction_id);

int vecfs_ann_import_node(
    struct super_block *sb,
    u64 transaction_id,
    const struct vecfs_ann_import_node *node);

int vecfs_ann_import_commit(
    struct super_block *sb,
    u64 transaction_id,
    u32 entry_node,
    u16 current_max_level,
    u64 *imported_nodes,
    u64 *new_epoch);

int vecfs_ann_import_abort(
    struct super_block *sb,
    u64 transaction_id);

int vecfs_ann_export_info(
    struct super_block *sb,
    u64 *node_count,
    u32 *capacity,
    u32 *entry_node,
    u16 *max_level,
    u64 *epoch);

int vecfs_ann_export_node(
    struct super_block *sb,
    u32 node_id,
    struct vecfs_ann_export_node *node);

#endif
