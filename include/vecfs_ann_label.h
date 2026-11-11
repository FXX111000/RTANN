#ifndef _VECFS_ANN_LABEL_H
#define _VECFS_ANN_LABEL_H

#include <linux/rhashtable.h>
#include <linux/types.h>

struct super_block;
struct vecfs_ann_snapshot;

struct vecfs_ann_label_entry {
    struct rhash_head hash_node;

    u64 external_id;
    u32 node_id;
};

struct vecfs_ann_label_index {
    struct rhashtable table;
    bool initialized;
};

int vecfs_ann_label_index_init(
    struct super_block *sb);

void vecfs_ann_label_index_destroy(
    struct super_block *sb);

/*
 * 这些修改接口必须持有 cache->writer_lock。
 */
int vecfs_ann_label_insert_locked(
    struct super_block *sb,
    u64 external_id,
    u32 node_id);

int vecfs_ann_label_remove_locked(
    struct super_block *sb,
    u64 external_id,
    u32 expected_node_id);

int vecfs_ann_label_lookup_locked(
    struct super_block *sb,
    u64 external_id,
    u32 *node_id);

int vecfs_ann_label_rebuild_locked(
    struct super_block *sb,
    const struct vecfs_ann_snapshot *snapshot);

#endif
