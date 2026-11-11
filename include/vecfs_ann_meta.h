#ifndef _VECFS_ANN_META_H
#define _VECFS_ANN_META_H

#include <linux/types.h>
#include <linux/mutex.h>

#include "vecfs_disk.h"

struct super_block;

struct vecfs_ann_config {
    u32 dimension;
    u16 element_type;
    u16 distance_type;

    u16 m;
    u16 m0;
    u16 max_levels;
    u16 current_max_level;

    u32 ef_construction;
    u32 default_ef_search;

    u64 random_seed;
    u64 level_mult_q32;

    u64 object_stride;
    u64 vector_bytes;
};

struct vecfs_ann_meta {
    struct mutex lock;

    struct vecfs_ann_config config;

    u64 sequence;
    u64 epoch;

    u32 state;
    u32 flags;

    u64 live_nodes;
    u64 deleted_nodes;
    u64 next_node_id;
    u64 next_external_id;

    u32 entry_node_id;
    u32 active_arena_ino;

    u32 rebuild_arena_ino;
    u32 rebuild_state;

    u64 committed_txid;

    /*
     * 当前选中的 metadata 副本：
     * 0 = A, 1 = B。
     */
    u8 active_copy;
    bool dirty;
};

int vecfs_ann_validate_config(const struct vecfs_ann_config *cfg);

u64 vecfs_ann_calculate_object_stride(
    u32 dimension,
    u16 element_type,
    u16 m,
    u16 m0,
    u16 max_levels);

int vecfs_ann_meta_load(struct super_block *sb);
int vecfs_ann_meta_commit(struct super_block *sb, bool sync);
void vecfs_ann_meta_destroy(struct super_block *sb);

#endif
