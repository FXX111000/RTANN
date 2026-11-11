#ifndef _VECFS_ANN_STORE_H
#define _VECFS_ANN_STORE_H

#include <linux/types.h>

#include "vecfs_disk.h"

struct inode;
struct super_block;

struct vecfs_ann_object_layout {
    u64 stride;

    u64 level0_offset;
    u64 vector_offset;
    u64 upper_counts_offset;
    u64 upper_links_offset;

    u64 vector_bytes;

    u32 dimension;
    u16 element_type;
    u16 m;
    u16 m0;
    u16 max_levels;
};

struct vecfs_ann_store {
    struct super_block *sb;
    struct inode *arena_inode;

    struct mutex allocation_lock;
    struct rw_semaphore io_lock;

    struct vecfs_ann_object_layout layout;

    u64 scanned_slots;
    u64 corrupt_objects;
};

struct vecfs_ann_object_create {
    u64 external_id;

    u64 owner_ino;
    u32 owner_generation;
    u32 vector_ordinal;

    u64 content_offset;
    u64 content_length;

    u16 level;
    u16 owner_type;
    u16 flags;

    const void *vector;
    size_t vector_bytes;
};

struct vecfs_ann_object_info {
    u32 node_id;
    u32 state;

    u64 external_id;

    u64 owner_ino;
    u32 owner_generation;
    u32 vector_ordinal;

    u64 content_offset;
    u64 content_length;

    u16 level;
    u16 level0_count;
    u16 owner_type;
    u16 flags;

    u64 object_epoch;
};

int vecfs_ann_store_mount(struct super_block *sb);
void vecfs_ann_store_unmount(struct super_block *sb);

int vecfs_ann_store_scan(struct super_block *sb);

int vecfs_ann_object_prepare(
    struct super_block *sb,
    const struct vecfs_ann_object_create *create,
    u32 *node_id);

int vecfs_ann_object_publish(
    struct super_block *sb,
    u32 node_id,
    bool sync);

int vecfs_ann_object_mark_deleted(
    struct super_block *sb,
    u32 node_id);

int vecfs_ann_object_read(
    struct super_block *sb,
    u32 node_id,
    void *object,
    size_t object_size);

int vecfs_ann_object_read_info(
    struct super_block *sb,
    u32 node_id,
    struct vecfs_ann_object_info *info);

int vecfs_ann_object_read_vector(
    struct super_block *sb,
    u32 node_id,
    void *vector,
    size_t vector_size);

int vecfs_ann_object_write_vector(
    struct super_block *sb,
    u32 node_id,
    const void *vector,
    size_t vector_size);

int vecfs_ann_object_read_level0(
    struct super_block *sb,
    u32 node_id,
    u32 *neighbors,
    u16 capacity,
    u16 *count);

int vecfs_ann_object_write_level0(
    struct super_block *sb,
    u32 node_id,
    const u32 *neighbors,
    u16 count);

int vecfs_ann_object_read_upper(
    struct super_block *sb,
    u32 node_id,
    u16 level,
    u32 *neighbors,
    u16 capacity,
    u16 *count);

int vecfs_ann_object_write_upper(
    struct super_block *sb,
    u32 node_id,
    u16 level,
    const u32 *neighbors,
    u16 count);
    
int vecfs_ann_object_unmark_deleted(
    struct super_block *sb,
    u32 node_id);

struct vecfs_ann_snapshot;

int vecfs_ann_store_rewrite_snapshot(
    struct super_block *sb,
    u32 arena_inode_number,
    const struct vecfs_ann_snapshot *snapshot,
    u64 epoch,
    struct vecfs_ann_store **result_store);

int vecfs_ann_object_update_flags(
    struct super_block *sb,
    u32 node_id,
    u16 clear_mask,
    u16 set_mask);

int vecfs_ann_object_update_binding(
    struct super_block *sb,
    u32 node_id,
    u64 content_offset,
    u64 content_length,
    u16 flags);

int vecfs_ann_object_read_raw(
    struct super_block *sb,
    u32 node_id,
    void *object,
    size_t object_size);

int vecfs_ann_object_restore_raw(
    struct super_block *sb,
    u32 node_id,
    const void *object,
    size_t object_size);

void vecfs_ann_store_release_instance(
    struct vecfs_ann_store *store,
    bool write_inode);

#endif
