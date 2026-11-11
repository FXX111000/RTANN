#ifndef _VECFS_ANN_STORE_INTERNAL_H
#define _VECFS_ANN_STORE_INTERNAL_H

#include <linux/types.h>

#include "../../include/vecfs_ann_store.h"

struct vecfs_sb_info;

int vecfs_ann_layout_init(
    struct vecfs_ann_object_layout *layout,
    const struct vecfs_ann_config *config);

int vecfs_ann_arena_read_locked(
    struct vecfs_ann_store *store,
    loff_t position,
    void *buffer,
    size_t length);

int vecfs_ann_arena_write_locked(
    struct vecfs_ann_store *store,
    loff_t position,
    const void *buffer,
    size_t length,
    bool sync);

int vecfs_ann_object_load_locked(
    struct vecfs_ann_store *store,
    u32 node_id,
    void *object);

int vecfs_ann_object_store_locked(
    struct vecfs_ann_store *store,
    u32 node_id,
    void *object,
    bool sync);

bool vecfs_ann_object_valid(
    struct vecfs_ann_store *store,
    u32 expected_id,
    const void *object);

u32 vecfs_ann_object_crc(
    struct vecfs_ann_store *store,
    void *object);

u32 vecfs_ann_vector_crc(
    struct vecfs_ann_store *store,
    const void *vector);

void vecfs_ann_object_to_info(
    const struct vecfs_ann_object_disk *disk,
    struct vecfs_ann_object_info *info);

int vecfs_ann_object_quarantine_locked(
    struct vecfs_ann_store *store,
    u32 node_id);

int vecfs_ann_get_arena_inode(
    struct super_block *sb,
    u32 inode_number,
    struct inode **result);

void vecfs_ann_store_release_instance(
    struct vecfs_ann_store *store,
    bool write_inode);

#endif
