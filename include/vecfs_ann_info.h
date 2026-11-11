#ifndef _VECFS_ANN_INFO_H
#define _VECFS_ANN_INFO_H

#include <linux/types.h>

#include "vecfs_ann_store.h"

struct inode;
struct super_block;

struct vecfs_ann_vector_record {
    struct vecfs_ann_object_info info;

    void *vector;
    size_t vector_bytes;
};

int vecfs_ann_get_by_label(
    struct super_block *sb,
    u64 external_id,
    bool include_deleted,
    bool allow_file_owner,
    struct vecfs_ann_vector_record *record);

int vecfs_ann_get_file_vector(
    struct inode *inode,
    u32 ordinal,
    bool include_deleted,
    struct vecfs_ann_vector_record *record);

void vecfs_ann_vector_record_destroy(
    struct vecfs_ann_vector_record *record);

#endif
