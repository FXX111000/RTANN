#ifndef _VECFS_ANN_MUTATION_H
#define _VECFS_ANN_MUTATION_H

#include <linux/types.h>

struct inode;
struct super_block;
struct vecfs_ann_snapshot;

enum vecfs_ann_access_class {
    VECFS_ANN_ACCESS_STANDALONE = 1,
    VECFS_ANN_ACCESS_FILE = 2,
    VECFS_ANN_ACCESS_INTERNAL = 3,
};

struct vecfs_ann_update_request {
    const void *vector;
    size_t vector_bytes;

    u32 neighbor_probability_ppm;

    /*
     * FILE vector update 时是否同步修改绑定范围。
     */
    bool update_binding;

    u16 binding_flags;

    u64 content_offset;
    u64 content_length;
};



int vecfs_ann_mark_delete_by_label(
    struct super_block *sb,
    u64 external_id,
    enum vecfs_ann_access_class access);

int vecfs_ann_unmark_delete_by_label(
    struct super_block *sb,
    u64 external_id,
    enum vecfs_ann_access_class access);

int vecfs_ann_update_by_label(
    struct super_block *sb,
    u64 external_id,
    const struct vecfs_ann_update_request *request,
    enum vecfs_ann_access_class access);

int vecfs_ann_file_mark_delete(
    struct inode *inode,
    u32 ordinal);

int vecfs_ann_file_update(
    struct inode *inode,
    u32 ordinal,
    const struct vecfs_ann_update_request *request);

bool vecfs_ann_snapshot_has_deletions(
    const struct vecfs_ann_snapshot *snapshot);

#endif
