#ifndef _VECFS_ANN_VALIDATE_H
#define _VECFS_ANN_VALIDATE_H

#include <linux/types.h>

struct super_block;

#define VECFS_ANN_VALIDATE_F_REPAIR       (1U << 0)
#define VECFS_ANN_VALIDATE_F_STRICT       (1U << 1)
#define VECFS_ANN_VALIDATE_F_CHECK_LINKS  (1U << 2)

struct vecfs_ann_validate_report {
    u64 objects_checked;
    u64 live_objects;
    u64 deleted_objects;
    u64 corrupt_objects;

    u64 file_objects;
    u64 standalone_objects;
    u64 imported_objects;

    u64 inode_mappings_checked;

    u64 missing_inodes;
    u64 generation_mismatches;
    u64 missing_inode_mappings;
    u64 stale_inode_mappings;
    u64 owner_mismatches;
    u64 ordinal_mismatches;
    u64 duplicate_external_ids;
    u64 invalid_neighbors;

    u64 objects_marked_deleted;
    u64 mappings_cleared;
    u64 metadata_repairs;

    u32 recovered_entry_node;
    u16 recovered_max_level;
    u16 reserved;
};

int vecfs_ann_validate(
    struct super_block *sb,
    u32 flags,
    struct vecfs_ann_validate_report *report);

int vecfs_ann_validate_mount(
    struct super_block *sb);

#endif
