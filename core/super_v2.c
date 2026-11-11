#include <linux/buffer_head.h>
#include <linux/crc32.h>
#include <linux/slab.h>

#include "vecfs.h"
#include "include/vecfs_disk.h"

static u32 vecfs_super_crc(const struct vecfs_super_disk_v2 *disk)
{
    struct vecfs_super_disk_v2 *copy;
    u32 crc;

    copy = kmemdup(disk, sizeof(*copy), GFP_KERNEL);
    if (!copy)
        return 0;

    copy->checksum = 0;
    crc = crc32_le(~0U, (const u8 *)copy, sizeof(*copy)) ^ ~0U;

    kfree(copy);
    return crc;
}

static int __maybe_unused vecfs_validate_super(
    struct super_block *sb,
    const struct vecfs_super_disk_v2 *disk)
{
    u64 blocks;
    u64 inodes;
    u64 inode_table;
    u64 data_start;
    u64 ann_a;
    u64 ann_b;
    u32 stored_crc;
    u32 calculated_crc;

    if (le32_to_cpu(disk->magic) != VECFS_SUPER_MAGIC)
        return -EINVAL;

    if (le16_to_cpu(disk->major) != VECFS_SUPER_MAJOR)
        return -EOPNOTSUPP;

    if (le32_to_cpu(disk->header_size) != sizeof(*disk))
        return -EUCLEAN;

    if (le32_to_cpu(disk->block_size) !=
        VECFS_DISK_BLOCK_SIZE)
        return -EINVAL;

    if (le32_to_cpu(disk->inode_size) !=
        VECFS_DISK_INODE_SIZE)
        return -EOPNOTSUPP;

    stored_crc = le32_to_cpu(disk->checksum);
    calculated_crc = vecfs_super_crc(disk);

    if (!calculated_crc || stored_crc != calculated_crc)
        return -EBADMSG;

    blocks = le64_to_cpu(disk->blocks_count);
    inodes = le64_to_cpu(disk->inodes_count);
    inode_table = le64_to_cpu(disk->inode_table);
    data_start = le64_to_cpu(disk->data_start);
    ann_a = le64_to_cpu(disk->ann_meta_a);
    ann_b = le64_to_cpu(disk->ann_meta_b);

    /*
     * 当前块指针和 node ID 都是 u32。
     */
    if (!blocks || blocks > U32_MAX)
        return -EFBIG;

    if (!inodes || inodes > U32_MAX)
        return -EFBIG;

    if (!inode_table || inode_table >= blocks)
        return -EUCLEAN;

    if (!data_start || data_start >= blocks)
        return -EUCLEAN;

    if (!ann_a || !ann_b || ann_a == ann_b)
        return -EUCLEAN;

    if (ann_a >= data_start || ann_b >= data_start)
        return -EUCLEAN;

    if (le32_to_cpu(disk->root_ino) != VECFS_ROOT_INO)
        return -EUCLEAN;

    return 0;
}
