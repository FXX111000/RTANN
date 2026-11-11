#ifndef _VECFS_INDIRECT_H
#define _VECFS_INDIRECT_H

#include <linux/blkdev.h>
#include <linux/buffer_head.h>
#include <linux/fs.h>
#include <linux/types.h>

typedef unsigned long long vecfs_fsblk_t;

struct inode;

sector_t vecfs_bmap(
    struct address_space *mapping,
    sector_t logical);

int vecfs_get_block(
    struct inode *inode,
    sector_t logical,
    struct buffer_head *result,
    int create);

long vecfs_fallocate(
    struct file *file,
    int mode,
    loff_t offset,
    loff_t length);

int vecfs_truncate_blocks(
    struct inode *inode,
    loff_t new_size);

int vecfs_alloc_fs_block(
    struct inode *inode,
    u32 *block);

int vecfs_zero_fs_block(
    struct inode *inode,
    u32 block);

void vecfs_free_block(
    struct inode *inode,
    unsigned long block);

#endif
