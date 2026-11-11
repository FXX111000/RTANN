// SPDX-License-Identifier: GPL-2.0

#include <linux/bitmap.h>
#include <linux/blkdev.h>
#include <linux/buffer_head.h>
#include <linux/errno.h>
#include <linux/falloc.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/limits.h>
#include <linux/pagemap.h>

#include "vecfs.h"
#include "vecfs_Indirect.h"
#include "include/vecfs_disk.h"
#include "../include/vecfs_ann_stale.h"

#define VECFS_PTRS_PER_BLOCK \
    (VECFS_DISK_BLOCK_SIZE / sizeof(__le32))

struct vecfs_block_path {
    unsigned int depth;
    unsigned int root;
    unsigned int offsets[3];
};

static bool vecfs_block_reserved(const struct vecfs_sb_info *sbi, u32 block)
{
    u32 inode_table_blocks;

    if (!sbi)
        return true;

    if (!block || block >= sbi->s_blocks_count)
        return true;

    /*
     * V2 的 superblock、bitmap、inode table 和 ANN metadata A/B
     * 全部位于 data_start 之前。
     */
    if (block < sbi->s_data_start)
        return true;

    inode_table_blocks = DIV_ROUND_UP(
        sbi->s_inodes_count,
        sbi->s_blocksize / VECFS_DISK_INODE_SIZE);

    if (block >= sbi->s_inode_table &&
        block < sbi->s_inode_table + inode_table_blocks)
        return true;

    return false;
}

static int vecfs_logical_block_to_path(
    sector_t logical,
    struct vecfs_block_path *path)
{
    u64 ptrs = VECFS_PTRS_PER_BLOCK;
    u64 double_capacity = ptrs * ptrs;
    u64 triple_capacity = double_capacity * ptrs;
    u64 block = logical;

    memset(path, 0, sizeof(*path));

    if (block < VECFS_N_DIRECT) {
        path->depth = 0;
        path->root = block;
        return 0;
    }

    block -= VECFS_N_DIRECT;

    if (block < ptrs) {
        path->depth = 1;
        path->root = VECFS_IND_BLOCK;
        path->offsets[0] = block;
        return 0;
    }

    block -= ptrs;

    if (block < double_capacity) {
        path->depth = 2;
        path->root = VECFS_DIND_BLOCK;
        path->offsets[0] = div_u64(block, ptrs);
        path->offsets[1] = block % ptrs;
        return 0;
    }

    block -= double_capacity;

    if (block < triple_capacity) {
        path->depth = 3;
        path->root = VECFS_TIND_BLOCK;
        path->offsets[0] = div_u64(block, double_capacity);
        block %= double_capacity;
        path->offsets[1] = div_u64(block, ptrs);
        path->offsets[2] = block % ptrs;
        return 0;
    }

    return -EFBIG;
}

int vecfs_alloc_fs_block(struct inode *inode, u32 *out)
{
    struct super_block *sb;
    struct vecfs_sb_info *sbi;
    unsigned long start;
    unsigned long block;

    if (!inode || !inode->i_sb || !out)
        return -EINVAL;

    sb = inode->i_sb;
    sbi = VECFS_SB(sb);
    if (!sbi)
        return -EINVAL;

    spin_lock(&sbi->s_block_lock);

    if (!sbi->s_free_blocks) {
        spin_unlock(&sbi->s_block_lock);
        return -ENOSPC;
    }

    start = max_t(
        unsigned long,
        sbi->s_next_free_block,
        sbi->s_data_start);

    block = find_next_zero_bit(
        sbi->s_bmap,
        sbi->s_blocks_count,
        start);

    if (block >= sbi->s_blocks_count) {
        block = find_next_zero_bit(
            sbi->s_bmap,
            sbi->s_blocks_count,
            sbi->s_data_start);

        if (block >= start) {
            spin_unlock(&sbi->s_block_lock);
            return -ENOSPC;
        }
    }

    if (unlikely(vecfs_block_reserved(sbi, block))) {
        spin_unlock(&sbi->s_block_lock);
        pr_err("vecfs: allocator selected reserved block %lu\n",
               block);
        return -EUCLEAN;
    }

    __set_bit(block, sbi->s_bmap);
    sbi->s_free_blocks--;

    sbi->s_next_free_block = block + 1;
    if (sbi->s_next_free_block >= sbi->s_blocks_count)
        sbi->s_next_free_block = sbi->s_data_start;

    spin_unlock(&sbi->s_block_lock);

    inode->i_blocks += inode->i_sb->s_blocksize >> SECTOR_SHIFT;
    *out = block;

    mark_inode_dirty(inode);
    return 0;
}

int vecfs_zero_fs_block(struct inode *inode, u32 block)
{
    struct buffer_head *bh;

    bh = sb_getblk(inode->i_sb, block);
    if (!bh)
        return -ENOMEM;

    lock_buffer(bh);
    memset(bh->b_data, 0, inode->i_sb->s_blocksize);
    set_buffer_uptodate(bh);
    clear_buffer_new(bh);
    unlock_buffer(bh);

    mark_buffer_dirty_inode(bh, inode);
    brelse(bh);

    return 0;
}

static int vecfs_alloc_zeroed_block(struct inode *inode, u32 *out)
{
    u32 block;
    int ret;

    ret = vecfs_alloc_fs_block(inode, &block);
    if (ret)
        return ret;

    ret = vecfs_zero_fs_block(inode, block);
    if (ret) {
        vecfs_free_block(inode, block);
        return ret;
    }

    *out = block;
    return 0;
}

void vecfs_free_block(struct inode *inode, unsigned long block)
{
    struct vecfs_sb_info *sbi;

    if (!inode || !inode->i_sb)
        return;

    sbi = VECFS_SB(inode->i_sb);
    if (!sbi)
        return;

    if (block < sbi->s_data_start ||
        block >= sbi->s_blocks_count) {
        pr_err("vecfs: refusing to free invalid block %lu\n",
               block);
        return;
    }

    spin_lock(&sbi->s_block_lock);

    if (test_bit(block, sbi->s_bmap)) {
        __clear_bit(block, sbi->s_bmap);
        sbi->s_free_blocks++;

        if (block < sbi->s_next_free_block)
            sbi->s_next_free_block = block;
    }

    spin_unlock(&sbi->s_block_lock);

    if (inode->i_blocks >=
        (inode->i_sb->s_blocksize >> SECTOR_SHIFT))
        inode->i_blocks -=
            inode->i_sb->s_blocksize >> SECTOR_SHIFT;
    else
        inode->i_blocks = 0;

    mark_inode_dirty(inode);
}

static int vecfs_get_indirect(
    struct inode *inode,
    u32 *root,
    const unsigned int *offsets,
    unsigned int depth,
    struct buffer_head *result,
    bool create)
{
    struct super_block *sb;
    struct buffer_head *bh = NULL;
    __le32 *entries;
    u32 cur;
    unsigned int level;
    int ret;

    if (!inode || !inode->i_sb)
        return -EINVAL;

    sb = inode->i_sb;

    if (!depth)
        return -EINVAL;

    cur = *root;

    if (!cur) {
        if (!create)
            return 0;

        ret = vecfs_alloc_zeroed_block(inode, &cur);
        if (ret)
            return ret;

        *root = cur;
        mark_inode_dirty(inode);
    }

    for (level = 0; level < depth; level++) {
        unsigned int index = offsets[level];
        u32 next;

        if (index >= VECFS_PTRS_PER_BLOCK)
            return -EFBIG;

        bh = sb_bread(sb, cur);
        if (!bh)
            return -EIO;

        entries = (__le32 *)bh->b_data;
        next = le32_to_cpu(entries[index]);

        if (level == depth - 1) {
            if (!next && create) {
                ret = vecfs_alloc_fs_block(inode, &next);
                if (ret) {
                    brelse(bh);
                    return ret;
                }

                entries[index] = cpu_to_le32(next);
                mark_buffer_dirty_inode(bh, inode);
                set_buffer_new(result);
            }

            if (next)
                map_bh(result, sb, next);

            brelse(bh);
            return 0;
        }

        if (!next) {
            if (!create) {
                brelse(bh);
                return 0;
            }

            ret = vecfs_alloc_zeroed_block(inode, &next);
            if (ret) {
                brelse(bh);
                return ret;
            }

            entries[index] = cpu_to_le32(next);
            mark_buffer_dirty_inode(bh, inode);
        }

        brelse(bh);
        bh = NULL;
        cur = next;
    }

    return -EUCLEAN;
}

int vecfs_get_block(
    struct inode *inode,
    sector_t logical,
    struct buffer_head *result,
    int create)
{
    struct vecfs_inode *vi = VECFS_I(inode);
    struct vecfs_block_path path;
    u32 *root;
    int ret;

    ret = vecfs_logical_block_to_path(logical, &path);
    if (ret)
        return ret;

    down_write(&vi->meta_lock);

    root = &vi->i_block[path.root];

    if (!path.depth) {
        if (!*root && create) {
            ret = vecfs_alloc_fs_block(inode, root);
            if (ret)
                goto out;

            set_buffer_new(result);
        }

        if (*root)
            map_bh(result, inode->i_sb, *root);

        ret = 0;
        goto out;
    }

    ret = vecfs_get_indirect(
        inode,
        root,
        path.offsets,
        path.depth,
        result,
        create);

out:
    up_write(&vi->meta_lock);
    return ret;
}

static bool vecfs_indirect_block_empty(
    const __le32 *entries,
    unsigned int count)
{
    unsigned int i;

    for (i = 0; i < count; i++) {
        if (entries[i])
            return false;
    }

    return true;
}

static int vecfs_free_indirect_entry(
    struct inode *inode,
    u32 *root,
    const unsigned int *offsets,
    unsigned int depth)
{
    struct super_block *sb = inode->i_sb;
    struct buffer_head *bh;
    __le32 *entries;
    u32 block;
    u32 child;
    unsigned int index;
    int ret = 0;

    if (!*root || !depth)
        return 0;

    block = *root;
    index = offsets[0];

    if (index >= VECFS_PTRS_PER_BLOCK)
        return -EFBIG;

    bh = sb_bread(sb, block);
    if (!bh)
        return -EIO;

    entries = (__le32 *)bh->b_data;
    child = le32_to_cpu(entries[index]);

    if (!child) {
        brelse(bh);
        return 0;
    }

    if (depth == 1) {
        entries[index] = 0;
        mark_buffer_dirty_inode(bh, inode);
        vecfs_free_block(inode, child);
    } else {
        u32 child_root = child;

        ret = vecfs_free_indirect_entry(
            inode,
            &child_root,
            offsets + 1,
            depth - 1);
        if (ret) {
            brelse(bh);
            return ret;
        }

        if (!child_root) {
            entries[index] = 0;
            mark_buffer_dirty_inode(bh, inode);
        }
    }

    if (vecfs_indirect_block_empty(
            entries, VECFS_PTRS_PER_BLOCK)) {
        brelse(bh);
        vecfs_free_block(inode, block);
        *root = 0;
        mark_inode_dirty(inode);
        return 0;
    }

    brelse(bh);
    return 0;
}

static int vecfs_free_single_logical_block(
    struct inode *inode,
    sector_t logical)
{
    struct vecfs_inode *vi = VECFS_I(inode);
    struct vecfs_block_path path;
    u32 *root;
    int ret;

    ret = vecfs_logical_block_to_path(logical, &path);
    if (ret)
        return ret;

    root = &vi->i_block[path.root];

    if (!path.depth) {
        if (*root) {
            u32 physical = *root;

            *root = 0;
            vecfs_free_block(inode, physical);
            mark_inode_dirty(inode);
        }

        return 0;
    }

    return vecfs_free_indirect_entry(
        inode,
        root,
        path.offsets,
        path.depth);
}

static int vecfs_free_block_range(
    struct inode *inode,
    sector_t first,
    sector_t last)
{
    struct vecfs_inode *vi = VECFS_I(inode);
    sector_t block;
    int ret = 0;

    if (last < first)
        return 0;

    down_write(&vi->meta_lock);

    for (block = first; block <= last; block++) {
        ret = vecfs_free_single_logical_block(
            inode, block);
        if (ret)
            break;

        if (block == (sector_t)-1)
            break;
    }

    up_write(&vi->meta_lock);
    return ret;
}

sector_t vecfs_bmap(
    struct address_space *mapping,
    sector_t logical)
{
    struct buffer_head map = { 0 };
    int ret;

    ret = vecfs_get_block(
        mapping->host,
        logical,
        &map,
        0);
    if (ret || !buffer_mapped(&map))
        return 0;

    return map.b_blocknr;
}

static int vecfs_allocate_blocks(
    struct inode *inode,
    loff_t offset,
    loff_t length)
{
    struct buffer_head map;
    sector_t first;
    sector_t last;
    sector_t logical;
    int ret;

    if (length <= 0)
        return -EINVAL;

    first = offset >> inode->i_blkbits;
    last = (offset + length - 1) >> inode->i_blkbits;

    for (logical = first; logical <= last; logical++) {
        memset(&map, 0, sizeof(map));

        ret = vecfs_get_block(inode, logical, &map, 1);
        if (ret)
            return ret;

        if (!buffer_mapped(&map))
            return -EIO;

        if (buffer_new(&map)) {
            ret = vecfs_zero_fs_block(
                inode, map.b_blocknr);
            if (ret)
                return ret;
        }
    }

    return 0;
}

static int vecfs_punch_hole(
    struct inode *inode,
    loff_t offset,
    loff_t length)
{
    sector_t first;
    sector_t last;
    loff_t size;
    int ret;

    if (!inode)
        return -EINVAL;

    if (offset < 0 ||
        length <= 0)
        return -EINVAL;

    if (!IS_ALIGNED(
            offset,
            inode->i_sb->s_blocksize) ||
        !IS_ALIGNED(
            length,
            inode->i_sb->s_blocksize))
        return -EOPNOTSUPP;

    size = i_size_read(inode);

    if (offset >= size)
        return 0;

    if (length > size - offset)
        length = size - offset;

    if (!length)
        return 0;

    /*
     * 在释放数据块之前持久化 stale 状态。
     */
    ret = vecfs_ann_mark_content_stale(
        inode,
        offset,
        length);
    if (ret)
        return ret;

    first =
        offset >> inode->i_blkbits;

    last =
        (offset + length - 1) >>
        inode->i_blkbits;

    truncate_pagecache_range(
        inode,
        offset,
        offset + length - 1);

    return vecfs_free_block_range(
        inode,
        first,
        last);
}


long vecfs_fallocate(
    struct file *file,
    int mode,
    loff_t offset,
    loff_t length)
{
    struct inode *inode = file_inode(file);
    loff_t end;
    int ret;

    if (length <= 0 || offset < 0)
        return -EINVAL;

    if (offset > LLONG_MAX - length)
        return -EOVERFLOW;

    end = offset + length;

    if (end > inode->i_sb->s_maxbytes)
        return -EFBIG;

    if (S_ISDIR(inode->i_mode))
        return -EISDIR;

    if (mode & ~(FALLOC_FL_KEEP_SIZE |
                 FALLOC_FL_PUNCH_HOLE))
        return -EOPNOTSUPP;

    if ((mode & FALLOC_FL_PUNCH_HOLE) &&
        !(mode & FALLOC_FL_KEEP_SIZE))
        return -EOPNOTSUPP;

    inode_lock(inode);

    if (mode & FALLOC_FL_PUNCH_HOLE) {
        ret = vecfs_punch_hole(
            inode, offset, length);
        goto out;
    }

    ret = inode_newsize_ok(inode, end);
    if (ret)
        goto out;

    ret = vecfs_allocate_blocks(
        inode, offset, length);
    if (ret)
        goto out;

    if (!(mode & FALLOC_FL_KEEP_SIZE) &&
        end > i_size_read(inode)) {
        i_size_write(inode, end);
        inode_set_mtime_to_ts(
            inode,
            inode_set_ctime_current(inode));
        mark_inode_dirty(inode);
    }

out:
    inode_unlock(inode);
    return ret;
}

int vecfs_truncate_blocks(
    struct inode *inode,
    loff_t new_size)
{
    loff_t old_size;
    sector_t old_blocks;
    sector_t new_blocks;
    int ret;

    if (!inode)
        return -EINVAL;

    if (new_size < 0)
        return -EINVAL;

    old_size = i_size_read(inode);

    if (new_size >= old_size)
        return 0;

    /*
     * 隐藏 arena inode 没有 FILE vector 绑定；stale helper 会快速
     * 跳过它们。
     */
    ret = vecfs_ann_mark_content_stale_after(
        inode,
        new_size);
    if (ret)
        return ret;

    old_blocks = DIV_ROUND_UP_ULL(
        old_size,
        inode->i_sb->s_blocksize);

    new_blocks = DIV_ROUND_UP_ULL(
        new_size,
        inode->i_sb->s_blocksize);

    if (old_blocks > new_blocks) {
        ret = vecfs_free_block_range(
            inode,
            new_blocks,
            old_blocks - 1);
        if (ret)
            return ret;
    }

    truncate_setsize(
        inode,
        new_size);

    inode_set_mtime_to_ts(
        inode,
        inode_set_ctime_current(inode));

    mark_inode_dirty(inode);

    return 0;
}
