// SPDX-License-Identifier: GPL-2.0

#include <linux/buffer_head.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/random.h>
#include <linux/slab.h>

#include "../../include/vecfs.h"
#include "../../include/vecfs_Indirect.h"
#include "../../include/vecfs_ann_meta.h"
#include "../../include/vecfs_ann_store.h"
#include "internal.h"

#define VECFS_ANN_OBJECT_ALIGNMENT 64ULL

static u64 vecfs_ann_align(u64 value)
{
    return ALIGN(value, VECFS_ANN_OBJECT_ALIGNMENT);
}

int vecfs_ann_layout_init(
    struct vecfs_ann_object_layout *layout,
    const struct vecfs_ann_config *config)
{
    u64 offset;
    u64 upper_levels;

    if (!layout || !config)
        return -EINVAL;

    if (vecfs_ann_validate_config(config))
        return -EINVAL;

    memset(layout, 0, sizeof(*layout));

    upper_levels = config->max_levels - 1;

    offset = vecfs_ann_align(
        sizeof(struct vecfs_ann_object_disk));

    layout->level0_offset = offset;

    offset += (u64)config->m0 * sizeof(__le32);
    offset = vecfs_ann_align(offset);

    layout->vector_offset = offset;

    offset += config->vector_bytes;
    offset = vecfs_ann_align(offset);

    layout->upper_counts_offset = offset;

    offset += upper_levels * sizeof(__le16);
    offset = vecfs_ann_align(offset);

    layout->upper_links_offset = offset;

    offset += upper_levels * config->m *
              sizeof(__le32);
    offset = vecfs_ann_align(offset);

    if (offset != config->object_stride)
        return -EUCLEAN;

    layout->stride = config->object_stride;
    layout->vector_bytes = config->vector_bytes;

    layout->dimension = config->dimension;
    layout->element_type = config->element_type;
    layout->m = config->m;
    layout->m0 = config->m0;
    layout->max_levels = config->max_levels;

    return 0;
}

static int vecfs_ann_initialize_arena_inode(
    struct inode *inode)
{
    if (sb_rdonly(inode->i_sb))
        return -EROFS;

    inode_init_owner(
        VECFS_IDMAP_NOP,
        inode,
        NULL,
        S_IFREG | 0600);

    inode->i_generation = get_random_u32();
    inode->i_blocks = 0;

    i_size_write(inode, 0);
    set_nlink(inode, 1);

    simple_inode_init_ts(inode);

    inode->i_mapping->a_ops =
        &vecfs_address_space_ops;

    mark_inode_dirty(inode);
    return write_inode_now(inode, 1);
}

int vecfs_ann_get_arena_inode(
    struct super_block *sb,
    u32 inode_number,
    struct inode **result)
{
    struct inode *inode;
    int ret;

    if (!sb || !result)
        return -EINVAL;

    if (inode_number !=
            VECFS_ANN_ARENA_INO &&
        inode_number !=
            VECFS_ANN_REBUILD_ARENA_INO)
        return -EINVAL;

    inode = iget_locked(
        sb,
        inode_number);
    if (!inode)
        return -ENOMEM;

    if (VECFS_INODE_IS_NEW(inode)) {
        ret = vecfs_read_inode(inode);
        if (ret) {
            iget_failed(inode);
            return ret;
        }

        if (!inode->i_mode) {
            ret =
                vecfs_ann_initialize_arena_inode(
                    inode);
            if (ret) {
                iget_failed(inode);
                return ret;
            }
        }

        unlock_new_inode(inode);
    }

    if (!S_ISREG(inode->i_mode)) {
        iput(inode);
        return -EUCLEAN;
    }

    if (!inode->i_nlink) {
        if (sb_rdonly(sb)) {
            iput(inode);
            return -EUCLEAN;
        }

        set_nlink(inode, 1);
        mark_inode_dirty(inode);
    }

    *result = inode;
    return 0;
}


int vecfs_ann_arena_read_locked(
    struct vecfs_ann_store *store,
    loff_t position,
    void *buffer,
    size_t length)
{
    struct inode *inode;
    struct super_block *sb;
    size_t copied = 0;

    if (!store || !buffer)
        return -EINVAL;

    inode = store->arena_inode;
    if (!inode)
        return -ENODEV;
    sb = store->sb;

    if (position < 0)
        return -EINVAL;

    if ((u64)position + length >
        (u64)i_size_read(inode))
        return -ENODATA;

    while (copied < length) {
        struct buffer_head map = { 0 };
        struct buffer_head *bh;
        loff_t cur = position + copied;
        sector_t logical =
            cur >> sb->s_blocksize_bits;
        size_t offset =
            cur & (sb->s_blocksize - 1);
        size_t chunk =
            min_t(size_t,
                  length - copied,
                  sb->s_blocksize - offset);
        int ret;

        ret = vecfs_get_block(
            inode, logical, &map, 0);
        if (ret)
            return ret;

        if (!buffer_mapped(&map)) {
            memset((u8 *)buffer + copied, 0, chunk);
            copied += chunk;
            continue;
        }

        bh = sb_bread(sb, map.b_blocknr);
        if (!bh)
            return -EIO;

        memcpy(
            (u8 *)buffer + copied,
            bh->b_data + offset,
            chunk);

        brelse(bh);
        copied += chunk;
    }

    return 0;
}

int vecfs_ann_arena_write_locked(
    struct vecfs_ann_store *store,
    loff_t position,
    const void *buffer,
    size_t length,
    bool sync)
{
    struct inode *inode;
    struct super_block *sb;
    size_t copied = 0;
    loff_t end;

    if (!store || !buffer)
        return -EINVAL;

    if (sb_rdonly(store->sb))
        return -EROFS;

    if (position < 0)
        return -EINVAL;

    if ((u64)position > LLONG_MAX - length)
        return -EOVERFLOW;

    inode = store->arena_inode;
    if (!inode)
        return -ENODEV;
    sb = store->sb;
    end = position + length;

    while (copied < length) {
        struct buffer_head map = { 0 };
        struct buffer_head *bh;
        loff_t cur = position + copied;
        sector_t logical =
            cur >> sb->s_blocksize_bits;
        size_t offset =
            cur & (sb->s_blocksize - 1);
        size_t chunk =
            min_t(size_t,
                  length - copied,
                  sb->s_blocksize - offset);
        int ret;

        ret = vecfs_get_block(
            inode, logical, &map, 1);
        if (ret)
            return ret;

        if (!buffer_mapped(&map))
            return -EIO;

        if (buffer_new(&map)) {
            bh = sb_getblk(sb, map.b_blocknr);
            if (!bh)
                return -ENOMEM;

            lock_buffer(bh);
            memset(bh->b_data, 0, sb->s_blocksize);
            set_buffer_uptodate(bh);
            clear_buffer_new(bh);
            unlock_buffer(bh);
        } else {
            bh = sb_bread(sb, map.b_blocknr);
            if (!bh)
                return -EIO;
        }

        lock_buffer(bh);

        memcpy(
            bh->b_data + offset,
            (const u8 *)buffer + copied,
            chunk);

        set_buffer_uptodate(bh);
        unlock_buffer(bh);

        mark_buffer_dirty_inode(bh, inode);

        if (sync) {
            sync_dirty_buffer(bh);

            if (buffer_write_io_error(bh)) {
                brelse(bh);
                return -EIO;
            }
        }

        brelse(bh);
        copied += chunk;
    }

    if (end > i_size_read(inode)) {
        i_size_write(inode, end);
        mark_inode_dirty(inode);
    }

    return 0;
}

int vecfs_ann_store_mount(struct super_block *sb)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_store *store;
    struct inode *arena_inode;
    u32 arena_ino;
    int ret;

    if (!sbi)
        return -EINVAL;

    if (sbi->s_ann_store)
        return -EBUSY;

    arena_ino = sbi->s_ann.active_arena_ino;

    if (!arena_ino) {
        if (sb_rdonly(sb))
            return -EROFS;

        arena_ino = VECFS_ANN_ARENA_INO;
    }

    if (arena_ino != VECFS_ANN_ARENA_INO &&
        arena_ino != VECFS_ANN_REBUILD_ARENA_INO)
        return -EUCLEAN;

    ret = vecfs_ann_get_arena_inode(
        sb, arena_ino, &arena_inode);
    if (ret)
        return ret;

    store = kzalloc(sizeof(*store), GFP_KERNEL);
    if (!store) {
        iput(arena_inode);
        return -ENOMEM;
    }

    store->sb = sb;
    store->arena_inode = arena_inode;

    mutex_init(&store->allocation_lock);
    init_rwsem(&store->io_lock);

    ret = vecfs_ann_layout_init(
        &store->layout,
        &sbi->s_ann.config);
    if (ret)
        goto fail_store;

    sbi->s_ann_store = store;

    if (!sbi->s_ann.active_arena_ino) {
        mutex_lock(&sbi->s_ann.lock);

        sbi->s_ann.active_arena_ino = arena_ino;
        sbi->s_ann.dirty = true;

        mutex_unlock(&sbi->s_ann.lock);

        ret = vecfs_ann_meta_commit(sb, true);
        if (ret)
            goto fail_attached;
    }

    ret = vecfs_ann_store_scan(sb);
    if (ret)
        goto fail_attached;

    return 0;

fail_attached:
    sbi->s_ann_store = NULL;
fail_store:
    mutex_destroy(&store->allocation_lock);
    iput(arena_inode);
    kfree(store);
    return ret;
}

void vecfs_ann_store_unmount(
    struct super_block *sb)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_store *store;

    if (!sb)
        return;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return;

    store = sbi->s_ann_store;
    if (!store)
        return;

    sbi->s_ann_store = NULL;

    vecfs_ann_store_release_instance(
        store,
        true);
}


void vecfs_ann_store_release_instance(
    struct vecfs_ann_store *store,
    bool write_inode)
{
    if (!store)
        return;

    if (store->arena_inode) {
        if (write_inode)
            write_inode_now(
                store->arena_inode,
                1);

        iput(store->arena_inode);
        store->arena_inode = NULL;
    }

    mutex_destroy(
        &store->allocation_lock);

    kfree(store);
}
