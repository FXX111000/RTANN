// SPDX-License-Identifier: GPL-2.0

#include <linux/buffer_head.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/list.h>
#include <linux/namei.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/xarray.h>

#include "../include/vecfs.h"
#include "../include/vecfs_Indirect.h"
#include "../include/vecfs_path.h"

struct vecfs_path_scan_item {
    struct list_head list;
    u64 ino;
};

static bool vecfs_path_name_is_dot(
    const char *name,
    u16 length)
{
    return length == 1 &&
           name[0] == '.';
}

static bool vecfs_path_name_is_dotdot(
    const char *name,
    u16 length)
{
    return length == 2 &&
           name[0] == '.' &&
           name[1] == '.';
}

static void vecfs_path_entry_free(
    struct vecfs_path_entry *entry)
{
    kfree(entry);
}

static struct vecfs_path_entry *
vecfs_path_entry_allocate(
    u64 ino,
    u32 inode_generation,
    u64 parent_ino,
    const char *name,
    u16 name_length,
    bool directory)
{
    struct vecfs_path_entry *entry;
    size_t allocation_size;

    if (!ino || !parent_ino)
        return ERR_PTR(-EINVAL);

    if (!name || !name_length ||
        name_length > VECFS_MAX_NAME_LEN)
        return ERR_PTR(-EINVAL);

    if (check_add_overflow(
            sizeof(*entry),
            (size_t)name_length + 1,
            &allocation_size))
        return ERR_PTR(-EOVERFLOW);

    entry = kzalloc(
        allocation_size,
        GFP_NOFS);
    if (!entry)
        return ERR_PTR(-ENOMEM);

    entry->ino = ino;
    entry->parent_ino = parent_ino;
    entry->inode_generation =
        inode_generation;
    entry->name_length =
        name_length;

    if (directory)
        entry->flags |=
            VECFS_PATH_ENTRY_F_DIRECTORY;

    memcpy(
        entry->name,
        name,
        name_length);

    entry->name[name_length] = '\0';

    return entry;
}

static struct inode *vecfs_path_iget(
    struct super_block *sb,
    u64 ino)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct inode *inode;
    int ret;

    if (!ino ||
        ino > sbi->s_inodes_count)
        return ERR_PTR(-ENOENT);

    inode = iget_locked(
        sb,
        ino);
    if (!inode)
        return ERR_PTR(-ENOMEM);

    if (VECFS_INODE_IS_NEW(inode)) {
        ret = vecfs_read_inode(inode);
        if (ret) {
            iget_failed(inode);
            return ERR_PTR(ret);
        }

        unlock_new_inode(inode);
    }

    if (!inode->i_mode) {
        iput(inode);
        return ERR_PTR(-ENOENT);
    }

    return inode;
}

static int vecfs_path_cache_store_locked(
    struct vecfs_path_cache *cache,
    struct vecfs_path_entry *entry)
{
    struct vecfs_path_entry *old;
    void *stored;

    lockdep_assert_held(
        &cache->lock);

    old = xa_load(
        &cache->entries,
        entry->ino);

    stored = xa_store(
        &cache->entries,
        entry->ino,
        entry,
        GFP_NOFS);

    if (xa_is_err(stored))
        return xa_err(stored);

    if (!old)
        cache->entry_count++;
    else
        vecfs_path_entry_free(old);

    return 0;
}

static int vecfs_path_cache_set_raw(
    struct super_block *sb,
    u64 ino,
    u32 inode_generation,
    u64 parent_ino,
    const char *name,
    u16 name_length,
    bool directory)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_path_cache *cache;
    struct vecfs_path_entry *entry;
    int ret;

    if (!sbi)
        return -EINVAL;

    cache = &sbi->s_path_cache;

    if (!cache->initialized)
        return -ENODEV;

    entry = vecfs_path_entry_allocate(
        ino,
        inode_generation,
        parent_ino,
        name,
        name_length,
        directory);
    if (IS_ERR(entry))
        return PTR_ERR(entry);

    mutex_lock(&cache->lock);

    ret = vecfs_path_cache_store_locked(
        cache,
        entry);

    mutex_unlock(&cache->lock);

    if (ret)
        vecfs_path_entry_free(entry);

    return ret;
}

int vecfs_path_cache_set(
    struct super_block *sb,
    u64 ino,
    u32 inode_generation,
    u64 parent_ino,
    const struct qstr *name,
    bool directory)
{
    if (!sb || !name)
        return -EINVAL;

    if (!name->len ||
        name->len > VECFS_MAX_NAME_LEN)
        return -EINVAL;

    return vecfs_path_cache_set_raw(
        sb,
        ino,
        inode_generation,
        parent_ino,
        name->name,
        name->len,
        directory);
}

int vecfs_path_cache_remove(
    struct super_block *sb,
    u64 ino,
    u32 inode_generation)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_path_cache *cache;
    struct vecfs_path_entry *entry;

    if (!sb || !ino)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return -EINVAL;

    cache = &sbi->s_path_cache;

    if (!cache->initialized)
        return -ENODEV;

    mutex_lock(&cache->lock);

    entry = xa_load(
        &cache->entries,
        ino);

    if (!entry) {
        mutex_unlock(&cache->lock);
        return -ENOENT;
    }

    if (inode_generation &&
        entry->inode_generation !=
            inode_generation) {
        mutex_unlock(&cache->lock);
        return -ESTALE;
    }

    entry = xa_erase(
        &cache->entries,
        ino);

    if (entry && cache->entry_count)
        cache->entry_count--;

    mutex_unlock(&cache->lock);

    vecfs_path_entry_free(entry);
    return 0;
}

int vecfs_path_cache_rename(
    struct super_block *sb,
    u64 ino,
    u32 inode_generation,
    u64 new_parent_ino,
    const struct qstr *new_name,
    bool directory)
{
    /*
     * entry 为不可变对象。rename 通过分配新 entry 后原子替换，
     * 避免路径读取观察到部分更新的名称。
     */
    return vecfs_path_cache_set(
        sb,
        ino,
        inode_generation,
        new_parent_ino,
        new_name,
        directory);
}

static void vecfs_path_cache_clear_locked(
    struct vecfs_path_cache *cache)
{
    unsigned long index;
    struct vecfs_path_entry *entry;

    lockdep_assert_held(
        &cache->lock);

    xa_for_each(
        &cache->entries,
        index,
        entry) {
        xa_erase(
            &cache->entries,
            index);

        vecfs_path_entry_free(entry);
    }

    cache->entry_count = 0;
    cache->complete = false;
}

int vecfs_path_cache_init(
    struct super_block *sb)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_path_cache *cache;

    if (!sb)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return -EINVAL;

    cache = &sbi->s_path_cache;

    memset(
        cache,
        0,
        sizeof(*cache));

    mutex_init(
        &cache->lock);

    xa_init(
        &cache->entries);

    cache->initialized = true;
    return 0;
}

void vecfs_path_cache_destroy(
    struct super_block *sb)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_path_cache *cache;

    if (!sb)
        return;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return;

    cache = &sbi->s_path_cache;

    if (!cache->initialized)
        return;

    mutex_lock(&cache->lock);

    vecfs_path_cache_clear_locked(
        cache);

    mutex_unlock(&cache->lock);

    xa_destroy(
        &cache->entries);

    mutex_destroy(
        &cache->lock);

    memset(
        cache,
        0,
        sizeof(*cache));
}

static int vecfs_path_scan_enqueue(
    struct list_head *queue,
    u64 ino)
{
    struct vecfs_path_scan_item *item;

    item = kmalloc(
        sizeof(*item),
        GFP_NOFS);
    if (!item)
        return -ENOMEM;

    INIT_LIST_HEAD(
        &item->list);

    item->ino = ino;

    list_add_tail(
        &item->list,
        queue);

    return 0;
}

static int vecfs_path_scan_directory(
    struct super_block *sb,
    struct inode *directory,
    struct list_head *queue,
    unsigned long *visited)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    sector_t blocks;
    sector_t logical;
    int ret = 0;

    u64 size = i_size_read(directory);
    u64 max_file_bytes =
        (u64)VECFS_SB(sb)->s_blocks_count *
        sb->s_blocksize;

    if (size > max_file_bytes)
        return -EUCLEAN;

    if (size % sb->s_blocksize)
        return -EUCLEAN;

    blocks = DIV_ROUND_UP_ULL(
        i_size_read(directory),
        sb->s_blocksize);

    for (logical = 0;
         logical < blocks;
         logical++) {
        struct buffer_head map = { 0 };
        struct buffer_head *bh;
        unsigned int offset = 0;

        ret = vecfs_get_block(
            directory,
            logical,
            &map,
            0);
        if (ret)
            return ret;

        if (!buffer_mapped(&map))
            continue;

        bh = sb_bread(
            sb,
            map.b_blocknr);
        if (!bh)
            return -EIO;

        while (offset + 8 <=
               sb->s_blocksize) {
            struct vecfs_dir_entry *disk_entry;
            u16 record_length;
            u16 name_length;
            u32 child_ino;
            struct inode *child;

            disk_entry =
                (struct vecfs_dir_entry *)
                (bh->b_data + offset);

            record_length =
                le16_to_cpu(
                    disk_entry->rec_len);

            name_length =
                le16_to_cpu(
                    disk_entry->name_len);

            child_ino =
                le32_to_cpu(
                    disk_entry->inode);

            if (record_length < 8 ||
                offset + record_length >
                    sb->s_blocksize) {
                ret = -EUCLEAN;
                break;
            }

            offset += record_length;

            if (!child_ino)
                continue;

            if (!name_length ||
                name_length >
                    VECFS_MAX_NAME_LEN) {
                ret = -EUCLEAN;
                break;
            }

            if (vecfs_path_name_is_dot(
                    disk_entry->name,
                    name_length) ||
                vecfs_path_name_is_dotdot(
                    disk_entry->name,
                    name_length))
                continue;

            if (child_ino >
                sbi->s_inodes_count) {
                ret = -EUCLEAN;
                break;
            }

            child = vecfs_path_iget(
                sb,
                child_ino);
            if (IS_ERR(child)) {
                ret = PTR_ERR(child);

                /*
                 * 挂载恢复允许跳过目录项指向的空 inode，传统目录
                 * validate/fsck 后续负责清理该目录项。
                 */
                if (ret == -ENOENT) {
                    ret = 0;
                    continue;
                }

                break;
            }

            ret = vecfs_path_cache_set_raw(
                sb,
                child_ino,
                child->i_generation,
                directory->i_ino,
                disk_entry->name,
                name_length,
                S_ISDIR(child->i_mode));
            if (ret) {
                iput(child);
                break;
            }

            if (S_ISDIR(child->i_mode) &&
                !test_and_set_bit(
                    child_ino - 1,
                    visited)) {
                ret = vecfs_path_scan_enqueue(
                    queue,
                    child_ino);

                if (ret) {
                    iput(child);
                    break;
                }
            }

            iput(child);
        }

        brelse(bh);

        if (ret)
            return ret;

        cond_resched();
    }

    return 0;
}

int vecfs_path_cache_rebuild(
    struct super_block *sb)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_path_cache *cache;
    LIST_HEAD(queue);
    unsigned long *visited;
    struct vecfs_path_scan_item *item;
    struct vecfs_path_scan_item *temporary;
    int ret;

    if (!sb)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return -EINVAL;

    cache = &sbi->s_path_cache;

    if (!cache->initialized)
        return -ENODEV;

    visited = bitmap_zalloc(
        sbi->s_inodes_count,
        GFP_KERNEL);
    if (!visited)
        return -ENOMEM;

    mutex_lock(&cache->lock);

    vecfs_path_cache_clear_locked(
        cache);

    mutex_unlock(&cache->lock);

    /*
     * 根目录也放入缓存，使任意 parent 链最终都能显式终止。
     */
    ret = vecfs_path_cache_set_raw(
        sb,
        sbi->s_root_ino,
        0,
        sbi->s_root_ino,
        ".",
        1,
        true);
    if (ret)
        goto out_visited;

    __set_bit(
        sbi->s_root_ino - 1,
        visited);

    ret = vecfs_path_scan_enqueue(
        &queue,
        sbi->s_root_ino);
    if (ret)
        goto out_visited;

    while (!list_empty(&queue)) {
        struct inode *directory;

        item = list_first_entry(
            &queue,
            struct vecfs_path_scan_item,
            list);

        list_del(
            &item->list);

        directory = vecfs_path_iget(
            sb,
            item->ino);

        kfree(item);

        if (IS_ERR(directory)) {
            ret = PTR_ERR(directory);
            goto out_queue;
        }

        if (!S_ISDIR(directory->i_mode)) {
            iput(directory);
            ret = -EUCLEAN;
            goto out_queue;
        }

        ret = vecfs_path_scan_directory(
            sb,
            directory,
            &queue,
            visited);

        iput(directory);

        if (ret)
            goto out_queue;
    }

    mutex_lock(&cache->lock);
    cache->complete = true;
    mutex_unlock(&cache->lock);

    ret = 0;
    goto out_visited;

out_queue:
    list_for_each_entry_safe(
        item,
        temporary,
        &queue,
        list) {
        list_del(
            &item->list);

        kfree(item);
    }

out_visited:
    bitmap_free(visited);
    return ret;
}
