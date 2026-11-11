// SPDX-License-Identifier: GPL-2.0

#include <linux/bitmap.h>
#include <linux/blkdev.h>
#include <linux/buffer_head.h>
#include <linux/crc32.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/mpage.h>
#include <linux/mount.h>
#include <linux/namei.h>
#include <linux/pagemap.h>
#include <linux/poll.h>
#include <linux/random.h>
#include <linux/slab.h>
#include <linux/timekeeping.h>
#include <linux/uaccess.h>
#include <linux/writeback.h>

#include "vecfs.h"
#include "vecfs_Indirect.h"
#include "include/vecfs_ann_meta.h"
#include "include/vecfs_ann_indirect.h"
#include "include/vecfs_ann_search.h"
#include "include/vecfs_ann_validate.h"
#include "include/vecfs_disk.h"
#include "include/vecfs_tail.h"
#include "../include/vecfs_interface.h"
#include "../include/vecfs_ann_stale.h"
#include "../include/vecfs_embed_upcall.h"
#include <linux/mm.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("fxx");
MODULE_DESCRIPTION("VecFS V2 vector file system");

#define VECFS_VEC_TAIL_INPROGRESS  (1U << 2)
#define VECFS_VEC_TAIL_DONE        (1U << 3)

struct kmem_cache *vecfs_inode_cachep;

static void vecfs_path_cache_set_best_effort(
    struct inode *parent,
    struct inode *inode,
    const struct qstr *name);
static void vecfs_path_cache_remove_best_effort(
    struct inode *inode);
static void vecfs_init_ann_roots(
    struct vecfs_inode *vi);
static int vecfs_sync_metadata(
    struct super_block *sb);

static u32 vecfs_super_crc(
    const struct vecfs_super_disk_v2 *disk)
{
    struct vecfs_super_disk_v2 *copy;
    u32 crc;

    copy = kmemdup(disk, sizeof(*copy), GFP_KERNEL);
    if (!copy)
        return 0;

    copy->checksum = 0;
    crc = crc32_le(
               ~0U,
               (const u8 *)copy,
               sizeof(*copy)) ^ ~0U;

    kfree(copy);
    return crc;
}

static int vecfs_validate_super(
    const struct vecfs_super_disk_v2 *disk)
{
    u64 blocks;
    u64 inodes;
    u64 data_start;
    u64 inode_table;
    u64 ann_a;
    u64 ann_b;
    u32 stored_crc;

    if (le32_to_cpu(disk->magic) !=
        VECFS_SUPER_MAGIC)
        return -EINVAL;

    if (le16_to_cpu(disk->major) !=
        VECFS_SUPER_MAJOR)
        return -EOPNOTSUPP;

    /* 新格式（minor 1 引入 superblock 嵌入配置）；拒绝更新的未知格式。 */
    if (le16_to_cpu(disk->minor) > VECFS_SUPER_MINOR)
        return -EOPNOTSUPP;

    if (le32_to_cpu(disk->header_size) !=
        sizeof(*disk))
        return -EUCLEAN;

    if (le32_to_cpu(disk->block_size) !=
        VECFS_DISK_BLOCK_SIZE)
        return -EINVAL;

    if (le32_to_cpu(disk->inode_size) !=
        VECFS_DISK_INODE_SIZE)
        return -EOPNOTSUPP;

    stored_crc = le32_to_cpu(disk->checksum);
    if (stored_crc != vecfs_super_crc(disk))
        return -EBADMSG;

    blocks = le64_to_cpu(disk->blocks_count);
    inodes = le64_to_cpu(disk->inodes_count);
    data_start = le64_to_cpu(disk->data_start);
    inode_table = le64_to_cpu(disk->inode_table);
    ann_a = le64_to_cpu(disk->ann_meta_a);
    ann_b = le64_to_cpu(disk->ann_meta_b);

    if (!blocks || blocks > U32_MAX)
        return -EFBIG;

    if (!inodes || inodes > U32_MAX)
        return -EFBIG;

    if (!data_start || data_start >= blocks)
        return -EUCLEAN;

    if (!inode_table || inode_table >= data_start)
        return -EUCLEAN;

    if (!ann_a || !ann_b || ann_a == ann_b)
        return -EUCLEAN;

    if (ann_a >= data_start || ann_b >= data_start)
        return -EUCLEAN;

    if (le32_to_cpu(disk->root_ino) !=
        VECFS_ROOT_INO)
        return -EUCLEAN;

    return 0;
}

int vecfs_setattr(
    VECFS_IDMAP idmap,
    struct dentry *dentry,
    struct iattr *attr)
{
    struct inode *inode;
    loff_t old_size;
    loff_t new_size;
    int ret;

    if (!dentry || !attr)
        return -EINVAL;

    inode = d_inode(dentry);
    if (!inode)
        return -EINVAL;

    ret = setattr_prepare(
        idmap,
        dentry,
        attr);
    if (ret)
        return ret;

    old_size = i_size_read(inode);

    if (attr->ia_valid & ATTR_SIZE) {
        new_size = attr->ia_size;

        if (new_size < 0)
            return -EINVAL;

        if (new_size < old_size) {
            ret = vecfs_truncate_blocks(
                inode,
                new_size);
            if (ret)
                return ret;
        } else if (new_size > old_size) {
            /*
             * 文件扩展采用 sparse semantics，不立即分配数据块。
             */
            truncate_setsize(
                inode,
                new_size);
        }

        if (new_size != old_size)
            vecfs_shield_version_bump(inode);
    }

    setattr_copy(
        idmap,
        inode,
        attr);

    mark_inode_dirty(inode);

    return 0;
}


unsigned long *load_bitmap(
    struct super_block *sb,
    sector_t start_block,
    unsigned int total_entries)
{
    unsigned int blocks;
    size_t bytes;
    unsigned long *bitmap;
    unsigned int i;

    blocks = DIV_ROUND_UP(
        total_entries,
        sb->s_blocksize * 8);

    bytes = (size_t)blocks * sb->s_blocksize;

    bitmap = kvzalloc(bytes, GFP_KERNEL);
    if (!bitmap)
        return NULL;

    for (i = 0; i < blocks; i++) {
        struct buffer_head *bh;

        bh = sb_bread(sb, start_block + i);
        if (!bh) {
            kvfree(bitmap);
            return NULL;
        }

        memcpy(
            (u8 *)bitmap +
                (size_t)i * sb->s_blocksize,
            bh->b_data,
            sb->s_blocksize);

        brelse(bh);
    }

    return bitmap;
}

static int write_bitmap(
    struct super_block *sb,
    sector_t start_block,
    unsigned int total_entries,
    unsigned long *bitmap)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    unsigned int blocks;
    size_t total_bytes;
    unsigned int i;

    if (!sbi || !bitmap)
        return -EINVAL;

    blocks = DIV_ROUND_UP(
        total_entries,
        sb->s_blocksize * 8);

    if (start_block >= sbi->s_blocks_count ||
        blocks > sbi->s_blocks_count - start_block)
        return -EOVERFLOW;

    total_bytes = BITS_TO_LONGS(total_entries) *
                  sizeof(unsigned long);

    for (i = 0; i < blocks; i++) {
        struct buffer_head *bh;
        size_t offset =
            (size_t)i * sb->s_blocksize;
        size_t copy = 0;

        if (offset < total_bytes)
            copy = min_t(
                size_t,
                total_bytes - offset,
                sb->s_blocksize);

        bh = sb_getblk(sb, start_block + i);
        if (!bh)
            return -ENOMEM;

        lock_buffer(bh);

        memset(bh->b_data, 0, sb->s_blocksize);
        if (copy)
            memcpy(
                bh->b_data,
                (u8 *)bitmap + offset,
                copy);

        set_buffer_uptodate(bh);
        unlock_buffer(bh);

        mark_buffer_dirty(bh);
        sync_dirty_buffer(bh);

        if (buffer_write_io_error(bh)) {
            brelse(bh);
            return -EIO;
        }

        brelse(bh);
    }

    return 0;
}

static void vecfs_inode_init_once(void *object)
{
    struct vecfs_inode *vi = object;

    inode_init_once(&vi->vfs_inode);
}


struct inode *vecfs_alloc_inode(
    struct super_block *sb)
{
    struct vecfs_inode *vi;

    vi = kmem_cache_alloc(
        vecfs_inode_cachep,
        GFP_KERNEL);
    if (!vi)
        return NULL;

    memset(vi->i_block, 0, sizeof(vi->i_block));
    vecfs_init_ann_roots(vi);

    atomic_set(&vi->open_count, 0);
    atomic64_set(&vi->access_count, 0);

    vi->i_state_flags = 0;
    vi->vec_flags = 0;
    vi->shield_version = 0;
    vecfs_shield_file_init(vi);

    init_rwsem(&vi->meta_lock);
    spin_lock_init(&vi->i_raw_lock);
    init_rwsem(&vi->xattr_sem);
    return &vi->vfs_inode;
}


void vecfs_destroy_inode(struct inode *inode)
{
    kmem_cache_free(
        vecfs_inode_cachep,
        VECFS_I(inode));
}

sector_t vecfs_inode_to_block(
    struct super_block *sb,
    unsigned long ino)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    unsigned int inodes_per_block;
    unsigned long index;
    sector_t block;

    if (!sbi || ino < 1 ||
        ino > sbi->s_inodes_count)
        return 0;

    inodes_per_block =
        sb->s_blocksize / VECFS_DISK_INODE_SIZE;

    index = ino - 1;
    block = sbi->s_inode_table +
            index / inodes_per_block;

    if (block >= sbi->s_data_start)
        return 0;

    return block;
}

unsigned int vecfs_inode_offset(
    struct super_block *sb,
    unsigned long ino)
{
    unsigned int inodes_per_block =
        sb->s_blocksize / VECFS_DISK_INODE_SIZE;

    return ((ino - 1) % inodes_per_block) *
           VECFS_DISK_INODE_SIZE;
}

int vecfs_write_inode(
    struct inode *inode,
    struct writeback_control *wbc)
{
    struct vecfs_inode *vi = VECFS_I(inode);
    struct vecfs_inode_disk_v2 *disk;
    struct buffer_head *bh;
    sector_t block;
    u64 size;
    int i;

    block = vecfs_inode_to_block(
        inode->i_sb, inode->i_ino);
    if (!block)
        return -EUCLEAN;

    bh = sb_bread(inode->i_sb, block);
    if (!bh)
        return -EIO;

    disk = (struct vecfs_inode_disk_v2 *)
        (bh->b_data +
         vecfs_inode_offset(
             inode->i_sb, inode->i_ino));

    down_read(&vi->meta_lock);

    size = i_size_read(inode);

    disk->mode = cpu_to_le16(inode->i_mode);
    disk->uid = cpu_to_le16(i_uid_read(inode));
    disk->gid = cpu_to_le16(i_gid_read(inode));

    disk->size_lo = cpu_to_le32(size);
    disk->size_high = cpu_to_le32(size >> 32);

    disk->atime =
        cpu_to_le32(inode_get_atime_sec(inode));
    disk->ctime =
        cpu_to_le32(inode_get_ctime_sec(inode));
    disk->mtime =
        cpu_to_le32(inode_get_mtime_sec(inode));

    disk->links_count =
        cpu_to_le16(inode->i_nlink);
    disk->blocks_lo =
        cpu_to_le32(inode->i_blocks);
    disk->flags =
        cpu_to_le32(vi->vec_flags);
    disk->generation =
        cpu_to_le32(inode->i_generation);

    for (i = 0; i < VECFS_N_BLOCK_PTRS; i++) {
        disk->i_block[i] =
            cpu_to_le32(vi->i_block[i]);
        disk->i_index[i] =
            cpu_to_le32(vi->i_index[i]);
    }

    up_read(&vi->meta_lock);

    mark_buffer_dirty(bh);

    if (wbc && wbc->sync_mode == WB_SYNC_ALL)
        sync_dirty_buffer(bh);

    brelse(bh);
    return 0;
}

static const struct inode_operations vecfs_symlink_inode_ops = {
    .get_link = page_get_link,
    .getattr = simple_getattr,
    .setattr = simple_setattr,
};

int vecfs_read_inode(struct inode *inode)
{
    struct vecfs_inode *vi = VECFS_I(inode);
    struct vecfs_inode_disk_v2 *disk;
    struct buffer_head *bh;
    sector_t block;
    u64 size;
    int i;

    block = vecfs_inode_to_block(
        inode->i_sb, inode->i_ino);
    if (!block)
        return -EUCLEAN;

    bh = sb_bread(inode->i_sb, block);
    if (!bh)
        return -EIO;

    disk = (struct vecfs_inode_disk_v2 *)
        (bh->b_data +
         vecfs_inode_offset(
             inode->i_sb, inode->i_ino));

    down_write(&vi->meta_lock);

    inode->i_mode = le16_to_cpu(disk->mode);
    i_uid_write(inode, le16_to_cpu(disk->uid));
    i_gid_write(inode, le16_to_cpu(disk->gid));

    size = le32_to_cpu(disk->size_lo);
    size |= (u64)le32_to_cpu(disk->size_high) << 32;
    i_size_write(inode, size);

    inode->i_blocks =
        le32_to_cpu(disk->blocks_lo);

    inode_set_atime(
        inode, le32_to_cpu(disk->atime), 0);
    inode_set_ctime(
        inode, le32_to_cpu(disk->ctime), 0);
    inode_set_mtime(
        inode, le32_to_cpu(disk->mtime), 0);

    set_nlink(
        inode, le16_to_cpu(disk->links_count));

    inode->i_generation =
        le32_to_cpu(disk->generation);

    for (i = 0; i < VECFS_N_BLOCK_PTRS; i++) {
        vi->i_block[i] =
            le32_to_cpu(disk->i_block[i]);

        vi->i_index[i] =
            le32_to_cpu(disk->i_index[i]);

        /*
        * V2 阶段一格式化时 inode table 为零。
        * 旧的零值需要解释为“没有 ANN 映射”。
        *
        * node ID 从 1 开始，因此 node ID 0 不被使用。
        */
        if (vi->i_index[i] == 0)
            vi->i_index[i] =
                VECFS_ANN_EMPTY_NODE_ID;
    }


    vi->vec_flags =
        (u8)le32_to_cpu(disk->flags);

    if (S_ISREG(inode->i_mode)) {
        inode->i_op = &vecfs_inode_ops;
        inode->i_fop = &vecfs_file_ops;
    } else if (S_ISDIR(inode->i_mode)) {
        inode->i_op = &vecfs_inode_dir_ops;
        inode->i_fop = &vecfs_file_dir_ops;
    } else if (S_ISLNK(inode->i_mode)) {
        inode->i_op = &vecfs_symlink_inode_ops;
        inode->i_fop = NULL;
    }

    inode->i_mapping->a_ops =
        &vecfs_address_space_ops;

    up_write(&vi->meta_lock);
    brelse(bh);

    return 0;
}

ino_t get_new_ino(struct super_block *sb)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    unsigned long bit;

    mutex_lock(&sbi->s_lock);

    if (!sbi->s_free_inodes) {
        mutex_unlock(&sbi->s_lock);
        return 0;
    }

    bit = find_next_zero_bit(
        sbi->s_imap,
        sbi->s_inodes_count,
        sbi->s_next_ino);

    if (bit >= sbi->s_inodes_count) {
        bit = find_next_zero_bit(
            sbi->s_imap,
            sbi->s_inodes_count,
            VECFS_FIRST_USER_INO);
    }

    if (bit >= sbi->s_inodes_count) {
        mutex_unlock(&sbi->s_lock);
        return 0;
    }

    __set_bit(bit, sbi->s_imap);
    sbi->s_free_inodes--;

    sbi->s_next_ino = bit + 1;
    if (sbi->s_next_ino >= sbi->s_inodes_count)
        sbi->s_next_ino = VECFS_FIRST_USER_INO;

    mutex_unlock(&sbi->s_lock);

    return bit + 1;
}

void vecfs_free_inode(
    struct super_block *sb,
    struct inode *inode)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    unsigned long bit;

    if (!sbi || !inode || !inode->i_ino)
        return;

    /*
     * inode 1..VECFS_FIRST_USER_INO 是格式化时永久保留的。
     * 包括 root、ANN arena 和保留扩展 inode。
     */
    if (inode->i_ino <= VECFS_FIRST_USER_INO)
        return;

    bit = inode->i_ino - 1;
    if (bit >= sbi->s_inodes_count)
        return;

    mutex_lock(&sbi->s_lock);

    if (test_bit(bit, sbi->s_imap)) {
        __clear_bit(bit, sbi->s_imap);
        sbi->s_free_inodes++;

        if (bit < sbi->s_next_ino)
            sbi->s_next_ino = bit;
    }

    mutex_unlock(&sbi->s_lock);
}


struct inode *vecfs_create_inode(
    VECFS_IDMAP idmap,
    struct super_block *sb,
    const struct inode *dir,
    umode_t mode,
    const struct qstr *qname)
{
    struct inode *inode;
    ino_t ino;
    int ret;

    inode = new_inode(sb);
    if (!inode)
        return ERR_PTR(-ENOMEM);

    /*
     * new_inode 路径不会经过 vecfs_read_inode，因此需要在此
     * 显式初始化锁。内核 alloc_inode() 在 vecfs_alloc_inode
     * 返回之后会调用 inode_init_always()，因此构造函数中完成的
     * 锁初始化可能已被覆盖。
     */
    init_rwsem(&VECFS_I(inode)->meta_lock);
    spin_lock_init(&VECFS_I(inode)->i_raw_lock);
    init_rwsem(&VECFS_I(inode)->xattr_sem);

    ino = get_new_ino(sb);
    if (!ino) {
        iput(inode);
        return ERR_PTR(-ENOSPC);
    }

    inode->i_ino = ino;
    inode_init_owner(idmap, inode, dir, mode);
    inode->i_generation = get_random_u32();
    inode->i_blocks = 0;
    i_size_write(inode, 0);
    vecfs_init_ann_roots(VECFS_I(inode));

    simple_inode_init_ts(inode);

    inode->i_mapping->a_ops =
        &vecfs_address_space_ops;

    if (S_ISDIR(mode)) {
        inode->i_op = &vecfs_inode_dir_ops;
        inode->i_fop = &vecfs_file_dir_ops;
    } else if (S_ISREG(mode)) {
        inode->i_op = &vecfs_inode_ops;
        inode->i_fop = &vecfs_file_ops;
    } else if (S_ISLNK(mode)) {
        inode->i_op = &vecfs_symlink_inode_ops;
    }

    ret = insert_inode_locked(inode);
    if (ret) {
        vecfs_free_inode(sb, inode);
        iput(inode);
        return ERR_PTR(ret);
    }

    return inode;
}


int vecfs_find_entry(
    struct inode *dir,
    const struct qstr *name,
    ino_t *ino)
{
    struct super_block *sb = dir->i_sb;
    sector_t blocks;
    sector_t logical;

    blocks = DIV_ROUND_UP_ULL(
        i_size_read(dir), sb->s_blocksize);

    for (logical = 0; logical < blocks; logical++) {
        struct buffer_head map = { 0 };
        struct buffer_head *bh;
        unsigned int offset = 0;
        int ret;

        ret = vecfs_get_block(
            dir, logical, &map, 0);
        if (ret)
            return ret;

        if (!buffer_mapped(&map))
            continue;

        bh = sb_bread(sb, map.b_blocknr);
        if (!bh)
            return -EIO;

        while (offset + 8 <= sb->s_blocksize) {
            struct vecfs_dir_entry *entry;
            u16 record_length;
            u16 name_length;

            entry = (struct vecfs_dir_entry *)
                (bh->b_data + offset);

            record_length =
                le16_to_cpu(entry->rec_len);
            name_length =
                le16_to_cpu(entry->name_len);

            if (record_length < 8 ||
                offset + record_length >
                    sb->s_blocksize)
                break;

            if (le32_to_cpu(entry->inode) &&
                name_length == name->len &&
                !memcmp(
                    entry->name,
                    name->name,
                    name->len)) {
                *ino = le32_to_cpu(entry->inode);
                brelse(bh);
                return 0;
            }

            offset += record_length;
        }

        brelse(bh);
    }

    return -ENOENT;
}

struct dentry *vecfs_lookup(
    struct inode *dir,
    struct dentry *dentry,
    unsigned int flags)
{
    struct inode *inode;
    ino_t ino;
    int ret;

    if (dentry->d_name.len > VECFS_MAX_NAME_LEN)
        return ERR_PTR(-ENAMETOOLONG);

    ret = vecfs_find_entry(
        dir, &dentry->d_name, &ino);
    if (ret == -ENOENT)
        return NULL;
    if (ret)
        return ERR_PTR(ret);

    inode = iget_locked(dir->i_sb, ino);
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

    return d_splice_alias(inode, dentry);
}

static int vecfs_extend_directory(struct inode *dir)
{
    struct buffer_head map = { 0 };
    struct buffer_head *bh;
    sector_t logical;
    int ret;

    logical = i_size_read(dir) >> dir->i_blkbits;

    ret = vecfs_get_block(dir, logical, &map, 1);
    if (ret)
        return ret;

    if (!buffer_mapped(&map))
        return -EIO;

    bh = sb_getblk(dir->i_sb, map.b_blocknr);
    if (!bh)
        return -ENOMEM;

    lock_buffer(bh);
    memset(bh->b_data, 0, dir->i_sb->s_blocksize);
    set_buffer_uptodate(bh);
    unlock_buffer(bh);

    mark_buffer_dirty_inode(bh, dir);
    brelse(bh);

    i_size_write(
        dir,
        i_size_read(dir) + dir->i_sb->s_blocksize);

    inode_set_mtime_to_ts(
        dir, inode_set_ctime_current(dir));
    mark_inode_dirty(dir);

    return 0;
}

int vecfs_add_entry(
    struct inode *dir,
    const struct qstr *name,
    ino_t ino)
{
    struct super_block *sb = dir->i_sb;
    unsigned int needed = VECFS_DIR_REC_LEN(name->len);
    bool extended = false;

retry:
    {
        sector_t blocks = DIV_ROUND_UP_ULL(
            i_size_read(dir), sb->s_blocksize);
        sector_t logical;

        for (logical = 0; logical < blocks; logical++) {
            struct buffer_head map = { 0 };
            struct buffer_head *bh;
            unsigned int offset = 0;
            int ret;

            ret = vecfs_get_block(
                dir, logical, &map, 0);
            if (ret)
                return ret;
            if (!buffer_mapped(&map))
                continue;

            bh = sb_bread(sb, map.b_blocknr);
            if (!bh)
                return -EIO;

            while (offset + 8 <= sb->s_blocksize) {
                struct vecfs_dir_entry *entry;
                unsigned int record_length;

                entry = (struct vecfs_dir_entry *)
                    (bh->b_data + offset);

                record_length =
                    le16_to_cpu(entry->rec_len);

                if (record_length < 8 ||
                    offset + record_length >
                        sb->s_blocksize)
                    break;

                if (!le32_to_cpu(entry->inode) &&
                    record_length >= needed) {
                    unsigned int remaining =
                        record_length - needed;

                    /*
                    * 只能清理当前目录项范围，不能按固定 name[] 大小写入，
                    * 因为磁盘目录项使用可变 rec_len。
                    */
                    memset(entry, 0, needed);

                    entry->inode = cpu_to_le32(ino);
                    entry->name_len = cpu_to_le16(name->len);
                    memcpy(entry->name, name->name, name->len);

                    if (remaining >= VECFS_DIR_REC_LEN(1)) {
                        struct vecfs_dir_entry *free_entry;

                        entry->rec_len = cpu_to_le16(needed);

                        free_entry =
                            (struct vecfs_dir_entry *)
                            ((u8 *)entry + needed);

                        memset(free_entry, 0, remaining);
                        free_entry->rec_len =
                            cpu_to_le16(remaining);
                    } else {
                        /*
                        * 剩余空间不足以形成合法目录项时，当前项占用整个旧记录。
                        */
                        entry->rec_len =
                            cpu_to_le16(record_length);
                    }

                    mark_buffer_dirty_inode(bh, dir);
                    brelse(bh);

                    inode_set_mtime_to_ts(
                        dir,
                        inode_set_ctime_current(dir));
                    mark_inode_dirty(dir);
                    return 0;
                }


                if (le32_to_cpu(entry->inode)) {
                    unsigned int minimal =
                        VECFS_DIR_REC_LEN(
                            le16_to_cpu(
                                entry->name_len));

                    if (record_length >=
                        minimal + needed) {
                        struct vecfs_dir_entry *new_entry;

                        new_entry =
                            (struct vecfs_dir_entry *)
                            ((u8 *)entry + minimal);

                        memset(new_entry, 0, needed);

                        new_entry->inode =
                            cpu_to_le32(ino);
                        new_entry->name_len =
                            cpu_to_le16(name->len);
                        new_entry->rec_len =
                            cpu_to_le16(
                                record_length -
                                minimal);

                        memcpy(new_entry->name,
                               name->name,
                               name->len);

                        entry->rec_len =
                            cpu_to_le16(minimal);

                        mark_buffer_dirty_inode(
                            bh, dir);
                        brelse(bh);

                        inode_set_mtime_to_ts(
                            dir,
                            inode_set_ctime_current(
                                dir));
                        mark_inode_dirty(dir);
                        return 0;
                    }
                }

                offset += record_length;
            }

            brelse(bh);
        }
    }

    if (extended)
        return -ENOSPC;

    if (vecfs_extend_directory(dir))
        return -ENOSPC;

    {
        struct buffer_head map = { 0 };
        struct buffer_head *bh;
        struct vecfs_dir_entry *entry;
        sector_t logical =
            (i_size_read(dir) - 1) >>
            dir->i_blkbits;
        int ret;

        ret = vecfs_get_block(
            dir, logical, &map, 0);
        if (ret || !buffer_mapped(&map))
            return ret ? ret : -EIO;

        bh = sb_bread(sb, map.b_blocknr);
        if (!bh)
            return -EIO;

        entry = (struct vecfs_dir_entry *)bh->b_data;
        entry->inode = 0;
        entry->name_len = 0;
        entry->rec_len =
            cpu_to_le16(sb->s_blocksize);

        mark_buffer_dirty_inode(bh, dir);
        brelse(bh);
    }

    extended = true;
    goto retry;
}

int vecfs_remove_entry(
    struct inode *dir,
    const struct qstr *name)
{
    struct super_block *sb = dir->i_sb;
    sector_t blocks;
    sector_t logical;

    blocks = DIV_ROUND_UP_ULL(
        i_size_read(dir), sb->s_blocksize);

    for (logical = 0; logical < blocks; logical++) {
        struct buffer_head map = { 0 };
        struct buffer_head *bh;
        struct vecfs_dir_entry *previous = NULL;
        unsigned int offset = 0;
        int ret;

        ret = vecfs_get_block(
            dir, logical, &map, 0);
        if (ret)
            return ret;
        if (!buffer_mapped(&map))
            continue;

        bh = sb_bread(sb, map.b_blocknr);
        if (!bh)
            return -EIO;

        while (offset + 8 <= sb->s_blocksize) {
            struct vecfs_dir_entry *entry;
            unsigned int record_length;

            entry = (struct vecfs_dir_entry *)
                (bh->b_data + offset);

            record_length =
                le16_to_cpu(entry->rec_len);

            if (record_length < 8 ||
                offset + record_length >
                    sb->s_blocksize)
                break;

            if (le32_to_cpu(entry->inode) &&
                le16_to_cpu(entry->name_len) ==
                    name->len &&
                !memcmp(entry->name,
                        name->name,
                        name->len)) {
                entry->inode = 0;
                entry->name_len = 0;

                if (previous &&
                    !le32_to_cpu(previous->inode)) {
                    previous->rec_len =
                        cpu_to_le16(
                            le16_to_cpu(
                                previous->rec_len) +
                            record_length);
                }

                mark_buffer_dirty_inode(bh, dir);
                brelse(bh);

                inode_set_mtime_to_ts(
                    dir,
                    inode_set_ctime_current(dir));
                mark_inode_dirty(dir);
                return 0;
            }

            previous = entry;
            offset += record_length;
        }

        brelse(bh);
    }

    return -ENOENT;
}

int vecfs_init_directory(
    struct inode *dir,
    struct inode *parent)
{
    struct buffer_head map = { 0 };
    struct buffer_head *bh;
    struct vecfs_dir_entry *entry;
    unsigned int dot_length =
        VECFS_DIR_REC_LEN(1);
    int ret;

    ret = vecfs_get_block(dir, 0, &map, 1);
    if (ret)
        return ret;

    if (!buffer_mapped(&map))
        return -EIO;

    bh = sb_getblk(dir->i_sb, map.b_blocknr);
    if (!bh)
        return -ENOMEM;

    lock_buffer(bh);
    memset(bh->b_data, 0, dir->i_sb->s_blocksize);

    entry = (struct vecfs_dir_entry *)bh->b_data;
    entry->inode = cpu_to_le32(dir->i_ino);
    entry->name_len = cpu_to_le16(1);
    entry->rec_len = cpu_to_le16(dot_length);
    entry->name[0] = '.';

    entry = (struct vecfs_dir_entry *)
        (bh->b_data + dot_length);

    entry->inode = cpu_to_le32(parent->i_ino);
    entry->name_len = cpu_to_le16(2);
    entry->rec_len = cpu_to_le16(
        dir->i_sb->s_blocksize - dot_length);
    entry->name[0] = '.';
    entry->name[1] = '.';

    set_buffer_uptodate(bh);
    unlock_buffer(bh);

    mark_buffer_dirty_inode(bh, dir);
    brelse(bh);

    i_size_write(dir, dir->i_sb->s_blocksize);
    set_nlink(dir, 2);

    inode_set_mtime_to_ts(
        dir, inode_set_ctime_current(dir));
    mark_inode_dirty(dir);

    return 0;
}

int vecfs_create(
    VECFS_IDMAP idmap,
    struct inode *dir,
    struct dentry *dentry,
    umode_t mode,
    bool excl)
{
    struct inode *inode;
    int ret;

    inode = vecfs_create_inode(
        idmap,
        dir->i_sb,
        dir,
        mode | S_IFREG,
        &dentry->d_name);
    if (IS_ERR(inode))
        return PTR_ERR(inode);

    ret = vecfs_add_entry(
        dir,
        &dentry->d_name,
        inode->i_ino);
    if (ret)
        goto fail;

    set_nlink(inode, 1);
    mark_inode_dirty(inode);

    vecfs_path_cache_set_best_effort(
        dir,
        inode,
        &dentry->d_name);

    d_instantiate(
        dentry,
        inode);

    unlock_new_inode(inode);
    return 0;

fail:
    clear_nlink(inode);
    discard_new_inode(inode);

    return ret;
}


VECFS_MKDIR_RET vecfs_mkdir(
    VECFS_IDMAP idmap,
    struct inode *dir,
    struct dentry *dentry,
    umode_t mode)
{
    struct inode *inode;
    int ret;

    inode = vecfs_create_inode(
        idmap,
        dir->i_sb,
        dir,
        S_IFDIR | mode,
        &dentry->d_name);
    if (IS_ERR(inode))
        return VECFS_MKDIR_ERR(PTR_ERR(inode));

    ret = vecfs_init_directory(
        inode,
        dir);
    if (ret)
        goto fail;

    ret = vecfs_add_entry(
        dir,
        &dentry->d_name,
        inode->i_ino);
    if (ret)
        goto fail;

    inc_nlink(dir);
    mark_inode_dirty(dir);

    vecfs_path_cache_set_best_effort(
        dir,
        inode,
        &dentry->d_name);

    d_instantiate(
        dentry,
        inode);

    unlock_new_inode(inode);
    return VECFS_MKDIR_OK;

fail:
    clear_nlink(inode);
    discard_new_inode(inode);

    return VECFS_MKDIR_ERR(ret);
}


bool vecfs_dir_empty(struct inode *dir)
{
    struct super_block *sb = dir->i_sb;
    sector_t blocks;
    sector_t logical;

    blocks = DIV_ROUND_UP_ULL(
        i_size_read(dir), sb->s_blocksize);

    for (logical = 0; logical < blocks; logical++) {
        struct buffer_head map = { 0 };
        struct buffer_head *bh;
        unsigned int offset = 0;

        if (vecfs_get_block(
                dir, logical, &map, 0) ||
            !buffer_mapped(&map))
            continue;

        bh = sb_bread(sb, map.b_blocknr);
        if (!bh)
            return false;

        while (offset + 8 <= sb->s_blocksize) {
            struct vecfs_dir_entry *entry;
            unsigned int length;
            unsigned int name_length;

            entry = (struct vecfs_dir_entry *)
                (bh->b_data + offset);

            length = le16_to_cpu(entry->rec_len);
            name_length =
                le16_to_cpu(entry->name_len);

            if (length < 8 ||
                offset + length > sb->s_blocksize)
                break;

            if (le32_to_cpu(entry->inode) &&
                !(name_length == 1 &&
                  entry->name[0] == '.') &&
                !(name_length == 2 &&
                  entry->name[0] == '.' &&
                  entry->name[1] == '.')) {
                brelse(bh);
                return false;
            }

            offset += length;
        }

        brelse(bh);
    }

    return true;
}

int vecfs_rmdir(
    struct inode *dir,
    struct dentry *dentry)
{
    struct inode *inode =
        d_inode(dentry);
    int ret;

    if (!vecfs_dir_empty(inode))
        return -ENOTEMPTY;

    ret = vecfs_remove_entry(
        dir,
        &dentry->d_name);
    if (ret)
        return ret;

    drop_nlink(dir);
    clear_nlink(inode);

    inode_set_ctime_current(inode);

    mark_inode_dirty(dir);
    mark_inode_dirty(inode);

    vecfs_path_cache_remove_best_effort(
        inode);

    return 0;
}


int vecfs_unlink(
    struct inode *dir,
    struct dentry *dentry)
{
    struct inode *inode =
        d_inode(dentry);
    int ret;

    ret = vecfs_remove_entry(
        dir,
        &dentry->d_name);
    if (ret)
        return ret;

    drop_nlink(inode);
    inode_set_ctime_current(inode);

    mark_inode_dirty(dir);
    mark_inode_dirty(inode);

    /*
     * 第一版不支持 hard link，因此目录项移除后即可移除规范路径。
     */
    vecfs_path_cache_remove_best_effort(
        inode);

    /*
     * 立即移除 FILE owner ANN 节点（标记 DELETED 并更新
     * 内存 snapshot），确保查询立刻看不到已 unlink 的文件向量。
     * evict_inode 会在 inode 释放时再次调用，作为安全兜底。
     */
    if (S_ISREG(inode->i_mode)) {
        ret = vecfs_ann_remove_all_nodes(inode);
        if (ret)
            pr_warn(
                "vecfs: removing ANN nodes for unlinked "
                "inode %lu failed: %d\n",
                inode->i_ino, ret);
    }

    return 0;
}


static void vecfs_path_cache_mark_incomplete(
    struct super_block *sb)
{
    struct vecfs_path_cache *cache =
        &VECFS_SB(sb)->s_path_cache;

    if (!cache->initialized)
        return;

    mutex_lock(&cache->lock);
    cache->complete = false;
    mutex_unlock(&cache->lock);
}

static void vecfs_path_cache_set_best_effort(
    struct inode *parent,
    struct inode *inode,
    const struct qstr *name)
{
    int ret;

    if (!parent ||
        !inode ||
        !name)
        return;

    ret = vecfs_path_cache_set(
        inode->i_sb,
        inode->i_ino,
        inode->i_generation,
        parent->i_ino,
        name,
        S_ISDIR(inode->i_mode));

    if (ret) {
        vecfs_path_cache_mark_incomplete(
            inode->i_sb);

        pr_warn(
            "vecfs: failed to cache path for "
            "inode %lu: %d\n",
            inode->i_ino,
            ret);
    }
}

static void vecfs_path_cache_remove_best_effort(
    struct inode *inode)
{
    int ret;

    if (!inode)
        return;

    ret = vecfs_path_cache_remove(
        inode->i_sb,
        inode->i_ino,
        inode->i_generation);

    if (ret && ret != -ENOENT) {
        vecfs_path_cache_mark_incomplete(
            inode->i_sb);

        pr_warn(
            "vecfs: failed to remove cached path for "
            "inode %lu: %d\n",
            inode->i_ino,
            ret);
    }
}


int vecfs_rename(
    VECFS_IDMAP idmap,
    struct inode *old_dir,
    struct dentry *old_dentry,
    struct inode *new_dir,
    struct dentry *new_dentry,
    unsigned int flags)
{
    struct inode *inode =
        d_inode(old_dentry);
    struct inode *target =
        d_inode(new_dentry);
    bool directory =
        S_ISDIR(inode->i_mode);
    int ret;

    if (flags & ~RENAME_NOREPLACE)
        return -EINVAL;

    if ((flags & RENAME_NOREPLACE) &&
        target)
        return -EEXIST;

    if (target &&
        S_ISDIR(target->i_mode) &&
        !vecfs_dir_empty(target))
        return -ENOTEMPTY;

    /*
     * 先删除目标目录项。当前文件系统不支持 exchange/whiteout。
     */
    if (target) {
        ret = vecfs_remove_entry(
            new_dir,
            &new_dentry->d_name);
        if (ret)
            return ret;
    }

    ret = vecfs_remove_entry(
        old_dir,
        &old_dentry->d_name);
    if (ret) {
        if (target) {
            vecfs_add_entry(
                new_dir,
                &new_dentry->d_name,
                target->i_ino);
        }

        return ret;
    }

    ret = vecfs_add_entry(
        new_dir,
        &new_dentry->d_name,
        inode->i_ino);
    if (ret) {
        int restore_ret;

        restore_ret = vecfs_add_entry(
            old_dir,
            &old_dentry->d_name,
            inode->i_ino);

        if (restore_ret)
            pr_err(
                "vecfs: rename rollback failed for "
                "inode %lu: %d\n",
                inode->i_ino,
                restore_ret);

        if (target) {
            restore_ret = vecfs_add_entry(
                new_dir,
                &new_dentry->d_name,
                target->i_ino);

            if (restore_ret)
                pr_err(
                    "vecfs: rename target rollback "
                    "failed for inode %lu: %d\n",
                    target->i_ino,
                    restore_ret);
        }

        return ret;
    }

    if (target) {
        if (S_ISDIR(target->i_mode))
            clear_nlink(target);
        else
            drop_nlink(target);

        inode_set_ctime_current(target);
        mark_inode_dirty(target);

        vecfs_path_cache_remove_best_effort(
            target);
    }

    if (directory &&
        old_dir != new_dir) {
        drop_nlink(old_dir);
        inc_nlink(new_dir);
    }

    inode_set_ctime_current(inode);

    inode_set_mtime_to_ts(
        old_dir,
        inode_set_ctime_current(old_dir));

    inode_set_mtime_to_ts(
        new_dir,
        inode_set_ctime_current(new_dir));

    mark_inode_dirty(inode);
    mark_inode_dirty(old_dir);
    mark_inode_dirty(new_dir);

    /*
     * 只更新被 rename inode。若它是目录，后代 parent 链仍指向该
     * inode，因此无需递归更新所有后代。
     */
    ret = vecfs_path_cache_rename(
        inode->i_sb,
        inode->i_ino,
        inode->i_generation,
        new_dir->i_ino,
        &new_dentry->d_name,
        directory);

    if (ret) {
        vecfs_path_cache_mark_incomplete(
            inode->i_sb);

        pr_warn(
            "vecfs: rename path cache update failed "
            "for inode %lu: %d\n",
            inode->i_ino,
            ret);
    }

    return 0;
}


int vecfs_store_symlink(
    struct inode *inode,
    const char *target)
{
    size_t length = strlen(target);

    return page_symlink(inode, target, length);
}

int vecfs_symlink(
    VECFS_IDMAP idmap,
    struct inode *dir,
    struct dentry *dentry,
    const char *target)
{
    struct inode *inode;
    int ret;

    inode = vecfs_create_inode(
        idmap,
        dir->i_sb,
        dir,
        S_IFLNK | 0777,
        &dentry->d_name);
    if (IS_ERR(inode))
        return PTR_ERR(inode);

    ret = vecfs_store_symlink(
        inode,
        target);
    if (ret)
        goto fail;

    ret = vecfs_add_entry(
        dir,
        &dentry->d_name,
        inode->i_ino);
    if (ret)
        goto fail;

    set_nlink(inode, 1);
    mark_inode_dirty(inode);

    vecfs_path_cache_set_best_effort(
        dir,
        inode,
        &dentry->d_name);

    d_instantiate(
        dentry,
        inode);

    unlock_new_inode(inode);
    return 0;

fail:
    clear_nlink(inode);
    discard_new_inode(inode);

    return ret;
}


int vecfs_readdir(
    struct file *file,
    struct dir_context *ctx)
{
    struct inode *inode = file_inode(file);
    struct super_block *sb = inode->i_sb;

    while (ctx->pos < i_size_read(inode)) {
        sector_t logical =
            ctx->pos >> sb->s_blocksize_bits;
        unsigned int offset =
            ctx->pos & (sb->s_blocksize - 1);
        struct buffer_head map = { 0 };
        struct buffer_head *bh;
        int ret;

        ret = vecfs_get_block(
            inode, logical, &map, 0);
        if (ret)
            return ret;

        if (!buffer_mapped(&map))
            return 0;

        bh = sb_bread(sb, map.b_blocknr);
        if (!bh)
            return -EIO;

        while (offset + 8 <= sb->s_blocksize &&
               ctx->pos < i_size_read(inode)) {
            struct vecfs_dir_entry *entry;
            unsigned int length;

            entry = (struct vecfs_dir_entry *)
                (bh->b_data + offset);

            length = le16_to_cpu(entry->rec_len);
            if (length < 8 ||
                offset + length > sb->s_blocksize) {
                brelse(bh);
                return -EUCLEAN;
            }

            if (le32_to_cpu(entry->inode) &&
                !dir_emit(
                    ctx,
                    entry->name,
                    le16_to_cpu(entry->name_len),
                    le32_to_cpu(entry->inode),
                    DT_UNKNOWN)) {
                brelse(bh);
                return 0;
            }

            offset += length;
            ctx->pos += length;
        }

        brelse(bh);
    }

    return 0;
}

int vecfs_readpage(
    struct file *file,
    struct folio *folio)
{
    return mpage_read_folio(
        folio, vecfs_get_block);
}

#if LINUX_VERSION_CODE < KERNEL_VERSION(7, 0, 0)
int vecfs_writepage(
    struct page *page,
    struct writeback_control *wbc)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 4, 0)
    return __block_write_full_folio(
        page->mapping->host,
        page_folio(page),
        vecfs_get_block,
        wbc);
#else
    return __block_write_full_page(
        page->mapping->host,
        page,
        vecfs_get_block,
        wbc,
        end_buffer_async_write);
#endif
}
#endif /* LINUX_VERSION_CODE < KERNEL_VERSION(7, 0, 0) */

int vecfs_writepages(
    struct address_space *mapping,
    struct writeback_control *wbc)
{
    return mpage_writepages(
        mapping, wbc, vecfs_get_block);
}

int vecfs_write_begin(
    VECFS_WB_FIRST file,
    struct address_space *mapping,
    loff_t position,
    unsigned int length,
    VECFS_WB_PAGEP folio,
    void **fsdata)
{
    return block_write_begin(
        mapping,
        position,
        length,
        folio,
        vecfs_get_block);
}

int vecfs_write_end(
    VECFS_WB_FIRST file,
    struct address_space *mapping,
    loff_t position,
    unsigned int length,
    unsigned int copied,
    VECFS_WB_PAGE folio,
    void *fsdata)
{
    struct inode *inode = mapping->host;
    int ret;

    /*
     * generic_write_end() 完整执行：
     *
     *   block_write_end()
     *   i_size 更新
     *   folio_unlock()
     *   folio_put()
     *
     * 不要在返回后再次 unlock/put。
     */
    ret = generic_write_end(
        file,
        mapping,
        position,
        length,
        copied,
        folio,
        fsdata);

    if (ret > 0) {
        inode_set_mtime_to_ts(
            inode,
            inode_set_ctime_current(inode));

        mark_inode_dirty(inode);
    }

    return ret;
}



ssize_t vecfs_read_iter(
    struct kiocb *iocb,
    struct iov_iter *to)
{
    return generic_file_read_iter(iocb, to);
}

ssize_t vecfs_write_iter(
    struct kiocb *iocb,
    struct iov_iter *from)
{
    struct file *file;
    struct inode *inode;
    size_t requested;
    loff_t start;
    int ret;

    if (!iocb || !from)
        return -EINVAL;

    file = iocb->ki_filp;
    if (!file)
        return -EINVAL;

    inode = file_inode(file);
    requested = iov_iter_count(from);

    if (!requested)
        return 0;

    /*
     * 对 O_APPEND 使用当前 EOF 作为保守起点。generic write checks
     * 会再次确定实际写入位置；如果存在并发 append，本次可能多标记
     * 一些范围，但不会漏掉旧内容绑定。
     */
    if (file->f_flags & O_APPEND)
        start = i_size_read(inode);
    else
        start = iocb->ki_pos;

    if (start < 0)
        return -EINVAL;

    if (vecfs_shield_write_marks_stale(inode->i_sb)) {
        /*
         * SHIELD/VALIDATED：先推进版本、再标记旧向量 stale（二者先于
         * 数据写入，见 vecfs_shield_write_begin）。版本先行推进使进行中
         * 的并发派生在提交复核时立即失效，杜绝 TOCTOU 无效结果泄露。
         */
        ret = vecfs_shield_write_begin(
            inode,
            start,
            requested);
        if (ret)
            return ret;
    }

    ret = generic_file_write_iter(
        iocb,
        from);

    if (ret > 0) {
        /*
         * 基线模式（EVENT/CONDITIONAL）不标记 stale，仍在写入成功后
         * 推进版本以支撑其过期判定。
         */
        if (!vecfs_shield_write_marks_stale(inode->i_sb))
            vecfs_shield_version_bump(inode);

        /*
         * SHIELD：写路径推进状态机并调度派生（其他模式为空操作）。
         * 旧向量已在上方 vecfs_shield_write_begin 中被置为 stale。
         */
        vecfs_shield_write_hook(inode, start, requested);
    }

    return ret;
}


/*
 * 尾部协议入口保留。
 *
 * 阶段二/阶段六实现 node object 和文件绑定事务后，
 * 该函数会解析 V2 footer 并提交对象。现在不能截断文件，
 * 否则会在未存储向量时丢失 payload。
 */

int vecfs_fsync(
    struct file *file,
    loff_t start,
    loff_t end,
    int datasync)
{
    int ret;

    /*
     * 先解析 page cache 中最新的尾部并提交 ANN。尾部提交成功后会
     * 截断文件，再由 generic_file_fsync() 统一写回文件数据、inode
     * 和间接块。
     */
    ret = vecfs_process_embedded_vectors(
        file);
    if (ret)
        return ret;

    return generic_file_fsync(
        file,
        start,
        end,
        datasync);
}


int vecfs_release(
    struct inode *inode,
    struct file *file)
{
    int ret;

    if (!inode || !file)
        return -EINVAL;

    /*
     * release 是尾部协议的兼容提交点。显式 fsync 已经截断尾部时，
     * 检测函数不会再次发现 magic，因此是幂等的。
     */
    ret = vecfs_process_embedded_vectors(
        file);

    return ret;
}



void vecfs_evict_inode(struct inode *inode)
{
    bool deleting =
        inode->i_ino > VECFS_FIRST_USER_INO &&
        !inode->i_nlink &&
        !is_bad_inode(inode);

    int ret;

    truncate_inode_pages_final(&inode->i_data);

    if (deleting) {
        /*
         * ANN 映射必须先处理，因为之后 inode generation 和
         * i_index[] 将不再可靠。
         */
        if (S_ISREG(inode->i_mode)) {
            ret = vecfs_ann_remove_all_nodes(inode);
            if (ret)
                pr_warn(
                    "vecfs: removing ANN bindings for "
                    "inode %lu failed: %d\n",
                    inode->i_ino,
                    ret);
        }

        ret = vecfs_truncate_blocks(inode, 0);
        if (ret)
            pr_warn(
                "vecfs: truncating inode %lu failed: %d\n",
                inode->i_ino,
                ret);

        vecfs_free_inode(
            inode->i_sb,
            inode);
    }

    if (S_ISREG(inode->i_mode))
        vecfs_shield_file_destroy(inode);

    /*
     * 失效关联缓冲必须放在所有截断/释放之后、clear_inode 之前：
     * vecfs_truncate_blocks 释放间接块时会对间接块 bh 调用
     * mark_buffer_dirty_inode()，把它重新挂回 i_data.i_private_list。
     * 若列表仍非空，clear_inode 会触发
     * BUG_ON(!list_empty(&inode->i_data.i_private_list)) 直接崩掉整机。
     * （对照 ext2：truncate 之后才 invalidate_inode_buffers。）
     */
    invalidate_inode_buffers(inode);

    /*
     * 保守兜底：若失效后仍有残留缓冲（理论上不应发生，仅在出现新的
     * 重挂路径时触发），打印诊断并再失效一次，而不是让 VFS BUG_ON 崩溃。
     */
    if (unlikely(!list_empty(vecfs_mapping_private_list(&inode->i_data)))) {
        pr_err(
            "vecfs: evict_inode ino=%lu mode=0%o nlink=%u icount=%d "
            "still has attached buffers after invalidation\n",
            inode->i_ino, inode->i_mode, inode->i_nlink,
            atomic_read(&inode->i_count));
        dump_stack();
        invalidate_inode_buffers(inode);
    }

    clear_inode(inode);
}



static int vecfs_sync_metadata(
    struct super_block *sb)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_super_disk_v2 *disk;
    int ret;

    ret = write_bitmap(
        sb,
        sbi->s_bmap_start,
        sbi->s_blocks_count,
        sbi->s_bmap);
    if (ret)
        return ret;

    ret = write_bitmap(
        sb,
        sbi->s_imap_start,
        sbi->s_inodes_count,
        sbi->s_imap);
    if (ret)
        return ret;

    if (sbi->s_ann.dirty) {
        ret = vecfs_ann_meta_commit(sb, true);
        if (ret)
            return ret;
    }

    disk = (struct vecfs_super_disk_v2 *)
        sbi->sbh->b_data;

    disk->free_blocks =
        cpu_to_le64(sbi->s_free_blocks);
    disk->free_inodes =
        cpu_to_le64(sbi->s_free_inodes);
    disk->write_time_sec =
        cpu_to_le64(ktime_get_real_seconds());
    disk->ann_active_hint =
        cpu_to_le32(sbi->s_ann.active_copy);

    disk->checksum = 0;
    disk->checksum =
        cpu_to_le32(vecfs_super_crc(disk));

    mark_buffer_dirty(sbi->sbh);
    sync_dirty_buffer(sbi->sbh);

    if (buffer_write_io_error(sbi->sbh))
        return -EIO;

    return 0;
}

/*
 * Phase D：改写 superblock 的干净卸载标记并同步落盘。
 *
 * clean=true 写入 VECFS_SUPER_CLEAN_MAGIC（卸载 drain 完成后）；
 * clean=false 写入 0（首条延迟 stale 落盘前置 dirty，确保崩溃可被检测）。
 * 调用方负责在 writer_lock 下同步访问（stale.c 的 mark 与卸载 drain）。
 */
int vecfs_super_set_clean(struct super_block *sb, bool clean)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_super_disk_v2 *disk;

    if (!sbi || !sbi->sbh)
        return -EINVAL;

    disk = (struct vecfs_super_disk_v2 *)
        sbi->sbh->b_data;

    disk->clean_magic =
        cpu_to_le64(clean ? VECFS_SUPER_CLEAN_MAGIC : 0);

    disk->checksum = 0;
    disk->checksum =
        cpu_to_le32(vecfs_super_crc(disk));

    mark_buffer_dirty(sbi->sbh);
    sync_dirty_buffer(sbi->sbh);

    if (buffer_write_io_error(sbi->sbh))
        return -EIO;

    return 0;
}

static void vecfs_init_ann_roots(
    struct vecfs_inode *vi)
{
    unsigned int i;

    for (i = 0; i < VECFS_N_BLOCK_PTRS; i++)
        vi->i_index[i] =
            VECFS_ANN_EMPTY_NODE_ID;
}


void vecfs_put_super(struct super_block *sb)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    int ret;

    if (!sbi)
        return;

    /*
     * 先关闭 SHIELD 派生并排空所有派生 worker（置 shutting_down、唤醒 upcall、
     * 等待 derive_refs 归零），再做任何同步/落盘/销毁。否则同步与卸载路径会
     * 与仍在运行的派生 worker 交互（它们读写 ANN 存储、持有 writer_lock 等），
     * 可能导致卸载永久卡死；且释放 rt/sbi/ANN 前必须确保没有 worker 运行中/
     * 排队中/计时器待发，否则 use-after-free。
     */
    vecfs_shield_shutdown(sb);

    ret = sync_filesystem(sb);
    if (ret)
        pr_warn("vecfs: sync_filesystem failed: %d\n",
                ret);

    /*
     * arena inode 仍然有效时先写回普通 inode 和对象块。
     */
    if (sbi->s_ann_store &&
        sbi->s_ann_store->arena_inode)
        write_inode_now(
            sbi->s_ann_store->arena_inode, 1);

    /*
     * Phase D：卸载前先 drain 延迟 stale 落盘，再写干净卸载标记并同步。
     * drain 失败仍继续（尽力而为），但此时不写 clean 标记，下次挂载走保守
     * 恢复兜底。
     */
    ret = vecfs_ann_stale_drain(sb);
    if (ret < 0)
        pr_err("vecfs: stale drain failed: %d\n", ret);
    else {
        ret = vecfs_super_set_clean(sb, true);
        if (ret)
            pr_err("vecfs: clean-marker write failed: %d\n", ret);
        else
            sbi->s_stale_dirty_marked = false;
    }

    ret = vecfs_sync_metadata(sb);
    if (ret)
        pr_err("vecfs: metadata sync failed: %d\n",
               ret);

    vecfs_ann_rebuild_async_destroy(sb);
    vecfs_ann_sched_destroy(sb);
    vecfs_ann_search_pool_destroy(sb);
    vecfs_ann_exchange_destroy(sb);
    vecfs_ann_label_index_destroy(sb);
    vecfs_ann_cache_unmount(sb);
    vecfs_ann_runtime_destroy(sb);
    vecfs_ann_distance_destroy(sb);
    vecfs_ann_store_unmount(sb);
    vecfs_ann_meta_destroy(sb);

    vecfs_path_cache_destroy(sb);




    kvfree(sbi->s_imap);
    kvfree(sbi->s_bmap);

    if (sbi->sbh)
        brelse(sbi->sbh);

    mutex_destroy(&sbi->s_lock);

    sb->s_fs_info = NULL;
    kfree(sbi);
}

int vecfs_mmap(
    struct file *file,
    struct vm_area_struct *vma)
{
    struct inode *inode;
    u64 offset;
    u64 length;
    int ret;

    if (!file || !vma)
        return -EINVAL;

    inode = file_inode(file);

    /*
     * MAP_PRIVATE 的写入只触发 COW，不修改文件。
     *
     * 使用 VM_MAYWRITE 而不是 VM_WRITE，因为用户之后可以通过
     * mprotect() 为 MAP_SHARED mapping 增加写权限。映射建立时
     * 保守标记 stale，避免后续 page fault 路径执行持久 ANN 更新。
     */
    if ((vma->vm_flags & VM_SHARED) &&
        (vma->vm_flags & VM_MAYWRITE)) {
        if ((u64)vma->vm_pgoff >
            (U64_MAX >> PAGE_SHIFT))
            return -EOVERFLOW;

        offset =
            (u64)vma->vm_pgoff <<
            PAGE_SHIFT;

        length =
            (u64)vma->vm_end -
            (u64)vma->vm_start;

        if (length) {
            ret = vecfs_ann_mark_content_stale(
                inode,
                offset,
                length);
            if (ret)
                return ret;
        }
    }

    return generic_file_mmap(
        file,
        vma);
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(7, 0, 0)
int vecfs_fill_super(
    struct super_block *sb,
    struct fs_context *fc)
#else
int vecfs_fill_super(
    struct super_block *sb,
    void *data,
    int silent)
#endif
{
    struct vecfs_sb_info *sbi;
    struct vecfs_super_disk_v2 *disk;
    struct buffer_head *bh;
    struct inode *root_inode;
    int ret;

    sbi = kzalloc(sizeof(*sbi), GFP_KERNEL);
    if (!sbi)
        return -ENOMEM;

    sb->s_fs_info = sbi;

    mutex_init(&sbi->s_lock);
    spin_lock_init(&sbi->s_block_lock);
    INIT_LIST_HEAD(&sbi->s_dirty_dirents);
    atomic_set(&sbi->s_dirty_count, 0);

    /*
     * 默认 VALIDATED（与现有写路径 stale 标记行为一致）。
     */
    sbi->s_shield.mode = VECFS_SHIELD_MODE_VALIDATED;
    vecfs_shield_runtime_init(&sbi->s_shield);

    if (!sb_set_blocksize(
            sb, VECFS_DISK_BLOCK_SIZE)) {
        ret = -EINVAL;
        goto fail_sbi;
    }

    bh = sb_bread(sb, 0);
    if (!bh) {
        ret = -EIO;
        goto fail_sbi;
    }

    disk = (struct vecfs_super_disk_v2 *)
        bh->b_data;

    ret = vecfs_validate_super(disk);
    if (ret)
        goto fail_bh;

    sbi->sbh = bh;
    sbi->s_magic =
        le32_to_cpu(disk->magic);
    sbi->s_disk_major =
        le16_to_cpu(disk->major);
    sbi->s_disk_minor =
        le16_to_cpu(disk->minor);
    sbi->s_features =
        le64_to_cpu(disk->features);

    /*
     * Phase D：读取干净卸载标记，决定挂载后是否保守恢复。
     */
    sbi->s_clean_unmount =
        le64_to_cpu(disk->clean_magic) ==
            VECFS_SUPER_CLEAN_MAGIC;
    sbi->s_stale_dirty_marked =
        !sbi->s_clean_unmount;

    /* 嵌入服务配置（superblock 镜像，NUL 结尾；url 非空 = 挂载时拉起守护）。 */
    strscpy(sbi->s_embed_url, (const char *)disk->embed_url,
            sizeof(sbi->s_embed_url));
    strscpy(sbi->s_embed_key, (const char *)disk->embed_key,
            sizeof(sbi->s_embed_key));
    strscpy(sbi->s_embed_model, (const char *)disk->embed_model,
            sizeof(sbi->s_embed_model));
    strscpy(sbi->s_embed_protocol, (const char *)disk->embed_protocol,
            sizeof(sbi->s_embed_protocol));
    sbi->s_embed_configured = (disk->embed_url[0] != 0);

    sbi->s_blocksize =
        le32_to_cpu(disk->block_size);
    sbi->s_root_ino =
        le32_to_cpu(disk->root_ino);

    sbi->s_inodes_count =
        le64_to_cpu(disk->inodes_count);
    sbi->s_blocks_count =
        le64_to_cpu(disk->blocks_count);

    sbi->s_data_start =
        le64_to_cpu(disk->data_start);
    sbi->s_inode_table =
        le64_to_cpu(disk->inode_table);

    sbi->s_free_inodes =
        le64_to_cpu(disk->free_inodes);
    sbi->s_free_blocks =
        le64_to_cpu(disk->free_blocks);

    sbi->s_imap_start =
        le64_to_cpu(disk->imap_start);
    sbi->s_bmap_start =
        le64_to_cpu(disk->bmap_start);

    sbi->s_imap_blocks =
        le32_to_cpu(disk->imap_blocks);
    sbi->s_bmap_blocks =
        le32_to_cpu(disk->bmap_blocks);

    sbi->s_ann_meta_block[0] =
        le64_to_cpu(disk->ann_meta_a);
    sbi->s_ann_meta_block[1] =
        le64_to_cpu(disk->ann_meta_b);

    sbi->s_next_ino = VECFS_FIRST_USER_INO;
    sbi->s_next_free_block = sbi->s_data_start;

    sbi->s_bmap = load_bitmap(
        sb,
        sbi->s_bmap_start,
        sbi->s_blocks_count);
    if (!sbi->s_bmap) {
        ret = -ENOMEM;
        goto fail_bh;
    }

    sbi->s_imap = load_bitmap(
        sb,
        sbi->s_imap_start,
        sbi->s_inodes_count);
    if (!sbi->s_imap) {
        ret = -ENOMEM;
        goto fail_bmap;
    }
    sb->s_magic = VECFS_SUPER_MAGIC;
    sb->s_maxbytes = MAX_LFS_FILESIZE;
    sb->s_op = &vecfs_super_ops;
    
    ret = vecfs_ann_meta_load(sb);
    if (ret)
        goto fail_imap;
    ret = vecfs_ann_store_mount(sb);
    if (ret)
        goto fail_ann_meta;
    ret = vecfs_ann_validate_mount(sb);
    if (ret)
        goto fail_ann_store;
    ret = vecfs_ann_distance_init(sb);
    if (ret)
        goto fail_ann_store;
    ret = vecfs_ann_runtime_init(sb);
    if (ret)
        goto fail_distance;
    ret = vecfs_ann_cache_mount(sb);
    if (ret)
        goto fail_distance;

    /*
     * Phase D：非干净卸载（崩溃）→ 保守失效所有 file-bound 向量，恢复
     * 正确性（查询不再返回过期向量）。召回由 set_mode(SHIELD) 触发
     * 重派生恢复。
     */
    if (!sbi->s_clean_unmount) {
        ret = vecfs_ann_recover_file_vectors(sb);
        if (ret < 0) {
            pr_err("vecfs: crash recovery failed: %d\n", ret);
            goto fail_cache;
        }
        pr_warn("vecfs: dirty unmount detected, conservatively "
                "invalidated %d file vectors\n", ret);
    }

    ret = vecfs_ann_label_index_init(sb);
    if (ret)
        goto fail_cache;
    ret = vecfs_ann_exchange_init(sb);
    if (ret)
        goto fail_labels;
   ret = vecfs_ann_search_pool_init(
        sb,
        atomic_read(
            &sbi->s_ann_runtime.maximum_queries));
    if (ret)
        goto fail_exchange;
    ret = vecfs_ann_sched_init(sb);
    if (ret)
        goto fail_search_pool;
    ret = vecfs_ann_rebuild_async_init(sb);
    if (ret)
        goto fail_sched;



    root_inode = iget_locked(
        sb, sbi->s_root_ino);
    if (!root_inode) {
        ret = -ENOMEM;
        goto fail_search_pool;
    }

    if (VECFS_INODE_IS_NEW(root_inode)) {
        ret = vecfs_read_inode(root_inode);
        if (ret) {
            iget_failed(root_inode);
            goto fail_search_pool;
        }

        if (!root_inode->i_mode) {
            inode_init_owner(
                VECFS_IDMAP_NOP,
                root_inode,
                NULL,
                S_IFDIR | 0755);

            root_inode->i_op =
                &vecfs_inode_dir_ops;
            root_inode->i_fop =
                &vecfs_file_dir_ops;
            root_inode->i_mapping->a_ops =
                &vecfs_address_space_ops;

            root_inode->i_generation =
                get_random_u32();

            ret = vecfs_init_directory(
                root_inode, root_inode);
            if (ret) {
                iget_failed(root_inode);
                goto fail_search_pool;
            }

            mark_inode_dirty(root_inode);
            write_inode_now(root_inode, 1);
        }

        unlock_new_inode(root_inode);
    }
    ret = vecfs_path_cache_init(sb);
    if (ret) {
        iput(root_inode);
        goto fail_search_pool;
    }

    ret = vecfs_path_cache_rebuild(sb);
    if (ret) {
        pr_err(
            "vecfs: path cache rebuild failed: %d\n",
            ret);

        vecfs_path_cache_destroy(sb);
        iput(root_inode);

        goto fail_search_pool;
    }

    /*
     * 嵌入配置存在则拉起守护（call_usermodehelper）。失败 = 无法履行嵌入
     * 契约，拒绝挂载（与 path_cache_rebuild 失败同样回到 fail_search_pool）。
     */
    if (sbi->s_embed_configured) {
        ret = vecfs_embed_upcall_start(sbi->s_embed_url, sbi->s_embed_key,
                                       sbi->s_embed_model,
                                       sbi->s_embed_protocol);
        if (ret) {
            pr_err("vecfs: embed helper start failed: %d\n", ret);
            vecfs_path_cache_destroy(sb);
            iput(root_inode);
            goto fail_search_pool;
        }
    }

    sb->s_root = d_make_root(root_inode);
    if (!sb->s_root) {
        ret = -ENOMEM;
        goto fail_ann_store;
    }

    pr_info(
        "vecfs: mounted V2, blocks=%u inodes=%u "
        "dim=%u distance=%u M=%u M0=%u\n",
        sbi->s_blocks_count,
        sbi->s_inodes_count,
        sbi->s_ann.config.dimension,
        sbi->s_ann.config.distance_type,
        sbi->s_ann.config.m,
        sbi->s_ann.config.m0);

    return 0;

fail_sched:
    vecfs_ann_sched_destroy(sb);
fail_search_pool:
    vecfs_ann_search_pool_destroy(sb);
fail_exchange:
    vecfs_ann_exchange_destroy(sb);
fail_labels:
    vecfs_ann_label_index_destroy(sb);
fail_cache:
    vecfs_ann_cache_unmount(sb);
fail_distance:
    vecfs_ann_runtime_destroy(sb);
    vecfs_ann_distance_destroy(sb);
fail_ann_store:
    vecfs_ann_store_unmount(sb);
fail_ann_meta:
    vecfs_ann_meta_destroy(sb);
fail_imap:
    kvfree(sbi->s_imap);
fail_bmap:
    kvfree(sbi->s_bmap);
fail_bh:
    brelse(bh);
fail_sbi:
    mutex_destroy(&sbi->s_lock);
    kfree(sbi);
    sb->s_fs_info = NULL;
    return ret;

}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(7, 0, 0)
static int vecfs_get_tree(struct fs_context *fc)
{
    return get_tree_bdev(fc, vecfs_fill_super);
}

static const struct fs_context_operations vecfs_context_ops = {
    .get_tree = vecfs_get_tree,
};

static int vecfs_init_fs_context(struct fs_context *fc)
{
    fc->ops = &vecfs_context_ops;
    return 0;
}
#else
struct dentry *vecfs_mount(
    struct file_system_type *type,
    int flags,
    const char *device,
    void *data)
{
    return mount_bdev(
        type,
        flags,
        device,
        data,
        vecfs_fill_super);
}
#endif

void vecfs_kill_sb(struct super_block *sb)
{
    struct vecfs_sb_info *sbi = sb ? VECFS_SB(sb) : NULL;

    pr_err("vecfs: kill_sb sb=%p fs_info=%p root=%p bdev=%p cache=%p\n",
           sb, sb->s_fs_info, sb->s_root, sb->s_bdev,
           sbi ? sbi->s_ann_cache : NULL);

    /*
     * 在 kill_block_super 之前排空派生 worker：VFS 的 generic_shutdown_super
     * 先 evict_inodes（只回收 i_count==0 的 inode）再回调 put_super，故若把
     * 排空只放在 put_super（如 vecfs_shield_shutdown），派生子代持有的 igrab
     * 会把 inode 的 i_count 撑在 >0，使它们被 evict_inodes 跳过、永不回收，
     * 最终在 rmmod 时触发 vecfs_inode 缓存残留的 BUG。这里提前置 shutting_down、
     * 唤醒 upcall 并等待 derive_refs 归零，让 worker 的 igrab 经 iput 归还，
     * inode 走正常 iput_final 释放，evict_inodes 随后能全部回收。
     * put_super 里的 shutdown 保留为兜底（此时已归零，幂等即时返回）。
     */
    vecfs_shield_shutdown(sb);

    kill_block_super(sb);
}

struct file_system_type vecfs_type = {
    .owner = THIS_MODULE,
    .name = VECFS_MOD_NAME,
#if LINUX_VERSION_CODE >= KERNEL_VERSION(7, 0, 0)
    .init_fs_context = vecfs_init_fs_context,
#else
    .mount = vecfs_mount,
#endif
    .kill_sb = vecfs_kill_sb,
    .fs_flags = FS_REQUIRES_DEV,
};

static int __init vecfs_init(void)
{
    int ret;

    pr_info("vecfs: sizeof(struct inode)=%zu "
            "sizeof(struct vecfs_inode)=%zu "
            "meta_lock_off=%zu\n",
            sizeof(struct inode),
            sizeof(struct vecfs_inode),
            offsetof(struct vecfs_inode, meta_lock));

    vecfs_inode_cachep = kmem_cache_create(
        VECFS_MOD_NAME "_inode",
        sizeof(struct vecfs_inode),
        0,
        SLAB_RECLAIM_ACCOUNT,
        vecfs_inode_init_once);
    if (!vecfs_inode_cachep)
        return -ENOMEM;

    ret = vecfs_ann_cache_init();
    if (ret) {
        kmem_cache_destroy(vecfs_inode_cachep);
        vecfs_inode_cachep = NULL;
        return ret;
    }

    ret = vecfs_embed_upcall_init();
    if (ret) {
        vecfs_ann_cache_exit();
        kmem_cache_destroy(vecfs_inode_cachep);
        vecfs_inode_cachep = NULL;
        return ret;
    }

    ret = register_filesystem(&vecfs_type);
    if (ret) {
        vecfs_embed_upcall_exit();
        vecfs_ann_cache_exit();
        kmem_cache_destroy(vecfs_inode_cachep);
        vecfs_inode_cachep = NULL;
        return ret;
    }

    pr_info("vecfs: V2 filesystem registered\n");
    return 0;
}

static void __exit vecfs_exit(void)
{
    int ret;

    ret = unregister_filesystem(&vecfs_type);
    if (ret)
        pr_err("vecfs: unregister failed: %d\n", ret);

    vecfs_embed_upcall_exit();

    rcu_barrier();

    /*
     * rcu_barrier 已等完所有 call_rcu 回调（快照 free 已排进 free_wq），
     * 这里 flush 这些 free 真正跑完再销毁 workqueue，避免模块卸载后
     * workqueue 还回调模块代码。
     */
    vecfs_ann_cache_exit();

    if (vecfs_inode_cachep) {
        kmem_cache_destroy(vecfs_inode_cachep);
        vecfs_inode_cachep = NULL;
    }

    pr_info("vecfs: V2 filesystem unregistered\n");
}

/*
 * SHIELD 事件通知：poll() 阻塞等待文件派生状态回到 CLEAN（或其它终态）。
 * 仅在 SHIELD 模式下有意义；非 SHIELD 模式按普通文件语义（恒可读）。
 * 利用 rt->state_wq 阻塞等待，替代用户态 100ms 轮询 file_state ioctl。
 */
static __poll_t vecfs_file_poll(struct file *file, poll_table *wait)
{
    struct inode *inode = file_inode(file);
    struct vecfs_shield_runtime *rt = vecfs_shield_runtime(inode->i_sb);
    struct vecfs_shield_file_state_info info;

    if (!rt || vecfs_shield_mode(inode->i_sb) != VECFS_SHIELD_MODE_SHIELD)
        return DEFAULT_POLLMASK;

    poll_wait(file, &rt->state_wq, wait);

    if (vecfs_shield_file_state_get(inode, &info))
        return EPOLLERR;

    if (info.state == VECFS_SHIELD_FS_CLEAN)
        return EPOLLIN | EPOLLRDNORM;

    return 0;
}

const struct file_operations vecfs_file_ops = {
    .owner = THIS_MODULE,

    .mmap = vecfs_mmap,

    .fsync = vecfs_fsync,
    .open = generic_file_open,
    .llseek = generic_file_llseek,
    .poll = vecfs_file_poll,

    .read_iter = vecfs_read_iter,
    .write_iter = vecfs_write_iter,

    .fallocate = vecfs_fallocate,
    .release = vecfs_release,

    .unlocked_ioctl = vecfs_ioctl,

#ifdef CONFIG_COMPAT
    .compat_ioctl = vecfs_compat_ioctl,
#endif
};


const struct file_operations vecfs_file_dir_ops = {
    .owner = THIS_MODULE,
    .llseek = generic_file_llseek,
    .read = generic_read_dir,
    .fsync = generic_file_fsync,
    .iterate_shared = vecfs_readdir,
    .unlocked_ioctl = vecfs_ioctl,
#ifdef CONFIG_COMPAT
    .compat_ioctl = vecfs_compat_ioctl,
#endif
};

const struct inode_operations vecfs_inode_ops = {
    .setattr = vecfs_setattr,
    .getattr = simple_getattr,
};

const struct inode_operations vecfs_inode_dir_ops = {
    .create = vecfs_create,
    .mkdir = vecfs_mkdir,
    .rmdir = vecfs_rmdir,
    .unlink = vecfs_unlink,
    .lookup = vecfs_lookup,
    .rename = vecfs_rename,
    .getattr = simple_getattr,
    .setattr = vecfs_setattr,
    .symlink = vecfs_symlink,
};


const struct super_operations vecfs_super_ops = {
    .statfs = simple_statfs,
    .destroy_inode = vecfs_destroy_inode,
    .put_super = vecfs_put_super,
    .alloc_inode = vecfs_alloc_inode,
    .write_inode = vecfs_write_inode,
    .evict_inode = vecfs_evict_inode,
};

const struct address_space_operations vecfs_address_space_ops = {
    .read_folio = vecfs_readpage,
#if LINUX_VERSION_CODE < KERNEL_VERSION(7, 0, 0)
    .writepage = vecfs_writepage,
#endif
    .writepages = vecfs_writepages,
    .write_begin = vecfs_write_begin,
    .write_end = vecfs_write_end,
    .bmap = vecfs_bmap,
    .dirty_folio = block_dirty_folio,
};


module_init(vecfs_init);
module_exit(vecfs_exit);
