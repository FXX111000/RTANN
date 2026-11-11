// SPDX-License-Identifier: GPL-2.0

#include <linux/bitmap.h>
#include <linux/blkdev.h>
#include <linux/buffer_head.h>
#include <linux/err.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/mutex.h>
#include <linux/rcupdate.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

#include "vecfs.h"
#include "vecfs_Indirect.h"
#include "vecfs_ann_indirect.h"
#include "include/vecfs_ann_store.h"
#include "include/vecfs_disk.h"


struct vecfs_ann_path {
    unsigned int depth;
    unsigned int root;
    unsigned int offsets[3];
};

static unsigned int vecfs_ann_ptrs_per_block(
    struct super_block *sb)
{
    return sb->s_blocksize / sizeof(__le32);
}

static int vecfs_ann_make_path(
    struct inode *inode,
    u32 ordinal,
    struct vecfs_ann_path *path)
{
    struct super_block *sb = inode->i_sb;
    u64 ptrs;
    u64 single_capacity;
    u64 double_capacity;
    u64 triple_capacity;
    u64 value;

    if (!path)
        return -EINVAL;

    ptrs = vecfs_ann_ptrs_per_block(sb);
    if (!ptrs)
        return -EINVAL;

    memset(path, 0, sizeof(*path));

    value = ordinal;

    if (value < VECFS_N_DIRECT) {
        path->depth = 0;
        path->root = value;
        return 0;
    }

    value -= VECFS_N_DIRECT;

    single_capacity = ptrs;
    double_capacity = ptrs * ptrs;
    triple_capacity = double_capacity * ptrs;

    if (value < single_capacity) {
        path->depth = 1;
        path->root = VECFS_IND_BLOCK;
        path->offsets[0] = value;
        return 0;
    }

    value -= single_capacity;

    if (value < double_capacity) {
        path->depth = 2;
        path->root = VECFS_DIND_BLOCK;
        path->offsets[0] = div_u64(value, ptrs);
        path->offsets[1] = value % ptrs;
        return 0;
    }

    value -= double_capacity;

    if (value < triple_capacity) {
        path->depth = 3;
        path->root = VECFS_TIND_BLOCK;
        path->offsets[0] =
            div_u64(value, double_capacity);

        value %= double_capacity;

        path->offsets[1] =
            div_u64(value, ptrs);
        path->offsets[2] =
            value % ptrs;

        return 0;
    }

    return -EFBIG;
}

static bool vecfs_ann_block_is_empty(
    const __le32 *entries,
    unsigned int count)
{
    unsigned int i;

    for (i = 0; i < count; i++) {
        if (le32_to_cpu(entries[i]) != 0)
            return false;
    }

    return true;
}

static int vecfs_ann_read_indirect(
    struct inode *inode,
    u32 root,
    const unsigned int *offsets,
    unsigned int depth,
    u32 *node_id)
{
    struct super_block *sb = inode->i_sb;
    u32 cur = root;
    unsigned int level;

    if (!root || !offsets || !node_id || !depth)
        return -ENOENT;

    for (level = 0; level < depth; level++) {
        struct buffer_head *bh;
        __le32 *entries;
        u32 next;
        unsigned int index = offsets[level];

        if (index >= vecfs_ann_ptrs_per_block(sb))
            return -EFBIG;

        bh = sb_bread(sb, cur);
        if (!bh)
            return -EIO;

        entries = (__le32 *)bh->b_data;
        next = le32_to_cpu(entries[index]);

        brelse(bh);

        if (!next)
            return -ENOENT;

        if (level == depth - 1) {
            if (next == VECFS_ANN_EMPTY_NODE_ID)
                return -ENOENT;

            *node_id = next;
            return 0;
        }

        cur = next;
    }

    return -ENOENT;
}

static int vecfs_ann_create_indirect_block(
    struct inode *inode,
    u32 *block)
{
    int ret;

    ret = vecfs_alloc_fs_block(inode, block);
    if (ret)
        return ret;

    ret = vecfs_zero_fs_block(inode, *block);
    if (ret) {
        vecfs_free_block(inode, *block);
        *block = 0;
        return ret;
    }

    return 0;
}

static int vecfs_ann_write_indirect(
    struct inode *inode,
    u32 *root,
    const unsigned int *offsets,
    unsigned int depth,
    u32 node_id,
    bool replace)
{
    struct super_block *sb = inode->i_sb;
    u32 cur;
    unsigned int level;
    int ret = 0;

    if (!root || !offsets || !depth)
        return -EINVAL;

    if (node_id == VECFS_ANN_EMPTY_NODE_ID)
        return -EINVAL;

    cur = *root;

    if (!cur) {
        ret = vecfs_ann_create_indirect_block(
            inode, &cur);
        if (ret)
            return ret;

        *root = cur;
        mark_inode_dirty(inode);
    }

    for (level = 0; level < depth; level++) {
        struct buffer_head *bh;
        __le32 *entries;
        unsigned int index = offsets[level];
        u32 next;

        if (index >= vecfs_ann_ptrs_per_block(sb))
            return -EFBIG;

        bh = sb_bread(sb, cur);
        if (!bh)
            return -EIO;

        entries = (__le32 *)bh->b_data;
        next = le32_to_cpu(entries[index]);

        if (level == depth - 1) {
            if (next && !replace) {
                brelse(bh);
                return -EEXIST;
            }

            entries[index] =
                cpu_to_le32(node_id);

            mark_buffer_dirty_inode(bh, inode);
            brelse(bh);
            return 0;
        }

        if (!next) {
            ret = vecfs_ann_create_indirect_block(
                inode, &next);
            if (ret) {
                brelse(bh);
                return ret;
            }

            entries[index] =
                cpu_to_le32(next);
            mark_buffer_dirty_inode(bh, inode);
        }

        brelse(bh);
        cur = next;
    }

    return -EUCLEAN;
}

static int vecfs_ann_clear_indirect(
    struct inode *inode,
    u32 *root,
    const unsigned int *offsets,
    unsigned int depth)
{
    struct super_block *sb = inode->i_sb;
    struct buffer_head *bh;
    __le32 *entries;
    u32 child;
    unsigned int index;
    int ret;

    if (!root || !*root || !depth)
        return 0;

    index = offsets[0];
    if (index >= vecfs_ann_ptrs_per_block(sb))
        return -EFBIG;

    bh = sb_bread(sb, *root);
    if (!bh)
        return -EIO;

    entries = (__le32 *)bh->b_data;
    child = le32_to_cpu(entries[index]);

    if (!child) {
        brelse(bh);
        return 0;
    }

    if (depth == 1) {
        /*
         * 叶子值是 node ID，不是物理块号。
         * 这里只清除映射，不能释放 child。
         */
        entries[index] = 0;
        mark_buffer_dirty_inode(bh, inode);
    } else {
        u32 child_root = child;

        ret = vecfs_ann_clear_indirect(
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

    if (vecfs_ann_block_is_empty(
            entries,
            vecfs_ann_ptrs_per_block(sb))) {
        u32 physical = *root;

        brelse(bh);

        *root = 0;
        vecfs_free_block(inode, physical);
        mark_inode_dirty(inode);
        return 0;
    }

    brelse(bh);
    return 0;
}


int vecfs_ann_get_node(
    struct inode *inode,
    u32 ordinal,
    u32 *node_id)
{
    struct vecfs_inode *vi;
    struct vecfs_ann_path path;
    u32 root;
    int ret;

    if (!inode || !node_id)
        return -EINVAL;

    *node_id = VECFS_ANN_EMPTY_NODE_ID;

    vi = VECFS_I(inode);

    ret = vecfs_ann_make_path(
        inode, ordinal, &path);
    if (ret)
        return ret;

    down_read(&vi->meta_lock);

    root = vi->i_index[path.root];

    if (!root) {
        ret = -ENOENT;
        goto out;
    }

    if (!path.depth) {
        if (root == VECFS_ANN_EMPTY_NODE_ID) {
            ret = -ENOENT;
            goto out;
        }

        *node_id = root;
        ret = 0;
        goto out;
    }

    ret = vecfs_ann_read_indirect(
        inode,
        root,
        path.offsets,
        path.depth,
        node_id);

out:
    up_read(&vi->meta_lock);
    return ret;
}

int vecfs_ann_set_node(
    struct inode *inode,
    u32 ordinal,
    u32 node_id,
    bool replace)
{
    struct vecfs_inode *vi;
    struct vecfs_ann_path path;
    u32 *root;
    int ret;

    if (!inode ||
        node_id == VECFS_ANN_EMPTY_NODE_ID)
        return -EINVAL;

    vi = VECFS_I(inode);

    ret = vecfs_ann_make_path(
        inode, ordinal, &path);
    if (ret)
        return ret;

    down_write(&vi->meta_lock);

    root = &vi->i_index[path.root];

    if (!path.depth) {
        if (*root && !replace) {
            ret = -EEXIST;
            goto out;
        }

        *root = node_id;
        mark_inode_dirty(inode);
        ret = 0;
        goto out;
    }

    ret = vecfs_ann_write_indirect(
        inode,
        root,
        path.offsets,
        path.depth,
        node_id,
        replace);

out:
    up_write(&vi->meta_lock);
    return ret;
}

int vecfs_ann_clear_node(
    struct inode *inode,
    u32 ordinal)
{
    struct vecfs_inode *vi;
    struct vecfs_ann_path path;
    u32 *root;
    int ret;

    if (!inode)
        return -EINVAL;

    vi = VECFS_I(inode);

    ret = vecfs_ann_make_path(
        inode, ordinal, &path);
    if (ret)
        return ret;

    down_write(&vi->meta_lock);

    root = &vi->i_index[path.root];

    if (!*root) {
        ret = 0;
        goto out;
    }

    if (!path.depth) {
        *root = 0;
        mark_inode_dirty(inode);
        ret = 0;
        goto out;
    }

    ret = vecfs_ann_clear_indirect(
        inode,
        root,
        path.offsets,
        path.depth);

out:
    up_write(&vi->meta_lock);
    return ret;
}

static u64 vecfs_ann_level_span(
    struct super_block *sb,
    unsigned int depth)
{
    u64 ptrs = vecfs_ann_ptrs_per_block(sb);
    u64 span = 1;
    unsigned int i;

    for (i = 1; i < depth; i++)
        span *= ptrs;

    return span;
}

static int vecfs_ann_walk_block(
    struct inode *inode,
    u32 physical,
    unsigned int depth,
    u64 ordinal_base,
    vecfs_ann_node_iter_fn callback,
    void *private_data)
{
    struct super_block *sb = inode->i_sb;
    struct buffer_head *bh;
    __le32 *entries;
    unsigned int i;
    u64 span;
    int ret = 0;

    if (!physical || !depth)
        return 0;

    bh = sb_bread(sb, physical);
    if (!bh)
        return -EIO;

    entries = (__le32 *)bh->b_data;
    span = vecfs_ann_level_span(sb, depth);

    for (i = 0;
         i < vecfs_ann_ptrs_per_block(sb);
         i++) {
        u32 value = le32_to_cpu(entries[i]);

        if (!value)
            continue;

        if (depth == 1) {
            u64 ordinal = ordinal_base + i;

            if (value == VECFS_ANN_EMPTY_NODE_ID)
                continue;

            if (ordinal > U32_MAX) {
                ret = -EFBIG;
                break;
            }

            ret = callback(
                inode,
                (u32)ordinal,
                value,
                private_data);
            if (ret)
                break;
        } else {
            u64 child_base =
                ordinal_base + (u64)i * span;

            ret = vecfs_ann_walk_block(
                inode,
                value,
                depth - 1,
                child_base,
                callback,
                private_data);
            if (ret)
                break;
        }
    }

    brelse(bh);
    return ret;
}

int vecfs_ann_for_each_node(
    struct inode *inode,
    vecfs_ann_node_iter_fn callback,
    void *private_data)
{
    struct vecfs_inode *vi;
    struct super_block *sb;
    unsigned int i;
    int ret = 0;

    if (!inode || !callback)
        return -EINVAL;

    vi = VECFS_I(inode);
    sb = inode->i_sb;

    down_read(&vi->meta_lock);

    for (i = 0; i < VECFS_N_DIRECT; i++) {
        u32 node_id = vi->i_index[i];

        if (!node_id)
            continue;

        if (node_id == VECFS_ANN_EMPTY_NODE_ID)
            continue;

        ret = callback(
            inode,
            i,
            node_id,
            private_data);
        if (ret)
            goto out;
    }

    if (vi->i_index[VECFS_IND_BLOCK]) {
        ret = vecfs_ann_walk_block(
            inode,
            vi->i_index[VECFS_IND_BLOCK],
            1,
            VECFS_N_DIRECT,
            callback,
            private_data);
        if (ret)
            goto out;
    }

    if (vi->i_index[VECFS_DIND_BLOCK]) {
        u64 base =
            VECFS_N_DIRECT +
            vecfs_ann_ptrs_per_block(sb);

        ret = vecfs_ann_walk_block(
            inode,
            vi->i_index[VECFS_DIND_BLOCK],
            2,
            base,
            callback,
            private_data);
        if (ret)
            goto out;
    }

    if (vi->i_index[VECFS_TIND_BLOCK]) {
        u64 base =
            VECFS_N_DIRECT +
            vecfs_ann_ptrs_per_block(sb) +
            (u64)vecfs_ann_ptrs_per_block(sb) *
            vecfs_ann_ptrs_per_block(sb);

        ret = vecfs_ann_walk_block(
            inode,
            vi->i_index[VECFS_TIND_BLOCK],
            3,
            base,
            callback,
            private_data);
    }

out:
    up_read(&vi->meta_lock);
    return ret;
}

int vecfs_ann_validate_file_binding(
    struct inode *inode,
    u32 ordinal,
    u32 node_id)
{
    struct vecfs_ann_object_info info;
    int ret;

    if (!inode)
        return -EINVAL;

    ret = vecfs_ann_object_read_info(
        inode->i_sb,
        node_id,
        &info);
    if (ret)
        return ret;

    if (info.state != VECFS_ANN_OBJ_PREPARED &&
        info.state != VECFS_ANN_OBJ_LIVE)
        return -ESTALE;

    if (info.owner_type !=
        VECFS_ANN_OWNER_FILE)
        return -EPERM;

    if (info.owner_ino != inode->i_ino)
        return -EXDEV;

    if (info.owner_generation !=
        inode->i_generation)
        return -ESTALE;

    if (info.vector_ordinal != ordinal)
        return -EINVAL;

    return 0;
}

int vecfs_ann_bind_file_node(
    struct inode *inode,
    u32 ordinal,
    u32 node_id,
    bool replace)
{
    u32 cur;
    int ret;

    ret = vecfs_ann_validate_file_binding(
        inode, ordinal, node_id);
    if (ret)
        return ret;

    ret = vecfs_ann_get_node(
        inode, ordinal, &cur);
    if (!ret) {
        if (cur == node_id)
            return 0;

        if (!replace)
            return -EEXIST;
    } else if (ret != -ENOENT) {
        return ret;
    }

    return vecfs_ann_set_node(
        inode,
        ordinal,
        node_id,
        replace);
}

struct vecfs_ann_remove_entry {
    u32 ordinal;
    u32 node_id;
};

struct vecfs_ann_remove_context {
    struct vecfs_ann_remove_entry *entries;
    size_t count;
    size_t capacity;
};

static int vecfs_ann_collect_remove_entry(
    struct inode *inode,
    u32 ordinal,
    u32 node_id,
    void *private_data)
{
    struct vecfs_ann_remove_context *ctx = private_data;
    struct vecfs_ann_remove_entry *entries;

    if (ctx->count == ctx->capacity) {
        size_t new_capacity =
            ctx->capacity ?
                ctx->capacity * 2 : 16;

        entries = krealloc_array(
            ctx->entries,
            new_capacity,
            sizeof(*entries),
            GFP_NOFS);
        if (!entries)
            return -ENOMEM;

        ctx->entries = entries;
        ctx->capacity = new_capacity;
    }

    ctx->entries[ctx->count].ordinal = ordinal;
    ctx->entries[ctx->count].node_id = node_id;
    ctx->count++;

    return 0;
}

int vecfs_ann_remove_all_nodes(
    struct inode *inode)
{
    struct super_block *sb = inode->i_sb;
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *old;
    struct vecfs_ann_snapshot *next = NULL;
    struct vecfs_ann_remove_context context = {};
    u32 node_id;
    size_t i;
    int ret;

    if (!inode || !sbi)
        return -EINVAL;

    cache = sbi->s_ann_cache;

    ret = vecfs_ann_for_each_node(
        inode,
        vecfs_ann_collect_remove_entry,
        &context);
    if (ret)
        goto out_free;
    if (context.count == 0)
        goto out_free;

    /*
     * 必须同时更新磁盘对象和内存 snapshot。
     * 只有在两者都能成功的前提下才允许推进。
     *
     * 策略：
     *   1. 磁盘对象优先标记为 DELETED（尽量早提交到稳定存储）
     *   2. 内存 snapshot clone + 标记 DELETED
     *   3. meta_commit 确保在磁盘和 snapshot 一致后再持久化元数据
     *   4. publish snapshot（查询立即看不到这些节点）
     *   5. 清除 inode 映射（与 snapshot 已无关）
     */

    /*
     * Step 1: 磁盘标记。如果全部成功（或可忽略的错误），才可以
     * 继续修改 snapshot；否则 snapshot 保持不变。
     */
    for (i = 0; i < context.count; i++) {
        ret = vecfs_ann_object_mark_deleted(
            sb,
            context.entries[i].node_id);

        /*
         * 允许部分已 DELETED 的节点（-EALREADY / -ESTALE），
         * 但不允许硬错误。
         */
        if (ret && ret != -EALREADY &&
            ret != -ENOENT &&
            ret != -ESTALE)
            goto out_free;
    }

    /*
     * Step 2: 更新内存 snapshot。
     */
    if (!cache) {
        ret = -ENODEV;
        goto out_free;
    }

    mutex_lock(&cache->writer_lock);

    old = rcu_dereference_protected(
        cache->snapshot,
        lockdep_is_held(&cache->writer_lock));

    if (!old) {
        ret = -ENODEV;
        goto out_unlock;
    }

    next = vecfs_ann_snapshot_clone_locked(
        old, old->capacity);
    if (IS_ERR(next)) {
        ret = PTR_ERR(next);
        next = NULL;
        goto out_unlock;
    }

    for (i = 0; i < context.count; i++) {
        node_id = context.entries[i].node_id;

        if (!vecfs_ann_snapshot_id_valid(
                next, node_id))
            continue;

        if (vecfs_ann_node_at(next, node_id)->state ==
            VECFS_ANN_OBJ_LIVE) {
            vecfs_ann_node_mutable(next, node_id)->state =
                VECFS_ANN_OBJ_DELETED;
            if (next->live_count)
                next->live_count--;
        }
    }

    next->has_deletions = true;

    ret = vecfs_ann_meta_commit(sb, true);
    if (ret) {
        vecfs_ann_snapshot_free(next);
        next = NULL;
        goto out_unlock;
    }

    ret = vecfs_ann_cache_publish_locked(
        cache, next);
    if (ret) {
        /*
         * publish 失败：meta_commit 已经提交，磁盘元数据
         * 序号已递增，但 snapshot 未更新。
         * mount 时 validate 会修正不一致。
         */
        vecfs_ann_snapshot_free(next);
        next = NULL;
        goto out_unlock;
    }

    /* publish 成功，next 所有权已转移给 cache */
    next = NULL;

out_unlock:
    mutex_unlock(&cache->writer_lock);

    if (next)
        vecfs_ann_snapshot_free(next);
    if (ret)
        goto out_free;

    /*
     * Step 3: 清除 inode 映射。
     */
    for (i = 0; i < context.count; i++) {
        int clear_ret;

        clear_ret = vecfs_ann_clear_node(
            inode,
            context.entries[i].ordinal);
        if (clear_ret) {
            pr_warn(
                "vecfs: clear_node ino=%lu ordinal=%u "
                "failed: %d\n",
                inode->i_ino,
                context.entries[i].ordinal,
                clear_ret);
        }
    }

out_free:
    kfree(context.entries);
    return ret;
}

int vecfs_ann_get_block(
    struct inode *inode,
    sector_t ordinal,
    void *vector,
    int create)
{
    u32 node_id;
    int ret;

    if (ordinal > U32_MAX)
        return -EFBIG;

    if (create)
        return -EOPNOTSUPP;

    ret = vecfs_ann_get_node(
        inode,
        (u32)ordinal,
        &node_id);

    return ret;
}

int vecfs_ann_alloc_block(
    struct inode *inode,
    u32 *block)
{
    return vecfs_alloc_fs_block(
        inode, block);
}

void vecfs_ann_free_block(
    struct inode *inode,
    u32 block)
{
    vecfs_free_block(inode, block);
}

int vecfs_ann_truncate_blocks(
    struct inode *inode,
    loff_t newsize)
{
    if (!inode)
        return -EINVAL;

    /*
     * ANN 节点不再与文件大小或文件逻辑块绑定。
     * 只有 eviction/delete 路径允许通过 newsize=0 清空。
     */
    if (newsize != 0)
        return -EOPNOTSUPP;

    return vecfs_ann_remove_all_nodes(inode);
}
