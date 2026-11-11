// SPDX-License-Identifier: GPL-2.0

#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/kernel.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_meta.h"
#include "../include/vecfs_ann_runtime.h"
#include "../include/vecfs_ann_search.h"

int vecfs_ann_runtime_init(struct super_block *sb)
{
    struct vecfs_sb_info *sbi;

    if (!sb)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return -EINVAL;

    atomic_set(
        &sbi->s_ann_runtime.default_ef_search,
        sbi->s_ann.config.default_ef_search);

    atomic_set(
        &sbi->s_ann_runtime.maximum_queries,
        max_t(int, 1, num_online_cpus()));

    return 0;
}

void vecfs_ann_runtime_destroy(struct super_block *sb)
{
    struct vecfs_sb_info *sbi;

    if (!sb)
        return;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return;

    atomic_set(
        &sbi->s_ann_runtime.default_ef_search,
        0);

    atomic_set(
        &sbi->s_ann_runtime.maximum_queries,
        0);
}

int vecfs_ann_runtime_set_ef(
    struct super_block *sb,
    u32 ef_search,
    bool persist)
{
    struct vecfs_sb_info *sbi;
    int ret = 0;

    if (!sb)
        return -EINVAL;

    if (!ef_search ||
        ef_search > VECFS_ANN_SEARCH_MAX_EF)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return -EINVAL;

    /*
     * 只影响之后开始的查询。已经开始的查询已捕获旧值。
     */
    atomic_set(
        &sbi->s_ann_runtime.default_ef_search,
        ef_search);

    if (!persist)
        return 0;

    mutex_lock(&sbi->s_ann.lock);

    sbi->s_ann.config.default_ef_search =
        ef_search;
    sbi->s_ann.dirty = true;

    mutex_unlock(&sbi->s_ann.lock);

    if (!sb_rdonly(sb))
        ret = vecfs_ann_meta_commit(sb, true);

    return ret;
}

u32 vecfs_ann_runtime_capture_ef(
    struct super_block *sb,
    u32 request_ef,
    u32 k)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    u32 ef;

    /*
     * 这是整个查询唯一一次读取默认 ef_search。
     */
    if (request_ef)
        ef = request_ef;
    else
        ef = atomic_read(
            &sbi->s_ann_runtime.default_ef_search);

    return max(ef, k);
}
