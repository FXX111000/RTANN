// SPDX-License-Identifier: GPL-2.0

#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/rcupdate.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_cache.h"
#include "../include/vecfs_ann_label.h"
#include "../include/vecfs_ann_maintenance.h"
#include "../include/vecfs_ann_validate.h"
#include "../include/vecfs_path.h"

int vecfs_ann_validate_and_refresh(
    struct super_block *sb,
    u32 flags,
    struct vecfs_ann_validate_report *report)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *new_snapshot = NULL;
    struct vecfs_ann_snapshot *old_snapshot;
    bool repair;
    int path_ret;
    int validate_ret;
    int ret;

    if (!sb || !report)
        return -EINVAL;

    sbi = VECFS_SB(sb);

    if (!sbi ||
        !sbi->s_ann_cache)
        return -ENODEV;

    repair =
        flags &
        VECFS_ANN_VALIDATE_F_REPAIR;

    if (repair &&
        sb_rdonly(sb))
        return -EROFS;

    cache = sbi->s_ann_cache;

    mutex_lock(
        &cache->writer_lock);

    validate_ret = vecfs_ann_validate(
        sb,
        flags,
        report);

    if (validate_ret &&
        validate_ret != -EUCLEAN) {
        ret = validate_ret;
        goto out_unlock;
    }

    if (!repair) {
        ret = validate_ret;
        goto out_unlock;
    }

    new_snapshot =
        vecfs_ann_snapshot_build(sb);

    if (IS_ERR(new_snapshot)) {
        ret = PTR_ERR(new_snapshot);
        new_snapshot = NULL;
        goto out_unlock;
    }

    ret = vecfs_ann_label_rebuild_locked(
        sb,
        new_snapshot);
    if (ret)
        goto out_free_snapshot;

    old_snapshot =
        rcu_dereference_protected(
            cache->snapshot,
            lockdep_is_held(
                &cache->writer_lock));

    rcu_assign_pointer(
        cache->snapshot,
        new_snapshot);

    new_snapshot = NULL;

    /*
     * 旧快照经 kref 归还（call_rcu 延迟回收），不再直接 free——RTANN
     * 调度器可能正持有 kref，直接 free 是 use-after-free。
     */
    vecfs_ann_snapshot_put(old_snapshot);

    ret = validate_ret;
    goto out_unlock;

out_free_snapshot:
    vecfs_ann_snapshot_free(
        new_snapshot);

out_unlock:
    mutex_unlock(
        &cache->writer_lock);

    /*
     * 路径缓存与 ANN writer lock 无关，必须在释放 writer lock 后
     * 重建，避免目录扫描过程中产生不必要的长时间 writer 阻塞。
     */
    if (repair &&
        (!ret || ret == -EUCLEAN)) {
        path_ret =
            vecfs_path_cache_rebuild(sb);

        if (path_ret)
            pr_warn(
                "vecfs: path cache rebuild after "
                "validation failed: %d\n",
                path_ret);
    }

    return ret;
}
