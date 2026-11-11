// SPDX-License-Identifier: GPL-2.0

#include <linux/errno.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/workqueue.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_rebuild.h"

static void vecfs_ann_rebuild_workfn(
    struct work_struct *work)
{
    struct vecfs_ann_rebuild_async *async =
        container_of(
            work,
            struct vecfs_ann_rebuild_async,
            work);

    struct vecfs_ann_rebuild_report report;
    int ret;

    memset(&report, 0, sizeof(report));

    ret = vecfs_ann_rebuild(
        async->sb,
        async->flags & ~VECFS_REBUILD_F_ASYNC,
        &report);

    mutex_lock(&async->lock);

    async->status = ret;
    async->report = report;

    if (atomic_read(&async->cancel) &&
        ret == -ECANCELED)
        async->state =
            VECFS_ANN_REBUILD_CANCELLED;
    else if (ret)
        async->state =
            VECFS_ANN_REBUILD_FAILED;
    else
        async->state =
            VECFS_ANN_REBUILD_DONE;

    mutex_unlock(&async->lock);

    wake_up_all(&async->wait);
}

int vecfs_ann_rebuild_async_init(
    struct super_block *sb)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_rebuild_async *async;

    if (!sb)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return -EINVAL;

    async = &sbi->s_ann_rebuild_async;

    memset(async, 0, sizeof(*async));

    mutex_init(&async->lock);
    init_waitqueue_head(&async->wait);
    INIT_WORK(&async->work,
              vecfs_ann_rebuild_workfn);

    async->sb = sb;
    async->state =
        VECFS_ANN_REBUILD_IDLE;

    atomic_set(&async->cancel, 0);
    async->initialized = true;

    return 0;
}

void vecfs_ann_rebuild_async_destroy(
    struct super_block *sb)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_rebuild_async *async;

    if (!sb)
        return;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return;

    async = &sbi->s_ann_rebuild_async;

    if (!async->initialized)
        return;

    atomic_set(&async->cancel, 1);
    cancel_work_sync(&async->work);

    mutex_lock(&async->lock);

    async->state =
        VECFS_ANN_REBUILD_IDLE;
    async->status = 0;
    async->initialized = false;

    mutex_unlock(&async->lock);

    mutex_destroy(&async->lock);
}

bool vecfs_ann_rebuild_should_cancel(
    struct super_block *sb)
{
    struct vecfs_sb_info *sbi;

    if (!sb)
        return true;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return true;

    return atomic_read(
        &sbi->s_ann_rebuild_async.cancel) != 0;
}

int vecfs_ann_rebuild_submit(
    struct super_block *sb,
    u32 flags,
    u64 *request_id)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_rebuild_async *async;

    if (!sb || !request_id)
        return -EINVAL;

    if (sb_rdonly(sb))
        return -EROFS;

    if (flags &
        ~(VECFS_REBUILD_F_SYNC |
          VECFS_REBUILD_F_PRESERVE_LEVELS |
          VECFS_REBUILD_F_ASYNC |
          VECFS_REBUILD_F_PARALLEL))
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return -EINVAL;

    async = &sbi->s_ann_rebuild_async;

    if (!async->initialized)
        return -ENODEV;

    mutex_lock(&async->lock);

    if (async->state ==
            VECFS_ANN_REBUILD_RUNNING) {
        mutex_unlock(&async->lock);
        return -EBUSY;
    }

    async->request_id++;

    if (!async->request_id)
        async->request_id = 1;

    async->flags =
        flags | VECFS_REBUILD_F_ASYNC;

    async->status = -EINPROGRESS;
    memset(&async->report, 0,
           sizeof(async->report));

    atomic_set(&async->cancel, 0);

    async->state =
        VECFS_ANN_REBUILD_RUNNING;

    *request_id = async->request_id;

    mutex_unlock(&async->lock);

    if (!queue_work(
            system_long_wq,
            &async->work)) {
        mutex_lock(&async->lock);

        async->state =
            VECFS_ANN_REBUILD_FAILED;
        async->status = -EALREADY;

        mutex_unlock(&async->lock);
        return -EALREADY;
    }

    return 0;
}

int vecfs_ann_rebuild_cancel(
    struct super_block *sb,
    u64 request_id)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_rebuild_async *async;

    if (!sb || !request_id)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return -EINVAL;

    async = &sbi->s_ann_rebuild_async;

    mutex_lock(&async->lock);

    if (async->request_id != request_id) {
        mutex_unlock(&async->lock);
        return -ENOENT;
    }

    if (async->state !=
        VECFS_ANN_REBUILD_RUNNING) {
        mutex_unlock(&async->lock);
        return -EALREADY;
    }

    atomic_set(&async->cancel, 1);

    mutex_unlock(&async->lock);
    return 0;
}

int vecfs_ann_rebuild_get_status(
    struct super_block *sb,
    u64 request_id,
    enum vecfs_ann_rebuild_state *state,
    int *status,
    struct vecfs_ann_rebuild_report *report)
{
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_rebuild_async *async;

    if (!sb || !state || !status || !report)
        return -EINVAL;

    sbi = VECFS_SB(sb);
    if (!sbi)
        return -EINVAL;

    async = &sbi->s_ann_rebuild_async;

    mutex_lock(&async->lock);

    if (request_id &&
        async->request_id != request_id) {
        mutex_unlock(&async->lock);
        return -ENOENT;
    }

    *state = async->state;
    *status = async->status;
    *report = async->report;

    mutex_unlock(&async->lock);
    return 0;
}
