#ifndef _VECFS_ANN_REBUILD_H
#define _VECFS_ANN_REBUILD_H

#include <linux/atomic.h>
#include <linux/mutex.h>
#include <linux/types.h>
#include <linux/wait.h>
#include <linux/workqueue.h>

struct super_block;

enum vecfs_ann_rebuild_state {
    VECFS_ANN_REBUILD_IDLE = 0,
    VECFS_ANN_REBUILD_RUNNING,
    VECFS_ANN_REBUILD_DONE,
    VECFS_ANN_REBUILD_FAILED,
    VECFS_ANN_REBUILD_CANCELLED,
};

#define VECFS_REBUILD_F_SYNC              (1U << 0)
#define VECFS_REBUILD_F_PRESERVE_LEVELS   (1U << 1)
#define VECFS_REBUILD_F_ASYNC             (1U << 2)
/* 并行构图：workqueue 多 worker 并发插入（hnswlib 式每节点链锁）。 */
#define VECFS_REBUILD_F_PARALLEL          (1U << 3)

struct vecfs_ann_rebuild_report {
    u64 source_live_nodes;
    u64 source_deleted_nodes;

    u64 rebuilt_nodes;
    u64 removed_nodes;

    u64 old_epoch;
    u64 new_epoch;

    u32 old_arena_inode;
    u32 new_arena_inode;

    u64 duration_ns;
};

struct vecfs_ann_rebuild_async {
    struct mutex lock;
    wait_queue_head_t wait;
    struct work_struct work;

    struct super_block *sb;

    u64 request_id;
    u32 flags;

    atomic_t cancel;

    enum vecfs_ann_rebuild_state state;
    int status;

    struct vecfs_ann_rebuild_report report;
    bool initialized;
};

int vecfs_ann_rebuild_async_init(
    struct super_block *sb);

void vecfs_ann_rebuild_async_destroy(
    struct super_block *sb);

int vecfs_ann_rebuild(
    struct super_block *sb,
    u32 flags,
    struct vecfs_ann_rebuild_report *report);

int vecfs_ann_rebuild_submit(
    struct super_block *sb,
    u32 flags,
    u64 *request_id);

int vecfs_ann_rebuild_cancel(
    struct super_block *sb,
    u64 request_id);

int vecfs_ann_rebuild_get_status(
    struct super_block *sb,
    u64 request_id,
    enum vecfs_ann_rebuild_state *state,
    int *status,
    struct vecfs_ann_rebuild_report *report);

bool vecfs_ann_rebuild_should_cancel(
    struct super_block *sb);

#endif
