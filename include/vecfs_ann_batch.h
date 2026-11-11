#ifndef _VECFS_ANN_BATCH_H
#define _VECFS_ANN_BATCH_H

#include <linux/types.h>

#include "vecfs_ann_build.h"

struct inode;
struct super_block;

#define VECFS_ANN_BATCH_MAX_ITEMS 4096U

struct vecfs_ann_batch_item {
    struct vecfs_ann_insert_request request;
    struct vecfs_ann_insert_result result;
};

int vecfs_ann_batch_insert_file(
    struct inode *inode,
    struct vecfs_ann_batch_item *items,
    u32 count);

/*
 * 事务式发布（C4）：发布前版本复核回调。在 vecfs_ann_batch_insert_file_guarded
 * 的发布临界区（writer_lock 持有期间、publish_locked 之前）调用；返回 false
 * 表示版本已过期或被取消，本事务将被回滚——新节点不进入公共图（消除
 * start_invalid 泄露），函数以 -ESTALE 返回。回调在 writer_lock 下执行，可
 * 安全获取文件级 fs->lock（写路径只在释放 fs->lock 后才获取 writer_lock，
 * 不构成锁环）。
 */
typedef bool (*vecfs_ann_batch_publish_guard_fn)(
    struct inode *inode,
    void *ctx);

int vecfs_ann_batch_insert_file_guarded(
    struct inode *inode,
    struct vecfs_ann_batch_item *items,
    u32 count,
    vecfs_ann_batch_publish_guard_fn guard,
    void *ctx);

/*
 * 多文件批量插入（按窗口聚批，消除每文件一次 O(N) 快照克隆）。
 *
 * files[] 每个文件独立校验/回滚，但共享一次 snapshot clone、一次
 * build_context、一次 persist 与一次 publish。关键约束：所有文件的
 * prepare_objects（node id 分配）统一在 clone 之前完成，故 final_capacity
 * = 跨文件最大 node id + 1；publish_guard 逐文件调用（每文件一个 guard_ctx，
 * 对应其自身的 derive_version）。
 *
 * total_items 为跨文件 item 总数，必须 ≤ VECFS_ANN_BATCH_MAX_ITEMS。任一文件
 * publish_guard 失败（返回 false）则整批回滚并以 -ESTALE 返回；其余错误同样
 * 整批回滚（已 prepare 的对象统一 mark_deleted）。所有文件须属于同一 sb。
 */
struct vecfs_ann_batch_file {
    struct inode *inode;
    struct vecfs_ann_batch_item *items;
    u32 count;
    void *guard_ctx;
};

int vecfs_ann_batch_insert_files_many(
    struct super_block *sb,
    struct vecfs_ann_batch_file *files,
    u32 n_files,
    u32 total_items,
    vecfs_ann_batch_publish_guard_fn publish_guard,
    bool copy_full);

#endif
