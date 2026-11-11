#ifndef _VECFS_ANN_RUNTIME_H
#define _VECFS_ANN_RUNTIME_H

#include <linux/atomic.h>
#include <linux/types.h>

struct super_block;

struct vecfs_ann_runtime {
    /*
     * 后续查询的默认 ef_search。
     *
     * 查询开始时通过 atomic_read() 捕获一次，查询过程中不再访问。
     */
    atomic_t default_ef_search;

    atomic_t maximum_queries;
};

int vecfs_ann_runtime_init(struct super_block *sb);

void vecfs_ann_runtime_destroy(struct super_block *sb);

int vecfs_ann_runtime_set_ef(
    struct super_block *sb,
    u32 ef_search,
    bool persist);

u32 vecfs_ann_runtime_capture_ef(
    struct super_block *sb,
    u32 request_ef,
    u32 k);

#endif
