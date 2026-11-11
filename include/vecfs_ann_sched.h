#ifndef _VECFS_ANN_SCHED_H
#define _VECFS_ANN_SCHED_H

#include <linux/types.h>

struct super_block;

/*
 * 定义在 vecfs_ioctl.h 中；这里只前置声明指针成员，避免把 SHIELD 的
 * UAPI 宏提前拉到 vecfs.h（与 vecfs_shield.h 的枚举同名冲突）。
 */
struct vecfs_ioctl_sched_trace;

/*
 * RTANN 调度器（workqueue 实现）。
 *
 * 检索路径是「纯净」的 bare HNSW：不携带 stale/valid/invalid/deleted/owner/
 * SHIELD/path 语义，inner loop 只有 visited + 距离 + 堆维护。调度器把一次
 * base-layer 搜索分解为 microstep（每次展开一个 level-0 candidate），用两个
 * unbound workqueue 做并发与优先级隔离：
 *
 *   norm_wq  : WQ_UNBOUND —— 普通微批
 *   boost_wq : WQ_UNBOUND | WQ_HIGHPRI —— victim 补偿（RTC 预算）
 *
 * 并发上限由 workqueue 的 max_active 给出（静态 cohort），动态塑造通过
 * effective_high 与 active 计数在提交路径做节流；victim 由 EWMA 进度率识别。
 *
 * 提交是同步的：submit() 把所有查询跑完后返回。快照在提交时 kref 固定一次、
 * 全部查询完成后归还一次；worker（可睡眠、不在 RCU 临界区）因此可以直接使用
 * 快照指针而不需要 rcu_read_lock。
 */
struct vecfs_ann_sched_req {
    u32 n_queries;
    u32 dim;
    u32 k;
    u32 ef;
    u32 sched_mode;

    /* [n_queries, dim] fp32 */
    const float *queries;

    /* [n_queries, k] node id */
    u32 *results;

    /* [n_queries, k] distance_bits；可为 NULL */
    u32 *distances;

    /* [n_queries] 距离计算次数 */
    u64 *dist_counts;

    /* [n_queries] 调度轨迹；可为 NULL */
    struct vecfs_ioctl_sched_trace *traces;

    /* 输出：首个错误码（0 = 全部成功） */
    int status;
};

int vecfs_ann_sched_init(struct super_block *sb);
void vecfs_ann_sched_destroy(struct super_block *sb);

/*
 * 同步提交一批查询。返回 0 或负 errno；失败时 req->status 可能带有
 * 首个失败的查询错误码。成功返回后 results/distances/dist_counts/traces
 * 均已填充。
 */
int vecfs_ann_sched_submit(
    struct super_block *sb,
    struct vecfs_ann_sched_req *req);

#endif
