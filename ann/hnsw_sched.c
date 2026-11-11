// SPDX-License-Identifier: GPL-2.0

/*
 * RTANN 调度器（workqueue 实现）。
 *
 * 检索路径是「纯净」bare HNSW：不访问 state/owner/stale/valid/SHIELD/path，
 * inner loop 只有 visited 判定、一次距离计算、两个堆的维护。调度器把一次
 * base-layer 搜索拆成 microstep（每次展开一个 level-0 candidate），交给三个
 * unbound workqueue 并发执行：
 *
 *   norm_wq  : 普通微批（WQ_UNBOUND）
 *   boost_wq : victim 补偿微批（WQ_UNBOUND | WQ_HIGHPRI）
 *   tail_wq  : tail pod 救援队列，单并发（WQ_UNBOUND | WQ_HIGHPRI, max_active=1）
 *
 * P1 microstep   —— 每个 work item 执行至多 budget 个 microstep 后 requeue，
 *                   用 workqueue 的抢占/公平调度实现细粒度交织。
 * P2 EWMA 进度率  —— 每 slice 结束采样一次进度率，作为争用代理信号；并据此
 *                   自适应调整普通 slice 的预算，对准 ~80us 目标时长。
 * P3 补偿 + 塑造  —— 进度率低于 cohort 平均的 victim 迁到 boost_wq 以固定
 *                   boost 预算补偿；争用持续时把并发上界 effective_high 缩到
 *                   cohort_low（并按 victim 数进一步缩到 high/2、high/4）。
 *   tail pod      —— victim 连续低速 slice 数达阈值后迁入 tail_wq，以 RTC
 *                   预算一次性收尾（V3 救援层）。
 *   低负载 RTC    —— 在飞查询不超过 cohort_low 时直接 run-to-completion，
 *                   避免无排队时白付 RR 切换开销。
 *   OFF 基线      —— sched_mode=0 不做 microstep/塑造/补偿/tail，每个 work
 *                   item 以 U32_MAX 预算 run-to-completion（并行裸跑、零策略），
 *                   作为对照 hnswlib num_threads=N 的消融基线。
 *
 * 提交是同步的：submit() 把所有查询跑完返回。快照在提交时 kref 固定一次、
 * 完成后归还一次（worker 可睡眠、不在 RCU 临界区，可直接使用快照指针）。
 */

#include <linux/completion.h>
#include <linux/cpumask.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/ktime.h>
#include <linux/math64.h>
#include <linux/prefetch.h>
#include <linux/rcupdate.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/timekeeping.h>
#include <linux/vmalloc.h>
#include <linux/workqueue.h>

#include "../include/vecfs.h"
#include "../include/vecfs_ann_cache.h"
#include "../include/vecfs_ann_distance.h"
#include "../include/vecfs_ann_heap.h"
#include "../include/vecfs_ann_runtime.h"
#include "../include/vecfs_ann_sched.h"
#include "../include/vecfs_ann_search.h"
#include "../include/vecfs_ann_visited.h"
#include "../include/vecfs_ioctl.h"

#define VECFS_ANN_FLOAT_MAX 3.402823466e+38F

/* 单个普通 slice 的目标运行时长（自适应预算对准它，P2）。 */
#define VECFS_ANN_SCHED_TARGET_NS 80000ULL

/* 自适应 microstep 预算边界（P2，KVFS 同款 32..96）。 */
#define VECFS_ANN_SCHED_BUDGET_INIT 48U
#define VECFS_ANN_SCHED_BUDGET_MIN  32U
#define VECFS_ANN_SCHED_BUDGET_MAX  96U

/* 低负载 / tail pod 的 run-to-completion 预算（P3）。 */
#define VECFS_ANN_SCHED_BUDGET_RTC 4096U

/* victim 补偿 slice 的固定预算（P3，论文 B_boost）。 */
#define VECFS_ANN_SCHED_BOOST_BUDGET 192U

/* Q16 定点进度率基（2^16），与 update_rate 的 65536000000 一致。 */
#define VECFS_ANN_SCHED_RATE_ONE (1ULL << 16)

/* EWMA 平滑因子 alpha = 1 / EWMA_ALPHA。 */
#define VECFS_ANN_SCHED_EWMA_ALPHA 8U

/* candidate 堆上界 = CAND_FACTOR * m0 * ef，溢出即 drop-new。 */
#define VECFS_ANN_SCHED_CAND_FACTOR 4U

/* 连续争用/缓解样本数达到该值才切换并发上界（滞回）。 */
#define VECFS_ANN_SCHED_CONTEND_HYST 4U

/* victim 连续低速 slice 数达到该值才迁入 tail pod（P3 V3）。 */
#define VECFS_ANN_SCHED_TAIL_ENTER_STREAK 3U

struct vecfs_ann_sched_ctx {
    struct list_head pool_node;

    struct vecfs_ann_visited visited;

    /* MIN 堆：最近 candidate 在顶。 */
    struct vecfs_ann_heap candidates;

    /* MAX 堆：最远 result 在顶。 */
    struct vecfs_ann_heap results;

    struct vecfs_ann_heap_item *sorted;

    u32 visited_capacity;
    u32 candidate_capacity;
    u32 result_capacity;
    u32 sorted_capacity;
};

struct vecfs_ann_sched_pool {
    struct mutex lock;
    wait_queue_head_t wait;
    struct list_head free;

    u32 total;
    u32 free_count;
    u32 maximum;

    bool shutting_down;
};

struct vecfs_ann_sched;
struct vecfs_ann_sched_batch;

struct vecfs_ann_sched_query {
    struct work_struct work;

    struct vecfs_ann_sched *sched;
    struct vecfs_ann_sched_batch *batch;
    struct vecfs_ann_snapshot *snapshot;
    struct vecfs_ann_sched_ctx *ctx;

    u32 index;
    const float *query;

    u32 k;
    u32 ef;

    /* 搜索状态。 */
    u32 entry_node;
    float entry_distance;
    bool upper_done;
    bool base_done;
    float lower_bound;

    /* RTANN 记账。 */
    u32 progress_units;
    u32 dist_compute_cnt;

    u64 avg_rate_q16;
    u64 min_rate_q16;
    u64 max_rate_q16;

    u32 dispatches;
    u32 requeues;
    u32 zero_progress_batches;
    u32 stall_count;

    bool victim;
    u32 victim_entries;
    u32 victim_exits;

    /* tail pod（P3 V3）：victim 持续低速后迁入专用 rescue 队列。 */
    bool in_tail;
    u32 tail_low_streak;

    u32 wave_id_submit;
    u32 wave_id_admit;
    u32 wave_id_finish;

    u32 tail_reason;

    /* 时间戳（ns）。 */
    u64 submitted_ns;
    u64 admitted_ns;
    u64 first_run_ns;
    u64 finished_ns;
    u64 enqueue_ns;
    bool enqueue_boost;

    u64 run_ns_acc;
    u64 queue_wait_acc;
    u64 normal_wait_acc;
    u64 boost_wait_acc;
    u64 ctx_wait_acc;

    int status;
};

struct vecfs_ann_sched_batch {
    struct completion done;
    atomic_t remaining;

    struct vecfs_ann_snapshot *snapshot;
    struct vecfs_ann_sched_req *req;
    struct vecfs_ann_sched_query *queries;
};

struct vecfs_ann_sched {
    struct super_block *sb;
    struct vecfs_ann_distance *distance;

    struct workqueue_struct *norm_wq;
    struct workqueue_struct *boost_wq;
    struct workqueue_struct *tail_wq;

    struct vecfs_ann_sched_pool pool;

    /* 并发塑造门：提交路径在 active >= effective_high 时等待。 */
    atomic_t active;
    wait_queue_head_t gate_wait;

    int cohort_high;
    int cohort_low;
    int cohort_min_high;
    int effective_high;

    /* 当前 victim 查询数（P3 shaping：缩 cohort 上界）。 */
    atomic_t victim_count;

    /* cohort 平均进度率 EWMA 与争用滞回计数。 */
    spinlock_t rate_lock;
    u64 cohort_rate_q16;
    int contend_streak;

    atomic_t wave_id;
};

static void vecfs_ann_sched_finish_query(
    struct vecfs_ann_sched_query *query);

static void vecfs_ann_sched_ctx_put(
    struct vecfs_ann_sched *sched,
    struct vecfs_ann_sched_ctx *ctx);

/* ------------------------------------------------------------------ */
/* 上下文池                                                           */
/* ------------------------------------------------------------------ */

static int vecfs_ann_sched_ctx_init(
    struct vecfs_ann_sched_ctx *ctx,
    u32 visited_capacity,
    u32 candidate_capacity,
    u32 result_capacity)
{
    int ret;

    memset(ctx, 0, sizeof(*ctx));
    INIT_LIST_HEAD(&ctx->pool_node);

    ret = vecfs_ann_visited_init(
        &ctx->visited,
        visited_capacity,
        GFP_KERNEL);
    if (ret)
        return ret;

    ret = vecfs_ann_heap_init(
        &ctx->candidates,
        candidate_capacity,
        VECFS_ANN_HEAP_MIN,
        GFP_KERNEL);
    if (ret)
        goto fail_visited;

    ret = vecfs_ann_heap_init(
        &ctx->results,
        result_capacity,
        VECFS_ANN_HEAP_MAX,
        GFP_KERNEL);
    if (ret)
        goto fail_candidates;

    ctx->sorted = kvmalloc_array(
        result_capacity,
        sizeof(*ctx->sorted),
        GFP_KERNEL);
    if (!ctx->sorted) {
        ret = -ENOMEM;
        goto fail_results;
    }

    ctx->visited_capacity = visited_capacity;
    ctx->candidate_capacity = candidate_capacity;
    ctx->result_capacity = result_capacity;
    ctx->sorted_capacity = result_capacity;

    return 0;

fail_results:
    vecfs_ann_heap_destroy(&ctx->results);
fail_candidates:
    vecfs_ann_heap_destroy(&ctx->candidates);
fail_visited:
    vecfs_ann_visited_destroy(&ctx->visited);
    return ret;
}

static void vecfs_ann_sched_ctx_destroy(
    struct vecfs_ann_sched_ctx *ctx)
{
    if (!ctx)
        return;

    kvfree(ctx->sorted);
    vecfs_ann_heap_destroy(&ctx->results);
    vecfs_ann_heap_destroy(&ctx->candidates);
    vecfs_ann_visited_destroy(&ctx->visited);
    memset(ctx, 0, sizeof(*ctx));
}

/*
 * 保证 ctx 容量满足需求。容量增长（图扩容 / ef 增大）时重建；不变则直接复用。
 */
static int vecfs_ann_sched_ctx_prepare(
    struct vecfs_ann_sched_ctx *ctx,
    u32 visited_capacity,
    u32 candidate_capacity,
    u32 result_capacity)
{
    struct vecfs_ann_sched_ctx replacement;
    int ret;

    if (ctx->visited_capacity >= visited_capacity &&
        ctx->candidate_capacity >= candidate_capacity &&
        ctx->result_capacity >= result_capacity) {
        vecfs_ann_visited_reset(&ctx->visited);
        vecfs_ann_heap_reset(&ctx->candidates);
        vecfs_ann_heap_reset(&ctx->results);
        return 0;
    }

    visited_capacity = max(visited_capacity, ctx->visited_capacity);
    candidate_capacity = max(candidate_capacity, ctx->candidate_capacity);
    result_capacity = max(result_capacity, ctx->result_capacity);

    ret = vecfs_ann_sched_ctx_init(
        &replacement,
        visited_capacity,
        candidate_capacity,
        result_capacity);
    if (ret)
        return ret;

    vecfs_ann_sched_ctx_destroy(ctx);
    *ctx = replacement;
    INIT_LIST_HEAD(&ctx->pool_node);

    return 0;
}

static void vecfs_ann_sched_pool_destroy(
    struct vecfs_ann_sched *sched)
{
    struct vecfs_ann_sched_pool *pool = &sched->pool;
    struct vecfs_ann_sched_ctx *ctx;
    struct vecfs_ann_sched_ctx *tmp;

    mutex_lock(&pool->lock);
    pool->shutting_down = true;
    mutex_unlock(&pool->lock);

    wake_up_all(&pool->wait);

    mutex_lock(&pool->lock);

    list_for_each_entry_safe(
        ctx,
        tmp,
        &pool->free,
        pool_node) {
        list_del_init(&ctx->pool_node);
        vecfs_ann_sched_ctx_destroy(ctx);
        kfree(ctx);
    }

    mutex_unlock(&pool->lock);
    mutex_destroy(&pool->lock);
}

static int vecfs_ann_sched_ctx_acquire(
    struct vecfs_ann_sched *sched,
    u32 visited_capacity,
    u32 candidate_capacity,
    u32 result_capacity,
    struct vecfs_ann_sched_ctx **result)
{
    struct vecfs_ann_sched_pool *pool = &sched->pool;
    struct vecfs_ann_sched_ctx *ctx;
    int ret;

retry:
    mutex_lock(&pool->lock);

    if (pool->shutting_down) {
        mutex_unlock(&pool->lock);
        return -ESHUTDOWN;
    }

    if (!list_empty(&pool->free)) {
        ctx = list_first_entry(
            &pool->free,
            struct vecfs_ann_sched_ctx,
            pool_node);

        list_del_init(&ctx->pool_node);
        pool->free_count--;

        mutex_unlock(&pool->lock);

        ret = vecfs_ann_sched_ctx_prepare(
            ctx,
            visited_capacity,
            candidate_capacity,
            result_capacity);
        if (ret) {
            vecfs_ann_sched_ctx_put(sched, ctx);
            return ret;
        }

        *result = ctx;
        return 0;
    }

    if (pool->total < pool->maximum) {
        pool->total++;
        mutex_unlock(&pool->lock);

        ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
        if (!ctx) {
            ret = -ENOMEM;
            goto undo_count;
        }

        ret = vecfs_ann_sched_ctx_init(
            ctx,
            visited_capacity,
            candidate_capacity,
            result_capacity);
        if (ret) {
            kfree(ctx);
            goto undo_count;
        }

        *result = ctx;
        return 0;
    }

    mutex_unlock(&pool->lock);

    wait_event(
        pool->wait,
        READ_ONCE(pool->shutting_down) ||
        READ_ONCE(pool->free_count));

    goto retry;

undo_count:
    mutex_lock(&pool->lock);
    pool->total--;
    mutex_unlock(&pool->lock);
    wake_up(&pool->wait);
    return ret;
}

static void vecfs_ann_sched_ctx_put(
    struct vecfs_ann_sched *sched,
    struct vecfs_ann_sched_ctx *ctx)
{
    struct vecfs_ann_sched_pool *pool = &sched->pool;

    if (!ctx)
        return;

    mutex_lock(&pool->lock);

    if (pool->shutting_down) {
        if (pool->total)
            pool->total--;

        mutex_unlock(&pool->lock);

        vecfs_ann_sched_ctx_destroy(ctx);
        kfree(ctx);
        wake_up_all(&pool->wait);
        return;
    }

    list_add(&ctx->pool_node, &pool->free);
    pool->free_count++;

    mutex_unlock(&pool->lock);
    wake_up(&pool->wait);
}

/* ------------------------------------------------------------------ */
/* 距离位模式                                                         */
/* ------------------------------------------------------------------ */

static __always_inline u32 vecfs_ann_sched_float_bits(float value)
{
    u32 bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/* ------------------------------------------------------------------ */
/* 纯净 microstep 执行器                                              */
/* ------------------------------------------------------------------ */

static __always_inline void
vecfs_ann_sched_prefetch_neighbor(
    const struct vecfs_ann_snapshot *snapshot,
    struct vecfs_ann_visited *visited,
    u32 node_id)
{
    prefetch(&visited->generation[node_id]);
    prefetch(vecfs_ann_snapshot_vector(snapshot, node_id));
}

/*
 * 上层贪心下降（纯净路径，等价 search.c 的 vecfs_ann_greedy_upper）。
 */
static void vecfs_ann_sched_greedy_upper(
    struct vecfs_ann_sched_query *q,
    const struct vecfs_ann_distance_session *session)
{
    const struct vecfs_ann_snapshot *snapshot = q->snapshot;
    u32 cur = q->entry_node;
    float curd = q->entry_distance;
    int level;

    for (level = snapshot->max_level;
         level > 0;
         level--) {
        bool changed = true;

        while (changed) {
            const u32 *links;
            u16 count;
            u16 i;

            changed = false;

            links = vecfs_ann_upper_links(
                snapshot, cur, level);
            count = vecfs_ann_upper_count(
                snapshot, cur, level);

            if (unlikely(!links))
                break;

            for (i = 0; i < count; i++) {
                u32 candidate = links[i];
                float candidate_distance;

                candidate_distance =
                    vecfs_ann_distance_calculate(
                        session,
                        q->query,
                        vecfs_ann_snapshot_vector(
                            snapshot,
                            candidate));

                q->dist_compute_cnt++;

                if (candidate_distance < curd) {
                    curd = candidate_distance;
                    cur = candidate;
                    changed = true;
                }
            }
        }
    }

    q->entry_node = cur;
    q->entry_distance = curd;
}

/*
 * 执行至多 budget 个 base-layer microstep。首次调用先做上层下降并初始化
 * base 状态。返回 true 表示搜索完成（base_done）。
 *
 * 每个 microstep：弹出最近 candidate，展开其 level-0 邻居，计算距离并
 * 维护 candidate（MIN，有界 drop-new）/ results（MAX，size==ef）两个堆。
 */
static bool vecfs_ann_sched_run_slice(
    struct vecfs_ann_sched_query *q,
    const struct vecfs_ann_distance_session *session,
    u32 budget)
{
    const struct vecfs_ann_snapshot *snapshot = q->snapshot;
    struct vecfs_ann_sched_ctx *ctx = q->ctx;
    struct vecfs_ann_visited *visited = &ctx->visited;
    u32 ef = q->ef;
    u32 step;

    if (!q->upper_done) {
        /*
         * entry 距离必须在此处（FPU 临界区内）计算：SIMD raw_fn
         * 依赖 kernel_fpu 上下文。
         */
        q->entry_distance =
            vecfs_ann_distance_calculate(
                session,
                q->query,
                vecfs_ann_snapshot_vector(
                    snapshot, q->entry_node));

        q->dist_compute_cnt++;

        vecfs_ann_sched_greedy_upper(q, session);
        q->upper_done = true;

        vecfs_ann_visited_reset(visited);
        vecfs_ann_heap_reset(&ctx->candidates);
        vecfs_ann_heap_reset(&ctx->results);

        vecfs_ann_visited_test_and_set(
            visited, q->entry_node);

        vecfs_ann_heap_push(
            &ctx->results,
            q->entry_node,
            q->entry_distance);

        vecfs_ann_heap_push(
            &ctx->candidates,
            q->entry_node,
            q->entry_distance);

        q->lower_bound = q->entry_distance;
        q->base_done = false;
    }

    if (q->base_done)
        return true;

    for (step = 0; step < budget; step++) {
        const struct vecfs_ann_heap_item *nearest;
        struct vecfs_ann_heap_item cur;
        const u32 *links;
        u16 count;
        u16 i;

        nearest = vecfs_ann_heap_top(
            &ctx->candidates);

        if (unlikely(!nearest)) {
            q->base_done = true;
            break;
        }

        /* 纯净 bare 终止条件：最近 candidate 已远于下界。 */
        if (nearest->distance > q->lower_bound) {
            q->base_done = true;
            break;
        }

        vecfs_ann_heap_pop(
            &ctx->candidates, &cur);

        /*
         * 预取下一个 candidate 的 level0 元素（count+links）与 vector，
         * 隐藏下一次迭代 pop 后读 links 的延迟。对齐 hnswlib
         * searchBaseLayerST 里 candidate_set.top() 的预取。
         */
        {
            const struct vecfs_ann_heap_item *next =
                vecfs_ann_heap_top(&ctx->candidates);

            if (next) {
                prefetch(vecfs_ann_level0_element(
                    snapshot, next->node_id));
                prefetch(vecfs_ann_snapshot_vector(
                    snapshot, next->node_id));
            }
        }

        links = vecfs_ann_level0_links(
            snapshot, cur.node_id);
        count = vecfs_ann_level0_count(
            snapshot, cur.node_id);

        if (likely(count))
            vecfs_ann_sched_prefetch_neighbor(
                snapshot,
                visited,
                links[0]);

        for (i = 0; i < count; i++) {
            u32 candidate = links[i];
            float candidate_distance;

            if (i + 1 < count)
                vecfs_ann_sched_prefetch_neighbor(
                    snapshot,
                    visited,
                    links[i + 1]);

            if (likely(
                    vecfs_ann_visited_test_and_set(
                        visited, candidate)))
                continue;

            candidate_distance =
                vecfs_ann_distance_calculate(
                    session,
                    q->query,
                    vecfs_ann_snapshot_vector(
                        snapshot,
                        candidate));

            q->dist_compute_cnt++;

            if (ctx->results.size >= ef &&
                q->lower_bound <= candidate_distance)
                continue;

            /*
             * candidate 堆有界（4*m0*ef）：满了 drop-new，仍保证导航
             * 前进（已被 visited 标记，不会再次入堆）。
             */
            vecfs_ann_heap_push(
                &ctx->candidates,
                candidate,
                candidate_distance);

            if (ctx->results.size < ef) {
                vecfs_ann_heap_push(
                    &ctx->results,
                    candidate,
                    candidate_distance);
            } else {
                const struct vecfs_ann_heap_item *worst =
                    vecfs_ann_heap_top(&ctx->results);

                if (candidate_distance <
                    worst->distance)
                    vecfs_ann_heap_replace_top(
                        &ctx->results,
                        candidate,
                        candidate_distance);
            }

            q->lower_bound =
                vecfs_ann_heap_top(
                    &ctx->results)->distance;
        }

        q->progress_units++;
    }

    return q->base_done;
}

/* ------------------------------------------------------------------ */
/* 结果与轨迹回填                                                     */
/* ------------------------------------------------------------------ */

static void vecfs_ann_sched_fill_trace(
    struct vecfs_ann_sched_query *q,
    struct vecfs_ioctl_sched_trace *t)
{
    memset(t, 0, sizeof(*t));

    t->qid = q->index;

    t->total_us = (q->finished_ns - q->submitted_ns) / 1000ULL;
    t->backlog_wait_us = 0;
    t->queue_wait_us = q->queue_wait_acc / 1000ULL;
    t->run_us = q->run_ns_acc / 1000ULL;
    t->ctx_wait_us = q->ctx_wait_acc / 1000ULL;
    t->rescue_wait_us = 0;
    t->normal_wait_us = q->normal_wait_acc / 1000ULL;
    t->boost_wait_us = q->boost_wait_acc / 1000ULL;

    t->first_run_wait_us = q->first_run_ns ?
        (q->first_run_ns - q->admitted_ns) / 1000ULL : 0;

    t->dispatches = q->dispatches;
    t->requeues = q->requeues;
    t->ctx_misses = 0;
    t->zero_progress_batches = q->zero_progress_batches;

    t->wave_id_submit = q->wave_id_submit;
    t->wave_id_admit = q->wave_id_admit;
    t->wave_id_finish = q->wave_id_finish;

    t->victim_entries = q->victim_entries;
    t->victim_exits = q->victim_exits;
    t->rescue_entries_seen = 0;

    t->progress_units = q->progress_units;
    t->dist_compute_cnt = q->dist_compute_cnt;
    t->tail_reason = q->tail_reason;

    t->avg_rate_q16 = q->avg_rate_q16;
    t->min_rate_q16 = q->min_rate_q16;
    t->max_rate_q16 = q->max_rate_q16;
}

static void vecfs_ann_sched_finalize_results(
    struct vecfs_ann_sched_query *q)
{
    struct vecfs_ann_sched_req *req = q->batch->req;
    struct vecfs_ann_sched_ctx *ctx = q->ctx;
    u32 base = q->index * q->k;
    u32 sorted_count = 0;
    u32 n = 0;
    u32 j;

    if (!q->status && ctx) {
        vecfs_ann_heap_copy_sorted(
            &ctx->results,
            ctx->sorted,
            ctx->sorted_capacity,
            &sorted_count);

        n = min(q->k, sorted_count);

        for (j = 0; j < n; j++) {
            const struct vecfs_ann_heap_item *item =
                &ctx->sorted[j];

            req->results[base + j] = item->node_id;

            if (req->distances)
                req->distances[base + j] =
                    vecfs_ann_sched_float_bits(
                        item->distance);
        }
    }

    for (j = n; j < q->k; j++) {
        req->results[base + j] = 0;

        if (req->distances)
            req->distances[base + j] = 0;
    }

    req->dist_counts[q->index] = q->dist_compute_cnt;

    if (req->traces)
        vecfs_ann_sched_fill_trace(
            q, &req->traces[q->index]);

    if (q->status && !READ_ONCE(req->status))
        WRITE_ONCE(req->status, q->status);
}

/* ------------------------------------------------------------------ */
/* EWMA 进度率与争用塑造                                              */
/* ------------------------------------------------------------------ */

static void vecfs_ann_sched_update_rate(
    struct vecfs_ann_sched_query *q,
    u32 delta_progress,
    u64 slice_ns)
{
    u64 rate_q16 = 0;

    if (slice_ns)
        rate_q16 =
            (u64)delta_progress * 65536000000ULL /
            slice_ns;

    if (!q->avg_rate_q16) {
        q->avg_rate_q16 = rate_q16;
        q->min_rate_q16 = rate_q16;
        q->max_rate_q16 = rate_q16;
    } else {
        q->avg_rate_q16 =
            (q->avg_rate_q16 *
                 (VECFS_ANN_SCHED_EWMA_ALPHA - 1) +
             rate_q16) /
            VECFS_ANN_SCHED_EWMA_ALPHA;

        q->min_rate_q16 =
            min(q->min_rate_q16, rate_q16);
        q->max_rate_q16 =
            max(q->max_rate_q16, rate_q16);
    }
}

/*
 * 用本 slice 进度率更新 cohort 平均，并按滞回调整并发上界（P3 shaping）。
 */
static void vecfs_ann_sched_adjust_concurrency(
    struct vecfs_ann_sched *sched,
    struct vecfs_ann_sched_query *q,
    u64 rate_q16)
{
    unsigned long flags;
    bool contended;

    spin_lock_irqsave(&sched->rate_lock, flags);

    if (!sched->cohort_rate_q16)
        sched->cohort_rate_q16 = rate_q16;
    else
        sched->cohort_rate_q16 =
            (sched->cohort_rate_q16 *
                 (VECFS_ANN_SCHED_EWMA_ALPHA - 1) +
             rate_q16) /
            VECFS_ANN_SCHED_EWMA_ALPHA;

    contended = rate_q16 &&
        rate_q16 < (sched->cohort_rate_q16 >> 1);

    if (contended) {
        if (sched->contend_streak <
            VECFS_ANN_SCHED_CONTEND_HYST)
            sched->contend_streak++;
    } else {
        if (sched->contend_streak >
            -VECFS_ANN_SCHED_CONTEND_HYST)
            sched->contend_streak--;
    }

    if (sched->contend_streak >=
            VECFS_ANN_SCHED_CONTEND_HYST &&
        sched->effective_high > sched->cohort_low) {
        sched->effective_high = sched->cohort_low;
        sched->contend_streak = 0;
    } else if (
        sched->contend_streak <=
            -VECFS_ANN_SCHED_CONTEND_HYST &&
        sched->effective_high < sched->cohort_high) {
        sched->effective_high = sched->cohort_high;
        sched->contend_streak = 0;
    }

    spin_unlock_irqrestore(&sched->rate_lock, flags);
}

/*
 * victim 判定：进度率低于 cohort 平均的 1/4，或连续多 slice 停滞。
 * 只用于 BOOST / FULL 模式。
 */
static bool vecfs_ann_sched_is_victim(
    struct vecfs_ann_sched_query *q)
{
    struct vecfs_ann_sched *sched = q->sched;
    unsigned long flags;
    u64 cohort;

    if (q->stall_count >= 2)
        return true;

    spin_lock_irqsave(&sched->rate_lock, flags);
    cohort = sched->cohort_rate_q16;
    spin_unlock_irqrestore(&sched->rate_lock, flags);

    return cohort &&
        q->avg_rate_q16 &&
        q->avg_rate_q16 < (cohort >> 2);
}

/*
 * 有效并发上界：在 contention-streak 塑造（adjust_concurrency 写 effective_high）
 * 的基础上，再按当前 victim 数收缩（P3 shaping，KVFS V2 同款 high/2/high/4）。
 */
static int vecfs_ann_sched_effective_high(
    struct vecfs_ann_sched *sched)
{
    int high = READ_ONCE(sched->effective_high);
    int victims = atomic_read(&sched->victim_count);

    if (victims >= 2)
        high = min(high,
            max(sched->cohort_min_high,
                sched->cohort_high / 4));
    else if (victims == 1)
        high = min(high,
            max(sched->cohort_min_high,
                sched->cohort_high / 2));

    return max(1, high);
}

/*
 * 低负载判定：在飞查询不超过 cohort_low 时认为无排队压力，
 * 直接 run-to-completion 以免白付 RR 切换开销（KVFS low-load RTC）。
 */
static bool vecfs_ann_sched_low_load(
    struct vecfs_ann_sched *sched)
{
    return atomic_read(&sched->active) <= sched->cohort_low;
}

/*
 * 由 EWMA 进度率（Q16 microstep/ms）反推 ~80us 的 slice 预算（P2）。
 */
static u32 vecfs_ann_sched_budget_from_rate(u64 rate_q16)
{
    u64 budget;

    if (!rate_q16)
        return VECFS_ANN_SCHED_BUDGET_INIT;

    budget = div64_u64(
        rate_q16 * VECFS_ANN_SCHED_TARGET_NS,
        1000000ULL * VECFS_ANN_SCHED_RATE_ONE);

    return clamp_t(u32, (u32)budget,
        VECFS_ANN_SCHED_BUDGET_MIN,
        VECFS_ANN_SCHED_BUDGET_MAX);
}

/*
 * 预算选择：OFF(RTC) > tail pod > 低负载 > victim 补偿 > 自适应。
 */
static u32 vecfs_ann_sched_select_budget(
    struct vecfs_ann_sched *sched,
    struct vecfs_ann_sched_query *q)
{
    if (q->batch->req->sched_mode == VECFS_SCHED_MODE_OFF)
        return U32_MAX;

    if (q->in_tail)
        return VECFS_ANN_SCHED_BUDGET_RTC;

    if (vecfs_ann_sched_low_load(sched))
        return VECFS_ANN_SCHED_BUDGET_RTC;

    if (q->victim)
        return VECFS_ANN_SCHED_BOOST_BUDGET;

    return vecfs_ann_sched_budget_from_rate(q->avg_rate_q16);
}

/* ------------------------------------------------------------------ */
/* work item 主体                                                     */
/* ------------------------------------------------------------------ */

static void vecfs_ann_sched_work_fn(
    struct work_struct *work)
{
    struct vecfs_ann_sched_query *q =
        container_of(work, struct vecfs_ann_sched_query, work);

    struct vecfs_ann_sched *sched = q->sched;
    struct vecfs_ann_distance_session session;

    u32 mode = q->batch->req->sched_mode;
    u64 wait_ns;
    u64 slice_start;
    u64 slice_ns;
    u32 before;
    u32 after;
    u32 delta;
    bool done;

    if (q->status)
        goto finish;

    wait_ns = ktime_get_ns() - q->enqueue_ns;
    q->queue_wait_acc += wait_ns;

    if (q->enqueue_boost)
        q->boost_wait_acc += wait_ns;
    else
        q->normal_wait_acc += wait_ns;

    if (!q->first_run_ns)
        q->first_run_ns = ktime_get_ns();

    q->dispatches++;

    vecfs_ann_distance_begin(
        &session, sched->distance);

    if (unlikely(!session.active)) {
        q->status = -ENODEV;
        vecfs_ann_distance_end(&session);
        goto finish;
    }

    slice_start = ktime_get_ns();
    before = q->progress_units;

    done = vecfs_ann_sched_run_slice(
        q,
        &session,
        vecfs_ann_sched_select_budget(sched, q));

    after = q->progress_units;

    vecfs_ann_distance_end(&session);

    slice_ns = ktime_get_ns() - slice_start;
    q->run_ns_acc += slice_ns;

    delta = after - before;
    if (!delta)
        q->zero_progress_batches++;

    if (delta)
        q->stall_count = 0;
    else
        q->stall_count++;

    vecfs_ann_sched_update_rate(q, delta, slice_ns);

    if (done || q->status)
        goto finish;

    q->requeues++;

    /* P2/P3：victim 判定 + 迁到 boost_wq。 */
    if (mode == VECFS_SCHED_MODE_BOOST ||
        mode == VECFS_SCHED_MODE_FULL) {
        if (vecfs_ann_sched_is_victim(q)) {
            if (!q->victim) {
                q->victim = true;
                q->victim_entries++;
                atomic_inc(&sched->victim_count);
            }
        } else if (q->victim) {
            q->victim = false;
            q->victim_exits++;
            atomic_dec(&sched->victim_count);
        }
    }

    if (mode == VECFS_SCHED_MODE_SHAPING ||
        mode == VECFS_SCHED_MODE_FULL)
        vecfs_ann_sched_adjust_concurrency(
            sched, q, q->avg_rate_q16);

    /*
     * tail pod（P3 V3）：victim 连续低速 slice 数达到阈值后迁到专用
     * tail_wq，以 RTC 预算一次性收尾；否则继续停在 boost_wq 补偿。
     */
    if (q->victim) {
        if (!q->in_tail &&
            ++q->tail_low_streak >=
                VECFS_ANN_SCHED_TAIL_ENTER_STREAK) {
            q->in_tail = true;
            q->tail_reason = VECFS_SCHED_TAIL_LOW_RATE;
        }
    } else {
        q->tail_low_streak = 0;
    }

    q->enqueue_ns = ktime_get_ns();

    if (q->in_tail) {
        q->enqueue_boost = true;
        queue_work(sched->tail_wq, &q->work);
    } else if (q->victim) {
        q->enqueue_boost = true;
        queue_work(sched->boost_wq, &q->work);
    } else {
        q->enqueue_boost = false;
        queue_work(sched->norm_wq, &q->work);
    }

    return;

finish:
    vecfs_ann_sched_finish_query(q);
}

static void vecfs_ann_sched_finish_query(
    struct vecfs_ann_sched_query *q)
{
    struct vecfs_ann_sched *sched = q->sched;

    q->finished_ns = ktime_get_ns();

    /*
     * 出错时用 CTX_WAIT 覆盖；否则保留 work_fn 里记录的 tail_reason
     * （如迁入 tail pod 的 LOW_RATE），未迁入时保持 NONE。
     */
    if (q->status)
        q->tail_reason = VECFS_SCHED_TAIL_CTX_WAIT;

    /* 若查询以 victim 状态结束，释放它的 victim_count 计数。 */
    if (q->victim)
        atomic_dec(&sched->victim_count);

    vecfs_ann_sched_finalize_results(q);

    if (q->ctx) {
        vecfs_ann_sched_ctx_put(sched, q->ctx);
        q->ctx = NULL;
    }

    atomic_dec(&sched->active);
    wake_up_all(&sched->gate_wait);

    if (atomic_dec_and_test(
            &q->batch->remaining))
        complete(&q->batch->done);
}

/* ------------------------------------------------------------------ */
/* 提交                                                               */
/* ------------------------------------------------------------------ */

int vecfs_ann_sched_submit(
    struct super_block *sb,
    struct vecfs_ann_sched_req *req)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_sched *sched;
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *snapshot;
    struct vecfs_ann_sched_batch *batch = NULL;
    u32 ef;
    u32 candidate_capacity;
    u32 i;
    int ret = 0;

    if (!sb || !req)
        return -EINVAL;

    if (!req->n_queries || !req->k)
        return -EINVAL;

    sched = sbi->s_ann_sched;
    if (!sched)
        return -ENODEV;

    if (req->sched_mode > VECFS_SCHED_MODE_FULL)
        return -EINVAL;

    cache = sbi->s_ann_cache;
    if (!cache)
        return -ENODEV;

    if (req->dim != sbi->s_ann.config.dimension)
        return -EINVAL;

    ef = vecfs_ann_runtime_capture_ef(
        sb, req->ef, req->k);
    if (!ef || ef > VECFS_ANN_SEARCH_MAX_EF)
        return -EINVAL;

    /*
     * 固定快照：worker 在可睡眠上下文中直接使用，故用 kref 而非
     * rcu_read_lock。整个批次完成后归还。
     */
    rcu_read_lock();
    snapshot = rcu_dereference(cache->snapshot);
    if (unlikely(!snapshot)) {
        rcu_read_unlock();
        return -ENODEV;
    }
    vecfs_ann_snapshot_get(snapshot);
    rcu_read_unlock();

    if (snapshot->vector_bytes !=
        (size_t)req->dim * sizeof(float)) {
        ret = -EINVAL;
        goto out_put;
    }

    if (!snapshot->live_count ||
        snapshot->entry_node ==
            VECFS_ANN_INVALID_NODE_ID) {
        /*
         * 空图：全部结果置零，dist_counts 置 0。
         */
        memset(req->results, 0,
               (size_t)req->n_queries * req->k *
                   sizeof(u32));
        if (req->distances)
            memset(req->distances, 0,
                   (size_t)req->n_queries * req->k *
                       sizeof(u32));
        memset(req->dist_counts, 0,
               (size_t)req->n_queries * sizeof(u64));
        ret = 0;
        goto out_put;
    }

    batch = kzalloc(sizeof(*batch), GFP_KERNEL);
    if (!batch) {
        ret = -ENOMEM;
        goto out_put;
    }

    batch->req = req;
    batch->snapshot = snapshot;

    batch->queries = kcalloc(
        req->n_queries,
        sizeof(*batch->queries),
        GFP_KERNEL);
    if (!batch->queries) {
        ret = -ENOMEM;
        goto out_batch;
    }

    init_completion(&batch->done);
    atomic_set(&batch->remaining, req->n_queries);

    candidate_capacity =
        VECFS_ANN_SCHED_CAND_FACTOR *
        snapshot->m0 * ef;

    for (i = 0; i < req->n_queries; i++) {
        struct vecfs_ann_sched_query *q =
            &batch->queries[i];

        /*
         * 并发塑造门：active >= effective_high 时提交路径等待，
         * 直到有查询完成释放槽位。这是「admit = acquire slot」。
         */
        wait_event(
            sched->gate_wait,
            atomic_read(&sched->active) <
                vecfs_ann_sched_effective_high(sched));

        q->sched = sched;
        q->batch = batch;
        q->snapshot = snapshot;
        q->index = i;
        q->query = &req->queries[
            (size_t)i * req->dim];
        q->k = req->k;
        q->ef = ef;

        q->entry_node = snapshot->entry_node;
        q->entry_distance = 0.0f;

        q->wave_id_submit =
            atomic_read(&sched->wave_id);

        q->submitted_ns = ktime_get_ns();

        q->ctx = NULL;
        ret = vecfs_ann_sched_ctx_acquire(
            sched,
            snapshot->capacity,
            candidate_capacity,
            ef,
            &q->ctx);
        if (ret) {
            q->status = ret;
            /*
             * 拿不到 ctx：该查询立刻以失败结束，不占槽位。
             */
            q->ctx_wait_acc = 0;
            q->finished_ns = ktime_get_ns();
            vecfs_ann_sched_finalize_results(q);

            if (atomic_dec_and_test(
                    &batch->remaining))
                complete(&batch->done);

            continue;
        }

        q->ctx_wait_acc =
            ktime_get_ns() - q->submitted_ns;
        q->admitted_ns = ktime_get_ns();
        q->wave_id_admit = q->wave_id_submit;
        q->enqueue_ns = q->admitted_ns;
        q->enqueue_boost = false;

        atomic_inc(&sched->active);

        INIT_WORK(&q->work, vecfs_ann_sched_work_fn);
        queue_work(sched->norm_wq, &q->work);
    }

    wait_for_completion(&batch->done);

    atomic_inc(&sched->wave_id);

    ret = READ_ONCE(req->status);

    kfree(batch->queries);

out_batch:
    kfree(batch);

out_put:
    vecfs_ann_snapshot_put(snapshot);
    return ret;
}

/* ------------------------------------------------------------------ */
/* 初始化 / 销毁                                                      */
/* ------------------------------------------------------------------ */

int vecfs_ann_sched_init(struct super_block *sb)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_sched *sched;
    u32 maximum;

    if (!sbi)
        return -EINVAL;

    sched = kzalloc(sizeof(*sched), GFP_KERNEL);
    if (!sched)
        return -ENOMEM;

    sched->sb = sb;
    sched->distance = &sbi->s_ann_distance;

    sched->cohort_high = max(1, num_online_cpus());
    sched->cohort_low = max(1, sched->cohort_high / 2);
    sched->cohort_min_high = max(1, sched->cohort_high / 4);
    sched->effective_high = sched->cohort_high;

    atomic_set(&sched->active, 0);
    atomic_set(&sched->victim_count, 0);
    atomic_set(&sched->wave_id, 0);

    init_waitqueue_head(&sched->gate_wait);
    spin_lock_init(&sched->rate_lock);

    sched->norm_wq = alloc_workqueue(
        "vecfs_ann_norm", WQ_UNBOUND, 0);
    sched->boost_wq = alloc_workqueue(
        "vecfs_ann_boost", WQ_UNBOUND | WQ_HIGHPRI, 0);
    /* tail pod：单并发 rescue 队列（P3 V3）。 */
    sched->tail_wq = alloc_workqueue(
        "vecfs_ann_tail", WQ_UNBOUND | WQ_HIGHPRI, 1);

    if (!sched->norm_wq || !sched->boost_wq || !sched->tail_wq) {
        if (sched->norm_wq)
            destroy_workqueue(sched->norm_wq);
        if (sched->boost_wq)
            destroy_workqueue(sched->boost_wq);
        if (sched->tail_wq)
            destroy_workqueue(sched->tail_wq);
        kfree(sched);
        return -ENOMEM;
    }

    mutex_init(&sched->pool.lock);
    init_waitqueue_head(&sched->pool.wait);
    INIT_LIST_HEAD(&sched->pool.free);

    maximum = max_t(u32,
        sched->cohort_high,
        (u32)max(1,
            atomic_read(
                &sbi->s_ann_runtime.maximum_queries)));

    sched->pool.maximum = maximum;

    sbi->s_ann_sched = sched;

    pr_info(
        "vecfs: RTANN scheduler ready cohort=%d norm/boost/tail workqueues\n",
        sched->cohort_high);

    return 0;
}

void vecfs_ann_sched_destroy(struct super_block *sb)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_sched *sched;

    if (!sbi)
        return;

    sched = sbi->s_ann_sched;
    if (!sched)
        return;

    sbi->s_ann_sched = NULL;

    flush_workqueue(sched->norm_wq);
    flush_workqueue(sched->boost_wq);
    flush_workqueue(sched->tail_wq);

    destroy_workqueue(sched->norm_wq);
    destroy_workqueue(sched->boost_wq);
    destroy_workqueue(sched->tail_wq);

    vecfs_ann_sched_pool_destroy(sched);

    kfree(sched);
}
