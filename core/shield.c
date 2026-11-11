// SPDX-License-Identifier: GPL-2.0
//
// SHIELD 派生有效性的实现。
//
// P0：三种基线模式（Event/Conditional/Validated）的统计观测与
//     每文件源状态版本 v_f。
// P1：SHIELD 状态机（CLEAN→PENDING→DERIVING→CLEAN）、每文件去抖
//     调度、内核侧假嵌入（确定性哈希 → fp32 向量）与条件发布。
// P2：策略接口（classify/chunk/embed）+ 内置默认策略（文本判别 +
//     定长分块 + 哈希嵌入）+ 块级复用表 + 二进制 SKIP 处置。
//
// 设计约束：文件系统掌握全部控制权，用户态是被调用方。SHIELD 模式下
// 内核在写路径自动标记旧向量 stale、推进状态机并调度派生；派生完成后
// 内核以「目标版本仍等于当前 v_f」为前提条件发布新向量，杜绝把过期
// 派生结果写回索引。P2 的块级复用表让未变块免于重嵌入与重入 HNSW，
// 是「语义派生放大」在内容粒度上的体现。

#include <linux/atomic.h>
#include <linux/bitmap.h>
#include <linux/delay.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/highmem.h>
#include <linux/jiffies.h>
#include <linux/ktime.h>
#include <linux/overflow.h>
#include <linux/pagemap.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

#include "../include/vecfs.h"
#include "../include/vecfs_disk.h"
#include "../include/vecfs_shield.h"
#include "../include/vecfs_embed_upcall.h"
#include "../include/vecfs_ann_cache.h"
#include "../include/vecfs_ann_batch.h"
#include "../include/vecfs_ann_build.h"
#include "../include/vecfs_ann_indirect.h"
#include "../include/vecfs_ann_stale.h"

/* SHIELD 默认去抖窗口（微秒）。set_mode 可通过 window_us 覆盖。 */
#define VECFS_SHIELD_DEFAULT_WINDOW_US (50 * 1000UL)

/* 假嵌入读取的内容上限，避免大文件占用过多内存。
 * 必须 ≥ 单文件最大块数×块大小（含追加尾块），否则超过上限的尾部块会被静默丢弃；
 * 同时与 VECFS_EMBED_UP_MAX_DATA 对齐，保证 reuse=OFF 的全量重嵌能整批发出。 */
#define VECFS_SHIELD_MAX_EMBED_BYTES (256 * 1024UL)

/* 定长分块默认大小（字节）。 */
#define VECFS_SHIELD_DEFAULT_CHUNK_SIZE 256UL

/* 单文件派生块数上限（防御性）。 */
#define VECFS_SHIELD_MAX_CHUNKS 4096U

struct vecfs_shield_runtime *
vecfs_shield_runtime(struct super_block *sb)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);

    return sbi ? &sbi->s_shield : NULL;
}

int vecfs_shield_mode(struct super_block *sb)
{
    struct vecfs_shield_runtime *rt = vecfs_shield_runtime(sb);

    if (!rt)
        return VECFS_SHIELD_MODE_NONE;

    return READ_ONCE(rt->mode);
}

/* 事件通知：文件派生状态转换（→CLEAN 等终态）时唤醒 poll() 等待者。 */
static void vecfs_shield_state_wake(struct inode *inode)
{
    struct vecfs_shield_runtime *rt = vecfs_shield_runtime(inode->i_sb);

    if (rt)
        wake_up_all(&rt->state_wq);
}

int vecfs_shield_set_mode(struct super_block *sb, int mode, u64 window_us)
{
    struct vecfs_shield_runtime *rt = vecfs_shield_runtime(sb);

    if (!rt)
        return -ENODEV;

    switch (mode) {
    case VECFS_SHIELD_MODE_EVENT:
    case VECFS_SHIELD_MODE_CONDITIONAL:
    case VECFS_SHIELD_MODE_VALIDATED:
    case VECFS_SHIELD_MODE_SHIELD:
        break;
    default:
        return -EINVAL;
    }

    if (!window_us)
        window_us = VECFS_SHIELD_DEFAULT_WINDOW_US;

    WRITE_ONCE(rt->window_us, window_us);
    WRITE_ONCE(rt->window_jiffies,
               max_t(u64, usecs_to_jiffies(window_us), 1));
    WRITE_ONCE(rt->mode, mode);
    vecfs_shield_stats_reset(sb);

    /*
     * Phase D：进入 SHIELD 模式时，把当前 stale 的 file-bound 向量调度
     * 重派生（覆盖崩溃恢复后重新启用 SHIELD、以及从 VALIDATED 切 SHIELD
     * 两种场景）。
     */
    if (mode == VECFS_SHIELD_MODE_SHIELD)
        vecfs_shield_schedule_stale_rederive(sb);

    return 0;
}

/* ---------------------------------------------------------------------
 * P2：策略接口与内置默认策略
 * ------------------------------------------------------------------- */

static u64 vecfs_shield_splitmix64(u64 *state)
{
    u64 z = (*state += 0x9E3779B97F4A7C15ULL);

    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* FNV-1a 内容哈希：同时用于假嵌入种子与块级复用表的比较键。 */
static u64 vecfs_shield_content_hash(const void *data, size_t len)
{
    const u8 *p = data;
    u64 hash = 14695981039346656037ULL;
    size_t i;

    for (i = 0; i < len; i++) {
        hash ^= p[i];
        hash *= 1099511628211ULL;
    }

    return hash;
}

/*
 * 完整 L1 在途复用：输入键 K(c)=H(θ‖X(c))。把块偏移 θ 与内容哈希 X(c) 拼成
 * 16 字节做 FNV-1a，得到块在「位置 × 内容」上的内容寻址身份。写路径据此在
 * 在途注册表里定位被新写覆盖的块；同一块内容跨版本重排也可按键区分。
 */
static u64 vecfs_shield_input_key(u64 offset, u64 content_hash)
{
    u64 hash = 14695981039346656037ULL;
    int i;

    for (i = 0; i < 8; i++) {
        hash ^= (u8)(offset >> (i * 8));
        hash *= 1099511628211ULL;
    }
    for (i = 0; i < 8; i++) {
        hash ^= (u8)(content_hash >> (i * 8));
        hash *= 1099511628211ULL;
    }

    return hash;
}

/*
 * 假嵌入：对块内容做 FNV-1a 哈希，再以 (hash ^ dim) 为种子生成每个
 * 维度的 fp32 位模式，落在 [-2,-1) ∪ [1,2)（有限、确定性）。
 * 内核代码默认禁用 FPU，故用整数构造 IEEE-754 位模式而非浮点运算。
 * 这里是内置默认策略；日后多模态/真实嵌入器可注册用户态策略替换。
 */
static int vecfs_shield_embed_hash(const void *data, size_t len,
                                   u32 dim, u32 *out)
{
    u64 hash = vecfs_shield_content_hash(data, len);
    u32 d;

    for (d = 0; d < dim; d++) {
        u64 state = hash ^ ((u64)d << 32) ^ d;
        u32 lo = (u32)vecfs_shield_splitmix64(&state);
        u32 bits;

        /* [1.0, 2.0)：指数 127，尾数取 lo 的高 23 位。 */
        bits = 0x3F800000u | ((lo >> 8) & 0x007FFFFFu);
        /* 随机符号位 → [-2,-1) ∪ [1,2)。 */
        if (lo & 0x00000080u)
            bits |= 0x80000000u;

        out[d] = bits;
    }

    return VECFS_SHIELD_EMBED_OK;
}

/*
 * 批量假嵌入：逐块调用 embed_hash，把 nchunks×dim 个 fp32 连续写入 out
 * （chunks[j] → out[j*dim..]）。供 derive 批量路径与 embed_batch 接口对齐。
 */
static int vecfs_shield_embed_hash_batch(struct inode *inode, u64 generation,
                                         const void *base,
                                         const struct vecfs_shield_chunk *chunks,
                                         u32 nchunks, u32 dim, u32 *out,
                                         u64 *req_id_out)
{
    u32 j;

    for (j = 0; j < nchunks; j++)
        vecfs_shield_embed_hash(base + chunks[j].offset, chunks[j].length,
                                dim, out + (size_t)j * dim);

    return VECFS_SHIELD_EMBED_OK;
}

/*
 * 文本/二进制判别：NUL 字节直接判二进制；控制字符（<0x20，排除
 * \n \t \r）占比超过 10% 判二进制。高位字节（>=0x80）是 UTF-8 多字节
 * 码点的正常组成，不作为二进制信号（否则 CJK/重音文本会被误判 SKIP）。
 * 留出日后接入更精细用户态分类策略的空间。
 */
static int vecfs_shield_classify_text(const void *data, size_t len)
{
    const u8 *p = data;
    size_t sample = min_t(size_t, len, 4096);
    size_t non_printable = 0;
    size_t i;

    for (i = 0; i < sample; i++) {
        u8 c = p[i];

        if (c == 0)
            return VECFS_SHIELD_DISP_SKIP;

        if (c < 0x20 && c != '\n' && c != '\t' && c != '\r')
            non_printable++;
    }

    if (sample && non_printable * 100 / sample > 10)
        return VECFS_SHIELD_DISP_SKIP;

    return VECFS_SHIELD_DISP_EMBED;
}

/*
 * 定长分块（码点对齐）：chunk i = [clamp(i*C), clamp((i+1)*C))，
 * clamp 把边界向前钳过 UTF-8 续字节（0b10xxxxxx），避免在多字节码点
 * 中间切开。块长在 [C-3, C+3] 内浮动；查询不分块、仅文件存储分块，
 * 且 stale 范围映射按内容区间重叠判定，变长块无需其它改动。
 */
static int vecfs_shield_chunk_fixed(const void *data, size_t size,
                                    u32 chunk_size,
                                    struct vecfs_shield_chunk *out,
                                    u32 max_chunks, u32 *count)
{
    const u8 *p = data;
    u32 n = (u32)((size + chunk_size - 1) / chunk_size);
    u32 i;

    if (n > max_chunks)
        n = max_chunks;

    for (i = 0; i < n; i++) {
        u64 off = (u64)i * chunk_size;
        u64 next = (u64)(i + 1) * chunk_size;

        while (off < size && (p[off] & 0xC0) == 0x80)
            off++;

        while (next < size && (p[next] & 0xC0) == 0x80)
            next++;

        out[i].offset = off;
        out[i].length = min_t(u64, next - off, size - off);
    }

    *count = n;
    return 0;
}

static const struct vecfs_shield_strategy_ops vecfs_shield_builtin_strategy = {
    .name = "builtin-text-fixed-hash",
    .classify = vecfs_shield_classify_text,
    .chunk = vecfs_shield_chunk_fixed,
    .embed = vecfs_shield_embed_hash,
    .embed_batch = vecfs_shield_embed_hash_batch,
};

/*
 * 用户态嵌入 upcall 策略：classify/chunk 仍由内核内置（符合「内核做简单分块/
 * 判别」），仅 embed 交给用户态嵌入服务（/dev/vecfs-embed 守护进程）。文件
 * 系统仍是决策者——内核决定何时嵌入并处置 -EAGAIN 重试，用户态只应答。
 * 走批量路径：一次派生收集所有已变块、一次 upcall 嵌入（对齐 baseline embed_many）。
 */
static int vecfs_shield_upcall_embed_batch(struct inode *inode, u64 generation,
                                           const void *base,
                                           const struct vecfs_shield_chunk *chunks,
                                           u32 nchunks, u32 dim, u32 *out,
                                           u64 *req_id_out)
{
    return vecfs_embed_upcall_embed_batch(base, chunks, nchunks, dim, out,
                                          generation, req_id_out);
}

static const struct vecfs_shield_strategy_ops vecfs_shield_upcall_strategy = {
    .name = "upcall-text-fixed-ollama",
    .classify = vecfs_shield_classify_text,
    .chunk = vecfs_shield_chunk_fixed,
    .embed_batch = vecfs_shield_upcall_embed_batch,
};

static void vecfs_shield_drain_work(struct work_struct *work);
static void vecfs_shield_derive_put_ref(struct vecfs_shield_file_state *fs,
                                        struct inode *inode);

void vecfs_shield_runtime_init(struct vecfs_shield_runtime *rt)
{
    if (!rt)
        return;

    rt->chunk_size = VECFS_SHIELD_DEFAULT_CHUNK_SIZE;
    rt->strat_flags =
        VECFS_SHIELD_STRAT_F_REUSE |
        VECFS_SHIELD_STRAT_F_CLASSIFY |
        VECFS_SHIELD_STRAT_F_VALIDATE |
        VECFS_SHIELD_STRAT_F_CANCEL |
        VECFS_SHIELD_STRAT_F_COW |
        VECFS_SHIELD_STRAT_F_ASYNC_STALE |
        VECFS_SHIELD_STRAT_F_PARTIAL |
        VECFS_SHIELD_STRAT_F_ADAPTIVE_WINDOW;
    rt->strategy = &vecfs_shield_builtin_strategy;
    rt->shutting_down = false;
    atomic64_set(&rt->derive_refs_total, 0);

    spin_lock_init(&rt->pending_lock);
    INIT_LIST_HEAD(&rt->pending_files);
    INIT_DELAYED_WORK(&rt->drain_work, vecfs_shield_drain_work);
    init_waitqueue_head(&rt->state_wq);
}

/*
 * 卸载时关闭 SHIELD 派生。必须在释放 sbi/ANN 运行时之前调用：
 *   1) 置 shutting_down——此后写路径不再排队、worker requeue 不再重新武装；
 *   2) vecfs_embed_upcall_stop()——最后一次卸载时 refcount 归零，立即完成所有
 *      在途 upcall 请求（-EAGAIN），唤醒阻塞等待应答的 worker（不等 30s 超时）；
 *   3) 等待 derive_refs_total 归零——即所有运行中/排队中/计时器待发的派生
 *      work 都已终结并 iput。归零后 sbi/ANN 运行时即可安全释放。
 */
void vecfs_shield_shutdown(struct super_block *sb)
{
    struct vecfs_shield_runtime *rt = vecfs_shield_runtime(sb);
    struct vecfs_shield_file_state *fs;
    struct vecfs_shield_file_state *tmp;
    LIST_HEAD(drain_list);
    unsigned long deadline;

    if (!rt)
        return;

    WRITE_ONCE(rt->shutting_down, true);

    vecfs_embed_upcall_stop();

    /*
     * 取消 drain worker：先等一个正在运行的 drain 结束（upcall 已排空，其
     * 阻塞的 embed 已 -EAGAIN 唤醒），此后 drain 不再被重新武装。
     */
    cancel_delayed_work_sync(&rt->drain_work);

    /*
     * 清空仍挂起的待派生队列：每个 PENDING 文件在队列中持有一份 igrab 引用
     * （derive_refs≥1），逐个归还，让 derive_refs_total 归零。先把队列整体
     * 摘到局部链表再释放锁——put_ref 里 iput 可能阻塞/触发 evict，不能在
     * 自旋锁内执行。shutting_down 已置位，写路径不再入队，队列在 cancel 后稳定。
     */
    spin_lock(&rt->pending_lock);
    list_splice_init(&rt->pending_files, &drain_list);
    spin_unlock(&rt->pending_lock);

    list_for_each_entry_safe(fs, tmp, &drain_list, pending_node) {
        struct vecfs_inode *vi = container_of(fs, struct vecfs_inode, shield);

        list_del_init(&fs->pending_node);
        mutex_lock(&fs->lock);
        fs->state = VECFS_SHIELD_FS_CLEAN;
        mutex_unlock(&fs->lock);
        vecfs_shield_derive_put_ref(fs, &vi->vfs_inode);
    }

    /*
     * 兜底等待：cancel_sync + 手动排空后理论上已归零；保留硬上限（upcall 超时
     * 30s + 余量）以防卡死，避免卸载永久挂起（此前曾因此需要重启整机）。
     */
    deadline = jiffies + msecs_to_jiffies(40000);
    while (atomic64_read(&rt->derive_refs_total) > 0) {
        if (time_after(jiffies, deadline)) {
            pr_err("vecfs: shield shutdown: %lld derive refs still "
                   "outstanding after 40s; proceeding (worker stuck)\n",
                   (long long)atomic64_read(&rt->derive_refs_total));
            break;
        }
        msleep(10);
    }
}

/*
 * Phase D：扫描当前快照中所有 stale 的 file-bound LIVE 向量，按唯一 inode
 * 调度重派生。用于非干净卸载恢复后（或从 VALIDATED 切到 SHIELD 时）把被
 * 保守失效的向量重新嵌入恢复召回。write_hook 自带去抖，重复调度同一文件
 * 只会顺延窗口，不会二次派生。
 */
void vecfs_shield_schedule_stale_rederive(struct super_block *sb)
{
    struct vecfs_sb_info *sbi = VECFS_SB(sb);
    struct vecfs_ann_cache *cache;
    struct vecfs_ann_snapshot *snapshot;
    const struct vecfs_ann_cache_node *node;
    unsigned long *seen;
    u32 node_id;
    u32 ino;

    if (!sbi || !sbi->s_ann_cache)
        return;

    cache = sbi->s_ann_cache;

    seen = bitmap_zalloc(
        sbi->s_inodes_count + 1, GFP_KERNEL);
    if (!seen)
        return;

    rcu_read_lock();

    snapshot = rcu_dereference(cache->snapshot);
    if (!snapshot)
        goto out_unlock;

    for (node_id = VECFS_ANN_FIRST_NODE_ID;
         node_id < snapshot->capacity;
         node_id++) {
        node = vecfs_ann_node_at(snapshot, node_id);

        if (node->state != VECFS_ANN_OBJ_LIVE)
            continue;

        if (node->owner_type != VECFS_ANN_OWNER_FILE)
            continue;

        if (!vecfs_ann_snapshot_content_stale(
                snapshot, node_id))
            continue;

        ino = (u32)node->owner_ino;
        if (!ino || ino > sbi->s_inodes_count)
            continue;

        __set_bit(ino, seen);
    }

    rcu_read_unlock();

    for_each_set_bit(ino, seen, sbi->s_inodes_count + 1) {
        struct inode *inode = iget_locked(sb, ino);

        if (!inode)
            continue;

        if (VECFS_INODE_IS_NEW(inode)) {
            if (vecfs_read_inode(inode)) {
                iget_failed(inode);
                continue;
            }
            unlock_new_inode(inode);
        }

        if (S_ISREG(inode->i_mode) && inode->i_nlink)
            vecfs_shield_write_hook(
                inode, 0, i_size_read(inode));

        iput(inode);
    }

    bitmap_free(seen);
    return;

out_unlock:
    rcu_read_unlock();
    bitmap_free(seen);
}

int vecfs_shield_set_strategy(struct super_block *sb, u32 strategy_id,
                              u32 flags, u64 chunk_size)
{
    struct vecfs_shield_runtime *rt = vecfs_shield_runtime(sb);

    if (!rt)
        return -ENODEV;

    /* 0 = 内置假嵌入；1 = 用户态嵌入 upcall（embed 交给 /dev/vecfs-embed）。 */
    if (strategy_id != VECFS_SHIELD_STRATEGY_BUILTIN &&
        strategy_id != VECFS_SHIELD_STRATEGY_UPCALL)
        return -EINVAL;

    /* upcall 依赖 superblock 里 mkfs 固化的嵌入配置（url 非空）；否则永久 -EAGAIN。 */
    if (strategy_id == VECFS_SHIELD_STRATEGY_UPCALL) {
        struct vecfs_sb_info *sbi = VECFS_SB(sb);

        if (!sbi || !sbi->s_embed_configured)
            return -EINVAL;
    }

    if (flags &
        ~(VECFS_SHIELD_STRAT_F_REUSE |
          VECFS_SHIELD_STRAT_F_CLASSIFY |
          VECFS_SHIELD_STRAT_F_VALIDATE |
          VECFS_SHIELD_STRAT_F_CANCEL |
          VECFS_SHIELD_STRAT_F_COW |
          VECFS_SHIELD_STRAT_F_ASYNC_STALE |
          VECFS_SHIELD_STRAT_F_PARTIAL |
          VECFS_SHIELD_STRAT_F_ADAPTIVE_WINDOW))
        return -EINVAL;

    if (!chunk_size)
        chunk_size = VECFS_SHIELD_DEFAULT_CHUNK_SIZE;
    if (chunk_size > VECFS_SHIELD_MAX_EMBED_BYTES)
        chunk_size = VECFS_SHIELD_MAX_EMBED_BYTES;

    WRITE_ONCE(rt->chunk_size, (u32)chunk_size);
    WRITE_ONCE(rt->strat_flags, flags);
    WRITE_ONCE(rt->strategy,
               strategy_id == VECFS_SHIELD_STRATEGY_UPCALL
                   ? &vecfs_shield_upcall_strategy
                   : &vecfs_shield_builtin_strategy);

    /* 异步 stale 落盘开关透传到 ANN cache（写路径 mark_snapshot_stale 消费）。 */
    {
        struct vecfs_sb_info *sbi = VECFS_SB(sb);

        if (sbi && sbi->s_ann_cache)
            sbi->s_ann_cache->async_stale =
                !!(flags & VECFS_SHIELD_STRAT_F_ASYNC_STALE);
    }

    return 0;
}

bool vecfs_shield_write_marks_stale(struct super_block *sb)
{
    int mode = vecfs_shield_mode(sb);

    return mode == VECFS_SHIELD_MODE_VALIDATED ||
           mode == VECFS_SHIELD_MODE_SHIELD;
}

u64 vecfs_shield_version(struct inode *inode)
{
    struct vecfs_inode *vi = VECFS_I(inode);

    return READ_ONCE(vi->shield_version);
}

void vecfs_shield_version_bump(struct inode *inode)
{
    struct vecfs_inode *vi = VECFS_I(inode);
    struct vecfs_shield_file_state *fs = &vi->shield;
    struct vecfs_shield_runtime *rt =
        vecfs_shield_runtime(inode->i_sb);

    /*
     * 版本推进持 fs->lock，与派生 worker 提交阶段的「版本复核」
     * 互斥：复核通过后不可能再有版本推进插入到提交之前。版本号
     * 只用于过期判断（不决定数据正确性）；溢出在实验时间尺度内
     * 不可能。
     */
    mutex_lock(&fs->lock);
    vi->shield_version++;
    mutex_unlock(&fs->lock);

    if (rt)
        atomic64_inc(&rt->counters.writes);
}

/*
 * 块独立可见性（L2）：推进版本并记录写区间覆盖的每块写版本，二者在同一
 * fs->lock 临界区内完成，保证与派生提交阶段的按块复核互斥（复核读到的是
 * 「版本推进 + 块版本标记」的原子一致快照，避免复核在版本推进后、块标记
 * 前看到旧块版本而误判该块仍有效）。
 */
static void vecfs_shield_version_bump_range(struct inode *inode,
                                            u64 start, u64 len)
{
    struct vecfs_inode *vi = VECFS_I(inode);
    struct vecfs_shield_file_state *fs = &vi->shield;
    struct vecfs_shield_runtime *rt = vecfs_shield_runtime(inode->i_sb);
    u32 chunk_size;
    u32 first, last, i;

    mutex_lock(&fs->lock);
    vi->shield_version++;

    chunk_size = rt ? READ_ONCE(rt->chunk_size) : 0;
    if (!chunk_size)
        chunk_size = VECFS_SHIELD_DEFAULT_CHUNK_SIZE;

    /* chunk_size 变化：旧块版本按旧寻址，全部失效，整表重建。 */
    if (fs->chunk_written_version &&
        fs->chunk_version_chunk_size != chunk_size) {
        kfree(fs->chunk_written_version);
        fs->chunk_written_version = NULL;
        fs->chunk_version_count = 0;
    }

    /* 只记录会进入嵌入范围的块（读内容上限 MAX_EMBED_BYTES），越界块永不派生。 */
    if (len && start < VECFS_SHIELD_MAX_EMBED_BYTES) {
        u64 end = start + len;

        if (end > VECFS_SHIELD_MAX_EMBED_BYTES)
            end = VECFS_SHIELD_MAX_EMBED_BYTES;
        /*
         * 定长分块边界做 UTF-8 码点对齐时会向前钳最多 3 字节，块 i 的起点
         * 在 [i*C, i*C+3] 内浮动。写区间可能落在上一块的「借走」续字节里，
         * 故 lower bound 向前扩 3 字节，保证不遗漏任何被写块（多记一块仅导致
         * 保守拒绝，不会泄漏过期向量）。
         */
        first = (u32)((start >= 3 ? start - 3 : 0) / chunk_size);
        last = (u32)((end - 1) / chunk_size);
        if (last >= VECFS_SHIELD_MAX_CHUNKS)
            last = VECFS_SHIELD_MAX_CHUNKS - 1;
        if (first <= last) {
            if (fs->chunk_version_count <= last) {
                u32 ncnt = last + 1;
                u64 *nv = krealloc(fs->chunk_written_version,
                                   (size_t)ncnt * sizeof(*nv), GFP_KERNEL);

                if (nv) {
                    memset(nv + fs->chunk_version_count, 0,
                           (size_t)(ncnt - fs->chunk_version_count) *
                               sizeof(*nv));
                    fs->chunk_written_version = nv;
                    fs->chunk_version_count = ncnt;
                    fs->chunk_version_chunk_size = chunk_size;
                }
            }
            if (last < fs->chunk_version_count) {
                for (i = first; i <= last; i++)
                    fs->chunk_written_version[i] = vi->shield_version;
            }
        }
    }
    mutex_unlock(&fs->lock);

    if (rt)
        atomic64_inc(&rt->counters.writes);
}

/*
 * 写路径前段（SHIELD/VALIDATED 模式）。先推进版本、再标记旧向量
 * stale，二者均先于数据写入执行。
 *
 * 这是 P3 原子条件发布正确性的关键：若版本推进放在数据写入之后，
 * 派生 worker 可能在「mark_content_stale 之后、版本推进之前」提交，
 * 使写入前的旧内容向量既未被 stale 命中、又被标记为有效，造成无效
 * 结果泄露。版本先行推进后，worker 提交复核（持同一 fs->lock）会
 * 捕获任何已开始的并发写。
 */
/*
 * 主动取消（P1）：版本推进后，若本文件有在途 upcall 请求，立即取消。
 * 只读 in_flight_upcall 提示（WRITE_ONCE/READ_ONCE，不加锁）——读不到或
 * 读到已结束的 id 都退化为既有「版本复核」兜底；取消仅作提前唤醒以省算力，
 * 正确性仍由「版本先行推进 + 提交复核」保证。
 *
 * 完整 L1 在途复用（需求驱动取消）：不再「区间重叠即整批取消」，而是按输入键
 * 注册表逐块判定。写无关块 → 不取消；只覆盖部分在途块 → 保留在途请求，让稳定
 * 块嵌入完成后按块过滤部分发布；仅当全部在途块被覆盖（或注册表超界）才整批
 * 取消。
 */
static void vecfs_shield_cancel_inflight(struct inode *inode,
                                         u64 start, u64 len)
{
    struct vecfs_shield_file_state *fs = &VECFS_I(inode)->shield;
    u64 id = READ_ONCE(fs->in_flight_upcall);
    u64 ifirst, ilast;
    u32 nkeys, overflow;
    u32 overlap = 0;
    u32 i;

    if (!id)
        return;

    ifirst = READ_ONCE(fs->in_flight_first);
    ilast = READ_ONCE(fs->in_flight_last);
    if (ilast > ifirst &&
        !(start < ilast && start + len > ifirst))
        return; /* 无关块变更：不打断在途稳定块嵌入 */

    /*
     * 与在途区间重叠：按块注册表逐块判定。注册表未就绪（nkeys==0）或超界
     * （overflow）时退回整批取消的保守行为，与 scoped L1 一致。
     */
    nkeys = READ_ONCE(fs->in_flight_nkeys);
    overflow = READ_ONCE(fs->in_flight_overflow);
    if (!nkeys || overflow) {
        vecfs_embed_upcall_cancel(id);
        return;
    }
    if (nkeys > VECFS_SHIELD_INFLIGHT_MAX)
        nkeys = VECFS_SHIELD_INFLIGHT_MAX;

    for (i = 0; i < nkeys; i++) {
        u64 off = READ_ONCE(fs->in_flight_keys[i].offset);
        u64 l = READ_ONCE(fs->in_flight_keys[i].length);

        if (l && start < off + l && start + len > off)
            overlap++;
    }

    /* 全部在途块被新写覆盖才整批取消；部分覆盖保留在途、按块过滤部分发布。 */
    if (overlap >= nkeys)
        vecfs_embed_upcall_cancel(id);
}

int vecfs_shield_write_begin(struct inode *inode, u64 start, u64 len)
{
    struct vecfs_shield_runtime *rt =
        vecfs_shield_runtime(inode->i_sb);

    /* 版本推进 + 每块写版本标记（同一 fs->lock 内，供按块发布复核）。 */
    vecfs_shield_version_bump_range(inode, start, len);

    /* 版本已推进：取消与本次写区间重叠的在途嵌入（提前唤醒，避免后端为
     * 过期块继续算）。消融开关 F_CANCEL 关闭时跳过——退化为纯「版本复核」
     * 兜底，正确性不变。 */
    if (rt && (rt->strat_flags & VECFS_SHIELD_STRAT_F_CANCEL))
        vecfs_shield_cancel_inflight(inode, start, len);

    return vecfs_ann_mark_content_stale(
        inode,
        start,
        len);
}

/* ---------------------------------------------------------------------
 * P2：块级复用表
 * ------------------------------------------------------------------- */

static void vecfs_shield_reuse_free(struct vecfs_shield_file_state *fs)
{
    kfree(fs->reuse);
    fs->reuse = NULL;
}

/* 释放块级复用表（evict_inode 调用）。 */
void vecfs_shield_file_destroy(struct inode *inode)
{
    struct vecfs_inode *vi = VECFS_I(inode);
    struct vecfs_shield_file_state *fs = &vi->shield;

    /*
     * inode 进入 evict 说明 i_count 已归零——所有 igrab 引用（含派生
     * worker 持有的）都已 iput 归还，故 derive_refs 必须为 0。若非 0，
     * 说明引用计数与实际 igrab/iput 失衡（泄漏或账目错位），在此告警，
     * 定位比事后追 oops 便宜得多。复用表仍照常释放，避免二次泄漏。
     */
    if (WARN_ON_ONCE(READ_ONCE(fs->derive_refs) != 0))
        pr_err_ratelimited(
            "vecfs: evict with derive_refs=%d ino=%lu icount=%d\n",
            READ_ONCE(fs->derive_refs), inode->i_ino,
            atomic_read(&inode->i_count));

    /*
     * 先持锁置死标记：此后写路径与 drain worker 的 requeue 在持同一把
     * fs->lock 时检查 dead，发现已置位便不再入队，从而把「置死之后仍重新
     * 入队」的竞态关死。
     */
    mutex_lock(&fs->lock);
    fs->dead = true;
    mutex_unlock(&fs->lock);

    /*
     * 无需 cancel_delayed_work_sync(&fs->work)：每文件 work 已删除，改为全局
     * drain_work + pending_files 队列。进入 evict 说明 i_count 已归零——所有
     * igrab 引用（含队列持有的）都已归还，故文件此刻不在 pending_files 里、
     * 也不在某个 drain worker 的在途批内；这里只释放复用表与块版本表即可。
     */
    vecfs_shield_reuse_free(fs);
    kfree(fs->chunk_written_version);
    fs->chunk_written_version = NULL;
    fs->chunk_version_count = 0;
    fs->chunk_version_chunk_size = 0;
}

/*
 * 确保复用表与当前 chunk_size / 块数匹配。
 *
 * 关键约束：定长分块下块边界只由 chunk_size 决定、与文件长度无关，故追加
 * （nchunks 增大）时旧块的下标与内容哈希保持不变，只有新增块需要新条目。
 * 因此 nchunks 增大时必须在原地扩容并保留已有条目，而非整表重建——旧实现
 * 在「count != nchunks」时一律 free+realloc，而每次 append 都会让 nchunks 增长，
 * 导致复用表每次都被清空、整文件被重新嵌入，这是「大量块未被嵌入」的真正根因。
 * 仅当 chunk_size 变化（分块寻址整体改变）时才整表重建。
 */
static int vecfs_shield_reuse_ensure(struct vecfs_shield_file_state *fs,
                                     u32 chunk_size, u32 nchunks)
{
    struct vecfs_shield_reuse_table *tab = fs->reuse;

    if (tab && tab->chunk_size == chunk_size && tab->count == nchunks)
        return 0;

    /* chunk_size 变化：旧哈希按旧寻址计算，全部失效，整表重建。 */
    if (tab && tab->chunk_size != chunk_size) {
        vecfs_shield_reuse_free(fs);
        tab = NULL;
    }

    if (!tab) {
        tab = kzalloc(struct_size(tab, entries, nchunks), GFP_KERNEL);
        if (!tab)
            return -ENOMEM;
    } else if (nchunks > tab->count) {
        struct vecfs_shield_reuse_table *ntab;
        size_t old_sz = struct_size(tab, entries, tab->count);
        size_t new_sz = struct_size(tab, entries, nchunks);

        /* 追加：保留旧条目，仅清零新增区间。krealloc 失败保留旧表重试。 */
        ntab = krealloc(tab, new_sz, GFP_KERNEL);
        if (!ntab)
            return -ENOMEM;
        memset((u8 *)ntab + old_sz, 0, new_sz - old_sz);
        tab = ntab;
    }
    /* nchunks <= tab->count：收缩，仅截断 count（分配保留，无害）。 */

    tab->chunk_size = chunk_size;
    tab->count = nchunks;
    fs->reuse = tab;

    return 0;
}

static bool vecfs_shield_reuse_lookup(struct vecfs_shield_file_state *fs,
                                      u32 index, u64 hash, u32 *ordinal)
{
    struct vecfs_shield_reuse_table *tab = fs->reuse;

    if (!tab || index >= tab->count)
        return false;

    if (tab->entries[index].hash != hash)
        return false;

    *ordinal = tab->entries[index].ordinal;
    return true;
}

static void vecfs_shield_reuse_update(struct vecfs_shield_file_state *fs,
                                      u32 index, u64 hash, u32 ordinal)
{
    struct vecfs_shield_reuse_table *tab = fs->reuse;

    if (!tab || index >= tab->count)
        return;

    tab->entries[index].hash = hash;
    tab->entries[index].ordinal = ordinal;
}

/* ---------------------------------------------------------------------
 * 内容读取
 * ------------------------------------------------------------------- */

/* 逐页读取 [0, len) 内容到用户提供的缓冲区（进程上下文）。 */
static int vecfs_shield_read_pages(struct inode *inode, void *buf, size_t len)
{
    struct address_space *mapping = inode->i_mapping;
    struct vecfs_shield_runtime *rt = vecfs_shield_runtime(inode->i_sb);
    size_t pos = 0;

    while (pos < len) {
        struct folio *folio;
        size_t off = pos & (PAGE_SIZE - 1);
        size_t chunk = min_t(size_t, PAGE_SIZE - off, len - pos);
        void *kaddr;

        /*
         * 卸载关闭中：读内容可能触发块 I/O（页不在缓存时 read_folio →
         * mpage_read_folio 同步读盘），无超时、不受 stop() 影响。逐页前再
         * 查一次 shutting_down，让已越过 2b 早退、正在读盘的 worker 尽快
         * 归还引用，避免 shutdown 的 40s 排空被同步块 I/O 拖住。
         */
        if (rt && READ_ONCE(rt->shutting_down))
            return -EAGAIN;

        folio = read_mapping_folio(mapping, pos >> PAGE_SHIFT, NULL);
        if (IS_ERR(folio))
            return PTR_ERR(folio);

        kaddr = kmap_local_folio(folio, off);
        memcpy(buf + pos, kaddr, chunk);
        kunmap_local(kaddr);
        folio_put(folio);

        pos += chunk;
    }

    return 0;
}

/* 读取整文件内容（上限截断）用于嵌入。 */
static int vecfs_shield_read_content(struct inode *inode,
                                     void **out, size_t *out_len)
{
    loff_t size = i_size_read(inode);
    size_t len;
    void *buf;
    int ret;

    if (size <= 0)
        return -EINVAL;

    len = min_t(loff_t, size, VECFS_SHIELD_MAX_EMBED_BYTES);

    buf = kvmalloc(len, GFP_KERNEL);
    if (!buf)
        return -ENOMEM;

    ret = vecfs_shield_read_pages(inode, buf, len);
    if (ret) {
        kvfree(buf);
        return ret;
    }

    *out = buf;
    *out_len = len;
    return 0;
}

/* ---------------------------------------------------------------------
 * 发布
 * ------------------------------------------------------------------- */

/*
 * 最小空闲 ordinal 分配（可跳过本批次已分配的 ordinal）。
 *
 * 批量插入在同一次文件派生中分配多个 ordinal，而 ordinal→node 绑定要等
 * 整个批量事务（vecfs_ann_batch_insert_file）建立后才可见。若沿用单条插入
 * 逐次调用本函数，会因绑定尚未建立而反复返回同一个 ordinal。故把本批次
 * 已分配的 ordinal（skip[0..skip_count)）一并传入跳过，保证批内 ordinal
 * 唯一；对已绑定到已有节点的 ordinal，仍靠 vecfs_ann_get_node 判重。
 */
static int vecfs_shield_alloc_ordinal(struct inode *inode,
                                      const u32 *skip, u32 skip_count,
                                      u32 *ordinal)
{
    u32 value;

    if (!inode || !ordinal)
        return -EINVAL;

    for (value = 0; value < VECFS_ANN_ORDINAL_AUTO; value++) {
        u32 node_id;
        u32 k;
        bool taken = false;
        int ret;

        for (k = 0; k < skip_count; k++) {
            if (skip[k] == value) {
                taken = true;
                break;
            }
        }
        if (taken)
            continue;

        ret = vecfs_ann_get_node(inode, value, &node_id);

        if (ret == -ENOENT) {
            *ordinal = value;
            return 0;
        }
        if (ret)
            return ret;

        cond_resched();
    }

    return -ENOSPC;
}

/*
 * 保守释放写路径 igrab 的 inode 引用：持 fs->lock 递减 derive_refs。
 *
 * 归零前再减 = 重复 iput（一次 igrab 被释放两次）——这是把 VFS 的
 * `BUG_ON` 触发成内核 oops 的典型路径。这里不直接 iput，而是 WARN +
 * 打印 inode 关键状态后跳过，把「崩内核」降级成「日志告警 + 一次引用
 * 泄漏」，代价远小于重启。计数为正则正常 iput。
 */
static void vecfs_shield_derive_put_ref(struct vecfs_shield_file_state *fs,
                                        struct inode *inode)
{
    struct vecfs_shield_runtime *rt = vecfs_shield_runtime(inode->i_sb);
    bool held;
    u32 state;
    int refs;

    mutex_lock(&fs->lock);
    if (fs->derive_refs > 0) {
        fs->derive_refs--;
        held = true;
    } else {
        held = false;
    }
    state = fs->state;
    refs = fs->derive_refs;
    mutex_unlock(&fs->lock);

    if (WARN_ON_ONCE(!held)) {
        pr_err_ratelimited(
            "vecfs: shield derive double-drop ref ino=%lu state=%u "
            "nlink=%u icount=%d refs=%d\n",
            inode->i_ino, state, inode->i_nlink,
            atomic_read(&inode->i_count), refs);
        return;
    }

    /* 与写路径 igrab 的 +1 配对：每释放一个 igrab 引用就递减总账。 */
    if (rt)
        atomic64_dec(&rt->derive_refs_total);

    iput(inode);
}

/*
 * 策略判定「不可嵌入/分块失败」的终态：清空复用表、推进语义版本、回到
 * CLEAN 并释放内容缓冲与 inode 引用。旧向量已由写路径置 stale，无需
 * 在此显式清除。调用后派生 worker 直接 return。
 */
static void vecfs_shield_derive_skip_file(struct vecfs_shield_file_state *fs,
                                          struct vecfs_shield_runtime *rt,
                                          struct inode *inode,
                                          u64 derive_version,
                                          void *content,
                                          struct vecfs_shield_chunk *chunks)
{
    vecfs_shield_reuse_free(fs);
    atomic64_inc(&rt->counters.files_skipped);

    mutex_lock(&fs->lock);
    fs->semantic_version = derive_version;
    fs->state = VECFS_SHIELD_FS_CLEAN;
    mutex_unlock(&fs->lock);
    vecfs_shield_state_wake(inode);

    kfree(chunks);
    kvfree(content);
    vecfs_shield_derive_put_ref(fs, inode);
}

/*
 * 派生 worker（workqueue 进程上下文，可睡眠）。
 * 完成后 iput 释放首次写时 igrab 的引用；过期/失败则重排队。
 *
 * P3 原子条件发布：新向量「插入即无效」，提交阶段持 fs->lock 复核
 * 版本后，再逐个把新向量翻转为有效（mark_node_valid）。版本复核与
 * 写路径的版本推进（同样持 fs->lock）互斥，因此复核通过后不会有
 * 并发写插入到提交之前；写入先推进版本、再标记 stale（见
 * vecfs_shield_write_begin），stale 标记作为兜底捕获提交竞态中遗漏
 * 的向量。
 */
/*
 * 事务式发布（C4）的版本复核回调：在批量插入的发布临界区内（writer_lock
 * 持有、publish_locked 之前）持 fs->lock 复核版本。返回 false 则整批回滚
 * （新节点不进入公共图），消除「插入即无效」节点在复核失败后永久残留的泄露。
 *
 * 该回调在 writer_lock 下执行、再取 fs->lock（writer_lock → fs->lock 锁序）；
 * 写路径只在释放 fs->lock 后才取 writer_lock（见 vecfs_shield_write_begin），
 * 因此不构成锁环。
 */
struct vecfs_shield_publish_guard_ctx {
    u64 derive_version;
    u32 *chunk_idx;   /* 本批嵌入块下标（to_embed_idx 前 count 项） */
    u32 count;        /* 本批嵌入块数（new_count） */
};

static bool vecfs_shield_publish_guard(struct inode *inode, void *ctx)
{
    struct vecfs_shield_file_state *fs = &VECFS_I(inode)->shield;
    struct vecfs_shield_publish_guard_ctx *gc = ctx;
    bool ok;
    u32 j;

    mutex_lock(&fs->lock);
    if (!inode->i_nlink) {
        ok = false;
    } else if (fs->target_version == gc->derive_version &&
               vecfs_shield_version(inode) == gc->derive_version) {
        /* 快路径：无并发写，整文件仍干净。 */
        ok = true;
    } else {
        /*
         * 慢路径（块独立可见性）：整文件版本已推进（无关块被并发写），但
         * 本批嵌入块若都未在派生版本之后被重写，仍可原子发布——稳定块不
         * 被无关块变更拖住。chunk_version 缺失条目视为 0（从未被写 → 有效）。
         */
        ok = true;
        for (j = 0; j < gc->count; j++) {
            u32 idx = gc->chunk_idx[j];

            if (idx < fs->chunk_version_count &&
                fs->chunk_written_version[idx] > gc->derive_version) {
                ok = false;
                break;
            }
        }
    }
    mutex_unlock(&fs->lock);

    return ok;
}

/* 幂等入队（须持 rt->pending_lock）。 */
static void vecfs_shield_pending_enqueue_locked(struct vecfs_shield_runtime *rt,
                                                struct vecfs_shield_file_state *fs)
{
    if (list_empty(&fs->pending_node))
        list_add_tail(&fs->pending_node, &rt->pending_files);
}

/* 回到 PENDING 重试：引用保留、重新入队并武装 drain；否则终结并归还引用。 */
static void vecfs_shield_requeue(struct vecfs_shield_file_state *fs,
                                 struct inode *inode,
                                 struct vecfs_shield_runtime *rt)
{
    mutex_lock(&fs->lock);
    if (inode->i_nlink && !fs->dead &&
        !READ_ONCE(rt->shutting_down)) {
        fs->state = VECFS_SHIELD_FS_PENDING;
        spin_lock(&rt->pending_lock);
        vecfs_shield_pending_enqueue_locked(rt, fs);
        spin_unlock(&rt->pending_lock);
        mod_delayed_work(system_long_wq, &rt->drain_work,
                         READ_ONCE(rt->window_jiffies));
        mutex_unlock(&fs->lock);
        return;
    }
    fs->state = VECFS_SHIELD_FS_CLEAN;
    mutex_unlock(&fs->lock);
    vecfs_shield_state_wake(inode);
    vecfs_shield_derive_put_ref(fs, inode);
}

/*
 * 复核版本并提交 CLEAN（无新节点入图的成功路径）。返回 true=已提交（调用方
 * 随后 mark_node_valid→释放缓冲→put_ref）；false=已 requeue（引用保留，调用方
 * 只需释放缓冲）。mark_node_valid 需单独取 writer_lock（不能持 fs->lock，否则
 * 与 publish_guard 的 writer_lock→fs->lock 锁序相逆），故由调用方在 CLEAN 提交
 * 之后、put_ref 之前执行。
 */
static bool vecfs_shield_commit_clean(struct vecfs_shield_file_state *fs,
                                      struct inode *inode,
                                      struct vecfs_shield_runtime *rt,
                                      u64 derive_version)
{
    mutex_lock(&fs->lock);
    if (fs->target_version != derive_version || !inode->i_nlink) {
        mutex_unlock(&fs->lock);
        atomic64_inc(&rt->counters.tasks_expired);
        atomic64_inc(&rt->counters.publishes_rejected);
        vecfs_shield_requeue(fs, inode, rt);
        return false;
    }
    fs->semantic_version = derive_version;
    fs->state = VECFS_SHIELD_FS_CLEAN;
    mutex_unlock(&fs->lock);
    vecfs_shield_state_wake(inode);
    return true;
}

/* 单个文件派生准备的结果。 */
enum vecfs_shield_prepare_outcome {
    VECFS_SHIELD_PREP_READY = 0,   /* new_count>0，batch_items 已填好，待批量插入 */
    VECFS_SHIELD_PREP_HANDLED = 1, /* 已终态（skip/全部复用/requeue/删除/关闭） */
};

struct vecfs_shield_prepare {
    struct vecfs_shield_file_state *fs;
    struct inode *inode;
    u64 derive_version;
    struct vecfs_shield_publish_guard_ctx guard_ctx;

    /* READY 时持有、由 drain worker 在插入后释放的缓冲。 */
    struct vecfs_ann_batch_item *batch_items;
    u8 *batch_vecs;
    u32 *new_ordinals;
    u32 *to_embed_idx;
    u64 *chunk_hash;
    u32 new_count;

    /* 完整 L1：嵌入返回后按块过滤剔除过 stale 块 → 稳定块发布后须 requeue
     * （而非 commit_clean 提交整文件），让 stale 块以新内容重新派生。 */
    bool partial;

    int outcome;
};

/*
 * 单个文件的派生准备（读/判别/分块/两遍派生/嵌入/填批量项）。对应旧
 * derive_work 的 1~7.3 步，但不执行批量插入与 node id 分配（那些统一由
 * vecfs_ann_batch_insert_files_many 在批量边界完成）。返回 prep->outcome。
 */
static void vecfs_shield_prepare_file(struct vecfs_shield_runtime *rt,
                                      struct inode *inode,
                                      struct vecfs_shield_prepare *prep)
{
    struct vecfs_shield_file_state *fs = &VECFS_I(inode)->shield;
    struct vecfs_sb_info *sbi = VECFS_SB(inode->i_sb);
    const struct vecfs_shield_strategy_ops *strategy;
    u32 dim = sbi->s_ann.config.dimension;
    size_t vec_bytes = (size_t)sbi->s_ann.config.vector_bytes;
    u32 chunk_size;
    u32 strat_flags;
    void *content = NULL;
    size_t content_len = 0;
    struct vecfs_shield_chunk *chunks = NULL;
    struct vecfs_ann_batch_item *batch_items = NULL;
    u8 *batch_vecs = NULL;
    u32 *new_ordinals = NULL;
    u32 *to_embed_idx = NULL;
    u64 *chunk_hash = NULL;
    u32 new_count = 0;
    u32 n_embed = 0;
    u32 max_chunks;
    u32 nchunks;
    u64 derive_version;
    u64 t_embed;
    bool publish;
    int ret;
    u32 i;
    u32 j;

    memset(prep, 0, sizeof(*prep));
    prep->fs = fs;
    prep->inode = inode;
    prep->outcome = VECFS_SHIELD_PREP_HANDLED;

    /* 1) 抓取目标版本，转 DERIVING。 */
    mutex_lock(&fs->lock);
    if (fs->state != VECFS_SHIELD_FS_PENDING) {
        mutex_unlock(&fs->lock);
        vecfs_shield_derive_put_ref(fs, inode);
        return;
    }
    derive_version = fs->target_version;
    fs->state = VECFS_SHIELD_FS_DERIVING;
    mutex_unlock(&fs->lock);

    prep->derive_version = derive_version;
    prep->guard_ctx.derive_version = derive_version;

    atomic64_inc(&rt->counters.tasks_begun);

    /* 2) 文件已删除则放弃。 */
    if (!inode->i_nlink) {
        mutex_lock(&fs->lock);
        fs->state = VECFS_SHIELD_FS_CLEAN;
        mutex_unlock(&fs->lock);
        vecfs_shield_state_wake(inode);
        vecfs_shield_derive_put_ref(fs, inode);
        return;
    }

    /* 2b) 卸载关闭：不再派生，直接归还引用。 */
    if (READ_ONCE(rt->shutting_down)) {
        mutex_lock(&fs->lock);
        fs->state = VECFS_SHIELD_FS_CLEAN;
        mutex_unlock(&fs->lock);
        vecfs_shield_state_wake(inode);
        vecfs_shield_derive_put_ref(fs, inode);
        return;
    }

    /* 3) 读内容（可睡眠，不持锁）。 */
    ret = vecfs_shield_read_content(inode, &content, &content_len);
    if (ret)
        goto requeue;

    /* 快照策略配置。 */
    chunk_size = READ_ONCE(rt->chunk_size);
    if (!chunk_size)
        chunk_size = VECFS_SHIELD_DEFAULT_CHUNK_SIZE;
    strat_flags = READ_ONCE(rt->strat_flags);
    strategy = READ_ONCE(rt->strategy);
    if (!strategy)
        strategy = &vecfs_shield_builtin_strategy;

    /* 4) 条件发布预检（权威检查在提交阶段，持 fs->lock 复核）。 */
    mutex_lock(&fs->lock);
    publish = inode->i_nlink &&
              fs->target_version == derive_version &&
              vecfs_shield_version(inode) == derive_version;
    mutex_unlock(&fs->lock);

    atomic64_inc(&rt->counters.tasks_ended);

    if (!publish) {
        atomic64_inc(&rt->counters.tasks_expired);
        atomic64_inc(&rt->counters.publishes_rejected);
        goto requeue;
    }

    /* 5) 处置判别：二进制文件 SKIP，不产生向量。 */
    if (strat_flags & VECFS_SHIELD_STRAT_F_CLASSIFY) {
        int disp = strategy->classify(content, content_len);

        if (disp < 0) {
            if (disp == -EAGAIN)
                goto requeue;
            disp = VECFS_SHIELD_DISP_SKIP;
        }

        if (disp == VECFS_SHIELD_DISP_SKIP) {
            vecfs_shield_derive_skip_file(fs, rt, inode, derive_version,
                                          content, NULL);
            return;
        }
    }

    /* 6) 分块。 */
    max_chunks = (u32)((content_len + chunk_size - 1) / chunk_size);
    if (!max_chunks)
        max_chunks = 1;
    if (max_chunks > VECFS_SHIELD_MAX_CHUNKS)
        max_chunks = VECFS_SHIELD_MAX_CHUNKS;

    chunks = kcalloc(max_chunks, sizeof(*chunks), GFP_KERNEL);
    if (!chunks) {
        ret = -ENOMEM;
        goto requeue;
    }

    {
        u32 count = 0;

        ret = strategy->chunk(content, content_len, chunk_size, chunks,
                              max_chunks, &count);
        if (ret < 0) {
            if (ret == -EAGAIN)
                goto requeue;
            vecfs_shield_derive_skip_file(fs, rt, inode, derive_version,
                                          content, chunks);
            return;
        }
        nchunks = count;
    }
    if (!nchunks) {
        vecfs_shield_derive_skip_file(fs, rt, inode, derive_version,
                                      content, chunks);
        return;
    }

    batch_items = kcalloc(nchunks, sizeof(*batch_items), GFP_KERNEL);
    if (!batch_items) {
        ret = -ENOMEM;
        goto requeue;
    }

    batch_vecs = kvmalloc((size_t)nchunks * vec_bytes, GFP_KERNEL);
    if (!batch_vecs) {
        ret = -ENOMEM;
        goto requeue;
    }

    new_ordinals = kcalloc(nchunks, sizeof(*new_ordinals), GFP_KERNEL);
    if (!new_ordinals) {
        ret = -ENOMEM;
        goto requeue;
    }

    /*
     * 复用表无论开关都保留：复用关闭时也照常维护上一版本内容哈希，
     * 以便第一遍判定「内容未变但被重嵌」的冗余块（chunks_redundant 消融口径）。
     */
    ret = vecfs_shield_reuse_ensure(fs, chunk_size, nchunks);
    if (ret)
        goto requeue;

    /* 7.1) 第一遍：clamp + 哈希 + 复用判定，收集已变块下标。 */
    to_embed_idx = kcalloc(nchunks, sizeof(*to_embed_idx), GFP_KERNEL);
    chunk_hash = kcalloc(nchunks, sizeof(*chunk_hash), GFP_KERNEL);
    if (!to_embed_idx || !chunk_hash) {
        ret = -ENOMEM;
        goto requeue;
    }

    for (i = 0; i < nchunks; i++) {
        struct vecfs_shield_chunk *ch = &chunks[i];
        u64 hash;
        u32 ordinal;

        if (ch->offset >= content_len)
            break;
        if (ch->length > content_len - ch->offset)
            ch->length = content_len - ch->offset;
        if (!ch->length)
            continue;

        hash = vecfs_shield_content_hash(content + ch->offset, ch->length);
        chunk_hash[i] = hash;

        if (vecfs_shield_reuse_lookup(fs, i, hash, &ordinal)) {
            if (strat_flags & VECFS_SHIELD_STRAT_F_REUSE) {
                vecfs_ann_clear_file_vector_stale(inode, ordinal,
                                                  derive_version);
                atomic64_inc(&rt->counters.chunks_reused);
            } else {
                /* 复用关闭：内容未变却仍需重嵌 → 计入冗余消融口径。 */
                atomic64_inc(&rt->counters.chunks_redundant);
                to_embed_idx[n_embed++] = i;
            }
        } else {
            to_embed_idx[n_embed++] = i;
        }
    }

    /* 7.2) 第二遍：一次批量嵌入所有已变块。 */
    if (n_embed) {
        struct vecfs_shield_chunk *to_embed;
        int eret;

        to_embed = kcalloc(n_embed, sizeof(*to_embed), GFP_KERNEL);
        if (!to_embed) {
            ret = -ENOMEM;
            goto requeue;
        }
        for (j = 0; j < n_embed; j++)
            to_embed[j] = chunks[to_embed_idx[j]];

        if (strategy->embed_batch) {
            u64 ifirst = U64_MAX, ilast = 0;
            u32 nkeys;

            /* 记录在途批量嵌入的内容区间（供写路径按重叠取消判定）。 */
            for (j = 0; j < n_embed; j++) {
                if (to_embed[j].offset < ifirst)
                    ifirst = to_embed[j].offset;
                if (to_embed[j].offset + to_embed[j].length > ilast)
                    ilast = to_embed[j].offset + to_embed[j].length;
            }

            /* 完整 L1：发布在途块输入键注册表（先填表、再发布 nkeys，写路径
             * 读 nkeys 后按块做需求驱动取消）。超过上限退回整批取消。 */
            nkeys = n_embed > VECFS_SHIELD_INFLIGHT_MAX ?
                        VECFS_SHIELD_INFLIGHT_MAX : n_embed;
            for (j = 0; j < nkeys; j++) {
                u32 ci = to_embed_idx[j];

                WRITE_ONCE(fs->in_flight_keys[j].key,
                           vecfs_shield_input_key(to_embed[j].offset,
                                                  chunk_hash[ci]));
                WRITE_ONCE(fs->in_flight_keys[j].offset, to_embed[j].offset);
                WRITE_ONCE(fs->in_flight_keys[j].length, to_embed[j].length);
            }
            WRITE_ONCE(fs->in_flight_overflow,
                       n_embed > VECFS_SHIELD_INFLIGHT_MAX ? 1 : 0);
            WRITE_ONCE(fs->in_flight_nkeys, nkeys);

            WRITE_ONCE(fs->in_flight_first, ifirst);
            WRITE_ONCE(fs->in_flight_last, ilast);
            WRITE_ONCE(fs->in_flight_upcall, 0);
            t_embed = ktime_get_ns();
            eret = strategy->embed_batch(inode, derive_version, content,
                                         to_embed, n_embed, dim,
                                         (u32 *)batch_vecs,
                                         &fs->in_flight_upcall);
            /* 消融观测：upcall 阻塞等待耗时（含 Ollama 计算，对齐用户态 embed）。 */
            atomic64_add(ktime_get_ns() - t_embed,
                         &rt->counters.upcall_wait_ns);
            atomic64_inc(&rt->counters.upcall_count);
            WRITE_ONCE(fs->in_flight_nkeys, 0);
            WRITE_ONCE(fs->in_flight_overflow, 0);
            WRITE_ONCE(fs->in_flight_first, 0);
            WRITE_ONCE(fs->in_flight_last, 0);
        } else {
            eret = 0;
            for (j = 0; j < n_embed; j++) {
                u8 *dst = batch_vecs + (size_t)j * vec_bytes;

                eret = strategy->embed(content + to_embed[j].offset,
                                       to_embed[j].length, dim, (u32 *)dst);
                if (eret)
                    break;
            }
        }
        kfree(to_embed);

        if (eret == -EAGAIN || eret == -ECANCELED) {
            ret = eret;
            goto requeue;
        }
        if (eret) {
            n_embed = 0;
        }
    }

    if (READ_ONCE(rt->shutting_down) || !inode->i_nlink) {
        ret = -EAGAIN;
        goto requeue;
    }

    /*
     * 7.2b) 完整 L1 部分发布：嵌入返回后按块过滤——把派生版本之后被重写的块
     * （chunk_written_version > derive_version，L2 权威）从本批剔除，只发布
     * 稳定块；被剔除块由 requeue 以新内容重新派生。向量随 to_embed_idx 一起
     * 前移紧凑（w <= j，memmove 处理重叠安全）。全部在途块被覆盖则直接
     * requeue，不发布任何块。
     * 消融开关 F_PARTIAL 关闭时跳过本过滤：退回 scoped L1 的整批取消行为
     * （重叠写整批作废，稳定块也不发布），供部分重叠写微基准 A/B。
     */
    if ((strat_flags & VECFS_SHIELD_STRAT_F_PARTIAL) && n_embed) {
        u32 w = 0;
        bool had_stale = false;

        mutex_lock(&fs->lock);
        for (j = 0; j < n_embed; j++) {
            u32 idx = to_embed_idx[j];
            bool stale = (idx < fs->chunk_version_count &&
                          fs->chunk_written_version[idx] > derive_version);

            if (stale) {
                had_stale = true;
                continue;
            }
            if (w != j) {
                to_embed_idx[w] = idx;
                memmove(batch_vecs + (size_t)w * vec_bytes,
                        batch_vecs + (size_t)j * vec_bytes, vec_bytes);
            }
            w++;
        }
        mutex_unlock(&fs->lock);

        if (had_stale && !w) {
            /* 全部在途块都被新写覆盖：结果全作废，requeue 重新派生。 */
            goto requeue;
        }
        n_embed = w;
        if (had_stale)
            prep->partial = true;
    }

    /* 7.3) 逐个（仅已变块）：分配 ordinal + 填批量项。 */
    for (j = 0; j < n_embed; j++) {
        struct vecfs_shield_chunk *ch = &chunks[to_embed_idx[j]];
        struct vecfs_ann_batch_item *bi = &batch_items[new_count];
        u8 *dst = batch_vecs + (size_t)new_count * vec_bytes;
        u32 ordinal;

        ret = vecfs_shield_alloc_ordinal(
            inode, new_ordinals, new_count, &ordinal);
        if (ret)
            goto requeue;

        memset(bi, 0, sizeof(*bi));
        bi->request.vector = dst;
        bi->request.vector_bytes = vec_bytes;
        bi->request.external_id = VECFS_ANN_EXTERNAL_ID_AUTO;
        bi->request.owner_ino = inode->i_ino;
        bi->request.owner_generation = inode->i_generation;
        bi->request.vector_ordinal = ordinal;
        bi->request.content_offset = ch->offset;
        bi->request.content_length = ch->length;
        bi->request.owner_type = VECFS_ANN_OWNER_FILE;
        bi->request.flags = VECFS_ANN_OBJ_F_HAS_CONTENT_RANGE;
        bi->request.start_invalid =
            (strat_flags & VECFS_SHIELD_STRAT_F_VALIDATE) != 0;

        /* chunks_embedded 计数已移至 drain worker 成功发布后（回滚的不计）。 */
        new_ordinals[new_count++] = ordinal;
    }

    /* 全部复用（无新节点）：提交 CLEAN（版本复核失败则 requeue）。 */
    if (!new_count) {
        kfree(new_ordinals);
        kfree(to_embed_idx);
        kfree(chunk_hash);
        kfree(batch_items);
        kvfree(batch_vecs);
        kfree(chunks);
        kvfree(content);
        if (vecfs_shield_commit_clean(fs, inode, rt, derive_version))
            vecfs_shield_derive_put_ref(fs, inode);
        return;
    }

    /* content/chunks 不再需要，立即释放；其余缓冲由 drain worker 在插入后释放。 */
    kfree(chunks);
    kvfree(content);

    prep->batch_items = batch_items;
    prep->batch_vecs = batch_vecs;
    prep->new_ordinals = new_ordinals;
    prep->to_embed_idx = to_embed_idx;
    prep->chunk_hash = chunk_hash;
    prep->new_count = new_count;
    prep->guard_ctx.chunk_idx = to_embed_idx;
    prep->guard_ctx.count = new_count;
    prep->outcome = VECFS_SHIELD_PREP_READY;
    return;

requeue:
    kfree(new_ordinals);
    kfree(to_embed_idx);
    kfree(chunk_hash);
    kfree(batch_items);
    kvfree(batch_vecs);
    kfree(chunks);
    kvfree(content);
    vecfs_shield_requeue(fs, inode, rt);
}

/*
 * 全局 drain worker：把待派生队列里的一批文件聚合成一次（或几次）多文件批量
 * 插入，消除每文件一次 O(N) 快照克隆。替代旧的每文件 derive_work。
 */
static void vecfs_shield_drain_work(struct work_struct *work)
{
    struct delayed_work *dwork = to_delayed_work(work);
    struct vecfs_shield_runtime *rt =
        container_of(dwork, struct vecfs_shield_runtime, drain_work);
    struct super_block *sb = NULL;
    struct vecfs_shield_file_state *fs;
    struct vecfs_shield_file_state *tmp;
    struct vecfs_shield_prepare *preps = NULL;
    struct vecfs_shield_prepare **ready = NULL;
    struct vecfs_ann_batch_file *files = NULL;
    LIST_HEAD(batch);
    u32 n = 0;
    u32 n_ready = 0;
    u32 cursor;
    u32 i;
    u32 j;
    u32 kept;
    int ret;

    if (READ_ONCE(rt->shutting_down)) {
        spin_lock(&rt->pending_lock);
        list_splice_init(&rt->pending_files, &batch);
        spin_unlock(&rt->pending_lock);

        list_for_each_entry_safe(fs, tmp, &batch, pending_node) {
            struct vecfs_inode *vi = container_of(fs, struct vecfs_inode, shield);

            list_del_init(&fs->pending_node);
            mutex_lock(&fs->lock);
            fs->state = VECFS_SHIELD_FS_CLEAN;
            mutex_unlock(&fs->lock);
            vecfs_shield_derive_put_ref(fs, &vi->vfs_inode);
        }
        return;
    }

    /* 一次性摘出当前所有待派生文件（不跨轮反复重处理同一 requeue）。 */
    spin_lock(&rt->pending_lock);
    list_splice_init(&rt->pending_files, &batch);
    spin_unlock(&rt->pending_lock);

    if (list_empty(&batch))
        return;

    list_for_each_entry(fs, &batch, pending_node)
        n++;

    preps = kcalloc(n, sizeof(*preps), GFP_KERNEL);
    ready = kcalloc(n, sizeof(*ready), GFP_KERNEL);
    files = kcalloc(n, sizeof(*files), GFP_KERNEL);
    if (!preps || !ready || !files) {
        /* 分配失败：逐个终结（归还引用），后续写会重新触发。 */
        list_for_each_entry_safe(fs, tmp, &batch, pending_node) {
            struct vecfs_inode *vi = container_of(fs, struct vecfs_inode, shield);

            list_del_init(&fs->pending_node);
            mutex_lock(&fs->lock);
            fs->state = VECFS_SHIELD_FS_CLEAN;
            mutex_unlock(&fs->lock);
            vecfs_shield_derive_put_ref(fs, &vi->vfs_inode);
        }
        kfree(files);
        kfree(ready);
        kfree(preps);
        return;
    }

    i = 0;
    list_for_each_entry_safe(fs, tmp, &batch, pending_node) {
        struct vecfs_inode *vi = container_of(fs, struct vecfs_inode, shield);

        list_del_init(&fs->pending_node);
        vecfs_shield_prepare_file(rt, &vi->vfs_inode, &preps[i]);
        i++;
    }

    for (i = 0; i < n; i++) {
        if (preps[i].outcome != VECFS_SHIELD_PREP_READY)
            continue;

        if (!sb)
            sb = preps[i].inode->i_sb;

        ready[n_ready] = &preps[i];
        files[n_ready].inode = preps[i].inode;
        files[n_ready].items = preps[i].batch_items;
        files[n_ready].count = preps[i].new_count;
        files[n_ready].guard_ctx = &preps[i].guard_ctx;
        n_ready++;
    }

    /* 按 ≤ VECFS_ANN_BATCH_MAX_ITEMS 分批插入：每批一次 clone。 */
    cursor = 0;
    while (cursor < n_ready) {
        u32 total = 0;
        u32 end = cursor;

        while (end < n_ready) {
            u32 cnt = files[end].count;

            if (total && total + cnt > VECFS_ANN_BATCH_MAX_ITEMS)
                break;
            total += cnt;
            end++;
        }

        /*
         * 预筛（批级块独立可见性）：同批文件顺序派生，先派生的文件会被后面
         * 文件的派生耗时拖出很长的「prepare 完成 → 提交」窗口。若某文件在这
         * 段窗口里又被写入，其 guard 已不成立——单独 requeue 它、就地压缩出
         * 本批，而非让 batch_insert 末尾「任一 stale → 整批回滚」把同批干净
         * 文件也一并作废重嵌（级联放大，λ 高时显著）。batch_insert 末尾的
         * guard 仍保留，作为「预筛到提交」极小窗口的权威兜底。
         */
        kept = 0;
        for (j = cursor; j < end; j++) {
            struct vecfs_shield_prepare *p = ready[j];

            if (!vecfs_shield_publish_guard(p->inode, &p->guard_ctx)) {
                atomic64_inc(&rt->counters.tasks_expired);
                atomic64_inc(&rt->counters.publishes_rejected);
                kfree(p->new_ordinals);
                kfree(p->to_embed_idx);
                kfree(p->chunk_hash);
                kfree(p->batch_items);
                kvfree(p->batch_vecs);
                vecfs_shield_requeue(p->fs, p->inode, rt);
                continue;
            }
            files[cursor + kept] = files[j];
            ready[cursor + kept] = ready[j];
            kept++;
        }

        if (!kept) {
            cursor = end;
            continue;
        }

        total = 0;
        for (j = 0; j < kept; j++)
            total += files[cursor + j].count;

        ret = vecfs_ann_batch_insert_files_many(
            sb, &files[cursor], kept, total,
            vecfs_shield_publish_guard,
            !(rt->strat_flags & VECFS_SHIELD_STRAT_F_COW));

        if (ret == -ESTALE) {
            atomic64_inc(&rt->counters.tasks_expired);
            atomic64_inc(&rt->counters.publishes_rejected);
        }

        for (j = 0; j < kept; j++) {
            struct vecfs_shield_prepare *p = ready[cursor + j];
            u32 *ord = p->new_ordinals;
            u32 *tei = p->to_embed_idx;
            u64 *ch = p->chunk_hash;
            u32 cnt = p->new_count;
            u32 k;

            if (ret) {
                /* 残差回滚（预筛后仍过期）：复用表/valid 均不更新，逐文件 requeue。 */
                kfree(ord);
                kfree(tei);
                kfree(ch);
                kfree(p->batch_items);
                kvfree(p->batch_vecs);
                vecfs_shield_requeue(p->fs, p->inode, rt);
                continue;
            }

            /*
             * 插入成功：新节点已由 publish_guard 按块复核通过、内容有效，
             * 立即翻 valid（块独立可见性：即使整文件仍需派生其它变更块，本批
             * 稳定块也应立即可见）。并发写的 mark_content_stale 会兜底把被重写
             * 块重新置 stale，故此处提前翻 valid 不会泄露过期向量。
             */
            if (rt->strat_flags & VECFS_SHIELD_STRAT_F_VALIDATE)
                for (k = 0; k < cnt; k++)
                    vecfs_ann_mark_node_valid(p->inode, ord[k]);

            /*
             * 再更新复用表（nodes 已发布），随后提交 CLEAN。复用关闭时也更新：
             * chunks_redundant 需对照「上一版本内容哈希」判定，表缺失会退化为
             * 全量重嵌、无法统计冗余（见 derive 第一遍）。commit_clean 复核整
             * 文件版本：若仍有更新的 target（无关块并发写），返回 false 并
             * requeue 重新派生剩余变更块（本批已发布块会经复用表命中）。
             */
            for (k = 0; k < cnt; k++)
                vecfs_shield_reuse_update(p->fs, tei[k], ch[tei[k]], ord[k]);

            /* 成功发布才计数：cnt 块是「已嵌入且已发布」的块（回滚的不计）。 */
            atomic64_add(cnt, &rt->counters.chunks_embedded);

            if (p->partial) {
                /*
                 * 完整 L1：本文件有被按块过滤剔除的 stale 块。稳定块已发布
                 * 并更新复用表，requeue 让 stale 块以新内容重新派生——下一轮
                 * 派生经复用表命中稳定块、只重嵌 stale 块。不 commit_clean：
                 * 整文件仍有待派生变更，semantic_version 保持旧值。
                 */
                vecfs_shield_requeue(p->fs, p->inode, rt);
            } else if (vecfs_shield_commit_clean(p->fs, p->inode, rt,
                                                p->derive_version)) {
                vecfs_shield_derive_put_ref(p->fs, p->inode);
            }

            kfree(ord);
            kfree(tei);
            kfree(ch);
            kfree(p->batch_items);
            kvfree(p->batch_vecs);
        }

        cursor = end;
    }

    kfree(files);
    kfree(ready);
    kfree(preps);
}

void vecfs_shield_file_init(struct vecfs_inode *vi)
{
    struct vecfs_shield_file_state *fs = &vi->shield;

    mutex_init(&fs->lock);
    fs->state = VECFS_SHIELD_FS_CLEAN;
    fs->target_version = 0;
    fs->semantic_version = 0;
    fs->pending_since = 0;
    fs->first_change = 0;
    fs->last_change = 0;
    fs->derive_refs = 0;
    fs->dead = false;
    fs->in_flight_upcall = 0;
    fs->chunk_written_version = NULL;
    fs->chunk_version_count = 0;
    fs->chunk_version_chunk_size = 0;
    fs->in_flight_first = 0;
    fs->in_flight_last = 0;
    fs->in_flight_nkeys = 0;
    fs->in_flight_overflow = 0;
    fs->reuse = NULL;
    INIT_LIST_HEAD(&fs->pending_node);
}

void vecfs_shield_write_hook(struct inode *inode, u64 start, u64 len)
{
    struct vecfs_inode *vi = VECFS_I(inode);
    struct vecfs_shield_file_state *fs = &vi->shield;
    struct vecfs_shield_runtime *rt =
        vecfs_shield_runtime(inode->i_sb);
    u64 version;
    u64 w_jiffies;
    bool first;
    bool idle;
    bool only;

    if (!rt || vecfs_shield_mode(inode->i_sb) != VECFS_SHIELD_MODE_SHIELD)
        return;
    if (!S_ISREG(inode->i_mode))
        return;

    version = vecfs_shield_version(inode);
    w_jiffies = READ_ONCE(rt->window_jiffies);
    if (!w_jiffies)
        w_jiffies = usecs_to_jiffies(VECFS_SHIELD_DEFAULT_WINDOW_US);

    /*
     * 全程持 fs->lock 完成「状态推进 + 排队」。igrab / queue/mod_delayed_work
     * 均非阻塞，可安全在锁内执行；把排队放进锁内、与 dead / shutting_down
     * 检查同临界区，才能在 file_destroy 置 dead 后杜绝再武装 work。
     */
    mutex_lock(&fs->lock);
    if (fs->dead || READ_ONCE(rt->shutting_down)) {
        mutex_unlock(&fs->lock);
        return;
    }

    first = (fs->state == VECFS_SHIELD_FS_CLEAN);
    /*
     * 自适应去抖（idle fast-path）：文件写前已空闲超过一个窗口（距上次写 > ω），
     * 说明没有正在进行的突发要合并，改为立即派生（ω_eff=0），省掉欠载时的 ω
     * 陈旧税。pending_since 记录上一 PENDING 时段的最后一次写时刻（CLEAN 后不清），
     * 首写（从未写过）pending_since=0 → 恒 idle。仅在 F_ADAPTIVE_WINDOW 开时生效。
     */
    idle = first && (READ_ONCE(rt->strat_flags) &
                     VECFS_SHIELD_STRAT_F_ADAPTIVE_WINDOW) &&
           (jiffies - fs->pending_since) > w_jiffies;
    fs->state = VECFS_SHIELD_FS_PENDING;
    fs->target_version = version;
    fs->pending_since = jiffies;
    if (first) {
        fs->first_change = start;
        fs->last_change = start + len;
    } else {
        if (start < fs->first_change)
            fs->first_change = start;
        if (start + len > fs->last_change)
            fs->last_change = start + len;
    }

    /*
     * 首次进入 PENDING 时 igrab 持引用并入全局待派生队列、武装 drain worker；
     * 后续写去抖：合并区间并顺延窗口（mod_delayed_work，不新增引用、不重复入队）。
     */
    if (first) {
        if (!igrab(inode)) {
            fs->state = VECFS_SHIELD_FS_CLEAN;
            mutex_unlock(&fs->lock);
            vecfs_shield_state_wake(inode);
            return;
        }
        fs->derive_refs++;
        atomic64_inc(&rt->derive_refs_total);

        spin_lock(&rt->pending_lock);
        vecfs_shield_pending_enqueue_locked(rt, fs);
        /* 自适应：仅当本文件是全局队列中唯一待派生文件（无其它文件处于突发）
         * 才允许立即派生。否则 0 延迟会提前触发全局 drain、打断其它文件去抖。 */
        only = list_is_singular(&rt->pending_files);
        spin_unlock(&rt->pending_lock);

        mod_delayed_work(system_long_wq, &rt->drain_work,
                         (idle && only) ? 0 : w_jiffies);
    } else {
        mod_delayed_work(system_long_wq, &rt->drain_work, w_jiffies);
    }
    mutex_unlock(&fs->lock);
}

int vecfs_shield_file_state_get(struct inode *inode,
                                struct vecfs_shield_file_state_info *out)
{
    struct vecfs_inode *vi = VECFS_I(inode);
    struct vecfs_shield_file_state *fs = &vi->shield;

    if (!inode || !out)
        return -EINVAL;
    if (!S_ISREG(inode->i_mode))
        return -EINVAL;

    memset(out, 0, sizeof(*out));

    mutex_lock(&fs->lock);
    out->state = fs->state;
    out->target_version = fs->target_version;
    out->semantic_version = fs->semantic_version;
    out->first_change = fs->first_change;
    out->last_change = fs->last_change;
    mutex_unlock(&fs->lock);

    out->shield_version = vecfs_shield_version(inode);
    return 0;
}

/* ---------------------------------------------------------------------
 * P0：基线任务生命周期 + 查询统计观测
 * ------------------------------------------------------------------- */

int vecfs_shield_task_begin(struct inode *inode, u64 *target_version)
{
    struct vecfs_shield_runtime *rt;
    u64 version;

    if (!inode || !target_version)
        return -EINVAL;

    if (!S_ISREG(inode->i_mode))
        return -EINVAL;

    rt = vecfs_shield_runtime(inode->i_sb);
    if (!rt)
        return -ENODEV;

    version = vecfs_shield_version(inode);

    atomic64_inc(&rt->counters.tasks_begun);

    *target_version = version;
    return 0;
}

int vecfs_shield_task_end(struct inode *inode, u64 target_version,
                          bool *expired)
{
    struct vecfs_shield_runtime *rt;
    u64 now;
    bool is_expired;

    if (!inode || !expired)
        return -EINVAL;

    if (!S_ISREG(inode->i_mode))
        return -EINVAL;

    rt = vecfs_shield_runtime(inode->i_sb);
    if (!rt)
        return -ENODEV;

    now = vecfs_shield_version(inode);
    is_expired = now != target_version;

    atomic64_inc(&rt->counters.tasks_ended);

    if (is_expired) {
        atomic64_inc(&rt->counters.tasks_expired);

        /*
         * CONDITIONAL：发布前校验。目标版本已过期，派生结果不应
         * 发布；这里只递增拒绝计数，真正跳过发布由调用方（派生
         * 流水线）依据 *expired 完成。后续阶段（版本化映射 + 条件
         * 发布）会把拒绝下沉到内核侧强制。注意这里刻意不调用
         * vecfs_ann_mark_content_stale——把已有向量回溯置 stale 是
         * VALIDATED（查询时校验）的职责，而非 CONDITIONAL。
         */
        if (vecfs_shield_mode(inode->i_sb) ==
            VECFS_SHIELD_MODE_CONDITIONAL)
            atomic64_inc(&rt->counters.publishes_rejected);
    }

    *expired = is_expired;
    return 0;
}

void vecfs_shield_count_query(struct super_block *sb, u32 result_count)
{
    struct vecfs_shield_runtime *rt = vecfs_shield_runtime(sb);

    if (!rt)
        return;

    atomic64_inc(&rt->counters.queries);
    atomic64_add(result_count, &rt->counters.results_returned);
}

void vecfs_shield_stats_snapshot(struct super_block *sb,
                                 struct vecfs_shield_stats *out)
{
    struct vecfs_shield_runtime *rt = vecfs_shield_runtime(sb);
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;

    if (!rt || !out)
        return;

    out->writes =
        atomic64_read(&rt->counters.writes);
    out->tasks_begun =
        atomic64_read(&rt->counters.tasks_begun);
    out->tasks_ended =
        atomic64_read(&rt->counters.tasks_ended);
    out->tasks_expired =
        atomic64_read(&rt->counters.tasks_expired);
    out->publishes_rejected =
        atomic64_read(&rt->counters.publishes_rejected);
    out->queries =
        atomic64_read(&rt->counters.queries);
    out->results_returned =
        atomic64_read(&rt->counters.results_returned);
    out->chunks_embedded =
        atomic64_read(&rt->counters.chunks_embedded);
    out->chunks_reused =
        atomic64_read(&rt->counters.chunks_reused);
    out->chunks_redundant =
        atomic64_read(&rt->counters.chunks_redundant);
    out->files_skipped =
        atomic64_read(&rt->counters.files_skipped);
    out->upcall_wait_ns =
        atomic64_read(&rt->counters.upcall_wait_ns);
    out->upcall_count =
        atomic64_read(&rt->counters.upcall_count);

    /*
     * R5f/R5g 消融观测计数器实际落在 ann cache（写路径 / 派生 clone 路径消费），
     * 这里从 cache 读出并入 stats 快照。cache 缺失时保持 0。
     */
    sbi = VECFS_SB(sb);
    cache = sbi ? sbi->s_ann_cache : NULL;
    if (cache) {
        out->stale_sync_writes =
            atomic64_read(&cache->stale_sync_writes);
        out->stale_deferred =
            atomic64_read(&cache->stale_deferred);
        out->clone_ns =
            atomic64_read(&cache->clone_ns);
        out->clone_count =
            atomic64_read(&cache->clone_count);
        out->persist_ns =
            atomic64_read(&cache->persist_ns);
        out->persist_count =
            atomic64_read(&cache->persist_count);
        out->build_ns =
            atomic64_read(&cache->build_ns);
        out->build_count =
            atomic64_read(&cache->build_count);
        out->publish_ns =
            atomic64_read(&cache->publish_ns);
        out->publish_count =
            atomic64_read(&cache->publish_count);
    } else {
        out->stale_sync_writes = 0;
        out->stale_deferred = 0;
        out->clone_ns = 0;
        out->clone_count = 0;
        out->persist_ns = 0;
        out->persist_count = 0;
        out->build_ns = 0;
        out->build_count = 0;
        out->publish_ns = 0;
        out->publish_count = 0;
    }
}

void vecfs_shield_stats_reset(struct super_block *sb)
{
    struct vecfs_shield_runtime *rt = vecfs_shield_runtime(sb);
    struct vecfs_sb_info *sbi;
    struct vecfs_ann_cache *cache;

    if (!rt)
        return;

    atomic64_set(&rt->counters.writes, 0);
    atomic64_set(&rt->counters.tasks_begun, 0);
    atomic64_set(&rt->counters.tasks_ended, 0);
    atomic64_set(&rt->counters.tasks_expired, 0);
    atomic64_set(&rt->counters.publishes_rejected, 0);
    atomic64_set(&rt->counters.queries, 0);
    atomic64_set(&rt->counters.results_returned, 0);
    atomic64_set(&rt->counters.chunks_embedded, 0);
    atomic64_set(&rt->counters.chunks_reused, 0);
    atomic64_set(&rt->counters.chunks_redundant, 0);
    atomic64_set(&rt->counters.files_skipped, 0);
    atomic64_set(&rt->counters.upcall_wait_ns, 0);
    atomic64_set(&rt->counters.upcall_count, 0);

    /* R5f/R5g 消融观测计数器（ann cache）一并归零。 */
    sbi = VECFS_SB(sb);
    cache = sbi ? sbi->s_ann_cache : NULL;
    if (cache) {
        atomic64_set(&cache->stale_sync_writes, 0);
        atomic64_set(&cache->stale_deferred, 0);
        atomic64_set(&cache->clone_ns, 0);
        atomic64_set(&cache->clone_count, 0);
        atomic64_set(&cache->persist_ns, 0);
        atomic64_set(&cache->persist_count, 0);
        atomic64_set(&cache->build_ns, 0);
        atomic64_set(&cache->build_count, 0);
        atomic64_set(&cache->publish_ns, 0);
        atomic64_set(&cache->publish_count, 0);
    }
}
