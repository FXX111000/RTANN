// SPDX-License-Identifier: GPL-2.0
#ifndef _VECFS_SHIELD_H
#define _VECFS_SHIELD_H

#include <linux/atomic.h>
#include <linux/list.h>
#include <linux/mutex.h>
#include <linux/spinlock.h>
#include <linux/types.h>
#include <linux/workqueue.h>
#include <linux/wait.h>

struct inode;
struct super_block;
struct vecfs_inode;

/*
 * 派生有效性模式。
 *
 * P0 实现三种基线模式（Event/Conditional/Validated），
 * VECFS_SHIELD_MODE_SHIELD 接入状态机、去抖调度、内核侧假嵌入与条件发布。
 */
enum vecfs_shield_mode {
    VECFS_SHIELD_MODE_NONE        = 0,
    VECFS_SHIELD_MODE_EVENT       = 1,
    VECFS_SHIELD_MODE_CONDITIONAL = 2,
    VECFS_SHIELD_MODE_VALIDATED   = 3,
    VECFS_SHIELD_MODE_SHIELD      = 4,
};

/*
 * 每文件派生状态机。
 *
 *   CLEAN ──write──▶ PENDING ──窗口到──▶ DERIVING
 *      ▲                                     │ 目标过期/发布失败
 *      └──────────── 条件发布成功 ────────────┘
 */
enum vecfs_shield_fs_state {
    VECFS_SHIELD_FS_CLEAN    = 0,
    VECFS_SHIELD_FS_PENDING  = 1,
    VECFS_SHIELD_FS_DERIVING = 2,
};

/*
 * 文件处置结果：EMBED = 可嵌入（文本等）；SKIP = 不可嵌入（二进制等）。
 */
enum vecfs_shield_disposition {
    VECFS_SHIELD_DISP_EMBED = 0,
    VECFS_SHIELD_DISP_SKIP  = 1,
};

/* 策略开关（消融用，各自可独立关闭）。 */
#define VECFS_SHIELD_STRAT_F_REUSE    (1U << 0) /* 块级复用表 */
#define VECFS_SHIELD_STRAT_F_CLASSIFY (1U << 1) /* 文本/二进制判别 */
#define VECFS_SHIELD_STRAT_F_VALIDATE (1U << 2) /* 查询端 Valid()：valid_bitmap 校验 */
#define VECFS_SHIELD_STRAT_F_CANCEL   (1U << 3) /* P1 主动取消在途 upcall 嵌入 */
#define VECFS_SHIELD_STRAT_F_COW      (1U << 4) /* 快照写时复制（off=全拷贝克隆） */
#define VECFS_SHIELD_STRAT_F_ASYNC_STALE (1U << 5) /* 异步 stale 落盘（off=写路径同步落盘） */
#define VECFS_SHIELD_STRAT_F_PARTIAL  (1U << 6) /* 完整 L1 部分发布（off=整批取消退回 scoped L1） */
#define VECFS_SHIELD_STRAT_F_ADAPTIVE_WINDOW (1U << 7) /* 自适应去抖：写前空闲则立即派生（off=固定窗口 ω） */

/* 派生策略 id（set_strategy 选用；消融可各自独立切换）。 */
#define VECFS_SHIELD_STRATEGY_BUILTIN 0 /* 内置：文本判别 + 定长分块 + 哈希假嵌入 */
#define VECFS_SHIELD_STRATEGY_UPCALL  1 /* 用户态嵌入 upcall：embed 交 /dev/vecfs-embed */

/*
 * P2 策略接口：文件系统掌握控制权，策略是被调用方。默认内置
 * 「文本判别 + 定长分块 + 哈希嵌入」；日后可注册用户态策略
 * （classify/chunk/embed 由用户态提供，内核决定何时调用）。
 *
 * 统一返回约定（任务4标准，三回调一致）：
 *   0     成功，载荷写入内核提供的输出缓冲（*count / out[]）。
 *   > 0   非错误处置（见各回调；如 SKIP / 本块拒绝嵌入）。
 *   < 0   -errno 失败。其中 -EAGAIN 表示「暂时失败，保留 PENDING 重试」，
 *         其余负值按永久失败处理（内核保守处置，不反复重试）。
 *
 * 关键约束：内核提供所有输出缓冲并决定何时调用、如何处置返回值；
 * 用户态策略只「回答」，不分配内存、不持有跨调用状态、不做发布决策。
 * 维度契约：embed 必须产出恰好 dim 个 fp32 位模式（索引维度在 mkfs
 * 时固定）；模型维度不符属于配置错误，以 -errno 报告而非隐式截断。
 */
struct vecfs_shield_chunk {
    u64 offset;
    u64 length;
};

/*
 * 嵌入回调的非错误处置（正返回值）。
 */
enum vecfs_shield_embed_result {
    VECFS_SHIELD_EMBED_OK   = 0, /* 成功：out[0..dim) 已填 fp32 位模式 */
    VECFS_SHIELD_EMBED_SKIP = 1, /* 本块拒绝嵌入：内核按 SKIP 处理（不发布向量） */
};

struct vecfs_shield_strategy_ops {
    const char *name;

    /*
     * 处置判别（文件级预处理）：
     *   返回 VECFS_SHIELD_DISP_EMBED(0) 或 VECFS_SHIELD_DISP_SKIP(1)；
     *   负 -errno = 判别失败（-EAGAIN 重试，其余按 SKIP 保守处理）。
     */
    int (*classify)(const void *data, size_t len);

    /*
     * 分块（文件级预处理）：
     *   填充 out[0..*count)，每项为升序、不重叠、覆盖 [0,size) 的区间；
     *   返回 0 成功（*count = 块数，0..max_chunks；0 = 空内容）；
     *   负 -errno = 分块失败（-EAGAIN 重试，其余按 SKIP 保守处理）。
     *   data 为文件内容缓冲（长度 size），分块器可在字节边界上做
     *   码点对齐（避免切开 UTF-8 多字节码点）。
     */
    int (*chunk)(const void *data, size_t size, u32 chunk_size,
                 struct vecfs_shield_chunk *out, u32 max_chunks,
                 u32 *count);

    /*
     * 嵌入（块级，嵌入模型调用返回标准）：
     *   返回 VECFS_SHIELD_EMBED_OK(0)：out[0..dim) 已填 fp32 位模式；
     *   VECFS_SHIELD_EMBED_SKIP(1)：本块拒绝嵌入；
     *   负 -errno：失败（-EAGAIN 保留 PENDING 重试，其余按永久失败跳过本块）。
     */
    int (*embed)(const void *data, size_t len, u32 dim, u32 *out);

    /*
     * 批量嵌入（文件级，一次嵌入 nchunks 个块，对齐 baseline embed_many 粒度）：
     *   out 为 nchunks*dim 个 fp32 位模式的连续缓冲（chunks[i] → out[i*dim..]）。
     *   返回 0 = 全部成功；-ECANCELED = 主动取消（走 requeue 重试）；
     *   -EAGAIN = 整体保留 PENDING 重试；其余负 -errno = 整体按永久失败跳过
     *   （对应旧逐块路径的「永久失败 skip 块」）。
     *   embed_batch 非 NULL 时 derive 走批量路径；NULL 时回退逐块 embed。
     *
     *   inode/generation/req_id_out 供主动取消（P1）使用：inode 为派生目标
     *   （只读上下文）；generation = 目标版本 target_version（请求代际标识）；
     *   req_id_out 指向内核提供的在途请求句柄槽（可为 NULL，内置策略忽略），
     *   upcall 策略在阻塞期间写入/清零，供写路径 READ_ONCE 发现在途请求。
     *   策略仍是被调用方：内核决定何时嵌入、如何处置，上述仅为上下文/输出槽。
     */
    int (*embed_batch)(struct inode *inode, u64 generation, const void *base,
                       const struct vecfs_shield_chunk *chunks,
                       u32 nchunks, u32 dim, u32 *out, u64 *req_id_out);
};

/*
 * 块级复用表：记录每块上次派生时的内容哈希与发布向量 ordinal。
 * 块内容未变（哈希相同）时清 stale 复用旧向量，避免重嵌入 + 重入 HNSW。
 */
struct vecfs_shield_reuse_entry {
    u64 hash;
    u32 ordinal;
};

struct vecfs_shield_reuse_table {
    u32 chunk_size;
    u32 count;
    struct vecfs_shield_reuse_entry entries[];
};

/*
 * 完整 L1 在途复用：在途块的输入键注册表。每个在途批量嵌入块以内容寻址键
 * K(c)=H(θ‖X(c))（块偏移 θ 与内容哈希 X(c) 的组合）登记，写路径据此做
 * 需求驱动取消——只按块重叠判定哪些在途块被新写覆盖，仅当全部在途块被覆盖
 * 才整批取消，部分覆盖时保留在途请求、由派生在嵌入返回后按块过滤部分发布。
 * 注册表是「取消提示」而非正确性权威（权威由 L2 的 chunk_written_version 保证）。
 */
#define VECFS_SHIELD_INFLIGHT_MAX 256

struct vecfs_shield_inflight_key {
    u64 key;    /* K(c) = H(θ ‖ X(c))：块的内容寻址身份 */
    u64 offset; /* 块偏移 θ（重叠判定用） */
    u64 length; /* 块长（重叠判定用） */
};

/*
 * 嵌入 struct vecfs_inode，寿命与 inode 一致。
 */
struct vecfs_shield_file_state {
    struct mutex lock;

    u32 state;             /* enum vecfs_shield_fs_state */
    u64 target_version;    /* 最新待派生 v_f */
    u64 semantic_version;  /* 已发布向量的 v_f */
    u64 pending_since;     /* jiffies，最近一次进入 PENDING */
    u64 first_change;      /* 待派生内容区间 [first_change, last_change) */
    u64 last_change;

    /*
     * 全局待派生队列节点（挂到 rt->pending_files）。文件处于 PENDING 且在
     * 队列中即持有一份 igrab 引用（derive_refs≥1）；仅在 pending_lock 保护下
     * 增删。
     */
    struct list_head pending_node;

    /*
     * 写路径 igrab 持有的 inode 引用计数（首写时 +1，派生 worker 终态 -1）。
     * 用计数而非 bool：首写 igrab 与上一轮 worker 的收尾 iput 存在合法竞态
     * （旧执行置 CLEAN 后、iput 前，新写可再次 igrab），瞬时可达 2。归零后
     * 再减即判为「重复 iput」，此时 WARN 并跳过 iput，避免触发 VFS 的
     * BUG_ON 把整机崩掉。始终在 fs->lock 内增减。
     */
    int derive_refs;

    /*
     * 置位后禁止再为本 inode 排队派生。在 file_destroy 持 fs->lock 置位。
     * 写路径与 drain worker 的 requeue 都在同一把 fs->lock 内检查该标志，
     * 从而把「置死之后仍重新入队」的竞态关死，杜绝 work 在 inode 释放后
     * 被调度执行（use-after-free）。
     */
    bool dead;

    /*
     * 在途 upcall 请求 id（P1 主动取消的在途句柄）。仅 WRITE_ONCE/READ_ONCE
     * 访问，不加锁：它是「取消提示」而非正确性权威——写路径读到非零即尝试
     * 取消，读不到或读到已结束的 id 都退化为既有「版本复核」兜底。0 = 无
     * 在途请求。
     */
    u64 in_flight_upcall;

    /*
     * 块独立可见性（L2）：每块最近一次写版本。写路径在 version_bump 时于
     * fs->lock 内按写区间更新（chunk_version_count = 已分配条目数，缺失条目
     * 视为 0 = 从未被写 → 恒 <= derive_version → 视为有效）。派生提交阶段按
     * 块复核「该块在派生版本之后是否被重写」，替代整文件版本门。
     * chunk_version_chunk_size 记录建表时的 chunk_size，变化时整表重置。
     */
    u64 *chunk_written_version;
    u32 chunk_version_count;
    u32 chunk_version_chunk_size;

    /*
     * 在途批量嵌入的内容区间 [in_flight_first, in_flight_last)。仅
     * WRITE_ONCE/READ_ONCE 访问，作取消判定提示（L1 在途复用）：写路径仅当
     * 本次写区间与之重叠才取消在途请求，从而让无关块变更不再打断在途的稳定
     * 块嵌入。0 表示无在途批量区间。
     */
    u64 in_flight_first;
    u64 in_flight_last;

    /*
     * 完整 L1 在途复用：在途块输入键注册表（详见 vecfs_shield_inflight_key）。
     * in_flight_nkeys 由派生 worker 在 embed_batch 前后 WRITE_ONCE 发布/清零；
     * 写路径 READ_ONCE 后在 [0, nkeys) 内按块重叠做需求驱动取消。全部仅作
     * 提示（不加锁），正确性由 L2 chunk_written_version（fs->lock 内）保证。
     * in_flight_overflow=1 表示本批 n_embed 超过 VECFS_SHIELD_INFLIGHT_MAX，
     * 无法逐块判定，退回整批取消的保守行为。
     */
    struct vecfs_shield_inflight_key in_flight_keys[VECFS_SHIELD_INFLIGHT_MAX];
    u32 in_flight_nkeys;
    u32 in_flight_overflow;

    struct vecfs_shield_reuse_table *reuse; /* P2 块级复用表 */
};

/* 面向用户态的统计快照。 */
struct vecfs_shield_stats {
    u64 writes;              /* 内容修改次数 */
    u64 tasks_begun;         /* 派生任务开始数 */
    u64 tasks_ended;         /* 派生任务完成数 */
    u64 tasks_expired;       /* 完成时目标版本已过期 */
    u64 publishes_rejected;  /* 条件发布拒绝数 */
    u64 queries;             /* KNN 查询数 */
    u64 results_returned;    /* 返回结果总数 */

    u64 chunks_embedded;     /* 实际重嵌入并发布的块数 */
    u64 chunks_reused;       /* 复用（未变）的块数 */
    u64 chunks_redundant;    /* 内容未变但被重嵌的块数（复用关闭消融） */
    u64 files_skipped;       /* 因二进制 SKIP 的文件数 */

    u64 stale_sync_writes;   /* R5g：写路径同步落盘 CONTENT_STALE 的次数 */
    u64 stale_deferred;      /* R5g：延迟到 dirty-set（卸载 drain）的节点数 */
    u64 clone_ns;            /* R5f：snapshot clone 累计耗时（ns） */
    u64 clone_count;         /* R5f：snapshot clone 次数 */

    u64 upcall_wait_ns;      /* upcall 嵌入阻塞等待累计耗时（ns，含 Ollama 计算） */
    u64 upcall_count;        /* upcall 嵌入批次数（= 派生次数） */
    u64 persist_ns;          /* 派生批量持久化累计耗时（ns：node links + meta commit） */
    u64 persist_count;       /* 持久化批次数 */

    u64 build_ns;            /* HNSW 图构建累计耗时（ns：逐节点插入 + ef_construction） */
    u64 build_count;         /* 图构建批次数 */
    u64 publish_ns;          /* 发布累计耗时（ns：bind + labels + publish_objects） */
    u64 publish_count;       /* 发布批次数 */
};

/* 内部原子计数器。 */
struct vecfs_shield_counters {
    atomic64_t writes;
    atomic64_t tasks_begun;
    atomic64_t tasks_ended;
    atomic64_t tasks_expired;
    atomic64_t publishes_rejected;
    atomic64_t queries;
    atomic64_t results_returned;

    atomic64_t chunks_embedded;
    atomic64_t chunks_reused;
    atomic64_t chunks_redundant;
    atomic64_t files_skipped;

    atomic64_t upcall_wait_ns;   /* upcall 嵌入阻塞等待累计（ns，含 Ollama） */
    atomic64_t upcall_count;
};

struct vecfs_shield_runtime {
    int mode;

    /* SHIELD 去抖窗口（微秒）。固定窗口；自适应窗口留作消融开关。 */
    u64 window_us;
    u64 window_jiffies;

    /* P2 策略配置。 */
    u32 chunk_size;
    u32 strat_flags;
    const struct vecfs_shield_strategy_ops *strategy;

    /*
     * 全局待派生队列 + 批量 drain worker（替代旧的每文件 delayed_work）。
     * pending_lock 保护 pending_files；drain_work 到点后把队列里的一批文件
     * 聚合成一次多文件批量插入（每窗口一次 O(N) clone，而非每文件一次）。
     */
    spinlock_t pending_lock;
    struct list_head pending_files;
    struct delayed_work drain_work;

    /*
     * 事件通知 waitqueue：文件派生状态转换（PENDING/DERIVING → CLEAN 等终态）
     * 时唤醒 poll() 等待者，用户态据此阻塞等待而非 100ms 轮询 file_state。
     */
    wait_queue_head_t state_wq;

    /*
     * 卸载关闭标志：put_super 先置位、再唤醒 upcall worker、最后等待
     * derive_refs_total 归零。置位后写路径不再排队、worker 的 requeue 不再
     * 重新武装，从而保证在释放 sbi/ANN 运行时之前所有派生 worker 已排空。
     */
    bool shutting_down;

    /*
     * 所有文件 derive_refs 之和（= 尚未终结的派生 worker/在途 work 数）。
     * 写路径 igrab 成功时 +1，worker 终态 put_ref 时 -1；归零即无任何
     * 运行中/排队中/计时器待发的派生 work。
     */
    atomic64_t derive_refs_total;

    struct vecfs_shield_counters counters;
};

/* 每文件派生状态快照（ioctl 回填）。 */
struct vecfs_shield_file_state_info {
    u32 state;
    u32 reserved;
    u64 shield_version;
    u64 target_version;
    u64 semantic_version;
    u64 first_change;
    u64 last_change;
};

int vecfs_shield_mode(struct super_block *sb);
int vecfs_shield_set_mode(struct super_block *sb, int mode, u64 window_us);
struct vecfs_shield_runtime *vecfs_shield_runtime(struct super_block *sb);

/* 初始化 SHIELD 运行时默认策略配置（chunk_size / flags / 内置策略）。 */
void vecfs_shield_runtime_init(struct vecfs_shield_runtime *rt);

/*
 * Phase D：扫描 stale 的 file-bound LIVE 向量并按唯一 inode 调度重派生
 * （非干净卸载恢复后 / 从 VALIDATED 切 SHIELD 时恢复召回）。
 */
void vecfs_shield_schedule_stale_rederive(struct super_block *sb);

/*
 * 卸载时关闭 SHIELD 派生：置 shutting_down、唤醒阻塞在 upcall 的 worker，
 * 并等待所有派生 worker 释放 igrab 引用。必须在释放 sbi/ANN 运行时之前调用。
 */
void vecfs_shield_shutdown(struct super_block *sb);

/*
 * 设置派生策略：strategy_id（当前仅 0 = 内置）、开关 flags、
 * 定长分块大小 chunk_size。
 */
int vecfs_shield_set_strategy(struct super_block *sb, u32 strategy_id,
                              u32 flags, u64 chunk_size);

bool vecfs_shield_write_marks_stale(struct super_block *sb);

u64 vecfs_shield_version(struct inode *inode);
void vecfs_shield_version_bump(struct inode *inode);

/*
 * 写路径前段（SHIELD/VALIDATED 模式）：先推进版本、再标记旧向量 stale，
 * 二者先于数据写入。用于杜绝派生 worker 在「标记 stale」与「版本推进」
 * 之间提交导致的无效结果泄露。
 */
int vecfs_shield_write_begin(struct inode *inode, u64 start, u64 len);

void vecfs_shield_file_init(struct vecfs_inode *vi);
void vecfs_shield_file_destroy(struct inode *inode);

void vecfs_shield_write_hook(struct inode *inode, u64 start, u64 len);

int vecfs_shield_file_state_get(struct inode *inode,
                                struct vecfs_shield_file_state_info *out);

int vecfs_shield_task_begin(struct inode *inode, u64 *target_version);
int vecfs_shield_task_end(struct inode *inode, u64 target_version,
                          bool *expired);

void vecfs_shield_count_query(struct super_block *sb, u32 result_count);

void vecfs_shield_stats_snapshot(struct super_block *sb,
                                 struct vecfs_shield_stats *out);
void vecfs_shield_stats_reset(struct super_block *sb);

#endif /* _VECFS_SHIELD_H */
