#ifndef _VECFS_ANN_CACHE_H
#define _VECFS_ANN_CACHE_H

#include <linux/kref.h>
#include <linux/mutex.h>
#include <linux/rcupdate.h>
#include <linux/refcount.h>
#include <linux/types.h>
#include <linux/workqueue.h>

#include "vecfs_disk.h"

struct super_block;
struct vecfs_ann_insert_request;

/*
 * 冷元数据与查询热数据分离。
 *
 * bare-bone 查询的 inner loop 不访问该结构；只有 deleted/filter
 * 路径以及最终结果组装才访问。
 */
struct vecfs_ann_cache_node {
    u64 external_id;

    u64 owner_ino;
    u64 content_offset;
    u64 content_length;

    u32 owner_generation;
    u32 vector_ordinal;

    u16 owner_type;
    u16 flags;

    u8 state;
    u8 level;
    u16 reserved;
};

/*
 * nodes[] 分页（C3）：扁平页表。每页容纳 2^k 个 node，页内索引用掩码定位；
 * capacity 有界 → 单层页表 O(1)，不做 radix。refcount 放页尾（而非页头），
 * 使 node 数组从页起始处自然对齐（struct 含 u64 字段，需 8 字节对齐），
 * clone 共享页时两个快照共享该 refcount，写侧经 node_mutable() COW。
 */
#define VECFS_ANN_NODE_PAGE_SHIFT 10U
#define VECFS_ANN_NODE_PAGE_SIZE  (1U << VECFS_ANN_NODE_PAGE_SHIFT)
#define VECFS_ANN_NODE_PAGE_MASK  (VECFS_ANN_NODE_PAGE_SIZE - 1)

struct vecfs_ann_node_page {
    struct vecfs_ann_cache_node nodes[VECFS_ANN_NODE_PAGE_SIZE];
    refcount_t refcount;
};

/*
 * upper level 与 hnswlib 类似，按节点实际 level 分配：
 *
 *   [u16 count][padding][M * u32 links]  level 1
 *   [u16 count][padding][M * u32 links]  level 2
 *   ...
 *
 * 每层 stride 固定，因此定位只需要一次乘法。
 */
struct vecfs_ann_upper_block {
    /* 指向块头 [refcount_t][层数据...]。clone 共享时两个快照的 upper 块
     * 指向同一块头、共享头部的 refcount；COW 时复制整块并递减旧块。 */
    void *memory;
};

static inline refcount_t *
vecfs_ann_upper_block_refs(const struct vecfs_ann_upper_block *block)
{
    return (refcount_t *)block->memory;
}

static inline void *
vecfs_ann_upper_block_data(const struct vecfs_ann_upper_block *block)
{
    return (u8 *)block->memory + sizeof(refcount_t);
}

/*
 * level0 内存分块：把整块 level0 对象内存切成固定字节目标大小的块，
 * 每块容纳 2^k 个节点（k 由 element_stride 决定）。超大图（GB 级）不再
 * 产生单次超大 vmalloc（此前 5GB 导入会因单块连续分配失败）；查询热路径
 * 用「右移定位块 + 掩码定位块内偏移」，仅多一次指针解引用，性能影响可忽略。
 */
#define VECFS_ANN_LEVEL0_CHUNK_TARGET_BYTES (4UL * 1024 * 1024)

struct vecfs_ann_level0_chunk {
    /* 指向块头 [refcount_t][元素数据...]。clone 共享时两个快照的 chunk
     * 指向同一块头、共享头部的 refcount；COW 时复制整块并递减旧块。 */
    void *memory;
};

static inline refcount_t *
vecfs_ann_level0_chunk_refs(const struct vecfs_ann_level0_chunk *chunk)
{
    return (refcount_t *)chunk->memory;
}

static inline void *
vecfs_ann_level0_chunk_data(const struct vecfs_ann_level0_chunk *chunk)
{
    return (u8 *)chunk->memory + sizeof(refcount_t);
}

static inline u32 vecfs_ann_level0_chunk_shift(u32 element_stride)
{
    u32 nodes;
    u32 shift = 0;

    if (!element_stride)
        return 0;

    nodes = (u32)(VECFS_ANN_LEVEL0_CHUNK_TARGET_BYTES / element_stride);
    if (nodes < 1)
        nodes = 1;

    /* floor(log2(nodes))：每块节点数取不超过目标字节数的最大 2 的幂。 */
    while (nodes >>= 1)
        shift++;

    return shift;
}

/*
 * 不可变查询快照。
 *
 * 发布后，任何字段都不能原地修改。writer 构建完整新快照后通过
 * rcu_assign_pointer() 发布，旧快照经 kref_put → call_rcu 延迟到
 * 宽限期结束后回收（无需阻塞式 synchronize_rcu()）。
 */
struct vecfs_ann_snapshot {
    u32 capacity;
    u32 live_count;

    u32 entry_node;
    u16 max_level;

    u16 m;
    u16 m0;
    u16 max_levels;
    u16 reserved16;

    u32 dimension;
    u32 vector_bytes;

    u32 level0_links_offset;
    u32 vector_offset;
    u32 element_stride;

    /* level0 对象内存分块表；分块几何由 level0_chunk_shift/mask 决定。 */
    struct vecfs_ann_level0_chunk *level0_chunks;
    u32 level0_chunk_shift;   /* log2(每块节点数)，2 的幂 */
    u32 level0_chunk_mask;    /* 每块节点数 - 1 */
    u32 level0_chunk_count;   /* 块数 */

    struct vecfs_ann_upper_block *upper;
    struct vecfs_ann_node_page **node_pages;
    u32 node_page_count;

    unsigned long *stale_bitmap;

    /*
     * SHIELD 查询端 Valid(x,f) = Current ∧ Referenced 的落地位图，
     * 与 stale_bitmap 并列。位为 1 表示该节点被某个「仍为当前的已提交
     * 语义根」引用（有效）；位为 0 表示无效（孤儿或根已失效）。
     *
     * 默认全 1（全部有效），因此非 SHIELD 模式（从不翻转该位）不受影响。
     * SHIELD 发布时置新根引用为 valid、置孤儿为 invalid；该位图只存在于
     * 内存中（不落盘），重挂载后随 in-memory 派生状态一并重建。
     */
    unsigned long *valid_bitmap;

    /*
     * stale.c 在 writer_lock 下修改 stale 状态，因此查询可以在
     * RCU snapshot 有效期间通过 atomic_read() 判断是否需要进入
     * filtered search。
     */
    atomic_t stale_count;

    /* valid_bitmap 中被置 0（无效）的节点计数，用于查询热路径快速判断。 */
    atomic_t invalid_count;

    bool has_deletions;

    /*
     * 查询快照引用计数。
     *
     * RTANN 调度器在 workqueue worker（可睡眠、不在 RCU 读侧临界区）中直接
     * 使用快照，因此不能依赖 rcu_read_lock 保护。提交时在 rcu_read_lock 内
     * kref_get 一次、完成时 kref_put 一次；writer 发布/卸载路径在
     * rcu_assign_pointer(NULL) 后 kref_put，归零后由 call_rcu 延迟到宽限期
     * 结束再释放，保证旧快照在最后一个引用者归还前不被释放。
     */
    struct kref refcount;

    /*
     * 延迟回收：kref 归零后经 call_rcu 延迟到宽限期结束，再转 workqueue
     * 释放快照内存。读侧（search / hnsw_sched / shield 的 rcu_read_lock
     * 临界区）无需等待同步宽限期；发布/卸载路径在 rcu_assign_pointer(NULL)
     * 后即可 kref_put，无需每文件 synchronize_rcu()。
     *
     * 注意 free() 内部用 kvfree（大块可能落到 vfree，可睡眠），不能直接在
     * RCU 软中断回调里跑，故 call_rcu 回调只负责排到 free_wq。
     */
    struct rcu_head rcu;
    struct work_struct free_work;
};

/*
 * nodes[] 分页读访问器：右移定位页、掩码定位页内偏移。返回 const 指针——
 * 任何经此修改 node 字段的写点都会编译报错，强制写侧改用 node_mutable()。
 * 定义置于 struct 之后、所有访问器之前，供 effective_flags 等早期内联使用。
 */
static __always_inline const struct vecfs_ann_cache_node *
vecfs_ann_node_at(
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    return &snapshot->node_pages[
               node_id >> VECFS_ANN_NODE_PAGE_SHIFT]
                ->nodes[node_id & VECFS_ANN_NODE_PAGE_MASK];
}

static __always_inline bool
vecfs_ann_snapshot_has_stale(
    const struct vecfs_ann_snapshot *snapshot)
{
    return snapshot &&
           atomic_read(
               &snapshot->stale_count) > 0;
}

static __always_inline bool
vecfs_ann_snapshot_has_invalid(
    const struct vecfs_ann_snapshot *snapshot)
{
    return snapshot &&
           atomic_read(
               &snapshot->invalid_count) > 0;
}


struct vecfs_ann_cache {
    struct super_block *sb;

    /*
     * 只允许 writer 获取。查询不获取该锁。
     */
    struct mutex writer_lock;

    struct vecfs_ann_snapshot __rcu *snapshot;

    /*
     * Phase D：已置内存 stale 位、但 CONTENT_STALE 尚未落盘的 node_id 脏集合。
     * 位 1 = 待落盘；drain 后清 0。与 snapshot 容量对齐，reload 扩容时重建。
     * 仅在 writer_lock 下访问。
     */
    unsigned long *stale_dirty;
    u32 stale_dirty_bits;

    /*
     * 异步 stale 落盘开关（消融）。true = 写路径只置内存位 + 记入脏集合，
     * CONTENT_STALE 磁盘标志延迟到卸载 drain；false = 写路径每节点同步落盘。
     * 由 SHIELD set_strategy 按 F_ASYNC_STALE 透传；默认 true（cache_mount 初始化）。
     */
    bool async_stale;

    /*
     * 消融观测计数器（由 SHIELD GET_STATS 读出，reset_stats 归零）。
     *   stale_sync_writes：写路径同步落盘 CONTENT_STALE 的次数（async OFF 时每
     *     失效节点一次，串行于 store->io_lock）；stale_deferred：延迟到 dirty-set
     *     的节点数（async ON）。二者之和 = 本次负载失效节点总数。
     *   clone_ns / clone_count：派生批量插入里 snapshot clone 的累计耗时（ns）与
     *     次数。COW ON=refcount（O(capacity)）、OFF=全拷贝（O(capacity×node)），
     *     二者成本差异集中暴露在 clone_ns 上，与图构建成本解耦。
     */
    atomic64_t stale_sync_writes;
    atomic64_t stale_deferred;
    atomic64_t clone_ns;
    atomic64_t clone_count;

    /* 派生批量持久化（node links + meta commit）累计耗时（ns）与批次数。 */
    atomic64_t persist_ns;
    atomic64_t persist_count;

    /*
     * 派生批量插入里「图构建」（HNSW 逐节点插入，ef_construction 贪心搜索）与
     * 「发布」（bind_mappings + insert_labels + publish_objects）的累计耗时（ns）
     * 与批次数。与 clone_ns / persist_ns 一起把「嵌入之后」的派生成本完整拆开，
     * 用于定位 arrival 负载下内核派生比 hnswlib 插入慢的那段。
     */
    atomic64_t build_ns;
    atomic64_t build_count;
    atomic64_t publish_ns;
    atomic64_t publish_count;
};

static __always_inline bool
vecfs_ann_snapshot_content_stale(
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    if (unlikely(
            !snapshot ||
            !snapshot->stale_bitmap ||
            node_id >= snapshot->capacity))
        return false;

    return test_bit(
        node_id,
        snapshot->stale_bitmap);
}

static __always_inline void
vecfs_ann_snapshot_set_content_stale(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    if (likely(
            snapshot &&
            snapshot->stale_bitmap &&
            node_id < snapshot->capacity))
        set_bit(
            node_id,
            snapshot->stale_bitmap);
}

static __always_inline void
vecfs_ann_snapshot_clear_content_stale(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    if (likely(
            snapshot &&
            snapshot->stale_bitmap &&
            node_id < snapshot->capacity))
        clear_bit(
            node_id,
            snapshot->stale_bitmap);
}

/*
 * 查询端 Valid(x,f) 的 valid 位读取：valid_bitmap 缺省时视作全 1
 * （全部有效），因此旧快照或非 SHIELD 环境不受影响。
 */
static __always_inline bool
vecfs_ann_snapshot_content_valid(
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    if (unlikely(
            !snapshot ||
            !snapshot->valid_bitmap ||
            node_id >= snapshot->capacity))
        return true;

    return test_bit(
        node_id,
        snapshot->valid_bitmap);
}

static __always_inline bool
vecfs_ann_snapshot_content_invalid(
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    return !vecfs_ann_snapshot_content_valid(
        snapshot,
        node_id);
}

static __always_inline u16
vecfs_ann_snapshot_effective_flags(
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    u16 flags;

    if (unlikely(
            !snapshot ||
            node_id >= snapshot->capacity))
        return 0;

    flags = READ_ONCE(
        vecfs_ann_node_at(snapshot, node_id)->flags);

    if (vecfs_ann_snapshot_content_stale(
            snapshot,
            node_id))
        flags |=
            VECFS_ANN_OBJ_F_CONTENT_STALE;
    else
        flags &=
            ~VECFS_ANN_OBJ_F_CONTENT_STALE;

    if (vecfs_ann_snapshot_content_invalid(
            snapshot,
            node_id))
        flags |=
            VECFS_ANN_OBJ_F_CONTENT_INVALID;
    else
        flags &=
            ~VECFS_ANN_OBJ_F_CONTENT_INVALID;

    return flags;
}


int vecfs_ann_cache_mount(struct super_block *sb);
void vecfs_ann_cache_unmount(struct super_block *sb);
int vecfs_ann_cache_reload(struct super_block *sb);

int vecfs_ann_snapshot_update_binding(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id,
    u64 content_offset,
    u64 content_length,
    u16 flags);

struct vecfs_ann_snapshot *
vecfs_ann_snapshot_build(struct super_block *sb);

void vecfs_ann_snapshot_free(
    struct vecfs_ann_snapshot *snapshot);

static inline void vecfs_ann_snapshot_get(
    struct vecfs_ann_snapshot *snapshot)
{
    if (snapshot)
        kref_get(&snapshot->refcount);
}

void vecfs_ann_snapshot_put(
    struct vecfs_ann_snapshot *snapshot);

/*
 * 延迟回收 workqueue 的生命周期（模块 init/exit）。vecfs_ann_cache_init 在
 * vecfs_init 中调用；vecfs_ann_cache_exit 在 vecfs_exit 的 rcu_barrier() 之后
 * 调用，确保所有 call_rcu → queue_work 的 free 都跑完。
 */
int vecfs_ann_cache_init(void);
void vecfs_ann_cache_exit(void);

/*
 * level0 分块内存的分配/释放。分块几何（shift/mask/count）由本函数根据
 * element_stride 计算并写回 snapshot，因此调用方只需保证 element_stride
 * 已就绪（init_layout / 结构体拷贝 / 显式拷贝）。定义见 ann/snapshot.c。
 */
int vecfs_ann_level0_alloc(
    struct vecfs_ann_snapshot *snapshot,
    u32 capacity);

void vecfs_ann_level0_free(
    struct vecfs_ann_snapshot *snapshot);

int vecfs_ann_cache_publish(
    struct super_block *sb,
    struct vecfs_ann_snapshot *snapshot);

static __always_inline bool
vecfs_ann_snapshot_id_valid(
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    return node_id >= VECFS_ANN_FIRST_NODE_ID &&
           node_id < snapshot->capacity;
}

static __always_inline void *
vecfs_ann_level0_element(
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    /* 分块定位：右移得块索引，掩码得块内节点号。块内偏移仍是一次乘法，
     * 热路径只多一次指针解引用，保持分支无关（__always_inline）。 */
    return (u8 *)vecfs_ann_level0_chunk_data(
               &snapshot->level0_chunks[
                   node_id >> snapshot->level0_chunk_shift]) +
           (size_t)(node_id & snapshot->level0_chunk_mask) *
               snapshot->element_stride;
}

static __always_inline const u32 *
vecfs_ann_level0_links(
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    return (const u32 *)(
        (const u8 *)vecfs_ann_level0_element(
            snapshot, node_id) +
        snapshot->level0_links_offset);
}

static __always_inline u16
vecfs_ann_level0_count(
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    return READ_ONCE(
        *(const u16 *)vecfs_ann_level0_element(
            snapshot, node_id));
}

static __always_inline const float *
vecfs_ann_snapshot_vector(
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    return (const float *)(
        (const u8 *)vecfs_ann_level0_element(
            snapshot, node_id) +
        snapshot->vector_offset);
}

static __always_inline const void *
vecfs_ann_upper_level(
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id,
    u16 level)
{
    const struct vecfs_ann_upper_block *upper;
    size_t stride;

    upper = &snapshot->upper[node_id];

    if (unlikely(!upper->memory || !level))
        return NULL;

    stride = ALIGN(
        sizeof(u16) + (size_t)snapshot->m * sizeof(u32),
        sizeof(u32));

    return (const u8 *)vecfs_ann_upper_block_data(upper) +
           (size_t)(level - 1) * stride;
}

static __always_inline u16
vecfs_ann_upper_count(
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id,
    u16 level)
{
    const void *block =
        vecfs_ann_upper_level(snapshot, node_id, level);

    return block ? READ_ONCE(*(const u16 *)block) : 0;
}

static __always_inline const u32 *
vecfs_ann_upper_links(
    const struct vecfs_ann_snapshot *snapshot,
    u32 node_id,
    u16 level)
{
    const void *block =
        vecfs_ann_upper_level(snapshot, node_id, level);

    if (unlikely(!block))
        return NULL;

    return (const u32 *)((const u8 *)block + sizeof(u32));
}
/*
 * 以下接口只允许在 cache->writer_lock 持有期间调用。
 */
struct vecfs_ann_snapshot *
vecfs_ann_snapshot_clone_locked(
    const struct vecfs_ann_snapshot *source,
    u32 minimum_capacity);

/*
 * 消融用：copy_full=true 时克隆整份拷贝（level0 元素 / node 页 / 上层块
 * 全部重新分配 + memcpy，不与 source 共享），用于度量 COW 带来的容量解耦。
 */
struct vecfs_ann_snapshot *
vecfs_ann_snapshot_clone_locked_copy(
    const struct vecfs_ann_snapshot *source,
    u32 minimum_capacity,
    bool copy_full);

/*
 * 写侧 COW 访问器（仅 writer_lock 下调用）。返回指向 node_id 的 level0
 * element / upper 层块的可写指针；若所在 chunk/block 被多个快照共享
 * （refcount>1），先写时复制该 chunk/block 再返回。分配失败返回 NULL。
 *
 * 读侧访问器（vecfs_ann_level0_element / vecfs_ann_upper_level）不变，
 * 仍是「右移定位块 + 掩码定位偏移」的免锁读路径。
 */
void *vecfs_ann_level0_element_mutable(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id);

void *vecfs_ann_upper_level_mutable(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id,
    u16 level);

/*
 * nodes[] 分页写侧 COW 访问器（仅 writer_lock 下调用）。返回 node_id 的
 * 可写 node 指针；若所在页被多个快照共享（refcount>1），先写时复制该页。
 * 分配失败返回 NULL。
 */
struct vecfs_ann_cache_node *
vecfs_ann_node_mutable(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id);

/* nodes[] 分页的整表分配 / 释放（cache.c 建快照 / 释放快照时调用）。 */
int vecfs_ann_node_pages_alloc(
    struct vecfs_ann_snapshot *snapshot,
    u32 capacity);

void vecfs_ann_node_pages_free(
    struct vecfs_ann_snapshot *snapshot);

int vecfs_ann_cache_publish_locked(
    struct vecfs_ann_cache *cache,
    struct vecfs_ann_snapshot *snapshot);

/* 延迟 stale 落盘脏集合随快照容量扩容（仅 writer_lock 下调用）。 */
int vecfs_ann_cache_stale_dirty_grow(
    struct vecfs_ann_cache *cache,
    u32 new_bits);

int vecfs_ann_snapshot_init_node(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id,
    u16 level,
    const void *vector,
    const struct vecfs_ann_insert_request *request);

static __always_inline u32 *
vecfs_ann_level0_links_mutable(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id)
{
    void *element =
        vecfs_ann_level0_element_mutable(snapshot, node_id);

    if (!element)
        return NULL;

    return (u32 *)((u8 *)element +
                   snapshot->level0_links_offset);
}

static __always_inline void
vecfs_ann_level0_set_count(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id,
    u16 count)
{
    void *element =
        vecfs_ann_level0_element_mutable(snapshot, node_id);

    if (!element)
        return;

    WRITE_ONCE(*(u16 *)element, count);
}

static __always_inline u32 *
vecfs_ann_upper_links_mutable(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id,
    u16 level)
{
    void *block =
        vecfs_ann_upper_level_mutable(snapshot, node_id, level);

    if (unlikely(!block))
        return NULL;

    return (u32 *)((u8 *)block + sizeof(u32));
}

static __always_inline void
vecfs_ann_upper_set_count(
    struct vecfs_ann_snapshot *snapshot,
    u32 node_id,
    u16 level,
    u16 count)
{
    void *block =
        vecfs_ann_upper_level_mutable(snapshot, node_id, level);

    if (block)
        WRITE_ONCE(*(u16 *)block, count);
}

struct vecfs_ann_snapshot *
vecfs_ann_snapshot_create_empty_like(
    const struct vecfs_ann_snapshot *source,
    u32 capacity);

#endif
