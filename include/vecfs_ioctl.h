#ifndef _UAPI_VECFS_IOCTL_H
#define _UAPI_VECFS_IOCTL_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define VECFS_IOCTL_MAGIC       0xb7
#define VECFS_IOCTL_ABI_MAJOR   2
#define VECFS_IOCTL_ABI_MINOR   0

#define VECFS_IOCTL_MAX_K             4096U
#define VECFS_IOCTL_MAX_PATH_BUFFER   (16U * 1024U * 1024U)

#define VECFS_EXTERNAL_ID_AUTO ((__u64)~0ULL)
#define VECFS_ORDINAL_AUTO     ((__u32)~0U)

/*
 * 通用 vector flags。
 */
#define VECFS_VECTOR_F_HAS_CONTENT_RANGE (1U << 0)
#define VECFS_VECTOR_F_CONTENT_STALE     (1U << 1)

/*
 * runtime flags。
 */
#define VECFS_RUNTIME_F_PERSIST_EF       (1U << 0)

/*
 * 查询结果 flags。
 */
#define VECFS_RESULT_F_HAS_PATH          (1U << 0)
#define VECFS_RESULT_F_HAS_CONTENT_RANGE (1U << 1)
#define VECFS_RESULT_F_CONTENT_STALE     (1U << 2)
#define VECFS_RESULT_F_CONTENT_INVALID   (1U << 3)

/*
 * 查询 flags，后续过滤器继续扩展。
 */
#define VECFS_QUERY_F_FILE_ONLY          (1U << 0)
#define VECFS_QUERY_F_STANDALONE_ONLY    (1U << 1)
#define VECFS_QUERY_F_INCLUDE_STALE      (1U << 2)

#define VECFS_VECTOR_GET_F_INCLUDE_DELETED  (1U << 0)

#define VECFS_FILE_LIST_F_INCLUDE_DELETED   (1U << 0)

#define VECFS_VALIDATE_F_REPAIR             (1U << 0)
#define VECFS_VALIDATE_F_STRICT             (1U << 1)
#define VECFS_VALIDATE_F_CHECK_LINKS        (1U << 2)

#define VECFS_REBUILD_F_SYNC              (1U << 0)
#define VECFS_REBUILD_F_PRESERVE_LEVELS   (1U << 1)

#define VECFS_EXCHANGE_MAJOR 1
#define VECFS_EXCHANGE_MINOR 0

#define VECFS_IMPORT_F_PRESERVE_NODE_IDS  (1U << 0)
#define VECFS_IMPORT_F_ABORT_EXISTING     (1U << 1)

#define VECFS_EXPORT_F_INCLUDE_VECTORS    (1U << 0)
#define VECFS_EXPORT_F_INCLUDE_GRAPH      (1U << 1)

#define VECFS_REBUILD_F_ASYNC             (1U << 2)
#define VECFS_REBUILD_F_PARALLEL          (1U << 3)

struct vecfs_ioctl_exchange_info {
    __u16 exchange_major;
    __u16 exchange_minor;

    __u16 element_type;
    __u16 distance_type;

    __u32 dimension;

    __u16 m;
    __u16 m0;
    __u16 max_levels;
    __u16 current_max_level;

    __u32 ef_construction;
    __u32 default_ef_search;

    __u64 node_count;
    __u64 node_capacity;

    __u32 entry_node_id;
    __u32 flags;

    __u64 epoch;
    __u64 random_seed;

    __u64 vector_bytes;
    __u64 reserved[8];
};

struct vecfs_ioctl_import_begin {
    struct vecfs_ioctl_exchange_info info;

    /*
     * 输入：预计导入的 LIVE node 数量。
     */
    __u64 expected_nodes;

    /*
     * 输入：最高 internal node ID + 1。
     *
     * node ID 0 保留；capacity 至少为 2。
     */
    __u32 node_capacity;

    __u32 flags;

    /*
     * 输出：本次导入事务 ID。
     */
    __u64 transaction_id;

    __u64 reserved[4];
};

struct vecfs_ioctl_import_node {
    __u64 transaction_id;

    __u32 node_id;
    __u16 level;
    __u16 flags;

    __u64 external_id;

    __u64 vector_ptr;
    __u32 vector_bytes;

    __u16 level0_count;
    __u16 upper_level_count;

    /*
     * level-0 邻居数组：
     *
     *     level0_count * __u32
     */
    __u64 level0_links_ptr;

    /*
     * upper_counts_ptr:
     *
     *     level * __u16
     *
     * upper_links_ptr:
     *
     *     level * M * __u32
     *
     * 即每个 upper level 固定 M 个槽位，只有 counts 中声明的
     * 前 count 个槽位有效。
     */
    __u64 upper_counts_ptr;
    __u64 upper_links_ptr;

    __u64 reserved[4];
};

struct vecfs_ioctl_import_commit {
    __u64 transaction_id;

    __u32 entry_node_id;
    __u16 current_max_level;
    __u16 flags;

    __s32 status;
    __u32 reserved32;

    __u64 imported_nodes;
    __u64 new_epoch;

    __u64 reserved[4];
};

struct vecfs_ioctl_import_abort {
    __u64 transaction_id;
    __u64 reserved[3];
};

struct vecfs_ioctl_export_node {
    /*
     * 输入 node ID。
     */
    __u32 node_id;

    /*
     * 输入 flags。
     */
    __u32 flags;

    /*
     * 输出基础信息。
     */
    __u64 external_id;

    __u16 level;
    __u16 level0_count;
    __u16 upper_level_count;
    __u16 reserved16;

    /*
     * 用户输出缓冲区。
     */
    __u64 vector_ptr;
    __u32 vector_capacity;
    __u32 vector_bytes;

    __u64 level0_links_ptr;
    __u32 level0_capacity;
    __u32 level0_count_out;

    __u64 upper_counts_ptr;
    __u32 upper_counts_capacity;
    __u32 upper_counts_count_out;

    __u64 upper_links_ptr;
    __u32 upper_links_capacity;
    __u32 upper_links_count_out;

    __u64 reserved[4];
};


struct vecfs_ioctl_rebuild {
    __u32 flags;
    __s32 status;

    __u64 request_id;

    __u64 source_live_nodes;
    __u64 source_deleted_nodes;

    __u64 rebuilt_nodes;
    __u64 removed_nodes;

    __u64 old_epoch;
    __u64 new_epoch;

    __u32 old_arena_inode;
    __u32 new_arena_inode;

    __u64 duration_ns;

    __u64 reserved[7];
};

#define VECFS_IOC_ANN_REBUILD \
    _IOWR(VECFS_IOCTL_MAGIC, 0x41, struct vecfs_ioctl_rebuild)


enum vecfs_uapi_element_type {
    VECFS_UAPI_ELEM_FP32 = 1,
    VECFS_UAPI_ELEM_FP16 = 2,
    VECFS_UAPI_ELEM_INT8 = 3,
    VECFS_UAPI_ELEM_INT16 = 4,
};

enum vecfs_uapi_distance_type {
    VECFS_UAPI_DIST_L2 = 1,
    VECFS_UAPI_DIST_IP = 2,
};

enum vecfs_uapi_owner_type {
    VECFS_UAPI_OWNER_STANDALONE = 1,
    VECFS_UAPI_OWNER_FILE = 2,
    VECFS_UAPI_OWNER_IMPORTED = 3,
};

struct vecfs_ioctl_info {
    __u16 abi_major;
    __u16 abi_minor;
    __u16 fs_major;
    __u16 fs_minor;

    __u32 dimension;
    __u16 element_type;
    __u16 distance_type;

    __u16 m;
    __u16 m0;
    __u16 max_levels;
    __u16 current_max_level;

    __u32 ef_construction;
    __u32 default_ef_search;

    __u64 live_nodes;
    __u64 deleted_nodes;
    __u64 next_node_id;

    __u32 entry_node_id;
    __u32 simd_mode;

    __u64 object_stride;
    __u64 vector_bytes;

    __u64 reserved[8];
};

struct vecfs_ioctl_runtime {
    __u32 ef_search;
    __u32 flags;
    __u32 maximum_queries;
    __u32 reserved;
};

struct vecfs_ioctl_vector_add {
    __u64 vector_ptr;
    __u32 vector_bytes;
    __u32 flags;

    __u64 external_id;

    /*
     * 输出字段。
     */
    __u64 result_external_id;
    __u32 result_node_id;
    __u16 result_level;
    __u16 reserved;
};

struct vecfs_ioctl_vector_update {
    __u64 vector_ptr;
    __u32 vector_bytes;

    /*
     * 0..1000000，默认建议 1000000。
     */
    __u32 neighbor_probability_ppm;

    __u64 external_id;
};

struct vecfs_ioctl_vector_label {
    __u64 external_id;
    __u32 flags;
    __u32 reserved;
};

struct vecfs_ioctl_file_vector_add {
    __u64 vector_ptr;
    __u32 vector_bytes;
    __u32 flags;

    __u64 external_id;
    __u32 ordinal;
    __u32 reserved0;

    __u64 content_offset;
    __u64 content_length;

    /*
     * 输出字段。
     */
    __u64 result_external_id;
    __u32 result_node_id;
    __u32 result_ordinal;

    __u16 result_level;
    __u16 reserved1;
    __u32 reserved2;
};

struct vecfs_ioctl_file_vector_update {
    __u64 vector_ptr;
    __u32 vector_bytes;
    __u32 neighbor_probability_ppm;

    __u32 ordinal;
    __u32 flags;

    __u64 content_offset;
    __u64 content_length;
};

struct vecfs_ioctl_file_vector_delete {
    __u32 ordinal;
    __u32 flags;
};

struct vecfs_ioctl_knn_result {
    __u64 external_id;

    __u64 ino;
    __u32 inode_generation;
    __u32 vector_ordinal;

    __u64 content_offset;
    __u64 content_length;

    __u32 owner_type;
    __u32 flags;

    /*
     * IEEE-754 FP32 位模式，避免在 UAPI 结构中直接使用 float。
     */
    __u32 distance_bits;

    /*
     * 相对于 path_buffer 的偏移和长度。
     * 路径不包含结尾 NUL，字符串表中实际会写入 NUL。
     */
    __u32 path_offset;
    __u32 path_length;
    __u32 reserved;
};

struct vecfs_ioctl_knn {
    __u64 query_ptr;
    __u32 query_bytes;

    __u32 k;
    __u32 ef_search;
    __u32 flags;

    __u64 results_ptr;
    __u32 result_capacity;
    __u32 result_count;

    __u64 path_buffer_ptr;
    __u32 path_buffer_capacity;
    __u32 path_buffer_bytes;
};

struct vecfs_ioctl_vector_info {
    __u64 external_id;

    __u64 ino;
    __u32 inode_generation;
    __u32 vector_ordinal;

    __u64 content_offset;
    __u64 content_length;

    __u32 node_id;
    __u32 state;

    __u16 level;
    __u16 owner_type;
    __u16 flags;
    __u16 reserved16;

    __u64 reserved[4];
};

struct vecfs_ioctl_vector_get {
    __u64 external_id;

    /*
     * 可选输出向量缓冲区。
     *
     * vector_ptr == 0 时只返回 metadata。
     */
    __u64 vector_ptr;
    __u32 vector_capacity;
    __u32 vector_bytes;

    __u32 flags;
    __u32 reserved;

    struct vecfs_ioctl_vector_info info;
};

struct vecfs_ioctl_file_vector_get {
    __u32 ordinal;
    __u32 flags;

    /*
     * 可选输出向量缓冲区。
     */
    __u64 vector_ptr;
    __u32 vector_capacity;
    __u32 vector_bytes;

    struct vecfs_ioctl_vector_info info;
};

struct vecfs_ioctl_file_vector_list {
    /*
     * 输入：从该 ordinal 开始，包括该 ordinal。
     */
    __u32 start_ordinal;

    /*
     * 输入：用户 entries_ptr 可容纳的条目数。
     */
    __u32 entry_capacity;

    /*
     * 输入标志。
     */
    __u32 flags;

    /*
     * 输出条目数。
     */
    __u32 entry_count;

    /*
     * 输出：下次调用应使用的 start_ordinal。
     *
     * done == 1 时该值无意义。
     */
    __u32 next_ordinal;

    /*
     * 输出：是否已经遍历完成。
     */
    __u32 done;

    __u64 entries_ptr;
};

struct vecfs_ioctl_validate_report {
    __u64 objects_checked;
    __u64 live_objects;
    __u64 deleted_objects;
    __u64 corrupt_objects;

    __u64 file_objects;
    __u64 standalone_objects;
    __u64 imported_objects;

    __u64 inode_mappings_checked;

    __u64 missing_inodes;
    __u64 generation_mismatches;
    __u64 missing_inode_mappings;
    __u64 stale_inode_mappings;
    __u64 owner_mismatches;
    __u64 ordinal_mismatches;
    __u64 duplicate_external_ids;
    __u64 invalid_neighbors;

    __u64 objects_marked_deleted;
    __u64 mappings_cleared;
    __u64 metadata_repairs;

    __u32 recovered_entry_node;
    __u16 recovered_max_level;
    __u16 reserved16;

    __u64 reserved[8];
};

struct vecfs_ioctl_validate {
    __u32 flags;
    __s32 status;

    struct vecfs_ioctl_validate_report report;
};


struct vecfs_ioctl_rebuild_status {
    __u64 request_id;

    __u32 state;
    __s32 status;

    __u64 source_live_nodes;
    __u64 source_deleted_nodes;
    __u64 rebuilt_nodes;
    __u64 removed_nodes;

    __u64 old_epoch;
    __u64 new_epoch;

    __u32 old_arena_inode;
    __u32 new_arena_inode;

    __u64 duration_ns;
};

struct vecfs_ioctl_rebuild_cancel {
    __u64 request_id;
};

#define VECFS_IOC_ANN_REBUILD_STATUS \
    _IOWR(VECFS_IOCTL_MAGIC, 0x42, \
          struct vecfs_ioctl_rebuild_status)

#define VECFS_IOC_ANN_REBUILD_CANCEL \
    _IOW(VECFS_IOCTL_MAGIC, 0x43, \
         struct vecfs_ioctl_rebuild_cancel)

#define VECFS_IOC_EXPORT_INFO \
    _IOR(VECFS_IOCTL_MAGIC, 0x50, struct vecfs_ioctl_exchange_info)

#define VECFS_IOC_EXPORT_NODE \
    _IOWR(VECFS_IOCTL_MAGIC, 0x51, struct vecfs_ioctl_export_node)

#define VECFS_IOC_IMPORT_BEGIN \
    _IOWR(VECFS_IOCTL_MAGIC, 0x58, struct vecfs_ioctl_import_begin)

#define VECFS_IOC_IMPORT_NODE \
    _IOW(VECFS_IOCTL_MAGIC, 0x59, struct vecfs_ioctl_import_node)

#define VECFS_IOC_IMPORT_COMMIT \
    _IOWR(VECFS_IOCTL_MAGIC, 0x5a, struct vecfs_ioctl_import_commit)

#define VECFS_IOC_IMPORT_ABORT \
    _IOW(VECFS_IOCTL_MAGIC, 0x5b, struct vecfs_ioctl_import_abort)


#define VECFS_IOC_VECTOR_GET \
    _IOWR(VECFS_IOCTL_MAGIC, 0x14, struct vecfs_ioctl_vector_get)

#define VECFS_IOC_FILE_VECTOR_GET \
    _IOWR(VECFS_IOCTL_MAGIC, 0x23, struct vecfs_ioctl_file_vector_get)

#define VECFS_IOC_FILE_VECTOR_LIST \
    _IOWR(VECFS_IOCTL_MAGIC, 0x24, struct vecfs_ioctl_file_vector_list)

#define VECFS_IOC_ANN_VALIDATE \
    _IOWR(VECFS_IOCTL_MAGIC, 0x40, struct vecfs_ioctl_validate)


#define VECFS_IOC_GET_INFO \
    _IOR(VECFS_IOCTL_MAGIC, 0x00, struct vecfs_ioctl_info)

#define VECFS_IOC_GET_RUNTIME \
    _IOR(VECFS_IOCTL_MAGIC, 0x01, struct vecfs_ioctl_runtime)

#define VECFS_IOC_SET_RUNTIME \
    _IOW(VECFS_IOCTL_MAGIC, 0x02, struct vecfs_ioctl_runtime)

#define VECFS_IOC_VECTOR_ADD \
    _IOWR(VECFS_IOCTL_MAGIC, 0x10, struct vecfs_ioctl_vector_add)

#define VECFS_IOC_VECTOR_UPDATE \
    _IOW(VECFS_IOCTL_MAGIC, 0x11, struct vecfs_ioctl_vector_update)

#define VECFS_IOC_VECTOR_DELETE \
    _IOW(VECFS_IOCTL_MAGIC, 0x12, struct vecfs_ioctl_vector_label)

#define VECFS_IOC_VECTOR_UNDELETE \
    _IOW(VECFS_IOCTL_MAGIC, 0x13, struct vecfs_ioctl_vector_label)

#define VECFS_IOC_FILE_VECTOR_ADD \
    _IOWR(VECFS_IOCTL_MAGIC, 0x20, struct vecfs_ioctl_file_vector_add)

#define VECFS_IOC_FILE_VECTOR_UPDATE \
    _IOW(VECFS_IOCTL_MAGIC, 0x21, struct vecfs_ioctl_file_vector_update)

#define VECFS_IOC_FILE_VECTOR_DELETE \
    _IOW(VECFS_IOCTL_MAGIC, 0x22, struct vecfs_ioctl_file_vector_delete)

#define VECFS_IOC_KNN_SEARCH \
    _IOWR(VECFS_IOCTL_MAGIC, 0x30, struct vecfs_ioctl_knn)

/*
 * RTANN 调度器模式（RQ2 消融实验用）。所有模式都通过 workqueue 并行执行；
 * OFF = 并行裸跑基线（run-to-completion、零策略）；其余模式在 microstep
 * 执行模型之上叠加并发/预算/塑造/补偿组件。
 */
#define VECFS_SCHED_MODE_OFF      0  /* 并行裸跑基线（workqueue RTC，零策略） */
#define VECFS_SCHED_MODE_MICRO    1  /* 仅 microstep（P1） */
#define VECFS_SCHED_MODE_SHAPING  2  /* microstep + 并发塑造（P1+P3 shaping） */
#define VECFS_SCHED_MODE_BOOST    3  /* microstep + victim 补偿（P1+P3 boost） */
#define VECFS_SCHED_MODE_FULL     4  /* 完整 RTANN（P1+P2+P3） */

enum vecfs_sched_tail_reason {
    VECFS_SCHED_TAIL_NONE          = 0,
    VECFS_SCHED_TAIL_QUEUE_WAIT    = 1,
    VECFS_SCHED_TAIL_FIRST_RUN     = 2,
    VECFS_SCHED_TAIL_RESCUE_WAIT   = 3,
    VECFS_SCHED_TAIL_WAVE_WAIT     = 4,
    VECFS_SCHED_TAIL_RUN_TIME      = 5,
    VECFS_SCHED_TAIL_WORKLOAD      = 6,
    VECFS_SCHED_TAIL_LOW_RATE      = 7,
    VECFS_SCHED_TAIL_CTX_WAIT      = 8,
    VECFS_SCHED_TAIL_FRAGMENTATION = 9,
};

/*
 * 每个查询的调度器观测轨迹。字段全部为定宽整数，用户态用同布局结构读取。
 */
struct vecfs_ioctl_sched_trace {
    __u64 qid;

    __u64 total_us;
    __u64 backlog_wait_us;
    __u64 queue_wait_us;
    __u64 run_us;
    __u64 ctx_wait_us;
    __u64 rescue_wait_us;
    __u64 normal_wait_us;
    __u64 boost_wait_us;
    __u64 first_run_wait_us;

    __u32 dispatches;
    __u32 requeues;
    __u32 ctx_misses;
    __u32 zero_progress_batches;

    __u32 wave_id_submit;
    __u32 wave_id_admit;
    __u32 wave_id_finish;

    __u32 victim_entries;
    __u32 victim_exits;
    __u32 rescue_entries_seen;

    __u64 progress_units;
    __u32 dist_compute_cnt;
    __u32 tail_reason;

    __u64 avg_rate_q16;
    __u64 min_rate_q16;
    __u64 max_rate_q16;
};

struct vecfs_ioctl_knn_batch {
    /* 输入：查询向量 [n_queries, dim] fp32。 */
    __u64 queries_ptr;
    __u32 dim;
    __u32 k;
    __u32 ef_search;
    __u32 n_queries;

    __u32 sched_mode;   /* VECFS_SCHED_MODE_* */
    __u32 flags;        /* 保留，必须为 0 */

    /* 输出：node id [n_queries, k]（未命中为 0）。 */
    __u64 results_ptr;

    /* 输出：distance_bits（IEEE-754 FP32 位模式）[n_queries, k]；0 表示跳过。 */
    __u64 distances_ptr;

    /* 输出：每个查询的距离计算次数 [n_queries]。 */
    __u64 dist_counts_ptr;

    /* 输出：每个查询的调度轨迹 [n_queries]；0 表示跳过。 */
    __u64 trace_ptr;

    __u64 reserved[2];
};

#define VECFS_IOC_KNN_SEARCH_BATCH \
    _IOWR(VECFS_IOCTL_MAGIC, 0x31, struct vecfs_ioctl_knn_batch)

/*
 * SHIELD 派生有效性：模式控制与统计观测。
 */
#define VECFS_SHIELD_MODE_EVENT       1
#define VECFS_SHIELD_MODE_CONDITIONAL 2
#define VECFS_SHIELD_MODE_VALIDATED   3
#define VECFS_SHIELD_MODE_SHIELD      4

#define VECFS_SHIELD_STATS_F_RESET    (1U << 0)

struct vecfs_ioctl_shield_set_mode {
    __u32 mode;          /* VECFS_SHIELD_MODE_* */
    __u32 flags;
    __u64 window_us;     /* SHIELD 去抖窗口（微秒）；0 = 使用默认 */
    __u64 reserved[3];
};

struct vecfs_ioctl_shield_stats {
    __u32 flags;         /* VECFS_SHIELD_STATS_F_RESET */
    __u32 reserved;

    __u64 writes;
    __u64 tasks_begun;
    __u64 tasks_ended;
    __u64 tasks_expired;
    __u64 publishes_rejected;
    __u64 queries;
    __u64 results_returned;

    __u64 chunks_embedded;   /* P2：实际重嵌入并发布的块数 */
    __u64 chunks_reused;     /* P2：复用（未变）的块数 */
    __u64 chunks_redundant;  /* P2：内容未变但被重嵌的块数（复用关闭消融） */
    __u64 files_skipped;     /* P2：因二进制 SKIP 的文件数 */

    __u64 stale_sync_writes; /* R5g：写路径同步落盘 CONTENT_STALE 的次数 */
    __u64 stale_deferred;    /* R5g：延迟到 dirty-set（卸载 drain）的节点数 */
    __u64 clone_ns;          /* R5f：snapshot clone 累计耗时（ns） */
    __u64 clone_count;       /* R5f：snapshot clone 次数 */

    __u64 upcall_wait_ns;    /* upcall 嵌入阻塞等待累计耗时（ns） */
    __u64 upcall_count;      /* upcall 嵌入批次数 */
    __u64 persist_ns;        /* 派生批量持久化累计耗时（ns） */
    __u64 persist_count;     /* 持久化批次数 */

    __u64 build_ns;          /* HNSW 图构建累计耗时（ns） */
    __u64 build_count;       /* 图构建批次数 */
    __u64 publish_ns;        /* 发布累计耗时（ns：bind + labels + publish_objects） */
    __u64 publish_count;     /* 发布批次数 */
};

/* P2 派生策略配置。 */
struct vecfs_ioctl_shield_strategy {
    __u32 strategy_id;   /* 0 = 内置哈希假嵌入；1 = 用户态嵌入 upcall */
    __u32 flags;         /* VECFS_SHIELD_STRAT_F_* */
    __u64 chunk_size;    /* 定长分块字节数；0 = 默认 */
    __u64 reserved[3];
};

struct vecfs_ioctl_shield_task_begin {
    __u64 target_version;   /* 输出：任务目标版本 = 当前 v_f */
    __u64 reserved[3];
};

struct vecfs_ioctl_shield_task_end {
    __u64 target_version;   /* 输入：任务开始时记录的版本 */
    __u64 expired;          /* 输出：完成时目标版本是否已过期 */
    __u32 flags;
    __u32 reserved;
    __u64 reserved64[2];
};

/* 每文件派生状态（SHIELD 模式） */
struct vecfs_ioctl_shield_file_state {
    __u32 state;            /* 输出：VECFS_SHIELD_FS_* */
    __u32 reserved;
    __u64 shield_version;   /* 输出：当前源状态 v_f */
    __u64 target_version;   /* 输出：待派生目标 v_f */
    __u64 semantic_version; /* 输出：已发布 v_f */
    __u64 first_change;     /* 输出：待派生区间 */
    __u64 last_change;
};

#define VECFS_IOC_SHIELD_SET_MODE \
    _IOW(VECFS_IOCTL_MAGIC, 0x60, struct vecfs_ioctl_shield_set_mode)

#define VECFS_IOC_SHIELD_GET_STATS \
    _IOWR(VECFS_IOCTL_MAGIC, 0x61, struct vecfs_ioctl_shield_stats)

#define VECFS_IOC_SHIELD_TASK_BEGIN \
    _IOWR(VECFS_IOCTL_MAGIC, 0x62, struct vecfs_ioctl_shield_task_begin)

#define VECFS_IOC_SHIELD_TASK_END \
    _IOWR(VECFS_IOCTL_MAGIC, 0x63, struct vecfs_ioctl_shield_task_end)

#define VECFS_IOC_SHIELD_FILE_STATE \
    _IOWR(VECFS_IOCTL_MAGIC, 0x64, struct vecfs_ioctl_shield_file_state)

#define VECFS_IOC_SHIELD_SET_STRATEGY \
    _IOW(VECFS_IOCTL_MAGIC, 0x65, struct vecfs_ioctl_shield_strategy)

#endif
