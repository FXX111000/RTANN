#ifndef _VECFS_DISK_H
#define _VECFS_DISK_H

/*
 * 该文件只包含固定宽度磁盘格式。
 * 不得在磁盘结构中使用：
 *
 *     size_t
 *     unsigned long
 *     sector_t
 *     enum
 *     指针
 */

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/build_bug.h>
#else
#include <stdint.h>
#include <endian.h>

/* If linux/types.h types are not yet available, define them.
 * Use _LINUX_TYPES_H (a #define guard, not a typedef guard) to detect. */
#ifndef _LINUX_TYPES_H
typedef uint8_t  __u8;
typedef uint16_t __u16;
typedef uint32_t __u32;
typedef uint64_t __u64;

typedef uint16_t __le16;
typedef uint32_t __le32;
typedef uint64_t __le64;
#endif

#ifndef __packed
#define __packed __attribute__((packed))
#endif

struct vecfs_ann_config {
    __u32 dimension;
    __u16 element_type;
    __u16 distance_type;

    __u16 m;
    __u16 m0;
    __u16 max_levels;
    __u16 current_max_level;

    __u32 ef_construction;
    __u32 default_ef_search;

    __u64 random_seed;
    __u64 level_mult_q32;

    __u64 object_stride;
    __u64 vector_bytes;
};

#endif

#define VECFS_DISK_BLOCK_SIZE              4096U
#define VECFS_DISK_BLOCK_BITS              12U

#define VECFS_SUPER_MAGIC                  0xEF53BEEF
#define VECFS_SUPER_MAJOR                  2
#define VECFS_SUPER_MINOR                  1

#define VECFS_ANN_META_MAGIC               0x56414E4E /* "VANN" */
#define VECFS_ANN_META_MAJOR               2
#define VECFS_ANN_META_MINOR               0

#define VECFS_ANN_OBJECT_MAGIC             0x564E4F44 /* "VNOD" */
#define VECFS_ANN_OBJECT_VERSION           1

/*
 * 干净卸载标记（Phase D）：写入 superblock clean_magic。等于该值表示上次
 * 卸载时所有 CONTENT_STALE 标志已落盘；其它值（含 mkfs 初值 0）表示可能
 * 崩溃，挂载时需对 file-bound 向量保守失效并触发重派生。
 */
#define VECFS_SUPER_CLEAN_MAGIC            0x5653434c45414e31ULL /* "VSCLEAN1" */

#define VECFS_ROOT_INO                     1
#define VECFS_FIRST_USER_INO               10

/*
 * 阶段 2 将使用这些内部 inode。
 * 本阶段只保留编号，active_arena_ino 仍为 0。
 */
#define VECFS_ANN_ARENA_INO                2
#define VECFS_ANN_REBUILD_ARENA_INO        3

#define VECFS_DISK_INODE_SIZE              128
#define VECFS_N_BLOCK_PTRS                 10
#define VECFS_N_DIRECT                     7
#define VECFS_IND_BLOCK                    7
#define VECFS_DIND_BLOCK                   8
#define VECFS_TIND_BLOCK                   9

#define VECFS_ANN_INVALID_NODE_ID          ((__u32)~0U)
#define VECFS_ANN_EXTERNAL_ID_AUTO         ((__u64)~0ULL)
#define VECFS_ANN_ORDINAL_AUTO             ((__u32)~0U)
/*
 * i_index[] 和 ANN 间接块中的 0 表示无映射。
 * 实际 node ID 从 1 开始分配。
 */
#define VECFS_ANN_EMPTY_NODE_ID 0U
#define VECFS_ANN_FIRST_NODE_ID 1U

#define VECFS_ANN_MAX_DIM                  4096
#define VECFS_ANN_MAX_LEVELS               32
#define VECFS_ANN_MAX_M                    128
#define VECFS_ANN_MAX_M0                   256

#define VECFS_ANN_DEFAULT_DIM              96
#define VECFS_ANN_DEFAULT_M                16
#define VECFS_ANN_DEFAULT_M0               32
#define VECFS_ANN_DEFAULT_MAX_LEVELS       16
#define VECFS_ANN_DEFAULT_EF_CONSTRUCTION  200
#define VECFS_ANN_DEFAULT_EF_SEARCH        64
#define VECFS_ANN_DEFAULT_RANDOM_SEED      100ULL

/*
 * superblock feature flags。
 */
#define VECFS_FEATURE_ANN                   (1ULL << 0)
#define VECFS_FEATURE_ANN_META_AB           (1ULL << 1)
#define VECFS_FEATURE_ANN_SHARED_DATA       (1ULL << 2)

/*
 * 向量元素格式。
 * V2 第一版只实现 FP32。
 */
enum vecfs_ann_element_type {
    VECFS_ANN_ELEM_INVALID = 0,
    VECFS_ANN_ELEM_FP32    = 1,
    VECFS_ANN_ELEM_FP16    = 2,
    VECFS_ANN_ELEM_INT8    = 3,
    VECFS_ANN_ELEM_INT16   = 4,
};

enum vecfs_ann_distance_type {
    VECFS_ANN_DIST_INVALID = 0,
    VECFS_ANN_DIST_L2      = 1,
    VECFS_ANN_DIST_IP      = 2,
};

enum vecfs_ann_meta_state {
    VECFS_ANN_META_EMPTY      = 0,
    VECFS_ANN_META_ACTIVE     = 1,
    VECFS_ANN_META_REBUILDING = 2,
    VECFS_ANN_META_IMPORTING  = 3,
    VECFS_ANN_META_ERROR      = 4,
};

enum vecfs_ann_rebuild_disk_state {
    VECFS_ANN_REBUILD_DISK_NONE     = 0,
    VECFS_ANN_REBUILD_DISK_PREPARED = 1,
    VECFS_ANN_REBUILD_DISK_COPYING  = 2,
    VECFS_ANN_REBUILD_DISK_READY    = 3,
    VECFS_ANN_REBUILD_DISK_SWITCHED = 4,
};

enum vecfs_ann_object_state {
    VECFS_ANN_OBJ_FREE     = 0,
    VECFS_ANN_OBJ_PREPARED = 1,
    VECFS_ANN_OBJ_LIVE     = 2,
    VECFS_ANN_OBJ_DELETED  = 3,
    VECFS_ANN_OBJ_CORRUPT  = 4,
};

enum vecfs_ann_owner_type {
    VECFS_ANN_OWNER_NONE       = 0,
    VECFS_ANN_OWNER_STANDALONE = 1,
    VECFS_ANN_OWNER_FILE       = 2,
    VECFS_ANN_OWNER_IMPORTED   = 3,
};

#define VECFS_ANN_OBJ_F_HAS_CONTENT_RANGE  (1U << 0)
#define VECFS_ANN_OBJ_F_CONTENT_STALE      (1U << 1)
#define VECFS_ANN_OBJ_F_EXTERNAL_ID_AUTO   (1U << 2)

/*
 * SHIELD 查询端 Valid(x,f) 的「无效」标记：仅由 effective_flags 在查询
 * 结果上动态计算，表示该节点不被任何当前已提交语义根引用。只存在于内存
 * 查询路径，绝不写入 node->flags 或落盘。
 */
#define VECFS_ANN_OBJ_F_CONTENT_INVALID    (1U << 3)

/*
 * VecFS V2 超级块。
 *
 * 块 0 存放整个结构。
 * checksum 为 CRC32，计算时该字段必须置 0。
 */
struct vecfs_super_disk_v2 {
    __le32 magic;
    __le16 major;
    __le16 minor;

    __le32 header_size;
    __le32 checksum;

    __le64 features;
    __le64 write_time_sec;

    __le32 block_size;
    __le32 root_ino;

    __le64 inodes_count;
    __le64 blocks_count;

    __le64 data_start;
    __le64 inode_table;

    __le64 free_inodes;
    __le64 free_blocks;

    __le64 imap_start;
    __le64 bmap_start;

    __le32 imap_blocks;
    __le32 bmap_blocks;

    /*
     * ANN metadata A/B 各占一个块。
     */
    __le64 ann_meta_a;
    __le64 ann_meta_b;

    /*
     * 仅作为快速提示，mount 仍必须校验 A/B 的 CRC 和 sequence。
     * 0 = A，1 = B。
     */
    __le32 ann_active_hint;

    __le32 inode_size;

    /*
     * 嵌入服务配置（mkfs 写入）。全 0 表示未配置，derive 落到内置哈希嵌入。
     * 统一为 url + key（本地模型 key 留空）+ model；protocol 预留扩展后端。
     */
    __u8 embed_url[256];       /* 例 http://127.0.0.1:11434 */
    __u8 embed_key[256];       /* API key；本地模型留空 */
    __u8 embed_model[64];      /* 例 embeddinggemma */
    __u8 embed_protocol[16];   /* 例 ollama；预留其它后端 */

    /*
     * 干净卸载标记（Phase D）。见 VECFS_SUPER_CLEAN_MAGIC。
     */
    __u8 reserved[3360];
    __le64 clean_magic;
} __packed;

/*
 * ANN 根元数据，一个完整磁盘块。
 */
struct vecfs_ann_meta_disk {
    __le32 magic;
    __le16 major;
    __le16 minor;

    __le32 header_size;
    __le32 checksum;

    __le64 sequence;
    __le64 epoch;

    __le32 state;
    __le32 flags;

    __le32 dimension;
    __le16 element_type;
    __le16 distance_type;

    __le16 m;
    __le16 m0;
    __le16 max_levels;
    __le16 current_max_level;

    __le32 ef_construction;
    __le32 default_ef_search;

    __le64 random_seed;

    /*
     * 1/log(M)，Q32.32。
     */
    __le64 level_mult_q32;

    __le64 live_nodes;
    __le64 deleted_nodes;
    __le64 next_node_id;
    __le64 next_external_id;

    __le32 entry_node_id;
    __le32 active_arena_ino;

    __le32 rebuild_arena_ino;
    __le32 rebuild_state;

    __le64 object_stride;
    __le64 vector_bytes;

    __le64 committed_txid;

    __u8 reserved[3944];
} __packed;

/*
 * 阶段 2 的对象头现在先冻结下来。
 * object_stride 根据格式化参数动态计算。
 */
struct vecfs_ann_object_disk {
    __le32 magic;
    __le16 version;
    __le16 header_size;

    __le32 node_id;
    __le32 state;

    __le64 external_id;

    __le64 owner_ino;
    __le32 owner_generation;
    __le32 vector_ordinal;

    __le64 content_offset;
    __le64 content_length;

    __le16 level;
    __le16 level0_count;
    __le16 owner_type;
    __le16 flags;

    __le64 object_epoch;

    __le32 vector_crc;
    __le32 object_crc;
} __packed;

/*
 * 保留原 128-byte inode 布局，避免修改传统文件系统数据映射。
 *
 * 注意：
 *   i_block[] 为普通文件块树；
 *   i_index[] 为文件向量 ordinal -> node ID 的多级间接树。
 */
struct vecfs_inode_disk_v2 {
    __le16 mode;
    __le16 uid;
    __le32 size_lo;

    __le32 atime;
    __le32 ctime;
    __le32 mtime;
    __le32 dtime;

    __le16 gid;
    __le16 links_count;

    __le32 blocks_lo;
    __le32 flags;
    __le32 generation;

    __le32 i_block[VECFS_N_BLOCK_PTRS];
    __le32 i_index[VECFS_N_BLOCK_PTRS];

    __le32 size_high;
    __le32 reserved;
} __packed;

#ifdef __KERNEL__
static_assert(sizeof(struct vecfs_super_disk_v2) == 4096);
static_assert(sizeof(struct vecfs_ann_meta_disk) == 4096);
static_assert(sizeof(struct vecfs_ann_object_disk) == 80);
static_assert(sizeof(struct vecfs_inode_disk_v2) == 128);
#else
_Static_assert(sizeof(struct vecfs_super_disk_v2) == 4096,
               "bad VecFS superblock size");
_Static_assert(sizeof(struct vecfs_ann_meta_disk) == 4096,
               "bad VecFS ANN metadata size");
_Static_assert(sizeof(struct vecfs_ann_object_disk) == 80,
               "bad VecFS ANN object header size");
_Static_assert(sizeof(struct vecfs_inode_disk_v2) == 128,
               "bad VecFS inode size");
#endif

#endif
