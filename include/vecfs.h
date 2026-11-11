#ifndef VECFS_H
#define VECFS_H

#include <linux/fs.h>
#include "vecfs_compat.h"
#include "vecfs_Indirect.h"
#include <linux/buffer_head.h>
#include <linux/blkdev.h>
#include <linux/writeback.h>
#include "include/vecfs_disk.h"
#include "include/vecfs_ann_meta.h"
#include "include/vecfs_ann_store.h"
#include "include/vecfs_ann_distance.h"
#include "include/vecfs_ann_cache.h"
#include "include/vecfs_ann_runtime.h"
#include "include/vecfs_ann_sched.h"
#include "include/vecfs_ann_label.h"
#include "vecfs_ann_exchange.h"
#include "include/vecfs_ann_rebuild.h"
#include "vecfs_path.h"
#include "vecfs_shield.h"


#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif
#define BLOCK_SIZE 4096
#define INODE_SIZE 128
#define BLOCK_BITS 12
#ifdef SECTOR_SIZE
#undef SECTOR_SIZE
#endif
#define SECTOR_SIZE 512
#define SECTOR_BITS 9
#define VECFS_MAGIC 0xEF53BEEF
#define VECFS_FIRST_INO 10
#define VECFS_DIR_REC_LEN(namelen) (((namelen) + 8 + 3) & ~3)
#define VECFS_MAX_NAME_LEN 30
#define VECFS_EMBED_MAGIC "VECFSVEC"
#define VECFS_EMBED_MAGIC_LEN 8


extern struct file_system_type vecfs_type;

#if LINUX_VERSION_CODE < KERNEL_VERSION(7, 0, 0)
struct dentry *vecfs_mount(struct file_system_type * fs_type, int flags, const char * devname, void * data);
#endif
void vecfs_kill_sb(struct super_block * sb);
int vecfs_readdir(struct file * file, struct dir_context * ctx);
VECFS_MKDIR_RET vecfs_mkdir(VECFS_IDMAP idmap, struct inode * dir, struct dentry * dentry, umode_t mode);
int vecfs_rmdir(struct inode * dir, struct dentry * dentry);
int vecfs_unlink(struct inode * dir, struct dentry * dentry);
struct dentry *vecfs_lookup(struct inode * dir, struct dentry * dentry, unsigned int flags);
int vecfs_rename(VECFS_IDMAP idmap, struct inode * old_dir, struct dentry * old_dentry, struct inode * new_dir, struct dentry * new_dentry, unsigned int flags);
int vecfs_symlink(VECFS_IDMAP idmap, struct inode * dir, struct dentry * dentry, const char * symname);void vecfs_destroy_inode(struct inode * inode);
void vecfs_put_super(struct super_block * sb);
int vecfs_super_set_clean(struct super_block *sb, bool clean);
struct inode *vecfs_alloc_inode(struct super_block * sb);
int vecfs_write_inode(struct inode * inode, struct writeback_control * wbc);
int vecfs_read_inode(struct inode *inode);
int vecfs_readpage(struct file * file, struct folio * folio);
#if LINUX_VERSION_CODE < KERNEL_VERSION(7, 0, 0)
int vecfs_writepage(struct page * page, struct writeback_control * wbc);
#endif
int vecfs_writepages(struct address_space *mapping, struct writeback_control *wbc);
int vecfs_write_begin(VECFS_WB_FIRST file, struct address_space *mapping, loff_t pos, unsigned len, VECFS_WB_PAGEP foliop, void **fsdata);
int vecfs_write_end(VECFS_WB_FIRST file, struct address_space *mapping, loff_t pos, unsigned len, unsigned copied, VECFS_WB_PAGE folio, void *fsdata);
void vecfs_evict_inode(struct inode *inode);
void vecfs_free_inode(struct super_block *sb, struct inode *inode);
int vecfs_init_directory(struct inode *dir, struct inode *parent);
ino_t get_new_ino(struct super_block *sb);
struct inode *vecfs_create_inode(VECFS_IDMAP idmap, struct super_block *sb,
                                const struct inode *dir,
                                umode_t mode,
                                const struct qstr *qname);
int vecfs_find_entry(struct inode *dir, const struct qstr *name, ino_t *ino);
int vecfs_add_entry(struct inode *dir, const struct qstr *name, ino_t ino);
int vecfs_remove_entry(struct inode *dir, const struct qstr *name);
bool vecfs_dir_empty(struct inode *dir);
int vecfs_store_symlink(struct inode *inode, const char *symname);
unsigned long *load_bitmap(struct super_block *sb, sector_t start_block, 
                                unsigned int total_entries);
#if LINUX_VERSION_CODE >= KERNEL_VERSION(7, 0, 0)
int vecfs_fill_super(struct super_block *sb, struct fs_context *fc);
#else
int vecfs_fill_super(struct super_block *sb, void *data, int silent);
#endif
sector_t vecfs_inode_to_block(struct super_block *sb, unsigned long ino);
unsigned int vecfs_inode_offset(struct super_block *sb, unsigned long ino);

// 读取/写入迭代器与 fsync/release 钩子
ssize_t vecfs_read_iter(struct kiocb *iocb, struct iov_iter *to);
ssize_t vecfs_write_iter(struct kiocb *iocb, struct iov_iter *from);
int vecfs_fsync(struct file *file, loff_t start, loff_t end, int datasync);
int vecfs_release(struct inode *inode, struct file *file);
int vecfs_create(
    VECFS_IDMAP idmap,
    struct inode *dir,
    struct dentry *dentry,
    umode_t mode,
    bool excl);
int vecfs_setattr(
    VECFS_IDMAP idmap,
    struct dentry *dentry,
    struct iattr *attr);
int vecfs_mmap(
    struct file *file,
    struct vm_area_struct *vma);


// 目录条目结构
struct vecfs_dir_entry {
    __le32 inode;           // inode 号
    __le16 rec_len;         // 目录条目长度
    __le16 name_len;        // 文件名长度
    char name[VECFS_MAX_NAME_LEN];         // 文件名（可变长度）
};


// 尾部标记与头部结构（固定大小，位于文件末尾）
#define VECFS_EMBED_MAGIC "VECFSVEC"
#define VECFS_EMBED_MAGIC_LEN 8

struct vecfs_tail_hdr {
__u8 magic[VECFS_EMBED_MAGIC_LEN]; // "VECFSVEC"
__le32 version; // 头部版本，建议取 1
__le32 vec_dim; // 向量维度
__le16 elem_size; // 每元素字节数（例如 4=fp32，1=int8）
__le32 vec_count; // 向量个数（通常 1，也可>1）
__le64 payload_size; // 向量负载总字节数（vec_count * vec_dim * elem_size）
__le64 content_size; // 原始可见内容长度（尾部分离后文件应截断到此）
__le32 crc32; // 可选校验，0 表示未使用
} __packed;


struct ino_allocator {
    spinlock_t lock;          // 分配器自旋锁
    unsigned long *bitmap;    // inode 位图
    unsigned long start_ino;  // 起始 inode 号
    
    unsigned long free_inos;  // 空闲 inode 计数
    unsigned long total_inos; // 总 inode 数
};

struct vecfs_sb_info {
    struct buffer_head *sbh;

    u32 s_magic;
    u32 s_blocksize;
    u32 s_root_ino;

    struct mutex s_lock;

    unsigned long *s_imap;
    u64 s_imap_start;
    unsigned long s_imap_blocks;
    u32 s_inodes_count;
    u32 s_inode_table;
    u32 s_free_inodes;
    unsigned long s_next_ino;

    unsigned long *s_bmap;
    u64 s_bmap_start;
    unsigned long s_data_start;
    unsigned long s_bmap_blocks;
    u32 s_blocks_count;
    u32 s_free_blocks;
    unsigned long long s_next_free_block;
    spinlock_t s_block_lock;

    struct vecfs_path_cache s_path_cache;

    /*
     * VecFS V2 ANN metadata。
     */
    sector_t s_ann_meta_block[2];
    struct vecfs_ann_meta s_ann;
    struct vecfs_ann_store *s_ann_store;
    struct vecfs_ann_distance s_ann_distance;
    struct vecfs_ann_runtime s_ann_runtime;
    struct vecfs_ann_label_index s_ann_labels;

    struct vecfs_ann_search_pool *s_ann_search_pool;
    struct vecfs_ann_cache *s_ann_cache;
    struct vecfs_ann_sched *s_ann_sched;
    struct vecfs_ann_exchange s_ann_exchange;
    struct vecfs_ann_rebuild_async s_ann_rebuild_async;

    struct vecfs_shield_runtime s_shield;

    u64 s_features;
    u16 s_disk_major;
    u16 s_disk_minor;

    /* 嵌入服务配置（superblock 镜像，NUL 结尾 C 串）。 */
    char s_embed_url[256];
    char s_embed_key[256];
    char s_embed_model[64];
    char s_embed_protocol[16];
    bool s_embed_configured;   /* url 非空 = 挂载时 spawn 嵌入守护 */

    struct list_head s_dirty_dirents;
    atomic_t s_dirty_count;

    /*
     * Phase D：异步 stale 持久化 + 干净卸载标记。
     *
     * s_clean_unmount     挂载时从 superblock clean_magic 读得，判断是否
     *                      需要保守恢复（崩溃兜底）。
     * s_stale_dirty_marked superblock 是否已被改写为 dirty 并落盘（首条
     *                      延迟 stale 落盘前置 dirty，卸载 drain 后置 clean）。
     */
    bool s_clean_unmount;
    bool s_stale_dirty_marked;
};


struct vecfs_inode {
    // 标准 inode 字段（约 160 字节）
    struct inode vfs_inode;
    
    // ================= 块映射状态 =================
    __u32 i_block[10]; // 多级索引块指针（7个直接 + 3个间接）
    // ================= ANN节点映射状态 =================
    __u32 i_index[10];
    
    /* 文件状态 */
    atomic_t open_count;             // 打开计数器
    
    // ================= 并发控制 =================
    spinlock_t i_raw_lock;           // 原始数据自旋锁
    struct rw_semaphore meta_lock;   // 元数据读写锁
    
    // ================= 缓存管理 =================
    struct address_space i_data;     // 页缓存映射
    //struct list_head i_orphan;       // 孤儿inode链表
    unsigned long i_state_flags;     // 状态位（如I_DIRTY）
    
    // ================= 扩展属性 =================
    struct rw_semaphore xattr_sem;   // 扩展属性锁
    
    // ================= 统计信息 =================
    atomic64_t access_count;         // 访问计数器

    /*
     * SHIELD 源状态版本 v_f。仅内存态、不持久化；每次提交的
     * 内容修改（write/truncate）单调递增。用于派生目标过期判断。
     */
    u64 shield_version;

    /* SHIELD 每文件派生状态机（见 vecfs_shield.h） */
    struct vecfs_shield_file_state shield;

    uint8_t vec_flags;               // 加密/压缩标志
    uint8_t padding[7];              // 内存对齐
};

// 声明 inode 缓存指针
extern struct kmem_cache *vecfs_inode_cachep;
extern const struct file_operations vecfs_file_ops;
extern const struct file_operations vecfs_file_dir_ops;
extern const struct inode_operations vecfs_inode_ops;
extern const struct inode_operations vecfs_inode_dir_ops;
extern const struct super_operations vecfs_super_ops;
extern const struct address_space_operations vecfs_address_space_ops;


static inline struct vecfs_inode *VECFS_I(struct inode *inode)
{
	return container_of(inode, struct vecfs_inode, vfs_inode);
}


static inline struct vecfs_sb_info *VECFS_SB(struct super_block *sb)
{
	return sb->s_fs_info;
}
#endif
