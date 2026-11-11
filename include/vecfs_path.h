#ifndef _VECFS_PATH_H
#define _VECFS_PATH_H

#include <linux/mutex.h>
#include <linux/types.h>
#include <linux/xarray.h>

struct inode;
struct qstr;
struct super_block;

struct vecfs_path_entry {
    u64 ino;
    u64 parent_ino;

    u32 inode_generation;
    u16 name_length;
    u16 flags;

    char name[];
};

#define VECFS_PATH_ENTRY_F_DIRECTORY  (1U << 0)

struct vecfs_path_cache {
    /*
     * xarray key 为 inode number，value 为 vecfs_path_entry *。
     *
     * 路径解析不在 HNSW 热循环中，因此这里使用 mutex 保证 rename、
     * unlink 与结果路径组装之间的一致性。不同 ANN 查询的图搜索仍然
     * 完全无锁；只有搜索结束后的路径组装可能短暂串行。
     */
    struct mutex lock;
    struct xarray entries;

    u64 entry_count;
    bool initialized;
    bool complete;
};

int vecfs_path_cache_init(
    struct super_block *sb);

void vecfs_path_cache_destroy(
    struct super_block *sb);

int vecfs_path_cache_rebuild(
    struct super_block *sb);

int vecfs_path_cache_set(
    struct super_block *sb,
    u64 ino,
    u32 inode_generation,
    u64 parent_ino,
    const struct qstr *name,
    bool directory);

int vecfs_path_cache_remove(
    struct super_block *sb,
    u64 ino,
    u32 inode_generation);

int vecfs_path_cache_rename(
    struct super_block *sb,
    u64 ino,
    u32 inode_generation,
    u64 new_parent_ino,
    const struct qstr *new_name,
    bool directory);

/*
 * 返回相对于 VecFS 根目录的路径，不带开头 '/'。
 *
 * 根目录返回 "."。
 *
 * destination 中始终写入结尾 NUL，但 length 不包含 NUL。
 */
int vecfs_result_relative_path(
    struct super_block *sb,
    u64 ino,
    u32 generation,
    char *destination,
    size_t capacity,
    size_t *length);

#endif
