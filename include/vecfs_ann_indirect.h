#ifndef _VECFS_ANN_INDIRECT_H
#define _VECFS_ANN_INDIRECT_H

#include <linux/fs.h>
#include <linux/types.h>

#include "vecfs_disk.h"

typedef int (*vecfs_ann_node_iter_fn)(
    struct inode *inode,
    u32 ordinal,
    u32 node_id,
    void *private_data);

/*
 * 读取文件内 vector ordinal 对应的 node ID。
 *
 * 返回：
 *   0       找到映射
 *   -ENOENT 映射不存在
 *   -EINVAL 参数错误
 */
int vecfs_ann_get_node(
    struct inode *inode,
    u32 ordinal,
    u32 *node_id);

/*
 * 建立 ordinal -> node_id 映射。
 *
 * replace=false 时，如果 ordinal 已有映射，返回 -EEXIST。
 * replace=true  时允许替换已有映射。
 */
int vecfs_ann_set_node(
    struct inode *inode,
    u32 ordinal,
    u32 node_id,
    bool replace);

/*
 * 删除 ordinal 映射。
 *
 * 只删除 inode ANN 间接树中的映射，不删除 node object。
 */
int vecfs_ann_clear_node(
    struct inode *inode,
    u32 ordinal);

/*
 * 遍历文件中的所有有效 ordinal -> node_id 映射。
 */
int vecfs_ann_for_each_node(
    struct inode *inode,
    vecfs_ann_node_iter_fn callback,
    void *private_data);

/*
 * 校验 node object 是否属于指定文件和 ordinal。
 */
int vecfs_ann_validate_file_binding(
    struct inode *inode,
    u32 ordinal,
    u32 node_id);

/*
 * 将已经创建的 FILE node 绑定到 inode。
 *
 * 该函数会先校验 object 的 owner_ino、generation 和 ordinal。
 */
int vecfs_ann_bind_file_node(
    struct inode *inode,
    u32 ordinal,
    u32 node_id,
    bool replace);

/*
 * 删除文件关联的全部向量：
 *
 * 1. 标记 node object 为 DELETED；
 * 2. 清除 inode ANN 间接树映射。
 */
int vecfs_ann_remove_all_nodes(
    struct inode *inode);

/*
 * 兼容旧代码。
 *
 * V2 不再通过该接口写入向量或自动调用 HNSW。
 * create=true 时返回 -EOPNOTSUPP。
 */
int vecfs_ann_get_block(
    struct inode *inode,
    sector_t ordinal,
    void *vector,
    int create);

/*
 * 兼容旧代码的块分配包装函数。
 */
int vecfs_ann_alloc_block(
    struct inode *inode,
    u32 *block);

void vecfs_ann_free_block(
    struct inode *inode,
    u32 block);

/*
 * 兼容旧的 inode eviction 调用。
 *
 * V2 中 newsize 只接受 0，表示删除该文件的全部向量。
 */
int vecfs_ann_truncate_blocks(
    struct inode *inode,
    loff_t newsize);

int vecfs_alloc_fs_block(
    struct inode *inode,
    u32 *block);

int vecfs_zero_fs_block(
    struct inode *inode,
    u32 block);

#endif
