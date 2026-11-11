#ifndef _VECFS_ANN_STALE_H
#define _VECFS_ANN_STALE_H

#include <linux/types.h>

struct inode;
struct super_block;

/*
 * 标记与 [offset, offset + length) 相交的 FILE 向量。
 */
int vecfs_ann_mark_content_stale(
    struct inode *inode,
    u64 offset,
    u64 length);

/*
 * 文件缩小到 new_size 时，标记所有越过 new_size 的内容绑定。
 */
int vecfs_ann_mark_content_stale_after(
    struct inode *inode,
    u64 new_size);

/*
 * 清除指定 FILE 向量的 stale 状态。
 *
 * 复用路径在「内容哈希未变」时调用，期望向量仍对应 expected_version 的
 * 内容。函数在 writer_lock 下复核 inode 当前 shield_version 是否仍等于
 * expected_version：若写路径已推进版本（并即将/已经保守标记该向量 stale），
 * 则返回 -ESTALE 且不清除 stale，杜绝「复用清除 stale」与「并发写重新标记
 * stale」之间的竞态把过期向量重新激活。expected_version 应为触发本次派生的
 * target_version。
 */
int vecfs_ann_clear_file_vector_stale(
    struct inode *inode,
    u32 ordinal,
    u64 expected_version);

/*
 * SHIELD 查询端 Valid(x,f) 的 valid 位操作。
 *
 * mark_node_valid 把节点置为「被当前语义根引用」（valid 位 = 1），
 * mark_node_invalid 把节点置为无效（valid 位 = 0，孤儿/根已失效）。
 * 二者在 writer_lock 下修改 in-memory valid_bitmap，不落盘。
 */
int vecfs_ann_mark_node_valid(
    struct inode *inode,
    u32 ordinal);

int vecfs_ann_mark_node_invalid(
    struct inode *inode,
    u32 ordinal);

/*
 * Phase D：把 stale_dirty 脏集合中待落盘的 CONTENT_STALE 标志同步写盘。
 * 仅在 writer_lock 下调用（drain 与 mark 的 __set_bit 互斥）。返回已落盘
 * 节点数或负 -errno。
 */
int vecfs_ann_stale_drain(
    struct super_block *sb);

/*
 * Phase D 崩溃恢复：把所有 file-bound LIVE 向量保守失效（内存 stale 位 +
 * 磁盘 CONTENT_STALE 标志同步落盘）。仅在挂载发现非干净卸载时调用。返回
 * 失效节点数或负 -errno。
 */
int vecfs_ann_recover_file_vectors(
    struct super_block *sb);

#endif
