// SPDX-License-Identifier: GPL-2.0
/*
 * vecfs 跨内核 6.x 兼容层。
 *
 * 把 6.1 → 6.14 之间变化的 VFS/内存 API 集中在这里，用
 * LINUX_VERSION_CODE 做守卫。当前只覆盖到 6.14（folio write 路径）；
 * 6.17 的 write_begin/write_end 改签名（struct file* → const struct
 * kiocb*、writepage 移除）暂未处理。
 *
 * 注意：6.1 以 shield-computon-v3 为准——该内核已回移 ctime 访问器
 * （inode_set_ctime / inode_set_ctime_current / inode_set_ctime_to_ts），
 * 故此处不为其提供 shim；vanilla 6.1.0（无回移）不在支持范围。
 */
#ifndef _VECFS_COMPAT_H
#define _VECFS_COMPAT_H

#include <linux/version.h>
#include <linux/fs.h>
#include <linux/fs_context.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/time.h>
#include <linux/buffer_head.h>
#include <linux/user_namespace.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 2, 0)
#include <linux/mnt_idmapping.h>
#endif

/* unaligned 访问器：6.9 起合并进 linux/unaligned.h，之前用 asm/unaligned.h */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 9, 0)
#include <linux/unaligned.h>
#else
#include <asm/unaligned.h>
#endif

/* ------------------------------------------------------------------ *
 * (1) idmapped mount：6.1 用 struct user_namespace *，6.2+ 用 struct
 *     mnt_idmap *（并配套 nop_mnt_idmap）。
 * ------------------------------------------------------------------ */
#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 2, 0)
#define VECFS_IDMAP      struct user_namespace *
#define VECFS_IDMAP_NOP  (&init_user_ns)
#else
#define VECFS_IDMAP      struct mnt_idmap *
#define VECFS_IDMAP_NOP  (&nop_mnt_idmap)
#endif

/* ------------------------------------------------------------------ *
 * (2) struct address_space.private_list 重命名为 i_private_list（6.7）。
 * ------------------------------------------------------------------ */
#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 7, 0)
#define vecfs_mapping_private_list(m)  (&(m)->private_list)
#else
#define vecfs_mapping_private_list(m)  (&(m)->i_private_list)
#endif

/* ------------------------------------------------------------------ *
 * (3) kvmemdup（6.2 引入）：kvmalloc + memcpy 的等价 shim。
 * ------------------------------------------------------------------ */
#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 2, 0)
static inline void *kvmemdup(const void *src, size_t len, gfp_t gfp)
{
	void *p = kvmalloc(len, gfp);

	if (p)
		memcpy(p, src, len);
	return p;
}
#endif

/* ------------------------------------------------------------------ *
 * (4) atime/mtime 时间戳访问器（6.7 引入；ctime 访问器 6.2 起已有）。
 *     6.1 的 inode 里 i_atime/i_mtime/i_ctime 仍是 struct timespec64，
 *     直接按字段读写。
 * ------------------------------------------------------------------ */
#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 7, 0)
static inline time64_t inode_get_atime_sec(const struct inode *inode)
{
	return inode->i_atime.tv_sec;
}

static inline time64_t inode_get_ctime_sec(const struct inode *inode)
{
	return inode->i_ctime.tv_sec;
}

static inline time64_t inode_get_mtime_sec(const struct inode *inode)
{
	return inode->i_mtime.tv_sec;
}

static inline void inode_set_atime(struct inode *inode, time64_t sec,
				   long nsec)
{
	inode->i_atime = (struct timespec64){ .tv_sec = sec,
					     .tv_nsec = nsec };
}

static inline void inode_set_mtime(struct inode *inode, time64_t sec,
				   long nsec)
{
	inode->i_mtime = (struct timespec64){ .tv_sec = sec,
					     .tv_nsec = nsec };
}

static inline struct timespec64 inode_set_mtime_to_ts(struct inode *inode,
						      struct timespec64 ts)
{
	inode->i_mtime = ts;
	return ts;
}

static inline struct timespec64 simple_inode_init_ts(struct inode *inode)
{
	struct timespec64 ts = current_time(inode);

	inode->i_atime = inode->i_mtime = inode->i_ctime = ts;
	return ts;
}
#endif

/* ------------------------------------------------------------------ *
 * (5) write_begin/write_end 的 page → folio 转换（6.11）。
 *     首个参数在 6.1/6.14 均为 struct file *（6.17 才换成 kiocb，暂不
 *     处理）。
 * ------------------------------------------------------------------ */
#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 11, 0)
#define VECFS_WB_PAGEP   struct page **
#define VECFS_WB_PAGE    struct page *
#else
#define VECFS_WB_PAGEP   struct folio **
#define VECFS_WB_PAGE    struct folio *
#endif

/* ------------------------------------------------------------------ *
 * (6) 内核 7.0（≈ mainline 6.17+）VFS 迁移：
 *     - inode->i_state 由 unsigned long 改为 struct inode_state_flags，
 *       须经 inode_state_read_once() 读取再按位判断。
 *     - write_begin/write_end 首个参数由 struct file* 改为
 *       const struct kiocb*；writepage 从 address_space_operations 移除。
 *     - ->mkdir() 返回值由 int 改为 struct dentry *（NULL=成功用传入
 *       dentry，ERR_PTR 传错误）。
 * ------------------------------------------------------------------ */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(7, 0, 0)
#define VECFS_INODE_IS_NEW(inode)  (inode_state_read_once(inode) & I_NEW)
#define VECFS_WB_FIRST             const struct kiocb *
#define VECFS_MKDIR_RET            struct dentry *
#define VECFS_MKDIR_OK             NULL
#define VECFS_MKDIR_ERR(r)         ERR_PTR(r)
#else
#define VECFS_INODE_IS_NEW(inode)  ((inode)->i_state & I_NEW)
#define VECFS_WB_FIRST             struct file *
#define VECFS_MKDIR_RET            int
#define VECFS_MKDIR_OK             0
#define VECFS_MKDIR_ERR(r)         (r)
#endif

#endif /* _VECFS_COMPAT_H */
