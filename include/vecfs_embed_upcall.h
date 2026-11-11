// SPDX-License-Identifier: GPL-2.0
#ifndef _VECFS_EMBED_UPCALL_H
#define _VECFS_EMBED_UPCALL_H

/*
 * 内核 → 用户态嵌入服务 upcall（批量）。
 *
 * 设计约束（与 SHIELD 一致）：文件系统是决策者，用户态是被调用方。内核在
 * SHIELD 派生时决定「何时嵌入、嵌入哪些块」，一次把 N 个已变块打包成一个
 * upcall；用户态守护进程（vecfs-embed-helper）应答后，一次 HTTP 调用嵌入 N 个
 * 块——对齐用户态 baseline 的 embed_many 粒度（每文件一次 /api/embed）。
 *
 * 传输：misc 字符设备 /dev/vecfs-embed。请求/响应各 16 字节头，其后变长载荷：
 *   请求载荷 = nchunks × [ __u32 len(le) + data[len] ]
 *   响应载荷（status==0）= nchunks × dim 个 fp32（le 位模式）
 *
 * 守护进程由文件系统自行管理：mount 时内核 call_usermodehelper 拉起 helper，
 * unmount（refcount 归零）时经设备 read 返回 0(EOF) 让 helper 自然退出。
 *
 * 本头文件同时供内核（core/embed_upcall.c）与用户态 helper
 * （tools/vecfs-embed-helper.c）使用。
 */

/*
 * 模块名（fs 类型 + misc 设备名前缀）。默认 "vecfs"；Makefile 以
 * -DVECFS_MOD_NAME=\"$(MODNAME)\" 覆盖（MODNAME=vecfs2 时即 "vecfs2"），
 * 用于临时改名与旧残留模块共存。改回只需把 Makefile 的 MODNAME 换回 vecfs
 * 并重编译，无需改动各处字符串字面量。放在这个头里是因为它是内核
 * core/vecfs_main.c 与 core/embed_upcall.c 共同包含的头文件。
 */
#ifndef VECFS_MOD_NAME
#define VECFS_MOD_NAME "vecfs"
#endif

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
typedef uint64_t __u64;
typedef uint32_t __u32;
typedef int32_t  __s32;
#ifndef __packed
#define __packed __attribute__((packed))
#endif
#endif

/* 单批块数上限，与 VECFS_SHIELD_MAX_CHUNKS 一致（防御性）。 */
#define VECFS_EMBED_UP_MAX_CHUNKS 4096U

/* 单批数据字节总和上限，与 VECFS_SHIELD_MAX_EMBED_BYTES 对齐。 */
#define VECFS_EMBED_UP_MAX_DATA (256 * 1024)

/* 响应向量维度上限，与 VECFS_ANN_MAX_DIM 一致（防御恶意守护进程）。 */
#define VECFS_EMBED_UP_MAX_DIM 4096

/* 内核等待守护进程应答的超时（毫秒）；超时按 -EAGAIN 返回，由 SHIELD 重试。 */
#define VECFS_EMBED_UP_TIMEOUT_MS (30 * 1000)

/*
 * 与守护进程共享的 wire ABI（各 16 字节头，__packed）。
 */
struct vecfs_embed_request {
    __u64 id;       /* 内核分配，用于应答匹配 */
    __u32 nchunks;  /* 本批块数（>=1） */
    __u32 dim;      /* 期望输出维度 = 索引维度 */
    /* 其后紧跟 nchunks × [ __u32 len(le); u8 data[len] ] */
} __packed;

struct vecfs_embed_response {
    __u64 id;       /* 回显请求 id */
    __s32 status;   /* 0 = 成功（其后跟 nchunks×dim 个 fp32）；负 -errno = 失败（无载荷） */
    __u32 nchunks;  /* 成功时 = 请求 nchunks */
    /* status==0 时其后紧跟 nchunks × dim 个 fp32（le 位模式） */
} __packed;

/*
 * 控制通道事件（16 字节，与请求/响应头同宽，单向：内核 → 守护）。
 *
 * 主动取消（P1）协议：守护对 /dev/vecfs-embed 二次 open（O_RDONLY）得到
 * 控制 fd，内核写路径在版本推进后向控制通道下发 CANCEL(id)，守护据此
 * 中止该请求的底层嵌入计算。协议只保证「结果不再返回 + 内核不再消费」；
 * 后端能否真停算力由后端语义决定（能力约束，分层表述）。
 */
#define VECFS_EMBED_CTRL_CANCEL 1

struct vecfs_embed_control {
    __u32 opcode;   /* VECFS_EMBED_CTRL_CANCEL */
    __u32 reserved;
    __u64 id;       /* 要取消的请求 id */
} __packed;

#ifdef __KERNEL__

struct vecfs_shield_chunk; /* 前置声明（定义见 vecfs_shield.h） */

/*
 * 批量嵌入：把 base 上 chunks[0..nchunks) 描述的块一次打包进单个 upcall，
 * 应答后把 nchunks×dim 个 fp32 位模式写入 out（chunks[i] → out[i*dim..]）。
 * 返回 0 成功；-ECANCELED 被主动取消；-EAGAIN 守护不可用/超时；其余负
 * -errno 永久失败。与 SHIELD 统一返回约定一致。
 *
 * generation：触发本次派生的目标版本（target_version），仅作请求代际标识
 *            与观测，不参与取消判定（取消按唯一 id 键控）。
 * req_id_out：可选（可 NULL）。非 NULL 时，内核在入队后把分配到的请求 id
 *             写入 *req_id_out（WRITE_ONCE），请求结束（应答/超时/取消）
 *             前清零——供写路径经 READ_ONCE 发现在途请求以触发主动取消。
 *             该槽位须在本次阻塞调用期间保持有效（通常指向 inode 内嵌的
 *             每文件在途句柄）。
 */
int vecfs_embed_upcall_embed_batch(const void *base,
                                   const struct vecfs_shield_chunk *chunks,
                                   u32 nchunks, u32 dim, u32 *out,
                                   u64 generation, u64 *req_id_out);

/*
 * 主动取消（P1）：令在途请求 id 立即以 -ECANCELED 结束（worker 唤醒走
 * requeue），并向控制通道（若守护已开）下发 CANCEL 事件，尽力让后端中止
 * 计算。不阻塞等待后端。id 不存在/已结束返回 -ENOENT（无副作用）。
 */
int vecfs_embed_upcall_cancel(u64 id);

/*
 * 守护进程生命周期（由 mount/unmount 驱动，文件系统自行管理）。
 * start：首次挂载（refcount 0→1）时 call_usermodehelper 拉起 helper，配置经
 *        env 传入（key 不进 cmdline）；后续挂载仅自增 refcount。
 * stop：refcount 递减；归零时置 stopping、唤醒 helper 使其退出。
 */
int vecfs_embed_upcall_start(const char *url, const char *key,
                             const char *model, const char *proto);
void vecfs_embed_upcall_stop(void);

int vecfs_embed_upcall_init(void);
void vecfs_embed_upcall_exit(void);

#endif /* __KERNEL__ */

#endif /* _VECFS_EMBED_UPCALL_H */
