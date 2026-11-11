// SPDX-License-Identifier: GPL-2.0
//
// 内核 → 用户态嵌入服务 upcall（批量）。
//
// 提供一个 misc 字符设备 /dev/vecfs-embed，让用户态嵌入守护进程
// （vecfs-embed-helper，Ollama embeddinggemma）被动应答内核 SHIELD 派生时
// 发来的「批量嵌入请求」：
//
//   守护进程：open(注册，单例) → read(阻塞取请求：16B 头 + nchunks×[len+data])
//             → 一次 Ollama /api/embed（input=[nchunks 个文本]）
//             → write(16B 头 + nchunks×dim 个 fp32 位模式)
//
// 内核侧：SHIELD derive worker 收集「已变需重嵌」的块，一次
// vecfs_embed_upcall_embed_batch() 打包入队、唤醒守护、阻塞等应答；超时/无守护
// 返回 -EAGAIN，由 SHIELD 既有 requeue 重试，不丢 PENDING。一次派生仅一次
// HTTP 调用，对齐用户态 baseline 的 embed_many 粒度。
//
// 守护进程生命周期由文件系统自行管理（mount/unmount 驱动）：
//   - vecfs_embed_upcall_start():首次挂载时 call_usermodehelper 拉起 helper，
//     配置（url/key/model/protocol）经 env 传入（key 不进 cmdline）；后续挂载
//     仅自增 refcount。
//   - vecfs_embed_upcall_stop():refcount 递减，归零置 stopping 并唤醒 helper，
//     read 返回 0(EOF) → helper 自然退出。
//
// 并发与所有权：
//   - 每个请求由调用者（derive worker）独占创建与释放，且是唯一释放者。
//   - 请求的链表节点在「应答者」（write 或 release/exit）与调用者之间通过
//     g_upcall.lock 互斥，list_del_init + list_empty 保证恰好摘除一次。
//   - 应答者从不访问请求的 wire/out_buf 之外的调用者缓冲，无 use-after-free。

#include "vecfs_compat.h"
#include <linux/completion.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/list.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/poll.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/uaccess.h>
#include <linux/umh.h>
#include <linux/vmalloc.h>
#include <linux/wait.h>

#include "../include/vecfs_shield.h"
#include "../include/vecfs_embed_upcall.h"

/*
 * 请求生命周期状态：
 *   PENDING 待守护 read；INFLIGHT 已交给守护、待 write；
 *   CANCEL_REQUESTED 写路径已请求取消（结果立即按 -ECANCELED 结束，不再
 *   接受应答）。INFLIGHT→CANCEL_REQUESTED 与 INFLIGHT→应答完成在
 *   g_upcall.lock 下互斥，只成其一。
 */
enum vecfs_embed_req_state {
	VECFS_EMBED_REQ_PENDING = 0,
	VECFS_EMBED_REQ_INFLIGHT = 1,
	VECFS_EMBED_REQ_CANCEL_REQUESTED = 2,
};

struct vecfs_embed_upcall_req {
	struct list_head list;
	u64 id;
	u64 generation; /* 触发派生的 target_version（仅标识/观测，不参与取消判定） */
	u32 nchunks;
	u32 dim;
	u8 *wire;      /* 扁平化请求载荷：nchunks × [le32(len) + data]（kvmalloc） */
	size_t wire_len;
	u32 *out_buf;  /* nchunks × dim 个 fp32 位模式（kvmalloc），应答写入这里 */
	int status;    /* 0 = 成功；负 -errno */
	int state;
	struct completion done;

	/* 观测计时（仅定位 upcall IPC 开销，ktime_get_ns 单调钟）。 */
	u64 submit_ns;  /* 提交时刻（wake_up 前） */
	u64 read_ns;    /* 守护 read 取走时刻 */
	u64 write_ns;   /* 守护 write 应答时刻 */
};

/* 控制通道事件环容量：单守护串行处理、同一时刻至多一个在途请求，16 足够。 */
#define VECFS_EMBED_CTRL_QUEUE_MAX 16

struct vecfs_embed_upcall_ctx {
	struct mutex lock;
	wait_queue_head_t wq;      /* 守护进程阻塞等待新 PENDING 请求 */
	struct list_head reqs;     /* PENDING + INFLIGHT + CANCEL_REQUESTED 在列 */
	u64 next_id;
	bool connected;            /* 单守护数据通道：open(O_RDWR) 置位，release 清除 */
	bool ctrl_connected;       /* 控制通道：open(O_RDONLY) 置位，release 清除 */
	int refcount;              /* 需要嵌入服务的挂载点数 */
	bool stopping;             /* refcount 归零：通知守护退出 */

	/* 控制通道事件环（内核 → 守护，单向，容量有界；溢出则丢弃——取消
	 * 效果已由「立即 -ECANCELED 完成」保证，事件只是尽力让后端停算）。 */
	wait_queue_head_t ctrl_wq;
	struct vecfs_embed_control ctrl_ring[VECFS_EMBED_CTRL_QUEUE_MAX];
	u32 ctrl_head;
	u32 ctrl_count;

	/* 当前守护配置（首次 start 记录，用于多挂载一致性检查）。 */
	char url[256];
	char key[256];
	char model[64];
	char proto[16];
};

static struct vecfs_embed_upcall_ctx g_upcall;

/* helper 二进制路径（module_param；开发期 insmod 传树内绝对路径）。 */
static char *vecfs_embed_helper = "/sbin/vecfs-embed-helper";
module_param(vecfs_embed_helper, charp, 0644);
MODULE_PARM_DESC(vecfs_embed_helper,
		 "path to the userspace embedding helper binary");

/* ---------------------------------------------------------------------------
 * 内部：链表与匹配（均须持 g_upcall.lock）
 * ------------------------------------------------------------------------- */

static bool vecfs_embed_pending_locked(void)
{
	struct vecfs_embed_upcall_req *r;

	list_for_each_entry(r, &g_upcall.reqs, list)
		if (r->state == VECFS_EMBED_REQ_PENDING)
			return true;
	return false;
}

/* 供 wait_event 条件使用：内部持锁检查（调用方未持锁）。 */
static bool vecfs_embed_pending(void)
{
	bool has;

	mutex_lock(&g_upcall.lock);
	has = vecfs_embed_pending_locked();
	mutex_unlock(&g_upcall.lock);
	return has;
}

static bool vecfs_embed_stopping(void)
{
	bool stop;

	mutex_lock(&g_upcall.lock);
	stop = g_upcall.stopping;
	mutex_unlock(&g_upcall.lock);
	return stop;
}

/* 取走队首 PENDING（改 INFLIGHT，不摘链）。 */
static struct vecfs_embed_upcall_req *vecfs_embed_take_locked(void)
{
	struct vecfs_embed_upcall_req *r;

	list_for_each_entry(r, &g_upcall.reqs, list) {
		if (r->state == VECFS_EMBED_REQ_PENDING) {
			r->state = VECFS_EMBED_REQ_INFLIGHT;
			return r;
		}
	}
	return NULL;
}

static struct vecfs_embed_upcall_req *vecfs_embed_find_locked(u64 id)
{
	struct vecfs_embed_upcall_req *r;

	list_for_each_entry(r, &g_upcall.reqs, list)
		if (r->id == id && r->state == VECFS_EMBED_REQ_INFLIGHT)
			return r;
	return NULL;
}

/* 取消用：匹配尚未终结的请求（PENDING 待派发或 INFLIGHT 已派发）。 */
static struct vecfs_embed_upcall_req *vecfs_embed_find_cancellable_locked(u64 id)
{
	struct vecfs_embed_upcall_req *r;

	list_for_each_entry(r, &g_upcall.reqs, list)
		if (r->id == id &&
		    (r->state == VECFS_EMBED_REQ_PENDING ||
		     r->state == VECFS_EMBED_REQ_INFLIGHT))
			return r;
	return NULL;
}

/* 恰好摘链一次（可被应答者或调用者调用，幂等）。 */
static void vecfs_embed_req_unlink_locked(struct vecfs_embed_upcall_req *req)
{
	if (!list_empty(&req->list))
		list_del_init(&req->list);
}

/* 控制通道事件环（均须持 g_upcall.lock）。溢出丢弃，不阻塞写路径。 */
static void vecfs_embed_ctrl_enqueue_locked(const struct vecfs_embed_control *c)
{
	u32 tail;

	if (g_upcall.ctrl_count >= VECFS_EMBED_CTRL_QUEUE_MAX)
		return; /* 环满：丢弃。取消的立即 -ECANCELED 完成已保证正确性。 */

	tail = (g_upcall.ctrl_head + g_upcall.ctrl_count) %
	       VECFS_EMBED_CTRL_QUEUE_MAX;
	g_upcall.ctrl_ring[tail] = *c;
	g_upcall.ctrl_count++;
}

/* 出队一个控制事件（调用方已确认非空）。 */
static void vecfs_embed_ctrl_dequeue_locked(struct vecfs_embed_control *out)
{
	*out = g_upcall.ctrl_ring[g_upcall.ctrl_head];
	g_upcall.ctrl_head = (g_upcall.ctrl_head + 1) %
			     VECFS_EMBED_CTRL_QUEUE_MAX;
	g_upcall.ctrl_count--;
}

static bool vecfs_embed_ctrl_pending_locked(void)
{
	return g_upcall.ctrl_count != 0;
}

/* 供 wait_event 条件使用：内部持锁检查（调用方未持锁）。 */
static bool vecfs_embed_ctrl_pending(void)
{
	bool has;

	mutex_lock(&g_upcall.lock);
	has = vecfs_embed_ctrl_pending_locked();
	mutex_unlock(&g_upcall.lock);
	return has;
}

/* ---------------------------------------------------------------------------
 * 批量嵌入入口（SHIELD derive worker 调用，可睡眠）
 * ------------------------------------------------------------------------- */

int vecfs_embed_upcall_embed_batch(const void *base,
				   const struct vecfs_shield_chunk *chunks,
				   u32 nchunks, u32 dim, u32 *out,
				   u64 generation, u64 *req_id_out)
{
	struct vecfs_embed_upcall_req *req;
	size_t data_sum = 0;
	size_t wire_len;
	size_t vec_bytes;
	u8 *p;
	u32 j;
	long rem;
	int ret;

	if (!out || !base || !chunks || nchunks == 0 ||
	    nchunks > VECFS_EMBED_UP_MAX_CHUNKS)
		return -EINVAL;
	if (dim == 0 || dim > VECFS_EMBED_UP_MAX_DIM)
		return -EINVAL;

	/* 校验并累计载荷尺寸（每块长度 + 总和均设防）。 */
	for (j = 0; j < nchunks; j++) {
		if (chunks[j].length == 0 ||
		    chunks[j].length > VECFS_EMBED_UP_MAX_DATA)
			return -EINVAL;
		if (data_sum > (size_t)VECFS_EMBED_UP_MAX_DATA - chunks[j].length)
			return -E2BIG;
		data_sum += chunks[j].length;
	}
	wire_len = (size_t)nchunks * sizeof(u32) + data_sum;
	vec_bytes = (size_t)nchunks * dim * sizeof(u32);

	req = kzalloc(sizeof(*req), GFP_KERNEL);
	if (!req)
		return -ENOMEM;
	req->wire = kvmalloc(wire_len, GFP_KERNEL);
	if (!req->wire) {
		kfree(req);
		return -ENOMEM;
	}
	req->out_buf = kvmalloc(vec_bytes, GFP_KERNEL);
	if (!req->out_buf) {
		kvfree(req->wire);
		kfree(req);
		return -ENOMEM;
	}

	/* 扁平化：nchunks × [ le32(len) + data ]。 */
	p = req->wire;
	for (j = 0; j < nchunks; j++) {
		put_unaligned_le32((u32)chunks[j].length, p);
		p += sizeof(u32);
		memcpy(p, base + chunks[j].offset, chunks[j].length);
		p += chunks[j].length;
	}
	req->wire_len = wire_len;
	req->nchunks = nchunks;
	req->dim = dim;
	req->generation = generation;
	req->state = VECFS_EMBED_REQ_PENDING;
	req->status = -EAGAIN;
	init_completion(&req->done);

	mutex_lock(&g_upcall.lock);
	/*
	 * 关闭排空中：stopping 已置位即拒绝新请求（-EAGAIN 立即返回），而非
	 * 等待 30s 超时。否则卸载时大量积压 worker 会在 stop() 完成既有请求后
	 * 陆续提交新请求、各自阻塞 30s，经 system_long_wq 有限 worker 线程
	 * 一轮轮串行超时，排空被拖到远超 40s 硬上限，导致 inode 泄漏。
	 */
	if (!g_upcall.connected || g_upcall.stopping) {
		mutex_unlock(&g_upcall.lock);
		ret = -EAGAIN;
		goto free_req;
	}
	req->id = ++g_upcall.next_id;
	if (req->id == 0)
		req->id = ++g_upcall.next_id;
	list_add_tail(&req->list, &g_upcall.reqs);
	/* 入列后（cancel 已可按 id 找到）才向写路径发布在途句柄。 */
	if (req_id_out)
		WRITE_ONCE(*req_id_out, req->id);
	mutex_unlock(&g_upcall.lock);

	req->submit_ns = ktime_get_ns();
	wake_up_interruptible(&g_upcall.wq);

	rem = wait_for_completion_interruptible_timeout(
		&req->done, msecs_to_jiffies(VECFS_EMBED_UP_TIMEOUT_MS));

	mutex_lock(&g_upcall.lock);
	if (rem > 0) {
		/* 守护已应答（write）、release/exit 按 -EAGAIN 结束，或已被取消
		 * （-ECANCELED）。 */
		ret = req->status;
		vecfs_embed_req_unlink_locked(req);
		mutex_unlock(&g_upcall.lock);
		if (!ret)
			memcpy(out, req->out_buf, vec_bytes);
		goto free_req;
	}

	/* 超时或被信号打断：摘链放弃，守护后续 write 将因找不到 id 而丢弃。 */
	vecfs_embed_req_unlink_locked(req);
	mutex_unlock(&g_upcall.lock);
	ret = -EAGAIN;

free_req:
	if (req_id_out)
		WRITE_ONCE(*req_id_out, 0);
	kvfree(req->out_buf);
	kvfree(req->wire);
	kfree(req);
	return ret;
}

/* ---------------------------------------------------------------------------
 * 主动取消（P1）
 * ------------------------------------------------------------------------- */

/*
 * 令在途请求 id 立即以 -ECANCELED 结束：INFLIGHT→CANCEL_REQUESTED，摘链并
 * complete（worker 唤醒走 requeue）；若守护已开控制通道，再下发 CANCEL(id)
 * 事件尽力让后端中止计算。不阻塞等待后端。
 *
 * 写路径语义分层：内核侧取消保证「结果不再返回 + 内核不再消费」（complete
 * 后应答 write 因找不到 INFLIGHT id 而丢弃）；后端能否真停算力是后端能力，
 * 由控制事件尽力触发。
 */
int vecfs_embed_upcall_cancel(u64 id)
{
	struct vecfs_embed_upcall_req *req;
	struct vecfs_embed_control c;
	int ret = -ENOENT;

	if (id == 0)
		return -EINVAL;

	mutex_lock(&g_upcall.lock);
	req = vecfs_embed_find_cancellable_locked(id);
	if (!req) {
		/* 已应答/超时/已取消：无副作用。 */
		mutex_unlock(&g_upcall.lock);
		return -ENOENT;
	}

	/* 仅 INFLIGHT（已派发给守护）需要下发控制事件让后端停算；PENDING
	 * 尚未被守护读到，直接摘除即可，守护 take 时不会见到它。 */
	c.opcode = VECFS_EMBED_CTRL_CANCEL;
	c.reserved = 0;
	c.id = id;
	if (req->state == VECFS_EMBED_REQ_INFLIGHT && g_upcall.ctrl_connected)
		vecfs_embed_ctrl_enqueue_locked(&c);

	req->state = VECFS_EMBED_REQ_CANCEL_REQUESTED;
	req->status = -ECANCELED;
	vecfs_embed_req_unlink_locked(req);
	complete(&req->done); /* 立即唤醒 worker（不阻塞等待后端） */
	ret = 0;
	mutex_unlock(&g_upcall.lock);

	if (g_upcall.ctrl_connected)
		wake_up_interruptible(&g_upcall.ctrl_wq);

	return ret;
}

/* ---------------------------------------------------------------------------
 * 守护进程生命周期（mount/unmount 驱动，文件系统自行管理）
 * ------------------------------------------------------------------------- */

int vecfs_embed_upcall_start(const char *url, const char *key,
			     const char *model, const char *proto)
{
	char *argv[2];
	char url_env[288], key_env[288], model_env[96], proto_env[64], dev_env[64];
	char *envp[8];
	int idx = 0;
	int ret = 0;

	if (!url || !url[0] || !model || !model[0])
		return -EINVAL;

	mutex_lock(&g_upcall.lock);

	if (g_upcall.refcount > 0) {
		/* 单模型假设：追加挂载的配置须一致；换模型需重 mkfs（全图重建）。 */
		if (strncmp(g_upcall.url, url, sizeof(g_upcall.url) - 1) ||
		    strncmp(g_upcall.model, model, sizeof(g_upcall.model) - 1) ||
		    strncmp(g_upcall.key, key ? key : "",
			    sizeof(g_upcall.key) - 1)) {
			pr_warn("vecfs-embed: embed config mismatch on additional mount\n");
			ret = -EBUSY;
			goto out;
		}
		g_upcall.refcount++;
		goto out;
	}

	/* 首次：记录配置并拉起 helper。 */
	strscpy(g_upcall.url, url, sizeof(g_upcall.url));
	strscpy(g_upcall.key, key ? key : "", sizeof(g_upcall.key));
	strscpy(g_upcall.model, model, sizeof(g_upcall.model));
	strscpy(g_upcall.proto, proto ? proto : "ollama",
		sizeof(g_upcall.proto));
	g_upcall.stopping = false;

	snprintf(url_env, sizeof(url_env), "VECFS_EMBED_URL=%s", url);
	snprintf(key_env, sizeof(key_env), "VECFS_EMBED_KEY=%s", key ? key : "");
	snprintf(model_env, sizeof(model_env), "VECFS_EMBED_MODEL=%s", model);
	snprintf(proto_env, sizeof(proto_env), "VECFS_EMBED_PROTO=%s",
		 proto ? proto : "ollama");
	snprintf(dev_env, sizeof(dev_env), "VECFS_EMBED_DEVICE=/dev/%s-embed",
		 VECFS_MOD_NAME);

	envp[idx++] = "HOME=/";
	envp[idx++] = "PATH=/sbin:/bin:/usr/sbin:/usr/bin";
	envp[idx++] = url_env;
	envp[idx++] = key_env;
	envp[idx++] = model_env;
	envp[idx++] = proto_env;
	envp[idx++] = dev_env;
	envp[idx] = NULL;

	argv[0] = vecfs_embed_helper;
	argv[1] = NULL;

	ret = call_usermodehelper(vecfs_embed_helper, argv, envp,
				  UMH_WAIT_EXEC);
	if (ret) {
		pr_err("vecfs-embed: call_usermodehelper(%s) failed: %d\n",
		       vecfs_embed_helper, ret);
		goto out; /* 不记 refcount，下次挂载可重试 */
	}

	g_upcall.refcount = 1;
	ret = 0;

out:
	mutex_unlock(&g_upcall.lock);
	return ret;
}

void vecfs_embed_upcall_stop(void)
{
	struct vecfs_embed_upcall_req *r, *tmp;

	mutex_lock(&g_upcall.lock);
	if (g_upcall.refcount > 0)
		g_upcall.refcount--;
	if (g_upcall.refcount == 0) {
		g_upcall.stopping = true;
		wake_up_interruptible(&g_upcall.wq);

		/*
		 * 立即完成所有在途请求（-EAGAIN），唤醒阻塞在
		 * wait_for_completion 的派生 worker。若不主动完成，worker 要
		 * 等到 30s 超时才醒，卸载排空（vecfs_shield_shutdown）会被
		 * 拖到 30s+。complete 在锁内调用安全：被唤醒方在应答路径会
		 * 重新争 g_upcall.lock，这里先摘链、释放锁后才由对方取锁。
		 */
		list_for_each_entry_safe(r, tmp, &g_upcall.reqs, list) {
			r->status = -EAGAIN;
			vecfs_embed_req_unlink_locked(r);
			complete(&r->done);
		}
	}
	mutex_unlock(&g_upcall.lock);
}

/* ---------------------------------------------------------------------------
 * misc 字符设备 file_operations
 * ------------------------------------------------------------------------- */

static int vecfs_embed_upcall_open(struct inode *inode, struct file *filp)
{
	mutex_lock(&g_upcall.lock);
	if (filp->f_mode & FMODE_WRITE) {
		/* 数据通道（O_RDWR）：单守护，读请求/写应答。 */
		if (g_upcall.connected) {
			mutex_unlock(&g_upcall.lock);
			return -EBUSY;
		}
		g_upcall.connected = true;
	} else {
		/* 控制通道（O_RDONLY）：单守护，只读 CANCEL 事件。 */
		if (g_upcall.ctrl_connected) {
			mutex_unlock(&g_upcall.lock);
			return -EBUSY;
		}
		g_upcall.ctrl_connected = true;
	}
	mutex_unlock(&g_upcall.lock);
	return 0;
}

static int vecfs_embed_upcall_release(struct inode *inode, struct file *filp)
{
	struct vecfs_embed_upcall_req *r, *tmp;

	mutex_lock(&g_upcall.lock);
	if (filp->f_mode & FMODE_WRITE) {
		/* 数据通道关闭：守护退出，结束所有未应答请求
		 * （-EAGAIN 让 derive 重试，守护可重连，worker 不挂死）。 */
		g_upcall.connected = false;
		list_for_each_entry_safe(r, tmp, &g_upcall.reqs, list) {
			r->status = -EAGAIN;
			vecfs_embed_req_unlink_locked(r);
			complete(&r->done);
		}
	} else {
		/* 控制通道关闭：清空事件环。 */
		g_upcall.ctrl_connected = false;
		g_upcall.ctrl_head = 0;
		g_upcall.ctrl_count = 0;
	}
	mutex_unlock(&g_upcall.lock);
	return 0;
}

static ssize_t vecfs_embed_upcall_read(struct file *filp, char __user *ubuf,
				       size_t count, loff_t *ppos)
{
	struct vecfs_embed_request hdr;
	struct vecfs_embed_upcall_req *req;
	struct vecfs_embed_control ctrl;
	size_t total;

	/* 控制通道：一次返回一个 16B 事件。 */
	if (!(filp->f_mode & FMODE_WRITE)) {
		if (count < sizeof(ctrl))
			return -EINVAL;

		mutex_lock(&g_upcall.lock);
		while (!vecfs_embed_ctrl_pending_locked()) {
			if (g_upcall.stopping) {
				mutex_unlock(&g_upcall.lock);
				return 0; /* EOF：通知 helper 退出 */
			}
			mutex_unlock(&g_upcall.lock);
			if (wait_event_interruptible(g_upcall.ctrl_wq,
						     vecfs_embed_ctrl_pending() ||
						     vecfs_embed_stopping()))
				return -ERESTARTSYS;
			mutex_lock(&g_upcall.lock);
		}
		vecfs_embed_ctrl_dequeue_locked(&ctrl);
		mutex_unlock(&g_upcall.lock);

		if (copy_to_user(ubuf, &ctrl, sizeof(ctrl)))
			return -EFAULT;
		return sizeof(ctrl);
	}

	mutex_lock(&g_upcall.lock);
	while (!(req = vecfs_embed_take_locked())) {
		if (g_upcall.stopping) {
			mutex_unlock(&g_upcall.lock);
			return 0; /* EOF：通知 helper 退出 */
		}
		mutex_unlock(&g_upcall.lock);
		if (wait_event_interruptible(g_upcall.wq,
					     vecfs_embed_pending() ||
					     vecfs_embed_stopping()))
			return -ERESTARTSYS;
		mutex_lock(&g_upcall.lock);
	}

	/*
	 * 持锁拷贝：worker 的超时路径（unlink + kvfree req->wire）也须持同一把
	 * 锁，持锁拷贝可避免它在本函数 copy_to_user 途中释放 wire（use-after-free）。
	 * copy_to_user 可能 page fault 睡眠，但 mutex 允许睡眠，安全。
	 */
	total = sizeof(hdr) + req->wire_len;
	if (count < total) {
		mutex_unlock(&g_upcall.lock);
		return -EINVAL;
	}

	hdr.id = req->id;
	hdr.nchunks = req->nchunks;
	hdr.dim = req->dim;

	if (copy_to_user(ubuf, &hdr, sizeof(hdr))) {
		mutex_unlock(&g_upcall.lock);
		return -EFAULT;
	}
	if (copy_to_user(ubuf + sizeof(hdr), req->wire, req->wire_len)) {
		mutex_unlock(&g_upcall.lock);
		return -EFAULT;
	}

	req->read_ns = ktime_get_ns();
	pr_info("vecfs-embed: read id=%llu nchunks=%u submit_to_read=%lldns\n",
		req->id, req->nchunks,
		(long long)(req->read_ns - req->submit_ns));
	mutex_unlock(&g_upcall.lock);
	return total;
}

static ssize_t vecfs_embed_upcall_write(struct file *filp,
					const char __user *ubuf, size_t count,
					loff_t *ppos)
{
	struct vecfs_embed_response hdr;
	struct vecfs_embed_upcall_req *req;
	u32 *vec = NULL;
	size_t vec_bytes;
	ssize_t ret = 0;

	/* 只有数据通道接受应答；控制通道只读。 */
	if (!(filp->f_mode & FMODE_WRITE))
		return -EBADF;

	if (count < sizeof(hdr))
		return -EINVAL;

	if (copy_from_user(&hdr, ubuf, sizeof(hdr)))
		return -EFAULT;

	/* 失败响应：无载荷。 */
	if (hdr.status != 0) {
		if (count != sizeof(hdr))
			return -EINVAL;
		mutex_lock(&g_upcall.lock);
		req = vecfs_embed_find_locked(hdr.id);
		if (!req) {
			mutex_unlock(&g_upcall.lock);
			return -EINVAL; /* 未知/已超时的 id */
		}
		req->status = hdr.status;
		vecfs_embed_req_unlink_locked(req);
		complete(&req->done);
		mutex_unlock(&g_upcall.lock);
		return count;
	}

	/* 成功响应：nchunks 须与请求一致，载荷 = nchunks × dim × 4。 */
	mutex_lock(&g_upcall.lock);
	req = vecfs_embed_find_locked(hdr.id);
	if (!req || hdr.nchunks != req->nchunks) {
		mutex_unlock(&g_upcall.lock);
		return -EINVAL;
	}
	vec_bytes = (size_t)req->nchunks * req->dim * sizeof(u32);
	mutex_unlock(&g_upcall.lock);

	if (count != sizeof(hdr) + vec_bytes)
		return -EINVAL;

	vec = kvmalloc(vec_bytes, GFP_KERNEL);
	if (!vec)
		return -ENOMEM;
	if (copy_from_user(vec, ubuf + sizeof(hdr), vec_bytes)) {
		ret = -EFAULT;
		goto out;
	}

	mutex_lock(&g_upcall.lock);
	req = vecfs_embed_find_locked(hdr.id);
	if (!req) {
		mutex_unlock(&g_upcall.lock);
		ret = -EINVAL; /* 超时已被 worker 摘除 */
		goto out;
	}
	memcpy(req->out_buf, vec, vec_bytes);
	req->status = 0;
	vecfs_embed_req_unlink_locked(req);
	req->write_ns = ktime_get_ns();
	pr_info("vecfs-embed: write id=%llu read_to_write=%lldns submit_to_write=%lldns\n",
		req->id,
		(long long)(req->write_ns - req->read_ns),
		(long long)(req->write_ns - req->submit_ns));
	complete(&req->done);
	mutex_unlock(&g_upcall.lock);

	ret = count;
out:
	kvfree(vec);
	return ret;
}

static __poll_t vecfs_embed_upcall_poll(struct file *filp, poll_table *wait)
{
	__poll_t mask = 0;

	if (filp->f_mode & FMODE_WRITE) {
		/* 数据通道：有新 PENDING 请求或停止通知时可读。 */
		poll_wait(filp, &g_upcall.wq, wait);
		mutex_lock(&g_upcall.lock);
		if (vecfs_embed_pending_locked() || g_upcall.stopping)
			mask |= EPOLLIN | EPOLLRDNORM;
		mutex_unlock(&g_upcall.lock);
	} else {
		/* 控制通道：有事件或停止通知时可读。 */
		poll_wait(filp, &g_upcall.ctrl_wq, wait);
		mutex_lock(&g_upcall.lock);
		if (vecfs_embed_ctrl_pending_locked() || g_upcall.stopping)
			mask |= EPOLLIN | EPOLLRDNORM;
		mutex_unlock(&g_upcall.lock);
	}
	return mask;
}

static const struct file_operations vecfs_embed_upcall_fops = {
	.owner = THIS_MODULE,
	.open = vecfs_embed_upcall_open,
	.release = vecfs_embed_upcall_release,
	.read = vecfs_embed_upcall_read,
	.write = vecfs_embed_upcall_write,
	.poll = vecfs_embed_upcall_poll,
	.llseek = noop_llseek,
};

static struct miscdevice vecfs_embed_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = VECFS_MOD_NAME "-embed",
	.fops = &vecfs_embed_upcall_fops,
	.mode = 0660,
};

/* ---------------------------------------------------------------------------
 * 模块生命周期
 * ------------------------------------------------------------------------- */

int vecfs_embed_upcall_init(void)
{
	mutex_init(&g_upcall.lock);
	init_waitqueue_head(&g_upcall.wq);
	init_waitqueue_head(&g_upcall.ctrl_wq);
	INIT_LIST_HEAD(&g_upcall.reqs);
	g_upcall.next_id = 0;
	g_upcall.connected = false;
	g_upcall.ctrl_connected = false;
	g_upcall.ctrl_head = 0;
	g_upcall.ctrl_count = 0;
	g_upcall.refcount = 0;
	g_upcall.stopping = false;

	return misc_register(&vecfs_embed_misc);
}

void vecfs_embed_upcall_exit(void)
{
	struct vecfs_embed_upcall_req *r, *tmp;

	/* 结束所有在途请求（兜底；正常流程由 release 处理）。 */
	mutex_lock(&g_upcall.lock);
	g_upcall.connected = false;
	g_upcall.ctrl_connected = false;
	g_upcall.ctrl_head = 0;
	g_upcall.ctrl_count = 0;
	g_upcall.stopping = true;
	list_for_each_entry_safe(r, tmp, &g_upcall.reqs, list) {
		r->status = -EAGAIN;
		vecfs_embed_req_unlink_locked(r);
		complete(&r->done);
	}
	mutex_unlock(&g_upcall.lock);

	misc_deregister(&vecfs_embed_misc);
}
