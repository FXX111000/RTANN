# 模块名：临时改名（如 vecfs2）以与旧残留模块共存，改回只需换这一处并重编译。
MODNAME ?= vecfs2

obj-m += $(MODNAME).o

$(MODNAME)-y := \
	core/vecfs_Indirect.o \
	core/vecfs_ann_indirect.o \
	core/vecfs_main.o \
	core/super_v2.o \
	core/path_cache.o \
	core/shield.o \
	core/embed_upcall.o \
	ann/metadata.o \
	ann/runtime.o \
	ann/label.o \
	ann/info.o \
	ann/maintenance.o \
	ann/stale.o \
	ann/cache.o \
	ann/snapshot.o \
	ann/heap.o \
	ann/visited.o \
	ann/search_context.o \
	ann/search.o \
	ann/select.o \
	ann/hnsw_sched.o \
	ann/build.o \
	ann/insert.o \
	ann/batch.o \
	ann/update_build.o \
	ann/update.o \
	ann/delete.o \
	ann/rebuild.o \
	ann/exchange.o \
	ann/distance/distance.o \
	ann/distance/scalar.o \
	ann/distance/x86.o \
	ann/store/arena.o \
	ann/store/object.o \
	ann/store/scan.o \
	ann/store/validate.o \
	ann/store/rewrite.o \
	interface/ioctl.o \
	interface/path.o \
	interface/tail.o \
	ann/rebuild_async.o



ccflags-y += -I$(src)
ccflags-y += -I$(src)/include
ccflags-y += -DVECFS_MOD_NAME=\"$(MODNAME)\"

VECFS_FP_CFLAGS := \
	-msse2 \
	-mfpmath=sse \
	-fno-tree-vectorize

CFLAGS_REMOVE_ann/rebuild.o := \
	-mno-sse -mno-mmx -mno-80387

CFLAGS_ann/rebuild.o += \
	$(VECFS_FP_CFLAGS)


CFLAGS_REMOVE_ann/batch.o := \
	-mno-sse -mno-mmx -mno-80387

CFLAGS_ann/batch.o += \
	$(VECFS_FP_CFLAGS)

CFLAGS_REMOVE_ann/heap.o := \
	-mno-sse -mno-mmx -mno-80387
CFLAGS_ann/heap.o += $(VECFS_FP_CFLAGS)

CFLAGS_REMOVE_ann/search.o := \
	-mno-sse -mno-mmx -mno-80387
CFLAGS_ann/search.o += $(VECFS_FP_CFLAGS)

CFLAGS_REMOVE_ann/hnsw_sched.o := \
	-mno-sse -mno-mmx -mno-80387
CFLAGS_ann/hnsw_sched.o += $(VECFS_FP_CFLAGS)

CFLAGS_REMOVE_ann/select.o := \
	-mno-sse -mno-mmx -mno-80387
CFLAGS_ann/select.o += $(VECFS_FP_CFLAGS)

CFLAGS_REMOVE_ann/build.o := \
	-mno-sse -mno-mmx -mno-80387
CFLAGS_ann/build.o += $(VECFS_FP_CFLAGS)

CFLAGS_REMOVE_ann/insert.o := \
	-mno-sse -mno-mmx -mno-80387
CFLAGS_ann/insert.o += $(VECFS_FP_CFLAGS)

CFLAGS_REMOVE_ann/update_build.o := \
	-mno-sse -mno-mmx -mno-80387
CFLAGS_ann/update_build.o += $(VECFS_FP_CFLAGS)

CFLAGS_REMOVE_ann/update.o := \
	-mno-sse -mno-mmx -mno-80387
CFLAGS_ann/update.o += $(VECFS_FP_CFLAGS)

CFLAGS_REMOVE_ann/distance/scalar.o := \
	-mno-sse -mno-mmx -mno-80387
CFLAGS_ann/distance/scalar.o += $(VECFS_FP_CFLAGS)

CFLAGS_REMOVE_ann/distance/distance.o := \
	-mno-sse -mno-mmx -mno-80387
CFLAGS_ann/distance/distance.o += $(VECFS_FP_CFLAGS)

CFLAGS_REMOVE_ann/distance/x86.o := \
	-mno-sse -mno-mmx -mno-80387
CFLAGS_ann/distance/x86.o += \
	-mavx2 \
	-mfpmath=sse \
	-fno-tree-vectorize

# 构建树：跟随运行中内核（/lib/modules/$(uname -r)/build 软链）。当前运行内核为
# -shield-computon-v3，其构建树在 /usr/src/linux-headers-6.1.183-shield-computon-v3。
# 旧的 /home/fxx/linux-6.1.183 是无后缀 6.1.183 的树，vermagic 不匹配当前内核会
# insmod 报 Invalid module format。KDIR 仍可命令行覆盖。
KDIR ?= /lib/modules/$(shell uname -r)/build

.PHONY: all clean tools

all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

tools:
	$(MAKE) -C tools

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
