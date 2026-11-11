# VecFS · RTANN

VecFS is a **vector filesystem that runs inside the Linux kernel**. On top of the
usual filesystem facilities it maintains, for every formatted block device, a single
global **HNSW** (Hierarchical Navigable Small World) secondary index used for
similarity search. The repository also contains **RTANN**, a real-time scheduling
framework inside the kernel that shapes query execution under bursty, SLO-bound
arrival traffic.

This repository is a self-contained release containing:

1. the **filesystem source** (kernel module),
2. the **formatting / import toolchain** (userspace tools),
3. the **RTANN benchmark scripts** (userspace load generators).

> **VecFS is experimental.** Use it only in a VM, on a loop device, or on dedicated
> test hardware. Do not use it for important data.

---

## 1. Overview

Per formatted device, VecFS keeps **one global ANN index**. The index configuration is
fixed at format time and includes:

- vector dimension and element type;
- distance type;
- `M` (upper-level degree), `M0` (level-0 degree);
- maximum number of levels;
- `ef_construction`;
- default `ef_search`;
- random seed.

The main tested UAPI paths are **FP32 + L2** and **FP32 + Inner Product (IP)**, with
distance defined as:

```
L2:  sum((a[i] - b[i])^2)
IP:  1 - dot(a, b)
```

Each persisted ANN node carries an external ID, owner information, a vector ordinal, a
content range, its HNSW level, level-0 and upper-level neighbours, the vector payload, a
CRC and an object state (`PREPARED`, `LIVE`, `DELETED`, `CORRUPT`).

The system supports normal file/directory operations, standalone vectors, file↔vector
binding, content-range association, online insert/update/delete/recovery, ANN query with
filtering and path return, validation/rebuild, and HNSW graph import/export.

---

## 2. Repository layout

```
Makefile                 kernel-module build (MODNAME ?= vecfs2)
Kconfig                  kernel configuration fragment
vecfs_compat.h           kernel-version compatibility shims
core/                    filesystem core, superblock, path cache, SHIELD, embed upcall
ann/                     HNSW index: search, insert/update/delete, rebuild, batch, …
  ann/distance/          distance kernels (scalar + x86 SIMD)
  ann/store/             on-disk object arena, scan, validation, rewrite
  ann/hnsw_sched.c       RTANN scheduler
include/                 UAPI + internal headers (incl. vecfs_ioctl.h)
interface/               VFS entry points: ioctl, path, tail protocol
tools/                   userspace toolchain (mkfs, import/export, fsck, converter)
tests/rtann/             RTANN benchmark + test scripts (see §7)
```

---

## 3. Requirements

- Linux kernel source/headers for the running kernel (`/lib/modules/$(uname -r)/build`),
  a C toolchain, and `make`.
- A C++17 compiler for `hnswlib-converter`.
- `libcurl` **development headers** for `vecfs-embed-helper` (only needed if you build
  that tool; see §4.2 for an override when headers are unavailable).
- Python 3 with `numpy`, `h5py` for the benchmark scripts; `hnswlib` is required only
  for the *baseline* load generator (`hnswlib_mmpp.py`) and the graph builder.

---

## 4. Build

### 4.1 Kernel module

```bash
make            # builds vecfs2.ko (see MODNAME below)
make clean
```

The module name is set by `MODNAME ?= vecfs2` at the top of the `Makefile`. It is
`vecfs2` by default so it can coexist with an older, still-loaded `vecfs` module on the
same machine. Rename it by editing that single line and rebuilding; the filesystem
mount type and the `import_hnsw.sh` helper both follow `MODNAME` automatically.

Load it with:

```bash
sudo insmod vecfs2.ko
```

### 4.2 Userspace tools

```bash
make -C tools
```

This builds `mkfs.vecfs`, `vecfs-index`, `hnswlib-converter`, `fsck.vecfs`, and
`vecfs-embed-helper`.

`vecfs-embed-helper` links against `libcurl`. If the system `libcurl` development headers
are not installed but a runtime library is present (for example, conda headers plus the
system `.so`), override the flags:

```bash
make -C tools \
  CURL_CFLAGS="-I/opt/anaconda3/include" \
  CURL_LIBS="-L/usr/lib/x86_64-linux-gnu -l:libcurl.so.4"
```

If you do not need embedding, you can skip building that one tool; the others build
without libcurl.

---

## 5. Quick start

### 5.1 Format and mount a device

```bash
sudo insmod vecfs2.ko

sudo tools/mkfs.vecfs \
    --dim 96 --distance l2 --m 16 --m0 32 --levels 16 \
    --ef-construction 400 --ef-search 200 \
    /dev/nvme1n1p2

sudo mkdir -p /mnt/vecfs
sudo mount -t vecfs2 /dev/nvme1n1p2 /mnt/vecfs
```

`mkfs.vecfs` options:

| option | meaning | default |
| --- | --- | --- |
| `--dim N` | vector dimension | 96 |
| `--distance l2\|ip` | distance type | l2 |
| `--m N` | upper-level degree M | 16 |
| `--m0 N` | level-0 degree M0 | 32 |
| `--levels N` | maximum number of levels | 16 |
| `--ef-construction N` | ef for construction | 200 |
| `--ef-search N` | default ef for search | 64 |

> **WARNING:** `mkfs.vecfs` irreversibly reformats the target device.

### 5.2 Import an hnswlib index (one step)

`tests/rtann/import_hnsw.sh` reads an hnswlib `.bin` index header, formats the device,
mounts it, converts the graph to VecFS exchange format, and imports it:

```bash
./tests/rtann/import_hnsw.sh \
    data/gist-960-euclidean-M16-M0-32-efc400.bin \
    /dev/nvme1n1p2 l2 /mnt/vecfs-gist
```

```
usage: ./import_hnsw.sh <bin-file> <device> [ip|l2] [mountpoint]
```

`dim`, `M`, `M0`, `ef_construction` and the level count are read from the `.bin` header.
The distance type is not stored in the header, so it is taken from the third argument
(or guessed from the file name: `gist`/`euclid` → `l2`, `deep`/`angular`/`cosine` → `ip`).
Set `YES=1` to skip the interactive confirmation.

### 5.3 Manual conversion + import

```bash
tools/hnswlib-converter from-hnsw \
    --distance l2 --max-levels 16 --ef-search 200 \
    index.bin index.exchange

tools/vecfs-index info index.exchange
tools/vecfs-index import index.exchange /mnt/vecfs/control
```

---

## 6. Query interface

VecFS exposes ANN search through an **ioctl** on a mounted file descriptor
(magic `0xB7`, command `0x31`, `VECFS_KNN_SEARCH_BATCH`). The request struct is:

```c
struct vecfs_hnsw_search_req {
    __u64 queries_ptr;      /* in:  FP32 query vectors, n_queries × dim */
    __u32 dim;
    __u32 k;
    __u32 ef_search;
    __u32 n_queries;
    __u32 sched_mode;       /* RTANN mode, 0..4 (see §7) */
    __u32 flags;
    __u64 results_ptr;      /* out: node IDs, n_queries × k */
    __u64 distances_ptr;    /* out: distances, n_queries × k */
    __u64 dist_counts_ptr;  /* out: distance computations per query */
    __u64 trace_ptr;        /* optional per-query RTANN trace */
    __u64 reserved[2];
};
```

`tests/rtann/vecfs_mmpp.py` is a complete reference implementation of this encoding
(see `VecfsHnswSearchReq`, `_IOWR`, and `ioctl_search_batch`); `tests/rtann/test_ioctl.py`
is a small standalone smoke test for the same interface.

---

## 7. RTANN scheduler

The kernel scheduler (`ann/hnsw_sched.c`) supports five modes, selected per request via
`sched_mode`:

| mode | name | behaviour |
| --- | --- | --- |
| 0 | `OFF` | bare parallel HNSW, no scheduling |
| 1 | `MICRO` | micro-batched execution |
| 2 | `SHAPING` | arrival shaping |
| 3 | `BOOST` | victim compensation on the real-time budget |
| 4 | `FULL` | all mechanisms combined |

Under **light load** all modes overlap (mode 4 ≈ mode 0). The scheduler's benefit appears
in the **contention knee** under bursty (MMPP) load, where it protects tail latency against
the arrival-driven throughput that a bare parallel HNSW would otherwise chase.

---

## 8. Benchmarks (`tests/rtann/`)

Two matched MMPP load generators measure the same SLO metrics so the in-kernel index can
be compared apples-to-apples against an out-of-the-box `hnswlib` index.

### 8.1 hnswlib baseline

```bash
python3 tests/rtann/hnswlib_mmpp.py \
    --index data/gist-960-euclidean-M16-M0-32-efc400.bin \
    --hdf5  data/gist-960-euclidean.hdf5 \
    --space l2 --dim 960 --no_norm \
    --mmpp_scale 0.2
```

### 8.2 VecFS (ioctl)

```bash
python3 tests/rtann/vecfs_mmpp.py \
    --dev_path /mnt/vecfs-gist \
    --hdf5 data/gist-960-euclidean.hdf5 \
    --dim 960 --no_norm \
    --sched-mode 4 \
    --mmpp_scale 0.2
```

Note: `vecfs_mmpp.py` L2-normalises queries **by default** (deep-image/IP convention);
pass `--no_norm` for Euclidean/L2 datasets such as gist. `hnswlib_mmpp.py` matches this
with the same `--no_norm` flag and requires `--space` to equal the index's space.

### 8.3 MMPP + SLO parameters (both scripts)

| flag | meaning | default |
| --- | --- | --- |
| `--mmpp_lambda_low` | low-load arrival rate (QPS) | 500 |
| `--mmpp_lambda_high` | high-load arrival rate (QPS) | 4000 |
| `--mmpp_scale` | scales both rates by a factor | 1.0 |
| `--mmpp_q01` | state 0→1 transition rate (1/s) | 0.5 |
| `--mmpp_q10` | state 1→0 transition rate (1/s) | 1.0 |
| `--mmpp_window_ms` | micro-batch window (ms) | 10 |
| `--mmpp_burnin_s` | burn-in before recording | 10 |
| `--deadline_ms` | primary deadline for state breakdown | 50 |
| `--deadline_list_ms` | deadlines reported together | `30,50,100` |
| `--slo_timeout_rate` | allowed timeout rate (SLO) | 0.001 |
| `--max_queries` | target query count | 100000 |
| `--warmup` | warm-up queries (excluded from stats) | 256 |
| `--k` / `--ef` | top-k / ef_search | 10 / 200 |
| `--save_json` | dump a summary JSON | off |

A query violates its SLO when its latency exceeds `deadline_ms`; the load is considered
in-SLO when the timeout rate stays below `slo_timeout_rate` (0.1% by default).

### 8.4 Other scripts

- `build_hnsw_graph.py` — build an hnswlib `.bin` graph from an HDF5 dataset
  (prerequisite for the baseline and for import).
- `hnsw_test.py` — basic hnswlib throughput/tail-latency test.
- `test_ioctl.py` — standalone VecFS ioctl smoke test.
- `hnswlib_mmpp_deadline.py` — hnswlib baseline with per-query deadlines.
- `hnswlib_mmpp_locality.py` / `vecfs_mmpp_locality.py` — locality experiments.
- `hnswlib_mmpp_mba.py` — Intel MBA (memory-bandwidth allocation) counter-experiment,
  showing that statically partitioning bandwidth does **not** remove tail-latency jitter.

---

## 9. Datasets

The load generators read queries from an HDF5 file with dataset keys such as `train`,
`test`, and `base` (the scripts search common key names automatically). Examples used in
the paper are `deep-image-96-angular.hdf5` (96-dim, IP) and `gist-960-euclidean.hdf5`
(960-dim, L2). Place them anywhere and point `--hdf5` (and `--index`, for the baseline)
at them.

---

## 10. Notes

- The module mounts as `-t vecfs2` by default (see §4.1).
- `mkfs.vecfs` reformats the device; `import_hnsw.sh` prints a summary and asks for
  confirmation first.
- The search ioctl is reachable by opening any file inside the mount (or the mount point
  itself) and issuing the ioctl; no privileged capability is required in the current
  interface.

---

## 11. License

No license has been chosen yet for this repository.
