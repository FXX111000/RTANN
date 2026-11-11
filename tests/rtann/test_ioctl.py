#!/usr/bin/env python3
import os
import time
import queue
import fcntl
import ctypes
import argparse
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict, Any

import h5py
import numpy as np


DEFAULT_DEV_PATH = "tests/mount_point"
DEV_PATH = DEFAULT_DEV_PATH


# ============================================================
# ioctl encoding
# ============================================================

def _IOWR(type_, nr, size):
    IOC_NRBITS = 8
    IOC_TYPEBITS = 8
    IOC_SIZEBITS = 14
    IOC_DIRBITS = 2

    IOC_NRSHIFT = 0
    IOC_TYPESHIFT = IOC_NRSHIFT + IOC_NRBITS
    IOC_SIZESHIFT = IOC_TYPESHIFT + IOC_TYPEBITS
    IOC_DIRSHIFT = IOC_SIZESHIFT + IOC_SIZEBITS

    IOC_WRITE = 1
    IOC_READ = 2
    dir_ = IOC_READ | IOC_WRITE

    return ((dir_ << IOC_DIRSHIFT) |
            (type_ << IOC_TYPESHIFT) |
            (nr << IOC_NRSHIFT) |
            (size << IOC_SIZESHIFT))


VECFS_IOCTL_MAGIC = 0xB7

# RTANN 调度器模式（与 include/vecfs_ioctl.h 的 VECFS_SCHED_MODE_* 对齐）
SCHED_MODE_OFF = 0
SCHED_MODE_MICRO = 1
SCHED_MODE_SHAPING = 2
SCHED_MODE_BOOST = 3
SCHED_MODE_FULL = 4


class VecfsHnswTraceUser(ctypes.Structure):
    _fields_ = [
        ("qid", ctypes.c_uint64),

        ("total_us", ctypes.c_uint64),
        ("backlog_wait_us", ctypes.c_uint64),
        ("queue_wait_us", ctypes.c_uint64),
        ("run_us", ctypes.c_uint64),
        ("ctx_wait_us", ctypes.c_uint64),
        ("rescue_wait_us", ctypes.c_uint64),
        ("normal_wait_us", ctypes.c_uint64),
        ("boost_wait_us", ctypes.c_uint64),
        ("first_run_wait_us", ctypes.c_uint64),

        ("dispatches", ctypes.c_uint32),
        ("requeues", ctypes.c_uint32),
        ("ctx_misses", ctypes.c_uint32),
        ("zero_progress_batches", ctypes.c_uint32),

        ("wave_id_submit", ctypes.c_uint32),
        ("wave_id_admit", ctypes.c_uint32),
        ("wave_id_finish", ctypes.c_uint32),

        ("victim_entries", ctypes.c_uint32),
        ("victim_exits", ctypes.c_uint32),
        ("rescue_entries_seen", ctypes.c_uint32),

        ("progress_units", ctypes.c_uint64),
        ("dist_compute_cnt", ctypes.c_uint32),
        ("tail_reason", ctypes.c_uint32),

        ("avg_rate_q16", ctypes.c_uint64),
        ("min_rate_q16", ctypes.c_uint64),
        ("max_rate_q16", ctypes.c_uint64),
    ]


class VecfsHnswSearchReq(ctypes.Structure):
    _fields_ = [
        ("queries_ptr", ctypes.c_uint64),
        ("dim", ctypes.c_uint32),
        ("k", ctypes.c_uint32),
        ("ef_search", ctypes.c_uint32),
        ("n_queries", ctypes.c_uint32),
        ("sched_mode", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("results_ptr", ctypes.c_uint64),
        ("distances_ptr", ctypes.c_uint64),
        ("dist_counts_ptr", ctypes.c_uint64),
        ("trace_ptr", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 2),
    ]


IOCTL_SEARCH_BATCH = _IOWR(
    VECFS_IOCTL_MAGIC,
    0x31,
    ctypes.sizeof(VecfsHnswSearchReq),
)


TAIL_REASON_NAMES = {
    0: "NONE",
    1: "QUEUE_WAIT",
    2: "FIRST_RUN_WAIT",
    3: "RESCUE_WAIT",
    4: "WAVE_WAIT",
    5: "RUN_TIME",
    6: "WORKLOAD",
    7: "LOW_RATE",
    8: "CTX_WAIT",
    9: "FRAGMENTATION",
}


# ============================================================
# data loading
# ============================================================

def find_dataset(f: h5py.File, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in f:
            return name
    return None


def load_deep_image_hdf5(hdf5_path: str,
                         max_queries: Optional[int] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(hdf5_path, 'r') as f:
        test_key = find_dataset(f, ['test', 'query', 'queries'])
        neighbors_key = find_dataset(f, ['neighbors', 'knn', 'groundtruth', 'true_neighbors'])

        if test_key is None or neighbors_key is None:
            raise RuntimeError("未在 HDF5 中找到 test/queries 或 neighbors 数据集")

        queries = f[test_key]
        neighbors = f[neighbors_key]

        if max_queries is not None:
            queries = queries[:max_queries]
            neighbors = neighbors[:max_queries]

        queries = np.asarray(queries, dtype=np.float32)
        neighbors = np.asarray(neighbors, dtype=np.int64)

        # dim 不再硬编码（deep-image=96, gist=960）；查询维度由 ioctl 按 query shape 透传，
        # 与设备 mkfs 的 --dim 不匹配时由内核返回错误。
        return queries, neighbors


def l2_normalize_queries_fp32(queries: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(queries, axis=1, keepdims=True).astype(np.float32)
    norms[norms == 0] = 1e-12
    q = queries / norms
    return np.ascontiguousarray(q.astype(np.float32, copy=False))


# ============================================================
# metrics
# ============================================================

def compute_recall_at_k(pred_ids, true_ids, k: int, id_offset: int = 1) -> float:
    pred = set(int(x) for x in pred_ids[:k])
    gt = set(int(x) + id_offset for x in true_ids[:k])
    if not gt:
        return 0.0
    return len(pred & gt) / float(k)


def calculate_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "avg": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "avg": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def print_stats_ms(title: str, values_ms: List[float]):
    s = calculate_stats(values_ms)
    print(f"\n[{title}] 延迟统计（ms）:")
    print(f"  avg = {s['avg']:.3f}")
    print(f"  p50 = {s['p50']:.3f}")
    print(f"  p90 = {s['p90']:.3f}")
    print(f"  p95 = {s['p95']:.3f}")
    print(f"  p99 = {s['p99']:.3f}")
    print(f"  min = {s['min']:.3f}")
    print(f"  max = {s['max']:.3f}")


def print_stats_num(title: str, values: List[float]):
    s = calculate_stats(values)
    print(f"\n[{title}] 统计:")
    print(f"  avg = {s['avg']:.2f}")
    print(f"  p50 = {s['p50']:.2f}")
    print(f"  p90 = {s['p90']:.2f}")
    print(f"  p95 = {s['p95']:.2f}")
    print(f"  p99 = {s['p99']:.2f}")
    print(f"  min = {s['min']:.0f}")
    print(f"  max = {s['max']:.0f}")


def q16_to_float(x: int) -> float:
    return float(x) / 65536.0


# ============================================================
# trace helpers
# ============================================================

def trace_to_dict(t: VecfsHnswTraceUser) -> Dict[str, Any]:
    return {
        "qid": int(t.qid),

        "total_us": int(t.total_us),
        "backlog_wait_us": int(t.backlog_wait_us),
        "queue_wait_us": int(t.queue_wait_us),
        "run_us": int(t.run_us),
        "ctx_wait_us": int(t.ctx_wait_us),
        "rescue_wait_us": int(t.rescue_wait_us),
        "normal_wait_us": int(t.normal_wait_us),
        "boost_wait_us": int(t.boost_wait_us),
        "first_run_wait_us": int(t.first_run_wait_us),

        "dispatches": int(t.dispatches),
        "requeues": int(t.requeues),
        "ctx_misses": int(t.ctx_misses),
        "zero_progress_batches": int(t.zero_progress_batches),

        "wave_id_submit": int(t.wave_id_submit),
        "wave_id_admit": int(t.wave_id_admit),
        "wave_id_finish": int(t.wave_id_finish),

        "victim_entries": int(t.victim_entries),
        "victim_exits": int(t.victim_exits),
        "rescue_entries_seen": int(t.rescue_entries_seen),

        "progress_units": int(t.progress_units),
        "dist_compute_cnt": int(t.dist_compute_cnt),
        "tail_reason": int(t.tail_reason),
        "tail_reason_name": TAIL_REASON_NAMES.get(int(t.tail_reason), f"UNKNOWN({int(t.tail_reason)})"),

        "avg_rate_q16": int(t.avg_rate_q16),
        "min_rate_q16": int(t.min_rate_q16),
        "max_rate_q16": int(t.max_rate_q16),
    }


def summarize_trace_records(records: List[Dict[str, Any]]):
    traces = [r.get("trace") for r in records if r.get("trace") is not None]
    if not traces:
        return

    reason_counter = Counter(t["tail_reason_name"] for t in traces)

    print("\n[trace] tail_reason 分布:")
    for reason, cnt in reason_counter.most_common():
        print(f"  {reason:18s}: {cnt}")

    total_us = [t["total_us"] for t in traces]
    backlog_us = [t["backlog_wait_us"] for t in traces]
    queue_us = [t["queue_wait_us"] for t in traces]
    run_us = [t["run_us"] for t in traces]
    rescue_us = [t["rescue_wait_us"] for t in traces]
    normal_us = [t["normal_wait_us"] for t in traces]
    boost_us = [t["boost_wait_us"] for t in traces]
    progress = [t["progress_units"] for t in traces]
    dist = [t["dist_compute_cnt"] for t in traces]
    dispatches = [t["dispatches"] for t in traces]

    print_stats_ms("trace.total", [x / 1000.0 for x in total_us])
    print_stats_ms("trace.backlog_wait", [x / 1000.0 for x in backlog_us])
    print_stats_ms("trace.queue_wait", [x / 1000.0 for x in queue_us])
    print_stats_ms("trace.run", [x / 1000.0 for x in run_us])
    print_stats_ms("trace.rescue_wait", [x / 1000.0 for x in rescue_us])
    print_stats_ms("trace.normal_wait", [x / 1000.0 for x in normal_us])
    print_stats_ms("trace.boost_wait", [x / 1000.0 for x in boost_us])

    print_stats_num("trace.progress_units / 步数", progress)
    print_stats_num("trace.dist_compute_cnt / 距离计算量", dist)
    print_stats_num("trace.dispatches", dispatches)


def record_latency_ms(record: Dict[str, Any], latency_source: str = "auto") -> float:
    tr = record.get("trace")

    if latency_source == "kernel" and tr is not None:
        return tr["total_us"] / 1000.0

    if latency_source == "user":
        return record["latency_ms"]

    # auto
    if tr is not None:
        return tr["total_us"] / 1000.0

    return record["latency_ms"]


def select_near_percentile(records: List[Dict[str, Any]],
                           percentile: float,
                           count: int,
                           latency_source: str = "auto") -> List[Dict[str, Any]]:
    if not records:
        return []

    sorted_records = sorted(records, key=lambda r: record_latency_ms(r, latency_source))
    n = len(sorted_records)

    center = int(round((percentile / 100.0) * (n - 1)))
    half = count // 2
    start = max(0, center - half)
    end = min(n, start + count)

    if end - start < count:
        start = max(0, end - count)

    return sorted_records[start:end]


def select_worst(records: List[Dict[str, Any]],
                 count: int,
                 latency_source: str = "auto") -> List[Dict[str, Any]]:
    return sorted(records, key=lambda r: record_latency_ms(r, latency_source), reverse=True)[:count]


def print_record_table(title: str,
                       records: List[Dict[str, Any]],
                       latency_source: str = "auto"):
    print(f"\n[{title}] 共 {len(records)} 条")
    if not records:
        return

    header = (
        f"{'rank':>4} {'idx':>8} {'lat(ms)':>10} {'user(ms)':>10} "
        f"{'dist':>9} {'steps':>9} {'disp':>6} {'reason':>16} "
        f"{'run(ms)':>9} {'qwait(ms)':>10} {'backlog(ms)':>11} "
        f"{'rescue(ms)':>11} {'victim':>6} {'rate(avg)':>10}"
    )
    print(header)
    print("-" * len(header))

    for rank, r in enumerate(records):
        tr = r.get("trace")
        lat = record_latency_ms(r, latency_source)
        user_lat = r.get("latency_ms", 0.0)
        idx = r.get("idx", -1)
        dist = int(r.get("dist_count", 0))

        if tr is None:
            print(
                f"{rank:4d} {idx:8d} {lat:10.3f} {user_lat:10.3f} "
                f"{dist:9d} {'-':>9} {'-':>6} {'NO_TRACE':>16} "
                f"{'-':>9} {'-':>10} {'-':>11} {'-':>11} {'-':>6} {'-':>10}"
            )
        else:
            steps = int(tr["progress_units"])
            disp = int(tr["dispatches"])
            reason = tr["tail_reason_name"]
            run_ms = tr["run_us"] / 1000.0
            qwait_ms = tr["queue_wait_us"] / 1000.0
            backlog_ms = tr["backlog_wait_us"] / 1000.0
            rescue_ms = tr["rescue_wait_us"] / 1000.0
            victim = int(tr["victim_entries"])
            avg_rate = q16_to_float(tr["avg_rate_q16"])

            print(
                f"{rank:4d} {idx:8d} {lat:10.3f} {user_lat:10.3f} "
                f"{dist:9d} {steps:9d} {disp:6d} {reason:>16} "
                f"{run_ms:9.3f} {qwait_ms:10.3f} {backlog_ms:11.3f} "
                f"{rescue_ms:11.3f} {victim:6d} {avg_rate:10.2f}"
            )


def print_filtered_records(records: List[Dict[str, Any]],
                           sample_count: int,
                           latency_source: str = "auto"):
    if not records:
        return

    p50_group = select_near_percentile(records, 50, sample_count, latency_source)
    worst_group = select_worst(records, sample_count, latency_source)

    print_record_table("P50 附近样本", p50_group, latency_source)
    print_record_table("最差尾部样本", worst_group, latency_source)


# ============================================================
# ioctl wrapper
# ============================================================

def ioctl_search_batch(fd: int,
                       queries: np.ndarray,
                       k: int,
                       ef: int,
                       sched_mode: int = SCHED_MODE_OFF,
                       want_trace: bool = False
                       ) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict[str, Any]]]]:
    """
    returns:
      results: [n_queries, k] uint32
      dist_counts: [n_queries] uint64
      traces: Optional[List[dict]]
    """
    if queries.dtype != np.float32:
        queries = queries.astype(np.float32, copy=False)

    if not queries.flags['C_CONTIGUOUS']:
        queries = np.ascontiguousarray(queries, dtype=np.float32)

    n_queries, dim = queries.shape

    results = np.zeros((n_queries, k), dtype=np.uint32)
    dist_counts = np.zeros((n_queries,), dtype=np.uint64)

    if not results.flags['C_CONTIGUOUS']:
        results = np.ascontiguousarray(results, dtype=np.uint32)

    if not dist_counts.flags['C_CONTIGUOUS']:
        dist_counts = np.ascontiguousarray(dist_counts, dtype=np.uint64)

    trace_arr = None
    trace_ptr = 0

    if want_trace:
        TraceArrayT = VecfsHnswTraceUser * n_queries
        trace_arr = TraceArrayT()
        trace_ptr = ctypes.addressof(trace_arr)

    req = VecfsHnswSearchReq()
    req.dim = int(dim)
    req.k = int(k)
    req.ef_search = int(ef)
    req.n_queries = int(n_queries)
    req.sched_mode = int(sched_mode)
    req.flags = 0
    req.queries_ptr = int(queries.ctypes.data)
    req.results_ptr = int(results.ctypes.data)
    req.distances_ptr = 0
    req.dist_counts_ptr = int(dist_counts.ctypes.data)
    req.trace_ptr = int(trace_ptr)

    fcntl.ioctl(fd, IOCTL_SEARCH_BATCH, req)

    traces = None
    if trace_arr is not None:
        traces = [trace_to_dict(trace_arr[i]) for i in range(n_queries)]

    return results, dist_counts, traces


def ioctl_search_single(fd: int,
                        query: np.ndarray,
                        k: int,
                        ef: int,
                        sched_mode: int = SCHED_MODE_OFF,
                        want_trace: bool = False
                        ) -> Tuple[List[int], int, Optional[Dict[str, Any]]]:
    q2 = query.reshape(1, -1).astype(np.float32, copy=False)
    q2 = np.ascontiguousarray(q2, dtype=np.float32)

    res, dist_counts, traces = ioctl_search_batch(
        fd,
        q2,
        k,
        ef,
        sched_mode=sched_mode,
        want_trace=want_trace,
    )

    trace = traces[0] if traces else None
    return res[0].tolist(), int(dist_counts[0]), trace


# ============================================================
# evaluation helpers
# ============================================================

def prepare_queries(hdf5_path: str,
                    max_queries: Optional[int],
                    no_norm: bool) -> Tuple[np.ndarray, np.ndarray]:
    print(f"[load] 加载数据集: {hdf5_path}")
    queries, neighbors = load_deep_image_hdf5(hdf5_path, max_queries=max_queries)
    print(f"[load] queries={queries.shape}, neighbors={neighbors.shape}")

    if no_norm:
        print("[load] 不进行 L2 归一化")
        q_fp32 = np.ascontiguousarray(queries.astype(np.float32, copy=False))
    else:
        print("[load] L2 归一化查询")
        q_fp32 = l2_normalize_queries_fp32(queries)

    return q_fp32, neighbors


def finalize_report(tag: str,
                    records: List[Dict[str, Any]],
                    recalls_sum: Dict[int, float],
                    k_list: List[int],
                    total_elapsed_s: float,
                    sample_count: int,
                    latency_source: str):
    n = len(records)
    if n == 0:
        print(f"[{tag}] 没有记录")
        return

    latency_ms = [record_latency_ms(r, latency_source) for r in records]
    user_latency_ms = [r["latency_ms"] for r in records]
    dist_counts = [int(r["dist_count"]) for r in records]

    print(f"\n[{tag}] 完成: n={n}, elapsed={total_elapsed_s:.3f}s, QPS={n / total_elapsed_s:.2f}")

    print("\n[recall]")
    for k in k_list:
        print(f"  recall@{k} = {recalls_sum[k] / n:.6f}")

    print_stats_ms(f"{tag}.latency_selected({latency_source})", latency_ms)
    print_stats_ms(f"{tag}.latency_user", user_latency_ms)
    print_stats_num(f"{tag}.dist_compute_cnt / 计算量", dist_counts)

    summarize_trace_records(records)

    print_filtered_records(
        records,
        sample_count=sample_count,
        latency_source=latency_source,
    )


# ============================================================
# modes
# ============================================================

def evaluate_single(hdf5_path: str,
                    k_list: List[int],
                    ef: int,
                    max_queries: Optional[int],
                    id_offset: int,
                    qps_print_every: int,
                    no_norm: bool,
                    sched_mode: int,
                    trace: bool,
                    sample_count: int,
                    latency_source: str):
    q_fp32, neighbors = prepare_queries(hdf5_path, max_queries, no_norm)

    max_k = max(k_list)
    if neighbors.shape[1] < max_k:
        raise RuntimeError(f"真值近邻数量不足以评估 k={max_k}")

    if not os.path.exists(DEV_PATH):
        raise RuntimeError(f"未找到 {DEV_PATH}")

    if trace and sched_mode == SCHED_MODE_OFF:
        print("[warn] --trace 在 sched_mode=OFF 下仍会返回 trace；基线也计入轨迹")

    fd = os.open(DEV_PATH, os.O_RDONLY)
    records = []
    recalls_sum = {k: 0.0 for k in k_list}

    print(f"[single] n={len(q_fp32)}, ef={ef}, k={max_k}, sched_mode={sched_mode}, trace={trace}")

    t_all0 = time.perf_counter()
    try:
        for i in range(len(q_fp32)):
            t0 = time.perf_counter_ns()
            pred, dc, tr = ioctl_search_single(
                fd,
                q_fp32[i],
                max_k,
                ef,
                sched_mode=sched_mode,
                want_trace=trace,
            )
            t1 = time.perf_counter_ns()

            for k in k_list:
                recalls_sum[k] += compute_recall_at_k(pred, neighbors[i], k, id_offset)

            records.append({
                "idx": i,
                "pred_ids": pred,
                "dist_count": dc,
                "latency_ms": (t1 - t0) / 1e6,
                "trace": tr,
            })

            if qps_print_every > 0 and ((i + 1) % qps_print_every == 0 or i + 1 == len(q_fp32)):
                elapsed = time.perf_counter() - t_all0
                print(f"[single {i + 1}/{len(q_fp32)}] QPS={((i + 1) / elapsed):.2f}")
    finally:
        os.close(fd)

    total_elapsed = time.perf_counter() - t_all0
    finalize_report(
        "single",
        records,
        recalls_sum,
        k_list,
        total_elapsed,
        sample_count,
        latency_source,
    )


def evaluate_batch(hdf5_path: str,
                   k_list: List[int],
                   ef: int,
                   max_queries: Optional[int],
                   id_offset: int,
                   qps_print_every: int,
                   no_norm: bool,
                   batch_size: int,
                   sched_mode: int,
                   trace: bool,
                   sample_count: int,
                   latency_source: str):
    q_fp32, neighbors = prepare_queries(hdf5_path, max_queries, no_norm)

    max_k = max(k_list)
    if neighbors.shape[1] < max_k:
        raise RuntimeError(f"真值近邻数量不足以评估 k={max_k}")

    if not os.path.exists(DEV_PATH):
        raise RuntimeError(f"未找到 {DEV_PATH}")

    if trace and sched_mode == SCHED_MODE_OFF:
        print("[warn] --trace 在 sched_mode=OFF 下仍会返回 trace；基线也计入轨迹")

    fd = os.open(DEV_PATH, os.O_RDONLY)
    records = []
    recalls_sum = {k: 0.0 for k in k_list}

    n = len(q_fp32)
    num_batches = (n + batch_size - 1) // batch_size

    print(f"[batch] n={n}, batch_size={batch_size}, ef={ef}, k={max_k}, sched_mode={sched_mode}, trace={trace}")

    t_all0 = time.perf_counter()
    try:
        processed = 0

        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, n)
            cur_n = end - start

            batch_q = q_fp32[start:end]
            batch_neighbors = neighbors[start:end]

            t0 = time.perf_counter_ns()
            batch_res, batch_dc, batch_traces = ioctl_search_batch(
                fd,
                batch_q,
                max_k,
                ef,
                sched_mode=sched_mode,
                want_trace=trace,
            )
            t1 = time.perf_counter_ns()

            batch_latency_ms = (t1 - t0) / 1e6

            for j in range(cur_n):
                global_idx = start + j
                pred = batch_res[j].tolist()
                dc = int(batch_dc[j])
                tr = batch_traces[j] if batch_traces else None

                for k in k_list:
                    recalls_sum[k] += compute_recall_at_k(pred, batch_neighbors[j], k, id_offset)

                records.append({
                    "idx": global_idx,
                    "pred_ids": pred,
                    "dist_count": dc,
                    # no per-query user latency in a batch; every query observes batch completion
                    "latency_ms": batch_latency_ms,
                    "batch_id": b,
                    "trace": tr,
                })

            processed += cur_n

            if qps_print_every > 0 and (processed % qps_print_every < cur_n or processed == n):
                elapsed = time.perf_counter() - t_all0
                print(f"[batch {processed}/{n}] QPS={processed / elapsed:.2f}, last_batch={batch_latency_ms:.3f}ms")
    finally:
        os.close(fd)

    total_elapsed = time.perf_counter() - t_all0
    finalize_report(
        "batch",
        records,
        recalls_sum,
        k_list,
        total_elapsed,
        sample_count,
        latency_source,
    )


def evaluate_fixed_concurrent(hdf5_path: str,
                              k_list: List[int],
                              ef: int,
                              max_queries: Optional[int],
                              id_offset: int,
                              qps_print_every: int,
                              no_norm: bool,
                              concurrency: int,
                              sched_mode: int,
                              trace: bool,
                              sample_count: int,
                              latency_source: str,
                              pin_all_queries: bool = False,
                              warmup: int = 0):
    q_fp32, neighbors = prepare_queries(hdf5_path, max_queries, no_norm)

    max_k = max(k_list)
    if neighbors.shape[1] < max_k:
        raise RuntimeError(f"真值近邻数量不足以评估 k={max_k}")

    if not os.path.exists(DEV_PATH):
        raise RuntimeError(f"未找到 {DEV_PATH}")

    if trace and sched_mode == SCHED_MODE_OFF:
        print("[warn] --trace 在 sched_mode=OFF 下仍会返回 trace；基线也计入轨迹")

    n = len(q_fp32)
    concurrency = max(1, min(concurrency, n))

    if pin_all_queries:
        print("[fixed] touch queries...")
        _ = float(np.sum(q_fp32))

    if warmup > 0:
        warm_n = min(warmup, n)
        print(f"[fixed] warmup {warm_n}")
        fd_w = os.open(DEV_PATH, os.O_RDONLY)
        try:
            for i in range(warm_n):
                ioctl_search_single(fd_w, q_fp32[i], max_k, ef, sched_mode=sched_mode, want_trace=False)
        finally:
            os.close(fd_w)

    print(f"[fixed] n={n}, concurrency={concurrency}, ef={ef}, k={max_k}, sched_mode={sched_mode}, trace={trace}")

    indices = np.arange(n, dtype=np.int64)
    chunks = np.array_split(indices, concurrency)

    records_parts: List[List[Dict[str, Any]]] = [[] for _ in range(concurrency)]
    recall_parts: List[Dict[int, float]] = [{k: 0.0 for k in k_list} for _ in range(concurrency)]

    progress_lock = threading.Lock()
    processed = 0
    t_global_start_ns = 0

    barrier = threading.Barrier(concurrency + 1)

    def worker_fn(tid: int, idxs: np.ndarray):
        nonlocal processed, t_global_start_ns

        fd = os.open(DEV_PATH, os.O_RDONLY)
        local_count = 0

        try:
            barrier.wait()

            for idx_np in idxs:
                i = int(idx_np)

                t0 = time.perf_counter_ns()
                pred, dc, tr = ioctl_search_single(
                    fd,
                    q_fp32[i],
                    max_k,
                    ef,
                    sched_mode=sched_mode,
                    want_trace=trace,
                )
                t1 = time.perf_counter_ns()

                for k in k_list:
                    recall_parts[tid][k] += compute_recall_at_k(pred, neighbors[i], k, id_offset)

                records_parts[tid].append({
                    "idx": i,
                    "pred_ids": pred,
                    "dist_count": dc,
                    "latency_ms": (t1 - t0) / 1e6,
                    "trace": tr,
                    "tid": tid,
                })

                local_count += 1

                if qps_print_every > 0 and local_count % qps_print_every == 0:
                    with progress_lock:
                        processed += qps_print_every
                        elapsed = (time.perf_counter_ns() - t_global_start_ns) / 1e9
                        qps = processed / elapsed if elapsed > 0 else 0.0
                        recent = records_parts[tid][-qps_print_every:]
                        recent_lat = [record_latency_ms(r, latency_source) for r in recent]
                        recent_avg = float(np.mean(recent_lat)) if recent_lat else 0.0
                        print(f"[fixed tid={tid}] progress~{processed}/{n}, QPS~{qps:.2f}, recent_avg={recent_avg:.3f}ms")
        finally:
            os.close(fd)

    threads = []
    for tid in range(concurrency):
        th = threading.Thread(target=worker_fn, args=(tid, chunks[tid]), daemon=True)
        th.start()
        threads.append(th)

    t_all0 = time.perf_counter()
    t_global_start_ns = time.perf_counter_ns()
    barrier.wait()

    for th in threads:
        th.join()

    total_elapsed = time.perf_counter() - t_all0

    records: List[Dict[str, Any]] = []
    recalls_sum = {k: 0.0 for k in k_list}

    for part in records_parts:
        records.extend(part)

    for k in k_list:
        recalls_sum[k] = sum(rp[k] for rp in recall_parts)

    records.sort(key=lambda r: r["idx"])

    finalize_report(
        f"fixed_concurrent_{concurrency}",
        records,
        recalls_sum,
        k_list,
        total_elapsed,
        sample_count,
        latency_source,
    )


def evaluate_executor_concurrent(hdf5_path: str,
                                 k_list: List[int],
                                 ef: int,
                                 max_queries: Optional[int],
                                 id_offset: int,
                                 qps_print_every: int,
                                 no_norm: bool,
                                 concurrency: int,
                                 sched_mode: int,
                                 trace: bool,
                                 sample_count: int,
                                 latency_source: str):
    q_fp32, neighbors = prepare_queries(hdf5_path, max_queries, no_norm)

    max_k = max(k_list)
    if neighbors.shape[1] < max_k:
        raise RuntimeError(f"真值近邻数量不足以评估 k={max_k}")

    if not os.path.exists(DEV_PATH):
        raise RuntimeError(f"未找到 {DEV_PATH}")

    if trace and sched_mode == SCHED_MODE_OFF:
        print("[warn] --trace 在 sched_mode=OFF 下仍会返回 trace；基线也计入轨迹")

    n = len(q_fp32)
    concurrency = max(1, min(concurrency, n))

    print(f"[executor] n={n}, concurrency={concurrency}, ef={ef}, k={max_k}, sched_mode={sched_mode}, trace={trace}")

    def task(i: int) -> Dict[str, Any]:
        fd = os.open(DEV_PATH, os.O_RDONLY)
        try:
            t0 = time.perf_counter_ns()
            pred, dc, tr = ioctl_search_single(
                fd,
                q_fp32[i],
                max_k,
                ef,
                sched_mode=sched_mode,
                want_trace=trace,
            )
            t1 = time.perf_counter_ns()

            return {
                "idx": i,
                "pred_ids": pred,
                "dist_count": dc,
                "latency_ms": (t1 - t0) / 1e6,
                "trace": tr,
            }
        finally:
            os.close(fd)

    records = []
    recalls_sum = {k: 0.0 for k in k_list}

    t_all0 = time.perf_counter()
    processed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(task, i): i for i in range(n)}

        for fut in as_completed(futures):
            i = futures[fut]
            r = fut.result()

            for k in k_list:
                recalls_sum[k] += compute_recall_at_k(r["pred_ids"], neighbors[i], k, id_offset)

            records.append(r)
            processed += 1

            if qps_print_every > 0 and (processed % qps_print_every == 0 or processed == n):
                elapsed = time.perf_counter() - t_all0
                print(f"[executor {processed}/{n}] QPS={processed / elapsed:.2f}")

    total_elapsed = time.perf_counter() - t_all0
    records.sort(key=lambda r: r["idx"])

    finalize_report(
        f"executor_concurrent_{concurrency}",
        records,
        recalls_sum,
        k_list,
        total_elapsed,
        sample_count,
        latency_source,
    )


# ============================================================
# main
# ============================================================

def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="VecFS HNSW ioctl benchmark with optional per-query trace"
    )

    parser.add_argument("--hdf5",type=str, default='/home/fxx/vecfs/data/deep-image-96-angular.hdf5',help="deep-image-96-angular.hdf5 文件路径")
    parser.add_argument("--ef", type=int, default=200)
    parser.add_argument("--k_list", type=str, default="10")
    parser.add_argument("--max_queries", type=int, default=10000)
    parser.add_argument("--id_offset", type=int, default=1)
    parser.add_argument("--qps_print_every", type=int, default=1000)
    parser.add_argument("--no-norm", action="store_true")

    parser.add_argument(
        "--mode",
        choices=["single", "batch", "fixed", "concurrent"],
        default="fixed",
    )

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--concurrency_list",
        type=str,
        default="",
        help="例如 1,2,4,8,16；非空时会按列表 sweep",
    )

    parser.add_argument(
        "--sched-mode",
        type=int,
        choices=[SCHED_MODE_OFF, SCHED_MODE_MICRO, SCHED_MODE_SHAPING, SCHED_MODE_BOOST, SCHED_MODE_FULL],
        default=SCHED_MODE_OFF,
        help="内核调度器模式：0=OFF(并行裸跑) 1=MICRO 2=SHAPING 3=BOOST 4=FULL",
    )
    parser.add_argument("--trace", action="store_true", help="请求内核返回 per-query trace")
    parser.add_argument("--dev_path", default=DEFAULT_DEV_PATH, help="VecFS 挂载点路径（文件或目录）")

    parser.add_argument(
        "--latency-source",
        choices=["auto", "user", "kernel"],
        default="auto",
        help="过滤/排序时使用的延迟来源。auto: 有 trace 用 kernel，否则 user",
    )

    parser.add_argument("--sample_count", type=int, default=6, help="P50附近/最差尾部各输出多少条")
    parser.add_argument("--pin_all_queries", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)

    args = parser.parse_args()

    global DEV_PATH
    DEV_PATH = args.dev_path

    k_list = parse_int_list(args.k_list)

    if args.trace and args.sched_mode == SCHED_MODE_OFF:
        print("[warn] --trace 在 sched_mode=OFF 下仍会返回 trace；基线也计入轨迹")

    # if args.concurrency_list.strip():
    #     concurrency_values = parse_int_list(args.concurrency_list)
    # else:
    #     concurrency_values = [args.concurrency]
    concurrency_values = [1,2,4,8,16,20,30,40,80,160,320]
    if args.mode == "single":
        evaluate_single(
            hdf5_path=args.hdf5,
            k_list=k_list,
            ef=args.ef,
            max_queries=args.max_queries,
            id_offset=args.id_offset,
            qps_print_every=args.qps_print_every,
            no_norm=args.no_norm,
            sched_mode=args.sched_mode,
            trace=args.trace,
            sample_count=args.sample_count,
            latency_source=args.latency_source,
        )

    elif args.mode == "batch":
        for b in concurrency_values:
            evaluate_batch(
            hdf5_path=args.hdf5,
            k_list=k_list,
            ef=args.ef,
            max_queries=args.max_queries,
            id_offset=args.id_offset,
            qps_print_every=args.qps_print_every,
            no_norm=args.no_norm,
            batch_size=b,
            sched_mode=args.sched_mode,
            trace=args.trace,
            sample_count=args.sample_count,
            latency_source=args.latency_source,
        )

    elif args.mode == "fixed":
        for c in concurrency_values:
            evaluate_fixed_concurrent(
                hdf5_path=args.hdf5,
                k_list=k_list,
                ef=args.ef,
                max_queries=args.max_queries,
                id_offset=args.id_offset,
                qps_print_every=args.qps_print_every,
                no_norm=args.no_norm,
                concurrency=c,
                sched_mode=args.sched_mode,
                trace=args.trace,
                sample_count=args.sample_count,
                latency_source=args.latency_source,
                pin_all_queries=args.pin_all_queries,
                warmup=args.warmup,
            )

    elif args.mode == "concurrent":
        for c in concurrency_values:
            evaluate_executor_concurrent(
                hdf5_path=args.hdf5,
                k_list=k_list,
                ef=args.ef,
                max_queries=args.max_queries,
                id_offset=args.id_offset,
                qps_print_every=args.qps_print_every,
                no_norm=args.no_norm,
                concurrency=c,
                sched_mode=args.sched_mode,
                trace=args.trace,
                sample_count=args.sample_count,
                latency_source=args.latency_source,
            )


if __name__ == "__main__":
    main()

