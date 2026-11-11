#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures as cf
import json
import os
import statistics
import threading
import time
from typing import List, Dict, Optional, Tuple, Any

import h5py
import numpy as np
import hnswlib


# ============================================================
# 数据集工具
# ============================================================

def find_dataset(f: h5py.File, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in f:
            return name
    return None


def load_deep_image_hdf5(hdf5_path: str, max_queries: Optional[int] = None):
    with h5py.File(hdf5_path, "r") as f:
        test_key = find_dataset(f, ["test", "query", "queries"])
        neighbors_key = find_dataset(f, ["neighbors", "knn", "groundtruth", "true_neighbors"])

        if test_key is None or neighbors_key is None:
            raise RuntimeError("未在 HDF5 中找到 test/queries 或 neighbors 数据集")

        queries = np.asarray(f[test_key], dtype=np.float32)
        neighbors = np.asarray(f[neighbors_key], dtype=np.int64)

        if max_queries is not None:
            queries = queries[:max_queries]
            neighbors = neighbors[:max_queries]

        return queries, neighbors


def l2_normalize_queries_fp32(queries: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(queries, axis=1, keepdims=True).astype(np.float32)
    norms[norms == 0] = 1e-12
    q = queries / norms
    q[~np.isfinite(q)] = 0.0
    return np.ascontiguousarray(q.astype(np.float32, copy=False))


# ============================================================
# 统计工具
# ============================================================

def percentile_sorted(v: List[float], p: float) -> float:
    if not v:
        return 0.0
    n = len(v)
    idx = int(round((p / 100.0) * (n - 1)))
    idx = max(0, min(n - 1, idx))
    return float(v[idx])


def calculate_stats_ms(values_s: List[float]) -> Dict[str, float]:
    if not values_s:
        return {
            "avg": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    v = sorted([x * 1000.0 for x in values_s])
    return {
        "avg": float(statistics.mean(v)),
        "p50": percentile_sorted(v, 50),
        "p90": percentile_sorted(v, 90),
        "p95": percentile_sorted(v, 95),
        "p99": percentile_sorted(v, 99),
        "min": float(v[0]),
        "max": float(v[-1]),
    }


def calculate_stats_ms_from_ns(values_ns: List[int]) -> Dict[str, float]:
    return calculate_stats_ms([x / 1e9 for x in values_ns])


def compute_recall_at_k(pred: np.ndarray, gt_row: np.ndarray, k: int, id_offset: int) -> float:
    gt_topk = (gt_row[:k] + id_offset).tolist()
    return len(set(pred[:k].tolist()) & set(int(x) for x in gt_topk)) / float(k)


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def print_latency_stats(prefix: str, stats: Dict[str, float]):
    print(f"\n[{prefix}] 延迟统计(ms):")
    print(f"  avg = {stats['avg']:.3f}")
    print(f"  p50 = {stats['p50']:.3f}")
    print(f"  p90 = {stats['p90']:.3f}")
    print(f"  p95 = {stats['p95']:.3f}")
    print(f"  p99 = {stats['p99']:.3f}")
    print(f"  min = {stats['min']:.3f}")
    print(f"  max = {stats['max']:.3f}")


def select_near_percentile(records: List[Dict[str, Any]],
                           percentile: float,
                           count: int) -> List[Dict[str, Any]]:
    if not records:
        return []

    s = sorted(records, key=lambda r: r["latency_ns"])
    n = len(s)

    center = int(round((percentile / 100.0) * (n - 1)))
    half = count // 2
    start = max(0, center - half)
    end = min(n, start + count)

    if end - start < count:
        start = max(0, end - count)

    return s[start:end]


def select_worst(records: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    return sorted(records, key=lambda r: r["latency_ns"], reverse=True)[:count]


def print_record_samples(title: str,
                         records: List[Dict[str, Any]],
                         sample_count: int = 6):
    if not records:
        return

    print(f"\n[{title}]")
    header = (
        f"{'rank':>4} {'idx':>8} {'lat(ms)':>10} {'driver':>6} "
        f"{'width':>5} {'batch_n':>7} {'recall':>8}"
    )
    print(header)
    print("-" * len(header))

    for rank, r in enumerate(records[:sample_count]):
        lat_ms = r["latency_ns"] / 1e6
        rec = r.get("recall", None)
        rec_s = f"{rec:.4f}" if rec is not None else "N/A"

        print(
            f"{rank:4d} {r['idx']:8d} {lat_ms:10.3f} "
            f"{r.get('driver_tid', -1):6d} {r.get('driver_width', -1):5d} "
            f"{r.get('batch_n', -1):7d} {rec_s:>8}"
        )


def print_filtered_samples(records: List[Dict[str, Any]], sample_count: int):
    if not records:
        return

    p50_samples = select_near_percentile(records, 50, sample_count)
    worst_samples = select_worst(records, sample_count)

    print_record_samples("P50 附近样本", p50_samples, sample_count)
    print_record_samples("最差尾部样本", worst_samples, sample_count)


# ============================================================
# 基础模式：single
# ============================================================

def evaluate_single(p,
                    ef: int,
                    k: int,
                    q: np.ndarray,
                    disable_recall: bool,
                    neighbors: np.ndarray,
                    id_offset: int,
                    qps_print_every: int,
                    threads: int):
    p.set_ef(ef)

    n = len(q)
    print(f"[single-hnsw] 开始单条查询评估，总查询数={n}, ef={ef}, k={k}, num_threads={threads}")

    print("[single-hnsw] 预热...")
    warm = min(20, n)
    for i in range(warm):
        _labels, _dists = p.knn_query(q[i], k=k, num_threads=threads)

    records: List[Dict[str, Any]] = []
    recall_sum = 0.0
    processed = 0

    t_all0 = time.perf_counter()

    for i in range(n):
        ts = time.perf_counter_ns()
        labels, _dists = p.knn_query(q[i], k=k, num_threads=threads)
        te = time.perf_counter_ns()

        rec = None
        if not disable_recall:
            rec = compute_recall_at_k(labels[0], neighbors[i], k=k, id_offset=id_offset)
            recall_sum += rec

        records.append({
            "idx": i,
            "latency_ns": te - ts,
            "driver_tid": 0,
            "driver_width": 1,
            "batch_n": 1,
            "recall": rec,
        })

        processed += 1

        if qps_print_every > 0 and (processed % qps_print_every == 0 or processed == n):
            elapsed = time.perf_counter() - t_all0
            qps = processed / elapsed if elapsed > 0 else 0.0
            recent = [r["latency_ns"] for r in records[-min(qps_print_every, len(records)):]]
            recent_avg_ms = (sum(recent) / len(recent)) / 1e6 if recent else 0.0
            msg = (
                f"[single-hnsw {processed}/{n}] elapsed={elapsed:.2f}s "
                f"QPS={qps:.2f} recent_avg_latency={recent_avg_ms:.2f}ms"
            )
            if not disable_recall:
                msg += f" recall@{k}={recall_sum / processed:.4f}"
            print(msg)

    total = time.perf_counter() - t_all0
    qps = processed / total if total > 0 else 0.0
    stats = calculate_stats_ms_from_ns([r["latency_ns"] for r in records])
    recall = recall_sum / processed if (not disable_recall and processed > 0) else None

    print("\n" + "=" * 70)
    print("[single-hnsw] 完成")
    print("=" * 70)
    print(f"总查询数={processed}, 总耗时={total:.3f}s, QPS={qps:.2f}")
    print_latency_stats("single-hnsw", stats)
    if recall is not None:
        print(f"recall@{k}={recall:.6f}")

    print_filtered_samples(records, sample_count=6)

    return {
        "total_queries": processed,
        "total_time": total,
        "qps": qps,
        "latency_stats": stats,
        "recall": recall,
    }


# ============================================================
# 原 concurrent 模式：ThreadPoolExecutor，保留作对照
# ============================================================

def evaluate_concurrent(p,
                        ef: int,
                        k: int,
                        concurrency: int,
                        q: np.ndarray,
                        disable_recall: bool,
                        neighbors: np.ndarray,
                        id_offset: int,
                        qps_print_every: int,
                        threads: int):
    p.set_ef(ef)

    n = len(q)
    conc = max(1, int(concurrency))

    print(
        f"[concurrent-hnsw] 开始评估，总查询数={n}, ef={ef}, k={k}, "
        f"并发数={conc}, num_threads={threads}"
    )

    print("[concurrent-hnsw] 预热...")
    warm = min(20, n)
    for i in range(warm):
        _labels, _dists = p.knn_query(q[i], k=k, num_threads=threads)

    records: List[Dict[str, Any]] = []
    recall_sum = 0.0
    processed = 0
    t_all0 = time.perf_counter()

    def one_query(i: int):
        ts = time.perf_counter_ns()
        labels, _dists = p.knn_query(q[i], k=k, num_threads=threads)
        te = time.perf_counter_ns()
        return i, labels[0], te - ts

    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        next_i = 0
        in_flight: Dict[Any, int] = {}

        while next_i < n and len(in_flight) < conc:
            fut = ex.submit(one_query, next_i)
            in_flight[fut] = next_i
            next_i += 1

        while in_flight:
            done, _ = cf.wait(in_flight, return_when=cf.FIRST_COMPLETED)

            for fut in done:
                _i = in_flight.pop(fut)
                qi, pred, lat_ns = fut.result()

                rec = None
                if not disable_recall:
                    rec = compute_recall_at_k(pred, neighbors[qi], k=k, id_offset=id_offset)
                    recall_sum += rec

                records.append({
                    "idx": qi,
                    "latency_ns": lat_ns,
                    "driver_tid": -1,
                    "driver_width": 1,
                    "batch_n": 1,
                    "recall": rec,
                })

                processed += 1

                if next_i < n:
                    nfut = ex.submit(one_query, next_i)
                    in_flight[nfut] = next_i
                    next_i += 1

                if qps_print_every > 0 and (processed % qps_print_every == 0 or processed == n):
                    elapsed = time.perf_counter() - t_all0
                    qps = processed / elapsed if elapsed > 0 else 0.0
                    recent = [r["latency_ns"] for r in records[-min(qps_print_every, len(records)):]]
                    recent_avg_ms = (sum(recent) / len(recent)) / 1e6 if recent else 0.0
                    msg = (
                        f"[concurrent-hnsw {processed}/{n}] elapsed={elapsed:.2f}s "
                        f"QPS={qps:.2f} recent_avg_latency={recent_avg_ms:.2f}ms"
                    )
                    if not disable_recall:
                        msg += f" recall@{k}={recall_sum / processed:.4f}"
                    print(msg)

    total = time.perf_counter() - t_all0
    qps = processed / total if total > 0 else 0.0
    stats = calculate_stats_ms_from_ns([r["latency_ns"] for r in records])
    recall = recall_sum / processed if (not disable_recall and processed > 0) else None

    print("\n" + "=" * 70)
    print("[concurrent-hnsw] 完成")
    print("=" * 70)
    print(f"总查询数={processed}, 总耗时={total:.3f}s, QPS={qps:.2f}")
    print_latency_stats("concurrent-hnsw", stats)
    if recall is not None:
        print(f"recall@{k}={recall:.6f}")

    print_filtered_samples(records, sample_count=6)

    return {
        "total_queries": processed,
        "total_time": total,
        "qps": qps,
        "latency_stats": stats,
        "recall": recall,
    }


# ============================================================
# batch 模式
# ============================================================

def evaluate_batch(p,
                   ef: int,
                   k: int,
                   batch_size: int,
                   q: np.ndarray,
                   disable_recall: bool,
                   neighbors: np.ndarray,
                   id_offset: int,
                   qps_print_every: int,
                   batch_threads: int):
    p.set_ef(ef)

    n = len(q)
    batch_size = max(1, int(batch_size))
    batch_threads = max(1, int(batch_threads))

    print(
        f"[batch-hnsw] 开始批量查询评估，总查询数={n}, ef={ef}, k={k}, "
        f"batch_size={batch_size}, batch_threads={batch_threads}"
    )

    print("[batch-hnsw] 预热...")
    warm_batch = min(batch_size, n)
    if warm_batch > 0:
        _labels, _dists = p.knn_query(q[:warm_batch], k=k, num_threads=batch_threads)

    records: List[Dict[str, Any]] = []
    total_recall_sum = 0.0
    processed = 0
    t_all0 = time.perf_counter()

    num_batches = (n + batch_size - 1) // batch_size

    for b in range(num_batches):
        start = b * batch_size
        end = min(start + batch_size, n)
        cur_n = end - start
        if cur_n <= 0:
            continue

        batch_q = q[start:end]
        batch_neighbors = neighbors[start:end]

        ts = time.perf_counter_ns()
        labels, _dists = p.knn_query(batch_q, k=k, num_threads=batch_threads)
        te = time.perf_counter_ns()

        batch_lat_ns = te - ts

        for j in range(cur_n):
            idx = start + j
            rec = None
            if not disable_recall:
                rec = compute_recall_at_k(labels[j], batch_neighbors[j], k=k, id_offset=id_offset)
                total_recall_sum += rec

            records.append({
                "idx": idx,
                "latency_ns": batch_lat_ns,
                "driver_tid": 0,
                "driver_width": batch_threads,
                "batch_n": cur_n,
                "recall": rec,
            })

        processed += cur_n

        if qps_print_every > 0 and (processed % qps_print_every < cur_n or processed == n):
            elapsed = time.perf_counter() - t_all0
            qps = processed / elapsed if elapsed > 0 else 0.0
            print(
                f"[batch-hnsw {processed}/{n}] elapsed={elapsed:.2f}s "
                f"QPS={qps:.2f} last_batch_latency={batch_lat_ns / 1e6:.3f}ms"
            )

    total = time.perf_counter() - t_all0
    qps = processed / total if total > 0 else 0.0
    stats = calculate_stats_ms_from_ns([r["latency_ns"] for r in records])
    recall = total_recall_sum / processed if (not disable_recall and processed > 0) else None

    print("\n" + "=" * 70)
    print("[batch-hnsw] 完成")
    print("=" * 70)
    print(f"总查询数={processed}, 总耗时={total:.3f}s, QPS={qps:.2f}")
    print(f"批次数量={num_batches}, batch_size={batch_size}, batch_threads={batch_threads}")
    print_latency_stats("batch-hnsw", stats)
    if recall is not None:
        print(f"recall@{k}={recall:.6f}")

    print_filtered_samples(records, sample_count=6)

    return {
        "total_queries": processed,
        "total_time": total,
        "qps": qps,
        "latency_stats": stats,
        "recall": recall,
        "num_batches": num_batches,
    }


# ============================================================
# fixed 模式：低并发直接 N 个线程，高并发最多 20 个 driver 线程
# ============================================================

def compute_driver_widths(concurrency: int, max_driver_threads: int = 20) -> List[int]:
    """
    concurrency <= max_driver_threads:
        driver_threads = concurrency
        widths = [1] * concurrency

    concurrency > max_driver_threads:
        driver_threads = max_driver_threads
        widths sum == concurrency
        尽量均匀分配。例如 concurrency=40 => 20 个线程，每个 width=2。
    """
    concurrency = max(1, int(concurrency))
    max_driver_threads = max(1, int(max_driver_threads))

    driver_threads = min(concurrency, max_driver_threads)

    base = concurrency // driver_threads
    rem = concurrency % driver_threads

    widths = []
    for i in range(driver_threads):
        widths.append(base + (1 if i < rem else 0))

    assert sum(widths) == concurrency
    assert len(widths) == driver_threads

    return widths


def evaluate_concurrent_fixed(p,
                              ef: int,
                              k: int,
                              concurrency: int,
                              q: np.ndarray,
                              disable_recall: bool,
                              neighbors: np.ndarray,
                              id_offset: int,
                              qps_print_every: int,
                              max_driver_threads: int = 20,
                              fixed_inner_threads: int = 0,
                              warmup: int = 256,
                              pin_all_queries: bool = True,
                              sample_count: int = 6):
    """
    fixed 模式，高并发降 Python 线程数版本。

    设计目标：
    - concurrency < 20:
        使用 concurrency 个真实 Python driver 线程，每个线程 width=1。
    - concurrency >= 20:
        最多使用 20 个真实 Python driver 线程。
        每个 driver 线程负责一个 width。
        例如 concurrency=40:
            20 个 driver 线程，每个 width=2。

    重要说明：
    - hnswlib 的 Python API 是同步阻塞接口，没有“单线程同时发射多个异步 knn_query”的原生能力。
    - 因此这里用每个 driver 线程调用 batch knn_query(q_batch, num_threads=width)
      来近似“该 driver 负责 width 个后端并发查询”。
    - 这样可以把 Python 层真实线程数限制到最多 20，避免 80/160/320 个 Python 线程造成
      GIL、OS 调度、future 管理等用户态噪声。
    - 每个 driver 是真实 threading.Thread，并通过 barrier 同步起跑，不使用 asyncio。
    """
    p.set_ef(ef)

    n = len(q)
    conc = max(1, int(concurrency))
    q = np.ascontiguousarray(q, dtype=np.float32)

    widths = compute_driver_widths(conc, max_driver_threads=max_driver_threads)
    driver_threads = len(widths)

    print(
        f"[concurrent-hnsw-fixed] 开始评估，总查询数={n}, ef={ef}, k={k}, "
        f"目标并发={conc}, driver_threads={driver_threads}, widths={widths}, "
        f"fixed_inner_threads={fixed_inner_threads}"
    )

    if pin_all_queries:
        print("[concurrent-hnsw-fixed] 触摸查询向量内存（降低 page fault 抖动）...")
        _ = float(np.sum(q))

    if warmup > 0:
        warm_n = min(warmup, n)
        print(f"[concurrent-hnsw-fixed] warmup {warm_n} 次（不计入统计）...")
        for i in range(warm_n):
            _labels, _dists = p.knn_query(q[i], k=k, num_threads=1)

    barrier = threading.Barrier(driver_threads + 1)

    records_parts: List[List[Dict[str, Any]]] = [[] for _ in range(driver_threads)]
    recall_parts = [0.0 for _ in range(driver_threads)]
    processed_parts = [0 for _ in range(driver_threads)]

    index_lock = threading.Lock()
    progress_lock = threading.Lock()

    next_idx = {"value": 0}
    progress_processed = {"value": 0}
    t0_ns_holder = {"t0": 0}

    def alloc_batch(width: int) -> Optional[np.ndarray]:
        with index_lock:
            start = next_idx["value"]
            if start >= n:
                return None

            end = min(start + width, n)
            next_idx["value"] = end

        return np.arange(start, end, dtype=np.int64)

    def driver_fn(tid: int, width: int):
        """
        每个 driver 是一个真实 Python 线程。
        width 表示它负责的后端并发宽度。

        对 width 个 query 使用一次 batch knn_query。
        batch 返回后立刻领取下一组 query，保持该 driver 维持 width 宽度。
        """
        barrier.wait()

        while True:
            idxs = alloc_batch(width)
            if idxs is None or len(idxs) == 0:
                break

            batch_q = q[idxs]
            batch_neighbors = neighbors[idxs]
            batch_n = len(idxs)

            if fixed_inner_threads > 0:
                num_threads = fixed_inner_threads
            else:
                # 默认用 width 个 hnswlib 内部线程，近似该 driver 维护 width 个后端并发查询。
                num_threads = max(1, int(width))

            ts = time.perf_counter_ns()
            labels, _dists = p.knn_query(batch_q, k=k, num_threads=num_threads)
            te = time.perf_counter_ns()

            batch_lat_ns = te - ts

            local_records = records_parts[tid]

            for j in range(batch_n):
                idx = int(idxs[j])
                rec = None

                if not disable_recall:
                    rec = compute_recall_at_k(labels[j], batch_neighbors[j], k=k, id_offset=id_offset)
                    recall_parts[tid] += rec

                local_records.append({
                    "idx": idx,
                    "latency_ns": batch_lat_ns,
                    "driver_tid": tid,
                    "driver_width": width,
                    "batch_n": batch_n,
                    "inner_threads": num_threads,
                    "recall": rec,
                })

            processed_parts[tid] += batch_n

            if qps_print_every > 0 and processed_parts[tid] % qps_print_every < batch_n:
                with progress_lock:
                    progress_processed["value"] += qps_print_every
                    approx_processed = min(progress_processed["value"], n)
                    elapsed_ns = time.perf_counter_ns() - t0_ns_holder["t0"]
                    qps = approx_processed / (elapsed_ns / 1e9) if elapsed_ns > 0 else 0.0
                    recent = local_records[-min(qps_print_every, len(local_records)):]
                    recent_avg_ms = (
                        sum(r["latency_ns"] for r in recent) / len(recent) / 1e6
                    ) if recent else 0.0

                    print(
                        f"[concurrent-hnsw-fixed tid={tid} width={width}] "
                        f"progress~{approx_processed}/{n}, QPS~{qps:.2f}, "
                        f"recent_avg_latency~{recent_avg_ms:.2f}ms"
                    )

    threads_list = []
    for tid, width in enumerate(widths):
        th = threading.Thread(target=driver_fn, args=(tid, width), daemon=True)
        th.start()
        threads_list.append(th)

    t0_ns_holder["t0"] = time.perf_counter_ns()
    barrier.wait()

    for th in threads_list:
        th.join()

    total_elapsed_s = (time.perf_counter_ns() - t0_ns_holder["t0"]) / 1e9

    records: List[Dict[str, Any]] = []
    for part in records_parts:
        records.extend(part)

    records.sort(key=lambda r: r["idx"])

    total_processed = len(records)
    overall_qps = total_processed / total_elapsed_s if total_elapsed_s > 0 else 0.0

    lat_ns = [r["latency_ns"] for r in records]
    stats = calculate_stats_ms_from_ns(lat_ns)

    recall_sum = float(sum(recall_parts))
    recall = (recall_sum / total_processed) if (not disable_recall and total_processed > 0) else None

    print("\n" + "=" * 70)
    print("[concurrent-hnsw-fixed] 完成")
    print("=" * 70)
    print(f"目标并发={conc}, driver_threads={driver_threads}, widths={widths}")
    print(f"总查询数={total_processed}, 总耗时={total_elapsed_s:.3f}s, QPS={overall_qps:.2f}")
    print_latency_stats("concurrent-hnsw-fixed", stats)

    if recall is not None:
        print(f"recall@{k}={recall:.6f}")

    print_filtered_samples(records, sample_count=sample_count)

    return {
        "total_queries": total_processed,
        "total_time": total_elapsed_s,
        "qps": overall_qps,
        "latency_stats": stats,
        "recall": recall,
        "target_concurrency": conc,
        "driver_threads": driver_threads,
        "widths": widths,
    }


# ============================================================
# comprehensive
# ============================================================

def run_comprehensive_test(p,
                           ef: int,
                           k: int,
                           q: np.ndarray,
                           neighbors: np.ndarray,
                           id_offset: int,
                           qps_print_every: int,
                           disable_recall: bool = False,
                           test_configs=None):
    if test_configs is None:
        test_configs = {
            "single": [{"threads": 1}],
            "concurrent": [
                {"concurrency": 1, "threads": 1},
                {"concurrency": 2, "threads": 1},
                {"concurrency": 4, "threads": 1},
                {"concurrency": 8, "threads": 1},
                {"concurrency": 16, "threads": 1},
                {"concurrency": 32, "threads": 1},
            ],
            "batch": [
                {"batch_size": 32, "batch_threads": 1},
                {"batch_size": 64, "batch_threads": 2},
                {"batch_size": 128, "batch_threads": 4},
                {"batch_size": 256, "batch_threads": 8},
            ],
        }

    results = {}

    print("\n" + "=" * 80)
    print("单条查询测试")
    print("=" * 80)
    single_results = []
    for config in test_configs.get("single", []):
        print(f"\n单条查询配置: {config}")
        result = evaluate_single(
            p=p,
            ef=ef,
            k=k,
            q=q,
            disable_recall=disable_recall,
            neighbors=neighbors,
            id_offset=id_offset,
            qps_print_every=qps_print_every,
            threads=config.get("threads", 1),
        )
        single_results.append((config, result))
    results["single"] = single_results

    print("\n" + "=" * 80)
    print("并发查询测试")
    print("=" * 80)
    concurrent_results = []
    for config in test_configs.get("concurrent", []):
        print(f"\n并发查询配置: {config}")
        result = evaluate_concurrent(
            p=p,
            ef=ef,
            k=k,
            concurrency=config.get("concurrency", 1),
            q=q,
            disable_recall=disable_recall,
            neighbors=neighbors,
            id_offset=id_offset,
            qps_print_every=qps_print_every,
            threads=config.get("threads", 1),
        )
        concurrent_results.append((config, result))
    results["concurrent"] = concurrent_results

    print("\n" + "=" * 80)
    print("批量查询测试")
    print("=" * 80)
    batch_results = []
    for config in test_configs.get("batch", []):
        print(f"\n批量查询配置: {config}")
        result = evaluate_batch(
            p=p,
            ef=ef,
            k=k,
            batch_size=config.get("batch_size", 128),
            q=q,
            disable_recall=disable_recall,
            neighbors=neighbors,
            id_offset=id_offset,
            qps_print_every=qps_print_every,
            batch_threads=config.get("batch_threads", 4),
        )
        batch_results.append((config, result))
    results["batch"] = batch_results

    print_summary_report(results)
    return results


def print_summary_report(results):
    print("\n" + "=" * 100)
    print("测试结果汇总报告")
    print("=" * 100)

    if "single" in results and results["single"]:
        print("\n单条查询结果:")
        print("-" * 100)
        print(f"{'配置':<24} {'QPS':<12} {'avg(ms)':<12} {'p99(ms)':<12} {'max(ms)':<12} {'recall':<12}")
        for config, result in results["single"]:
            stats = result["latency_stats"]
            recall = f"{result['recall']:.6f}" if result["recall"] is not None else "N/A"
            print(
                f"{str(config):<24} {result['qps']:<12.2f} "
                f"{stats['avg']:<12.3f} {stats['p99']:<12.3f} {stats['max']:<12.3f} {recall:<12}"
            )

    if "concurrent" in results and results["concurrent"]:
        print("\n并发查询结果:")
        print("-" * 100)
        print(f"{'配置':<24} {'QPS':<12} {'avg(ms)':<12} {'p99(ms)':<12} {'max(ms)':<12} {'recall':<12}")
        for config, result in results["concurrent"]:
            stats = result["latency_stats"]
            recall = f"{result['recall']:.6f}" if result["recall"] is not None else "N/A"
            print(
                f"{str(config):<24} {result['qps']:<12.2f} "
                f"{stats['avg']:<12.3f} {stats['p99']:<12.3f} {stats['max']:<12.3f} {recall:<12}"
            )

    if "batch" in results and results["batch"]:
        print("\n批量查询结果:")
        print("-" * 100)
        print(f"{'配置':<24} {'QPS':<12} {'avg(ms)':<12} {'p99(ms)':<12} {'max(ms)':<12} {'recall':<12}")
        for config, result in results["batch"]:
            stats = result["latency_stats"]
            recall = f"{result['recall']:.6f}" if result["recall"] is not None else "N/A"
            print(
                f"{str(config):<24} {result['qps']:<12.2f} "
                f"{stats['avg']:<12.3f} {stats['p99']:<12.3f} {stats['max']:<12.3f} {recall:<12}"
            )

    print("\n" + "=" * 100)


# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="HNSWLib 性能测试工具：single / concurrent / batch / fixed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  1. 单条查询:
     python hnsw_test.py --mode single --threads 1

  2. 原 ThreadPoolExecutor 并发:
     python hnsw_test.py --mode concurrent --concurrency 40 --threads 1

  3. 批量:
     python hnsw_test.py --mode batch --batch_size 128 --batch_threads 8

  4. fixed 模式，目标并发 320，但最多 20 个 Python driver 线程:
     python hnsw_test.py --mode fixed --concurrency_list 320 --fixed_max_driver_threads 20

fixed 模式说明:
  concurrency < 20:
      使用 concurrency 个真实 Python 线程，每个线程 width=1。
  concurrency >= 20:
      使用最多 20 个真实 Python driver 线程。
      每个 driver 线程负责 width 个后端查询宽度。
      例如 concurrency=40:
          20 个 driver 线程，每个线程 batch width=2。
      该 driver 调用 hnswlib batch knn_query(q_batch, num_threads=width)。
        """
    )

    ap.add_argument("--index", default="/home/fxx/vecfs/data/deep-image-96-angular-M16-M0-32-efc400.bin",
                    help="hnswlib 官方格式索引文件（build_hnsw_graph.py 输出）")
    ap.add_argument("--space", default="ip", choices=["ip", "cosine", "l2"],
                    help="必须与构建索引时一致")
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--hdf5", default="/home/fxx/vecfs/data/deep-image-96-angular.hdf5")

    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef", type=int, default=200)
    ap.add_argument("--max_queries", type=int, default=10000)

    ap.add_argument("--mode",
                    default="fixed",
                    choices=["single", "concurrent", "batch", "comprehensive", "fixed"],
                    help="测试模式")

    ap.add_argument("--threads", type=int, default=1,
                    help="single/concurrent 模式下每次 knn_query 的 num_threads")

    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--concurrency_list", type=str,
                    default="1,2,4,8,16,20,30,40,80,160,320",
                    help="fixed/comprehensive 使用的并发数列表")

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--batch_threads", type=int, default=8)
    ap.add_argument("--batch_size_list", type=str,
                    default="1,2,8,16,20,30,40,80,160,320",
                    help="batch/comprehensive 使用的批大小列表")

    ap.add_argument("--fixed_max_driver_threads", type=int, default=20,
                    help="fixed 模式下 Python driver 线程数上限")
    ap.add_argument("--fixed_inner_threads", type=int, default=0,
                    help="fixed 模式每个 driver 的 hnswlib num_threads。0 表示使用 driver width")
    ap.add_argument("--fixed_sample_count", type=int, default=6,
                    help="fixed 模式输出 P50 附近和最差尾部样本数量")

    ap.add_argument("--qps_print_every", type=int, default=1000)
    ap.add_argument("--id_offset", type=int, default=0,
                    help="GT ID 偏移。hnswlib 默认 label 与 GT 都是 0-based，用 0（VecFS 返回 1-based node_id，才用 1）")
    ap.add_argument("--no_norm", action="store_true",
                    help="不做 L2 归一化")
    ap.add_argument("--disable_recall", action="store_true",
                    help="只测吞吐/延迟，不计算 recall")
    ap.add_argument("--warmup", type=int, default=256)
    ap.add_argument("--no_pin_queries", action="store_true",
                    help="fixed 模式下不预触摸 query 数组")
    ap.add_argument("--save_json", action="store_true",
                    help="comprehensive 模式保存 json")

    args = ap.parse_args()

    print(f"[hnsw-test] 加载数据集: {args.hdf5}")
    queries, neighbors = load_deep_image_hdf5(args.hdf5, args.max_queries)
    print(f"[hnsw-test] 查询数量={len(queries)}, 维度={queries.shape[1]}")

    if queries.shape[1] != args.dim:
        raise RuntimeError(f"dim mismatch: queries dim={queries.shape[1]} vs --dim {args.dim}")

    if not args.no_norm and args.space in ("ip", "cosine"):
        print("[hnsw-test] L2 归一化查询向量为单位范数...")
        q = l2_normalize_queries_fp32(queries)
    else:
        q = np.ascontiguousarray(queries.astype(np.float32, copy=False))

    print(f"[hnsw-test] 加载索引: {args.index}")
    p = hnswlib.Index(space=args.space, dim=args.dim)
    p.load_index(args.index)

    if args.mode == "single":
        print("\n" + "=" * 70)
        print("单条查询模式")
        print("=" * 70)

        evaluate_single(
            p=p,
            ef=args.ef,
            k=args.k,
            q=q,
            disable_recall=args.disable_recall,
            neighbors=neighbors,
            id_offset=args.id_offset,
            qps_print_every=args.qps_print_every,
            threads=args.threads,
        )

    elif args.mode == "concurrent":
        print("\n" + "=" * 70)
        print("原 ThreadPoolExecutor 并发模式")
        print("=" * 70)

        evaluate_concurrent(
            p=p,
            ef=args.ef,
            k=args.k,
            concurrency=args.concurrency,
            q=q,
            disable_recall=args.disable_recall,
            neighbors=neighbors,
            id_offset=args.id_offset,
            qps_print_every=args.qps_print_every,
            threads=args.threads,
        )

    elif args.mode == "batch":
        print("\n" + "=" * 70)
        print("批量查询模式")
        print("=" * 70)

        batch_size_list = parse_int_list(args.batch_size_list)
        if not batch_size_list:
            batch_size_list = [args.batch_size]

        for bs in batch_size_list:
            bt = bs//8
            evaluate_batch(
                p=p,
                ef=args.ef,
                k=args.k,
                batch_size=bs,
                q=q,
                disable_recall=args.disable_recall,
                neighbors=neighbors,
                id_offset=args.id_offset,
                qps_print_every=args.qps_print_every,
                batch_threads=bt,
            )

    elif args.mode == "fixed":
        print("\n" + "=" * 70)
        print("fixed 模式：最多 20 个 Python driver 线程，高并发按 width 分摊")
        print("=" * 70)

        concurrency_list = parse_int_list(args.concurrency_list)
        if not concurrency_list:
            concurrency_list = [args.concurrency]

        for conc in concurrency_list:
            evaluate_concurrent_fixed(
                p=p,
                ef=args.ef,
                k=args.k,
                concurrency=conc,
                q=q,
                disable_recall=args.disable_recall,
                neighbors=neighbors,
                id_offset=args.id_offset,
                qps_print_every=args.qps_print_every,
                max_driver_threads=args.fixed_max_driver_threads,
                fixed_inner_threads=args.fixed_inner_threads,
                warmup=args.warmup,
                pin_all_queries=(not args.no_pin_queries),
                sample_count=args.fixed_sample_count,
            )

    elif args.mode == "comprehensive":
        print("\n" + "=" * 70)
        print("comprehensive 模式")
        print("=" * 70)

        concurrency_list = parse_int_list(args.concurrency_list)
        batch_size_list = parse_int_list(args.batch_size_list)

        test_configs = {
            "single": [{"threads": args.threads}],
            "concurrent": [
                {"concurrency": c, "threads": args.threads}
                for c in concurrency_list
            ],
            "batch": [
                {"batch_size": bs, "batch_threads": args.batch_threads}
                for bs in batch_size_list
            ],
        }

        results = run_comprehensive_test(
            p=p,
            ef=args.ef,
            k=args.k,
            q=q,
            neighbors=neighbors,
            id_offset=args.id_offset,
            qps_print_every=args.qps_print_every,
            disable_recall=args.disable_recall,
            test_configs=test_configs,
        )

        if args.save_json:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            result_file = f"hnsw_test_results_{timestamp}.json"

            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_to_serializable(x) for x in obj]
                if isinstance(obj, tuple):
                    return tuple(convert_to_serializable(x) for x in obj)
                return obj

            with open(result_file, "w") as f:
                json.dump(convert_to_serializable(results), f, indent=2)

            print(f"\n测试结果已保存到: {result_file}")


if __name__ == "__main__":
    main()

