#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import json
import fcntl
import ctypes
import argparse
import threading
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import List, Tuple, Optional, Dict, Any

import h5py
import numpy as np


DEFAULT_DEV_PATH = "/mnt/vecfs-deepimage"
VEC_DIM = 96

# RTANN 调度器模式（与 include/vecfs_ioctl.h 的 VECFS_SCHED_MODE_* 对齐）
SCHED_MODE_OFF = 0
SCHED_MODE_MICRO = 1
SCHED_MODE_SHAPING = 2
SCHED_MODE_BOOST = 3
SCHED_MODE_FULL = 4


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
# Dataset utils
# ============================================================

def find_dataset(f: h5py.File, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in f:
            return name
    return None


def load_deep_image_queries_extend_from_train(
    hdf5_path: str,
    max_queries: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    查询加载策略：

    1. 优先从 test/query/queries 中取；
    2. 如果 max_queries > test 数量，则继续从 train/base/database/data 中取；
    3. 由于训练集补充部分没有 ground truth，本脚本不计算 Recall。

    返回：
      queries: [n, dim] float32
      source:  [n] int32, 0=test, 1=train
      meta:    统计信息
    """
    with h5py.File(hdf5_path, "r") as f:
        test_key = find_dataset(f, ["test", "query", "queries"])
        train_key = find_dataset(f, ["train", "base", "database", "data"])

        if test_key is None:
            raise RuntimeError(
                "未在 HDF5 中找到 test/query/queries 数据集，无法加载测试查询"
            )

        test_ds = f[test_key]
        test_n = int(test_ds.shape[0])
        dim = int(test_ds.shape[1])

        if max_queries is None:
            target_n = test_n
        else:
            target_n = int(max_queries)

        if target_n <= 0:
            raise RuntimeError("--max_queries 必须 > 0")

        take_test = min(test_n, target_n)

        parts = []
        sources = []

        if take_test > 0:
            q_test = np.asarray(test_ds[:take_test], dtype=np.float32)
            parts.append(q_test)
            sources.append(np.zeros((take_test,), dtype=np.int32))

        need_train = target_n - take_test

        train_n = 0
        take_train = 0

        if need_train > 0:
            if train_key is None:
                raise RuntimeError(
                    f"测试集只有 {test_n} 条，不足 {target_n} 条，"
                    "但 HDF5 中未找到 train/base/database/data 数据集用于补充"
                )

            train_ds = f[train_key]
            train_n = int(train_ds.shape[0])

            if int(train_ds.shape[1]) != dim:
                raise RuntimeError(
                    f"train 维度 {train_ds.shape[1]} 与 test 维度 {dim} 不一致"
                )

            take_train = min(train_n, need_train)

            if take_train < need_train:
                raise RuntimeError(
                    f"test({test_n}) + train({train_n}) 仍不足 {target_n} 条查询"
                )

            q_train = np.asarray(train_ds[:take_train], dtype=np.float32)
            parts.append(q_train)
            sources.append(np.ones((take_train,), dtype=np.int32))

        queries = np.vstack(parts)
        source = np.concatenate(sources)

        meta = {
            "test_key": test_key,
            "train_key": train_key,
            "test_total": test_n,
            "train_total": train_n,
            "take_test": take_test,
            "take_train": take_train,
            "total_queries": int(queries.shape[0]),
            "dim": dim,
        }

        return queries, source, meta


def l2_normalize_queries_fp32(queries: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(queries, axis=1, keepdims=True).astype(np.float32)
    norms[norms == 0] = 1e-12
    q = queries / norms
    q[~np.isfinite(q)] = 0.0
    return np.ascontiguousarray(q.astype(np.float32, copy=False))


# ============================================================
# Stats utils
# ============================================================

def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def percentile_sorted(v: List[float], p: float) -> float:
    if not v:
        return 0.0

    n = len(v)
    idx = int(round((p / 100.0) * (n - 1)))
    idx = max(0, min(n - 1, idx))
    return float(v[idx])


def calculate_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "avg": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "p999": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    v = sorted(float(x) for x in values)

    return {
        "avg": float(statistics.mean(v)),
        "p50": percentile_sorted(v, 50),
        "p90": percentile_sorted(v, 90),
        "p95": percentile_sorted(v, 95),
        "p99": percentile_sorted(v, 99),
        "p999": percentile_sorted(v, 99.9),
        "min": float(v[0]),
        "max": float(v[-1]),
    }


def print_stats_ms(title: str, values_ms: List[float]) -> Dict[str, float]:
    s = calculate_stats(values_ms)

    print(f"\n[{title}] 延迟统计(ms):")
    print(f"  avg  = {s['avg']:.3f}")
    print(f"  p50  = {s['p50']:.3f}")
    print(f"  p90  = {s['p90']:.3f}")
    print(f"  p95  = {s['p95']:.3f}")
    print(f"  p99  = {s['p99']:.3f}")
    print(f"  p999 = {s['p999']:.3f}")
    print(f"  min  = {s['min']:.3f}")
    print(f"  max  = {s['max']:.3f}")

    return s


def print_stats_num(title: str, values: List[float]) -> Dict[str, float]:
    s = calculate_stats(values)

    print(f"\n[{title}] 统计:")
    print(f"  avg  = {s['avg']:.2f}")
    print(f"  p50  = {s['p50']:.2f}")
    print(f"  p90  = {s['p90']:.2f}")
    print(f"  p95  = {s['p95']:.2f}")
    print(f"  p99  = {s['p99']:.2f}")
    print(f"  p999 = {s['p999']:.2f}")
    print(f"  min  = {s['min']:.0f}")
    print(f"  max  = {s['max']:.0f}")

    return s


def compute_slo_metrics(
    records: List[Dict[str, Any]],
    total_elapsed_s: float,
    deadline_ms_list: List[float],
    slo_timeout_rate: float,
) -> Dict[str, Any]:
    out = {}

    latencies = [float(r["e2e_latency_ms"]) for r in records]
    n = len(latencies)

    if n == 0:
        return out

    overall_stats = calculate_stats(latencies)

    for d in deadline_ms_list:
        timeout_count = sum(1 for x in latencies if x > d)
        success_count = n - timeout_count
        timeout_rate = timeout_count / n
        goodput = success_count / total_elapsed_s if total_elapsed_s > 0 else 0.0

        pass_slo = (
            overall_stats["p99"] <= d and
            timeout_rate <= slo_timeout_rate
        )

        out[f"deadline_{d:g}ms"] = {
            "deadline_ms": d,
            "timeout_count": timeout_count,
            "success_count": success_count,
            "timeout_rate": timeout_rate,
            "success_rate": success_count / n,
            "goodput_qps": goodput,
            "p99_le_deadline": overall_stats["p99"] <= d,
            "timeout_rate_le_target": timeout_rate <= slo_timeout_rate,
            "pass_slo": pass_slo,
        }

    return out


def print_slo_metrics(
    records: List[Dict[str, Any]],
    total_elapsed_s: float,
    deadline_ms_list: List[float],
    slo_timeout_rate: float,
    tag: str,
) -> Dict[str, Any]:
    metrics = compute_slo_metrics(
        records=records,
        total_elapsed_s=total_elapsed_s,
        deadline_ms_list=deadline_ms_list,
        slo_timeout_rate=slo_timeout_rate,
    )

    print(f"\n[{tag}] SLO / 超时率 / Goodput:")

    for _, m in metrics.items():
        print(
            f"  deadline={m['deadline_ms']:.3f}ms "
            f"timeout={m['timeout_count']}/{len(records)} "
            f"timeout_rate={m['timeout_rate'] * 100:.4f}% "
            f"success_rate={m['success_rate'] * 100:.4f}% "
            f"goodput={m['goodput_qps']:.2f} qps "
            f"pass_slo={m['pass_slo']}"
        )

    return metrics


def print_state_breakdown(
    records: List[Dict[str, Any]],
    total_elapsed_s: float,
    deadline_ms: float,
) -> Dict[str, Any]:
    out = {}

    print(f"\n[state breakdown] 按 MMPP 状态统计，主 deadline={deadline_ms:.3f}ms")

    for state in [0, 1]:
        rs = [r for r in records if int(r.get("mmpp_state", -1)) == state]

        if not rs:
            print(f"  state={state}: n=0")
            continue

        lat = [float(r["e2e_latency_ms"]) for r in rs]
        st = calculate_stats(lat)

        timeout_count = sum(1 for x in lat if x > deadline_ms)
        timeout_rate = timeout_count / len(lat)
        goodput = (len(lat) - timeout_count) / total_elapsed_s if total_elapsed_s > 0 else 0.0

        print(
            f"  state={state} n={len(rs)} "
            f"avg={st['avg']:.3f}ms "
            f"p95={st['p95']:.3f}ms "
            f"p99={st['p99']:.3f}ms "
            f"p999={st['p999']:.3f}ms "
            f"min={st['min']:.3f}ms "
            f"max={st['max']:.3f}ms "
            f"timeout={timeout_count}/{len(rs)} "
            f"timeout_rate={timeout_rate * 100:.4f}% "
            f"goodput={goodput:.2f}qps"
        )

        out[f"state_{state}"] = {
            "n": len(rs),
            "latency_ms": st,
            "deadline_ms": deadline_ms,
            "timeout_count": timeout_count,
            "timeout_rate": timeout_rate,
            "goodput_qps": goodput,
        }

    return out


# ============================================================
# Trace utils
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
        "tail_reason_name": TAIL_REASON_NAMES.get(
            int(t.tail_reason),
            f"UNKNOWN({int(t.tail_reason)})"
        ),

        "avg_rate_q16": int(t.avg_rate_q16),
        "min_rate_q16": int(t.min_rate_q16),
        "max_rate_q16": int(t.max_rate_q16),
    }


def summarize_trace_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    traces = [r.get("trace") for r in records if r.get("trace") is not None]

    if not traces:
        return {}

    reason_counter = Counter(t["tail_reason_name"] for t in traces)

    print("\n[trace] tail_reason 分布:")
    for reason, cnt in reason_counter.most_common():
        print(f"  {reason:18s}: {cnt}")

    out = {}

    out["trace_total_ms"] = print_stats_ms(
        "trace.total",
        [t["total_us"] / 1000.0 for t in traces],
    )

    out["trace_backlog_wait_ms"] = print_stats_ms(
        "trace.backlog_wait",
        [t["backlog_wait_us"] / 1000.0 for t in traces],
    )

    out["trace_queue_wait_ms"] = print_stats_ms(
        "trace.queue_wait",
        [t["queue_wait_us"] / 1000.0 for t in traces],
    )

    out["trace_run_ms"] = print_stats_ms(
        "trace.run",
        [t["run_us"] / 1000.0 for t in traces],
    )

    out["trace_ctx_wait_ms"] = print_stats_ms(
        "trace.ctx_wait",
        [t["ctx_wait_us"] / 1000.0 for t in traces],
    )

    out["trace_rescue_wait_ms"] = print_stats_ms(
        "trace.rescue_wait",
        [t["rescue_wait_us"] / 1000.0 for t in traces],
    )

    out["trace_normal_wait_ms"] = print_stats_ms(
        "trace.normal_wait",
        [t["normal_wait_us"] / 1000.0 for t in traces],
    )

    out["trace_boost_wait_ms"] = print_stats_ms(
        "trace.boost_wait",
        [t["boost_wait_us"] / 1000.0 for t in traces],
    )

    out["trace_first_run_wait_ms"] = print_stats_ms(
        "trace.first_run_wait",
        [t["first_run_wait_us"] / 1000.0 for t in traces],
    )

    out["trace_progress_units"] = print_stats_num(
        "trace.progress_units",
        [t["progress_units"] for t in traces],
    )

    out["trace_dist_compute_cnt"] = print_stats_num(
        "trace.dist_compute_cnt",
        [t["dist_compute_cnt"] for t in traces],
    )

    out["trace_dispatches"] = print_stats_num(
        "trace.dispatches",
        [t["dispatches"] for t in traces],
    )

    out["trace_requeues"] = print_stats_num(
        "trace.requeues",
        [t["requeues"] for t in traces],
    )

    out["trace_ctx_misses"] = print_stats_num(
        "trace.ctx_misses",
        [t["ctx_misses"] for t in traces],
    )

    out["tail_reason_counter"] = dict(reason_counter)

    return out


# ============================================================
# ioctl wrapper
# ============================================================

def ioctl_search_batch(
    fd: int,
    queries: np.ndarray,
    k: int,
    ef: int,
    sched_mode: int = SCHED_MODE_OFF,
    want_trace: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict[str, Any]]]]:
    """
    returns:
      results:     [n_queries, k] uint32  (node id，未命中为 0)
      dist_counts: [n_queries] uint64     (每查询距离计算次数)
      traces:      Optional[List[dict]]
    """
    if queries.dtype != np.float32:
        queries = queries.astype(np.float32, copy=False)

    if not queries.flags["C_CONTIGUOUS"]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)

    n_queries, dim = queries.shape

    results = np.zeros((n_queries, k), dtype=np.uint32)
    dist_counts = np.zeros((n_queries,), dtype=np.uint64)

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
    req.distances_ptr = 0          # 不请求距离，保持接口最小化
    req.dist_counts_ptr = int(dist_counts.ctypes.data)
    req.trace_ptr = int(trace_ptr)

    fcntl.ioctl(fd, IOCTL_SEARCH_BATCH, req)

    traces = None
    if trace_arr is not None:
        traces = [trace_to_dict(trace_arr[i]) for i in range(n_queries)]

    return results, dist_counts, traces


def ioctl_search_single(
    fd: int,
    query: np.ndarray,
    k: int,
    ef: int,
    sched_mode: int = SCHED_MODE_OFF,
    want_trace: bool = False,
):
    q2 = np.ascontiguousarray(query.reshape(1, -1).astype(np.float32, copy=False))

    res, dist_counts, traces = ioctl_search_batch(
        fd,
        q2,
        k,
        ef,
        sched_mode=sched_mode,
        want_trace=want_trace,
    )

    return res[0].tolist(), int(dist_counts[0]), traces[0] if traces else None


# ============================================================
# MMPP generator
# ============================================================

def expected_mmpp_high_query_fraction(
    lambda_low: float,
    lambda_high: float,
    q01: float,
    q10: float,
) -> Tuple[float, float]:
    """
    返回：
      high_time_frac:
        稳态高负载时间占比。

      high_query_frac:
        稳态下到达查询中来自高负载状态的比例。
    """
    if q01 < 0 or q10 < 0:
        return 0.0, 0.0

    if q01 == 0 and q10 == 0:
        return 0.0, 0.0

    if q01 + q10 <= 0:
        return 0.0, 0.0

    pi1 = q01 / (q01 + q10)
    pi0 = 1.0 - pi1

    denom = pi0 * lambda_low + pi1 * lambda_high

    if denom <= 0:
        high_query_frac = 0.0
    else:
        high_query_frac = pi1 * lambda_high / denom

    return pi1, high_query_frac


def generate_mmpp_arrivals_two_state(
    num_queries: int,
    lambda_low: float,
    lambda_high: float,
    q01: float,
    q10: float,
    initial_state: int,
    seed: int,
    burnin_s: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    二状态 MMPP。

    state 0:
      低负载状态，到达率 lambda_low，单位 query/s。

    state 1:
      高负载状态，到达率 lambda_high，单位 query/s。

    q01:
      state0 -> state1 切换速率，单位 1/s。

    q10:
      state1 -> state0 切换速率，单位 1/s。

    burnin_s:
      先运行 burnin_s 秒，不记录查询，用于降低初始状态偏差。
    """
    if num_queries <= 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int32)

    if lambda_low <= 0 and lambda_high <= 0:
        raise ValueError("lambda_low 和 lambda_high 不能同时 <= 0")

    if q01 < 0 or q10 < 0:
        raise ValueError("q01/q10 不能为负数")

    rng = np.random.default_rng(seed)

    lambdas = [float(lambda_low), float(lambda_high)]
    trans_rates = [float(q01), float(q10)]

    state = int(initial_state)
    if state not in (0, 1):
        state = 0

    t = 0.0
    burnin_s = max(0.0, float(burnin_s))

    offsets = np.empty((num_queries,), dtype=np.float64)
    states = np.empty((num_queries,), dtype=np.int32)

    i = 0

    while i < num_queries:
        lam = lambdas[state]
        trans_rate = trans_rates[state]

        time_to_arrival = rng.exponential(1.0 / lam) if lam > 0 else math.inf
        time_to_transition = rng.exponential(1.0 / trans_rate) if trans_rate > 0 else math.inf

        if time_to_arrival <= time_to_transition:
            t += time_to_arrival

            if t >= burnin_s:
                offsets[i] = t - burnin_s
                states[i] = state
                i += 1
        else:
            t += time_to_transition
            state = 1 - state

    return offsets, states


def generate_mmpp_with_min_high_fraction(
    num_queries: int,
    lambda_low: float,
    lambda_high: float,
    q01: float,
    q10: float,
    initial_state: int,
    seed: int,
    burnin_s: float,
    min_high_query_frac: float,
    seed_retry: int,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """
    为了避免有限样本下高负载查询数量过少，支持自动换 seed 重试。

    如果 min_high_query_frac <= 0，则不强制。
    """
    min_high_query_frac = max(0.0, float(min_high_query_frac))
    seed_retry = max(0, int(seed_retry))

    best_offsets = None
    best_states = None
    best_seed = seed
    best_frac = -1.0

    for attempt in range(seed_retry + 1):
        cur_seed = seed + attempt

        offsets, states = generate_mmpp_arrivals_two_state(
            num_queries=num_queries,
            lambda_low=lambda_low,
            lambda_high=lambda_high,
            q01=q01,
            q10=q10,
            initial_state=initial_state,
            seed=cur_seed,
            burnin_s=burnin_s,
        )

        high_frac = float(np.mean(states == 1)) if num_queries > 0 else 0.0

        if high_frac > best_frac:
            best_offsets = offsets
            best_states = states
            best_seed = cur_seed
            best_frac = high_frac

        if min_high_query_frac <= 0 or high_frac >= min_high_query_frac:
            return offsets, states, cur_seed, high_frac

    print(
        f"[warn] 已重试 {seed_retry} 次，仍未达到 "
        f"min_high_query_frac={min_high_query_frac:.4f}，"
        f"使用目前最佳 seed={best_seed}, high_frac={best_frac:.4f}"
    )

    return best_offsets, best_states, best_seed, best_frac


def sleep_until_ns(target_ns: int):
    while True:
        now = time.perf_counter_ns()
        remain_ns = target_ns - now

        if remain_ns <= 0:
            return

        if remain_ns > 2_000_000:
            time.sleep((remain_ns - 1_000_000) / 1e9)
        else:
            time.sleep(0)


def virtual_batch_threads(batch_size: int) -> int:
    """
    VecFS ioctl 当前没有 batch_threads 参数。
    这里只保留 hnswlib 对照中的 virtual_batch_threads 字段。
    """
    return max(1, int(batch_size) // 8)


# ============================================================
# Sample printing
# ============================================================

def select_near_percentile(
    records: List[Dict[str, Any]],
    percentile: float,
    count: int,
) -> List[Dict[str, Any]]:
    if not records:
        return []

    s = sorted(records, key=lambda r: r["e2e_latency_ms"])
    n = len(s)

    center = int(round((percentile / 100.0) * (n - 1)))
    half = count // 2

    start = max(0, center - half)
    end = min(n, start + count)

    if end - start < count:
        start = max(0, end - count)

    return s[start:end]


def print_samples(records: List[Dict[str, Any]], sample_count: int):
    if not records:
        return

    groups = [
        ("P50 附近样本", select_near_percentile(records, 50, sample_count)),
        ("P99 附近样本", select_near_percentile(records, 99, sample_count)),
        ("最差尾部样本", sorted(records, key=lambda r: r["e2e_latency_ms"], reverse=True)[:sample_count]),
    ]

    for title, rs in groups:
        print(f"\n[{title}]")
        header = (
            f"{'rank':>4} {'idx':>8} {'src':>5} {'state':>5} "
            f"{'e2e':>10} {'win':>10} {'pyq':>10} {'ioctl':>10} "
            f"{'kernel':>10} {'batch':>7} {'vth':>4} "
            f"{'dist':>10} {'reason':>16}"
        )
        print(header)
        print("-" * len(header))

        for rank, r in enumerate(rs):
            src = "test" if r.get("source", 0) == 0 else "train"
            tr = r.get("trace")
            kernel_ms = tr["total_us"] / 1000.0 if tr is not None else 0.0
            reason = tr["tail_reason_name"] if tr is not None else "NO_TRACE"

            print(
                f"{rank:4d} {r['idx']:8d} {src:>5} {r['mmpp_state']:5d} "
                f"{r['e2e_latency_ms']:10.3f} "
                f"{r['window_wait_ms']:10.3f} "
                f"{r['python_queue_wait_ms']:10.3f} "
                f"{r['ioctl_latency_ms']:10.3f} "
                f"{kernel_ms:10.3f} "
                f"{r['batch_n']:7d} "
                f"{r['virtual_batch_threads']:4d} "
                f"{int(r['dist_count']):10d} "
                f"{reason:>16}"
            )


# ============================================================
# Main MMPP evaluation
# ============================================================

def evaluate_mmpp(args):
    dev_path = args.dev_path

    if not os.path.exists(dev_path):
        raise RuntimeError(f"未找到挂载点路径: {dev_path}")

    if args.trace and args.sched_mode == SCHED_MODE_OFF:
        print("[warn] --trace 在 sched_mode=OFF 下仍会返回 trace；基线也计入轨迹")

    print(f"[load] 加载数据集: {args.hdf5}")

    queries, query_source, dataset_meta = load_deep_image_queries_extend_from_train(
        hdf5_path=args.hdf5,
        max_queries=args.max_queries,
    )

    print(f"[load] queries={queries.shape}")
    print(f"[load] test_key={dataset_meta['test_key']}, train_key={dataset_meta['train_key']}")
    print(f"[load] test_total={dataset_meta['test_total']}, train_total={dataset_meta['train_total']}")
    print(f"[load] take_test={dataset_meta['take_test']}, take_train={dataset_meta['take_train']}")

    if queries.shape[1] != args.dim:
        raise RuntimeError(
            f"dim mismatch: queries dim={queries.shape[1]} vs --dim={args.dim}"
        )

    if args.no_norm:
        print("[load] 不进行 L2 normalize")
        q_fp32 = np.ascontiguousarray(queries.astype(np.float32, copy=False))
    else:
        print("[load] L2 normalize queries")
        q_fp32 = l2_normalize_queries_fp32(queries)

    print("[note] 本脚本使用 test + train 扩充查询集，不计算 Recall。")

    n = len(q_fp32)

    if n <= 0:
        raise RuntimeError("没有查询数据")

    worker_threads = 20

    window_s = float(args.mmpp_window_ms) / 1000.0
    if window_s <= 0:
        raise ValueError("--mmpp_window_ms 必须 > 0")

    effective_lambda_low = args.mmpp_lambda_low * args.mmpp_scale
    effective_lambda_high = args.mmpp_lambda_high * args.mmpp_scale

    if effective_lambda_low <= 0 and effective_lambda_high <= 0:
        raise ValueError("scale 后 lambda_low 和 lambda_high 不能同时 <= 0")

    high_time_frac_expected, high_query_frac_expected = expected_mmpp_high_query_fraction(
        lambda_low=effective_lambda_low,
        lambda_high=effective_lambda_high,
        q01=args.mmpp_q01,
        q10=args.mmpp_q10,
    )

    print("\n" + "=" * 90)
    print("[mmpp-vecfs] MMPP + large query set + SLO 测试")
    print("=" * 90)
    print(f"backend                         = {dev_path}")
    print(f"queries                         = {n}")
    print(f"ef                              = {args.ef}")
    print(f"k                               = {args.k}")
    print(f"sched_mode                      = {args.sched_mode}")
    print(f"trace                           = {args.trace}")
    print(f"worker_threads                  = {worker_threads}，固定为 20")
    print(f"micro-batching window           = {args.mmpp_window_ms:.3f} ms")
    print(f"virtual_batch_threads           = max(1, batch_size // 8)，仅记录，不传给 VecFS")
    print(f"mmpp_scale                      = {args.mmpp_scale:.3f}")
    print(f"lambda_low base                 = {args.mmpp_lambda_low:.3f} qps")
    print(f"lambda_high base                = {args.mmpp_lambda_high:.3f} qps")
    print(f"lambda_low effective            = {effective_lambda_low:.3f} qps")
    print(f"lambda_high effective           = {effective_lambda_high:.3f} qps")
    print(f"q01                             = {args.mmpp_q01:.6f} 1/s")
    print(f"q10                             = {args.mmpp_q10:.6f} 1/s")
    print(f"initial_state                   = {args.mmpp_initial_state}")
    print(f"seed                            = {args.mmpp_seed}")
    print(f"burnin_s                        = {args.mmpp_burnin_s:.3f}")
    print(f"min_high_query_frac             = {args.mmpp_min_high_query_frac:.4f}")
    print(f"seed_retry                      = {args.mmpp_seed_retry}")
    print(f"expected high time fraction     = {high_time_frac_expected * 100:.2f}%")
    print(f"expected high query fraction    = {high_query_frac_expected * 100:.2f}%")
    print(f"deadline_ms                     = {args.deadline_ms:.3f}")
    print(f"deadline_list_ms                = {args.deadline_list_ms}")
    print(f"slo_timeout_rate                = {args.slo_timeout_rate:.6f}")

    if args.pin_all_queries:
        print("[warm] touch queries")
        _ = float(np.sum(q_fp32))

    if args.warmup > 0:
        warm_n = min(args.warmup, n)
        print(f"[warm] warmup {warm_n} 次，不计入统计")

        fd_w = os.open(dev_path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            for i in range(warm_n):
                ioctl_search_single(
                    fd_w,
                    q_fp32[i],
                    args.k,
                    args.ef,
                    sched_mode=args.sched_mode,
                    want_trace=False,
                )
        finally:
            os.close(fd_w)

    print("[mmpp] 生成 MMPP 到达序列...")

    arrival_offsets_s, mmpp_states, used_seed, actual_high_frac = generate_mmpp_with_min_high_fraction(
        num_queries=n,
        lambda_low=effective_lambda_low,
        lambda_high=effective_lambda_high,
        q01=args.mmpp_q01,
        q10=args.mmpp_q10,
        initial_state=args.mmpp_initial_state,
        seed=args.mmpp_seed,
        burnin_s=args.mmpp_burnin_s,
        min_high_query_frac=args.mmpp_min_high_query_frac,
        seed_retry=args.mmpp_seed_retry,
    )

    offered_duration_s = float(arrival_offsets_s[-1] - arrival_offsets_s[0]) if n > 1 else 0.0
    offered_qps = n / offered_duration_s if offered_duration_s > 0 else 0.0

    high_state_count = int(np.sum(mmpp_states == 1))
    low_state_count = n - high_state_count

    print(f"[mmpp] used_seed               = {used_seed}")
    print(f"[mmpp] planned duration        = {offered_duration_s:.3f}s")
    print(f"[mmpp] offered QPS             = {offered_qps:.2f}")
    print(f"[mmpp] state0 queries          = {low_state_count}")
    print(f"[mmpp] state1 queries          = {high_state_count}")
    print(f"[mmpp] actual high query frac  = {actual_high_frac * 100:.3f}%")

    records: List[Dict[str, Any]] = []
    records_lock = threading.Lock()

    processed = 0
    completed_batches = 0
    submitted_batches = 0
    progress_next_print = args.qps_print_every if args.qps_print_every > 0 else 0

    worker_local = threading.local()
    worker_fds = []
    worker_fds_lock = threading.Lock()

    def get_worker_fd() -> int:
        fd = getattr(worker_local, "fd", None)

        if fd is None:
            fd = os.open(dev_path, os.O_RDONLY | os.O_CLOEXEC)
            worker_local.fd = fd

            with worker_fds_lock:
                worker_fds.append(fd)

        return fd

    def process_one_batch(
        batch_id: int,
        start: int,
        end: int,
        batch_arrival_ns: np.ndarray,
        batch_states: np.ndarray,
        batch_source: np.ndarray,
        batch_submit_ns: int,
    ) -> Dict[str, Any]:
        fd = get_worker_fd()

        batch_start_ns = time.perf_counter_ns()

        batch_n = end - start
        vth = virtual_batch_threads(batch_n)

        ioctl_start_ns = time.perf_counter_ns()

        batch_res, batch_dc, batch_traces = ioctl_search_batch(
            fd,
            q_fp32[start:end],
            args.k,
            args.ef,
            sched_mode=args.sched_mode,
            want_trace=args.trace,
        )

        ioctl_done_ns = time.perf_counter_ns()
        batch_done_ns = ioctl_done_ns

        ioctl_latency_ms = (ioctl_done_ns - ioctl_start_ns) / 1e6
        python_queue_wait_ms = (batch_start_ns - batch_submit_ns) / 1e6

        local_records = []

        for j in range(batch_n):
            idx = start + j
            arrival_ns = int(batch_arrival_ns[j])

            tr = batch_traces[j] if batch_traces else None

            local_records.append({
                "idx": int(idx),
                "source": int(batch_source[j]),

                "dist_count": int(batch_dc[j]),
                "trace": tr,

                "mmpp_state": int(batch_states[j]),
                "arrival_offset_s": float(arrival_offsets_s[idx]),

                "arrival_ns": int(arrival_ns),
                "submit_ns": int(batch_submit_ns),
                "batch_start_ns": int(batch_start_ns),
                "done_ns": int(batch_done_ns),

                "e2e_latency_ms": float((batch_done_ns - arrival_ns) / 1e6),
                "window_wait_ms": float((batch_submit_ns - arrival_ns) / 1e6),
                "python_queue_wait_ms": float(python_queue_wait_ms),
                "ioctl_latency_ms": float(ioctl_latency_ms),

                "batch_id": int(batch_id),
                "batch_n": int(batch_n),
                "virtual_batch_threads": int(vth),
                "worker_ident": int(threading.get_ident()),
            })

        return {
            "batch_id": int(batch_id),
            "batch_n": int(batch_n),
            "records": local_records,
            "ioctl_latency_ms": float(ioctl_latency_ms),
            "virtual_batch_threads": int(vth),
        }

    def consume_future_result(fut):
        nonlocal processed, completed_batches, progress_next_print

        result = fut.result()
        batch_records = result["records"]

        with records_lock:
            records.extend(batch_records)
            processed += len(batch_records)
            completed_batches += 1

            if args.qps_print_every > 0:
                while processed >= progress_next_print:
                    elapsed_s = (time.perf_counter_ns() - t0_ns) / 1e9
                    qps_done = processed / elapsed_s if elapsed_s > 0 else 0.0

                    recent = records[-min(args.qps_print_every, len(records)):]
                    recent_avg = (
                        sum(r["e2e_latency_ms"] for r in recent) / len(recent)
                        if recent else 0.0
                    )

                    print(
                        f"[mmpp-vecfs {processed}/{n}] "
                        f"elapsed={elapsed_s:.2f}s "
                        f"completed_batches={completed_batches} "
                        f"QPS_done={qps_done:.2f} "
                        f"recent_avg_e2e={recent_avg:.3f}ms"
                    )

                    progress_next_print += args.qps_print_every

    print("[mmpp] 开始按 MMPP 到达时间在线回放...")

    t0_ns = time.perf_counter_ns()
    pending_futures = set()

    try:
        with ThreadPoolExecutor(max_workers=worker_threads) as executor:
            next_i = 0
            batch_id = 0

            while next_i < n:
                first_offset = float(arrival_offsets_s[next_i])
                window_index = int(math.floor(first_offset / window_s))
                window_end_s = (window_index + 1) * window_s

                start = next_i

                while next_i < n and float(arrival_offsets_s[next_i]) < window_end_s:
                    next_i += 1

                end = next_i
                batch_n = end - start

                if batch_n <= 0:
                    continue

                target_submit_ns = t0_ns + int(window_end_s * 1e9)
                sleep_until_ns(target_submit_ns)

                batch_arrival_ns = np.array(
                    [
                        t0_ns + int(float(arrival_offsets_s[i]) * 1e9)
                        for i in range(start, end)
                    ],
                    dtype=np.int64,
                )

                batch_states = mmpp_states[start:end].astype(np.int32, copy=False)
                batch_source = query_source[start:end].astype(np.int32, copy=False)

                batch_submit_ns = time.perf_counter_ns()

                fut = executor.submit(
                    process_one_batch,
                    batch_id,
                    start,
                    end,
                    batch_arrival_ns,
                    batch_states,
                    batch_source,
                    batch_submit_ns,
                )

                pending_futures.add(fut)

                submitted_batches += 1
                batch_id += 1

                done, pending_futures = wait(
                    pending_futures,
                    timeout=0,
                    return_when=FIRST_COMPLETED,
                )

                for df in done:
                    consume_future_result(df)

            for fut in as_completed(pending_futures):
                consume_future_result(fut)

    finally:
        with worker_fds_lock:
            for fd in worker_fds:
                try:
                    os.close(fd)
                except OSError:
                    pass

            worker_fds.clear()

    total_elapsed_s = (time.perf_counter_ns() - t0_ns) / 1e9

    records.sort(key=lambda r: r["idx"])

    total_processed = len(records)
    completed_qps = total_processed / total_elapsed_s if total_elapsed_s > 0 else 0.0

    test_query_count = sum(1 for r in records if r["source"] == 0)
    train_query_count = sum(1 for r in records if r["source"] == 1)

    batch_info = {}

    for r in records:
        bid = r["batch_id"]

        if bid not in batch_info:
            batch_info[bid] = {
                "batch_n": r["batch_n"],
                "virtual_batch_threads": r["virtual_batch_threads"],
            }

    batch_sizes = [v["batch_n"] for v in batch_info.values()]
    vths = [v["virtual_batch_threads"] for v in batch_info.values()]

    print("\n" + "=" * 90)
    print("[mmpp-vecfs] 完成")
    print("=" * 90)
    print(f"total_queries                = {total_processed}")
    print(f"test_queries                 = {test_query_count}")
    print(f"train_queries                = {train_query_count}")
    print(f"submitted_batches            = {submitted_batches}")
    print(f"completed_batches            = {completed_batches}")
    print(f"total_elapsed_s              = {total_elapsed_s:.3f}")
    print(f"offered_duration_s           = {offered_duration_s:.3f}")
    print(f"offered_qps                  = {offered_qps:.2f}")
    print(f"completed_qps                = {completed_qps:.2f}")
    print(f"mmpp_scale                   = {args.mmpp_scale:.3f}")
    print(f"lambda_low_effective         = {effective_lambda_low:.3f}")
    print(f"lambda_high_effective        = {effective_lambda_high:.3f}")
    print(f"state0_queries               = {low_state_count}")
    print(f"state1_queries               = {high_state_count}")
    print(f"state1_query_fraction        = {high_state_count / total_processed * 100:.3f}%")
    print(f"sched_mode                   = {args.sched_mode}")
    print(f"trace                        = {args.trace}")
    print("[note] Recall 未计算，因为查询集包含 train 补充样本。")

    if batch_sizes:
        print("\n[batch 统计]")
        print(f"  batch_count                 = {len(batch_sizes)}")
        print(f"  batch_size avg              = {float(np.mean(batch_sizes)):.3f}")
        print(f"  batch_size p50              = {float(np.percentile(batch_sizes, 50)):.3f}")
        print(f"  batch_size p90              = {float(np.percentile(batch_sizes, 90)):.3f}")
        print(f"  batch_size p99              = {float(np.percentile(batch_sizes, 99)):.3f}")
        print(f"  batch_size min              = {int(min(batch_sizes))}")
        print(f"  batch_size max              = {int(max(batch_sizes))}")
        print(f"  virtual_batch_threads avg   = {float(np.mean(vths)):.3f}")
        print(f"  virtual_batch_threads min   = {int(min(vths))}")
        print(f"  virtual_batch_threads max   = {int(max(vths))}")

    e2e_stats = print_stats_ms(
        "mmpp.e2e，到达到返回",
        [r["e2e_latency_ms"] for r in records],
    )

    window_stats = print_stats_ms(
        "mmpp.window_wait，窗口等待",
        [r["window_wait_ms"] for r in records],
    )

    pyqueue_stats = print_stats_ms(
        "mmpp.python_queue，Python线程池排队",
        [r["python_queue_wait_ms"] for r in records],
    )

    ioctl_stats = print_stats_ms(
        "mmpp.ioctl_blocking，batch ioctl阻塞时间",
        [r["ioctl_latency_ms"] for r in records],
    )

    dist_stats = print_stats_num(
        "mmpp.dist_compute_cnt",
        [r["dist_count"] for r in records],
    )

    deadline_list = parse_float_list(args.deadline_list_ms)

    if args.deadline_ms not in deadline_list:
        deadline_list = [args.deadline_ms] + deadline_list

    deadline_list = sorted(set(float(x) for x in deadline_list))

    slo_metrics = print_slo_metrics(
        records=records,
        total_elapsed_s=total_elapsed_s,
        deadline_ms_list=deadline_list,
        slo_timeout_rate=args.slo_timeout_rate,
        tag="mmpp-vecfs",
    )

    state_metrics = print_state_breakdown(
        records=records,
        total_elapsed_s=total_elapsed_s,
        deadline_ms=args.deadline_ms,
    )

    trace_summary = summarize_trace_records(records)

    print_samples(records, args.sample_count)

    result = {
        "mode": "mmpp",
        "backend": dev_path,
        "dataset": dataset_meta,

        "total_queries": total_processed,
        "test_queries": test_query_count,
        "train_queries": train_query_count,

        "submitted_batches": submitted_batches,
        "completed_batches": completed_batches,
        "total_elapsed_s": total_elapsed_s,
        "offered_duration_s": offered_duration_s,
        "offered_qps": offered_qps,
        "completed_qps": completed_qps,

        "worker_threads": worker_threads,
        "window_ms": args.mmpp_window_ms,

        "mmpp_scale": args.mmpp_scale,
        "lambda_low_base": args.mmpp_lambda_low,
        "lambda_high_base": args.mmpp_lambda_high,
        "lambda_low_effective": effective_lambda_low,
        "lambda_high_effective": effective_lambda_high,
        "q01": args.mmpp_q01,
        "q10": args.mmpp_q10,
        "initial_state": args.mmpp_initial_state,
        "seed": args.mmpp_seed,
        "used_seed": used_seed,
        "burnin_s": args.mmpp_burnin_s,
        "min_high_query_frac": args.mmpp_min_high_query_frac,
        "expected_high_time_fraction": high_time_frac_expected,
        "expected_high_query_fraction": high_query_frac_expected,
        "actual_high_query_fraction": high_state_count / total_processed if total_processed > 0 else 0.0,
        "low_state_queries": low_state_count,
        "high_state_queries": high_state_count,

        "deadline_ms": args.deadline_ms,
        "deadline_list_ms": deadline_list,
        "slo_timeout_rate": args.slo_timeout_rate,

        "ef": args.ef,
        "k": args.k,
        "sched_mode": args.sched_mode,
        "trace": args.trace,

        "e2e_latency_ms": e2e_stats,
        "window_wait_ms": window_stats,
        "python_queue_wait_ms": pyqueue_stats,
        "ioctl_blocking_ms": ioctl_stats,
        "dist_compute_cnt": dist_stats,

        "slo_metrics": slo_metrics,
        "state_metrics": state_metrics,
        "trace_summary": trace_summary,

        "batch_count": len(batch_sizes),
        "batch_size_avg": float(np.mean(batch_sizes)) if batch_sizes else 0.0,
        "batch_size_min": int(min(batch_sizes)) if batch_sizes else 0,
        "batch_size_max": int(max(batch_sizes)) if batch_sizes else 0,
        "virtual_batch_threads_avg": float(np.mean(vths)) if vths else 0.0,
        "virtual_batch_threads_min": int(min(vths)) if vths else 0,
        "virtual_batch_threads_max": int(max(vths)) if vths else 0,

        "recall": None,
        "recall_note": "Recall not computed because queries may include training samples without ground truth.",
    }

    if args.save_records:
        print(f"\n[save] records -> {args.save_records}")

        with open(args.save_records, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": result,
                    "records": records,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    if args.save_json:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"vecfs_mmpp_slo_large_{ts}.json"

        print(f"\n[save] summary -> {path}")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ============================================================
# main
# ============================================================

def main():
    #scale_list = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.0]
    scale_list = [0.25,0.5,0.75,1,1.25,1.5,1.75,2]
    for scale in scale_list:
        ap = argparse.ArgumentParser(
            description="VecFS HNSW ioctl MMPP benchmark with large query set, SLO, timeout rate, goodput"
        )

        ap.add_argument(
            "--dev_path",
            default=DEFAULT_DEV_PATH,
            help="VecFS 挂载点路径",
        )

        ap.add_argument(
            "--hdf5",
            default="/home/fxx/vecfs/data/deep-image-96-angular.hdf5",
            help="deep-image hdf5 文件路径",
        )

        ap.add_argument("--dim", type=int, default=96)

        ap.add_argument("--k", type=int, default=10)
        ap.add_argument("--ef", type=int, default=200)

        ap.add_argument(
            "--max_queries",
            type=int,
            default=100000,
            help="目标查询数。超过 test 数量时，会从 train/base/database 中补充。",
        )

        ap.add_argument(
            "--no_norm",
            action="store_true",
            help="不做 L2 normalize",
        )

        ap.add_argument(
            "--sched-mode",
            type=int,
            default=SCHED_MODE_OFF,
            choices=[
                SCHED_MODE_OFF,
                SCHED_MODE_MICRO,
                SCHED_MODE_SHAPING,
                SCHED_MODE_BOOST,
                SCHED_MODE_FULL,
            ],
            help=(
                "RTANN 调度模式：0=OFF(bare HNSW) 1=MICRO 2=SHAPING "
                "3=BOOST 4=FULL"
            ),
        )

        ap.add_argument(
            "--trace",
            action="store_true",
            help="请求内核返回 per-query trace（所有模式都支持）",
        )

        ap.add_argument(
            "--qps_print_every",
            type=int,
            default=5000,
        )

        ap.add_argument(
            "--warmup",
            type=int,
            default=256,
            help="预热查询数，不计入统计",
        )

        ap.add_argument(
            "--pin_all_queries",
            action="store_true",
            help="预触摸 query 内存，降低 page fault 抖动",
        )

        ap.add_argument(
            "--sample_count",
            type=int,
            default=2,
            help="P50/P99/尾部样本输出数量",
        )

        # MMPP parameters
        ap.add_argument(
            "--mmpp_window_ms",
            type=float,
            default=10.0,
            help="micro-batching 时间窗口，单位 ms",
        )

        ap.add_argument(
            "--mmpp_lambda_low",
            type=float,
            default=500.0,
            help="低负载状态到达率，单位 qps",
        )

        ap.add_argument(
            "--mmpp_lambda_high",
            type=float,
            default=4000.0,
            help="高负载状态到达率，单位 qps",
        )

        ap.add_argument(
            "--mmpp_scale",
            type=float,
            default=scale,
            help="同时缩放 lambda_low 和 lambda_high，例如 2.0 表示整体流量放大 2 倍",
        )

        ap.add_argument(
            "--mmpp_q01",
            type=float,
            default=0.5,
            help="低负载 state0 -> 高负载 state1 的切换速率，单位 1/s",
        )

        ap.add_argument(
            "--mmpp_q10",
            type=float,
            default=1.0,
            help="高负载 state1 -> 低负载 state0 的切换速率，单位 1/s",
        )

        ap.add_argument(
            "--mmpp_initial_state",
            type=int,
            default=0,
            choices=[0, 1],
            help="MMPP 初始状态，0=低负载，1=高负载",
        )

        ap.add_argument(
            "--mmpp_seed",
            type=int,
            default=12345,
            help="MMPP 随机种子",
        )

        ap.add_argument(
            "--mmpp_burnin_s",
            type=float,
            default=10.0,
            help="MMPP burn-in 时间。先模拟这么多秒不记录请求，减少初始状态偏差。",
        )

        ap.add_argument(
            "--mmpp_min_high_query_frac",
            type=float,
            default=0.20,
            help=(
                "要求生成的高负载查询比例至少达到该值。"
                "例如 0.20 表示至少 20%% 查询来自高负载状态。"
                "设为 0 则不强制。"
            ),
        )

        ap.add_argument(
            "--mmpp_seed_retry",
            type=int,
            default=50,
            help="如果高负载查询比例不足，自动换 seed 重试次数",
        )

        # SLO parameters
        ap.add_argument(
            "--deadline_ms",
            type=float,
            default=50.0,
            help="主 deadline，用于 state breakdown 和 SLO 判断",
        )

        ap.add_argument(
            "--deadline_list_ms",
            type=str,
            default="30,50,100",
            help="多个 deadline，用于同时输出 timeout/goodput，例如 30,50,100",
        )

        ap.add_argument(
            "--slo_timeout_rate",
            type=float,
            default=0.001,
            help="SLO 允许的超时率，例如 0.001 表示 0.1%",
        )

        ap.add_argument(
            "--save_json",
            action="store_true",
            help="保存 summary json",
        )

        ap.add_argument(
            "--save_records",
            default="",
            help="保存每条 query 的详细记录到指定 JSON 文件",
        )

        args = ap.parse_args()

        if not os.path.exists(args.dev_path):
            raise RuntimeError(f"挂载点路径不存在: {args.dev_path}")

        if not os.path.exists(args.hdf5):
            raise RuntimeError(f"HDF5 文件不存在: {args.hdf5}")

        evaluate_mmpp(args)


if __name__ == "__main__":
    main()

