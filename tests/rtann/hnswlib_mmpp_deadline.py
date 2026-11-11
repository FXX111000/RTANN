#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures as cf
import ctypes
import json
import math
import os
import platform
import statistics
import time
from typing import List, Dict, Optional, Tuple, Any

import h5py
import numpy as np
import hnswlib


# ============================================================
# Linux SCHED_DEADLINE
# ============================================================

SCHED_DEADLINE = 6


class SchedAttr(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32),
        ("policy", ctypes.c_uint32),
        ("flags", ctypes.c_uint64),
        ("nice", ctypes.c_int32),
        ("priority", ctypes.c_uint32),
        ("runtime", ctypes.c_uint64),
        ("deadline", ctypes.c_uint64),
        ("period", ctypes.c_uint64),
    ]


def get_sys_sched_setattr_number() -> int:
    machine = platform.machine().lower()

    if machine in ("x86_64", "amd64"):
        return 314
    if machine in ("aarch64", "arm64"):
        return 274
    if machine in ("i386", "i686", "x86"):
        return 351
    if machine.startswith("arm"):
        return 380
    if machine in ("riscv64",):
        return 274

    raise RuntimeError(f"Unsupported arch for sched_setattr: {machine}")


SYS_SCHED_SETATTR = get_sys_sched_setattr_number()

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.syscall.restype = ctypes.c_long


def set_sched_deadline_ns(
    runtime_ns: int,
    deadline_ns: int,
    period_ns: int,
    tid: int = 0,
):
    """
    给当前 Linux 线程设置 SCHED_DEADLINE。

    tid=0 表示当前调用线程。
    Linux 调度策略是 per-thread，不是整个进程统一设置。
    """

    runtime_ns = int(runtime_ns)
    deadline_ns = int(deadline_ns)
    period_ns = int(period_ns)

    if not (0 < runtime_ns <= deadline_ns <= period_ns):
        raise ValueError(
            "SCHED_DEADLINE requires: 0 < runtime <= deadline <= period"
        )

    attr = SchedAttr()
    attr.size = ctypes.sizeof(SchedAttr)
    attr.policy = SCHED_DEADLINE
    attr.flags = 0
    attr.nice = 0
    attr.priority = 0
    attr.runtime = runtime_ns
    attr.deadline = deadline_ns
    attr.period = period_ns

    ctypes.set_errno(0)

    ret = libc.syscall(
        ctypes.c_long(SYS_SCHED_SETATTR),
        ctypes.c_long(tid),
        ctypes.c_void_p(ctypes.addressof(attr)),
        ctypes.c_uint(0),
    )

    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(
            err,
            f"sched_setattr(SCHED_DEADLINE) failed: {os.strerror(err)} "
            f"(syscall={SYS_SCHED_SETATTR}, tid={tid}, "
            f"runtime={runtime_ns}, deadline={deadline_ns}, period={period_ns})",
        )


def set_sched_deadline_ms(
    runtime_ms: float,
    deadline_ms: float,
    period_ms: float,
):
    set_sched_deadline_ns(
        runtime_ns=int(float(runtime_ms) * 1_000_000),
        deadline_ns=int(float(deadline_ms) * 1_000_000),
        period_ns=int(float(period_ms) * 1_000_000),
    )


# ============================================================
# Dataset
# ============================================================

def find_dataset(f: h5py.File, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in f:
            return name
    return None


def load_deep_image_hdf5_with_extra(
    hdf5_path: str,
    max_queries: Optional[int] = None,
):
    with h5py.File(hdf5_path, "r") as f:
        test_key = find_dataset(f, ["test", "query", "queries"])
        neighbors_key = find_dataset(
            f,
            ["neighbors", "knn", "groundtruth", "true_neighbors"],
        )

        if test_key is None or neighbors_key is None:
            raise RuntimeError("未找到测试查询或 groundtruth 数据集")

        queries_test = np.asarray(f[test_key], dtype=np.float32)
        neighbors_test = np.asarray(f[neighbors_key], dtype=np.int64)

        n_test = int(len(queries_test))

        if max_queries is None or int(max_queries) <= 0:
            target_n = n_test
        else:
            target_n = int(max_queries)

        if target_n <= n_test:
            queries = queries_test[:target_n]
            neighbors = neighbors_test[:target_n]
            has_recall = np.ones((target_n,), dtype=bool)
            return queries, neighbors, has_recall

        # 需要扩充查询集
        gt_width = int(neighbors_test.shape[1])
        remaining = target_n - n_test

        query_parts = [queries_test]
        neighbor_parts = [neighbors_test]
        has_recall_parts = [np.ones((n_test,), dtype=bool)]

        train_key = find_dataset(f, ["train", "base", "database", "data"])

        if train_key is not None:
            train_data = np.asarray(f[train_key], dtype=np.float32)
            n_train = int(len(train_data))

            if n_train <= 0:
                raise RuntimeError("训练集为空，无法扩充查询")

            if n_train >= remaining:
                indices = np.random.choice(n_train, remaining, replace=False)
            else:
                indices = np.random.choice(n_train, remaining, replace=True)

            extra_queries = train_data[indices]
        else:
            # 没有 train 时循环复用 test 查询，但不计算 recall
            repeats = (remaining + n_test - 1) // n_test
            extra_queries = np.tile(queries_test, (repeats, 1))[:remaining]

        extra_neighbors = np.full((remaining, gt_width), -1, dtype=np.int64)
        extra_has_recall = np.zeros((remaining,), dtype=bool)

        query_parts.append(extra_queries)
        neighbor_parts.append(extra_neighbors)
        has_recall_parts.append(extra_has_recall)

        queries = np.concatenate(query_parts, axis=0).astype(np.float32, copy=False)
        neighbors = np.concatenate(neighbor_parts, axis=0).astype(np.int64, copy=False)
        has_recall = np.concatenate(has_recall_parts, axis=0).astype(bool, copy=False)

        return queries, neighbors, has_recall


def l2_normalize_queries_fp32(queries: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(queries, axis=1, keepdims=True).astype(np.float32)
    norms[norms == 0] = 1e-12

    q = queries / norms
    q[~np.isfinite(q)] = 0.0

    return np.ascontiguousarray(q.astype(np.float32, copy=False))


# ============================================================
# Stats
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


def calculate_stats_ms(values_ms: List[float]) -> Dict[str, float]:
    if not values_ms:
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

    v = sorted(float(x) for x in values_ms)

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


def compute_slo_metrics(
    latencies_ms: List[float],
    total_elapsed_s: float,
    deadline_ms_list: List[float],
    slo_timeout_rate: float,
) -> Dict[str, Any]:
    out = {}

    n = len(latencies_ms)
    if n == 0:
        return out

    overall_stats = calculate_stats_ms(latencies_ms)

    for d in deadline_ms_list:
        d = float(d)

        timeout_count = sum(1 for x in latencies_ms if x > d)
        success_count = n - timeout_count
        timeout_rate = timeout_count / n
        success_rate = success_count / n
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
            "success_rate": success_rate,
            "goodput_qps": goodput,
            "p99_le_deadline": overall_stats["p99"] <= d,
            "timeout_rate_le_target": timeout_rate <= slo_timeout_rate,
            "pass_slo": pass_slo,
        }

    return out


def compute_recall_at_k(
    pred: np.ndarray,
    gt_row: np.ndarray,
    k: int,
    id_offset: int,
) -> float:
    gt_topk = (gt_row[:k] + id_offset).tolist()
    return len(set(pred[:k].tolist()) & set(int(x) for x in gt_topk)) / float(k)


def print_final_result(result: Dict[str, Any]):
    print("\n" + "=" * 90)
    print("[mmpp-hnswlib] result")
    print("=" * 90)

    print(f"total_queries             = {result['total_queries']}")
    print(f"submitted_queries         = {result['submitted_queries']}")
    print(f"total_elapsed_s           = {result['total_elapsed_s']:.6f}")
    print(f"offered_duration_s        = {result['offered_duration_s']:.6f}")
    print(f"offered_qps               = {result['offered_qps']:.2f}")
    print(f"completed_qps             = {result['completed_qps']:.2f}")
    print(f"mmpp_scale                = {result['mmpp_scale']:.3f}")
    print(f"lambda_low_effective      = {result['lambda_low_effective']:.3f}")
    print(f"lambda_high_effective     = {result['lambda_high_effective']:.3f}")
    print(f"state0_queries            = {result['low_state_queries']}")
    print(f"state1_queries            = {result['high_state_queries']}")

    if result["recall"] is not None:
        print(f"recall@{result['k']}                 = {result['recall']:.6f}")

    print(
        "sched_deadline           = "
        f"{result['linux_sched_deadline']['enabled']}"
    )

    if result["linux_sched_deadline"]["enabled"]:
        print(
            "sched_params             = "
            f"runtime={result['linux_sched_deadline']['runtime_ms']}ms, "
            f"deadline={result['linux_sched_deadline']['deadline_ms']}ms, "
            f"period={result['linux_sched_deadline']['period_ms']}ms"
        )

    print("-" * 90)

    for _, m in result["slo_metrics"].items():
        print(
            f"deadline={m['deadline_ms']:.3f}ms "
            f"timeout={m['timeout_count']}/{result['total_queries']} "
            f"timeout_rate={m['timeout_rate'] * 100:.4f}% "
            f"success_rate={m['success_rate'] * 100:.4f}% "
            f"goodput={m['goodput_qps']:.2f}qps "
            f"pass_slo={m['pass_slo']}"
        )


# ============================================================
# MMPP
# ============================================================

def generate_mmpp_arrivals_two_state(
    num_queries: int,
    lambda_low: float,
    lambda_high: float,
    q01: float,
    q10: float,
    initial_state: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
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
            offsets[i] = t
            states[i] = state
            i += 1
        else:
            t += time_to_transition
            state = 1 - state

    return offsets, states


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


# ============================================================
# evaluate MMPP
# ============================================================

def evaluate_mmpp(args):
    if not os.path.exists(args.index):
        raise RuntimeError(f"index 不存在: {args.index}")

    if not os.path.exists(args.hdf5):
        raise RuntimeError(f"hdf5 不存在: {args.hdf5}")

    if args.use_sched_deadline:
        if not (
            0 < args.sched_runtime_ms <=
            args.sched_deadline_ms <=
            args.sched_period_ms
        ):
            raise ValueError(
                "SCHED_DEADLINE 参数必须满足: "
                "0 < runtime_ms <= deadline_ms <= period_ms"
            )

    queries, neighbors, has_recall = load_deep_image_hdf5_with_extra(
        args.hdf5,
        args.max_queries,
    )

    total_queries = len(queries)
    recall_available_count = int(np.sum(has_recall))

    if queries.shape[1] != args.dim:
        raise RuntimeError(
            f"dim mismatch: queries dim={queries.shape[1]} vs --dim={args.dim}"
        )

    if not args.no_norm and args.space in ("ip", "cosine"):
        q = l2_normalize_queries_fp32(queries)
    else:
        q = np.ascontiguousarray(queries.astype(np.float32, copy=False))

    p = hnswlib.Index(space=args.space, dim=args.dim)
    p.load_index(args.index)
    p.set_ef(args.ef)

    n = len(q)
    if n <= 0:
        raise RuntimeError("没有查询数据")

    worker_threads = int(args.worker_threads)
    if worker_threads <= 0:
        raise ValueError("--worker_threads 必须 > 0")

    window_s = float(args.mmpp_window_ms) / 1000.0
    if window_s <= 0:
        raise ValueError("--mmpp_window_ms 必须 > 0")

    effective_lambda_low = float(args.mmpp_lambda_low) * float(args.mmpp_scale)
    effective_lambda_high = float(args.mmpp_lambda_high) * float(args.mmpp_scale)

    if effective_lambda_low <= 0 and effective_lambda_high <= 0:
        raise ValueError("scale 后 lambda_low 和 lambda_high 不能同时 <= 0")

    if args.pin_all_queries:
        _ = float(np.sum(q))

    if args.warmup > 0:
        warm_n = min(int(args.warmup), n)

        for i in range(warm_n):
            p.knn_query(q[i], k=args.k, num_threads=1)

    arrival_offsets_s, mmpp_states = generate_mmpp_arrivals_two_state(
        num_queries=n,
        lambda_low=effective_lambda_low,
        lambda_high=effective_lambda_high,
        q01=args.mmpp_q01,
        q10=args.mmpp_q10,
        initial_state=args.mmpp_initial_state,
        seed=args.mmpp_seed,
    )

    offered_duration_s = (
        float(arrival_offsets_s[-1] - arrival_offsets_s[0])
        if n > 1 else 0.0
    )

    offered_qps = n / offered_duration_s if offered_duration_s > 0 else 0.0

    high_state_count = int(np.sum(mmpp_states == 1))
    low_state_count = n - high_state_count

    latencies_ms: List[float] = []

    recall_sum = 0.0
    recall_count = 0

    processed = 0
    submitted_queries = 0

    def apply_deadline_if_needed():
        if not args.use_sched_deadline:
            return

        set_sched_deadline_ms(
            runtime_ms=args.sched_runtime_ms,
            deadline_ms=args.sched_deadline_ms,
            period_ms=args.sched_period_ms,
        )

    def process_one_query(
        idx: int,
        arrival_ns: int,
        submit_ns: int,
    ) -> Tuple[float, Optional[float]]:
        """
        每个 query 单独作为一个任务执行。
        如果启用 --use_sched_deadline，则在该 query 执行前对当前 worker 线程设置 SCHED_DEADLINE。
        hnswlib 内部 num_threads 固定为 1，避免查询跑到其他非 deadline 线程上。
        """

        if args.use_sched_deadline:
            apply_deadline_if_needed()

        labels, _dists = p.knn_query(q[idx], k=args.k, num_threads=1)

        done_ns = time.perf_counter_ns()
        e2e_latency_ms = float((done_ns - arrival_ns) / 1e6)

        rec = None

        if (
            not args.disable_recall and
            bool(has_recall[idx])
        ):
            rec = compute_recall_at_k(
                labels[0],
                neighbors[idx],
                args.k,
                args.id_offset,
            )

        return e2e_latency_ms, rec

    def consume_future_result(fut):
        nonlocal processed, recall_sum, recall_count

        e2e_latency_ms, rec = fut.result()

        latencies_ms.append(e2e_latency_ms)
        processed += 1

        if rec is not None:
            recall_sum += float(rec)
            recall_count += 1

    t0_ns = time.perf_counter_ns()
    pending_futures = set()

    with cf.ThreadPoolExecutor(max_workers=worker_threads) as executor:
        next_i = 0

        while next_i < n:
            first_offset = float(arrival_offsets_s[next_i])
            window_index = int(math.floor(first_offset / window_s))
            window_end_s = (window_index + 1) * window_s

            start = next_i

            while next_i < n and float(arrival_offsets_s[next_i]) < window_end_s:
                next_i += 1

            end = next_i

            if end <= start:
                continue

            target_submit_ns = t0_ns + int(window_end_s * 1e9)
            sleep_until_ns(target_submit_ns)

            submit_ns = time.perf_counter_ns()

            for idx in range(start, end):
                arrival_ns = t0_ns + int(float(arrival_offsets_s[idx]) * 1e9)

                fut = executor.submit(
                    process_one_query,
                    int(idx),
                    int(arrival_ns),
                    int(submit_ns),
                )

                pending_futures.add(fut)
                submitted_queries += 1

            done, pending_futures = cf.wait(
                pending_futures,
                timeout=0,
                return_when=cf.FIRST_COMPLETED,
            )

            for df in done:
                consume_future_result(df)

        for fut in cf.as_completed(pending_futures):
            consume_future_result(fut)

    total_elapsed_s = (time.perf_counter_ns() - t0_ns) / 1e9

    total_processed = len(latencies_ms)
    completed_qps = (
        total_processed / total_elapsed_s
        if total_elapsed_s > 0 else 0.0
    )

    recall = None
    if not args.disable_recall and recall_count > 0:
        recall = recall_sum / recall_count

    deadline_list = parse_float_list(args.deadline_list_ms)

    if args.deadline_ms not in deadline_list:
        deadline_list = [args.deadline_ms] + deadline_list

    deadline_list = sorted(set(float(x) for x in deadline_list))

    slo_metrics = compute_slo_metrics(
        latencies_ms=latencies_ms,
        total_elapsed_s=total_elapsed_s,
        deadline_ms_list=deadline_list,
        slo_timeout_rate=args.slo_timeout_rate,
    )

    result = {
        "mode": "mmpp",
        "backend": "hnswlib",

        "index": args.index,
        "hdf5": args.hdf5,
        "space": args.space,
        "dim": args.dim,
        "k": args.k,
        "ef": args.ef,

        "total_queries": total_processed,
        "submitted_queries": submitted_queries,

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
        "low_state_queries": low_state_count,
        "high_state_queries": high_state_count,

        "deadline_ms": args.deadline_ms,
        "deadline_list_ms": deadline_list,
        "slo_timeout_rate": args.slo_timeout_rate,

        "linux_sched_deadline": {
            "enabled": bool(args.use_sched_deadline),
            "runtime_ms": args.sched_runtime_ms,
            "deadline_ms": args.sched_deadline_ms,
            "period_ms": args.sched_period_ms,
            "syscall": SYS_SCHED_SETATTR,
        },

        "recall": recall,
        "recall_count": recall_count,
        "recall_available_count": recall_available_count,

        "slo_metrics": slo_metrics,
    }

    print_final_result(result)

    if args.save_json:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"hnswlib_mmpp_deadline_result_{ts}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n[save] result -> {path}")

    return result


# ============================================================
# main
# ============================================================

def main():
    scale_list = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    for scale in scale_list:
        ap = argparse.ArgumentParser(
            description="hnswlib MMPP benchmark with Linux SCHED_DEADLINE, timeout rate, goodput"
        )

        ap.add_argument(
            "--index",
            default="/home/yons/fxx/hnswlib/index_hnswlib.bin",
        )

        ap.add_argument(
            "--hdf5",
            default="/home/yons/fxx/hnswlib/deep-image-96-angular.hdf5",
        )

        ap.add_argument(
            "--space",
            default="ip",
            choices=["ip", "cosine", "l2"],
        )

        ap.add_argument("--dim", type=int, default=96)
        ap.add_argument("--k", type=int, default=10)
        ap.add_argument("--ef", type=int, default=200)
        ap.add_argument("--max_queries", type=int, default=10000)
        ap.add_argument("--id_offset", type=int, default=0)

        ap.add_argument("--no_norm", action="store_true")
        ap.add_argument("--disable_recall", action="store_true")
        ap.add_argument("--warmup", type=int, default=256)
        ap.add_argument("--pin_all_queries", action="store_true")

        ap.add_argument(
            "--worker_threads",
            type=int,
            default=20,
            help="Python ThreadPoolExecutor worker 数量",
        )

        # MMPP parameters
        ap.add_argument("--mmpp_window_ms", type=float, default=10.0)
        ap.add_argument("--mmpp_lambda_low", type=float, default=500.0)
        ap.add_argument("--mmpp_lambda_high", type=float, default=4000.0)

        ap.add_argument(
            "--mmpp_scale",
            type=float,
            default=scale,
            help="同时缩放 lambda_low 和 lambda_high，例如 1.5 表示整体流量放大 1.5 倍",
        )

        ap.add_argument("--mmpp_q01", type=float, default=0.2)
        ap.add_argument("--mmpp_q10", type=float, default=1.0)

        ap.add_argument(
            "--mmpp_initial_state",
            type=int,
            default=0,
            choices=[0, 1],
        )

        ap.add_argument("--mmpp_seed", type=int, default=12345)

        # SLO parameters
        ap.add_argument(
            "--deadline_ms",
            type=float,
            default=50.0,
            help="主 deadline，用于 SLO 判断",
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

        # Linux SCHED_DEADLINE parameters
        ap.add_argument(
            "--use_sched_deadline",
            action="store_true",
            help="对每个查询执行前设置 Linux SCHED_DEADLINE",
        )

        ap.add_argument(
            "--sched_runtime_ms",
            type=float,
            default=2.0,
            help="SCHED_DEADLINE runtime，单位 ms",
        )

        ap.add_argument(
            "--sched_deadline_ms",
            type=float,
            default=30.0,
            help="SCHED_DEADLINE deadline，单位 ms",
        )

        ap.add_argument(
            "--sched_period_ms",
            type=float,
            default=30.0,
            help="SCHED_DEADLINE period，单位 ms",
        )

        ap.add_argument("--save_json", action="store_true")

        args = ap.parse_args()

        evaluate_mmpp(args)


if __name__ == "__main__":
    main()

