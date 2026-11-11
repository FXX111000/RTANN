#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures as cf
import json
import math
import os
import statistics
import threading
import time
from typing import List, Dict, Optional, Tuple, Any

import h5py
import numpy as np
import hnswlib


VEC_DIM = 96


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
    max_queries: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    查询池加载：

    1. 优先使用 test/query/queries；
    2. 如果不足 max_queries，则从 train/base/database/data 中补；
    3. 不计算 Recall。

    返回：
      queries: [N, dim] float32
      source:  [N] int32, 0=test, 1=train
      meta:    数据集统计
    """
    with h5py.File(hdf5_path, "r") as f:
        test_key = find_dataset(f, ["test", "query", "queries"])
        train_key = find_dataset(f, ["train", "base", "database", "data"])

        if test_key is None:
            raise RuntimeError("未在 HDF5 中找到 test/query/queries 数据集")

        test_ds = f[test_key]
        test_n = int(test_ds.shape[0])
        dim = int(test_ds.shape[1])

        target_n = int(max_queries)
        if target_n <= 0:
            raise ValueError("--max_queries 必须 > 0")

        take_test = min(test_n, target_n)
        need_train = target_n - take_test

        parts = []
        sources = []

        if take_test > 0:
            q_test = np.asarray(test_ds[:take_test], dtype=np.float32)
            parts.append(q_test)
            sources.append(np.zeros((take_test,), dtype=np.int32))

        train_n = 0
        take_train = 0

        if need_train > 0:
            if train_key is None:
                raise RuntimeError(
                    f"test 只有 {test_n} 条，不足 {target_n} 条，且未找到 train/base/database/data"
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

    overall_stats = calculate_stats(latencies_ms)

    for d in deadline_ms_list:
        timeout_count = sum(1 for x in latencies_ms if x > d)
        success_count = n - timeout_count
        timeout_rate = timeout_count / n
        success_rate = success_count / n
        goodput_qps = success_count / total_elapsed_s if total_elapsed_s > 0 else 0.0

        pass_slo = (
            overall_stats["p99"] <= d and
            timeout_rate <= slo_timeout_rate
        )

        out[f"deadline_{d:g}ms"] = {
            "deadline_ms": float(d),
            "timeout_count": int(timeout_count),
            "success_count": int(success_count),
            "timeout_rate": float(timeout_rate),
            "success_rate": float(success_rate),
            "goodput_qps": float(goodput_qps),
            "p99_le_deadline": bool(overall_stats["p99"] <= d),
            "timeout_rate_le_target": bool(timeout_rate <= slo_timeout_rate),
            "pass_slo": bool(pass_slo),
        }

    return out


def print_latency_stats(name: str, stats: Dict[str, float]):
    print(f"\n[{name}]")
    print(f"  avg_ms   = {stats['avg']:.3f}")
    print(f"  p50_ms   = {stats['p50']:.3f}")
    print(f"  p90_ms   = {stats['p90']:.3f}")
    print(f"  p95_ms   = {stats['p95']:.3f}")
    print(f"  p99_ms   = {stats['p99']:.3f}")
    print(f"  p999_ms  = {stats['p999']:.3f}")
    print(f"  min_ms   = {stats['min']:.3f}")
    print(f"  max_ms   = {stats['max']:.3f}")


# ============================================================
# MMPP generator
# ============================================================

def expected_mmpp_high_query_fraction(
    lambda_low: float,
    lambda_high: float,
    q01: float,
    q10: float,
) -> Tuple[float, float]:
    if q01 < 0 or q10 < 0:
        return 0.0, 0.0

    if q01 == 0 and q10 == 0:
        return 0.0, 0.0

    if q01 + q10 <= 0:
        return 0.0, 0.0

    pi1 = q01 / (q01 + q10)
    pi0 = 1.0 - pi1

    denom = pi0 * lambda_low + pi1 * lambda_high
    high_query_frac = 0.0 if denom <= 0 else pi1 * lambda_high / denom

    return pi1, high_query_frac


def generate_mmpp_arrivals_two_state(
    num_requests: int,
    lambda_low: float,
    lambda_high: float,
    q01: float,
    q10: float,
    initial_state: int,
    seed: int,
    burnin_s: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if num_requests <= 0:
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

    offsets = np.empty((num_requests,), dtype=np.float64)
    states = np.empty((num_requests,), dtype=np.int32)

    i = 0

    while i < num_requests:
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
    num_requests: int,
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
    min_high_query_frac = max(0.0, float(min_high_query_frac))
    seed_retry = max(0, int(seed_retry))

    best_offsets = None
    best_states = None
    best_seed = seed
    best_frac = -1.0

    for attempt in range(seed_retry + 1):
        cur_seed = seed + attempt

        offsets, states = generate_mmpp_arrivals_two_state(
            num_requests=num_requests,
            lambda_low=lambda_low,
            lambda_high=lambda_high,
            q01=q01,
            q10=q10,
            initial_state=initial_state,
            seed=cur_seed,
            burnin_s=burnin_s,
        )

        high_frac = float(np.mean(states == 1)) if num_requests > 0 else 0.0

        if high_frac > best_frac:
            best_offsets = offsets
            best_states = states
            best_seed = cur_seed
            best_frac = high_frac

        if min_high_query_frac <= 0 or high_frac >= min_high_query_frac:
            return offsets, states, cur_seed, high_frac

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


# ============================================================
# Locality workload generator
# ============================================================

def build_query_nn_groups_hnswlib(
    q_pool: np.ndarray,
    num_groups: int,
    group_size: int,
    space: str,
    ef_construction: int,
    m: int,
    ef_search: int,
    seed: int,
    num_threads: int,
    cache_path: str = "",
) -> np.ndarray:
    n, dim = q_pool.shape
    num_groups = min(int(num_groups), n)
    group_size = min(int(group_size), n)

    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        groups = data["groups"].astype(np.int64, copy=False)

        if groups.ndim != 2:
            raise RuntimeError("groups cache 格式错误")

        if groups.shape[1] != group_size:
            raise RuntimeError(
                f"groups cache group_size={groups.shape[1]} 与当前 {group_size} 不一致"
            )

        if int(np.max(groups)) >= n or int(np.min(groups)) < 0:
            raise RuntimeError("groups cache 中 query id 越界")

        return groups

    ids = np.arange(n, dtype=np.int64)

    idx = hnswlib.Index(space=space, dim=dim)
    idx.init_index(
        max_elements=n,
        ef_construction=int(ef_construction),
        M=int(m),
        random_seed=int(seed),
    )

    idx.add_items(q_pool, ids, num_threads=int(num_threads))
    idx.set_ef(max(int(ef_search), group_size * 2))

    rng = np.random.default_rng(seed)
    anchors = rng.choice(n, size=num_groups, replace=False)

    groups = np.empty((num_groups, group_size), dtype=np.int64)

    for gi, anchor in enumerate(anchors):
        labels, _ = idx.knn_query(
            q_pool[int(anchor)].reshape(1, -1),
            k=group_size,
            num_threads=1,
        )
        groups[gi, :] = labels[0].astype(np.int64)

    if cache_path:
        np.savez_compressed(
            cache_path,
            groups=groups,
            num_groups=np.array([num_groups], dtype=np.int64),
            group_size=np.array([group_size], dtype=np.int64),
            seed=np.array([seed], dtype=np.int64),
        )

    return groups


def build_random_hot_groups(
    n_pool: int,
    num_groups: int,
    group_size: int,
    seed: int,
    cache_path: str = "",
) -> np.ndarray:
    num_groups = min(int(num_groups), n_pool)
    group_size = min(int(group_size), n_pool)

    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["groups"].astype(np.int64, copy=False)

    rng = np.random.default_rng(seed)
    groups = np.empty((num_groups, group_size), dtype=np.int64)

    for gi in range(num_groups):
        groups[gi, :] = rng.choice(n_pool, size=group_size, replace=False)

    if cache_path:
        np.savez_compressed(cache_path, groups=groups)

    return groups


def choose_new_group(rng: np.random.Generator, num_groups: int, old_group: int) -> int:
    if num_groups <= 1:
        return 0

    g = int(rng.integers(0, num_groups))
    if g == old_group:
        g = (g + 1) % num_groups
    return g


def generate_locality_query_order(
    arrival_offsets_s: np.ndarray,
    mmpp_states: np.ndarray,
    groups: np.ndarray,
    n_pool: int,
    period_ms: float,
    keep_prob: float,
    seed: int,
    apply_to: str,
) -> Tuple[np.ndarray, np.ndarray]:
    n_req = len(arrival_offsets_s)
    num_groups, group_size = groups.shape

    period_s = float(period_ms) / 1000.0
    if period_s <= 0:
        raise ValueError("--locality_period_ms 必须 > 0")

    keep_prob = float(keep_prob)
    if keep_prob < 0.0 or keep_prob > 1.0:
        raise ValueError("--locality_keep_prob 必须在 [0, 1]")

    if apply_to not in ("all", "high_only"):
        raise ValueError("--locality_apply_to 必须是 all 或 high_only")

    rng = np.random.default_rng(seed)

    query_order = np.empty((n_req,), dtype=np.int64)
    group_order = np.empty((n_req,), dtype=np.int32)

    current_period = -1
    current_group = -1

    for i, t in enumerate(arrival_offsets_s):
        period_id = int(math.floor(float(t) / period_s))

        while current_period < period_id:
            current_period += 1

            if current_group < 0:
                current_group = int(rng.integers(0, num_groups))
            else:
                if rng.random() > keep_prob:
                    current_group = choose_new_group(rng, num_groups, current_group)

        use_locality = True

        if apply_to == "high_only" and int(mmpp_states[i]) != 1:
            use_locality = False

        if use_locality:
            g = current_group
            pos = int(rng.integers(0, group_size))
            query_order[i] = int(groups[g, pos])
            group_order[i] = int(g)
        else:
            query_order[i] = int(rng.integers(0, n_pool))
            group_order[i] = -1

    return query_order, group_order


def generate_independent_query_order(
    num_requests: int,
    n_pool: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    query_order = rng.integers(0, n_pool, size=num_requests, dtype=np.int64)
    group_order = np.full((num_requests,), -1, dtype=np.int32)
    return query_order, group_order


def save_workload_npz(
    path: str,
    arrival_offsets_s: np.ndarray,
    mmpp_states: np.ndarray,
    query_order: np.ndarray,
    group_order: np.ndarray,
    meta: Dict[str, Any],
):
    np.savez_compressed(
        path,
        arrival_offsets_s=arrival_offsets_s.astype(np.float64),
        mmpp_states=mmpp_states.astype(np.int32),
        query_order=query_order.astype(np.int64),
        group_order=group_order.astype(np.int32),
        meta_json=np.array([json.dumps(meta, ensure_ascii=False)]),
    )


def load_workload_npz(
    path: str,
    expected_n: int,
    n_pool: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    data = np.load(path, allow_pickle=False)

    arrival_offsets_s = data["arrival_offsets_s"].astype(np.float64, copy=False)
    mmpp_states = data["mmpp_states"].astype(np.int32, copy=False)
    query_order = data["query_order"].astype(np.int64, copy=False)
    group_order = data["group_order"].astype(np.int32, copy=False)

    meta = {}
    if "meta_json" in data:
        meta = json.loads(str(data["meta_json"][0]))

    if len(arrival_offsets_s) != expected_n:
        raise RuntimeError(
            f"workload 请求数 {len(arrival_offsets_s)} != expected_n {expected_n}"
        )

    if len(mmpp_states) != expected_n or len(query_order) != expected_n:
        raise RuntimeError("workload 数组长度不一致")

    if int(np.min(query_order)) < 0 or int(np.max(query_order)) >= n_pool:
        raise RuntimeError("workload query_order 越界")

    return arrival_offsets_s, mmpp_states, query_order, group_order, meta


# ============================================================
# HNSW search helpers
# ============================================================

def compute_batch_threads(batch_size: int) -> int:
    return max(1, int(batch_size) // 8)


# ============================================================
# Evaluation
# ============================================================

def evaluate(args):
    # -----------------------------
    # Load query pool
    # -----------------------------
    queries, query_source_pool, dataset_meta = load_deep_image_queries_extend_from_train(
        hdf5_path=args.hdf5,
        max_queries=args.max_queries,
    )

    if queries.shape[1] != args.dim:
        raise RuntimeError(
            f"dim mismatch: queries dim={queries.shape[1]} vs --dim={args.dim}"
        )

    if args.no_norm:
        q_pool = np.ascontiguousarray(queries.astype(np.float32, copy=False))
    else:
        q_pool = l2_normalize_queries_fp32(queries)

    n_pool = len(q_pool)
    num_requests = int(args.num_requests)

    if num_requests <= 0:
        raise ValueError("--num_requests 必须 > 0")

    # -----------------------------
    # Load hnswlib search index
    # -----------------------------
    p = hnswlib.Index(space=args.space, dim=args.dim)
    p.load_index(args.index)
    p.set_ef(args.ef)

    # -----------------------------
    # Workload
    # -----------------------------
    effective_lambda_low = args.mmpp_lambda_low * args.mmpp_scale
    effective_lambda_high = args.mmpp_lambda_high * args.mmpp_scale

    high_time_frac_expected, high_query_frac_expected = expected_mmpp_high_query_fraction(
        lambda_low=effective_lambda_low,
        lambda_high=effective_lambda_high,
        q01=args.mmpp_q01,
        q10=args.mmpp_q10,
    )

    if args.load_workload:
        arrival_offsets_s, mmpp_states, query_order, group_order, workload_meta = load_workload_npz(
            path=args.load_workload,
            expected_n=num_requests,
            n_pool=n_pool,
        )
        used_seed = int(workload_meta.get("used_seed", args.mmpp_seed))
        actual_high_frac = float(np.mean(mmpp_states == 1))
    else:
        arrival_offsets_s, mmpp_states, used_seed, actual_high_frac = generate_mmpp_with_min_high_fraction(
            num_requests=num_requests,
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

        if args.query_stream == "independent":
            query_order, group_order = generate_independent_query_order(
                num_requests=num_requests,
                n_pool=n_pool,
                seed=args.query_seed,
            )
        elif args.query_stream == "locality":
            if args.locality_method == "hnsw_groups":
                groups = build_query_nn_groups_hnswlib(
                    q_pool=q_pool,
                    num_groups=args.locality_num_groups,
                    group_size=args.locality_group_size,
                    space=args.locality_space,
                    ef_construction=args.locality_ef_construction,
                    m=args.locality_m,
                    ef_search=args.locality_ef_search,
                    seed=args.locality_seed,
                    num_threads=args.locality_build_threads,
                    cache_path=args.locality_groups_cache,
                )
            elif args.locality_method == "random_hotset":
                groups = build_random_hot_groups(
                    n_pool=n_pool,
                    num_groups=args.locality_num_groups,
                    group_size=args.locality_group_size,
                    seed=args.locality_seed,
                    cache_path=args.locality_groups_cache,
                )
            else:
                raise ValueError(f"unknown locality_method: {args.locality_method}")

            query_order, group_order = generate_locality_query_order(
                arrival_offsets_s=arrival_offsets_s,
                mmpp_states=mmpp_states,
                groups=groups,
                n_pool=n_pool,
                period_ms=args.locality_period_ms,
                keep_prob=args.locality_keep_prob,
                seed=args.query_seed,
                apply_to=args.locality_apply_to,
            )
        else:
            raise ValueError(f"unknown query_stream: {args.query_stream}")

        workload_meta = {
            "num_requests": num_requests,
            "n_pool": n_pool,
            "query_stream": args.query_stream,
            "locality_method": args.locality_method,
            "locality_group_size": args.locality_group_size,
            "locality_num_groups": args.locality_num_groups,
            "locality_period_ms": args.locality_period_ms,
            "locality_keep_prob": args.locality_keep_prob,
            "locality_apply_to": args.locality_apply_to,
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
            "query_seed": args.query_seed,
            "locality_seed": args.locality_seed,
            "burnin_s": args.mmpp_burnin_s,
        }

        if args.save_workload:
            save_workload_npz(
                path=args.save_workload,
                arrival_offsets_s=arrival_offsets_s,
                mmpp_states=mmpp_states,
                query_order=query_order,
                group_order=group_order,
                meta=workload_meta,
            )

    # -----------------------------
    # Workload summary
    # -----------------------------
    offered_duration_s = float(arrival_offsets_s[-1] - arrival_offsets_s[0]) if num_requests > 1 else 0.0
    offered_qps = num_requests / offered_duration_s if offered_duration_s > 0 else 0.0

    high_state_count = int(np.sum(mmpp_states == 1))
    low_state_count = num_requests - high_state_count

    unique_queries = int(len(np.unique(query_order)))
    unique_query_frac = unique_queries / float(num_requests) if num_requests > 0 else 0.0

    local_group_used = group_order[group_order >= 0]
    unique_groups_used = int(len(np.unique(local_group_used))) if len(local_group_used) > 0 else 0

    test_request_count = int(np.sum(query_source_pool[query_order] == 0))
    train_request_count = int(np.sum(query_source_pool[query_order] == 1))

    # -----------------------------
    # Warmup
    # -----------------------------
    if args.pin_all_queries:
        _ = float(np.sum(q_pool))

    if args.warmup > 0:
        warm_n = min(args.warmup, n_pool)
        for i in range(warm_n):
            p.knn_query(q_pool[i], k=args.k, num_threads=1)

    # -----------------------------
    # Online replay
    # -----------------------------
    worker_threads = 20
    window_s = float(args.mmpp_window_ms) / 1000.0
    if window_s <= 0:
        raise ValueError("--mmpp_window_ms 必须 > 0")

    latency_e2e_ms = np.empty((num_requests,), dtype=np.float64)
    window_wait_ms = np.empty((num_requests,), dtype=np.float64)
    python_queue_wait_ms = np.empty((num_requests,), dtype=np.float64)
    service_ms = np.empty((num_requests,), dtype=np.float64)

    batch_ids_arr = np.empty((num_requests,), dtype=np.int32)
    batch_n_arr = np.empty((num_requests,), dtype=np.int32)
    batch_threads_arr = np.empty((num_requests,), dtype=np.int32)

    completed_batches = 0
    submitted_batches = 0
    processed = 0

    records_lock = threading.Lock()

    def process_one_batch(
        batch_id: int,
        start: int,
        end: int,
        batch_arrival_ns: np.ndarray,
        batch_query_ids: np.ndarray,
        batch_submit_ns: int,
    ) -> Dict[str, Any]:
        batch_start_ns = time.perf_counter_ns()

        batch_n = end - start
        batch_threads = compute_batch_threads(batch_n)

        batch_q = np.ascontiguousarray(q_pool[batch_query_ids], dtype=np.float32)

        p.knn_query(
            batch_q,
            k=args.k,
            num_threads=batch_threads,
        )

        batch_done_ns = time.perf_counter_ns()

        cur_service_ms = (batch_done_ns - batch_start_ns) / 1e6
        cur_pyqueue_ms = (batch_start_ns - batch_submit_ns) / 1e6

        local_e2e = np.empty((batch_n,), dtype=np.float64)
        local_window = np.empty((batch_n,), dtype=np.float64)

        for j in range(batch_n):
            arrival_ns = int(batch_arrival_ns[j])
            local_e2e[j] = (batch_done_ns - arrival_ns) / 1e6
            local_window[j] = (batch_submit_ns - arrival_ns) / 1e6

        return {
            "batch_id": int(batch_id),
            "start": int(start),
            "end": int(end),
            "batch_n": int(batch_n),
            "batch_threads": int(batch_threads),
            "e2e_ms": local_e2e,
            "window_ms": local_window,
            "python_queue_ms": float(cur_pyqueue_ms),
            "service_ms": float(cur_service_ms),
        }

    def consume_future_result(fut):
        nonlocal processed, completed_batches

        res = fut.result()

        start = res["start"]
        end = res["end"]
        batch_n = res["batch_n"]
        batch_id = res["batch_id"]
        batch_threads = res["batch_threads"]

        with records_lock:
            latency_e2e_ms[start:end] = res["e2e_ms"]
            window_wait_ms[start:end] = res["window_ms"]
            python_queue_wait_ms[start:end] = res["python_queue_ms"]
            service_ms[start:end] = res["service_ms"]

            batch_ids_arr[start:end] = batch_id
            batch_n_arr[start:end] = batch_n
            batch_threads_arr[start:end] = batch_threads

            processed += batch_n
            completed_batches += 1

    t0_ns = time.perf_counter_ns()
    pending_futures = set()

    with cf.ThreadPoolExecutor(max_workers=worker_threads) as executor:
        next_i = 0
        batch_id = 0

        while next_i < num_requests:
            first_offset = float(arrival_offsets_s[next_i])
            window_index = int(math.floor(first_offset / window_s))
            window_end_s = (window_index + 1) * window_s

            start = next_i

            while next_i < num_requests and float(arrival_offsets_s[next_i]) < window_end_s:
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

            batch_query_ids = query_order[start:end].astype(np.int64, copy=False)
            batch_submit_ns = time.perf_counter_ns()

            fut = executor.submit(
                process_one_batch,
                batch_id,
                start,
                end,
                batch_arrival_ns,
                batch_query_ids,
                batch_submit_ns,
            )

            pending_futures.add(fut)
            submitted_batches += 1
            batch_id += 1

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
    completed_qps = num_requests / total_elapsed_s if total_elapsed_s > 0 else 0.0

    # -----------------------------
    # Summary metrics
    # -----------------------------
    e2e_list = latency_e2e_ms.tolist()
    window_list = window_wait_ms.tolist()
    pyqueue_list = python_queue_wait_ms.tolist()
    service_list = service_ms.tolist()

    e2e_stats = calculate_stats(e2e_list)
    window_stats = calculate_stats(window_list)
    pyqueue_stats = calculate_stats(pyqueue_list)
    service_stats = calculate_stats(service_list)

    deadline_list = parse_float_list(args.deadline_list_ms)
    if args.deadline_ms not in deadline_list:
        deadline_list = [args.deadline_ms] + deadline_list
    deadline_list = sorted(set(float(x) for x in deadline_list))

    slo_metrics = compute_slo_metrics(
        latencies_ms=e2e_list,
        total_elapsed_s=total_elapsed_s,
        deadline_ms_list=deadline_list,
        slo_timeout_rate=args.slo_timeout_rate,
    )

    state_metrics = {}
    for state in [0, 1]:
        mask = (mmpp_states == state)
        state_lat = latency_e2e_ms[mask].tolist()
        state_n = len(state_lat)

        if state_n == 0:
            state_metrics[f"state_{state}"] = {
                "n": 0,
                "latency_ms": calculate_stats([]),
                "deadline_ms": float(args.deadline_ms),
                "timeout_count": 0,
                "timeout_rate": 0.0,
                "goodput_qps": 0.0,
            }
            continue

        timeout_count = int(np.sum(np.asarray(state_lat) > args.deadline_ms))
        timeout_rate = timeout_count / state_n
        goodput_qps = (state_n - timeout_count) / total_elapsed_s if total_elapsed_s > 0 else 0.0

        state_metrics[f"state_{state}"] = {
            "n": int(state_n),
            "latency_ms": calculate_stats(state_lat),
            "deadline_ms": float(args.deadline_ms),
            "timeout_count": int(timeout_count),
            "timeout_rate": float(timeout_rate),
            "goodput_qps": float(goodput_qps),
        }

    unique_batch_ids = np.unique(batch_ids_arr)
    batch_sizes = []
    batch_threads_used = []

    for bid in unique_batch_ids:
        idx = int(np.where(batch_ids_arr == bid)[0][0])
        batch_sizes.append(int(batch_n_arr[idx]))
        batch_threads_used.append(int(batch_threads_arr[idx]))

    batch_summary = {
        "batch_count": int(len(batch_sizes)),
        "batch_size_avg": float(np.mean(batch_sizes)) if batch_sizes else 0.0,
        "batch_size_p50": float(np.percentile(batch_sizes, 50)) if batch_sizes else 0.0,
        "batch_size_p90": float(np.percentile(batch_sizes, 90)) if batch_sizes else 0.0,
        "batch_size_p99": float(np.percentile(batch_sizes, 99)) if batch_sizes else 0.0,
        "batch_size_min": int(min(batch_sizes)) if batch_sizes else 0,
        "batch_size_max": int(max(batch_sizes)) if batch_sizes else 0,
        "batch_threads_avg": float(np.mean(batch_threads_used)) if batch_threads_used else 0.0,
        "batch_threads_min": int(min(batch_threads_used)) if batch_threads_used else 0,
        "batch_threads_max": int(max(batch_threads_used)) if batch_threads_used else 0,
    }

    summary = {
        "mode": "mmpp_locality",
        "backend": "hnswlib",
        "index": args.index,
        "dataset": dataset_meta,

        "query_pool_size": int(n_pool),
        "num_requests": int(num_requests),
        "unique_query_ids": int(unique_queries),
        "unique_query_fraction": float(unique_query_frac),
        "test_requests": int(test_request_count),
        "train_requests": int(train_request_count),

        "submitted_batches": int(submitted_batches),
        "completed_batches": int(completed_batches),
        "total_elapsed_s": float(total_elapsed_s),
        "offered_duration_s": float(offered_duration_s),
        "offered_qps": float(offered_qps),
        "completed_qps": float(completed_qps),

        "worker_threads": int(worker_threads),
        "window_ms": float(args.mmpp_window_ms),

        "query_stream": args.query_stream,
        "locality_method": args.locality_method,
        "locality_group_size": int(args.locality_group_size),
        "locality_num_groups": int(args.locality_num_groups),
        "locality_period_ms": float(args.locality_period_ms),
        "locality_keep_prob": float(args.locality_keep_prob),
        "locality_apply_to": args.locality_apply_to,
        "unique_groups_used": int(unique_groups_used),

        "mmpp_scale": float(args.mmpp_scale),
        "lambda_low_base": float(args.mmpp_lambda_low),
        "lambda_high_base": float(args.mmpp_lambda_high),
        "lambda_low_effective": float(effective_lambda_low),
        "lambda_high_effective": float(effective_lambda_high),
        "q01": float(args.mmpp_q01),
        "q10": float(args.mmpp_q10),
        "initial_state": int(args.mmpp_initial_state),
        "seed": int(args.mmpp_seed),
        "used_seed": int(used_seed),
        "query_seed": int(args.query_seed),
        "locality_seed": int(args.locality_seed),
        "burnin_s": float(args.mmpp_burnin_s),
        "min_high_query_frac": float(args.mmpp_min_high_query_frac),
        "expected_high_time_fraction": float(high_time_frac_expected),
        "expected_high_query_fraction": float(high_query_frac_expected),
        "actual_high_query_fraction": float(high_state_count / num_requests if num_requests > 0 else 0.0),
        "low_state_requests": int(low_state_count),
        "high_state_requests": int(high_state_count),

        "deadline_ms": float(args.deadline_ms),
        "deadline_list_ms": deadline_list,
        "slo_timeout_rate": float(args.slo_timeout_rate),

        "ef": int(args.ef),
        "k": int(args.k),

        "e2e_latency_ms": e2e_stats,
        "window_wait_ms": window_stats,
        "python_queue_wait_ms": pyqueue_stats,
        "service_ms": service_stats,

        "slo_metrics": slo_metrics,
        "state_metrics": state_metrics,
        "batch_summary": batch_summary,

        "recall": None,
        "recall_note": "Recall not computed because queries may include training samples without ground truth.",
        "workload_meta": workload_meta,
    }

    # -----------------------------
    # Print only result summary
    # -----------------------------
    print("\n" + "=" * 88)
    print("[hnswlib-mmpp-locality] RESULT")
    print("=" * 88)
    print(f"query_pool_size        = {n_pool}")
    print(f"num_requests           = {num_requests}")
    print(f"unique_query_ids       = {unique_queries} ({unique_query_frac * 100:.3f}%)")
    print(f"test_requests          = {test_request_count}")
    print(f"train_requests         = {train_request_count}")
    print(f"query_stream           = {args.query_stream}")
    print(f"locality_method        = {args.locality_method}")
    print(f"locality_keep_prob     = {args.locality_keep_prob:.4f}")
    print(f"locality_period_ms     = {args.locality_period_ms:.3f}")
    print(f"locality_apply_to      = {args.locality_apply_to}")
    print(f"unique_groups_used     = {unique_groups_used}")
    print(f"offered_qps            = {offered_qps:.2f}")
    print(f"completed_qps          = {completed_qps:.2f}")
    print(f"total_elapsed_s        = {total_elapsed_s:.3f}")
    print(f"offered_duration_s     = {offered_duration_s:.3f}")
    print(f"state0_requests        = {low_state_count}")
    print(f"state1_requests        = {high_state_count}")
    print(f"state1_fraction        = {high_state_count / num_requests * 100:.3f}%")
    print(f"submitted_batches      = {submitted_batches}")
    print(f"completed_batches      = {completed_batches}")

    print_latency_stats("e2e_latency_ms", e2e_stats)
    print_latency_stats("window_wait_ms", window_stats)
    print_latency_stats("python_queue_wait_ms", pyqueue_stats)
    print_latency_stats("service_ms", service_stats)

    print("\n[SLO / Timeout / Goodput]")
    for _, m in slo_metrics.items():
        print(
            f"  deadline={m['deadline_ms']:.3f}ms "
            f"timeout={m['timeout_count']}/{num_requests} "
            f"timeout_rate={m['timeout_rate'] * 100:.4f}% "
            f"success_rate={m['success_rate'] * 100:.4f}% "
            f"goodput={m['goodput_qps']:.2f} qps "
            f"pass_slo={m['pass_slo']}"
        )

    print("\n[state breakdown]")
    for state in [0, 1]:
        sm = state_metrics[f"state_{state}"]
        st = sm["latency_ms"]
        print(
            f"  state={state} n={sm['n']} "
            f"avg={st['avg']:.3f}ms "
            f"p95={st['p95']:.3f}ms "
            f"p99={st['p99']:.3f}ms "
            f"p999={st['p999']:.3f}ms "
            f"timeout={sm['timeout_count']}/{sm['n']} "
            f"timeout_rate={sm['timeout_rate'] * 100:.4f}% "
            f"goodput={sm['goodput_qps']:.2f}qps"
        )

    print("\n[batch]")
    print(f"  batch_count        = {batch_summary['batch_count']}")
    print(f"  batch_size_avg     = {batch_summary['batch_size_avg']:.3f}")
    print(f"  batch_size_p50     = {batch_summary['batch_size_p50']:.3f}")
    print(f"  batch_size_p90     = {batch_summary['batch_size_p90']:.3f}")
    print(f"  batch_size_p99     = {batch_summary['batch_size_p99']:.3f}")
    print(f"  batch_size_min     = {batch_summary['batch_size_min']}")
    print(f"  batch_size_max     = {batch_summary['batch_size_max']}")
    print(f"  batch_threads_avg  = {batch_summary['batch_threads_avg']:.3f}")
    print(f"  batch_threads_min  = {batch_summary['batch_threads_min']}")
    print(f"  batch_threads_max  = {batch_summary['batch_threads_max']}")

    if args.save_json:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"hnswlib_mmpp_locality_100k_summary_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nsummary saved to: {path}")

    if args.save_summary:
        with open(args.save_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nsummary saved to: {args.save_summary}")

    return summary


# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="hnswlib MMPP locality benchmark, summary-only output"
    )

    # hnswlib index
    ap.add_argument(
        "--index",
        default="/home/yons/fxx/hnswlib/index_hnswlib.bin",
        help="hnswlib index 文件",
    )
    ap.add_argument(
        "--space",
        default="ip",
        choices=["ip", "cosine", "l2"],
        help="必须与 hnswlib index 构建时一致",
    )
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef", type=int, default=200)

    # dataset
    ap.add_argument(
        "--hdf5",
        default="/home/yons/fxx/hnswlib/deep-image-96-angular.hdf5",
        help="deep-image hdf5 文件路径",
    )
    ap.add_argument(
        "--max_queries",
        type=int,
        default=100000,
        help="查询池大小。test 不足时从 train 补。",
    )
    ap.add_argument(
        "--num_requests",
        type=int,
        default=100000,
        help="在线请求总数。",
    )
    ap.add_argument(
        "--no_norm",
        action="store_true",
        help="不做 L2 normalize",
    )

    # run
    ap.add_argument(
        "--warmup",
        type=int,
        default=256,
        help="预热查询数，不计入统计",
    )
    ap.add_argument(
        "--pin_all_queries",
        action="store_true",
        help="预触摸 query pool 内存",
    )

    # MMPP
    ap.add_argument("--mmpp_window_ms", type=float, default=10.0)
    ap.add_argument("--mmpp_lambda_low", type=float, default=500.0)
    ap.add_argument("--mmpp_lambda_high", type=float, default=4000.0)
    ap.add_argument("--mmpp_scale", type=float, default=1.0)
    ap.add_argument("--mmpp_q01", type=float, default=0.2)
    ap.add_argument("--mmpp_q10", type=float, default=1.0)
    ap.add_argument("--mmpp_initial_state", type=int, default=0, choices=[0, 1])
    ap.add_argument("--mmpp_seed", type=int, default=12345)
    ap.add_argument("--mmpp_burnin_s", type=float, default=10.0)
    ap.add_argument("--mmpp_min_high_query_frac", type=float, default=0.20)
    ap.add_argument("--mmpp_seed_retry", type=int, default=50)

    # query stream / locality
    ap.add_argument(
        "--query_stream",
        choices=["independent", "locality"],
        default="locality",
    )
    ap.add_argument("--query_seed", type=int, default=20240520)
    ap.add_argument(
        "--locality_method",
        choices=["hnsw_groups", "random_hotset"],
        default="hnsw_groups",
    )
    ap.add_argument("--locality_group_size", type=int, default=128)
    ap.add_argument("--locality_num_groups", type=int, default=1000)
    ap.add_argument("--locality_period_ms", type=float, default=100.0)
    ap.add_argument("--locality_keep_prob", type=float, default=0.9)
    ap.add_argument(
        "--locality_apply_to",
        choices=["all", "high_only"],
        default="all",
    )
    ap.add_argument("--locality_seed", type=int, default=777)
    ap.add_argument(
        "--locality_space",
        choices=["ip", "cosine", "l2"],
        default="ip",
    )
    ap.add_argument("--locality_m", type=int, default=16)
    ap.add_argument("--locality_ef_construction", type=int, default=400)
    ap.add_argument("--locality_ef_search", type=int, default=200)
    ap.add_argument("--locality_build_threads", type=int, default=8)
    ap.add_argument(
        "--locality_groups_cache",
        default="",
        help="local groups cache .npz。存在则加载，不存在则构建并保存。",
    )

    # workload reuse
    ap.add_argument(
        "--load_workload",
        default="",
        help="加载 VecFS 或本脚本生成的 workload .npz，推荐用于公平对比。",
    )
    ap.add_argument(
        "--save_workload",
        default="",
        help="保存 arrival_offsets_s/mmpp_states/query_order/group_order 到 .npz。",
    )

    # SLO
    ap.add_argument("--deadline_ms", type=float, default=50.0)
    ap.add_argument("--deadline_list_ms", type=str, default="30,50,100")
    ap.add_argument(
        "--slo_timeout_rate",
        type=float,
        default=0.001,
        help="SLO 允许超时率，例如 0.001 表示 0.1%",
    )

    # save
    ap.add_argument("--save_json", action="store_true")
    ap.add_argument(
        "--save_summary",
        default="",
        help="保存 summary 到指定 JSON 文件",
    )

    args = ap.parse_args()

    if not os.path.exists(args.index):
        raise RuntimeError(f"index 文件不存在: {args.index}")

    if not os.path.exists(args.hdf5):
        raise RuntimeError(f"HDF5 文件不存在: {args.hdf5}")

    evaluate(args)


if __name__ == "__main__":
    main()

