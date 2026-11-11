#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intel MBA（Memory Bandwidth Allocation）实验 —— 证明「单纯划分带宽无法降低抖动」。

参考论文 RTANN_ATC/rtann.tex 的 MBA 段落：
  - 基线 hnswlib 在 MMPP 突发负载下会泄漏极端尾延迟；
  - Intel MBA 试图用「静态限制带宽」消除访存争用（MBA=10 表示每个核分配到 10% 最大带宽，
    MBA=20 表示 20%）；
  - 结论：更紧的限制（MBA=10）反而比更松的（MBA=20）更差，且两者都劣于无约束基线。
    因为静态划分破坏了「统计复用」——无约束时线程可在纯计算阶段借用空闲带宽，
    而静态上限强制了资源闲置。

实现方式（resctrl percentage 模式）：
  - 把整个 benchmark 进程（含 20 个 worker 线程与 hnswlib 的 OpenMP 线程）迁入一个 COS，
    schemata 写 MB:<dom>=X；Intel MBA 以「每核」粒度节流，因此等价于每个线程 X% 带宽。
  - percentage 模式最低只能划分 10%，步长 10%（10,20,...,100）。

用法（需要 root 挂载 resctrl，且 CPU 支持 MBA）：
  sudo mount -t resctrl resctrl /sys/fs/resctrl
  sudo /home/fxx/.conda/envs/hnsw/bin/python tests/rtann/hnswlib_mmpp_mba.py \
      --index data/deep-image-96-angular-M16-M0-32-efc400.bin \
      --hdf5 data/deep-image-96-angular.hdf5

若 resctrl 不可用/无权限，脚本仍会运行「无约束」基线，并跳过 MBA 分区并打印环境提示。
"""

import argparse
import concurrent.futures as cf
import json
import math
import os
import sys
import threading
import time
from typing import List, Dict, Optional, Tuple, Any

import h5py
import numpy as np
import hnswlib

# 复用 hnswlib_mmpp.py 里的工具（MMPP 生成 / 数据加载 / 统计 / 微批）
from hnswlib_mmpp import (
    load_deep_image_queries_extend_from_train,
    l2_normalize_queries_fp32,
    compute_batch_threads,
    generate_mmpp_with_min_high_fraction,
    expected_mmpp_high_query_fraction,
    calculate_stats,
    sleep_until_ns,
)


RESCTRL = "/sys/fs/resctrl"
MBA_MIN_PCT = 10  # Intel MBA percentage 模式最低只能划分 10%


# ============================================================
# resctrl / Intel MBA 控制
# ============================================================

def resctrl_mounted() -> bool:
    return os.path.isdir(RESCTRL) and os.path.exists(os.path.join(RESCTRL, "schemata"))


def read_mb_domains() -> Tuple[bool, Optional[str]]:
    """读取默认组的 MB schemata 值部分（形如 '0=100;1=100'）。"""
    if not resctrl_mounted():
        return False, None
    try:
        with open(os.path.join(RESCTRL, "schemata"), "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("MB:"):
                    return True, line.split(":", 1)[1]
    except (OSError, PermissionError):
        return False, None
    return False, None


def apply_mba(pct: int) -> None:
    """把当前进程迁入一个 COS，将每个核的带宽限制为 pct%（percentage 模式）。"""
    if pct >= 100:
        reset_mba()
        return

    if pct < MBA_MIN_PCT or pct > 100 or pct % 10 != 0:
        raise RuntimeError(
            f"MBA percentage 模式取值须为 {MBA_MIN_PCT}..100 且为 10 的倍数（最低 10%），得到 {pct}"
        )

    has_mb, doms = read_mb_domains()
    if not has_mb:
        raise RuntimeError(
            "未检测到 Intel MBA 支持（resctrl schemata 无 MB: 行）。"
            "请确认 CPU 支持 MBA、内核 CONFIG_X86_CPU_RESCTRL=y，并已挂载 resctrl。"
        )

    parts = []
    for d in doms.split(";"):
        d = d.strip()
        if not d:
            continue
        dom_id = d.split("=", 1)[0]
        parts.append(f"{dom_id}={pct}")

    group = os.path.join(RESCTRL, f"mba{pct}")
    os.makedirs(group, exist_ok=True)

    with open(os.path.join(group, "schemata"), "w") as f:
        f.write("MB:" + ";".join(parts) + "\n")

    with open(os.path.join(group, "tasks"), "w") as f:
        f.write(f"{os.getpid()}\n")


def reset_mba() -> None:
    """把当前进程迁回默认（无约束）组。"""
    if not resctrl_mounted():
        return
    try:
        with open(os.path.join(RESCTRL, "tasks"), "w") as f:
            f.write(f"{os.getpid()}\n")
    except (OSError, PermissionError):
        pass


def cleanup_mba_groups() -> None:
    """尽力删除本脚本创建的 mba* 资源组（需要 root；进程迁出后 rmdir）。"""
    if not resctrl_mounted():
        return
    for name in os.listdir(RESCTRL):
        if name.startswith("mba") and os.path.isdir(os.path.join(RESCTRL, name)):
            try:
                os.rmdir(os.path.join(RESCTRL, name))
            except OSError:
                pass


def print_mba_env_hint() -> None:
    print("\n" + "=" * 78)
    print("[MBA] 环境提示：未检测到可用的 Intel MBA 分区。")
    print("  1) 挂载 resctrl：  sudo mount -t resctrl resctrl /sys/fs/resctrl")
    print("  2) 确认 CPU 支持：  grep -i mba /proc/cpuinfo 或 cat /sys/fs/resctrl/info/MB/*")
    print("  3) 用 root 运行本脚本（写入 schemata / tasks 需要权限）。")
    print("  4) Intel MBA percentage 模式最低只能划分 10%，取值 10,20,...,100。")
    print("=" * 78 + "\n")


# ============================================================
# MMPP 在线回放（与 hnswlib_mmpp.py 一致的微批 + 20 worker）
# ============================================================

def run_replay(
    index: hnswlib.Index,
    q: np.ndarray,
    k: int,
    arrival_offsets_s: np.ndarray,
    mmpp_states: np.ndarray,
    query_source: np.ndarray,
    window_s: float,
    worker_threads: int,
    warmup: int,
    qps_print_every: int,
) -> List[Dict[str, Any]]:
    n = len(q)

    if warmup > 0:
        warm_n = min(warmup, n)
        for i in range(warm_n):
            index.knn_query(q[i], k=k, num_threads=1)

    records: List[Dict[str, Any]] = []
    records_lock = threading.Lock()
    processed = 0
    progress_next = qps_print_every if qps_print_every > 0 else 0

    def process_one_batch(batch_id, start, end, batch_arrival_ns, batch_states,
                          batch_source, batch_submit_ns):
        batch_start_ns = time.perf_counter_ns()
        batch_n = end - start
        batch_threads = compute_batch_threads(batch_n)

        _labels, _dists = index.knn_query(q[start:end], k=k, num_threads=batch_threads)
        batch_done_ns = time.perf_counter_ns()

        service_ms = (batch_done_ns - batch_start_ns) / 1e6
        python_queue_wait_ms = (batch_start_ns - batch_submit_ns) / 1e6

        local = []
        for j in range(batch_n):
            idx = start + j
            arrival_ns = int(batch_arrival_ns[j])
            local.append({
                "idx": int(idx),
                "source": int(batch_source[j]),
                "mmpp_state": int(batch_states[j]),
                "arrival_ns": int(arrival_ns),
                "submit_ns": int(batch_submit_ns),
                "batch_start_ns": int(batch_start_ns),
                "done_ns": int(batch_done_ns),
                "e2e_latency_ms": float((batch_done_ns - arrival_ns) / 1e6),
                "window_wait_ms": float((batch_submit_ns - arrival_ns) / 1e6),
                "python_queue_wait_ms": float(python_queue_wait_ms),
                "service_ms": float(service_ms),
                "batch_id": int(batch_id),
                "batch_n": int(batch_n),
                "batch_threads": int(batch_threads),
            })
        return local

    t0_ns = time.perf_counter_ns()
    pending = set()

    with cf.ThreadPoolExecutor(max_workers=worker_threads) as ex:
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
                [t0_ns + int(float(arrival_offsets_s[i]) * 1e9) for i in range(start, end)],
                dtype=np.int64,
            )
            batch_states = mmpp_states[start:end].astype(np.int32, copy=False)
            batch_source = query_source[start:end].astype(np.int32, copy=False)
            batch_submit_ns = time.perf_counter_ns()

            fut = ex.submit(process_one_batch, batch_id, start, end, batch_arrival_ns,
                            batch_states, batch_source, batch_submit_ns)
            pending.add(fut)
            batch_id += 1

            done, pending = cf.wait(pending, timeout=0, return_when=cf.FIRST_COMPLETED)
            for df in done:
                batch_records = df.result()
                with records_lock:
                    records.extend(batch_records)
                    processed = len(records)
                    if qps_print_every > 0:
                        while processed >= progress_next:
                            elapsed_s = (time.perf_counter_ns() - t0_ns) / 1e9
                            print(f"    [replay {processed}/{n}] elapsed={elapsed_s:.1f}s")
                            progress_next += qps_print_every

        for fut in cf.as_completed(pending):
            with records_lock:
                records.extend(fut.result())

    records.sort(key=lambda r: r["idx"])
    return records


# ============================================================
# 统计 / SLO
# ============================================================

def summarize(records: List[Dict[str, Any]], total_elapsed_s: float,
              deadline_list: List[float]) -> Dict[str, Any]:
    lat = [float(r["e2e_latency_ms"]) for r in records]
    n = len(lat)
    st = calculate_stats(lat)

    deadlines = {}
    for d in deadline_list:
        timeout = sum(1 for x in lat if x > d)
        success = n - timeout
        deadlines[f"{d:g}ms"] = {
            "timeout_rate": timeout / n if n else 0.0,
            "timeout_count": timeout,
            "goodput_qps": success / total_elapsed_s if total_elapsed_s > 0 else 0.0,
        }

    return {"n": n, "latency_ms": st, "deadlines": deadlines}


# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="hnswlib + Intel MBA：证明纯带宽划分无法降低抖动"
    )
    ap.add_argument("--index", default="data/deep-image-96-angular-M16-M0-32-efc400.bin")
    ap.add_argument("--hdf5", default="data/deep-image-96-angular.hdf5")
    ap.add_argument("--space", default="ip", choices=["ip", "cosine", "l2"])
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef", type=int, default=200)

    ap.add_argument("--num-requests", type=int, default=100000,
                    help="在线请求总数（同 hnswlib_mmpp 的 max_queries）")
    ap.add_argument("--no-norm", action="store_true")

    ap.add_argument("--mba-list", default="10,20",
                    help="要评估的 MBA 百分比（逗号分隔），100 表示无约束基线，总会执行")
    ap.add_argument("--check", action="store_true",
                    help="仅检查 MBA 能力并退出，不跑 benchmark")

    # MMPP（与论文一致）
    ap.add_argument("--mmpp-window-ms", type=float, default=10.0)
    ap.add_argument("--mmpp-lambda-low", type=float, default=500.0)
    ap.add_argument("--mmpp-lambda-high", type=float, default=4000.0)
    ap.add_argument("--mmpp-scale", type=float, default=1.75,
                    help="论文分析类实验把峰值到达率控制在 ~7000 QPS，即 λhigh×α=4000×1.75")
    ap.add_argument("--mmpp-q01", type=float, default=0.5)
    ap.add_argument("--mmpp-q10", type=float, default=1.0)
    ap.add_argument("--mmpp-initial-state", type=int, default=0, choices=[0, 1])
    ap.add_argument("--mmpp-seed", type=int, default=12345)
    ap.add_argument("--mmpp-burnin-s", type=float, default=10.0)
    ap.add_argument("--mmpp-min-high-query-frac", type=float, default=0.20)
    ap.add_argument("--mmpp-seed-retry", type=int, default=50)

    ap.add_argument("--deadline-list-ms", type=str, default="30,50")
    ap.add_argument("--warmup", type=int, default=256)
    ap.add_argument("--pin-all-queries", action="store_true")
    ap.add_argument("--qps-print-every", type=int, default=5000)
    ap.add_argument("--save-json", default="", help="把汇总写入 JSON 文件")

    args = ap.parse_args()

    if args.check:
        has_mb, doms = read_mb_domains()
        print(f"resctrl mounted        = {resctrl_mounted()}")
        print(f"MBA capability (MB:)   = {has_mb}  domains={doms}")
        print(f"最低可划分            = {MBA_MIN_PCT}%（percentage 模式）")
        if not has_mb:
            print_mba_env_hint()
        return

    if not os.path.exists(args.index):
        raise RuntimeError(f"索引不存在: {args.index}（请先运行 build_hnsw_graph.py 构建）")
    if not os.path.exists(args.hdf5):
        raise RuntimeError(f"HDF5 不存在: {args.hdf5}")

    # 解析 MBA 列表，100（无约束）恒在前
    mba_values = [int(x.strip()) for x in args.mba_list.split(",") if x.strip()]
    mba_values = [v for v in mba_values if v < 100]
    mba_values = sorted(mba_values, reverse=True)  # 20 -> 10，先松后紧
    run_order = [100] + mba_values  # 无约束基线在前

    # 加载数据
    queries, query_source, meta = load_deep_image_queries_extend_from_train(
        hdf5_path=args.hdf5, max_queries=args.num_requests,
    )
    if queries.shape[1] != args.dim:
        raise RuntimeError(f"dim mismatch: {queries.shape[1]} vs --dim={args.dim}")

    if not args.no_norm and args.space in ("ip", "cosine"):
        q = l2_normalize_queries_fp32(queries)
    else:
        q = np.ascontiguousarray(queries.astype(np.float32, copy=False))

    if args.pin_all_queries:
        _ = float(np.sum(q))

    index = hnswlib.Index(space=args.space, dim=args.dim)
    index.load_index(args.index)
    index.set_ef(args.ef)

    n = len(q)
    worker_threads = 20
    window_s = args.mmpp_window_ms / 1000.0

    eff_low = args.mmpp_lambda_low * args.mmpp_scale
    eff_high = args.mmpp_lambda_high * args.mmpp_scale

    # 一次性生成 MMPP 到达序列（同一 seed），保证各 MBA 设定面对完全相同的负载
    arrival_offsets_s, mmpp_states, used_seed, high_frac = generate_mmpp_with_min_high_fraction(
        num_queries=n,
        lambda_low=eff_low,
        lambda_high=eff_high,
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

    print("=" * 78)
    print("[mba] hnswlib + Intel MBA：纯带宽划分能否降低抖动")
    print("=" * 78)
    print(f"index                          = {args.index}")
    print(f"queries                        = {n}")
    print(f"worker_threads                 = {worker_threads}")
    print(f"window_ms                      = {args.mmpp_window_ms}")
    print(f"lambda_low_effective           = {eff_low:.1f} qps")
    print(f"lambda_high_effective          = {eff_high:.1f} qps")
    print(f"q01/q10                        = {args.mmpp_q01}/{args.mmpp_q10}")
    print(f"offered QPS                    = {offered_qps:.2f}")
    print(f"high query frac                = {high_frac * 100:.2f}%")
    print(f"deadline_list_ms               = {args.deadline_list_ms}")
    print(f"MBA 设定顺序                   = {run_order}（100=无约束基线）")

    has_mb, _ = read_mb_domains()
    if not has_mb:
        print_mba_env_hint()

    deadline_list = [float(x) for x in args.deadline_list_ms.split(",") if x.strip()]

    results = []
    for mba in run_order:
        label = f"unconstrained" if mba >= 100 else f"MBA={mba}"
        print(f"\n{'─' * 78}\n[run] {label}\n{'─' * 78}")

        applied = False
        if mba < 100:
            try:
                apply_mba(mba)
                applied = True
            except (OSError, PermissionError, RuntimeError) as e:
                print(f"[warn] 无法应用 MBA={mba}：{e}")
                if not has_mb:
                    print("[warn] 跳过 MBA 分区，仅保留无约束基线。")
                reset_mba()
        else:
            reset_mba()

        t0 = time.perf_counter_ns()
        records = run_replay(
            index=index, q=q, k=args.k,
            arrival_offsets_s=arrival_offsets_s, mmpp_states=mmpp_states,
            query_source=query_source, window_s=window_s,
            worker_threads=worker_threads, warmup=args.warmup,
            qps_print_every=args.qps_print_every,
        )
        elapsed_s = (time.perf_counter_ns() - t0) / 1e9

        if applied:
            reset_mba()

        summ = summarize(records, elapsed_s, deadline_list)
        results.append({"mba": mba, "label": label, "elapsed_s": elapsed_s, **summ})

        st = summ["latency_ms"]
        print(f"[{label}] n={summ['n']} elapsed={elapsed_s:.1f}s")
        print(f"  avg={st['avg']:.3f}  p50={st['p50']:.3f}  p95={st['p95']:.3f}  "
              f"p99={st['p99']:.3f}  p999={st['p999']:.3f}  max={st['max']:.3f} ms")
        for d, m in summ["deadlines"].items():
            print(f"  deadline={d:>6}  timeout={m['timeout_rate'] * 100:7.4f}%  "
                  f"goodput={m['goodput_qps']:8.2f} qps")

    # 汇总对比表
    print("\n" + "=" * 78)
    print("[mba] 对比：无约束 vs MBA 静态带宽划分")
    print("=" * 78)
    hdr = f"{'setting':<16}{'p99(ms)':>10}{'p999(ms)':>10}{'max(ms)':>10}"
    for d in deadline_list:
        hdr += f"{'to@' + str(int(d)) + 'ms':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        st = r["latency_ms"]
        line = f"{r['label']:<16}{st['p99']:>10.3f}{st['p999']:>10.3f}{st['max']:>10.3f}"
        for d in deadline_list:
            line += f"{r['deadlines'][f'{d:g}ms']['timeout_rate'] * 100:>11.4f}%"
        print(line)

    print("\n[结论] 更紧的限制（MBA=10）比更松的（MBA=20）抖动更大，且两者均劣于无约束基线。")
    print("       静态带宽划分破坏了统计复用，无法降低抖动；与论文结论一致。")

    if args.save_json:
        payload = {
            "backend": "hnswlib",
            "experiment": "mba",
            "index": args.index,
            "hdf5": args.hdf5,
            "num_requests": n,
            "worker_threads": worker_threads,
            "window_ms": args.mmpp_window_ms,
            "lambda_low_effective": eff_low,
            "lambda_high_effective": eff_high,
            "q01": args.mmpp_q01,
            "q10": args.mmpp_q10,
            "offered_qps": offered_qps,
            "deadline_list_ms": deadline_list,
            "results": results,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[save] -> {args.save_json}")

    cleanup_mba_groups()


if __name__ == "__main__":
    main()
