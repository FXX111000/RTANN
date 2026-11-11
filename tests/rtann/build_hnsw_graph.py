#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 hnswlib 从 data/ 下的公开数据集构建 HNSW 图，保存为 hnswlib 原生 save_index() 格式。
后续可经 tools/hnswlib-converter from-hnsw 转成 exchange 文件，再 vecfs-index import 进本系统。

用法示例（需使用带 hnswlib 的 conda 环境）：
  /home/fxx/.conda/envs/hnsw/bin/python tests/rtann/build_hnsw_graph.py \
      --hdf5 data/deep-image-96-angular.hdf5 \
      --space ip --M 16 --ef-construction 400 \
      --output data/deep-image-96-angular-M16-M0-32-efc400.bin

  /home/fxx/.conda/envs/hnsw/bin/python tests/rtann/build_hnsw_graph.py \
      --hdf5 data/gist-960-euclidean.hdf5 \
      --space l2 --M 16 --ef-construction 400 \
      --output data/gist-960-euclidean-M16-M0-32-efc400.bin

说明：
  - hnswlib Python API 未暴露 maxM0，其 C++ 内部固定 maxM0 = 2*M。
    因此 M=16 即对应 M0=32（与论文/用户历史参数一致）。
  - deep-image 为 angular（向量已 L2 归一化），用 space=ip（内积 = 余弦）。
    gist 为 euclidean，用 space=l2。
  - 向量维度自动从 HDF5 数据集推断；train/base/database/data 键自动探测。
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np

try:
    import hnswlib
except ModuleNotFoundError:
    sys.stderr.write(
        "[fatal] 未找到 hnswlib。请使用带 hnswlib 的环境，例如：\n"
        "  /home/fxx/.conda/envs/hnsw/bin/python tests/rtann/build_hnsw_graph.py ...\n"
    )
    sys.exit(1)


def find_dataset(f: h5py.File, candidates):
    for name in candidates:
        if name in f:
            return name
    return None


def l2_normalize_fp32(batch: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(batch, axis=1, keepdims=True).astype(np.float32)
    norms[norms == 0] = 1e-12
    out = batch / norms
    out[~np.isfinite(out)] = 0.0
    return np.ascontiguousarray(out, dtype=np.float32)


def compute_recall_at_k(pred, truth, k):
    hits = 0
    for i in range(pred.shape[0]):
        p = set(int(x) for x in pred[i][:k])
        t = set(int(x) for x in truth[i][:k])
        if not t:
            continue
        hits += len(p & t) / float(k)
    return hits / pred.shape[0]


def main():
    ap = argparse.ArgumentParser(description="用 hnswlib 构建 HNSW 图并保存为原生 .bin")
    ap.add_argument("--hdf5", required=True, help="输入 HDF5 数据集路径")
    ap.add_argument("--train-key", default="",
                    help="base 向量数据集键，留空自动探测 train/base/database/data")
    ap.add_argument("--space", choices=["ip", "cosine", "l2"], default="ip",
                    help="距离空间，必须与后续 converter --distance 一致")
    ap.add_argument("--M", type=int, default=16, help="HNSW M（M0=2*M 自动确定）")
    ap.add_argument("--ef-construction", type=int, default=400, help="构建时 ef")
    ap.add_argument("--ef-search", type=int, default=200, help="验证召回时查询 ef")
    ap.add_argument("--normalize", choices=["auto", "none", "always"], default="auto",
                    help="auto: ip/cosine 且向量未归一化时归一化")
    ap.add_argument("--max-elements", type=int, default=0,
                    help="最多构建多少条 base 向量（0=全部；用于快速自测）")
    ap.add_argument("--chunk", type=int, default=100000, help="分块 add_items 的块大小")
    ap.add_argument("--output", required=True, help="输出 .bin 路径")
    ap.add_argument("--random-seed", type=int, default=100)
    ap.add_argument("--verify-recall", action="store_true",
                    help="构建后加载 test/neighbors 计算 recall@k（需要 HDF5 含真值）")
    ap.add_argument("--k", type=int, default=10, help="验证 recall 时的 k")
    args = ap.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        key = args.train_key or find_dataset(f, ["train", "base", "database", "data"])
        if key is None:
            raise RuntimeError("未在 HDF5 中找到 train/base/database/data 数据集")

        ds = f[key]
        n_total, dim = int(ds.shape[0]), int(ds.shape[1])
        n = n_total if args.max_elements <= 0 else min(n_total, args.max_elements)

        print(f"[load] dataset={key} shape=({n_total}, {dim}) build={n}")
        print(f"[cfg ] space={args.space} M={args.M} M0={2 * args.M} "
              f"ef_construction={args.ef_construction}")

        # 归一化决策
        do_normalize = False
        if args.normalize == "always":
            do_normalize = True
        elif args.normalize == "auto":
            if args.space in ("ip", "cosine"):
                sample = np.asarray(ds[:min(500, n_total)], dtype=np.float32)
                mean_norm = float(np.mean(np.linalg.norm(sample, axis=1)))
                do_normalize = abs(mean_norm - 1.0) > 1e-3
                print(f"[norm] sample mean norm={mean_norm:.6f} -> normalize={do_normalize}")
            else:
                do_normalize = False

        index = hnswlib.Index(space=args.space, dim=dim)
        index.init_index(
            max_elements=n,
            M=args.M,
            ef_construction=args.ef_construction,
            random_seed=args.random_seed,
        )

        t0 = time.time()
        for start in range(0, n, args.chunk):
            end = min(start + args.chunk, n)
            batch = np.asarray(ds[start:end], dtype=np.float32)
            if do_normalize:
                batch = l2_normalize_fp32(batch)
            batch = np.ascontiguousarray(batch, dtype=np.float32)
            index.add_items(batch)

            elapsed = time.time() - t0
            rate = (end) / elapsed if elapsed > 0 else 0.0
            print(f"[build] {end}/{n} ({end / n * 100:.1f}%) "
                  f"elapsed={elapsed:.0f}s rate={rate:.0f} vec/s", flush=True)

        build_s = time.time() - t0
        print(f"[build] 完成，耗时 {build_s:.1f}s")

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        index.save_index(args.output)
        print(f"[save] -> {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)")

        # 可选：召回验证
        if args.verify_recall:
            test_key = find_dataset(f, ["test", "query", "queries"])
            nb_key = find_dataset(f, ["neighbors", "knn", "groundtruth", "true_neighbors"])
            if test_key is None or nb_key is None:
                print("[verify] 跳过：HDF5 缺 test/neighbors 真值")
            else:
                q = np.asarray(f[test_key], dtype=np.float32)
                truth = np.asarray(f[nb_key], dtype=np.int64)
                if do_normalize:
                    q = l2_normalize_fp32(q)
                q = np.ascontiguousarray(q, dtype=np.float32)
                index.set_ef(args.ef_search)
                pred, _ = index.knn_query(q, k=args.k)
                rec = compute_recall_at_k(pred, truth, args.k)
                print(f"[verify] recall@{args.k} (ef_search={args.ef_search}) = {rec * 100:.3f}%")


if __name__ == "__main__":
    main()
