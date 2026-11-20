#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cchunk_sweep.py — 检验"每 chunk 固定开销"对决策的影响（零 GPU）

第①步的结论是：cChunk 的"值"无法从现有数据里干净反解（斜率随尺寸 162→527 ns 变化）。
但这不一定妨碍使用它——真正要问的是：
    **决策对 cChunk 的取值敏感吗？**
若在 [100, 400] ns 内决策稳定（512/500/1024/syrk/symm 全部转到 TK=64，且 4096 不动），
那就可以把它写成"有界项"（物理下界 + 敏感性），而不是"标定出的拟合常数"。
"""
import os
import statistics as st
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import decide_dev as D  # noqa: E402
import replay_dev as R  # noqa: E402

ROWS = ["gemm4096", "gemm1024", "gemm512", "gemm500", "syrk", "symm"]
GRID = [0, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]
LOG = {n: (p, f) for n, p, f, _ in R.ROWS}


def gap_of(name, cfg, fam):
    path, f = LOG[name]
    tbl = R.parse(path, f)
    if not tbl or not cfg:
        return None
    best = min(tbl.values())
    t = tbl.get(R.key_of(cfg, f))
    return None if t is None else t / best - 1.0


def main():
    print("cChunk(ns) | " + " | ".join(f"{n:>18s}" for n in ROWS))
    print("-" * (12 + 21 * len(ROWS)))
    base = None
    for cc in GRID:
        cells, decs = [], {}
        for n in ROWS:
            r = D.decide(D.DEV_KERNELS[n][0], "hard", cc)
            c = r["config"] or {}
            key = tuple(c.get(k) for k in ("BM", "BN", "TK", "S"))
            decs[n] = key
            g = gap_of(n, c, LOG[n][1])
            cells.append(f"{'/'.join(str(x) for x in key if x):>13s} {'' if g is None else f'{g*100:+5.1f}%':>5s}")
        if base is None:
            base = decs
        flips = sum(1 for n in ROWS if decs[n] != base[n])
        print(f"{cc:>10d} | " + " | ".join(cells) + f"   (flips={flips})")

    print("\n逐行看：决策在哪个 cChunk 区间开始稳定？")
    for n in ROWS:
        seq = []
        for cc in GRID:
            c = D.decide(D.DEV_KERNELS[n][0], "hard", cc)["config"] or {}
            seq.append(tuple(c.get(k) for k in ("BM", "BN", "TK", "S")))
        # 找到最后一个变化点之后的稳定值
        stable_from = 0
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                stable_from = GRID[i]
        tk = seq[-1][2]
        print(f"  {n:9s} 末值 TK={tk:<3d} 最后一次变化在 cChunk={stable_from:>3d} ns → "
              f"{'稳定' if stable_from <= 200 else '⚠ 仍在变'}")


if __name__ == "__main__":
    main()
