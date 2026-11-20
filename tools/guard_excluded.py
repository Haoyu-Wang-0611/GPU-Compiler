#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""guard_excluded.py — 被占用守卫排除的配置，实测到底差多少？（零 GPU）

用途：如果我们要保留"块数 ≥ SM/2"这条限制，必须能回答两个问题：
  1) 它改变决策吗？（已知：8/8 不变）
  2) 被它排除的配置，实测有多差？（本脚本回答）
只有当"被排除的最快配置也明显更慢"时，保留它才是安全的搜索空间收缩，
而不是"误杀了好配置"。
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import replay_dev as R  # noqa: E402

SHAPE = {
    "gemm4096": (4096, 4096), "gemm1024": (1024, 1024), "gemm512": (512, 512),
    "gemm500": (500, 500), "twomm": (512, 512), "three_mm": (512, 512),
    "syrk": (512, 512), "symm": (512, 512),
}


def blocks(key, shape):
    BM, BN = key[0], key[1]
    return math.ceil(shape[0] / BM) * math.ceil(shape[1] / BN)


def main():
    print(f"{'kernel':10s} {'守卫线':>6s} {'排除数':>6s} {'被排除最快(µs)':>14s} "
          f"{'该配置':>16s} {'相对最优':>9s} {'最优(µs)':>9s}")
    worst = []
    for name, shape in SHAPE.items():
        path, fam = next((p, f) for n, p, f, _ in R.ROWS if n == name)
        tbl = R.parse(path, fam)
        if not tbl:
            continue
        best_t = min(tbl.values())
        best_k = min(tbl, key=lambda k: tbl[k])
        exc = {k: v for k, v in tbl.items() if blocks(k, shape) < 64}
        if not exc:
            print(f"{name:10s} {64:>6d} {0:>6d} {'—':>14s} {'—':>16s} {'—':>9s} {best_t:>9.1f}")
            continue
        ek = min(exc, key=lambda k: exc[k])
        et = exc[ek]
        gap = et / best_t - 1.0
        worst.append(gap)
        print(f"{name:10s} {64:>6d} {len(exc):>6d} {et:>14.1f} "
              f"{'/'.join(str(x) for x in ek):>16s} {gap*100:>+8.1f}% {best_t:>9.1f}")
    if worst:
        print(f"\n被排除配置中「最快者」相对最优：最小 {min(worst)*100:+.1f}%，"
              f"中位 {sorted(worst)[len(worst)//2]*100:+.1f}%，最大 {max(worst)*100:+.1f}%")
        print("判读：若最小值也 ≥50%，保留守卫是安全的搜索空间收缩，而非误杀。")


if __name__ == "__main__":
    main()
