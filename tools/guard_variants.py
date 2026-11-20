#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""guard_variants.py — 占用守卫的三个变体（零 GPU）

  G1 hard : 块数 ≥ SM/2 = 64（现模型）
  G2 full : 块数 ≥ SM = 128（占满机；一个都不可行时退回 64）
  G3 none : 不设占用门槛，完全靠成本模型自己排序

问题：守卫是必要的，还是冗余的？
若 G3 与 G1 决策相同 → 守卫冗余，可删（老师的"启发式混进可行域"批评自动消失）。
"""
import os
import statistics as st
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import decide_dev as D  # noqa: E402
import replay_dev as R  # noqa: E402

# 只跑 GEMM 族（GEMV/stencil 的守卫在 l1model 内部分支，且它们本来就是 rank1）
ROWS = ["gemm4096", "gemm1024", "gemm512", "gemm500", "twomm", "three_mm", "syrk", "symm"]
LOG = {n: (p, f) for n, p, f, _ in R.ROWS}
VARIANTS = [("G1 hard", "hard"), ("G2 full", "full"), ("G3 none", "none")]


def fmt(c):
    return "/".join(str(c[k]) for k in ("BM", "BN", "TK", "T") if k in c) + \
           (f"/{c['S']}" if "S" in c else "")


def gap(name, cfg):
    path, fam = LOG[name]
    tbl = R.parse(path, fam)
    if not tbl or not cfg:
        return None
    best = min(tbl.values())
    t = tbl.get(R.key_of(cfg, fam))
    return None if t is None else t / best - 1.0


def main():
    print(f"{'kernel':10s} " + " | ".join(f"{v[0]:>20s}" for v in VARIANTS))
    print("-" * (11 + 23 * len(VARIANTS)))
    per = {v[1]: [] for v in VARIANTS}
    dec = {v[1]: {} for v in VARIANTS}
    for n in ROWS:
        cells = []
        for label, occ in VARIANTS:
            r = D.decide(D.DEV_KERNELS[n][0], occ)
            c = r["config"] or {}
            dec[occ][n] = fmt(c)
            g = gap(n, c)
            per[occ].append(g)
            cells.append(f"{fmt(c):>14s} {(f'{g*100:+5.1f}%' if g is not None else '未测'):>5s}")
        print(f"{n:10s} " + " | ".join(cells))
    print()
    for label, occ in VARIANTS:
        v = [x for x in per[occ] if x is not None]
        print(f"{label:8s} 中位 {st.median(v)*100:+5.1f}%   最差 {max(v)*100:+5.1f}%   "
              f"n={len(v)}")
    print("\n与 G1 相比决策变化：")
    for label, occ in VARIANTS[1:]:
        diff = [n for n in ROWS if dec[occ][n] != dec["hard"][n]]
        print(f"  {label}: {len(diff)}/{len(ROWS)}  " + (", ".join(diff) or "（无）"))


if __name__ == "__main__":
    main()
