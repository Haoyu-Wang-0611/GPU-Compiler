#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""l2_regime.py — L2 驻留 regime 项：改进绝对误差，但不动决策？（零 GPU）

两个问题
  Q1 加 L2_BOOST 后，决策变不变？
  Q2 加 L2_BOOST 后，预测/实测的比值（绝对误差）改善多少？
若 Q1=不变、Q2=明显改善，则该项是纯收益：它不承担决策职责，只修正量纲。
"""
import os
import statistics as st
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import decide_dev as D  # noqa: E402
import replay_dev as R  # noqa: E402

BOOSTS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def measured(name, cfg):
    path, fam = LOG[name]
    tbl = R.parse(path, fam)
    if not tbl or not cfg:
        return None
    return tbl.get(R.key_of(cfg, fam))


LOG = {n: (p, f) for n, p, f, _ in R.ROWS}
NAMES = list(LOG.keys())


def main():
    print("Q1 决策是否随 L2_BOOST 变化")
    base = None
    for b in BOOSTS:
        D.L2_BOOST = b
        dec = {n: tuple(sorted((D.decide(D.DEV_KERNELS[n][0], "hard")["config"] or {}).items()))
               for n in NAMES}
        if base is None:
            base = dec
        diff = [n for n in NAMES if dec[n] != base[n]]
        print(f"  L2_BOOST={b:<4} 变化的行：{len(diff)}  {', '.join(diff) or '（无）'}")

    print("\nQ2 绝对误差（预测/实测 − 1）")
    print(f"{'kernel':10s} {'实测(µs)':>9s} {'β=1':>8s} {'β=2':>8s} {'β=3':>8s} {'β=6':>8s}")
    ratios = {1.0: [], 2.0: [], 3.0: [], 6.0: []}
    for n in NAMES:
        prof = D.DEV_KERNELS[n][0]
        row = []
        for b in (1.0, 2.0, 3.0, 6.0):
            D.L2_BOOST = b
            r = D.decide(prof, "hard")
            t = measured(n, r["config"])
            if t is None or r["predicted_ns"] is None:
                row.append(None)
                continue
            ratio = r["predicted_ns"] / (t * 1000.0) - 1.0
            ratios[b].append(ratio)
            row.append(ratio)
        t0 = measured(n, D.decide(prof, "hard")["config"])
        cells = " ".join(f"{x*100:+7.1f}%" if x is not None else "     —" for x in row)
        print(f"{n:10s} {(t0 if t0 else 0):9.1f} {cells}")

    print()
    for b in (1.0, 2.0, 3.0, 6.0):
        v = [abs(x) for x in ratios[b]]
        if v:
            print(f"  β={b:<4} 绝对误差中位 {st.median(v)*100:5.1f}%   最差 {max(v)*100:5.1f}%")


if __name__ == "__main__":
    main()
