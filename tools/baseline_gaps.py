#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""baseline_gaps.py — "模型 vs 固定默认"对照（零 GPU，纯查日志）

回答审稿人的第一个问题：你离最优 4%，那"不思考"会多差？
  模型：decide_dev 的决策
  固定默认：每个族一个"教科书式"配置（GEMM 128×128/TK32/S2；GEMV T64/TK32/S2；stencil T64/S2）
"""
import os
import statistics as st
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import decide_dev as D  # noqa: E402
import replay_dev as R  # noqa: E402

DEFAULT = {"gemm": (128, 128, 32, 2), "gemv": (64, 32, 2), "stencil1": (64, 2)}


def main():
    print(f"{'kernel':10s} {'最优':>9s} {'模型':>9s} {'模型距优':>9s} "
          f"{'固定默认':>9s} {'默认距优':>9s}")
    gm, gd = [], []
    for name, path, fam, note in R.ROWS:
        cfg = D.decide(D.DEV_KERNELS[name][0], "hard")["config"]
        tbl = R.parse(path, fam)
        if not cfg or not tbl:
            continue
        best = min(tbl.values())
        tm = tbl.get(R.key_of(cfg, fam))
        td = tbl.get(DEFAULT[fam])
        if tm is None:
            continue
        g_model = tm / best - 1.0
        gm.append(g_model)
        if td is None:
            print(f"{name:10s} {best:9.1f} {tm:9.1f} {g_model*100:8.1f}% "
                  f"{'—':>9s} {'不可行/未测':>9s}")
            continue
        g_def = td / best - 1.0
        gd.append(g_def)
        print(f"{name:10s} {best:9.1f} {tm:9.1f} {g_model*100:8.1f}% "
              f"{td:9.1f} {g_def*100:8.1f}%")
    print()
    print(f"模型距优：中位 {st.median(gm)*100:+.1f}%  最差 {max(gm)*100:+.1f}%  n={len(gm)}")
    if gd:
        print(f"固定默认：中位 {st.median(gd)*100:+.1f}%  最差 {max(gd)*100:+.1f}%  n={len(gd)}")
    print(f"\n固定默认（各族一个固定配置）：GEMM {DEFAULT['gemm']} / "
          f"GEMV {DEFAULT['gemv']} / stencil {DEFAULT['stencil1']}")


if __name__ == "__main__":
    main()
