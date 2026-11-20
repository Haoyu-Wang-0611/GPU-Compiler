#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sensitivity.py — 常数敏感性（零 GPU）：7 个标定常数各动 ±20%，决策变不变？

为什么要做：留出集防"对 kernel 过拟合"，防不了"常数是对着开发集调的"。
若决策对常数不敏感，就能说"决策由约束与流量结构主导，不由常数值主导"。

区分两类量（这点决定分析是否严谨）：
  硬件事实（不扰动）：smemCap 99KB、numSms 128、每线程 4×4/256 线程映射
  标定常数（扰动）：smBw、smCompute、rExposed、pExp、deltaFloor、launch、Lmin(Gemm/Gemv)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import decide_dev as D  # noqa: E402
import l1model as L  # noqa: E402

CONST = ["smBw", "smCompute", "rExposed", "pExp", "deltaFloor", "launch",
         "LminGemm", "LminGemv"]


def decisions():
    out = {}
    for name, (prof, *_rest) in D.DEV_KERNELS.items():
        r = D.decide(prof, "hard")
        out[name] = tuple(sorted((r["config"] or {}).items()))
    return out


def main():
    base = decisions()
    print(f"{'常数':12s} {'-20%':>10s} {'+20%':>10s}   变化的 kernel")
    total = 0
    for c in CONST:
        orig = L.HW[c]
        changed = []
        for factor, tag in ((0.8, "-"), (1.2, "+")):
            L.HW[c] = orig * factor
            d = decisions()
            diff = [k for k in base if d[k] != base[k]]
            if tag == "-":
                minus = diff
            else:
                plus = diff
            L.HW[c] = orig
        allch = sorted(set(minus) | set(plus))
        total += len(allch)
        print(f"{c:12s} {len(minus):>10d} {len(plus):>10d}   {', '.join(allch) or '（无）'}")
    print(f"\n合计「常数±20% 会改变决策」的 (常数, kernel) 组合：{total}")


if __name__ == "__main__":
    main()
