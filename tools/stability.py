#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stability.py — 同一 kernel、同一配置、跨多次会话的复现性，按"活跃块数"分组。

动机：3mm 的 128×128/TK16 家族三轮测出三套完全不同的排名（471/348/244/330 →
255/459/243/243 → 284/244/337/249），而 64×64 的胜者三轮都是 75.1/75.3/75.3。
怀疑是"活跃块数远小于 SM 数"的配置对共享卡干扰极其敏感。

用法：python tools/stability.py 3mm_512.log 3mm_rerun1.log 3mm_rerun2.log
       （文件名相对 daima/xfer_2026-11-15，或给绝对路径；支持 --dir）
"""
import os
import re
import statistics as st
import sys

BASE = "daima/xfer_2026-11-15"
CFG = re.compile(r"BM=(\d+)\s+BN=(\d+)\s+TK=(\d+)\s+S=(\d+)\s+smem=\s*[\d.]+KB\s+"
                 r"耗时:\s+([\d.]+)\s+us")


def parse(p):
    out = {}
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for m in CFG.finditer(f.read()):
            k = tuple(int(m.group(i)) for i in range(1, 5))
            out[k] = float(m.group(5))
    return out


def main():
    paths = [a for a in sys.argv[1:] if a.endswith(".log")]
    if not paths:
        print("用法: python tools/stability.py <log1> <log2> ...")
        return
    runs = []
    for p in paths:
        fp = p if os.path.exists(p) else os.path.join(BASE, p)
        runs.append((os.path.basename(fp), parse(fp)))
    keys = set(runs[0][1]) & set(runs[1][1])
    for _, d in runs[2:]:
        keys &= set(d)
    if not keys:
        print("没有共同配置")
        return

    size = 512
    rows = []
    for k in keys:
        us = [d[k] for _, d in runs]
        blocks = (size // k[0]) * (size // k[1])
        rows.append((blocks, k, us, max(us) / min(us) - 1.0))

    print("=" * 96)
    print(f"跨 {len(runs)} 次会话的逐配置复现性（{', '.join(n for n, _ in runs)}）")
    print("=" * 96)
    print(f"{'活跃块':>6s} {'配置':22s} {'各轮耗时(us)':38s} {'极差':>7s}")
    for blocks, k, us, sp in sorted(rows, key=lambda r: -r[3]):
        s = " / ".join(f"{u:6.1f}" for u in us)
        print(f"{blocks:6d} {str(k):22s} {s:38s} {sp*100:6.1f}%")

    print()
    print("=" * 96)
    print("按活跃块数分组的极差（这是'低占用=噪声放大器'的直接证据）")
    print("=" * 96)
    for lo, hi, tag in [(0, 16, "≤16 块（≤1/8 卡）"), (17, 63, "17–63 块"),
                        (64, 10**9, "≥64 块（≥半卡）")]:
        g = [sp for b, _, _, sp in rows if lo <= b <= hi]
        if g:
            print(f"  {tag:18s} n={len(g):3d}  中位极差 {st.median(g)*100:5.1f}%  "
                  f"最大 {max(g)*100:5.1f}%")
    print("\n判读：若低块数组的极差显著高于高块数组，则'名次'在低占用配置上不可复现，")
    print("      任何跨会话的 rank/距优 结论都必须限定在该配置的活跃块数 ≥ 半卡。")


if __name__ == "__main__":
    main()
