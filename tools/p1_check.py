#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p1_check.py — 用所有 oracle 扫参日志检验 b+c 的三条命题（零 GPU、只读日志）。

检验对象（对应 docs/B线 里的 P1/P3）：
  derived_S     : 每个 tile 组取"该组最大可行 S"，再在组间取最优 —— b+c 提议的决策规则
  oracle        : 全空间最优（上界）
  tile_only_S1  : 只用 S=1 的配置（"完全不流水"的极限）
  seq           : 先用 S=1 选 tile，再在该 tile 上选 S（顺序决策）

对每个日志输出：空间跨度、四策略的距优、以及"逐 tile 组"的
`max-S 相对该组最优 S` 的偏差分布（这是 P1 的直接证据）。

用法：python tools/p1_check.py daima/xfer_2026-11-15/*.log
"""
import glob
import os
import re
import statistics as st

G = re.compile(r"BM=(\d+)\s+BN=(\d+)\s+TK=(\d+)\s+S=(\d+)\s+smem=\s*([\d.]+)KB\s+"
               r"耗时:\s*([\d.]+)\s*us")
V = re.compile(r"T=(\d+)\s+TK=(\d+)\s+S=(\d+)\s+smem=\s*([\d.]+)KB\s+"
               r"实测:\s*([\d.]+)\s*us")
S1 = re.compile(r"T=(\d+)\s+S=(\d+)\s+smem=\s*([\d.]+)KB\s+"
                r"实测:\s*([\d.]+)\s*us")
SIZE = re.compile(r"M=N=K=(\d+)")
NPAT = re.compile(r"N=(\d+)")


def parse(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    fam, cfgs = None, []
    for m in G.finditer(text):
        fam = "gemm"
        cfgs.append(dict(axes=(int(m.group(1)), int(m.group(2)), int(m.group(3))),
                         S=int(m.group(4)), us=float(m.group(6))))
    if not cfgs:
        for m in V.finditer(text):
            fam = "gemv"
            cfgs.append(dict(axes=(int(m.group(1)), int(m.group(2))),
                             S=int(m.group(3)), us=float(m.group(5))))
    if not cfgs:
        for m in S1.finditer(text):
            fam = "stencil"
            cfgs.append(dict(axes=(int(m.group(1)),), S=int(m.group(2)),
                             us=float(m.group(4))))
    sz = SIZE.search(text) or NPAT.search(text)
    return dict(path=path, fam=fam, cfgs=cfgs,
                size=int(sz.group(1)) if sz else None,
                title=os.path.basename(path))


def analyse(d):
    cfgs = d["cfgs"]
    if len(cfgs) < 4:
        return None
    best = min(cfgs, key=lambda c: c["us"])
    worst = max(cfgs, key=lambda c: c["us"])
    groups = {}
    for c in cfgs:
        groups.setdefault(c["axes"], []).append(c)

    # 逐组：最大可行 S vs 该组最优 S
    gaps, maxs = [], []
    for ax, cs in groups.items():
        mx = max(cs, key=lambda c: c["S"])          # 该组最大 S（= 最大可行）
        bs = min(cs, key=lambda c: c["us"])
        gaps.append((ax, mx["S"], mx["us"] / bs["us"] - 1.0, len(cs)))
        maxs.append(mx)
    gaps.sort(key=lambda g: -g[2])

    # derived_S：每组取最大 S，再组间取最优
    der_us = min(c["us"] for c in maxs)

    s1 = [c for c in cfgs if c["S"] == 1]
    t1 = min(s1, key=lambda c: c["us"]) if s1 else None
    seq = min([c for c in cfgs if c["axes"] == t1["axes"]], key=lambda c: c["us"]) \
        if t1 else None

    # 最优 tile 组上，"S=1" 相对"该组最优 S"损失多少（二值深度的价值）
    bt = min(groups, key=lambda ax: min(c["us"] for c in groups[ax]))
    bt_cs = groups[bt]
    bt_s1 = [c for c in bt_cs if c["S"] == 1]
    binary = (bt_s1[0]["us"] / min(c["us"] for c in bt_cs) - 1.0) if bt_s1 else None

    return dict(d=d, n=len(cfgs), best=best, worst=worst, groups=len(groups),
                spread=worst["us"] / best["us"], gaps=gaps, derived=der_us,
                tile_only=(t1["us"] if t1 else None),
                seq=(seq["us"] if seq else None), binary=binary,
                best_tile=bt, top_tile_span=(
                    min(c["us"] for c in bt_cs) / best["us"] - 1.0))


def main():
    import sys
    paths = sys.argv[1:] or glob.glob("daima/xfer_2026-11-15/*.log")
    rows = []
    for p in sorted(paths):
        d = parse(p)
        r = analyse(d)
        if r:
            rows.append(r)

    print("=" * 104)
    print("表 1  逐日志：空间跨度与四策略（全部相对该日志的实测最优）")
    print("=" * 104)
    print(f"{'日志':26s} {'族':8s} {'尺寸':6s} {'n':>4s} {'组':>3s} {'跨度':>7s} "
          f"{'derived':>9s} {'tile_only':>10s} {'seq':>8s} {'二值S=1':>8s} "
          f"{'最优tile组':>10s}  最优配置")
    for r in rows:
        nm = r["d"]["title"].replace(".log", "")
        gap = lambda x: f"{x*100:+.1f}%" if x is not None else "—"
        b = r["best"]["us"]
        bx = r["best"]["axes"]
        print(f"{nm:26s} {str(r['d']['fam']):8s} {str(r['d']['size']):6s} "
              f"{r['n']:4d} {r['groups']:3d} {r['spread']:6.2f}x "
              f"{gap(r['derived']/b - 1):>9s} "
              f"{gap((r['tile_only']/b - 1) if r['tile_only'] else None):>10s} "
              f"{gap((r['seq']/b - 1) if r['seq'] else None):>8s} "
              f"{gap(r['binary']):>8s} "
              f"{gap(r['top_tile_span']):>10s}  "
              f"{bx}/S{r['best']['S']} @ {b:.1f}us")

    print()
    print("=" * 104)
    print("表 2  P1 直接证据：每个 tile 组「最大可行 S」相对「该组最优 S」的偏差")
    print("=" * 104)
    allg = []
    for r in rows:
        g = [x[2] for x in r["gaps"]]
        allg += g
        nm = r["d"]["title"].replace(".log", "")
        print(f"{nm:26s} 组数={len(g):3d}  中位={st.median(g)*100:5.1f}%  "
              f"最大={max(g)*100:5.1f}%  ≤2%的组={sum(1 for x in g if x<=0.02)}"
              f"/{len(g)}  最坏组={r['gaps'][0][0]}/S={r['gaps'][0][1]}")
    print(f"\n合计 {len(allg)} 个 tile 组：中位 {st.median(allg)*100:.1f}%  "
          f"均值 {st.mean(allg)*100:.1f}%  最大 {max(allg)*100:.1f}%  "
          f"≤2% 占 {sum(1 for x in allg if x<=0.02)/len(allg)*100:.0f}%  "
          f"≤5% 占 {sum(1 for x in allg if x<=0.05)/len(allg)*100:.0f}%")

    print()
    print("=" * 104)
    print("表 3  最坏 10 个 tile 组（P1 若被证伪，就死在这些点上）")
    print("=" * 104)
    flat = [(r["d"]["title"], g) for r in rows for g in r["gaps"]]
    flat.sort(key=lambda t: -t[1][2])
    for nm, g in flat[:10]:
        print(f"  {nm:26s} tile={g[0]}  最大S={g[1]}  "
              f"相对该组最优 +{g[2]*100:.1f}%  (组内 {g[3]} 配置)")


if __name__ == "__main__":
    main()
