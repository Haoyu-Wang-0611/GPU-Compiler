#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cchunk_probe.py — 从实测日志反解"每 chunk 固定开销" cChunk（零 GPU）

原理
  对固定的 (tile, S)，把 block 时间写成 chunks 的仿射函数：
      block = a + chunks · (每 chunk 代价)
  斜率 = 每 chunk 代价 = 模型里已知的 copy/comp 部分 + 未知的固定开销 cChunk
  于是 cChunk = 斜率 − 模型每 chunk 工作项

为什么要跨尺寸
  如果 512 / 500 / 1024（都在 L2 内）反解出同一个 cChunk，它就是物理常数
  （cp.async commit/wait + barrier 的成本）；只有 512 对得上就说明是拟合。
  4096 作为负对照：工作集 192 MB 走 DRAM，cChunk 允许不同（会单独解释）。
"""
import os
import re
import statistics as st
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import l1model as L  # noqa: E402
import replay_dev as R  # noqa: E402

HW = L.HW
G = re.compile(r"BM=(\d+)\s+BN=(\d+)\s+TK=(\d+)\s+S=(\d+)\s+smem=\s*([\d.]+)KB\s+"
               r"耗时:\s+([\d.]+)\s+us")

LOGS = [
    (512, os.path.join(WS, "daima", "oracle_logs", "2026-11-15", "sweep_512_clean.log"), "干净卡"),
    (500, os.path.join(WS, "daima", "xfer_2026-11-15", "sweep_500.log"), "⚠ 会话未标注"),
    (1024, os.path.join(WS, "daima", "oracle_logs", "2026-11-15", "sweep_1024.log"), "干净卡"),
    (4096, os.path.join(WS, "daima", "xfer_2026-11-15", "sweep_run1.log"), "负对照（DRAM）"),
]


def load(path):
    out = {}
    if not os.path.exists(path):
        return None
    txt = open(path, encoding="utf-8", errors="replace").read()
    for m in G.finditer(txt):
        BM, BN, TK, S = (int(m.group(i)) for i in range(1, 5))
        out[(BM, BN, TK, S)] = float(m.group(6))
    return out or None


def per_chunk_work(BM, BN, TK, S, K):
    """模型里该配置"每 chunk"已知的工作量（ns）"""
    copy = (BM * TK + TK * BN) * 4.0 / HW["smBw"]
    comp = 2.0 * BM * BN * TK / HW["smCompute"]
    if S == 1:
        return copy + comp
    return max(copy, comp, min(HW["LminGemm"], copy + comp))


def slope(n_chunks, times):
    """最小二乘 y = a + b·x 的 b"""
    mx = st.mean(n_chunks)
    my = st.mean(times)
    num = sum((x - mx) * (y - my) for x, y in zip(n_chunks, times))
    den = sum((x - mx) ** 2 for x in n_chunks)
    return num / den if den else None


def main():
    all_rows = []
    for K, path, note in LOGS:
        tbl = load(path)
        if not tbl:
            print(f"[skip] {path}")
            continue
        groups = {}
        for (BM, BN, TK, S), us in tbl.items():
            groups.setdefault((BM, BN, S), []).append((TK, us))

        print(f"\n=== N=K={K}  ({note}) ===")
        print(f"{'tile':>9s} {'S':>2s} {'TK→µs':>26s} {'斜率ns/chunk':>13s} "
              f"{'模型工作':>9s} {'cChunk ns':>9s}")
        size_rows = []
        for (BM, BN, S), pts in sorted(groups.items()):
            pts = sorted(pts)
            if len(pts) < 2:
                continue
            chunks = [-(-K // tk) for tk, _ in pts]
            times = [us * 1000.0 for _, us in pts]
            b = slope(chunks, times)
            if b is None:
                continue
            work = st.mean([per_chunk_work(BM, BN, tk, S, K) for tk, _ in pts])
            c = b - work
            size_rows.append(c)
            all_rows.append((K, BM, BN, S, c))
            ptxt = " ".join(f"{tk}:{us:.1f}" for tk, us in pts)
            print(f"{f'{BM}x{BN}':>9s} {S:>2d} {ptxt:>26s} {b:>13.0f} "
                  f"{work:>9.0f} {c:>9.0f}")
        if size_rows:
            print(f"  → N={K} 汇总：中位 cChunk = {st.median(size_rows):.0f} ns "
                  f"（IQR {min(size_rows):.0f}–{max(size_rows):.0f}，n={len(size_rows)} 组）")

    print("\n=== 跨尺寸一致性（核心判据）===")
    by_size = {}
    for K, BM, BN, S, c in all_rows:
        by_size.setdefault(K, []).append(c)
    for K in sorted(by_size):
        v = by_size[K]
        print(f"N={K:<5d} 中位 {st.median(v):>7.0f} ns   IQR "
              f"{st.quantiles(v)[0]:>7.0f}–{st.quantiles(v)[2]:>7.0f}   n={len(v)}")
    l2 = [c for K, _, _, _, c in all_rows if K in (500, 512, 1024)]
    if l2:
        med = st.median(l2)
        spread = (max(l2) - min(l2)) / abs(med) if med else float("inf")
        print(f"\nL2 三尺寸合并：中位 {med:.0f} ns，极差/中位 = {spread*100:.0f}%")
        print("判据：三尺寸中位相差 ≤15% → 视为物理常数；否则视为拟合，不应加入模型。")


if __name__ == "__main__":
    main()
