#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_l1.py — 核对 xfer 包里的决策输出（l1_*.txt）与 golden。

输出三块：
  1) 每份 l1_*.txt 的 detected 行 / Decision 行 / 搜索计数（一眼看主表口径）
  2) l1_*.txt 与 goldens/*.golden 的逐行一致（排序后，模拟 regress.sh 的 diff）
  3) 覆盖缺口：regress.sh 清单一共 16 发，哪些缺 txt / 缺 golden
"""
import glob
import os
import re

BASE = "daima/xfer_2026-11-15"
LIST = ("heat1d jacobi1d_big2 jacobi2d gemm4096 gemm512 gemm500 gemm1024 atax_big "
        "probe_dyn twomm mvt_probe bicg_probe symm_true_probe symm_full_probe "
        "three_mm_probe syrk_probe").split()


def lines(p):
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return [l.rstrip("\n") for l in f]


def pick(ls, pat):
    for l in ls:
        if re.search(pat, l):
            return l.strip()
    return "—"


print("=" * 110)
print("1) 决策输出总览（l1_*.txt）")
print("=" * 110)
print(f"{'文件':24s} {'detected':52s} {'Decision（截断）'}")
for p in sorted(glob.glob(os.path.join(BASE, "l1_*.txt"))):
    ls = lines(p)
    det = pick(ls, r"\[L1\] detected:")
    dec = pick(ls, r"\[L1 Decision\]")
    det = det.replace("[L1] detected:", "").strip()
    print(f"{os.path.basename(p):24s} {det[:52]:52s} {dec[:70]}")

print()
print("=" * 110)
print("2) txt vs golden（排序后比较，等价 regress.sh 的判据）")
print("=" * 110)
for name in LIST:
    t = os.path.join(BASE, f"l1_{name}.txt")
    g = os.path.join(BASE, "goldens", f"l1_{name}.golden")
    if not os.path.exists(t) or not os.path.exists(g):
        print(f"[缺] {name:18s} txt={os.path.exists(t)} golden={os.path.exists(g)}")
        continue
    a, b = sorted(lines(g)), sorted(lines(t))
    print(f"[{'PASS' if a == b else 'DIFF'}] {name}")
    if a != b:
        import difflib
        for d in list(difflib.unified_diff(a, b, "golden", "txt", lineterm=""))[:12]:
            print("   ", d)

print()
print("=" * 110)
print("3) 覆盖缺口 / 多余文件")
print("=" * 110)
have_txt = {os.path.basename(p)[3:-4] for p in glob.glob(os.path.join(BASE, "l1_*.txt"))}
have_g = {os.path.basename(p)[3:-7] for p in glob.glob(os.path.join(BASE, "goldens", "*.golden"))}
print("regress 清单中缺 txt   :", [n for n in LIST if n not in have_txt] or "无")
print("regress 清单中缺 golden:", [n for n in LIST if n not in have_g] or "无")
print("清单外的 txt           :", sorted(have_txt - set(LIST)) or "无")
print("清单外的 golden        :", sorted(have_g - set(LIST)) or "无")

# ---------------------------------------------------------------- 4) 决策行的实测核对
print()
print("=" * 110)
print("4) 决策配置在对应 sweep 日志里的实测值 / 名次 / 距优")
print("=" * 110)
DEC = [
    # (日志, 决策配置(轴元组,S), 说明)
    ("sweep_run1.log", ((128, 256, 16), 3), "gemm4096 决策 (run1)"),
    ("sweep_friday.log", ((128, 256, 16), 3), "gemm4096 决策 (friday)"),
    ("sweep_512_S1.log", ((64, 64, 32), 4), "gemm512 决策"),
    ("sweep_500.log", ((64, 64, 32), 4), "gemm500 决策"),
    ("twomm_512.log", ((64, 64, 32), 4), "twomm 决策"),
    ("3mm_512.log", ((64, 64, 32), 4), "3mm 决策"),
]
for log, want, tag in DEC:
    path = os.path.join(BASE, log)
    if not os.path.exists(path):
        continue
    rows = []
    for m in re.finditer(r"BM=(\d+)\s+BN=(\d+)\s+TK=(\d+)\s+S=(\d+)\s+"
                         r"smem=\s*([\d.]+)KB\s+耗时:\s*([\d.]+)\s*us", "\n".join(lines(path))):
        rows.append((((int(m.group(1)), int(m.group(2)), int(m.group(3))),
                      int(m.group(4))), float(m.group(6))))
    if not rows:
        continue
    rows.sort(key=lambda r: r[1])
    hit = [(i + 1, u) for i, (k, u) in enumerate(rows) if k == want]
    print(f"{tag:22s} {log:20s} n={len(rows):3d} 最优={rows[0][1]:8.1f}us  "
          f"决策实测={hit[0][1] if hit else float('nan'):8.1f}us  "
          f"名次={hit[0][0] if hit else '—'}/{len(rows)}  "
          f"距优={((hit[0][1]/rows[0][1]-1)*100) if hit else float('nan'):+.1f}%")

print()
print("5) [搜索: 可行 X/枚举 Y] 双列计数（与台账口径核对）")
print("=" * 110)
for p in sorted(glob.glob(os.path.join(BASE, "l1_*.txt"))):
    ls = lines(p)
    dec = pick(ls, r"\[L1 Decision\]")
    m = re.search(r"\[搜索:\s*可行\s*(\d+)/枚举\s*(\d+)\]", dec)
    if m:
        print(f"  {os.path.basename(p):24s} 可行 {m.group(1):>3s} / 枚举 {m.group(2):>3s}")

# ---------------------------------------------------------------- 6) 其余家族决策行核对
print()
print("=" * 110)
print("6) stencil / GEMV 家族的决策行与实测（T/TK/S 格式）")
print("=" * 110)


def parse_ts(log):
    """返回 {(T, TK?, S): us}，兼容带/不带 TK 两种格式。"""
    out = {}
    for l in lines(os.path.join(BASE, log)):
        m = re.search(r"T=(\d+)\s+TK=(\d+)\s+S=(\d+)\s+smem=\s*[\d.]+KB\s+"
                      r"实测:\s*([\d.]+)\s*us", l)
        if m:
            out[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = float(m.group(4))
            continue
        m = re.search(r"T=(\d+)\s+S=(\d+)\s+smem=\s*[\d.]+KB\s+"
                      r"实测:\s*([\d.]+)\s*us", l)
        if m:
            out[(int(m.group(1)), None, int(m.group(2)))] = float(m.group(3))
    return out


KPI = [
    ("atax_sweep2.log", (64, 64, 1), "atax8192 决策 T64/TK64/S1"),
    ("atax_sweep.log", (64, 64, 1), "atax8192 决策（buggy 轮）"),
    ("mvt_512.log", (8, 64, 1), "mvt 决策 T8/TK64/S1"),
    ("jacobi2d_sweep.log", (64, None, 1), "jacobi2d 决策 T64/S1"),
    ("jacobi1d_2p24_sweep.log", (4096, None, 1), "jacobi1d@2^24 决策 T4096/S1"),
    ("heat1d_2^24.log", (4096, None, 1), "heat1d@2^24 决策 T4096/S1"),
    ("heat1d_2^20_redo.log", (2048, None, 1), "heat1d@2^20 决策 T2048/S1"),
]
for log, want, tag in KPI:
    if not os.path.exists(os.path.join(BASE, log)):
        continue
    d = parse_ts(log)
    if not d:
        continue
    ordered = sorted(d.items(), key=lambda kv: kv[1])
    best = ordered[0]
    hit = [(i + 1, u) for i, (k, u) in enumerate(ordered) if k == want]
    print(f"{tag:30s} {log:24s} n={len(d):3d} 最优={best[1]:9.1f}us({best[0]})  "
          f"决策实测={hit[0][1] if hit else float('nan'):9.1f}us  "
          f"名次={hit[0][0] if hit else '—'}/{len(d)}  "
          f"距优={((hit[0][1]/best[1]-1)*100) if hit else float('nan'):+.1f}%")
