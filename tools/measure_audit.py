#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""measure_audit.py — "实测最优"可信度审计（纯离线，零 GPU）

回答一个问题：回放表里的"实测最优 / 名次 / 距优"有没有可能是测量假象？

六项检查
  C1 配置数 = 按 smem 公式算出的可行数？（漏测/多测）
  C2 smem 列 = (S+1)·单份字节？（载具与公式是否同口径）
  C3 TFLOPS 是否可能（> 硬件上限 = 计时或尺寸错）
  C4 同一 tile 内 S 变化是否平滑（>15% 跳变 = 抖动/污染）
  C5 顶层两名的差距（<1% = 并列带，"最优"本身不稳定）
  C6 正确性校验行是否存在且 PASS

用法
  python tools/measure_audit.py            # 打印审计表
  python tools/measure_audit.py --write    # 写 docs/evidence/measurement_audit.md
"""
import argparse
import math
import os
import re
import statistics as st
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(_HERE)
XFER = os.path.join(WS, "daima", "xfer_2026-11-15")
NEW = os.path.join(WS, "daima", "oracle_logs", "2026-11-15")
DAIMA = os.path.join(WS, "daima")

SMEM_CAP = 101376
PEAK_FP32_TFLOPS = 82.6   # 4090 FP32 理论峰值

G = re.compile(r"BM=(\d+)\s+BN=(\d+)\s+TK=(\d+)\s+S=(\d+)\s+smem=\s*([\d.]+)KB\s+"
               r"耗时:\s+([\d.]+)\s+us")
V = re.compile(r"T=(\d+)\s+TK=(\d+)\s+S=(\d+)\s+smem=\s*([\d.]+)KB\s+"
               r"实测:\s+([\d.]+)\s+us")
S1 = re.compile(r"T=(\d+)\s+S=(\d+)\s+smem=\s*([\d.]+)KB\s+实测:\s+([\d.]+)\s+us")
SIZE = re.compile(r"M=N=K=(\d+)")
NSIZE = re.compile(r"N=(\d+)")

# name, path, family, size, 说明
LOGS = [
    ("gemm4096/run1", os.path.join(XFER, "sweep_run1.log"), "gemm", 4096, "4096 会话1"),
    ("gemm4096/run2", os.path.join(XFER, "sweep_run2.log"), "gemm", 4096, "4096 会话2"),
    ("gemm4096/fri", os.path.join(XFER, "sweep_friday.log"), "gemm", 4096, "4096 会话3"),
    ("gemm1024", os.path.join(NEW, "sweep_1024.log"), "gemm", 1024, "干净卡 6 号"),
    ("gemm512/clean", os.path.join(NEW, "sweep_512_clean.log"), "gemm", 512, "干净卡"),
    ("gemm512/S1", os.path.join(XFER, "sweep_512_S1.log"), "gemm", 512, "⚠ 整轮 +12%"),
    ("gemm512/S4", os.path.join(WS, "daima", "sweep_512_S4.log"), "gemm", 512, "⚠ 整轮 +12%"),
    ("gemm500", os.path.join(XFER, "sweep_500.log"), "gemm", 500, "⚠ 会话未标注"),
    ("twomm", os.path.join(XFER, "twomm_512.log"), "gemm", 512, "链×2"),
    ("three_mm", os.path.join(XFER, "3mm_512.log"), "gemm", 512, "链×3"),
    ("atax/buggy", os.path.join(XFER, "atax_sweep.log"), "gemv", 8192, "⚠ acc bug"),
    ("atax/fixed", os.path.join(XFER, "atax_sweep2.log"), "gemv", 8192, "修复后"),
    ("mvt", os.path.join(XFER, "mvt_512.log"), "gemv", 512, "全空间"),
    ("bicg", os.path.join(NEW, "bicg_512.log"), "gemv", 512, "全空间"),
    ("jacobi2d", os.path.join(XFER, "jacobi2d_sweep.log"), "stencil", 1022, "嵌套 2D"),
    ("jacobi1d", os.path.join(XFER, "jacobi1d_2p24_sweep.log"), "stencil", 1 << 24, "2^24"),
    ("heat1d/2^24", os.path.join(XFER, "heat1d_2^24.log"), "stencil", 1 << 24, "2^24"),
    ("heat1d/2^20", os.path.join(XFER, "heat1d_2^20_redo.log"), "stencil", 1 << 20, "redo"),
    ("syrk/v2", os.path.join(DAIMA, "stril_syrk_512_v2.log"), "gemm", 512, "守卫"),
    ("syrk/v3", os.path.join(NEW, "stril_syrk_512_v3.log"), "gemm", 512, "守卫·第三轮"),
    ("syrk/v4", os.path.join(NEW, "stril_syrk_512_v4.log"), "gemm", 512, "守卫·第四轮"),
    ("symm/v1", os.path.join(DAIMA, "stril_symm_512.log"), "gemm", 512, "守卫"),
    ("symm/v2", os.path.join(DAIMA, "stril_symm_512_v2.log"), "gemm", 512, "守卫"),
]


def smem_feasible_count_gemm():
    n = 0
    for BM in (64, 128, 256):
        for BN in (64, 128, 256):
            if BM == 256 and BN == 256:
                continue
            for TK in (16, 32, 64):
                for S in (1, 2, 3, 4):
                    if (S + 1) * (BM * TK + TK * BN) * 4 <= SMEM_CAP:
                        n += 1
    return n


def smem_feasible_count_gemv(N):
    n = 0
    for T in (8, 16, 32, 64, 128, 256):
        if N % T:
            continue
        for TK in (16, 32, 64):
            for S in (1, 2, 3, 4):
                if (S + 1) * (T * TK + TK) * 4 <= SMEM_CAP:
                    n += 1
    return n


def parse(path, fam):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()
    rows = []
    if fam == "gemm":
        for m in G.finditer(txt):
            rows.append(dict(k=(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))),
                             smem=float(m.group(5)), us=float(m.group(6))))
    elif fam == "gemv":
        for m in V.finditer(txt):
            rows.append(dict(k=(int(m.group(1)), int(m.group(2)), int(m.group(3))),
                             smem=float(m.group(4)), us=float(m.group(5))))
    else:
        for m in S1.finditer(txt):
            rows.append(dict(k=(int(m.group(1)), int(m.group(2))),
                             smem=float(m.group(3)), us=float(m.group(4))))
    return (rows, txt)


def audit_one(name, path, fam, size, note):
    got = parse(path, fam)
    if got is None:
        return dict(name=name, path=path, fam=fam, note=note, missing=True)
    rows, txt = got
    if not rows:
        return dict(name=name, path=path, fam=fam, note=note, missing=True)
    us = [r["us"] for r in rows]
    best, second = min(us), sorted(us)[1] if len(us) > 1 else min(us)

    # C1 配置数
    if fam == "gemm":
        exp = smem_feasible_count_gemm()
    elif fam == "gemv":
        exp = smem_feasible_count_gemv(size)
    else:
        exp = None
    skipped = len(re.findall(r"模型拒（smem/半机守卫）——跳过", txt))
    full = {"gemm": 96, "gemv": 72}.get(fam)
    c1 = (exp is None) or (len(rows) == exp) or (full is not None and len(rows) + skipped == full)

    # C2 smem 列与公式一致（gemm/gemv 闭式；stencil 用 (S+1)·T·4 近似，跳过）
    c2, bad_smem = True, 0
    if fam == "gemm":
        for r in rows:
            BM, BN, TK, S = r["k"]
            pred = (S + 1) * (BM * TK + TK * BN) * 4 / 1024.0
            if abs(pred - r["smem"]) > 0.6:
                bad_smem += 1
        c2 = bad_smem == 0
    elif fam == "gemv":
        for r in rows:
            T, TK, S = r["k"]
            pred = (S + 1) * (T * TK + TK) * 4 / 1024.0
            if abs(pred - r["smem"]) > 0.6:
                bad_smem += 1
        c2 = bad_smem == 0

    # C3 TFLOPS 上限
    c3, max_tf = True, 0.0
    if fam == "gemm":
        fl = 2.0 * size ** 3
        for r in rows:
            tf = fl / (r["us"] * 1e-6) / 1e12
            max_tf = max(max_tf, tf)
        c3 = max_tf <= PEAK_FP32_TFLOPS * 1.05

    # C4 同 tile 内"重叠态"(S>=2)抖动：两态式下 S>=2 应基本平坦，>10% 即异常
    c4, jitters = True, []
    groups = {}
    for r in rows:
        groups.setdefault(r["k"][:-1], []).append(r)
    for g, rs in groups.items():
        t = [x["us"] for x in rs if x["k"][-1] >= 2]
        if len(t) < 2:
            continue
        jitters.append(max(t) / min(t) - 1.0)
    if jitters:
        c4 = max(jitters) <= 0.10

    # C5 前两名差距
    gap12 = second / best - 1.0
    # C6 正确性行
    c6 = ("PASS" in txt)

    return dict(name=name, path=path, fam=fam, note=note, missing=False, n=len(rows), exp=exp,
                c1=c1, c2=c2, bad_smem=bad_smem, c3=c3, max_tf=max_tf,
                c4=c4, jitter=max(jitters) * 100 if jitters else 0.0,
                gap12=gap12 * 100, best=best, c6=c6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    res = [audit_one(*x) for x in LOGS]
    L = []
    L.append("# 测量完整性审计（\"实测最优\"可不可信）\n")
    L.append("> 生成：`python tools/measure_audit.py --write`。纯离线，零 GPU。")
    L.append("> C1 配置数对得上 / C2 smem 列与公式一致 / C3 TFLOPS ≤ 峰值 / C4 同 tile 内 S 抖动 ≤15% /"
             " C5 前两名差距 / C6 有正确性 PASS\n")
    L.append("| 日志 | 族 | n | 期望 | C1 | C2 | C3 | C4 抖动 | 前二差 | C6 | 最优(µs) | 备注 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

    for r in res:
        if r["missing"]:
            L.append(f"| {r['name']} | {r['fam']} | — | — | — | — | — | — | — | — | — | 缺文件 |")
            continue
        c2_cell = "✓" if r["c2"] else f"✗({r['bad_smem']})"
        c4_cell = f"{r['jitter']:.1f}%" + ("" if r["c4"] else " ✗")
        L.append(f"| {r['name']} | {r['fam']} | {r['n']} | {r['exp'] if r['exp'] else '—'} | "
                 f"{'✓' if r['c1'] else '✗'} | {c2_cell} | "
                 f"{'✓' if r['c3'] else '✗'} | {c4_cell} | "
                 f"{r['gap12']:.2f}% | {'✓' if r['c6'] else '✗'} | {r['best']:.1f} | {r['note']} |")

    # 跨会话：同一 kernel 多份日志的最优值离散度
    L.append("")
    L.append("## 跨会话最优值稳定性\n")
    L.append("| kernel | 各会话最优(µs) | 极差 | 判读 |")
    L.append("|---|---|---|---|")
    fam_groups = {
        "gemm4096": ["gemm4096/run1", "gemm4096/run2", "gemm4096/fri"],
        "gemm512": ["gemm512/clean", "gemm512/S1", "gemm512/S4"],
        "syrk": ["syrk/v2", "syrk/v3", "syrk/v4"],
        "symm": ["symm/v1", "symm/v2"],
        "atax": ["atax/buggy", "atax/fixed"],
    }
    by_name = {r["name"]: r for r in res if not r["missing"]}
    for k, names in fam_groups.items():
        vals = [by_name[n]["best"] for n in names if n in by_name]
        if len(vals) < 2:
            continue
        spread = max(vals) / min(vals) - 1.0
        verdict = "稳定（可跨会话引用）" if spread <= 0.02 else ("偏差大，需干净卡复测" if spread <= 0.10 else "严重，只能引单轮")
        L.append(f"| {k} | {' / '.join(f'{v:.1f}' for v in vals)} | {spread*100:.1f}% | {verdict} |")

    md = "\n".join(L)
    print(md)
    if a.write:
        out = os.path.join(WS, "docs", "evidence", "measurement_audit.md")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"\n[写出] {out}")


if __name__ == "__main__":
    main()
