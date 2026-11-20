#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""replay_dev.py — 开发集回放：模型决策 vs 实测最优（A4 第二件交付）

它只做一件事：对每一行开发集，把 `decide_dev` 选的配置拿到 oracle 日志里查实测值，
然后报四个数：名次（严格/并列）、距优、以及日志里的实测最优是谁。

为什么必须离线做：日志里只有"配置 → 时间"，没有"规则 → 配置"。
服务器能告诉你 128×256 多慢，但告诉不了你模型会不会选它。

用法
  python tools/replay_dev.py                 # 打印 markdown 表
  python tools/replay_dev.py --write         # 同时写 docs/evidence/replay_hard.md
"""
import argparse
import math
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import decide_dev as D  # noqa: E402

WS = os.path.dirname(_HERE)
XFER = os.path.join(WS, "daima", "xfer_2026-11-15")
NEW = os.path.join(WS, "daima", "oracle_logs", "2026-11-15")
DAIMA = os.path.join(WS, "daima")

TIE = 0.02  # 并列带：与最优相差 ≤2%

G = re.compile(r"BM=(\d+)\s+BN=(\d+)\s+TK=(\d+)\s+S=(\d+)\s+smem=\s*[\d.]+KB\s+"
               r"耗时:\s+([\d.]+)\s+us")
V = re.compile(r"T=(\d+)\s+TK=(\d+)\s+S=(\d+)\s+smem=\s*[\d.]+KB\s+"
               r"实测:\s+([\d.]+)\s+us")
S1 = re.compile(r"T=(\d+)\s+S=(\d+)\s+smem=\s*[\d.]+KB\s+实测:\s+([\d.]+)\s+us")

# name: (log path, family, 污染/来源备注)
ROWS = [
    ("gemm4096", os.path.join(XFER, "sweep_run1.log"), "gemm", "干净（4096 跨会话 0.23%）"),
    ("gemm1024", os.path.join(NEW, "sweep_1024.log"), "gemm", "干净（6 号卡）"),
    ("gemm512", os.path.join(NEW, "sweep_512_clean.log"), "gemm", "干净（复现 11-14 的 24.1→24.2）"),
    ("gemm500", os.path.join(XFER, "sweep_500.log"), "gemm", "⚠ 会话未标注，可能同期污染"),
    ("twomm", os.path.join(XFER, "twomm_512.log"), "gemm", "链×2"),
    ("three_mm", os.path.join(XFER, "3mm_512.log"), "gemm", "链×3（该日志 128×128 族有抖动，赢家族稳定）"),
    ("atax", os.path.join(XFER, "atax_sweep2.log"), "gemv", "修复后轮"),
    ("mvt", os.path.join(XFER, "mvt_512.log"), "gemv", "全空间 64 配置"),
    ("bicg", os.path.join(NEW, "bicg_512.log"), "gemv", "全空间 64 配置"),
    ("jacobi2d", os.path.join(XFER, "jacobi2d_sweep.log"), "stencil1", "12 可行配置"),
    ("jacobi1d", os.path.join(XFER, "jacobi1d_2p24_sweep.log"), "stencil1", "2^24"),
    ("heat1d24", os.path.join(XFER, "heat1d_2^24.log"), "stencil1", "2^24"),
    ("heat1d20", os.path.join(XFER, "heat1d_2^20_redo.log"), "stencil1", "2^20（redo，干净）"),
    ("syrk", os.path.join(NEW, "stril_syrk_512_v3.log"), "gemm", "守卫形态（干净轮，第三轮）"),
    ("symm", os.path.join(DAIMA, "stril_symm_512_v2.log"), "gemm", "守卫形态（干净轮；模型侧等价于 plain GEMM）"),
]


def parse(path, family):
    out = {}
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()
    if family == "gemm":
        for m in G.finditer(txt):
            out[(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))] = float(m.group(5))
    elif family == "gemv":
        for m in V.finditer(txt):
            out[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = float(m.group(4))
    else:
        for m in S1.finditer(txt):
            out[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
    return out or None


def key_of(cfg, family):
    if family == "gemm":
        return (cfg["BM"], cfg["BN"], cfg["TK"], cfg["S"])
    if family == "gemv":
        return (cfg["T"], cfg["TK"], cfg["S"])
    return (cfg["T"], cfg["S"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    lines = []
    lines.append("# 开发集回放（A4）— 模型决策 vs 实测最优\n")
    lines.append("> 生成：`python tools/replay_dev.py --out <file>`；决策由 `tools/decide_dev.py`"
                 "（v1.8.5 + 每 chunk 固定开销 cChunk=150 ns，名义值；25–500 ns 决策不变）给出，"
                 "实测值查自各 oracle 日志。")
    lines.append("> 名字含义：严格名次 = 比它更快的配置数 +1；并列 = 与最优差 ≤2% 视作同一档；"
                 "距优 = 该配置实测/最优 − 1。\n")
    lines.append("| kernel | 模型决策 | 实测(µs) | 严格名次 | 距优 | 实测最优 | 并列带 | 日志 |")
    lines.append("|---|---|---|---|---|---|---|---|")

    stats = []
    for name, path, fam, note in ROWS:
        prof = D.DEV_KERNELS[name][0]
        r = D.decide(prof, "hard")
        cfg = r["config"]
        if cfg is None:
            lines.append(f"| {name} | 无可行 | — | — | — | — | — | {note} |")
            continue
        tbl = parse(path, fam)
        if not tbl:
            lines.append(f"| {name} | {cfg} | 缺日志 | — | — | — | — | {note} |")
            continue
        k = key_of(cfg, fam)
        if k not in tbl:
            lines.append(f"| {name} | {cfg} | 不在日志中 | — | — | — | — | {note} |")
            continue
        best_k = min(tbl, key=lambda x: tbl[x])
        best_t = tbl[best_k]
        t = tbl[k]
        rank = 1 + sum(1 for v in tbl.values() if v < t - 1e-9)
        regret = t / best_t - 1.0
        tie_band = sum(1 for v in tbl.values() if v <= best_t * (1 + TIE))
        stats.append((name, rank, len(tbl), t, best_t, regret))
        cfg_s = "/".join(str(cfg[x]) for x in ("BM", "BN", "TK", "T", "S") if x in cfg)
        best_s = "/".join(str(x) for x in best_k)
        lines.append(f"| {name} | {cfg_s} | {t:.1f} | {rank}/{len(tbl)} | "
                     f"{regret*100:+.1f}% | {best_s} | {tie_band} | {note} |")

    if stats:
        med_rank = sorted(s[1] / s[2] for s in stats)[len(stats) // 2]
        med_reg = sorted(s[5] for s in stats)[len(stats) // 2]
        top1 = sum(1 for s in stats if s[1] == 1)
        top3 = sum(1 for s in stats if s[1] <= 3)
        within20 = sum(1 for s in stats if s[1] / s[2] <= 0.2 and s[5] <= 0.2)
        lines.append("")
        lines.append(f"**汇总**：{len(stats)} 行；严格第 1 名 **{top1}** 行，前三名 **{top3}** 行；"
                     f"名次百分位中位 **{med_rank*100:.1f}%**，距优中位 **{med_reg*100:+.1f}%**；"
                     f"双达标（前 20% 且 ≤20%）**{within20}/{len(stats)}**。")
        lines.append("")
        lines.append("> 注：本表 15 行 = 11-15 规范主表 13 行 + 11-16 后新增的 gemm1024 与 heat1d@2²⁰。`gemm500` 的日志来自一个未标注环境的会话（与 512 污染轮同期），"
                     "其数字只作参考。**本表是开发集（in-sample）**：15 行都参与过模型开发，"
                     "因此只能说明复现与排序，不能作为泛化证据；泛化必须看留出套件。"
                     "唯二未命中「距优 ≤20%」的行是 heat1d：2²⁴ 的 rank 9/38 落在 4% 密集带（距优 +4.0%）；"
                     "2²⁰ 的 +18% 是相对一个**只在 S=4 出现的 6.1 µs 带**（2 个 cell），"
                     "而模型的选择落在**主并列带**（8 个 cell 均为 7.2 µs）——该 6.1 带无法由任何已建模项解释，"
                     "已标记待复测，不据此调模型（见 docs/evidence/stencil1d_regime.md）。")

    md = "\n".join(lines)
    print(md)
    if args.write or args.out:
        out = args.out or os.path.join(WS, "docs", "evidence", "replay_hard.md")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"\n[写出] {out}")


if __name__ == "__main__":
    main()
