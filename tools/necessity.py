#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""necessity.py — 用 oracle 扫参日志重算"joint 是否必要"，零 GPU。

读 cuda_oracle 的输出日志（sgemm_tiled 格式），对每个日志产出：
  1) 五种策略的 regret 表（相对同日志穷举最优）
  2) 并列带（与最优相差 ≤1% 的配置数）——判断"名次"是否有意义
  3) 若同一 kernel 有两份日志，产出逐配置漂移表（测量可复现性）

策略定义（全部只用日志里已有的实测点，不引入模型）：
  oracle         : 全空间最优
  joint          : 模型决策（日志头 "模型标注: BM,BN,TK,S"）
  tile_only_S1   : 只在 S=1 的配置里选最优（"不用流水"的极限）
  seq            : S=1 选 tile → 在该 tile 上再选最优 (TK,S)（顺序决策）
  depth_only     : 固定模型给出的 tile（BM,BN,TK），只选 S（depth 的边际价值）
  depth_only@def : 固定 128×128 与各 TK，只选 S（常见"默认 tile"基线）
  max_tile       : 取可行配置里 BM×BN 最大的那族，再在其中选最优（容量规则基线）

用法：
  python tools/necessity.py daima/sweep_512.log daima/sweep_512_S4.log
  python tools/necessity.py --json out.json <logs...>
"""
import argparse
import json
import re
import sys
from collections import OrderedDict

CFG_RE = re.compile(
    r"BM=(\d+)\s+BN=(\d+)\s+TK=(\d+)\s+S=(\d+)\s+smem=\s*([\d.]+)KB\s+"
    r"耗时:\s*([\d.]+)\s*us")
MODEL_RE = re.compile(r"模型标注:\s*(\d+),(\d+),(\d+),(\d+)")
SIZE_RE = re.compile(r"M=N=K=(\d+)")
TIE_TOL = 0.01          # 并列带：与最优相差 ≤1%


def parse(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    cfgs = OrderedDict()
    for m in CFG_RE.finditer(text):
        bm, bn, tk, s = (int(m.group(i)) for i in range(1, 5))
        us = float(m.group(6))
        cfgs[(bm, bn, tk, s)] = us
    mm = MODEL_RE.search(text)
    model = tuple(int(mm.group(i)) for i in range(1, 5)) if mm else None
    sm = SIZE_RE.search(text)
    size = int(sm.group(1)) if sm else None
    return dict(path=path, size=size, cfg=cfgs, model=model)


def best_of(items):
    """items: [(key, us)] → (key, us)；同分保留先出现者。"""
    bk, bu = None, float("inf")
    for k, u in items:
        if u < bu:
            bk, bu = k, u
    return bk, bu


def strategies(d):
    cfgs = list(d["cfg"].items())
    ok, us_oracle = best_of(cfgs)
    res = [("oracle", ok, us_oracle)]
    if d["model"] and d["model"] in d["cfg"]:
        mk = d["model"]
        res.append(("joint", mk, d["cfg"][mk]))

    # tile_only_S1
    k1, u1 = best_of([(k, u) for k, u in cfgs if k[3] == 1])
    res.append(("tile_only_S1", k1, u1))

    # seq：S=1 的赢家 → 该 (BM,BN) 上最优 (TK,S)
    if k1:
        k2, u2 = best_of([(k, u) for k, u in cfgs if k[:2] == k1[:2]])
        res.append(("seq", k2, u2))

    # depth_only：固定模型的 (BM,BN,TK)
    if d["model"]:
        m = d["model"]
        k3, u3 = best_of([(k, u) for k, u in cfgs if k[:3] == m[:3]])
        if k3:
            res.append(("depth_only", k3, u3))

    # depth_only@def：128x128 各 TK
    k4, u4 = best_of([(k, u) for k, u in cfgs if k[:2] == (128, 128)])
    if k4:
        res.append(("depth_only@def", k4, u4))

    # max_tile：最大 BM×BN 族内最优
    if cfgs:
        maxarea = max(k[0] * k[1] for k, _ in cfgs)
        k5, u5 = best_of([(k, u) for k, u in cfgs if k[0] * k[1] == maxarea])
        res.append(("max_tile", k5, u5))

    out = []
    for name, key, us in res:
        out.append(dict(name=name, cfg=key, us=us,
                        regret=us / us_oracle - 1.0,
                        rank=1 + sum(1 for _, u in cfgs if u < us - 1e-9)))
    return out, us_oracle


def tie_band(d):
    cfgs = d["cfg"]
    if not cfgs:
        return 0, 0.0
    best = min(cfgs.values())
    n = sum(1 for u in cfgs.values() if u / best - 1.0 <= TIE_TOL)
    return n, best


def fmt_cfg(k):
    if not k:
        return "—"
    return f"{k[0]}×{k[1]}/TK{k[2]}/S{k[3]}"


def report(paths):
    parsed = [parse(p) for p in paths]
    allout = []
    for d in parsed:
        if not d["cfg"]:
            print(f"[skip] {d['path']}: 未解析到配置行")
            continue
        res, best = strategies(d)
        n, _ = tie_band(d)
        print("=" * 92)
        print(f"{d['path']}   N={d['size']}   配置数={len(d['cfg'])}   "
              f"最优={best:.1f}us   并列带(≤1%)={n} 个   模型={fmt_cfg(d['model'])}")
        print("=" * 92)
        print(f"{'策略':16s} {'配置':24s} {'实测(us)':>10s} {'距优':>9s} {'名次':>7s}")
        for r in res:
            print(f"{r['name']:16s} {fmt_cfg(r['cfg']):24s} {r['us']:10.1f} "
                  f"{r['regret']*100:+8.1f}% {r['rank']:5d}/{len(d['cfg'])}")
        allout.append(dict(path=d["path"], size=d["size"], n=len(d["cfg"]),
                           best=best, tie=n, model=d["model"], rows=res))
        print()

    # 漂移表：同 size、同配置集合的两份日志
    by_size = {}
    for d in parsed:
        if d["cfg"]:
            by_size.setdefault(d["size"], []).append(d)
    for size, ds in by_size.items():
        if len(ds) < 2:
            continue
        a, b = ds[0], ds[1]
        common = [k for k in a["cfg"] if k in b["cfg"]]
        if not common:
            continue
        print("=" * 92)
        print(f"漂移表：N={size}  {a['path'].split('/')[-1]} → {b['path'].split('/')[-1]}"
              f"   （同配置逐项对比，共 {len(common)} 项）")
        print("=" * 92)
        deltas = []
        for k in common:
            x, y = a["cfg"][k], b["cfg"][k]
            deltas.append((y / x - 1.0, k, x, y))
        deltas.sort()
        print(f"{'配置':24s} {'前(us)':>9s} {'后(us)':>9s} {'变化':>9s}")
        for dv, k, x, y in (deltas[:3] + deltas[-3:]):
            print(f"{fmt_cfg(k):24s} {x:9.1f} {y:9.1f} {dv*100:+8.1f}%")
        med = sorted(d[0] for d in deltas)[len(deltas) // 2]
        lo, hi = deltas[0][0], deltas[-1][0]
        print(f"\n中位漂移 {med*100:+.1f}%   范围 [{lo*100:+.1f}%, {hi*100:+.1f}%]"
              f"   → 若 |中位| 或上界 >5%，同尺寸的 5% 级'距优'结论不可复现")
        print()
    return allout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = report(a.logs)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1, default=str)
        print(f"[写出] {a.json}")


if __name__ == "__main__":
    main()
