#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""holdout_v19_probe.py — 在首轮留出日志上回放 v1.9（F1+F2），核对修复效果

这是**开发集回放**（同一批日志已经看过结果），不是验证；只用于：
  · 确认 v1.9 的决策确实落到预期（S3/S7/R5/R6/R8 命中，S1 仍 S=1）；
  · 确认 v1.9 的前沿不再误杀（false-pruning 0）。

v1.9 = v1.8.5 + F1（1D numTiles 512→128）+ F2（GEMV T<16 仅 DRAM 罚）。
"""
import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import decide_dev as D  # noqa: E402
import holdout_profiles as P  # noqa: E402
import holdout_profiles_v19 as P19  # noqa: E402
import four_arms as FA  # noqa: E402

WS = os.path.dirname(_HERE)
SUITE = os.path.join(WS, "daima", "holdout", "suite.json")
DEC = os.path.join(WS, "docs", "evidence", "holdout_decisions.json")
LOGS = os.path.join(WS, "ceshi-shuchu3", "logs")


def key_of(cfg):
    if not cfg:
        return None
    if "BM" in cfg:
        return (cfg["BM"], cfg["BN"], cfg["TK"], cfg["S"])
    if "TK" in cfg:
        return (cfg["T"], cfg["TK"], cfg["S"])
    return (cfg["T"], cfg["S"])


def fmt(cfg):
    if not cfg:
        return "—"
    if "BM" in cfg:
        return f"{cfg['BM']}/{cfg['BN']}/{cfg['TK']}/{cfg['S']}"
    if "TK" in cfg:
        return f"T{cfg['T']}/TK{cfg['TK']}/S{cfg['S']}"
    return f"T{cfg['T']}/S{cfg['S']}"


def main():
    suite = json.load(open(SUITE, encoding="utf-8"))
    dec = json.load(open(DEC, encoding="utf-8"))
    by_id = {r["id"]: r for r in dec["kernels"]}
    D.L2_BOOST = 2.0
    rows = []
    for e in suite["kernels"]:
        rec = by_id[e["id"]]
        log = os.path.join(LOGS, rec["name"] + ".log")
        if not os.path.exists(log):
            continue
        tbl = FA.parse_log(log)
        if not tbl:
            continue
        tau, best = FA.tau_of(tbl)
        best_key = min(tbl, key=lambda k: tbl[k])
        prof = P.build_profile(e)
        cur = D.decide(prof, "hard", D.CCHUNK, model="v1.8.5")["config"]
        fix = D.decide(prof, "hard", D.CCHUNK, model="v1.9")["config"]
        # 旧前沿（v1.8.5，512）→ 新前沿（v1.9，128）；用补丁切换，不改 frozen 文件
        old_fn = P._sten1d_hard
        old_keys = {P.cfg_key(c) for c in P.frontier(e)[2]}
        P._sten1d_hard = P19._sten1d_hard_v19
        fkeys = {P.cfg_key(c) for c in P.frontier(e)[2]}
        P._sten1d_hard = old_fn
        def m(cfg):
            k = key_of(cfg)
            if k is None or k not in tbl:
                return None
            return FA.metrics(tbl, k, tau, best)
        mc, mf = m(cur), m(fix)
        fp_cur = (best_key not in old_keys)
        fp_fix = (best_key not in fkeys)
        rows.append((e["id"], e["primary"], rec["name"], cur, fix, mc, mf, fp_cur, fp_fix, tau))
    print(f"{'id':3s} {'prim':5s} {'v1.8+A2':18s} {'rank/reg':>13s} "
          f"{'v1.9':18s} {'rank/reg':>13s} {'fprune':>7s}")
    hc = hf = 0
    fpc = fpf = 0
    for id_, pri, name, cur, fix, mc, mf, f1, f2, tau in rows:
        def mr(m):
            return f"{m['rank']}/{m['n']} {m['regret']*100:+.1f}%" if m else "未测"
        hc += bool(mc and mc["hit"])
        hf += bool(mf and mf["hit"])
        fpc += bool(f1)
        fpf += bool(f2)
        print(f"{id_:3s} {str(pri):5s} {fmt(cur):18s} {mr(mc):>13s} "
              f"{fmt(fix):18s} {mr(mf):>13s} {str(f1)[0]}->{str(f2)[0]:>5s}")
    n_pri = sum(1 for r in rows if r[1])
    print(f"\n全 {len(rows)}：命中 {hc}→{hf}；false-pruning {fpc}→{fpf}")
    print(f"主集 {n_pri}：命中 {sum(1 for r in rows if r[1] and r[5] and r[5]['hit'])}"
          f"→{sum(1 for r in rows if r[1] and r[6] and r[6]['hit'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
