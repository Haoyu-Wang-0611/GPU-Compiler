#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""holdout_fix_probe.py — 离线 what-if：只修两个明确模型 bug，看同一批日志上能救回多少

修复（仅本探针；不改 decide_dev/l1model/L1Feas）：
  F1 1D stencil 老守卫：`numTiles >= 512` → `>= 128`（半机/满机量级，而非 512）
  F2 GEMV 薄流水惩罚：`T<16 → Lmin=1260` 改为 **仅 DRAM regime**（A 工作集 > L2）时惩罚
其余族沿用 A2 当前决策。日志用 ceshi-shuchu3/logs。

这不是验证（同一批留出集已经看过结果），只用于判断"修完后大概能到哪"。
"""
import json
import math
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import decide_dev as D  # noqa: E402
import holdout_profiles as P  # noqa: E402
import four_arms as FA  # noqa: E402

WS = os.path.dirname(_HERE)
SUITE = os.path.join(WS, "daima", "holdout", "suite.json")
DEC = os.path.join(WS, "docs", "evidence", "holdout_decisions.json")
LOGS = os.path.join(WS, "ceshi-shuchu3", "logs")


def decide_sten1d_fixed(prof, guard=128):
    N = prof["N"]
    W = 2 * N * 4
    bw = D._bw_for_ws(W)
    best = None
    for S in (1, 2, 3, 4):
        for T in [16 << i for i in range(40)]:
            if T > N:
                break
            bytes_ = 0
            for base_elems, elem in prof["legacy_base"]:
                bytes_ += base_elems * elem + (T - 1) * elem
            smem = bytes_ * (S + 1)
            if smem > D.HW["smemCap"]:
                continue
            numTiles = math.ceil(N / T)
            if N >= 8192 and numTiles < guard:
                continue
            waves = math.ceil(numTiles / D.HW["numSms"])
            eff = min(float(S), max(1.0, float(waves - 1)))
            copy = bytes_ / bw
            comp = prof["flops_per_iter"] * T / D.HW["smCompute"]
            R = max(D.HW["rExposed"] - D.HW["aT"] * T, 1.0)
            expo = max(R / (eff ** D.HW["pExp"]), D.HW["deltaFloor"])
            total = D.HW["launch"] + waves * (copy + comp + expo)
            if best is None or total < best["total"]:
                best = dict(total=total, T=T, S=S)
    return best


def decide_gemv_fixed(prof, l2_bytes=72 * 1024 * 1024):
    P_ = len(prof["passes"])
    best = None
    for T in (256, 128, 64, 32, 16, 8):
        for TK in (64, 32, 16):
            for S in (1, 2, 3, 4):
                tiles = [0] * prof["nloops"]
                for p in prof["passes"]:
                    tiles[p["out"]] = T
                    tiles[p["red"]] = TK
                smem, ok = 0, True
                reds = [p["red"] for p in prof["passes"]]
                for acc in prof["accs"]:
                    v = D.L.window(acc, tiles)
                    if v < 0:
                        ok = False
                        break
                    smem += v * ((S + 1) if D.L.is_stream(acc, reds) else 1)
                if not ok or smem > D.HW["smemCap"]:
                    continue
                total = D.HW["launch"] * P_
                guard = True
                for p in prof["passes"]:
                    M, K = p["M"], p["K"]
                    nb = math.ceil(M / T)
                    if nb < D.HW["numSms"] // 2:      # 保留半机守卫
                        guard = False
                        break
                    chunks = math.ceil(K / TK)
                    # F2：只有 DRAM regime 才罚 T<16
                    W = (M * K) * 4.0
                    thin = (T < 16 and W > l2_bytes)
                    Lmin = D.HW.get("LminGemvThin", 1260.0) if thin else D.HW.get("LminGemv", 820.0)
                    occ_ = min(1.0, nb / D.HW["numSms"])
                    bwEff = D.HW["numSms"] * D.HW["smBw"] * occ_ * max(0.6, 1.0 - 2.4 / T)
                    streamB = M * K * 4.0
                    pt = max(chunks * Lmin, streamB / bwEff) + T * 4.0 / D.HW["smBw"]
                    total += pt
                if guard and (best is None or total < best["total"]):
                    best = dict(total=total, T=T, TK=TK, S=S)
    return best


def key_of(cfg):
    if not cfg:
        return None
    if "BM" in cfg:
        return (cfg["BM"], cfg["BN"], cfg["TK"], cfg["S"])
    if "TK" in cfg:
        return (cfg["T"], cfg["TK"], cfg["S"])
    return (cfg["T"], cfg["S"])


def main():
    suite = json.load(open(SUITE, encoding="utf-8"))
    dec = json.load(open(DEC, encoding="utf-8"))
    by_id = {r["id"]: r for r in dec["kernels"]}
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
        prof = P.build_profile(e)
        # 当前 A2
        D.L2_BOOST = 2.0
        cur = D.decide(prof, "hard", D.CCHUNK)["config"]
        # what-if
        if e["form"] in ("sten1d", "sibling"):
            fix = decide_sten1d_fixed(prof) if e["form"] == "sten1d" else D.decide(prof, "hard", D.CCHUNK)["config"]
        elif e["form"] in ("mvt", "bicg"):
            fix = decide_gemv_fixed(prof)
        else:
            fix = cur
        def m(cfg):
            k = key_of(cfg)
            if k is None or k not in tbl:
                return None
            return FA.metrics(tbl, k, tau, best)
        mc, mf = m(cur), m(fix)
        rows.append((e["id"], e["primary"], rec["name"], key_of(cur), key_of(fix),
                     mc, mf, tau))
    print(f"{'id':3s} {'primary':7s} {'current A2':18s} {'rank/reg':>14s} "
          f"{'fixed':18s} {'rank/reg':>14s} {'Δhit':>5s}")
    hit_cur = hit_fix = 0
    for r in rows:
        id_, pri, name, kc, kf, mc, mf, tau = r
        def fmt(k):
            return "/".join(str(x) for x in k) if k else "—"
        def mr(m):
            return f"{m['rank']}/{m['n']} {m['regret']*100:+.1f}%" if m else "未测"
        dc = mf["hit"] and not mc["hit"]
        hit_cur += bool(mc and mc["hit"])
        hit_fix += bool(mf and mf["hit"])
        print(f"{id_:3s} {str(pri):7s} {fmt(kc):18s} {mr(mc):>14s} "
              f"{fmt(kf):18s} {mr(mf):>14s} {('+1' if dc else ''):>5s}")
    print(f"\n命中：current {hit_cur}/{len(rows)} → fixed {hit_fix}/{len(rows)}")
    pri = [r for r in rows if r[1]]
    print(f"主集：current {sum(1 for r in pri if r[5] and r[5]['hit'])}/{len(pri)} → "
          f"fixed {sum(1 for r in pri if r[6] and r[6]['hit'])}/{len(pri)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
