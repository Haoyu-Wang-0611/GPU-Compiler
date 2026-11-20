#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""holdout_decide.py — 留出套件的离线决策（A1/A2 两臂 + 前沿）

只做三件事（零 GPU）：
  1. 读 `daima/holdout/suite.json`，为每个 kernel 构造模型 profile；
  2. 跑 A1（v1.8.5：cchunk=0、β=1）与 A2（cchunk=150、β=2）的冻结决策；
  3. 输出每个 kernel 的前沿（硬可行 / 收缩后 / 实际），供 false-pruning 用。

用法：
  python tools/holdout_decide.py                  # 打印表 + 写 docs/evidence/holdout_decisions.json
  python tools/holdout_decide.py --no-write       # 只打印
  python tools/holdout_decide.py --reps 20        # 决策耗时采样次数
"""
import argparse
import datetime
import json
import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import decide_dev as D  # noqa: E402
import holdout_profiles as P  # noqa: E402

WS = os.path.dirname(_HERE)
SUITE = os.path.join(WS, "daima", "holdout", "suite.json")
OUT = os.path.join(WS, "docs", "evidence", "holdout_decisions.json")


def fmt_cfg(cfg):
    if cfg is None:
        return "REJECT"
    if "BM" in cfg:
        return f"{cfg['BM']}/{cfg['BN']}/{cfg['TK']}/{cfg['S']}"
    if "TK" in cfg:
        return f"T{cfg['T']}/TK{cfg['TK']}/S{cfg['S']}"
    return f"T{cfg['T']}/S{cfg['S']}"


def blocks_of(entry, cfg):
    form, p = entry["form"], entry["params"]
    if form in ("gemm", "gemm_rect"):
        return math.ceil(p["M"] / cfg["BM"]) * math.ceil(p["N"] / cfg["BN"])
    if form in ("chain2", "chain3"):
        return math.ceil(p["SZ"] / cfg["BM"]) * math.ceil(p["SZ"] / cfg["BN"])
    if form in ("mvt", "bicg"):
        return math.ceil(p["N"] / cfg["T"])
    if form == "sten1d":
        return math.ceil(p["N"] / cfg["T"])
    if form == "sten2d":
        d = p["N"] - 2 * p["r"]
        return math.ceil(d / cfg["T"]) ** 2
    if form == "sibling":
        return math.ceil((p["N"] - 2) / cfg["T"])
    raise ValueError(form)


def smem_of(entry, cfg):
    r = P.feasible(entry, cfg, apply_occ=False)
    return None if r is None else r["smem"]


def run_arm(entry, arm, reps=1):
    """arm: 'a1'|'a2'。返回 dict（含 config/predicted/blocks/smem/耗时）。"""
    prof = P.build_profile(entry)
    if arm == "a1":
        D.L2_BOOST = 1.0
        cchunk = 0.0
    elif arm == "a2":
        D.L2_BOOST = 2.0
        cchunk = D.CCHUNK  # 150
    else:
        raise ValueError(arm)
    t0 = time.perf_counter()
    out = D.decide(prof, "hard", cchunk)
    dt = (time.perf_counter() - t0) * 1000.0
    cfg = out.get("config")
    rec = dict(config=cfg, predicted_ns=out.get("predicted_ns"),
               n_feas=out.get("n_feas"), n_enum=out.get("n_enum"),
               blocks=(blocks_of(entry, cfg) if cfg else None),
               smem=(smem_of(entry, cfg) if cfg else None),
               decision_ms=dt)
    if reps > 1:
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            D.decide(prof, "hard", cchunk)
            samples.append((time.perf_counter() - t0) * 1000.0)
        rec["decision_ms_median"] = statistics.median(samples)
        rec["decision_ms_reps"] = reps
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    if not os.path.exists(SUITE):
        raise SystemExit(f"缺 suite.json：{SUITE}（先跑 tools/holdout_gen.py）")
    with open(SUITE, "r", encoding="utf-8") as f:
        suite = json.load(f)

    rows = []
    for e in suite["kernels"]:
        a1 = run_arm(e, "a1", args.reps)
        a2 = run_arm(e, "a2", args.reps)
        hard, contracted, used = P.frontier(e)
        rows.append(dict(
            id=e["id"], name=e["name"], form=e["form"], family=e["family"],
            primary=e["primary"], params=e["params"],
            a1=a1, a2=a2,
            frontier=dict(n_enum=len(P.enumerate_configs(e)), n_hard=len(hard),
                          n_frontier=len(used),
                          n_contracted=len(contracted),
                          fallback=(len(contracted) == 0),
                          keys=[list(P.cfg_key(c)) for c in used]),
        ))
    D.L2_BOOST = 2.0

    # 表
    print(f"{'id':3s} {'family':12s} {'A1(v1.8.5)':20s} {'A2(c150/β2)':20s} "
          f"{'frontier':>10s}  {'blocks':>6s} {'smemKB':>7s}  {'A2 ms':>7s}")
    for r in rows:
        a1, a2, f = r["a1"], r["a2"], r["frontier"]
        print(f"{r['id']:3s} {r['family']:12s} {fmt_cfg(a1['config']):20s} "
              f"{fmt_cfg(a2['config']):20s} "
              f"{f['n_frontier']:>4d}/{f['n_hard']:<4d} "
              f"{str(a2['blocks']):>6s} "
              f"{(a2['smem'] or 0)/1024:>7.1f} "
              f"{a2.get('decision_ms_median', a2['decision_ms']):>7.2f}")

    n_diff = sum(1 for r in rows if r["a1"]["config"] != r["a2"]["config"])
    n_reject = sum(1 for r in rows if r["a1"]["config"] is None or r["a2"]["config"] is None)
    n_fallback = sum(1 for r in rows if r["frontier"]["fallback"])
    n_primary = sum(1 for r in rows if r["primary"])
    print(f"\n汇总：{len(rows)} kernel（主 {n_primary}）；A1/A2 决策不同 {n_diff}；"
          f"拒因 {n_reject}；空集回退 {n_fallback}；"
          f"A2 中位决策 {statistics.median([r['a2'].get('decision_ms_median', r['a2']['decision_ms']) for r in rows]):.2f} ms")

    payload = dict(version="holdout_decisions v1", generated_at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                   suite_sha256=None, arms=dict(
                       a0="R_full rule (four_arms.py)",
                       a1="v1.8.5: cchunk=0, L2_BOOST=1",
                       a2="v1.8.5+cChunk150+L2_BOOST=2",
                       a3="oracle (sweep logs)"),
                   kernels=rows)
    import hashlib
    payload["suite_sha256"] = hashlib.sha256(open(SUITE, "rb").read()).hexdigest()
    if not args.no_write:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"[写出] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
