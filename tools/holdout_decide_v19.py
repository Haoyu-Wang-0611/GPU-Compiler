#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""holdout_decide_v19.py — 生成 v1.9 的留出集决策 + 前沿（不覆盖 frozen A1/A2）

v1.9 = v1.8.5 + F1（1D numTiles 512→128）+ F2（GEMV T<16 仅 DRAM 罚）。
输出：docs/evidence/holdout_decisions_v19.json
  {id, name, family, primary, params, v19:{config,predicted_ns,blocks,smem,...},
   frontier:{n_enum,n_hard,n_frontier,keys}}

用法：
  python tools/holdout_decide_v19.py
  python tools/holdout_decide_v19.py --no-write
"""
import argparse
import datetime
import hashlib
import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import decide_dev as D  # noqa: E402
import holdout_profiles as P  # noqa: E402
import holdout_profiles_v19 as P19  # noqa: E402
import holdout_decide as HD  # noqa: E402

WS = os.path.dirname(_HERE)
SUITE = os.path.join(WS, "daima", "holdout", "suite.json")
OUT = os.path.join(WS, "docs", "evidence", "holdout_decisions_v19.json")


def fmt(cfg):
    if not cfg:
        return "REJECT"
    if "BM" in cfg:
        return f"{cfg['BM']}/{cfg['BN']}/{cfg['TK']}/{cfg['S']}"
    if "TK" in cfg:
        return f"T{cfg['T']}/TK{cfg['TK']}/S{cfg['S']}"
    return f"T{cfg['T']}/S{cfg['S']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    suite = json.load(open(SUITE, encoding="utf-8"))
    P19.install()          # 之后 P.frontier/P.feasible 用 128 守卫
    rows = []
    D.L2_BOOST = 2.0
    for e in suite["kernels"]:
        prof = P.build_profile(e)
        out = D.decide(prof, "hard", D.CCHUNK, model="v1.9")
        cfg = out["config"]
        hard, contracted, used = P.frontier(e)
        rec = dict(
            id=e["id"], name=e["name"], form=e["form"], family=e["family"],
            primary=e["primary"], params=e["params"],
            v19=dict(config=cfg, predicted_ns=out.get("predicted_ns"),
                     n_feas=out.get("n_feas"), n_enum=out.get("n_enum"),
                     blocks=(HD.blocks_of(e, cfg) if cfg else None),
                     smem=(HD.smem_of(e, cfg) if cfg else None)),
            frontier=dict(n_enum=len(P.enumerate_configs(e)), n_hard=len(hard),
                          n_contracted=len(contracted), n_frontier=len(used),
                          fallback=(len(contracted) == 0),
                          keys=[list(P.cfg_key(c)) for c in used]),
        )
        rows.append(rec)

    print(f"{'id':3s} {'family':12s} {'v1.9':20s} {'pred(ns)':>12s} "
          f"{'blocks':>6s} {'smemKB':>7s} {'front':>8s}")
    for r in rows:
        v = r["v19"]
        print(f"{r['id']:3s} {r['family']:12s} {fmt(v['config']):20s} "
              f"{(v['predicted_ns'] or 0):12.1f} {str(v['blocks']):>6s} "
              f"{(v['smem'] or 0)/1024:7.1f} "
              f"{r['frontier']['n_frontier']:3d}/{r['frontier']['n_hard']:<3d}")

    payload = dict(version="v1.9 (F1 128 + F2 DRAM-only)",
                   generated_at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                   suite_sha256=hashlib.sha256(open(SUITE, "rb").read()).hexdigest(),
                   model=dict(f1="sten1d numTiles 512→128",
                              f2="gemv T<16 thin only if A working set > 72MB L2"),
                   kernels=rows)
    if not args.no_write:
        with open(OUT, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"\n[写出] {OUT}")
    print(f"汇总：{len(rows)} kernel；v1.9 决策/前沿已生成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
