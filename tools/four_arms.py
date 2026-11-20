#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""four_arms.py — 留出集四臂裁判（A0 规则 / A1 v1.8.5 / A2 当前 / A3 oracle）

零 GPU：只读 `daima/holdout/suite.json`、`docs/evidence/holdout_decisions.json`
与 `--logs` 目录下的 sweep 日志，算 tie-aware 名次/regret/命中/false-pruning。

日志格式：
  新（优先）：`RESULT kernel=<id> family=<...> BM=.. BN=.. TK=.. T=.. S=.. time_us=.. pass=.. sample=..`
  旧（兼容）：`BM=.. BN=.. TK=.. S=.. smem=..KB 耗时: X us`
             `T=.. TK=.. S=.. smem=..KB 实测: X us`
             `T=.. S=.. smem=..KB 实测: X us`
  同一配置多行 → 取中位数；含 FAIL 的行跳过。

用法：
  python tools/four_arms.py --selftest
  python tools/four_arms.py --logs daima/oracle_logs/holdout
  python tools/four_arms.py --logs <dir> --report docs/holdout/四臂裁判-结果.md
"""
import argparse
import datetime
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import holdout_profiles as P  # noqa: E402
import holdout_decide as HD  # noqa: E402

WS = os.path.dirname(_HERE)
SUITE = os.path.join(WS, "daima", "holdout", "suite.json")
DECISIONS = os.path.join(WS, "docs", "evidence", "holdout_decisions.json")
DEFAULT_LOGS = os.path.join(WS, "daima", "oracle_logs", "holdout")
DEFAULT_REPORT = os.path.join(WS, "docs", "holdout", "四臂裁判-结果.md")
SM = P.SM
CAP = P.CAP

# 旧格式
RE_G = re.compile(r"BM=(\d+)\s+BN=(\d+)\s+TK=(\d+)\s+S=(\d+)\s+smem=\s*[\d.]+KB\s+"
                  r"(?:耗时|实测):\s*([\d.]+)\s*us")
RE_V = re.compile(r"T=(\d+)\s+TK=(\d+)\s+S=(\d+)\s+smem=\s*[\d.]+KB\s+"
                  r"(?:实测|耗时):\s*([\d.]+)\s*us")
RE_S = re.compile(r"T=(\d+)\s+S=(\d+)\s+smem=\s*[\d.]+KB\s+"
                  r"(?:实测|耗时):\s*([\d.]+)\s*us")
RE_RESULT = re.compile(r"^RESULT\s+(.*)$")
RE_KV = re.compile(r"(\w+)=(\S+)")


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def _key_from_kv(kv):
    try:
        if "BM" in kv:
            return (int(kv["BM"]), int(kv["BN"]), int(kv["TK"]), int(kv["S"]))
        if "TK" in kv:
            return (int(kv["T"]), int(kv["TK"]), int(kv["S"]))
        if "T" in kv:
            return (int(kv["T"]), int(kv["S"]))
    except (KeyError, ValueError):
        return None
    return None


def parse_log(path):
    """返回 {key: median_time_us}；无有效行返回 {}。"""
    samples = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or "FAIL" in s:
                continue
            m = RE_RESULT.match(s)
            if m:
                kv = dict(RE_KV.findall(m.group(1)))
                k = _key_from_kv(kv)
                try:
                    t = float(kv.get("time_us", "nan"))
                except ValueError:
                    continue
                if k and math.isfinite(t) and t > 0:
                    samples[k].append(t)
                continue
            m = RE_G.search(s)
            if m:
                k = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
                t = float(m.group(5))
                if t > 0:
                    samples[k].append(t)
                continue
            m = RE_V.search(s)
            if m:
                k = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                t = float(m.group(4))
                if t > 0:
                    samples[k].append(t)
                continue
            m = RE_S.search(s)
            if m:
                k = (int(m.group(1)), int(m.group(2)))
                t = float(m.group(3))
                if t > 0:
                    samples[k].append(t)
    return {k: statistics.median(v) for k, v in samples.items() if v}


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def tau_of(tbl):
    """τ = max(2%, 前 10% 配置的 IQR 相对值)。"""
    times = sorted(tbl.values())
    best = times[0]
    k = max(1, int(math.ceil(0.1 * len(times))))
    fast = times[:k]
    if len(fast) < 2:
        return 0.02, best
    # 四分位（小样本用线性插值）
    def q(p):
        pos = (len(fast) - 1) * p
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        return fast[lo] + (fast[hi] - fast[lo]) * (pos - lo)
    iqr = q(0.75) - q(0.25)
    return max(0.02, iqr / best), best


def metrics(tbl, key, tau, best=None):
    """返回 dict(rank, pct, regret, hit) 或 None（配置不在日志）。"""
    if key is None or key not in tbl:
        return None
    if best is None:
        best = min(tbl.values())
    t = tbl[key]
    rank = 1 + sum(1 for v in tbl.values() if v < t * (1 - tau))
    return dict(time=t, rank=rank, n=len(tbl), pct=rank / len(tbl),
                regret=t / best - 1.0, hit=(t <= best * (1 + tau)))


def blocks_of(entry, key):
    form, p = entry["form"], entry["params"]
    if form in ("gemm", "gemm_rect"):
        return math.ceil(p["M"] / key[0]) * math.ceil(p["N"] / key[1])
    if form in ("chain2", "chain3"):
        return math.ceil(p["SZ"] / key[0]) * math.ceil(p["SZ"] / key[1])
    if form in ("mvt", "bicg"):
        return math.ceil(p["N"] / key[0])
    if form == "sten1d":
        return math.ceil(p["N"] / key[0])
    if form == "sten2d":
        d = p["N"] - 2 * p["r"]
        return math.ceil(d / key[0]) ** 2
    if form == "sibling":
        return math.ceil((p["N"] - 2) / key[0])
    raise ValueError(form)


def area_of(entry, key):
    if entry["form"] in ("gemm", "gemm_rect", "chain2", "chain3"):
        return key[0] * key[1]
    if entry["form"] == "sten2d":
        return key[0] * key[0]
    return key[0]


def tk_of(entry, key):
    if entry["form"] in ("gemm", "gemm_rect", "chain2", "chain3"):
        return key[2]
    if entry["form"] in ("mvt", "bicg"):
        return key[1]
    return 0


def rule_pick(entry, keys):
    """A0 = R_full：块数≥SM（不可达退 SM/2，再退 0）→ 最大面积 → 最大 TK → S 升序。"""
    scored = []
    for k in keys:
        scored.append((blocks_of(entry, k), area_of(entry, k), tk_of(entry, k), k))
    for thr in (SM, SM // 2, 0):
        cand = [x for x in scored if x[0] >= thr]
        if cand:
            break
    if not cand:
        return None
    cand.sort(key=lambda x: (-x[1], -x[2], x[3][-1], x[3]))
    return cand[0][3]


# ---------------------------------------------------------------------------
# 单 kernel 分析
# ---------------------------------------------------------------------------
def analyze_kernel(entry, rec, tbl):
    """rec = holdout_decisions 的 kernel 记录；tbl = {key: time}。返回结果 dict。"""
    tau, best = tau_of(tbl) if tbl else (0.02, None)
    best_key = min(tbl, key=lambda k: tbl[k]) if tbl else None
    # 显式转 key（dict.values() 顺序不可靠）
    def to_key(cfg):
        if not cfg:
            return None
        if "BM" in cfg:
            return (cfg["BM"], cfg["BN"], cfg["TK"], cfg["S"])
        if "TK" in cfg:
            return (cfg["T"], cfg["TK"], cfg["S"])
        return (cfg["T"], cfg["S"])
    a_keys = {a: to_key(rec[a]["config"]) for a in ("a1", "a2")}
    a0_key = rule_pick(entry, list(tbl.keys())) if tbl else None

    arms = {}
    for name, key in (("a0", a0_key), ("a1", a_keys["a1"]), ("a2", a_keys["a2"]),
                      ("a3", best_key)):
        if key is None:
            arms[name] = dict(config=None, rank=None, pct=None, regret=None,
                              hit=False, note="REJECT/未测")
        elif not tbl:
            arms[name] = dict(config=key, rank=None, pct=None, regret=None,
                              hit=False, note="无日志")
        else:
            m = metrics(tbl, key, tau, best)
            if m is None:
                arms[name] = dict(config=key, rank=None, pct=None, regret=None,
                                  hit=False, note="配置不在日志（未测）")
            else:
                arms[name] = dict(config=key, rank=m["rank"], n=m["n"],
                                  pct=m["pct"], regret=m["regret"], hit=m["hit"],
                                  time_us=m["time"], note="")

    # 前沿 / false-pruning
    fkeys = [tuple(x) for x in rec["frontier"]["keys"]]
    enum_keys = {P.cfg_key(c) for c in P.enumerate_configs(entry)}
    false_prune = None
    outside = None
    if best_key is not None:
        if best_key not in enum_keys:
            outside = True
        else:
            outside = False
            false_prune = best_key not in set(fkeys)
    return dict(id=entry["id"], name=entry["name"], family=entry["family"],
                primary=entry["primary"], tau=tau, best_key=best_key,
                best_time=(best if tbl else None), n=len(tbl),
                arms=arms, frontier=rec["frontier"],
                frontier_keys=fkeys, false_prune=false_prune,
                oracle_outside_space=outside,
                double_hit={a: (None if arms[a]["pct"] is None else
                                (arms[a]["pct"] <= 0.2 and arms[a]["regret"] <= 0.2))
                            for a in arms})


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
def selftest():
    entry = dict(id="T", name="test_gemm", form="gemm", family="gemm",
                 params=dict(M=1152, N=1152, K=1152), primary=True)
    cfgs = P.enumerate_configs(entry)
    # 造一个明确的赢家：128/128/32/1 最快；同 tile 的 S 越深越慢；其他 tile 更慢
    tbl = {}
    for c in cfgs:
        base = 100.0
        base += 50.0 if (c["BM"], c["BN"]) != (128, 128) else 0.0
        base += 0.5 * abs(c["TK"] - 32)
        base += 9.0 * (c["S"] - 1)  # 深度越深越慢，赢家 S=1
        tbl[P.cfg_key(c)] = base
    best_key = min(tbl, key=lambda k: tbl[k])
    assert best_key == (128, 128, 32, 1), best_key
    tau, best = tau_of(tbl)
    assert abs(best - 100.0) < 1e-9
    m1 = metrics(tbl, (128, 128, 32, 1), tau)
    assert m1["rank"] == 1 and abs(m1["regret"]) < 1e-12 and m1["hit"]
    m2 = metrics(tbl, (128, 128, 32, 4), tau)
    assert m2["rank"] > 1 and m2["regret"] > 0.2 and not m2["hit"]
    k = rule_pick(entry, list(tbl.keys()))
    assert blocks_of(entry, k) >= SM, (k, blocks_of(entry, k))
    # 前沿：全候选可行 → false-prune 0
    recs = dict(a1=dict(config=dict(BM=128, BN=128, TK=32, S=1)),
                a2=dict(config=dict(BM=128, BN=128, TK=32, S=1)),
                frontier=dict(keys=[list(x) for x in tbl.keys()]))
    r = analyze_kernel(entry, recs, tbl)
    assert r["false_prune"] is False and r["arms"]["a2"]["hit"]
    print(f"[selftest] OK：τ={tau:.3f}，A3={best_key}，A0={k}，"
          f"A2 rank={r['arms']['a2']['rank']}/{r['arms']['a2']['n']}")
    return 0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_key(k):
    if k is None:
        return "REJECT"
    return "/".join(str(x) for x in k)


def report_md(results, suite, decisions, logs_dir):
    lines = []
    lines.append("# 四臂裁判结果（留出集）\n")
    lines.append(f"> 生成：`python tools/four_arms.py --logs {os.path.relpath(logs_dir, WS)}`；"
                 f"{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}")
    lines.append(f"> suite sha256：`{decisions.get('suite_sha256','?')}`；"
                 f"网格：τ=max(2%, 前10% IQR)；严格名次=比它快超过τ的配置数+1。\n")
    if not any(r["n"] for r in results):
        lines.append("> ⚠ **尚无日志**：本文件是空表模板；把日志放进 `--logs` 目录后重跑本脚本覆盖。\n")
    lines.append("## 1. 决策与前沿\n")
    lines.append("| id | family | A1(v1.8.5) | A2(当前) | A0(规则) | A3(oracle) | τ | 前沿/硬可行 | false-prune |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        arms = r["arms"]
        f = r["frontier"]
        lines.append(f"| {r['id']} | {r['family']} | {fmt_key(arms['a1']['config'])} | "
                     f"{fmt_key(arms['a2']['config'])} | {fmt_key(arms['a0']['config'])} | "
                     f"{fmt_key(arms['a3']['config'])} | {r['tau']*100:.1f}% | "
                     f"{f['n_frontier']}/{f['n_hard']} | "
                     f"{'—' if r['false_prune'] is None else ('**YES**' if r['false_prune'] else '0')} |")
    lines.append("\n## 2. 质量（tie-aware）\n")
    lines.append("| id | A1 名次/距优/命中 | A2 名次/距优/命中 | A0 名次/距优/命中 | A2 双达标 |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        def cell(a):
            m = r["arms"][a]
            if m["rank"] is None:
                return "—"
            return f"{m['rank']}/{m['n']} · {m['regret']*100:+.1f}% · {'✓' if m['hit'] else '✗'}"
        lines.append(f"| {r['id']} | {cell('a1')} | {cell('a2')} | {cell('a0')} | "
                     f"{'✓' if r['double_hit']['a2'] else '✗'} |")
    lines.append("\n## 3. 汇总\n")
    run = [r for r in results if r["n"] > 0]
    lines.append(f"- 有日志 kernel：{len(run)}/{len(results)}（主集 "
                 f"{sum(1 for r in run if r['primary'])}/12）")
    for a in ("a0", "a1", "a2"):
        ms = [r["arms"][a] for r in run if r["arms"][a]["rank"] is not None]
        if not ms:
            continue
        hits = sum(1 for m in ms if m["hit"])
        med_pct = statistics.median([m["pct"] for m in ms])
        med_reg = statistics.median([m["regret"] for m in ms])
        dh = sum(1 for r in run if r["double_hit"][a])
        lines.append(f"- {a.upper()}：命中 {hits}/{len(ms)}；名次百分位中位 "
                     f"{med_pct*100:.1f}%；距优中位 {med_reg*100:+.1f}%；双达标 {dh}/{len(run)}")
    fp = [r for r in run if r["false_prune"]]
    lines.append(f"- false-pruning：{len(fp)}/{len(run)}"
                 + ("（**未通过**）" if fp else "（0，通过）"))
    outside = [r for r in run if r["oracle_outside_space"]]
    if outside:
        lines.append(f"- ⚠ oracle 不在候选空间：{[r['id'] for r in outside]}（协议问题，先查）")
    lines.append("\n## 4. 判据对照（预注册 §7）\n")
    lines.append("| 判据 | 结果 |")
    lines.append("|---|---|")
    a2 = [r for r in run if r["primary"] and r["arms"]["a2"]["rank"] is not None]
    hits = sum(1 for r in a2 if r["arms"]["a2"]["hit"])
    dh = sum(1 for r in a2 if r["double_hit"]["a2"])
    ok_test = len(a2) >= 10
    lines.append(f"| 主集可测 ≥10 | {len(a2)}/12 {'✓' if ok_test else '✗'} |")
    lines.append(f"| A2 命中 ≥9/12 | {hits}/{len(a2) or 12} {'✓' if hits >= 9 else '✗'} |")
    lines.append(f"| 双达标 ≥80% | {dh}/{len(a2) or 12} {'✓' if dh >= 0.8 * max(1, len(a2)) else '✗'} |")
    lines.append(f"| false-pruning=0 | {len(fp)} {'✓' if not fp else '✗'} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--logs", default=DEFAULT_LOGS)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--json", default=os.path.join(WS, "docs", "evidence", "four_arms.json"))
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    suite = load_json(SUITE)
    dec = load_json(DECISIONS)
    by_id = {r["id"]: r for r in dec["kernels"]}
    results = []
    for e in suite["kernels"]:
        rec = by_id[e["id"]]
        log = os.path.join(args.logs, rec["name"] + ".log")
        tbl = parse_log(log) if os.path.exists(log) else {}
        results.append(analyze_kernel(e, rec, tbl))
    md = report_md(results, suite, dec, args.logs)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(md)
    print(f"[写出] {args.report}\n[写出] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
