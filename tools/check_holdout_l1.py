#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_holdout_l1.py — 留出集冒烟校验（C++ v1.8.5 vs Python A1）

用途：在跑 GPU 之前，用服务器上的 `mlir-l1feas` 跑 16 个留出 `.mlir`，
把输出里的 `Best:` 与 `docs/evidence/holdout_decisions.json` 的 **A1 列**逐条对比。
期望 16/16 一致；不一致 = 生成的 IR / 分类 / Python 档案口径有偏差，先修再上 GPU。

用法：
  # 1) 服务器上生成 16 份输出（见 docs/holdout/服务器跑批手册.md）
  # 2) 本机（或服务器）校验：
  python tools/check_holdout_l1.py --l1-dir l1_holdout
  python tools/check_holdout_l1.py --l1-dir l1_holdout --write

输出文件命名：<kernel name>.txt（如 gemm_1152.txt；也接受 .log）。
"""
import argparse
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(_HERE)
SUITE = os.path.join(WS, "daima", "holdout", "suite.json")
DECISIONS = os.path.join(WS, "docs", "evidence", "holdout_decisions.json")
RE_BEST = re.compile(r"Best:\s*([^\n]+)")
RE_FEAS = re.compile(r"可行\s*(\d+)\s*/\s*枚举\s*(\d+)")
RE_DETECT = re.compile(r"detected:\s*([^\n]+)")


def parse_l1(path):
    """返回 dict(best_raw, keys, feas, unit, detected)。"""
    txt = open(path, "r", encoding="utf-8", errors="replace").read()
    out = dict(best_raw=None, keys={}, feas=None, detected=None)
    m = RE_DETECT.search(txt)
    if m:
        out["detected"] = m.group(1).strip()
    m = RE_BEST.search(txt)
    if m:
        raw = m.group(1).strip()
        out["best_raw"] = raw
        for k, v in re.findall(r"(\w+)=(\d+)", raw):
            out["keys"][k] = int(v)
    m = RE_FEAS.search(txt)
    if m:
        out["feas"] = (int(m.group(1)), int(m.group(2)))
    return out


def norm_expected(entry, cfg):
    """A1 config dict → 与 C++ `keys` 同名的 dict。"""
    form = entry["form"]
    if form in ("gemm", "gemm_rect", "chain2", "chain3"):
        return dict(BM=cfg["BM"], BN=cfg["BN"], TK=cfg["TK"], Stages=cfg["S"])
    if form in ("mvt", "bicg"):
        return dict(T=cfg["T"], TK=cfg["TK"], Stages=cfg["S"])
    if form == "sten2d":
        return dict(Tile2d=cfg["T"], Stages=cfg["S"])
    if form in ("sten1d", "sibling"):
        return dict(Tile=cfg["T"], Stages=cfg["S"])
    raise ValueError(form)


def compare(entry, rec, parsed):
    """返回 (ok, note, expected_keys, actual_keys)。"""
    cfg = rec["a1"]["config"]
    if cfg is None:
        return False, "A1=REJECT（预期应有解）", None, None
    exp = norm_expected(entry, cfg)
    act = parsed["keys"]
    # 只比较 expected 里的键（C++ 可能多打印别的键）
    diffs = [f"{k}:{exp[k]}→{act.get(k)}" for k in exp if act.get(k) != exp[k]]
    if not act:
        return False, "未解析到 Best（可能 REJECT/语法错）", exp, act
    if diffs:
        return False, "不一致：" + ", ".join(diffs), exp, act
    return True, "", exp, act


def find_file(l1_dir, name):
    for ext in (".txt", ".log"):
        p = os.path.join(l1_dir, name + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1-dir", default="l1_holdout")
    ap.add_argument("--suite", default=SUITE)
    ap.add_argument("--decisions", default=DECISIONS)
    ap.add_argument("--out", default=os.path.join(WS, "docs", "evidence", "holdout_smoke.md"))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    suite = json.load(open(args.suite, "r", encoding="utf-8"))
    dec = json.load(open(args.decisions, "r", encoding="utf-8"))
    by_id = {r["id"]: r for r in dec["kernels"]}
    rows, bad = [], 0
    for e in suite["kernels"]:
        rec = by_id[e["id"]]
        path = find_file(args.l1_dir, rec["name"])
        if not path:
            rows.append((e["id"], rec["name"], "缺文件", "—", "—", False))
            bad += 1
            continue
        p = parse_l1(path)
        ok, note, exp, act = compare(e, rec, p)
        # 可行计数（副校验，不单独判失败）
        fnote = ""
        if p["feas"] and rec["a1"].get("n_feas") is not None:
            fnote = f"feas {p['feas'][0]}/{p['feas'][1]} vs {rec['a1']['n_feas']}/{rec['a1']['n_enum']}"
        rows.append((e["id"], rec["name"],
                     "/".join(str(exp[k]) for k in exp) if exp else "—",
                     "/".join(str(act.get(k)) for k in exp) if act and exp else "—",
                     (p["detected"] or "") + ("；" + fnote if fnote else ""),
                     ok))
        if not ok:
            bad += 1
            print(f"[FAIL] {e['id']} {rec['name']}: {note}  (detected={p['detected']})")

    print(f"\n{'id':3s} {'name':24s} {'期望(A1)':22s} {'实际':22s} {'detected/feas':28s} ok")
    for r in rows:
        print(f"{r[0]:3s} {r[1]:24s} {r[2]:22s} {r[3]:22s} {r[4]:28s} {'✓' if r[5] else '✗'}")
    print(f"\n汇总：{len(rows)} 个 kernel，匹配 {len(rows)-bad}，不匹配/缺文件 {bad}")
    if args.write:
        out = args.out
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write("# 留出集冒烟校验（C++ v1.8.5 vs Python A1）\n\n")
            f.write(f"> 生成：`python tools/check_holdout_l1.py --l1-dir {args.l1_dir} --write`\n\n")
            f.write("| id | name | 期望(A1) | 实际 | detected/feas | ok |\n|---|---|---|---|---|---|\n")
            for r in rows:
                f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {'✓' if r[5] else '✗'} |\n")
            f.write(f"\n**汇总**：{len(rows)} 个，匹配 {len(rows)-bad}，不匹配 {bad}。\n")
        print(f"[写出] {out}")
    return 0 if bad == 0 else 1


def selftest():
    entry = dict(id="T", name="t", form="gemm", params={})
    rec = dict(a1=dict(config=dict(BM=128, BN=128, TK=16, S=4), n_feas=40, n_enum=96))
    line = ("[L1] detected: matmul\n"
            "[L1 Decision] Best: BM=128 BN=128 TK=16 Stages=4 predicted = 1.0e5 ns "
            "[搜索: 可行 40/枚举 96]\n"
            "[L1 Decision] 注意: 最优配置非整除（余数吸收）\n")
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(line)
        p = f.name
    parsed = parse_l1(p)
    os.unlink(p)
    ok, note, exp, act = compare(entry, rec, parsed)
    assert ok, note
    assert parsed["feas"] == (40, 96), parsed
    # 反例
    rec2 = dict(a1=dict(config=dict(BM=64, BN=64, TK=32, S=4), n_feas=10, n_enum=96))
    ok2, note2, _, _ = compare(entry, rec2, parsed)
    assert not ok2 and "不一致" in note2, (ok2, note2)
    print("[selftest] OK：解析/比较/计数 均通过")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main())
