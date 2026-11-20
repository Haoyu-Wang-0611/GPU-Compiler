#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rule_baseline.py — "只用约束、不用成本模型"的三行规则 baseline（零 GPU）

规则（三个变体，都必须报，因为变体选择本身就是结论的一部分）
  R_half : 候选(日志里已测=smem 可行) → 块数 ≥ 64 → 按 (tile 面积, TK) 降序 → 取 S=1
  R_full : 同上，但块数 ≥ 128（不可达时退回 ≥64）
  R_naive: 不加占用守卫（直接最大面积）—— 这是"稻草人"，用来说明守卫不可省

评分：规则选中的配置 → 查日志里的实测时间 → rank / 距优。
若规则选中的配置不在日志中（未测/不可行），标"未测"。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import decide_dev as D  # noqa: E402
import replay_dev as R  # noqa: E402

# kernel → (M, N) 或 (N,)；用于算块数
SHAPE = {
    "gemm4096": (4096, 4096), "gemm1024": (1024, 1024), "gemm512": (512, 512),
    "gemm500": (500, 500), "twomm": (512, 512), "three_mm": (512, 512),
    "syrk": (512, 512), "symm": (512, 512),
    "atax": (8192,), "mvt": (512,), "bicg": (512,),
    "jacobi2d": (1022,), "jacobi1d": (16777214,),
    "heat1d24": (1 << 24,), "heat1d20": (1 << 20,),
}


def blocks_of(fam, key, shape):
    if fam == "gemm":
        BM, BN = key[0], key[1]
        return -(-shape[0] // BM) * (-(-shape[1] // BN))
    T = key[0]
    n = shape[0]
    if fam == "stencil1":
        # jacobi2d 是二维：用 n² 近似块数（interior=1022 是边长）
        return -(-n // T) ** 2 if n < 5000 else -(-n // T)
    return -(-n // T)


def area_of(fam, key):
    return key[0] * key[1] if fam == "gemm" else key[0]


def sort_key(fam, key):
    # 面积降序、TK 降序、S 升序（S=1 优先）
    tk = key[2] if fam == "gemm" else key[1]
    s = key[-1]
    return (-area_of(fam, key), -tk, s)


def pick(fam, tbl, shape, min_blocks):
    cand = [k for k in tbl if blocks_of(fam, k, shape) >= min_blocks]
    if not cand:
        return None
    return sorted(cand, key=lambda k: sort_key(fam, k))[0]


def main():
    print(f"{'kernel':10s} {'模型配置':>18s} {'模型距优':>9s} | "
          f"{'R_half':>16s} {'距优':>8s} | {'R_naive':>16s} {'距优':>8s}")
    rows = []
    for name, path, fam, note in R.ROWS:
        shape = SHAPE[name]
        tbl = R.parse(path, fam)
        if not tbl:
            continue
        best = min(tbl.values())
        mk = R.key_of(D.decide(D.DEV_KERNELS[name][0], "hard")["config"], fam)
        mt = tbl.get(mk)
        if mt is None:
            continue
        g_model = mt / best - 1.0

        def score(k):
            if k is None:
                return None, None
            t = tbl.get(k)
            return k, (t / best - 1.0) if t is not None else None

        rh = pick(fam, {k: 0 for k in tbl}, shape, 64)
        rf = pick(fam, {k: 0 for k in tbl}, shape, 128) or rh
        rn = pick(fam, {k: 0 for k in tbl}, shape, 0)
        kh, gh = score(rh)
        kf, gf = score(rf)
        kn, gn = score(rn)
        rows.append((name, g_model, gh, gn, gf))
        f = lambda k: "/".join(str(x) for x in k) if k else "—"
        g = lambda x: f"{x*100:+7.1f}%" if x is not None else "   未测"
        print(f"{name:10s} {f(mk):>18s} {g(g_model):>9s} | {f(kh):>16s} {g(gh):>8s} | "
              f"{f(kf):>16s} {g(gf):>8s} | {f(kn):>16s} {g(gn):>8s}")

    import statistics as st
    gm = [r[1] for r in rows]
    gh = [r[2] for r in rows if r[2] is not None]
    gf = [r[4] for r in rows if r[4] is not None]
    gn = [r[3] for r in rows if r[3] is not None]
    print()
    print(f"样本 {len(rows)} 行")
    print(f"模型   ：中位 {st.median(gm)*100:+.1f}%  最差 {max(gm)*100:+.1f}%")
    print(f"R_half ：中位 {st.median(gh)*100:+.1f}%  最差 {max(gh)*100:+.1f}%  （可用 {len(gh)}/{len(rows)}）")
    print(f"R_full ：中位 {st.median(gf)*100:+.1f}%  最差 {max(gf)*100:+.1f}%  （可用 {len(gf)}/{len(rows)}）")
    print(f"R_naive：中位 {st.median(gn)*100:+.1f}%  最差 {max(gn)*100:+.1f}%  （可用 {len(gn)}/{len(rows)}）")
    win = sum(1 for r in rows if r[2] is not None and r[1] < r[2])
    print(f"\n模型优于 R_half 的行数：{win}/{len([r for r in rows if r[2] is not None])}")


if __name__ == "__main__":
    main()
