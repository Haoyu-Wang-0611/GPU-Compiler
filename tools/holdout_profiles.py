#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""holdout_profiles.py — 留出套件的"模型档案 + 前沿"构造（离线）

职责：
  1. 把 `daima/holdout/suite.json` 的每个 entry 变成 `decide_dev` 能吃的 profile；
  2. 枚举该族的候选空间，给出**硬可行集**与**收缩后前沿**（供 false-pruning 用）。

纪律：
  - 不 import GPU / 不跑 MLIR；
  - 前沿判定必须与 `tools/decide_dev.py` 的可行性口径一致（同一份公式，复制而非改它）；
  - 只做"是否可行 + 是否在收缩线内"，不做打分（打分仍由 decide_dev 负责）。
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import decide_dev as D  # noqa: E402
import l1model as L  # noqa: E402

HW = L.HW
SM = HW["numSms"]
CAP = HW["smemCap"]


# ---------------------------------------------------------------------------
# profile 构造
# ---------------------------------------------------------------------------
def _acc(name, dims, access, shape, read=True, written=False, elem=4, role=None):
    return L.Acc(name, elem, access, shape, read=read, written=written, role=role)


def profile_sten1d(N, r):
    offs = [(0, d) for d in range(-r, r + 1)]
    A = _acc("A", 1, [[(0, o) for (_, o) in offs]], [N], role="A")
    B = _acc("B", 1, [[(0, 0)]], [N], read=False, written=True, role="B")
    return dict(kind="stencil1d", N=N, flops_per_iter=float(2 * r + 1),
                legacy_base=[(2 * r + 1, 4), (1, 4)], accs=[A, B])


def profile_sten2d(N, r, T):
    d = N - 2 * r
    offs = [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)]
    A = _acc("A", 2, [[(1, di) for (di, _) in offs], [(2, dj) for (_, dj) in offs]],
             [d, d], role="A")
    B = _acc("B", 2, [[(1, 0)], [(2, 0)]], [d, d], read=False, written=True, role="B")
    return dict(kind="stencil2d", nloops=3, spatial=[1, 2], tsteps=T,
                dim1=d, dim2=d, flops_per_iter=float((2 * r + 1) ** 2), accs=[A, B])


def profile_sibling(N, T):
    interior = N - 2
    A = _acc("A", 1, [[(1, -1), (1, 0), (1, 1)]], [interior], role="A")
    B = _acc("B", 1, [[(2, -1), (2, 0), (2, 1)]], [interior], role="B")
    return dict(kind="sibling", nloops=3, spatial=[1, 2], tsteps=T,
                interior=interior, flops_per_iter=6.0 / interior,
                access_expanded=[], accs=[A, B])


def build_profile(entry):
    form, p = entry["form"], entry["params"]
    if form == "sten1d":
        return profile_sten1d(p["N"], p["r"])
    if form == "sten2d":
        return profile_sten2d(p["N"], p["r"], p["T"])
    if form == "sibling":
        return profile_sibling(p["N"], p["T"])
    if form == "gemm":
        return L._gemm(p["M"], p["N"], p["K"])
    if form == "gemm_rect":
        return L._gemm(p["M"], p["N"], p["K"])
    if form == "chain2":
        return L._twomm(p["SZ"])
    if form == "chain3":
        return D.profile_three_mm(p["SZ"])
    if form == "mvt" or form == "bicg":
        return L._atax(p["N"])
    raise ValueError(f"unknown form: {form}")


def family_of(entry):
    return entry["family"]


# ---------------------------------------------------------------------------
# 候选枚举
# ---------------------------------------------------------------------------
def enumerate_configs(entry):
    """返回该族的候选配置列表（dict）。与 decide_dev 的枚举止一致。"""
    form, p = entry["form"], entry["params"]
    out = []
    if form in ("gemm", "gemm_rect", "chain2", "chain3"):
        for BM in (64, 128, 256):
            for BN in (64, 128, 256):
                if BM == 256 and BN == 256:
                    continue
                for TK in (16, 32, 64):
                    for S in (1, 2, 3, 4):
                        out.append(dict(BM=BM, BN=BN, TK=TK, S=S))
    elif form in ("mvt", "bicg"):
        for T in (256, 128, 64, 32, 16, 8):
            for TK in (64, 32, 16):
                for S in (1, 2, 3, 4):
                    out.append(dict(T=T, TK=TK, S=S))
    elif form == "sten2d":
        for T in (16, 32, 64, 128):
            for S in (1, 2, 3, 4):
                out.append(dict(T=T, S=S))
    elif form in ("sten1d", "sibling"):
        N = p["N"]
        if form == "sibling":
            limit = N - 1
        else:
            limit = N
        T = 16
        while T <= limit:
            for S in (1, 2, 3, 4):
                out.append(dict(T=T, S=S))
            T *= 2
    return out


# ---------------------------------------------------------------------------
# 可行性（与模型同口径）
# ---------------------------------------------------------------------------
def _gemm_hard(prof, cfg, occ_ok=True):
    a, b, c = prof["A"], prof["B"], prof["C"]
    M, N, K = prof["M"], prof["N"], prof["K"]
    tiles = [0] * prof["nloops"]
    tiles[prof["red"]] = cfg["TK"]
    tiles[prof["out"][0]] = cfg["BM"]
    tiles[prof["out"][1]] = cfg["BN"]
    smem = 0
    seen = set()
    for acc in (a, b, c):
        if acc.name in seen:
            continue
        seen.add(acc.name)
        v = L.window(acc, tiles)
        if v < 0:
            return None
        if L.is_stream(acc, prof["reds"]):
            smem += v * (cfg["S"] + 1)
    if smem > CAP:
        return None
    blocks = math.ceil(M / cfg["BM"]) * math.ceil(N / cfg["BN"])
    if occ_ok and blocks < SM // 2:
        return None
    return dict(smem=smem, blocks=blocks)


def _chain_hard(prof, cfg):
    ok = True
    blocks = []
    for p in prof["passes"]:
        tiles = [0] * prof["nloops"]
        tiles[p["red"]] = cfg["TK"]
        tiles[p["out"][0]] = cfg["BM"]
        tiles[p["out"][1]] = cfg["BN"]
        agg = D._chain_panels(p, tiles)
        if agg is None:
            return None
        smem, copyB, outB = agg
        if smem * (cfg["S"] + 1) > CAP:
            return None
        nb = math.ceil(p["M"] / cfg["BM"]) * math.ceil(p["N"] / cfg["BN"])
        blocks.append(nb)
    return dict(blocks=min(blocks), smem=smem * (cfg["S"] + 1))


def _gemv_hard(prof, cfg):
    tiles = [0] * prof["nloops"]
    for p in prof["passes"]:
        tiles[p["out"]] = cfg["T"]
        tiles[p["red"]] = cfg["TK"]
    smem = 0
    reds = [p["red"] for p in prof["passes"]]
    for acc in prof["accs"]:
        v = L.window(acc, tiles)
        if v < 0:
            return None
        smem += v * ((cfg["S"] + 1) if L.is_stream(acc, reds) else 1)
    if smem > CAP:
        return None
    blocks = min(math.ceil(p["M"] / cfg["T"]) for p in prof["passes"])
    return dict(smem=smem, blocks=blocks)


def _sten1d_hard(prof, cfg):
    N, T = prof["N"], cfg["T"]
    if T > N:
        return None
    bytes_ = 0
    for base_elems, elem in prof["legacy_base"]:
        bytes_ += base_elems * elem + (T - 1) * elem
    smem = bytes_ * (cfg["S"] + 1)
    if smem > CAP:
        return None
    numTiles = math.ceil(N / T)
    if N >= 8192 and numTiles < 512:
        return None
    return dict(smem=smem, blocks=numTiles)


def _sten2d_hard(prof, cfg):
    T, S = cfg["T"], cfg["S"]
    tiles = [0] * prof["nloops"]
    tiles[prof["spatial"][0]] = T
    tiles[prof["spatial"][1]] = T
    smem = 0
    for acc in prof["accs"]:
        v = L.window(acc, tiles)
        if v < 0:
            return None
        if acc.read:
            smem += v * (S + 1)
    if smem > CAP:
        return None
    nb = math.ceil(prof["dim1"] / T) * math.ceil(prof["dim2"] / T)
    return dict(smem=smem, blocks=nb)


def _sibling_hard(prof, cfg):
    T, S = cfg["T"], cfg["S"]
    if T > prof["interior"] + 1:
        return None
    tiles = [0] * prof["nloops"]
    tiles[prof["spatial"][0]] = T
    tiles[prof["spatial"][1]] = T
    smem = 0
    for acc in prof["accs"]:
        v = L.window(acc, tiles)
        if v < 0:
            return None
        if acc.read:
            smem += v * (S + 1)
    if smem > CAP:
        return None
    return dict(smem=smem, blocks=math.ceil(prof["interior"] / T))


def feasible(entry, cfg, apply_occ=True):
    """返回 dict(smem, blocks) 或 None。apply_occ=True 时叠加占用收缩（块数≥SM/2）。"""
    prof = build_profile(entry)
    form = entry["form"]
    if form in ("gemm", "gemm_rect"):
        r = _gemm_hard(prof, cfg, occ_ok=apply_occ)
        return None if r is None else dict(smem=r["smem"], blocks=r["blocks"])
    if form in ("chain2", "chain3"):
        r = _chain_hard(prof, cfg)
        if r is None:
            return None
        if apply_occ and r["blocks"] < SM // 2:
            return None
        return r
    if form in ("mvt", "bicg"):
        r = _gemv_hard(prof, cfg)
        if r is None:
            return None
        if apply_occ and r["blocks"] < SM // 2:
            return None
        return r
    if form == "sten1d":
        r = _sten1d_hard(prof, cfg)
        return r  # legacy 1D 分支本来就没有占用门槛
    if form == "sten2d":
        r = _sten2d_hard(prof, cfg)
        if r is None:
            return None
        if apply_occ and r["blocks"] < SM // 2:
            return None
        return r
    if form == "sibling":
        return _sibling_hard(prof, cfg)
    raise ValueError(form)


def frontier(entry):
    """返回 (hard, contracted, used)：hard=硬可行全集；contracted=占用收缩；used=实际前沿。"""
    configs = enumerate_configs(entry)
    hard = [c for c in configs if feasible(entry, c, apply_occ=False) is not None]
    contracted = [c for c in configs if feasible(entry, c, apply_occ=True) is not None]
    used = contracted if contracted else hard
    return hard, contracted, used


def cfg_key(cfg):
    if "BM" in cfg:
        return (cfg["BM"], cfg["BN"], cfg["TK"], cfg["S"])
    if "TK" in cfg:
        return (cfg["T"], cfg["TK"], cfg["S"])
    return (cfg["T"], cfg["S"])


if __name__ == "__main__":
    import json
    suite = os.path.join(os.path.dirname(_HERE), "daima", "holdout", "suite.json")
    with open(suite, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for e in payload["kernels"]:
        prof = build_profile(e)
        hard, contracted, used = frontier(e)
        print(f"{e['id']:3s} {e['name']:24s} enum={len(enumerate_configs(e)):3d} "
              f"hard={len(hard):3d} contracted={len(contracted):3d} used={len(used):3d}")
