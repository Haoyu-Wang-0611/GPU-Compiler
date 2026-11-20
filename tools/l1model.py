#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
l1model.py — L1Feas v1.8.2 决策模型的独立 Python 复现（B 线原型工作台）

用途
  1) 逐项复现 C++ 决策器的 predicted 值（自检：数值必须逐位对上服务器输出，
     对上即说明本文件与 L1Feas.cpp 同构；对不上就是复现有偏差，禁止用于结论）；
  2) 在不动 C++ 的前提下试做 B 线扩展（每 operand 独立流水深度 / 聚合带宽墙）；
  3) 给出"哪个 regime"的判读（带宽饱和 / 流水受限 / 延迟受限），
     这是判断 depth 决策在何处才有意义的工具。

模型口径与 L1Feas.cpp 对应关系（行号指 daima/L1Feas.cpp）
  - 机器常数            : :306-316
  - 窗口足迹            : windowFootprint :150-179（Tmax + (hi-lo)，dimCap 封顶）
  - GEMM 两态式 + L_min : :836-843
  - GEMV 两资源         : :670-685
  - stencil-2D          : :950-955
  - stencil 兄弟交换    : :1004-1030
  - stencil 1D legacy   : :1050-1069
  - 候选枚举与 tie-break : 严格 `<` 保留先枚举者；GEMM 升序、GEMV 降序

用法
  python tools/l1model.py --selftest          # 与服务器已记录 predicted 对表
  python tools/l1model.py --regime            # 三类 regime 判读（B 线机理前提）
  python tools/l1model.py --asym              # 非对称缓冲原型（M2 / 容量前沿）
  python tools/l1model.py --frontier          # 非对称释放出的可行前沿
"""
import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------- 机器常数
HW = dict(
    smBw=7.21,        # GB/s per SM（naive 922.8/128，10-24 复现）
    smCompute=630.0,  # GFLOPS per SM
    rExposed=330.0,   # ns
    aT=0.60,          # ns/elem
    pExp=1.20,
    deltaFloor=52.0,  # ns
    launch=3000.0,    # ns
    numSms=128,
    smemCap=99 * 1024,
    LminGemm=1739.0,  # ns
    LminGemv=820.0,
    LminGemvThin=1260.0,
)

NAN = float("-inf")
L2_BYTES = 72 * 1024 * 1024   # 4090 L2（v1.9 GEMV thin 惩罚的 regime 判据）

# 机器常数预设。4090 = 已标定（唯一可用于决策的口径）；
# a100 = 仅按规格换算的"未标定"预设，只用于 regime 判读（比例关系），不可用于决策。
PRESETS = {
    "4090": dict(),
    "a100": dict(
        smBw=1555.0 / 108,     # GB/s per SM（HBM2e 1555 GB/s / 108 SM）
        smCompute=19500.0 / 108,  # GFLOPS per SM（FP32 19.5 TFLOPS / 108 SM）
        numSms=108,
        smemCap=163 * 1024,
    ),
    "r6000": dict(
        smBw=960.0 / 142,      # GDDR6 960 GB/s / 142 SM
        smCompute=91100.0 / 142,
        numSms=142,
        smemCap=99 * 1024,
    ),
}


# ---------------------------------------------------------------- 数据结构
@dataclass
class Acc:
    """一个 memref 的访问描述（同 C++ AccessDesc）。"""
    name: str
    elem: int                          # 元素字节
    dims: List[List[Tuple[int, int]]]  # dims[d] = [(loop_idx, const_offset), ...]
    caps: List[int]                    # 该 memref 每维尺寸（dimCap）
    read: bool = True
    written: bool = False
    role: str = ""                     # GEMM 用："A"/"B"/"C"；仅 B 线原型使用


def window(acc: Acc, tiles: List[int]) -> int:
    """窗口足迹（字节）；-1 = 不可算（对应 C++ 返回 -1）。"""
    b = acc.elem
    for d, dim in enumerate(acc.dims):
        if not dim:
            continue
        tmax = -1
        lo = hi = dim[0][1]
        for (l, c) in dim:
            if l < 0 or l >= len(tiles):
                return -1
            tmax = max(tmax, tiles[l])
            lo = min(lo, c)
            hi = max(hi, c)
        b *= min(tmax + (hi - lo), acc.caps[d])
    return b


def is_stream(acc: Acc, red_loops) -> bool:
    """窗口维是否被归约环携带（→ 多缓冲 ×(S+1)）。"""
    for dim in acc.dims:
        for (l, _) in dim:
            if l in red_loops:
                return True
    return False


# ---------------------------------------------------------------- GEMM（单 nest）
def gemm_eval(prof, BM, BN, TK, S, hw=HW):
    """复现 L1Feas GEMM 模板（:750-848）。返回 dict 或 None（不可行）。"""
    a, b, c = prof["A"], prof["B"], prof["C"]
    M, N, K = prof["M"], prof["N"], prof["K"]
    tiles = [0] * prof["nloops"]
    tiles[prof["red"]] = TK
    tiles[prof["out"][0]] = BM
    tiles[prof["out"][1]] = BN

    smem, copyB, outB = 0, 0.0, 0.0
    fp = {}
    # 同 memref 只记一次（C++ 按 AccessDesc 合并，见 :101-110；syrk 的 A 双读即此情形）
    seen = set()
    uniq = []
    for acc in (a, b, c):
        if acc.name in seen:
            continue
        seen.add(acc.name)
        uniq.append(acc)
    for acc in uniq:
        v = window(acc, tiles)
        if v < 0:
            return None
        fp[acc.name] = v
        if is_stream(acc, prof["reds"]):
            smem += v * (S + 1)
            copyB += v
        else:
            outB += v
    if smem > hw["smemCap"]:
        return None

    numBlocks = math.ceil(M / BM) * math.ceil(N / BN)
    if numBlocks < hw["numSms"] // 2:
        return None

    waves = math.ceil(numBlocks / hw["numSms"])
    chunks = math.ceil(K / TK)
    copy = copyB / hw["smBw"]
    comp = 2.0 * BM * BN * TK / hw["smCompute"]
    eff = min(float(S), max(1.0, float(chunks - 1)))
    expo = max(hw["rExposed"] / (eff ** hw["pExp"]), hw["deltaFloor"])
    cWrite = outB / hw["smBw"]

    if S == 1:
        block = chunks * (copy + comp) + expo + cWrite
    else:
        Lcap = min(hw["LminGemm"], copy + comp)
        block = chunks * max(copy, comp, Lcap) + expo + cWrite
    # 每 chunk 固定开销（sync/wait/发射）：与深度无关的加性项。
    # v1.8.2 把这类开销写成 max() 的下界（Lcap），因此在 copy ≫ Lcap 的大 tile 上
    # 完全不可见 → TK 维零分辨率（512³ 实测 TK32→TK64 差 1.3 µs 正是此项）。
    block += chunks * hw.get("cChunk", 0.0)
    total = hw["launch"] + waves * block
    return dict(total=total, smem=smem, numBlocks=numBlocks, waves=waves, chunks=chunks,
                copy=copy, comp=comp, expo=expo, cWrite=cWrite, fp=fp,
                traffic=copyB, nonDiv=(K % TK != 0 or M % BM != 0 or N % BN != 0))


def decide_gemm(prof, hw=HW):
    best, nEnum, nFeas = None, 0, 0
    for BM in (64, 128, 256):
        for BN in (64, 128, 256):
            if BM == 256 and BN == 256:
                continue                      # 寄存器约束 :780
            for TK in (16, 32, 64):
                for S in (1, 2, 3, 4):
                    nEnum += 1
                    r = gemm_eval(prof, BM, BN, TK, S, hw)
                    if r is None:
                        continue
                    nFeas += 1
                    if best is None or r["total"] < best["total"]:
                        best = dict(r, BM=BM, BN=BN, TK=TK, S=S)
    return best, nFeas, nEnum


# ---------------------------------------------------------------- GEMM 组化链
def decide_gemm_chain(prof, hw=HW):
    """复现组化 pass-chain 分支（:466-543）：全 pass 共享 (BM,BN,TK,S)。"""
    P = len(prof["passes"])
    best, nEnum, nFeas = None, 0, 0
    for BM in (64, 128, 256):
        for BN in (64, 128, 256):
            if BM == 256 and BN == 256:
                continue
            for TK in (16, 32, 64):
                for S in (1, 2, 3, 4):
                    nEnum += 1
                    total = hw["launch"] * P
                    ok, nonDiv = True, False
                    for p in prof["passes"]:
                        tiles = [0] * prof["nloops"]
                        tiles[p["red"]] = TK
                        tiles[p["out"][0]] = BM
                        tiles[p["out"][1]] = BN
                        smem, copyB, outB = 0, 0.0, 0.0
                        for acc in p["accs"]:
                            v = window(acc, tiles)
                            if v < 0:
                                ok = False
                                break
                            if is_stream(acc, [p["red"]]):
                                smem += v * (S + 1)
                                copyB += v
                            else:
                                outB += v
                        if not ok or smem > hw["smemCap"]:
                            ok = False
                            break
                        M, N, K = p["M"], p["N"], p["K"]
                        nb = math.ceil(M / BM) * math.ceil(N / BN)
                        if nb < hw["numSms"] // 2:
                            ok = False
                            break
                        waves = math.ceil(nb / hw["numSms"])
                        chunks = math.ceil(K / TK)
                        copy = copyB / hw["smBw"]
                        comp = 2.0 * BM * BN * TK / hw["smCompute"]
                        eff = min(float(S), max(1.0, float(chunks - 1)))
                        expo = max(hw["rExposed"] / (eff ** hw["pExp"]), hw["deltaFloor"])
                        cWrite = outB / hw["smBw"]
                        if S == 1:
                            block = chunks * (copy + comp) + expo + cWrite
                        else:
                            block = chunks * max(copy, comp,
                                                 min(hw["LminGemm"], copy + comp)) + expo + cWrite
                        block += chunks * hw.get("cChunk", 0.0)
                        total += waves * block
                        if K % TK or M % BM or N % BN:
                            nonDiv = True
                    if not ok:
                        continue
                    nFeas += 1
                    if best is None or total < best["total"]:
                        best = dict(total=total, BM=BM, BN=BN, TK=TK, S=S, nonDiv=nonDiv)
    return best, nFeas, nEnum


# ---------------------------------------------------------------- GEMV 链
def decide_gemv(prof, hw=HW, thin_dram_only=False):
    """复现 pass-chain GEMV（:633-690），T 降序枚举（:628）。

    thin_dram_only=False → v1.8.5 行为：T<16 一律罚 1260；
    thin_dram_only=True  → v1.9：仅当该 pass 的 A 工作集 > L2 时才罚（lane×DRAM 双条件）。
    """
    P = len(prof["passes"])
    best, nEnum, nFeas = None, 0, 0
    for T in (256, 128, 64, 32, 16, 8):
        for TK in (64, 32, 16):
            for S in (1, 2, 3, 4):
                nEnum += 1
                tiles = [0] * prof["nloops"]
                for p in prof["passes"]:
                    tiles[p["out"]] = T
                    tiles[p["red"]] = TK
                smem = 0
                ok = True
                reds = [p["red"] for p in prof["passes"]]
                for acc in prof["accs"]:
                    v = window(acc, tiles)
                    if v < 0:
                        ok = False
                        break
                    smem += v * ((S + 1) if is_stream(acc, reds) else 1)
                if not ok or smem > hw["smemCap"]:
                    continue
                total = hw["launch"] * P
                guard = True
                for p in prof["passes"]:
                    M, K = p["M"], p["K"]
                    nb = math.ceil(M / T)
                    if nb < hw["numSms"] // 2:
                        guard = False
                        break
                    chunks = math.ceil(K / TK)
                    W = M * K * 4.0
                    thin = (T < 16) and ((not thin_dram_only) or W > L2_BYTES)
                    Lmin = hw["LminGemvThin"] if thin else hw["LminGemv"]
                    occ = min(1.0, nb / hw["numSms"])
                    bwEff = hw["numSms"] * hw["smBw"] * occ * max(0.6, 1.0 - 2.4 / T)
                    streamB = M * K * 4.0
                    total += max(chunks * Lmin, streamB / bwEff) + T * 4.0 / hw["smBw"]
                if guard:
                    nFeas += 1
                    if best is None or total < best["total"]:
                        best = dict(total=total, T=T, TK=TK, S=S)
    return best, nFeas, nEnum


# ---------------------------------------------------------------- stencil-2D
def decide_stencil2d(prof, hw=HW):
    best = None
    for T in (16, 32, 64, 128):
        for S in (1, 2, 3, 4):
            tiles = [0] * prof["nloops"]
            tiles[prof["spatial"][0]] = T
            tiles[prof["spatial"][1]] = T
            smem, copyB, ok = 0, 0, True
            for acc in prof["accs"]:
                v = window(acc, tiles)
                if v < 0:
                    ok = False
                    break
                copyB += v
                if acc.read:
                    smem += v * (S + 1)
            if not ok or smem > hw["smemCap"]:
                continue
            d1, d2 = prof["dim1"], prof["dim2"]
            nb = math.ceil(d1 / T) * math.ceil(d2 / T)
            if nb < hw["numSms"] // 2:
                continue
            waves = math.ceil(nb / hw["numSms"])
            copy = copyB / hw["smBw"]
            comp = prof["flops_per_iter"] * T * T / hw["smCompute"]
            eff = min(float(S), max(1.0, float(waves - 1)))
            R = max(hw["rExposed"] - hw["aT"] * T, 1.0)
            expo = max(R / (eff ** hw["pExp"]), hw["deltaFloor"])
            total = hw["launch"] + prof["tsteps"] * waves * (copy + comp + expo)
            if best is None or total < best["total"]:
                best = dict(total=total, T=T, S=S, smem=smem, waves=waves)
    return best


# ---------------------------------------------------------------- stencil 兄弟交换
def decide_sibling(prof, hw=HW):
    interior, tsteps = prof["interior"], prof["tsteps"]
    bits = []
    for m in prof["access_expanded"]:
        bits.append(m)
    best = None
    for S in (1, 2, 3, 4):
        T = 16
        while T <= interior + 1:
            tiles = [0] * prof["nloops"]
            tiles[prof["spatial"][0]] = T
            tiles[prof["spatial"][1]] = T
            smem, fpSum, ok = 0, 0, True
            for acc in prof["accs"]:
                v = window(acc, tiles)
                if v < 0:
                    ok = False
                    break
                fpSum += v
                if acc.read:
                    smem += v * (S + 1)
            if ok and smem <= hw["smemCap"]:
                flopPerElemPass = prof["flops_per_iter"] * interior / 2.0
                copyStep = 2 * fpSum
                numTiles = math.ceil(interior / T)
                waves = math.ceil(numTiles / hw["numSms"])
                eff = min(float(S), max(1.0, float(waves - 1)))
                copy = copyStep / hw["smBw"]
                comp = 2.0 * flopPerElemPass * T / hw["smCompute"]
                R = max(hw["rExposed"] - hw["aT"] * T, 1.0)
                expo = max(R / (eff ** hw["pExp"]), hw["deltaFloor"])
                total = hw["launch"] + tsteps * waves * (copy + comp + expo)
                if best is None or total < best["total"]:
                    best = dict(total=total, T=T, S=S, smem=smem, waves=waves)
            T *= 2
    return best


# ---------------------------------------------------------------- stencil 1D legacy
def decide_stencil1d(prof, hw=HW, min_tiles=512):
    """legacy 1D 分支（:1050-1069）：base = loopDepth=1 区域，非窗口法。

    min_tiles=512 → v1.8.5 行为；v1.9 传 numSms（128）。
    """
    N = prof["N"]
    best = None
    for S in (1, 2, 3, 4):
        for T in [16 << i for i in range(40)]:
            if T > N:
                break
            bytes_ = 0
            for base_elems, elem in prof["legacy_base"]:
                bytes_ += base_elems * elem + (T - 1) * elem
            smem = bytes_ * (S + 1)
            if smem > hw["smemCap"]:
                continue
            numTiles = math.ceil(N / T)
            if N >= 8192 and numTiles < min_tiles:
                continue
            waves = math.ceil(numTiles / hw["numSms"])
            eff = min(float(S), max(1.0, float(waves - 1)))
            copy = bytes_ / hw["smBw"]
            comp = prof["flops_per_iter"] * T / hw["smCompute"]
            R = max(hw["rExposed"] - hw["aT"] * T, 1.0)
            expo = max(R / (eff ** hw["pExp"]), hw["deltaFloor"])
            total = hw["launch"] + waves * (copy + comp + expo)
            if best is None or total < best["total"]:
                best = dict(total=total, T=T, S=S, smem=smem)
    return best


# ---------------------------------------------------------------- B 线：非对称深度
def asym_eval(prof, BM, BN, TK, SA, SB, hw=HW, variant="min"):
    """每 operand 独立缓冲深度。

    variant="min"   : 时间只认有效深度 S_eff = min(SA,SB)（保守、与现模型同构）；
    variant="little": Little 律 T_chunk >= copy_i/(S_i-1)（激进、可被现有数据证伪）。
    """
    a, b, c = prof["A"], prof["B"], prof["C"]
    M, N, K = prof["M"], prof["N"], prof["K"]
    tiles = [0] * prof["nloops"]
    tiles[prof["red"]] = TK
    tiles[prof["out"][0]] = BM
    tiles[prof["out"][1]] = BN
    fpA, fpB, fpC = window(a, tiles), window(b, tiles), window(c, tiles)
    if min(fpA, fpB, fpC) < 0:
        return None
    parts = [(fpA, SA), (fpB, SB)]
    smem = sum(f * (s + 1) for f, s in parts)          # 非流式 C 计 0（同现模型）
    if smem > hw["smemCap"]:
        return None
    numBlocks = math.ceil(M / BM) * math.ceil(N / BN)
    if numBlocks < hw["numSms"] // 2:
        return None
    waves = math.ceil(numBlocks / hw["numSms"])
    chunks = math.ceil(K / TK)
    copyA, copyB_ = fpA / hw["smBw"], fpB / hw["smBw"]
    comp = 2.0 * BM * BN * TK / hw["smCompute"]
    cWrite = fpC / hw["smBw"]
    if SA == 1 or SB == 1:
        # 任一 operand 无预取 → 该 operand 的装载完整暴露（串行态）
        load = (copyA if SA == 1 else 0.0) + (copyB_ if SB == 1 else 0.0)
        block = chunks * (load + comp) + max(hw["rExposed"] / max(1.0, min(SA, SB)) ** hw["pExp"],
                                            hw["deltaFloor"]) + cWrite
    else:
        if variant == "min":
            Se = min(SA, SB)
            eff = min(float(Se), max(1.0, float(chunks - 1)))
            expo = max(hw["rExposed"] / (eff ** hw["pExp"]), hw["deltaFloor"])
            Lcap = min(hw["LminGemm"], copyA + copyB_ + comp)
            block = chunks * max(copyA + copyB_, comp, Lcap) + expo + cWrite
        else:  # little
            Se = min(SA, SB)
            eff = min(float(Se), max(1.0, float(chunks - 1)))
            expo = max(hw["rExposed"] / (eff ** hw["pExp"]), hw["deltaFloor"])
            rateA = copyA / (SA - 1)
            rateB = copyB_ / (SB - 1)
            block = chunks * max(comp, rateA, rateB, hw["LminGemm"]) + expo + cWrite
    return dict(total=hw["launch"] + waves * block, smem=smem, waves=waves,
                chunks=chunks, copyA=copyA, copyB=copyB_, comp=comp,
                block=block, traffic=fpA + fpB)


def decide_asym(prof, hw=HW, variant="min", symmetric_only=False):
    best, nEnum, nFeas = None, 0, 0
    for BM in (64, 128, 256):
        for BN in (64, 128, 256):
            if BM == 256 and BN == 256:
                continue
            for TK in (16, 32, 64):
                pairs = [(s, s) for s in (1, 2, 3, 4)] if symmetric_only else \
                        [(sa, sb) for sa in (1, 2, 3, 4) for sb in (1, 2, 3, 4)]
                for (SA, SB) in pairs:
                    nEnum += 1
                    r = asym_eval(prof, BM, BN, TK, SA, SB, hw, variant)
                    if r is None:
                        continue
                    nFeas += 1
                    if best is None or r["total"] < best["total"]:
                        best = dict(r, BM=BM, BN=BN, TK=TK, SA=SA, SB=SB)
    return best, nFeas, nEnum


# ---------------------------------------------------------------- regime 判读
def regime(prof, BM, BN, TK, S, hw=HW):
    """把一次评估拆成三个"墙"，判读谁在支配。

    bw_wall : 该配置每 chunk 需搬的字节 × 并发块数（=128）折算到整卡带宽
              → 这是聚合带宽墙；若它 ≈ copy，说明 copy 项实际就是整卡带宽约束，
                增加深度不能降低它（深度不改变总流量）。
    comp    : 计算墙
    fixed   : 每 chunk 固定开销（L_min）与暴露延迟
    """
    r = gemm_eval(prof, BM, BN, TK, S, hw)
    if r is None:
        return None
    chipBw = hw["smBw"] * hw["numSms"]           # GB/s
    per_chunk_traffic = r["traffic"]             # 每块每 chunk 字节
    agg_ns = (per_chunk_traffic * hw["numSms"]) / chipBw
    Lmin = hw["LminGemm"] if S > 1 else 0.0
    return dict(copy_ns=r["copy"], agg_ns=agg_ns, comp_ns=r["comp"],
                Lmin_ns=Lmin, expo_ns=r["expo"], chunks=r["chunks"],
                waves=r["waves"], total=r["total"])


# ---------------------------------------------------------------- 已知 kernel profile
def mk(builder):
    return builder()


def _gemm(M, N, K):
    # 循环编号：L0=i(外) L1=j L2=k(内)，与 [L1] detected: reduction dim L2 一致
    A = Acc("A", 4, [[(0, 0)], [(2, 0)]], [M, K], role="A")
    B = Acc("B", 4, [[(2, 0)], [(1, 0)]], [K, N], role="B")
    C = Acc("C", 4, [[(0, 0)], [(1, 0)]], [M, N], read=False, written=True, role="C")
    return dict(kind="gemm", nloops=3, red=2, reds=[2], out=[0, 1],
                M=M, N=N, K=K, A=A, B=B, C=C)


def _syrk(n):
    # k 进读不进写 → 单归约；A 双读（A[i,k] 与 A[j,k]）→ dim0 双载体
    A = Acc("A", 4, [[(0, 0), (1, 0)], [(2, 0)]], [n, n], role="A")
    C = Acc("C", 4, [[(0, 0)], [(1, 0)]], [n, n], read=False, written=True, role="C")
    return dict(kind="gemm", nloops=3, red=2, reds=[2], out=[0, 1],
                M=n, N=n, K=n, A=A, B=A, C=C)


def _twomm(n):
    p1 = _gemm(n, n, n)
    pid = dict(red=2, out=[0, 1], M=n, N=n, K=n, accs=[p1["A"], p1["B"], p1["C"]])
    return dict(kind="gemm_chain", nloops=3, passes=[pid, pid])


def _jacobi2d(d1, d2, steps):
    A = Acc("A", 4, [[(1, -1), (1, 0), (1, 1)], [(2, -1), (2, 0), (2, 1)]], [d1, d2])
    B = Acc("B", 4, [[(1, 0)], [(2, 0)]], [d1, d2], read=False, written=True)
    return dict(kind="stencil2d", nloops=3, spatial=[1, 2], tsteps=steps,
                dim1=d1, dim2=d2, flops_per_iter=5.0, accs=[A, B])


def _sibling(interior, steps):
    A = Acc("A", 4, [[(1, -1), (1, 0), (1, 1)]], [interior], role="A")
    B = Acc("B", 4, [[(2, -1), (2, 0), (2, 1)]], [interior], role="B")
    return dict(kind="sibling", nloops=3, spatial=[1, 2], tsteps=steps,
                interior=interior, flops_per_iter=6.0 / interior,
                access_expanded=[], accs=[A, B])


def _heat1d(N, base=(3, 1)):
    return dict(kind="stencil1d", N=N, flops_per_iter=3.0,
                legacy_base=[(base[0], 4), (base[1], 4)],
                accs=[Acc("A", 4, [[(0, -1), (0, 0), (0, 1)]], [N]),
                      Acc("B", 4, [[(0, 0)]], [N], read=False, written=True)])


def _atax(n):
    A = Acc("A", 4, [[(0, 0)], [(1, 0)]], [n, n], role="A")
    tmp = Acc("tmp", 4, [[(0, 0)]], [n])
    x = Acc("x", 4, [[(1, 0)]], [n])
    p1 = dict(out=0, red=1, M=n, K=n)
    A2 = Acc("A", 4, [[(3, 0)], [(2, 0)]], [n, n], role="A")
    tmp2 = Acc("tmp", 4, [[(2, 0)]], [n])
    y = Acc("y", 4, [[(3, 0)]], [n], read=True)
    p2 = dict(out=3, red=2, M=n, K=n)
    return dict(kind="gemv_chain", nloops=4, passes=[p1, p2],
                accs=[A, tmp, x, A2, tmp2, y])


# 已记录的服务器 predicted（单位 ns），用于自检
KNOWN = {
    "gemm512":      (4.169331e4, dict(BM=64, BN=64, TK=32, S=4)),
    "gemm500":      (4.169331e4, dict(BM=64, BN=64, TK=32, S=4)),
    "gemm4096":     (3.566476e6, dict(BM=128, BN=256, TK=16, S=3)),
    "syrk512":      (2.351412e4, dict(BM=64, BN=64, TK=64, S=4)),
    "twomm512":     (8.338663e4, dict(BM=64, BN=64, TK=32, S=4)),
    "atax8192":     (6.104702e5, dict(T=64, TK=64, S=1)),
    "jacobi2d":     (1.634208e5, dict(T=64, S=1)),
    "jacobi1d_big2": (2.648293e6, dict(T=4096, S=1)),
    "heat1d_1024":  (3339.339, dict(T=16, S=1)),
    "heat1d_2p20":  (1.234105e4, dict(T=2048, S=1)),
    # 11-14 记忆记作 "predicted 150.8 µs"（四舍五入到 0.1 µs），故此处容差放宽
    "heat1d_2p24":  (1.508e5, dict(T=4096, S=1)),
}


def selftest():
    print("=" * 78)
    print("l1model 自检：与服务器已记录 predicted 对表（相对误差阈值 1e-4）")
    print("=" * 78)
    rows = []
    p = _gemm(512, 512, 512)
    b, nf, ne = decide_gemm(p)
    rows.append(("gemm512", b, nf, ne))
    b, nf, ne = decide_gemm(_gemm(500, 500, 500))
    rows.append(("gemm500", b, nf, ne))
    b, nf, ne = decide_gemm(_gemm(4096, 4096, 4096))
    rows.append(("gemm4096", b, nf, ne))
    b, nf, ne = decide_gemm(_syrk(512))
    rows.append(("syrk512", b, nf, ne))
    b, nf, ne = decide_gemm_chain(_twomm(512))
    rows.append(("twomm512", b, nf, ne))
    b, nf, ne = decide_gemv(_atax(8192))
    rows.append(("atax8192", b, nf, ne))
    j2 = decide_stencil2d(_jacobi2d(1022, 1022, 16))
    rows.append(("jacobi2d", j2, 12, 16))
    sib = decide_sibling(_sibling(16777214, 9))
    rows.append(("jacobi1d_big2", sib, None, None))
    h1 = decide_stencil1d(_heat1d(1024))
    rows.append(("heat1d_1024", h1, None, None))
    h2 = decide_stencil1d(_heat1d(1 << 20))
    rows.append(("heat1d_2p20", h2, None, None))
    h3 = decide_stencil1d(_heat1d(1 << 24))
    rows.append(("heat1d_2p24", h3, None, None))

    ok = 0
    tol = {"heat1d_2p24": 1e-3}   # 记录值本身是四舍五入的 150.8 µs
    for name, b, nf, ne in rows:
        exp, _ = KNOWN[name]
        got = b["total"] if b else NAN
        rel = abs(got - exp) / exp if b else float("inf")
        mark = "PASS" if rel <= tol.get(name, 1e-4) else "FAIL"
        ok += rel <= tol.get(name, 1e-4)
        cfg = ""
        for k in ("BM", "BN", "TK", "T", "S", "SA", "SB"):
            if b and k in b:
                cfg += f"{k}={b[k]} "
        print(f"[{mark}] {name:14s} predicted={got:14.6e}  记录={exp:14.6e} "
              f"rel={rel:8.2e}  {cfg}")
    print(f"\n自检：{ok}/{len(rows)} 通过")
    return ok == len(rows)


def show_regime():
    print("=" * 78)
    print("B 线机理前提：三类墙的对照（单位 ns，per chunk / 整块）")
    print("=" * 78)
    print(f"{'kernel':10s} {'cfg':22s} {'copy':>9s} {'aggBW':>9s} {'comp':>9s} "
          f"{'Lmin':>7s} {'expo':>7s} {'支配项'}")
    cases = [
        ("gemm4096", _gemm(4096, 4096, 4096), (128, 256, 16, 3)),
        ("gemm4096", _gemm(4096, 4096, 4096), (64, 64, 16, 2)),
        ("gemm512", _gemm(512, 512, 512), (64, 64, 32, 4)),
        ("gemm4096", _gemm(4096, 4096, 4096), (256, 128, 16, 2)),
    ]
    for name, p, (BM, BN, TK, S) in cases:
        r = regime(p, BM, BN, TK, S)
        dom = max([("copy", r["copy_ns"]), ("comp", r["comp_ns"]),
                   ("Lmin", r["Lmin_ns"]), ("expo", r["expo_ns"])], key=lambda x: x[1])[0]
        print(f"{name:10s} {f'{BM}x{BN}/{TK}/S{S}':22s} {r['copy_ns']:9.1f} "
              f"{r['agg_ns']:9.1f} {r['comp_ns']:9.1f} {r['Lmin_ns']:7.1f} "
              f"{r['expo_ns']:7.1f} {dom}")
    print("\n判读要点：copy 项 = 该块每 chunk 字节 / 单 SM 带宽(7.21)；")
    print("aggBW = 同一字节数 × 128 块 / 整卡带宽 —— 两者恒等 ⇒ copy 项在满占用时")
    print("就是整卡带宽约束；深度不改变总流量，故满占用+搬运主导区里 depth 无稳态收益。")
    print("⇒ depth 只在 (a) 计算主导 (comp>copy) 或 (b) 占用不足 (blocks<128) 时有意义。")


def show_asym():
    print("=" * 78)
    print("B 线原型：非对称缓冲（每 operand 独立深度）")
    print("=" * 78)
    for name, p in (("gemm512", _gemm(512, 512, 512)),
                    ("gemm4096", _gemm(4096, 4096, 4096))):
        sym, nfs, nes = decide_asym(p, symmetric_only=True)
        print(f"\n--- {name} ---")
        print(f"  对称  : {sym['BM']}x{sym['BN']}/TK={sym['TK']} S={sym['SA']} "
              f"@ {sym['total']:.6e} ns  smem={sym['smem']/1024:.1f}KB  ({nfs}/{nes})")
        for variant in ("min", "little"):
            b, nf, ne = decide_asym(p, variant=variant)
            print(f"  {variant:6s}: {b['BM']}x{b['BN']}/TK={b['TK']} "
                  f"S=({b['SA']},{b['SB']}) @ {b['total']:.6e} ns  "
                  f"smem={b['smem']/1024:.1f}KB  ({nf}/{ne})")


def show_frontier():
    print("=" * 78)
    print("B 线：非对称缓冲释放的容量前沿（symmetric 不可行 / asymmetric 可行）")
    print("=" * 78)
    p = _gemm(4096, 4096, 4096)
    found = 0
    for BM in (64, 128, 256):
        for BN in (64, 128, 256):
            if BM == 256 and BN == 256:
                continue
            for TK in (16, 32, 64):
                sym_ok = [s for s in (1, 2, 3, 4)
                          if asym_eval(p, BM, BN, TK, s, s, variant="min") is not None]
                asym_best = None
                for SA in (1, 2, 3, 4):
                    for SB in (1, 2, 3, 4):
                        r = asym_eval(p, BM, BN, TK, SA, SB, variant="min")
                        if r and (asym_best is None or r["total"] < asym_best[0]["total"]):
                            asym_best = (r, SA, SB)
                if asym_best and not sym_ok:
                    r, SA, SB = asym_best
                    print(f"  {BM}x{BN}/TK={TK}: 对称全不可行 → 非对称 S=({SA},{SB}) "
                          f"可行 @ {r['total']:.4e} ns  smem={r['smem']/1024:.1f}KB")
                    found += 1
    if not found:
        print("  （无：99KB 上限下 4096 的对称前沿没有任何非对称独占的配置）")


def show_compare_hw():
    """同一份代码、两块卡：regime 是否翻转、决策是否随之改变。

    注意：a100/r6000 的常数是规格换算的未标定值，这里只演示"机理判读"，
    真实结论必须重标 8 个常数后再跑（见 docs/B线-机理判读）。
    """
    print("=" * 84)
    print("跨架构 regime 判读（a100/r6000 常数为规格换算，未标定，仅看比例关系）")
    print("=" * 84)
    for preset in ("4090", "a100", "r6000"):
        HW.update({"smBw": 7.21, "smCompute": 630.0, "numSms": 128,
                   "smemCap": 99 * 1024})
        HW.update(PRESETS[preset])
        print(f"\n--- {preset}: numSms={HW['numSms']} smBw={HW['smBw']:.2f} GB/s/SM "
              f"smCompute={HW['smCompute']:.1f} GFLOPS/SM ---")
        for name, p in (("gemm512", _gemm(512, 512, 512)),
                        ("gemm4096", _gemm(4096, 4096, 4096))):
            b, nf, ne = decide_gemm(p)
            if b is None:
                print(f"  {name:9s} 无可行配置（半机守卫：tile≥64 时块数 < numSms/2）"
                      f" 可行 {nf}/{ne}")
                continue
            r = regime(p, b["BM"], b["BN"], b["TK"], b["S"])
            ratio = r["comp_ns"] / r["copy_ns"]
            kind = "计算主导" if ratio > 1.0 else "搬运主导"
            print(f"  {name:9s} 决策 {b['BM']}x{b['BN']}/TK={b['TK']}/S={b['S']} "
                  f"@ {b['total']:.3e} ns  可行 {nf}/{ne}  "
                  f"comp/copy={ratio:.2f} → {kind}")
    HW.update({"smBw": 7.21, "smCompute": 630.0, "numSms": 128,
               "smemCap": 99 * 1024})


def show_calib_chunk():
    """每 chunk 固定开销 cChunk 的标定：看 512³ 的 TK 选择是否翻到 64。

    实测锚点（sweep_512_S4.log）：最优 64×64/TK=64/S=2 @ 27.2 µs；
    模型当前选 64×64/TK=32/S=4 @ 29.7 µs（+9.2%）。TK32→TK64 的实测差 = 1.3 µs，
    chunks 从 16 降到 8 → 反解 cChunk ≈ 160 ns。本函数验证该值能否让决策翻到 TK=64。
    """
    print("=" * 84)
    print("cChunk 标定：每 chunk 固定开销（ns），看 512³/4096³ 决策如何移动")
    print("=" * 84)
    base = HW.get("cChunk", 0.0)
    for c in (0.0, 80.0, 120.0, 160.0, 200.0, 240.0):
        HW["cChunk"] = c
        b5, _, _ = decide_gemm(_gemm(512, 512, 512))
        b4, _, _ = decide_gemm(_gemm(4096, 4096, 4096))
        bs, _, _ = decide_gemm(_syrk(512))
        print(f"  cChunk={c:5.0f}  512³ → {b5['BM']}x{b5['BN']}/TK={b5['TK']}/S={b5['S']}"
              f" @ {b5['total']:.4e}   |   4096³ → {b4['BM']}x{b4['BN']}/TK={b4['TK']}/S={b4['S']}"
              f"   |   syrk → TK={bs['TK']}/S={bs['S']}")
    HW["cChunk"] = base
    print("\n判读：512³ 的实测最优是 TK=64（27.2 µs），模型需 cChunk≈160 ns 才翻过去；")
    print("4096³ 的决策在 cChunk ∈ [0,240] 内应保持 128×256/TK=16/S=3（copy 3409 ≫ 160）。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", default="4090", choices=list(PRESETS))
    ap.add_argument("--chunk-cost", type=float, default=0.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--regime", action="store_true")
    ap.add_argument("--asym", action="store_true")
    ap.add_argument("--frontier", action="store_true")
    ap.add_argument("--compare-hw", action="store_true")
    ap.add_argument("--calib-chunk", action="store_true")
    args = ap.parse_args()
    HW.update(PRESETS[args.hw])
    HW["cChunk"] = args.chunk_cost
    if args.selftest:
        selftest()
    elif args.regime:
        show_regime()
    elif args.asym:
        show_asym()
    elif args.frontier:
        show_frontier()
    elif args.compare_hw:
        show_compare_hw()
    elif args.calib_chunk:
        show_calib_chunk()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
