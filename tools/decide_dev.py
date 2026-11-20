#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""decide_dev.py — 唯一决策函数（A4 最小版：冻结 v1.8.5 模型，可复现 13 行）

定位
  这不是新模型。它把 `l1model.py`（已验证到 1e-7）里的五支公式收成**一个入口**，
  加上两样 l1model 没有的东西：
    ① 面板感知的 GEMM 评估（syrk 读 A 两条行带 → 两块面板，v1.8.3 口径）
    ② 占用率模式开关：--occ=hard（块数 ≥ 64，现模型）| soft（占用不足按比例扣时）

统一输出（所有族同一 schema）：
  {family, config, predicted_ns, blocks, smem, smem_kb, s_max, chunks, waves, occ_mode}

约束边界（本文件刻意不做的事）
  - 不改 l1model.py（共享文件，另一个 agent 可能在用）
  - 不改 L1Feas.cpp（冻结 v1.8.5）
  - 不引入任何拟合项（cChunk 等一律不在此处）
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import l1model as L  # noqa: E402

HW = L.HW

# 每 chunk 固定开销（cp.async commit/wait + __syncthreads 的同步路径）。
# 它不是"标定出来的常数"：
#   · 物理下界 ≳100 ns（barrier + commit/wait 必然存在）
#   · 决策在 25–500 ns 内完全不敏感（见 tools/cchunk_sweep.py）
# 取 150 ns 仅为名义值；改它不改变任何决策。
CCHUNK = 150.0

# ---- v1.9 修复开关（A1/A2 仍走 v1.8.5 默认）--------------------------------
# F1：1D stencil 的 numTiles 守卫 512 → numSms（128）
STEN1D_MIN_TILES_V185 = 512
STEN1D_MIN_TILES_V19 = HW["numSms"]          # 128
# F2：GEMV 的 T<16 薄流水惩罚，仅在 DRAM regime（A 工作集 > L2）生效
#     （v1.8.5 无条件罚；详见 gemv_guard.md 的 lane×DRAM 双条件）

# ---- L2 驻留 regime（#6）----------------------------------------------------
# 模型原来只有一个带宽常数 smBw=7.21（= 922.8/128，DRAM 均分）。
# 但工作集装得进 L2 时，读 A/B 的流量由 L2 服务，远比 DRAM 便宜：
#   512³ 工作集 3 MB、1024³ 12 MB、2048³ 48 MB（都 < 72 MB）；
#   4096³ 192 MB（> 72 MB，真走 DRAM）。
# 这正是"512 模型算慢 63%、4096 算快 17%"的同一原因，也是 1024 上 S=1 被
# 过度惩罚的根源（S=1 是串行 copy+comp，copy 被高估 3 倍就必然输给 S≥2）。
#
# L2_BYTES 是硬件属性（待下一步用 cudaDeviceProp.l2CacheSize 记录）；
# L2_BOOST 是"L2 服务流量的加速倍数"，只用于**输入流式读**，不用于 C 写回
# （写回仍走 DRAM 口径，保守）。其取值不影响决策（见 l2_regime_sweep.py）。
L2_BYTES = 72 * 1024 * 1024
L2_BOOST = 2.0


def _working_set(M, N, K):
    """A+B+C 的总字节数（判 L2 是否装得下）。"""
    return (M * K + K * N + M * N) * 4


def _bw_for(M, N, K):
    """输入流式读的有效每 SM 带宽（GB/s）。"""
    return _bw_for_ws(_working_set(M, N, K))


def _bw_for_ws(W):
    """按工作集字节数选带宽 regime。"""
    return HW["smBw"] * (L2_BOOST if W <= L2_BYTES else 1.0)

# ---------------------------------------------------------------------------
# 两层结构（论文口径必须分开写，这是老师第 2 条批评的解法）
#
#   ① 硬可行域（可证明）：共享内存 ≤ smemCap、寄存器 ≤245（0 spill）、结构可接
#      —— 96 个枚举 → 48 个（GEMM）
#   ② 搜索空间收缩（经验，不是可行域）：块数 ≥ numSms/2
#      —— 48 → 10（512/500/链/syrk/symm）
#      依据：8/8 决策不变（tools/guard_variants.py）；被收缩掉的配置里最快的也
#            比最优慢 +69.4%（tools/guard_excluded.py）
#      回退：若收缩后为空（小问题只有 1 个块），退回①，保证永远有决策
#
# 代码里 occ="hard" = ①+②（带回退）；"none" = 只①；"full" = ①+②但收缩线取 SM 数
# ---------------------------------------------------------------------------


# ------------------------------------------------------------------ 面板工具
def _eval_panels(prof, BM, BN, TK, S, occ="hard", cchunk=0.0, min_blocks=None):
    """按 (memref, 载体环集合) 分组后的面板做 GEMM 两态式评估。

    cchunk = 每 chunk 与 TK 无关的固定开销（cp.async commit/wait + barrier）。
    注意：它只加在"每 chunk 代价"里，用来打破 TK 的零分辨率。

    occ: "hard"=块数≥SM/2（现模型）| "none"=不设占用门槛 | "full"=按 min_blocks 门槛
    """
    tiles = [0] * prof["nloops"]
    tiles[prof["red"]] = TK
    tiles[prof["out"][0]] = BM
    tiles[prof["out"][1]] = BN
    reds = {prof["red"]}

    smem, copyB, outB = 0, 0.0, 0.0
    for pn in prof["panels"]:
        fp = L.window(pn, tiles)
        if fp < 0:
            return None
        if L.is_stream(pn, reds):
            smem += fp * (S + 1)
            copyB += fp
        else:
            outB += fp
    if smem > HW["smemCap"]:
        return None

    M, N, K = prof["M"], prof["N"], prof["K"]
    nb = math.ceil(M / BM) * math.ceil(N / BN)
    if occ == "hard" and nb < HW["numSms"] // 2:
        return None
    if occ == "full" and nb < (min_blocks or HW["numSms"]):
        return None
    waves = math.ceil(nb / HW["numSms"])
    chunks = math.ceil(K / TK)
    copy = copyB / _bw_for(M, N, K) + cchunk
    comp = 2.0 * BM * BN * TK / HW["smCompute"]
    eff = min(float(S), max(1.0, float(chunks - 1)))
    expo = max(HW["rExposed"] / (eff ** HW["pExp"]), HW["deltaFloor"])
    cWrite = outB / HW["smBw"]
    if S == 1:
        block = chunks * (copy + comp) + expo + cWrite
    else:
        Lcap = min(HW["LminGemm"], copy + comp)
        block = chunks * max(copy, comp, Lcap) + expo + cWrite
    total = HW["launch"] + waves * block
    if occ == "soft":
        total *= max(1.0, HW["numSms"] / max(nb, 1))
    return dict(total=total, smem=smem, blocks=nb, waves=waves, chunks=chunks,
                copy=copy, comp=comp)


def decide_gemm_panels(prof, occ="hard", cchunk=0.0):
    """单 GEMM（面板感知）。返回 (best, n_enumerated, n_feasible)。

    occ="hard"（默认）= 两层：
        ① 硬可行域：容量/寄存器/结构（_eval_panels 内的 smemCap 检查）
        ② 搜索空间收缩：块数 ≥ SM/2；**若收缩后为空，退回只做硬约束**（小问题不会无解）
    occ="full" = ①②但收缩线用 SM 数（占满机），同样带回退
    occ="none" = 只做硬约束，不做 occupy 收缩
    """
    phases = [("none", None)]
    if occ == "hard":
        phases = [("hard", None), ("none", None)]
    elif occ == "full":
        phases = [("full", HW["numSms"]), ("full", HW["numSms"] // 2), ("none", None)]
    for occ_mode, min_blocks in phases:
        mode = occ_mode
        best, n_enum, n_feas = None, 0, 0
        for BM in (64, 128, 256):
            for BN in (64, 128, 256):
                if BM == 256 and BN == 256:
                    continue
                for TK in (16, 32, 64):
                    for S in (1, 2, 3, 4):
                        n_enum += 1
                        r = _eval_panels(prof, BM, BN, TK, S, mode, cchunk, min_blocks)
                        if r is None:
                            continue
                        n_feas += 1
                        if best is None or r["total"] < best["total"]:
                            best = dict(r, family="gemm", BM=BM, BN=BN, TK=TK, S=S)
        if best is not None:
            return best, n_enum, n_feas
    return None, n_enum, n_feas


def _chain_panels(p, tiles):
    """把一个 pass 的 accs 按 (流式/输出) 汇总成 (smem, copyB, outB)。"""
    smem, copyB, outB = 0, 0.0, 0.0
    for acc in p["accs"]:
        v = L.window(acc, tiles)
        if v < 0:
            return None
        if L.is_stream(acc, [p["red"]]):
            smem += v
            copyB += v
        else:
            outB += v
    return smem, copyB, outB


def _decide_chain(prof, occ="hard", cchunk=0.0):
    """组化 pass-chain：镜像 l1model.decide_gemm_chain，但支持 occ 与 cchunk。

    occ 语义见 decide_gemm_panels：hard = 硬可行域 + 占用收缩（空则退回），
    none = 只做硬约束，full = 收缩线取 SM 数（带回退）。
    """
    P = len(prof["passes"])
    phases = [("none", None)]
    if occ == "hard":
        phases = [("hard", None), ("none", None)]
    elif occ == "full":
        phases = [("full", HW["numSms"]), ("full", HW["numSms"] // 2), ("none", None)]
    for occ_mode, min_blocks in phases:
        mode = occ_mode
        best, n_enum, n_feas = None, 0, 0
        for BM in (64, 128, 256):
            for BN in (64, 128, 256):
                if BM == 256 and BN == 256:
                    continue
                for TK in (16, 32, 64):
                    for S in (1, 2, 3, 4):
                        n_enum += 1
                        total = HW["launch"] * P
                        ok = True
                        for p in prof["passes"]:
                            tiles = [0] * prof["nloops"]
                            tiles[p["red"]] = TK
                            tiles[p["out"][0]] = BM
                            tiles[p["out"][1]] = BN
                            agg = _chain_panels(p, tiles)
                            if agg is None:
                                ok = False
                                break
                            smem, copyB, outB = agg
                            if smem * (S + 1) > HW["smemCap"]:
                                ok = False
                                break
                            M, N, K = p["M"], p["N"], p["K"]
                            nb = math.ceil(M / BM) * math.ceil(N / BN)
                            if mode == "hard" and nb < HW["numSms"] // 2:
                                ok = False
                                break
                            if mode == "full" and nb < (min_blocks or HW["numSms"]):
                                ok = False
                                break
                            waves = math.ceil(nb / HW["numSms"])
                            chunks = math.ceil(K / TK)
                            copy = copyB / _bw_for(M, N, K) + cchunk
                            comp = 2.0 * BM * BN * TK / HW["smCompute"]
                            eff = min(float(S), max(1.0, float(chunks - 1)))
                            expo = max(HW["rExposed"] / (eff ** HW["pExp"]), HW["deltaFloor"])
                            cWrite = outB / HW["smBw"]
                            if S == 1:
                                block = chunks * (copy + comp) + expo + cWrite
                            else:
                                block = chunks * max(copy, comp,
                                                     min(HW["LminGemm"], copy + comp)) + expo + cWrite
                            total += waves * block
                        if not ok:
                            continue
                        n_feas += 1
                        if best is None or total < best["total"]:
                            best = dict(total=total, family="gemm_chain",
                                        BM=BM, BN=BN, TK=TK, S=S)
        if best is not None:
            return best, n_feas, n_enum
    return None, n_feas, n_enum


def _decide_stencil1d(prof, occ="hard", cchunk=0.0, min_tiles=STEN1D_MIN_TILES_V185):
    """stencil-1D（legacy 形态）：镜像 l1model.decide_stencil1d，但
       ① 输入流的带宽按工作集走 L2/DRAM regime（#6）；
       ② 保留原有 T 过滤器与 expo 形式（R = max(330-0.6T,1)）。
    min_tiles：v1.8.5=512；v1.9=numSms（128）。cchunk 不进入本族。
    """
    N = prof["N"]
    W = 2 * N * 4                      # A 读 + B 写
    bw = _bw_for_ws(W)
    best = None
    for S in (1, 2, 3, 4):
        for T in [16 << i for i in range(40)]:
            if T > N:
                break
            bytes_ = 0
            for base_elems, elem in prof["legacy_base"]:
                bytes_ += base_elems * elem + (T - 1) * elem
            smem = bytes_ * (S + 1)
            if smem > HW["smemCap"]:
                continue
            numTiles = math.ceil(N / T)
            if N >= 8192 and numTiles < min_tiles:
                continue
            waves = math.ceil(numTiles / HW["numSms"])
            eff = min(float(S), max(1.0, float(waves - 1)))
            copy = bytes_ / bw
            comp = prof["flops_per_iter"] * T / HW["smCompute"]
            R = max(HW["rExposed"] - HW["aT"] * T, 1.0)
            expo = max(R / (eff ** HW["pExp"]), HW["deltaFloor"])
            total = HW["launch"] + waves * (copy + comp + expo)
            if best is None or total < best["total"]:
                best = dict(total=total, T=T, S=S, smem=smem, waves=waves, blocks=numTiles)
    return best

def _decide_gemv(prof, occ="hard", cchunk=0.0, wave_pen=False, thin_dram_only=False):
    """GEMV pass-chain：镜像 l1model.decide_gemv，但把占用"守卫"参数化。

    occ="hard" = 块数 ≥ SM/2（现模型，空集退回）；"full" = ≥ SM（退 SM/2 再退无门槛）；
    "none" = 不设占用门槛（带宽项里的 occ 仍生效）。
    cchunk 不进入本族（GEMV 已有每 chunk 下界 L_min）。

    wave_pen=True 是"部分波次效率"惩罚，**已否证**：它能让无守卫版本恢复 T=8，
    但把预测值放大 2 倍，与实测矛盾（mvt 的 2.1 MB/19.5 µs 仅 107 GB/s，是延迟受限
    而非吞吐受限）。保留参数只为记录该实验，默认关闭。见 docs/evidence/gemv_guard.md
    """
    P = len(prof["passes"])
    phases = [("none",)]
    if occ == "hard":
        phases = [("hard",), ("none",)]
    elif occ == "full":
        phases = [("full128",), ("full64",), ("none",)]
    for (mode,) in phases:
        best, n_enum, n_feas = None, 0, 0
        for T in (256, 128, 64, 32, 16, 8):
            for TK in (64, 32, 16):
                for S in (1, 2, 3, 4):
                    n_enum += 1
                    tiles = [0] * prof["nloops"]
                    for p in prof["passes"]:
                        tiles[p["out"]] = T
                        tiles[p["red"]] = TK
                    smem, ok = 0, True
                    reds = [p["red"] for p in prof["passes"]]
                    for acc in prof["accs"]:
                        v = L.window(acc, tiles)
                        if v < 0:
                            ok = False
                            break
                        smem += v * ((S + 1) if L.is_stream(acc, reds) else 1)
                    if not ok or smem > HW["smemCap"]:
                        continue
                    total = HW["launch"] * P
                    guard = True
                    for p in prof["passes"]:
                        M, K = p["M"], p["K"]
                        nb = math.ceil(M / T)
                        thr = {"hard": HW["numSms"] // 2,
                               "none": 0,
                               "full128": HW["numSms"],
                               "full64": HW["numSms"] // 2}[mode]
                        if nb < thr:
                            guard = False
                            break
                        chunks = math.ceil(K / TK)
                        W = M * K * 4.0
                        thin = (T < 16) and ((not thin_dram_only) or W > L2_BYTES)
                        Lmin = HW.get("LminGemvThin", 1260.0) if thin else HW.get("LminGemv", 820.0)
                        # 占用只计一次：wave_pen 打开时，用"部分波次效率"表达占用，
                        # 带宽项不再重复乘 occ（否则在带宽主导区会双重惩罚）。
                        occ_ = 1.0 if wave_pen else min(1.0, nb / HW["numSms"])
                        bwEff = HW["numSms"] * HW["smBw"] * occ_ * max(0.6, 1.0 - 2.4 / T)
                        streamB = M * K * 4.0
                        pt = max(chunks * Lmin, streamB / bwEff) + T * 4.0 / HW["smBw"]
                        if wave_pen:
                            pt *= max(1.0, HW["numSms"] / max(nb, 1))
                        total += pt
                    if guard:
                        n_feas += 1
                        if best is None or total < best["total"]:
                            best = dict(total=total, T=T, TK=TK, S=S)
        if best is not None:
            return best, n_feas, n_enum
    return None, n_feas, n_enum

def _pack(best, family, occ, n_enum=None, n_feas=None):
    if best is None:
        return dict(family=family, config=None, predicted_ns=None, occ_mode=occ)
    cfg = {k: best[k] for k in ("BM", "BN", "TK", "T", "S", "SA", "SB") if k in best}
    return dict(family=family, config=cfg, predicted_ns=best["total"],
                blocks=best.get("blocks"), smem=best.get("smem"),
                smem_kb=(best.get("smem") or 0) / 1024.0,
                chunks=best.get("chunks"), waves=best.get("waves"),
                n_enum=n_enum, n_feas=n_feas, occ_mode=occ)


# ------------------------------------------------------------------ 统一入口
def decide(profile, occ="hard", cchunk=None, model="v1.8.5"):
    """model: "v1.8.5"（A1/A2 默认）| "v1.9"（F1 守卫 128 + F2 GEMV DRAM-only）"""
    if cchunk is None:
        cchunk = CCHUNK
    if model not in ("v1.8.5", "v1.9"):
        raise ValueError(f"unknown model: {model}")
    v19 = (model == "v1.9")
    """profile.kind ∈ {gemm_panels, gemm, gemm_chain, gemv_chain, stencil2d,
    sibling, stencil1d}"""
    kind = profile["kind"]
    if kind == "gemm_panels":
        b, ne, nf = decide_gemm_panels(profile, occ, cchunk)
        return _pack(b, "gemm", occ, ne, nf)
    if kind == "gemm":
        if cchunk or occ != "hard":
            b, ne, nf = decide_gemm_panels(
                profile_gemm_panels(profile["M"], profile["N"], profile["K"],
                                    _panels_for_gemm(profile["M"], profile["N"], profile["K"])),
                occ, cchunk)
        else:
            b, nf, ne = L.decide_gemm(profile)
        return _pack(b, "gemm", occ, ne, nf)
    if kind == "gemm_chain":
        b, nf, ne = _decide_chain(profile, occ, cchunk)
        return _pack(b, "gemm_chain", occ, ne, nf)
    if kind == "gemv_chain":
        b, nf, ne = _decide_gemv(profile, occ, thin_dram_only=v19)
        return _pack(b, "gemv_chain", occ, ne, nf)
    if kind == "stencil2d":
        b = L.decide_stencil2d(profile)
        return _pack(b, "stencil2d", occ)
    if kind == "sibling":
        b = L.decide_sibling(profile)
        return _pack(b, "sibling", occ)
    if kind == "stencil1d":
        mt = STEN1D_MIN_TILES_V19 if v19 else STEN1D_MIN_TILES_V185
        b = _decide_stencil1d(profile, occ, cchunk, min_tiles=mt)
        return _pack(b, "stencil1d", occ)
    raise ValueError(f"unknown kind: {kind}")


# ------------------------------------------------------------------ 开发集画像
def _panels_for_gemm(M, N, K):
    """plain GEMM 的三块面板（与 l1model._gemm 的合并视图等价）。"""
    A = L.Acc("A", 4, [[(0, 0)], [(2, 0)]], [M, K], role="A")
    B = L.Acc("B", 4, [[(2, 0)], [(1, 0)]], [K, N], role="B")
    C = L.Acc("C", 4, [[(0, 0)], [(1, 0)]], [M, N], read=False, written=True, role="C")
    return [A, B, C]


def _panels_for_syrk(n):
    """syrk：A[i,k] 与 A[j,k] 是两块独立面板（v1.8.3 口径），C 不占 smem。"""
    A1 = L.Acc("A", 4, [[(0, 0)], [(2, 0)]], [n, n], role="A")
    A2 = L.Acc("A", 4, [[(1, 0)], [(2, 0)]], [n, n], role="A")
    C = L.Acc("C", 4, [[(0, 0)], [(1, 0)]], [n, n], read=False, written=True, role="C")
    return [A1, A2, C]


def profile_gemm_panels(M, N, K, panels):
    return dict(kind="gemm_panels", nloops=3, red=2, reds=[2], out=[0, 1],
                M=M, N=N, K=K, panels=panels)


def profile_three_mm(n):
    p = L._twomm(n)
    p["passes"] = list(p["passes"]) * 1  # 2 passes
    p["passes"] = p["passes"] + [p["passes"][0]]  # → 3 passes（三遍同构）
    p["kind"] = "gemm_chain"
    return p


DEV_KERNELS = {
    # name: (profile, expected_config, expected_predicted_ns, expected_source)
    "gemm4096": (L._gemm(4096, 4096, 4096), dict(BM=128, BN=256, TK=16, S=3), 3.566476e6, "l1_gemm4096.txt"),
    "gemm1024": (L._gemm(1024, 1024, 1024), dict(BM=64, BN=128, TK=32, S=3), 1.167083e5, "l1_gemm1024.txt"),
    "gemm512":  (L._gemm(512, 512, 512),   dict(BM=64, BN=64, TK=32, S=4),  4.169331e4, "l1_gemm512.txt"),
    "gemm500":  (L._gemm(500, 500, 500),   dict(BM=64, BN=64, TK=32, S=4),  4.169331e4, "l1_gemm500.txt"),
    "twomm":    (L._twomm(512),             dict(BM=64, BN=64, TK=32, S=4),  8.338663e4, "l1_twomm.txt"),
    "three_mm": (profile_three_mm(512),     dict(BM=64, BN=64, TK=32, S=4),  1.250799e5, "l1_three_mm_probe.txt"),
    "atax":     (L._atax(8192),             dict(T=64, TK=64, S=1),          6.104702e5, "l1_atax_big.txt"),
    "mvt":      (L._atax(512),              dict(T=8, TK=64, S=1),           2.616888e4, "l1_mvt_probe.txt"),
    "bicg":     (L._atax(512),              dict(T=8, TK=64, S=1),           2.616888e4, "l1_bicg_probe.txt"),
    "jacobi2d": (L._jacobi2d(1022, 1022, 16), dict(T=64, S=1),                1.634208e5, "l1_jacobi2d.txt"),
    "jacobi1d": (L._sibling(16777214, 9),   dict(T=4096, S=1),                2.648293e6, "l1_jacobi1d_big2.txt"),
    "heat1d24": (L._heat1d(1 << 24),        dict(T=4096, S=1),                1.507572e5, "l1_heat1d_big24.txt"),
    "heat1d20": (L._heat1d(1 << 20),        dict(T=2048, S=1),                1.234105e4, "l1_heat1d_big.txt"),
    "syrk":     (profile_gemm_panels(512, 512, 512, _panels_for_syrk(512)),
                 dict(BM=64, BN=64, TK=32, S=4), None, "memory/2026-11-15（v1.8.3 后无 l1_*.txt 归档）"),
    "symm":     (L._gemm(512, 512, 512),    dict(BM=64, BN=64, TK=32, S=4),  4.169331e4, "l1_symm_true_probe.txt"),
}




def main():
    """两段：① cchunk=0 复现 v1.8.5（保真性自检）；② 新模型（CCHUNK）给出决策。"""
    print(f"# decide_dev — CCHUNK={CCHUNK:.0f} ns（名义值；25–500 ns 决策不变）")
    print("\n[1] 保真性自检：cchunk=0 且 L2_BOOST=1 应逐位复现 v1.8.5 的历史决策")
    ok = bad = 0
    _save = L2_BOOST
    globals()["L2_BOOST"] = 1.0
    for name, (prof, exp_cfg, exp_ns, src) in DEV_KERNELS.items():
        r = decide(prof, "hard", 0.0)
        c = r.get("config") or {}
        hit = all(c.get(k) == v for k, v in exp_cfg.items())
        ns_ok = (exp_ns is None) or (r["predicted_ns"] is not None
                                     and abs(r["predicted_ns"] - exp_ns) <= 1e-6 * exp_ns)
        ok += bool(hit and ns_ok)
        bad += not (hit and ns_ok)
    globals()["L2_BOOST"] = _save
    print(f"    -> {ok} PASS / {bad} FAIL")

    print("\n[2] 新模型决策（加每 chunk 固定开销后）")
    print(f"{'kernel':10s} {'v1.8.5':>18s} {'新决策':>18s}  changed?")
    for name, (prof, *_r) in DEV_KERNELS.items():
        old = decide(prof, "hard", 0.0)["config"] or {}
        new = decide(prof, "hard")["config"] or {}
        def fmt(c):
            return "/".join(str(c[k]) for k in ("BM", "BN", "TK", "T") if k in c) + \
                   (f"/{c['S']}" if "S" in c else "")
        fo, fn = fmt(old), fmt(new)
        print(f"{name:10s} {fo:>18s} {fn:>18s}  {'<- changed' if fo != fn else ''}")


if __name__ == "__main__":
    main()
