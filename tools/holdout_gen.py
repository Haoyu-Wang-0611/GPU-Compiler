#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""holdout_gen.py — 留出套件生成器（spec v2，2026-11-20）

做三件事，全部离线（不跑 MLIR、不跑 GPU）：
  1. 按 spec v2 的 12 主 + 4 副，生成 `daima/holdout/<id>.mlir` 与 `<id>.json`；
  2. 汇总 `daima/holdout/suite.json`（冻结入口，含每个文件的 sha256）；
  3. `--check` 校验：主/副计数、id/名字唯一、(类,尺寸) 唯一、文件与哈希一致。

纪律：
  - MLIR 语法逐字镜像 `daima/` 下已验证的模板（gemm512/twomm/three_mm_probe/
    mvt_probe/bicg_probe/jacobi1d_big2/jacobi2d/heat1d_big24）。
  - 生成一次即冻结；批内不再重生成（改了就必须全量重跑）。
  - 不 import l1model / decide_dev：本文件只产出 IR 与元数据。

用法：
  python tools/holdout_gen.py            # 生成/覆盖
  python tools/holdout_gen.py --check    # 只校验
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, "daima", "holdout")
GEN_VERSION = "holdout_gen v1.0 (2026-11-20)"
SEED = 20261115

# ---------------------------------------------------------------------------
# 套件表（spec v2 §3）
# ---------------------------------------------------------------------------
KERNELS = [
    # ---------------- 主集：流式 6 ----------------
    dict(id="S1", name="sten1d_r2_n350k", func="sten1d_r2_n350k",
         class_name="streaming", form="sten1d", family="stencil1",
         params=dict(N=350000, r=2), primary=True,
         vehicle="stencil1d_general (new)", notes="1D stencil r=2，非 2 幂；halo 由窗口法吸收"),
    dict(id="S2", name="sten1d_r3_n600k", func="sten1d_r3_n600k",
         class_name="streaming", form="sten1d", family="stencil1",
         params=dict(N=600000, r=3), primary=True,
         vehicle="stencil1d_general (new)", notes="1D stencil r=3；7 点读"),
    dict(id="S3", name="sten1d_r1_n999989", func="sten1d_r1_n999989",
         class_name="streaming", form="sten1d", family="stencil1",
         params=dict(N=999989, r=1), primary=True,
         vehicle="stencil1d_general (new)", notes="素数 N=999989，非整除"),
    dict(id="S4", name="sten2d_r1_n768", func="sten2d_r1_n768",
         class_name="streaming", form="sten2d", family="stencil2",
         params=dict(N=768, r=1, T=8), primary=True,
         vehicle="jacobi2d_tiled (param N)", notes="2D 5 点；8 时间步"),
    dict(id="S5", name="sten2d_r2_n400", func="sten2d_r2_n400",
         class_name="streaming", form="sten2d", family="stencil2",
         params=dict(N=400, r=2, T=8), primary=True,
         vehicle="stencil2d_general (new, conditional)", notes="2D 9 点；载具未就绪时用 S7 替换"),
    dict(id="S6", name="sibling_n12m", func="sibling_n12m",
         class_name="streaming", form="sibling", family="stencil1",
         params=dict(N=12000000, T=8), primary=True,
         vehicle="stencil1d_general (new)", notes="jacobi-1d 形：N=12M，双数组 96MB > L2 → DRAM 流区"),
    # ---------------- 主集：重用 6 ----------------
    dict(id="R1", name="gemm_1152", func="gemm_1152",
         class_name="reuse", form="gemm", family="gemm",
         params=dict(M=1152, N=1152, K=1152), primary=True,
         vehicle="sgemm_tiled (SIZE)", notes="新尺寸；64/128 两档 tile 都可满半机（18²/9²）"),
    dict(id="R2", name="gemm_999", func="gemm_999",
         class_name="reuse", form="gemm", family="gemm",
         params=dict(M=999, N=999, K=999), primary=True,
         vehicle="sgemm_tiled (SIZE, ceil)", notes="真非整除：999 不是 64/128 的倍数；128×128 恰好 64 块"),
    dict(id="R3", name="chain2_1152", func="chain2_1152",
         class_name="reuse", form="chain2", family="gemm_chain",
         params=dict(SZ=1152, P=2), primary=True,
         vehicle="twomm_tiled (SIZE)", notes="组化链 P=2；与 R1/R4 同尺寸，链长对照"),
    dict(id="R4", name="chain3_1152", func="chain3_1152",
         class_name="reuse", form="chain3", family="gemm_chain",
         params=dict(SZ=1152, P=3), primary=True,
         vehicle="3mm_tiled (SIZE)", notes="组化链 P=3；与 R1/R3 同尺寸，链长对照"),
    dict(id="R5", name="gemv_mvt_1536", func="gemv_mvt_1536",
         class_name="reuse", form="mvt", family="gemv_chain",
         params=dict(N=1536), primary=True,
         vehicle="mvt_tiled (N)", notes="mvt 形 2 pass；T=8/16 两档过半机"),
    dict(id="R6", name="gemv_bicg_1536", func="gemv_bicg_1536",
         class_name="reuse", form="bicg", family="gemv_chain",
         params=dict(N=1536), primary=True,
         vehicle="bicg_tiled (N)", notes="bicg 形 2 pass（pass1 红外/写内）；T=8/16 两档"),
    # ---------------- 副集 4 ----------------
    dict(id="S7", name="sten1d_r2_n770k", func="sten1d_r2_n770k",
         class_name="streaming", form="sten1d", family="stencil1",
         params=dict(N=770000, r=2), primary=False,
         vehicle="stencil1d_general (new)", notes="S5 的替补；同形不同尺寸"),
    dict(id="R7", name="gemm_768", func="gemm_768",
         class_name="reuse", form="gemm", family="gemm",
         params=dict(M=768, N=768, K=768), primary=False,
         vehicle="sgemm_tiled (SIZE)", notes="小尺寸（仅 64×64 过半机）；与 R4 同尺寸不同类"),
    dict(id="R8", name="gemv_mvt_2048", func="gemv_mvt_2048",
         class_name="reuse", form="mvt", family="gemv_chain",
         params=dict(N=2048), primary=False,
         vehicle="mvt_tiled (N)", notes="同族不同尺寸；T=8/16/32 三档"),
    dict(id="R9", name="gemm_rect_1024x1536x1280", func="gemm_rect_1024x1536x1280",
         class_name="reuse", form="gemm_rect", family="gemm",
         params=dict(M=1024, N=1536, K=1280), primary=False,
         vehicle="sgemm_rect (new)", notes="矩形；载具未就绪则只做决策层"),
]


# ---------------------------------------------------------------------------
# MLIR 发射器
# ---------------------------------------------------------------------------
def memref(*dims):
    return "memref<" + "x".join(str(d) for d in dims) + "xf32>"


def _header(k):
    return (f"// {k['id']} {k['name']} — holdout spec v2；{k['notes']}\n"
            f"// 生成器：{GEN_VERSION}；seed={SEED}；生成一次即冻结，批内不得重生成\n")


def emit_sten1d(k):
    N, r = k["params"]["N"], k["params"]["r"]
    lo, hi = r, N - r
    m = memref(N)
    L = [_header(k), "module {",
         f"  func.func @{k['func']}(%A: {m}, %B: {m}) {{",
         f"    %c{r} = arith.constant {r} : index",
         f"    %c{hi} = arith.constant {hi} : index",
         "    %cf = arith.constant 0.25 : f32",
         f"    affine.for %i = %c{r} to %c{hi} {{"]
    for d in range(-r, r + 1):
        idx = "%i" if d == 0 else (f"%i - {-d}" if d < 0 else f"%i + {d}")
        L.append(f"      %a{d + r} = affine.load %A[{idx}] : {m}")
    prev = "%a0"
    for j in range(1, 2 * r + 1):
        L.append(f"      %s{j} = arith.addf {prev}, %a{j} : f32")
        prev = f"%s{j}"
    L.append(f"      %res = arith.mulf {prev}, %cf : f32")
    L.append(f"      affine.store %res, %B[%i] : {m}")
    L += ["    }", "    return", "  }", "}"]
    return "\n".join(L) + "\n"


def emit_sten2d(k):
    N, r, T = k["params"]["N"], k["params"]["r"], k["params"]["T"]
    lo, hi = r, N - r
    m = memref(N, N)
    L = [_header(k), "module {",
         f"  func.func @{k['func']}(%A: {m}, %B: {m}) {{",
         "    %c0 = arith.constant 0 : index",
         f"    %cT = arith.constant {T} : index",
         f"    %c{lo} = arith.constant {lo} : index",
         f"    %c{hi} = arith.constant {hi} : index",
         "    %cst = arith.constant 0.2 : f32",
         "    affine.for %t = %c0 to %cT {",
         f"      affine.for %i = %c{lo} to %c{hi} {{",
         f"        affine.for %j = %c{lo} to %c{hi} {{"]
    names = []
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            n = f"%a{di + r}{dj + r}"
            names.append(n)
            ii = "%i" if di == 0 else (f"%i - {-di}" if di < 0 else f"%i + {di}")
            jj = "%j" if dj == 0 else (f"%j - {-dj}" if dj < 0 else f"%j + {dj}")
            L.append(f"          {n} = affine.load %A[{ii}, {jj}] : {m}")
    prev = names[0]
    for j, n in enumerate(names[1:], 1):
        L.append(f"          %s{j} = arith.addf {prev}, {n} : f32")
        prev = f"%s{j}"
    L.append(f"          %res = arith.mulf {prev}, %cst : f32")
    L.append(f"          affine.store %res, %B[%i, %j] : {m}")
    L += ["        }", "      }", "    }", "    return", "  }", "}"]
    return "\n".join(L) + "\n"


def emit_sibling(k):
    N, T = k["params"]["N"], k["params"]["T"]
    hi = N - 1
    m = memref(N)
    L = [_header(k), "module {",
         f"  func.func @{k['func']}(%A: {m}, %B: {m}) {{",
         "    %c1 = arith.constant 1 : index",
         f"    %cT = arith.constant {T} : index",
         f"    %c{hi} = arith.constant {hi} : index",
         "    %cst = arith.constant 0.333333 : f32",
         "    affine.for %t = %c1 to %cT {",
         f"      affine.for %i = %c1 to %c{hi} {{",
         f"        %a0 = affine.load %A[%i - 1] : {m}",
         f"        %a1 = affine.load %A[%i] : {m}",
         f"        %a2 = affine.load %A[%i + 1] : {m}",
         "        %s1 = arith.addf %a0, %a1 : f32",
         "        %s2 = arith.addf %s1, %a2 : f32",
         "        %r = arith.mulf %s2, %cst : f32",
         f"        affine.store %r, %B[%i] : {m}",
         "      }",
         f"      affine.for %i = %c1 to %c{hi} {{",
         f"        %b0 = affine.load %B[%i - 1] : {m}",
         f"        %b1 = affine.load %B[%i] : {m}",
         f"        %b2 = affine.load %B[%i + 1] : {m}",
         "        %s1 = arith.addf %b0, %b1 : f32",
         "        %s2 = arith.addf %s1, %b2 : f32",
         "        %r = arith.mulf %s2, %cst : f32",
         f"        affine.store %r, %A[%i] : {m}",
         "      }",
         "    }", "    return", "  }", "}"]
    return "\n".join(L) + "\n"


def _gemm_body(L, sz, A, B, C, tag, zero_inside=False):
    """一个 pass：(i,j,k) 累加；A=left, B=right, C=out（SSA 名后缀 tag）。"""
    L.append(f"    affine.for %i{tag} = %c0 to %c{sz} {{")
    L.append(f"      affine.for %j{tag} = %c0 to %c{sz} {{")
    if zero_inside:
        L.append(f"        affine.store %cz, %{C}[%i{tag}, %j{tag}] : {memref(sz, sz)}")
    L.append(f"        affine.for %k{tag} = %c0 to %c{sz} {{")
    L.append(f"          %a{tag} = affine.load %{A}[%i{tag}, %k{tag}] : {memref(sz, sz)}")
    L.append(f"          %b{tag} = affine.load %{B}[%k{tag}, %j{tag}] : {memref(sz, sz)}")
    L.append(f"          %c{tag} = affine.load %{C}[%i{tag}, %j{tag}] : {memref(sz, sz)}")
    L.append(f"          %m{tag} = arith.mulf %a{tag}, %b{tag} : f32")
    L.append(f"          %s{tag} = arith.addf %c{tag}, %m{tag} : f32")
    L.append(f"          affine.store %s{tag}, %{C}[%i{tag}, %j{tag}] : {memref(sz, sz)}")
    L.append("        }")
    L.append("      }")
    L.append("    }")


def emit_gemm(k):
    M, N, K = k["params"]["M"], k["params"]["N"], k["params"]["K"]
    # 方形走 sgemm 模板；矩形走 generic 三维
    if M == N == K:
        sz = M
        m = memref(sz, sz)
        L = [_header(k), "module {",
             f"  func.func @{k['func']}(%A: {m}, %B: {m}, %C: {m}) {{",
             f"    affine.for %i = 0 to {sz} {{",
             f"      affine.for %j = 0 to {sz} {{",
             f"        affine.for %k = 0 to {sz} {{",
             f"          %a = affine.load %A[%i, %k] : {m}",
             f"          %b = affine.load %B[%k, %j] : {m}",
             f"          %c = affine.load %C[%i, %j] : {m}",
             "          %p = arith.mulf %a, %b : f32",
             "          %s = arith.addf %p, %c : f32",
             f"          affine.store %s, %C[%i, %j] : {m}",
             "        }", "      }", "    }",
             "    return", "  }", "}"]
        return "\n".join(L) + "\n"
    # 矩形：M×K · K×N → M×N
    L = [_header(k), "module {",
         f"  func.func @{k['func']}(%A: {memref(M, K)}, %B: {memref(K, N)}, %C: {memref(M, N)}) {{",
         f"    affine.for %i = 0 to {M} {{",
         f"      affine.for %j = 0 to {N} {{",
         f"        affine.for %k = 0 to {K} {{",
         f"          %a = affine.load %A[%i, %k] : {memref(M, K)}",
         f"          %b = affine.load %B[%k, %j] : {memref(K, N)}",
         f"          %c = affine.load %C[%i, %j] : {memref(M, N)}",
         "          %p = arith.mulf %a, %b : f32",
         "          %s = arith.addf %p, %c : f32",
         f"          affine.store %s, %C[%i, %j] : {memref(M, N)}",
         "        }", "      }", "    }",
         "    return", "  }", "}"]
    return "\n".join(L) + "\n"


def emit_chain2(k):
    sz = k["params"]["SZ"]
    m = memref(sz, sz)
    L = [_header(k), "module {",
         f"  func.func @{k['func']}(%A: {m}, %B: {m}, %C: {m}, %D: {m}, %E: {m}) {{",
         "    %c0 = arith.constant 0 : index",
         f"    %c{sz} = arith.constant {sz} : index",
         "    %cz = arith.constant 0.0 : f32",
         f"    affine.for %i0 = %c0 to %c{sz} {{",
         f"      affine.for %j0 = %c0 to %c{sz} {{",
         f"        affine.store %cz, %E[%i0, %j0] : {m}",
         "      }", "    }",
         f"    affine.for %i1 = %c0 to %c{sz} {{",
         f"      affine.for %j1 = %c0 to %c{sz} {{",
         f"        affine.store %cz, %D[%i1, %j1] : {m}",
         "      }", "    }",
         f"    affine.for %i2 = %c0 to %c{sz} {{",
         f"      affine.for %j2 = %c0 to %c{sz} {{",
         f"        affine.for %k2 = %c0 to %c{sz} {{",
         f"          %a2 = affine.load %A[%i2, %k2] : {m}",
         f"          %b2 = affine.load %B[%k2, %j2] : {m}",
         f"          %e2 = affine.load %E[%i2, %j2] : {m}",
         "          %m2 = arith.mulf %a2, %b2 : f32",
         "          %s2 = arith.addf %e2, %m2 : f32",
         f"          affine.store %s2, %E[%i2, %j2] : {m}",
         "        }", "      }", "    }",
         f"    affine.for %i3 = %c0 to %c{sz} {{",
         f"      affine.for %j3 = %c0 to %c{sz} {{",
         f"        affine.for %k3 = %c0 to %c{sz} {{",
         f"          %e3 = affine.load %E[%i3, %k3] : {m}",
         f"          %c3 = affine.load %C[%k3, %j3] : {m}",
         f"          %d3 = affine.load %D[%i3, %j3] : {m}",
         "          %m3 = arith.mulf %e3, %c3 : f32",
         "          %s3 = arith.addf %d3, %m3 : f32",
         f"          affine.store %s3, %D[%i3, %j3] : {m}",
         "        }", "      }", "    }",
         "    return", "  }", "}"]
    return "\n".join(L) + "\n"


def emit_chain3(k):
    sz = k["params"]["SZ"]
    m = memref(sz, sz)
    L = [_header(k), "module {",
         (f"  func.func @{k['func']}(%A: {m}, %B: {m}, %C: {m}, %D: {m}, "
          f"%E: {m}, %F: {m}, %G: {m}) {{"),
         "    %c0 = arith.constant 0 : index",
         f"    %c{sz} = arith.constant {sz} : index",
         "    %cz = arith.constant 0.0 : f32"]
    # pass1: E=A·B；pass2: F=C·D；pass3: G=E·F（清零均在 (i,j) 层，镜像 three_mm_probe）
    for tag, (LA, LB, LC) in (("1", ("A", "B", "E")), ("2", ("C", "D", "F")), ("3", ("E", "F", "G"))):
        L.append(f"    affine.for %i{tag} = %c0 to %c{sz} {{")
        L.append(f"      affine.for %j{tag} = %c0 to %c{sz} {{")
        L.append(f"        affine.store %cz, %{LC}[%i{tag}, %j{tag}] : {m}")
        L.append(f"        affine.for %k{tag} = %c0 to %c{sz} {{")
        L.append(f"          %a{tag} = affine.load %{LA}[%i{tag}, %k{tag}] : {m}")
        L.append(f"          %b{tag} = affine.load %{LB}[%k{tag}, %j{tag}] : {m}")
        L.append(f"          %c{tag} = affine.load %{LC}[%i{tag}, %j{tag}] : {m}")
        L.append(f"          %m{tag} = arith.mulf %a{tag}, %b{tag} : f32")
        L.append(f"          %s{tag} = arith.addf %c{tag}, %m{tag} : f32")
        L.append(f"          affine.store %s{tag}, %{LC}[%i{tag}, %j{tag}] : {m}")
        L.append("        }")
        L.append("      }")
        L.append("    }")
    L += ["    return", "  }", "}"]
    return "\n".join(L) + "\n"


def emit_mvt(k):
    n = k["params"]["N"]
    mv = memref(n)
    mm = memref(n, n)
    L = [_header(k), "module {",
         (f"  func.func @{k['func']}(%A: {mm}, %x1: {mv}, %x2: {mv}, "
          f"%y1: {mv}, %y2: {mv}) {{"),
         "    %c0 = arith.constant 0 : index",
         f"    %c{n} = arith.constant {n} : index"]
    for tag, (xs, ys) in (("1", ("x1", "y1")), ("2", ("x2", "y2"))):
        L.append(f"    affine.for %i{tag} = %c0 to %c{n} {{")
        L.append(f"      affine.for %j{tag} = %c0 to %c{n} {{")
        L.append(f"        %xv{tag} = affine.load %{xs}[%i{tag}] : {mv}")
        L.append(f"        %a{tag} = affine.load %A[%i{tag}, %j{tag}] : {mm}")
        L.append(f"        %yv{tag} = affine.load %{ys}[%j{tag}] : {mv}")
        L.append(f"        %m{tag} = arith.mulf %a{tag}, %yv{tag} : f32")
        L.append(f"        %s{tag} = arith.addf %xv{tag}, %m{tag} : f32")
        L.append(f"        affine.store %s{tag}, %{xs}[%i{tag}] : {mv}")
        L.append("      }")
        L.append("    }")
    L += ["    return", "  }", "}"]
    return "\n".join(L) + "\n"


def emit_bicg(k):
    n = k["params"]["N"]
    mv = memref(n)
    mm = memref(n, n)
    L = [_header(k), "module {",
         (f"  func.func @{k['func']}(%A: {mm}, %s: {mv}, %q: {mv}, "
          f"%p: {mv}, %r: {mv}) {{"),
         "    %c0 = arith.constant 0 : index",
         f"    %c{n} = arith.constant {n} : index",
         # pass1: s[j] += r[i]·A[i,j]（红外/写内，镜像 bicg_probe）
         f"    affine.for %i1 = %c0 to %c{n} {{",
         f"      affine.for %j1 = %c0 to %c{n} {{",
         f"        %sv = affine.load %s[%j1] : {mv}",
         f"        %rv = affine.load %r[%i1] : {mv}",
         f"        %a1 = affine.load %A[%i1, %j1] : {mm}",
         "        %m1 = arith.mulf %rv, %a1 : f32",
         "        %s1 = arith.addf %sv, %m1 : f32",
         f"        affine.store %s1, %s[%j1] : {mv}",
         "      }", "    }",
         # pass2: q[i] += A[i,j]·p[j]（标准）
         f"    affine.for %i2 = %c0 to %c{n} {{",
         f"      affine.for %j2 = %c0 to %c{n} {{",
         f"        %qv = affine.load %q[%i2] : {mv}",
         f"        %a2 = affine.load %A[%i2, %j2] : {mm}",
         f"        %pv = affine.load %p[%j2] : {mv}",
         "        %m2 = arith.mulf %a2, %pv : f32",
         "        %s2 = arith.addf %qv, %m2 : f32",
         f"        affine.store %s2, %q[%i2] : {mv}",
         "      }", "    }",
         "    return", "  }", "}"]
    return "\n".join(L) + "\n"


EMITTERS = {
    "sten1d": emit_sten1d,
    "sten2d": emit_sten2d,
    "sibling": emit_sibling,
    "gemm": emit_gemm,
    "gemm_rect": emit_gemm,
    "chain2": emit_chain2,
    "chain3": emit_chain3,
    "mvt": emit_mvt,
    "bicg": emit_bicg,
}


# ---------------------------------------------------------------------------
# 元数据
# ---------------------------------------------------------------------------
def sidecar(k, mlir_text, mlir_path, gen_at):
    h = hashlib.sha256(mlir_text.encode("utf-8")).hexdigest()
    return dict(
        id=k["id"], name=k["name"], func=k["func"], class_name=k["class_name"],
        form=k["form"], family=k["family"], params=k["params"], primary=k["primary"],
        vehicle=k["vehicle"], notes=k["notes"], seed=SEED, generator=GEN_VERSION,
        generated_at=gen_at, mlir=os.path.basename(mlir_path), mlir_sha256=h,
    )


def generate(force=True):
    os.makedirs(OUT, exist_ok=True)
    gen_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    suite = []
    for k in KERNELS:
        emitter = EMITTERS[k["form"]]
        text = emitter(k)
        path = os.path.join(OUT, k["name"] + ".mlir")
        jpath = os.path.join(OUT, k["name"] + ".json")
        if not force and (os.path.exists(path) or os.path.exists(jpath)):
            raise SystemExit(f"文件已存在（用 --force 覆盖）：{path}")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        sc = sidecar(k, text, path, gen_at)
        with open(jpath, "w", encoding="utf-8", newline="\n") as f:
            json.dump(sc, f, ensure_ascii=False, indent=2)
            f.write("\n")
        suite.append(sc)
    payload = dict(version="v2", generator=GEN_VERSION, seed=SEED,
                   generated_at=gen_at, n_primary=sum(1 for k in KERNELS if k["primary"]),
                   n_secondary=sum(1 for k in KERNELS if not k["primary"]),
                   kernels=suite)
    suite_path = os.path.join(OUT, "suite.json")
    with open(suite_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[生成] {len(KERNELS)} 个 kernel → {OUT}")
    print(f"       主集 {payload['n_primary']} / 副集 {payload['n_secondary']}")
    print(f"[suite] {suite_path}")
    print(f"        sha256 = {hashlib.sha256(open(suite_path,'rb').read()).hexdigest()}")
    return payload


def check():
    suite_path = os.path.join(OUT, "suite.json")
    if not os.path.exists(suite_path):
        raise SystemExit(f"缺 suite.json：{suite_path}（先跑生成）")
    with open(suite_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    errs = []
    ids, names, pairs = set(), set(), set()
    for k in payload["kernels"]:
        if k["id"] in ids:
            errs.append(f"重复 id: {k['id']}")
        if k["name"] in names:
            errs.append(f"重复 name: {k['name']}")
        ids.add(k["id"]); names.add(k["name"])
        # (形态, 尺寸) 唯一：尺寸取 params 的字符串形式（形态 = 算子类，不是 streaming/reuse）
        pair = (k["form"], json.dumps(k["params"], sort_keys=True))
        if pair in pairs:
            errs.append(f"重复 (类,尺寸): {pair}")
        pairs.add(pair)
        p = os.path.join(OUT, k["mlir"])
        if not os.path.exists(p):
            errs.append(f"缺 mlir: {p}")
            continue
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if h != k["mlir_sha256"]:
            errs.append(f"哈希不一致: {k['mlir']}")
    n_pri = sum(1 for k in payload["kernels"] if k["primary"])
    if n_pri != 12:
        errs.append(f"主集数 {n_pri} != 12")
    if len(payload["kernels"]) != 16:
        errs.append(f"总数 {len(payload['kernels'])} != 16")
    if errs:
        print("[check] FAIL")
        for e in errs:
            print("  -", e)
        return 1
    print(f"[check] OK：{len(payload['kernels'])} kernels（主 {n_pri}），哈希全一致")
    print(f"{'id':4s} {'name':24s} {'class':9s} {'form':10s} {'primary':7s} params")
    for k in payload["kernels"]:
        print(f"{k['id']:4s} {k['name']:24s} {k['class_name']:9s} "
              f"{k['form']:10s} {str(k['primary']):7s} {k['params']}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-force", action="store_true")
    args = ap.parse_args()
    if args.check:
        return check()
    generate(force=not args.no_force)
    return check()


if __name__ == "__main__":
    sys.exit(main())
