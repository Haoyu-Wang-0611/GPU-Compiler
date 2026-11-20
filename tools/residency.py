#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""residency.py — 用 ptxas 详单 + 扫参日志，离线算"驻留块数"，检验四层取 min 的不等式。

数据来源（都不需要跑 GPU）：
  compile.log            : ptxas 的 "Used N registers"（每个模板实例一行）
  扫参日志               : 每配置的 smem 与实测耗时
  设备规格（待机上核实）  : regs/SM、smem/SM、threads/block

驻留块数 = min( floor(regsPerSM / (regs×threads)),
                floor(smemPerSM / smemPerBlock),
                maxBlocksPerSM )

用法：python tools/residency.py 3mm_512.log [其它日志...]
"""
import re
import sys
import os

BASE = "daima/xfer_2026-11-15"
REGS_RE = re.compile(
    r"Compiling entry function '_Z\w*sgemm_tiledILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E\w*'")
USED_RE = re.compile(r"Used (\d+) registers")
CFG_RE = re.compile(r"BM=(\d+)\s+BN=(\d+)\s+TK=(\d+)\s+S=(\d+)\s+"
                    r"smem=\s*([\d.]+)KB\s+耗时:\s*([\d.]+)\s+us")

# 4090（sm_89, AD102）规格。**注意：smem/SM 与 regs/SM 必须机上核实**，
# 一条 cudaDeviceGetAttribute 即可（见文末命令）。
SPEC = dict(regsPerSM=65536, smemPerSM=128 * 1024, threadsPerBlock=256,
            maxBlocksPerSM=32)


def parse_regs(path):
    regs, cur = {}, None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = REGS_RE.search(line)
            if m:
                cur = tuple(int(m.group(i)) for i in range(1, 5))
                continue
            m = USED_RE.search(line)
            if m and cur:
                regs[cur] = int(m.group(1))
                cur = None
    return regs


def parse_sweep(path):
    out = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for m in CFG_RE.finditer(f.read()):
            k = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
            out[k] = dict(smem=float(m.group(5)) * 1024, us=float(m.group(6)))
    return out


def main():
    logs = sys.argv[1:] or ["3mm_512.log", "sweep_friday.log"]
    regs = parse_regs(os.path.join(BASE, "compile.log"))
    print(f"compile.log 解析到 {len(regs)} 个模板实例的寄存器数")
    if not regs:
        print("解析失败：检查 compile.log 是否为 ptxas -v 输出")
        return

    for log in logs:
        path = os.path.join(BASE, log)
        if not os.path.exists(path):
            continue
        sw = parse_sweep(path)
        print()
        print("=" * 104)
        print(f"{log}: 驻留块数 vs 实测（按耗时升序，只列前 14 + 关注项）")
        print("=" * 104)
        rows = []
        for k, v in sw.items():
            r = regs.get(k)
            if r is None:
                continue
            reg_lim = SPEC["regsPerSM"] // (r * SPEC["threadsPerBlock"])
            smem_lim = int(SPEC["smemPerSM"] // v["smem"]) if v["smem"] > 0 else 99
            res = max(1, min(reg_lim, smem_lim, SPEC["maxBlocksPerSM"]))
            rows.append((v["us"], k, r, v["smem"], reg_lim, smem_lim, res))
        rows.sort()
        print(f"{'BM':>4s} {'BN':>4s} {'TK':>3s} {'S':>2s} {'us':>9s} "
              f"{'regs':>5s} {'smem':>7s} {'寄存器限':>8s} {'smem限':>7s} {'驻留':>4s}")
        for us, k, r, sm, rl, sl, res in rows[:14]:
            print(f"{k[0]:4d} {k[1]:4d} {k[2]:3d} {k[3]:2d} {us:9.1f} "
                  f"{r:5d} {sm/1024:6.0f}K {rl:8d} {sl:7d} {res:4d}")
        print("  ...")
        # 同 tile 不同 S 的"受控对照"：只有 smem 越过台阶
        groups = {}
        for us, k, r, sm, rl, sl, res in rows:
            groups.setdefault(k[:3], []).append((us, k[3], sm, res))
        print(f"\n{'同 tile 受控对照（只有 S 变，按 S 升序）':44s} {'耗时':>18s} {'驻留':>8s}")
        for tile, g in groups.items():
            if len(g) < 2:
                continue
            g.sort(key=lambda x: x[1])                 # 按 S 升序
            resset = {x[3] for x in g}
            span = max(x[0] for x in g) / min(x[0] for x in g)
            if len(resset) < 2 and span < 1.10:
                continue                               # 既无驻留变化又无显著耗时差 → 跳过
            seq = " → ".join(f"S{x[1]}:{x[0]:.1f}us/r{x[3]}" for x in g)
            print(f"  {tile}  {seq}")

    print()
    print("机上核实命令（一条，10 秒）：")
    print("  python - <<'EOF'  # 或写个小 .cu")
    print("  cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);")
    print("  cudaDeviceGetAttribute(&v, cudaDevAttrMaxRegistersPerMultiprocessor, 0);")
    print("  cudaDeviceGetAttribute(&v, cudaDevAttrMaxBlocksPerMultiprocessor, 0);")
    print("  EOF")


if __name__ == "__main__":
    main()
