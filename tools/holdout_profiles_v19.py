#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""holdout_profiles_v19.py — v1.9 的前沿补丁（不改 frozen holdout_profiles.py）

唯一差异：1D stencil 的 `numTiles < 512` 守卫改为 `numTiles < numSms`（128）。
用法：
    import holdout_profiles as P
    import holdout_profiles_v19 as P19
    P19.install()          # 之后 P.frontier / P.feasible 用 v1.9 守卫
"""
import math

import holdout_profiles as P


def _sten1d_hard_v19(prof, cfg, min_tiles=128):
    N, T = prof["N"], cfg["T"]
    if T > N:
        return None
    bytes_ = 0
    for base_elems, elem in prof["legacy_base"]:
        bytes_ += base_elems * elem + (T - 1) * elem
    smem = bytes_ * (cfg["S"] + 1)
    if smem > P.CAP:
        return None
    numTiles = math.ceil(N / T)
    if N >= 8192 and numTiles < min_tiles:
        return None
    return dict(smem=smem, blocks=numTiles)


def install():
    """把 P._sten1d_hard 替换为 v1.9 版本；返回旧函数供恢复。"""
    old = P._sten1d_hard
    P._sten1d_hard = _sten1d_hard_v19
    return old


if __name__ == "__main__":
    import json
    import os
    P19_old = install()
    suite = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "daima", "holdout", "suite.json")
    for e in json.load(open(suite, encoding="utf-8"))["kernels"]:
        hard, contracted, used = P.frontier(e)
        print(f"{e['id']:3s} {e['name']:24s} enum={len(P.enumerate_configs(e)):3d} "
              f"hard={len(hard):3d} contracted={len(contracted):3d} used={len(used):3d}")
