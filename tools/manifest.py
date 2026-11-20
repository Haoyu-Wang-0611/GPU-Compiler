#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""manifest.py — 证据树（48h 审计 A1 的落地）

扫描工作区的日志/载具/IR/决策输出/文档，生成 `docs/evidence/manifest.json`：
  path, sha256, bytes, mtime, kind, kernel, status, note

状态口径（与 README 红线、measurement_audit 一致）：
  clean / contaminated / buggy / invalid / jitter-only / no-pass /
  unlabeled-session / needs-review / superseded / probe / unknown

用法：
  python tools/manifest.py                 # 生成 manifest.json + 摘要
  python tools/manifest.py --check         # 校验：15 行主表日志齐全、无 unknown 日志
  python tools/manifest.py --external      # 额外扫描 D:\\CC_GPU_1（只读）
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import sys
from collections import Counter

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, "docs", "evidence", "manifest.json")
EXT_ROOTS = [r"D:\CC_GPU_1"]

# 主表 15 行 → 规范日志文件名（与 docs/evidence/replay_*.md 一致）
MAIN_TABLE = {
    "gemm4096": "sweep_run1.log",
    "gemm1024": "sweep_1024.log",
    "gemm512": "sweep_512_clean.log",
    "gemm500": "sweep_500.log",
    "twomm": "twomm_512.log",
    "three_mm": "3mm_512.log",
    "atax": "atax_sweep2.log",
    "mvt": "mvt_512.log",
    "bicg": "bicg_512.log",
    "jacobi2d": "jacobi2d_sweep.log",
    "jacobi1d": "jacobi1d_2p24_sweep.log",
    "heat1d24": "heat1d_2^24.log",
    "heat1d20": "heat1d_2^20_redo.log",
    "syrk": "stril_syrk_512_v3.log",
    "symm": "stril_symm_512_v2.log",
}

# 文件名 → (status, note)
REDLINE = {
    "sweep_512_S1.log": ("contaminated", "整轮 +12%（README 红线；用 sweep_512_clean）"),
    "sweep_512_S4.log": ("contaminated", "整轮 +12%（README 红线；用 sweep_512_clean）"),
    "atax_sweep.log": ("buggy", "acc 作用域 bug（用 atax_sweep2）"),
    "heat1d_2^20.log": ("invalid", "4 行 2.2µs launch 失败假数据（用 _redo）"),
    "3mm_rerun1.log": ("jitter-only", "低块数同会话抖动；只作现象证据"),
    "3mm_rerun2.log": ("jitter-only", "低块数同会话抖动；只作现象证据"),
    "stril_symm_512_v3.log": ("needs-review", "README 称实为 syrk；dual-mode 修复后待人工确认"),
    "sweep_500.log": ("unlabeled-session", "会话未标注，可能同期污染"),
    "mvt_512.log": ("no-pass", "measurement_audit C6：未打印 PASS"),
    "bicg_512.log": ("no-pass", "measurement_audit C6：未打印 PASS"),
    "heat1d_2^24.log": ("no-pass", "measurement_audit C6：未打印 PASS"),
    "heat1d_2^20_redo.log": ("no-pass", "measurement_audit C6：未打印 PASS"),
    "stril_symm_512.log": ("needs-review", "v1；与 v2 的 5.9% 漂移未裁决"),
    "stril_symm_512_v2.log": ("needs-review", "v2；与 v1 的 5.9% 漂移未裁决"),
    "stril_syrk_512.log": ("superseded", "v1（污染轮）；用 v3/v4"),
    "stril_syrk_512_v2.log": ("contaminated", "v2 为污染轮（36.1 vs 38.8）"),
    "jacobi1d_sweep.log": ("superseded", "旧小尺寸；主表用 2p24"),
    "jacobi1d_big_sweep.log": ("superseded", "旧大尺寸；主表用 2p24"),
    "sweep_512.log": ("superseded", "11-14 首轮；干净复现用 sweep_512_clean"),
    "gemm512_clean.log": ("clean", ""),
}

CLEAN = {
    "sweep_512_clean.log", "sweep_1024.log", "sweep_run1.log", "sweep_run2.log",
    "sweep_friday.log", "twomm_512.log", "3mm_512.log", "jacobi2d_sweep.log",
    "jacobi1d_2p24_sweep.log", "atax_sweep2.log", "bufprobe_512.log",
    "bufprobe_1024.log", "bufprobe_ptxas.log", "guard6.log", "guard7.log",
    "compile.log", "stril_syrk_512_v3.log", "stril_syrk_512_v4.log",
    "stril_symm_512_v3.log",
}

KERNEL_PATTERNS = [
    ("gemm4096", r"gemm4096|sweep_run[12]|sweep_friday"),
    ("gemm1024", r"gemm1024|sweep_1024"),
    ("gemm512", r"gemm512|sweep_512|sweep_500|gemm500"),
    ("gemm500", r"gemm500|sweep_500"),
    ("twomm", r"twomm"),
    ("three_mm", r"3mm|three_mm"),
    ("atax", r"atax"),
    ("mvt", r"mvt"),
    ("bicg", r"bicg"),
    ("jacobi1d", r"jacobi1d"),
    ("jacobi2d", r"jacobi2d"),
    ("heat1d", r"heat1d"),
    ("syrk", r"syrk"),
    ("symm", r"symm"),
    ("bufprobe", r"bufprobe"),
    ("guard", r"guard"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_kernel(name):
    low = name.lower()
    for k, pat in KERNEL_PATTERNS:
        if re.search(pat, low):
            return k
    return ""


def classify(rel, name):
    ext = os.path.splitext(name)[1].lower()
    if name in REDLINE:
        st, note = REDLINE[name]
        kind = "log" if ext == ".log" else "artifact"
        return kind, st, note
    if ext == ".log":
        st = "clean" if name in CLEAN else "unknown"
        return "log", st, ""
    if ext == ".mlir":
        return "kernel_input", "clean", ""
    if ext == ".cu":
        return "vehicle", "clean", ""
    if ext == ".json":
        return "metadata", "clean", ""
    if ext == ".py":
        return "tool", "clean", ""
    if ext == ".md":
        return "doc", "clean", ""
    if ext == ".txt":
        return "decision_or_note", "clean", ""
    return "other", "unknown", ""


def scan(root, label):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in
                       {".git", ".deps", ".vscode", "__pycache__", "node_modules"}]
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace("\\", "/")
            try:
                st = os.stat(full)
                h = sha256(full)
            except OSError:
                continue
            kind, status, note = classify(rel, name)
            out.append(dict(root=label, path=rel, name=name, bytes=st.st_size,
                            mtime=datetime.datetime.fromtimestamp(st.st_mtime)
                            .astimezone().isoformat(timespec="seconds"),
                            sha256=h, kind=kind, kernel=infer_kernel(name),
                            status=status, note=note))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--external", action="store_true")
    args = ap.parse_args()

    files = []
    for sub in ("daima", "docs", "tools", "ceshi-shuchu", "ceshi-shuchu2"):
        p = os.path.join(WS, sub)
        if os.path.isdir(p):
            files += scan(p, sub)
    # 根目录的 md/txt（README、CHANGELOG、Untitled）
    for name in os.listdir(WS):
        full = os.path.join(WS, name)
        if os.path.isfile(full) and os.path.splitext(name)[1].lower() in (".md", ".txt"):
            st = os.stat(full)
            kind, status, note = classify(name, name)
            files.append(dict(root=".", path=name, name=name, bytes=st.st_size,
                              mtime=datetime.datetime.fromtimestamp(st.st_mtime)
                              .astimezone().isoformat(timespec="seconds"),
                              sha256=sha256(full), kind=kind, kernel=infer_kernel(name),
                              status=status, note=note))
    external = []
    if args.external:
        for r in EXT_ROOTS:
            if os.path.isdir(r):
                external = scan(r, r)
                files += external

    files.sort(key=lambda x: (x["root"], x["path"]))
    by_status = Counter(x["status"] for x in files)
    by_kind = Counter(x["kind"] for x in files)
    logs = [x for x in files if x["kind"] == "log"]
    unknown_logs = [x["path"] for x in logs if x["status"] == "unknown"]
    # 重复指纹
    seen, dups = {}, []
    for x in files:
        if x["sha256"] in seen:
            dups.append([seen[x["sha256"]], x["path"]])
        else:
            seen[x["sha256"]] = x["path"]

    payload = dict(version="manifest v0.1", generated_at=datetime.datetime.now()
                   .astimezone().isoformat(timespec="seconds"),
                   roots=["daima", "docs", "tools", "ceshi-shuchu", "ceshi-shuchu2", "."] +
                         ([r for r in EXT_ROOTS if os.path.isdir(r)] if args.external else []),
                   summary=dict(total=len(files), logs=len(logs), by_status=dict(by_status),
                                by_kind=dict(by_kind), unknown_logs=len(unknown_logs),
                                duplicate_hashes=len(dups)),
                   main_table=MAIN_TABLE, files=files)

    if args.check:
        names = {x["name"] for x in files}
        missing = [f"{k}:{v}" for k, v in MAIN_TABLE.items() if v not in names]
        print(f"[check] 文件 {len(files)}；日志 {len(logs)}；unknown 日志 {len(unknown_logs)}；"
              f"重复指纹 {len(dups)}")
        if unknown_logs:
            print("  unknown 日志：")
            for p in unknown_logs:
                print("   -", p)
        if missing:
            print("  主表缺失：", missing)
        if dups:
            print("  重复指纹（前 10）：")
            for a, b in dups[:10]:
                print(f"   - {a} == {b}")
        ok = not unknown_logs and not missing
        print("[check]", "OK" if ok else "FAIL")
        return 0 if ok else 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[写出] {OUT}")
    print(f"  总 {len(files)}；日志 {len(logs)}；kind={dict(by_kind)}")
    print(f"  status={dict(by_status)}")
    if unknown_logs:
        print(f"  ⚠ unknown 日志 {len(unknown_logs)}：{unknown_logs[:5]}...")
    if dups:
        print(f"  ⚠ 重复指纹 {len(dups)} 组")
    return 0


if __name__ == "__main__":
    sys.exit(main())
