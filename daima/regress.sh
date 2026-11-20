#!/bin/bash
# regress.sh — 十二发回归一键脚本（v1.8 起，基线 = v1.7.1 11-12 21:00 输出）
# 用法: ./regress.sh            # 正常回归（生成本轮输出 + 与 golden 逐字节 diff）
#       ./regress.sh --rebase   # 重新生成 golden（仅在有"预期内差异"并核销后使用）
#
# 依赖: $L1 与 $PASS 环境变量（见 ~/.bashrc），goldens/ 与本脚本同目录
set -u
L1=${L1:-~/haoyu/l1-feas/build/mlir-l1feas}
PASS=${PASS:-'builtin.module(func.func(l1-feasibility{num-stages=3 tile-size=64}))'}
DIR=$(cd "$(dirname "$0")" && pwd)
GOLD="$DIR/goldens"
mkdir -p "$GOLD"

# 十四发清单（顺序 = 基线顺序）
LIST="heat1d jacobi1d_big2 jacobi2d gemm4096 gemm512 gemm500 gemm1024 atax_big probe_dyn twomm mvt_probe bicg_probe symm_true_probe symm_full_probe three_mm_probe syrk_probe"

fail=0
for f in $LIST; do
  src=~/haoyu/scratch/$f.mlir
  [ -f "$src" ] || { echo "[skip] $f.mlir 不存在"; continue; }
  "$L1" "$src" -pass-pipeline "$PASS" -o /dev/null 2> "$DIR/l1_$f.txt"
  if [ ! -f "$GOLD/l1_$f.golden" ]; then
    cp "$DIR/l1_$f.txt" "$GOLD/l1_$f.golden"
    echo "[golden 新建] $f"
    continue
  fi
  # 比较用"排序后 diff"：[L1 Anchor] 行的内容确定但顺序受 map 迭代序影响
  #   （11-13 发现，登记待修：L1Feas Anchor 打印应改确定性排序）；
  #   Decision 行本身单行无序问题。排序 diff = 内容敏感、行序不敏感。
  if diff <(sort "$GOLD/l1_$f.golden") <(sort "$DIR/l1_$f.txt") > /dev/null; then
    echo "[PASS] $f（内容一致）"
  else
    echo "[DIFF] $f —— 排序后逐行差异："
    diff <(sort "$GOLD/l1_$f.golden") <(sort "$DIR/l1_$f.txt") | head -10
    fail=1
  fi
done

if [ "${1:-}" = "--rebase" ]; then
  echo "[rebase] 以本轮输出重建 golden……"
  for f in $LIST; do
    [ -f "$DIR/l1_$f.txt" ] && cp "$DIR/l1_$f.txt" "$GOLD/l1_$f.golden"
  done
  echo "[rebase] 完成（请把本次差异原因写进台账后再 rebase）"
  exit 0
fi

if [ $fail -eq 0 ]; then echo "===== 回归全部通过 ====="; else echo "===== 存在差异，先核对是否预期内 ====="; fi
