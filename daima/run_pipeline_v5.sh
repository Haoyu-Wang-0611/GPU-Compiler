#!/bin/bash
# run_pipeline.sh v5 — 四类 kernel 发射适配（stencil / stencil-2d / GEMM / pass-chain GEMV）
# 用法:
#   ./run_pipeline.sh                    # heat1d (stencil)，用 L1Feas 决策 (T,S)
#   ./run_pipeline.sh <src.mlir> [S]     # 指定输入；S>0 强制流水深度（验证用）
#
# v5 新增（11-08）:
#   stencil-2d 分支（L1Feas v1.2 配套）:
#     解析: Best: Tile2d=T,T Stages=S（区别于 1D 的 Best: Tile=T）
#     tile 命令: tile-sizes=KINF,T,T —— 时间环 L0 退化切分出壳（unroll 消壳，
#       时间环保持整条，与 GEMM 的 k 处理同套路）；空间环切 T×T
#     流水化目标 = 最内层空间环（mulf→get_parent_op 现成）；maxCycle 自动标定现成
#   其余: stencil-1D / GEMM / pass-chain 路径与 v4 逐字节一致
set -e

MLOPT=~/haoyu/llvm-project/build/bin/mlir-opt
MTOPT=~/haoyu/llvm-project/build/bin/mlir-transform-opt
L1FEAS=~/haoyu/l1-feas/build/mlir-l1feas

SRC=${1:-~/haoyu/scratch/heat1d.mlir}
FORCE_S=${2:-0}          # 0 = 使用决策器输出；>0 = 强制指定 S（验证用）
KINF=1073741824          # 2^30: 退化切分哨兵（须大于对应环 trip）
WORK=$(dirname $(realpath $SRC))

# ---------- 发射脚本生成器: $1 = iteration_interval ----------
gen_pipeline_script() {
  cat > $WORK/pipeline_auto.mlir <<EOF
module attributes {transform.with_named_sequence} {
transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
  %mul = transform.structured.match ops{["arith.mulf"]} in %root
    : (!transform.any_op) -> !transform.any_op
  %kloop = transform.get_parent_op %mul
    : (!transform.any_op) -> !transform.op<"scf.for">
  transform.loop.pipeline %kloop iteration_interval = $1
    : (!transform.op<"scf.for">) -> !transform.any_op
  transform.yield
}
}
EOF
}

# ---------- 依赖链自动标定: II=1 探测深度 D → maxCycle = D-1 ----------
# 克隆数按"每迭代 mulf 基数"归一（GEMM/2D-stencil 基数 1，atax 基数 2）
calib_maxcycle() {
  BASE=$(grep -c arith.mulf $WORK/lowered.mlir)
  [ -z "$BASE" ] || [ "$BASE" -eq 0 ] && BASE=1
  gen_pipeline_script 1
  if ! $MTOPT $WORK/lowered.mlir -transform=$WORK/pipeline_auto.mlir > $WORK/probe_depth.mlir 2> $WORK/emit_err.txt; then
    echo "[glue] 标定发射失败（II=1）——emit_err.txt:"
    cat $WORK/emit_err.txt
    exit 1
  fi
  CLONES=$(grep -c arith.mulf $WORK/probe_depth.mlir)
  if [ "$CLONES" -le "$BASE" ]; then
    echo "[glue] II=1 探测深度=1：无跨迭代依赖链，流水无收益，输出降级 IR"
    cp $WORK/lowered.mlir $WORK/final.mlir
    echo "[glue] 完成: $WORK/final.mlir（未流水）"
    exit 0
  fi
  DEPTH=$((CLONES / BASE))
  MC=$((DEPTH - 1))
  echo "[glue] 自动标定: 基数=$BASE, II=1 → 克隆=$CLONES → 探测深度=$DEPTH → maxCycle=$MC"
}

# ---------- 1. L1 分析 ----------
$L1FEAS $SRC -pass-pipeline 'builtin.module(func.func(l1-feasibility{num-stages=3 tile-size=64}))' -o /dev/null 2> $WORK/l1_out.txt

# ---------- 2. 决策解析（v5 四分支）----------
KERNEL=$(grep -oP '\[L1\] detected: \K(stencil|matmul)' $WORK/l1_out.txt | head -1)
if [ -z "$KERNEL" ]; then
  echo "[glue] 决策器无 kernel 探测输出，终止"
  exit 1
fi
BEST_S=$(grep -oP 'Stages=\K[0-9]+' $WORK/l1_out.txt | head -1)
MODE=""

if echo "$KERNEL" | grep -q stencil; then
  BEST_T2=$(grep -oP 'Best: Tile2d=\K[0-9]+' $WORK/l1_out.txt | head -1)
  if [ -n "$BEST_T2" ]; then
    MODE=stencil2d
    echo "[glue] kernel 类: stencil-2d, 决策: Tile=$BEST_T2 x $BEST_T2 Stages=$BEST_S"
  else
    BEST_T=$(grep -oP 'Best: Tile=\K[0-9]+' $WORK/l1_out.txt | head -1)
    if [ -z "$BEST_T" ]; then
      echo "[glue] stencil 无可行配置，决策器拒因:"
      grep -m1 'REJECT' $WORK/l1_out.txt
      exit 1
    fi
    MODE=stencil
    echo "[glue] kernel 类: stencil, 决策: Tile=$BEST_T Stages=$BEST_S"
  fi
else
  if grep -q 'pass-chain' $WORK/l1_out.txt; then
    T_OUT=$(grep -oP 'Best: T=\K[0-9]+' $WORK/l1_out.txt | head -1)
    TK_R=$(grep -oP ' TK=\K[0-9]+' $WORK/l1_out.txt | head -1)
    if [ -z "$T_OUT" ]; then
      echo "[glue] pass-chain 无可行配置，决策器拒因:"
      grep -m1 'REJECT' $WORK/l1_out.txt
      exit 1
    fi
    MODE=passchain
    echo "[glue] kernel 类: matmul(pass-chain GEMV), 决策: T=$T_OUT TK=$TK_R Stages=$BEST_S"
  else
    BM=$(grep -oP 'Best: BM=\K[0-9]+' $WORK/l1_out.txt | head -1)
    BN=$(grep -oP ' BN=\K[0-9]+' $WORK/l1_out.txt | head -1)
    TK=$(grep -oP ' TK=\K[0-9]+' $WORK/l1_out.txt | head -1)
    if [ -z "$BM" ]; then
      echo "[glue] matmul 无可行配置，决策器拒因:"
      grep -m1 'REJECT' $WORK/l1_out.txt
      exit 1
    fi
    MODE=gemm
    echo "[glue] kernel 类: matmul(GEMM), 决策: BM=$BM BN=$BN TK=$TK Stages=$BEST_S"
  fi
fi

# ============================================================
# stencil 1D 分支（v1 原样保留，回归锚点）
# ============================================================
if [ "$MODE" = "stencil" ]; then
  $MLOPT -lower-affine $SRC -o $WORK/lowered.mlir

  if [ "$FORCE_S" -gt 0 ]; then BEST_S=$FORCE_S; fi
  if [ "$BEST_S" -le 1 ]; then
    echo "[glue] S=1: 不流水（平凡发射），输出 = 降级 IR"
    cp $WORK/lowered.mlir $WORK/final.mlir
    echo "[glue] 完成: $WORK/final.mlir"
    exit 0
  fi

  calib_maxcycle
  II=$(( (MC + BEST_S - 2) / (BEST_S - 1) ))
  if [ "$II" -lt 1 ]; then II=1; fi
  if [ "$II" -eq 1 ] && [ "$BEST_S" -gt "$DEPTH" ]; then
    echo "[glue] 注意: 模型 S=$BEST_S 超过依赖链允许深度 $DEPTH，实际深度=$DEPTH"
  fi
  gen_pipeline_script $II
  echo "[glue] 深度 S=$BEST_S → II=$II (maxCycle=$MC)"

  $MTOPT $WORK/lowered.mlir -transform=$WORK/pipeline_auto.mlir > $WORK/final.mlir
  echo "[glue] 完成: $WORK/final.mlir（流水化 IR）"
  exit 0
fi

# ============================================================
# 三参数 tile 骨架（GEMM / pass-chain / stencil-2d 共用，tile 参数按模式）
# ============================================================
if [ "$FORCE_S" -gt 0 ]; then BEST_S=$FORCE_S; fi

# (1) tile 命令按模式拼装
if [ "$MODE" = "gemm" ]; then
  TILE_SIZES="$BM,$BN,$KINF"          # i/j 切 BM/BN，k 退化切分保整条
elif [ "$MODE" = "stencil2d" ]; then
  # 11-08 发射形式修正：流水机器只接受常数界（探测矩阵+本次失败一致证实）。
  # 空间双维都切 T 会让 j1 带动态 min 界（父环 i0 是真环，折叠不掉）→ 拒收。
  # 解法：只切 i 方向（行带），j 用 KINF 壳折叠成常数界整条 → 流水 j。
  # 【已登记偏差】发射形态 = 行带流式（时间整条 + i 带 T + j 整条），
  #   与模型的 2D tile 语义不一致——模型决策仍由手写 CUDA oracle 验证；
  #   完整 2D tile 发射需上游动态上界支持（"改流水机器"清单）。
  TILE_SIZES="$KINF,$BEST_T2,$KINF"
else
  TILE_SIZES="$T_OUT,$KINF"           # band 对称假设: out 恒在 band 首位
fi
$MLOPT -affine-loop-tile="tile-sizes=$TILE_SIZES" $SRC -o $WORK/tiled.mlir
echo "[glue] tile 完成: tile-sizes=$TILE_SIZES"

# (2) 壳消除: unroll-full 只动 trip<=1 的壳循环 → 内层界常数化
#     注意: unroll 是函数级 pass，须用 -pass-pipeline 包 func.func
$MLOPT -pass-pipeline 'builtin.module(func.func(affine-loop-unroll{unroll-factor=-1 unroll-full-threshold=1}))' \
    $WORK/tiled.mlir -o $WORK/unrolled.mlir
$MLOPT -canonicalize $WORK/unrolled.mlir -o $WORK/constbounds.mlir
echo "[glue] 壳消除完成"

# (3) 降级 affine → scf
$MLOPT -lower-affine -canonicalize $WORK/constbounds.mlir -o $WORK/lowered.mlir
echo "[glue] 降级完成: $WORK/lowered.mlir"

# (4) S=1: 不流水
if [ "$BEST_S" -le 1 ]; then
  echo "[glue] S=1: 不流水（平凡发射），输出 = 降级 IR"
  cp $WORK/lowered.mlir $WORK/final.mlir
  echo "[glue] 完成: $WORK/final.mlir"
  exit 0
fi

# (5) 自动标定 + 发射
calib_maxcycle
II=$(( (MC + BEST_S - 2) / (BEST_S - 1) ))
if [ "$II" -lt 1 ]; then II=1; fi
if [ "$II" -eq 1 ] && [ "$BEST_S" -gt "$DEPTH" ]; then
  echo "[glue] 注意: 模型 S=$BEST_S 超过依赖链允许深度 $DEPTH，实际深度=$DEPTH"
fi
gen_pipeline_script $II
echo "[glue] 深度 S=$BEST_S → II=$II (maxCycle=$MC)"

# (6) 发射（mlir-transform-opt 输出用 > 重定向，-o 无效）
if ! $MTOPT $WORK/lowered.mlir -transform=$WORK/pipeline_auto.mlir > $WORK/final.mlir 2> $WORK/emit_err.txt; then
  echo "[glue] 发射失败——emit_err.txt 内容:"
  cat $WORK/emit_err.txt
  exit 1
fi
echo "[glue] 完成: $WORK/final.mlir（流水化 IR）"
