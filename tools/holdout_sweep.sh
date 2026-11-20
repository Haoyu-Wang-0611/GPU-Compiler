#!/usr/bin/env bash
# holdout_sweep.sh — 留出集跑批（R=5；单配置型 + 整轮扫参型两种载具）
#
# 变量不写死：每次会话在终端 export（脚本不保存任何服务器路径）
#   export WORK=~/haoyu/scratch
#   export BIN=$WORK            # 载具二进制目录；源码/二进制在 scratch 根就用 $WORK
#   export GPU=6
#
# 用法：
#   bash holdout_sweep.sh --list
#   DRY=1 bash holdout_sweep.sh all        # 只打印命令（不写日志）
#   bash holdout_sweep.sh all
#   bash holdout_sweep.sh gemm_1152
#
# 载具接口（按现有源码核对）：
#   单配置：sgemm_tiled/twomm_tiled/3mm_tiled   SIZE BM BN TK S
#           stencil1d_tiled/stencil2d_tiled     N T S R（新写）
#           sgemm_rect                          M N K BM BN TK S（新写）
#   整轮扫参：mvt_tiled/bicg_tiled              N            （内部 T×TK×S）
#             jacobi1d_8                    N            （内部 T×S，STEPS 编译期）
#             jacobi2d_768                      （无参；N/STEPS 编译期）
: "${WORK:?请先 export WORK（例：export WORK=~/haoyu/scratch）}"
: "${BIN:?请先 export BIN（例：export BIN=$WORK/cuda_oracle）}"
: "${GPU:?请先 export GPU（例：export GPU=6）}"
DATE=${DATE:-$(date +%F)}
OUT=${OUT:-$WORK/holdout_$DATE}
LOGS=${LOGS:-$OUT/logs}
mkdir -p "$LOGS"
exec 3>&1

log_header() {
  local log=$1 kernel=$2
  {
    echo "=== kernel=$kernel date=$(date -Is) host=$(hostname) ==="
    nvidia-smi -i "$GPU" --query-gpu=index,name,memory.total,memory.used,utilization.gpu,clocks.sm --format=csv,noheader 2>/dev/null
    echo "CUDA_VISIBLE_DEVICES=$GPU"
  } >> "$log"
}

run() {
  if [[ "${DRY:-0}" == "1" ]]; then
    echo "DRY: CUDA_VISIBLE_DEVICES=$GPU $*" >&3
  else
    CUDA_VISIBLE_DEVICES=$GPU "$@"
  fi
}

need() {
  [[ -x "$1" ]] || { echo "[skip] 缺可执行文件：$1" >&2; return 1; }
}

# 注意：sgemm_tiled/twomm_tiled/3mm_tiled 也是**整轮扫参型**
#      （传 SIZE 后自己跑完 48 配置，再用传入的模型配置做排名）；
#      用下面的 sweep_pass，不要按每配置循环。

# ---- 单配置：1D stencil（新载具；N T S R）----
sweep_sten1d() {
  local bin=$1 n=$2 rad=$3 log=$4
  need "$bin" || return 0
  log_header "$log" "$(basename "$bin")_${n}_r${rad}"
  for r in 1 2 3 4 5; do
    for T in 16 32 64 128 256 512 1024 2048 4096 8192; do
      [[ $T -gt $n ]] && break
      for S in 1 2 3 4; do
        run "$bin" "$n" "$T" "$S" "$rad" >> "$log" 2>&1 \
          || echo "# CMD_FAIL $bin $n $T $S $rad round=$r" >> "$log"
      done
    done
  done
}

# ---- 单配置：2D stencil（新载具；N T S R）----
sweep_sten2d() {
  local bin=$1 n=$2 rad=$3 log=$4
  need "$bin" || return 0
  log_header "$log" "$(basename "$bin")_${n}_r${rad}"
  for r in 1 2 3 4 5; do
    for T in 16 32 64 128; do
      for S in 1 2 3 4; do
        run "$bin" "$n" "$T" "$S" "$rad" >> "$log" 2>&1 \
          || echo "# CMD_FAIL $bin $n $T $S $rad round=$r" >> "$log"
      done
    done
  done
}

# ---- 整轮扫参：外层 5 遍 = R=5 ----
sweep_pass() {
  local bin=$1 log=$2; shift 2
  need "$bin" || return 0
  log_header "$log" "$(basename "$bin")"
  for r in 1 2 3 4 5; do
    run "$bin" "$@" >> "$log" 2>&1 \
      || echo "# CMD_FAIL $bin $* round=$r" >> "$log"
  done
}

# ---- 单配置：矩形 GEMM（新载具；M N K BM BN TK S）----
sweep_rect() {
  local bin=$1 m=$2 n=$3 k=$4 log=$5
  need "$bin" || return 0
  log_header "$log" "$(basename "$bin")"
  for r in 1 2 3 4 5; do
    for BM in 64 128 256; do
      for BN in 64 128 256; do
        [[ $BM -eq 256 && $BN -eq 256 ]] && continue
        for TK in 16 32 64; do
          for S in 1 2 3 4; do
            run "$bin" "$m" "$n" "$k" "$BM" "$BN" "$TK" "$S" >> "$log" 2>&1 \
              || echo "# CMD_FAIL $bin $m $n $k $BM $BN $TK $S round=$r" >> "$log"
          done
        done
      done
    done
  done
}

case "${1:-}" in
  --list)
    cat <<'EOF'
整轮：  gemm_1152 gemm_999 gemm_768 chain2_1152 chain3_1152
        gemv_mvt_1536 gemv_bicg_1536 gemv_mvt_2048
        sten2d_r1_n768（jacobi2d_768，N=768/STEPS=8）
        sibling_n12m（jacobi1d_8，STEPS=8）
单配置：sten1d_r2_n350k sten1d_r3_n600k sten1d_r1_n999989 sten1d_r2_n770k
        sten2d_r2_n400（新 2D 载具）
        rect_1024x1536x1280（sgemm_rect）
EOF
    exit 0
    ;;
  all)
    # 整轮扫参（含 sgemm/twomm/3mm）
    sweep_pass   "$BIN/sgemm_tiled"     "$LOGS/gemm_1152.log"   1152
    sweep_pass   "$BIN/sgemm_tiled"     "$LOGS/gemm_999.log"     999
    sweep_pass   "$BIN/twomm_tiled"     "$LOGS/chain2_1152.log"  1152
    sweep_pass   "$BIN/3mm_tiled"       "$LOGS/chain3_1152.log"  1152
    sweep_pass   "$BIN/mvt_tiled"       "$LOGS/gemv_mvt_1536.log"  1536
    sweep_pass   "$BIN/bicg_tiled"      "$LOGS/gemv_bicg_1536.log" 1536
    sweep_pass   "$BIN/jacobi2d_768"    "$LOGS/sten2d_r1_n768.log"
    sweep_pass   "$BIN/jacobi1d_8"      "$LOGS/sibling_n12m.log"   12000000
    # 单配置（新载具）
    sweep_sten1d "$BIN/stencil1d_tiled" 350000 2 "$LOGS/sten1d_r2_n350k.log"
    sweep_sten1d "$BIN/stencil1d_tiled" 600000 3 "$LOGS/sten1d_r3_n600k.log"
    sweep_sten1d "$BIN/stencil1d_tiled" 999989 1 "$LOGS/sten1d_r1_n999989.log"
    sweep_sten2d "$BIN/stencil2d_tiled"    400 2 "$LOGS/sten2d_r2_n400.log"
    # 副集
    sweep_sten1d "$BIN/stencil1d_tiled" 770000 2 "$LOGS/sten1d_r2_n770k.log"
    sweep_pass   "$BIN/sgemm_tiled"     "$LOGS/gemm_768.log"     768
    sweep_pass   "$BIN/mvt_tiled"       "$LOGS/gemv_mvt_2048.log"  2048
    ;;
  gemm_1152)   sweep_pass "$BIN/sgemm_tiled" "$LOGS/gemm_1152.log" 1152 ;;
  gemm_999)    sweep_pass "$BIN/sgemm_tiled" "$LOGS/gemm_999.log"   999 ;;
  gemm_768)    sweep_pass "$BIN/sgemm_tiled" "$LOGS/gemm_768.log"   768 ;;
  chain2_1152) sweep_pass "$BIN/twomm_tiled" "$LOGS/chain2_1152.log" 1152 ;;
  chain3_1152) sweep_pass "$BIN/3mm_tiled"   "$LOGS/chain3_1152.log" 1152 ;;
  sten1d_r2_n350k)   sweep_sten1d "$BIN/stencil1d_tiled" 350000 2 "$LOGS/sten1d_r2_n350k.log" ;;
  sten1d_r3_n600k)   sweep_sten1d "$BIN/stencil1d_tiled" 600000 3 "$LOGS/sten1d_r3_n600k.log" ;;
  sten1d_r1_n999989) sweep_sten1d "$BIN/stencil1d_tiled" 999989 1 "$LOGS/sten1d_r1_n999989.log" ;;
  sten1d_r2_n770k)   sweep_sten1d "$BIN/stencil1d_tiled" 770000 2 "$LOGS/sten1d_r2_n770k.log" ;;
  sten2d_r2_n400)    sweep_sten2d "$BIN/stencil2d_tiled" 400 2 "$LOGS/sten2d_r2_n400.log" ;;
  gemv_mvt_1536)     sweep_pass "$BIN/mvt_tiled"  "$LOGS/gemv_mvt_1536.log"  1536 ;;
  gemv_bicg_1536)    sweep_pass "$BIN/bicg_tiled" "$LOGS/gemv_bicg_1536.log" 1536 ;;
  gemv_mvt_2048)     sweep_pass "$BIN/mvt_tiled"  "$LOGS/gemv_mvt_2048.log"  2048 ;;
  sten2d_r1_n768)    sweep_pass "$BIN/jacobi2d_768"   "$LOGS/sten2d_r1_n768.log" ;;
  sibling_n12m)      sweep_pass "$BIN/jacobi1d_8" "$LOGS/sibling_n12m.log" 12000000 ;;
  rect_1024x1536x1280)
    sweep_rect "$BIN/sgemm_rect" 1024 1536 1280 "$LOGS/gemm_rect_1024x1536x1280.log" ;;
  *)
    echo "用法：bash holdout_sweep.sh --list | all | <kernel>" >&2
    exit 1
    ;;
esac

echo "[done] logs: $LOGS"
