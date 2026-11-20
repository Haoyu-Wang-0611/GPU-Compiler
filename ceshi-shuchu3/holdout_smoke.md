# 留出集冒烟校验（C++ v1.8.5 vs Python A1）

> 生成：`python tools/check_holdout_l1.py --l1-dir /home/user24/haoyu/scratch/holdout_2026-11-20/l1_holdout --write`

| id | name | 期望(A1) | 实际 | detected/feas | ok |
|---|---|---|---|---|---|
| S1 | sten1d_r2_n350k | 256/3 | 256/3 | stencil (no reduction dim; spatial: L0 ) | ✓ |
| S2 | sten1d_r3_n600k | 1024/1 | 1024/1 | stencil (no reduction dim; spatial: L0 ) | ✓ |
| S3 | sten1d_r1_n999989 | 1024/1 | 1024/1 | stencil (no reduction dim; spatial: L0 ) | ✓ |
| S4 | sten2d_r1_n768 | 32/4 | 32/4 | stencil (no reduction dim; spatial: L1 L2 ) | ✓ |
| S5 | sten2d_r2_n400 | 16/4 | 16/4 | stencil (no reduction dim; spatial: L1 L2 ) | ✓ |
| S6 | sibling_n12m | 4096/1 | 4096/1 | stencil (no reduction dim; spatial: L1 L2 ) | ✓ |
| R1 | gemm_1152 | 128/128/16/4 | 128/128/16/4 | matmul (reduction dim: L2 )；feas 40/96 vs 40/96 | ✓ |
| R2 | gemm_999 | 64/128/32/3 | 64/128/32/3 | matmul (reduction dim: L2 )；feas 40/96 vs 40/96 | ✓ |
| R3 | chain2_1152 | 128/128/16/4 | 128/128/16/4 | matmul (reduction dim: L2 L5 )；feas 40/96 vs 40/96 | ✓ |
| R4 | chain3_1152 | 128/128/16/4 | 128/128/16/4 | matmul (reduction dim: L2 L5 L8 )；feas 40/96 vs 40/96 | ✓ |
| R5 | gemv_mvt_1536 | 16/64/1 | 16/64/1 | matmul (reduction dim: L1 L3 )；feas 24/72 vs 24/72 | ✓ |
| R6 | gemv_bicg_1536 | 16/64/1 | 16/64/1 | matmul (reduction dim: L1 L2 )；feas 24/72 vs 24/72 | ✓ |
| S7 | sten1d_r2_n770k | 1024/1 | 1024/1 | stencil (no reduction dim; spatial: L0 ) | ✓ |
| R7 | gemm_768 | 64/128/32/3 | 64/128/32/3 | matmul (reduction dim: L2 )；feas 26/96 vs 26/96 | ✓ |
| R8 | gemv_mvt_2048 | 16/64/1 | 16/64/1 | matmul (reduction dim: L1 L3 )；feas 36/72 vs 36/72 | ✓ |
| R9 | gemm_rect_1024x1536x1280 | 128/128/16/4 | 128/128/16/4 | matmul (reduction dim: L2 )；feas 40/96 vs 40/96 | ✓ |

**汇总**：16 个，匹配 16，不匹配 0。
