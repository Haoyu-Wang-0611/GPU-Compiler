// heat1d 大尺寸输入 v2（D1，11-14）——N = 2^24（16777216 元素，双数组 128 MB）
// 【选 N=2^24 的理由】N=2^20 时工作集 8 MB 完全装入 4090 的 72 MB L2 →
//   实验跑在 L2 区，守卫（DRAM 流区标定）被证伪 3.3×（11-14 发现，已登记）；
//   N=2^24 工作集 128 MB >> L2 → 真 DRAM 流区 = 守卫的标定 regime，
//   也是 11-08 regime 表的"DRAM 流式 ≥2^24"边界——主表第 4 行的正确口径。
// 【2^20 证伪数据保留】本身就是 regime-aware 叙事的实验证据（L2 区最优参数
//   结构与 DRAM 区不同：大 tile 更优），论文可用。
// 预注册（定性）：DRAM 流区决策预期 Tile 中等偏大、S=1；绝对值以实测为准。
func.func @heat1d(%A: memref<16777216xf32>, %B: memref<16777216xf32>) {
  %cf = arith.constant 0.25 : f32
  affine.for %i = 1 to 16777215 {
    %a = affine.load %A[%i - 1] : memref<16777216xf32>
    %b = affine.load %A[%i]     : memref<16777216xf32>
    %c = affine.load %A[%i + 1] : memref<16777216xf32>
    %s1 = arith.addf %a, %c : f32
    %s2 = arith.addf %s1, %b : f32
    %r = arith.mulf %s2, %cf : f32
    affine.store %r, %B[%i] : memref<16777216xf32>
  }
  return
}
