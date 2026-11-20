// R1 gemm_1152 — holdout spec v2；新尺寸；64/128 两档 tile 都可满半机（18²/9²）
// 生成器：holdout_gen v1.0 (2026-11-20)；seed=20261115；生成一次即冻结，批内不得重生成

module {
  func.func @gemm_1152(%A: memref<1152x1152xf32>, %B: memref<1152x1152xf32>, %C: memref<1152x1152xf32>) {
    affine.for %i = 0 to 1152 {
      affine.for %j = 0 to 1152 {
        affine.for %k = 0 to 1152 {
          %a = affine.load %A[%i, %k] : memref<1152x1152xf32>
          %b = affine.load %B[%k, %j] : memref<1152x1152xf32>
          %c = affine.load %C[%i, %j] : memref<1152x1152xf32>
          %p = arith.mulf %a, %b : f32
          %s = arith.addf %p, %c : f32
          affine.store %s, %C[%i, %j] : memref<1152x1152xf32>
        }
      }
    }
    return
  }
}
