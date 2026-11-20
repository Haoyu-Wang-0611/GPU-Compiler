// R2 gemm_999 — holdout spec v2；真非整除：999 不是 64/128 的倍数；128×128 恰好 64 块
// 生成器：holdout_gen v1.0 (2026-11-20)；seed=20261115；生成一次即冻结，批内不得重生成

module {
  func.func @gemm_999(%A: memref<999x999xf32>, %B: memref<999x999xf32>, %C: memref<999x999xf32>) {
    affine.for %i = 0 to 999 {
      affine.for %j = 0 to 999 {
        affine.for %k = 0 to 999 {
          %a = affine.load %A[%i, %k] : memref<999x999xf32>
          %b = affine.load %B[%k, %j] : memref<999x999xf32>
          %c = affine.load %C[%i, %j] : memref<999x999xf32>
          %p = arith.mulf %a, %b : f32
          %s = arith.addf %p, %c : f32
          affine.store %s, %C[%i, %j] : memref<999x999xf32>
        }
      }
    }
    return
  }
}
