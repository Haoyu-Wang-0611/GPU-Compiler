// R9 gemm_rect_1024x1536x1280 — holdout spec v2；矩形；载具未就绪则只做决策层
// 生成器：holdout_gen v1.0 (2026-11-20)；seed=20261115；生成一次即冻结，批内不得重生成

module {
  func.func @gemm_rect_1024x1536x1280(%A: memref<1024x1280xf32>, %B: memref<1280x1536xf32>, %C: memref<1024x1536xf32>) {
    affine.for %i = 0 to 1024 {
      affine.for %j = 0 to 1536 {
        affine.for %k = 0 to 1280 {
          %a = affine.load %A[%i, %k] : memref<1024x1280xf32>
          %b = affine.load %B[%k, %j] : memref<1280x1536xf32>
          %c = affine.load %C[%i, %j] : memref<1024x1536xf32>
          %p = arith.mulf %a, %b : f32
          %s = arith.addf %p, %c : f32
          affine.store %s, %C[%i, %j] : memref<1024x1536xf32>
        }
      }
    }
    return
  }
}
