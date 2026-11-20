// R7 gemm_768 — holdout spec v2；小尺寸（仅 64×64 过半机）；与 R4 同尺寸不同类
// 生成器：holdout_gen v1.0 (2026-11-20)；seed=20261115；生成一次即冻结，批内不得重生成

module {
  func.func @gemm_768(%A: memref<768x768xf32>, %B: memref<768x768xf32>, %C: memref<768x768xf32>) {
    affine.for %i = 0 to 768 {
      affine.for %j = 0 to 768 {
        affine.for %k = 0 to 768 {
          %a = affine.load %A[%i, %k] : memref<768x768xf32>
          %b = affine.load %B[%k, %j] : memref<768x768xf32>
          %c = affine.load %C[%i, %j] : memref<768x768xf32>
          %p = arith.mulf %a, %b : f32
          %s = arith.addf %p, %c : f32
          affine.store %s, %C[%i, %j] : memref<768x768xf32>
        }
      }
    }
    return
  }
}
