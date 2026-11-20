// S4 sten2d_r1_n768 — holdout spec v2；2D 5 点；8 时间步
// 生成器：holdout_gen v1.0 (2026-11-20)；seed=20261115；生成一次即冻结，批内不得重生成

module {
  func.func @sten2d_r1_n768(%A: memref<768x768xf32>, %B: memref<768x768xf32>) {
    %c0 = arith.constant 0 : index
    %cT = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c767 = arith.constant 767 : index
    %cst = arith.constant 0.2 : f32
    affine.for %t = %c0 to %cT {
      affine.for %i = %c1 to %c767 {
        affine.for %j = %c1 to %c767 {
          %a00 = affine.load %A[%i - 1, %j - 1] : memref<768x768xf32>
          %a01 = affine.load %A[%i - 1, %j] : memref<768x768xf32>
          %a02 = affine.load %A[%i - 1, %j + 1] : memref<768x768xf32>
          %a10 = affine.load %A[%i, %j - 1] : memref<768x768xf32>
          %a11 = affine.load %A[%i, %j] : memref<768x768xf32>
          %a12 = affine.load %A[%i, %j + 1] : memref<768x768xf32>
          %a20 = affine.load %A[%i + 1, %j - 1] : memref<768x768xf32>
          %a21 = affine.load %A[%i + 1, %j] : memref<768x768xf32>
          %a22 = affine.load %A[%i + 1, %j + 1] : memref<768x768xf32>
          %s1 = arith.addf %a00, %a01 : f32
          %s2 = arith.addf %s1, %a02 : f32
          %s3 = arith.addf %s2, %a10 : f32
          %s4 = arith.addf %s3, %a11 : f32
          %s5 = arith.addf %s4, %a12 : f32
          %s6 = arith.addf %s5, %a20 : f32
          %s7 = arith.addf %s6, %a21 : f32
          %s8 = arith.addf %s7, %a22 : f32
          %res = arith.mulf %s8, %cst : f32
          affine.store %res, %B[%i, %j] : memref<768x768xf32>
        }
      }
    }
    return
  }
}
