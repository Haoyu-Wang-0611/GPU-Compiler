// S5 sten2d_r2_n400 — holdout spec v2；2D 9 点；载具未就绪时用 S7 替换
// 生成器：holdout_gen v1.0 (2026-11-20)；seed=20261115；生成一次即冻结，批内不得重生成

module {
  func.func @sten2d_r2_n400(%A: memref<400x400xf32>, %B: memref<400x400xf32>) {
    %c0 = arith.constant 0 : index
    %cT = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c398 = arith.constant 398 : index
    %cst = arith.constant 0.2 : f32
    affine.for %t = %c0 to %cT {
      affine.for %i = %c2 to %c398 {
        affine.for %j = %c2 to %c398 {
          %a00 = affine.load %A[%i - 2, %j - 2] : memref<400x400xf32>
          %a01 = affine.load %A[%i - 2, %j - 1] : memref<400x400xf32>
          %a02 = affine.load %A[%i - 2, %j] : memref<400x400xf32>
          %a03 = affine.load %A[%i - 2, %j + 1] : memref<400x400xf32>
          %a04 = affine.load %A[%i - 2, %j + 2] : memref<400x400xf32>
          %a10 = affine.load %A[%i - 1, %j - 2] : memref<400x400xf32>
          %a11 = affine.load %A[%i - 1, %j - 1] : memref<400x400xf32>
          %a12 = affine.load %A[%i - 1, %j] : memref<400x400xf32>
          %a13 = affine.load %A[%i - 1, %j + 1] : memref<400x400xf32>
          %a14 = affine.load %A[%i - 1, %j + 2] : memref<400x400xf32>
          %a20 = affine.load %A[%i, %j - 2] : memref<400x400xf32>
          %a21 = affine.load %A[%i, %j - 1] : memref<400x400xf32>
          %a22 = affine.load %A[%i, %j] : memref<400x400xf32>
          %a23 = affine.load %A[%i, %j + 1] : memref<400x400xf32>
          %a24 = affine.load %A[%i, %j + 2] : memref<400x400xf32>
          %a30 = affine.load %A[%i + 1, %j - 2] : memref<400x400xf32>
          %a31 = affine.load %A[%i + 1, %j - 1] : memref<400x400xf32>
          %a32 = affine.load %A[%i + 1, %j] : memref<400x400xf32>
          %a33 = affine.load %A[%i + 1, %j + 1] : memref<400x400xf32>
          %a34 = affine.load %A[%i + 1, %j + 2] : memref<400x400xf32>
          %a40 = affine.load %A[%i + 2, %j - 2] : memref<400x400xf32>
          %a41 = affine.load %A[%i + 2, %j - 1] : memref<400x400xf32>
          %a42 = affine.load %A[%i + 2, %j] : memref<400x400xf32>
          %a43 = affine.load %A[%i + 2, %j + 1] : memref<400x400xf32>
          %a44 = affine.load %A[%i + 2, %j + 2] : memref<400x400xf32>
          %s1 = arith.addf %a00, %a01 : f32
          %s2 = arith.addf %s1, %a02 : f32
          %s3 = arith.addf %s2, %a03 : f32
          %s4 = arith.addf %s3, %a04 : f32
          %s5 = arith.addf %s4, %a10 : f32
          %s6 = arith.addf %s5, %a11 : f32
          %s7 = arith.addf %s6, %a12 : f32
          %s8 = arith.addf %s7, %a13 : f32
          %s9 = arith.addf %s8, %a14 : f32
          %s10 = arith.addf %s9, %a20 : f32
          %s11 = arith.addf %s10, %a21 : f32
          %s12 = arith.addf %s11, %a22 : f32
          %s13 = arith.addf %s12, %a23 : f32
          %s14 = arith.addf %s13, %a24 : f32
          %s15 = arith.addf %s14, %a30 : f32
          %s16 = arith.addf %s15, %a31 : f32
          %s17 = arith.addf %s16, %a32 : f32
          %s18 = arith.addf %s17, %a33 : f32
          %s19 = arith.addf %s18, %a34 : f32
          %s20 = arith.addf %s19, %a40 : f32
          %s21 = arith.addf %s20, %a41 : f32
          %s22 = arith.addf %s21, %a42 : f32
          %s23 = arith.addf %s22, %a43 : f32
          %s24 = arith.addf %s23, %a44 : f32
          %res = arith.mulf %s24, %cst : f32
          affine.store %res, %B[%i, %j] : memref<400x400xf32>
        }
      }
    }
    return
  }
}
