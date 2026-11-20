// S2 sten1d_r3_n600k — holdout spec v2；1D stencil r=3；7 点读
// 生成器：holdout_gen v1.0 (2026-11-20)；seed=20261115；生成一次即冻结，批内不得重生成

module {
  func.func @sten1d_r3_n600k(%A: memref<600000xf32>, %B: memref<600000xf32>) {
    %c3 = arith.constant 3 : index
    %c599997 = arith.constant 599997 : index
    %cf = arith.constant 0.25 : f32
    affine.for %i = %c3 to %c599997 {
      %a0 = affine.load %A[%i - 3] : memref<600000xf32>
      %a1 = affine.load %A[%i - 2] : memref<600000xf32>
      %a2 = affine.load %A[%i - 1] : memref<600000xf32>
      %a3 = affine.load %A[%i] : memref<600000xf32>
      %a4 = affine.load %A[%i + 1] : memref<600000xf32>
      %a5 = affine.load %A[%i + 2] : memref<600000xf32>
      %a6 = affine.load %A[%i + 3] : memref<600000xf32>
      %s1 = arith.addf %a0, %a1 : f32
      %s2 = arith.addf %s1, %a2 : f32
      %s3 = arith.addf %s2, %a3 : f32
      %s4 = arith.addf %s3, %a4 : f32
      %s5 = arith.addf %s4, %a5 : f32
      %s6 = arith.addf %s5, %a6 : f32
      %res = arith.mulf %s6, %cf : f32
      affine.store %res, %B[%i] : memref<600000xf32>
    }
    return
  }
}
