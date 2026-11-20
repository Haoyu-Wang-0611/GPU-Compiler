// S1 sten1d_r2_n350k — holdout spec v2；1D stencil r=2，非 2 幂；halo 由窗口法吸收
// 生成器：holdout_gen v1.0 (2026-11-20)；seed=20261115；生成一次即冻结，批内不得重生成

module {
  func.func @sten1d_r2_n350k(%A: memref<350000xf32>, %B: memref<350000xf32>) {
    %c2 = arith.constant 2 : index
    %c349998 = arith.constant 349998 : index
    %cf = arith.constant 0.25 : f32
    affine.for %i = %c2 to %c349998 {
      %a0 = affine.load %A[%i - 2] : memref<350000xf32>
      %a1 = affine.load %A[%i - 1] : memref<350000xf32>
      %a2 = affine.load %A[%i] : memref<350000xf32>
      %a3 = affine.load %A[%i + 1] : memref<350000xf32>
      %a4 = affine.load %A[%i + 2] : memref<350000xf32>
      %s1 = arith.addf %a0, %a1 : f32
      %s2 = arith.addf %s1, %a2 : f32
      %s3 = arith.addf %s2, %a3 : f32
      %s4 = arith.addf %s3, %a4 : f32
      %res = arith.mulf %s4, %cf : f32
      affine.store %res, %B[%i] : memref<350000xf32>
    }
    return
  }
}
