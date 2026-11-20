// S3 sten1d_r1_n999989 — holdout spec v2；素数 N=999989，非整除
// 生成器：holdout_gen v1.0 (2026-11-20)；seed=20261115；生成一次即冻结，批内不得重生成

module {
  func.func @sten1d_r1_n999989(%A: memref<999989xf32>, %B: memref<999989xf32>) {
    %c1 = arith.constant 1 : index
    %c999988 = arith.constant 999988 : index
    %cf = arith.constant 0.25 : f32
    affine.for %i = %c1 to %c999988 {
      %a0 = affine.load %A[%i - 1] : memref<999989xf32>
      %a1 = affine.load %A[%i] : memref<999989xf32>
      %a2 = affine.load %A[%i + 1] : memref<999989xf32>
      %s1 = arith.addf %a0, %a1 : f32
      %s2 = arith.addf %s1, %a2 : f32
      %res = arith.mulf %s2, %cf : f32
      affine.store %res, %B[%i] : memref<999989xf32>
    }
    return
  }
}
