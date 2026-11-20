// S6 sibling_n12m — holdout spec v2；jacobi-1d 形：N=12M，双数组 96MB > L2 → DRAM 流区
// 生成器：holdout_gen v1.0 (2026-11-20)；seed=20261115；生成一次即冻结，批内不得重生成

module {
  func.func @sibling_n12m(%A: memref<12000000xf32>, %B: memref<12000000xf32>) {
    %c1 = arith.constant 1 : index
    %cT = arith.constant 8 : index
    %c11999999 = arith.constant 11999999 : index
    %cst = arith.constant 0.333333 : f32
    affine.for %t = %c1 to %cT {
      affine.for %i = %c1 to %c11999999 {
        %a0 = affine.load %A[%i - 1] : memref<12000000xf32>
        %a1 = affine.load %A[%i] : memref<12000000xf32>
        %a2 = affine.load %A[%i + 1] : memref<12000000xf32>
        %s1 = arith.addf %a0, %a1 : f32
        %s2 = arith.addf %s1, %a2 : f32
        %r = arith.mulf %s2, %cst : f32
        affine.store %r, %B[%i] : memref<12000000xf32>
      }
      affine.for %i = %c1 to %c11999999 {
        %b0 = affine.load %B[%i - 1] : memref<12000000xf32>
        %b1 = affine.load %B[%i] : memref<12000000xf32>
        %b2 = affine.load %B[%i + 1] : memref<12000000xf32>
        %s1 = arith.addf %b0, %b1 : f32
        %s2 = arith.addf %s1, %b2 : f32
        %r = arith.mulf %s2, %cst : f32
        affine.store %r, %A[%i] : memref<12000000xf32>
      }
    }
    return
  }
}
