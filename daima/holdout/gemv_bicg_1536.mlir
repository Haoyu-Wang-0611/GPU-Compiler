// R6 gemv_bicg_1536 — holdout spec v2；bicg 形 2 pass（pass1 红外/写内）；T=8/16 两档
// 生成器：holdout_gen v1.0 (2026-11-20)；seed=20261115；生成一次即冻结，批内不得重生成

module {
  func.func @gemv_bicg_1536(%A: memref<1536x1536xf32>, %s: memref<1536xf32>, %q: memref<1536xf32>, %p: memref<1536xf32>, %r: memref<1536xf32>) {
    %c0 = arith.constant 0 : index
    %c1536 = arith.constant 1536 : index
    affine.for %i1 = %c0 to %c1536 {
      affine.for %j1 = %c0 to %c1536 {
        %sv = affine.load %s[%j1] : memref<1536xf32>
        %rv = affine.load %r[%i1] : memref<1536xf32>
        %a1 = affine.load %A[%i1, %j1] : memref<1536x1536xf32>
        %m1 = arith.mulf %rv, %a1 : f32
        %s1 = arith.addf %sv, %m1 : f32
        affine.store %s1, %s[%j1] : memref<1536xf32>
      }
    }
    affine.for %i2 = %c0 to %c1536 {
      affine.for %j2 = %c0 to %c1536 {
        %qv = affine.load %q[%i2] : memref<1536xf32>
        %a2 = affine.load %A[%i2, %j2] : memref<1536x1536xf32>
        %pv = affine.load %p[%j2] : memref<1536xf32>
        %m2 = arith.mulf %a2, %pv : f32
        %s2 = arith.addf %qv, %m2 : f32
        affine.store %s2, %q[%i2] : memref<1536xf32>
      }
    }
    return
  }
}
