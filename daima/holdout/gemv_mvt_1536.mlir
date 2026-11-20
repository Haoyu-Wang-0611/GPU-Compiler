// R5 gemv_mvt_1536 — holdout spec v2；mvt 形 2 pass；T=8/16 两档过半机
// 生成器：holdout_gen v1.0 (2026-11-20)；seed=20261115；生成一次即冻结，批内不得重生成

module {
  func.func @gemv_mvt_1536(%A: memref<1536x1536xf32>, %x1: memref<1536xf32>, %x2: memref<1536xf32>, %y1: memref<1536xf32>, %y2: memref<1536xf32>) {
    %c0 = arith.constant 0 : index
    %c1536 = arith.constant 1536 : index
    affine.for %i1 = %c0 to %c1536 {
      affine.for %j1 = %c0 to %c1536 {
        %xv1 = affine.load %x1[%i1] : memref<1536xf32>
        %a1 = affine.load %A[%i1, %j1] : memref<1536x1536xf32>
        %yv1 = affine.load %y1[%j1] : memref<1536xf32>
        %m1 = arith.mulf %a1, %yv1 : f32
        %s1 = arith.addf %xv1, %m1 : f32
        affine.store %s1, %x1[%i1] : memref<1536xf32>
      }
    }
    affine.for %i2 = %c0 to %c1536 {
      affine.for %j2 = %c0 to %c1536 {
        %xv2 = affine.load %x2[%i2] : memref<1536xf32>
        %a2 = affine.load %A[%i2, %j2] : memref<1536x1536xf32>
        %yv2 = affine.load %y2[%j2] : memref<1536xf32>
        %m2 = arith.mulf %a2, %yv2 : f32
        %s2 = arith.addf %xv2, %m2 : f32
        affine.store %s2, %x2[%i2] : memref<1536xf32>
      }
    }
    return
  }
}
