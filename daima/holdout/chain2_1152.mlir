// R3 chain2_1152 — holdout spec v2；组化链 P=2；与 R1/R4 同尺寸，链长对照
// 生成器：holdout_gen v1.0 (2026-11-20)；seed=20261115；生成一次即冻结，批内不得重生成

module {
  func.func @chain2_1152(%A: memref<1152x1152xf32>, %B: memref<1152x1152xf32>, %C: memref<1152x1152xf32>, %D: memref<1152x1152xf32>, %E: memref<1152x1152xf32>) {
    %c0 = arith.constant 0 : index
    %c1152 = arith.constant 1152 : index
    %cz = arith.constant 0.0 : f32
    affine.for %i0 = %c0 to %c1152 {
      affine.for %j0 = %c0 to %c1152 {
        affine.store %cz, %E[%i0, %j0] : memref<1152x1152xf32>
      }
    }
    affine.for %i1 = %c0 to %c1152 {
      affine.for %j1 = %c0 to %c1152 {
        affine.store %cz, %D[%i1, %j1] : memref<1152x1152xf32>
      }
    }
    affine.for %i2 = %c0 to %c1152 {
      affine.for %j2 = %c0 to %c1152 {
        affine.for %k2 = %c0 to %c1152 {
          %a2 = affine.load %A[%i2, %k2] : memref<1152x1152xf32>
          %b2 = affine.load %B[%k2, %j2] : memref<1152x1152xf32>
          %e2 = affine.load %E[%i2, %j2] : memref<1152x1152xf32>
          %m2 = arith.mulf %a2, %b2 : f32
          %s2 = arith.addf %e2, %m2 : f32
          affine.store %s2, %E[%i2, %j2] : memref<1152x1152xf32>
        }
      }
    }
    affine.for %i3 = %c0 to %c1152 {
      affine.for %j3 = %c0 to %c1152 {
        affine.for %k3 = %c0 to %c1152 {
          %e3 = affine.load %E[%i3, %k3] : memref<1152x1152xf32>
          %c3 = affine.load %C[%k3, %j3] : memref<1152x1152xf32>
          %d3 = affine.load %D[%i3, %j3] : memref<1152x1152xf32>
          %m3 = arith.mulf %e3, %c3 : f32
          %s3 = arith.addf %d3, %m3 : f32
          affine.store %s3, %D[%i3, %j3] : memref<1152x1152xf32>
        }
      }
    }
    return
  }
}
