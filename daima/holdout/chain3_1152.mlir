// R4 chain3_1152 — holdout spec v2；组化链 P=3；与 R1/R3 同尺寸，链长对照
// 生成器：holdout_gen v1.0 (2026-11-20)；seed=20261115；生成一次即冻结，批内不得重生成

module {
  func.func @chain3_1152(%A: memref<1152x1152xf32>, %B: memref<1152x1152xf32>, %C: memref<1152x1152xf32>, %D: memref<1152x1152xf32>, %E: memref<1152x1152xf32>, %F: memref<1152x1152xf32>, %G: memref<1152x1152xf32>) {
    %c0 = arith.constant 0 : index
    %c1152 = arith.constant 1152 : index
    %cz = arith.constant 0.0 : f32
    affine.for %i1 = %c0 to %c1152 {
      affine.for %j1 = %c0 to %c1152 {
        affine.store %cz, %E[%i1, %j1] : memref<1152x1152xf32>
        affine.for %k1 = %c0 to %c1152 {
          %a1 = affine.load %A[%i1, %k1] : memref<1152x1152xf32>
          %b1 = affine.load %B[%k1, %j1] : memref<1152x1152xf32>
          %c1 = affine.load %E[%i1, %j1] : memref<1152x1152xf32>
          %m1 = arith.mulf %a1, %b1 : f32
          %s1 = arith.addf %c1, %m1 : f32
          affine.store %s1, %E[%i1, %j1] : memref<1152x1152xf32>
        }
      }
    }
    affine.for %i2 = %c0 to %c1152 {
      affine.for %j2 = %c0 to %c1152 {
        affine.store %cz, %F[%i2, %j2] : memref<1152x1152xf32>
        affine.for %k2 = %c0 to %c1152 {
          %a2 = affine.load %C[%i2, %k2] : memref<1152x1152xf32>
          %b2 = affine.load %D[%k2, %j2] : memref<1152x1152xf32>
          %c2 = affine.load %F[%i2, %j2] : memref<1152x1152xf32>
          %m2 = arith.mulf %a2, %b2 : f32
          %s2 = arith.addf %c2, %m2 : f32
          affine.store %s2, %F[%i2, %j2] : memref<1152x1152xf32>
        }
      }
    }
    affine.for %i3 = %c0 to %c1152 {
      affine.for %j3 = %c0 to %c1152 {
        affine.store %cz, %G[%i3, %j3] : memref<1152x1152xf32>
        affine.for %k3 = %c0 to %c1152 {
          %a3 = affine.load %E[%i3, %k3] : memref<1152x1152xf32>
          %b3 = affine.load %F[%k3, %j3] : memref<1152x1152xf32>
          %c3 = affine.load %G[%i3, %j3] : memref<1152x1152xf32>
          %m3 = arith.mulf %a3, %b3 : f32
          %s3 = arith.addf %c3, %m3 : f32
          affine.store %s3, %G[%i3, %j3] : memref<1152x1152xf32>
        }
      }
    }
    return
  }
}
