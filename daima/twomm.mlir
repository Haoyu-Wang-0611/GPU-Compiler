// PolyBench 探路 kernel（11-02）——重用类：2mm 简化版（E = A·B；D = E·C），512³ 常数化
module {
  func.func @twomm(%A: memref<512x512xf32>, %B: memref<512x512xf32>, %C: memref<512x512xf32>, %D: memref<512x512xf32>, %E: memref<512x512xf32>) {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %cz = arith.constant 0.0 : f32
    affine.for %i = %c0 to %c512 {
      affine.for %j = %c0 to %c512 {
        affine.store %cz, %E[%i, %j] : memref<512x512xf32>
      }
    }
    affine.for %i = %c0 to %c512 {
      affine.for %j = %c0 to %c512 {
        affine.store %cz, %D[%i, %j] : memref<512x512xf32>
      }
    }
    affine.for %i = %c0 to %c512 {
      affine.for %j = %c0 to %c512 {
        affine.for %k = %c0 to %c512 {
          %a = affine.load %A[%i, %k] : memref<512x512xf32>
          %b = affine.load %B[%k, %j] : memref<512x512xf32>
          %e = affine.load %E[%i, %j] : memref<512x512xf32>
          %m = arith.mulf %a, %b : f32
          %s = arith.addf %e, %m : f32
          affine.store %s, %E[%i, %j] : memref<512x512xf32>
        }
      }
    }
    affine.for %i = %c0 to %c512 {
      affine.for %j = %c0 to %c512 {
        affine.for %k = %c0 to %c512 {
          %e = affine.load %E[%i, %k] : memref<512x512xf32>
          %c = affine.load %C[%k, %j] : memref<512x512xf32>
          %d = affine.load %D[%i, %j] : memref<512x512xf32>
          %m = arith.mulf %e, %c : f32
          %s = arith.addf %d, %m : f32
          affine.store %s, %D[%i, %j] : memref<512x512xf32>
        }
      }
    }
    return
  }
}
