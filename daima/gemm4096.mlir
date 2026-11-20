module {
  func.func @gemm4096(%A: memref<4096x4096xf32>, %B: memref<4096x4096xf32>, %C: memref<4096x4096xf32>) {
    %c0 = arith.constant 0 : index
    %c4096 = arith.constant 4096 : index
    affine.for %i = %c0 to %c4096 {
      affine.for %j = %c0 to %c4096 {
        affine.for %k = %c0 to %c4096 {
          %a = affine.load %A[%i, %k] : memref<4096x4096xf32>
          %b = affine.load %B[%k, %j] : memref<4096x4096xf32>
          %c = affine.load %C[%i, %j] : memref<4096x4096xf32>
          %m = arith.mulf %a, %b : f32
          %s = arith.addf %c, %m : f32
          affine.store %s, %C[%i, %j] : memref<4096x4096xf32>
        }
      }
    }
    return
  }
}
