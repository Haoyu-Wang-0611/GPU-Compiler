module {
  func.func @gemm500(%A: memref<500x500xf32>, %B: memref<500x500xf32>, %C: memref<500x500xf32>) {
    %c0 = arith.constant 0 : index
    %c500 = arith.constant 500 : index
    affine.for %i = %c0 to %c500 {
      affine.for %j = %c0 to %c500 {
        affine.for %k = %c0 to %c500 {
          %a = affine.load %A[%i, %k] : memref<500x500xf32>
          %b = affine.load %B[%k, %j] : memref<500x500xf32>
          %c = affine.load %C[%i, %j] : memref<500x500xf32>
          %m = arith.mulf %a, %b : f32
          %s = arith.addf %c, %m : f32
          affine.store %s, %C[%i, %j] : memref<500x500xf32>
        }
      }
    }
    return
  }
}
