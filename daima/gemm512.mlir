module {
  func.func @matmul(%A: memref<512x512xf32>, %B: memref<512x512xf32>, %C: memref<512x512xf32>) {
    affine.for %i = 0 to 512 {
      affine.for %j = 0 to 512 {
        affine.for %k = 0 to 512 {
          %a = affine.load %A[%i, %k] : memref<512x512xf32>
          %b = affine.load %B[%k, %j] : memref<512x512xf32>
          %c = affine.load %C[%i, %j] : memref<512x512xf32>
          %p = arith.mulf %a, %b : f32
          %s = arith.addf %p, %c : f32
          affine.store %s, %C[%i, %j] : memref<512x512xf32>
        }
      }
    }
    return
  }
}
