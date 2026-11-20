func.func @heat1d(%A: memref<1024xf32>, %B: memref<1024xf32>) {
  %cf = arith.constant 0.25 : f32
  affine.for %i = 1 to 1023 {
    %a = affine.load %A[%i - 1] : memref<1024xf32>
    %b = affine.load %A[%i]     : memref<1024xf32>
    %c = affine.load %A[%i + 1] : memref<1024xf32>
    %s1 = arith.addf %a, %c : f32
    %s2 = arith.addf %s1, %b : f32
    %r = arith.mulf %s2, %cf : f32
    affine.store %r, %B[%i] : memref<1024xf32>
  }
  return
}
