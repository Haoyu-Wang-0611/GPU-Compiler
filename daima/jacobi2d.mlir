// jacobi-2d 探路（PolyBench 流式类 2D；尺寸 1024×1024、时间步 16，全部常数化）
// 结构：L0 = 时间环（IV t 不进下标），L1/L2 = 空间环；读 A 五点 stencil，写 B
module {
  func.func @jacobi2d(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c1023 = arith.constant 1023 : index
    %cst = arith.constant 2.000000e-01 : f32
    affine.for %t = %c0 to %c16 {
      affine.for %i = %c1 to %c1023 {
        affine.for %j = %c1 to %c1023 {
          %a0 = affine.load %A[%i, %j] : memref<1024x1024xf32>
          %am = affine.load %A[%i - 1, %j] : memref<1024x1024xf32>
          %ap = affine.load %A[%i + 1, %j] : memref<1024x1024xf32>
          %al = affine.load %A[%i, %j - 1] : memref<1024x1024xf32>
          %ar = affine.load %A[%i, %j + 1] : memref<1024x1024xf32>
          %s1 = arith.addf %a0, %am : f32
          %s2 = arith.addf %s1, %ap : f32
          %s3 = arith.addf %s2, %al : f32
          %s4 = arith.addf %s3, %ar : f32
          %r = arith.mulf %s4, %cst : f32
          affine.store %r, %B[%i, %j] : memref<1024x1024xf32>
        }
      }
    }
    return
  }
}
