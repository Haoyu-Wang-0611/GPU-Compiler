// atax 大尺寸版（11-10，oracle 用：N=8192，A=256MB >> L2，2 pass GEMV）——重用类：atax（tmp = A·x；y = A^T·tmp），M=N=8192 常数化
module {
  func.func @atax(%A: memref<8192x8192xf32>, %x: memref<8192xf32>, %y: memref<8192xf32>, %tmp: memref<8192xf32>) {
    %c0 = arith.constant 0 : index
    %c8192 = arith.constant 8192 : index
    %cz = arith.constant 0.0 : f32
    affine.for %i = %c0 to %c8192 {
      affine.store %cz, %tmp[%i] : memref<8192xf32>
      affine.for %j = %c0 to %c8192 {
        %a = affine.load %A[%i, %j] : memref<8192x8192xf32>
        %xv = affine.load %x[%j] : memref<8192xf32>
        %tv = affine.load %tmp[%i] : memref<8192xf32>
        %m = arith.mulf %a, %xv : f32
        %s = arith.addf %tv, %m : f32
        affine.store %s, %tmp[%i] : memref<8192xf32>
      }
    }
    affine.for %j = %c0 to %c8192 {
      affine.store %cz, %y[%j] : memref<8192xf32>
      affine.for %i = %c0 to %c8192 {
        %a = affine.load %A[%i, %j] : memref<8192x8192xf32>
        %tv = affine.load %tmp[%i] : memref<8192xf32>
        %yv = affine.load %y[%j] : memref<8192xf32>
        %m = arith.mulf %a, %tv : f32
        %s = arith.addf %yv, %m : f32
        affine.store %s, %y[%j] : memref<8192xf32>
      }
    }
    return
  }
}
