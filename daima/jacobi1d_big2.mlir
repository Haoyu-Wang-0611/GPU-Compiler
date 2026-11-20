// jacobi-1d 超大尺寸版（11-09，兄弟窗口化验证用：N=2^24，数组 64MB >> L2）（11-02，喂 L1Feas 看探测器反应）
// jacobi-1d：流式类探路（PolyBench stamper-v1.0，尺寸 N=16777216、T=10 常数化）
module {
  func.func @jacobi1d(%A: memref<16777216xf32>, %B: memref<16777216xf32>) {
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c16777215 = arith.constant 16777215 : index
    %cst = arith.constant 3.333333e-01 : f32
    affine.for %t = %c1 to %c10 {
      affine.for %i = %c1 to %c16777215 {
        %a0 = affine.load %A[%i - 1] : memref<16777216xf32>
        %a1 = affine.load %A[%i] : memref<16777216xf32>
        %a2 = affine.load %A[%i + 1] : memref<16777216xf32>
        %s1 = arith.addf %a0, %a1 : f32
        %s2 = arith.addf %s1, %a2 : f32
        %r = arith.mulf %s2, %cst : f32
        affine.store %r, %B[%i] : memref<16777216xf32>
      }
      affine.for %i = %c1 to %c16777215 {
        %b0 = affine.load %B[%i - 1] : memref<16777216xf32>
        %b1 = affine.load %B[%i] : memref<16777216xf32>
        %b2 = affine.load %B[%i + 1] : memref<16777216xf32>
        %s1 = arith.addf %b0, %b1 : f32
        %s2 = arith.addf %s1, %b2 : f32
        %r = arith.mulf %s2, %cst : f32
        affine.store %r, %A[%i] : memref<16777216xf32>
      }
    }
    return
  }
}
