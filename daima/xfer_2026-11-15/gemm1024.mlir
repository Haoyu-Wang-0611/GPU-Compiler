// 现场演示输入（11-04）——1024³ GEMM，整除，系统从未见过
// 预期决策 ≠ gemm512 的 64×64：规模翻倍后守卫放行更大 tile，模型应选 64×128
module {
  func.func @gemm1024(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    affine.for %i = %c0 to %c1024 {
      affine.for %j = %c0 to %c1024 {
        affine.for %k = %c0 to %c1024 {
          %a = affine.load %A[%i, %k] : memref<1024x1024xf32>
          %b = affine.load %B[%k, %j] : memref<1024x1024xf32>
          %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
          %m = arith.mulf %a, %b : f32
          %s = arith.addf %c, %m : f32
          affine.store %s, %C[%i, %j] : memref<1024x1024xf32>
        }
      }
    }
    return
  }
}
