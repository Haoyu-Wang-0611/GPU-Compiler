// symm 全 trip 对照探针（11-11，A1）——单 pass out-out-red（k 全域 512）
// 用途：正对照——验证单 GEMM 模板对"2 out + 1 red 单嵌套"照常出决策。
// 预期：与 gemm512 同构决策 (64,64,TK=32,S=4)（A 下标 [k,i] 与 [i,k]
//       窗口尺寸相同：BM×TK 流式 / TK×BN 流式 / C 非流式 ×0）。
// 仅测决策层结构，不执行（C 未清零，累加语义不完整）。
module {
  func.func @symm_full(%A: memref<512x512xf32>, %B: memref<512x512xf32>, %C: memref<512x512xf32>) {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    affine.for %i = %c0 to %c512 {
      affine.for %j = %c0 to %c512 {
        affine.for %k = %c0 to %c512 {
          %a = affine.load %A[%k, %i] : memref<512x512xf32>
          %b = affine.load %B[%k, %j] : memref<512x512xf32>
          %c = affine.load %C[%i, %j] : memref<512x512xf32>
          %m = arith.mulf %a, %b : f32
          %s = arith.addf %c, %m : f32
          affine.store %s, %C[%i, %j] : memref<512x512xf32>
        }
      }
    }
    return
  }
}
