// 3mm 探针（11-12，A3 探测排序）——官方 PolyBench 4.2 形态：三个独立 (i,j,k) 嵌套，
// E/F 清零语句在 (i,j) 层（k 外），G = E·F。
// 结构：每嵌套 = out(i)-out(j)-red(k)，与 twomm 同构 → 组化分支 P=3。
// 【预注册】Best: BM=64 BN=64 TK=32 Stages=4，predicted = 3×3000 + 3×38693.31
//   = 1.250799e+05 ns（grouped pass-chain GEMM x3，每 pass 与 gemm512 逐位同构）；
//   TFLOPS 6.438333e+00（与 gemm512/twomm 同比）。
// 仅测决策层结构，不执行。
module {
  func.func @three_mm(%A: memref<512x512xf32>, %B: memref<512x512xf32>,
                      %C: memref<512x512xf32>, %D: memref<512x512xf32>,
                      %E: memref<512x512xf32>, %F: memref<512x512xf32>,
                      %G: memref<512x512xf32>) {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %cz = arith.constant 0.0 : f32
    // pass1: E = A·B（清零在 (i,j) 层）
    affine.for %i = %c0 to %c512 {
      affine.for %j = %c0 to %c512 {
        affine.store %cz, %E[%i, %j] : memref<512x512xf32>
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
    // pass2: F = C·D（清零在 (i,j) 层）
    affine.for %i = %c0 to %c512 {
      affine.for %j = %c0 to %c512 {
        affine.store %cz, %F[%i, %j] : memref<512x512xf32>
        affine.for %k = %c0 to %c512 {
          %c = affine.load %C[%i, %k] : memref<512x512xf32>
          %d = affine.load %D[%k, %j] : memref<512x512xf32>
          %f = affine.load %F[%i, %j] : memref<512x512xf32>
          %m = arith.mulf %c, %d : f32
          %s = arith.addf %f, %m : f32
          affine.store %s, %F[%i, %j] : memref<512x512xf32>
        }
      }
    }
    // pass3: G = E·F（清零在 (i,j) 层）
    affine.for %i = %c0 to %c512 {
      affine.for %j = %c0 to %c512 {
        affine.store %cz, %G[%i, %j] : memref<512x512xf32>
        affine.for %k = %c0 to %c512 {
          %e = affine.load %E[%i, %k] : memref<512x512xf32>
          %f = affine.load %F[%k, %j] : memref<512x512xf32>
          %g = affine.load %G[%i, %j] : memref<512x512xf32>
          %m = arith.mulf %e, %f : f32
          %s = arith.addf %g, %m : f32
          affine.store %s, %G[%i, %j] : memref<512x512xf32>
        }
      }
    }
    return
  }
}
