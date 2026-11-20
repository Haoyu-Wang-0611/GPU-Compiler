// symm 真形态探针 v2（11-11 晚）——k<i 三角语义以 affine 守卫形式表达
// 【v1 教训】affine.for 上界不能用外层 IV（IV 只能作 dim 不能作 symbol，
//   "operand cannot be used as a symbol" 已证）→ 三角 trip 无直接表达，
//   惯用形式 = 常 trip 512 + affine.if (k<i) 守卫。Polygeist 产出形态
//   是常 trip + 守卫还是其他形式，待 A3/Polygeist 验证——本探针即对照基准。
// 预期：单 GEMM 分支拒因（独立清零段 → outLoops 计 4 != 2）——
//   登记发现：单模板清零段盲区；守卫浪费不进模型，登记。
// 注意：下行 #lt 为 MLIR 属性定义（合法语法），不是 # 注释。
#lt = affine_set<(d0, d1) : (d1 - d0 - 1 >= 0)>   // d0=k, d1=i, 即 k < i
module {
  func.func @symm(%A: memref<512x512xf32>, %B: memref<512x512xf32>, %C: memref<512x512xf32>) {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %cz = arith.constant 0.0 : f32
    affine.for %i = %c0 to %c512 {
      affine.for %j = %c0 to %c512 {
        affine.store %cz, %C[%i, %j] : memref<512x512xf32>
      }
    }
    affine.for %i = %c0 to %c512 {
      affine.for %j = %c0 to %c512 {
        affine.for %k = %c0 to %c512 {
          affine.if #lt(%k, %i) {
            %a = affine.load %A[%k, %i] : memref<512x512xf32>
            %b = affine.load %B[%k, %j] : memref<512x512xf32>
            %c = affine.load %C[%i, %j] : memref<512x512xf32>
            %m = arith.mulf %a, %b : f32
            %s = arith.addf %c, %m : f32
            affine.store %s, %C[%i, %j] : memref<512x512xf32>
          }
        }
      }
    }
    return
  }
}
