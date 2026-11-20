// syrk 探针（11-14，A3）——官方 PolyBench syrk 守卫近似形态：
//   for i { for j { for k<j（守卫）: C[i,j] += A[i,k]·A[j,k] } }
//   （官方为 dot 标量 + C 写在 k 外——affine 不可表达标量 loop-carried，
//     取守卫 + C 累加近似；登记近似：k 内多一次 C load/store 访存）
// 结构：单嵌套 (i,j,k)，写 C[i,j] → i/j = out，读 A[i,k]/A[j,k]（k 进读不进写）
//   → k = 归约环（单归约）→ 单 GEMM 分支（v1.7 同段过滤）。
// 【模型自适应点（预注册）】A 双读合并 → A 窗口 = max(BM,BN)×TK（行带并集，
//   v1.8.2 max-tile 语义）→ 流式流量只有 gemm512 的一半（只流 A，无 B）→
//   copyB 减半 → Lcap 支配解除 → **TK=64 反超 TK=32**。
// 【预注册】Best: BM=64 BN=64 TK=64 Stages=4, predicted ≈ 2.3514e+04 ns
//   （与 gemm512 的 (64,64,32,4)/4.17e4 不同 = 模型对流量结构的自适应展示）；
//   D4 告警预期触发（A dim0 双载体在 BM≠BN 候选上异值）。
// 仅测决策层结构，不执行（C 未清零）。
// 注意：下行 #lt 为 MLIR 属性定义（合法语法），不是 # 注释。
#lt = affine_set<(d0, d1) : (d1 - d0 - 1 >= 0)>   // d0=k, d1=j, 即 k < j
module {
  func.func @syrk(%A: memref<512x512xf32>, %C: memref<512x512xf32>) {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    affine.for %i = %c0 to %c512 {
      affine.for %j = %c0 to %c512 {
        affine.for %k = %c0 to %c512 {
          affine.if #lt(%k, %j) {
            %a1 = affine.load %A[%i, %k] : memref<512x512xf32>
            %a2 = affine.load %A[%j, %k] : memref<512x512xf32>
            %c = affine.load %C[%i, %j] : memref<512x512xf32>
            %m = arith.mulf %a1, %a2 : f32
            %s = arith.addf %c, %m : f32
            affine.store %s, %C[%i, %j] : memref<512x512xf32>
          }
        }
      }
    }
    return
  }
}
