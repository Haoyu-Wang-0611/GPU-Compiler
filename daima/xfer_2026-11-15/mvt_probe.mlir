// mvt 探针（11-12，A3 探测排序）——官方 PolyBench 4.2 形态：两个独立 (i,j) 嵌套，
// 均读 A[i][j]：pass1 x1[i] += A[i][j]·y1[j]，pass2 x2[i] += A[i][j]·y2[j]。
// 结构：每嵌套 = out(i)-red(j) → GEMV 形 → 落回 v1.1 交替分支。
// 【关键变量：半机填充守卫】numBlocks = ceil(512/T) ≥ 64 → 仅 T=8 可行
//   （T=16 → 32 块 < 64 被拒）；T=8 → L_min=1260 ns。
// 【预注册】Best: T=8 TK=64 Stages=1，predicted = 2×3000 + 2×(8×1260 + 4.44)
//   = 2.616888e+04 ns（pass-chain GEMV x2；TK=64 因 chunks×L_min 支配流式项）。
// 【登记（v1.8 待办）】若前端产出融合形态（x1/x2 同嵌套双输出，A 只读一遍），
//   分类 = 单 GEMM 分支拒因"output loops != 2"——GEMV 单遍多输出形态不支持。
// 仅测决策层结构，不执行。
module {
  func.func @mvt(%A: memref<512x512xf32>, %x1: memref<512xf32>,
                 %x2: memref<512xf32>, %y1: memref<512xf32>, %y2: memref<512xf32>) {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    // pass1: x1 = x1 + A·y1（i 外层输出，j 内层归约）
    affine.for %i = %c0 to %c512 {
      affine.for %j = %c0 to %c512 {
        %x = affine.load %x1[%i] : memref<512xf32>
        %a = affine.load %A[%i, %j] : memref<512x512xf32>
        %y = affine.load %y1[%j] : memref<512xf32>
        %m = arith.mulf %a, %y : f32
        %s = arith.addf %x, %m : f32
        affine.store %s, %x1[%i] : memref<512xf32>
      }
    }
    // pass2: x2 = x2 + A·y2（同构）
    affine.for %i = %c0 to %c512 {
      affine.for %j = %c0 to %c512 {
        %x = affine.load %x2[%i] : memref<512xf32>
        %a = affine.load %A[%i, %j] : memref<512x512xf32>
        %y = affine.load %y2[%j] : memref<512xf32>
        %m = arith.mulf %a, %y : f32
        %s = arith.addf %x, %m : f32
        affine.store %s, %x2[%i] : memref<512xf32>
      }
    }
    return
  }
}
