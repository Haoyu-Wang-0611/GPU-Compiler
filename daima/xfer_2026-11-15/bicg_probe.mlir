// bicg 探针（11-12，A3 探测排序）——官方 PolyBench 4.2 形态：两个独立 i-外层嵌套。
// pass1: s[j] += r[i]·A[i][j]——【注意】j 是输出但在内层，i 是归约在外层
//   （红在外/写在内，与 atax 相反）；pass2: q[i] += A[i][j]·p[j]（标准 out 外/red 内）。
// 分类推演：反转编号后 L0=p2-i(out)/L1=p2-j(red)/L2=p1-j(out)/L3=p1-i(red)
//   → 偶=out、奇=red 交替成立 → v1.1 GEMV 交替分支。
// 【预注册】与 mvt 同构决策：Best: T=8 TK=64 Stages=1，predicted = 2.616888e+04 ns
//   （pass-chain GEMV x2；仅 T=8 过半机填充守卫，理由同 mvt 探针）。
// 仅测决策层结构，不执行（s/q 未清零，累加语义不完整）。
module {
  func.func @bicg(%A: memref<512x512xf32>, %s: memref<512xf32>,
                  %q: memref<512xf32>, %p: memref<512xf32>, %r: memref<512xf32>) {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    // pass1: s = s + r·A（j 内层输出，i 外层归约——红外/写内）
    affine.for %i = %c0 to %c512 {
      affine.for %j = %c0 to %c512 {
        %sv = affine.load %s[%j] : memref<512xf32>
        %rv = affine.load %r[%i] : memref<512xf32>
        %a = affine.load %A[%i, %j] : memref<512x512xf32>
        %m = arith.mulf %rv, %a : f32
        %s2 = arith.addf %sv, %m : f32
        affine.store %s2, %s[%j] : memref<512xf32>
      }
    }
    // pass2: q = q + A·p（i 外层输出，j 内层归约——标准形态）
    affine.for %i = %c0 to %c512 {
      affine.for %j = %c0 to %c512 {
        %qv = affine.load %q[%i] : memref<512xf32>
        %a = affine.load %A[%i, %j] : memref<512x512xf32>
        %pv = affine.load %p[%j] : memref<512xf32>
        %m = arith.mulf %a, %pv : f32
        %q2 = arith.addf %qv, %m : f32
        affine.store %q2, %q[%i] : memref<512xf32>
      }
    }
    return
  }
}
