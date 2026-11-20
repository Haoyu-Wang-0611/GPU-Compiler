module {
  func.func @probe(%A: memref<512xf32>, %lb: index, %ub: index) {
    %c1 = arith.constant 1 : index
    scf.for %k = %lb to %ub step %c1 {
      %a = memref.load %A[%k] : memref<512xf32>
      %s = arith.mulf %a, %a : f32
      memref.store %s, %A[%k] : memref<512xf32>
    }
    return
  }
}
