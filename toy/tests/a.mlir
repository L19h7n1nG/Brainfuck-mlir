module {
  func.func @main() {
    %cst = arith.constant 0.0 : f32
    %cst_0 = arith.constant 1.300e+01 : f32
    %cst_1 = arith.constant 1.200e+01 : f32
    %cst_2 = arith.constant 1.100e+01 : f32
    %cst_3 = arith.constant 1.000e+01 : f32
    %cst_4 = arith.constant 9.000e+00 : f32
    %cst_5 = arith.constant 8.000e+00 : f32
    %cst_6 = arith.constant 7.000e+00 : f32
    %cst_7 = arith.constant 6.000e+00 : f32
    %cst_8 = arith.constant 5.000e+00 : f32
    %cst_9 = arith.constant 4.000e+00 : f32
    %cst_10 = arith.constant 3.000e+00 : f32
    %cst_11 = arith.constant 2.000e+00 : f32
   
    %cst_12 = arith.constant 1.000000e+00 : f32
    %alloc = memref.alloc() : memref<4x2xf32>
    %alloc_13 = memref.alloc() : memref<4x3xf32>
    %alloc_14 = memref.alloc() : memref<3x2xf32>
    affine.store %cst_12, %alloc_14[0, 0] : memref<3x2xf32>
    toy.print %alloc_14: memref<3x2xf32>
    // affine.store %cst_11, %alloc_14[0, 1] : memref<3x2xf32>
    // affine.store %cst_10, %alloc_14[1, 0] : memref<3x2xf32>
    // affine.store %cst_9, %alloc_14[1, 1] : memref<3x2xf32>
    // affine.store %cst_8, %alloc_14[2, 0] : memref<3x2xf32>
    // affine.store %cst_7, %alloc_14[2, 1] : memref<3x2xf32>
    // affine.store %cst_11, %alloc_13[0, 0] : memref<4x3xf32>
    // affine.store %cst_10, %alloc_13[0, 1] : memref<4x3xf32>
    // affine.store %cst_9, %alloc_13[0, 2] : memref<4x3xf32>
    // affine.store %cst_8, %alloc_13[1, 0] : memref<4x3xf32>
    // affine.store %cst_7, %alloc_13[1, 1] : memref<4x3xf32>
    // affine.store %cst_6, %alloc_13[1, 2] : memref<4x3xf32>
    // affine.store %cst_5, %alloc_13[2, 0] : memref<4x3xf32>
    // affine.store %cst_4, %alloc_13[2, 1] : memref<4x3xf32>
    // affine.store %cst_3, %alloc_13[2, 2] : memref<4x3xf32>
    // affine.store %cst_2, %alloc_13[3, 0] : memref<4x3xf32>
    // affine.store %cst_1, %alloc_13[3, 1] : memref<4x3xf32>
    // affine.store %cst_0, %alloc_13[3, 2] : memref<4x3xf32>
    // affine.for %arg0 = 0 to 4 {
    //   affine.for %arg1 = 0 to 2 {
    //     affine.store %cst, %alloc[%arg0, %arg1] : memref<4x2xf32>
    //   }
    // }
    // affine.for %arg0 = 0 to 4 {
    //   affine.for %arg1 = 0 to 3 {
    //     affine.for %arg2 = 0 to 2 {
    //       %0 = affine.load %alloc_13[%arg0, %arg1] : memref<4x3xf32>
    //       %1 = affine.load %alloc_14[%arg1, %arg2] : memref<3x2xf32>
    //       %2 = arith.mulf %0, %1 : f32
    //       %3 = affine.load %alloc[%arg0, %arg2] : memref<4x2xf32>
    //       %4 = arith.addf %2, %3 : f32
    //       affine.store %4, %alloc[%arg0, %arg2] : memref<4x2xf32>
    //     }
    //   }
    // }
    // toy.print %alloc : memref<4x2xf32>
    // memref.dealloc %alloc_14 : memref<3x2xf32>
    // memref.dealloc %alloc_13 : memref<4x3xf32>
    // memref.dealloc %alloc : memref<4x2xf32>
    return
  }
}