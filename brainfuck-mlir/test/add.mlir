module {
  // func.func @test_ptr(%arg0: !brainfuck.ptr, %arg1: i32) -> i32 {

  //   %1 = brainfuck.shr %arg0, %arg1: (!brainfuck.ptr, i32) -> !brainfuck.ptr

  //   return %arg1: i32
  // } 

  func.func @main() {
     %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %1 = "brainfuck.cell"() {pos = 0 : i32, val = 0 : i32} : () -> !brainfuck.cell<pos = 0, val = 0>
  %2 = "brainfuck.add"(%1, %0) : (!brainfuck.cell<pos = 0, val = 0>, i32) -> !brainfuck.cell<pos = 0, val = 1>
      // %4 = brainfuck.add %arg0, %2 : (!brainfuck.cell<pos = 0, val = 0>, i32) -> !brainfuck.cell<pos = 0, val = 2>


    return
  }
}