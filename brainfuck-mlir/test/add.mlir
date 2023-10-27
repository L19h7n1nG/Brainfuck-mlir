module {
  func.func @test_ptr(%arg0: !brainfuck.ptr, %arg1: i32) -> i32 {

    %1 = brainfuck.shr %arg0, %arg1: (!brainfuck.ptr, i32) -> !brainfuck.ptr

    return %arg1: i32
  } 

  func.func @test_cell(%arg0: !brainfuck.cell, %arg1: i32) -> !brainfuck.cell {
    
    
      %1 = brainfuck.add %arg0, %arg1 : (!brainfuck.cell, i32) -> !brainfuck.cell
      %2 = brainfuck.add %1, %arg1 : (!brainfuck.cell, i32) -> !brainfuck.cell

    return %arg0 : !brainfuck.cell
  }
}