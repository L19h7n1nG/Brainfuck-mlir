module {
  func.func @main() {
    %0 = "brainfuck.cell"() {pos = 0 : i32, val = 0 : i32} : () -> !brainfuck.cell<pos = 0, val = 0>
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %1 = "brainfuck.cell"() {pos = 0 : i32, val = 1 : i32} : () -> !brainfuck.cell<pos = 0, val = 1>
    %2 = "brainfuck.cell"() {pos = 0 : i32, val = 2 : i32} : () -> !brainfuck.cell<pos = 0, val = 2>
    %3 = "brainfuck.cell"() {pos = 0 : i32, val = 3 : i32} : () -> !brainfuck.cell<pos = 0, val = 3>
    %4 = "brainfuck.cell"() {pos = 0 : i32, val = 4 : i32} : () -> !brainfuck.cell<pos = 0, val = 4>
    %5 = "brainfuck.cell"() {pos = 0 : i32, val = 5 : i32} : () -> !brainfuck.cell<pos = 0, val = 5>
    %6 = "brainfuck.cell"() {pos = 0 : i32, val = 6 : i32} : () -> !brainfuck.cell<pos = 0, val = 6>
    %7:2 = scf.while (%arg0 = %6, %arg1 = %c0_i32) : (!brainfuck.cell<pos = 0, val = 6>, i32) -> (!brainfuck.cell<pos = 0, val = 6>, i32) {
      %c6_i32 = arith.constant 6 : i32
      %8 = arith.cmpi ne, %c6_i32, %c0_i32 : i32
      scf.condition(%8) %arg0, %arg1 : !brainfuck.cell<pos = 0, val = 6>, i32
    } do {
    ^bb0(%arg0: !brainfuck.cell<pos = 0, val = 6>, %arg1: i32):
      scf.yield %6, %c0_i32 : !brainfuck.cell<pos = 0, val = 6>, i32
    }
    return
  }
}