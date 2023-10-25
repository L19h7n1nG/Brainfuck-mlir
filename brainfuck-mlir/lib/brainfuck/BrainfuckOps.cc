#include "brainfuck/BrainfuckOps.h"
#include "brainfuck/BrainfuckDialect.h"

#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "brainfuck/BrainfuckOps.cpp.inc"
