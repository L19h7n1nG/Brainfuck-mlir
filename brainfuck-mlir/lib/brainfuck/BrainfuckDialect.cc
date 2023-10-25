#include "brainfuck/BrainfuckDialect.h"
#include "brainfuck/BrainfuckOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/TypeID.h"
#include "brainfuck/BrainfuckOpsDialect.cpp.inc"


using namespace mlir::lightning::brainfuck;

void BrainfuckDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "brainfuck/BrainfuckOps.cpp.inc"
            >();
}

