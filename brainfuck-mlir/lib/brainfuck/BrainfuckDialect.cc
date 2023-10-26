#include "brainfuck/BrainfuckDialect.h"
#include "brainfuck/BrainfuckOps.h"
#include "brainfuck/BrainfuckTypes.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/TypeID.h"

#include "llvm/ADT/TypeSwitch.h"

#include "brainfuck/BrainfuckOpsDialect.cpp.inc"

using namespace mlir::lightning::brainfuck;
void BrainfuckDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "brainfuck/BrainfuckOpsTypes.cpp.inc"
            >();
    addOperations<
#define GET_OP_LIST
#include "brainfuck/BrainfuckOps.cpp.inc"
            >();
}

#define GET_OP_CLASSES
#include "brainfuck/BrainfuckOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "brainfuck/BrainfuckOpsTypes.cpp.inc"
