#include "brainfuck/BrainfuckDialect.h"
#include "brainfuck/BrainfuckOps.h"
#include "brainfuck/BrainfuckTypes.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/TypeID.h"

#include "llvm/ADT/TypeSwitch.h"





#define GET_TYPEDEF_CLASSES
#include "brainfuck/BrainfuckOpsTypes.cpp.inc"


using namespace mlir::lightning::brainfuck;





