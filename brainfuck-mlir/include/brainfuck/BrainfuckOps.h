#ifndef BRAINFUCKOPS_H
#define BRAINFUCKOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"


#include "BrainfuckTypes.h"
#define GET_OP_CLASSES
#include "brainfuck/BrainfuckOps.h.inc"
#endif