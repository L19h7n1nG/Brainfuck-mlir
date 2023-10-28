#include "brainfuck/BrainfuckOps.h"
#include <cstdio>
#include "brainfuck/BrainfuckDialect.h"
#include "brainfuck/BrainfuckTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/AsyncTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/TypeID.h"

#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "brainfuck/BrainfuckOps.cpp.inc"

using namespace mlir::lightning::brainfuck;

void CellOp::build(
        ::mlir::OpBuilder&      odsBuilder,
        ::mlir::OperationState& odsState,
        int                     pos,
        int                     val) {
    auto cell_type = CellType::get(odsBuilder.getContext(), pos, val);
    odsState.addTypes(cell_type);

    odsState.addAttribute(
            "pos", odsBuilder.getIntegerAttr(odsBuilder.getI32Type(), pos));
    odsState.addAttribute(
            "val", odsBuilder.getIntegerAttr(odsBuilder.getI32Type(), val));
}

void AddOp::build(
        ::mlir::OpBuilder&      odsBuilder,
        ::mlir::OperationState& odsState,
        mlir::Value             lhs,
        int                     rhs) {
    if (lhs.getType().isa<CellType>()) {

       
        // printf("%p: %p", lhs.getContext(), odsBuilder.getContext());
       

    } else {
        llvm_unreachable("");
    }
}

// void AddOp::build(
//         ::mlir::OpBuilder&      odsBuilder,
//         ::mlir::OperationState& odsState,
//         mlir::Type              lhs,
//         int                     rhs) {
//     if (lhs.isa<CellType>()) {
//         auto cell_lhs = lhs.cast<CellType>();
//         auto new_type = CellType::get(
//                 cell_lhs.getContext(),
//                 cell_lhs.getPos(),
//                 cell_lhs.getVal() + 1);
//         odsState.addTypes(new_type);

//         auto operands_rhs = odsBuilder.create<mlir::arith::ConstantIntOp>(
//                 odsBuilder.getUnknownLoc(), 1, 32);
//         auto operands_lhs =
//         odsBuilder.create<CellOp>(odsBuilder.getUnknownLoc(),
//         cell_lhs.getPos(), cell_lhs.getVal());

//         odsState.addOperands({operands_lhs, operands_rhs});

//     } else {
//         llvm_unreachable("Type should be CellType");
//     }
// }
