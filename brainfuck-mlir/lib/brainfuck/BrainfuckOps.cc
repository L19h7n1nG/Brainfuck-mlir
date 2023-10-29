#include "brainfuck/BrainfuckOps.h"
#include <cstdio>
#include "brainfuck/BrainfuckDialect.h"
#include "brainfuck/BrainfuckTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
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
        ::mlir::OpBuilder& odsBuilder,
        ::mlir::OperationState& odsState,
        int pos,
        int val) {
    auto cell_type = CellType::get(odsBuilder.getContext(), pos, val);
    odsState.addTypes(cell_type);
    odsState.addAttribute("pos", odsBuilder.getI32IntegerAttr(pos));
    odsState.addAttribute("val", odsBuilder.getI32IntegerAttr(val));
}

void AddOp::build(
        ::mlir::OpBuilder& odsBuilder,
        ::mlir::OperationState& odsState,
        mlir::Value lhs,
        mlir::Value rhs) {
    if (lhs.getType().isa<CellType>() && rhs.getType().isa<IntegerType>()) {
        auto cur_type = lhs.getType().cast<CellType>();
        rhs.getDefiningOp()->getResult(0);
        auto new_type = CellType::get(
                cur_type.getContext(),
                cur_type.getPos(),
                cur_type.getVal() +
                        llvm::cast<arith::ConstantIntOp>(rhs.getDefiningOp())
                                .value());

        odsState.addTypes(new_type);
        odsState.addOperands({lhs, rhs});
    } else {
        llvm_unreachable("");
    }
}

void ShrOp::build(
        ::mlir::OpBuilder& odsBuilder,
        ::mlir::OperationState& odsState,
        mlir::Value lhs,
        mlir::Value rhs) {
    if (lhs.getType().isa<CellType>() && rhs.getType().isa<IntegerType>()) {
        auto cur_type = lhs.getType().cast<CellType>();
        rhs.getDefiningOp()->getResult(0);
        auto new_type = CellType::get(
                cur_type.getContext(),
                cur_type.getPos() +
                        llvm::cast<arith::ConstantIntOp>(rhs.getDefiningOp())
                                .value(),
                cur_type.getVal());

        odsState.addTypes(new_type);
        odsState.addOperands({lhs, rhs});
    } else {
        llvm_unreachable("");
    }
}