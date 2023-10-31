#include "brainfuck/BrainfuckOps.h"
#include <sys/types.h>
#include <cstdint>
#include <cstdio>
#include "brainfuck/BrainfuckDialect.h"
#include "brainfuck/BrainfuckTypeInterfaces.h"
#include "brainfuck/BrainfuckTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
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

CellType CellType::get(
        ::mlir::MLIRContext* context,
        llvm::StringRef pos,
        llvm::StringRef val) {
    int inner_pos = 0, inner_val = 0;

    if (pos.compare("unknown"))
        inner_pos = CellLikeType::kDynamic;
    else
        pos.getAsInteger(10, inner_pos);

    if (val.compare("unknown"))
        inner_val = CellLikeType::kDynamic;
    else
        val.getAsInteger(10, inner_val);

    return CellType::get(context, inner_pos, inner_val);
}

void CellOp::build(
        ::mlir::OpBuilder& odsBuilder,
        ::mlir::OperationState& odsState,
        int pos,
        int val) {
    auto cell_type = CellType::get(odsBuilder.getContext(), pos, val);
    odsState.addTypes(cell_type);
    odsState.addAttribute("pos", odsBuilder.getSI32IntegerAttr(pos));
    odsState.addAttribute("val", odsBuilder.getSI32IntegerAttr(val));
}

llvm::ArrayRef<int> CellType::getCell() const {
    return llvm::ArrayRef<int>{getPos(), getVal()};
}