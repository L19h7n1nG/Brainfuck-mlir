#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "brainfuck/BrainfuckTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "brainfuck/BrainfuckDialect.h"
#include "brainfuck/BrainfuckOps.h"

template <typename... Args>
void regisrtyDialects(mlir::DialectRegistry& registry) {
    registry.insert<Args...>();
}

using namespace mlir::lightning::brainfuck;

mlir::OwningOpRef<mlir::ModuleOp> parse_code(
        mlir::MLIRContext& context,
        std::string_view   code) {
    mlir::OpBuilder builder(&context);

    auto theModule = mlir::ModuleOp::create(builder.getUnknownLoc());


    auto cur_type = mlir::lightning::brainfuck::CellType::get(&context, 0, 0);

    auto const_one = builder.create<mlir::arith::ConstantIntOp>(
            builder.getUnknownLoc(), 1, 32);
    auto const_nega_one = builder.create<mlir::arith::ConstantIntOp>(
            builder.getUnknownLoc(), -1, 32);
    auto cur_cell =
            builder.create<CellOp>(builder.getUnknownLoc(), cur_type, 0, 0);

    theModule.push_back(const_one);
    theModule.push_back(cur_cell);

    auto new_cell = builder.create<AddOp>(builder.getUnknownLoc(), cur_type, cur_cell, const_one);

    theModule.push_back(new_cell);

    if (failed(mlir::verify(theModule))) {
        theModule.emitError("module verification error");
        return nullptr;
    }
    return theModule;
}

int main(int argc, char** argv) {
    auto s =
            "++++++++++++++++++++++++>>>>>>>>>>>>>++++++++++++++++<<<<<<<<<<<<+++++++-------";
    mlir::MLIRContext context;
    context.loadDialect<
            mlir::func::FuncDialect,
            mlir::lightning::brainfuck::BrainfuckDialect,
            mlir::arith::ArithDialect>();

    mlir::OwningOpRef<mlir::ModuleOp> module = parse_code(context, s);

    if (module)
        module->dump();

    mlir::DialectRegistry registry;
    regisrtyDialects<
            mlir::lightning::brainfuck::BrainfuckDialect,
            mlir::func::FuncDialect,
            mlir::arith::ArithDialect>(registry);

    return mlir::asMainReturnCode(mlir::MlirOptMain(
            argc, argv, "Brainfuck optimizer driver\n", registry));
}
