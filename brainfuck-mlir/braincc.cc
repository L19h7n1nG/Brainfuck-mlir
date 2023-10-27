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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
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

    auto funcType = builder.getFunctionType(std::nullopt, std::nullopt);

    // auto mainfunction = builder.create<mlir::func::FuncOp>(
    //         builder.getUnknownLoc(), "main", funcType);

    // theModule.push_back(mainfunction);

    // auto entryBlock = mainfunction.addEntryBlock();

    // builder.setInsertionPointToEnd(entryBlock);

    // builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    // builder.setInsertionPointToEnd(entryBlock);

    for (auto&& ch : code) {
        if (ch == '<') {
            // builder.create<mlir::lightning::brainfuck::AddOp>(
            //         builder.getUnknownLoc());
            // auto lhsType = builder.create()
            // mlir::Value lhs = mlir::Value::setType(builder.getI32Type());
            mlir::Value lhs;
            mlir::Value rhs;
            auto        outType =
                    mlir::lightning::brainfuck::PtrType::get(&context, 0);
            // auto rhs =
            auto op = builder.create<mlir::lightning::brainfuck::ShiftOp>(
                    builder.getUnknownLoc(), outType, lhs, rhs);
            // theModule.push_back(op);
        }
    }

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
            mlir::lightning::brainfuck::BrainfuckDialect>();

    mlir::OwningOpRef<mlir::ModuleOp> module = parse_code(context, s);

    module->dump();

    // mlir::DialectRegistry registry;
    // regisrtyDialects<
    //         mlir::lightning::brainfuck::BrainfuckDialect,
    //         mlir::func::FuncDialect>(registry);

    // return mlir::asMainReturnCode(mlir::MlirOptMain(
    //         argc, argv, "Brainfuck optimizer driver\n", registry));
}
