#include <iostream>
#include <string>
#include <string_view>
#include <vector>

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
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
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
// void mlirGen(
//         std::string_view inputs,
//         mlir::ModuleOp& themodule,
//         mlir::OpBuilder& builder) {
//     // int len = inputs.size();
//     themodule = mlir::ModuleOp::create(builder.getUnknownLoc());
//     for (auto&& e : inputs) {
//         switch (e) {
//             case '+':

//                 // builder.create<AddOp>()
//             default:
//                 break;
//         }
//     }
// }

int main(int argc, char** argv) {
    // mlir::registerAllPasses();

    // mlir::lightning::brainfuck
    mlir::DialectRegistry registry;
    regisrtyDialects<
            mlir::lightning::brainfuck::BrainfuckDialect, mlir::func::FuncDialect>(registry);

    return mlir::asMainReturnCode(mlir::MlirOptMain(
            argc, argv, "Brainfuck optimizer driver\n", registry));
}
