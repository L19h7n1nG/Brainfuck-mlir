#include <iostream>
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
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "brainfuck/BrainfuckDialect.h"

template <typename... Args>
void regisrtyDialects(mlir::DialectRegistry& registry) {
    registry.insert<Args...>();
}

int main(int argc, char** argv) {
    // mlir::registerAllPasses();

    mlir::DialectRegistry registry;

    regisrtyDialects<mlir::lightning::brainfuck::BrainfuckDialect>(registry);

    return mlir::asMainReturnCode(mlir::MlirOptMain(
            argc, argv, "Brainfuck optimizer driver\n", registry));
}
