#include "brainfuck/BrainfuckDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;
int main(int argc, char** argv) {
    DialectRegistry registry;
    registry
            .insert<func::FuncDialect,
                    mlir::lightning::brainfuck::BrainfuckDialect>();

    return failed(MlirLspServerMain(argc, argv, registry));
}