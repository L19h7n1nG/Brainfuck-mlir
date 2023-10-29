#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stack>
#include <string>
#include <string_view>
#include <vector>

#include "brainfuck/BrainfuckTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
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

struct Loop {
    mlir::Block *Entry, *Body, *Exit;
};
using namespace mlir::lightning::brainfuck;

mlir::OwningOpRef<mlir::ModuleOp> parse_code(
        mlir::MLIRContext& context,
        std::string_view   code) {
    mlir::OpBuilder builder(&context);

    auto theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    auto mainFunc = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(),
            "main",
            builder.getFunctionType(std::nullopt, std::nullopt));
    theModule.push_back(mainFunc);
    auto entry = mainFunc.addEntryBlock();
    builder.setInsertionPointToEnd(entry);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(entry);

    auto cur_cell = builder.create<CellOp>(builder.getUnknownLoc(), 0, 0);
    auto const_negaone = builder.create<mlir::arith::ConstantIntOp>(
            builder.getUnknownLoc(), -1, 32);
    auto const_one = builder.create<mlir::arith::ConstantIntOp>(
            builder.getUnknownLoc(), 1, 32);

    auto const_zero = builder.create<mlir::arith::ConstantIntOp>(
            builder.getUnknownLoc(), 0, 32);
    std::stack<Loop, llvm::SmallVector<Loop>> loops;
    Loop                                      ThisLoop;

    mlir::Block* thisBlock = builder.getInsertionBlock();
    for (auto&& ch : code) {
        if (ch == '+') {
            //     builder.create<AddOp>(builder.getUnknownLoc(), cur_cell,
            //     const_one);
            cur_cell = builder.create<CellOp>(
                    builder.getUnknownLoc(),
                    cur_cell.getPos(),
                    cur_cell.getVal() + 1);

        } else if (ch == '-') {
            //     builder.create<AddOp>(
            //             builder.getUnknownLoc(), cur_cell, const_negaone);
            cur_cell = builder.create<CellOp>(
                    builder.getUnknownLoc(),
                    cur_cell.getPos(),
                    cur_cell.getVal() - 1);
        } else if (ch == '>') {
            //     builder.create<ShrOp>(builder.getUnknownLoc(), cur_cell,
            //     const_one);
            cur_cell = builder.create<CellOp>(
                    builder.getUnknownLoc(),
                    cur_cell.getPos() + 1,
                    cur_cell.getVal());
        } else if (ch == '<') {
            //     builder.create<ShrOp>(
            //             builder.getUnknownLoc(), cur_cell, const_negaone);
            cur_cell = builder.create<CellOp>(
                    builder.getUnknownLoc(),
                    cur_cell.getPos() - 1,
                    cur_cell.getVal());
        } else if (ch == '[') {
            ThisLoop.Body = builder.createBlock(thisBlock);
            builder.setInsertionPointToStart(ThisLoop.Body);


            loops.push(ThisLoop);
        } else if (ch == ']') {
            if (loops.empty()) {
                llvm::errs() << "Error\n";
                exit(0);
            }

            ThisLoop = loops.top();
            loops.pop();
        //     builder.setInsertionPointToEnd(ThisLoop.Body);
            builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
        }
    }

    if (failed(mlir::verify(theModule))) {
        theModule.emitError("module verification error");
        return nullptr;
    }
    return theModule;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        return 0;
    }
    auto              filepath = argv[1];
    std::ifstream     ifs(filepath);
    std::stringstream buffer;
    buffer << ifs.rdbuf();

    mlir::MLIRContext context;
    context.loadDialect<
            mlir::func::FuncDialect,
            mlir::lightning::brainfuck::BrainfuckDialect,
            mlir::arith::ArithDialect,
            mlir::scf::SCFDialect>();

    mlir::OwningOpRef<mlir::ModuleOp> module =
            parse_code(context, buffer.str());

    if (module)
        module->dump();

    //     mlir::DialectRegistry registry;
    //     regisrtyDialects<
    //             mlir::lightning::brainfuck::BrainfuckDialect,
    //             mlir::func::FuncDialect,
    //             mlir::arith::ArithDialect>(registry);

    //     return mlir::asMainReturnCode(mlir::MlirOptMain(
    //             argc, argv, "Brainfuck optimizer driver\n", registry));

    return 0;
}
