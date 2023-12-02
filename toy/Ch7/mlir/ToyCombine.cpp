//===- ToyCombine.cpp - Toy High Level Optimizer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the Toy dialect.
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "toy/Dialect.h"
using namespace mlir;
using namespace toy;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "ToyCombine.inc"
} // namespace

/// Fold constants.
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
    return getValue();
}

/// Fold struct constants.
OpFoldResult StructConstantOp::fold(FoldAdaptor adaptor) {
    return getValue();
}

/// Fold simple struct access operations that access into a constant.
OpFoldResult StructAccessOp::fold(FoldAdaptor adaptor) {
    auto structAttr =
            llvm::dyn_cast_if_present<mlir::ArrayAttr>(adaptor.getInput());
    if (!structAttr)
        return nullptr;

    size_t elementIndex = getIndex();
    return structAttr[elementIndex];
}

/// This is an example of a c++ rewrite pattern for the TransposeOp. It
/// optimizes the following scenario: transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
    /// We register this pattern to match every toy.transpose in the IR.
    /// The "benefit" is used by the framework to order the patterns and process
    /// them in order of profitability.
    SimplifyRedundantTranspose(mlir::MLIRContext* context)
            : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

    /// This method attempts to match a pattern and rewrite it. The rewriter
    /// argument is the orchestrator of the sequence of rewrites. The pattern is
    /// expected to interact with it to perform any changes to the IR from here.
    mlir::LogicalResult matchAndRewrite(
            TransposeOp            op,
            mlir::PatternRewriter& rewriter) const override {
        // Look through the input of the current transpose.
        mlir::Value transposeInput = op.getOperand();
        TransposeOp transposeInputOp =
                transposeInput.getDefiningOp<TransposeOp>();

        // Input defined by another transpose? If not, no match.
        if (!transposeInputOp)
            return failure();

        // Otherwise, we have a redundant transpose. Use the rewriter.
        rewriter.replaceOp(op, {transposeInputOp.getOperand()});
        return success();
    }
};

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(
        RewritePatternSet& results,
        MLIRContext*       context) {
    results.add<SimplifyRedundantTranspose>(context);
}

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(
        RewritePatternSet& results,
        MLIRContext*       context) {
    results
            .add<ReshapeReshapeOptPattern,
                 RedundantReshapeOptPattern,
                 FoldConstantReshapeOptPattern>(context);
}

struct AssociativeMatMul : public mlir::OpRewritePattern<MatMulOp> {
    AssociativeMatMul(mlir::MLIRContext* ctx)
            : mlir::OpRewritePattern<MatMulOp>(ctx, 2) {}

    mlir::LogicalResult matchAndRewrite(
            MatMulOp               op,
            mlir::PatternRewriter& rewriter) const override {
        mlir::Value MatmulLhs = op.getOperands()[0];
        mlir::Value MatmulRhs = op.getOperands()[1];
        MatMulOp    matmulLhsOp = MatmulLhs.getDefiningOp<MatMulOp>();
        if (!matmulLhsOp)
            return failure();
        auto BxC = rewriter.create<MatMulOp>(
                op.getLoc(),
                matmulLhsOp.getOperands()[1],
                MatmulRhs); // 重现创建Op
        auto AxBC = rewriter.create<MatMulOp>(
                op.getLoc(), matmulLhsOp.getOperands()[0], BxC);
        rewriter.replaceOp(op, mlir::ValueRange{AxBC});
        return success();
    }
};

void MatMulOp::getCanonicalizationPatterns(
        RewritePatternSet& results,
        MLIRContext*       context) {
    results.add<AssociativeMatMul>(context);
}