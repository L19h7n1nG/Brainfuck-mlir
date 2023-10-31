#ifndef BRAINFUCK_INTERFACE
#define BRAINFUCK_INTERFACE



include "mlir/IR/OpBase.td"


def CellLikeTypeInterface: TypeInterface<"CellLikeType"> {
    let cppNamespace = "mlir::lightning::brainfuck";
    let methods = [
        InterfaceMethod<[{}],
            "int" ,"getPos">,
        InterfaceMethod<[{}],
            "int", "getVal">,
        InterfaceMethod<[{}],
            "llvm::ArrayRef<int>", "getCell">,
    ];

    let extraClassDeclaration = [{
        static constexpr int kDynamic = 0x3f3f3f3f;
        static constexpr bool isDynamic(int dValue) {
            return dValue = kDynamic;
        }
        static bool isDynamicCell(llvm::ArrayRef<int> cell) {
            return any_of(cell, [](int val) { return isDynamic(val); } );
        }
    }];

    let extraSharedClassDeclaration = [{
        bool isDynamicIdx(unsigned idx) const {
            return mlir::lightning::brainfuck::CellLikeType::isDynamic($_type.getCell()[idx]);
        }
        bool isStaticCell() const {
            return !mlir::lightning::brainfuck::CellLikeType::isDynamicCell($_type.getCell());
        }
    }];
}




#endif //