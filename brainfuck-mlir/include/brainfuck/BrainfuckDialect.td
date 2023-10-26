#ifndef BRAINFUCK_DIALECT
#define BRAINFUCK_DIALECT

include "mlir/IR/OpBase.td"




def Brainfuck_Dialect: Dialect {
    let name = "brainfuck";
    let summary = "Brainfuck MLIR dialect";
    let description = [{
       None
    }];
    let cppNamespace = "::mlir::lightning::brainfuck";


    let useDefaultTypePrinterParser = 1;

}



#endif //