#ifndef BRAINFUCK_OPS
#define BRAINFUCK_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "BrainfuckDialect.td"
include "BrainfuckTypes.td"


class Brainfuck_Op<string mnemonic, list<Trait> traits = []> :
        Op<Brainfuck_Dialect, mnemonic, traits>;





def Brainfuck_AddOp : Brainfuck_Op<"add", []> {
        let summary = "Add op";
        let arguments = (ins Brainfuck_CellType:$lhs, I<32>:$rhs);
        let results = (outs Brainfuck_CellType:$output);
}



#endif //