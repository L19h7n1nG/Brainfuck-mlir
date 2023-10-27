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
        let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)";
}


// def Brainfuck_IncOp : Brainfuck_Op<"inc", []> {
//         let summary = "inc operator of cell";
//         let arguments = (ins Brainfuck_CellType:$lhs, I<32>:$rhs);

// }  

def Brainfuck_ShiftOp : Brainfuck_Op<"shr",[]> {
        let summary = "shifting a ptr op";
        let arguments = (ins Brainfuck_PtrType:$lhs, I<32>:$rhs);
        let results = (outs Brainfuck_PtrType:$output);
        let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)";

        let builders = [
                OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
        ];

}


#endif //