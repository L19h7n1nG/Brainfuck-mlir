#ifndef BRAINFUCK_TYPES
#define BRAINFUCK_TYPES


include "BrainfuckDialect.td"
include "BrainfuckAttributes.td"
include "mlir/IR/AttrTypeBase.td"

class Brainfuck_Type<string name, string typeMnemonic>
    : TypeDef<Brainfuck_Dialect, name> {
        let mnemonic = typeMnemonic; 
    }

// def Brainfuck_CellType: Brainfuck_Type<"Cell", "cell"> {
//     let summary = "The cell type of brainfuck";
//     let description = [{}];
// }


// def Brainfuck_PtrType: Brainfuck_Type<"Ptr", "ptr"> {
//     let summary = "The pointer type of brainfuck";
//     let description = [{}];

//     let parameters = (ins "int":$initParam);

//     let builders = [
//         TypeBuilder<(ins "int":$initParam)>,
//         TypeBuilder<(ins CArg<"int", "0">:$initParam)>
//     ];

//     let assemblyFormat = "`<` $initParam `>`";
// }





// def Brainfuck_GenericType: AnyTypeOf<[Brainfuck_CellType, Brainfuck_PtrType]>;


def Brainfuck_CellType: Brainfuck_Type<"Cell", "cell"> {
    let summary = "The cell type of brainfuck";
    let parameters = (ins "int":$pos, "int":$val);

    let assemblyFormat = "`<`  struct($pos, $val) `>`";

    let builders = [
        TypeBuilder<(ins "int":$pos, "int":$val)>
    ];
}

#endif //