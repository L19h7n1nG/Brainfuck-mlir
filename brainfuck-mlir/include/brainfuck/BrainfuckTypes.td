#ifndef BRAINFUCK_TYPES
#define BRAINFUCK_TYPES


include "BrainfuckDialect.td"
include "BrainfuckAttributes.td"
include "mlir/IR/AttrTypeBase.td"
include "BrainfuckTypeInterfaces.td"




class Brainfuck_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Brainfuck_Dialect, name, traits> {
        let mnemonic = typeMnemonic; 
    }


def Brainfuck_CellType: Brainfuck_Type<"Cell", "cell" , [DeclareTypeInterfaceMethods<CellLikeTypeInterface>]
> {
    let summary = "The cell type of brainfuck";
    let parameters = (ins "int":$pos, "int":$val);

    let assemblyFormat = "`<`  struct($pos, $val) `>`";


    let builders = [
        TypeBuilder<(ins "llvm::StringRef":$pos, "llvm::StringRef":$val)>,
    ];
}



#endif //