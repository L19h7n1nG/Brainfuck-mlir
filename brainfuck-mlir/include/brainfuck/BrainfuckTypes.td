#ifndef BRAINFUCK_TYPES
#define BRAINFUCK_TYPES


include "BrainfuckDialect.td"
include "BrainfuckAttributes.td"

class Brainfuck_Type<string name, string typeMnemonic>
    : TypeDef<Brainfuck_Dialect, name> {
        let mnemonic = typeMnemonic; 
    }

def Brainfuck_CellType: Brainfuck_Type<"Cell", "cell"> {
    let summary = "The cell type of brainfuck";
    let description = [{}];
}


def Brainfuck_PtrType: Brainfuck_Type<"Ptr", "ptr"> {
    let summary = "The pointer type of brainfuck";
    let description = [{}];
}


// def Brainfuck_GenericTypes

#endif //