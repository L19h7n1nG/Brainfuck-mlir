#ifndef BRAINFUCK_ATTRIBUTES
#define BRAINFUCK_ATTRIBUTES



include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "BrainfuckDialect.td"


class Brainfuck_Attr<string name, string attrMnemonic, list<Trait> traits =[]>
    : AttrDef<Brainfuck_Dialect, name, traits> {
        let mnemonic = attrMnemonic;
    }



#endif //