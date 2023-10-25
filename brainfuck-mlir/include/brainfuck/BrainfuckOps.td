#ifndef BRAINFUCK_OPS
#define BRAINFUCK_OPS


include "BrainfuckDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class Brainfuck_Op<string mnemonic, list<Trait> traits = []> :
        Op<Brainfuck_Dialect, mnemonic, traits>;




#endif //