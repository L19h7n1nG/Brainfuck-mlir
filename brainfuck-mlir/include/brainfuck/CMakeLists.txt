add_mlir_dialect(BrainfuckOps brainfuck)


set(LLVM_TARGET_DEFINITIONS BrainfuckAttributes.td)

mlir_tablegen(BrainfuckAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(BrainfuckAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRBrainfuckAttributesIncGen)
